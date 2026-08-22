// logits_dump.cpp — deterministic byte-identical differential harness.
//
// Uses ONLY APIs stable across pre-refactor `main` and the typed-graph-inputs
// branch (Model load, create_forward_pass, run_prefill, decode_step,
// GreedySampler). Compile the SAME file in both trees, run on the same model
// with the same fixed token prompt, and `cmp` the output files: any divergence
// in input-tensor population perturbs logits and fails the gate.
//
// Usage: logits_dump <model.gguf> <out.bin> [options]
//
// With no options the behaviour is the historical one: a fixed synthetic
// 24-token prompt, 9 greedy decode steps, byte-comparable output file. The
// options below exist for the cross-engine divergence probe (Objective 2 of
// docs/note-engine-divergence-probe.md) and do not alter the default path.
//
//   --tokens a,b,c   prefill these token IDs instead of the synthetic prompt
//   --text "..."     tokenize this string (engine tokenizer, no BOS added)
//   --topk N         print top-N logits per prefill position to stdout
//   --steps N        greedy decode steps (default 9)
//   --bos            prepend the GGUF-declared bos_token_id (fails if -1)
//   --dtopk          print top-k from the DECODE graph at each greedy step,
//                    so decode-path and prefill-path logits for the same
//                    context can be compared directly
//   --kv-f16         build the attention KV cache as F16 instead of F32
//   --mem            report per-slot KV and recurrent-state bytes (the figure
//                    that actually bounds the server's slot count)
//   --bench          time prefill and N argmax decode steps separately and
//                    report tok/s (no sampler heuristics in the loop)
//   --tf             teacher-forced sweep: re-prefill every growing prefix of
//                    the prompt and print top-k at each, so the "positions
//                    deeper" leg needs no sampler and no decode path

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

#include "ggml-backend.h"
#include "../../src/core/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/core/decode_step.h"
#include "../../src/sampling/sampling.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/models/qwen35.h"
#include "../../src/models/qwen36.h"

namespace {

std::vector<int32_t> parse_csv_ids(const std::string& s) {
    std::vector<int32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        if (!tok.empty()) out.push_back(static_cast<int32_t>(std::strtol(tok.c_str(), nullptr, 10)));
        i = j + 1;
    }
    return out;
}

// Print the top-k entries of one logit row, with the softmax probability so
// near-ties are visible as ties rather than as raw-magnitude noise.
void print_topk(const float* row, uint32_t vocab_size, int k,
                const std::vector<std::string>& vocab, const char* label) {
    std::vector<uint32_t> idx(vocab_size);
    std::iota(idx.begin(), idx.end(), 0u);
    const int kk = std::min<int>(k, static_cast<int>(vocab_size));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });

    const float max_logit = row[idx[0]];
    double sum = 0.0;
    for (uint32_t v = 0; v < vocab_size; ++v) sum += std::exp(static_cast<double>(row[v] - max_logit));

    std::printf("%s\n", label);
    for (int r = 0; r < kk; ++r) {
        const uint32_t t = idx[r];
        const double p = std::exp(static_cast<double>(row[t] - max_logit)) / sum;
        const std::string piece = t < vocab.size() ? vocab[t] : std::string("<oob>");
        std::printf("  %d. id=%-7u logit=%12.6f  p=%.6f  %s\n", r + 1, t, row[t], p, piece.c_str());
    }
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <out.bin> [--tokens a,b,c] [--text S] [--topk N] [--steps N] [--bos] [--tf]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string out_path = argv[2];

    std::string tokens_csv, text_arg;
    int topk = 0;
    int steps = 9;          // historical: 8 loop iterations + the trailing push
    bool want_bos = false;
    bool want_tf  = false;
    bool want_dtopk = false;
    bool want_bench = false;
    bool want_mem   = false;
    bool want_kv_f16= false;

    for (int a = 3; a < argc; ++a) {
        const std::string opt = argv[a];
        auto need = [&](const char* name) -> const char* {
            if (a + 1 >= argc) {
                std::fprintf(stderr, "error: option %s: expected a value, got end of arguments\n", name);
                std::exit(2);
            }
            return argv[++a];
        };
        if (opt == "--tokens")      tokens_csv = need("--tokens");
        else if (opt == "--text")   text_arg   = need("--text");
        else if (opt == "--topk")   topk       = std::atoi(need("--topk"));
        else if (opt == "--steps")  steps      = std::atoi(need("--steps"));
        else if (opt == "--bos")    want_bos   = true;
        else if (opt == "--tf")     want_tf    = true;
        else if (opt == "--dtopk")  want_dtopk = true;
        else if (opt == "--bench")  want_bench = true;
        else if (opt == "--mem")    want_mem   = true;
        else if (opt == "--kv-f16") want_kv_f16= true;
        else {
            std::fprintf(stderr, "error: argv[%d]: expected one of --tokens/--text/--topk/--steps/--bos/--tf/--dtopk/--bench/--mem/--kv-f16, got '%s'\n",
                         a, opt.c_str());
            return 2;
        }
    }
    if (!tokens_csv.empty() && !text_arg.empty()) {
        std::fprintf(stderr, "error: prompt source: expected exactly one of --tokens or --text, got both\n");
        return 2;
    }

    ggml_backend_load_all();
    register_builtin_models();

    Model model;
    model.load_metadata(model_path);
    model.load_tensors();

    const auto& meta = model.get_metadata();
    const uint32_t vocab_size = meta.vocab_size;

    auto fp = create_forward_pass(model, &meta, 4096, 1,
                                  want_kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    ggml_backend_sched_t sched = model.get_scheduler();

    const auto vocab = model.get_tokenizer()->get_vocabulary();

    std::vector<int32_t> tokens;
    if (!tokens_csv.empty()) {
        tokens = parse_csv_ids(tokens_csv);
    } else if (!text_arg.empty()) {
        tokens = model.get_tokenizer()->encode(text_arg);
    } else {
        // Fixed deterministic prompt (token ids only — no tokenizer variance).
        for (int32_t i = 0; i < 24; ++i) tokens.push_back((i * 7 + 3) % 1000);
    }
    if (want_bos) {
        if (meta.bos_token_id < 0) {
            std::fprintf(stderr, "error: --bos: expected a declared bos_token_id >= 0, got %d\n", meta.bos_token_id);
            return 2;
        }
        tokens.insert(tokens.begin(), meta.bos_token_id);
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "error: prompt: expected at least 1 token, got 0\n");
        return 2;
    }

    if (topk > 0) {
        std::printf("# model            : %s\n", model_path.c_str());
        std::printf("# vocab_size       : %u\n", vocab_size);
        std::printf("# bos_token_id     : %d  (add_bos_token=%d)\n", meta.bos_token_id, (int)meta.add_bos_token);
        std::printf("# eos_token_id     : %d\n", meta.eos_token_id);
        std::printf("# prompt tokens (%zu): ", tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) std::printf("%d%s", tokens[i], i + 1 == tokens.size() ? "\n" : ",");
        std::printf("# prompt text      : %s\n", model.get_tokenizer()->decode(tokens).c_str());
        std::fflush(stdout);
    }

    std::vector<float> prefill = fp->run_prefill(tokens, 0, 0, sched);

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::perror("fopen"); return 1; }

    // 1. Full prefill logits buffer (exercises Tokens/Positions/AttnMask —
    //    incl. Gemma per-layer interleaved local/global windows).
    uint64_t n = prefill.size();
    std::fwrite(&n, sizeof(n), 1, f);
    std::fwrite(prefill.data(), sizeof(float), prefill.size(), f);

    // The recipe's prefill graph applies the LM head to the LAST position
    // only, so `prefill` holds exactly one logit row: the distribution over
    // the token that follows the whole prompt. Label it as such — treating
    // row 0 as "position 0" would misattribute it to the first prompt token.
    if (topk > 0) {
        const size_t n_rows = prefill.size() / vocab_size;
        if (n_rows * vocab_size != prefill.size()) {
            std::fprintf(stderr, "error: prefill logits: expected a multiple of vocab_size %u, got %zu floats\n",
                         vocab_size, prefill.size());
            return 1;
        }
        for (size_t r = 0; r < n_rows; ++r) {
            const size_t p = tokens.size() - n_rows + r;   // absolute prompt index
            char label[512];
            std::snprintf(label, sizeof(label),
                          "\n[pos %zu] ctx=%zu tok, last id=%d '%s' -> predicts next:",
                          p, p + 1, tokens[p], model.get_tokenizer()->decode(tokens[p]).c_str());
            print_topk(prefill.data() + r * vocab_size, vocab_size, topk, vocab, label);
        }
    }

    // Teacher-forced sweep. Re-prefill each growing prefix from a cleared
    // slot, so every position's distribution comes from the same prefill code
    // path with no sampler and no decode graph in the comparison. clear_slot
    // resets attention KV *and* DeltaNet recurrent state, so prefixes do not
    // bleed into one another.
    if (want_tf) {
        if (topk <= 0) {
            std::fprintf(stderr, "error: --tf: expected --topk N with N > 0, got %d\n", topk);
            return 2;
        }
        std::printf("\n# ---- teacher-forced sweep (%zu prefixes) ----\n", tokens.size());
        for (size_t len = 1; len <= tokens.size(); ++len) {
            fp->clear_slot(0);
            fp->set_cache_pos(0, 0);
            std::vector<int32_t> prefix(tokens.begin(), tokens.begin() + len);
            std::vector<float> lg = fp->run_prefill(prefix, 0, 0, sched);
            if (lg.size() < vocab_size) {
                std::fprintf(stderr, "error: tf prefix len %zu: expected >= %u logits, got %zu\n",
                             len, vocab_size, lg.size());
                return 1;
            }
            const float* row = lg.data() + lg.size() - vocab_size;
            const int32_t actual = len < tokens.size() ? tokens[len] : -1;
            char label[512];
            std::snprintf(label, sizeof(label),
                          "\n[tf pos %zu] ctx='%s' -> predicts next (actual next id=%d '%s'):",
                          len - 1, model.get_tokenizer()->decode(prefix).c_str(),
                          actual,
                          actual >= 0 ? model.get_tokenizer()->decode(actual).c_str() : "<end>");
            print_topk(row, vocab_size, topk, vocab, label);
        }
        // Leave the slot holding the full prompt so the greedy leg below is
        // unaffected by the sweep.
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
        prefill = fp->run_prefill(tokens, 0, 0, sched);
    }

    // 2. Greedy decode continuation (exercises kq_mask_b + gather_indices via
    //    decode_step). Recipes whose decode graph is unimplemented throw —
    //    record a sentinel so the comparison is still well-defined.
    // State-memory report. KV grows with context; recurrent state does not —
    // it is a fixed per-slot cost. Reporting them separately is the point:
    // only one of them scales with the prompt.
    if (want_mem) {
        simple_kv_cache* kv = fp->snapshot_kv_cache();
        DeltaNetState*   dn = fp->snapshot_recurrent();
        const double MB = 1024.0 * 1024.0;
        std::printf("\n# ---- state memory (ctx 4096, 1 slot) ----\n");
        if (kv) std::printf("# kv_cache_MB     : %.2f\n", kv->memory_bytes() / MB);
        else    std::printf("# kv_cache_MB     : <recipe exposes none>\n");
        if (dn) std::printf("# recurrent_MB    : %.2f   (per slot, context-independent)\n", dn->memory_bytes() / MB);
        else    std::printf("# recurrent_MB    : <recipe exposes none>\n");
        if (kv && dn) {
            const double per_slot = (kv->memory_bytes() + dn->memory_bytes()) / MB;
            std::printf("# per_slot_total_MB: %.2f\n", per_slot);
            std::printf("# at_10_slots_GB   : %.2f\n", per_slot * 10 / 1024.0);
        }
        std::fflush(stdout);
    }

    // Throughput bench. Prefill and decode are timed separately because they
    // are different regimes: prefill is compute-bound over the whole prompt,
    // decode is launch/bandwidth-bound at one token per step. Reporting a
    // single blended tok/s hides which one an engine is actually good at.
    if (want_bench) {
        using clk = std::chrono::steady_clock;
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);

        const auto t0 = clk::now();
        std::vector<float> lg = fp->run_prefill(tokens, 0, 0, sched);
        const auto t1 = clk::now();
        const double prefill_s = std::chrono::duration<double>(t1 - t0).count();

        const float* row = lg.data() + lg.size() - vocab_size;
        int32_t tok = static_cast<int32_t>(std::max_element(row, row + vocab_size) - row);

        const auto t2 = clk::now();
        for (int step = 0; step < steps; ++step) {
            ggml_backend_sched_reset(sched);
            const std::vector<int32_t>  dt = {tok};
            const std::vector<uint32_t> ds = {0u};
            const std::vector<int32_t>  dp = {static_cast<int32_t>(fp->get_cache_pos(0))};
            ggml_cgraph* dg = fp->build_decoding_graph(dt, ds, dp);
            ggml_backend_sched_alloc_graph(sched, dg);
            fp->set_decode_inputs(dg, dt, ds, dp);
            ggml_backend_sched_graph_compute(sched, dg);
            fp->advance_cache(1, 0);
            std::vector<float> dl = fp->get_output_logits(dg);
            tok = static_cast<int32_t>(std::max_element(dl.begin(), dl.begin() + vocab_size) - dl.begin());
        }
        const auto t3 = clk::now();
        const double decode_s = std::chrono::duration<double>(t3 - t2).count();

        std::printf("\n# ---- bench ----\n");
        std::printf("# prompt_tokens   : %zu\n", tokens.size());
        std::printf("# prefill_s       : %.4f\n", prefill_s);
        std::printf("# prefill_tok_s   : %.2f\n", tokens.size() / prefill_s);
        std::printf("# decode_steps    : %d\n", steps);
        std::printf("# decode_s        : %.4f\n", decode_s);
        std::printf("# decode_tok_s    : %.2f\n", steps / decode_s);
        std::printf("# decode_ms_step  : %.3f\n", 1000.0 * decode_s / steps);
        std::fflush(stdout);

        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
        prefill = fp->run_prefill(tokens, 0, 0, sched);
    }

    // Decode-path probe. Runs the dense decode route inline (rather than
    // through decode_step, whose signature returns only a token) so the
    // decode graph's logits for a given context can be diffed against the
    // prefill graph's logits for the same context.
    if (want_dtopk) {
        if (topk <= 0) {
            std::fprintf(stderr, "error: --dtopk: expected --topk N with N > 0, got %d\n", topk);
            return 2;
        }
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
        std::vector<float> lg = fp->run_prefill(tokens, 0, 0, sched);
        std::vector<int32_t> ctx = tokens;
        std::printf("\n# ---- decode-path sweep (%d steps) ----\n", steps);
        const float* row = lg.data() + lg.size() - vocab_size;
        for (int step = 0; step < steps; ++step) {
            const uint32_t argmax =
                static_cast<uint32_t>(std::max_element(row, row + vocab_size) - row);
            char label[512];
            std::snprintf(label, sizeof(label),
                          "\n[decode step %d] pos %zu, ctx='%s' -> predicts next:",
                          step, ctx.size() - 1,
                          model.get_tokenizer()->decode(ctx).c_str());
            print_topk(row, vocab_size, topk, vocab, label);

            const int32_t tok = static_cast<int32_t>(argmax);
            ctx.push_back(tok);

            ggml_backend_sched_reset(sched);
            const std::vector<int32_t>  dt = {tok};
            const std::vector<uint32_t> ds = {0u};
            const std::vector<int32_t>  dp = {static_cast<int32_t>(fp->get_cache_pos(0))};
            ggml_cgraph* dg = fp->build_decoding_graph(dt, ds, dp);
            ggml_backend_sched_alloc_graph(sched, dg);
            fp->set_decode_inputs(dg, dt, ds, dp);
            ggml_backend_sched_graph_compute(sched, dg);
            fp->advance_cache(1, 0);
            lg = fp->get_output_logits(dg);
            if (lg.size() < vocab_size) {
                std::fprintf(stderr, "error: decode step %d: expected >= %u logits, got %zu\n",
                             step, vocab_size, lg.size());
                return 1;
            }
            row = lg.data();
        }
        std::printf("\n# decode-path text : %s\n", model.get_tokenizer()->decode(ctx).c_str());
        std::fflush(stdout);
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
        prefill = fp->run_prefill(tokens, 0, 0, sched);
    }

    qwenium::GreedySampler sampler;
    sampler.build_token_trie(vocab);
    std::vector<int32_t> history = tokens;
    std::vector<int32_t> gen;
    bool decode_ok = true;
    try {
        std::vector<float> last(prefill.end() - vocab_size, prefill.end());
        int32_t tok = static_cast<int32_t>(sampler.sample(last, history, vocab));
        for (int step = 0; step < steps - 1; ++step) {
            gen.push_back(tok);
            history.push_back(tok);
            tok = decode_step(fp.get(), sched, &sampler, tok, 0,
                              history, vocab, vocab_size, true);
        }
        gen.push_back(tok);
    } catch (const std::exception& e) {
        decode_ok = false;
    }
    uint8_t ok = decode_ok ? 1 : 0;
    std::fwrite(&ok, 1, 1, f);
    uint64_t g = gen.size();
    std::fwrite(&g, sizeof(g), 1, f);
    std::fwrite(gen.data(), sizeof(int32_t), gen.size(), f);

    std::fclose(f);

    if (topk > 0) {
        std::printf("\n# greedy continuation (%zu tokens, decode_ok=%d): ", gen.size(), (int)ok);
        for (size_t i = 0; i < gen.size(); ++i) std::printf("%d%s", gen[i], i + 1 == gen.size() ? "\n" : ",");
        std::printf("# greedy text      : %s\n", model.get_tokenizer()->decode(gen).c_str());
        std::fflush(stdout);
    }

    std::fprintf(stderr, "wrote %s: %llu prefill floats, decode_ok=%d, %llu gen\n",
                 out_path.c_str(), (unsigned long long)n, (int)ok,
                 (unsigned long long)g);
    return 0;
}
