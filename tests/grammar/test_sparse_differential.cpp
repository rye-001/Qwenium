// Phase A done criterion #3 — byte-identical token streams across sparse/dense
// decode paths under a fixed grammar and greedy sampling.
//
// Three configurations, same model + grammar + prompt + greedy:
//   A: decode_step normal (sparse fires when |valid_set| < vocab/8)
//   B: decode_step force_dense=true (peek_valid_set still primes cache, but
//      the dense forward + apply_grammar_constraints consume it)
//   C: pre-Phase-A path — no peek_valid_set call. Dense forward, then
//      sampler->sample() runs apply_grammar_constraints fresh.
//
// Outputs:
//   - Token-stream comparison A vs B vs C.
//   - First divergence index, with the three candidates and the valid-set
//     sizes at that step, if any.
//   - "Stuck" status per configuration: did sample() ever hit the
//     no-valid-tokens code path?

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/core/decode_step.h"
#include "../../src/core/model.h"
#include "../../src/loader/chat_template.h"
#include "../../src/loader/gguf_loader.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/models/model_registry.h"
#include "../../src/sampling/grammar_vocab.h"
#include "../../src/sampling/sampling.h"

#include "ggml-backend.h"

#include "ggml.h"
#include <cstdint>
#include <cstdio>

// Tree-portable decode-input application. This one source file is compiled in
// BOTH pre-refactor `main` (@7fac6f7) and the typed-graph-inputs branch so the
// sparse dump is byte-comparable. The branch folded the sparse one-off
// (set_sparse_decode_ids / upload_sparse_indices / consume-on-use clear) into
// SparseHeadInput, applied via set_decode_inputs; `main` uses the legacy
// upload_sparse_indices + set_batched_inputs. Select by header presence so the
// SAME file compiles and produces identical bytes on each tree.
#if defined(__has_include)
#  if __has_include("../../src/graph_inputs/graph_input.h")
#    define QINF_HAVE_TYPED_INPUTS 1
#  endif
#endif

namespace {

void apply_decode_inputs(ForwardPassBase* fp, ggml_cgraph* gf,
                         const std::vector<int32_t>& toks,
                         const std::vector<uint32_t>& slts,
                         const std::vector<int32_t>& poss) {
#ifdef QINF_HAVE_TYPED_INPUTS
    fp->set_decode_inputs(gf, toks, slts, poss);
#else
    fp->upload_sparse_indices();
    fp->set_batched_inputs(gf, toks, slts, poss);
#endif
}


std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(1);
    }
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

struct RunResult {
    std::vector<int32_t> tokens;
    std::vector<size_t>  valid_set_sizes;  // |valid_ids| seen at each step (B/C only)
    bool                 stuck = false;
};

// Run N decode steps with the given mode. Caller must have:
//   - reset the grammar
//   - run prefill so cache_pos(slot) is at end of prompt
//   - sampled the first token (passed in as `first_token`) and called
//     grammar->accept_token on it
// Returns the generated token IDs (NOT including first_token).
//
// mode:
//   0 = decode_step normal (sparse may fire)
//   1 = decode_step force_dense=true
//   2 = manual dense path: forward + sample() (no peek_valid_set)
RunResult run_n(
    ForwardPassBase* fp,
    ggml_backend_sched_t scheduler,
    qwenium::Sampler* sampler,
    qwenium::GrammarVocab* grammar,
    const std::vector<std::string>& vocab,
    uint32_t vocab_size,
    int mode,
    int32_t first_token,
    uint32_t slot,
    int n_tokens,
    const std::vector<int32_t>& stop_ids,
    bool probe = false,
    const char* tag = ""
) {
    RunResult res;
    int32_t next = first_token;
    constexpr int32_t PROBE_TOK = 318;  // ' (' — observed C-divergence token

    for (int i = 0; i < n_tokens; ++i) {
        std::vector<int32_t> history;

        // Pre-step probe: capture the valid set at this exact grammar state.
        // Pure inspection — does not modify grammar.
        if (probe) {
            auto vset = grammar->get_valid_tokens(vocab);
            bool has_probe_tok = false;
            for (int32_t id : vset) if (id == PROBE_TOK) { has_probe_tok = true; break; }
            std::cerr << "[probe " << tag << " step=" << i
                      << " ver=" << grammar->state_version()
                      << "] |valid|=" << vset.size()
                      << " has_318=" << (has_probe_tok ? "Y" : "N") << "\n";
        }

        if (mode == 0 || mode == 1) {
            const bool force_dense = (mode == 1);
            // For the probe, we want to inspect logits after grammar masking.
            // decode_step encapsulates this; for probe, we inline the body.
            if (probe) {
                std::vector<int32_t> valid_ids = sampler->peek_valid_set();
                const bool use_sparse = !force_dense &&
                    !valid_ids.empty() &&
                    valid_ids.size() < vocab_size / 8;
                fp->set_sparse_decode_ids(use_sparse ? valid_ids : std::vector<int32_t>{});

                const int32_t pos = static_cast<int32_t>(fp->get_cache_pos(slot));
                const std::vector<int32_t>  toks = {next};
                const std::vector<uint32_t> slts = {slot};
                const std::vector<int32_t>  poss = {pos};
                ggml_backend_sched_reset(scheduler);
                ggml_cgraph* gf = fp->build_decoding_graph(toks, slts, poss);
                ggml_backend_sched_alloc_graph(scheduler, gf);
                apply_decode_inputs(fp, gf, toks, slts, poss);
                ggml_backend_sched_graph_compute(scheduler, gf);
                fp->advance_cache(1, slot);
                std::vector<float> logits = fp->get_output_logits(gf);

                if (use_sparse) {
                    // sparse path: dump top-3 by sparse_logits, mapped to ids
                    std::vector<size_t> idx(logits.size());
                    for (size_t k = 0; k < idx.size(); ++k) idx[k] = k;
                    std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(3, idx.size()), idx.end(),
                        [&](size_t a, size_t b){ return logits[a] > logits[b]; });
                    std::cerr << "[probe " << tag << " step=" << i << " sparse top]";
                    for (size_t k = 0; k < std::min<size_t>(3, idx.size()); ++k)
                        std::cerr << " " << valid_ids[idx[k]] << "('" << vocab[valid_ids[idx[k]]]
                                  << "')=" << logits[idx[k]];
                    std::cerr << "\n";
                    next = sampler->sample_sparse(logits, valid_ids, history);
                } else {
                    // dense path: dump top-3 raw, then top-3 after grammar masking
                    auto top3 = [&](const std::vector<float>& v, const char* lab) {
                        std::vector<size_t> idx(v.size());
                        for (size_t k = 0; k < idx.size(); ++k) idx[k] = k;
                        std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(),
                            [&](size_t a, size_t b){ return v[a] > v[b]; });
                        std::cerr << "[probe " << tag << " step=" << i << " " << lab << " top]";
                        for (size_t k = 0; k < 3; ++k)
                            std::cerr << " " << idx[k] << "('" << vocab[idx[k]] << "')=" << v[idx[k]];
                        std::cerr << "\n";
                    };
                    auto raw_logits = logits;  // copy for raw top
                    top3(raw_logits, "raw");
                    next = static_cast<int32_t>(sampler->sample(logits, history, vocab));
                    top3(logits, "masked");
                    sampler->accept_token(next);
                }
            } else {
                next = decode_step(fp, scheduler, sampler, next, slot,
                                   history, vocab, vocab_size, force_dense);
            }
        } else {
            const int32_t pos = static_cast<int32_t>(fp->get_cache_pos(slot));
            const std::vector<int32_t>  tokens = {next};
            const std::vector<uint32_t> slots  = {slot};
            const std::vector<int32_t>  positions = {pos};

            fp->set_sparse_decode_ids({});
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);
            ggml_backend_sched_alloc_graph(scheduler, gf);
            apply_decode_inputs(fp, gf, tokens, slots, positions);
            ggml_backend_sched_graph_compute(scheduler, gf);
            fp->advance_cache(1, slot);

            std::vector<float> logits = fp->get_output_logits(gf);
            if (probe) {
                auto raw_logits = logits;
                auto top3 = [&](const std::vector<float>& v, const char* lab) {
                    std::vector<size_t> idx(v.size());
                    for (size_t k = 0; k < idx.size(); ++k) idx[k] = k;
                    std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(),
                        [&](size_t a, size_t b){ return v[a] > v[b]; });
                    std::cerr << "[probe " << tag << " step=" << i << " " << lab << " top]";
                    for (size_t k = 0; k < 3; ++k)
                        std::cerr << " " << idx[k] << "('" << vocab[idx[k]] << "')=" << v[idx[k]];
                    std::cerr << "\n";
                };
                top3(raw_logits, "raw");
                next = static_cast<int32_t>(sampler->sample(logits, history, vocab));
                top3(logits, "masked");
            } else {
                next = static_cast<int32_t>(sampler->sample(logits, history, vocab));
            }
            sampler->accept_token(next);
        }

        res.tokens.push_back(next);

        // Stop on EOS.
        bool is_eos = false;
        for (int32_t s : stop_ids) if (next == s) { is_eos = true; break; }
        if (is_eos) break;
    }

    return res;
}

// ── Sparse byte-identical dump (Phase 1 gate, 7th row) ────────────────────
// Replicates decode_step's sparse branch EXACTLY (force_dense=false), but
// captures the sparse path's actual internal state to a deterministic binary:
// the peek_valid_set ids, the `valid_indices` graph-node readback (the direct
// tripwire for the SparseHeadInput set/clear ORDERING bug — uninitialized or
// prematurely-cleared indices show up here, invisible in post-mapped vocab
// logits), the pre-mapping sparse [k] logits, and the chosen token. Same
// format on both trees via apply_decode_inputs.

void fwr_u64(FILE* f, uint64_t v) { std::fwrite(&v, sizeof(v), 1, f); }

int32_t capture_sparse_step(FILE* f, ForwardPassBase* fp,
                            ggml_backend_sched_t scheduler,
                            qwenium::Sampler* sampler,
                            const std::vector<std::string>& vocab,
                            uint32_t vocab_size, int32_t token, uint32_t slot) {
    // decode_step sparse decision (SPARSE_HEAD_FRACTION_DENOM = 8),
    // force_dense=false (force_dense_param is the sole diagnostic seam).
    std::vector<int32_t> valid_ids = sampler->peek_valid_set();
    const bool use_sparse = !valid_ids.empty() &&
                            valid_ids.size() < vocab_size / 8;
    fp->set_sparse_decode_ids(use_sparse ? valid_ids
                                          : std::vector<int32_t>{});

    const int32_t pos = static_cast<int32_t>(fp->get_cache_pos(slot));
    const std::vector<int32_t>  toks = {token};
    const std::vector<uint32_t> slts = {slot};
    const std::vector<int32_t>  poss = {pos};

    ggml_backend_sched_reset(scheduler);
    ggml_cgraph* gf = fp->build_decoding_graph(toks, slts, poss);
    ggml_backend_sched_alloc_graph(scheduler, gf);
    apply_decode_inputs(fp, gf, toks, slts, poss);
    ggml_backend_sched_graph_compute(scheduler, gf);
    fp->advance_cache(1, slot);

    std::vector<float> logits = fp->get_output_logits(gf);

    // Read back the actual `valid_indices` graph node (post-compute, before
    // the next reset). This is the ordering-bug tripwire.
    std::vector<int32_t> node;
    ggml_tensor* vi = ggml_graph_get_tensor(gf, "valid_indices");
    if (vi) {
        node.resize(static_cast<size_t>(vi->ne[0]));
        ggml_backend_tensor_get(vi, node.data(), 0,
                                node.size() * sizeof(int32_t));
    }

    std::vector<int32_t> history;
    int32_t next = use_sparse
        ? sampler->sample_sparse(logits, valid_ids, history)
        : static_cast<int32_t>(sampler->sample(logits, history, vocab));
    sampler->accept_token(next);

    uint8_t us = use_sparse ? 1 : 0;
    std::fwrite(&us, 1, 1, f);
    fwr_u64(f, valid_ids.size());
    std::fwrite(valid_ids.data(), sizeof(int32_t), valid_ids.size(), f);
    fwr_u64(f, node.size());
    std::fwrite(node.data(), sizeof(int32_t), node.size(), f);
    fwr_u64(f, logits.size());
    std::fwrite(logits.data(), sizeof(float), logits.size(), f);
    std::fwrite(&next, sizeof(int32_t), 1, f);
    return next;
}

} // namespace

int main(int argc, char** argv) {
    // ---- Configuration ----
    const char* env_model = std::getenv("QWENIUM_MODEL_PATH");
    std::string model_path = env_model ? env_model
                                       : "./Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf";
    std::string grammar_path = "grammar/order-management.gbnf";
    std::string system_prompt_path = "tests/system_prompt_order_mngmt.txt";
    std::string user_query = "get all platinum customers";
    int n_tokens = 30;
    uint32_t context_length = 4096;
    uint32_t kv_quant_bits = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--grammar") == 0 && i+1 < argc) grammar_path = argv[++i];
        else if (std::strcmp(argv[i], "--system-prompt") == 0 && i+1 < argc) system_prompt_path = argv[++i];
        else if (std::strcmp(argv[i], "--query") == 0 && i+1 < argc) user_query = argv[++i];
        else if (std::strcmp(argv[i], "-n") == 0 && i+1 < argc) n_tokens = std::stoi(argv[++i]);
    }

    std::cerr << "[diff] model: " << model_path << "\n"
              << "[diff] grammar: " << grammar_path << "\n"
              << "[diff] n_tokens: " << n_tokens << "\n";

    ggml_log_set([](ggml_log_level lvl, const char* text, void*) {
        if (lvl == GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", text);
    }, nullptr);
    ggml_backend_load_all();

    // ---- Load model ----
    Model model;
    register_builtin_models();
    try {
        model.load_metadata(model_path);
        model.load_tensors();
    } catch (const std::exception& e) {
        std::cerr << "ERROR loading model: " << e.what() << "\n";
        return 1;
    }
    Tokenizer* tokenizer = model.get_tokenizer();
    const auto raw_vocab = tokenizer->get_vocabulary();
    std::vector<std::string> decoded_vocab;
    decoded_vocab.reserve(raw_vocab.size());
    for (size_t i = 0; i < raw_vocab.size(); ++i) {
        decoded_vocab.push_back(tokenizer->decode(static_cast<int32_t>(i)));
    }
    const auto& meta = model.get_metadata();
    const uint32_t vocab_size = meta.vocab_size;

    // ---- Load grammar ----
    auto grammar_str = slurp(grammar_path);
    auto grammar = qwenium::GrammarVocab::parse_impl(grammar_str);
    if (!grammar) {
        std::cerr << "ERROR: failed to parse grammar\n";
        return 1;
    }

    // ---- Render prompt ----
    const ChatTemplate* tmpl = lookup_chat_template(meta.architecture);
    if (!tmpl) { std::cerr << "ERROR: no chat template\n"; return 1; }
    std::string system_content = slurp(system_prompt_path);
    std::string system_turn = tmpl->render({{"system", system_content}}, false);
    std::string user_turn   = tmpl->render({{"user", user_query}}, true);
    auto system_tokens = tokenizer->encode(system_turn);
    auto user_tokens   = tokenizer->encode(user_turn);

    std::cerr << "[diff] system_tokens=" << system_tokens.size()
              << " user_tokens=" << user_tokens.size() << "\n";

    // ---- Forward pass + scheduler ----
    // 4 slots: 0 = system prefill source, 1/2/3 = per-config session slots.
    // Dedicating a slot per config prevents DeltaNet recurrent-state contamination
    // across runs (overwrite-semantics state can't be rewound by set_cache_pos alone).
    auto forward_pass = create_forward_pass(model, &meta, context_length, 4, kv_quant_bits);
    ggml_backend_sched_t scheduler = model.get_scheduler();

    const uint32_t prefill_slot = 0;
    const uint32_t slot_A = 1;
    const uint32_t slot_B = 2;
    const uint32_t slot_C = 3;

    forward_pass->run_prefill(system_tokens, 0, prefill_slot, scheduler);
    forward_pass->clone_slot(prefill_slot, slot_A, system_tokens.size());
    forward_pass->clone_slot(prefill_slot, slot_B, system_tokens.size());
    forward_pass->clone_slot(prefill_slot, slot_C, system_tokens.size());

    const std::vector<int32_t> stop_ids = meta.stop_token_ids;

    auto fresh_start = [&](qwenium::Sampler* sampler, uint32_t slot) -> int32_t {
        grammar->reset();
        std::vector<float> logits = forward_pass->run_prefill(
            user_tokens, static_cast<int32_t>(system_tokens.size()),
            slot, scheduler);
        std::vector<float> last(logits.end() - vocab_size, logits.end());
        std::vector<int32_t> history;
        int32_t tok = static_cast<int32_t>(sampler->sample(last, history, decoded_vocab));
        grammar->accept_token(tok, decoded_vocab);
        return tok;
    };

    // ---- Run three configurations ----
    auto make_sampler = [&]() {
        auto s = std::make_unique<qwenium::GreedySampler>();
        s->set_grammar(grammar.get());
        s->build_token_trie(decoded_vocab);
        for (int32_t id : stop_ids) s->add_eos_token_id(id);
        return s;
    };

    // ── Sparse byte-identical dump mode (Phase 1 gate, 7th row) ───────────
    // QWENIUM_SPARSE_DUMP=<path>: run ONLY the sparse config (Run A: real
    // grammar, force_dense=false) deterministically and write a binary the
    // same way logits_dump.cpp does, so `cmp`/sha256 works branch-vs-main.
    if (const char* dump_path = std::getenv("QWENIUM_SPARSE_DUMP")) {
        auto s = make_sampler();
        grammar->reset();
        std::vector<float> prefill = forward_pass->run_prefill(
            user_tokens, static_cast<int32_t>(system_tokens.size()),
            slot_A, scheduler);
        std::vector<float> last(prefill.end() - vocab_size, prefill.end());
        std::vector<int32_t> hist0;
        int32_t first = static_cast<int32_t>(
            s->sample(last, hist0, decoded_vocab));
        grammar->accept_token(first, decoded_vocab);

        FILE* f = std::fopen(dump_path, "wb");
        if (!f) { std::perror("fopen"); return 1; }
        // 1. Full dense prefill logits (sanity; same path as existing gate).
        fwr_u64(f, prefill.size());
        std::fwrite(prefill.data(), sizeof(float), prefill.size(), f);
        // 2. First token (chosen at prefill under grammar).
        std::fwrite(&first, sizeof(int32_t), 1, f);
        // 3. Eight grammar-constrained sparse decode steps, full internal
        //    state per step (valid ids, valid_indices node, [k] logits, tok).
        int32_t tok = first;
        uint8_t n_steps = 0;
        long count_pos = std::ftell(f);
        std::fwrite(&n_steps, 1, 1, f);  // placeholder, rewritten below
        for (int i = 0; i < 8; ++i) {
            tok = capture_sparse_step(f, forward_pass.get(), scheduler,
                                      s.get(), decoded_vocab, vocab_size,
                                      tok, slot_A);
            ++n_steps;
            bool eos = false;
            for (int32_t sid : stop_ids) if (tok == sid) { eos = true; break; }
            if (eos) break;
        }
        std::fseek(f, count_pos, SEEK_SET);
        std::fwrite(&n_steps, 1, 1, f);
        std::fclose(f);
        std::fprintf(stderr,
            "[sparse-dump] wrote %s: %zu prefill floats, first=%d, %u steps\n",
            dump_path, prefill.size(), first, (unsigned)n_steps);
        return 0;
    }

    const char* probe_env = std::getenv("QWENIUM_DIFF_PROBE");
    const bool do_probe = probe_env && std::string(probe_env) == "1";

    std::cerr << "\n[diff] --- Run A: decode_step (sparse fires when narrow) ---\n";
    auto sA = make_sampler();
    int32_t firstA = fresh_start(sA.get(), slot_A);
    auto resA = run_n(forward_pass.get(), scheduler, sA.get(), grammar.get(),
                     decoded_vocab, vocab_size, 0, firstA, slot_A, n_tokens, stop_ids,
                     do_probe, "A");
    std::cerr << "[diff] A: " << (1 + resA.tokens.size()) << " tokens\n";

    std::cerr << "\n[diff] --- Run B: decode_step force_dense=true ---\n";
    auto sB = make_sampler();
    int32_t firstB = fresh_start(sB.get(), slot_B);
    auto resB = run_n(forward_pass.get(), scheduler, sB.get(), grammar.get(),
                     decoded_vocab, vocab_size, 1, firstB, slot_B, n_tokens, stop_ids,
                     do_probe, "B");
    std::cerr << "[diff] B: " << (1 + resB.tokens.size()) << " tokens\n";

    std::cerr << "\n[diff] --- Run C: manual dense, no peek_valid_set ---\n";
    auto sC = make_sampler();
    int32_t firstC = fresh_start(sC.get(), slot_C);
    auto resC = run_n(forward_pass.get(), scheduler, sC.get(), grammar.get(),
                     decoded_vocab, vocab_size, 2, firstC, slot_C, n_tokens, stop_ids,
                     do_probe, "C");
    std::cerr << "[diff] C: " << (1 + resC.tokens.size()) << " tokens\n";

    // ---- Compare ----
    auto dump = [&](const char* tag, int32_t first, const RunResult& r) {
        std::cerr << "[" << tag << "] " << first;
        for (int32_t t : r.tokens) std::cerr << "," << t;
        std::cerr << "\n";
        std::cerr << "[" << tag << " decoded] '" << decoded_vocab[first] << "'";
        for (int32_t t : r.tokens) std::cerr << " '" << decoded_vocab[t] << "'";
        std::cerr << "\n";
    };
    dump("A", firstA, resA);
    dump("B", firstB, resB);
    dump("C", firstC, resC);

    auto compare = [&](const char* tagX, int32_t fX, const RunResult& rX,
                       const char* tagY, int32_t fY, const RunResult& rY) {
        if (fX != fY) {
            std::cerr << "[diverge " << tagX << " vs " << tagY << "] @0 "
                      << fX << " ('" << decoded_vocab[fX] << "') vs "
                      << fY << " ('" << decoded_vocab[fY] << "')\n";
            return false;
        }
        const size_t n = std::min(rX.tokens.size(), rY.tokens.size());
        for (size_t i = 0; i < n; ++i) {
            if (rX.tokens[i] != rY.tokens[i]) {
                std::cerr << "[diverge " << tagX << " vs " << tagY << "] @" << (i+1)
                          << " " << rX.tokens[i] << " ('"
                          << decoded_vocab[rX.tokens[i]] << "') vs "
                          << rY.tokens[i] << " ('"
                          << decoded_vocab[rY.tokens[i]] << "')\n";
                return false;
            }
        }
        if (rX.tokens.size() != rY.tokens.size()) {
            std::cerr << "[diverge " << tagX << " vs " << tagY << "] length "
                      << rX.tokens.size() << " vs " << rY.tokens.size() << "\n";
            return false;
        }
        std::cerr << "[match " << tagX << " == " << tagY << "] "
                  << (1 + rX.tokens.size()) << " tokens identical\n";
        return true;
    };

    bool ab = compare("A", firstA, resA, "B", firstB, resB);
    bool bc = compare("B", firstB, resB, "C", firstC, resC);
    bool ac = compare("A", firstA, resA, "C", firstC, resC);

    std::cerr << "\n[diff] === Verdict ===\n";
    std::cerr << "[diff] A==B (sparse path matches dense, Phase A criterion #3): "
              << (ab ? "PASS" : "FAIL") << "\n";
    std::cerr << "[diff] B==C (peek_valid_set cache is safe): "
              << (bc ? "PASS" : "FAIL") << "\n";
    std::cerr << "[diff] A==C: " << (ac ? "PASS" : "FAIL") << "\n";

    if (ab && bc && ac) {
        std::cerr << "[diff] All three identical. Stuck-state bug (if any) is "
                     "pre-existing in GrammarVocab.\n";
        return 0;
    }
    return 1;
}
