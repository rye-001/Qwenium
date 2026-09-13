// Closed-form draft head probe — capture stage.
//
// Runs greedy (argmax, temperature 0) decoding over a few fixed prompts and
// dumps, for every generated step t, the POST-final-norm hidden state h_t
// alongside the token actually generated from it. h_t is the "final_norm"
// tensor build_output_head names unconditionally in every recipe (base
// class, src/models/forward_pass_base.cpp) — NOT the D3 "hidden_out" seam,
// which only qwen36.cpp populates (qwen35 has no IMtpDraftable head, so
// set_output_hidden(true) throws fail-loud there: "'hidden_out' tensor not
// found in graph"). Using final_norm instead of the pre-norm D3 tap needs no
// recipe-file change: this harness calls build_prefill_graph() directly (the
// same public entry run_prefill wraps) and marks the already-built
// "final_norm" node as a graph output itself, exactly the pattern
// mark_attention_taps() uses for the lens tap. Because h_t is already
// normed, argmax(lm_head(h_t)) reduces to a single matmul with the output
// weight (no norm replay needed) — one fewer moving part for the numpy
// scoring stage.
//
// Offline (numpy) fits ridge-regression heads W: h_t -> h_{t+1} and
// W2: h_t -> h_{t+2} and scores argmax(output_weight . (W . h_t)) against
// the true x_{t+2} (p2) and argmax(output_weight . (W2 . h_t)) against the
// true x_{t+3} (p3). See docs/note-closed-form-draft-head.md.
//
//   QWEN36_MODEL_PATH=models/Qwen3.5-0.8B-BF16.gguf ./bin/draft-head-capture
//
// (No argv parsing — model path resolves ONLY from QWEN36_MODEL_PATH, else
// the qwen35moe default below. Mirrors verify_cost_curve.cpp.)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#include "ggml.h"
#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/loader/tokenizer.h"

namespace {

int32_t argmax_last(const std::vector<float>& logits, size_t vocab_size) {
    const float* row = logits.data() + (logits.size() - vocab_size);
    size_t best = 0;
    float best_v = row[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (row[i] > best_v) { best_v = row[i]; best = i; }
    }
    return (int32_t)best;
}

// Replicates ForwardPassBase::run_prefill's body, but additionally taps the
// "final_norm" tensor (built unconditionally by build_output_head, base
// class) instead of the recipe-gated D3 "hidden_out" seam. Returns logits;
// *final_norm_out receives the [hidden, n_tokens] post-norm hidden state.
std::vector<float> run_prefill_tap_final_norm(
        ForwardPassBase* fp, const std::vector<int32_t>& tokens, int pos,
        uint32_t slot_idx, ggml_backend_sched_t sched, size_t hidden,
        std::vector<float>* final_norm_out) {
    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = fp->build_prefill_graph(tokens, pos, slot_idx);
    ggml_tensor* fn = ggml_graph_get_tensor(gf, "final_norm");
    if (!fn) throw std::runtime_error("final_norm tensor not found in graph");
    ggml_set_output(fn);
    ggml_backend_sched_alloc_graph(sched, gf);
    fp->set_prefill_inputs(gf, tokens, pos);
    qinf::engine::require_compute_success(
        ggml_backend_sched_graph_compute(sched, gf), "run_prefill_tap_final_norm");
    fp->advance_cache((uint32_t)tokens.size(), slot_idx);

    size_t n = ggml_nbytes(fn);
    final_norm_out->resize(n / sizeof(float));
    ggml_backend_tensor_get(fn, final_norm_out->data(), 0, n);
    if (final_norm_out->size() < hidden) {
        throw std::runtime_error("final_norm smaller than hidden_size — layout assumption wrong");
    }
    return fp->get_output_logits(gf);
}

}  // namespace

int main() {
    const char* env = std::getenv("QWEN36_MODEL_PATH");
    std::string path = env ? env : "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";
    const uint32_t CTX = 2048;
    const int GEN_PER_PROMPT = 1500;

    register_builtin_models();
    std::cerr << "Loading " << path << " ...\n";
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    std::cerr << "architecture=" << meta.architecture
              << " embedding_length=" << meta.embedding_length
              << " vocab_size=" << meta.vocab_size << "\n";

    auto fp = create_forward_pass(model, &meta, CTX, 1);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();
    const size_t vocab_size = meta.vocab_size;
    const size_t hidden = meta.embedding_length;

    // ── Teacher-forced capture over REAL text ──────────────────────────────
    // Free-running greedy generation DEGENERATES: a 0.8B base model loops
    // within a few hundred tokens (measured: 18-116 distinct tokens per 1500,
    // 95-100% 8-gram self-match), which makes h_{t+1} trivially predictable
    // from h_t and the whole probe vacuous (it scored p2=0.963 on pure loop).
    // Teacher forcing cannot degenerate: the input is a fixed corpus, not the
    // model's own output. Labels remain the MODEL's argmax at each position --
    // what a draft head must actually predict -- not the corpus token.
    const int    CHUNK      = 128;
    const size_t MAX_CHUNKS = 180;

    const char* corpus_files[] = {
        "docs/architecture.md",
        "docs/decode-gap-status.md",
        "docs/modular-layer-architecture.md",
        "src/layers/attention.cpp",
        "src/models/qwen35_family.cpp",
        "src/sampling/grammar_vocab.cpp",
        "src/server/inference_server.h",
    };
    std::string corpus;
    for (const char* cf : corpus_files) {
        std::ifstream in(cf);
        if (!in) { std::cerr << "skip (missing): " << cf << "\n"; continue; }
        std::stringstream ss; ss << in.rdbuf();
        corpus += ss.str();
        corpus += "\n\n";
    }
    if (corpus.empty()) { std::cerr << "empty corpus\n"; return 1; }
    std::cerr << "corpus bytes: " << corpus.size() << "\n";

    std::vector<int32_t> all_tokens = tok->encode(corpus);
    std::cerr << "corpus tokens: " << all_tokens.size() << "\n";

    size_t n_chunks = all_tokens.size() / CHUNK;
    if (n_chunks > MAX_CHUNKS) n_chunks = MAX_CHUNKS;
    if (n_chunks == 0) { std::cerr << "corpus too small\n"; return 1; }

    // All-position logits: the prefill head slice must be OFF or only the last
    // position's logits come back.
    fp->set_slice_prefill_head(false);

    std::string out_path = "/private/tmp/claude-987549057/-Users-sfadaei-dev-jprojs-fas-github-qwen-inference/d592d7bb-53fc-4dfb-809b-b234be37593f/scratchpad/draft_head_capture.bin";
    std::ofstream out(out_path, std::ios::binary);
    if (!out) { std::cerr << "failed to open output file\n"; return 1; }

    int32_t hidden_i32 = (int32_t)hidden;
    int32_t vocab_i32  = (int32_t)vocab_size;
    int32_t n_seq_i32  = (int32_t)n_chunks;
    out.write((char*)&hidden_i32, 4);
    out.write((char*)&vocab_i32, 4);
    out.write((char*)&n_seq_i32, 4);

    int total_records = 0;
    for (size_t ci = 0; ci < n_chunks; ++ci) {
        std::vector<int32_t> chunk(all_tokens.begin() + ci * CHUNK,
                                   all_tokens.begin() + (ci + 1) * CHUNK);
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);

        std::vector<float> fn;
        std::vector<float> logits =
            run_prefill_tap_final_norm(fp.get(), chunk, 0, 0, sched, hidden, &fn);

        if (fn.size() < hidden * chunk.size()) {
            std::cerr << "final_norm has " << fn.size() << " floats, expected >= "
                      << hidden * chunk.size() << " -- layout assumption wrong\n";
            return 1;
        }
        if (logits.size() < vocab_size * chunk.size()) {
            std::cerr << "logits has " << logits.size() << " floats, expected >= "
                      << vocab_size * chunk.size()
                      << " -- head slice still on?\n";
            return 1;
        }

        int32_t n_records = (int32_t)chunk.size();
        out.write((char*)&n_records, 4);
        for (int32_t r = 0; r < n_records; ++r) {
            const float* row = logits.data() + (size_t)r * vocab_size;
            int32_t best = 0; float best_v = row[0];
            for (size_t v = 1; v < vocab_size; ++v) {
                if (row[v] > best_v) { best_v = row[v]; best = (int32_t)v; }
            }
            out.write((char*)(fn.data() + (size_t)r * hidden),
                      (std::streamsize)(hidden * sizeof(float)));
            out.write((char*)&best, 4);
        }
        total_records += n_records;
        if (ci % 20 == 0) std::cerr << "chunk " << ci << "/" << n_chunks << "\n";
    }

    out.close();
    std::cerr << "total records: " << total_records << " -> " << out_path << "\n";
    return 0;
}
