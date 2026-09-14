// Closed-form draft head probe — scoring stage.
//
// Loads the model (needed only for model.get_output_weight()), reads a
// binary file of ridge-regression predictions produced offline in numpy
// (from draft_head_capture.cpp's dump), applies the model's OWN output
// weight matmul (ggml_mul_mat, quantized/bf16-correct — no numpy dequant
// risk) to each predicted vector, argmaxes, and reports accuracy against
// the true next-next / next-next-next token. This is the "have the C++
// harness itself apply the model's own output head" route from
// docs/note-closed-form-draft-head.md (chosen over dumping the full
// [vocab x hidden] weight matrix — fewer moving parts, no giant file, no
// separate dequant code path to get right).
//
// Input binary format (written by fit_draft_heads.py):
//   int32 hidden
//   int32 vocab_size (sanity check against the loaded model)
//   int32 N   (held-out sample count, same N for W and W2 blocks)
//   block 1 (p2): N * (hidden float32 predicted-h_{t+1} + int32 true x_{t+2})
//   block 2 (p3): N * (hidden float32 predicted-h_{t+2} + int32 true x_{t+3})
//
//   QWEN36_MODEL_PATH=models/Qwen3.5-0.8B-BF16.gguf ./bin/draft-head-score \
//       <input.bin>
//
// argv[1] is the ONE exception to the no-argv-parsing convention elsewhere
// in tests/perf/ — this stage has no sensible hardcoded default because its
// input is produced by a prior numpy step. Model path still resolves only
// from QWEN36_MODEL_PATH, same as draft_head_capture.cpp.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ggml.h"
#include "engine/model.h"
#include "../../src/models/model_registry.h"

namespace {

int argmax_col(const float* data, size_t vocab, size_t col) {
    const float* row = data + col * vocab;
    size_t best = 0;
    float best_v = row[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (row[i] > best_v) { best_v = row[i]; best = i; }
    }
    return (int)best;
}

// Runs [hidden, N] input through ggml_mul_mat(output_weight, input) on the
// model's own scheduler/backend and returns [vocab, N] logits, row-major by
// column (ggml layout: fastest-varying axis is vocab).
std::vector<float> apply_output_head(Model& model, const std::vector<float>& input,
                                      size_t hidden, size_t n, size_t vocab) {
    ggml_tensor* weight = model.get_output_weight();
    // Tied-embedding models (e.g. Qwen3.5-0.8B) carry no separate output.weight —
    // the output head IS token_embd.weight. Same convention the recipes use.
    if (!weight) weight = model.get_token_embedding_weight();
    if (!weight) throw std::runtime_error("model has neither output_weight nor token_embd");

    size_t buf_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_init_params params{buf_size, nullptr, true};
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* gf = ggml_new_graph(ctx);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)hidden, (int64_t)n);
    ggml_set_input(inp);
    ggml_set_name(inp, "score_input");

    ggml_tensor* logits = ggml_mul_mat(ctx, weight, inp);
    ggml_set_output(logits);
    ggml_set_name(logits, "score_logits");
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_t sched = model.get_scheduler();
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(inp, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> out(vocab * n);
    ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: draft-head-score <input.bin> (produced by fit_draft_heads.py)\n";
        return 1;
    }
    const char* env = std::getenv("QWEN36_MODEL_PATH");
    std::string path = env ? env : "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";

    register_builtin_models();
    std::cerr << "Loading " << path << " ...\n";
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    std::cerr << "architecture=" << meta.architecture
              << " embedding_length=" << meta.embedding_length
              << " vocab_size=" << meta.vocab_size << "\n";

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 1; }
    int32_t hidden, vocab_check, n;
    in.read((char*)&hidden, 4);
    in.read((char*)&vocab_check, 4);
    in.read((char*)&n, 4);
    if ((size_t)hidden != meta.embedding_length) {
        std::cerr << "hidden mismatch: file=" << hidden
                  << " model=" << meta.embedding_length << "\n";
        return 1;
    }
    if ((size_t)vocab_check != meta.vocab_size) {
        std::cerr << "vocab mismatch: file=" << vocab_check
                  << " model=" << meta.vocab_size << "\n";
        return 1;
    }
    const size_t vocab = meta.vocab_size;

    auto read_block = [&](std::vector<float>* preds, std::vector<int32_t>* labels) {
        preds->resize((size_t)hidden * n);
        labels->resize(n);
        for (int i = 0; i < n; ++i) {
            in.read((char*)(preds->data() + (size_t)i * hidden), hidden * sizeof(float));
            in.read((char*)&(*labels)[i], 4);
        }
    };

    std::vector<float> preds2, preds3;
    std::vector<int32_t> labels2, labels3;
    read_block(&preds2, &labels2);
    read_block(&preds3, &labels3);
    if (!in) { std::cerr << "short read on " << argv[1] << "\n"; return 1; }

    std::vector<float> logits2 = apply_output_head(model, preds2, hidden, n, vocab);
    std::vector<float> logits3 = apply_output_head(model, preds3, hidden, n, vocab);

    int correct2 = 0, correct3 = 0;
    for (int i = 0; i < n; ++i) {
        if (argmax_col(logits2.data(), vocab, i) == labels2[i]) correct2++;
        if (argmax_col(logits3.data(), vocab, i) == labels3[i]) correct3++;
    }
    double p2 = (double)correct2 / n;
    double p3 = (double)correct3 / n;
    std::cout << "N=" << n << " p2=" << p2 << " (" << correct2 << "/" << n << ")"
              << " p3=" << p3 << " (" << correct3 << "/" << n << ")\n";
    return 0;
}
