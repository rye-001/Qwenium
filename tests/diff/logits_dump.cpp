// logits_dump.cpp — deterministic byte-identical differential harness.
//
// Uses ONLY APIs stable across pre-refactor `main` and the typed-graph-inputs
// branch (Model load, create_forward_pass, run_prefill, decode_step,
// GreedySampler). Compile the SAME file in both trees, run on the same model
// with the same fixed token prompt, and `cmp` the output files: any divergence
// in input-tensor population perturbs logits and fails the gate.
//
// Usage: logits_dump <model.gguf> <kv_quant_bits> <out.bin>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include "ggml-backend.h"
#include "../../src/core/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/core/decode_step.h"
#include "../../src/sampling/sampling.h"
#include "../../src/loader/tokenizer.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <kv_bits> <out.bin>\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const int kv_bits = std::atoi(argv[2]);
    const std::string out_path = argv[3];

    ggml_backend_load_all();
    register_builtin_models();

    Model model;
    model.load_metadata(model_path);
    model.load_tensors();

    const auto& meta = model.get_metadata();
    const uint32_t vocab_size = meta.vocab_size;

    auto fp = create_forward_pass(model, &meta, 4096, 1, kv_bits);
    ggml_backend_sched_t sched = model.get_scheduler();

    // Fixed deterministic prompt (token ids only — no tokenizer variance).
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 24; ++i) tokens.push_back((i * 7 + 3) % 1000);

    std::vector<float> prefill = fp->run_prefill(tokens, 0, 0, sched);

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::perror("fopen"); return 1; }

    // 1. Full prefill logits buffer (exercises Tokens/Positions/AttnMask —
    //    incl. Gemma per-layer interleaved local/global windows).
    uint64_t n = prefill.size();
    std::fwrite(&n, sizeof(n), 1, f);
    std::fwrite(prefill.data(), sizeof(float), prefill.size(), f);

    // 2. Greedy decode continuation (exercises kq_mask_b + gather_indices via
    //    decode_step). Recipes whose decode graph is unimplemented throw —
    //    record a sentinel so the comparison is still well-defined.
    qwenium::GreedySampler sampler;
    const auto vocab = model.get_tokenizer()->get_vocabulary();
    sampler.build_token_trie(vocab);
    std::vector<int32_t> history = tokens;
    std::vector<int32_t> gen;
    bool decode_ok = true;
    try {
        std::vector<float> last(prefill.end() - vocab_size, prefill.end());
        int32_t tok = static_cast<int32_t>(sampler.sample(last, history, vocab));
        for (int step = 0; step < 8; ++step) {
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
    std::fprintf(stderr, "wrote %s: %llu prefill floats, decode_ok=%d, %llu gen\n",
                 out_path.c_str(), (unsigned long long)n, (int)ok,
                 (unsigned long long)g);
    return 0;
}
