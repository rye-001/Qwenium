// Batch-scaling probe — the falsifiable test for the "empty seats" hypothesis.
//
// Claim: on the qwen36 hybrid, one decode step is dominated by kernel-launch
// overhead (DeltaNet) and weight reads, NOT per-sequence math — so decoding B
// sequences in one batched step costs far less than B× one. If true, "N futures
// per step, priced as one" is a real primitive (best-of-N, tree speculation,
// ghost-slot prefill). If batch-8 ≈ 8× batch-1, the hypothesis is dead.
//
// Method: prefill one prompt into slot 0, clone it O(1) into slots 1..B-1
// (clone_slot), then time D batched decode steps at B = 1,2,4,8,10. Each lane
// is fed a DIFFERENT token id per step so MoE expert routing diverges across
// lanes (the realistic, weight-read-heavy case — identical tokens would route
// every lane to the same experts and flatter the result).
//
// Reports, per B: ms/step (all B lanes) and ms/step ÷ B (amortized per-token).
// The headline is ms_step(8) / ms_step(1).
//
//   QWEN36_MODEL_PATH=models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf ./bin/batch-scaling

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/loader/tokenizer.h"

using Clock = std::chrono::steady_clock;

int main() {
    const char* env = std::getenv("QWEN36_MODEL_PATH");
    std::string path = env ? env : "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";

    const uint32_t MAXB    = 10;   // largest batch tested
    const uint32_t CTX     = 1024; // KV ctx per slot (enough for prompt + decode)
    const int      WARMUP  = 6;    // steps to absorb galloc reserve/first-run cost
    const int      TIMED   = 24;   // timed steps
    const std::vector<uint32_t> batches = {1, 2, 4, 8, 10};

    register_builtin_models();
    std::cerr << "Loading " << path << " ...\n";
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    if (meta.architecture != "qwen35moe") {
        std::cerr << "batch-scaling: expected qwen35moe, got " << meta.architecture << "\n";
        return 1;
    }
    auto fp = create_forward_pass(model, &meta, CTX, MAXB);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();

    // A short prompt — decode cost is what we measure, prefill is one-off.
    std::vector<int32_t> prompt = tok->encode("Summarize the following order:");
    const auto vsz = (int32_t)meta.vocab_size;

    std::printf("\n batch |  ms/step | ms/step/lane | vs B=1 (step) | speedup vs serial\n");
    std::printf("-------+----------+--------------+---------------+------------------\n");

    double ms_step_b1 = 0.0;
    for (uint32_t B : batches) {
        // Fresh state: clear every slot, prefill slot 0, clone into 1..B-1.
        for (uint32_t s = 0; s < MAXB; ++s) { fp->clear_slot(s); fp->set_cache_pos(0, s); }
        fp->run_prefill(prompt, 0, 0, sched);
        const int P = (int)fp->get_cache_pos(0);
        for (uint32_t s = 1; s < B; ++s) fp->clone_slot(0, s, (uint32_t)P);

        std::vector<uint32_t> slots(B);
        for (uint32_t s = 0; s < B; ++s) slots[s] = s;

        auto run_step = [&](int step) {
            std::vector<int32_t> tokens(B);
            std::vector<int32_t> positions(B);
            const int pos = P + step;
            // QINF_BATCH_IDENTICAL=1 feeds every lane the SAME token → maximal
            // MoE expert-read sharing (best case). Default: distinct per lane →
            // maximal routing diversity (worst case). The two bracket real
            // best-of-N, which shares a context and so lands in between.
            static const bool identical = std::getenv("QINF_BATCH_IDENTICAL");
            for (uint32_t s = 0; s < B; ++s) {
                tokens[s]    = identical ? (int32_t)((1000 + step * 7) % vsz)
                                         : (int32_t)((1000 + step * 7 + s * 131) % vsz);
                positions[s] = pos;
            }
            ggml_backend_sched_reset(sched);
            ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);
            ggml_backend_sched_alloc_graph(sched, gf);
            fp->set_decode_inputs(gf, tokens, slots, positions);
            ggml_backend_sched_graph_compute(sched, gf);
            for (uint32_t s = 0; s < B; ++s) fp->advance_cache(1, s);
        };

        for (int i = 0; i < WARMUP; ++i) run_step(i);
        auto t0 = Clock::now();
        for (int i = 0; i < TIMED; ++i) run_step(WARMUP + i);
        auto t1 = Clock::now();

        double ms_step = std::chrono::duration<double, std::milli>(t1 - t0).count() / TIMED;
        if (B == 1) ms_step_b1 = ms_step;
        std::printf(" %5u | %8.2f | %12.2f | %13.2fx | %15.2fx\n",
                    B, ms_step, ms_step / B, ms_step / ms_step_b1,
                    (ms_step_b1 * B) / ms_step);
        std::fflush(stdout);
    }
    std::printf("\nHeadline: ms/step(B) grows as ~1x if seats are free, ~Bx if full.\n"
                "'speedup vs serial' = tokens/wall you'd get running B futures batched\n"
                "instead of one at a time.\n");
    return 0;
}
