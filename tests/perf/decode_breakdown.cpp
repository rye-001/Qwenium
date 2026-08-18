// Decode-step phase decomposition — where does the ~20ms fixed intercept live?
//
// The batch-scaling probe fit ms/step ≈ 20 + 25.6·B. The ~20ms fixed part is
// paid every single-user token (44% of B=1 latency) and is what graph-reuse
// would attack. But only if it's CPU-side REPLANNING (rebuild the identical
// graph + re-plan scratch every step). If instead it lives inside GPU compute
// (launch bubbles between hundreds of tiny DeltaNet kernels), reuse won't help
// and the answer is fusion. This probe times each phase separately to decide.
//
// Phases (one B=1 decode step, the single-user case that matters):
//   build   — build_decoding_graph: re-derive ~2400 ggml nodes (CPU)
//   alloc   — sched_reset + alloc_graph: re-plan scratch memory (CPU)
//   set     — set_decode_inputs: fill typed input slots + upload (CPU→GPU)
//   compute — sched_graph_compute: encode + dispatch + GPU run + sync
//   read    — get_output_logits: readback logits (GPU→CPU sync)
//
//   QWEN36_MODEL_PATH=... ./bin/decode-breakdown

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../../src/core/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/loader/tokenizer.h"

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main() {
    const char* env = std::getenv("QWEN36_MODEL_PATH");
    std::string path = env ? env : "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";
    const uint32_t CTX = 1024;
    const int WARMUP = 8, TIMED = 40;

    register_builtin_models();
    std::cerr << "Loading " << path << " ...\n";
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    auto fp = create_forward_pass(model, &meta, CTX, 1);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();

    std::vector<int32_t> prompt = tok->encode("Summarize the following order:");
    fp->run_prefill(prompt, 0, 0, sched);
    const int32_t vsz = (int32_t)meta.vocab_size;

    std::vector<double> t_build, t_alloc, t_set, t_compute, t_read;
    const int base_pos = (int)fp->get_cache_pos(0);

    for (int step = 0; step < WARMUP + TIMED; ++step) {
        const std::vector<int32_t>  tokens    = {(int32_t)((1000 + step * 7) % vsz)};
        const std::vector<uint32_t> slots     = {0};
        const std::vector<int32_t>  positions = {(int)fp->get_cache_pos(0)};

        auto t0 = Clock::now();
        ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);
        double d_build = ms_since(t0);

        auto t1 = Clock::now();
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        double d_alloc = ms_since(t1);

        auto t2 = Clock::now();
        fp->set_decode_inputs(gf, tokens, slots, positions);
        double d_set = ms_since(t2);

        auto t3 = Clock::now();
        ggml_backend_sched_graph_compute(sched, gf);
        double d_compute = ms_since(t3);

        auto t4 = Clock::now();
        volatile std::vector<float> logits = fp->get_output_logits(gf);
        double d_read = ms_since(t4);

        fp->advance_cache(1, 0);

        if (step >= WARMUP) {
            t_build.push_back(d_build);   t_alloc.push_back(d_alloc);
            t_set.push_back(d_set);       t_compute.push_back(d_compute);
            t_read.push_back(d_read);
        }
        (void)base_pos;
    }

    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    double b = med(t_build), a = med(t_alloc), s = med(t_set),
           c = med(t_compute), r = med(t_read);
    double total = b + a + s + c + r;
    double cpu_fixed = b + a + s + r;  // everything graph-reuse could remove

    std::printf("\n  phase                 median ms    %% of step\n");
    std::printf("  --------------------- ---------    ---------\n");
    std::printf("  build_decoding_graph  %8.2f    %6.1f%%\n", b, 100*b/total);
    std::printf("  sched alloc (replan)  %8.2f    %6.1f%%\n", a, 100*a/total);
    std::printf("  set_decode_inputs     %8.2f    %6.1f%%\n", s, 100*s/total);
    std::printf("  graph_compute (GPU)   %8.2f    %6.1f%%\n", c, 100*c/total);
    std::printf("  logits readback       %8.2f    %6.1f%%\n", r, 100*r/total);
    std::printf("  --------------------- ---------    ---------\n");
    std::printf("  TOTAL / step          %8.2f    100.0%%\n", total);
    std::printf("\n  CPU-side fixed (build+alloc+set+read) = %.2f ms (%.1f%%)\n",
                cpu_fixed, 100*cpu_fixed/total);
    std::printf("  GPU compute                           = %.2f ms (%.1f%%)\n",
                c, 100*c/total);
    std::printf("\n  Verdict: if CPU-fixed ~ the 20ms intercept, graph-reuse wins.\n"
                "           if GPU compute holds the intercept, it's launch bubbles → fusion.\n");
    return 0;
}
