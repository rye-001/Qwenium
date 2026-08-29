// Alloc-reuse confirmation — does stabilizing the decode graph shape recover
// the ~12ms galloc replanning that decode_breakdown localized?
//
// Hypothesis: galloc re-plans every step because the KV view bakes n_kv into
// the shape and n_kv grows by one each step, so galloc never sees an identical
// graph twice. If true, holding the shape constant should make alloc drop to
// near-zero after the first call (galloc's reuse fast-path). If alloc stays
// expensive even at a constant shape, the wall is the sched_reset re-plan
// itself, not the shape — and bucketing alone won't help.
//
// A/B, with sched_reset and alloc_graph timed SEPARATELY:
//   GROW  — advance the cache each step (real decode): n_kv grows, shape churns.
//   FIXED — rebuild at the SAME position each step: n_kv constant, shape stable.
//
//   QWEN36_MODEL_PATH=... ./bin/alloc-reuse

#include <algorithm>
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
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
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
    const int32_t vsz = (int32_t)meta.vocab_size;

    fp->run_prefill(tok->encode("Summarize the following order:"), 0, 0, sched);
    const int P = (int)fp->get_cache_pos(0);

    // mode 0 = GROW (advance each step), mode 1 = FIXED (rebuild at pos P).
    auto measure = [&](bool grow, std::vector<double>& reset_ms,
                       std::vector<double>& alloc_ms) {
        fp->set_cache_pos((uint32_t)P, 0);
        for (int step = 0; step < WARMUP + TIMED; ++step) {
            const int pos = grow ? (int)fp->get_cache_pos(0) : P;
            const std::vector<int32_t>  tokens    = {(int32_t)((1000 + step * 7) % vsz)};
            const std::vector<uint32_t> slots     = {0};
            const std::vector<int32_t>  positions = {pos};

            ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);

            auto tr = Clock::now();
            ggml_backend_sched_reset(sched);
            double d_reset = ms_since(tr);

            auto ta = Clock::now();
            ggml_backend_sched_alloc_graph(sched, gf);
            double d_alloc = ms_since(ta);

            fp->set_decode_inputs(gf, tokens, slots, positions);
            ggml_backend_sched_graph_compute(sched, gf);
            if (grow) fp->advance_cache(1, 0);

            if (step >= WARMUP) { reset_ms.push_back(d_reset); alloc_ms.push_back(d_alloc); }
        }
    };

    std::vector<double> g_reset, g_alloc, f_reset, f_alloc;
    measure(/*grow=*/true,  g_reset, g_alloc);
    measure(/*grow=*/false, f_reset, f_alloc);

    // PERSIST: the actual fix. Build+alloc the decode graph ONCE at a fixed
    // shape, then reuse it — only refill inputs + recompute each step, never
    // re-alloc. This is the ceiling of "stop replanning". (Fixed position, so
    // KV is overwritten in place — fine for timing; a real impl buckets n_kv so
    // the one allocation stays valid for the whole bucket, mask gating the tail.)
    std::vector<double> persist_step, grow_step;
    {
        fp->set_cache_pos((uint32_t)P, 0);
        const std::vector<int32_t>  tokens0   = {(int32_t)(1000 % vsz)};
        const std::vector<uint32_t> slots     = {0};
        const std::vector<int32_t>  positions = {P};
        ggml_backend_sched_reset(sched);
        ggml_cgraph* gf = fp->build_decoding_graph(tokens0, slots, positions);
        ggml_backend_sched_alloc_graph(sched, gf);      // the ONE allocation
        for (int step = 0; step < WARMUP + TIMED; ++step) {
            const std::vector<int32_t> tokens = {(int32_t)((1000 + step * 7) % vsz)};
            auto t0 = Clock::now();
            fp->set_decode_inputs(gf, tokens, slots, positions);
            ggml_backend_sched_graph_compute(sched, gf);
            double d = ms_since(t0);
            if (step >= WARMUP) persist_step.push_back(d);
        }
    }
    // GROW full-step (build+reset+alloc+set+compute) for the honest comparison.
    {
        fp->set_cache_pos((uint32_t)P, 0);
        for (int step = 0; step < WARMUP + TIMED; ++step) {
            const std::vector<int32_t>  tokens    = {(int32_t)((1000 + step * 7) % vsz)};
            const std::vector<uint32_t> slots     = {0};
            const std::vector<int32_t>  positions = {(int)fp->get_cache_pos(0)};
            auto t0 = Clock::now();
            ggml_backend_sched_reset(sched);
            ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);
            ggml_backend_sched_alloc_graph(sched, gf);
            fp->set_decode_inputs(gf, tokens, slots, positions);
            ggml_backend_sched_graph_compute(sched, gf);
            double d = ms_since(t0);
            fp->advance_cache(1, 0);
            if (step >= WARMUP) grow_step.push_back(d);
        }
    }

    std::printf("\n  mode   | sched_reset ms | alloc_graph ms\n");
    std::printf("  -------+----------------+----------------\n");
    std::printf("  GROW   | %14.3f | %14.3f\n", med(g_reset), med(g_alloc));
    std::printf("  FIXED  | %14.3f | %14.3f\n", med(f_reset), med(f_alloc));
    std::printf("\n  Shape is NOT the lever: FIXED alloc (%.1f) ~ GROW alloc (%.1f).\n"
                "  alloc_graph re-plans ~%.0fms every call regardless of shape.\n",
                med(f_alloc), med(g_alloc), med(g_alloc));

    double gs = med(grow_step), ps = med(persist_step);
    std::printf("\n  full decode step (build+reset+alloc+set+compute) = %.2f ms  (%.1f tok/s)\n",
                gs, 1000.0 / gs);
    std::printf("  PERSIST step (reuse one allocation, set+compute)  = %.2f ms  (%.1f tok/s)\n",
                ps, 1000.0 / ps);
    std::printf("  => reusing the allocation recovers %.2f ms/step = %.2fx decode.\n",
                gs - ps, gs / ps);
    return 0;
}
