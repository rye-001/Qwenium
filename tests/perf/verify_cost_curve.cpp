// Verify-pass cost curve — is a width-K speculative-decoding verify (a
// K-token prefill on ONE slot) flat or rising in K?
//
// Speculative decoding drafts K tokens then verifies them in one causal
// prefill pass (src/sampling/speculative.h: verify() feeds run_prefill).
// This harness times run_prefill directly at K = 1, 2, 4, 8 tokens on a
// single slot, warm model, and reports each as a multiple of the B=1
// decode-step baseline (measured elsewhere: 13.36ms qwen35, 32.61ms
// qwen35moe).
//
//   QWEN36_MODEL_PATH=models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf ./bin/verify-cost-curve
//
// (No argv parsing — model path resolves ONLY from QWEN36_MODEL_PATH, else
// the qwen35moe default below. This mirrors decode_breakdown.cpp/batch_scaling.cpp.)

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

int main() {
    const char* env = std::getenv("QWEN36_MODEL_PATH");
    std::string path = env ? env : "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";
    const uint32_t CTX = 1024;
    const int WARMUP = 3, TIMED = 20;

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
    (void)tok;

    std::vector<int> ks = {1, 2, 4, 8};
    std::cerr << "\nK\tmean_ms\tmin_ms\tmax_ms\n";
    for (int K : ks) {
        std::vector<double> times;
        for (int iter = 0; iter < WARMUP + TIMED; ++iter) {
            fp->clear_slot(0);
            fp->set_cache_pos(0, 0);
            std::vector<int32_t> tokens;
            for (int i = 0; i < K; ++i) tokens.push_back((int32_t)((i * 7 + 3) % vsz));

            auto t0 = Clock::now();
            auto logits = fp->run_prefill(tokens, 0, 0, sched);
            double dt = ms_since(t0);
            (void)logits;
            if (iter >= WARMUP) times.push_back(dt);
        }
        double sum = 0, mn = times[0], mx = times[0];
        for (double t : times) { sum += t; mn = std::min(mn, t); mx = std::max(mx, t); }
        double mean = sum / times.size();
        std::cout << K << "\t" << mean << "\t" << mn << "\t" << mx << std::endl;
    }
    return 0;
}
