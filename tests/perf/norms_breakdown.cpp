// Phase 4 investigation — RMSNorm cost at decode shape.
//
// Why: a Qwen 3.6 forward pass has a lot of RMSNorms (≥2 per layer × 40 layers
// + final pre-LM-head norm + Q/K norms inside attention layers). Each is a
// tiny tensor-shape op on Metal, so launch / barrier overhead can dominate the
// arithmetic. We need a per-norm μs figure to multiply against site count and
// rank against attention/DeltaNet/MoE.
//
// Method: cumulative sub-graphs with 0, 1, 2, 4, 8, 16 chained RMSNorms at
// shape [n_embd, 1]. The 1→2 delta is the marginal cost of one additional
// norm; the 0→1 delta absorbs scheduler launch overhead. Stage-cost
// methodology mirrors moe_breakdown.cpp / attention_breakdown.cpp.
//
// No model weights required — this is a pure-shape benchmark; the only
// dependency on the model is to learn n_embd. We still load the model so the
// scheduler / Metal backend match the production environment exactly.
//
// Usage:
//   QWEN36_MODEL_PATH=/path/to/qwen36.gguf \
//   ./bin/norms-breakdown > tests/fixtures/norms_breakdown.json

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include <nlohmann/json.hpp>

#include "ggml.h"
#include "ggml-backend.h"

#include "engine/model.h"
#include "../../src/models/model_registry.h"

#include "breakdown_harness.h"

using json = nlohmann::json;
using breakdown::BuiltGraph;
using breakdown::make_graph;
using breakdown::time_subgraph;

static BuiltGraph build_chain(int n_embd, int n_norms, float rms_norm_eps) {
    BuiltGraph bg = make_graph(/*meta_bytes=*/16 * 1024 * 1024, /*max_nodes=*/4096);

    bg.input = ggml_new_tensor_2d(bg.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(bg.input);
    ggml_set_name(bg.input, "x");

    ggml_tensor* cur = bg.input;
    for (int i = 0; i < n_norms; ++i) {
        cur = ggml_rms_norm(bg.ctx, cur, rms_norm_eps);
    }
    bg.output = cur;
    ggml_build_forward_expand(bg.gf, cur);
    return bg;
}

int main(int /*argc*/, char** /*argv*/) {
    const char* qwen36_path = std::getenv("QWEN36_MODEL_PATH");
    const char* warmup_env  = std::getenv("NORMS_WARMUP");
    const char* iters_env   = std::getenv("NORMS_ITERS");

    const int warmup = warmup_env ? std::atoi(warmup_env) : 10;
    const int iters  = iters_env  ? std::atoi(iters_env)  : 200;

    if (!qwen36_path) {
        std::cerr << "norms_breakdown: QWEN36_MODEL_PATH not set\n";
        return 1;
    }

    register_builtin_models();
    Model model;
    model.load_metadata(qwen36_path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    const int n_embd = static_cast<int>(meta.embedding_length);
    const float eps  = meta.rms_norm_eps;

    ggml_backend_sched_t sched = model.get_scheduler();

    std::vector<int> chain_lengths = { 0, 1, 2, 4, 8, 16 };

    std::vector<float> input_data(static_cast<size_t>(n_embd));
    std::mt19937 rng(0xCA11AB1Eu);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : input_data) v = dist(rng);

    json points = json::array();
    double prev_ms = 0.0;
    int    prev_n  = 0;

    for (int n : chain_lengths) {
        BuiltGraph bg = build_chain(n_embd, n, eps);
        ggml_tensor* x = ggml_graph_get_tensor(bg.gf, "x");
        auto upload = [x, &input_data]() {
            if (x) ggml_backend_tensor_set(x, input_data.data(), 0,
                                            input_data.size() * sizeof(float));
        };
        double ms = time_subgraph(bg, sched, upload, warmup, iters,
                                  /*label=*/("rms_chain_" + std::to_string(n)).c_str());
        ggml_free(bg.ctx);

        const double per_norm_us = (n > prev_n)
            ? ((ms - prev_ms) / static_cast<double>(n - prev_n)) * 1000.0
            : 0.0;

        std::cerr << "  chain=" << n << ": cum " << ms
                  << " ms, marginal per-norm " << per_norm_us << " μs\n";
        points.push_back({
            {"n_norms",         n},
            {"cumulative_ms",   ms},
            {"per_norm_us",     per_norm_us},
        });
        prev_ms = ms;
        prev_n  = n;
    }

    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));

    json out = {
        {"_note",
            "Phase 4 investigation: RMSNorm marginal cost at decode shape "
            "[n_embd, 1]. Chains of 0/1/2/4/8/16 norms — per-norm μs is the "
            "delta divided by chain-length increment. 0→1 absorbs scheduler "
            "launch overhead; 1→2, 2→4, 4→8 deltas are the load-bearing values."},
        {"generated_at",  ts_buf},
        {"model_path",    qwen36_path},
        {"architecture",  meta.architecture},
        {"n_embd",        n_embd},
        {"rms_norm_eps",  eps},
        {"warmup_iters",  warmup},
        {"timed_iters",   iters},
        {"chains",        points},
    };
    std::cout << out.dump(2) << "\n";
    return 0;
}
