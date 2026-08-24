// Phase 4 investigation — total per-layer DeltaNet cost on the decode path.
//
// Why: Qwen 3.6 is a hybrid — 30 of its 40 layers are GatedDeltaNet. If those
// layers individually cost more than the attention or MoE layers, DeltaNet is
// the right Phase 4 PR-1 target rather than attention or MoE dispatch fusion.
// We need a per-layer μs figure, even if coarse, to multiply against 30 and
// rank against the other measurements.
//
// Method: a coarse two-point measurement. Stage 0 builds an empty
// scheduled-graph that just uploads the residual input — this isolates the
// per-iter scheduler / launch overhead floor. Stage 1 runs one full
// DeltaNetLayer::build(Phase::Decode) on a single slot. Stage cost = Stage1 −
// Stage0. No internal sub-stage split: DeltaNet's recurrent + conv + gating
// path is a single conceptual block; if total cost lands large enough to be
// the main 3× target, a sub-stage breakdown is the obvious follow-up.
//
// Usage:
//   QWEN36_MODEL_PATH=/path/to/qwen36.gguf \
//   ./bin/deltanet-breakdown > tests/fixtures/deltanet_breakdown.json
//
// Env knobs: DN_WARMUP (default 10), DN_ITERS (default 200),
// DN_LAYER (default 0 — first DeltaNet layer in the recipe's layout).
//
// Keys are read under the model's OWN architecture prefix, so this runs on any
// DeltaNet-carrying recipe (qwen35 and qwen35moe), not just Qwen 3.6.

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

#include "../../src/core/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/state/deltanet_state.h"
#include "../../src/layers/deltanet.h"
#include "../../src/layers/layer.h"

#include "breakdown_harness.h"

using json = nlohmann::json;
using breakdown::BuiltGraph;
using breakdown::make_graph;
using breakdown::time_subgraph;

struct DnConfig {
    uint32_t d_inner;
    uint32_t num_v_heads;
    uint32_t num_k_heads;
    uint32_t head_v_dim;
    uint32_t head_k_dim;
    uint32_t conv_channels;
    uint32_t conv_kernel;
    float    rms_norm_eps;
};

static DnConfig read_dn_config(const ModelMetadata& meta) {
    DnConfig c{};
    // Read under the loaded model's architecture prefix rather than a hardcoded
    // one -- qwen35 and qwen35moe both carry DeltaNet and must both be probeable.
    const std::string A = meta.architecture;
    c.d_inner       = meta.raw_kv.get_uint32(A + ".ssm.inner_size");
    c.num_v_heads   = meta.raw_kv.get_uint32(A + ".ssm.time_step_rank");
    c.num_k_heads   = meta.raw_kv.get_uint32(A + ".ssm.group_count");
    c.head_v_dim    = c.d_inner / c.num_v_heads;
    c.head_k_dim    = meta.raw_kv.get_uint32(A + ".ssm.state_size");
    c.conv_channels = c.d_inner + 2 * c.num_k_heads * c.head_k_dim;
    c.conv_kernel   = meta.raw_kv.get_uint32(A + ".ssm.conv_kernel");
    c.rms_norm_eps  = meta.rms_norm_eps;
    return c;
}

int main(int /*argc*/, char** /*argv*/) {
    const char* qwen36_path = std::getenv("QWEN36_MODEL_PATH");
    const char* warmup_env  = std::getenv("DN_WARMUP");
    const char* iters_env   = std::getenv("DN_ITERS");
    const char* layer_env   = std::getenv("DN_LAYER");

    const int warmup = warmup_env ? std::atoi(warmup_env) : 10;
    const int iters  = iters_env  ? std::atoi(iters_env)  : 200;
    const int layer  = layer_env  ? std::atoi(layer_env)  : 0;

    if (!qwen36_path) {
        std::cerr << "deltanet_breakdown: QWEN36_MODEL_PATH not set\n";
        return 1;
    }

    register_builtin_models();
    Model model;
    model.load_metadata(qwen36_path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    const auto& blk  = model.get_block(layer);

    if (!blk.attn_qkv_weight || !blk.ssm_a || !blk.ssm_conv1d_weight) {
        std::cerr << "deltanet_breakdown: layer " << layer
                  << " is not a DeltaNet layer — try DN_LAYER=0,1,2,4,...\n";
        return 1;
    }

    const DnConfig cfg  = read_dn_config(meta);
    const int      n_embd = static_cast<int>(meta.embedding_length);

    ggml_backend_t backend = const_cast<ggml_backend_t>(model.get_backend_metal());
    if (!backend) backend = const_cast<ggml_backend_t>(model.get_backend_cpu());

    // Standalone DeltaNet state with one logical layer, one slot.
    DeltaNetState::Hparams sh{
        /*n_dn_layers=*/1, /*n_slots=*/1,
        cfg.head_v_dim, cfg.head_k_dim, cfg.num_v_heads,
        cfg.conv_channels, cfg.conv_kernel,
        backend
    };
    DeltaNetState dn_state(sh);

    DeltaNetLayer dn_layer(
        blk.attn_qkv_weight,
        blk.attn_gate_weight,
        blk.ssm_beta_weight,
        blk.ssm_alpha_weight,
        blk.ssm_dt_bias,
        blk.ssm_a,
        blk.ssm_conv1d_weight,
        blk.ssm_norm_weight,
        blk.ssm_out_weight,
        &dn_state,
        DeltaNetLayer::Hparams{
            n_embd,
            static_cast<int>(cfg.head_v_dim * cfg.num_v_heads),  // d_inner
            static_cast<int>(cfg.head_k_dim),
            static_cast<int>(cfg.num_k_heads),
            static_cast<int>(cfg.num_v_heads),
            static_cast<int>(cfg.head_v_dim),
            static_cast<int>(cfg.conv_channels),
            static_cast<int>(cfg.conv_kernel),
            cfg.rms_norm_eps
        });

    ggml_backend_sched_t sched = model.get_scheduler();

    std::vector<float> input_data(static_cast<size_t>(n_embd));
    std::mt19937 rng(0xCA11AB1Eu);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : input_data) v = dist(rng);

    auto build_baseline = [&]() {
        BuiltGraph bg = make_graph();
        bg.input = ggml_new_tensor_2d(bg.ctx, GGML_TYPE_F32, n_embd, 1);
        ggml_set_input(bg.input);
        ggml_set_name(bg.input, "x");
        // Cheap no-op-ish output: identity-shape add against the input.
        // ggml_dup keeps the graph non-empty so reserve() does not balk.
        bg.output = ggml_dup(bg.ctx, bg.input);
        ggml_build_forward_expand(bg.gf, bg.output);
        return bg;
    };

    auto build_full = [&]() {
        BuiltGraph bg = make_graph();
        bg.input = ggml_new_tensor_2d(bg.ctx, GGML_TYPE_F32, n_embd, 1);
        ggml_set_input(bg.input);
        ggml_set_name(bg.input, "x");

        DeltaNetLayer::DecodeArgs da{ /*slots=*/{0u} };
        DeltaNetLayer::PrefillArgs pa_unused{1, 0};
        bg.output = dn_layer.build(bg.ctx, bg.gf, bg.input,
                                   /*dn_idx=*/0, Phase::Decode, pa_unused, &da);
        ggml_build_forward_expand(bg.gf, bg.output);
        return bg;
    };

    auto run = [&](const char* name, BuiltGraph& bg) {
        ggml_tensor* x = ggml_graph_get_tensor(bg.gf, "x");
        auto upload = [x, &input_data]() {
            if (x) ggml_backend_tensor_set(x, input_data.data(), 0,
                                            input_data.size() * sizeof(float));
        };
        return time_subgraph(bg, sched, upload, warmup, iters, name);
    };

    BuiltGraph bg0 = build_baseline();
    double base_ms = run("baseline_dup", bg0);
    ggml_free(bg0.ctx);

    BuiltGraph bg1 = build_full();
    double full_ms = run("dn_decode_full", bg1);
    ggml_free(bg1.ctx);

    const double layer_ms = full_ms - base_ms;

    std::cerr << "  layer=" << layer
              << " baseline " << base_ms << " ms"
              << ", dn_decode_full " << full_ms << " ms"
              << ", per-layer " << layer_ms << " ms\n";

    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));

    json out = {
        {"_note",
            "Phase 4 investigation: coarse total per-layer DeltaNet decode cost. "
            "Computed as full−baseline to cancel the per-iter scheduler launch "
            "overhead floor. Sub-stage attribution is intentionally not provided; "
            "if DeltaNet ranks as the largest budget chunk, a sub-stage breakdown "
            "is the next step."},
        {"generated_at",   ts_buf},
        {"model_path",     qwen36_path},
        {"architecture",   meta.architecture},
        {"layer_idx",      layer},
        {"n_embd",         n_embd},
        {"d_inner",        cfg.d_inner},
        {"num_v_heads",    cfg.num_v_heads},
        {"num_k_heads",    cfg.num_k_heads},
        {"head_v_dim",     cfg.head_v_dim},
        {"head_k_dim",     cfg.head_k_dim},
        {"conv_channels",  cfg.conv_channels},
        {"conv_kernel",    cfg.conv_kernel},
        {"warmup_iters",   warmup},
        {"timed_iters",    iters},
        {"baseline_ms",    base_ms},
        {"dn_decode_ms",   full_ms},
        {"per_layer_ms",   layer_ms},
    };
    std::cout << out.dump(2) << "\n";
    return 0;
}
