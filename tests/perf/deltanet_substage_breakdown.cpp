// Phase 4 PR-1 — DeltaNet sub-stage breakdown (decode shape, n_tokens=1).
//
// Resolves Q1 in docs/phase4-investigation.md: is the 0.97 ms/layer DeltaNet
// decode cost launch-bound (lots of small ops × per-dispatch overhead) or
// matmul-bound (the [2048×4096] output projection is irreducible)?
//
// Method: cumulative-subgraph timing. Six graphs are built, each terminating
// one stage later than the previous. Stage cost = median(g_N) − median(g_{N-1}).
// Mirrors the moe_breakdown.cpp pattern — see breakdown_harness.h for the
// shared timing harness and caveats.
//
// Stages:
//   qkv_proj      — mul_mat(w_qkv, input) + reshape
//   gate_proj     — + z (w_gate), β (w_beta + sigmoid), α→decay (w_a + softplus·A_log)
//   conv_update   — + conv-window concat + cpy writeback + ssm_conv + silu
//   state_update  — + Q/K l2_norm + repeat_4d + gated_delta_net + recurrent cpy writeback
//   out_norm      — + rms_norm + mul(w_norm) + silu(z) + mul (gated norm)
//   out_proj      — + mul_mat(w_out, …)  ← the [2048×4096] GEMV under suspicion
//
// Verdict criterion (computed in-binary, written to fixture):
//   • out_proj_pct ≥ 40%  → matmul-bound; recommend "no action — baseline preserved".
//   • out_proj_pct < 40%  → launch-bound; recommend a fused-kernel PR.
//
// Usage:
//   QWEN36_MODEL_PATH=/path/to/qwen36.gguf \
//   ./bin/deltanet-substage-breakdown > tests/fixtures/deltanet_substage_breakdown_qwen36.json
//
// Env knobs: DN_WARMUP (default 10), DN_ITERS (default 200), DN_LAYER (default 0).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ggml.h"
#include "ggml-backend.h"

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/state/deltanet_state.h"

#include "breakdown_harness.h"

using json = nlohmann::json;
using breakdown::BuiltGraph;
using breakdown::make_graph;
using breakdown::time_subgraph;

// ── Stages ───────────────────────────────────────────────────────────────────

enum class Stage : int {
    QkvProj     = 0,
    GateProj    = 1,
    ConvUpdate  = 2,
    StateUpdate = 3,
    OutNorm     = 4,
    OutProj     = 5,
};

static const char* stage_name(int s) {
    switch (s) {
        case 0: return "qkv_proj";
        case 1: return "gate_proj";
        case 2: return "conv_update";
        case 3: return "state_update";
        case 4: return "out_norm";
        case 5: return "out_proj";
    }
    return "?";
}

// ── Config + weights handles ─────────────────────────────────────────────────

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

struct DnWeights {
    ggml_tensor* w_qkv;
    ggml_tensor* w_gate;
    ggml_tensor* w_beta;
    ggml_tensor* w_a;
    ggml_tensor* w_dt_bias;
    ggml_tensor* w_a_log;
    ggml_tensor* w_conv;
    ggml_tensor* w_norm;
    ggml_tensor* w_out;
};

static DnConfig read_dn_config(const ModelMetadata& meta) {
    DnConfig c{};
    // Read under the loaded model's architecture prefix -- qwen35 and qwen35moe
    // both carry DeltaNet and must both be probeable.
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

// ── Subgraph construction ────────────────────────────────────────────────────
//
// Mirrors the prefix of build_deltanet_layer in src/layers/deltanet.cpp,
// inserting early-termination points. The recurrent + conv state tensors are
// owned by `dn_state` (its own ggml_context); cross-context references work
// for DeltaNet (the failure mode that blocked attention-breakdown does not
// apply here — see docs/phase4-investigation.md).

static BuiltGraph build_subgraph(
    int               n_embd,
    Stage             stop,
    const DnConfig&   cfg,
    const DnWeights&  W,
    DeltaNetState&    dn_state,
    uint32_t          dn_idx,
    uint32_t          slot_idx)
{
    BuiltGraph bg = make_graph();
    bg.input = ggml_new_tensor_2d(bg.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(bg.input);
    ggml_set_name(bg.input, "x");

    ggml_context* ctx = bg.ctx;
    ggml_cgraph*  gf  = bg.gf;
    ggml_tensor*  cur = bg.input;

    const int64_t n_seq_tokens   = 1;
    const int64_t n_seqs         = 1;
    const int     num_v_heads    = static_cast<int>(cfg.num_v_heads);
    const int     num_k_heads    = static_cast<int>(cfg.num_k_heads);
    const int     head_k_dim     = static_cast<int>(cfg.head_k_dim);
    const int     head_v_dim     = static_cast<int>(cfg.head_v_dim);
    const int     conv_channels  = static_cast<int>(cfg.conv_channels);
    const int     conv_kernel    = static_cast<int>(cfg.conv_kernel);
    const float   rms_norm_eps   = cfg.rms_norm_eps;

    // ── Stage QkvProj ────────────────────────────────────────────────────────
    ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, W.w_qkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    if (stop == Stage::QkvProj) {
        bg.output = qkv_mixed;
        ggml_build_forward_expand(gf, qkv_mixed);
        return bg;
    }

    // ── Stage GateProj — z + β + α→decay ─────────────────────────────────────
    ggml_tensor* z = ggml_mul_mat(ctx, W.w_gate, cur);

    ggml_tensor* beta = ggml_mul_mat(ctx, W.w_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);

    ggml_tensor* alpha = ggml_mul_mat(ctx, W.w_a, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* alpha_biased = ggml_add(ctx, alpha, W.w_dt_bias);
    ggml_tensor* alpha_sp     = ggml_softplus(ctx, alpha_biased);
    ggml_tensor* decay_gate   = ggml_mul(ctx, alpha_sp, W.w_a_log);
    decay_gate = ggml_reshape_4d(ctx, decay_gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    if (stop == Stage::GateProj) {
        bg.output = decay_gate;
        ggml_build_forward_expand(gf, qkv_mixed);
        ggml_build_forward_expand(gf, z);
        ggml_build_forward_expand(gf, beta);
        ggml_build_forward_expand(gf, decay_gate);
        return bg;
    }

    // ── Stage ConvUpdate — conv window cpy + ssm_conv + silu ─────────────────
    ggml_tensor* conv_all = dn_state.conv_tensor(dn_idx);
    const int64_t conv_state_elems =
        static_cast<int64_t>(conv_kernel - 1) * conv_channels;

    ggml_tensor* conv_states = ggml_view_1d(ctx, conv_all,
        conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    conv_states = ggml_reshape_3d(ctx, conv_states, conv_kernel - 1, conv_channels, n_seqs);

    ggml_tensor* qkv_t      = ggml_transpose(ctx, qkv_mixed);
    ggml_tensor* conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);

    ggml_tensor* last_conv = ggml_view_3d(ctx, conv_input,
        conv_kernel - 1, conv_channels, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        static_cast<size_t>(conv_input->ne[0] - (conv_kernel - 1))
            * ggml_element_size(conv_input));
    ggml_tensor* conv_dst = ggml_view_1d(ctx, conv_all,
        conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    ggml_tensor* conv_writeback = ggml_cpy(ctx, last_conv, conv_dst);

    ggml_tensor* conv_out = ggml_ssm_conv(ctx, conv_input, W.w_conv);
    conv_out = ggml_silu(ctx, conv_out);

    if (stop == Stage::ConvUpdate) {
        bg.output = conv_out;
        ggml_build_forward_expand(gf, z);
        ggml_build_forward_expand(gf, beta);
        ggml_build_forward_expand(gf, decay_gate);
        ggml_build_forward_expand(gf, conv_writeback);
        ggml_build_forward_expand(gf, conv_out);
        return bg;
    }

    // ── Stage StateUpdate — split, l2_norm, repeat, gated_delta_net + cpy ───
    const int64_t qkv_dim = static_cast<int64_t>(head_k_dim) * num_k_heads * 2
                            + static_cast<int64_t>(head_v_dim) * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor* q_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens, 0);

    ggml_tensor* k_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        static_cast<size_t>(head_k_dim) * num_k_heads * ggml_element_size(conv_out));

    ggml_tensor* v_conv = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_v_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        ggml_row_size(conv_out->type, 2LL * head_k_dim * num_k_heads));

    q_conv = ggml_l2_norm(ctx, q_conv, rms_norm_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, rms_norm_eps);

    if (num_k_heads != num_v_heads) {
        q_conv = ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor* rec_all = dn_state.recurrent_tensor(dn_idx);
    const int64_t rec_slot_floats =
        static_cast<int64_t>(head_v_dim) * head_k_dim * num_v_heads;

    ggml_tensor* S = ggml_view_1d(ctx, rec_all,
        rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    S = ggml_reshape_4d(ctx, S, head_v_dim, head_k_dim, num_v_heads, n_seqs);

    // K = 1 state snapshot, matching src/layers/deltanet.cpp — see the comment
    // there for why. ggml b10582 made K explicit; K=1 is the pre-b10582 layout.
    ggml_tensor* result = ggml_gated_delta_net(ctx,
        q_conv, k_conv, v_conv, decay_gate, beta, S, /*K=*/1);

    ggml_tensor* output = ggml_view_4d(ctx, result,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens),
        0);

    ggml_tensor* new_state = ggml_view_4d(ctx, result,
        head_v_dim, head_k_dim, num_v_heads, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim * num_v_heads),
        ggml_row_size(result->type,
            static_cast<int64_t>(head_v_dim) * num_v_heads * n_seq_tokens * n_seqs));

    ggml_tensor* rec_dst = ggml_view_1d(ctx, rec_all,
        rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    ggml_tensor* state_writeback = ggml_cpy(ctx,
        ggml_reshape_1d(ctx, new_state, rec_slot_floats), rec_dst);

    if (stop == Stage::StateUpdate) {
        bg.output = output;
        ggml_build_forward_expand(gf, z);
        ggml_build_forward_expand(gf, conv_writeback);
        ggml_build_forward_expand(gf, state_writeback);
        ggml_build_forward_expand(gf, output);
        return bg;
    }

    // ── Stage OutNorm — rms_norm + w_norm + silu(z) + mul ────────────────────
    ggml_tensor* z_4d   = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* normed = ggml_rms_norm(ctx, output, rms_norm_eps);
    normed = ggml_mul(ctx, normed, W.w_norm);
    ggml_tensor* z_silu = ggml_silu(ctx, z_4d);
    ggml_tensor* gated  = ggml_mul(ctx, normed, z_silu);

    if (stop == Stage::OutNorm) {
        bg.output = gated;
        ggml_build_forward_expand(gf, conv_writeback);
        ggml_build_forward_expand(gf, state_writeback);
        ggml_build_forward_expand(gf, gated);
        return bg;
    }

    // ── Stage OutProj — final mul_mat(w_out, …) ──────────────────────────────
    ggml_tensor* flat = ggml_reshape_3d(ctx, gated,
        static_cast<int64_t>(head_v_dim) * num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* out_proj = ggml_mul_mat(ctx, W.w_out, flat);
    out_proj = ggml_reshape_2d(ctx, out_proj, n_embd, n_seq_tokens * n_seqs);

    bg.output = out_proj;
    ggml_build_forward_expand(gf, conv_writeback);
    ggml_build_forward_expand(gf, state_writeback);
    ggml_build_forward_expand(gf, out_proj);
    return bg;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int /*argc*/, char** /*argv*/) {
    const char* qwen36_path = std::getenv("QWEN36_MODEL_PATH");
    const char* warmup_env  = std::getenv("DN_WARMUP");
    const char* iters_env   = std::getenv("DN_ITERS");
    const char* layer_env   = std::getenv("DN_LAYER");

    const int warmup = warmup_env ? std::atoi(warmup_env) : 10;
    const int iters  = iters_env  ? std::atoi(iters_env)  : 200;
    const int layer  = layer_env  ? std::atoi(layer_env)  : 0;

    if (!qwen36_path) {
        std::cerr << "deltanet_substage_breakdown: QWEN36_MODEL_PATH not set\n";
        return 1;
    }

    register_builtin_models();
    Model model;
    model.load_metadata(qwen36_path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    const auto& blk  = model.get_block(layer);

    if (!blk.attn_qkv_weight || !blk.ssm_a || !blk.ssm_conv1d_weight) {
        std::cerr << "deltanet_substage_breakdown: layer " << layer
                  << " is not a DeltaNet layer — try DN_LAYER=0,1,2,4,...\n";
        return 1;
    }

    const DnConfig cfg    = read_dn_config(meta);
    const int      n_embd = static_cast<int>(meta.embedding_length);

    ggml_backend_t backend = const_cast<ggml_backend_t>(model.get_backend_metal());
    if (!backend) backend = const_cast<ggml_backend_t>(model.get_backend_cpu());

    DeltaNetState::Hparams sh{
        /*n_dn_layers=*/1, /*n_slots=*/1,
        cfg.head_v_dim, cfg.head_k_dim, cfg.num_v_heads,
        cfg.conv_channels, cfg.conv_kernel,
        backend
    };
    DeltaNetState dn_state(sh);

    DnWeights W{};
    W.w_qkv     = blk.attn_qkv_weight;
    W.w_gate    = blk.attn_gate_weight;
    W.w_beta    = blk.ssm_beta_weight;
    W.w_a       = blk.ssm_alpha_weight;
    W.w_dt_bias = blk.ssm_dt_bias;
    W.w_a_log   = blk.ssm_a;
    W.w_conv    = blk.ssm_conv1d_weight;
    W.w_norm    = blk.ssm_norm_weight;
    W.w_out     = blk.ssm_out_weight;

    ggml_backend_sched_t sched = model.get_scheduler();

    std::vector<float> input_data(static_cast<size_t>(n_embd));
    std::mt19937 rng(0xCA11AB1Eu);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : input_data) v = dist(rng);

    std::cerr << "  layer=" << layer
              << " n_embd=" << n_embd
              << " d_inner=" << cfg.d_inner
              << " num_v_heads=" << cfg.num_v_heads
              << " num_k_heads=" << cfg.num_k_heads
              << " head_v_dim=" << cfg.head_v_dim
              << " head_k_dim=" << cfg.head_k_dim
              << " conv_channels=" << cfg.conv_channels
              << " conv_kernel=" << cfg.conv_kernel << "\n";

    // ── Baseline subgraph (per-iter scheduler floor) ─────────────────────────
    // Mirrors the baseline_dup pattern in deltanet_breakdown.cpp: a near-no-op
    // graph that still triggers the full sched_reset/alloc/compute/sync cycle
    // each iteration. This isolates the scheduler floor so per-layer cost can
    // be reported as `total_cumulative − baseline` rather than absorbing the
    // floor into stage 0 (which would deflate downstream stage percentages).
    BuiltGraph bg_base = make_graph();
    bg_base.input = ggml_new_tensor_2d(bg_base.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(bg_base.input);
    ggml_set_name(bg_base.input, "x");
    bg_base.output = ggml_dup(bg_base.ctx, bg_base.input);
    ggml_build_forward_expand(bg_base.gf, bg_base.output);
    ggml_tensor* xb = ggml_graph_get_tensor(bg_base.gf, "x");
    std::function<void()> upload_b = [xb, &input_data]() {
        if (xb) ggml_backend_tensor_set(xb, input_data.data(), 0,
                                         input_data.size() * sizeof(float));
    };
    const double baseline_ms = time_subgraph(bg_base, sched, upload_b,
                                             warmup, iters, "baseline_dup");
    ggml_free(bg_base.ctx);
    std::cerr << "    baseline (sched floor): " << baseline_ms << " ms\n";

    // ── Time each cumulative stage ───────────────────────────────────────────
    json stages = json::array();
    double prev_ms = 0.0;
    for (int s = 0; s <= static_cast<int>(Stage::OutProj); ++s) {
        BuiltGraph bg = build_subgraph(n_embd, static_cast<Stage>(s),
                                       cfg, W, dn_state,
                                       /*dn_idx=*/0, /*slot_idx=*/0);
        ggml_tensor* x = ggml_graph_get_tensor(bg.gf, "x");
        std::function<void()> upload = [x, &input_data]() {
            if (x) ggml_backend_tensor_set(x, input_data.data(), 0,
                                            input_data.size() * sizeof(float));
        };
        const double ms = time_subgraph(bg, sched, upload, warmup, iters,
                                        stage_name(s));
        ggml_free(bg.ctx);

        const double delta = ms - prev_ms;
        std::cerr << "    stage " << s << " (" << stage_name(s)
                  << "): cumulative " << ms << " ms, delta " << delta * 1000.0 << " µs\n";
        stages.push_back({
            {"stage",         stage_name(s)},
            {"cumulative_ms", ms},
            {"delta_ms",      delta},
            {"delta_us",      delta * 1000.0},
        });
        prev_ms = ms;
    }

    const double total_ms     = prev_ms;
    const double per_layer_ms = total_ms - baseline_ms;  // honest per-layer cost
    for (auto& s : stages) {
        s["pct_of_total"] = (total_ms > 0.0)
            ? (s["delta_ms"].get<double>() / total_ms) * 100.0
            : 0.0;
        s["pct_of_per_layer"] = (per_layer_ms > 0.0)
            ? (s["delta_ms"].get<double>() / per_layer_ms) * 100.0
            : 0.0;
    }

    // ── Verdict ──────────────────────────────────────────────────────────────
    double out_proj_ms = 0.0;
    for (auto& s : stages) {
        if (s["stage"].get<std::string>() == "out_proj") {
            out_proj_ms = s["delta_ms"].get<double>();
            break;
        }
    }
    const double out_proj_pct = (per_layer_ms > 0.0)
        ? (out_proj_ms / per_layer_ms) * 100.0
        : 0.0;

    std::string verdict;
    std::string recommendation;
    if (out_proj_pct >= 40.0) {
        verdict        = "matmul-bound";
        recommendation = "no-action: out_proj dominates per-layer cost; "
                         "fusion of the small ops cannot reach a useful ceiling. "
                         "Close PR 4.2 with baseline preserved.";
    } else {
        verdict        = "launch-bound";
        recommendation = "fused-kernel: small ops (excluding out_proj) collectively "
                         "dominate, suggesting per-dispatch overhead is fusable. "
                         "Proceed with PR 4.2 (fused DeltaNet decode-step kernel).";
    }

    std::cerr << "  total_cumulative=" << total_ms << " ms"
              << ", baseline=" << baseline_ms << " ms"
              << ", per_layer=" << per_layer_ms << " ms"
              << ", out_proj=" << out_proj_ms << " ms (" << out_proj_pct << "%)"
              << " → " << verdict << "\n";

    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));

    json out = {
        {"_note",
            "Phase 4 PR-1 sub-stage breakdown of DeltaNet decode-path cost. "
            "Stage costs computed as deltas between cumulative sub-graph timings. "
            "per_layer_ms = total_cumulative_ms − baseline_ms; verdict evaluates "
            "out_proj as a fraction of per_layer_ms. Individual stage deltas are "
            "indicative — Metal can run adjacent ops in parallel within a single "
            "command buffer, so attribution to a specific stage is fuzzy; totals "
            "are robust. The first stage (qkv_proj) additionally absorbs JIT "
            "kernel-pipeline compilation on first run, which can drive subsequent "
            "stage deltas slightly negative."},
        {"generated_at",          ts_buf},
        {"model_path",            qwen36_path},
        {"architecture",          meta.architecture},
        {"layer_idx",             layer},
        {"n_embd",                n_embd},
        {"d_inner",               cfg.d_inner},
        {"num_v_heads",           cfg.num_v_heads},
        {"num_k_heads",           cfg.num_k_heads},
        {"head_v_dim",            cfg.head_v_dim},
        {"head_k_dim",            cfg.head_k_dim},
        {"conv_channels",         cfg.conv_channels},
        {"conv_kernel",           cfg.conv_kernel},
        {"warmup_iters",          warmup},
        {"timed_iters",           iters},
        {"baseline_ms",           baseline_ms},
        {"total_cumulative_ms",   total_ms},
        {"per_layer_ms",          per_layer_ms},
        {"stages",                stages},
        {"verdict", {
            {"out_proj_ms",            out_proj_ms},
            {"out_proj_pct_of_layer",  out_proj_pct},
            {"threshold_pct",          40.0},
            {"classification",         verdict},
            {"recommendation",         recommendation},
        }},
    };
    std::cout << out.dump(2) << "\n";
    return 0;
}
