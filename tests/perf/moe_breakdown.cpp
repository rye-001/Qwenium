// PR 4.1 investigation — per-stage timing of one MoE layer.
//
// Why: PR 4.1 spec ("single fused custom op vs chained mul_mat_id per expert")
// rests on the assumption that launch / view / scatter overhead dominates the
// matmul work in the current fallback. ggml_custom_op is CPU-only on Metal, so
// the planned implementation path is not viable; before redesigning PR 4.1 we
// need data on whether fusion can plausibly hit the 3× target at all.
//
// Method: build six cumulative sub-graphs of one MoE layer's decode-shape
// dispatch, each ending one stage later than the previous. Time each by
// repeated compute + sched_synchronize. Stage cost = median(g_N) − median(g_{N-1}).
//
// Production code is not modified; this file replicates the relevant
// graph-building prefix from MoELayer::build() so the stages can be cut at
// arbitrary points.
//
// Usage:
//   QWEN36_MODEL_PATH=/path/to/qwen36.gguf \
//   GEMMA4_MODEL_PATH=/path/to/gemma4.gguf \
//   ./bin/moe-breakdown > tests/fixtures/moe_breakdown.json
//
// Env knobs: MOE_WARMUP (default 10), MOE_ITERS (default 200),
// MOE_LAYER (default 0), MOE_TOP_K (default 8 — both target families today).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ggml.h"
#include "ggml-backend.h"

#include "engine/model.h"
#include "../../src/models/model_registry.h"

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// ── Stages ───────────────────────────────────────────────────────────────────

enum class Stage : int {
    Route   = 0,  // router matmul + argsort + top-k softmax
    Gate    = 1,  // + mul_mat_id(w_exp_gate)
    Up      = 2,  // + mul_mat_id(w_exp_up) + silu + mul
    Down    = 3,  // + mul_mat_id(w_exp_down)
    Sum     = 4,  // + weighted scatter-add over top_k
    Shared  = 5,  // + shared SwiGLU + sigmoid blend (only if has_shared)
};

static const char* stage_name(int s) {
    switch (s) {
        case 0: return "route";
        case 1: return "gate";
        case 2: return "up_silu_mul";
        case 3: return "down";
        case 4: return "weighted_sum";
        case 5: return "shared_expert";
    }
    return "?";
}

// ── Weights handle ───────────────────────────────────────────────────────────

struct MoEWeights {
    ggml_tensor* w_router;
    ggml_tensor* w_exp_gate;
    ggml_tensor* w_exp_up;
    ggml_tensor* w_exp_down;
    ggml_tensor* w_sh_gate;
    ggml_tensor* w_sh_up;
    ggml_tensor* w_sh_down;
    ggml_tensor* w_sh_norm;
    int  n_experts;
    int  top_k;
    int  ffn_dim;
    bool has_shared;
};

// ── Subgraph construction ────────────────────────────────────────────────────

struct BuiltGraph {
    ggml_context* ctx;
    ggml_cgraph*  gf;
    ggml_tensor*  input;
    ggml_tensor*  output;
    std::vector<uint8_t> meta_buf;
};

// Build a sub-graph that terminates after stage `stop`. Mirrors the
// graph-building prefix of MoELayer::build() in src/layers/moe.cpp.
static BuiltGraph build_subgraph(int n_embd, Stage stop, const MoEWeights& W) {
    BuiltGraph bg;
    bg.meta_buf.resize(64 * 1024 * 1024);

    ggml_init_params ip{};
    ip.mem_size   = bg.meta_buf.size();
    ip.mem_buffer = bg.meta_buf.data();
    ip.no_alloc   = true;
    bg.ctx = ggml_init(ip);
    bg.gf  = ggml_new_graph_custom(bg.ctx, 16384, false);

    bg.input = ggml_new_tensor_2d(bg.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(bg.input);
    ggml_set_name(bg.input, "moe_input");

    const int n_exp    = W.n_experts;
    const int top_k    = W.top_k;
    const int n_tokens = 1;

    // ── Stage Route ──────────────────────────────────────────────────────────
    ggml_tensor* logits     = ggml_mul_mat(bg.ctx, W.w_router, bg.input);
    ggml_tensor* sorted_idx = ggml_argsort(bg.ctx, logits, GGML_SORT_ORDER_DESC);
    ggml_tensor* expert_idx = ggml_view_2d(bg.ctx, sorted_idx, top_k, n_tokens, sorted_idx->nb[1], 0);

    ggml_tensor* logits_3d     = ggml_reshape_3d(bg.ctx, logits, 1, n_exp, n_tokens);
    ggml_tensor* expert_logits = ggml_get_rows(bg.ctx, logits_3d, expert_idx);
    expert_logits = ggml_reshape_2d(bg.ctx, expert_logits, top_k, n_tokens);
    ggml_tensor* expert_weights = ggml_soft_max(bg.ctx, expert_logits);

    auto seal_route_only = [&]() {
        bg.output = expert_weights;
        ggml_build_forward_expand(bg.gf, expert_weights);
        ggml_build_forward_expand(bg.gf, expert_idx);
    };
    if (stop == Stage::Route) { seal_route_only(); return bg; }

    // ── Expert dispatch matmuls ──────────────────────────────────────────────
    ggml_tensor* input_3d = ggml_reshape_3d(bg.ctx, bg.input, n_embd, 1, n_tokens);

    ggml_tensor* exp_gate_out = ggml_mul_mat_id(bg.ctx, W.w_exp_gate, input_3d, expert_idx);
    if (stop == Stage::Gate) {
        bg.output = exp_gate_out;
        ggml_build_forward_expand(bg.gf, exp_gate_out);
        ggml_build_forward_expand(bg.gf, expert_weights);
        return bg;
    }

    ggml_tensor* exp_up_out = ggml_mul_mat_id(bg.ctx, W.w_exp_up, input_3d, expert_idx);
    ggml_tensor* exp_act    = ggml_mul(bg.ctx, ggml_silu(bg.ctx, exp_gate_out), exp_up_out);
    if (stop == Stage::Up) {
        bg.output = exp_act;
        ggml_build_forward_expand(bg.gf, exp_act);
        ggml_build_forward_expand(bg.gf, expert_weights);
        return bg;
    }

    ggml_tensor* exp_down_out = ggml_mul_mat_id(bg.ctx, W.w_exp_down, exp_act, expert_idx);
    if (stop == Stage::Down) {
        bg.output = exp_down_out;
        ggml_build_forward_expand(bg.gf, exp_down_out);
        ggml_build_forward_expand(bg.gf, expert_weights);
        return bg;
    }

    // ── Weighted scatter-add (Sum stage) ─────────────────────────────────────
    ggml_tensor* w_expanded = ggml_reshape_3d(bg.ctx, expert_weights, 1, top_k, n_tokens);
    ggml_tensor* weighted   = ggml_mul(bg.ctx, exp_down_out, w_expanded);
    ggml_tensor* routed_out = ggml_view_2d(bg.ctx, weighted, n_embd, n_tokens, weighted->nb[2], 0);
    for (int k = 1; k < top_k; ++k) {
        ggml_tensor* expert_k = ggml_view_2d(bg.ctx, weighted, n_embd, n_tokens,
            weighted->nb[2], static_cast<size_t>(k) * weighted->nb[1]);
        routed_out = ggml_add(bg.ctx, routed_out, expert_k);
    }

    if (stop == Stage::Sum || !W.has_shared) {
        bg.output = routed_out;
        ggml_build_forward_expand(bg.gf, routed_out);
        return bg;
    }

    // ── Shared expert ────────────────────────────────────────────────────────
    ggml_tensor* sh_gate_out = ggml_mul_mat(bg.ctx, W.w_sh_gate, bg.input);
    ggml_tensor* sh_up_out   = ggml_mul_mat(bg.ctx, W.w_sh_up,   bg.input);
    ggml_tensor* sh_act      = ggml_mul(bg.ctx, ggml_silu(bg.ctx, sh_gate_out), sh_up_out);
    ggml_tensor* sh_down_out = ggml_mul_mat(bg.ctx, W.w_sh_down, sh_act);

    ggml_tensor* sh_gate_logit   = ggml_mul_mat(bg.ctx, W.w_sh_norm, bg.input);
    ggml_tensor* sh_gate         = ggml_sigmoid(bg.ctx, sh_gate_logit);
    ggml_tensor* sh_contribution = ggml_mul(bg.ctx, sh_down_out, sh_gate);

    ggml_tensor* combined = ggml_add(bg.ctx, routed_out, sh_contribution);
    bg.output = combined;
    ggml_build_forward_expand(bg.gf, combined);
    return bg;
}

// ── Timing ───────────────────────────────────────────────────────────────────

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double time_subgraph(
    BuiltGraph&                bg,
    ggml_backend_sched_t       sched,
    const std::vector<float>&  input_data,
    int                        warmup,
    int                        iters)
{
    // Reserve scheduler buffers for this subgraph shape. The scheduler was
    // initially reserved for the full forward graph; our partial MoE graphs
    // have different topology, so a fresh reserve is required before alloc.
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_reserve(sched, bg.gf)) {
        std::cerr << "moe_breakdown: ggml_backend_sched_reserve failed\n";
        std::exit(1);
    }

    auto run_once = [&]() {
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, bg.gf);
        ggml_backend_tensor_set(bg.input, input_data.data(), 0,
            input_data.size() * sizeof(float));
        ggml_backend_sched_graph_compute(sched, bg.gf);
        ggml_backend_sched_synchronize(sched);
    };

    for (int i = 0; i < warmup; ++i) run_once();

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        run_once();
        auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median(std::move(samples));
}

// ── Per-model entry ──────────────────────────────────────────────────────────

static MoEWeights extract_weights(const Model& model, int layer_idx, int top_k) {
    const auto& blk = model.get_block(layer_idx);
    MoEWeights W{};
    W.w_router    = blk.moe_router_weight;
    W.w_exp_gate  = blk.moe_exp_gate_weight;
    W.w_exp_up    = blk.moe_exp_up_weight;
    W.w_exp_down  = blk.moe_exp_down_weight;
    W.w_sh_gate   = blk.moe_shexp_gate_w;
    W.w_sh_up     = blk.moe_shexp_up_weight;
    W.w_sh_down   = blk.moe_shexp_down_weight;
    W.w_sh_norm   = blk.moe_shexp_gate;
    W.has_shared  = (W.w_sh_gate && W.w_sh_up && W.w_sh_down);
    W.top_k       = top_k;
    // Expert weight tensor shape: [n_embd, ffn_dim, n_experts]
    W.ffn_dim     = W.w_exp_gate ? static_cast<int>(W.w_exp_gate->ne[1]) : 0;
    W.n_experts   = W.w_exp_gate ? static_cast<int>(W.w_exp_gate->ne[2]) : 0;
    return W;
}

static json benchmark_model(
    const std::string& model_path,
    const std::string& family,
    int                top_k,
    int                layer_idx,
    int                warmup,
    int                iters)
{
    if (model_path.empty()) {
        return {{"status", "skipped"}, {"reason", "env var not set"}};
    }

    register_builtin_models();

    std::cerr << "Loading: " << model_path << "\n";
    Model model;
    model.load_metadata(model_path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    ggml_backend_sched_t sched = model.get_scheduler();

    const int n_embd = static_cast<int>(meta.embedding_length);
    MoEWeights W = extract_weights(model, layer_idx, top_k);

    if (!W.w_router || !W.w_exp_gate || !W.w_exp_up || !W.w_exp_down) {
        return {{"status", "error"},
                {"reason", "MoE weights missing on layer " + std::to_string(layer_idx)}};
    }
    if (W.n_experts <= 0 || W.ffn_dim <= 0) {
        return {{"status", "error"},
                {"reason", "could not infer n_experts/ffn_dim from expert weights"}};
    }

    std::cerr << "  family=" << family
              << " layer=" << layer_idx
              << " n_embd=" << n_embd
              << " n_exp=" << W.n_experts
              << " top_k=" << W.top_k
              << " ffn_dim=" << W.ffn_dim
              << " has_shared=" << (W.has_shared ? "true" : "false") << "\n";

    std::vector<float> input_data(static_cast<size_t>(n_embd));
    std::mt19937 rng(0xCA11AB1Eu);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : input_data) v = dist(rng);

    json stages = json::array();
    double prev_ms = 0.0;
    int max_stage = W.has_shared ? static_cast<int>(Stage::Shared)
                                 : static_cast<int>(Stage::Sum);

    for (int s = 0; s <= max_stage; ++s) {
        BuiltGraph bg = build_subgraph(n_embd, static_cast<Stage>(s), W);
        double ms = time_subgraph(bg, sched, input_data, warmup, iters);
        ggml_free(bg.ctx);

        double delta = ms - prev_ms;
        std::cerr << "    stage " << s << " (" << stage_name(s)
                  << "): cumulative " << ms << " ms, delta " << delta << " ms\n";
        stages.push_back({
            {"stage",         stage_name(s)},
            {"cumulative_ms", ms},
            {"delta_ms",      delta},
        });
        prev_ms = ms;
    }

    const double total_ms = prev_ms;
    for (auto& s : stages) {
        s["pct_of_total"] = (total_ms > 0.0)
            ? (s["delta_ms"].get<double>() / total_ms) * 100.0
            : 0.0;
    }

    return {
        {"status",        "ok"},
        {"model_path",    model_path},
        {"family",        family},
        {"architecture",  meta.architecture},
        {"n_embd",        n_embd},
        {"n_experts",     W.n_experts},
        {"top_k",         W.top_k},
        {"ffn_dim",       W.ffn_dim},
        {"has_shared",    W.has_shared},
        {"layer_idx",     layer_idx},
        {"warmup_iters",  warmup},
        {"timed_iters",   iters},
        {"total_ms",      total_ms},
        {"stages",        stages},
    };
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int /*argc*/, char** /*argv*/) {
    const char* qwen36_path = std::getenv("QWEN36_MODEL_PATH");
    const char* gemma4_path = std::getenv("GEMMA4_MODEL_PATH");
    const char* warmup_env  = std::getenv("MOE_WARMUP");
    const char* iters_env   = std::getenv("MOE_ITERS");
    const char* layer_env   = std::getenv("MOE_LAYER");
    const char* topk_env    = std::getenv("MOE_TOP_K");

    const int warmup = warmup_env ? std::atoi(warmup_env) : 10;
    const int iters  = iters_env  ? std::atoi(iters_env)  : 200;
    const int layer  = layer_env  ? std::atoi(layer_env)  : 0;
    const int top_k  = topk_env   ? std::atoi(topk_env)   : 8;

    if (!qwen36_path && !gemma4_path) {
        std::cerr << "moe_breakdown: set QWEN36_MODEL_PATH and/or GEMMA4_MODEL_PATH\n";
        std::cerr << "  example:\n";
        std::cerr << "    QWEN36_MODEL_PATH=/path/to/qwen36.gguf \\\n";
        std::cerr << "    GEMMA4_MODEL_PATH=/path/to/gemma4.gguf \\\n";
        std::cerr << "    ./bin/moe-breakdown > tests/fixtures/moe_breakdown.json\n";
        return 1;
    }

    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));

    json out = {
        {"_note",
            "PR 4.1 investigation: per-stage MoE timing for the decode shape "
            "(n_tokens=1). Stage costs computed as deltas between cumulative "
            "sub-graph timings. Generated by tests/perf/moe_breakdown.cpp."},
        {"generated_at", ts_buf},
        {"top_k_used",   top_k},
    };

    if (qwen36_path) {
        out["qwen36"] = benchmark_model(qwen36_path, "qwen36", top_k, layer, warmup, iters);
    }
    if (gemma4_path) {
        out["gemma4"] = benchmark_model(gemma4_path, "gemma4", top_k, layer, warmup, iters);
    }

    std::cout << out.dump(2) << "\n";
    return 0;
}
