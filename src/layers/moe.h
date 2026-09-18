#pragma once
// moe.h — Mixture-of-Experts layer: top-k gating + expert SwiGLU dispatch.
//
// Responsibility: construct the MoE subgraph for one transformer layer.
//   Implements: gating network (top-k softmax over expert scores),
//   per-expert SwiGLU FFN dispatch via three batched ggml_mul_mat_id calls
//   (native Metal MUL_MAT_ID; launch count is O(1) in n_experts),
//   and optional sigmoid-gated shared expert blending.
// Public surface:
//     MoELayer::build() — unified entry point; Phase arg is accepted for
//       interface uniformity but MoE has no phase-dependent graph topology.
// State owned: none — stateless, all expert weights are passed by the caller.
// Invariants:
//   - All tensors are appended to the caller's ggml_cgraph; no ggml_context
//     is created inside this module.
//   - The fallback dispatch uses ggml_mul_mat_id which keeps kernel dispatch
//     count at O(1) per layer (not O(top_k)).
//   - When has_shared_expert == true, the shared expert contribution is added
//     with sigmoid gating after the routed experts are summed.
//   - RoutingSource::Router (the default) is byte-identical to the behaviour
//     that predates routing replay. RoutingSource::Replay swaps ONE tensor:
//     the top-k index operand ggml_mul_mat_id already takes. Nothing
//     downstream changes or needs to know — the gating weights are still
//     gathered from the REAL router logits at the replayed indices, so only
//     the discrete choice is pinned, not the gate. See routing_trace.h.
//   - Expert weight tensors must be 3D: [in_dim, out_dim, n_experts].
// Reference: llama.cpp qwen35moe.cpp; Mixtral MoE pattern.
// Unit test: tests/unit/test_moe.cpp

#include "layer.h"
#include "routing_trace.h"
#include "ggml.h"

#include <cstdint>

struct ggml_context;
struct ggml_cgraph;

// The top-k index operand ggml_mul_mat_id takes, from whichever source the
// policy names. FREE FUNCTION on purpose: Qwen's MoELayer and Gemma 4's
// build_moe_geglu are structurally different layers (dual-FFN vs plain MoE,
// GeGLU vs SwiGLU, shared expert vs none) that nonetheless select experts with
// the identical three lines. Routing replay has to hold in both families
// (CLAUDE.md cross-family rule) and both models fail the drift gate today, so
// the selection lives in one place rather than being branched twice.
//
// Router  -> argsort the logits, view the top_k prefix, name it "moe_idx.<il>".
// Replay  -> an I32 graph input named "moe_routing.<il>", filled by
//            RoutingReplayInput before compute. The caller must still build
//            `logits`: the gating weights are gathered from the REAL router
//            output at the replayed indices, so replay pins the discrete
//            choice only, never the gate.
ggml_tensor* moe_build_expert_idx(ggml_context* ctx,
                                  ggml_cgraph*  gf,
                                  ggml_tensor*  logits,     // [n_experts, n_tokens]
                                  int           top_k,
                                  int64_t       n_tokens,
                                  RoutingSource routing,
                                  int           il);

class MoELayer {
public:
    struct Hparams {
        int  n_experts;           // total expert count
        int  top_k;               // routed experts per token
        int  ffn_dim;             // expert intermediate dimension
        bool has_shared_expert;   // whether a shared expert is added
    };

    // All weight tensors are borrowed references; MoELayer does not own them.
    // w_sh_* and w_sh_norm must be non-null iff hp.has_shared_expert == true.
    // Expert weight tensors are 3D: [in_dim, out_dim, n_experts].
    MoELayer(
        ggml_tensor* w_router,     // [n_embd, n_experts] — routing logits
        ggml_tensor* w_exp_gate,   // [n_embd, ffn_dim, n_experts]
        ggml_tensor* w_exp_up,     // [n_embd, ffn_dim, n_experts]
        ggml_tensor* w_exp_down,   // [ffn_dim, n_embd, n_experts]
        ggml_tensor* w_sh_gate,    // [n_embd, ffn_dim] shared (nullptr if no shared)
        ggml_tensor* w_sh_up,      // [n_embd, ffn_dim] shared
        ggml_tensor* w_sh_down,    // [ffn_dim, n_embd] shared
        ggml_tensor* w_sh_norm,    // [1] shared expert weight scalar
        const Hparams& hp,
        // Router = the layer chooses (default, byte-identical to before).
        // Replay = the top-k operand becomes a graph input named
        // "moe_routing.<il>", filled by RoutingReplayInput. The router matmul
        // still runs: its logits are what the gate weights are gathered from.
        RoutingSource routing = RoutingSource::Router);

    // Build the MoE subgraph. Phase is accepted for interface uniformity;
    // MoE has one graph shape (no prefill/decode distinction).
    // Returns output tensor [n_embd, n_tokens].
    ggml_tensor* build(
        ggml_context* ctx,
        ggml_cgraph*  gf,
        ggml_tensor*  input,   // [n_embd, n_tokens]
        Phase         phase,
        int           il);     // physical layer index (for tensor naming)

private:
    ggml_tensor* w_router_;
    ggml_tensor* w_exp_gate_;
    ggml_tensor* w_exp_up_;
    ggml_tensor* w_exp_down_;
    ggml_tensor* w_sh_gate_;
    ggml_tensor* w_sh_up_;
    ggml_tensor* w_sh_down_;
    ggml_tensor*  w_sh_norm_;
    Hparams       hp_;
    RoutingSource routing_;
};
