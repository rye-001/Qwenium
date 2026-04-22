#pragma once
// moe.h — Mixture-of-Experts layer: top-k gating + expert SwiGLU dispatch.
//
// Responsibility: construct the MoE subgraph for one transformer layer.
//   Implements: gating network (top-k softmax over expert scores),
//   per-expert SwiGLU FFN dispatch via ggml_mul_mat_id (fallback path,
//   gated behind QINF_MOE_FALLBACK; the fused Metal kernel lives in Phase 4),
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
//   - Expert weight tensors must be 3D: [in_dim, out_dim, n_experts].
// Reference: llama.cpp qwen35moe.cpp; Mixtral MoE pattern.
// Unit test: tests/unit/test_moe.cpp

#include "layer.h"
#include "ggml.h"

#include <cstdint>

struct ggml_context;
struct ggml_cgraph;

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
        const Hparams& hp);

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
    ggml_tensor* w_sh_norm_;
    Hparams      hp_;
};
