#pragma once
// deltanet.h — Gated DeltaNet layer graph-building: free functions + DeltaNetLayer class.
//
// Responsibility: construct the recurrent-state-backed DeltaNet subgraph for
//   one hybrid-model layer. Implements the Gated Delta Rule recurrence:
//   S = S * decay + k * beta * (v - S*k), with 1D causal conv on Q/K/V.
// Public surface (free function):
//     build_deltanet_layer — full pipeline: projections → conv → L2-norm →
//         ggml_gated_delta_net → state write-back → gated RMSNorm → out-proj.
// Public surface (class):
//     DeltaNetLayer::build() — unified entry point dispatching on Phase enum.
//       Prefill: processes n_tokens tokens from a single slot.
//       Decode:  processes one token per entry in the slots vector.
// State owned: none — DeltaNetState and weight tensors are passed by the caller.
// Invariants:
//   - All tensors are appended to the caller's ggml_cgraph; no ggml_context
//     is created inside this module.
//   - The recurrent state is written back to DeltaNetState inside the graph
//     via ggml_cpy into the state tensor view; the caller must execute the
//     graph before reading updated state.
//   - ggml_cont() is used only where Metal kernels require contiguity;
//     each use is documented in the implementation.
// Reference: Qwen35ForwardPass::build_ssm_layer in src/models/qwen35.cpp;
//   llama.cpp delta-net-base.cpp (https://github.com/ggml-org/llama.cpp).
// Unit test: tests/unit/test_deltanet.cpp

#include "layer.h"
#include "../state/deltanet_state.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_cgraph;

// ── Free function (internal / legacy call sites) ──────────────────────────────

// Build the full GatedDeltaNet subgraph for one layer, single-slot prefill.
// Performs: QKV proj → Z-gate proj → beta/alpha proj → conv1d → L2-norm →
//   ggml_gated_delta_net (fused) → state write-back → gated RMSNorm → out-proj.
// Returns output tensor [n_embd, n_tokens].
ggml_tensor* build_deltanet_layer(
    ggml_context*   ctx,
    ggml_cgraph*    gf,
    ggml_tensor*    cur,          // normed input [n_embd, n_tokens]
    DeltaNetState*  dn_state,
    uint32_t        dn_idx,       // DeltaNet layer index within dn_state
    uint32_t        slot_idx,     // sequence slot
    uint32_t        n_tokens,
    ggml_tensor*    w_qkv,        // [n_embd, conv_channels] — joint QKV projection
    ggml_tensor*    w_gate,       // [n_embd, d_inner]       — output gate (Z)
    ggml_tensor*    w_beta,       // [n_embd, num_v_heads]   — beta gate (scalar per head)
    ggml_tensor*    w_a,          // [n_embd, num_v_heads]   — alpha for decay gate
    ggml_tensor*    w_dt_bias,    // [num_v_heads]           — dt bias
    ggml_tensor*    w_a_log,      // [num_v_heads]            — A_log scalar (1D)
    ggml_tensor*    w_conv,       // [conv_channels, 1, conv_kernel] — depthwise conv
    ggml_tensor*    w_norm,       // [d_inner]               — RMSNorm gamma
    ggml_tensor*    w_out,        // [d_inner, n_embd]       — output projection
    int             n_embd,
    int             d_inner,
    int             head_k_dim,
    int             num_k_heads,
    int             num_v_heads,
    int             head_v_dim,
    int             conv_channels,
    int             conv_kernel,
    float           rms_norm_eps,
    int             il);          // physical layer index (for tensor naming)

// ── DeltaNetLayer class (canonical Phase 3 interface) ────────────────────────

class DeltaNetLayer {
public:
    struct Hparams {
        int   n_embd;
        int   d_inner;
        int   head_k_dim;
        int   num_k_heads;
        int   num_v_heads;
        int   head_v_dim;
        int   conv_channels;
        int   conv_kernel;
        float rms_norm_eps;
    };

    // All weight tensors are borrowed references; DeltaNetLayer does not own them.
    DeltaNetLayer(
        ggml_tensor*   w_qkv,
        ggml_tensor*   w_gate,
        ggml_tensor*   w_beta,
        ggml_tensor*   w_a,
        ggml_tensor*   w_dt_bias,
        ggml_tensor*   w_a_log,
        ggml_tensor*   w_conv,
        ggml_tensor*   w_norm,
        ggml_tensor*   w_out,
        DeltaNetState* dn_state,
        const Hparams& hp);

    // Args for Phase::Prefill — single slot, multiple tokens.
    struct PrefillArgs {
        uint32_t n_tokens;
        uint32_t slot_idx;
    };

    // Args for Phase::Decode — one token per slot in the batch.
    struct DecodeArgs {
        std::vector<uint32_t> slots;  // sequence slot per batch element
    };

    // Unified build entry point. Returns output [n_embd, n_tokens/n_batch].
    ggml_tensor* build(
        ggml_context*      ctx,
        ggml_cgraph*       gf,
        ggml_tensor*       input,       // normed residual [n_embd, n_tokens/n_batch]
        uint32_t           dn_idx,      // DeltaNet layer index within dn_state
        Phase              phase,
        const PrefillArgs& prefill_args,
        const DecodeArgs*  decode_args  = nullptr);

private:
    ggml_tensor*   w_qkv_;
    ggml_tensor*   w_gate_;
    ggml_tensor*   w_beta_;
    ggml_tensor*   w_a_;
    ggml_tensor*   w_dt_bias_;
    ggml_tensor*   w_a_log_;
    ggml_tensor*   w_conv_;
    ggml_tensor*   w_norm_;
    ggml_tensor*   w_out_;
    DeltaNetState* dn_state_;
    Hparams        hp_;

    ggml_tensor* build_prefill(ggml_context*, ggml_cgraph*, ggml_tensor* input,
                                uint32_t dn_idx, const PrefillArgs&);
    ggml_tensor* build_decode(ggml_context*, ggml_cgraph*, ggml_tensor* input,
                               uint32_t dn_idx, const DecodeArgs&);
};
