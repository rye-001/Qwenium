#pragma once
// layer.h — common types shared by all layer modules.
//
// Responsibility: vocabulary that every layer module uses; kept minimal so
//   including this header does not drag in heavy dependencies.
// Contents:
//   Phase        — controls prefill vs decode graph topology.
//   AttentionKind — sliding-window vs full (global) attention mask.
//   LayerSpec    — per-layer attention shape; recipes build a vector of these
//                  and destructure into TransformerBlockHparams per iteration.
// Reference: docs/modular-layer-architecture.md § Prefill vs Decode.

#include "../state/layer_state.h"   // KvSharingSpec
#include <cstdint>

// Controls which graph topology a layer module emits.
// Passed explicitly so the memory allocator sees a single-topology graph per
// forward pass — mixing both paths in one graph defeats ggml_gallocr reuse.
enum class Phase {
    Prefill,  // batched prompt processing: full causal attention over all input tokens
    Decode,   // single-step autoregressive: one token per slot, attending to KV cache
};

// Attention mask topology for one layer.
// Global  — full causal attention over the entire KV cache.
// Sliding — causal attention limited to the last `window` tokens.
// Used by LayerSpec and by the recipe's set_inputs() to fill the KQ mask.
enum class AttentionKind {
    Global,   // full causal mask (window == 0)
    Sliding,  // sliding-window mask (window > 0)
};

// RoPE rotation variant for one attention layer.
// Standard — all rope_dim head dimensions receive RoPE rotation (rope_dim == head_dim_k).
// Pruned   — only the first rope_dim dimensions are rotated; the remaining
//             head_dim_k - rope_dim dimensions pass through unchanged.
//             This is "p-RoPE" used by Gemma 4 global layers.
//             When rope_dim == head_dim_k the two variants produce identical output.
enum class RopeKind {
    Standard,  // full-head rotation (default; all existing recipes)
    Pruned,    // partial rotation: rope_dim < head_dim_k
};

// Per-layer attention shape descriptor.
//
// Recipes (Gemma 4, future models) build a std::vector<LayerSpec> from GGUF
// metadata and destructure each element into TransformerBlockHparams before
// calling build_transformer_layer().  Uniform-shape models (Qwen, Gemma 1/2/3)
// do not need this struct — their per-call hparams are identical every iteration.
//
// Field naming follows the plan-gemma-impl.md convention:
//   head_dim_q/k/v — per-head dimensions for Q, K, V respectively.
//   n_head         — query head count.
//   n_head_kv      — KV head count (GQA).
//   rope_dim       — number of head dimensions that receive RoPE rotation.
//   rope_base      — RoPE frequency base for this layer.
//   rope_kind      — Standard (full) or Pruned (partial) RoPE variant.
//   attn_kind      — Global or Sliding.
//   window         — sliding window size in tokens (0 when attn_kind == Global).
struct LayerSpec {
    int           head_dim_q  = 0;
    int           head_dim_k  = 0;
    int           head_dim_v  = 0;
    int           n_head      = 0;
    int           n_head_kv   = 0;
    int           rope_dim    = 0;
    float         rope_base   = 10000.0f;
    RopeKind      rope_kind   = RopeKind::Standard;
    AttentionKind attn_kind   = AttentionKind::Global;
    uint32_t      window      = 0;

    // G4.4: KV cache slot ownership.  Default = owning slot (existing behavior
    // for every recipe).  Gemma 4's shared-KV feature populates this when
    // shared_kv_layers > 0; in 26B-A4B it stays default for every layer.
    KvSharingSpec kv_sharing  = {};
};
