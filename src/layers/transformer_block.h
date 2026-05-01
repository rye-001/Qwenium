#pragma once
// transformer_block.h — standard Qwen2/Qwen3 per-layer graph builder.
//
// Responsibility: construct the complete transformer layer graph for one block:
//   norm → QKV proj → optional QK norms → RoPE → attention → output proj →
//   residual → FFN norm → SwiGLU FFN → residual.
// Public surface: build_transformer_layer free function.
// State owned: none — ggml_context, KV cache, and weights are passed by the caller.
// Invariants: appends to the caller's ggml_cgraph; creates no ggml_context.
// Unit test: tests/unit/test_transformer_block.cpp
//
// Tensor naming: verbose (matches build_prefill_graph convention).
// Build-time only — negligible cost on the TQ hot path.

#include "layer.h"
#include "ggml.h"
#include <cstdint>

struct ggml_context;
struct ggml_cgraph;
class simple_kv_cache;

// Architecture hyperparameters for one transformer layer.
// The caller (model recipe) extracts these from ModelMetadata before the loop.
struct TransformerBlockHparams {
    bool  is_qwen2;        // true → apply QKV biases and skip QK RMS norms (Qwen2)
    int   n_head;          // query head count
    int   n_head_kv;       // KV head count (GQA)
    int   n_embd_head;     // Q per-head dimension (= hidden_dim / n_head for Qwen/Gemma 1-3)
    float freq_base;       // RoPE frequency base
    int   context_length;  // original context length for RoPE ext
    float rms_norm_eps;    // epsilon for RMS norm

    // ── Gemma-1 knobs (PR G1.5) ─────────────────────────────────────────
    // Defaults preserve bit-identical Qwen behavior. Flipping either flag
    // selects a Gemma-shaped op at the same call site — no parallel
    // module hierarchy.
    bool  gemma_rms_norm = false;  // true → (x / rms(x)) * (1 + w) instead of x * w
    bool  gemma_geglu    = false;  // true → GeGLU-tanh FFN instead of SwiGLU

    // ── Gemma-2 knobs (PR G2) ────────────────────────────────────────────
    // attn_softcap: attention logit soft-capping value (cap * tanh(x/cap)).
    //   0.0 = off (Qwen/Gemma-1 path — behavior is bit-identical).
    //   50.0 for Gemma 2.
    float attn_softcap = 0.0f;

    // ── G4.1: per-layer attention shape overrides ────────────────────────
    // When non-zero, override n_embd_head for K and V head dimensions
    // independently.  All existing recipes leave these at 0 (degenerate
    // case: head_dim_k = head_dim_v = n_embd_head — behavior unchanged).
    // Gemma 4 sets these per-layer from LayerSpec: SWA layers have
    // head_dim_k = head_dim_v = 256 while n_embd_head = 256 (also 256);
    // global layers have head_dim_k = head_dim_v = 512 while n_embd_head = 256.
    int      head_dim_k  = 0;   // K head dim; 0 → n_embd_head
    int      head_dim_v  = 0;   // V head dim; 0 → n_embd_head

    // rope_dim: number of head dimensions that receive RoPE rotation.
    //   0 → n_embd_head (full-head RoPE — all existing recipes).
    //   Gemma 4 global layers use 512 (= head_dim_k); SWA layers use 256.
    //   p-RoPE (G4.5) further prunes within rope_dim — handled in G4.5.
    int      rope_dim    = 0;
};

// Weight tensors for one transformer block.
// The caller extracts these from TransformerBlock before the loop.
struct TransformerBlockWeights {
    ggml_tensor* attn_norm;  // pre-attention RMS norm weight
    ggml_tensor* q;          // attn_q_weight
    ggml_tensor* k;          // attn_k_weight
    ggml_tensor* v;          // attn_v_weight
    ggml_tensor* q_bias;     // attn_q_bias  (nullptr when !is_qwen2)
    ggml_tensor* k_bias;     // attn_k_bias
    ggml_tensor* v_bias;     // attn_v_bias
    ggml_tensor* q_norm;     // attn_q_norm_weight (nullptr when is_qwen2)
    ggml_tensor* k_norm;     // attn_k_norm_weight
    ggml_tensor* out;        // attn_output_weight
    ggml_tensor* ffn_norm;   // pre-FFN RMS norm weight
    ggml_tensor* ffn_gate;   // ffn_gate_weight
    ggml_tensor* ffn_up;     // ffn_up_weight
    ggml_tensor* ffn_down;   // ffn_down_weight

    // ── Gemma-2 sandwich norm weights (PR G2.1) ──────────────────────────
    // nullptr → standard pre-norm only (Qwen/Gemma-1 path unchanged).
    // Non-null → sandwich norm: post-norm applied after attn/FFN output,
    //   before the residual add.  x + post_norm(op(pre_norm(x))).
    ggml_tensor* post_attn_norm = nullptr;  // blk.N.post_attention_norm.weight
    ggml_tensor* post_ffn_norm  = nullptr;  // blk.N.post_ffw_norm.weight
};

// Build one transformer layer's ggml nodes into an existing graph.
//
// cur:           hidden state entering this layer [n_embd, n_tokens]
// inp_pos:       token position indices [n_tokens]
// il:            physical layer index (used for KV cache slot selection and tensor naming)
// ple_residual:  optional secondary residual contribution added to the
//                residual stream at the block-entry injection point — i.e.
//                inpSA = cur + ple_residual is what subsequent norms/attn/FFN
//                see and what the final residual adds back into.  Shape must
//                match cur ([n_embd, n_tokens]).  Pass nullptr for the
//                Qwen / Gemma 1/2/3 path (behavior bit-identical when null).
//                When the input is all-zeros the non-null path is also
//                bit-identical to the null path, by construction
//                (ggml_add of zeros is the identity).
//                Used by Gemma 4 PLE (G4.3); inactive in 26B-A4B.
//
// Returns the updated hidden state tensor after the full layer.
ggml_tensor* build_transformer_layer(
    ggml_context*                  ctx,
    ggml_cgraph*                   gf,
    simple_kv_cache*               kv_cache,
    ggml_tensor*                   cur,
    ggml_tensor*                   inp_pos,
    const TransformerBlockWeights& w,
    const TransformerBlockHparams& hp,
    uint32_t                       il,
    uint32_t                       slot_idx,
    uint32_t                       n_tokens,
    ggml_tensor*                   ple_residual = nullptr);
