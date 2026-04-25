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

#include "ggml.h"
#include <cstdint>

struct ggml_context;
struct ggml_cgraph;
class simple_kv_cache;

// Architecture hyperparameters for one transformer layer.
// The caller (model recipe) extracts these from Qwen3Metadata before the loop.
struct TransformerBlockHparams {
    bool  is_qwen2;        // true → apply QKV biases and skip QK RMS norms (Qwen2)
    int   n_head;          // query head count
    int   n_head_kv;       // KV head count (GQA)
    int   n_embd_head;     // per-head embedding dimension
    float freq_base;       // RoPE frequency base
    int   context_length;  // original context length for RoPE ext
    float rms_norm_eps;    // epsilon for RMS norm
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
};

// Build one transformer layer's ggml nodes into an existing graph.
//
// cur:      hidden state entering this layer [n_embd, n_tokens]
// inp_pos:  token position indices [n_tokens]
// il:       physical layer index (used for KV cache slot selection and tensor naming)
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
    uint32_t                       n_tokens);
