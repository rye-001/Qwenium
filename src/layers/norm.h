#pragma once

struct ggml_context;
struct ggml_tensor;

// Canonical RMS norm + weight scaling shared by all layer types.
// Names the intermediate tensor "cur_rms_normed.il" (or "cur_rms_normed" when
// il < 0) to match the ForwardPassBase::build_norm convention so that graph
// debug output and tensor-name-based lookups in set_inputs remain stable.
ggml_tensor* build_rms_norm(
    ggml_context* ctx,
    ggml_tensor*  cur,
    ggml_tensor*  weight,
    float         eps,
    int           il);

// Gemma-style RMS norm: output = (x / rms(x)) * (1 + w).
// Why: Gemma stores the norm weight as a delta-from-unity. Implementing
// the standard `x * w` form silently produces near-zero outputs in the
// first few layers — the most common Gemma porting bug.
ggml_tensor* build_rms_norm_gemma(
    ggml_context* ctx,
    ggml_tensor*  cur,
    ggml_tensor*  weight,
    float         eps,
    int           il);

// Embedding-scale hook (Gemma): output = cur * scale.
// The recipe passes scale = sqrtf(hidden_size). Multiplication is element-wise
// in fp32; the caller is responsible for keeping the embedding lookup result
// in fp32 before this call (the scale must apply in fp32, not bf16/f16, or
// downstream attention probabilities collapse).
ggml_tensor* build_embed_scale(
    ggml_context* ctx,
    ggml_tensor*  cur,
    float         scale);
