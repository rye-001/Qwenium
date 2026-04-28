#include "ffn.h"
#include "ggml.h"
#include <cstdio>

// Extracted from ForwardPassBase::ffn_swiglu (src/models/forward_pass_base.cpp).
// Logic is identical — only ctx_ → ctx parameter.
ggml_tensor* build_ffn_swiglu(
    ggml_context* ctx,
    ggml_cgraph*  /*gf*/,
    ggml_tensor*  cur,
    ggml_tensor*  gate,
    ggml_tensor*  up,
    ggml_tensor*  down,
    int           il,
    Phase         /*phase*/)
{
    char name[128];

    ggml_tensor* tmp = ggml_mul_mat(ctx, up, cur);
    snprintf(name, sizeof(name), "ffn_up.%d", il);
    ggml_set_name(tmp, name);

    cur = ggml_mul_mat(ctx, gate, cur);
    snprintf(name, sizeof(name), "ffn_gate.%d", il);
    ggml_set_name(cur, name);

    cur = ggml_swiglu_split(ctx, cur, tmp);
    snprintf(name, sizeof(name), "ffn_swiglu.%d", il);
    ggml_set_name(cur, name);

    cur = ggml_mul_mat(ctx, down, cur);
    return cur;
}

ggml_tensor* build_ffn_geglu_tanh(
    ggml_context* ctx,
    ggml_cgraph*  /*gf*/,
    ggml_tensor*  cur,
    ggml_tensor*  gate,
    ggml_tensor*  up,
    ggml_tensor*  down,
    int           il,
    Phase         /*phase*/)
{
    char name[128];

    ggml_tensor* tmp = ggml_mul_mat(ctx, up, cur);
    snprintf(name, sizeof(name), "ffn_up.%d", il);
    ggml_set_name(tmp, name);

    cur = ggml_mul_mat(ctx, gate, cur);
    snprintf(name, sizeof(name), "ffn_gate.%d", il);
    ggml_set_name(cur, name);

    // ggml_geglu_split(a, b) = gelu(a) * b. Default ggml_gelu uses the
    // tanh approximation (matches PyTorch gelu_pytorch_tanh).
    cur = ggml_geglu_split(ctx, cur, tmp);
    snprintf(name, sizeof(name), "ffn_geglu.%d", il);
    ggml_set_name(cur, name);

    cur = ggml_mul_mat(ctx, down, cur);
    return cur;
}
