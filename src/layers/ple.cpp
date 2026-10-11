#include "ple.h"

#include "norm.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

// Same naming helper as moe.cpp / attention.cpp: a deterministic ".<layer>"
// suffix keeps debug-graph dumps unambiguous across layers.
static void ple_set_name(ggml_tensor* t, const char* base, int il)
{
    char name[64];
    if (il >= 0) std::snprintf(name, sizeof(name), "%s.%d", base, il);
    else         std::snprintf(name, sizeof(name), "%s", base);
    ggml_set_name(t, name);
}

ggml_tensor* build_ple_inputs(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  x_scaled,
    ggml_tensor*  model_proj,
    ggml_tensor*  proj_norm,
    int64_t       ple_dim,
    int64_t       n_layers,
    float         eps)
{
    if (model_proj->ne[1] != ple_dim * n_layers)
        throw std::runtime_error(
            "build_ple_inputs: 'per_layer_model_proj' width: expected ple_dim * n_layers = " +
            std::to_string(ple_dim * n_layers) + ", got: " + std::to_string(model_proj->ne[1]));
    const int64_t n_embd   = x_scaled->ne[0];
    const int64_t n_tokens = x_scaled->ne[1];

    ggml_tensor* cur = ggml_mul_mat(ctx, model_proj, x_scaled);   // [ple_dim*n_layers, n_tokens]
    cur = ggml_scale(ctx, cur, 1.0f / std::sqrt(static_cast<float>(n_embd)));
    cur = ggml_reshape_3d(ctx, cur, ple_dim, n_layers, n_tokens);
    cur = build_rms_norm(ctx, cur, proj_norm, eps, /*il=*/-1);
    // [ple_dim, n_layers, n_tokens] → [ple_dim, n_tokens, n_layers], so each
    // layer's slice is contiguous.
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3));
    ple_set_name(cur, "ple_inputs", -1);
    ggml_build_forward_expand(gf, cur);
    return cur;
}

ggml_tensor* build_ple_inject(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  cur,
    ggml_tensor*  ple_inputs,
    ggml_tensor*  inp_gate,
    ggml_tensor*  proj,
    ggml_tensor*  post_norm,
    float         eps,
    int           il)
{
    if (il < 0 || il >= ple_inputs->ne[2])
        throw std::runtime_error(
            "build_ple_inject: 'il': expected in [0, " + std::to_string(ple_inputs->ne[2]) +
            "), got: " + std::to_string(il));
    if (ple_inputs->ne[1] != cur->ne[1])
        throw std::runtime_error(
            "build_ple_inject: layer " + std::to_string(il) + " 'n_tokens': expected " +
            std::to_string(cur->ne[1]) + " (the residual's), got: " + std::to_string(ple_inputs->ne[1]));

    ggml_tensor* gate = ggml_mul_mat(ctx, inp_gate, cur);           // [ple_dim, n_tokens]
    gate = ggml_gelu(ctx, gate);
    ggml_tensor* layer_in = ggml_view_2d(ctx, ple_inputs,
        ple_inputs->ne[0], ple_inputs->ne[1], ple_inputs->nb[1],
        static_cast<size_t>(il) * ple_inputs->nb[2]);
    ggml_tensor* x = ggml_mul(ctx, gate, layer_in);
    x = ggml_mul_mat(ctx, proj, x);                                 // [n_embd, n_tokens]
    x = build_rms_norm(ctx, x, post_norm, eps, il);
    ple_set_name(x, "ple_out", il);
    ggml_tensor* out = ggml_add(ctx, cur, x);
    ple_set_name(out, "ple_residual", il);
    ggml_build_forward_expand(gf, out);
    return out;
}
