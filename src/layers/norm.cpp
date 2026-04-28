#include "norm.h"
#include "ggml.h"
#include <cstdio>

static void set_norm_name(ggml_tensor* t, const char* base, int il) {
    if (il >= 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s.%d", base, il);
        ggml_set_name(t, buf);
    } else {
        ggml_set_name(t, base);
    }
}

ggml_tensor* build_rms_norm(
    ggml_context* ctx,
    ggml_tensor*  cur,
    ggml_tensor*  weight,
    float         eps,
    int           il)
{
    cur = ggml_rms_norm(ctx, cur, eps);
    set_norm_name(cur, "cur_rms_normed", il);
    cur = ggml_mul(ctx, cur, weight);
    return cur;
}

ggml_tensor* build_rms_norm_gemma(
    ggml_context* ctx,
    ggml_tensor*  cur,
    ggml_tensor*  weight,
    float         eps,
    int           il)
{
    cur = ggml_rms_norm(ctx, cur, eps);
    set_norm_name(cur, "cur_rms_normed", il);
    // (1 + w): scale_bias computes s*a + b, so 1*weight + 1.
    ggml_tensor* one_plus_w = ggml_scale_bias(ctx, weight, 1.0f, 1.0f);
    set_norm_name(one_plus_w, "gemma_one_plus_w", il);
    cur = ggml_mul(ctx, cur, one_plus_w);
    return cur;
}

ggml_tensor* build_embed_scale(
    ggml_context* ctx,
    ggml_tensor*  cur,
    float         scale)
{
    cur = ggml_scale(ctx, cur, scale);
    ggml_set_name(cur, "embed_scaled");
    return cur;
}
