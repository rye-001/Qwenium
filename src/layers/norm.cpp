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
