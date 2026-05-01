#include "ple.h"

#include "ggml.h"

#include <cstdio>

// Mirrors the small naming helper in moe.cpp / attention.cpp — gives every
// node a deterministic ".<layer>" suffix so debug-graph dumps remain
// unambiguous when this op is invoked from multiple layers in one graph.
static void ple_set_name(ggml_cgraph* gf, ggml_tensor* t,
                         const char* base, int il)
{
    char name[64];
    std::snprintf(name, sizeof(name), "%s.%d", base, il);
    ggml_set_name(t, name);
    ggml_build_forward_expand(gf, t);
}

ggml_tensor* build_ple(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  token_ids,
    ggml_tensor*  ple_table,
    ggml_tensor*  ple_proj,
    int           il)
{
    // 1. Per-layer embedding lookup: gather the row for each token.
    //    ple_table layout: [ple_dim (ne[0]), vocab_size (ne[1])].
    //    ggml_get_rows selects rows from ne[1] by token_ids → [ple_dim, n_tokens].
    ggml_tensor* ple_emb = ggml_get_rows(ctx, ple_table, token_ids);
    ple_set_name(gf, ple_emb, "ple_emb", il);

    // 2. Project into the residual-stream width.
    //    ple_proj: [ple_dim (ne[0]), hidden_dim (ne[1])] in ggml's matmul
    //    convention → ggml_mul_mat(ple_proj, ple_emb) yields [hidden_dim, n_tokens].
    ggml_tensor* projected = ggml_mul_mat(ctx, ple_proj, ple_emb);
    ple_set_name(gf, projected, "ple_projected", il);

    return projected;
}
