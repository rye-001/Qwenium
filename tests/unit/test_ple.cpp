// test_ple.cpp — PR G4.3
//
// Unit tests for build_ple (src/layers/ple.cpp).
//
// PLE is a Gemma 4 feature that takes per-layer token embeddings, projects
// them, and returns a residual contribution at hidden_dim resolution.
// The 26B-A4B checkpoint has embedding_length_per_layer_input=0, so PLE is
// dormant in production; these tests are the synthetic-config gate the plan
// pre-commits to.
//
// Tests:
//   1. SyntheticOracle      — small (vocab=4, ple_dim=2, hidden_dim=3,
//                             n_tokens=2) configuration with hand-computed
//                             expected output.
//   2. ZeroProjectionIsZero — ple_proj filled with zeros ⇒ output is zero.
//   3. IdentityProjection   — when ple_dim == hidden_dim and ple_proj is the
//                             identity matrix, build_ple == ggml_get_rows.
//
// No model file required.
// Build target: ple-tests (see tests/CMakeLists.txt).

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../src/layers/ple.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

namespace {

// Run build_ple with the given table/proj weights and tokens.  Returns the
// flat output tensor [hidden_dim * n_tokens] in ggml's column-major layout.
struct PleConfig {
    int            vocab;
    int            ple_dim;
    int            hidden_dim;
    int            n_tokens;
    std::vector<int32_t> tokens;        // [n_tokens]
    std::vector<float>   table;         // [ple_dim * vocab]   (column-major)
    std::vector<float>   proj;          // [ple_dim * hidden]  (ggml mul_mat
                                        //   weight layout: [ne0=ple_dim,
                                        //   ne1=hidden_dim])
};

std::vector<float> run_ple(const PleConfig& cfg) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    EXPECT_NE(backend, nullptr);

    // Weight context (real allocation — ggml_get_rows needs row-major data).
    const size_t weight_bytes = 64 * ggml_tensor_overhead()
        + (cfg.table.size() + cfg.proj.size()) * sizeof(float);
    ggml_init_params wp{ weight_bytes, nullptr, /*no_alloc=*/false };
    ggml_context* wctx = ggml_init(wp);

    ggml_tensor* table_t = ggml_new_tensor_2d(
        wctx, GGML_TYPE_F32, cfg.ple_dim, cfg.vocab);
    std::memcpy(table_t->data, cfg.table.data(),
                cfg.table.size() * sizeof(float));

    ggml_tensor* proj_t = ggml_new_tensor_2d(
        wctx, GGML_TYPE_F32, cfg.ple_dim, cfg.hidden_dim);
    std::memcpy(proj_t->data, cfg.proj.data(),
                cfg.proj.size() * sizeof(float));

    // Compute graph context.
    const size_t cctx_bytes = 256 * ggml_tensor_overhead()
        + 64 * cfg.hidden_dim * cfg.n_tokens * sizeof(float);
    ggml_init_params cp{ cctx_bytes, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(cp);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* token_ids = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, cfg.n_tokens);
    ggml_set_input(token_ids);

    ggml_tensor* out = build_ple(ctx, gf, token_ids, table_t, proj_t,
                                 /*il=*/0);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(token_ids, cfg.tokens.data(), 0,
                            cfg.tokens.size() * sizeof(int32_t));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> result(cfg.hidden_dim * cfg.n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0,
                            result.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return result;
}

} // namespace

// 1. Synthetic oracle.
//   ple_table (vocab=4, ple_dim=2):
//      row 0: (1, 2)
//      row 1: (3, 4)
//      row 2: (5, 6)
//      row 3: (7, 8)
//   tokens = [2, 0]  → looked-up rows: (5,6) and (1,2)
//   ple_proj (hidden_dim=3, ple_dim=2), ggml-layout [ple_dim=ne0, hidden_dim=ne1]:
//       proj[h, p]: h=0: (1, 0)  h=1: (0, 1)  h=2: (1, 1)
//   For each token, output[h] = sum_p proj[h, p] * lookup[p]:
//       token 2 (5,6): out = (5*1+6*0, 5*0+6*1, 5*1+6*1) = (5, 6, 11)
//       token 0 (1,2): out = (1*1+2*0, 1*0+2*1, 1*1+2*1) = (1, 2, 3)
TEST(PleTest, SyntheticOracle) {
    PleConfig cfg;
    cfg.vocab      = 4;
    cfg.ple_dim    = 2;
    cfg.hidden_dim = 3;
    cfg.n_tokens   = 2;
    cfg.tokens     = { 2, 0 };
    // table column-major (ne0=ple_dim, ne1=vocab):
    //   row 0 (vocab 0): (1, 2);  row 1 (vocab 1): (3, 4);
    //   row 2 (vocab 2): (5, 6);  row 3 (vocab 3): (7, 8);
    cfg.table = {
        1.0f, 2.0f,   // vocab 0
        3.0f, 4.0f,   // vocab 1
        5.0f, 6.0f,   // vocab 2
        7.0f, 8.0f,   // vocab 3
    };
    // proj layout [ne0=ple_dim, ne1=hidden_dim]:
    //   hidden 0: (1, 0)
    //   hidden 1: (0, 1)
    //   hidden 2: (1, 1)
    cfg.proj = {
        1.0f, 0.0f,   // hidden 0
        0.0f, 1.0f,   // hidden 1
        1.0f, 1.0f,   // hidden 2
    };

    auto out = run_ple(cfg);
    ASSERT_EQ(out.size(), static_cast<size_t>(cfg.hidden_dim * cfg.n_tokens));

    // ggml output layout [ne0=hidden_dim, ne1=n_tokens] → flat index t*hidden + h.
    // Token 0 is the first looked-up token (id=2) → expected (5, 6, 11).
    EXPECT_FLOAT_EQ(out[0],  5.0f);
    EXPECT_FLOAT_EQ(out[1],  6.0f);
    EXPECT_FLOAT_EQ(out[2], 11.0f);
    // Token 1 is the second looked-up token (id=0) → expected (1, 2, 3).
    EXPECT_FLOAT_EQ(out[3], 1.0f);
    EXPECT_FLOAT_EQ(out[4], 2.0f);
    EXPECT_FLOAT_EQ(out[5], 3.0f);
}

// 2. Zero projection ⇒ zero output regardless of table contents.
TEST(PleTest, ZeroProjectionIsZero) {
    PleConfig cfg;
    cfg.vocab      = 4;
    cfg.ple_dim    = 3;
    cfg.hidden_dim = 5;
    cfg.n_tokens   = 3;
    cfg.tokens     = { 0, 2, 3 };
    cfg.table.assign(cfg.vocab * cfg.ple_dim, 1.0f);  // arbitrary non-zero
    cfg.proj.assign(cfg.hidden_dim * cfg.ple_dim, 0.0f);

    auto out = run_ple(cfg);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f) << "non-zero at flat index " << i;
    }
}

// 3. ple_dim == hidden_dim and proj == identity ⇒ output is the
//    looked-up table row, i.e. equivalent to a plain ggml_get_rows.
TEST(PleTest, IdentityProjection) {
    constexpr int D = 4;
    PleConfig cfg;
    cfg.vocab      = 3;
    cfg.ple_dim    = D;
    cfg.hidden_dim = D;
    cfg.n_tokens   = 2;
    cfg.tokens     = { 1, 2 };
    cfg.table = {
        // vocab 0: (10, 20, 30, 40)
        10.0f, 20.0f, 30.0f, 40.0f,
        // vocab 1: (1, 2, 3, 4)
         1.0f,  2.0f,  3.0f,  4.0f,
        // vocab 2: (-1, -2, -3, -4)
        -1.0f, -2.0f, -3.0f, -4.0f,
    };
    cfg.proj.assign(D * D, 0.0f);
    for (int i = 0; i < D; ++i) cfg.proj[i * D + i] = 1.0f;  // identity

    auto out = run_ple(cfg);
    ASSERT_EQ(out.size(), static_cast<size_t>(D * cfg.n_tokens));
    // token 0 → vocab 1 → (1, 2, 3, 4)
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);
    // token 1 → vocab 2 → (-1, -2, -3, -4)
    EXPECT_FLOAT_EQ(out[4], -1.0f);
    EXPECT_FLOAT_EQ(out[5], -2.0f);
    EXPECT_FLOAT_EQ(out[6], -3.0f);
    EXPECT_FLOAT_EQ(out[7], -4.0f);
}
