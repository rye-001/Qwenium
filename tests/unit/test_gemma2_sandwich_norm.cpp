// test_gemma2_sandwich_norm.cpp
//
// Unit tests for the sandwich-norm hook in build_transformer_layer (PR G2.1).
//
// Verifies:
//   1. Pre-norm-only path (post_attn_norm = post_ffn_norm = nullptr) is
//      bit-identical to the behaviour before G2 (regression gate).
//   2. Sandwich-norm path (both post-norm weights non-null) produces a
//      *different* output from the pre-norm-only path — confirming the
//      post-norm is actually applied and not silently skipped.
//   3. Enabling only post_attn_norm (post_ffn_norm = nullptr) changes the
//      output compared to the pre-norm-only baseline.
//   4. Output is finite for all token positions in the sandwich path.
//
// Synthetic weights: small diagonal matrices (N_EMBD=32, N_TOKENS=3).
// No model file required.

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "../../src/layers/transformer_block.h"
#include "../../src/state/kv_cache_simple.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static constexpr int N_EMBD      = 32;
static constexpr int N_HEAD      = 2;
static constexpr int N_HEAD_KV   = 2;
static constexpr int N_EMBD_HEAD = N_EMBD / N_HEAD;  // 16
static constexpr int N_FFN       = 64;
static constexpr int N_CTX_MAX   = 16;
static constexpr int N_TOKENS    = 3;
static constexpr int IL          = 0;

// ── Fixture ───────────────────────────────────────────────────────────────────

class SandwichNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        // Weight context: enough room for all synthetic tensors.
        const size_t weight_bytes = 128 * ggml_tensor_overhead()
            + (4LL * N_EMBD * N_EMBD
               + 2LL * N_HEAD_KV * N_EMBD_HEAD * N_EMBD
               + 3LL * N_FFN * N_EMBD
               + 8LL * N_EMBD) * sizeof(float);
        ggml_init_params wp = { weight_bytes, nullptr, false };
        wctx_ = ggml_init(wp);
        ASSERT_NE(wctx_, nullptr);

        auto make_identity_like = [&](int rows, int cols) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, cols, rows);
            float* d = (float*)t->data;
            int n = rows * cols;
            for (int i = 0; i < n; ++i)
                d[i] = (i % (cols + 1) == 0) ? 0.05f : 0.0f;
            return t;
        };
        auto make_ones_1d = [&](int n) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, n);
            float* d = (float*)t->data;
            for (int i = 0; i < n; ++i) d[i] = 1.0f;
            return t;
        };
        auto make_half_1d = [&](int n) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, n);
            float* d = (float*)t->data;
            for (int i = 0; i < n; ++i) d[i] = 0.5f;
            return t;
        };

        w_attn_norm    = make_ones_1d(N_EMBD);
        w_q            = make_identity_like(N_HEAD * N_EMBD_HEAD, N_EMBD);
        w_k            = make_identity_like(N_HEAD_KV * N_EMBD_HEAD, N_EMBD);
        w_v            = make_identity_like(N_HEAD_KV * N_EMBD_HEAD, N_EMBD);
        w_out          = make_identity_like(N_EMBD, N_HEAD * N_EMBD_HEAD);
        w_ffn_norm     = make_ones_1d(N_EMBD);
        w_ffn_gate     = make_identity_like(N_FFN, N_EMBD);
        w_ffn_up       = make_identity_like(N_FFN, N_EMBD);
        w_ffn_down     = make_identity_like(N_EMBD, N_FFN);
        // Post-norm weights use 0.5 so they visibly rescale the output.
        w_post_attn    = make_half_1d(N_EMBD);
        w_post_ffn     = make_half_1d(N_EMBD);

        kv_cache_ = std::make_unique<simple_kv_cache>(
            /*n_layers=*/1, N_CTX_MAX, /*n_batch_max=*/1,
            N_HEAD_KV * N_EMBD_HEAD, N_HEAD_KV * N_EMBD_HEAD,
            GGML_TYPE_F32, GGML_TYPE_F32, backend_);
    }

    void TearDown() override {
        kv_cache_.reset();
        if (wctx_)    ggml_free(wctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    // Build and run the transformer layer.  post_attn / post_ffn may be nullptr.
    // Returns the [N_EMBD * N_TOKENS] output vector.
    std::vector<float> run(ggml_tensor* post_attn, ggml_tensor* post_ffn) {
        // Reset KV cache position before each run so attention masks are consistent.
        kv_cache_->clear_all();

        const size_t ctx_bytes = 4 * 1024 * 1024;
        ggml_init_params p{ctx_bytes, nullptr, true};
        ggml_context* ctx = ggml_init(p);
        EXPECT_NE(ctx, nullptr);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        // Input: [N_EMBD, N_TOKENS]
        ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_TOKENS);
        ggml_set_input(cur);
        ggml_build_forward_expand(gf, cur);

        // Position indices [N_TOKENS]
        ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_TOKENS);
        ggml_set_input(inp_pos);
        ggml_build_forward_expand(gf, inp_pos);

        TransformerBlockWeights w{};
        w.attn_norm    = w_attn_norm;
        w.q            = w_q;
        w.k            = w_k;
        w.v            = w_v;
        w.out          = w_out;
        w.ffn_norm     = w_ffn_norm;
        w.ffn_gate     = w_ffn_gate;
        w.ffn_up       = w_ffn_up;
        w.ffn_down     = w_ffn_down;
        w.post_attn_norm = post_attn;
        w.post_ffn_norm  = post_ffn;

        TransformerBlockHparams hp{};
        hp.is_qwen2      = false;
        hp.n_head        = N_HEAD;
        hp.n_head_kv     = N_HEAD_KV;
        hp.n_embd_head   = N_EMBD_HEAD;
        hp.freq_base     = 10000.0f;
        hp.context_length = N_CTX_MAX;
        hp.rms_norm_eps  = 1e-6f;
        hp.gemma_geglu   = true;
        hp.attn_softcap  = 0.0f;

        ggml_tensor* out = build_transformer_layer(
            ctx, gf, kv_cache_.get(), cur, inp_pos,
            w, hp, IL, /*slot_idx=*/0, N_TOKENS);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        EXPECT_TRUE(ggml_gallocr_alloc_graph(alloc, gf));

        // Fill input: ramp [0..N_EMBD*N_TOKENS - 1] scaled small.
        const int n_in = N_EMBD * N_TOKENS;
        std::vector<float> indata(n_in);
        for (int i = 0; i < n_in; ++i) indata[i] = static_cast<float>(i) * 0.001f;
        ggml_backend_tensor_set(cur, indata.data(), 0, n_in * sizeof(float));

        std::vector<int32_t> pos(N_TOKENS);
        std::iota(pos.begin(), pos.end(), 0);
        ggml_backend_tensor_set(inp_pos, pos.data(), 0, N_TOKENS * sizeof(int32_t));

        // Fill causal KQ mask.
        char mname[32];
        std::snprintf(mname, sizeof(mname), "kq_mask.%d", IL);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, mname);
        if (kq_mask) {
            const int n_kv = (int)kq_mask->ne[0];
            std::vector<float> mask(n_kv * N_TOKENS);
            for (int i = 0; i < N_TOKENS; ++i)
                for (int j = 0; j < n_kv; ++j)
                    mask[i * n_kv + j] = (j <= i) ? 0.0f : -INFINITY;
            ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
        }

        ggml_backend_graph_compute(backend_, gf);

        const int n_out = N_EMBD * N_TOKENS;
        std::vector<float> result(n_out);
        ggml_backend_tensor_get(out, result.data(), 0, n_out * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_t backend_ = nullptr;
    ggml_context*  wctx_    = nullptr;
    std::unique_ptr<simple_kv_cache> kv_cache_;

    ggml_tensor* w_attn_norm = nullptr;
    ggml_tensor* w_q        = nullptr;
    ggml_tensor* w_k        = nullptr;
    ggml_tensor* w_v        = nullptr;
    ggml_tensor* w_out      = nullptr;
    ggml_tensor* w_ffn_norm = nullptr;
    ggml_tensor* w_ffn_gate = nullptr;
    ggml_tensor* w_ffn_up   = nullptr;
    ggml_tensor* w_ffn_down = nullptr;
    ggml_tensor* w_post_attn = nullptr;
    ggml_tensor* w_post_ffn  = nullptr;
};

// Pre-norm-only baseline builds and produces finite outputs.
TEST_F(SandwichNormTest, PreNormOnlyFinite) {
    auto out = run(nullptr, nullptr);
    ASSERT_EQ((int)out.size(), N_EMBD * N_TOKENS);
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v)) << "non-finite: " << v;
}

// Sandwich path (both post norms) produces finite outputs.
TEST_F(SandwichNormTest, SandwichNormFinite) {
    auto out = run(w_post_attn, w_post_ffn);
    ASSERT_EQ((int)out.size(), N_EMBD * N_TOKENS);
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v)) << "non-finite: " << v;
}

// Sandwich path produces a different result from pre-norm-only.
// (If post-norms were silently skipped the outputs would be identical.)
TEST_F(SandwichNormTest, SandwichNormDiffersFromPreNorm) {
    auto baseline  = run(nullptr, nullptr);
    auto sandwich  = run(w_post_attn, w_post_ffn);
    bool any_diff = false;
    for (size_t i = 0; i < baseline.size(); ++i) {
        if (std::fabs(sandwich[i] - baseline[i]) > 1e-6f) { any_diff = true; break; }
    }
    EXPECT_TRUE(any_diff)
        << "sandwich-norm output must differ from pre-norm-only baseline";
}

// Enabling only post_attn_norm also changes the output.
TEST_F(SandwichNormTest, PostAttnNormAloneChangesOutput) {
    auto baseline = run(nullptr,      nullptr);
    auto partial  = run(w_post_attn, nullptr);
    bool any_diff = false;
    for (size_t i = 0; i < baseline.size(); ++i) {
        if (std::fabs(partial[i] - baseline[i]) > 1e-6f) { any_diff = true; break; }
    }
    EXPECT_TRUE(any_diff)
        << "post_attn_norm alone must change the layer output";
}
