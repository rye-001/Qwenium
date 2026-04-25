// test_transformer_block.cpp
//
// Unit tests for build_transformer_layer.
// Uses synthetic weight tensors (small n_embd) — no model file required.
// Tests:
//   1. BuildSucceeds          — graph builds without error for a Qwen3-style layer.
//   2. OutputFinite           — computed output is finite for all tokens.
//   3. OutputShapeCorrect     — output shape matches [n_embd, n_tokens].
//   4. ScratchBudgetRecorded  — ggml_gallocr_get_buffer_size is recorded (regression gate).
//   5. Qwen2StyleBiasPath     — is_qwen2=true path builds and produces finite output.
//
// Run: build target qwen3-transformer-block-tests, then invoke the binary.

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "../../src/layers/transformer_block.h"
#include "../../src/state/kv_cache_simple.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// Small synthetic dimensions — large enough to exercise GQA reshaping.
static constexpr int N_EMBD      = 32;
static constexpr int N_HEAD      = 2;
static constexpr int N_HEAD_KV   = 2;
static constexpr int N_EMBD_HEAD = N_EMBD / N_HEAD;  // 16
static constexpr int N_FFN       = 64;
static constexpr int N_CTX_MAX   = 64;
static constexpr int N_TOKENS    = 4;
static constexpr int IL          = 0;

// ── Fixture ───────────────────────────────────────────────────────────────────

class TransformerBlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        // Weight context: holds all synthetic weight tensors (no_alloc=false).
        const size_t weight_bytes = 64 * ggml_tensor_overhead()
            + (2*N_EMBD*N_EMBD + 2*N_HEAD_KV*N_EMBD_HEAD*N_EMBD
               + 3*N_FFN*N_EMBD
               + 4*N_EMBD + 2*N_EMBD_HEAD) * sizeof(float);
        ggml_init_params wp = { weight_bytes, nullptr, false };
        wctx_ = ggml_init(wp);
        ASSERT_NE(wctx_, nullptr);

        auto make_2d = [&](int rows, int cols) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, cols, rows);
            float* d = (float*)t->data;
            // Small diagonal-ish initialisation keeps outputs numerically stable.
            for (int i = 0; i < rows * cols; ++i)
                d[i] = (i % (cols + 1) == 0) ? 0.05f : 0.0f;
            return t;
        };
        auto make_1d = [&](int n, float val = 1.0f) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, n);
            float* d = (float*)t->data;
            for (int i = 0; i < n; ++i) d[i] = val;
            return t;
        };

        w_.attn_norm = make_1d(N_EMBD);
        w_.q         = make_2d(N_EMBD, N_EMBD);
        w_.k         = make_2d(N_HEAD_KV * N_EMBD_HEAD, N_EMBD);
        w_.v         = make_2d(N_HEAD_KV * N_EMBD_HEAD, N_EMBD);
        w_.q_bias    = nullptr;
        w_.k_bias    = nullptr;
        w_.v_bias    = nullptr;
        w_.q_norm    = make_1d(N_EMBD_HEAD);
        w_.k_norm    = make_1d(N_EMBD_HEAD);
        w_.out       = make_2d(N_EMBD, N_EMBD);
        w_.ffn_norm  = make_1d(N_EMBD);
        w_.ffn_gate  = make_2d(N_FFN, N_EMBD);
        w_.ffn_up    = make_2d(N_FFN, N_EMBD);
        w_.ffn_down  = make_2d(N_EMBD, N_FFN);

        hp_.is_qwen2      = false;
        hp_.n_head        = N_HEAD;
        hp_.n_head_kv     = N_HEAD_KV;
        hp_.n_embd_head   = N_EMBD_HEAD;
        hp_.freq_base     = 10000.0f;
        hp_.context_length = N_CTX_MAX;
        hp_.rms_norm_eps  = 1e-6f;

        kv_cache_ = std::make_unique<simple_kv_cache>(
            /*n_layers=*/1, N_CTX_MAX, /*n_batch_max=*/2,
            N_HEAD_KV * N_EMBD_HEAD, N_HEAD_KV * N_EMBD_HEAD,
            GGML_TYPE_F32, GGML_TYPE_F32, backend_);
    }

    void TearDown() override {
        kv_cache_.reset();
        if (wctx_)    ggml_free(wctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    // Build + allocate + set inputs + compute; return output floats.
    // pos: starting token position (for causal mask)
    std::vector<float> run(int n_tokens, int pos = 0,
                           bool is_qwen2 = false) {
        TransformerBlockHparams hp = hp_;
        hp.is_qwen2 = is_qwen2;

        TransformerBlockWeights w = w_;
        if (is_qwen2) {
            // Provide non-null biases for the Qwen2 path.
            auto make_1d_bias = [&](int n) -> ggml_tensor* {
                ggml_tensor* t = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, n);
                float* d = (float*)t->data;
                for (int i = 0; i < n; ++i) d[i] = 0.01f;
                return t;
            };
            w.q_bias = make_1d_bias(N_EMBD);
            w.k_bias = make_1d_bias(N_HEAD_KV * N_EMBD_HEAD);
            w.v_bias = make_1d_bias(N_HEAD_KV * N_EMBD_HEAD);
            w.q_norm = nullptr;
            w.k_norm = nullptr;
        }

        const size_t ctx_bytes = 1024 * ggml_tensor_overhead()
                                + 64 * n_tokens * N_EMBD * sizeof(float);
        ggml_init_params p = { ctx_bytes, nullptr, true };
        ggml_context* ctx = ggml_init(p);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n_tokens);
        ggml_set_input(input);
        ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp_pos);

        ggml_tensor* out = build_transformer_layer(
            ctx, gf, kv_cache_.get(), input, inp_pos,
            w, hp, IL, /*slot_idx=*/0, static_cast<uint32_t>(n_tokens));
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        ggml_gallocr_alloc_graph(alloc, gf);

        // Record scratch budget (regression gate per task requirement).
        size_t scratch = ggml_gallocr_get_buffer_size(alloc, 0);
        RecordProperty("scratch_bytes", static_cast<int64_t>(scratch));

        // Set inputs.
        {
            std::vector<float> inp(N_EMBD * n_tokens, 0.5f);
            ggml_backend_tensor_set(input, inp.data(), 0, inp.size() * sizeof(float));
            std::vector<int32_t> pv(n_tokens);
            for (int i = 0; i < n_tokens; ++i) pv[i] = pos + i;
            ggml_backend_tensor_set(inp_pos, pv.data(), 0, pv.size() * sizeof(int32_t));
        }

        // Set the causal attention mask (looked up by name after allocation).
        {
            char mask_name[32];
            snprintf(mask_name, sizeof(mask_name), "kq_mask.%d", IL);
            ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, mask_name);
            EXPECT_NE(kq_mask, nullptr) << "kq_mask tensor not found in graph";
            if (kq_mask) {
                uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
                std::vector<float> mask(n_kv * n_tokens);
                for (int i = 0; i < n_tokens; ++i) {
                    int q_pos = pos + i;
                    for (uint32_t j = 0; j < n_kv; ++j)
                        mask[i * n_kv + j] = (static_cast<int>(j) <= q_pos) ? 0.0f : -INFINITY;
                }
                ggml_backend_tensor_set(kq_mask, mask.data(), 0,
                    n_kv * n_tokens * sizeof(float));
            }
        }

        ggml_backend_graph_compute(backend_, gf);
        kv_cache_->advance(static_cast<uint32_t>(n_tokens), 0);

        std::vector<float> result(N_EMBD * n_tokens);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_t backend_ = nullptr;
    ggml_context*  wctx_    = nullptr;
    TransformerBlockWeights  w_{};
    TransformerBlockHparams  hp_{};
    std::unique_ptr<simple_kv_cache> kv_cache_;
};

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_F(TransformerBlockTest, BuildSucceeds) {
    // Graph construction must not throw or crash.
    const size_t ctx_bytes = 1024 * ggml_tensor_overhead()
                            + 64 * N_TOKENS * N_EMBD * sizeof(float);
    ggml_init_params p = { ctx_bytes, nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf   = ggml_new_graph(ctx);

    ggml_tensor* input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_TOKENS);
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_TOKENS);
    ggml_set_input(input);
    ggml_set_input(inp_pos);

    ggml_tensor* out = build_transformer_layer(
        ctx, gf, kv_cache_.get(), input, inp_pos,
        w_, hp_, IL, 0, N_TOKENS);

    ASSERT_NE(out, nullptr);
    ggml_build_forward_expand(gf, out);
    EXPECT_GT(ggml_graph_n_nodes(gf), 0);

    ggml_free(ctx);
}

TEST_F(TransformerBlockTest, OutputFinite) {
    std::vector<float> out = run(N_TOKENS);
    ASSERT_EQ(out.size(), static_cast<size_t>(N_EMBD * N_TOKENS));
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v)) << "non-finite value in transformer layer output";
}

TEST_F(TransformerBlockTest, OutputShapeCorrect) {
    // Single token: output must be [N_EMBD].
    std::vector<float> out = run(1);
    EXPECT_EQ(out.size(), static_cast<size_t>(N_EMBD));
}

TEST_F(TransformerBlockTest, ScratchBudgetRecorded) {
    // Allocate a fresh cache for this test (avoid position collision with other tests).
    kv_cache_ = std::make_unique<simple_kv_cache>(
        1, N_CTX_MAX, 2,
        N_HEAD_KV * N_EMBD_HEAD, N_HEAD_KV * N_EMBD_HEAD,
        GGML_TYPE_F32, GGML_TYPE_F32, backend_);

    const size_t ctx_bytes = 1024 * ggml_tensor_overhead()
                            + 64 * N_TOKENS * N_EMBD * sizeof(float);
    ggml_init_params p = { ctx_bytes, nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf   = ggml_new_graph(ctx);

    ggml_tensor* input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_TOKENS);
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_TOKENS);
    ggml_set_input(input);
    ggml_set_input(inp_pos);

    ggml_tensor* out = build_transformer_layer(
        ctx, gf, kv_cache_.get(), input, inp_pos,
        w_, hp_, IL, 0, N_TOKENS);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend_));
    ggml_gallocr_alloc_graph(alloc, gf);

    size_t scratch = ggml_gallocr_get_buffer_size(alloc, 0);
    RecordProperty("scratch_bytes", static_cast<int64_t>(scratch));
    EXPECT_GT(scratch, 0u);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

TEST_F(TransformerBlockTest, Qwen2StyleBiasPath) {
    // Qwen2 path uses additive biases and skips QK norms.
    // Allocate a fresh cache so position is 0.
    kv_cache_ = std::make_unique<simple_kv_cache>(
        1, N_CTX_MAX, 2,
        N_HEAD_KV * N_EMBD_HEAD, N_HEAD_KV * N_EMBD_HEAD,
        GGML_TYPE_F32, GGML_TYPE_F32, backend_);

    std::vector<float> out = run(N_TOKENS, 0, /*is_qwen2=*/true);
    ASSERT_EQ(out.size(), static_cast<size_t>(N_EMBD * N_TOKENS));
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v)) << "non-finite value in Qwen2-style output";
}
