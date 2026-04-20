// test_attention.cpp — PR 2.4 / PR 2.8
//
// Unit tests for AttentionLayer::build().
// Uses synthetic weight tensors (small n_embd) — no model file required.
// Tests: build succeeds for both phases, output shape is correct, outputs
//        are finite, and scratch budget is recorded.
//
// Oracle: self-consistency — a single-token prefill followed by a single-token
// decode produces a finite output. Full logit-level oracle against llama.cpp
// lives in the integration test suite.
//
// Run: ./qwen3-layer-tests --gtest_filter="AttentionLayer*"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "../../src/layers/attention.h"
#include "../../src/layers/layer.h"
#include "../../src/state/kv_cache_simple.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// Small synthetic dimensions — large enough to exercise GQA reshaping.
static constexpr int N_EMBD      = 32;
static constexpr int N_HEAD      = 2;
static constexpr int N_HEAD_KV   = 2;
static constexpr int N_EMBD_HEAD = N_EMBD / N_HEAD;   // 16
static constexpr int N_CTX_MAX   = 64;
static constexpr int N_TOKENS    = 4;
static constexpr int IL          = 0;  // single layer

// ── Fixture ───────────────────────────────────────────────────────────────────

class AttentionLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        // Weight context: holds synthetic weight tensors.
        ggml_init_params wp = {
            .mem_size   = 256 * ggml_tensor_overhead() + 8 * N_EMBD * N_EMBD * sizeof(float),
            .mem_buffer = nullptr,
            .no_alloc   = false,
        };
        wctx_ = ggml_init(wp);
        ASSERT_NE(wctx_, nullptr);

        // Allocate synthetic weight matrices and fill with small values to
        // keep outputs numerically stable (near-identity + small perturbation).
        auto make_weight = [&](int rows, int cols) -> ggml_tensor* {
            ggml_tensor* t = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, cols, rows);
            float* d = (float*)t->data;
            for (int i = 0; i < rows * cols; ++i)
                d[i] = (i == (i / cols) * cols + (i % cols) && i / cols < cols) ? 0.1f : 0.0f;
            return t;
        };

        w_q_     = make_weight(N_EMBD, N_EMBD);
        w_k_     = make_weight(N_HEAD_KV * N_EMBD_HEAD, N_EMBD);
        w_v_     = make_weight(N_HEAD_KV * N_EMBD_HEAD, N_EMBD);
        w_out_   = make_weight(N_EMBD, N_EMBD);
        w_qnorm_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, N_EMBD_HEAD);
        w_knorm_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, N_EMBD_HEAD);
        {
            float* d = (float*)w_qnorm_->data;
            for (int i = 0; i < N_EMBD_HEAD; ++i) d[i] = 1.0f;
            d = (float*)w_knorm_->data;
            for (int i = 0; i < N_EMBD_HEAD; ++i) d[i] = 1.0f;
        }

        kv_cache_ = std::make_unique<simple_kv_cache>(
            /*n_layers=*/1, N_CTX_MAX, /*n_batch_max=*/2,
            N_HEAD_KV * N_EMBD_HEAD, N_HEAD_KV * N_EMBD_HEAD,
            GGML_TYPE_F32, GGML_TYPE_F32, backend_);

        AttentionLayer::Hparams hp;
        hp.n_head       = N_HEAD;
        hp.n_head_kv    = N_HEAD_KV;
        hp.n_embd_head  = N_EMBD_HEAD;
        hp.kq_scale     = 1.0f / sqrtf(float(N_EMBD_HEAD));
        hp.n_rot        = N_EMBD_HEAD;
        hp.freq_base    = 10000.0f;
        hp.context_len  = N_CTX_MAX;
        hp.rms_norm_eps = 1e-6f;
        hp.has_q_norm   = true;
        hp.has_bias     = false;

        layer_ = std::make_unique<AttentionLayer>(
            w_q_, w_k_, w_v_, w_out_,
            w_qnorm_, w_knorm_,
            /*q_bias=*/nullptr, /*k_bias=*/nullptr, /*v_bias=*/nullptr,
            kv_cache_.get(), hp);
    }

    void TearDown() override {
        layer_.reset();
        kv_cache_.reset();
        if (wctx_)   ggml_free(wctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    // Build a compute context, allocate, execute, and return output float data.
    std::vector<float> run_prefill(int n_tokens) {
        const size_t ctx_bytes = 512 * ggml_tensor_overhead()
                                + 32 * n_tokens * N_EMBD * sizeof(float);
        ggml_init_params p = {ctx_bytes, nullptr, true};
        ggml_context* ctx = ggml_init(p);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        // Synthetic input: ones.
        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n_tokens);
        ggml_set_input(input);
        ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp_pos);

        AttentionLayer::PrefillArgs args;
        args.inp_pos  = inp_pos;
        args.n_tokens = static_cast<uint32_t>(n_tokens);
        args.slot_idx = 0;

        ggml_tensor* out = layer_->build(ctx, gf, input, IL, Phase::Prefill, args);
        ggml_build_forward_expand(gf, out);

        // Allocate and compute.
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        ggml_gallocr_alloc_graph(alloc, gf);

        // Record scratch budget (PR 2.8 requirement).
        size_t scratch = ggml_gallocr_get_buffer_size(alloc, 0);
        RecordProperty("scratch_bytes", static_cast<int64_t>(scratch));

        // Set inputs.
        {
            std::vector<float> inp(N_EMBD * n_tokens, 1.0f);
            ggml_backend_tensor_set(input, inp.data(), 0, inp.size() * sizeof(float));
            std::vector<int32_t> pos(n_tokens);
            std::iota(pos.begin(), pos.end(), 0);
            ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
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
    ggml_tensor*   w_q_     = nullptr;
    ggml_tensor*   w_k_     = nullptr;
    ggml_tensor*   w_v_     = nullptr;
    ggml_tensor*   w_out_   = nullptr;
    ggml_tensor*   w_qnorm_ = nullptr;
    ggml_tensor*   w_knorm_ = nullptr;
    std::unique_ptr<simple_kv_cache> kv_cache_;
    std::unique_ptr<AttentionLayer>  layer_;
};

// ── Build tests ───────────────────────────────────────────────────────────────

TEST_F(AttentionLayerTest, PrefillBuildsAndProducesFiniteOutput) {
    std::vector<float> out = run_prefill(N_TOKENS);
    ASSERT_EQ(out.size(), static_cast<size_t>(N_EMBD * N_TOKENS));
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v)) << "non-finite value in prefill output";
}

TEST_F(AttentionLayerTest, OutputShapeMatchesInput) {
    // Output of AttentionLayer::build() is [N_EMBD, n_tokens].
    std::vector<float> out = run_prefill(1);
    EXPECT_EQ(out.size(), static_cast<size_t>(N_EMBD));
}

TEST_F(AttentionLayerTest, PrefillAdvancesThenDecodeBuilds) {
    // Prime the KV cache with a 2-token prefill.
    run_prefill(2);  // advances cache to pos 2

    // Now run a single-token decode.
    const size_t ctx_bytes = 512 * ggml_tensor_overhead()
                            + 32 * N_EMBD * sizeof(float);
    ggml_init_params p = {ctx_bytes, nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf   = ggml_new_graph(ctx);

    ggml_tensor* input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, 1);
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(input);
    ggml_set_input(inp_pos);

    uint32_t     slot_vec[1]  = {0};
    int32_t      pos_vec[1]   = {2};

    // Build a simple kq_mask for the decode (1 query, 3 kv positions including new one).
    uint32_t n_kv = kv_cache_->get_pos(0) + 1;  // 3
    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, 1, 1, 1);
    ggml_set_input(kq_mask);

    // Gather indices: trivial [0,1,2] for this test.
    ggml_tensor* gather_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
    ggml_set_input(gather_idx);

    const std::vector<uint32_t> slots    = {0};
    const std::vector<int32_t>  positions = {2};

    AttentionLayer::DecodeArgs dargs;
    dargs.inp_pos        = inp_pos;
    dargs.slots          = &slots;
    dargs.positions      = &positions;
    dargs.kq_mask        = kq_mask;
    dargs.gather_indices = gather_idx;

    AttentionLayer::PrefillArgs dummy_prefill{};
    ggml_tensor* out = layer_->build(ctx, gf, input, IL, Phase::Decode,
                                     dummy_prefill, &dargs);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    ggml_gallocr_alloc_graph(alloc, gf);

    // Set inputs.
    {
        std::vector<float> inp(N_EMBD, 1.0f);
        ggml_backend_tensor_set(input, inp.data(), 0, inp.size() * sizeof(float));
        int32_t pos_val = 2;
        ggml_backend_tensor_set(inp_pos, &pos_val, 0, sizeof(int32_t));
        std::vector<float> mask(n_kv, 0.0f);  // no masking (upper triangle handled internally)
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
        std::vector<int32_t> gidx(n_kv);
        std::iota(gidx.begin(), gidx.end(), 0);
        ggml_backend_tensor_set(gather_idx, gidx.data(), 0, gidx.size() * sizeof(int32_t));
    }

    ggml_backend_graph_compute(backend_, gf);

    std::vector<float> result(N_EMBD);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    for (float v : result)
        EXPECT_TRUE(std::isfinite(v)) << "non-finite value in decode output";

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}
