// test_ffn.cpp — PR 2.5 / PR 2.8
//
// Unit tests for build_ffn_swiglu.
// Verifies: correct output shape, finite outputs under both Phase values,
// and records the scratch budget.
//
// Run: ./qwen3-layer-tests --gtest_filter="FFN*"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../../src/layers/ffn.h"
#include "../../src/layers/layer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static constexpr int N_EMBD     = 32;
static constexpr int N_FFN_INTER = 64;  // intermediate dim (up/gate)
static constexpr int N_TOKENS   = 4;

class FFNTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        const size_t w_bytes = (N_EMBD * N_FFN_INTER + N_FFN_INTER * N_EMBD) * 2 * sizeof(float)
                             + 32 * ggml_tensor_overhead();
        ggml_init_params wp{w_bytes, nullptr, false};
        wctx_ = ggml_init(wp);
        ASSERT_NE(wctx_, nullptr);

        auto fill_weight = [&](ggml_tensor* t, float val) {
            float* d = (float*)t->data;
            int n = (int)(ggml_nbytes(t) / sizeof(float));
            for (int i = 0; i < n; ++i) d[i] = val;
        };

        // gate: [N_FFN_INTER, N_EMBD], up: [N_FFN_INTER, N_EMBD], down: [N_EMBD, N_FFN_INTER]
        w_gate_ = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, N_EMBD, N_FFN_INTER);
        w_up_   = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, N_EMBD, N_FFN_INTER);
        w_down_ = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, N_FFN_INTER, N_EMBD);
        fill_weight(w_gate_, 0.01f);
        fill_weight(w_up_,   0.01f);
        fill_weight(w_down_,  0.01f);
    }

    void TearDown() override {
        if (wctx_)    ggml_free(wctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    std::vector<float> run(Phase phase) {
        // 512 KB covers ggml_new_graph's node-table overhead + tensor metadata + data.
        const size_t ctx_bytes = 512 * 1024;
        ggml_init_params p{ctx_bytes, nullptr, true};
        ggml_context* ctx = ggml_init(p);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, N_TOKENS);
        ggml_set_input(input);

        ggml_tensor* out = build_ffn_swiglu(ctx, gf, input,
                                            w_gate_, w_up_, w_down_,
                                            /*il=*/0, phase);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        ggml_gallocr_alloc_graph(alloc, gf);

        RecordProperty("scratch_bytes", static_cast<int64_t>(ggml_gallocr_get_buffer_size(alloc, 0)));

        std::vector<float> inp(N_EMBD * N_TOKENS, 1.0f);
        ggml_backend_tensor_set(input, inp.data(), 0, inp.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, gf);

        std::vector<float> result(N_EMBD * N_TOKENS);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_t backend_ = nullptr;
    ggml_context*  wctx_    = nullptr;
    ggml_tensor*   w_gate_  = nullptr;
    ggml_tensor*   w_up_    = nullptr;
    ggml_tensor*   w_down_  = nullptr;
};

TEST_F(FFNTest, PrefillProducesCorrectShapeAndFiniteOutput) {
    auto out = run(Phase::Prefill);
    ASSERT_EQ(out.size(), static_cast<size_t>(N_EMBD * N_TOKENS));
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v));
}

TEST_F(FFNTest, DecodeProducesCorrectShapeAndFiniteOutput) {
    // Dense FFN has one graph shape — Phase::Decode output is identical.
    auto out = run(Phase::Decode);
    ASSERT_EQ(out.size(), static_cast<size_t>(N_EMBD * N_TOKENS));
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v));
}

TEST_F(FFNTest, PrefillAndDecodeOutputsMatch) {
    // The Phase argument is accepted but ignored for Dense FFN.
    auto p = run(Phase::Prefill);
    auto d = run(Phase::Decode);
    ASSERT_EQ(p.size(), d.size());
    for (size_t i = 0; i < p.size(); ++i)
        EXPECT_FLOAT_EQ(p[i], d[i]) << "mismatch at i=" << i;
}
