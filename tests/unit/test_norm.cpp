// test_norm.cpp
//
// Unit tests for build_rms_norm (src/layers/norm.cpp).
// Verifies: correct output values against hand-computed reference,
// tensor naming with and without layer index, finite outputs, and
// records the scratch budget per the Phase 2 PR 2.8 pattern.
//
// Run: ./qwen3-norm-tests

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../src/layers/norm.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// Hand-computed reference for build_rms_norm([3, 4], weight=[2, 0.5], eps=0):
//   mean_sq = (9 + 16) / 2 = 12.5
//   rms     = sqrt(12.5)  ≈ 3.535534
//   normed  = [3/rms, 4/rms] = [0.848528, 1.131371]
//   scaled  = [0.848528*2, 1.131371*0.5] = [1.697056, 0.565685]
static constexpr float REF_A = 1.697056f;
static constexpr float REF_B = 0.565685f;
static constexpr float TOLS  = 1e-4f;

class NormTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        // Weight context (persistent across graph builds)
        const size_t w_bytes = 2 * sizeof(float) + 8 * ggml_tensor_overhead();
        ggml_init_params wp{w_bytes, nullptr, false};
        wctx_ = ggml_init(wp);
        ASSERT_NE(wctx_, nullptr);

        w_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, 2);
        float wdata[2] = {2.0f, 0.5f};
        memcpy(w_->data, wdata, sizeof(wdata));
    }

    void TearDown() override {
        if (wctx_)    ggml_free(wctx_);
        if (backend_) ggml_backend_free(backend_);
    }

    // Build and execute a single build_rms_norm call; return output values.
    std::vector<float> run(float eps, int il,
                           const char** out_tensor_name = nullptr) {
        const size_t ctx_bytes = 256 * 1024;
        ggml_init_params p{ctx_bytes, nullptr, true};
        ggml_context* ctx = ggml_init(p);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
        ggml_set_input(input);

        ggml_tensor* out = build_rms_norm(ctx, input, w_, eps, il);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        ggml_gallocr_alloc_graph(alloc, gf);

        RecordProperty("scratch_bytes",
            static_cast<int64_t>(ggml_gallocr_get_buffer_size(alloc, 0)));

        float indata[2] = {3.0f, 4.0f};
        ggml_backend_tensor_set(input, indata, 0, sizeof(indata));
        ggml_backend_graph_compute(backend_, gf);

        std::vector<float> result(2);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

        // Locate the intermediate normed tensor by name to verify naming.
        if (out_tensor_name) {
            // Find the rms_norm tensor just before the mul output.
            // It should be named "cur_rms_normed[.il]".
            char expected[128];
            if (il >= 0)
                snprintf(expected, sizeof(expected), "cur_rms_normed.%d", il);
            else
                snprintf(expected, sizeof(expected), "cur_rms_normed");
            *out_tensor_name = ggml_graph_get_tensor(gf, expected)
                                   ? expected : nullptr;
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_t backend_ = nullptr;
    ggml_context*  wctx_    = nullptr;
    ggml_tensor*   w_       = nullptr;
};

// Known-answer: verify output matches hand-computed reference.
TEST_F(NormTest, KnownAnswerEpsZero) {
    auto out = run(0.0f, /*il=*/-1);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_NEAR(out[0], REF_A, TOLS);
    EXPECT_NEAR(out[1], REF_B, TOLS);
}

// A small epsilon must not change the answer materially.
TEST_F(NormTest, KnownAnswerSmallEps) {
    auto out = run(1e-6f, /*il=*/-1);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_NEAR(out[0], REF_A, TOLS);
    EXPECT_NEAR(out[1], REF_B, TOLS);
}

// Tensor naming with layer index: intermediate should be "cur_rms_normed.3".
TEST_F(NormTest, TensorNamingWithLayerIndex) {
    const char* found_name = nullptr;
    run(0.0f, /*il=*/3, &found_name);
    EXPECT_NE(found_name, nullptr)
        << "expected tensor named 'cur_rms_normed.3' in graph";
}

// Tensor naming without layer index: intermediate should be "cur_rms_normed".
TEST_F(NormTest, TensorNamingWithoutLayerIndex) {
    const char* found_name = nullptr;
    run(0.0f, /*il=*/-1, &found_name);
    EXPECT_NE(found_name, nullptr)
        << "expected tensor named 'cur_rms_normed' in graph";
}

// All-ones input + all-ones weight → output should be all-ones (exact).
TEST_F(NormTest, AllOnesInputAndWeight) {
    // Override weight to [1, 1].
    float ones[2] = {1.0f, 1.0f};
    memcpy(w_->data, ones, sizeof(ones));

    const size_t ctx_bytes = 256 * 1024;
    ggml_init_params p{ctx_bytes, nullptr, true};
    ggml_context* ctx = ggml_init(p);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_set_input(input);

    ggml_tensor* out = build_rms_norm(ctx, input, w_, 0.0f, /*il=*/0);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend_));
    ggml_gallocr_alloc_graph(alloc, gf);

    float indata[2] = {1.0f, 1.0f};
    ggml_backend_tensor_set(input, indata, 0, sizeof(indata));
    ggml_backend_graph_compute(backend_, gf);

    float result[2];
    ggml_backend_tensor_get(out, result, 0, sizeof(result));

    EXPECT_NEAR(result[0], 1.0f, 1e-6f);
    EXPECT_NEAR(result[1], 1.0f, 1e-6f);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    // Restore weight for subsequent tests.
    float wdata[2] = {2.0f, 0.5f};
    memcpy(w_->data, wdata, sizeof(wdata));
}

// Finite-output guard: no NaN or Inf in result.
TEST_F(NormTest, OutputIsFinite) {
    auto out = run(1e-5f, /*il=*/0);
    for (float v : out)
        EXPECT_TRUE(std::isfinite(v)) << "non-finite output: " << v;
}
