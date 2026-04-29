// test_gemma2_softcap.cpp
//
// Unit tests for build_softcap (src/layers/attention.cpp, PR G2.4).
//
// Oracle: cap * tanh(x / cap)
// For x=2.0, cap=50.0:
//   tanh(2/50) = tanh(0.04) ≈ 0.039979
//   result     = 50 * 0.039979 ≈ 1.999867
// For x=50.0, cap=50.0:
//   tanh(50/50) = tanh(1.0) ≈ 0.761594
//   result      = 50 * 0.761594 = 38.0797
// For x=0.0: result = 0.0 (tanh(0) = 0).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../../src/layers/attention.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static constexpr float CAP  = 50.0f;
static constexpr float TOLS = 1e-4f;

class SoftcapTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);
    }
    void TearDown() override {
        if (backend_) ggml_backend_free(backend_);
    }

    // Allocate a 1-D float context with n elements, run build_softcap(cap),
    // fill with `values`, compute, and return output.
    std::vector<float> run(const std::vector<float>& values, float cap) {
        const size_t n = values.size();
        const size_t ctx_bytes = 256 * 1024;
        ggml_init_params p{ctx_bytes, nullptr, true};
        ggml_context* ctx = ggml_init(p);
        EXPECT_NE(ctx, nullptr);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)n);
        ggml_set_input(x);
        ggml_tensor* out = build_softcap(ctx, x, cap);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(x, values.data(), 0, n * sizeof(float));
        ggml_backend_graph_compute(backend_, gf);

        std::vector<float> result(n);
        ggml_backend_tensor_get(out, result.data(), 0, n * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_t backend_ = nullptr;
};

// Zero input → zero output (tanh(0) = 0).
TEST_F(SoftcapTest, ZeroInput) {
    auto out = run({0.0f}, CAP);
    EXPECT_NEAR(out[0], 0.0f, TOLS);
}

// Small positive value: cap * tanh(x/cap) ≈ x for x << cap.
TEST_F(SoftcapTest, SmallPositiveApproxIdentity) {
    const float x = 2.0f;
    // tanh(2/50) = tanh(0.04); reference computed with std::tanh
    const float ref = CAP * std::tanh(x / CAP);
    auto out = run({x}, CAP);
    EXPECT_NEAR(out[0], ref, TOLS);
}

// Large positive value: result saturates towards cap.
TEST_F(SoftcapTest, LargePositiveSaturates) {
    const float x = 50.0f;
    const float ref = CAP * std::tanh(x / CAP);
    auto out = run({x}, CAP);
    EXPECT_NEAR(out[0], ref, TOLS);
    // Must be strictly less than cap.
    EXPECT_LT(out[0], CAP);
}

// Negative input: antisymmetric — result equals -(positive result).
TEST_F(SoftcapTest, NegativeInput) {
    const float x = 10.0f;
    auto pos = run({ x}, CAP);
    auto neg = run({-x}, CAP);
    EXPECT_NEAR(neg[0], -pos[0], TOLS);
}

// Final-logit cap = 30.0: spot-check.
TEST_F(SoftcapTest, FinalLogitCap30) {
    const float cap = 30.0f;
    const float x = 1.0f;
    const float ref = cap * std::tanh(x / cap);
    auto out = run({x}, cap);
    EXPECT_NEAR(out[0], ref, TOLS);
}

// Multiple elements processed independently.
TEST_F(SoftcapTest, MultipleElements) {
    std::vector<float> xs = {-100.0f, -1.0f, 0.0f, 1.0f, 100.0f};
    auto out = run(xs, CAP);
    ASSERT_EQ(out.size(), xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        const float ref = CAP * std::tanh(xs[i] / CAP);
        EXPECT_NEAR(out[i], ref, TOLS) << "index " << i;
    }
}
