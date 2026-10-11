// test_bidirectional_mask_input.cpp — co-located test for
// src/graph_inputs/bidirectional_mask_input.cpp (the encoder mask).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/bidirectional_mask_input.h"

namespace {

// Allocates one [n, n] F32 mask named `name` on the CPU backend, fills it
// through `in`, and returns its contents.
std::vector<float> fill(BidirectionalMaskInput in, uint32_t n, const char* name) {
    ggml_init_params p{ ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_set_input(mask);
    ggml_set_name(mask, name);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mask);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::vector<int32_t> toks(n, 0);
    StepContext step;
    step.gf = gf;
    step.tokens = &toks;
    step.pos = 0;
    in.set_input(step);

    std::vector<float> got((size_t)n * n);
    ggml_backend_tensor_get(mask, got.data(), 0, got.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return got;
}

}  // namespace

// Every query sees every key, including keys AFTER it — what a causal mask
// never allows.
TEST(BidirectionalMaskInput, OpenSeesEveryKey) {
    const uint32_t n = 7;
    const auto got = fill(BidirectionalMaskInput::open("kq_mask_global"), n, "kq_mask_global");
    for (uint32_t r = 0; r < n; ++r)
        for (uint32_t j = 0; j < n; ++j)
            EXPECT_EQ(got[r * n + j], 0.0f) << "r=" << r << " j=" << j;
}

// Radius R: key j visible from row q iff |q - j| <= R (HF's inclusive radius).
// Checked at the edge on both sides: R visible, R + 1 masked.
TEST(BidirectionalMaskInput, BandIsInclusiveRadiusBothWays) {
    const uint32_t n = 20, R = 4;
    const auto got = fill(BidirectionalMaskInput::band("kq_mask_swa", R), n, "kq_mask_swa");
    for (uint32_t q = 0; q < n; ++q)
        for (uint32_t j = 0; j < n; ++j) {
            const int64_t d = (int64_t)q - (int64_t)j;
            const float want = (d >= -(int64_t)R && d <= (int64_t)R) ? 0.0f : -INFINITY;
            EXPECT_EQ(got[q * n + j], want) << "q=" << q << " j=" << j;
        }
    EXPECT_EQ(got[10 * n + 14], 0.0f);        // +4
    EXPECT_EQ(got[10 * n + 15], -INFINITY);   // +5
    EXPECT_EQ(got[10 * n + 6], 0.0f);         // -4
    EXPECT_EQ(got[10 * n + 5], -INFINITY);    // -5
}

TEST(BidirectionalMaskInput, BandRefusesZeroRadius) {
    EXPECT_THROW(BidirectionalMaskInput::band("kq_mask_swa", 0), std::runtime_error);
}
