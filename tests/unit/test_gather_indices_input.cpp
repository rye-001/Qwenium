// test_gather_indices_input.cpp — both stride policies:
//   NCtxMax: index = slot * n_ctx_max + t   (qwen3/qwen35)
//   NKvLen : index = slot * n_kv_len  + t   (qwen36)

#include <gtest/gtest.h>

#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/gather_indices_input.h"

namespace {
struct H {
    ggml_context* ctx; ggml_backend_t be; ggml_backend_buffer_t buf;
    ggml_cgraph* gf; ggml_tensor* t;
    explicit H(size_t n) {
        ggml_init_params p{ ggml_tensor_overhead()*4 + ggml_graph_overhead(),
                            nullptr, true };
        ctx = ggml_init(p);
        t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(t); ggml_set_name(t, "gather_indices");
        gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, t);
        be = ggml_backend_cpu_init();
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    }
    ~H(){ ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx); }
};
}

TEST(GatherIndicesInput, NCtxMaxStride) {
    const uint32_t n_batch = 2, n_kv_len = 4, n_ctx_max = 100;
    std::vector<int32_t> toks(n_batch, 0);
    std::vector<uint32_t> slots{1, 3};
    H h(n_batch * n_kv_len);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.slots = &slots;

    GatherIndicesInput in(n_ctx_max);
    in.set_input(step);

    std::vector<int32_t> got(n_batch * n_kv_len);
    ggml_backend_tensor_get(h.t, got.data(), 0, got.size()*sizeof(int32_t));
    for (uint32_t b = 0; b < n_batch; ++b)
        for (uint32_t k = 0; k < n_kv_len; ++k)
            EXPECT_EQ(got[b*n_kv_len + k],
                      (int32_t)(slots[b]*n_ctx_max + k));
}

// The regression this file exists to prevent. A second stride policy
// (slot * n_kv_len + t) used to be selectable and qwen36 selected it; it agrees
// with the correct one at slot 0 and diverges everywhere else, which is why it
// survived — every qwen36 run was single-slot. Pin the multi-slot rows against
// the cache's real layout: gather_k flattens to [n_embd, n_ctx_max*n_batch_max].
TEST(GatherIndicesInput, MultiSlotRowsUseTheCacheNCtxMaxStride) {
    const uint32_t n_batch = 3, n_kv_len = 5, n_ctx_max = 64;
    std::vector<int32_t> toks(n_batch, 0);
    std::vector<uint32_t> slots{0, 2, 4};
    H h(n_batch * n_kv_len);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.slots = &slots;

    GatherIndicesInput in(n_ctx_max);
    in.set_input(step);

    std::vector<int32_t> got(n_batch * n_kv_len);
    ggml_backend_tensor_get(h.t, got.data(), 0, got.size()*sizeof(int32_t));
    for (uint32_t b = 0; b < n_batch; ++b)
        for (uint32_t k = 0; k < n_kv_len; ++k)
            EXPECT_EQ(got[b*n_kv_len + k], (int32_t)(slots[b]*n_ctx_max + k))
                << "slot " << slots[b] << " row " << k
                << " must index the cache by n_ctx_max, not n_kv_len";

    // Slot 0 is where the old wrong policy agreed with the right one — the
    // reason the defect stayed latent. Assert it explicitly.
    for (uint32_t k = 0; k < n_kv_len; ++k)
        EXPECT_EQ(got[k], (int32_t)k) << "slot 0 must be identity";
}

TEST(GatherIndicesInput, FailLoudWhenSlotsNull) {
    std::vector<int32_t> toks(2, 0);
    H h(8);
    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.slots = nullptr;
    GatherIndicesInput in(64);
    EXPECT_THROW(in.set_input(step), std::runtime_error);
}
