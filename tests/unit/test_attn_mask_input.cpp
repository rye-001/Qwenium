// test_attn_mask_input.cpp — AttnMaskInput produces the exact causal /
// sliding-window mask the recipes' hand-rolled loops did (bit-identical).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/attn_mask_input.h"

namespace {

struct Harness {
    ggml_context*        ctx  = nullptr;
    ggml_backend_t       be   = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph*         gf   = nullptr;
    ggml_tensor*         mask = nullptr;

    Harness(uint32_t n_kv, uint32_t n_rows, const char* name) {
        ggml_init_params p{ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
                            nullptr, true };
        ctx  = ggml_init(p);
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, 1, 1, n_rows);
        ggml_set_input(mask);
        ggml_set_name(mask, name);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, mask);
        be  = ggml_backend_cpu_init();
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    }
    ~Harness() {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(be);
        ggml_free(ctx);
    }
};

float ref(int64_t q_pos, uint32_t j, uint32_t window) {
    bool causal = (int64_t)j <= q_pos;
    bool in_win = (window == 0) ||
                  (q_pos - (int64_t)j < (int64_t)window);
    return (causal && in_win) ? 0.0f : -INFINITY;
}

}  // namespace

TEST(AttnMaskInput, PrefillCausalNoWindow) {
    const uint32_t n_kv = 6, n_rows = 4;
    Harness h(n_kv, n_rows, "kq_mask.0");

    std::vector<int32_t> toks(n_rows, 0);
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;
    step.pos = 0;  // contiguous: row r -> q_pos = r

    AttnMaskInput in("kq_mask.0", 0u);
    in.set_input(step);

    std::vector<float> got(n_kv * n_rows);
    ggml_backend_tensor_get(h.mask, got.data(), 0, got.size() * sizeof(float));
    for (uint32_t r = 0; r < n_rows; ++r)
        for (uint32_t j = 0; j < n_kv; ++j)
            EXPECT_FLOAT_EQ(got[r * n_kv + j], ref(r, j, 0))
                << "r=" << r << " j=" << j;
}

TEST(AttnMaskInput, SlidingWindowMatchesReference) {
    const uint32_t n_kv = 12, n_rows = 8, window = 3;
    Harness h(n_kv, n_rows, "kq_mask.1");

    std::vector<int32_t> toks(n_rows, 0);
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;
    step.pos = 0;

    AttnMaskInput in("kq_mask.1", window);
    in.set_input(step);

    std::vector<float> got(n_kv * n_rows);
    ggml_backend_tensor_get(h.mask, got.data(), 0, got.size() * sizeof(float));
    for (uint32_t r = 0; r < n_rows; ++r)
        for (uint32_t j = 0; j < n_kv; ++j)
            EXPECT_FLOAT_EQ(got[r * n_kv + j], ref(r, j, window));
}

TEST(AttnMaskInput, BatchedExplicitPositions) {
    const uint32_t n_kv = 10, n_rows = 3;
    Harness h(n_kv, n_rows, "kq_mask_b");

    std::vector<int32_t> toks(n_rows, 0);
    std::vector<int32_t> positions{7, 2, 9};
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;
    step.positions = &positions;  // batched: q_pos = positions[r]

    AttnMaskInput in("kq_mask_b", 0u);
    in.set_input(step);

    std::vector<float> got(n_kv * n_rows);
    ggml_backend_tensor_get(h.mask, got.data(), 0, got.size() * sizeof(float));
    for (uint32_t r = 0; r < n_rows; ++r)
        for (uint32_t j = 0; j < n_kv; ++j)
            EXPECT_FLOAT_EQ(got[r * n_kv + j], ref(positions[r], j, 0));
}

TEST(AttnMaskInput, FailLoudOnAbsentSlot) {
    Harness h(4, 2, "kq_mask.0");
    std::vector<int32_t> toks(2, 0);
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;

    AttnMaskInput in("kq_mask.99", 0u);  // not in graph
    EXPECT_THROW(in.set_input(step), std::runtime_error);
}
