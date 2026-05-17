// test_positions_input.cpp — contiguous (pos + r) and explicit-positions
// (batched) modes both fold through StepContext::row_pos.

#include <gtest/gtest.h>

#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/positions_input.h"

namespace {
struct H {
    ggml_context* ctx; ggml_backend_t be; ggml_backend_buffer_t buf;
    ggml_cgraph* gf; ggml_tensor* t;
    explicit H(size_t n) {
        ggml_init_params p{ ggml_tensor_overhead()*4 + ggml_graph_overhead(),
                            nullptr, true };
        ctx = ggml_init(p);
        t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(t); ggml_set_name(t, "inp_pos");
        gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, t);
        be = ggml_backend_cpu_init();
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    }
    ~H(){ ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx); }
};
}

TEST(PositionsInput, ContiguousPrefill) {
    std::vector<int32_t> toks(5, 0);
    H h(toks.size());
    StepContext step; step.gf = h.gf; step.tokens = &toks; step.pos = 100;

    PositionsInput in;
    in.set_input(step);

    std::vector<int32_t> got(5);
    ggml_backend_tensor_get(h.t, got.data(), 0, got.size()*sizeof(int32_t));
    EXPECT_EQ(got, (std::vector<int32_t>{100,101,102,103,104}));
}

TEST(PositionsInput, ExplicitBatched) {
    std::vector<int32_t> toks(3, 0);
    std::vector<int32_t> positions{42, 7, 99};
    H h(toks.size());
    StepContext step;
    step.gf = h.gf; step.tokens = &toks; step.positions = &positions;

    PositionsInput in;
    in.set_input(step);

    std::vector<int32_t> got(3);
    ggml_backend_tensor_get(h.t, got.data(), 0, got.size()*sizeof(int32_t));
    EXPECT_EQ(got, positions);
}
