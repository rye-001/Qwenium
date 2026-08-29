// test_mrope_positions_input.cpp — co-located unit test for
// src/graph_inputs/mrope_positions_input.cpp (CLAUDE.md test co-location).
//
// The thing worth pinning is the LAYOUT. ggml reads the four position
// components as pos[i], pos[i+n], pos[i+2n], pos[i+3n] — component-major, four
// contiguous blocks. Interleaving per token compiles, runs, and silently
// rotates against the wrong component; nothing downstream would complain. So
// these tests assert the block layout explicitly, not just the values.

#include <gtest/gtest.h>

#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/mrope_positions_input.h"

namespace {

// Harness sized for the 4-component tensor.
struct H {
    ggml_context* ctx; ggml_backend_t be; ggml_backend_buffer_t buf;
    ggml_cgraph* gf; ggml_tensor* t;
    explicit H(size_t n_rows) {
        ggml_init_params p{ ggml_tensor_overhead()*4 + ggml_graph_overhead(),
                            nullptr, true };
        ctx = ggml_init(p);
        t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                               n_rows * MRopePositionsInput::kComponents);
        ggml_set_input(t); ggml_set_name(t, "inp_pos");
        gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, t);
        be = ggml_backend_cpu_init();
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    }
    std::vector<int32_t> read(size_t n_rows) {
        std::vector<int32_t> out(n_rows * MRopePositionsInput::kComponents);
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(int32_t));
        return out;
    }
    ~H(){ ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx); }
};

}  // namespace

TEST(MRopePositionsInput, ComponentsAreFourBlocksNotInterleaved) {
    const size_t n = 3;
    H h(n);
    std::vector<int32_t> tokens(n, 0);

    StepContext step;
    step.gf = h.gf;
    step.tokens = &tokens;
    step.pos = 10;   // contiguous mode ⇒ rows are 10, 11, 12

    MRopePositionsInput().set_input(step);
    const auto v = h.read(n);

    ASSERT_EQ(v.size(), n * 4);
    // Component-major: [10,11,12 | 10,11,12 | 10,11,12 | 10,11,12].
    // Interleaved would be [10,10,10,10, 11,11,11,11, ...] — same multiset,
    // different meaning, which is exactly why this is asserted positionally.
    for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(v[k * n + 0], 10) << "component " << k;
        EXPECT_EQ(v[k * n + 1], 11) << "component " << k;
        EXPECT_EQ(v[k * n + 2], 12) << "component " << k;
    }
}

// The P2 invariant: text-only steps give every component the SAME position.
// That equality is what makes ggml_rope_multi reduce to the NEOX rotation and
// keeps output byte-identical. If a future change breaks it accidentally, the
// model degrades quietly rather than failing — so pin it.
TEST(MRopePositionsInput, AllFourComponentsAreEqualForTextOnly) {
    const size_t n = 5;
    H h(n);
    std::vector<int32_t> tokens(n, 0);

    StepContext step;
    step.gf = h.gf;
    step.pos = 0;
    step.tokens = &tokens;

    MRopePositionsInput().set_input(step);
    const auto v = h.read(n);

    for (size_t i = 0; i < n; ++i) {
        const int32_t t = v[0 * n + i];
        EXPECT_EQ(v[1 * n + i], t) << "row " << i << " h != t";
        EXPECT_EQ(v[2 * n + i], t) << "row " << i << " w != t";
        EXPECT_EQ(v[3 * n + i], t) << "row " << i << " e != t";
    }
}

// Batched decode supplies an explicit per-row position vector rather than
// pos+r; M-RoPE must honour it the same way PositionsInput does.
TEST(MRopePositionsInput, HonoursExplicitPerRowPositions) {
    const size_t n = 4;
    H h(n);
    std::vector<int32_t> tokens(n, 0);
    const std::vector<int32_t> positions{7, 0, 42, 3};

    StepContext step;
    step.gf = h.gf;
    step.tokens = &tokens;
    step.pos = 999;               // must be ignored when positions is set
    step.positions = &positions;

    MRopePositionsInput().set_input(step);
    const auto v = h.read(n);

    for (int k = 0; k < 4; ++k)
        for (size_t i = 0; i < n; ++i)
            EXPECT_EQ(v[k * n + i], positions[i])
                << "component " << k << " row " << i;
}

TEST(MRopePositionsInput, SingleRowStillWritesFourComponents) {
    H h(1);
    std::vector<int32_t> tokens(1, 0);
    StepContext step;
    step.gf = h.gf;
    step.tokens = &tokens;
    step.pos = 123;

    MRopePositionsInput().set_input(step);
    const auto v = h.read(1);
    ASSERT_EQ(v.size(), 4u);
    for (int k = 0; k < 4; ++k) EXPECT_EQ(v[k], 123);
}
