// test_mrope_image_positions.cpp — the M-RoPE image position construction
// (P4 of docs/plan-qwen35-vision-impl.md).
//
// §8.4 named this the quiet-failure case: get the position layout wrong and
// nothing errors, the model just answers worse. So every value here is pinned
// against the reference rather than against intuition.
//
// Reference (vendored, build-*/_deps/ggml-src/tools/mtmd/):
//   mtmd.cpp  mtmd_image_tokens_get_decoder_pos, MTMD_POS_TYPE_MROPE:
//       pos.t = pos_0;  pos.x = pos_0 + (i % nx);  pos.y = pos_0 + (i / nx);
//       pos.z = 0;
//   mtmd-helper-common.h  set_position_mrope_2d:
//       component 0 <- .t,  component 1 <- .y,  component 2 <- .x,  3 <- .z
//   => t = pos0, h = pos0 + row, w = pos0 + col, e = 0.
//
// Co-located with mrope_positions_input.cpp alongside test_mrope_positions_input;
// split out because the image path is a distinct behaviour with its own oracle.

#include <gtest/gtest.h>

#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/mrope_positions_input.h"

namespace {

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

// Run the input over an image chunk of nx*ny tokens starting at pos0.
std::vector<int32_t> image_positions(uint32_t nx, uint32_t ny, int32_t pos0) {
    const size_t n = static_cast<size_t>(nx) * ny;
    static H* h = nullptr; (void)h;
    H harness(n);
    std::vector<int32_t> tokens(n, 0);
    StepContext step;
    step.gf         = harness.gf;
    step.tokens     = &tokens;
    step.pos        = pos0;
    step.img_grid_w = nx;
    MRopePositionsInput().set_input(step);
    return harness.read(n);
}

}  // namespace

// The exact reference layout, checked cell by cell on a deliberately
// NON-SQUARE grid so a row/column swap cannot hide.
TEST(MRopeImagePositions, MatchesReferenceLayoutOnANonSquareGrid) {
    const uint32_t nx = 4, ny = 3;      // 4 wide, 3 tall
    const int32_t  pos0 = 100;
    const size_t   n = nx * ny;
    const auto v = image_positions(nx, ny, pos0);
    ASSERT_EQ(v.size(), n * 4);

    for (size_t i = 0; i < n; ++i) {
        const int32_t row = static_cast<int32_t>(i / nx);
        const int32_t col = static_cast<int32_t>(i % nx);
        EXPECT_EQ(v[0 * n + i], pos0)       << "t, token " << i;
        EXPECT_EQ(v[1 * n + i], pos0 + row) << "h(row), token " << i;
        EXPECT_EQ(v[2 * n + i], pos0 + col) << "w(col), token " << i;
        EXPECT_EQ(v[3 * n + i], 0)          << "e, token " << i;
    }
}

// h carries the ROW and w carries the COLUMN, not the reverse. On a
// non-square grid the two have different ranges, so this is decisive.
TEST(MRopeImagePositions, RowAndColumnAreNotTransposed) {
    const uint32_t nx = 8, ny = 2;
    const size_t   n = nx * ny;
    const auto v = image_positions(nx, ny, 0);

    int32_t h_max = 0, w_max = 0;
    for (size_t i = 0; i < n; ++i) {
        h_max = std::max(h_max, v[1 * n + i]);
        w_max = std::max(w_max, v[2 * n + i]);
    }
    EXPECT_EQ(h_max, static_cast<int32_t>(ny) - 1) << "h should span the rows";
    EXPECT_EQ(w_max, static_cast<int32_t>(nx) - 1) << "w should span the columns";
}

// The t component is CONSTANT across an image (it is the temporal axis, and a
// still image is one frame). If it advanced per token the image would consume
// n_tokens positions and the max(nx,ny) advance would be wrong.
TEST(MRopeImagePositions, TemporalComponentIsConstantAcrossTheImage) {
    const uint32_t nx = 5, ny = 4;
    const size_t   n = nx * ny;
    const auto v = image_positions(nx, ny, 42);
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(v[0 * n + i], 42) << "token " << i;
}

// The span's position footprint is max(nx, ny) — the largest component value
// reached is max(nx,ny)-1, so the next token may start at pos0 + max(nx,ny)
// without colliding. This is the invariant behind the advance in
// prefill_multimodal; if it broke, image and following text would share
// positions.
TEST(MRopeImagePositions, PositionFootprintIsMaxOfGridDims) {
    for (auto dims : std::vector<std::pair<uint32_t,uint32_t>>{
             {4,3}, {3,4}, {8,2}, {1,7}, {6,6}}) {
        const uint32_t nx = dims.first, ny = dims.second;
        const size_t   n  = static_cast<size_t>(nx) * ny;
        const int32_t  pos0 = 10;
        const auto v = image_positions(nx, ny, pos0);

        int32_t used_max = pos0;
        for (size_t i = 0; i < n; ++i) {
            used_max = std::max(used_max, v[0 * n + i]);
            used_max = std::max(used_max, v[1 * n + i]);
            used_max = std::max(used_max, v[2 * n + i]);
            // component 3 is 0 by reference and is not a sequence position
        }
        const int32_t advance = static_cast<int32_t>(std::max(nx, ny));
        EXPECT_EQ(used_max, pos0 + advance - 1)
            << "grid " << nx << "x" << ny;
        EXPECT_LT(used_max, pos0 + advance)
            << "next token would collide, grid " << nx << "x" << ny;
    }
}

// A 1-wide image is still 2-D: every token is its own row.
TEST(MRopeImagePositions, SingleColumnImageDegradesToRows) {
    const uint32_t nx = 1, ny = 5;
    const size_t   n = ny;
    const auto v = image_positions(nx, ny, 0);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(v[1 * n + i], static_cast<int32_t>(i));  // h = row
        EXPECT_EQ(v[2 * n + i], 0);                        // w = col, always 0
    }
}

// img_grid_w == 0 must keep the text behaviour exactly (all four equal), so a
// recipe that never sets it is unaffected by P4.
TEST(MRopeImagePositions, ZeroGridWidthIsTheTextPath) {
    H h(3);
    std::vector<int32_t> tokens(3, 0);
    StepContext step;
    step.gf = h.gf; step.tokens = &tokens; step.pos = 7;
    step.img_grid_w = 0;
    MRopePositionsInput().set_input(step);
    const auto v = h.read(3);
    for (size_t i = 0; i < 3; ++i)
        for (int k = 0; k < 4; ++k)
            EXPECT_EQ(v[k * 3 + i], 7 + static_cast<int32_t>(i));
}
