// test_graph_arena.cpp — the per-forward-pass ggml context, as a value.
//
// The point of extracting GraphArena out of ForwardPassBase is that context
// lifetime becomes testable WITHOUT constructing a model, a backend or a
// recipe. These tests are the evidence that it did.

#include <gtest/gtest.h>

#include "../../src/models/graph_arena.h"

TEST(GraphArena, ConstructsALiveContextOverItsOwnBuffer) {
    GraphArena arena;
    EXPECT_NE(arena.ctx(), nullptr);
    EXPECT_EQ(arena.buffer_bytes(), FP_GRAPH_SIZE_METADATA);
}

TEST(GraphArena, NewGraphReturnsAnEmptyGraph) {
    GraphArena arena;
    ggml_cgraph* gf = arena.new_graph();
    ASSERT_NE(gf, nullptr);
    EXPECT_EQ(ggml_graph_n_nodes(gf), 0);
}

// The invariant the whole design rests on: reset() reuses the SAME buffer, so a
// steady-state decode loop does no heap churn no matter how many passes it runs.
TEST(GraphArena, ResetReusesTheSameBufferAndStaysUsable) {
    GraphArena arena;
    const size_t bytes_before = arena.buffer_bytes();

    for (int i = 0; i < 50; ++i) {
        arena.reset();
        ASSERT_NE(arena.ctx(), nullptr) << "context died on reset " << i;
        ggml_cgraph* gf = arena.new_graph();
        ASSERT_NE(gf, nullptr) << "graph alloc failed after reset " << i;
        ggml_tensor* t = ggml_new_tensor_1d(arena.ctx(), GGML_TYPE_F32, 8);
        ASSERT_NE(t, nullptr) << "tensor alloc failed after reset " << i;
    }
    EXPECT_EQ(arena.buffer_bytes(), bytes_before)
        << "reset must re-init over the existing buffer, not reallocate";
}

// no_alloc: this context describes graphs, it does not hold numbers. A tensor
// built here has no data buffer — assuming otherwise is the classic ggml bug.
TEST(GraphArena, ContextIsMetadataOnly) {
    GraphArena arena;
    ggml_tensor* t = ggml_new_tensor_2d(arena.ctx(), GGML_TYPE_F32, 4, 4);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->data, nullptr)
        << "no_alloc=true means tensor data lives in backend buffers, not here";
}

// Single ownership of a raw ggml_context*, and the buffer address is baked into
// it — so copying or moving must not compile.
TEST(GraphArena, IsNeitherCopyableNorMovable) {
    EXPECT_FALSE(std::is_copy_constructible<GraphArena>::value);
    EXPECT_FALSE(std::is_move_constructible<GraphArena>::value);
    EXPECT_FALSE(std::is_copy_assignable<GraphArena>::value);
    EXPECT_FALSE(std::is_move_assignable<GraphArena>::value);
}
