// test_recurrent_state.cpp — PR 2.3
//
// Tests RecurrentState: fixed-size per-slot float state with
// checkpoint/restore/release semantics.  No ggml dependency — pure CPU.
//
// Run: ./qwen3-layer-tests --gtest_filter="RecurrentState*"

#include <gtest/gtest.h>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../../src/state/recurrent_state.h"

static constexpr size_t STATE_FLOATS = 64;
static constexpr int    N_SLOTS      = 2;

class RecurrentStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        rs_ = std::make_unique<RecurrentState>(STATE_FLOATS, N_SLOTS);
    }
    std::unique_ptr<RecurrentState> rs_;
};

// ── LayerState conformance ────────────────────────────────────────────────────

TEST_F(RecurrentStateTest, InheritsLayerState) {
    LayerState* ls = rs_.get();
    EXPECT_GT(ls->memory_bytes(), 0u);
}

TEST_F(RecurrentStateTest, ResetSequenceZerosSlot) {
    float* s = rs_->state_data(0);
    std::fill(s, s + STATE_FLOATS, 1.0f);

    rs_->reset_sequence(0);

    for (size_t i = 0; i < STATE_FLOATS; ++i)
        EXPECT_EQ(rs_->state_data(0)[i], 0.0f) << "at i=" << i;
}

TEST_F(RecurrentStateTest, MemoryBytesCoversAllSlots) {
    // At minimum: n_slots * state_floats * sizeof(float)
    size_t min_bytes = N_SLOTS * STATE_FLOATS * sizeof(float);
    EXPECT_GE(rs_->memory_bytes(), min_bytes);
}

// ── Checkpoint / restore ──────────────────────────────────────────────────────

TEST_F(RecurrentStateTest, CheckpointAndRestore) {
    // Fill slot 0 with a known pattern.
    float* s = rs_->state_data(0);
    std::iota(s, s + STATE_FLOATS, 1.0f);

    CheckpointId id = rs_->checkpoint(0);
    EXPECT_NE(id, kInvalidCheckpoint);

    // Mutate the state.
    std::fill(s, s + STATE_FLOATS, 0.0f);

    // Restore should recover the original values.
    rs_->restore(id);
    for (size_t i = 0; i < STATE_FLOATS; ++i)
        EXPECT_EQ(rs_->state_data(0)[i], static_cast<float>(i + 1)) << "at i=" << i;

    rs_->release(id);
}

TEST_F(RecurrentStateTest, CheckpointsArePerSlot) {
    // Slot 0 and slot 1 are independent.
    std::fill(rs_->state_data(0), rs_->state_data(0) + STATE_FLOATS, 10.0f);
    std::fill(rs_->state_data(1), rs_->state_data(1) + STATE_FLOATS, 20.0f);

    CheckpointId id0 = rs_->checkpoint(0);
    CheckpointId id1 = rs_->checkpoint(1);

    std::fill(rs_->state_data(0), rs_->state_data(0) + STATE_FLOATS, 0.0f);
    std::fill(rs_->state_data(1), rs_->state_data(1) + STATE_FLOATS, 0.0f);

    rs_->restore(id0);
    rs_->restore(id1);

    EXPECT_EQ(rs_->state_data(0)[0], 10.0f);
    EXPECT_EQ(rs_->state_data(1)[0], 20.0f);

    rs_->release(id0);
    rs_->release(id1);
}

TEST_F(RecurrentStateTest, CheckpointIdsAreRecycled) {
    // Releasing a checkpoint should make its id reusable.
    CheckpointId id1 = rs_->checkpoint(0);
    rs_->release(id1);
    CheckpointId id2 = rs_->checkpoint(0);
    // Recycled id equals the released one (implementation detail we can observe).
    EXPECT_EQ(id1, id2);
    rs_->release(id2);
}

// ── Fail-loud: double-release ─────────────────────────────────────────────────

TEST_F(RecurrentStateTest, DoubleReleaseThrows) {
    CheckpointId id = rs_->checkpoint(0);
    rs_->release(id);
    EXPECT_THROW(rs_->release(id), std::runtime_error);
}

TEST_F(RecurrentStateTest, ReleaseInvalidIdThrows) {
    EXPECT_THROW(rs_->release(kInvalidCheckpoint), std::runtime_error);
}
