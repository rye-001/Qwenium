// test_deltanet_state.cpp — PR 3.D.1
//
// Tests DeltaNetState: backend-backed fixed-size per-slot state for DeltaNet
// layers, with checkpoint/restore semantics for grammar backtracking and
// speculative-decoding rejection.
//
// Run: ./qwen3-deltanet-state-tests --gtest_filter="DeltaNetState*"

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../../src/state/deltanet_state.h"
#include "../../src/state/recurrent_state.h"  // for CheckpointId / kInvalidCheckpoint
#include "ggml.h"
#include "ggml-cpu.h"

// Qwen3.5-0.8B-like dimensions (small enough for a unit test)
static constexpr uint32_t DN_LAYERS    = 4;    // DeltaNet layer count
static constexpr uint32_t N_SLOTS      = 2;    // parallel sequences
static constexpr uint32_t HEAD_V_DIM   = 8;    // d_inner / num_v_heads
static constexpr uint32_t HEAD_K_DIM   = 8;    // ssm_state_size
static constexpr uint32_t NUM_V_HEADS  = 4;    // num groups
static constexpr uint32_t CONV_CHANNELS = 32;  // d_inner + 2 * num_k_heads * head_k_dim
static constexpr uint32_t CONV_KERNEL  = 4;    // causal conv kernel

class DeltaNetStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        ASSERT_NE(backend_, nullptr);

        DeltaNetState::Hparams hp;
        hp.n_dn_layers   = DN_LAYERS;
        hp.n_slots       = N_SLOTS;
        hp.head_v_dim    = HEAD_V_DIM;
        hp.head_k_dim    = HEAD_K_DIM;
        hp.num_v_heads   = NUM_V_HEADS;
        hp.conv_channels = CONV_CHANNELS;
        hp.conv_kernel   = CONV_KERNEL;
        hp.backend       = backend_;

        state_ = std::make_unique<DeltaNetState>(hp);

        rec_floats_  = HEAD_V_DIM * HEAD_K_DIM * NUM_V_HEADS;  // 4 * 4 * 4 = 256
        conv_floats_ = (CONV_KERNEL - 1) * CONV_CHANNELS;       // 3 * 32 = 96
    }

    void TearDown() override {
        state_.reset();
        if (backend_) ggml_backend_free(backend_);
    }

    ggml_backend_t               backend_ = nullptr;
    std::unique_ptr<DeltaNetState> state_;
    size_t rec_floats_  = 0;
    size_t conv_floats_ = 0;
};

// ── Construction ──────────────────────────────────────────────────────────────

TEST_F(DeltaNetStateTest, ConstructionSucceeds) {
    EXPECT_NE(state_, nullptr);
}

TEST_F(DeltaNetStateTest, InheritsLayerState) {
    LayerState* ls = state_.get();
    EXPECT_GT(ls->memory_bytes(), 0u);
}

// ── Initial state is zeroed ───────────────────────────────────────────────────

TEST_F(DeltaNetStateTest, RecurrentStateStartsZeroed) {
    std::vector<float> buf(rec_floats_);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->get_recurrent(il, 0, buf.data());
        for (size_t i = 0; i < rec_floats_; ++i)
            EXPECT_EQ(buf[i], 0.0f) << "layer=" << il << " i=" << i;
    }
}

TEST_F(DeltaNetStateTest, ConvStateStartsZeroed) {
    std::vector<float> buf(conv_floats_);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->get_conv(il, 0, buf.data());
        for (size_t i = 0; i < conv_floats_; ++i)
            EXPECT_EQ(buf[i], 0.0f) << "layer=" << il << " i=" << i;
    }
}

// ── Read / write ──────────────────────────────────────────────────────────────

TEST_F(DeltaNetStateTest, RecurrentStateWriteRead) {
    std::vector<float> write_data(rec_floats_);
    for (size_t i = 0; i < rec_floats_; ++i)
        write_data[i] = sinf(static_cast<float>(i) * 0.01f);

    state_->set_recurrent(1, 0, write_data.data());

    std::vector<float> read_data(rec_floats_);
    state_->get_recurrent(1, 0, read_data.data());

    for (size_t i = 0; i < rec_floats_; ++i)
        EXPECT_FLOAT_EQ(read_data[i], write_data[i]) << "i=" << i;
}

TEST_F(DeltaNetStateTest, ConvStateWriteRead) {
    std::vector<float> write_data(conv_floats_);
    for (size_t i = 0; i < conv_floats_; ++i)
        write_data[i] = cosf(static_cast<float>(i) * 0.02f);

    state_->set_conv(2, 1, write_data.data());

    std::vector<float> read_data(conv_floats_);
    state_->get_conv(2, 1, read_data.data());

    for (size_t i = 0; i < conv_floats_; ++i)
        EXPECT_FLOAT_EQ(read_data[i], write_data[i]) << "i=" << i;
}

// ── Isolation: slots and layers are independent ───────────────────────────────

TEST_F(DeltaNetStateTest, SlotIsolation) {
    std::vector<float> d0(rec_floats_, 1.0f);
    std::vector<float> d1(rec_floats_, 2.0f);
    state_->set_recurrent(0, 0, d0.data());
    state_->set_recurrent(0, 1, d1.data());

    std::vector<float> r0(rec_floats_);
    state_->get_recurrent(0, 0, r0.data());
    for (size_t i = 0; i < rec_floats_; ++i)
        EXPECT_EQ(r0[i], 1.0f) << "slot 0 corrupted at i=" << i;

    std::vector<float> r1(rec_floats_);
    state_->get_recurrent(0, 1, r1.data());
    for (size_t i = 0; i < rec_floats_; ++i)
        EXPECT_EQ(r1[i], 2.0f) << "slot 1 wrong at i=" << i;
}

TEST_F(DeltaNetStateTest, LayerIsolation) {
    std::vector<float> d0(rec_floats_, 3.0f);
    std::vector<float> d1(rec_floats_, 7.0f);
    state_->set_recurrent(0, 0, d0.data());
    state_->set_recurrent(1, 0, d1.data());

    std::vector<float> r0(rec_floats_);
    state_->get_recurrent(0, 0, r0.data());
    for (size_t i = 0; i < rec_floats_; ++i)
        EXPECT_EQ(r0[i], 3.0f) << "layer 0 corrupted at i=" << i;

    std::vector<float> r1(rec_floats_);
    state_->get_recurrent(1, 0, r1.data());
    for (size_t i = 0; i < rec_floats_; ++i)
        EXPECT_EQ(r1[i], 7.0f) << "layer 1 wrong at i=" << i;
}

// ── LayerState: reset_sequence / memory_bytes ─────────────────────────────────

TEST_F(DeltaNetStateTest, ResetSequenceZerosAllLayersForSlot) {
    // Write non-zero state to slot 0, all layers
    std::vector<float> rec_data(rec_floats_, 42.0f);
    std::vector<float> conv_data(conv_floats_, 99.0f);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->set_recurrent(il, 0, rec_data.data());
        state_->set_conv(il, 0, conv_data.data());
    }

    state_->reset_sequence(0);

    std::vector<float> r(rec_floats_);
    std::vector<float> c(conv_floats_);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->get_recurrent(il, 0, r.data());
        for (size_t i = 0; i < rec_floats_; ++i)
            EXPECT_EQ(r[i], 0.0f) << "recurrent not cleared: layer=" << il << " i=" << i;

        state_->get_conv(il, 0, c.data());
        for (size_t i = 0; i < conv_floats_; ++i)
            EXPECT_EQ(c[i], 0.0f) << "conv not cleared: layer=" << il << " i=" << i;
    }
}

TEST_F(DeltaNetStateTest, ResetSequenceDoesNotAffectOtherSlot) {
    std::vector<float> data(rec_floats_, 5.0f);
    state_->set_recurrent(0, 1, data.data());  // write to slot 1
    state_->reset_sequence(0);                  // reset slot 0 only

    std::vector<float> r(rec_floats_);
    state_->get_recurrent(0, 1, r.data());
    for (size_t i = 0; i < rec_floats_; ++i)
        EXPECT_EQ(r[i], 5.0f) << "slot 1 incorrectly reset at i=" << i;
}

TEST_F(DeltaNetStateTest, MemoryBytesCoversAllData) {
    size_t min_bytes =
        static_cast<size_t>(DN_LAYERS) * N_SLOTS *
        (rec_floats_ + conv_floats_) * sizeof(float);
    EXPECT_GE(state_->memory_bytes(), min_bytes);
}

// ── clone_slot / clear_slot ───────────────────────────────────────────────────

TEST_F(DeltaNetStateTest, ClearSlotZerosBothStates) {
    std::vector<float> rec_data(rec_floats_, 11.0f);
    std::vector<float> conv_data(conv_floats_, 22.0f);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->set_recurrent(il, 1, rec_data.data());
        state_->set_conv(il, 1, conv_data.data());
    }

    state_->clear_slot(1);

    std::vector<float> r(rec_floats_);
    std::vector<float> c(conv_floats_);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->get_recurrent(il, 1, r.data());
        for (size_t i = 0; i < rec_floats_; ++i)
            EXPECT_EQ(r[i], 0.0f) << "layer=" << il << " rec i=" << i;
        state_->get_conv(il, 1, c.data());
        for (size_t i = 0; i < conv_floats_; ++i)
            EXPECT_EQ(c[i], 0.0f) << "layer=" << il << " conv i=" << i;
    }
}

TEST_F(DeltaNetStateTest, CloneSlotCopiesAllData) {
    std::vector<float> rec_data(rec_floats_);
    std::vector<float> conv_data(conv_floats_);
    for (size_t i = 0; i < rec_floats_; ++i)  rec_data[i]  = sinf(i * 0.01f);
    for (size_t i = 0; i < conv_floats_; ++i) conv_data[i] = cosf(i * 0.02f);

    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->set_recurrent(il, 0, rec_data.data());
        state_->set_conv(il, 0, conv_data.data());
    }

    state_->clone_slot(0, 1);

    std::vector<float> r(rec_floats_);
    std::vector<float> c(conv_floats_);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->get_recurrent(il, 1, r.data());
        for (size_t i = 0; i < rec_floats_; ++i)
            EXPECT_FLOAT_EQ(r[i], rec_data[i]) << "layer=" << il << " rec i=" << i;
        state_->get_conv(il, 1, c.data());
        for (size_t i = 0; i < conv_floats_; ++i)
            EXPECT_FLOAT_EQ(c[i], conv_data[i]) << "layer=" << il << " conv i=" << i;
    }
}

// ── Checkpoint / restore ──────────────────────────────────────────────────────

TEST_F(DeltaNetStateTest, CheckpointAndRestore) {
    // Write a known pattern to slot 0
    std::vector<float> initial_rec(rec_floats_);
    std::iota(initial_rec.begin(), initial_rec.end(), 1.0f);
    for (uint32_t il = 0; il < DN_LAYERS; ++il)
        state_->set_recurrent(il, 0, initial_rec.data());

    CheckpointId id = state_->checkpoint(0);
    EXPECT_NE(id, kInvalidCheckpoint);

    // Overwrite with zeros
    std::vector<float> zeros(rec_floats_, 0.0f);
    for (uint32_t il = 0; il < DN_LAYERS; ++il)
        state_->set_recurrent(il, 0, zeros.data());

    // Restore should recover original values
    state_->restore(id);

    std::vector<float> r(rec_floats_);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->get_recurrent(il, 0, r.data());
        for (size_t i = 0; i < rec_floats_; ++i)
            EXPECT_FLOAT_EQ(r[i], initial_rec[i]) << "layer=" << il << " i=" << i;
    }

    state_->release(id);
}

TEST_F(DeltaNetStateTest, CheckpointIncludesConvState) {
    std::vector<float> init_conv(conv_floats_);
    std::iota(init_conv.begin(), init_conv.end(), 100.0f);
    for (uint32_t il = 0; il < DN_LAYERS; ++il)
        state_->set_conv(il, 0, init_conv.data());

    CheckpointId id = state_->checkpoint(0);

    std::vector<float> zeros(conv_floats_, 0.0f);
    for (uint32_t il = 0; il < DN_LAYERS; ++il)
        state_->set_conv(il, 0, zeros.data());

    state_->restore(id);

    std::vector<float> c(conv_floats_);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->get_conv(il, 0, c.data());
        for (size_t i = 0; i < conv_floats_; ++i)
            EXPECT_FLOAT_EQ(c[i], init_conv[i]) << "layer=" << il << " i=" << i;
    }

    state_->release(id);
}

TEST_F(DeltaNetStateTest, CheckpointsArePerSlot) {
    std::vector<float> data0(rec_floats_, 10.0f);
    std::vector<float> data1(rec_floats_, 20.0f);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->set_recurrent(il, 0, data0.data());
        state_->set_recurrent(il, 1, data1.data());
    }

    CheckpointId id0 = state_->checkpoint(0);
    CheckpointId id1 = state_->checkpoint(1);

    std::vector<float> zeros(rec_floats_, 0.0f);
    for (uint32_t il = 0; il < DN_LAYERS; ++il) {
        state_->set_recurrent(il, 0, zeros.data());
        state_->set_recurrent(il, 1, zeros.data());
    }

    state_->restore(id0);
    state_->restore(id1);

    std::vector<float> r(rec_floats_);
    state_->get_recurrent(0, 0, r.data());
    EXPECT_EQ(r[0], 10.0f);
    state_->get_recurrent(0, 1, r.data());
    EXPECT_EQ(r[0], 20.0f);

    state_->release(id0);
    state_->release(id1);
}

TEST_F(DeltaNetStateTest, CheckpointIdsAreRecycled) {
    CheckpointId id1 = state_->checkpoint(0);
    state_->release(id1);
    CheckpointId id2 = state_->checkpoint(0);
    EXPECT_EQ(id1, id2);
    state_->release(id2);
}

// ── Fail-loud: double-release and invalid id ──────────────────────────────────

TEST_F(DeltaNetStateTest, DoubleReleaseThrows) {
    CheckpointId id = state_->checkpoint(0);
    state_->release(id);
    EXPECT_THROW(state_->release(id), std::runtime_error);
}

TEST_F(DeltaNetStateTest, ReleaseInvalidIdThrows) {
    EXPECT_THROW(state_->release(kInvalidCheckpoint), std::runtime_error);
}

// ── Tensor accessor returns non-null with correct shape ───────────────────────

TEST_F(DeltaNetStateTest, RecurrentTensorShape) {
    ggml_tensor* t = state_->recurrent_tensor(0);
    ASSERT_NE(t, nullptr);
    // [rec_slot_floats, n_slots]
    EXPECT_EQ(static_cast<size_t>(t->ne[0]), rec_floats_);
    EXPECT_EQ(static_cast<uint32_t>(t->ne[1]), N_SLOTS);
}

TEST_F(DeltaNetStateTest, ConvTensorShape) {
    ggml_tensor* t = state_->conv_tensor(0);
    ASSERT_NE(t, nullptr);
    // [conv_slot_floats, n_slots]
    EXPECT_EQ(static_cast<size_t>(t->ne[0]), conv_floats_);
    EXPECT_EQ(static_cast<uint32_t>(t->ne[1]), N_SLOTS);
}
