// test_layer_state.cpp — PR 2.1 / PR 2.2
//
// Tests the LayerState abstract interface via a mock implementation,
// and verifies that simple_kv_cache and CompressedKVStore conform to it.
//
// Run: ./qwen3-layer-tests --gtest_filter="LayerState*"

#include <gtest/gtest.h>
#include <cstddef>
#include <vector>

#include "../../src/state/layer_state.h"
#include "../../src/state/kv_cache_simple.h"
#include "../../src/state/kv_cache_compressed.h"

#include "ggml.h"
#include "ggml-cpu.h"

// ── Mock implementation ───────────────────────────────────────────────────────

class MockLayerState : public LayerState {
public:
    int    reset_calls = 0;
    int    last_reset_seq_id = -1;
    size_t bytes = 128;

    void   reset_sequence(int seq_id) override { reset_calls++; last_reset_seq_id = seq_id; }
    size_t memory_bytes() const override       { return bytes; }
};

// ── PR 2.1: abstract interface ─────────────────────────────────────────────────

TEST(LayerState, PolymorphicDispatch) {
    MockLayerState mock;
    LayerState* ls = &mock;

    ls->reset_sequence(3);
    EXPECT_EQ(mock.reset_calls, 1);
    EXPECT_EQ(mock.last_reset_seq_id, 3);

    EXPECT_EQ(ls->memory_bytes(), 128u);
}

TEST(LayerState, VirtualDestructorFires) {
    // Deleting through base pointer must not leak.
    // We verify via unique_ptr with base type — no sanitizer error expected.
    std::unique_ptr<LayerState> ls = std::make_unique<MockLayerState>();
    ls->reset_sequence(0);
    // Destructor runs at end of scope; test passes if no crash.
}

// ── PR 2.2: simple_kv_cache conforms ─────────────────────────────────────────

class KVCacheLayerStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        cache_ = std::make_unique<simple_kv_cache>(
            /*n_layers=*/2, /*n_ctx_max=*/16, /*n_batch_max=*/2,
            /*n_embd_k=*/8, /*n_embd_v=*/8,
            GGML_TYPE_F32, GGML_TYPE_F32, backend_);
    }
    void TearDown() override {
        cache_.reset();
        if (backend_) ggml_backend_free(backend_);
    }
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<simple_kv_cache> cache_;
};

TEST_F(KVCacheLayerStateTest, InheritsLayerState) {
    // Holding via base pointer must compile and dispatch correctly.
    LayerState* ls = cache_.get();
    ASSERT_NE(ls, nullptr);
    EXPECT_GT(ls->memory_bytes(), 0u);
}

TEST_F(KVCacheLayerStateTest, ResetSequenceClearsSlot) {
    cache_->advance(4, /*slot=*/0);
    EXPECT_EQ(cache_->get_pos(0), 4u);

    LayerState* ls = cache_.get();
    ls->reset_sequence(0);
    EXPECT_EQ(cache_->get_pos(0), 0u);
}

TEST_F(KVCacheLayerStateTest, MemoryBytesNonzero) {
    EXPECT_GT(cache_->memory_bytes(), 0u);
}

TEST_F(KVCacheLayerStateTest, TruncateToPosition) {
    cache_->advance(10, 0);
    ASSERT_EQ(cache_->get_pos(0), 10u);

    // O(1) head-pointer truncation: discard positions 5–9.
    cache_->truncate_to_position(5, 0);
    EXPECT_EQ(cache_->get_pos(0), 5u);

    // Truncating to current position is a no-op.
    cache_->truncate_to_position(5, 0);
    EXPECT_EQ(cache_->get_pos(0), 5u);
}

// ── PR 2.2: CompressedKVStore conforms ───────────────────────────────────────

TEST(CompressedKVStoreLayerState, InheritsLayerState) {
    // n_embd_k=128, head_dim=64 (power of 2), block_size=32 → 128/32=4 blocks per head.
    CompressedKVStore store(/*n_layers=*/2, /*n_slots=*/2, /*n_ctx_max=*/16,
                            /*n_embd_k=*/128, /*n_embd_v=*/128,
                            /*head_dim=*/64, /*bits=*/2);
    LayerState* ls = &store;
    ASSERT_NE(ls, nullptr);
    EXPECT_GT(ls->memory_bytes(), 0u);
}

TEST(CompressedKVStoreLayerState, ResetSequenceResetsPosition) {
    CompressedKVStore store(2, 2, 16, 128, 128, 64, 2);
    store.advance(/*slot=*/0, /*n_tokens=*/6);
    EXPECT_EQ(store.get_pos(0), 6u);

    LayerState* ls = &store;
    ls->reset_sequence(0);
    EXPECT_EQ(store.get_pos(0), 0u);
}
