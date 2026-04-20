// test_kv_cache.cpp — PR 2.8
//
// Unit tests for simple_kv_cache: build test, advance/position management,
// truncate_to_position, LayerState conformance, and scratch-budget note.
// No model file required.
//
// Run: ./qwen3-layer-tests --gtest_filter="KVCache*"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "../../src/state/kv_cache_simple.h"
#include "../../src/state/layer_state.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static constexpr int N_LAYERS   = 2;
static constexpr int N_CTX      = 32;
static constexpr int N_BATCH    = 2;
static constexpr int N_EMBD_KV  = 16;

class KVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend_ = ggml_backend_cpu_init();
        cache_ = std::make_unique<simple_kv_cache>(
            N_LAYERS, N_CTX, N_BATCH,
            N_EMBD_KV, N_EMBD_KV,
            GGML_TYPE_F32, GGML_TYPE_F32, backend_);
    }
    void TearDown() override {
        cache_.reset();
        if (backend_) ggml_backend_free(backend_);
    }
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<simple_kv_cache> cache_;
};

// ── Build test ────────────────────────────────────────────────────────────────

TEST_F(KVCacheTest, ConstructionSucceeds) {
    EXPECT_GT(cache_->memory_bytes(), 0u);
    EXPECT_EQ(cache_->get_pos(0), 0u);
}

// ── Advance and position tracking ─────────────────────────────────────────────

TEST_F(KVCacheTest, AdvanceUpdatesPosition) {
    cache_->advance(5, 0);
    EXPECT_EQ(cache_->get_pos(0), 5u);
    cache_->advance(3, 0);
    EXPECT_EQ(cache_->get_pos(0), 8u);
}

TEST_F(KVCacheTest, SlotsAreIndependent) {
    cache_->advance(5, 0);
    cache_->advance(10, 1);
    EXPECT_EQ(cache_->get_pos(0), 5u);
    EXPECT_EQ(cache_->get_pos(1), 10u);
}

// ── truncate_to_position ──────────────────────────────────────────────────────

TEST_F(KVCacheTest, TruncateToPositionIsO1) {
    cache_->advance(20, 0);
    cache_->truncate_to_position(7, 0);
    EXPECT_EQ(cache_->get_pos(0), 7u);
}

TEST_F(KVCacheTest, TruncateToCurrentPositionIsNoop) {
    cache_->advance(5, 0);
    cache_->truncate_to_position(5, 0);
    EXPECT_EQ(cache_->get_pos(0), 5u);
}

// ── clear_slot via LayerState::reset_sequence ─────────────────────────────────

TEST_F(KVCacheTest, ResetSequenceClearsPosition) {
    cache_->advance(15, 1);
    static_cast<LayerState*>(cache_.get())->reset_sequence(1);
    EXPECT_EQ(cache_->get_pos(1), 0u);
}

// ── KV tensor views build cleanly ─────────────────────────────────────────────

TEST_F(KVCacheTest, GetKAndVTensorsAreNonNull) {
    cache_->advance(4, 0);

    const size_t ctx_bytes = 64 * ggml_tensor_overhead();
    ggml_init_params p{ctx_bytes, nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* k = cache_->get_k(ctx, 0, 4, 0);
    ggml_tensor* v = cache_->get_v(ctx, 0, 4, 0);
    EXPECT_NE(k, nullptr);
    EXPECT_NE(v, nullptr);
    EXPECT_EQ(k->ne[0], N_EMBD_KV);
    EXPECT_EQ(k->ne[1], 4);

    // Scratch budget note (PR 2.8 requirement):
    // simple_kv_cache tensors are persistent (not scratch) — they are the KV
    // store itself.  Views built over them add zero scratch allocation;
    // the memory cost is reported by memory_bytes() = cache buffer size.
    RecordProperty("memory_bytes", static_cast<int64_t>(cache_->memory_bytes()));

    ggml_free(ctx);
}
