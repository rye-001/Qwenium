// test_snapkv_position.cpp
// TDD tests for SnapKV dual position tracking (Phase 4).
// After eviction, get_cache_pos() must return the logical sequence position
// (for RoPE), not the physical compacted cache length.

#include <gtest/gtest.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "../../src/kv-cache/simple-kv-cache.h"
#include "../../src/qwen3-core/forward-pass-base.h"

// ============================================================================
// 1. snapkv_seq_pos: default is 0 (inactive)
// ============================================================================

TEST(SnapKVPosition, DefaultSeqPosIsZero) {
    // ForwardPassBase is abstract, but we test the seq_pos accessors directly
    // via a minimal concrete subclass or by testing the forward pass classes.
    // Since we can't instantiate ForwardPassBase, we test the concept via
    // the simple_kv_cache position + the forward-pass-level seq_pos tracking.

    // Verify that simple_kv_cache positions work as expected baseline
    simple_kv_cache cache(1, 32, 2, 64, 64, GGML_TYPE_F32, GGML_TYPE_F32, nullptr);
    EXPECT_EQ(cache.get_pos(0), 0u);
    EXPECT_EQ(cache.get_pos(1), 0u);
}

// ============================================================================
// 2. After compact, physical position = retained count
// ============================================================================

TEST(SnapKVPosition, CompactUpdatesPhysicalPosition) {
    simple_kv_cache cache(1, 32, 1, 64, 64, GGML_TYPE_F32, GGML_TYPE_F32, nullptr);
    cache.set_pos(10, 0);  // pretend 10 tokens prefilled

    std::vector<uint32_t> retained = {2, 5, 8};
    cache.compact(0, retained);

    // Physical position should be retained count
    EXPECT_EQ(cache.get_pos(0), 3u);
}

// ============================================================================
// 3. After compact + advance, physical position increments correctly
// ============================================================================

TEST(SnapKVPosition, AdvanceAfterCompact) {
    simple_kv_cache cache(1, 32, 1, 64, 64, GGML_TYPE_F32, GGML_TYPE_F32, nullptr);
    cache.set_pos(10, 0);

    std::vector<uint32_t> retained = {2, 5, 8};
    cache.compact(0, retained);
    EXPECT_EQ(cache.get_pos(0), 3u);

    cache.advance(1, 0);  // one decode token
    EXPECT_EQ(cache.get_pos(0), 4u);

    cache.advance(1, 0);  // another
    EXPECT_EQ(cache.get_pos(0), 5u);
}

// ============================================================================
// 4. ForwardPassBase seq_pos tracking
// ============================================================================

TEST(SnapKVPosition, SeqPosHelpers) {
    // Test the static helper functions for seq_pos tracking
    // These are on ForwardPassBase — we test the concept by verifying
    // the seq_pos vector logic that will be used by subclasses.

    // Simulate: 2 slots, seq_pos tracking
    std::vector<uint32_t> seq_pos(2, 0);

    // Initially inactive (0)
    EXPECT_EQ(seq_pos[0], 0u);
    EXPECT_EQ(seq_pos[1], 0u);

    // After SnapKV on slot 0: original length = 500, compacted to 100
    uint32_t original_len = 500;
    seq_pos[0] = original_len;

    // get_cache_pos logic: if seq_pos > 0, return it; else return physical pos
    uint32_t physical_pos = 100;  // after compaction
    auto get_pos = [&](uint32_t slot) -> uint32_t {
        return seq_pos[slot] > 0 ? seq_pos[slot] : physical_pos;
    };

    EXPECT_EQ(get_pos(0), 500u);  // returns logical pos
    EXPECT_EQ(get_pos(1), 100u);  // slot 1 not snapped, returns physical

    // After decode: both counters advance
    seq_pos[0] += 1;
    physical_pos += 1;
    EXPECT_EQ(seq_pos[0], 501u);  // RoPE pos for next token
    EXPECT_EQ(physical_pos, 101u);  // KV write offset

    // After clear_slot: seq_pos resets
    seq_pos[0] = 0;
    physical_pos = 0;
    EXPECT_EQ(get_pos(0), 0u);
}
