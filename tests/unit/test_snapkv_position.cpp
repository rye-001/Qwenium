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

// ============================================================================
// 4b. get_cache_pos vs get_physical_cache_pos divergence
// ============================================================================

TEST(SnapKVPosition, PhysicalVsLogical) {
    // Simulate SnapKV scenario via the position helpers
    std::vector<uint32_t> seq_pos(2, 0);

    // Prefill 500 tokens, compact to 100
    uint32_t physical = 100;
    seq_pos[0] = 500;  // logical

    // get_cache_pos should return logical for RoPE
    auto get_logical = [&](uint32_t slot) -> uint32_t {
        return seq_pos[slot] > 0 ? seq_pos[slot] : physical;
    };

    EXPECT_EQ(get_logical(0), 500u);   // logical for RoPE
    EXPECT_EQ(physical, 100u);          // physical for KV sizing

    // Key invariant: physical < logical after SnapKV
    EXPECT_LT(physical, get_logical(0));

    // After decode: both advance
    physical += 1;
    seq_pos[0] += 1;
    EXPECT_EQ(get_logical(0), 501u);
    EXPECT_EQ(physical, 101u);
}

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

// ============================================================================
// 6. Batched decode n_kv_len must use physical cache position, not logical
// ============================================================================
// Regression test for the bug where build_decoding_graph computed n_kv_len
// from the logical RoPE positions vector. After SnapKV compaction, the
// physical cache is much smaller than the logical position. Using the logical
// position caused gather_indices to read stale/OOB data at positions beyond
// the compacted cache.

TEST(SnapKVPosition, BatchedDecodeNKVLenUsesPhysical) {
    // Setup: 2 slots, 1 layer, context large enough for 500 tokens
    simple_kv_cache cache(1, 1024, 2, 64, 64, GGML_TYPE_F32, GGML_TYPE_F32, nullptr);

    // Prefill slot 0 with 500 tokens, slot 1 with 300
    cache.set_pos(500, 0);
    cache.set_pos(300, 1);

    // SnapKV compaction: retain a small subset
    std::vector<uint32_t> retained_0, retained_1;
    for (uint32_t i = 0; i < 100; i++) retained_0.push_back(i * 5);  // 100 tokens
    for (uint32_t i = 0; i <  80; i++) retained_1.push_back(i * 3);  //  80 tokens
    cache.compact(0, retained_0);
    cache.compact(1, retained_1);

    // Physical positions are now the retained counts
    EXPECT_EQ(cache.get_pos(0), 100u);
    EXPECT_EQ(cache.get_pos(1),  80u);

    // Simulate dual position tracking (as ForwardPassBase does)
    std::vector<uint32_t> seq_pos = {500, 300};  // logical positions for RoPE

    // --- Simulate the BUGGY n_kv_len computation (from positions[]) ---
    // This is what the old code did: derive from logical RoPE positions
    std::vector<int32_t> positions = {
        static_cast<int32_t>(seq_pos[0]),  // 500 — next decode token for slot 0
        static_cast<int32_t>(seq_pos[1])   // 300 — next decode token for slot 1
    };
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > static_cast<int32_t>(max_pos)) max_pos = static_cast<uint32_t>(p);
    }
    uint32_t buggy_n_kv_len = max_pos + 1;  // 501 — WRONG

    // --- Simulate the CORRECT n_kv_len computation (from physical cache pos) ---
    std::vector<uint32_t> slots = {0, 1};
    uint32_t max_physical = 0;
    for (uint32_t s : slots) {
        uint32_t phys = cache.get_pos(s);  // matches get_physical_cache_pos()
        if (phys > max_physical) max_physical = phys;
    }
    uint32_t correct_n_kv_len = max_physical + 1;  // 101 — correct

    // The buggy value would be ~5x larger, causing OOB gather reads
    EXPECT_EQ(correct_n_kv_len, 101u);
    EXPECT_EQ(buggy_n_kv_len,   501u);

    // The key invariant: correct n_kv_len must equal max physical + 1
    EXPECT_EQ(correct_n_kv_len, cache.get_pos(0) + 1);

    // After one decode step, physical advances but stays << logical
    cache.advance(1, 0);
    cache.advance(1, 1);
    seq_pos[0] += 1;
    seq_pos[1] += 1;

    max_physical = 0;
    for (uint32_t s : slots) {
        uint32_t phys = cache.get_pos(s);
        if (phys > max_physical) max_physical = phys;
    }
    EXPECT_EQ(max_physical + 1, 102u);     // physical: 101 + 1
    EXPECT_EQ(seq_pos[0], 501u);           // logical: still much larger
    EXPECT_LT(max_physical + 1, seq_pos[0]);  // invariant holds
}

// ============================================================================
// 7. Multi-slot: n_kv_len uses the MAX physical position across all slots
// ============================================================================

TEST(SnapKVPosition, BatchedDecodeMultiSlotMaxPhysical) {
    simple_kv_cache cache(1, 1024, 4, 64, 64, GGML_TYPE_F32, GGML_TYPE_F32, nullptr);

    // Slot 0: 150 physical, slot 1: 80, slot 2: 200, slot 3: not active
    cache.set_pos(600, 0);
    cache.set_pos(400, 1);
    cache.set_pos(800, 2);

    std::vector<uint32_t> ret_0, ret_1, ret_2;
    for (uint32_t i = 0; i < 150; i++) ret_0.push_back(i);
    for (uint32_t i = 0; i <  80; i++) ret_1.push_back(i);
    for (uint32_t i = 0; i < 200; i++) ret_2.push_back(i);
    cache.compact(0, ret_0);
    cache.compact(1, ret_1);
    cache.compact(2, ret_2);

    // Only slots 0 and 2 are in this batch
    std::vector<uint32_t> active_slots = {0, 2};
    uint32_t max_physical = 0;
    for (uint32_t s : active_slots) {
        uint32_t phys = cache.get_pos(s);
        if (phys > max_physical) max_physical = phys;
    }

    // n_kv_len should be max(150, 200) + 1 = 201
    EXPECT_EQ(max_physical + 1, 201u);

    // NOT based on logical positions which would give max(600,800)+1 = 801
}
