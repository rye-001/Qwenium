// test_moe_residency.cpp — PR 3.M.4
//
// Tests MoEResidencyTracker: pure C++ expert frequency bookkeeping and
// cold-path dispatch counter. No model file or ggml compute needed.
//
// Run: ./qwen3-moe-residency-tests --gtest_filter="MoEResidency*"

#include <gtest/gtest.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../../src/layers/moe_residency.h"

// ── Construction ──────────────────────────────────────────────────────────────

TEST(MoEResidencyTest, ConstructionSucceeds) {
    MoEResidencyTracker tracker(8, 3);
    EXPECT_EQ(tracker.cold_dispatch_count(), 0u);
    EXPECT_TRUE(tracker.resident_set().empty());
}

TEST(MoEResidencyTest, FrequenciesStartAtZero) {
    MoEResidencyTracker tracker(4, 2);
    for (uint64_t f : tracker.frequencies())
        EXPECT_EQ(f, 0u);
}

TEST(MoEResidencyTest, InvalidNExpertsThrows) {
    EXPECT_THROW(MoEResidencyTracker(0, 1), std::runtime_error);
}

TEST(MoEResidencyTest, InvalidResidentKThrows) {
    EXPECT_THROW(MoEResidencyTracker(8, 0), std::runtime_error);
}

// ── Resident set tracks top-K by frequency ────────────────────────────────────

TEST(MoEResidencyTest, ResidentSetTracksTopKByFrequency) {
    // 8 experts, resident_k=2
    MoEResidencyTracker tracker(8, 2);

    // Build up frequency: expert 0 x4, expert 1 x3, expert 2 x2, expert 3 x1
    for (int i = 0; i < 4; ++i) tracker.record_dispatch(0);
    for (int i = 0; i < 3; ++i) tracker.record_dispatch(1);
    for (int i = 0; i < 2; ++i) tracker.record_dispatch(2);
    tracker.record_dispatch(3);

    // Top-2 should be experts 0 and 1
    EXPECT_EQ(tracker.resident_set().size(), 2u);
    EXPECT_TRUE(tracker.is_resident(0));
    EXPECT_TRUE(tracker.is_resident(1));
    EXPECT_FALSE(tracker.is_resident(2));
    EXPECT_FALSE(tracker.is_resident(3));
}

TEST(MoEResidencyTest, ResidentSetRespectsClamping) {
    // resident_k > n_experts — clamped to n_experts
    MoEResidencyTracker tracker(3, 10);
    tracker.record_dispatch(0);
    tracker.record_dispatch(1);
    tracker.record_dispatch(2);
    // All 3 experts dispatched; resident_k clamped to 3
    EXPECT_EQ(tracker.resident_set().size(), 3u);
    EXPECT_TRUE(tracker.is_resident(0));
    EXPECT_TRUE(tracker.is_resident(1));
    EXPECT_TRUE(tracker.is_resident(2));
}

TEST(MoEResidencyTest, ResidentSetExcludesZeroFrequencyExperts) {
    // With resident_k=3 and only 1 expert ever dispatched, resident_set has 1 entry
    MoEResidencyTracker tracker(8, 3);
    tracker.record_dispatch(4);
    EXPECT_EQ(tracker.resident_set().size(), 1u);
    EXPECT_TRUE(tracker.is_resident(4));
}

// ── Cold-path counter ─────────────────────────────────────────────────────────

TEST(MoEResidencyTest, ColdCounterIncrementsOnNonResidentDispatch) {
    // Build residency for experts 0, 1, 2 with resident_k=3.
    MoEResidencyTracker tracker(8, 3);
    for (int i = 0; i < 5; ++i) tracker.record_dispatch(0);
    for (int i = 0; i < 4; ++i) tracker.record_dispatch(1);
    for (int i = 0; i < 3; ++i) tracker.record_dispatch(2);

    // Now 0, 1, 2 are all in resident set
    ASSERT_TRUE(tracker.is_resident(0));
    ASSERT_TRUE(tracker.is_resident(1));
    ASSERT_TRUE(tracker.is_resident(2));

    // Dispatch non-resident expert 7 — cold counter should increment by 1
    const uint64_t cold_before = tracker.cold_dispatch_count();
    tracker.record_dispatch(7);
    EXPECT_EQ(tracker.cold_dispatch_count(), cold_before + 1u);
}

TEST(MoEResidencyTest, ColdCounterStaysStableOnResidentDispatch) {
    MoEResidencyTracker tracker(8, 3);
    for (int i = 0; i < 5; ++i) tracker.record_dispatch(0);
    for (int i = 0; i < 4; ++i) tracker.record_dispatch(1);
    for (int i = 0; i < 3; ++i) tracker.record_dispatch(2);

    ASSERT_TRUE(tracker.is_resident(0));

    // Dispatch resident expert 0 — cold counter must not change
    const uint64_t cold_before = tracker.cold_dispatch_count();
    tracker.record_dispatch(0);
    EXPECT_EQ(tracker.cold_dispatch_count(), cold_before);
}

TEST(MoEResidencyTest, ColdCounterEarlyDispatchesAreAllCold) {
    // At construction the resident set is empty, so the very first dispatch
    // is always cold (nothing is pinned yet).
    MoEResidencyTracker tracker(4, 2);
    tracker.record_dispatch(0);
    EXPECT_EQ(tracker.cold_dispatch_count(), 1u);
}

// ── snapshot_metrics ──────────────────────────────────────────────────────────

TEST(MoEResidencyTest, SnapshotMetricsMatchesColdCount) {
    MoEResidencyTracker tracker(8, 3);
    for (int i = 0; i < 5; ++i) tracker.record_dispatch(0);
    tracker.record_dispatch(5);  // cold dispatch

    MoEMetrics m = tracker.snapshot_metrics();
    EXPECT_EQ(m.cold_dispatch_count, tracker.cold_dispatch_count());
}

// ── reset() ───────────────────────────────────────────────────────────────────

TEST(MoEResidencyTest, ResetClearsAllState) {
    MoEResidencyTracker tracker(8, 3);
    for (int i = 0; i < 4; ++i) tracker.record_dispatch(0);
    tracker.record_dispatch(5);

    tracker.reset();

    EXPECT_EQ(tracker.cold_dispatch_count(), 0u);
    EXPECT_TRUE(tracker.resident_set().empty());
    for (uint64_t f : tracker.frequencies())
        EXPECT_EQ(f, 0u);
}

TEST(MoEResidencyTest, ResetAllowsReuseFromCleanState) {
    MoEResidencyTracker tracker(4, 1);
    tracker.record_dispatch(0);
    tracker.reset();

    // After reset, first dispatch should again count as cold
    tracker.record_dispatch(2);
    EXPECT_EQ(tracker.cold_dispatch_count(), 1u);
    EXPECT_TRUE(tracker.is_resident(2));
}

// ── Fail-loud: out-of-range expert_id ────────────────────────────────────────

TEST(MoEResidencyTest, OutOfRangeExpertIdThrowsLow) {
    MoEResidencyTracker tracker(8, 3);
    EXPECT_THROW(tracker.record_dispatch(-1), std::runtime_error);
}

TEST(MoEResidencyTest, OutOfRangeExpertIdThrowsHigh) {
    MoEResidencyTracker tracker(8, 3);
    EXPECT_THROW(tracker.record_dispatch(8), std::runtime_error);
}

// ── Frequency tracking ────────────────────────────────────────────────────────

TEST(MoEResidencyTest, FrequenciesAccumulateCorrectly) {
    MoEResidencyTracker tracker(4, 2);
    tracker.record_dispatch(0);
    tracker.record_dispatch(0);
    tracker.record_dispatch(1);

    const auto& freq = tracker.frequencies();
    EXPECT_EQ(freq[0], 2u);
    EXPECT_EQ(freq[1], 1u);
    EXPECT_EQ(freq[2], 0u);
    EXPECT_EQ(freq[3], 0u);
}
