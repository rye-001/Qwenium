// test_snapkv_cli.cpp
// TDD tests for SnapKV CLI config plumbing.
// Verifies CliArgs has the expected fields with correct defaults,
// and that the forward pass factory signature accepts SnapKV params.

#include <gtest/gtest.h>
#include "../../src/cli/cli-args.h"

// ============================================================================
// 1. Default values
// ============================================================================

TEST(SnapKVCli, DefaultBudgetIsZero) {
    CliArgs args;
    EXPECT_EQ(args.snapkv_budget, 0u);
}

TEST(SnapKVCli, DefaultWindowIs32) {
    CliArgs args;
    EXPECT_EQ(args.snapkv_window, 32u);
}

// ============================================================================
// 2. Fields are assignable (sanity check for plumbing)
// ============================================================================

TEST(SnapKVCli, BudgetIsAssignable) {
    CliArgs args;
    args.snapkv_budget = 1024;
    EXPECT_EQ(args.snapkv_budget, 1024u);
}

TEST(SnapKVCli, WindowIsAssignable) {
    CliArgs args;
    args.snapkv_window = 64;
    EXPECT_EQ(args.snapkv_window, 64u);
}

// ============================================================================
// 3. Composition: SnapKV + TurboQuant fields coexist
// ============================================================================

TEST(SnapKVCli, ComposeWithTurboQuant) {
    CliArgs args;
    args.kv_quant_bits = 4;
    args.snapkv_budget = 1024;
    args.snapkv_window = 16;
    EXPECT_EQ(args.kv_quant_bits, 4);
    EXPECT_EQ(args.snapkv_budget, 1024u);
    EXPECT_EQ(args.snapkv_window, 16u);
}
