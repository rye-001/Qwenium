// test_attention.cpp — MRopeSections, the validated part of layers/attention.
//
// This file used to test AttentionLayer::build(). That class was deleted on
// 2026-08-29: it had no production caller and its project_qkv duplicated the
// projection logic recipes do inline, so the tests were exercising a dead
// parallel implementation and giving false confidence about attention.
//
// What remains is MRopeSections::from_widths, which IS live (qwen35/qwen36 read
// rope.dimension_sections through it). The attention free functions the recipes
// actually call have no direct unit test; they are covered at recipe level by
// test_qwen35_forward_attn, test_qwen36_forward and the bitwise recipe gates.

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "../../src/layers/attention.h"
#include "../../src/layers/layer.h"

// ── MRopeSections::from_widths (P2 of docs/plan-qwen35-vision-impl.md) ───────
//
// Absence of the GGUF key is a legitimate state (text-only checkpoint) and is
// handled by the caller. What this validates is a key that IS present but
// contradicts itself — a real defect in the file, where fail-loud applies.

TEST(MRopeSections, DefaultIsInactiveSoRopeStaysNeox) {
    MRopeSections m;
    EXPECT_FALSE(m.active);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(m.widths[i], 0);
}

TEST(MRopeSections, AcceptsTheQwen35FamilyShape) {
    // Every Qwen 3.5-family GGUF: [11,11,10,0] against rope dim 64.
    const auto m = MRopeSections::from_widths({11, 11, 10, 0},
                                              "qwen35.rope.dimension_sections", 64);
    EXPECT_TRUE(m.active);
    EXPECT_EQ(m.widths[0], 11);
    EXPECT_EQ(m.widths[1], 11);
    EXPECT_EQ(m.widths[2], 10);
    EXPECT_EQ(m.widths[3], 0);
}

TEST(MRopeSections, RejectsWrongCount) {
    EXPECT_THROW(MRopeSections::from_widths({11, 11, 10}, "k", 64),
                 std::runtime_error);
    EXPECT_THROW(MRopeSections::from_widths({11, 11, 10, 0, 0}, "k", 64),
                 std::runtime_error);
}

// The dangerous one: widths that do not cover n_rot/2 make ggml's
// `sector % sect_dims` wrap, so dimensions silently rotate against the wrong
// position component. Nothing downstream errors — output just degrades.
TEST(MRopeSections, RejectsWidthsThatDoNotSumToNRotHalf) {
    try {
        MRopeSections::from_widths({8, 8, 8, 0}, "qwen35.rope.dimension_sections", 64);
        FAIL() << "expected a throw: 24 != 64/2";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("qwen35.rope.dimension_sections"), std::string::npos) << msg;
        EXPECT_NE(msg.find("32"), std::string::npos) << msg;  // expected
        EXPECT_NE(msg.find("24"), std::string::npos) << msg;  // actual
    }
}

TEST(MRopeSections, RejectsNegativeWidth) {
    EXPECT_THROW(MRopeSections::from_widths({-1, 12, 10, 11}, "k", 64),
                 std::runtime_error);
}

// ggml_rope_multi asserts sections[0]||sections[1]||sections[2] > 0; refuse
// before the assert fires so the message names the key instead of aborting.
TEST(MRopeSections, RejectsAllZeroLeadingSections) {
    EXPECT_THROW(MRopeSections::from_widths({0, 0, 0, 32}, "k", 64),
                 std::runtime_error);
}

// A different rope width is fine as long as the sum tracks it.
TEST(MRopeSections, AcceptsAnyWidthConsistentWithNRot) {
    const auto m = MRopeSections::from_widths({16, 16, 16, 16}, "k", 128);
    EXPECT_TRUE(m.active);
}
