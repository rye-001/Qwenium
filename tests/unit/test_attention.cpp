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
#include <stdexcept>
#include <vector>

#include "ggml.h"

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


// ── build_attn_mha, flash-attention preconditions (--flash-attn) ─────────────
//
// The flash branch refuses two things rather than guessing, and both refusals
// are load-bearing:
//   * an F32 mask — ggml_flash_attn_ext hard-asserts F16, so without this check
//     a recipe that forgot the cast would abort inside ggml with no mention of
//     which recipe or layer;
// The F32-mask refusal is load-bearing: ggml_flash_attn_ext hard-asserts F16,
// so without it a recipe that forgot the cast would abort inside ggml with no
// mention of which recipe or layer. Softcap, by contrast, is FORWARDED — see
// ForwardsSoftcapToTheFlashKernel below. Both are pure graph-build checks, so
// they need no backend and no model.

namespace {
struct MhaCtx {
    ggml_context* ctx;
    ggml_cgraph*  gf;
    ggml_tensor  *q, *k, *v;
    MhaCtx() {
        ggml_init_params p{ 64 * ggml_tensor_overhead() + ggml_graph_overhead(),
                            nullptr, /*no_alloc=*/true };
        ctx = ggml_init(p);
        gf  = ggml_new_graph(ctx);
        const int d = 64, n_q = 1, n_head = 8, n_head_kv = 4, n_kv = 32;
        q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head,    n_q);
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_head_kv, n_kv, 1);
        v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_head_kv, n_kv, 1);
    }
    ~MhaCtx() { ggml_free(ctx); }
    ggml_tensor* mask(ggml_type t) { return ggml_new_tensor_2d(ctx, t, 32, 1); }
};
}  // namespace

TEST(BuildAttnMhaFlash, RefusesF32MaskNamingTheSlot) {
    MhaCtx c;
    EXPECT_THROW(build_attn_mha(c.ctx, c.gf, c.q, c.k, c.v, c.mask(GGML_TYPE_F32),
                                nullptr, 0.125f, 0, /*il=*/3, /*softcap=*/0.0f,
                                /*use_flash=*/true),
                 std::runtime_error);
}

TEST(BuildAttnMhaFlash, ForwardsSoftcapToTheFlashKernel) {
    // Gemma 2's attention softcap used to be refused here, while ggml's clamp
    // convention was unverified. It was then checked in both backends: the host
    // pre-divides (scale /= logit_softcap) and the kernel computes
    // logit_softcap*tanh(s*scale) — the scale applied BEFORE the clamp, which
    // is what build_softcap composed after ggml_scale does on the materialized
    // path. So it is forwarded, and a non-zero softcap must NOT throw.
    MhaCtx c;
    ggml_tensor* out = nullptr;
    EXPECT_NO_THROW(out = build_attn_mha(c.ctx, c.gf, c.q, c.k, c.v,
                                         c.mask(GGML_TYPE_F16), nullptr, 0.125f,
                                         0, /*il=*/3, /*softcap=*/30.0f,
                                         /*use_flash=*/true));
    ASSERT_NE(out, nullptr);
}

TEST(BuildAttnMhaFlash, MaterializedPathAcceptsF32MaskAndSoftcap) {
    // The refusals above must be flash-only: the default path is unchanged.
    MhaCtx c;
    EXPECT_NO_THROW(build_attn_mha(c.ctx, c.gf, c.q, c.k, c.v, c.mask(GGML_TYPE_F32),
                                   nullptr, 0.125f, 0, /*il=*/3, /*softcap=*/30.0f,
                                   /*use_flash=*/false));
}
