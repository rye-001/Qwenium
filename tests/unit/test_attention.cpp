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
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"

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

// ── M-RoPE layout: interleaved vs blocks (docs/note-verdict-img-ground.md) ───
//
// The Qwen 3.5 family sets MRopeSections::interleaved (ggml IMROPE), as trained;
// until 2026-10-02 it ran MROPE (blocks). Two facts that choice rests on, on the
// CPU kernel:
//   * text cannot tell the layouts apart — with all four components equal, both
//     equal plain NEOX bit for bit, so the switch changed no text output;
//   * an image can — with row and column components, the layouts rotate
//     different dimensions, so every image prefill moved.
namespace {
std::vector<float> rope_cpu(int mode, const std::vector<int32_t>& pos4, int n_tok) {
    const int d = 128, n_head = 2, n_rot = 64;
    ggml_init_params p{ 8 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, n_tok);
    float* xd = static_cast<float*>(x->data);
    for (int64_t i = 0; i < ggml_nelements(x); ++i) xd[i] = std::sin(0.37f * static_cast<float>(i));
    ggml_tensor* out = nullptr;
    if (mode == GGML_ROPE_TYPE_NEOX) {
        ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
        std::memcpy(pos->data, pos4.data(), n_tok * sizeof(int32_t));   // the t component
        out = ggml_rope_ext(ctx, x, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                            262144, 1.0e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    } else {
        ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n_tok);
        std::memcpy(pos->data, pos4.data(), 4 * n_tok * sizeof(int32_t));
        int sections[4] = {11, 11, 10, 0};   // the Qwen 3.5 family's widths
        out = ggml_rope_multi(ctx, x, pos, nullptr, n_rot, sections, mode,
                              262144, 1.0e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    const float* od = static_cast<const float*>(out->data);
    std::vector<float> res(od, od + ggml_nelements(out));
    ggml_free(ctx);
    return res;
}
}  // namespace

TEST(MRopeLayout, TextIsBitIdenticalToNeoxUnderBothLayouts) {
    const int n_tok = 5;
    std::vector<int32_t> pos4(4 * n_tok);
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < n_tok; ++i) pos4[k * n_tok + i] = 1000 + i;   // component-major
    const auto neox   = rope_cpu(GGML_ROPE_TYPE_NEOX,   pos4, n_tok);
    const auto blocks = rope_cpu(GGML_ROPE_TYPE_MROPE,  pos4, n_tok);
    const auto inter  = rope_cpu(GGML_ROPE_TYPE_IMROPE, pos4, n_tok);
    ASSERT_EQ(neox.size(), inter.size());
    EXPECT_EQ(std::memcmp(neox.data(), blocks.data(), neox.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(neox.data(), inter.data(),  neox.size() * sizeof(float)), 0);
}

TEST(MRopeLayout, AnImageTellsTheLayoutsApart) {
    const int nx = 3, ny = 2, n_tok = nx * ny, pos0 = 10;
    std::vector<int32_t> pos4(4 * n_tok);
    for (int i = 0; i < n_tok; ++i) {             // MRopePositionsInput's image layout
        pos4[0 * n_tok + i] = pos0;
        pos4[1 * n_tok + i] = pos0 + i / nx;
        pos4[2 * n_tok + i] = pos0 + i % nx;
        pos4[3 * n_tok + i] = 0;
    }
    const auto blocks = rope_cpu(GGML_ROPE_TYPE_MROPE,  pos4, n_tok);
    const auto inter  = rope_cpu(GGML_ROPE_TYPE_IMROPE, pos4, n_tok);
    EXPECT_NE(std::memcmp(blocks.data(), inter.data(), blocks.size() * sizeof(float)), 0);
}

TEST(MRopeSections, LayoutDefaultsToBlocks) {
    EXPECT_FALSE(MRopeSections{}.interleaved);
    EXPECT_FALSE(MRopeSections::from_widths({11, 11, 10, 0}, "k", 64).interleaved);
}
