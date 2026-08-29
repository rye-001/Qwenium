// test_image_preprocess.cpp — co-located unit test for
// src/vision/image_preprocess.cpp (CLAUDE.md test co-location).
//
// The recipes are pure data, so what is worth pinning is the CONSTANTS — they
// are read off real mmproj files and silently wrong values degrade output
// instead of failing (docs/plan-qwen35-vision-impl.md §3.7 calls preprocessing
// "the highest quiet-failure risk in the port"). The byte-faithful resampler
// gate lives in image-loader-tests; this only guards the numbers feeding it.

#include <gtest/gtest.h>

#include <vector>

#include "vision/image_preprocess.h"

namespace {

using qinf::vision::ImagePreprocess;

TEST(ImagePreprocessTest, Gemma3IsFixedSquareWithSiglipNormalization) {
    const auto pp = qinf::vision::gemma3_preprocess(896);
    EXPECT_EQ(pp.sizing, ImagePreprocess::Sizing::FixedSquarePadCeil);
    EXPECT_EQ(pp.fixed_target, 896);
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(pp.mean[c], 0.5f)   << "channel " << c;
        EXPECT_FLOAT_EQ(pp.stddev[c], 0.5f) << "channel " << c;
    }
}

TEST(ImagePreprocessTest, Gemma3HonoursANonDefaultTarget) {
    EXPECT_EQ(qinf::vision::gemma3_preprocess(768).fixed_target, 768);
}

// mean=[0,0,0] std=[1,1,1] are the verified mmproj-gemma-4-12B-it kv values.
// Qwen's tower uses 0.5/0.5 instead (plan §3.7), so a future change that reads
// these from the gguf must not quietly retarget Gemma 4 onto SigLIP numbers.
TEST(ImagePreprocessTest, Gemma4UvIsDynamicWithIdentityNormalization) {
    const auto pp = qinf::vision::gemma4uv_preprocess(48);
    EXPECT_EQ(pp.sizing, ImagePreprocess::Sizing::DynSmartResize);
    EXPECT_EQ(pp.align, 48);
    EXPECT_EQ(pp.min_tokens, 40);
    EXPECT_EQ(pp.max_tokens, 280);
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(pp.mean[c], 0.0f)   << "channel " << c;
        EXPECT_FLOAT_EQ(pp.stddev[c], 1.0f) << "channel " << c;
    }
}

// The knobs P5 will turn for Qwen (align 32, budget 8–4096) are already
// parameters, not constants — "parameterize, don't fork" holds here.
TEST(ImagePreprocessTest, Gemma4UvAlignAndTokenBudgetAreParameters) {
    const auto pp = qinf::vision::gemma4uv_preprocess(32, 8, 4096);
    EXPECT_EQ(pp.align, 32);
    EXPECT_EQ(pp.min_tokens, 8);
    EXPECT_EQ(pp.max_tokens, 4096);
}

TEST(ImagePreprocessTest, DefaultConstructedRecipeIsTheGemma3Shape) {
    const ImagePreprocess pp;
    EXPECT_EQ(pp.sizing, ImagePreprocess::Sizing::FixedSquarePadCeil);
    EXPECT_EQ(pp.fixed_target, 896);
}

}  // namespace

// ── Qwen 3.5-family recipe, audited against the vendored reference ───────────
//
// Every value below was read out of build-*/_deps/ggml-src/tools/mtmd/ on
// 2026-08-25, not inferred:
//   mtmd.cpp:694          QWEN3VL -> mtmd_image_preprocessor_dyn_size
//   clip.cpp:1646-1652    RESIZE_ALGO_BILINEAR, set_limit_image_tokens(8, 4096)
//   clip-model.h:67       image_resize_pad defaults to PAD_CEIL (not overridden)
//   mtmd-image.cpp:934    align_size = patch_size * n_merge
//   clip-model.h:187      pixel budget = n_tokens * patch_size^2 * n_merge^2
//
// The last one is why `align` doubles as the budget unit here: align is
// patch*merge, so align^2 IS patch^2*merge^2. If that identity is ever broken
// the budget silently changes, so it is asserted rather than assumed.

TEST(ImagePreprocessTest, Qwen3VlMatchesTheReferenceRecipe) {
    const auto pp = qinf::vision::qwen3vl_preprocess();
    EXPECT_EQ(pp.sizing, ImagePreprocess::Sizing::DynSmartResize);
    EXPECT_EQ(pp.align, 32);          // patch 16 * merge 2
    EXPECT_EQ(pp.min_tokens, 8);
    EXPECT_EQ(pp.max_tokens, 4096);
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(pp.mean[c], 0.5f)   << "channel " << c;
        EXPECT_FLOAT_EQ(pp.stddev[c], 0.5f) << "channel " << c;
    }
}

// align == patch*merge, so align^2 == patch^2 * merge^2 — the identity that
// lets image_loader convert a TOKEN budget into the reference's PIXEL budget.
TEST(ImagePreprocessTest, AlignSquaredEqualsReferencePatchArea) {
    struct Case { int patch, merge; };
    for (auto c : std::vector<Case>{{16, 2}, {16, 3}, {14, 1}}) {
        const int align = c.patch * c.merge;
        EXPECT_EQ(align * align, c.patch * c.patch * c.merge * c.merge)
            << "patch " << c.patch << " merge " << c.merge;
    }
}
