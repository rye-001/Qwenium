// test_gemma4uv_encoder.cpp — Gemma 4 unified-vision (`gemma4uv`) encoder
// (docs/plan-gemma4-vision-impl.md §4, §8).
//
// Smoke / shape / finiteness + loader acceptance gates. The authoritative
// bit-level encoder differential vs llama.cpp's clip_graph_gemma4uv lives in
// the DISABLED_ block below (needs a captured reference fixture); the coarse
// gate here proves the blockless graph builds, allocates, computes on the
// shared backend (Metal when QINF_TEST_METAL is set), and reads back a finite
// output of the per-image token count.
//
// Defaults to the on-disk mmproj-gemma-4-12B-it-Q8_0.gguf; self-skips if absent.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef QINF_TEST_HAS_METAL
#include "ggml-metal.h"
#endif

#include "../../src/vision/bitmap.h"
#include "../../src/vision/gemma4uv_encoder.h"
#include "../../src/vision/vision_loader.h"
#include "../../src/vision/vision_model.h"

using qinf::vision::Bitmap;
using qinf::vision::Gemma4UvEncoder;
using qinf::vision::VisionLoader;
using qinf::vision::VisionModel;
using qinf::vision::VisionProjectorType;

namespace {

std::string mmproj_path() {
    if (const char* e = std::getenv("QINF_GEMMA4_MMPROJ_PATH"))
        if (e[0]) return std::string(e);
    return "./mmproj-gemma-4-12B-it-Q8_0.gguf";
}

#define SKIP_IF_NO_MMPROJ()                                                   \
    do {                                                                      \
        FILE* _f = std::fopen(mmproj_path().c_str(), "rb");                   \
        if (!_f) GTEST_SKIP() << "gemma4uv mmproj not found at "              \
                              << mmproj_path()                                \
                              << " (set QINF_GEMMA4_MMPROJ_PATH to override)";\
        std::fclose(_f);                                                      \
    } while (0)

ggml_backend_t make_test_backend() {
#ifdef QINF_TEST_HAS_METAL
    if (const char* e = std::getenv("QINF_TEST_METAL"); e && e[0])
        return ggml_backend_metal_init();
#endif
    return ggml_backend_cpu_init();
}

// A uniform-gray bitmap at W×H, channel-planar [3, H, W]. W,H must be multiples
// of the effective patch (48).
Bitmap make_gray_bitmap(int w, int h, float v = 0.5f) {
    Bitmap bmp;
    bmp.channels = 3;
    bmp.width    = w;
    bmp.height   = h;
    bmp.pixels.assign(static_cast<size_t>(3) * w * h, v);
    return bmp;
}

}  // namespace

// ── Loader acceptance: the gemma4uv mmproj parses into a Gemma4Uv config ──────
TEST(Gemma4UvLoader, AcceptsGemma4UvMmprojConfig) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(mmproj_path(), model);

    const auto& cfg = model.config();
    EXPECT_EQ(cfg.projector_type, VisionProjectorType::Gemma4Uv);
    EXPECT_EQ(cfg.hidden_size, 3840u);
    EXPECT_EQ(cfg.projection_dim, 3840u);
    EXPECT_EQ(cfg.patch_size, 16u);
    EXPECT_EQ(cfg.n_merge, 3u);          // effective patch = 48
    EXPECT_EQ(cfg.num_layers, 0u);       // blockless
    EXPECT_EQ(cfg.mm_tokens_per_image, 0u);  // variable, not a constant
}

// ── mm_tokens_for: pure (W/P)·(H/P), P = 48 ──────────────────────────────────
TEST(Gemma4UvEncoder, MmTokensForDerivesFromDims) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(mmproj_path(), model);
    ggml_backend_t backend = make_test_backend();
    ASSERT_NE(backend, nullptr);
    loader.load_tensors(model, backend);

    Gemma4UvEncoder encoder(model, backend, model.config().projection_dim);

    EXPECT_EQ(encoder.mm_tokens_for(make_gray_bitmap(480, 480)), 100u);  // 10·10
    EXPECT_EQ(encoder.mm_tokens_for(make_gray_bitmap(48 * 16, 48 * 8)), 128u);  // 16·8
    EXPECT_EQ(encoder.projection_dim(), 3840u);

    ggml_backend_free(backend);
}

// ── Smoke: the blockless graph computes a finite output of the right shape ────
TEST(Gemma4UvEncoder, GraySmokeProducesFiniteShape) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(mmproj_path(), model);
    ggml_backend_t backend = make_test_backend();
    ASSERT_NE(backend, nullptr);
    loader.load_tensors(model, backend);

    const auto& cfg = model.config();
    Gemma4UvEncoder encoder(model, backend, cfg.projection_dim);

    // 480×480 → 10×10 = 100 tokens (within the 40..280 budget).
    Bitmap bmp = make_gray_bitmap(480, 480);
    const uint32_t expect_tokens = encoder.mm_tokens_for(bmp);   // 100
    ASSERT_EQ(expect_tokens, 100u);

    std::vector<float> emb = encoder.encode(bmp);

    ASSERT_EQ(emb.size(),
              static_cast<size_t>(expect_tokens) * cfg.projection_dim);
    for (size_t i = 0; i < emb.size(); ++i)
        ASSERT_TRUE(std::isfinite(emb[i]))
            << "non-finite at index " << i << " = " << emb[i];

    ggml_backend_free(backend);
}

// ── Fail-loud: a non-square in-budget image still encodes the right count ─────
TEST(Gemma4UvEncoder, NonSquareTokenCountMatchesEncode) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(mmproj_path(), model);
    ggml_backend_t backend = make_test_backend();
    ASSERT_NE(backend, nullptr);
    loader.load_tensors(model, backend);

    Gemma4UvEncoder encoder(model, backend, model.config().projection_dim);

    Bitmap bmp = make_gray_bitmap(48 * 12, 48 * 6);   // 12×6 = 72 tokens
    const uint32_t expect = encoder.mm_tokens_for(bmp);
    ASSERT_EQ(expect, 72u);
    std::vector<float> emb = encoder.encode(bmp);
    EXPECT_EQ(emb.size() / model.config().projection_dim, expect);

    ggml_backend_free(backend);
}

// ── Authoritative encoder differential vs llama.cpp clip_graph_gemma4uv ───────
// DISABLED pending a captured reference fixture (cheap to produce — the graph
// is blockless — via llama-mtmd-cli with a clip-output dump on a fixed image).
// When wired, gate coarse (min_cos > 0.999, rel_l2 < 0.02) per the mm-vs-mv /
// BF16 reduction-order fork; keep a strict bitwise variant DISABLED.
TEST(Gemma4UvEncoder, DISABLED_DifferentialAgainstLlamaCppReference) {
    GTEST_SKIP() << "needs tests/fixtures/vision/gemma4uv_*.bin reference "
                    "capture (plan §8 gate 1)";
}
