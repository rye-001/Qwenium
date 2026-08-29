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
#include <cstring>
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
#include "session/persistent_image_embedding_store.h"
#include "../../src/graph_inputs/image_embedding_section.h"
#include "../../src/session/compat_header.h"

using qinf::vision::Bitmap;
using qinf::vision::Gemma4UvEncoder;
using qinf::vision::VisionLoader;
using qinf::vision::VisionModel;
using qinf::vision::VisionProjectorType;
using qinf::session::CompatHeader;

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

// ── Vision V1: embedding cache is a valid ViT-skip substitute ────────────────
// The gemma4uv counterpart of SiglipEncoder.V1EmbeddingCacheIsByteExactVitSkip
// Substitute, closing the plan's open question for the SECOND encoder family
// (docs/plan-session-snapshot.md, "Encode batch-shape invariance"). gemma4uv's
// token count is VARIABLE across images (causal, raw pre-LM form) — but for ONE
// fixed image the batch shape is fixed, so a fixed backend should still encode
// byte-for-byte run-to-run, making the cached raw embedding a BYTE (not merely
// token-stable) ViT-skip substitute. This test is the falsifier for that claim:
// (1) two encodes of the same bitmap are byte-identical, (2) the persist
// round-trip is byte (pure memcpy), (3) a fresh store HITS byte-identical to the
// original encode.
TEST(Gemma4UvEncoder, V1EmbeddingCacheIsByteExactVitSkipSubstitute) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(mmproj_path(), model);
    ggml_backend_t backend = make_test_backend();
    ASSERT_NE(backend, nullptr);
    loader.load_tensors(model, backend);

    const auto& cfg = model.config();
    Gemma4UvEncoder encoder(model, backend, cfg.projection_dim);

    Bitmap bmp = make_gray_bitmap(480, 480);   // fixed dims → 10·10 = 100 tokens
    bmp.content_id = 0xC0FFEEULL;               // producer-set key (encoder is content-blind)
    const uint32_t n_tokens = encoder.mm_tokens_for(bmp);
    ASSERT_EQ(n_tokens, 100u);

    std::vector<float> e1 = encoder.encode(bmp);
    std::vector<float> e2 = encoder.encode(bmp);
    ASSERT_EQ(e1.size(), e2.size());
    ASSERT_EQ(e1.size(), static_cast<size_t>(n_tokens) * cfg.projection_dim);

    // (1) Encode determinism — the byte-vs-token-stable hinge. Fixed image ⇒
    // fixed batch shape ⇒ a fixed backend should reproduce bit-for-bit.
    const bool encode_byte =
        std::memcmp(e1.data(), e2.data(), e1.size() * sizeof(float)) == 0;
    std::fprintf(stderr, "[v1] backend=%s gemma4uv encode determinism: %s (%zu floats)\n",
                 ggml_backend_name(backend),
                 encode_byte ? "BYTE-identical" : "DIFFERS", e1.size());
    EXPECT_TRUE(encode_byte)
        << "gemma4uv encode is not run-to-run byte-deterministic on this backend "
           "— V1 cache is only a token-stable substitute for this family, not byte.";

    // (2) Persist + (3) a second loader HITS without re-encoding, byte-identical
    // to the fresh encode. n_tokens is per-image here (no fixed constant).
    const std::string dir = std::string(testing::TempDir()) + "/v1_gemma4uv_store";
    CompatHeader h = make_vision_header("gemma4uv", cfg.projection_dim,
                                        ggml_backend_name(backend));

    int encode_calls = 0;
    auto make = [&]() {
        ++encode_calls;
        qinf::vision::ImageEmbedding e;
        e.content_id = bmp.content_id;
        e.n_embd = cfg.projection_dim;
        e.n_tokens = n_tokens;
        e.data = encoder.encode(bmp);
        return e;
    };
    qinf::vision::ImageEmbedding first =
        PersistentImageEmbeddingStore(dir, h).get_or_encode(bmp.content_id, make);
    EXPECT_EQ(encode_calls, 1) << "first lookup should miss and encode once";

    qinf::vision::ImageEmbedding hit;
    ASSERT_TRUE(PersistentImageEmbeddingStore(dir, h).try_load(bmp.content_id, hit))
        << "second lookup (fresh store, same identity) should hit disk";
    ASSERT_EQ(hit.data.size(), e1.size());
    EXPECT_EQ(std::memcmp(hit.data.data(), e1.data(), e1.size() * sizeof(float)), 0)
        << "reloaded gemma4uv embedding differs from a fresh encode — ViT-skip invalid";

    std::remove(
        (dir + "/img-" + std::to_string(bmp.content_id) + ".embd").c_str());
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
