// test_qwen3vl_encoder.cpp — co-located unit test for
// src/vision/qwen3vl_encoder.cpp (CLAUDE.md test co-location).
//
// SCOPE, stated honestly: these are structural and behavioural invariants, not
// a numerical oracle. A captured llama.cpp encoder reference is the gate P3's
// plan asks for and it is NOT here — see docs/plan-qwen35-vision-impl.md §8.6.
// What these tests do catch is the class of bug that would otherwise pass
// silently: wrong token counts, a dead position input, an encoder that ignores
// its image, non-determinism, and every fail-loud precondition.
//
// Model-gated: needs the Qwen mmproj. Skips (never fails) without it, matching
// every other model-backed suite here.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef QINF_TEST_HAS_METAL
#include "ggml-metal.h"
#endif

#include "../../src/vision/bitmap.h"
#include "../../src/vision/image_preprocess.h"
#include "../../src/vision/qwen3vl_encoder.h"
#include "../../src/vision/vision_loader.h"
#include "../../src/vision/vision_model.h"

namespace {

using qinf::vision::Bitmap;
using qinf::vision::Qwen3VlEncoder;
using qinf::vision::VisionLoader;
using qinf::vision::VisionModel;

std::string qwen_mmproj_path() {
    if (const char* e = std::getenv("QINF_QWEN_MMPROJ_PATH"))
        if (e[0]) return std::string(e);
    return "./models/Qwen3.6-mtp-mmproj-BF16.gguf";
}

#define SKIP_IF_NO_MMPROJ()                                                   \
    do {                                                                      \
        FILE* _f = std::fopen(qwen_mmproj_path().c_str(), "rb");              \
        if (!_f) GTEST_SKIP() << "Qwen mmproj not found at "                   \
                              << qwen_mmproj_path()                           \
                              << " (set QINF_QWEN_MMPROJ_PATH)";              \
        std::fclose(_f);                                                      \
    } while (0)

// Encoder backend. CPU by default (portable, deterministic, but a 768² encode
// is ~4 minutes); QINF_TEST_METAL runs the production path in seconds. Mirrors
// test_multimodal_prefill.cpp::make_encoder_backend.
ggml_backend_t make_encoder_backend() {
#ifdef QINF_TEST_HAS_METAL
    if (const char* e = std::getenv("QINF_TEST_METAL"); e && e[0])
        return ggml_backend_metal_init();
#endif
    return ggml_backend_cpu_init();
}

// Loaded once — parsing + copying 334 tensors per test would dominate runtime.
struct Fixture {
    VisionModel    model;
    VisionLoader   loader;
    ggml_backend_t backend = nullptr;

    Fixture() {
        backend = make_encoder_backend();
        loader.parse_metadata(qwen_mmproj_path(), model);
        loader.load_tensors(model, backend);
    }
    ~Fixture() { if (backend) ggml_backend_free(backend); }
};

Fixture& fixture() { static Fixture f; return f; }

// Synthetic bitmap: channel-planar [C,H,W], deterministic non-uniform content.
Bitmap make_bitmap(int w, int h, float phase = 0.0f) {
    Bitmap b;
    b.width = w; b.height = h; b.channels = 3;
    b.pixels.resize(static_cast<size_t>(3) * w * h);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                b.pixels[(static_cast<size_t>(c) * h + y) * w + x] =
                    std::sin(0.05f * (x + 3 * y + 17 * c) + phase);
    return b;
}

}  // namespace

// ── Token accounting ─────────────────────────────────────────────────────────
// (W/32)·(H/32): patch 16, then the 2×2 merge. Getting this wrong misaligns the
// reserved placeholder span against the embeddings — a silent corruption.
TEST(Qwen3VlEncoder, TokenCountIsMergedPatchGrid) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);

    EXPECT_EQ(enc.mm_tokens_for(make_bitmap(64, 64)),   (64/32) * (64/32));
    EXPECT_EQ(enc.mm_tokens_for(make_bitmap(128, 64)),  (128/32) * (64/32));
    EXPECT_EQ(enc.mm_tokens_for(make_bitmap(256, 224)), (256/32) * (224/32));
}

TEST(Qwen3VlEncoder, ProjectionDimIsTheMergerOutputWidth) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);
    // 2048, not 8192 — mm.2.weight is [n_embd*4, projection_dim]. llama.cpp
    // computes the latter here and refuses to load (#20899).
    EXPECT_EQ(enc.projection_dim(), 2048u);
}

// ── Fail-loud preconditions ──────────────────────────────────────────────────
TEST(Qwen3VlEncoder, RefusesUnalignedDimensions) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);
    // 48 is a multiple of the patch (16) but NOT of patch·merge (32); the merge
    // would silently drop a half-block.
    EXPECT_THROW(enc.encode(make_bitmap(48, 64)), std::runtime_error);
    EXPECT_THROW(enc.encode(make_bitmap(64, 48)), std::runtime_error);
}

TEST(Qwen3VlEncoder, RefusesWrongChannelCountAndBufferSize) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);

    Bitmap wrong_c = make_bitmap(64, 64);
    wrong_c.channels = 1;
    EXPECT_THROW(enc.encode(wrong_c), std::runtime_error);

    Bitmap short_buf = make_bitmap(64, 64);
    short_buf.pixels.pop_back();
    EXPECT_THROW(enc.encode(short_buf), std::runtime_error);
}

TEST(Qwen3VlEncoder, RefusesMismatchedTextEmbedDim) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    // Seam A: mmproj projection_dim must equal the host model's width.
    EXPECT_THROW(Qwen3VlEncoder(f.model, f.backend, 4096u), std::runtime_error);
}

// ── Output shape and health ──────────────────────────────────────────────────
TEST(Qwen3VlEncoder, ProducesFiniteEmbeddingsOfTheDeclaredShape) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);

    const Bitmap b = make_bitmap(64, 64);
    const auto emb = enc.encode(b);

    ASSERT_EQ(emb.size(),
              static_cast<size_t>(enc.mm_tokens_for(b)) * enc.projection_dim());
    for (float x : emb) ASSERT_TRUE(std::isfinite(x)) << "non-finite output";

    // Not collapsed to a constant — a dead graph would still be "finite".
    double mean = 0; for (float x : emb) mean += x; mean /= emb.size();
    double var = 0;  for (float x : emb) var += (x - mean) * (x - mean);
    EXPECT_GT(var / emb.size(), 1e-8) << "embeddings are constant";
}

TEST(Qwen3VlEncoder, IsDeterministic) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);
    const Bitmap b = make_bitmap(64, 64);
    EXPECT_EQ(enc.encode(b), enc.encode(b));
}

// The encoder must actually READ the image.
TEST(Qwen3VlEncoder, DifferentImagesGiveDifferentEmbeddings) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);

    const auto a = enc.encode(make_bitmap(64, 64, 0.0f));
    const auto c = enc.encode(make_bitmap(64, 64, 1.3f));
    ASSERT_EQ(a.size(), c.size());
    double d = 0; for (size_t i = 0; i < a.size(); ++i) d += std::fabs(a[i] - c[i]);
    EXPECT_GT(d / a.size(), 1e-4) << "encoder appears to ignore its input";
}

// Position information must reach the output. If M-RoPE or the learned
// position embeddings were dropped, a spatially permuted image would encode to
// the same multiset of tokens; this catches a silently dead position path.
TEST(Qwen3VlEncoder, PositionsAffectTheOutput) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);

    Bitmap base = make_bitmap(64, 64);
    // Mirror horizontally: identical pixel multiset, different arrangement.
    Bitmap flipped = base;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x)
                flipped.pixels[(static_cast<size_t>(c) * 64 + y) * 64 + x] =
                    base.pixels[(static_cast<size_t>(c) * 64 + y) * 64 + (63 - x)];

    const auto a = enc.encode(base);
    const auto b = enc.encode(flipped);
    double d = 0; for (size_t i = 0; i < a.size(); ++i) d += std::fabs(a[i] - b[i]);
    EXPECT_GT(d / a.size(), 1e-4) << "spatial arrangement did not change output";
}

// The learned position embeddings live on a 48×48 grid. A 768×768 image hits
// that grid exactly and SHORT-CIRCUITS the bilinear resize; anything else takes
// the ggml_interpolate path. Both must work — this exercises the branch the
// other tests never reach.
TEST(Qwen3VlEncoder, HandlesTheNativePosEmbedGridWithoutResize) {
    SKIP_IF_NO_MMPROJ();
    // 2304 patches: ~4 min on CPU, seconds under QINF_TEST_METAL=1.
    auto& f = fixture();
    const auto& cfg = f.model.config();
    const int native = static_cast<int>(cfg.image_size);   // 768
    Qwen3VlEncoder enc(f.model, f.backend, cfg.projection_dim);

    const Bitmap b = make_bitmap(native, native);
    EXPECT_EQ(enc.mm_tokens_for(b),
              static_cast<uint32_t>((native / 32) * (native / 32)));
    const auto emb = enc.encode(b);
    ASSERT_EQ(emb.size(),
              static_cast<size_t>(enc.mm_tokens_for(b)) * enc.projection_dim());
    for (float x : emb) ASSERT_TRUE(std::isfinite(x));
}

// Non-square images must not transpose the grid.
TEST(Qwen3VlEncoder, HandlesNonSquareImages) {
    SKIP_IF_NO_MMPROJ();
    auto& f = fixture();
    Qwen3VlEncoder enc(f.model, f.backend, f.model.config().projection_dim);

    const Bitmap wide = make_bitmap(128, 64);
    const Bitmap tall = make_bitmap(64, 128);
    EXPECT_EQ(enc.mm_tokens_for(wide), enc.mm_tokens_for(tall));
    EXPECT_EQ(enc.encode(wide).size(), enc.encode(tall).size());
}
