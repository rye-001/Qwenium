// test_vision_encoder.cpp — Phase 2.3 + 2.4 of docs/plan-gemma-vision-impl.md.
//
// P2.3 smoke gate: build the SigLIP ViT forward graph, run it on a fixed
// gray-0.5 bitmap, assert a finite output of the expected shape. Proves the
// graph builds, allocates, computes on the shared backend, and reads back.
//
// P2.4 differential: run the FULL pipeline (ViT → 4×4 pool → RMS soft-emb-norm
// → projection) on the same gray input and compare against the captured
// llama.cpp reference (tests/fixtures/vision/siglip_gray_896.bin, [256, 2560]
// row-major n_embd-fastest). Structured after the feed_tokens precedent
// (docs/plan-feed-tokens.md): a primary RUNS test with a COARSE correctness
// gate, plus a DISABLED_ strict-bitwise variant that carries the measured
// FP noise floor (not red in CI; run with --gtest_also_run_disabled_tests).
//
// Reproducibility decision (bitwise vs coarse-ε): the encoder is BF16 weights
// through a long matmul/conv chain; vs llama.cpp's own ggml graph the expected
// divergence is low-bit FP reduction-order noise — the same class as
// feed_tokens (which resolved to token-stable + coarse ceiling, not bitwise).
// We therefore gate coarse (cosine-per-token + relative L2) and keep the
// bitwise check as a disabled diagnostic. The exact measured numbers are
// recorded in the disabled test's failure message once run.
//
// Self-skips when the mmproj file is absent (mirrors test_vision_loader.cpp).

#include <gtest/gtest.h>

#include <algorithm>
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
#include "../../src/vision/siglip_encoder.h"
#include "../../src/vision/vision_loader.h"
#include "../../src/vision/vision_model.h"
#include "session/persistent_image_embedding_store.h"
#include "../../src/graph_inputs/image_embedding_section.h"
#include "../../src/session/compat_header.h"

using qinf::vision::Bitmap;
using qinf::vision::SiglipEncoder;
using qinf::vision::VisionLoader;
using qinf::vision::VisionModel;
using qinf::session::CompatHeader;

namespace {

std::string get_mmproj_path() {
    if (const char* e = std::getenv("QINF_MMPROJ_PATH"))
        if (e[0]) return std::string(e);
    return "./mmproj-BF16.gguf";
}

std::string get_fixture_path() {
    if (const char* e = std::getenv("QINF_VISION_FIXTURE"))
        if (e[0]) return std::string(e);
    return "tests/fixtures/vision/siglip_gray_896.bin";
}

#define SKIP_IF_NO_MMPROJ()                                                \
    do {                                                                   \
        FILE* _f = std::fopen(get_mmproj_path().c_str(), "rb");            \
        if (!_f) GTEST_SKIP() << "mmproj GGUF not found at "                \
                              << get_mmproj_path()                         \
                              << " — skipping (set QINF_MMPROJ_PATH to "   \
                                 "override)";                              \
        std::fclose(_f);                                                   \
    } while (0)

// The same deterministic input the llama.cpp reference was captured with:
// gray 0.5 everywhere, channel-planar [3, 896, 896].
Bitmap make_gray_bitmap(const VisionModel& model) {
    const auto& cfg = model.config();
    Bitmap bmp;
    bmp.channels = static_cast<int>(cfg.num_channels);
    bmp.height   = static_cast<int>(cfg.image_size);
    bmp.width    = static_cast<int>(cfg.image_size);
    bmp.pixels.assign(
        static_cast<size_t>(cfg.num_channels) * cfg.image_size * cfg.image_size,
        0.5f);
    return bmp;
}

// Reads the captured llama.cpp reference. Returns empty on any failure so the
// caller can GTEST_SKIP with a precise message.
std::vector<float> read_fixture(const std::string& path, size_t expect_floats) {
    std::vector<float> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes > 0 && static_cast<size_t>(bytes) == expect_floats * sizeof(float)) {
        out.resize(expect_floats);
        size_t got = std::fread(out.data(), sizeof(float), expect_floats, f);
        if (got != expect_floats) out.clear();
    }
    std::fclose(f);
    return out;
}

struct DiffMetrics {
    size_t n            = 0;
    size_t mismatches   = 0;    // bitwise (memcmp) mismatched floats
    float  max_abs_diff = 0.0f;
    double rel_l2       = 0.0;  // ||a-b||_2 / ||b||_2
    double min_cos      = 1.0;  // worst per-token cosine similarity
    double mean_cos     = 1.0;  // mean per-token cosine similarity
};

// `a` = our encoder output, `b` = reference. Both [n_tokens, n_embd] row-major
// (n_embd-fastest), which is the layout of both our returned vector and the
// captured fixture.
DiffMetrics compare(const std::vector<float>& a, const std::vector<float>& b,
                    size_t n_tokens, size_t n_embd) {
    DiffMetrics m;
    m.n = a.size();
    double sum_sq_diff = 0.0, sum_sq_ref = 0.0, sum_cos = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            ++m.mismatches;
            m.max_abs_diff = std::max(m.max_abs_diff, std::fabs(a[i] - b[i]));
        }
        const double d = static_cast<double>(a[i]) - b[i];
        sum_sq_diff += d * d;
        sum_sq_ref  += static_cast<double>(b[i]) * b[i];
    }
    m.rel_l2 = sum_sq_ref > 0 ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0;

    for (size_t t = 0; t < n_tokens; ++t) {
        double dot = 0, na = 0, nb = 0;
        for (size_t j = 0; j < n_embd; ++j) {
            const double av = a[t * n_embd + j];
            const double bv = b[t * n_embd + j];
            dot += av * bv; na += av * av; nb += bv * bv;
        }
        const double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 1.0;
        m.min_cos = std::min(m.min_cos, cos);
        sum_cos += cos;
    }
    m.mean_cos = n_tokens ? sum_cos / static_cast<double>(n_tokens) : 1.0;
    return m;
}

// Coarse correctness ceilings (analogous to feed_tokens' kFeedTokensMaxAbsDiff).
// These gate "the encoder computes the right thing", robust to BF16 +
// reduction-order FP noise vs the llama.cpp reference. Tightened/loosened only
// with a recorded measurement, mirroring the feed_tokens owner decision.
constexpr double kMinCosCeiling = 0.999;   // worst-token cosine must exceed this
constexpr double kRelL2Ceiling  = 0.02;    // global relative L2 must be under this

// Backend for the encoder under test. Defaults to CPU (the captured reference's
// backend). With QINF_TEST_METAL set (and a Metal build), runs on Metal — used
// to validate the previously-unverified Metal encode path against the reference.
ggml_backend_t make_test_backend() {
#ifdef QINF_TEST_HAS_METAL
    if (const char* e = std::getenv("QINF_TEST_METAL"); e && e[0])
        return ggml_backend_metal_init();
#endif
    return ggml_backend_cpu_init();
}

}  // namespace

// ── P2.3 smoke gate: shape + finiteness on the gray input ────────────────────

TEST(SiglipEncoder, GraySmokeProducesFiniteProjectedShape) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_mmproj_path(), model);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr) << "ggml_backend_cpu_init failed";
    loader.load_tensors(model, backend);

    const auto& cfg = model.config();
    SiglipEncoder encoder(model, backend, cfg.projection_dim);

    Bitmap bmp = make_gray_bitmap(model);
    std::vector<float> emb = encoder.encode(bmp);

    // Full pipeline output: [mm_tokens_per_image, projection_dim] flattened.
    const size_t expect =
        static_cast<size_t>(cfg.mm_tokens_per_image) * cfg.projection_dim;
    EXPECT_EQ(emb.size(), expect)
        << "expected [mm_tokens_per_image=" << cfg.mm_tokens_per_image
        << " × projection_dim=" << cfg.projection_dim << "] = " << expect;

    size_t n_nonzero = 0;
    for (float x : emb) {
        ASSERT_TRUE(std::isfinite(x)) << "encoder produced NaN/Inf";
        if (x != 0.0f) ++n_nonzero;
    }
    EXPECT_GT(n_nonzero, 0u) << "encoder output is all zeros — graph likely "
                                "did not compute";

    ggml_backend_free(backend);
}

// ── Fail-loud: a bitmap that disagrees with the config is rejected ───────────

TEST(SiglipEncoder, RejectsMismatchedBitmapDimensions) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_mmproj_path(), model);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr);
    loader.load_tensors(model, backend);

    SiglipEncoder encoder(model, backend, model.config().projection_dim);

    Bitmap bad;
    bad.channels = 3;
    bad.height   = 448;   // wrong: config expects 896
    bad.width    = 448;
    bad.pixels.assign(static_cast<size_t>(3) * 448 * 448, 0.5f);

    EXPECT_THROW(encoder.encode(bad), std::runtime_error);

    ggml_backend_free(backend);
}

// ── P2.4 differential vs llama.cpp reference — COARSE gate (RUNS) ─────────────

TEST(SiglipEncoder, GrayDifferentialAgainstLlamaCppReferenceCoarse) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_mmproj_path(), model);

    const auto& cfg = model.config();
    const size_t n_tokens = cfg.mm_tokens_per_image;     // 256
    const size_t n_embd   = cfg.projection_dim;          // 2560
    const size_t expect   = n_tokens * n_embd;

    std::vector<float> ref = read_fixture(get_fixture_path(), expect);
    if (ref.empty()) {
        GTEST_SKIP() << "reference fixture not found / wrong size at "
                     << get_fixture_path() << " (expected " << expect
                     << " floats). Set QINF_VISION_FIXTURE to override.";
    }

    ggml_backend_t backend = make_test_backend();
    ASSERT_NE(backend, nullptr);
    loader.load_tensors(model, backend);

    SiglipEncoder encoder(model, backend, cfg.projection_dim);
    Bitmap bmp = make_gray_bitmap(model);
    std::vector<float> out = encoder.encode(bmp);
    ASSERT_EQ(out.size(), expect);

    DiffMetrics m = compare(out, ref, n_tokens, n_embd);

    // Always print the measured numbers so a tightening decision has evidence.
    std::fprintf(stderr,
        "[vision-diff] n=%zu mismatches=%zu (%.1f%%) max_abs=%.3e "
        "rel_l2=%.3e min_cos=%.6f mean_cos=%.6f\n",
        m.n, m.mismatches,
        m.n ? 100.0 * static_cast<double>(m.mismatches) / m.n : 0.0,
        m.max_abs_diff, m.rel_l2, m.min_cos, m.mean_cos);

    EXPECT_GT(m.min_cos, kMinCosCeiling)
        << "worst-token cosine similarity " << m.min_cos << " ≤ ceiling "
        << kMinCosCeiling << " vs the llama.cpp reference — the encoder is "
        << "computing a structurally different result, not low-bit FP noise.";
    EXPECT_LT(m.rel_l2, kRelL2Ceiling)
        << "relative L2 " << m.rel_l2 << " ≥ ceiling " << kRelL2Ceiling
        << " vs the llama.cpp reference.";

    ggml_backend_free(backend);
}

// ── Vision V1: embedding cache is a valid ViT-skip substitute ────────────────
// The cheap win (docs/plan-session-snapshot.md, V1): persist the encoder output
// and reuse it to skip the SigLIP pass. Validity hinges on (1) encode being
// run-to-run byte-deterministic on a fixed backend (so a cached embedding equals
// a fresh encode), and (2) the persist round-trip being byte (a pure memcpy).
// With both, a cache hit is indistinguishable from re-encoding.

TEST(SiglipEncoder, V1EmbeddingCacheIsByteExactVitSkipSubstitute) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_mmproj_path(), model);
    ggml_backend_t backend = make_test_backend();
    ASSERT_NE(backend, nullptr);
    loader.load_tensors(model, backend);

    const auto& cfg = model.config();
    SiglipEncoder encoder(model, backend, cfg.projection_dim);

    Bitmap bmp = make_gray_bitmap(model);
    bmp.content_id = 0xC0FFEEULL;  // producer-set cache key (encoder is content-blind)

    std::vector<float> e1 = encoder.encode(bmp);
    std::vector<float> e2 = encoder.encode(bmp);
    ASSERT_EQ(e1.size(), e2.size());

    // (1) Encode determinism — the byte-vs-token-stable hinge (the plan's
    // "encode batch-shape invariance" open question). SigLIP is a fixed 256-token
    // shape, so a fixed backend should reproduce bit-for-bit.
    const bool encode_byte =
        std::memcmp(e1.data(), e2.data(), e1.size() * sizeof(float)) == 0;
    std::fprintf(stderr, "[v1] backend=%s encode determinism: %s (%zu floats)\n",
                 ggml_backend_name(backend),
                 encode_byte ? "BYTE-identical" : "DIFFERS", e1.size());
    EXPECT_TRUE(encode_byte)
        << "SigLIP encode is not run-to-run byte-deterministic on this backend — "
           "V1 cache would be only a token-stable substitute, not byte.";

    // (2) Persist + reload via the store, and (3) confirm a second loader HITS
    // without re-encoding, byte-identical to the fresh encode.
    const std::string dir = std::string(testing::TempDir()) + "/v1_siglip_store";
    CompatHeader h = make_vision_header("gemma3-siglip", cfg.projection_dim,
                                        ggml_backend_name(backend));

    int encode_calls = 0;
    auto make = [&]() {
        ++encode_calls;
        qinf::vision::ImageEmbedding e;
        e.content_id = bmp.content_id;
        e.n_embd = cfg.projection_dim;
        e.n_tokens = cfg.mm_tokens_per_image;
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
        << "reloaded embedding differs from a fresh encode — ViT-skip invalid";

    std::remove(
        (dir + "/img-" + std::to_string(bmp.content_id) + ".embd").c_str());
    ggml_backend_free(backend);
}

// ── P2.4 strict bitwise — DISABLED diagnostic (noise-floor measurement) ──────

TEST(SiglipEncoder, DISABLED_GrayDifferentialAgainstLlamaCppReferenceBitwise) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_mmproj_path(), model);

    const auto& cfg = model.config();
    const size_t n_tokens = cfg.mm_tokens_per_image;
    const size_t n_embd   = cfg.projection_dim;
    const size_t expect   = n_tokens * n_embd;

    std::vector<float> ref = read_fixture(get_fixture_path(), expect);
    if (ref.empty()) GTEST_SKIP() << "reference fixture not found / wrong size.";

    ggml_backend_t backend = make_test_backend();
    ASSERT_NE(backend, nullptr);
    loader.load_tensors(model, backend);

    SiglipEncoder encoder(model, backend, cfg.projection_dim);
    Bitmap bmp = make_gray_bitmap(model);
    std::vector<float> out = encoder.encode(bmp);
    ASSERT_EQ(out.size(), expect);

    DiffMetrics m = compare(out, ref, n_tokens, n_embd);

    EXPECT_EQ(m.mismatches, 0u)
        << "NOISE-FLOOR MEASUREMENT (docs/plan-gemma-vision-impl.md P2.4): "
        << "our SigLIP encoder is NOT bitwise-reproducible vs the captured "
        << "llama.cpp reference. mismatched=" << m.mismatches << "/" << m.n
        << " max_abs=" << m.max_abs_diff << " rel_l2=" << m.rel_l2
        << " min_cos=" << m.min_cos << " mean_cos=" << m.mean_cos
        << ". Expected: low-bit FP reduction-order divergence (BF16 weights, "
        << "conv2d im2col + long matmul chain), same class as feed_tokens — "
        << "the shipping gate is the coarse cosine/rel-L2 test, not bitwise.";

    ggml_backend_free(backend);
}
