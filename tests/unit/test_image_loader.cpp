// test_image_loader.cpp — host-side Bitmap producer gate.
//
// Structural gates (SHAPE, VALUE RANGE, CHANNEL ORDER/LAYOUT, DETERMINISM) use
// synthesized solid-color BMPs (stb decodes BMP) to assert exact normalized
// values for known pixels without committing a binary fixture.
//
// MatchesLlamaCppGemma3Reference (Task 2) adds the real differential: a small
// non-square PNG fixture + probe points captured from llama.cpp mtmd's exact
// gemma3 encoder input, locking aspect-preserving resize + letterbox pad +
// normalization byte-for-byte against the reference.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/image/image_loader.h"

using qinf::image::load_image_to_bitmap;

namespace {

void put_u32(std::vector<unsigned char>& b, uint32_t v) {
    b.push_back(v & 0xFF); b.push_back((v >> 8) & 0xFF);
    b.push_back((v >> 16) & 0xFF); b.push_back((v >> 24) & 0xFF);
}
void put_u16(std::vector<unsigned char>& b, uint16_t v) {
    b.push_back(v & 0xFF); b.push_back((v >> 8) & 0xFF);
}

// Minimal 24bpp BMP, solid (r,g,b), w×h. Rows are bottom-up BGR, padded to 4.
std::string write_solid_bmp(int w, int h, unsigned char r, unsigned char g,
                            unsigned char b_, const std::string& path) {
    const int row_bytes = ((w * 3 + 3) / 4) * 4;
    const uint32_t pixel_off = 54;
    const uint32_t img_size = static_cast<uint32_t>(row_bytes) * h;

    std::vector<unsigned char> buf;
    buf.push_back('B'); buf.push_back('M');
    put_u32(buf, pixel_off + img_size);  // file size
    put_u32(buf, 0);                     // reserved
    put_u32(buf, pixel_off);             // pixel data offset
    put_u32(buf, 40);                    // info header size
    put_u32(buf, static_cast<uint32_t>(w));
    put_u32(buf, static_cast<uint32_t>(h));
    put_u16(buf, 1);                     // planes
    put_u16(buf, 24);                    // bpp
    put_u32(buf, 0);                     // compression (BI_RGB)
    put_u32(buf, img_size);
    put_u32(buf, 2835); put_u32(buf, 2835);  // ppm
    put_u32(buf, 0); put_u32(buf, 0);        // palette
    for (int y = 0; y < h; ++y) {
        int x = 0;
        for (; x < w; ++x) { buf.push_back(b_); buf.push_back(g); buf.push_back(r); }
        for (int p = w * 3; p < row_bytes; ++p) buf.push_back(0);  // row padding
    }

    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    f.close();
    return path;
}

std::string tmp_path(const char* name) {
    return std::string("/tmp/qinf_imgtest_") + name + ".bmp";
}

}  // namespace

// Solid RED → R-plane = +1, G/B-planes = -1. Pins normalization, RGB order
// (not BGR), channel-planar layout, and the output shape after resize.
TEST(ImageLoader, SolidRedNormalizesAndLaysOutPlanar) {
    const std::string path = write_solid_bmp(8, 8, 255, 0, 0, tmp_path("red"));
    const int T = 896;
    auto bmp = load_image_to_bitmap(path, T);

    ASSERT_EQ(bmp.channels, 3);
    ASSERT_EQ(bmp.height, T);
    ASSERT_EQ(bmp.width, T);
    ASSERT_EQ(bmp.pixels.size(), static_cast<size_t>(3) * T * T);

    const size_t plane = static_cast<size_t>(T) * T;
    for (size_t i = 0; i < plane; ++i) {
        ASSERT_TRUE(std::isfinite(bmp.pixels[i]));
        EXPECT_NEAR(bmp.pixels[i],              1.0f, 1e-3f);  // R = 255 → +1
        EXPECT_NEAR(bmp.pixels[plane + i],     -1.0f, 1e-3f);  // G = 0   → -1
        EXPECT_NEAR(bmp.pixels[2 * plane + i], -1.0f, 1e-3f);  // B = 0   → -1
    }
    std::remove(path.c_str());
}

// Same file twice → byte-identical pixels and identical content_id (Phase 7).
TEST(ImageLoader, IsDeterministic) {
    const std::string path = write_solid_bmp(13, 7, 30, 90, 200, tmp_path("det"));
    auto a = load_image_to_bitmap(path, 64);
    auto b = load_image_to_bitmap(path, 64);
    ASSERT_EQ(a.pixels.size(), b.pixels.size());
    EXPECT_EQ(a.content_id, b.content_id);
    EXPECT_NE(a.content_id, 0u);
    for (size_t i = 0; i < a.pixels.size(); ++i)
        ASSERT_EQ(a.pixels[i], b.pixels[i]) << "i=" << i;
    std::remove(path.c_str());
}

// Values stay in the SigLIP [-1, 1] range for an arbitrary color.
TEST(ImageLoader, ValuesInNormalizedRange) {
    const std::string path = write_solid_bmp(20, 20, 17, 200, 123, tmp_path("rng"));
    auto bmp = load_image_to_bitmap(path, 128);
    for (float v : bmp.pixels) {
        ASSERT_TRUE(std::isfinite(v));
        EXPECT_GE(v, -1.0001f);
        EXPECT_LE(v,  1.0001f);
    }
    std::remove(path.c_str());
}

TEST(ImageLoader, FailLoudOnMissingFile) {
    EXPECT_THROW(load_image_to_bitmap("/nonexistent/qinf_no_such_image.png", 896),
                 std::runtime_error);
}

// ── Differential vs llama.cpp mtmd reference (Task 2) ────────────────────────
//
// Locks our preprocessing to llama.cpp's gemma3 path byte-for-byte. The fixture
// (tests/fixtures/vision/) is a small non-square PNG + a set of probe points
// sampled from the EXACT [3,896,896] tensor llama-mtmd-cli feeds the SigLIP
// encoder (captured via an inp_raw dump on PROJECTOR_TYPE_GEMMA3). PNG is
// lossless, so stb decode is bit-identical across stb versions — the only
// variables are resize + pad + normalize, which is exactly what we gate.
//
// This catches the stretch-vs-aspect-pad bug decisively: ~40% of the probes lie
// in the black letterbox pad (value -1.0), which a stretch-to-square loader
// would fill with gradient content instead. ε is tight (same algorithm, same
// uint8 intermediate) but non-zero to absorb float-order differences.
namespace {
std::string find_fixture(const char* name) {
    std::vector<std::string> dirs;
    if (const char* e = std::getenv("QINF_VISION_FIXTURE_DIR"); e && e[0])
        dirs.push_back(e);
    dirs.insert(dirs.end(), {"tests/fixtures/vision", "../tests/fixtures/vision",
                             "../../tests/fixtures/vision",
                             "../../../tests/fixtures/vision"});
    for (const auto& d : dirs) {
        std::string p = d + "/" + name;
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return {};
}
}  // namespace

TEST(ImageLoader, MatchesLlamaCppGemma3Reference) {
    const std::string img    = find_fixture("preproc_input.png");
    const std::string probes = find_fixture("preproc_input.probes");
    if (img.empty() || probes.empty())
        GTEST_SKIP() << "vision fixtures not found (set QINF_VISION_FIXTURE_DIR "
                        "or run from repo root)";

    const int T = 896;
    auto bmp = load_image_to_bitmap(img, T);
    ASSERT_EQ(bmp.height, T);
    ASSERT_EQ(bmp.width, T);
    const size_t plane = static_cast<size_t>(T) * T;

    std::ifstream f(probes);
    std::string line;
    int C = 0, H = 0, W = 0, checked = 0, pad = 0;
    constexpr float kEps = 2e-3f;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tag;
        if (line.rfind("dims", 0) == 0) {
            ss >> tag >> C >> H >> W;
            ASSERT_EQ(C, 3); ASSERT_EQ(H, T); ASSERT_EQ(W, T)
                << "fixture dims must match target";
            continue;
        }
        int c, y, x; float expected;
        ss >> c >> y >> x >> expected;
        ASSERT_FALSE(ss.fail()) << "bad probe line: " << line;
        const float got = bmp.pixels[c * plane + static_cast<size_t>(y) * T + x];
        EXPECT_NEAR(got, expected, kEps)
            << "probe c=" << c << " y=" << y << " x=" << x;
        ++checked;
        if (std::abs(expected + 1.0f) < 1e-6f) ++pad;
    }
    ASSERT_GT(checked, 1000) << "fixture under-populated";
    ASSERT_GT(pad, 100) << "expected letterbox-pad probes (aspect-preserve gate)";
}

// ── Gemma 4 unified-vision preprocessing (dyn-size + /255 normalize) ──────────

using qinf::vision::gemma4uv_preprocess;

// A square input fully fills its smart_resize canvas (no pad), so every pixel is
// the solid color: red → R-plane = 255/255 = 1.0, G/B-planes = 0/255 = 0.0.
// Pins the /255 normalization (mean 0, std 1) — distinct from Gemma 3's −1..1.
TEST(ImageLoader, Gemma4UvSolidRedNormalizesTo255Scale) {
    const std::string path = write_solid_bmp(480, 480, 255, 0, 0,
                                             tmp_path("g4_red"));
    auto bmp = load_image_to_bitmap(path, gemma4uv_preprocess(/*align=*/48));
    std::remove(path.c_str());

    // 480×480 is in budget (100 tokens) and already aligned → canvas stays 480².
    ASSERT_EQ(bmp.width, 480);
    ASSERT_EQ(bmp.height, 480);
    const size_t plane = static_cast<size_t>(480) * 480;
    ASSERT_EQ(bmp.pixels.size(), 3 * plane);
    for (size_t i = 0; i < plane; ++i) {
        EXPECT_NEAR(bmp.pixels[i],             1.0f, 1e-3f) << "R i=" << i;
        EXPECT_NEAR(bmp.pixels[plane + i],     0.0f, 1e-3f) << "G i=" << i;
        EXPECT_NEAR(bmp.pixels[2 * plane + i], 0.0f, 1e-3f) << "B i=" << i;
    }
}

// Dynamic-size: dims are multiples of the effective patch 48, the token count
// (W/48)·(H/48) lands in the [40, 280] budget, and a non-square image gets a
// black letterbox pad at the normalized value 0.0 (mean 0, std 1).
TEST(ImageLoader, Gemma4UvDynSizeBudgetAndPad) {
    const std::string path = write_solid_bmp(1200, 400, 0, 0, 255,
                                             tmp_path("g4_blue"));
    auto bmp = load_image_to_bitmap(path, gemma4uv_preprocess(/*align=*/48));
    std::remove(path.c_str());

    ASSERT_EQ(bmp.width % 48, 0);
    ASSERT_EQ(bmp.height % 48, 0);
    const uint32_t tokens = static_cast<uint32_t>((bmp.width / 48) * (bmp.height / 48));
    EXPECT_GE(tokens, 40u);
    EXPECT_LE(tokens, 280u);

    // 3:1 aspect → vertical fit, horizontal pad. The pad pixels are black → 0.0
    // on every channel. Scan the top row (which contains pad for this geometry).
    const size_t plane = static_cast<size_t>(bmp.width) * bmp.height;
    ASSERT_EQ(bmp.pixels.size(), 3 * plane);
    int pad_pixels = 0, content_pixels = 0;
    for (int x = 0; x < bmp.width; ++x) {
        const float b = bmp.pixels[2 * plane + x];  // B plane, row 0
        if (std::fabs(b) < 1e-3f) ++pad_pixels;       // black pad → 0.0
        else if (std::fabs(b - 1.0f) < 1e-2f) ++content_pixels;  // blue content → 1.0
    }
    EXPECT_GT(pad_pixels, 0) << "expected a horizontal black pad for a 3:1 image";
}

// ── Differential vs llama.cpp mtmd reference, qwen3vl (P5) ───────────────────
//
// The Gemma 3 gate above locks the FIXED-SQUARE path. This locks the
// DYN_SMART_RESIZE path as the Qwen 3.5 family's projector parameterizes it —
// align 32, token budget 8…4096, mean/std 0.5 — against the tensor
// `mtmd_image_preprocessor_dyn_size` actually produces for the same PNG.
//
// §3.7 of docs/plan-qwen35-vision-impl.md calls preprocessing "the highest
// quiet-failure risk in the port", because wrong preprocessing DEGRADES output
// instead of erroring. That is why this gate is whole-tensor rather than
// probe-sampled: unlike Gemma 3's [3,896,896] (9.6 MB), a dyn-size canvas for a
// small image is ~180 KB, so nothing has to be sampled away.
//
// Two fixtures, one per branch of smart_resize:
//   preproc_input.png       157×97 → 160×96 (15 tokens) — the round-to-align
//                           branch, plus a 4-column letterbox pad.
//   preproc_input_tiny.png  21×53  → 64×160 (10 tokens) — the min_pixels
//                           UPSCALE branch (1113 px < the 8192 px floor),
//                           which the first fixture never reaches.
//
// Reference format: 3 int32 dims (C, H, W) then C·H·W f32 PLANAR — our Bitmap
// layout, de-interleaved at capture time from clip_image_f32's [H][W][C].
namespace {
void check_qwen3vl_reference(const char* png_name, const char* ref_name,
                             int want_w, int want_h) {
    const std::string img = find_fixture(png_name);
    const std::string ref = find_fixture(ref_name);
    if (img.empty() || ref.empty())
        GTEST_SKIP() << "vision fixtures not found (set QINF_VISION_FIXTURE_DIR "
                        "or run from repo root)";

    std::ifstream f(ref, std::ios::binary);
    ASSERT_TRUE(f.good()) << "cannot open reference " << ref;
    int32_t dims[3] = {0, 0, 0};
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    ASSERT_EQ(dims[0], 3) << "reference must be 3-channel";
    ASSERT_EQ(dims[1], want_h) << "reference height moved — re-capture, don't relax";
    ASSERT_EQ(dims[2], want_w) << "reference width moved — re-capture, don't relax";

    const size_t n = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
    std::vector<float> expected(n);
    f.read(reinterpret_cast<char*>(expected.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    ASSERT_EQ(static_cast<size_t>(f.gcount()), n * sizeof(float))
        << "reference truncated: " << ref;

    auto bmp = load_image_to_bitmap(img, qinf::vision::qwen3vl_preprocess());
    ASSERT_EQ(bmp.width, want_w);
    ASSERT_EQ(bmp.height, want_h);
    ASSERT_EQ(bmp.pixels.size(), n);

    // Both sides do the same arithmetic in the same order (uint8 intermediate,
    // then (v/255 − mean)/stddev), and at capture time every value in both
    // fixtures matched EXACTLY (76,800 in total) — so the gate is bit-equality,
    // not a tolerance.
    // If a future toolchain contracts this arithmetic and the gate starts
    // failing on a last-bit margin, that is a re-capture / documented-ε event
    // (see tests/fixtures/vision/README.md), not a licence to widen it quietly.
    constexpr float kEps = 0.0f;
    double max_diff = 0.0;
    size_t worst = 0, mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(bmp.pixels[i]) - expected[i]);
        if (d > max_diff) { max_diff = d; worst = i; }
        if (d > kEps) ++mismatches;
    }
    const size_t plane = static_cast<size_t>(want_w) * want_h;
    EXPECT_EQ(mismatches, 0u)
        << png_name << ": " << mismatches << " of " << n
        << " values differ from the llama.cpp reference; worst at c="
        << worst / plane << " y=" << (worst % plane) / want_w
        << " x=" << (worst % plane) % want_w << " (got " << bmp.pixels[worst]
        << ", want " << expected[worst] << ", |Δ| " << max_diff << ")";
}
}  // namespace

// Round-to-align branch, with a letterbox pad: 157×97 → 160×96, 5×3 = 15 tokens.
TEST(ImageLoader, MatchesLlamaCppQwen3VlReference) {
    check_qwen3vl_reference("preproc_input.png", "preproc_input.qwen3vl.ref",
                            /*want_w=*/160, /*want_h=*/96);
}

// min_pixels UPSCALE branch: 21×53 is 1113 px, below the 8192 px floor, so
// smart_resize scales UP to 64×160 (2×5 = 10 tokens). Unreachable from the
// fixture above, and it is the branch with no max(align, ·) clamp.
TEST(ImageLoader, MatchesLlamaCppQwen3VlReferenceUpscaled) {
    check_qwen3vl_reference("preproc_input_tiny.png",
                            "preproc_input_tiny.qwen3vl.ref",
                            /*want_w=*/64, /*want_h=*/160);
}

// The budget in `qwen3vl_preprocess` is expressed in TOKENS; smart_resize works
// in PIXELS. The captured reference reports min_px 8192 / max_px 4194304 for
// this mmproj, which is exactly budget·align² — the identity image_loader
// relies on to convert one into the other. Pinned here because a silent change
// would move every canvas above without failing any shape assertion.
TEST(ImageLoader, Qwen3VlTokenBudgetMatchesReferencePixelBudget) {
    const auto pp = qinf::vision::qwen3vl_preprocess();
    EXPECT_EQ(pp.min_tokens * pp.align * pp.align, 8192);
    EXPECT_EQ(pp.max_tokens * pp.align * pp.align, 4194304);
}
