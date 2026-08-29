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

#include "../../src/cli/image_loader.h"

using qinf::cli::load_image_to_bitmap;

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
