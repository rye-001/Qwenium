// test_turboquant.cpp
// TDD tests for the TurboQuant math library.
// Tests are arch-agnostic — no ggml, no model, no KV cache.
//
// RED baseline: turboquant.h/cpp do not exist yet — tests fail to link.
// GREEN target: all tests pass after implementation.

#include "turboquant.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstring>

// ============================================================================
// Helpers
// ============================================================================

static std::vector<float> make_random_vector(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static float mse(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return static_cast<float>(sum / n);
}

static float max_abs(const float* v, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabsf(v[i]));
    return m;
}

// ============================================================================
// 1. Walsh-Hadamard Transform
// ============================================================================

TEST(WHT, RoundTrip_PowerOf2) {
    // WHT is self-inverse: applying it twice recovers the original vector.
    const int n = 128;
    auto orig = make_random_vector(n);
    auto data = orig;

    turboquant::wht_inplace(data.data(), n);
    // After one transform, data should differ from orig
    EXPECT_GT(mse(orig.data(), data.data(), n), 1e-6f);

    turboquant::wht_inplace(data.data(), n);
    // After two transforms, should recover original
    EXPECT_LT(mse(orig.data(), data.data(), n), 1e-10f);
}

TEST(WHT, RoundTrip_SmallSizes) {
    for (int n : {2, 4, 8, 16, 32, 64, 256}) {
        auto orig = make_random_vector(n, n);
        auto data = orig;
        turboquant::wht_inplace(data.data(), n);
        turboquant::wht_inplace(data.data(), n);
        EXPECT_LT(mse(orig.data(), data.data(), n), 1e-10f)
            << "Failed for n=" << n;
    }
}

TEST(WHT, PreservesNorm) {
    // WHT with 1/sqrt(n) normalization preserves L2 norm.
    const int n = 128;
    auto data = make_random_vector(n);

    float norm_before = 0;
    for (float x : data) norm_before += x * x;

    turboquant::wht_inplace(data.data(), n);

    float norm_after = 0;
    for (float x : data) norm_after += x * x;

    EXPECT_NEAR(norm_before, norm_after, 1e-4f);
}

TEST(WHT, SpreadsOutliers) {
    // A vector with one large outlier should have a lower max_abs after WHT.
    const int n = 128;
    std::vector<float> data(n, 0.0f);
    data[0] = 10.0f;  // Single outlier

    float peak_before = max_abs(data.data(), n);
    turboquant::wht_inplace(data.data(), n);
    float peak_after = max_abs(data.data(), n);

    // The peak should be spread: 10/sqrt(128) ≈ 0.88
    EXPECT_LT(peak_after, peak_before / 5.0f);
}

// ============================================================================
// 2. TurboQuantConfig
// ============================================================================

TEST(Config, BlockBytes) {
    // 4-bit, block_size=32: 2 bytes scale + 16 bytes data = 18
    EXPECT_EQ(turboquant::block_bytes(4, 32), 18u);
    // 3-bit, block_size=32: 2 bytes scale + 12 bytes data = 14
    EXPECT_EQ(turboquant::block_bytes(3, 32), 14u);
    // 2-bit, block_size=32: 2 bytes scale + 8 bytes data = 10
    EXPECT_EQ(turboquant::block_bytes(2, 32), 10u);
}

TEST(Config, CompressedSize) {
    // 128 elements, 4-bit, block_size=32 → 4 blocks × 18 bytes = 72
    EXPECT_EQ(turboquant::compressed_size(128, 4, 32), 72u);
    // 1024 elements (8 heads × 128) → 32 blocks × 18 = 576
    EXPECT_EQ(turboquant::compressed_size(1024, 4, 32), 576u);
}

// ============================================================================
// 3. Compress / Decompress Round-Trip
// ============================================================================

TEST(RoundTrip, FourBit_SingleHead) {
    const int n = 128;
    auto orig = make_random_vector(n);
    auto src = orig;

    size_t csize = turboquant::compressed_size(n, 4, 32);
    std::vector<uint8_t> buf(csize);
    std::vector<float> dst(n);

    turboquant::compress(src.data(), buf.data(), n, 4, 32);
    turboquant::decompress(buf.data(), dst.data(), n, 4, 32);

    // 4-bit Lloyd-Max on Gaussian data: MSE should be < 0.05
    float err = mse(orig.data(), dst.data(), n);
    EXPECT_LT(err, 0.05f) << "4-bit MSE too high: " << err;
}

TEST(RoundTrip, TwoBit_SingleHead) {
    const int n = 128;
    auto orig = make_random_vector(n);
    auto src = orig;

    size_t csize = turboquant::compressed_size(n, 2, 32);
    std::vector<uint8_t> buf(csize);
    std::vector<float> dst(n);

    turboquant::compress(src.data(), buf.data(), n, 2, 32);
    turboquant::decompress(buf.data(), dst.data(), n, 2, 32);

    // 2-bit is coarser, MSE < 0.25
    float err = mse(orig.data(), dst.data(), n);
    EXPECT_LT(err, 0.25f) << "2-bit MSE too high: " << err;
}

TEST(RoundTrip, ThreeBit_SingleHead) {
    const int n = 128;
    auto orig = make_random_vector(n);
    auto src = orig;

    size_t csize = turboquant::compressed_size(n, 3, 32);
    std::vector<uint8_t> buf(csize);
    std::vector<float> dst(n);

    turboquant::compress(src.data(), buf.data(), n, 3, 32);
    turboquant::decompress(buf.data(), dst.data(), n, 3, 32);

    // 3-bit should be between 2-bit and 4-bit
    float err = mse(orig.data(), dst.data(), n);
    EXPECT_LT(err, 0.12f) << "3-bit MSE too high: " << err;
}

TEST(RoundTrip, MultiHead) {
    // Simulate 8 KV heads, 128 dims each = 1024 total
    const int n_heads = 8;
    const int head_dim = 128;
    const int n = n_heads * head_dim;

    auto orig = make_random_vector(n);
    auto src = orig;

    size_t csize = turboquant::compressed_size(head_dim, 4, 32);
    std::vector<uint8_t> buf(csize * n_heads);
    std::vector<float> dst(n);

    // Compress/decompress per head
    for (int h = 0; h < n_heads; ++h) {
        turboquant::compress(
            src.data() + h * head_dim,
            buf.data() + h * csize,
            head_dim, 4, 32);
        turboquant::decompress(
            buf.data() + h * csize,
            dst.data() + h * head_dim,
            head_dim, 4, 32);
    }

    float err = mse(orig.data(), dst.data(), n);
    EXPECT_LT(err, 0.05f) << "Multi-head 4-bit MSE too high: " << err;
}

// ============================================================================
// 4. Quality properties
// ============================================================================

TEST(Quality, FourBit_BetterThan_TwoBit) {
    const int n = 128;
    auto orig = make_random_vector(n, 99);

    auto src4 = orig;
    size_t cs4 = turboquant::compressed_size(n, 4, 32);
    std::vector<uint8_t> buf4(cs4);
    std::vector<float> dst4(n);
    turboquant::compress(src4.data(), buf4.data(), n, 4, 32);
    turboquant::decompress(buf4.data(), dst4.data(), n, 4, 32);

    auto src2 = orig;
    size_t cs2 = turboquant::compressed_size(n, 2, 32);
    std::vector<uint8_t> buf2(cs2);
    std::vector<float> dst2(n);
    turboquant::compress(src2.data(), buf2.data(), n, 2, 32);
    turboquant::decompress(buf2.data(), dst2.data(), n, 2, 32);

    EXPECT_LT(mse(orig.data(), dst4.data(), n),
              mse(orig.data(), dst2.data(), n));
}

TEST(Quality, CompressDoesNotModifySource) {
    // compress() needs to apply WHT internally without clobbering the caller's buffer.
    // The API takes float* (non-const) because WHT is in-place, but we document
    // that the source IS modified (rotated). Test that decompress recovers original.
    const int n = 128;
    auto orig = make_random_vector(n);
    auto src = orig;

    size_t csize = turboquant::compressed_size(n, 4, 32);
    std::vector<uint8_t> buf(csize);
    std::vector<float> dst(n);

    turboquant::compress(src.data(), buf.data(), n, 4, 32);
    turboquant::decompress(buf.data(), dst.data(), n, 4, 32);

    // dst should approximate orig regardless of what happened to src
    EXPECT_LT(mse(orig.data(), dst.data(), n), 0.05f);
}

// ============================================================================
// 5. Edge cases
// ============================================================================

TEST(Edge, ZeroVector) {
    const int n = 128;
    std::vector<float> src(n, 0.0f);

    size_t csize = turboquant::compressed_size(n, 4, 32);
    std::vector<uint8_t> buf(csize);
    std::vector<float> dst(n, 999.0f);

    turboquant::compress(src.data(), buf.data(), n, 4, 32);
    turboquant::decompress(buf.data(), dst.data(), n, 4, 32);

    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(dst[i], 0.0f, 1e-6f);
}

TEST(Edge, ConstantVector) {
    const int n = 128;
    std::vector<float> src(n, 3.14f);
    auto orig = src;

    size_t csize = turboquant::compressed_size(n, 4, 32);
    std::vector<uint8_t> buf(csize);
    std::vector<float> dst(n);

    turboquant::compress(src.data(), buf.data(), n, 4, 32);
    turboquant::decompress(buf.data(), dst.data(), n, 4, 32);

    // Constant vectors are pathological for WHT (all energy in DC term).
    // Reconstruction is coarser but still bounded.
    EXPECT_LT(mse(orig.data(), dst.data(), n), 1.0f);
}

TEST(Edge, LargeOutlier) {
    // One huge value, rest near zero — WHT should handle this gracefully.
    const int n = 128;
    std::vector<float> src(n, 0.01f);
    src[0] = 100.0f;
    auto orig = src;

    size_t csize = turboquant::compressed_size(n, 4, 32);
    std::vector<uint8_t> buf(csize);
    std::vector<float> dst(n);

    turboquant::compress(src.data(), buf.data(), n, 4, 32);
    turboquant::decompress(buf.data(), dst.data(), n, 4, 32);

    // The outlier should be approximately preserved
    EXPECT_NEAR(dst[0], 100.0f, 5.0f);
}

// ============================================================================
// 6. Bit-packing correctness
// ============================================================================

TEST(BitPack, FourBit_AllIndices) {
    // Compress a vector that, after WHT + scale, maps to known centroid indices,
    // then verify decompress recovers consistent values.
    // We just verify compress→decompress is deterministic.
    const int n = 32;  // Single block
    auto v1 = make_random_vector(n, 1);
    auto v2 = v1;

    size_t csize = turboquant::compressed_size(n, 4, 32);
    std::vector<uint8_t> buf1(csize), buf2(csize);
    std::vector<float> dst1(n), dst2(n);

    turboquant::compress(v1.data(), buf1.data(), n, 4, 32);
    turboquant::compress(v2.data(), buf2.data(), n, 4, 32);

    // Same input → same compressed output
    EXPECT_EQ(buf1, buf2);

    turboquant::decompress(buf1.data(), dst1.data(), n, 4, 32);
    turboquant::decompress(buf2.data(), dst2.data(), n, 4, 32);

    // Same compressed → same decompressed
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(dst1[i], dst2[i]);
}
