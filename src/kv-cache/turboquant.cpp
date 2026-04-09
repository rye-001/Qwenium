#include "turboquant.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace turboquant {

// ============================================================================
// Walsh-Hadamard Transform (in-place, normalized)
// ============================================================================

void wht_inplace(float* data, int n) {
    assert(n > 0 && (n & (n - 1)) == 0 && "n must be a power of 2");
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
    const float norm = 1.0f / std::sqrt(static_cast<float>(n));
    for (int i = 0; i < n; ++i)
        data[i] *= norm;
}

// ============================================================================
// Lloyd-Max centroids for N(0,1)
//
// Precomputed via 1-D k-means on the standard normal distribution.
// Boundaries sit midway between adjacent centroids (sufficient for symmetric
// Gaussian). Indexed by bits: 2-bit → 4 levels, 3-bit → 8, 4-bit → 16.
// ============================================================================

// 2-bit: 4 centroids for N(0,1)
static constexpr float centroids_2bit[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};
static constexpr float boundaries_2bit[3] = {
    -0.9816f, 0.0f, 0.9816f
};

// 3-bit: 8 centroids for N(0,1)
static constexpr float centroids_3bit[8] = {
    -2.1519f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1519f
};
static constexpr float boundaries_3bit[7] = {
    -1.7480f, -1.0500f, -0.5006f, 0.0f, 0.5006f, 1.0500f, 1.7480f
};

// 4-bit: 16 centroids for N(0,1)
static constexpr float centroids_4bit[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};
static constexpr float boundaries_4bit[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7996f, -0.5224f, -0.2582f,  0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f
};

struct Codebook {
    const float* centroids;
    const float* boundaries;
    int n_levels;
    int n_boundaries;
};

static Codebook get_codebook(int bits) {
    switch (bits) {
        case 2: return {centroids_2bit, boundaries_2bit, 4, 3};
        case 3: return {centroids_3bit, boundaries_3bit, 8, 7};
        case 4: return {centroids_4bit, boundaries_4bit, 16, 15};
        default: assert(false && "bits must be 2, 3, or 4"); return {};
    }
}

// Find nearest centroid index via boundary search (sorted, fast).
static inline int quantize_scalar(float x, const Codebook& cb) {
    int idx = 0;
    for (int i = 0; i < cb.n_boundaries; ++i) {
        if (x > cb.boundaries[i]) idx = i + 1;
        else break;
    }
    return idx;
}

// ============================================================================
// Bit packing utilities
// ============================================================================

static inline void pack_4bit(uint8_t* dst, int idx, uint8_t val) {
    int byte = idx / 2;
    if (idx % 2 == 0)
        dst[byte] = (dst[byte] & 0xF0) | (val & 0x0F);
    else
        dst[byte] = (dst[byte] & 0x0F) | ((val & 0x0F) << 4);
}

static inline uint8_t unpack_4bit(const uint8_t* src, int idx) {
    int byte = idx / 2;
    return (idx % 2 == 0) ? (src[byte] & 0x0F) : ((src[byte] >> 4) & 0x0F);
}

static inline void pack_3bit(uint8_t* dst, int idx, uint8_t val) {
    int bit_offset = idx * 3;
    int byte = bit_offset / 8;
    int shift = bit_offset % 8;
    // Clear and set the bits (may span two bytes)
    uint16_t word;
    std::memcpy(&word, dst + byte, 2);
    word &= ~(uint16_t(0x7) << shift);
    word |= (uint16_t(val & 0x7) << shift);
    std::memcpy(dst + byte, &word, 2);
}

static inline uint8_t unpack_3bit(const uint8_t* src, int idx) {
    int bit_offset = idx * 3;
    int byte = bit_offset / 8;
    int shift = bit_offset % 8;
    uint16_t word;
    std::memcpy(&word, src + byte, 2);
    return (word >> shift) & 0x7;
}

static inline void pack_2bit(uint8_t* dst, int idx, uint8_t val) {
    int byte = idx / 4;
    int shift = (idx % 4) * 2;
    dst[byte] = (dst[byte] & ~(0x3 << shift)) | ((val & 0x3) << shift);
}

static inline uint8_t unpack_2bit(const uint8_t* src, int idx) {
    int byte = idx / 4;
    int shift = (idx % 4) * 2;
    return (src[byte] >> shift) & 0x3;
}

// ============================================================================
// Block geometry
// ============================================================================

size_t block_bytes(int bits, int block_size) {
    // 2 bytes for F16 scale + packed data
    size_t data_bytes = (static_cast<size_t>(block_size) * bits + 7) / 8;
    return 2 + data_bytes;
}

size_t compressed_size(int n_elements, int bits, int block_size) {
    assert(n_elements % block_size == 0);
    size_t n_blocks = n_elements / block_size;
    return n_blocks * block_bytes(bits, block_size);
}

// ============================================================================
// F16 ↔ F32 conversion (minimal, no ggml dependency)
// ============================================================================

// IEEE 754 half-precision. We only need round-trip fidelity for scale factors,
// so a simple software path is fine.

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (x >> 13) & 0x3FF;

    if (exp <= 0) return static_cast<uint16_t>(sign);         // flush to zero
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00); // inf
    return static_cast<uint16_t>(sign | (exp << 10) | frac);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    if (exp == 0) {
        // Zero / subnormal → flush to zero
        float r;
        uint32_t bits = sign;
        std::memcpy(&r, &bits, 4);
        return r;
    }
    if (exp == 31) exp = 255; else exp = exp - 15 + 127;

    uint32_t bits = sign | (exp << 23) | (frac << 13);
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}

// ============================================================================
// Quantize / dequantize a single block
// ============================================================================

static void quantize_block(const float* src, uint8_t* dst,
                           int block_size, int bits) {
    Codebook cb = get_codebook(bits);

    // 1. Find per-block scale: max(|x|) / max_centroid
    float amax = 0.0f;
    for (int i = 0; i < block_size; ++i)
        amax = std::max(amax, std::fabsf(src[i]));

    float max_centroid = cb.centroids[cb.n_levels - 1]; // positive extreme
    float scale = (amax > 1e-10f) ? (amax / max_centroid) : 0.0f;

    // Write scale as F16
    uint16_t scale_f16 = f32_to_f16(scale);
    std::memcpy(dst, &scale_f16, 2);
    uint8_t* packed = dst + 2;

    // 2. Clear the packed data region
    size_t data_bytes = (static_cast<size_t>(block_size) * bits + 7) / 8;
    std::memset(packed, 0, data_bytes);

    // 3. Quantize each element
    float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;
    for (int i = 0; i < block_size; ++i) {
        float normalized = src[i] * inv_scale;
        int idx = quantize_scalar(normalized, cb);
        switch (bits) {
            case 4: pack_4bit(packed, i, static_cast<uint8_t>(idx)); break;
            case 3: pack_3bit(packed, i, static_cast<uint8_t>(idx)); break;
            case 2: pack_2bit(packed, i, static_cast<uint8_t>(idx)); break;
        }
    }
}

static void dequantize_block(const uint8_t* src, float* dst,
                             int block_size, int bits) {
    Codebook cb = get_codebook(bits);

    // Read scale
    uint16_t scale_f16;
    std::memcpy(&scale_f16, src, 2);
    float scale = f16_to_f32(scale_f16);

    const uint8_t* packed = src + 2;
    for (int i = 0; i < block_size; ++i) {
        int idx;
        switch (bits) {
            case 4: idx = unpack_4bit(packed, i); break;
            case 3: idx = unpack_3bit(packed, i); break;
            case 2: idx = unpack_2bit(packed, i); break;
            default: idx = 0;
        }
        dst[i] = cb.centroids[idx] * scale;
    }
}

// ============================================================================
// Public API: compress / decompress
// ============================================================================

void compress(float* src, void* dst, int n, int bits, int block_size) {
    assert(n > 0 && (n & (n - 1)) == 0 && "n must be a power of 2");
    assert(n % block_size == 0);

    // 1. Apply WHT (in-place, modifies src)
    wht_inplace(src, n);

    // 2. Quantize block by block
    size_t bbytes = block_bytes(bits, block_size);
    int n_blocks = n / block_size;
    auto* out = static_cast<uint8_t*>(dst);

    for (int b = 0; b < n_blocks; ++b)
        quantize_block(src + b * block_size, out + b * bbytes, block_size, bits);
}

void decompress(const void* src, float* dst, int n, int bits, int block_size) {
    assert(n > 0 && (n & (n - 1)) == 0 && "n must be a power of 2");
    assert(n % block_size == 0);

    // 1. Dequantize block by block
    size_t bbytes = block_bytes(bits, block_size);
    int n_blocks = n / block_size;
    const auto* in = static_cast<const uint8_t*>(src);

    for (int b = 0; b < n_blocks; ++b)
        dequantize_block(in + b * bbytes, dst + b * block_size, block_size, bits);

    // 2. Inverse WHT (same as forward WHT — self-inverse)
    wht_inplace(dst, n);
}

} // namespace turboquant
