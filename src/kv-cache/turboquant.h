#pragma once
// turboquant.h — Architecture-agnostic TurboQuant compression.
//
// Implements the core TurboQuant pipeline:
//   1. Fast Walsh-Hadamard Transform (spreads outliers)
//   2. Per-block Lloyd-Max scalar quantization (precomputed centroids for N(0,1))
//
// No ggml dependency. No model dependency. Pure math on float arrays.

#include <cstddef>
#include <cstdint>

namespace turboquant {

// ── Walsh-Hadamard Transform ─────────────────────────────────────────────────
// In-place, normalized by 1/sqrt(n). Self-inverse: apply twice = identity.
// n must be a power of 2.
void wht_inplace(float* data, int n);

// ── Block geometry ───────────────────────────────────────────────────────────
// Block layout: [2 bytes F16 scale | ceil(block_size * bits / 8) bytes packed]
//   4-bit, bs=32 → 18 bytes    (7.1× vs F32)
//   3-bit, bs=32 → 14 bytes    (9.1× vs F32)
//   2-bit, bs=32 → 10 bytes   (12.8× vs F32)

size_t block_bytes(int bits, int block_size);
size_t compressed_size(int n_elements, int bits, int block_size);

// ── Compress / Decompress ────────────────────────────────────────────────────
// These apply WHT internally — caller passes raw (unrotated) floats.
//
// compress(): src is modified in-place (WHT rotation) before quantization.
//             dst must have at least compressed_size(n, bits, block_size) bytes.
//             n must be a power of 2 (and a multiple of block_size).
//
// decompress(): src is the compressed buffer.
//               dst receives the reconstructed float vector.
//               n must match the original compress() call.

void compress(float* src, void* dst, int n, int bits, int block_size);
void decompress(const void* src, float* dst, int n, int bits, int block_size);

} // namespace turboquant
