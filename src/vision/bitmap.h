#pragma once
// bitmap.h — POD-ish image input to VisionEncoder.
//
// The producer of a Bitmap (Phase 6 host-side image decoder) is responsible
// for decoding JPEG/PNG, resizing/letterboxing to 896×896, and normalizing
// pixel values to SigLIP's expected range. VisionEncoder consumes the
// already-prepared float32 pixel grid. Preprocessing lives outside this
// directory by design — `src/vision/` is the encoder subsystem, not the
// image-IO subsystem.
//
// Layout: pixels are stored contiguous, channel-last per pixel? No —
// SigLIP / ggml conv2d expects channel-first interleaving in memory.
// CONTRACT: `pixels` is laid out as [C, H, W] contiguous (channel-planar),
// row-major within each channel. Producer is responsible. VisionEncoder
// validates the size; layout violations are silent corruption, so document
// loudly here.
//
// Single-tile only in v1 (docs/plan-gemma-vision-impl.md → top-of-doc
// constraint). H == W == 896, C == 3. Pan & Scan is scoped out.

#include <cstdint>
#include <vector>

namespace qinf::vision {

struct Bitmap {
    // [C, H, W] channel-planar float32. SigLIP-normalized by the producer.
    std::vector<float> pixels;

    int height   = 0;     // expected: 896 (Gemma 3 single-tile)
    int width    = 0;     // expected: 896
    int channels = 0;     // expected: 3 (RGB)

    // Forward-looking: content-derived identifier so the orchestrator's
    // image-embedding cache (Phase 7) can dedup re-encodings of the same
    // image across turns. Producer-set. 0 means "not set" — the cache
    // treats 0 as a non-cacheable bitmap rather than colliding everything
    // to one slot.
    //
    // Set by the chunk-list builder in Phase 6, *not* by VisionEncoder
    // (the encoder is content-blind; it does not hash pixels).
    uint64_t content_id = 0;
};

}  // namespace qinf::vision
