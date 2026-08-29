#pragma once
// image_loader.h — host-side Bitmap producer (Gemma 3 + Gemma 4 vision).
//
// Decodes a JPEG/PNG/etc. file (via vendored stb_image), resizes, applies the
// projector's normalization, and emits the channel-planar float32 Bitmap that an
// IVisionEncoder consumes. Image IO lives OUTSIDE src/vision/ by design — that
// directory is the encoder subsystem, not the image pipeline.
//
// The RECIPE this file applies (qinf::vision::ImagePreprocess and the two
// projector factories) moved to vision/image_preprocess.h — a recipe is
// projector knowledge, not CLI code. This file kept the IO, which is what the
// paragraph above was always about.
//
// Two projector preprocessings, ONE function parameterized by ImagePreprocess
// (plan §5: "parameterize image_loader rather than fork it"):
//   - Gemma 3 (gemma3_preprocess): fixed 896 square, aspect-preserving PAD_CEIL
//     letterbox, SigLIP normalize (v/255 − 0.5)/0.5 → [-1, 1]; black pad → -1.0.
//   - Gemma 4 (gemma4uv_preprocess): Qwen-VL "smart_resize" to variable dims
//     (multiples of the effective patch 48, within a [min,max]-token budget),
//     same PAD_CEIL letterbox, normalize v/255 → [0, 1]; black pad → 0.0.
//
// Both use the byte-faithful llama.cpp resampler: ALIGN-CORNERS bilinear with a
// uint8 intermediate. The Gemma 3 path is gated byte-faithful by
// image-loader-tests::MatchesLlamaCppGemma3Reference (kEps 2e-3).

#include <cstdint>
#include <string>

#include "../vision/bitmap.h"
#include "../vision/image_preprocess.h"

namespace qinf::image {

// Decode + resize + normalize `path` into a channel-planar Bitmap per `pp`.
// content_id is a content hash of the normalized pixels (encode-cache reuse).
// Fail-loud (CLAUDE.md): throws naming the path and the stb reason on failure.
qinf::vision::Bitmap load_image_to_bitmap(const std::string& path,
                                          const qinf::vision::ImagePreprocess& pp);

// Back-compat overload: Gemma 3 fixed-square preprocessing (existing callers).
qinf::vision::Bitmap load_image_to_bitmap(const std::string& path, int target = 896);

// Same as load_image_to_bitmap(path, pp) but the encoded image FILE bytes are
// already in memory (e.g. a base64-decoded `image_url` data URI from the HTTP
// chat endpoint) rather than on disk. stb decodes the format from the bytes.
// Fail-loud (CLAUDE.md): throws naming the source and the stb reason on failure.
qinf::vision::Bitmap load_image_to_bitmap_from_memory(const uint8_t* data,
                                                      size_t len,
                                                      const qinf::vision::ImagePreprocess& pp);

}  // namespace qinf::image
