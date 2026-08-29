#pragma once
// image_loader.h — host-side Bitmap producer for every vision family.
//
// Decodes a JPEG/PNG/etc. file (via vendored stb_image), resizes, applies the
// projector's normalization, and emits the channel-planar float32 Bitmap that an
// IVisionEncoder consumes. This is IO, and it lives in src/image/ rather than
// src/vision/ (the encoder subsystem, not the image pipeline) and rather than
// src/cli/ — both front ends need it, and the server compiles it directly.
//
// The RECIPE this file applies (qinf::vision::ImagePreprocess and its projector
// factories) lives in vision/image_preprocess.h — a recipe is projector
// knowledge. This file kept the IO. One function parameterized by
// ImagePreprocess, never forked per family (plan §5: "parameterize
// image_loader rather than fork it"); the recipe is chosen once, by projector
// type, in vision/vision_profile.
//
// Three recipes today, along two sizing branches:
//   - FixedSquarePadCeil — gemma3_preprocess: fixed 896 square, aspect-
//     preserving PAD_CEIL letterbox, SigLIP normalize (v/255 − 0.5)/0.5 →
//     [-1, 1]; black pad → -1.0.
//   - DynSmartResize — gemma4uv_preprocess (align 48) and qwen3vl_preprocess
//     (align 32): Qwen-VL "smart_resize" to variable dims that are multiples of
//     the effective patch, within a [min,max]-TOKEN budget converted to pixels
//     by align²; same PAD_CEIL letterbox; normalize v/255 → [0, 1], black pad
//     → 0.0.
//
// All use the byte-faithful llama.cpp resampler: ALIGN-CORNERS bilinear with a
// uint8 intermediate — the encoder must see exactly what it saw in training.
// Both sizing branches are gated bit-exact against captured mtmd references:
// image-loader-tests::MatchesLlamaCppGemma3Reference (kEps 2e-3) and
// ::MatchesLlamaCppQwen3VlReference{,Upscaled}.
// Unit test: tests/unit/test_image_loader.cpp

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
