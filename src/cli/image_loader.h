#pragma once
// image_loader.h — host-side Bitmap producer (Gemma 3 + Gemma 4 vision).
//
// Decodes a JPEG/PNG/etc. file (via vendored stb_image), resizes, applies the
// projector's normalization, and emits the channel-planar float32 Bitmap that an
// IVisionEncoder consumes. Image IO lives OUTSIDE src/vision/ by design — that
// directory is the encoder subsystem, not the image pipeline.
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

namespace qinf::cli {

// Preprocessing recipe. Normalization is unified as (v/255 − mean[c])/stddev[c]
// per channel (the SigLIP form); the projector factories below fill the rest.
struct ImagePreprocess {
    enum class Sizing {
        FixedSquarePadCeil,  // resize+pad to a fixed square (Gemma 3)
        DynSmartResize,      // Qwen-VL smart_resize to variable dims (Gemma 4)
    };
    Sizing sizing = Sizing::FixedSquarePadCeil;

    // FixedSquarePadCeil:
    int fixed_target = 896;

    // DynSmartResize (effective patch align; token budget is inclusive):
    int align      = 48;
    int min_tokens = 40;
    int max_tokens = 280;

    // (v/255 − mean)/stddev, per channel.
    float mean[3]   = {0.5f, 0.5f, 0.5f};
    float stddev[3] = {0.5f, 0.5f, 0.5f};
};

// Gemma 3 SigLIP preprocessing (fixed 896, (v/255−0.5)/0.5).
ImagePreprocess gemma3_preprocess(int target = 896);

// Gemma 4 unified-vision preprocessing. mean=[0,0,0] std=[1,1,1] are the
// verified mmproj-gemma-4-12B-it kv values (clip.vision.image_{mean,std}); they
// are not read from the gguf because GGUFKVBag has no float-array accessor and
// adding one is cross-cutting infra out of scope for this work. `align` is the
// effective patch (patch_size·n_merge = 16·3 = 48).
ImagePreprocess gemma4uv_preprocess(int align = 48,
                                    int min_tokens = 40, int max_tokens = 280);

// Decode + resize + normalize `path` into a channel-planar Bitmap per `pp`.
// content_id is a content hash of the normalized pixels (encode-cache reuse).
// Fail-loud (CLAUDE.md): throws naming the path and the stb reason on failure.
qinf::vision::Bitmap load_image_to_bitmap(const std::string& path,
                                          const ImagePreprocess& pp);

// Back-compat overload: Gemma 3 fixed-square preprocessing (existing callers).
qinf::vision::Bitmap load_image_to_bitmap(const std::string& path, int target = 896);

// Same as load_image_to_bitmap(path, pp) but the encoded image FILE bytes are
// already in memory (e.g. a base64-decoded `image_url` data URI from the HTTP
// chat endpoint) rather than on disk. stb decodes the format from the bytes.
// Fail-loud (CLAUDE.md): throws naming the source and the stb reason on failure.
qinf::vision::Bitmap load_image_to_bitmap_from_memory(const uint8_t* data,
                                                      size_t len,
                                                      const ImagePreprocess& pp);

}  // namespace qinf::cli
