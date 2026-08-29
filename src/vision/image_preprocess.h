#pragma once
// image_preprocess.h — the projector's preprocessing RECIPE.
//
// Split from image/image_loader.h (2026-08-25, P0 of
// docs/plan-qwen35-vision-impl.md). The division of labour:
//
//   - The recipe (this file) is PROJECTOR KNOWLEDGE — what geometry and
//     normalization the tower was trained against. It belongs beside the
//     encoder that imposes it, and it is pure data: no image IO, no ggml, no
//     backend. A recipe is chosen by projector type, never by call site.
//   - The pipeline (image/image_loader.h) is IO — decode a JPEG/PNG, resample,
//     normalize, emit a Bitmap. That stays outside src/vision/, exactly as
//     image_loader.h has always said it should.
//
// Before the split the recipe lived with the IO, which put `qinf::cli` (as it then was) in the
// type name of something the server, the encoder subsystem and the tests all
// needed. Nothing about a preprocessing recipe is CLI-shaped.
//
// Deliberately header-light so a consumer can compile image_preprocess.cpp
// directly without linking the vision library (image-loader-tests does).

namespace qinf::vision {

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
// are not read from the gguf because GGUFKVBag has no float-array accessor.
// (P1 of docs/plan-qwen35-vision-impl.md adds that accessor — at which point
// these constants should be read, not hardcoded.) `align` is the effective
// patch (patch_size·n_merge = 16·3 = 48).
ImagePreprocess gemma4uv_preprocess(int align = 48,
                                    int min_tokens = 40, int max_tokens = 280);

// Qwen 3.5-family (`qwen3vl_merger`) preprocessing. Same DynSmartResize
// machinery as Gemma 4 — this is a different set of PARAMETERS, not a
// different algorithm:
//   align       = patch_size · n_merge = 16·2 = 32 (the encoder refuses a grid
//                 that is not a multiple of the merged patch)
//   token budget= 8 … 4096, upstream's set_limit_image_tokens for this
//                 projector; our mmproj carries no min/max_pixels override, so
//                 those defaults apply. Note the documented ACCURACY FLOOR of
//                 ~1024 image tokens for grounding tasks
//                 (docs/plan-qwen35-vision-impl.md §5) — the budget permits
//                 fewer, the model just gets worse.
//   mean/std    = 0.5 / 0.5, matching clip.vision.image_{mean,std} in the
//                 mmproj (verified 2026-08-25), i.e. SigLIP-style [-1,1].
//
// P5 gates this byte-faithfully against a captured reference; P3 needs it to
// feed the encoder at all.
ImagePreprocess qwen3vl_preprocess(int align = 32,
                                   int min_tokens = 8, int max_tokens = 4096);

}  // namespace qinf::vision
