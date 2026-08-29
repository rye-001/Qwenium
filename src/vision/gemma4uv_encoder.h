#pragma once
// gemma4uv_encoder.h — Gemma 4 unified-vision (`gemma4uv`) image encoder.
// The second IVisionEncoder (Seam A — see i_vision_encoder.h); SiglipEncoder
// is the first. Port of llama.cpp tools/mtmd/models/gemma4uv.cpp.
//
// Structurally distinct from SigLIP and much smaller: there is NO transformer
// stack. The "encoder" is a blockless projector —
//   im2col(P=48 patches) → LayerNorm → patch-embed matmul → LayerNorm
//   → +2D learned position embeddings → LayerNorm → RMSNorm → projection.
// Every LayerNorm here is pytorch-default eps 1e-5 (hardcoded in the reference,
// NOT read from the mmproj); the single final RMSNorm uses the mmproj's
// clip.vision.attention.layer_norm_epsilon (1e-6). Image attention is causal on
// the text side — there is no bidirectional image mask (the single biggest
// simplification versus the Gemma 3 path).
//
// Token count is PER-IMAGE: the dynamic-size preprocessor resizes each image to
// a multiple of the effective patch size 48 within a [40, 280]-token budget, so
// the encode produces (W/48)·(H/48) tokens. mm_tokens_for() reports that count
// from the preprocessed bitmap's dims (pure, no encode) so the caller can size
// the placeholder span before encoding.
//
// Ownership / threading: identical contract to SiglipEncoder — owns its own
// ggml graph context + scheduler (with an owned CPU fallback when the shared
// backend is a device backend), borrows the engine backend and the VisionModel
// weight tensors, single-threaded. (The scheduler boilerplate is duplicated
// rather than hoisted into a shared base: two encoders is below the bar for a
// third abstraction — see plan §3 "extract exactly two seams.")

#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "i_vision_encoder.h"

namespace qinf::vision {

struct Bitmap;
class  VisionModel;

class Gemma4UvEncoder : public IVisionEncoder {
public:
    // `model` must already have been loaded by VisionLoader with projector type
    // gemma4uv. `backend` is the engine's Metal (or CPU) backend — shared, not
    // owned. `text_embed_dim` is the host text model's embedding_length, checked
    // fail-loud against the mmproj projection weight (loose coupling).
    Gemma4UvEncoder(const VisionModel& model,
                    ggml_backend_t     backend,
                    uint32_t           text_embed_dim);
    ~Gemma4UvEncoder() override;

    Gemma4UvEncoder(const Gemma4UvEncoder&) = delete;
    Gemma4UvEncoder& operator=(const Gemma4UvEncoder&) = delete;

    // Encode one preprocessed bitmap. Returns [projection_dim, n_tokens]
    // row-major (projection_dim fastest), n_tokens = mm_tokens_for(bitmap).
    // Fail-loud on shape/compute failure.
    std::vector<float> encode(const Bitmap& bitmap) override;

    // (W/P)·(H/P) with P = effective patch (48). Pure — no encode.
    uint32_t mm_tokens_for(const Bitmap& bitmap) const override;

    // (W/P) × (H/P) for the effective patch P — the same factors
    // mm_tokens_for multiplies together.
    void mm_grid_for(const Bitmap& bitmap,
                     uint32_t& nx, uint32_t& ny) const override;

    uint32_t projection_dim() const override;

private:
    const VisionModel&    model_;
    ggml_backend_t        backend_       = nullptr;   // shared, not owned
    ggml_backend_t        cpu_fallback_  = nullptr;   // owned; only when backend_ isn't CPU
    ggml_backend_sched_t  scheduler_     = nullptr;   // owned
    ggml_context*         graph_ctx_     = nullptr;   // owned

    uint32_t              text_embed_dim_ = 0;

    // Effective patch size = patch_size * n_merge (16·3 = 48). Cached from cfg.
    uint32_t              eff_patch_      = 0;
};

}  // namespace qinf::vision
