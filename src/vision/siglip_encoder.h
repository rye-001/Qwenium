#pragma once
// siglip_encoder.h — SigLIP ViT + 4×4 pool + projection + soft-emb-norm
// (Gemma 3 vision). One concrete IVisionEncoder (Seam A — see
// i_vision_encoder.h); Gemma4UvEncoder is the second. Renamed from
// `VisionEncoder` when Gemma 4 vision forced the interface extraction: the
// class name now states which encoder it is, rather than implying it is the
// only one.
//
// Bitmap in, host-side embeddings out. Standalone subsystem; no dependency
// on the text-side ForwardPassBase or Model (C3 contract — vision lives in
// `src/vision/`, connected to the text model only at the embedding boundary
// via ImageEmbeddingsInput in Phase 3).
//
// Ownership model (decided P2.0):
//   - SiglipEncoder owns its own ggml_context for the graph and its own
//     ggml_cgraph. Resets per encode() like the text-side prefill graph.
//   - SiglipEncoder owns its own ggml_backend_sched_t (its own scheduler).
//   - SiglipEncoder DOES NOT own the backend — the engine's existing Metal
//     ggml_backend_t is passed in and shared. One Metal device, two
//     schedulers (one for text, one for vision). This mirrors the
//     llama.cpp mtmd boundary: separate graphs, shared device.
//   - SiglipEncoder borrows weight tensors from VisionModel (which owns
//     them). It does NOT take ownership of those tensors.
//
// API decisions (P2.0):
//   - Bitmap in: caller pre-decodes JPEG/PNG and pre-resizes to 896×896
//     channel-planar float32. Image IO is Phase 6 — not this subsystem.
//   - Embeddings out: host-side std::vector<float>, shape
//     [mm_tokens_per_image * projection_dim] (256 * text_embed_dim for
//     Gemma 3). Same pattern as get_output_logits — copy out of the
//     backend, hand to caller, caller owns lifetime.
//   - text_embed_dim is a constructor sanity-check parameter (loose
//     coupling): SiglipEncoder asserts the mmproj's projection-weight
//     column count matches the caller's stated text_embed_dim. Mismatch
//     is a fail-loud CLAUDE.md error, not a silent reshape.
//   - Single image per encode() call in v1. Multi-image batching is a
//     Phase 7 question; the API can be widened then without breaking
//     callers (a single-image overload survives a batched API).
//   - Not thread-safe — single-threaded use, like the text-side
//     ForwardPass. Documented; no internal locking.
//
// Status: P2.3 + P2.4 implemented — encode() builds the full Gemma 3 vision
// pipeline as one graph: conv2d patch embed → +position embd → 27 pre-norm
// SigLIP transformer layers → post_ln → 4×4 average pool (4096→256 tokens) →
// RMS soft-emb-norm → linear projection. Returns the projected image
// embeddings [projection_dim, mm_tokens_per_image] = [2560, 256] for Gemma 3.

#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "i_vision_encoder.h"

namespace qinf::vision {

struct Bitmap;
class  VisionModel;

class SiglipEncoder : public IVisionEncoder {
public:
    // `model` must already have been loaded by VisionLoader.
    // `backend` is the engine's Metal (or CPU) backend — shared, not owned.
    // `text_embed_dim` is the host text model's embedding_length. Used as
    // a fail-loud sanity check against the mmproj's projection weight; a
    // mismatch means the mmproj does not match the text model and the
    // engine refuses to encode rather than producing garbled embeddings.
    SiglipEncoder(const VisionModel& model,
                  ggml_backend_t      backend,
                  uint32_t            text_embed_dim);
    ~SiglipEncoder() override;

    SiglipEncoder(const SiglipEncoder&) = delete;
    SiglipEncoder& operator=(const SiglipEncoder&) = delete;

    // Encode one bitmap into host-side image embeddings.
    //
    // Returned vector has size `mm_tokens_per_image() * projection_dim()`,
    // row-major: row r = embedding of image-soft-token r at the text-embed
    // dim. Phase 3's ImageEmbeddingsInput uploads this back into the
    // gemma3 prefill graph at the image-soft-token positions.
    //
    // Fail-loud contract (CLAUDE.md): throws std::runtime_error if
    //   - bitmap.{height,width,channels} disagree with SigLIPConfig, or
    //   - bitmap.pixels.size() != C*H*W, or
    //   - the backend graph compute fails.
    // No silent resize, no best-effort fallback.
    std::vector<float> encode(const Bitmap& bitmap) override;

    // SigLIP encodes a fixed 896 tile to a constant 256 tokens, independent of
    // the bitmap. (IVisionEncoder Seam A: Gemma 4's count is per-image.)
    uint32_t mm_tokens_for(const Bitmap& bitmap) const override;

    // Fixed grid: (image_size/patch)/pool per side — 16×16 = 256 for Gemma 3.
    void mm_grid_for(const Bitmap& bitmap,
                     uint32_t& nx, uint32_t& ny) const override;

    // Tokens-per-image after the 4×4 average pool (256 for Gemma 3).
    uint32_t mm_tokens_per_image() const;

    // Output embedding dim (= text model's embedding_length, verified
    // against the mmproj projection weight at construction).
    uint32_t projection_dim() const override;

private:
    const VisionModel&    model_;
    ggml_backend_t        backend_       = nullptr;   // shared, not owned
    ggml_backend_t        cpu_fallback_  = nullptr;   // owned; only when backend_ isn't CPU
    ggml_backend_sched_t  scheduler_     = nullptr;   // owned
    ggml_context*         graph_ctx_     = nullptr;   // owned
    std::vector<uint8_t>  graph_ctx_buf_;             // owned

    uint32_t              text_embed_dim_ = 0;
};

}  // namespace qinf::vision
