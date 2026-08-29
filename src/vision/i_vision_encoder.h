#pragma once
// i_vision_encoder.h — Seam A (docs/plan-gemma4-vision-impl.md §3).
//
// The vision-encoder boundary, extracted when the SECOND multimodal recipe
// (Gemma 4 `gemma4uv`) arrived as the forcing function. Before that, the
// concrete SigLIP encoder was the only implementation and was wired directly
// into the orchestrator ("presumed Gemma-shaped"). The interface is exactly
// the contract that already existed — no method was invented to anticipate a
// hypothetical encoder; the success metric (CLAUDE.md Architecture Pressure
// Test) is that hosting Gemma 4 required only a new implementation, not edits
// to SiglipEncoder's internals.
//
// Implementations:
//   - SiglipEncoder   (Gemma 3): 27-layer SigLIP ViT + 4×4 pool + projection.
//   - Gemma4UvEncoder (Gemma 4): blockless im2col projector, variable tokens.
//
// Both return host-side embeddings in the SAME layout — [projection_dim,
// n_tokens] row-major, projection_dim fastest — so the existing
// ImageEmbeddingsInput graph input and ImageEmbeddingCache plug into either
// without modification.
//
// Ownership / threading carry over from the concrete encoders: an encoder owns
// its own ggml graph + scheduler, shares (does not own) the engine backend,
// and is single-threaded.

#include <cstdint>
#include <vector>

namespace qinf::vision {

struct Bitmap;

class IVisionEncoder {
public:
    virtual ~IVisionEncoder() = default;

    // Encode one preprocessed bitmap into host-side image embeddings.
    // Returned vector size = mm_tokens_for(bitmap) * projection_dim(), row-major
    // with projection_dim fastest (row t = the embedding of image soft-token t).
    // Fail-loud (CLAUDE.md): throws std::runtime_error on shape/compute failure.
    virtual std::vector<float> encode(const Bitmap& bitmap) = 0;

    // How many image soft-tokens this bitmap encodes to. PURE — derived from the
    // preprocessed bitmap's dimensions, no encode performed — so a caller can
    // size the placeholder span BEFORE encoding. For SigLIP this is the constant
    // 256 (fixed 896 tile); for Gemma 4 `gemma4uv` it is (W/P)·(H/P), per-image.
    // The orchestrator cross-checks encode().size()/projection_dim() against
    // this value fail-loud (the encode is authoritative; this sizes the span).
    virtual uint32_t mm_tokens_for(const Bitmap& bitmap) const = 0;

    // The 2-D SHAPE of the soft-token grid this bitmap encodes to, in tokens:
    // nx columns by ny rows, with nx*ny == mm_tokens_for(bitmap). PURE, like
    // mm_tokens_for.
    //
    // Added for M-RoPE (P4, docs/plan-qwen35-vision-impl.md). Every encoder here
    // always had a grid — SigLIP's 16×16 pool, Gemma 4's (W/P)×(H/P), Qwen's
    // merged patch grid — but until a recipe needed 2-D positions, Seam A only
    // ever had to report the count, so the shape was flattened on the way out.
    // A recipe whose positions are scalar simply ignores this.
    //
    // Callers that need BOTH must not assume nx*ny == mm_tokens_for by faith:
    // the orchestrator cross-checks them fail-loud.
    virtual void mm_grid_for(const Bitmap& bitmap,
                             uint32_t& nx, uint32_t& ny) const = 0;

    // Output embedding dim — equals the host text model's embedding_length,
    // verified against the mmproj projection weight at construction.
    virtual uint32_t projection_dim() const = 0;
};

}  // namespace qinf::vision
