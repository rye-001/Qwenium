#pragma once
// multimodal_prefill.h — Phase 5 prefill orchestrator (docs/plan-gemma-vision-impl.md).
//
// THE single top-level call site that bridges the vision subsystem and the
// gemma3 text recipe. Recipes do not orchestrate vision; this does. The flow:
//   1. Encode each image once (VisionEncoder — its own graph, its own
//      scheduler, shared backend; C3 boundary).
//   2. Arm the recipe's image-embedding substitution + bidirectional mask
//      (Gemma3ForwardPass::set_image_embeddings — Phases 3 & 4).
//   3. Run the text prefill (the encoded soft-tokens substitute into the
//      residual stream at the placeholder span).
//
// Lives in src/engine/ but is compiled into consumers (tests, cli), not the
// `core` static lib — same arrangement as decode_step.cpp, because it depends
// on BOTH qinf-vision and qinf-models while `core` sits below qinf-models.
//
// v1 scope (Phase 5): single image, single turn, single tile. Image-embedding
// reuse across turns and multiple images per prompt are Phase 7; >1 image here
// throws fail-loud rather than silently encoding only the first.

#include <cstdint>
#include <vector>

#include "ggml-backend.h"

class ForwardPassBase;
class ImageEmbeddingCache;
class PersistentImageEmbeddingStore;

namespace qinf::vision {
class IVisionEncoder;
struct Bitmap;
}  // namespace qinf::vision

// One image in the prompt: a prepared bitmap and the LOCAL token index (into
// `tokens`) where its mm_tokens_per_image soft-token placeholders begin. Local
// because the recipe uses it as a column offset into this batch's residual
// stream; the recipe converts to an absolute position for the attention mask.
struct ImagePromptChunk {
    const qinf::vision::Bitmap* bitmap;      // borrowed, not owned
    int32_t                     span_start;  // local index of the first soft-token
};

// Encode the prompt's image(s), arm the text recipe, and run text prefill.
// Returns the prefill logits (same buffer shape as ForwardPassBase::run_prefill).
//
// `text_fp` drives the prefill (ForwardPassBase::run_prefill) and, when images
// are present, must also implement IImageEmbeddable (Seam B) so its image span
// can be armed — checked fail-loud via dynamic_cast. Both Gemma3ForwardPass and
// Gemma4ForwardPass qualify; the orchestrator names neither concrete type.
//
// `images` empty  -> plain text prefill (passthrough, no vision touched).
// `images` size 1 -> the Phase 5 path.
// `images` size >1 -> throws (Phase 7 multi-image-in-one-prompt not implemented).
//
// `cache` (Phase 7): when non-null, the image is encoded through the per-session
// reuse cache keyed by its content_id, so a repeat of the same image across
// turns reuses the encode instead of recomputing it (and the C4 image cap is
// enforced there). When null, the image is encoded directly every call.
//
// `embed_store` (vision V1, docs/plan-session-snapshot.md): when non-null, a
// per-session cache miss consults this disk-backed store BEFORE encoding, so an
// image encoded on a previous run/process is loaded instead of re-running the
// ViT pass (and the fresh encode is persisted on a store miss). Layering:
// in-memory `cache` -> `embed_store` (disk) -> encoder. Opportunistic — a
// version/identity mismatch is a miss, never an error.
//
// Fail-loud (CLAUDE.md): names the parameter, expected, actual. Span/size
// validation of the encoded embeddings is delegated to the recipe's
// substitution site (it owns the residual stream and the placeholder span).
std::vector<float> prefill_multimodal(
    ForwardPassBase&                     text_fp,
    qinf::vision::IVisionEncoder&        encoder,
    ggml_backend_sched_t                 scheduler,
    const std::vector<int32_t>&          tokens,
    const std::vector<ImagePromptChunk>& images,
    int                                  pos,
    uint32_t                             slot,
    ImageEmbeddingCache*                 cache = nullptr,
    const PersistentImageEmbeddingStore* embed_store = nullptr);
