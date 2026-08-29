#pragma once
// multimodal_prefill.h — the vision/text prefill orchestrator.
//
// THE single top-level call site that bridges the vision subsystem and a text
// recipe. Recipes do not orchestrate vision; this does. The flow:
//   1. Encode the image once through Seam A (IVisionEncoder — its own graph and
//      scheduler, sharing the device backend), or reuse a cached encoding.
//   2. Arm the recipe's image-embedding substitution through Seam B
//      (IImageEmbeddable::set_image_embeddings). Family differences are
//      parameters there, not branches here: Gemma 3 asks for a bidirectional
//      mask over the span, Gemma 4 is plain causal, and the two Qwen recipes
//      additionally pass the 2-D grid M-RoPE needs.
//   3. Drive the prefill as a chunk sequence over one shared KV — every chunk
//      but the last advances state only (feed_tokens), the last produces logits.
//      The image is its own chunk so matmul widths match the reference
//      implementation and the single-vs-chunked Metal precision divergence
//      cannot arise.
// Recipe-agnostic: gemma3, gemma4, qwen36 and qwen35 all host images through it
//   unchanged. It requires feed_tokens support and says so fail-loud.
//
// Lives in src/engine/ but is compiled into consumers (tests, cli, server), not
// the qinf-engine static lib — same arrangement as decode_step.cpp, because it
// depends on BOTH qinf-vision and qinf-models while qinf-engine sits below
// qinf-models.
//
// Scope: ONE image per prompt, single turn, single tile. Encoding is reused
// across turns and processes (session/image_embedding_cache and
// persistent_image_embedding_store); multi-image fan-out is not built — >1
// image throws fail-loud rather than silently encoding only the first.

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
