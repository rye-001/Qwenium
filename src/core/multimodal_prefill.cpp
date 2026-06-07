#include "multimodal_prefill.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "../models/forward_pass_base.h"
#include "../models/i_image_embeddable.h"
#include "../vision/i_vision_encoder.h"
#include "../vision/bitmap.h"
#include "image_embedding_cache.h"

std::vector<float> prefill_multimodal(
    ForwardPassBase&                     text_fp,
    qinf::vision::IVisionEncoder&        encoder,
    ggml_backend_sched_t                 scheduler,
    const std::vector<int32_t>&          tokens,
    const std::vector<ImagePromptChunk>& images,
    int                                  pos,
    uint32_t                             slot,
    ImageEmbeddingCache*                 cache)
{
    // No images: plain text prefill. The vision subsystem is never touched.
    if (images.empty())
        return text_fp.run_prefill(tokens, pos, slot, scheduler);

    // v1 single-image scope. Multi-image fan-out (multiple armed spans) is a
    // Phase 7 capability; refuse loudly rather than encode only the first.
    if (images.size() != 1)
        throw std::runtime_error(
            "prefill_multimodal: parameter 'images': expected exactly 1 "
            "(Phase 5 single-image scope; multi-image is Phase 7), got: " +
            std::to_string(images.size()));

    const ImagePromptChunk& chunk = images[0];
    if (chunk.bitmap == nullptr)
        throw std::runtime_error(
            "prefill_multimodal: parameter 'images[0].bitmap': expected "
            "non-null, got: null");

    // Seam B: the recipe must accept an armed image span. Any recipe that
    // implements IImageEmbeddable qualifies; the orchestrator names no concrete
    // recipe type (Gemma3ForwardPass / Gemma4ForwardPass both qualify).
    auto* embeddable = dynamic_cast<IImageEmbeddable*>(&text_fp);
    if (embeddable == nullptr)
        throw std::runtime_error(
            "prefill_multimodal: parameter 'text_fp': expected a recipe "
            "implementing IImageEmbeddable (Seam B) for image input, got: a "
            "recipe without the image-substitution hook");

    // 1. Vision encode — own graph / own scheduler / shared backend (C3). The
    //    encode fully completes (compute + host readback) before text prefill
    //    begins; the two schedulers never run concurrently. With a session
    //    cache, a repeat of the same image (by content_id) skips the encode.
    std::vector<float> embd =
        (cache != nullptr)
            ? cache->get_or_encode(chunk.bitmap->content_id,
                                   [&]{ return encoder.encode(*chunk.bitmap); })
            : encoder.encode(*chunk.bitmap);

    // Token count is authoritative from the encode output (projection_dim-wide
    // rows); cross-check it against the pre-encode span size (mm_tokens_for the
    // preprocessed bitmap) fail-loud. For SigLIP these always agree (256); for
    // Gemma 4 the per-image count must match the placeholder span exactly.
    const uint32_t proj_dim = encoder.projection_dim();
    if (proj_dim == 0 || embd.size() % proj_dim != 0)
        throw std::runtime_error(
            "prefill_multimodal: slot 'encode output': expected size divisible "
            "by projection_dim=" + std::to_string(proj_dim) + ", got: " +
            std::to_string(embd.size()));
    const uint32_t n_img_tokens = static_cast<uint32_t>(embd.size() / proj_dim);
    const uint32_t expected_tokens = encoder.mm_tokens_for(*chunk.bitmap);
    if (n_img_tokens != expected_tokens)
        throw std::runtime_error(
            "prefill_multimodal: slot 'n_img_tokens': expected " +
            std::to_string(expected_tokens) + " (mm_tokens_for the preprocessed "
            "bitmap), got: " + std::to_string(n_img_tokens) +
            " (from the encode output)");

    // 2. Arm the recipe (substitution span). The recipe validates
    //    embd.size() == hidden_dim * n_img_tokens and the span bounds at build
    //    time — fail-loud there, named to the slot.
    embeddable->set_image_embeddings(std::move(embd), chunk.span_start, n_img_tokens);

    // 3. Text prefill — the soft-tokens substitute into the residual stream at
    //    [span_start, span_start + n_img_tokens); the recipe consumes the arm.
    return text_fp.run_prefill(tokens, pos, slot, scheduler);
}
