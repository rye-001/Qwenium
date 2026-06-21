#include "multimodal_prefill.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../models/forward_pass_base.h"
#include "../models/i_image_embeddable.h"
#include "../vision/i_vision_encoder.h"
#include "../vision/bitmap.h"
#include "image_embedding_cache.h"
#include "persistent_image_embedding_store.h"

namespace {

// One prefill chunk: a contiguous span fed to the recipe over the shared KV.
// A text chunk carries only token ids; the image chunk additionally carries the
// encoded soft-token embeddings that overwrite its residual rows at build time.
struct PrefillChunk {
    std::vector<int32_t> tokens;      // span token ids (placeholders for image)
    std::vector<float>   image_embd;  // moved into the recipe when n_img > 0
    uint32_t             n_img;       // image soft-token count (0 ⇒ text chunk)
};

// Drive a chunk sequence over one shared KV, continuing positions, causal.
// Every chunk but the last advances state only (feed_tokens — KV append, no LM
// head); the last chunk produces the logits (run_prefill). An image chunk arms
// the recipe's substitution span at local span_start 0 before its build. This is
// our analogue of llama's mtmd_helper_eval_chunks: the image is its own decode
// batch, so our matmul widths match the reference and the single-vs-chunked Metal
// precision divergence cannot arise. Caller guarantees the slot KV is at base_pos.
std::vector<float> drive_prefill_chunks(
    ForwardPassBase&            text_fp,
    IImageEmbeddable&           embeddable,
    ggml_backend_sched_t        scheduler,
    std::vector<PrefillChunk>&  chunks,
    int                         base_pos,
    uint32_t                    slot)
{
    if (chunks.empty())
        throw std::runtime_error(
            "drive_prefill_chunks: slot 'chunks': expected >= 1, got: 0");

    int pos = base_pos;
    for (size_t i = 0; i < chunks.size(); ++i) {
        PrefillChunk& c = chunks[i];
        if (c.n_img > 0)
            embeddable.set_image_embeddings(std::move(c.image_embd), 0, c.n_img);
        if (i + 1 == chunks.size())
            return text_fp.run_prefill(c.tokens, pos, slot, scheduler);
        text_fp.feed_tokens(c.tokens, slot, scheduler);
        pos += static_cast<int>(c.tokens.size());
    }
    // Unreachable: the loop returns on the last chunk.
    throw std::runtime_error("drive_prefill_chunks: fell through chunk loop");
}

}  // namespace

std::vector<float> prefill_multimodal(
    ForwardPassBase&                     text_fp,
    qinf::vision::IVisionEncoder&        encoder,
    ggml_backend_sched_t                 scheduler,
    const std::vector<int32_t>&          tokens,
    const std::vector<ImagePromptChunk>& images,
    int                                  pos,
    uint32_t                             slot,
    ImageEmbeddingCache*                 cache,
    const PersistentImageEmbeddingStore* embed_store)
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
    //    With a persistent store (V1), even a first-this-session encode is
    //    skipped if the image was encoded on a previous run/process on this node.
    const uint64_t content_id = chunk.bitmap->content_id;
    auto encode_image = [&]() -> std::vector<float> {
        if (embed_store == nullptr)
            return encoder.encode(*chunk.bitmap);
        // Disk-backed: load the cached ViT output on a hit, else encode and
        // persist. The store demands the encode's content_id match the key.
        qinf::vision::ImageEmbedding e = embed_store->get_or_encode(
            content_id, [&]() -> qinf::vision::ImageEmbedding {
                std::vector<float> data = encoder.encode(*chunk.bitmap);
                const uint32_t pd = encoder.projection_dim();
                qinf::vision::ImageEmbedding out;
                out.content_id = content_id;
                out.n_embd     = pd;
                out.n_tokens   = pd ? static_cast<uint32_t>(data.size() / pd) : 0;
                out.data       = std::move(data);
                return out;
            });
        return std::move(e.data);
    };
    std::vector<float> embd =
        (cache != nullptr)
            ? cache->get_or_encode(content_id, encode_image)
            : encode_image();

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

    // 2. Chunked prefill. The prompt is decoded as an ordered chunk sequence —
    //    [text-prefix] [image] [text-suffix] — over one shared KV with continuing
    //    positions, causal throughout. This mirrors llama's mtmd_helper_eval_chunks
    //    (the image is always its own decode batch, never fused into one combined
    //    batch). Load-bearing: a single combined batch and the chunked sequence are
    //    NOT bit-identical on Metal (matmul width changes the rounding), and Gemma's
    //    high-magnitude image residual under scale-1.0 attention amplifies that
    //    divergence into token-soup. Chunking makes our matmul widths match the
    //    reference, so the divergence cannot arise. See [[project_gemma4_vision]].
    //
    //    The image chunk arms the recipe's substitution span at local span_start 0
    //    (the chunk IS the image), so no whole-sequence span arithmetic is needed.
    if (!text_fp.feed_tokens_supported())
        throw std::runtime_error(
            "prefill_multimodal: parameter 'text_fp': expected a recipe whose "
            "feed_tokens_supported()==true (image prefill is chunked "
            "text|image|text over a shared KV), got: a recipe without the "
            "want_logits=false head guard");

    const int32_t  span = chunk.span_start;
    const uint32_t nimg = n_img_tokens;
    if (span < 0 || static_cast<size_t>(span) + nimg > tokens.size())
        throw std::runtime_error(
            "prefill_multimodal: slot 'image_span': expected within [0, " +
            std::to_string(tokens.size()) + "), got: start=" +
            std::to_string(span) + " n_img=" + std::to_string(nimg));

    std::vector<PrefillChunk> chunks;
    if (span > 0)  // text prefix (absent when the image opens the turn)
        chunks.push_back({{tokens.begin(), tokens.begin() + span}, {}, 0});
    chunks.push_back({{tokens.begin() + span, tokens.begin() + span + nimg},
                      std::move(embd), nimg});
    if (static_cast<size_t>(span) + nimg < tokens.size())  // text suffix
        chunks.push_back({{tokens.begin() + span + nimg, tokens.end()}, {}, 0});

    return drive_prefill_chunks(text_fp, *embeddable, scheduler, chunks, pos, slot);
}
