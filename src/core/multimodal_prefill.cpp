#include <algorithm>
#include <cstdlib>
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
    uint32_t             grid_w = 0;  // image soft-token grid (M-RoPE); 0 = none
    uint32_t             grid_h = 0;

    // How far this chunk advances the ROPE POSITION, which is not always its
    // row count (P4). A text chunk advances by its token count. An image chunk
    // under M-RoPE advances by max(nx, ny) — llama.cpp
    // mtmd_image_tokens_get_n_pos, MTMD_POS_TYPE_MROPE — because its tokens
    // are laid out on a 2-D grid whose rows and columns share positions, so
    // the span consumes far fewer positions than it occupies KV rows.
    uint32_t n_pos_advance = 0;
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
            embeddable.set_image_embeddings(std::move(c.image_embd), 0, c.n_img,
                                            c.grid_w, c.grid_h);
        // Record any rows-vs-positions divergence this chunk introduces BEFORE
        // handing off, so the next turn's starting position is derivable from
        // the engine rather than from the KV row count (get_rope_pos).
        text_fp.note_span_rows_vs_positions(
            slot, static_cast<uint32_t>(c.tokens.size()), c.n_pos_advance);
        if (i + 1 == chunks.size())
            return text_fp.run_prefill(c.tokens, pos, slot, scheduler);
        // Pass `pos` explicitly: feed_tokens would otherwise derive it from the
        // KV row count, which stops equalling the rope position the moment an
        // image chunk advances by max(nx, ny) instead of its token count.
        text_fp.feed_tokens(c.tokens, slot, scheduler, pos);
        pos += static_cast<int>(c.n_pos_advance);
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

    // The image's soft-token GRID, needed both for the M-RoPE component
    // construction inside the recipe and for the position advance here. Seam A
    // reports it; cross-check it against the count rather than trusting that
    // nx*ny == mm_tokens_for (they come from different code paths).
    uint32_t grid_w = 0, grid_h = 0;
    encoder.mm_grid_for(*chunk.bitmap, grid_w, grid_h);
    if (static_cast<uint64_t>(grid_w) * grid_h != nimg)
        throw std::runtime_error(
            "prefill_multimodal: slot 'mm_grid_for': expected nx*ny == "
            "mm_tokens_for (" + std::to_string(nimg) + "), got: " +
            std::to_string(grid_w) + "*" + std::to_string(grid_h) + " = " +
            std::to_string(static_cast<uint64_t>(grid_w) * grid_h));

    // Position advance for the image span. max(nx, ny) under M-RoPE (the image
    // is 2-D, so rows and columns share positions); the token count otherwise.
    // Ported from llama.cpp mtmd_image_tokens_get_n_pos.
    const uint32_t img_pos_advance =
        embeddable->image_span_is_2d() ? std::max(grid_w, grid_h) : nimg;

    std::vector<PrefillChunk> chunks;

    if (span > 0) {  // text prefix (absent when the image opens the turn)
        const uint32_t n = static_cast<uint32_t>(span);
        chunks.push_back({{tokens.begin(), tokens.begin() + span}, {}, 0, 0, 0, n});
    }
    chunks.push_back({{tokens.begin() + span, tokens.begin() + span + nimg},
                      std::move(embd), nimg, grid_w, grid_h, img_pos_advance});
    if (static_cast<size_t>(span) + nimg < tokens.size()) {  // text suffix
        const uint32_t n =
            static_cast<uint32_t>(tokens.size() - (static_cast<size_t>(span) + nimg));
        chunks.push_back({{tokens.begin() + span + nimg, tokens.end()}, {}, 0, 0, 0, n});
    }

    return drive_prefill_chunks(text_fp, *embeddable, scheduler, chunks, pos, slot);
}
