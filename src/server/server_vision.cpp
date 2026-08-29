// server_vision.cpp — image-input pipeline for the HTTP server.
// Extracted verbatim from http_server.cpp (docs/handoff-server-state.md §5);
// pure refactor, no behavior change.

#include "server_vision.h"

#include "inference_server.h"  // qwenium::InferenceRequest

#include "core/model.h"
#include "loader/tokenizer.h"
#include "sampling/sampling.h"
#include "models/forward_pass_base.h"

#include "core/multimodal_prefill.h"
#include "core/persistent_image_embedding_store.h"  // vision V1 (image-embed cache)
#include "core/prefix_library.h"                    // vision V2 (image-prefix KV cache)
#include "core/slot_snapshot.h"                      // shared L2 capture/restore helpers
#include "models/i_image_embeddable.h"
#include "vision/vision_model.h"
#include "vision/vision_loader.h"
#include "vision/i_vision_encoder.h"
#include "vision/bitmap.h"
#include "cli/image_loader.h"   // load_image_to_bitmap_from_memory (IO)
#include "cli/image_prompt.h"
#include "vision/vision_profile.h"

#include "ggml-backend.h"

#include <iostream>
#include <stdexcept>

ServerVision::ServerVision(Model& model, ForwardPassBase& forward_pass,
                           Tokenizer& tokenizer,
                           std::mutex& model_mutex, int max_ctx_per_slot,
                           const std::string& mmproj_path,
                           const std::string& image_embed_cache_dir,
                           const std::string& image_prefix_cache_dir)
    : model_(model),
      forward_pass_(forward_pass),
      tokenizer_(tokenizer),
      model_mutex_(model_mutex),
      max_ctx_per_slot_(max_ctx_per_slot) {

    std::cout << "Vision: loading mmproj projector '" << mmproj_path << "'"
              << std::endl;

    // The text recipe must implement IImageEmbeddable (Seam B) so its image span
    // can be armed; refuse loudly otherwise (opt-in, explicit).
    auto* img_recipe = dynamic_cast<IImageEmbeddable*>(&forward_pass_);
    if (img_recipe == nullptr)
        throw std::runtime_error(
            "ServerVision: parameter '--mmproj': expected a recipe "
            "implementing IImageEmbeddable (Gemma 3, Gemma 4, or "
            "Qwen 3.5-family vision), "
            "actual: the loaded model (" + model_label() + ") has no image hook");

    ggml_backend_t backend = model_.has_metal_backend()
        ? model_.get_backend_metal() : model_.get_backend_cpu();

    vmodel_  = std::make_unique<qinf::vision::VisionModel>();
    vloader_ = std::make_unique<qinf::vision::VisionLoader>();
    vloader_->parse_metadata(mmproj_path, *vmodel_);
    vloader_->load_tensors(*vmodel_, backend);

    // Projector-specific setup lives in ONE place (P0) — shared verbatim with
    // the CLI path, so the two can no longer drift. An unhandled projector
    // throws here instead of silently falling through to Gemma4Uv.
    qinf::vision::VisionProfile vprofile = qinf::vision::make_vision_profile(
        *vmodel_, backend, tokenizer_.get_vocabulary(),
        "ServerVision: parameter '--mmproj'");

    const std::string projector_tag = vprofile.projector_tag;
    vencoder_             = std::move(vprofile.encoder);
    boi_id_               = vprofile.boi_id;
    eoi_id_               = vprofile.eoi_id;
    soft_id_              = vprofile.soft_id;
    image_marker_prefix_  = vprofile.marker_prefix;
    image_wants_thinking_ = vprofile.wants_thinking;
    preprocess_           = vprofile.preprocess;

    // V1 — opt-in disk-backed image-embedding cache (--image-embed-cache):
    // a recurring image is encoded once per node ever (ViT skip). Keyed by
    // content_id + a vision identity header (projector + dim + backend);
    // opportunistic — a mismatch is a miss, never an error.
    if (!image_embed_cache_dir.empty()) {
        embed_store_ = std::make_unique<PersistentImageEmbeddingStore>(
            image_embed_cache_dir,
            make_vision_header(projector_tag, vmodel_->config().projection_dim,
                               ggml_backend_name(backend)));
        std::cout << "Vision: image-embed cache '" << image_embed_cache_dir
                  << "' (V1: encode once per node)" << std::endl;
    }

    // V2 — opt-in image-prefix KV cache (--image-prefix-cache): a recurring
    // (context, image) skips BOTH the ViT encode and the image-position
    // prefill. Version-gated fail-loud (the F9 rule). Requires a recipe that
    // exposes its KV cache(s); refuse loudly at setup if not (opt-in explicit).
    if (!image_prefix_cache_dir.empty()) {
        if (forward_pass_.snapshot_kv_caches().empty())
            throw std::runtime_error(
                "ServerVision: parameter '--image-prefix-cache': "
                "expected a recipe that exposes its KV cache(s) "
                "(snapshot_kv_caches non-empty), actual: a recipe without L2 "
                "snapshot support (" + model_label() + ")");
        // A 2-D image span writes nx*ny KV rows while advancing the position by
        // only max(nx, ny). The snapshot blob records a row count and no rope
        // coordinate, so such a slot cannot be round-tripped (plan §4 decision 3
        // — VL sessions are non-snapshottable in v1). Two things go wrong without
        // this: the V2 suffix prefill below computes its position as
        // `start_pos + img_end_local`, which is ROWS, and a restored blob has no
        // rope coordinate to restore. `capture_slot` refuses too, but only AFTER
        // a full model load, mmproj load and ViT encode — and once per REQUEST,
        // which on a server means a 500 per image instead of a refusal to start.
        // Same refusal the CLI makes at setup (cli/chat.cpp).
        if (img_recipe->image_span_is_2d())
            throw std::runtime_error(
                "ServerVision: parameter '--image-prefix-cache': expected a "
                "recipe whose image span advances one position per KV row, "
                "actual: an M-RoPE recipe (" + model_label() + "), whose image "
                "span occupies nx*ny rows but max(nx, ny) positions. The "
                "snapshot format carries no rope coordinate, so VL sessions are "
                "not prefix-cacheable in v1 — drop --image-prefix-cache");
        image_prefix_lib_ = std::make_unique<PrefixLibrary>(
            image_prefix_cache_dir,
            qinf::snapshot::make_snapshot_header(
                model_.get_metadata(), forward_pass_.snapshot_kv_caches()));
        std::cout << "Vision: image-prefix cache '" << image_prefix_cache_dir
                  << "' (V2: skip ViT + image prefill)" << std::endl;
    }

    std::cout << "Vision: enabled (projector=" << projector_tag
              << ", projection_dim=" << vmodel_->config().projection_dim
              << ")" << std::endl;
}

ServerVision::~ServerVision() = default;

std::string ServerVision::model_label() const {
    const auto& m = model_.get_metadata();
    return "arch='" + m.architecture + "', name='" + m.model_name + "'";
}

int ServerVision::run_multimodal_prefill(int slot_id,
                                         const qwenium::InferenceRequest& req,
                                         int start_pos,
                                         std::vector<int32_t>& out_tokens,
                                         qwenium::Sampler& sampler) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    // v1 single-image scope: marker expansion + the recipe substitution each
    // arm exactly one span. The route enforces this too; guard fail-loud.
    if (req.image_bytes.size() != 1)
        throw std::runtime_error(
            "slot " + std::to_string(slot_id) + ": parameter 'image': expected "
            "exactly 1 image per request (single-image scope), actual: " +
            std::to_string(req.image_bytes.size()));

    // Decode the image FILE bytes + apply the projector's preprocessing.
    qinf::vision::Bitmap bitmap = qinf::cli::load_image_to_bitmap_from_memory(
        req.image_bytes[0].data(), req.image_bytes[0].size(), preprocess_);
    const uint32_t n_img_tokens = vencoder_->mm_tokens_for(bitmap);

    // Tokenize the already marker-rendered prompt, then expand the single
    // image marker into the soft-token span the encoder fills.
    std::vector<int32_t> raw = tokenizer_.encode(req.prompt);
    qinf::cli::ExpandedImagePrompt built = qinf::cli::expand_image_markers(
        raw, boi_id_, soft_id_, eoi_id_, n_img_tokens);
    out_tokens = std::move(built.tokens);
    int img_span_start = built.span_start;

    // Gemma needs a BOS at conversation start; encode() does not prepend it.
    // Chat requests prefill at pos 0, so add it on the image turn (mirrors the
    // CLI image path — the BOS shifts every position, incl. the span, by one).
    const int32_t bos_id = model_.get_metadata().bos_token_id;
    if (start_pos == 0 && bos_id >= 0) {
        out_tokens.insert(out_tokens.begin(), bos_id);
        img_span_start += 1;
    }

    // Fail-loud ceiling guard (same shape as the text path), BEFORE the encode
    // touches the KV cache — an over-ceiling prompt would overflow it.
    if (max_ctx_per_slot_ > 0 &&
        static_cast<int>(out_tokens.size()) > max_ctx_per_slot_)
        throw std::runtime_error(
            "slot " + std::to_string(slot_id) + ": prompt too large; expected: "
            "<= " + std::to_string(max_ctx_per_slot_) + " tokens, actual: " +
            std::to_string(out_tokens.size()) + " (incl. " +
            std::to_string(n_img_tokens) + " image soft tokens)");

    // Encode → splice → text prefill. The per-session in-memory reuse cache
    // (`cache`) is the wrong tool for a server (it sees many distinct images),
    // so it stays null; but the OPT-IN disk caches do apply across requests:
    //   V1 (embed_store_): skip the ViT re-encode for a recurring image.
    //   V2 (image_prefix_lib_): skip ViT + image-position prefill for a
    //       recurring (context, image), by caching the image-inclusive KV.
    const uint32_t slot = static_cast<uint32_t>(slot_id);
    const std::vector<ImagePromptChunk> chunks = {{&bitmap, img_span_start}};
    std::vector<float> logits;
    if (!image_prefix_lib_) {
        // No V2 cache: the original single-call path (+ V1 thread-through).
        logits = prefill_multimodal(
            forward_pass_, *vencoder_, model_.get_scheduler(), out_tokens,
            chunks, start_pos, slot, /*cache=*/nullptr, embed_store_.get());
    } else {
        // V2: split the chunked prefill at the post-image boundary. The
        // image-inclusive KV is the cacheable unit; the question text after
        // it is the per-request variable suffix. Stateless server ⇒ every
        // request prefills at start_pos (0 for chat), so `preceding` is just
        // this request's pre-image tokens — no system-clone entanglement.
        const int img_end_local = img_span_start + static_cast<int>(n_img_tokens);
        const std::vector<int32_t> image_inclusive(
            out_tokens.begin(), out_tokens.begin() + img_end_local);
        const std::vector<int32_t> suffix(
            out_tokens.begin() + img_end_local, out_tokens.end());
        const int img_end_pos = start_pos + img_end_local;

        const std::vector<int32_t> preceding(
            out_tokens.begin(), out_tokens.begin() + img_span_start);
        const uint64_t ikey =
            PrefixLibrary::key_for(preceding, bitmap.content_id);
        const auto header = qinf::snapshot::make_snapshot_header(
            model_.get_metadata(), forward_pass_.snapshot_kv_caches());

        std::vector<uint8_t> blob;
        bool ihit = false;
        try {
            ihit = image_prefix_lib_->try_load(ikey, blob);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "slot " + std::to_string(slot_id) + ": '--image-prefix-cache': "
                "a stored blob for this (context, image) was built under a "
                "different model / quant / backend and is refused (" + e.what() +
                "). Clear or re-point the image-prefix cache dir.");
        }
        if (ihit) {
            // HIT: restore the image-inclusive KV (skip ViT + image prefill).
            // Requests #2..N load KV BYTES into the slot and prefill only the
            // question text — the text path proven immune to the image-prefill
            // graph-reuse degeneration (server-image-multirequest-bug.md §5).
            qinf::snapshot::restore_slot(forward_pass_, slot, blob, header);
            std::cout << "[image-prefix-cache] HIT: skipped ViT + image prefill ("
                      << n_img_tokens << " soft tokens)" << std::endl;
        } else {
            // MISS: encode + chunked-prefill [pre-image | image], capture the
            // post-image KV, store it under the key (+ V1 store on the encode).
            prefill_multimodal(
                forward_pass_, *vencoder_, model_.get_scheduler(),
                image_inclusive, chunks, start_pos, slot, /*cache=*/nullptr,
                embed_store_.get());
            image_prefix_lib_->store(
                ikey, qinf::snapshot::capture_slot(forward_pass_, slot, header));
            std::cout << "[image-prefix-cache] MISS: encoded + prefilled + stored ("
                      << n_img_tokens << " soft tokens)" << std::endl;
        }
        // The question suffix rides the plain text path either way.
        logits = forward_pass_.run_prefill(suffix, img_end_pos, slot,
                                           model_.get_scheduler());
    }

    const size_t vocab_size = model_.get_metadata().vocab_size;
    std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
    std::vector<int32_t> context(out_tokens);
    return sampler.sample(last_token_logits, context, /*token_strs=*/{});
}
