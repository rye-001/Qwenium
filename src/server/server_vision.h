#pragma once
// server_vision.h — the HTTP server's image-input pipeline, extracted from
// http_server.cpp (docs/handoff-server-state.md §5).
//
// ServerVision owns *only* the vision/cache state the server needs to consume
// image input on /v1/chat/completions: the mmproj-loaded encoder, the image
// marker ids, the projector preprocessing, and the two opt-in disk caches
// (V1 image-embed, V2 image-prefix KV). It is constructed once, from the
// integration's ctor, only when the server is started with --mmproj; a
// text-only server never builds one.
//
// It is NOT a god object: it does not own the model, tokenizer, sampler, or
// scheduler — those are borrowed by reference from QweniumServerIntegration. It
// implements the body of the engine's MultimodalPrefillFunc callback and the
// small capability surface the chat route reads (marker prefix, thinking flag).
// The engine (InferenceServer) and the recipes are untouched.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vision/image_preprocess.h"  // qinf::vision::ImagePreprocess (value member)

class Model;
class ForwardPassBase;
class Tokenizer;
class PersistentImageEmbeddingStore;  // vision V1
class PrefixLibrary;                   // vision V2

namespace qwenium {
class Sampler;
struct InferenceRequest;
}  // namespace qwenium

namespace qinf::vision {
class VisionModel;
class VisionLoader;
class IVisionEncoder;
}  // namespace qinf::vision

// Owns the server's image pipeline + image caches. One instance per
// vision-enabled server (null on QweniumServerIntegration for a text-only server).
class ServerVision {
public:
    // Construct = the old setup_vision(): load the mmproj into a VisionModel,
    // pick the encoder + markers by projector type, resolve the image-marker
    // token ids, set the projector preprocessing + render prefix, and build the
    // V1/V2 caches when their dirs are given. Fail-loud (CLAUDE.md) if the
    // recipe has no image hook or the vocab lacks the markers.
    //
    // All references are borrowed from QweniumServerIntegration and must outlive
    // this object. `model_mutex` is the integration's single model mutex shared
    // across prefill/decode — run_multimodal_prefill takes it. The sampler is NOT
    // held here: it is per-slot (owned by the integration) and passed per-call.
    ServerVision(Model& model, ForwardPassBase& forward_pass, Tokenizer& tokenizer,
                 std::mutex& model_mutex,
                 int max_ctx_per_slot, const std::string& mmproj_path,
                 const std::string& image_embed_cache_dir,
                 const std::string& image_prefix_cache_dir);

    // Out-of-line (defaulted in the .cpp) so the unique_ptr members can be
    // destroyed where their forward-declared types (PersistentImageEmbeddingStore,
    // PrefixLibrary, vision types) are complete.
    ~ServerVision();

    // ── Capability surface read by the chat route ────────────────────────────
    // The projector's image-marker string the route prepends to the user turn
    // (e.g. "\n\n<start_of_image>\n\n" / "<|image>").
    const std::string& image_marker_prefix() const { return image_marker_prefix_; }
    // True when the image path must use the thinking branch (Gemma 4): a leading
    // system <|think|> turn + a generation prompt ending at "model\n". Gemma 3
    // keeps its no-think image path. See docs/server-image-multirequest-bug.md §5.
    bool image_wants_thinking() const { return image_wants_thinking_; }

    // ── The MultimodalPrefillFunc body ───────────────────────────────────────
    // Decode → preprocess → tokenize → expand the image marker → encode →
    // splice → text prefill, then sample the first token. Writes the full
    // expanded token stream into `out_tokens`. Fail-loud (throws) on a malformed
    // image, >1 image, or an over-ceiling prompt. `sampler` is this slot's
    // sampler (built by the integration from the request).
    int run_multimodal_prefill(int slot_id, const qwenium::InferenceRequest& req,
                               int start_pos, std::vector<int32_t>& out_tokens,
                               qwenium::Sampler& sampler);

private:
    // Human-readable identity for fail-loud errors: arch + model name.
    std::string model_label() const;

    // ── Borrowed (not owned) ─────────────────────────────────────────────────
    Model&                  model_;
    ForwardPassBase&        forward_pass_;
    Tokenizer&              tokenizer_;
    std::mutex&             model_mutex_;
    int                     max_ctx_per_slot_;

    // ── Owned vision state ───────────────────────────────────────────────────
    std::unique_ptr<qinf::vision::VisionModel>    vmodel_;
    std::unique_ptr<qinf::vision::VisionLoader>   vloader_;
    std::unique_ptr<qinf::vision::IVisionEncoder> vencoder_;
    int32_t boi_id_ = -1, eoi_id_ = -1, soft_id_ = -1;
    std::string image_marker_prefix_;
    bool image_wants_thinking_ = false;  // Gemma 4 image input → thinking branch
    qinf::vision::ImagePreprocess preprocess_;

    // ── Image caches (opt-in; null unless the dir flag is set) ───────────────
    // V1: skip the ViT re-encode for a recurring image (keyed by content_id).
    // V2: skip ViT + image-position prefill for a recurring (context, image).
    // Both are request-independent and transparent — they fit the server's
    // stateless/F9 model (only faster). docs/plan-session-snapshot.md.
    std::unique_ptr<PersistentImageEmbeddingStore> embed_store_;       // V1
    std::unique_ptr<PrefixLibrary>                 image_prefix_lib_;  // V2
};
