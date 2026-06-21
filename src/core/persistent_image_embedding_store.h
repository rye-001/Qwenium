#pragma once
// persistent_image_embedding_store.h — disk-backed, hash-keyed image-embedding
// cache (vision V1, docs/plan-session-snapshot.md).
//
// ImageEmbeddingCache (image_embedding_cache.h) memoizes encodings for ONE
// session. This store persists them to a directory keyed by content_id, so a
// recurring image is encoded once PER NODE EVER — the robust sweet spot from the
// plan: ship the image (universal), cache the embedding locally per node, pay
// the SigLIP/gemma4uv encode once. Image positions are still prefilled (that is
// V2's job); V1 only skips the ViT pass.
//
// Opportunistic + version-gated. A stored blob carries a vision CompatHeader
// (encoder identity + kernel path). On lookup, a header mismatch / corrupt blob
// / content_id mismatch is a MISS — the caller re-encodes, which is always the
// correct answer. This is deliberately NOT the hard fail-loud of the explicit
// cross-node prefix library (Phase 4): a local cache miss costs an encode, never
// a wrong result, so silent re-encode is safe here. A mismatched blob is never
// USED, only ignored.
//
// content_id 0 (Bitmap "not set") is non-cacheable: always encoded, never stored.

#include <cstdint>
#include <functional>
#include <string>

#include "graph_inputs/image_embedding_section.h"
#include "session/compat_header.h"

class PersistentImageEmbeddingStore {
public:
    // `dir` is created if absent. `header` is this node's encoder identity (see
    // make_vision_header); a stored blob must match it to be reused.
    PersistentImageEmbeddingStore(std::string dir,
                                  qinf::session::CompatHeader header);

    // Hit → return the stored embedding; miss → call `encode`, persist, return.
    // content_id 0 always encodes (never touches disk). `encode` runs at most
    // once per call and must return an embedding whose content_id == the key.
    qinf::vision::ImageEmbedding get_or_encode(
        uint64_t content_id,
        const std::function<qinf::vision::ImageEmbedding()>& encode) const;

    // Lower-level: true + fills `out` on a version-matching hit; false otherwise.
    bool try_load(uint64_t content_id, qinf::vision::ImageEmbedding& out) const;
    // Persist `e` under e.content_id (atomic: temp file + rename). No-op for id 0.
    void store(const qinf::vision::ImageEmbedding& e) const;

private:
    std::string path_for(uint64_t content_id) const;
    std::string dir_;
    qinf::session::CompatHeader header_;
};

// Encoder-identity header for the store: arch_id from the projector tag,
// embedding_length = projection_dim, build_path_tag = encode kernel path
// (backend). A different projector / dim / backend ⇒ different header ⇒ miss.
qinf::session::CompatHeader make_vision_header(const std::string& projector_tag,
                                               uint32_t projection_dim,
                                               const std::string& backend_tag);
