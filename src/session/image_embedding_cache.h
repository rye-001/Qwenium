#pragma once
// image_embedding_cache.h — per-session image-embedding reuse 
//
// The naive pipeline re-encodes the same image on every turn of a multi-turn
// conversation — wasted work (a SigLIP encode is the most expensive step in the
// whole pipeline). This cache memoizes encoded embeddings keyed by a
// content-derived id (Bitmap::content_id, an FNV hash of the normalized pixels
// set by the image loader), so a given image is encoded exactly once per
// session and reused thereafter. Mirrors llama.cpp mtmd's bitmap-id reuse.
//
// Lifetime = the session. clear()/destruction evicts. No cross-session sharing
// in v1 (a security/isolation surface, deliberately deferred).
//
// content_id 0 is the Bitmap "not set" sentinel — NON-CACHEABLE: always
// encoded, never stored, never counted against the cap (so unset bitmaps do not
// all collide to one slot; see bitmap.h).
//
// C4 workload envelope: at most `max_images` DISTINCT images per session
// (default 2 — single-tile, 2×256 = 512 image tokens ≈ 13% of the 4 K
// envelope). A (cap+1)th distinct image is refused fail-loud, not silently
// dropped — the workload-envelope boundary is a stated contract, not a surprise.

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

class ImageEmbeddingCache {
public:
    explicit ImageEmbeddingCache(uint32_t max_images = 2)
        : max_images_(max_images) {}

    // Return the embeddings for `content_id`, encoding via `encode` on a miss.
    // `encode` is invoked at most once per distinct non-zero id for the session.
    // Returned by value (the cached copy stays for the next reuse); the ~2.6 MB
    // copy is negligible against the encode it avoids. Throws (fail-loud,
    // CLAUDE.md) when a new distinct id would exceed max_images.
    std::vector<float> get_or_encode(
        uint64_t content_id,
        const std::function<std::vector<float>()>& encode);

    // Distinct images currently cached (excludes content_id 0).
    size_t distinct_images() const { return store_.size(); }

    void clear() { store_.clear(); }

private:
    uint32_t max_images_;
    std::unordered_map<uint64_t, std::vector<float>> store_;  // keyed by content_id != 0
};
