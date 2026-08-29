#include "image_embedding_cache.h"

#include <stdexcept>
#include <string>

std::vector<float> ImageEmbeddingCache::get_or_encode(
    uint64_t content_id,
    const std::function<std::vector<float>()>& encode)
{
    // Sentinel: an unset content id is non-cacheable — always encode fresh and
    // never store (so distinct unset bitmaps do not collide to one slot).
    if (content_id == 0)
        return encode();

    auto it = store_.find(content_id);
    if (it != store_.end())
        return it->second;  // reuse — no encode

    // Miss: a new distinct image. Enforce the C4 per-session cap before encoding.
    if (store_.size() >= max_images_)
        throw std::runtime_error(
            "ImageEmbeddingCache: parameter 'max_images': expected at most " +
            std::to_string(max_images_) +
            " distinct images per session (C4 workload envelope), got: a " +
            std::to_string(store_.size() + 1) + "th distinct image (id " +
            std::to_string(content_id) + ")");

    auto inserted = store_.emplace(content_id, encode());
    return inserted.first->second;
}
