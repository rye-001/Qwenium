// test_image_embedding_cache.cpp — Phase 7 reuse gate (docs/plan-gemma-vision-impl.md).
//
// THE concretely-assertable Phase 7 deliverable: "a fixed image referenced
// twice in the same session encodes exactly once — assert via encoder call
// count, not timing." The cache is decoupled from VisionEncoder (it memoizes an
// encode() std::function), so the gate runs with a counting lambda — no model,
// no real encode. Also covers the C4 per-session image cap and the content_id 0
// non-cacheable sentinel.

#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "../../src/core/image_embedding_cache.h"

namespace {
// An encode() stand-in that counts invocations and returns a recognizable
// payload so reuse can be checked by value, not just by call count.
struct CountingEncoder {
    int calls = 0;
    std::function<std::vector<float>()> with(float tag) {
        return [this, tag]() { ++calls; return std::vector<float>(4, tag); };
    }
};
}  // namespace

// Reuse: the same id encodes once and returns identical embeddings both times.
TEST(ImageEmbeddingCache, SameIdEncodesExactlyOnce) {
    ImageEmbeddingCache cache(/*max_images=*/2);
    CountingEncoder enc;

    auto a = cache.get_or_encode(0xABCD, enc.with(1.0f));
    auto b = cache.get_or_encode(0xABCD, enc.with(2.0f));  // tag ignored: cache hit

    EXPECT_EQ(enc.calls, 1) << "second reference re-encoded the same image";
    EXPECT_EQ(a, b);
    EXPECT_EQ(a, std::vector<float>(4, 1.0f));  // first encode's payload reused
    EXPECT_EQ(cache.distinct_images(), 1u);
}

// Distinct ids each encode once.
TEST(ImageEmbeddingCache, DistinctIdsEncodeSeparately) {
    ImageEmbeddingCache cache(2);
    CountingEncoder enc;
    cache.get_or_encode(1, enc.with(1.0f));
    cache.get_or_encode(2, enc.with(2.0f));
    cache.get_or_encode(1, enc.with(9.0f));  // hit
    EXPECT_EQ(enc.calls, 2);
    EXPECT_EQ(cache.distinct_images(), 2u);
}

// content_id 0 is the "not set" sentinel: never cached, always encodes.
TEST(ImageEmbeddingCache, ZeroIdIsNonCacheable) {
    ImageEmbeddingCache cache(8);
    CountingEncoder enc;
    cache.get_or_encode(0, enc.with(1.0f));
    cache.get_or_encode(0, enc.with(1.0f));
    EXPECT_EQ(enc.calls, 2) << "content_id 0 must not dedup";
    EXPECT_EQ(cache.distinct_images(), 0u);
}

// C4 cap: a (cap+1)th DISTINCT image is refused fail-loud.
TEST(ImageEmbeddingCache, RejectsBeyondCap) {
    ImageEmbeddingCache cache(/*max_images=*/2);
    CountingEncoder enc;
    cache.get_or_encode(10, enc.with(1.0f));
    cache.get_or_encode(20, enc.with(2.0f));
    EXPECT_THROW(cache.get_or_encode(30, enc.with(3.0f)), std::runtime_error);
    // A repeat of an already-cached id is still fine at the cap (it is a hit).
    EXPECT_NO_THROW(cache.get_or_encode(10, enc.with(9.0f)));
    EXPECT_EQ(enc.calls, 2);
}

// clear() evicts — the next reference re-encodes.
TEST(ImageEmbeddingCache, ClearEvicts) {
    ImageEmbeddingCache cache(2);
    CountingEncoder enc;
    cache.get_or_encode(5, enc.with(1.0f));
    cache.clear();
    EXPECT_EQ(cache.distinct_images(), 0u);
    cache.get_or_encode(5, enc.with(1.0f));
    EXPECT_EQ(enc.calls, 2);
}
