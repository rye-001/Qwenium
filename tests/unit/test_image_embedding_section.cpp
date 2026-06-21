// test_image_embedding_section.cpp — vision V1, model-free.
// Section byte round-trip + the persistent store's hit / miss / version-gating
// / non-cacheable semantics, all on synthetic embeddings (no encoder/model).

#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/persistent_image_embedding_store.h"
#include "graph_inputs/image_embedding_section.h"
#include "session/compat_header.h"
#include "session/snapshot_io.h"

using qinf::session::CompatHeader;
using qinf::session::SnapshotReader;
using qinf::session::SnapshotWriter;
using qinf::vision::ImageEmbedding;
using qinf::vision::ImageEmbeddingSection;

namespace {
ImageEmbedding make_embedding(uint64_t id, uint32_t n_embd, uint32_t n_tokens) {
    ImageEmbedding e;
    e.content_id = id;
    e.n_embd = n_embd;
    e.n_tokens = n_tokens;
    e.data.resize(static_cast<size_t>(n_embd) * n_tokens);
    for (size_t i = 0; i < e.data.size(); ++i)
        e.data[i] = static_cast<float>(i) * 0.25f - 3.0f;
    return e;
}
}  // namespace

TEST(ImageEmbeddingSection, ByteRoundTrip) {
    ImageEmbedding src = make_embedding(0xABCDEF12u, 8, 5);
    SnapshotWriter w;
    ImageEmbeddingSection(src).write(w);

    ImageEmbedding dst;
    SnapshotReader r(w.buffer());
    ImageEmbeddingSection(dst).read(r);

    EXPECT_EQ(dst.content_id, src.content_id);
    EXPECT_EQ(dst.n_embd, src.n_embd);
    EXPECT_EQ(dst.n_tokens, src.n_tokens);
    EXPECT_EQ(dst.data, src.data);  // exact float bytes
    EXPECT_EQ(r.remaining(), 0u);
}

TEST(PersistentImageEmbeddingStore, StoreThenLoadHit) {
    const std::string dir = std::string(testing::TempDir()) + "/v1_store_hit";
    CompatHeader h = make_vision_header("gemma3-siglip", 8, "CPU");
    PersistentImageEmbeddingStore store(dir, h);

    ImageEmbedding e = make_embedding(123456789ull, 8, 5);
    store.store(e);

    ImageEmbedding out;
    ASSERT_TRUE(store.try_load(e.content_id, out));
    EXPECT_EQ(out.data, e.data);
    EXPECT_EQ(out.n_tokens, e.n_tokens);
    std::remove((dir + "/img-123456789.embd").c_str());
}

TEST(PersistentImageEmbeddingStore, MissOnAbsent) {
    const std::string dir = std::string(testing::TempDir()) + "/v1_store_absent";
    PersistentImageEmbeddingStore store(dir, make_vision_header("p", 8, "CPU"));
    ImageEmbedding out;
    EXPECT_FALSE(store.try_load(999u, out));
}

TEST(PersistentImageEmbeddingStore, VersionMismatchIsMissNotError) {
    // A blob written under one encoder identity must be a MISS (not a throw) when
    // read under a different one — the cache re-encodes the correct embedding.
    const std::string dir = std::string(testing::TempDir()) + "/v1_store_ver";
    ImageEmbedding e = make_embedding(42u, 8, 5);
    PersistentImageEmbeddingStore(dir, make_vision_header("gemma3-siglip", 8, "Metal"))
        .store(e);

    PersistentImageEmbeddingStore other(dir, make_vision_header("gemma3-siglip", 8, "CPU"));
    ImageEmbedding out;
    EXPECT_FALSE(other.try_load(42u, out));  // backend differs ⇒ miss, no throw
    std::remove((dir + "/img-42.embd").c_str());
}

TEST(PersistentImageEmbeddingStore, ContentIdZeroNonCacheable) {
    const std::string dir = std::string(testing::TempDir()) + "/v1_store_zero";
    PersistentImageEmbeddingStore store(dir, make_vision_header("p", 8, "CPU"));
    ImageEmbedding e = make_embedding(0, 8, 5);
    store.store(e);  // no-op
    ImageEmbedding out;
    EXPECT_FALSE(store.try_load(0, out));
}

TEST(PersistentImageEmbeddingStore, GetOrEncodeEncodesOnceThenHits) {
    const std::string dir = std::string(testing::TempDir()) + "/v1_store_goe";
    PersistentImageEmbeddingStore store(dir, make_vision_header("p", 8, "CPU"));
    int calls = 0;
    auto enc = [&]() {
        ++calls;
        return make_embedding(7u, 8, 5);
    };
    ImageEmbedding a = store.get_or_encode(7u, enc);
    ImageEmbedding b = store.get_or_encode(7u, enc);  // second call must hit disk
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(a.data, b.data);
    std::remove((dir + "/img-7.embd").c_str());
}
