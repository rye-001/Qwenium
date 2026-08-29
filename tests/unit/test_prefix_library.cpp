// test_prefix_library.cpp — Phase 4 prefix library, model-free.
//
// Exercises the disk-backed warm-prefix store's contract WITHOUT a model:
//   - key_for is pure, stable, and span-sensitive (the prompt-hash key);
//   - store + try_load round-trips a blob (Hit);
//   - an absent key is a Miss (false), NOT an error (first-use prefill);
//   - a present blob whose embedded CompatHeader mismatches the node identity is
//     REFUSED FAIL-LOUD (throws, names the field) — the Phase 4 non-negotiable,
//     and the deliberate inversion of the opportunistic image store;
//   - key 0 (empty span) is non-cacheable;
//   - bounded eviction drops the oldest entry at capacity;
//   - store() publishes atomically (no leftover .tmp).
//
// Blobs are synthetic: a CompatHeader followed by arbitrary payload bytes. The
// library only reads/gates the header — KV section (de)serialization is the
// owning module's job and is covered by the L2 harness.

#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "session/prefix_library.h"
#include "session/compat_header.h"
#include "session/snapshot_io.h"

using qinf::session::CompatHeader;
using qinf::session::SnapshotWriter;

namespace {
namespace fs = std::filesystem;

// A node identity header: arch + shape + the load-bearing L2 discriminants.
CompatHeader make_node_header() {
    CompatHeader h;
    h.arch_id = 0xA11CE;
    h.block_count = 28;
    h.embedding_length = 3072;
    h.vocab_size = 262144;
    h.weights_hash = 0xDEADBEEFCAFEull;
    h.build_path_tag = 0x1234567890ull;
    return h;
}

// A capture blob = header + trailing payload (stands in for the manifest's
// sections, which the library never parses).
std::vector<uint8_t> make_blob(const CompatHeader& h, uint8_t fill, size_t n) {
    SnapshotWriter w;
    h.write(w);
    std::vector<uint8_t> payload(n, fill);
    w.put_bytes(payload.data(), payload.size());
    return w.buffer();
}

std::string fresh_dir(const std::string& name) {
    std::string d = (fs::path(testing::TempDir()) / name).string();
    std::error_code ec;
    fs::remove_all(d, ec);
    return d;
}
}  // namespace

TEST(PrefixLibrary, KeyForIsPureStableAndSpanSensitive) {
    std::vector<int32_t> a = {2, 18, 4, 9001, 7};
    std::vector<int32_t> b = {2, 18, 4, 9001, 7};
    std::vector<int32_t> c = {2, 18, 4, 9001, 8};  // last token differs
    EXPECT_EQ(PrefixLibrary::key_for(a), PrefixLibrary::key_for(b));
    EXPECT_NE(PrefixLibrary::key_for(a), PrefixLibrary::key_for(c));
    EXPECT_EQ(PrefixLibrary::key_for({}), 0u);  // empty span = non-cacheable
}

TEST(PrefixLibrary, ImageKeyForIsPureContextAndImageSensitive) {
    std::vector<int32_t> ctx_a = {2, 18, 4, 9001};
    std::vector<int32_t> ctx_b = {2, 18, 4, 9002};  // preceding context differs
    const uint64_t img1 = 0xABCDEF12u;
    const uint64_t img2 = 0x99887766u;  // a different image

    // Pure + stable: same (context, image) → same key, on any host.
    EXPECT_EQ(PrefixLibrary::key_for(ctx_a, img1),
              PrefixLibrary::key_for(ctx_a, img1));
    // Sensitive to the preceding context (the V2 reuse condition: everything
    // before the image must be fixed).
    EXPECT_NE(PrefixLibrary::key_for(ctx_a, img1),
              PrefixLibrary::key_for(ctx_b, img1));
    // Sensitive to WHICH image, with the same preceding context.
    EXPECT_NE(PrefixLibrary::key_for(ctx_a, img1),
              PrefixLibrary::key_for(ctx_a, img2));
    // An image that opens the turn (empty preceding context) is still cacheable.
    EXPECT_NE(PrefixLibrary::key_for({}, img1), 0u);
    // content_id 0 (an unset bitmap) is the non-cacheable sentinel.
    EXPECT_EQ(PrefixLibrary::key_for(ctx_a, 0), 0u);
    // The image overload does not collide with the text-only overload for the
    // same token span (distinct key spaces: image mixes in 8 content_id bytes).
    EXPECT_NE(PrefixLibrary::key_for(ctx_a, img1), PrefixLibrary::key_for(ctx_a));
}

TEST(PrefixLibrary, StoreThenLoadHits) {
    const std::string dir = fresh_dir("prefixlib_hit");
    CompatHeader h = make_node_header();
    PrefixLibrary lib(dir, h);

    const uint64_t key = PrefixLibrary::key_for({1, 2, 3, 4});
    std::vector<uint8_t> blob = make_blob(h, 0x5A, 64);
    lib.store(key, blob);

    std::vector<uint8_t> out;
    ASSERT_TRUE(lib.try_load(key, out));
    EXPECT_EQ(out, blob);  // byte-identical round-trip
}

TEST(PrefixLibrary, AbsentKeyIsMissNotError) {
    const std::string dir = fresh_dir("prefixlib_miss");
    PrefixLibrary lib(dir, make_node_header());
    std::vector<uint8_t> out;
    EXPECT_FALSE(lib.try_load(PrefixLibrary::key_for({9, 9, 9}), out));
    EXPECT_FALSE(lib.try_load(0, out));  // non-cacheable key always misses
}

TEST(PrefixLibrary, PresentButMismatchedHeaderIsRefusedFailLoud) {
    const std::string dir = fresh_dir("prefixlib_refuse");
    // Producer stored a blob under a DIFFERENT build path (e.g. CPU vs Metal).
    CompatHeader producer = make_node_header();
    producer.build_path_tag ^= 0x1ull;
    const uint64_t key = PrefixLibrary::key_for({5, 6, 7});
    PrefixLibrary(dir, producer).store(key, make_blob(producer, 0x11, 32));

    // This node has the canonical identity; the stale blob must be REFUSED, not
    // returned and not silently treated as a miss.
    PrefixLibrary lib(dir, make_node_header());
    std::vector<uint8_t> out;
    try {
        lib.try_load(key, out);
        FAIL() << "expected a fail-loud refusal on build_path_tag mismatch";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("build_path_tag"), std::string::npos)
            << "refusal must name the mismatched field, got: " << e.what();
    }
}

TEST(PrefixLibrary, MismatchedWeightsHashIsRefused) {
    const std::string dir = fresh_dir("prefixlib_weights");
    CompatHeader producer = make_node_header();
    producer.weights_hash ^= 0xFFull;  // a different finetune / quant
    const uint64_t key = PrefixLibrary::key_for({3, 1, 4, 1, 5});
    PrefixLibrary(dir, producer).store(key, make_blob(producer, 0x22, 16));

    PrefixLibrary lib(dir, make_node_header());
    std::vector<uint8_t> out;
    EXPECT_THROW(lib.try_load(key, out), std::exception);
}

TEST(PrefixLibrary, BoundedEvictionDropsOldest) {
    const std::string dir = fresh_dir("prefixlib_evict");
    CompatHeader h = make_node_header();
    PrefixLibrary lib(dir, h, /*max_entries=*/3);

    // Insert 3 distinct prefixes; a small sleep makes mtimes strictly ordered so
    // "oldest" is unambiguous.
    std::vector<uint64_t> keys;
    for (int i = 0; i < 3; ++i) {
        uint64_t k = PrefixLibrary::key_for({100 + i, 200 + i});
        keys.push_back(k);
        lib.store(k, make_blob(h, static_cast<uint8_t>(i), 8));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // A 4th store must evict the oldest (keys[0]) to stay at the cap.
    uint64_t k4 = PrefixLibrary::key_for({999, 998});
    lib.store(k4, make_blob(h, 0x99, 8));

    std::vector<uint8_t> out;
    EXPECT_FALSE(lib.try_load(keys[0], out)) << "oldest entry should be evicted";
    EXPECT_TRUE(lib.try_load(keys[1], out));
    EXPECT_TRUE(lib.try_load(keys[2], out));
    EXPECT_TRUE(lib.try_load(k4, out));
}

TEST(PrefixLibrary, StoreIsAtomicNoLeftoverTemp) {
    const std::string dir = fresh_dir("prefixlib_atomic");
    CompatHeader h = make_node_header();
    PrefixLibrary lib(dir, h);
    lib.store(PrefixLibrary::key_for({42}), make_blob(h, 0x7, 16));

    int snap = 0, tmp = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string ext = e.path().extension().string();
        if (ext == ".snap") ++snap;
        if (ext == ".tmp") ++tmp;
    }
    EXPECT_EQ(snap, 1);
    EXPECT_EQ(tmp, 0) << "atomic publish must leave no .tmp behind";
}

TEST(PrefixLibrary, Key0IsNonCacheable) {
    const std::string dir = fresh_dir("prefixlib_key0");
    PrefixLibrary lib(dir, make_node_header());
    lib.store(0, make_blob(make_node_header(), 0x1, 8));  // no-op
    std::vector<uint8_t> out;
    EXPECT_FALSE(lib.try_load(0, out));
    // Nothing written to disk.
    int files = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        (void)e;
        ++files;
    }
    EXPECT_EQ(files, 0);
}
