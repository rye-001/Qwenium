// test_decode_policy.cpp — the run-time policy of a forward pass.
//
// Extracting DecodePolicy out of ForwardPassBase made these testable without a
// model, a backend, or a recipe. Two things are worth pinning: the defaults are
// the byte-reproducible configuration (the receipts claims in architecture.md
// §11 are made about a pass in that state), and decode_kv_len's bucketing —
// whose cap is the part with a real edge case.

#include <gtest/gtest.h>

#include "../../src/models/decode_policy.h"

TEST(DecodePolicy, DefaultsAreTheByteReproduciblePath) {
    DecodePolicy p;
    EXPECT_TRUE(p.slice_prefill_head) << "prefill head slice is the default";
    EXPECT_FALSE(p.output_hidden);
    EXPECT_TRUE(p.attention_taps.empty()) << "an empty tap set marks no node";
    EXPECT_EQ(p.kv_write_mode, DecodePolicy::KvWriteMode::Cpy);
    EXPECT_EQ(p.decode_kv_bucket, 0u);
    EXPECT_TRUE(p.is_default_byte_reproducible());
}

// Each opt-in seam, on its own, takes the pass out of the default state.
TEST(DecodePolicy, AnyArmedSeamLeavesTheDefaultState) {
    { DecodePolicy p; p.output_hidden = true;
      EXPECT_FALSE(p.is_default_byte_reproducible()); }
    { DecodePolicy p; p.attention_taps = {3};
      EXPECT_FALSE(p.is_default_byte_reproducible()); }
    { DecodePolicy p; p.kv_write_mode = DecodePolicy::KvWriteMode::SetRows;
      EXPECT_FALSE(p.is_default_byte_reproducible()); }
    { DecodePolicy p; p.decode_kv_bucket = 256;
      EXPECT_FALSE(p.is_default_byte_reproducible()); }
    { DecodePolicy p; p.slice_prefill_head = false;
      EXPECT_FALSE(p.is_default_byte_reproducible()); }
}

// Bucket 0 is exact sizing — today's decode path, and the reason the default is
// byte-reproducible.
TEST(DecodePolicy, BucketZeroIsExactSizing) {
    DecodePolicy p;
    for (uint32_t n : {1u, 7u, 255u, 4096u})
        EXPECT_EQ(p.decode_kv_len(n, 8192), n);
}

TEST(DecodePolicy, BucketRoundsUpToTheNextMultiple) {
    DecodePolicy p;
    p.decode_kv_bucket = 256;
    EXPECT_EQ(p.decode_kv_len(1,    8192), 256u);
    EXPECT_EQ(p.decode_kv_len(255,  8192), 256u);
    EXPECT_EQ(p.decode_kv_len(256,  8192), 256u) << "an exact multiple must not grow";
    EXPECT_EQ(p.decode_kv_len(257,  8192), 512u);
    EXPECT_EQ(p.decode_kv_len(1000, 8192), 1024u);
}

// The cap is the edge case: near the end of the context the rounded-up width
// would run past the cache and size the graph to read rows that do not exist.
TEST(DecodePolicy, BucketIsCappedAtNCtxMax) {
    DecodePolicy p;
    p.decode_kv_bucket = 256;
    EXPECT_EQ(p.decode_kv_len(4000, 4096), 4096u) << "4096 would round to 4096, capped";
    EXPECT_EQ(p.decode_kv_len(4096, 4096), 4096u);
    EXPECT_EQ(p.decode_kv_len(4095, 4096), 4096u) << "rounds to 4096, at the cap";
    // A width already at the cache limit must never be widened past it.
    EXPECT_LE(p.decode_kv_len(4090, 4096), 4096u);
}

// Whatever the bucket, the result must cover the requested width — a narrower
// graph would drop KV columns the step needs.
TEST(DecodePolicy, ResultAlwaysCoversTheRequestedWidth) {
    for (uint32_t bucket : {0u, 64u, 256u, 1024u}) {
        DecodePolicy p;
        p.decode_kv_bucket = bucket;
        for (uint32_t n = 1; n <= 4096; n += 37) {
            const uint32_t got = p.decode_kv_len(n, 4096);
            EXPECT_GE(got, n) << "bucket " << bucket << ", n_kv " << n
                              << " — the graph must not be narrower than the step";
            EXPECT_LE(got, 4096u) << "bucket " << bucket << ", n_kv " << n;
        }
    }
}
