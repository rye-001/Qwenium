// test_snapkv_eviction.cpp
// TDD tests for SnapKV eviction mask computation and cache compaction.
// Pure C++ — no ggml, no model.
//
// RED baseline: snapkv-eviction.h does not exist yet — tests fail to compile.
// GREEN target: all tests pass after implementation.

#include <gtest/gtest.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "../../src/kv-cache/snapkv-eviction.h"
#include "../../src/kv-cache/compressed_kv_store.h"
#include "../../src/kv-cache/simple-kv-cache.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

// ============================================================================
// Helpers
// ============================================================================

static std::vector<float> make_random_vector(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Make a scores tensor: [n_layers][n_heads][n_positions]
// with controllable "hot" positions per head.
static std::vector<std::vector<std::vector<float>>> make_scores(
    uint32_t n_layers, uint32_t n_heads, uint32_t n_positions,
    const std::vector<std::vector<uint32_t>>& hot_per_head_layer0 = {})
{
    std::vector<std::vector<std::vector<float>>> scores(
        n_layers, std::vector<std::vector<float>>(
            n_heads, std::vector<float>(n_positions, 0.01f)));

    // Make hot positions very high-scoring for layer 0
    if (!hot_per_head_layer0.empty()) {
        for (uint32_t h = 0; h < n_heads && h < hot_per_head_layer0.size(); ++h) {
            for (uint32_t pos : hot_per_head_layer0[h]) {
                if (pos < n_positions)
                    scores[0][h][pos] = 10.0f;
            }
        }
    }
    return scores;
}

// ============================================================================
// 1. Basic eviction mask computation
// ============================================================================

TEST(SnapKVEviction, BasicEviction) {
    // 1 layer, 4 heads, 100 positions, budget=10, window=5
    // Head 0 likes positions {0,1,2,3,4}, head 1 likes {5,6,7,8,9}
    // Head 2 likes {0,5,10,15,20}, head 3 likes {50,51,52,53,54}
    std::vector<std::vector<uint32_t>> hot = {
        {0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9},
        {0, 5, 10, 15, 20},
        {50, 51, 52, 53, 54}
    };
    auto scores = make_scores(1, 4, 100, hot);

    SnapKVResult result = compute_eviction_mask(scores, /*budget=*/10, /*obs_window=*/5);

    ASSERT_EQ(result.retained_positions.size(), 1u);  // 1 layer

    const auto& retained = result.retained_positions[0];

    // Retained must include the observation window (last 5: 95-99)
    for (uint32_t i = 95; i < 100; ++i) {
        EXPECT_NE(std::find(retained.begin(), retained.end(), i), retained.end())
            << "Window position " << i << " must be retained";
    }

    // Retained must be sorted
    EXPECT_TRUE(std::is_sorted(retained.begin(), retained.end()));

    // The hot positions from all heads should be in the union
    for (uint32_t pos : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 15u, 20u, 50u, 51u, 52u, 53u, 54u}) {
        EXPECT_NE(std::find(retained.begin(), retained.end(), pos), retained.end())
            << "Hot position " << pos << " should be in the union";
    }

    EXPECT_EQ(result.original_length, 100u);
}

// ============================================================================
// 2. Budget respected — each head keeps at most `budget` positions
// ============================================================================

TEST(SnapKVEviction, BudgetRespected) {
    // 1 layer, 2 heads, 200 positions, budget=20, window=10
    auto scores = make_scores(1, 2, 200);

    // Give every position unique scores so the top-20 is unambiguous per head
    for (uint32_t h = 0; h < 2; ++h) {
        for (uint32_t p = 0; p < 200; ++p) {
            scores[0][h][p] = static_cast<float>(p) + h * 0.5f;
        }
    }

    SnapKVResult result = compute_eviction_mask(scores, /*budget=*/20, /*obs_window=*/10);

    const auto& retained = result.retained_positions[0];

    // Union of top-20 per head + window(last 10) should be <= 20+20+10 = 50
    // (with overlap it could be less)
    EXPECT_LE(retained.size(), 50u);
    // Must have at least window + budget worth of positions
    EXPECT_GE(retained.size(), 10u);
}

// ============================================================================
// 3. Window always retained even with tiny budget
// ============================================================================

TEST(SnapKVEviction, WindowAlwaysRetained) {
    auto scores = make_scores(1, 2, 50);
    // budget=0 means only the observation window survives
    SnapKVResult result = compute_eviction_mask(scores, /*budget=*/0, /*obs_window=*/8);

    const auto& retained = result.retained_positions[0];
    EXPECT_EQ(retained.size(), 8u);
    for (uint32_t i = 42; i < 50; ++i) {
        EXPECT_NE(std::find(retained.begin(), retained.end(), i), retained.end());
    }
}

// ============================================================================
// 4. Budget >= seq_len means no eviction
// ============================================================================

TEST(SnapKVEviction, NoBudgetEviction) {
    auto scores = make_scores(1, 2, 30);
    // budget=100 > seq_len=30 → keep everything
    SnapKVResult result = compute_eviction_mask(scores, /*budget=*/100, /*obs_window=*/5);

    const auto& retained = result.retained_positions[0];
    EXPECT_EQ(retained.size(), 30u);
    for (uint32_t i = 0; i < 30; ++i) {
        EXPECT_EQ(retained[i], i);
    }
}

// ============================================================================
// 5. Multi-layer: each layer gets its own mask
// ============================================================================

TEST(SnapKVEviction, MultiLayer) {
    // 3 layers, 2 heads, 50 positions
    auto scores = make_scores(3, 2, 50);

    // Make layer 1 have different hot spots than layer 0
    scores[1][0][0] = 100.0f;
    scores[1][1][25] = 100.0f;
    scores[2][0][10] = 100.0f;
    scores[2][1][40] = 100.0f;

    SnapKVResult result = compute_eviction_mask(scores, /*budget=*/5, /*obs_window=*/5);

    ASSERT_EQ(result.retained_positions.size(), 3u);
    // Each layer may have different retained sets
    // but all must include window (45-49) and be sorted
    for (uint32_t l = 0; l < 3; ++l) {
        const auto& ret = result.retained_positions[l];
        EXPECT_TRUE(std::is_sorted(ret.begin(), ret.end()));
        for (uint32_t i = 45; i < 50; ++i) {
            EXPECT_NE(std::find(ret.begin(), ret.end(), i), ret.end());
        }
    }
}

// ============================================================================
// 6. No duplicate positions in output
// ============================================================================

TEST(SnapKVEviction, NoDuplicates) {
    // Multiple heads all like the same positions → union should have no dupes
    std::vector<std::vector<uint32_t>> hot = {
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4},
    };
    auto scores = make_scores(1, 3, 20, hot);

    SnapKVResult result = compute_eviction_mask(scores, /*budget=*/5, /*obs_window=*/3);
    const auto& retained = result.retained_positions[0];

    // Check uniqueness
    std::vector<uint32_t> sorted = retained;
    std::sort(sorted.begin(), sorted.end());
    auto last = std::unique(sorted.begin(), sorted.end());
    EXPECT_EQ(last, sorted.end()) << "No duplicate positions allowed";
}

// ============================================================================
// 7. CompressedKVStore::compact — data integrity after compaction
// ============================================================================

TEST(SnapKVCompact, CompressedStoreCompact) {
    const uint32_t n_embd = 128;
    const uint32_t head_dim = 128;
    const uint32_t n_layers = 2;
    const uint32_t n_ctx = 64;

    CompressedKVStore store(n_layers, 1, n_ctx, n_embd, n_embd, head_dim, 4);

    // Fill 10 tokens at positions 0..9
    std::vector<std::vector<float>> k_orig(10), v_orig(10);
    for (uint32_t t = 0; t < 10; ++t) {
        k_orig[t] = make_random_vector(n_embd, 100 + t);
        v_orig[t] = make_random_vector(n_embd, 200 + t);
        for (uint32_t l = 0; l < n_layers; ++l) {
            store.compress_token_k(l, 0, t, k_orig[t].data());
            store.compress_token_v(l, 0, t, v_orig[t].data());
        }
    }
    store.set_pos(0, 10);

    // Keep positions {1, 3, 7, 9}
    std::vector<uint32_t> retained = {1, 3, 7, 9};

    for (uint32_t l = 0; l < n_layers; ++l) {
        store.compact(l, 0, retained);
    }
    store.set_pos(0, retained.size());

    // Verify: new position i should hold data from old position retained[i]
    for (uint32_t i = 0; i < retained.size(); ++i) {
        uint32_t old_pos = retained[i];
        std::vector<float> k_out(n_embd), v_out(n_embd);

        for (uint32_t l = 0; l < n_layers; ++l) {
            store.decompress_token_k(l, 0, i, k_out.data());
            store.decompress_token_v(l, 0, i, v_out.data());

            // Compare against original (lossy compression, so use MSE threshold)
            std::vector<float> k_ref(n_embd), v_ref(n_embd);
            // Re-compress/decompress the original to get the reference
            // (since we can't compare against raw floats due to quantization loss)
            CompressedKVStore ref_store(1, 1, 1, n_embd, n_embd, head_dim, 4);
            ref_store.compress_token_k(0, 0, 0, k_orig[old_pos].data());
            ref_store.compress_token_v(0, 0, 0, v_orig[old_pos].data());
            ref_store.decompress_token_k(0, 0, 0, k_ref.data());
            ref_store.decompress_token_v(0, 0, 0, v_ref.data());

            float k_diff = 0, v_diff = 0;
            for (uint32_t j = 0; j < n_embd; ++j) {
                k_diff += (k_out[j] - k_ref[j]) * (k_out[j] - k_ref[j]);
                v_diff += (v_out[j] - v_ref[j]) * (v_out[j] - v_ref[j]);
            }
            EXPECT_LT(k_diff / n_embd, 1e-10f) << "K mismatch at new pos " << i << " layer " << l;
            EXPECT_LT(v_diff / n_embd, 1e-10f) << "V mismatch at new pos " << i << " layer " << l;
        }
    }
}

// ============================================================================
// 8. CompressedKVStore::compact — full retention (identity compaction)
// ============================================================================

TEST(SnapKVCompact, CompressedStoreCompactIdentity) {
    const uint32_t n_embd = 128;
    const uint32_t head_dim = 128;

    CompressedKVStore store(1, 1, 32, n_embd, n_embd, head_dim, 4);

    // Fill 5 tokens
    for (uint32_t t = 0; t < 5; ++t) {
        auto k = make_random_vector(n_embd, t);
        store.compress_token_k(0, 0, t, k.data());
    }
    store.set_pos(0, 5);

    // Retain all positions
    std::vector<uint32_t> retained = {0, 1, 2, 3, 4};

    // Snapshot before compaction
    std::vector<std::vector<float>> before(5);
    for (uint32_t t = 0; t < 5; ++t) {
        before[t].resize(n_embd);
        store.decompress_token_k(0, 0, t, before[t].data());
    }

    store.compact(0, 0, retained);

    // After identity compaction, data should be identical
    for (uint32_t t = 0; t < 5; ++t) {
        std::vector<float> after(n_embd);
        store.decompress_token_k(0, 0, t, after.data());
        for (uint32_t j = 0; j < n_embd; ++j) {
            EXPECT_EQ(before[t][j], after[j]) << "pos " << t << " elem " << j;
        }
    }
}

// ============================================================================
// 9. simple_kv_cache::compact — data integrity (CPU backend)
// ============================================================================

TEST(SnapKVCompact, SimpleKVCacheCompact) {
    // Use CPU backend (nullptr) for testability
    const uint32_t n_layers = 2;
    const uint32_t n_ctx = 32;
    const uint32_t n_embd = 64;

    simple_kv_cache cache(n_layers, n_ctx, 1, n_embd, n_embd,
                          GGML_TYPE_F32, GGML_TYPE_F32, nullptr);

    // Write known data into the cache using ggml_backend_tensor_set
    // Each position gets a unique pattern: all elements = pos * 1.0f + layer * 0.01f
    for (uint32_t l = 0; l < n_layers; ++l) {
        ggml_tensor* kt = cache.get_k_cache_tensor(l);
        ggml_tensor* vt = cache.get_v_cache_tensor(l);

        for (uint32_t pos = 0; pos < 10; ++pos) {
            std::vector<float> data(n_embd);
            for (uint32_t j = 0; j < n_embd; ++j) {
                data[j] = static_cast<float>(pos) + l * 0.01f + j * 0.0001f;
            }
            size_t offset = pos * n_embd * sizeof(float);
            ggml_backend_tensor_set(kt, data.data(), offset, n_embd * sizeof(float));
            ggml_backend_tensor_set(vt, data.data(), offset, n_embd * sizeof(float));
        }
    }
    cache.set_pos(10, 0);

    // Compact: keep positions {2, 5, 8}
    std::vector<uint32_t> retained = {2, 5, 8};
    cache.compact(0, retained);

    EXPECT_EQ(cache.get_pos(0), 3u);

    // Verify data at new positions
    for (uint32_t l = 0; l < n_layers; ++l) {
        ggml_tensor* kt = cache.get_k_cache_tensor(l);
        ggml_tensor* vt = cache.get_v_cache_tensor(l);

        for (uint32_t i = 0; i < retained.size(); ++i) {
            uint32_t old_pos = retained[i];
            std::vector<float> k_data(n_embd), v_data(n_embd);
            size_t offset = i * n_embd * sizeof(float);
            ggml_backend_tensor_get(kt, k_data.data(), offset, n_embd * sizeof(float));
            ggml_backend_tensor_get(vt, v_data.data(), offset, n_embd * sizeof(float));

            for (uint32_t j = 0; j < n_embd; ++j) {
                float expected = static_cast<float>(old_pos) + l * 0.01f + j * 0.0001f;
                EXPECT_FLOAT_EQ(k_data[j], expected)
                    << "K mismatch at new pos " << i << " layer " << l << " elem " << j;
                EXPECT_FLOAT_EQ(v_data[j], expected)
                    << "V mismatch at new pos " << i << " layer " << l << " elem " << j;
            }
        }
    }
}

// ============================================================================
// 10. simple_kv_cache::compact — identity (keep all)
// ============================================================================

TEST(SnapKVCompact, SimpleKVCacheCompactIdentity) {
    const uint32_t n_layers = 1;
    const uint32_t n_ctx = 16;
    const uint32_t n_embd = 32;

    simple_kv_cache cache(n_layers, n_ctx, 1, n_embd, n_embd,
                          GGML_TYPE_F32, GGML_TYPE_F32, nullptr);

    for (uint32_t pos = 0; pos < 5; ++pos) {
        std::vector<float> data(n_embd);
        for (uint32_t j = 0; j < n_embd; ++j)
            data[j] = static_cast<float>(pos * 100 + j);
        size_t offset = pos * n_embd * sizeof(float);
        ggml_backend_tensor_set(cache.get_k_cache_tensor(0), data.data(), offset, n_embd * sizeof(float));
    }
    cache.set_pos(5, 0);

    std::vector<uint32_t> retained = {0, 1, 2, 3, 4};
    cache.compact(0, retained);
    EXPECT_EQ(cache.get_pos(0), 5u);

    // Data should be unchanged
    for (uint32_t pos = 0; pos < 5; ++pos) {
        std::vector<float> data(n_embd);
        size_t offset = pos * n_embd * sizeof(float);
        ggml_backend_tensor_get(cache.get_k_cache_tensor(0), data.data(), offset, n_embd * sizeof(float));
        for (uint32_t j = 0; j < n_embd; ++j) {
            EXPECT_FLOAT_EQ(data[j], static_cast<float>(pos * 100 + j));
        }
    }
}
