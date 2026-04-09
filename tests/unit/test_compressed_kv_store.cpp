// test_compressed_kv_store.cpp
// TDD tests for the compressed KV backing store.
// No ggml, no model — pure C++ data structure tests.

#include "compressed_kv_store.h"
#include "turboquant.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

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

static float mse(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return static_cast<float>(sum / n);
}

// ============================================================================
// 1. Construction
// ============================================================================

TEST(Store, ConstructionAndConfig) {
    // 2 layers, 4 slots, ctx=128, head_dim=64, n_heads=2 → embd=128
    CompressedKVStore store(2, 4, 128, 128, 128, 64, 4);

    EXPECT_EQ(store.n_layers(), 2u);
    EXPECT_EQ(store.n_slots(), 4u);
    EXPECT_EQ(store.n_ctx_max(), 128u);
    EXPECT_EQ(store.bits(), 4);
    EXPECT_GT(store.compressed_bytes_per_token_k(), 0u);
}

// ============================================================================
// 2. Store and retrieve a single token
// ============================================================================

TEST(Store, StoreAndRetrieveSingleToken) {
    const int n_embd = 128;
    const int head_dim = 128;
    CompressedKVStore store(1, 1, 64, n_embd, n_embd, head_dim, 4);

    auto k_orig = make_random_vector(n_embd, 1);
    auto v_orig = make_random_vector(n_embd, 2);

    // Store token at position 0
    store.compress_token_k(0, 0, 0, k_orig.data());
    store.compress_token_v(0, 0, 0, v_orig.data());

    // Retrieve
    std::vector<float> k_out(n_embd), v_out(n_embd);
    store.decompress_token_k(0, 0, 0, k_out.data());
    store.decompress_token_v(0, 0, 0, v_out.data());

    EXPECT_LT(mse(k_orig.data(), k_out.data(), n_embd), 0.05f);
    EXPECT_LT(mse(v_orig.data(), v_out.data(), n_embd), 0.05f);
}

// ============================================================================
// 3. Store a sequence of tokens
// ============================================================================

TEST(Store, StoreSequence) {
    const int n_embd = 128;
    const int head_dim = 128;
    const int seq_len = 16;
    CompressedKVStore store(1, 1, 64, n_embd, n_embd, head_dim, 4);

    // Store 16 tokens
    std::vector<std::vector<float>> originals(seq_len);
    for (int t = 0; t < seq_len; ++t) {
        originals[t] = make_random_vector(n_embd, t + 100);
        store.compress_token_k(0, 0, t, originals[t].data());
    }

    // Retrieve all and verify
    std::vector<float> out(n_embd);
    for (int t = 0; t < seq_len; ++t) {
        store.decompress_token_k(0, 0, t, out.data());
        EXPECT_LT(mse(originals[t].data(), out.data(), n_embd), 0.05f)
            << "Token " << t << " has high MSE";
    }
}

// ============================================================================
// 4. Multi-layer, multi-slot
// ============================================================================

TEST(Store, MultiLayerMultiSlot) {
    const int n_layers = 4;
    const int n_slots = 3;
    const int n_embd = 128;
    const int head_dim = 128;
    CompressedKVStore store(n_layers, n_slots, 32, n_embd, n_embd, head_dim, 4);

    // Store one token per layer per slot with different data
    for (int l = 0; l < n_layers; ++l) {
        for (int s = 0; s < n_slots; ++s) {
            auto data = make_random_vector(n_embd, l * 100 + s);
            store.compress_token_k(l, s, 0, data.data());
        }
    }

    // Verify each is independently correct
    for (int l = 0; l < n_layers; ++l) {
        for (int s = 0; s < n_slots; ++s) {
            auto expected = make_random_vector(n_embd, l * 100 + s);
            std::vector<float> out(n_embd);
            store.decompress_token_k(l, s, 0, out.data());
            EXPECT_LT(mse(expected.data(), out.data(), n_embd), 0.05f)
                << "Layer " << l << ", Slot " << s;
        }
    }
}

// ============================================================================
// 5. Clone slot
// ============================================================================

TEST(Store, CloneSlot) {
    const int n_embd = 128;
    const int head_dim = 128;
    const int n_tokens = 8;
    CompressedKVStore store(2, 4, 64, n_embd, n_embd, head_dim, 4);

    // Fill slot 0 with data
    std::vector<std::vector<float>> originals(n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        originals[t] = make_random_vector(n_embd, t + 200);
        store.compress_token_k(0, 0, t, originals[t].data());
        store.compress_token_v(0, 0, t, originals[t].data());
        store.compress_token_k(1, 0, t, originals[t].data());
        store.compress_token_v(1, 0, t, originals[t].data());
    }

    // Clone slot 0 → slot 2
    store.clone_slot(0, 2, n_tokens);

    // Verify slot 2 has same data
    std::vector<float> out(n_embd);
    for (int t = 0; t < n_tokens; ++t) {
        store.decompress_token_k(0, 2, t, out.data());
        EXPECT_LT(mse(originals[t].data(), out.data(), n_embd), 0.05f)
            << "Clone K mismatch at token " << t;
        store.decompress_token_v(0, 2, t, out.data());
        EXPECT_LT(mse(originals[t].data(), out.data(), n_embd), 0.05f)
            << "Clone V mismatch at token " << t;
    }
}

// ============================================================================
// 6. Clear slot
// ============================================================================

TEST(Store, ClearSlot) {
    const int n_embd = 128;
    const int head_dim = 128;
    CompressedKVStore store(1, 2, 32, n_embd, n_embd, head_dim, 4);

    auto data = make_random_vector(n_embd);
    store.compress_token_k(0, 0, 0, data.data());

    // After clear, reading should give zeros (or at least no crash)
    store.clear_slot(0);

    std::vector<float> out(n_embd);
    store.decompress_token_k(0, 0, 0, out.data());

    // Cleared data should be near zero
    float sum = 0;
    for (float x : out) sum += x * x;
    EXPECT_LT(sum, 1e-6f);
}

// ============================================================================
// 7. Memory accounting
// ============================================================================

TEST(Store, MemoryAccounting) {
    const int n_embd = 1024;  // 8 heads × 128
    const int head_dim = 128;
    const int n_ctx = 2048;
    const int n_layers = 28;
    const int n_slots = 10;

    CompressedKVStore store(n_layers, n_slots, n_ctx, n_embd, n_embd, head_dim, 4);

    size_t f32_bytes = static_cast<size_t>(n_layers) * n_slots * n_ctx * n_embd * 4 * 2; // K+V
    size_t tq_bytes = store.total_compressed_bytes();

    // Compressed should be significantly smaller than F32
    float ratio = static_cast<float>(f32_bytes) / tq_bytes;
    EXPECT_GT(ratio, 5.0f) << "Expected >5x compression, got " << ratio << "x";
    EXPECT_LT(ratio, 10.0f) << "Ratio suspiciously high: " << ratio << "x";
}

// ============================================================================
// 8. Multi-head compress (non-power-of-2 n_embd, but head_dim is power of 2)
// ============================================================================

TEST(Store, MultiHeadNonPow2Embd) {
    // 6 heads × 64 = 384 (not a power of 2, but head_dim=64 is)
    const int n_heads = 6;
    const int head_dim = 64;
    const int n_embd = n_heads * head_dim;

    CompressedKVStore store(1, 1, 32, n_embd, n_embd, head_dim, 4);

    auto orig = make_random_vector(n_embd, 77);
    store.compress_token_k(0, 0, 0, orig.data());

    std::vector<float> out(n_embd);
    store.decompress_token_k(0, 0, 0, out.data());

    EXPECT_LT(mse(orig.data(), out.data(), n_embd), 0.05f);
}

// ============================================================================
// 9. Different bit widths
// ============================================================================

TEST(Store, TwoBitStore) {
    const int n_embd = 128;
    const int head_dim = 128;
    CompressedKVStore store(1, 1, 32, n_embd, n_embd, head_dim, 2);

    auto orig = make_random_vector(n_embd);
    store.compress_token_k(0, 0, 0, orig.data());

    std::vector<float> out(n_embd);
    store.decompress_token_k(0, 0, 0, out.data());

    // 2-bit is coarser
    EXPECT_LT(mse(orig.data(), out.data(), n_embd), 0.25f);
}

// ============================================================================
// 10. Position tracking — replaces simple_kv_cache position management when
//     TQ is enabled. The store tracks one position counter per slot.
// ============================================================================

TEST(PositionTracking, InitiallyZero) {
    CompressedKVStore store(2, 4, 128, 128, 128, 64, 4);

    for (uint32_t s = 0; s < 4; ++s)
        EXPECT_EQ(store.get_pos(s), 0u) << "Slot " << s << " should start at 0";
}

TEST(PositionTracking, Advance) {
    CompressedKVStore store(1, 2, 64, 128, 128, 128, 4);

    store.advance(0, 5);
    EXPECT_EQ(store.get_pos(0), 5u);

    store.advance(0, 3);
    EXPECT_EQ(store.get_pos(0), 8u);

    // Slot 1 is independent
    EXPECT_EQ(store.get_pos(1), 0u);
    store.advance(1, 10);
    EXPECT_EQ(store.get_pos(1), 10u);
    EXPECT_EQ(store.get_pos(0), 8u);  // slot 0 unchanged
}

TEST(PositionTracking, SetPos) {
    CompressedKVStore store(1, 3, 64, 128, 128, 128, 4);

    store.set_pos(1, 42);
    EXPECT_EQ(store.get_pos(1), 42u);
    EXPECT_EQ(store.get_pos(0), 0u);  // others untouched
    EXPECT_EQ(store.get_pos(2), 0u);

    store.set_pos(1, 7);  // overwrite
    EXPECT_EQ(store.get_pos(1), 7u);
}

TEST(PositionTracking, ClearSlotResetsPosition) {
    CompressedKVStore store(1, 2, 64, 128, 128, 128, 4);

    store.advance(0, 20);
    EXPECT_EQ(store.get_pos(0), 20u);

    store.clear_slot(0);
    EXPECT_EQ(store.get_pos(0), 0u) << "clear_slot must reset position to 0";
    EXPECT_EQ(store.get_pos(1), 0u);  // other slot untouched
}

TEST(PositionTracking, CloneSlotCopiesPosition) {
    CompressedKVStore store(1, 4, 64, 128, 128, 128, 4);

    store.advance(0, 15);
    // Clone slot 0 → slot 2  (n_tokens = 15)
    store.clone_slot(0, 2, 15);

    EXPECT_EQ(store.get_pos(2), 15u) << "clone_slot must copy source position";
    EXPECT_EQ(store.get_pos(1), 0u);  // untouched slots unchanged
    EXPECT_EQ(store.get_pos(3), 0u);
}

TEST(PositionTracking, AdvanceAndRetrieveMatchesSequence) {
    // Ensures position tracks correctly across a typical prefill + decode loop:
    // prefill 8 tokens, then advance 1 token at a time for 4 decode steps.
    CompressedKVStore store(1, 1, 64, 128, 128, 128, 4);

    store.advance(0, 8);
    EXPECT_EQ(store.get_pos(0), 8u);

    for (uint32_t i = 0; i < 4; ++i) {
        store.advance(0, 1);
        EXPECT_EQ(store.get_pos(0), 8u + i + 1);
    }
    EXPECT_EQ(store.get_pos(0), 12u);
}
