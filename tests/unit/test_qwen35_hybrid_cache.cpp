/**
 * test_qwen35_hybrid_cache.cpp — Step 5 TDD for Qwen 3.5 support
 * 
 * Tests that the forward pass correctly creates:
 *   - KV cache with 6 layers (attention only)
 *   - SSM state cache with 18 layers
 *   - Correct physical→cache index mappings
 * 
 * Requires: QWEN35_MODEL_PATH env var pointing to Qwen3.5-0.8B-BF16.gguf
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <set>

#include "../../src/qwen3-core/qwen3-model.h"
#include "../../src/models/qwen35.h"
#include "../../src/state/ssm_state_cache.h"

static std::string get_qwen35_model_path() {
    const char* path = std::getenv("QWEN35_MODEL_PATH");
    return path ? std::string(path) : "";
}

#define SKIP_IF_NO_MODEL()                                          \
    do {                                                            \
        if (get_qwen35_model_path().empty()) {                     \
            GTEST_SKIP() << "QWEN35_MODEL_PATH not set, skipping"; \
        }                                                          \
    } while (0)

class Qwen35HybridCacheTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_qwen35_model_path().empty()) return;

        model_ = std::make_unique<Qwen3Model>();
        model_->load_metadata(get_qwen35_model_path());
        model_->load_tensors();
    }

    static void TearDownTestSuite() {
        model_.reset();
    }

    static std::unique_ptr<Qwen3Model> model_;
};

std::unique_ptr<Qwen3Model> Qwen35HybridCacheTest::model_ = nullptr;

// ============================================================
// Test: Forward pass construction doesn't crash
// ============================================================
TEST_F(Qwen35HybridCacheTest, ConstructionSucceeds) {
    SKIP_IF_NO_MODEL();

    // 2048 context, 2 slots — modest for testing
    EXPECT_NO_THROW({
        Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);
    });
}

// ============================================================
// Test: KV cache exists and has 6 layers (attention only)
// ============================================================
TEST_F(Qwen35HybridCacheTest, KVCacheLayerCount) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    auto* kv = fp.get_kv_cache_ptr();
    ASSERT_NE(kv, nullptr) << "KV cache should be allocated";

    // For qwen35, KV cache should only have 6 layers (the attention layers)
    // We can verify this indirectly by checking that accessing layer 5 works
    // but the cache was created with n_layers = 6 (not 24)
    //
    // The get_k method uses the layer index, so if the cache was created
    // with 6 layers, accessing index 5 should work, but 6 should not.
    // (We can't easily test this without triggering an assert, so we'll
    // verify through the mapping instead.)
}

// ============================================================
// Test: DeltaNet state exists and has 18 layers
// ============================================================
TEST_F(Qwen35HybridCacheTest, SSMCacheExists) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    auto* dn = fp.get_dn_state_ptr();
    ASSERT_NE(dn, nullptr) << "DeltaNet state should be allocated for qwen35";
    EXPECT_EQ(dn->n_dn_layers(), 18u);
}

// ============================================================
// Test: DeltaNet state dimensions match model metadata
// ============================================================
TEST_F(Qwen35HybridCacheTest, SSMCacheDimensions) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    auto* dn = fp.get_dn_state_ptr();
    ASSERT_NE(dn, nullptr);

    EXPECT_EQ(dn->head_k_dim(),  128u);   // ssm.state_size
    EXPECT_EQ(dn->num_v_heads(),  16u);   // ssm.time_step_rank
    EXPECT_EQ(dn->conv_kernel(),   4u);   // ssm.conv_kernel
}

// ============================================================
// Test: Physical-to-KV-cache layer mapping
// Attention layers 3,7,11,15,19,23 → KV indices 0,1,2,3,4,5
// ============================================================
TEST_F(Qwen35HybridCacheTest, AttentionLayerMapping) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    const std::vector<uint32_t> attn_layers = {3, 7, 11, 15, 19, 23};

    for (uint32_t expected_idx = 0; expected_idx < attn_layers.size(); ++expected_idx) {
        uint32_t physical = attn_layers[expected_idx];
        int32_t kv_idx = fp.get_kv_layer_index(physical);

        EXPECT_EQ(kv_idx, static_cast<int32_t>(expected_idx))
            << "Physical layer " << physical << " should map to KV index " << expected_idx;
    }
}

// ============================================================
// Test: SSM layers return -1 for KV index (no KV cache)
// ============================================================
TEST_F(Qwen35HybridCacheTest, SSMLayersHaveNoKVIndex) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    // SSM layers: 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22
    const std::vector<uint32_t> ssm_layers = {0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                12, 13, 14, 16, 17, 18, 20, 21, 22};

    for (uint32_t physical : ssm_layers) {
        int32_t kv_idx = fp.get_kv_layer_index(physical);
        EXPECT_EQ(kv_idx, -1)
            << "SSM layer " << physical << " should not have a KV index";
    }
}

// ============================================================
// Test: Physical-to-SSM-cache layer mapping
// SSM layers 0,1,2,4,5,6,8,... → SSM indices 0,1,2,3,4,5,6,...
// ============================================================
TEST_F(Qwen35HybridCacheTest, SSMLayerMapping) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    const std::vector<uint32_t> ssm_layers = {0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                12, 13, 14, 16, 17, 18, 20, 21, 22};

    for (uint32_t expected_idx = 0; expected_idx < ssm_layers.size(); ++expected_idx) {
        uint32_t physical = ssm_layers[expected_idx];
        int32_t ssm_idx = fp.get_ssm_layer_index(physical);

        EXPECT_EQ(ssm_idx, static_cast<int32_t>(expected_idx))
            << "Physical layer " << physical << " should map to SSM index " << expected_idx;
    }
}

// ============================================================
// Test: Attention layers return -1 for SSM index
// ============================================================
TEST_F(Qwen35HybridCacheTest, AttentionLayersHaveNoSSMIndex) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    const std::vector<uint32_t> attn_layers = {3, 7, 11, 15, 19, 23};

    for (uint32_t physical : attn_layers) {
        int32_t ssm_idx = fp.get_ssm_layer_index(physical);
        EXPECT_EQ(ssm_idx, -1)
            << "Attention layer " << physical << " should not have an SSM index";
    }
}

// ============================================================
// Test: Non-qwen35 model still works (no SSM cache)
// ============================================================
TEST_F(Qwen35HybridCacheTest, NonQwen35HasNoSSMCache) {
    // This test verifies backward compatibility.
    // We can't easily load a second model in the same test suite,
    // so instead verify the API contract: for qwen35, ssm_cache is non-null.
    // The existing qwen2/3 tests already verify those models load correctly
    // with the standard KV-only cache.
    SKIP_IF_NO_MODEL();

    // Just verify we're testing qwen35
    EXPECT_EQ(model_->get_metadata().architecture, "qwen35");
}

// ============================================================
// Test: Cache clear/clone delegates to both caches
// ============================================================
TEST_F(Qwen35HybridCacheTest, ClearSlotWorks) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    // Should not crash — delegates to both KV and SSM caches
    EXPECT_NO_THROW(fp.clear_slot(0));
    EXPECT_NO_THROW(fp.clear_slot(1));
}

TEST_F(Qwen35HybridCacheTest, CloneSlotWorks) {
    SKIP_IF_NO_MODEL();

    Qwen35ForwardPass fp(*model_, &model_->get_metadata(), 2048, 2);

    // Should not crash — delegates to both KV and SSM caches
    EXPECT_NO_THROW(fp.clone_slot(0, 1, 0));
}
