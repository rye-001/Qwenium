/**
 * test_qwen35_tensor_loading.cpp — Step 2 TDD for Qwen 3.5 support
 * 
 * Tests that all 320 tensors load correctly and that per-block
 * tensor pointers are assigned based on layer type (SSM vs attention).
 * 
 * Requires: QWEN35_MODEL_PATH env var pointing to Qwen3.5-0.8B-BF16.gguf
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <set>

#include "../../src/core/model.h"
#include "../../src/loader/gguf_loader.h"

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

// Shared fixture that loads the model once for all tests in this suite
class Qwen35TensorLoadingTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_qwen35_model_path().empty()) return;

        model_ = std::make_unique<Model>();
        model_->load_metadata(get_qwen35_model_path());
        model_->load_tensors();
    }

    static void TearDownTestSuite() {
        model_.reset();
    }

    static std::unique_ptr<Model> model_;
};

std::unique_ptr<Model> Qwen35TensorLoadingTest::model_ = nullptr;


// ============================================================
// Test: Model loads without crashing
// ============================================================
TEST_F(Qwen35TensorLoadingTest, LoadSucceeds) {
    SKIP_IF_NO_MODEL();
    ASSERT_NE(model_, nullptr) << "Model failed to load";
}

// ============================================================
// Test: Global tensor pointers
// ============================================================
TEST_F(Qwen35TensorLoadingTest, GlobalTensors) {
    SKIP_IF_NO_MODEL();

    // token_embd.weight must be present
    EXPECT_NE(model_->get_token_embedding_weight(), nullptr);

    // output_norm.weight must be present
    EXPECT_NE(model_->get_output_norm_weight(), nullptr);

    // output.weight must be nullptr (weight tying)
    EXPECT_EQ(model_->get_output_weight(), nullptr)
        << "qwen35 has no output.weight — LM head reuses token_embd.weight";
}

// ============================================================
// Test: token_embd.weight has correct shape [1024, 248320]
// ============================================================
TEST_F(Qwen35TensorLoadingTest, EmbeddingShape) {
    SKIP_IF_NO_MODEL();

    auto* embd = model_->get_token_embedding_weight();
    ASSERT_NE(embd, nullptr);

    // ggml shape: ne[0]=1024, ne[1]=248320
    EXPECT_EQ(embd->ne[0], 1024);
    EXPECT_EQ(embd->ne[1], 248320);
}

// ============================================================
// Test: SSM layer tensor pointers (layer 0)
// ============================================================
TEST_F(Qwen35TensorLoadingTest, SSMLayerTensorsNonNull) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    const uint32_t fai = meta.raw_kv.get_uint32("qwen35.full_attention_interval");
    const bool is_full_0 = (fai > 0) && ((0 % fai) == (fai - 1));
    ASSERT_FALSE(is_full_0);

    const auto& blk = model_->get_block(0);

    // Shared tensors
    EXPECT_NE(blk.attn_norm_weight, nullptr) << "blk.0.attn_norm.weight";
    EXPECT_NE(blk.ffn_norm_weight, nullptr)  << "blk.0 FFN norm (post_attention_norm)";
    EXPECT_NE(blk.ffn_gate_weight, nullptr)  << "blk.0.ffn_gate.weight";
    EXPECT_NE(blk.ffn_up_weight, nullptr)    << "blk.0.ffn_up.weight";
    EXPECT_NE(blk.ffn_down_weight, nullptr)  << "blk.0.ffn_down.weight";

    // SSM-specific tensors
    EXPECT_NE(blk.ssm_a, nullptr)              << "blk.0.ssm_a";
    EXPECT_NE(blk.ssm_conv1d_weight, nullptr)  << "blk.0.ssm_conv1d.weight";
    EXPECT_NE(blk.ssm_dt_bias, nullptr)        << "blk.0.ssm_dt.bias";
    EXPECT_NE(blk.ssm_alpha_weight, nullptr)   << "blk.0.ssm_alpha.weight";
    EXPECT_NE(blk.ssm_beta_weight, nullptr)    << "blk.0.ssm_beta.weight";
    EXPECT_NE(blk.attn_qkv_weight, nullptr)    << "blk.0.attn_qkv.weight";
    EXPECT_NE(blk.attn_gate_weight, nullptr)   << "blk.0.attn_gate.weight";
    EXPECT_NE(blk.ssm_norm_weight, nullptr)    << "blk.0.ssm_norm.weight";
    EXPECT_NE(blk.ssm_out_weight, nullptr)     << "blk.0.ssm_out.weight";

    // Attention-only tensors should be null on SSM layers
    EXPECT_EQ(blk.attn_q_weight, nullptr)      << "SSM layer should not have attn_q";
    EXPECT_EQ(blk.attn_k_weight, nullptr)      << "SSM layer should not have attn_k";
    EXPECT_EQ(blk.attn_v_weight, nullptr)      << "SSM layer should not have attn_v";
    EXPECT_EQ(blk.attn_output_weight, nullptr) << "SSM layer should not have attn_output";
    EXPECT_EQ(blk.attn_q_norm_weight, nullptr) << "SSM layer should not have attn_q_norm";
    EXPECT_EQ(blk.attn_k_norm_weight, nullptr) << "SSM layer should not have attn_k_norm";
}

// ============================================================
// Test: Full attention layer tensor pointers (layer 3)
// ============================================================
TEST_F(Qwen35TensorLoadingTest, AttentionLayerTensorsNonNull) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    const uint32_t fai2 = meta.raw_kv.get_uint32("qwen35.full_attention_interval");
    const bool is_full_3 = (fai2 > 0) && ((3 % fai2) == (fai2 - 1));
    ASSERT_TRUE(is_full_3);

    const auto& blk = model_->get_block(3);

    // Shared tensors
    EXPECT_NE(blk.attn_norm_weight, nullptr) << "blk.3.attn_norm.weight";
    EXPECT_NE(blk.ffn_norm_weight, nullptr)  << "blk.3 FFN norm (post_attention_norm)";
    EXPECT_NE(blk.ffn_gate_weight, nullptr)  << "blk.3.ffn_gate.weight";
    EXPECT_NE(blk.ffn_up_weight, nullptr)    << "blk.3.ffn_up.weight";
    EXPECT_NE(blk.ffn_down_weight, nullptr)  << "blk.3.ffn_down.weight";

    // Attention-specific tensors
    EXPECT_NE(blk.attn_q_weight, nullptr)      << "blk.3.attn_q.weight";
    EXPECT_NE(blk.attn_k_weight, nullptr)      << "blk.3.attn_k.weight";
    EXPECT_NE(blk.attn_v_weight, nullptr)      << "blk.3.attn_v.weight";
    EXPECT_NE(blk.attn_output_weight, nullptr) << "blk.3.attn_output.weight";
    EXPECT_NE(blk.attn_q_norm_weight, nullptr) << "blk.3.attn_q_norm.weight";
    EXPECT_NE(blk.attn_k_norm_weight, nullptr) << "blk.3.attn_k_norm.weight";

    // SSM-specific tensors should be null on attention layers
    EXPECT_EQ(blk.ssm_a, nullptr)              << "Attn layer should not have ssm_a";
    EXPECT_EQ(blk.ssm_conv1d_weight, nullptr)  << "Attn layer should not have ssm_conv1d";
    EXPECT_EQ(blk.ssm_dt_bias, nullptr)        << "Attn layer should not have ssm_dt";
    EXPECT_EQ(blk.ssm_alpha_weight, nullptr)   << "Attn layer should not have ssm_alpha";
    EXPECT_EQ(blk.ssm_beta_weight, nullptr)    << "Attn layer should not have ssm_beta";
    EXPECT_EQ(blk.attn_qkv_weight, nullptr)    << "Attn layer should not have attn_qkv";
    EXPECT_EQ(blk.attn_gate_weight, nullptr)   << "Attn layer should not have attn_gate";
    EXPECT_EQ(blk.ssm_norm_weight, nullptr)    << "Attn layer should not have ssm_norm";
    EXPECT_EQ(blk.ssm_out_weight, nullptr)     << "Attn layer should not have ssm_out";
}

// ============================================================
// Test: ALL 24 layers have correct pointer pattern
// ============================================================
TEST_F(Qwen35TensorLoadingTest, AllLayersCorrectPattern) {
    SKIP_IF_NO_MODEL();

    const auto& meta = model_->get_metadata();
    const std::set<uint32_t> attn_layers = {3, 7, 11, 15, 19, 23};

    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const auto& blk = model_->get_block(i);
        bool is_attn = attn_layers.count(i) > 0;

        // Shared: always non-null
        EXPECT_NE(blk.attn_norm_weight, nullptr) << "Layer " << i;
        EXPECT_NE(blk.ffn_norm_weight, nullptr)  << "Layer " << i;
        EXPECT_NE(blk.ffn_gate_weight, nullptr)  << "Layer " << i;
        EXPECT_NE(blk.ffn_up_weight, nullptr)    << "Layer " << i;
        EXPECT_NE(blk.ffn_down_weight, nullptr)  << "Layer " << i;

        if (is_attn) {
            // Attention pointers non-null
            EXPECT_NE(blk.attn_q_weight, nullptr)      << "Attn layer " << i;
            EXPECT_NE(blk.attn_k_weight, nullptr)      << "Attn layer " << i;
            EXPECT_NE(blk.attn_v_weight, nullptr)      << "Attn layer " << i;
            EXPECT_NE(blk.attn_output_weight, nullptr) << "Attn layer " << i;
            EXPECT_NE(blk.attn_q_norm_weight, nullptr) << "Attn layer " << i;
            EXPECT_NE(blk.attn_k_norm_weight, nullptr) << "Attn layer " << i;
            // SSM pointers null
            EXPECT_EQ(blk.ssm_a, nullptr)           << "Attn layer " << i;
            EXPECT_EQ(blk.attn_qkv_weight, nullptr) << "Attn layer " << i;
        } else {
            // SSM pointers non-null
            EXPECT_NE(blk.ssm_a, nullptr)             << "SSM layer " << i;
            EXPECT_NE(blk.ssm_conv1d_weight, nullptr) << "SSM layer " << i;
            EXPECT_NE(blk.ssm_dt_bias, nullptr)       << "SSM layer " << i;
            EXPECT_NE(blk.ssm_alpha_weight, nullptr)  << "SSM layer " << i;
            EXPECT_NE(blk.ssm_beta_weight, nullptr)   << "SSM layer " << i;
            EXPECT_NE(blk.attn_qkv_weight, nullptr)   << "SSM layer " << i;
            EXPECT_NE(blk.attn_gate_weight, nullptr)   << "SSM layer " << i;
            EXPECT_NE(blk.ssm_norm_weight, nullptr)   << "SSM layer " << i;
            EXPECT_NE(blk.ssm_out_weight, nullptr)    << "SSM layer " << i;
            // Attention pointers null
            EXPECT_EQ(blk.attn_q_weight, nullptr)  << "SSM layer " << i;
            EXPECT_EQ(blk.attn_k_weight, nullptr)  << "SSM layer " << i;
            EXPECT_EQ(blk.attn_v_weight, nullptr)  << "SSM layer " << i;
        }
    }
}

// ============================================================
// Test: SSM tensor shapes match debug_gguf (spot check layer 0)
// ============================================================
TEST_F(Qwen35TensorLoadingTest, SSMTensorShapesCorrect) {
    SKIP_IF_NO_MODEL();

    const auto& blk = model_->get_block(0);

    // ssm_a [16]
    ASSERT_NE(blk.ssm_a, nullptr);
    EXPECT_EQ(blk.ssm_a->ne[0], 16);

    // ssm_conv1d.weight [4, 6144]
    ASSERT_NE(blk.ssm_conv1d_weight, nullptr);
    EXPECT_EQ(blk.ssm_conv1d_weight->ne[0], 4);
    EXPECT_EQ(blk.ssm_conv1d_weight->ne[1], 6144);

    // ssm_dt.bias [16]
    ASSERT_NE(blk.ssm_dt_bias, nullptr);
    EXPECT_EQ(blk.ssm_dt_bias->ne[0], 16);

    // ssm_alpha.weight [1024, 16]
    ASSERT_NE(blk.ssm_alpha_weight, nullptr);
    EXPECT_EQ(blk.ssm_alpha_weight->ne[0], 1024);
    EXPECT_EQ(blk.ssm_alpha_weight->ne[1], 16);

    // ssm_beta.weight [1024, 16]
    ASSERT_NE(blk.ssm_beta_weight, nullptr);
    EXPECT_EQ(blk.ssm_beta_weight->ne[0], 1024);
    EXPECT_EQ(blk.ssm_beta_weight->ne[1], 16);

    // attn_qkv.weight [1024, 6144]
    ASSERT_NE(blk.attn_qkv_weight, nullptr);
    EXPECT_EQ(blk.attn_qkv_weight->ne[0], 1024);
    EXPECT_EQ(blk.attn_qkv_weight->ne[1], 6144);

    // attn_gate.weight [1024, 2048]
    ASSERT_NE(blk.attn_gate_weight, nullptr);
    EXPECT_EQ(blk.attn_gate_weight->ne[0], 1024);
    EXPECT_EQ(blk.attn_gate_weight->ne[1], 2048);

    // ssm_norm.weight [128]
    ASSERT_NE(blk.ssm_norm_weight, nullptr);
    EXPECT_EQ(blk.ssm_norm_weight->ne[0], 128);

    // ssm_out.weight [2048, 1024]
    ASSERT_NE(blk.ssm_out_weight, nullptr);
    EXPECT_EQ(blk.ssm_out_weight->ne[0], 2048);
    EXPECT_EQ(blk.ssm_out_weight->ne[1], 1024);
}

// ============================================================
// Test: Attention tensor shapes match debug_gguf (layer 3)
// ============================================================
TEST_F(Qwen35TensorLoadingTest, AttentionTensorShapesCorrect) {
    SKIP_IF_NO_MODEL();

    const auto& blk = model_->get_block(3);

    // attn_q.weight [1024, 4096]
    ASSERT_NE(blk.attn_q_weight, nullptr);
    EXPECT_EQ(blk.attn_q_weight->ne[0], 1024);
    EXPECT_EQ(blk.attn_q_weight->ne[1], 4096);

    // attn_k.weight [1024, 512]
    ASSERT_NE(blk.attn_k_weight, nullptr);
    EXPECT_EQ(blk.attn_k_weight->ne[0], 1024);
    EXPECT_EQ(blk.attn_k_weight->ne[1], 512);

    // attn_v.weight [1024, 512]
    ASSERT_NE(blk.attn_v_weight, nullptr);
    EXPECT_EQ(blk.attn_v_weight->ne[0], 1024);
    EXPECT_EQ(blk.attn_v_weight->ne[1], 512);

    // attn_output.weight [2048, 1024]
    ASSERT_NE(blk.attn_output_weight, nullptr);
    EXPECT_EQ(blk.attn_output_weight->ne[0], 2048);
    EXPECT_EQ(blk.attn_output_weight->ne[1], 1024);

    // attn_q_norm.weight [256]
    ASSERT_NE(blk.attn_q_norm_weight, nullptr);
    EXPECT_EQ(blk.attn_q_norm_weight->ne[0], 256);

    // attn_k_norm.weight [256]
    ASSERT_NE(blk.attn_k_norm_weight, nullptr);
    EXPECT_EQ(blk.attn_k_norm_weight->ne[0], 256);
}
