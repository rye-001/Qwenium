/**
 * test_qwen35_metadata.cpp — Step 1 TDD for Qwen 3.5 support
 * 
 * Tests metadata parsing and layer type classification against
 * the real Qwen3.5-0.8B-BF16.gguf file.
 * 
 * Run: ./qwen3-unit-tests --gtest_filter="Qwen35*"
 * 
 * Requires: QWEN35_MODEL_PATH env var pointing to Qwen3.5-0.8B-BF16.gguf
 *   export QWEN35_MODEL_PATH=/path/to/Qwen3.5-0.8B-BF16.gguf
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <set>

#include "../../src/core/model.h"
#include "../../src/loader/gguf_loader.h"
#include "../../src/models/qwen35.h"

// Helper to get model path from env
static std::string get_qwen35_model_path() {
    const char* path = std::getenv("QWEN35_MODEL_PATH");
    if (!path) {
        return "";
    }
    return std::string(path);
}

// Skip tests if model not available
#define SKIP_IF_NO_MODEL()                                          \
    do {                                                            \
        if (get_qwen35_model_path().empty()) {                     \
            GTEST_SKIP() << "QWEN35_MODEL_PATH not set, skipping"; \
        }                                                          \
    } while (0)


// ============================================================
// Test: Architecture detection
// ============================================================
TEST(Qwen35Metadata, ArchitectureIsQwen35) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    EXPECT_EQ(meta.architecture, "qwen35");
}

// ============================================================
// Test: Core model dimensions (from debug_gguf keys 14-24)
// ============================================================
TEST(Qwen35Metadata, CoreDimensionsMatchGGUF) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    // Key[14]: qwen35.block_count = 24
    EXPECT_EQ(meta.block_count, 24u);

    // Key[15]: qwen35.context_length = 262144
    EXPECT_EQ(meta.context_length, 262144u);

    // Key[16]: qwen35.embedding_length = 1024
    EXPECT_EQ(meta.embedding_length, 1024u);

    // Key[17]: qwen35.feed_forward_length = 3584
    EXPECT_EQ(meta.feed_forward_length, 3584u);

    // Key[18]: qwen35.attention.head_count = 8
    EXPECT_EQ(meta.attention_head_count, 8u);

    // Key[19]: qwen35.attention.head_count_kv = 2
    EXPECT_EQ(meta.attention_head_count_kv, 2u);

    // Key[21]: qwen35.rope.freq_base = 1e+07
    EXPECT_FLOAT_EQ(meta.rope_freq_base, 1e7f);

    // Key[22]: qwen35.attention.layer_norm_rms_epsilon = 1e-06
    EXPECT_FLOAT_EQ(meta.rms_norm_eps, 1e-6f);

    // Key[23]: qwen35.attention.key_length = 256
    EXPECT_EQ(meta.attention_key_length, 256u);

    // Key[24]: qwen35.attention.value_length = 256
    EXPECT_EQ(meta.attention_value_length, 256u);
}

// ============================================================
// Test: SSM-specific metadata (from debug_gguf keys 26-32)
// ============================================================
TEST(Qwen35Metadata, SSMMetadataParsed) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    // Key[26]: qwen35.ssm.conv_kernel = 4
    EXPECT_EQ(meta.raw_kv.get_uint32("qwen35.ssm.conv_kernel"), 4u);

    // Key[27]: qwen35.ssm.state_size = 128
    EXPECT_EQ(meta.raw_kv.get_uint32("qwen35.ssm.state_size"), 128u);

    // Key[28]: qwen35.ssm.group_count = 16
    EXPECT_EQ(meta.raw_kv.get_uint32("qwen35.ssm.group_count"), 16u);

    // Key[29]: qwen35.ssm.time_step_rank = 16
    EXPECT_EQ(meta.raw_kv.get_uint32("qwen35.ssm.time_step_rank"), 16u);

    // Key[30]: qwen35.ssm.inner_size = 2048
    EXPECT_EQ(meta.raw_kv.get_uint32("qwen35.ssm.inner_size"), 2048u);

    // Key[31]: qwen35.full_attention_interval = 4
    EXPECT_EQ(meta.raw_kv.get_uint32("qwen35.full_attention_interval"), 4u);

    // Key[32]: qwen35.rope.dimension_count = 64
    EXPECT_EQ(meta.raw_kv.get_uint32("qwen35.rope.dimension_count"), 64u);
}

// ============================================================
// Test: Tokenizer metadata (from debug_gguf keys 34-40)
// ============================================================
TEST(Qwen35Metadata, TokenizerMetadata) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    // Key[34]: tokenizer.ggml.model = 'gpt2'
    EXPECT_EQ(meta.tokenizer_type, "gpt2");

    // Key[35]: tokenizer.ggml.pre = 'qwen35'
    EXPECT_EQ(meta.tokenizer_pre, "qwen35");

    // Key[36]: tokenizer.ggml.tokens = [array of 248320 elements]
    EXPECT_EQ(meta.vocab_size, 248320u);
    EXPECT_EQ(meta.id_to_token.size(), 248320u);

    // Key[38]: tokenizer.ggml.merges = [array of 247587 elements]
    EXPECT_EQ(meta.merges.size(), 247587u);

    // Key[39]: tokenizer.ggml.eos_token_id = 248046
    EXPECT_EQ(meta.eos_token_id, 248046);

    // Key[40]: tokenizer.ggml.padding_token_id = 248055
    EXPECT_EQ(meta.padding_token_id, 248055);
}

// ============================================================
// Test: Weight tying — no output.weight tensor exists
// ============================================================
TEST(Qwen35Metadata, WeightTyingNoOutputWeight) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    // Verify output.weight is NOT in tensor inventory
    EXPECT_EQ(meta.tensor_inventory.count("output.weight"), 0u);

    // Verify token_embd.weight IS present (will be used as LM head)
    EXPECT_EQ(meta.tensor_inventory.count("token_embd.weight"), 1u);

    // Verify output_norm.weight IS present
    EXPECT_EQ(meta.tensor_inventory.count("output_norm.weight"), 1u);
}

// ============================================================
// Test: Total tensor count = 320
// ============================================================
TEST(Qwen35Metadata, TensorCount) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    EXPECT_EQ(meta.tensor_inventory.size(), 320u);
}

// ============================================================
// Test: Layer type classification
// Full attention layers: {3, 7, 11, 15, 19, 23}
// SSM layers: all others (0-2, 4-6, 8-10, 12-14, 16-18, 20-22)
// ============================================================
TEST(Qwen35Metadata, LayerTypeClassification) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    // full_attention_interval = 4 means layers 3,7,11,15,19,23
    const std::set<uint32_t> expected_attn_layers = {3, 7, 11, 15, 19, 23};
    const Qwen35Config cfg = Qwen35Config::from_metadata(meta);

    for (uint32_t i = 0; i < meta.block_count; ++i) {
        bool is_full_attn = expected_attn_layers.count(i) > 0;

        // Classification check via Qwen35Config — the canonical location after migration
        EXPECT_EQ(cfg.is_full_attention_layer(i), is_full_attn)
            << "Layer " << i << " classification mismatch";

        // Cross-check against actual tensors in GGUF
        std::string prefix = "blk." + std::to_string(i) + ".";

        if (is_full_attn) {
            // Full attention layers have separate Q/K/V and QK norms
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_q.weight"))
                << "Layer " << i << " missing attn_q.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_k.weight"))
                << "Layer " << i << " missing attn_k.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_v.weight"))
                << "Layer " << i << " missing attn_v.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_output.weight"))
                << "Layer " << i << " missing attn_output.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_q_norm.weight"))
                << "Layer " << i << " missing attn_q_norm.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_k_norm.weight"))
                << "Layer " << i << " missing attn_k_norm.weight";

            // Should NOT have SSM tensors
            EXPECT_FALSE(meta.tensor_inventory.count(prefix + "ssm_a"))
                << "Layer " << i << " should not have ssm_a";
            EXPECT_FALSE(meta.tensor_inventory.count(prefix + "attn_qkv.weight"))
                << "Layer " << i << " should not have attn_qkv.weight";
        } else {
            // SSM layers have fused QKV and SSM-specific tensors
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_qkv.weight"))
                << "Layer " << i << " missing attn_qkv.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ssm_a"))
                << "Layer " << i << " missing ssm_a";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ssm_conv1d.weight"))
                << "Layer " << i << " missing ssm_conv1d.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ssm_dt.bias"))
                << "Layer " << i << " missing ssm_dt.bias";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ssm_alpha.weight"))
                << "Layer " << i << " missing ssm_alpha.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ssm_beta.weight"))
                << "Layer " << i << " missing ssm_beta.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_gate.weight"))
                << "Layer " << i << " missing attn_gate.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ssm_norm.weight"))
                << "Layer " << i << " missing ssm_norm.weight";
            EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ssm_out.weight"))
                << "Layer " << i << " missing ssm_out.weight";

            // Should NOT have separate Q/K/V
            EXPECT_FALSE(meta.tensor_inventory.count(prefix + "attn_q.weight"))
                << "Layer " << i << " should not have attn_q.weight";
        }

        // Both layer types share these
        EXPECT_TRUE(meta.tensor_inventory.count(prefix + "attn_norm.weight"))
            << "Layer " << i << " missing attn_norm.weight";
        EXPECT_TRUE(meta.tensor_inventory.count(prefix + "post_attention_norm.weight"))
            << "Layer " << i << " missing post_attention_norm.weight";
        EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ffn_gate.weight"))
            << "Layer " << i << " missing ffn_gate.weight";
        EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ffn_up.weight"))
            << "Layer " << i << " missing ffn_up.weight";
        EXPECT_TRUE(meta.tensor_inventory.count(prefix + "ffn_down.weight"))
            << "Layer " << i << " missing ffn_down.weight";
    }
}

// ============================================================
// Test: SSM tensor shapes match expected dimensions
// ============================================================
TEST(Qwen35Metadata, SSMTensorShapes) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    // Check layer 0 (SSM layer) tensor shapes from debug_gguf
    auto check_shape = [&](const std::string& name,
                           const std::vector<uint64_t>& expected) {
        auto it = meta.tensor_inventory.find(name);
        ASSERT_NE(it, meta.tensor_inventory.end()) << "Missing tensor: " << name;
        EXPECT_EQ(it->second.shape, expected) << "Shape mismatch for: " << name;
    };

    // SSM layer 0 shapes
    check_shape("blk.0.ssm_a",              {16});
    check_shape("blk.0.ssm_conv1d.weight",  {4, 6144});
    check_shape("blk.0.ssm_dt.bias",        {16});
    check_shape("blk.0.ssm_alpha.weight",   {1024, 16});
    check_shape("blk.0.ssm_beta.weight",    {1024, 16});
    check_shape("blk.0.attn_qkv.weight",    {1024, 6144});
    check_shape("blk.0.attn_gate.weight",   {1024, 2048});
    check_shape("blk.0.ssm_norm.weight",    {128});
    check_shape("blk.0.ssm_out.weight",     {2048, 1024});

    // Full attention layer 3 shapes
    check_shape("blk.3.attn_q.weight",      {1024, 4096});
    check_shape("blk.3.attn_k.weight",      {1024, 512});
    check_shape("blk.3.attn_v.weight",      {1024, 512});
    check_shape("blk.3.attn_output.weight", {2048, 1024});
    check_shape("blk.3.attn_q_norm.weight", {256});
    check_shape("blk.3.attn_k_norm.weight", {256});
}

// ============================================================
// Test: Norm naming — qwen35 uses post_attention_norm, not ffn_norm
// ============================================================
TEST(Qwen35Metadata, NormNamingConvention) {
    SKIP_IF_NO_MODEL();

    GGUFLoader loader;
    loader.load_model(get_qwen35_model_path());

    ModelMetadata meta;
    loader.extract_metadata(meta);

    for (uint32_t i = 0; i < meta.block_count; ++i) {
        std::string prefix = "blk." + std::to_string(i) + ".";

        // qwen35 uses post_attention_norm (NOT ffn_norm)
        EXPECT_TRUE(meta.tensor_inventory.count(prefix + "post_attention_norm.weight"))
            << "Layer " << i << " missing post_attention_norm.weight";
        EXPECT_FALSE(meta.tensor_inventory.count(prefix + "ffn_norm.weight"))
            << "Layer " << i << " should not have ffn_norm.weight (qwen35 uses post_attention_norm)";
    }
}

// ============================================================
// Test: validate_architecture accepts qwen35
// ============================================================
TEST(Qwen35Metadata, ValidateArchitectureAcceptsQwen35) {
    SKIP_IF_NO_MODEL();

    Model model;
    model.load_metadata(get_qwen35_model_path());

    EXPECT_TRUE(model.validate_architecture());
}
