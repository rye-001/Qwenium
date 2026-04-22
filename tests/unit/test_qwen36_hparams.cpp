#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "../../src/loader/gguf_loader.h"
#include "../../src/qwen3-core/qwen3-model.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string model_path() {
    const char* p = std::getenv("QWEN_MODEL_PATH");
    return p ? std::string(p) : "";
}

// Build a minimal but complete qwen35moe Qwen3Metadata with a synthetic
// tensor inventory for n_blocks blocks (full_attention_interval = 4).
// Pass a non-empty drop_key to deliberately omit one tensor so that
// validate_qwen35moe_inventory() throws.
static Qwen3Metadata make_qwen35moe_meta(uint32_t n_blocks,
                                         const std::string& drop_key = "") {
    Qwen3Metadata m;
    m.architecture         = "qwen35moe";
    m.block_count          = n_blocks;
    m.full_attention_interval = 4;
    m.expert_count         = 256;
    m.expert_used_count    = 8;
    m.expert_feed_forward_length        = 512;
    m.expert_shared_feed_forward_length = 512;

    auto add = [&](const std::string& name) {
        if (name != drop_key)
            m.tensor_inventory[name] = TensorMetadata{name, GGML_TYPE_F32, {1}, 0};
    };

    add("token_embd.weight");
    add("output_norm.weight");

    const std::vector<std::string> moe = {
        "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"
    };
    const std::vector<std::string> attn = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"
    };
    const std::vector<std::string> dn = {
        "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias",
        "ssm_alpha.weight", "ssm_beta.weight",
        "attn_qkv.weight", "attn_gate.weight",
        "ssm_norm.weight", "ssm_out.weight"
    };

    for (uint32_t i = 0; i < n_blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        add(p + "attn_norm.weight");
        add(p + "post_attention_norm.weight");
        for (const auto& t : moe) add(p + t);
        if (m.is_full_attention_layer(i))
            for (const auto& t : attn) add(p + t);
        else
            for (const auto& t : dn)  add(p + t);
    }
    return m;
}

// ---------------------------------------------------------------------------
// Architecture validation (no model file required)
// ---------------------------------------------------------------------------

TEST(Qwen36Hparams, ArchValidationAcceptsQwen35moe) {
    QwenGGUFLoader loader;
    Qwen3Metadata m;
    m.architecture = "qwen35moe";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Qwen36Hparams, ArchValidationRejectsUnknown) {
    QwenGGUFLoader loader;
    Qwen3Metadata m;
    m.architecture = "unknown_arch";
    try {
        loader.validate_architecture(m);
        FAIL() << "Expected GGUFLoadError for unknown architecture";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("unknown_arch"), std::string::npos)
            << "Error message should name the bad architecture";
    }
}

// ---------------------------------------------------------------------------
// is_full_attention_layer — layer pattern for qwen35moe (interval = 4)
// ---------------------------------------------------------------------------

TEST(Qwen36Hparams, AttentionLayerPattern) {
    Qwen3Metadata m;
    m.architecture          = "qwen35moe";
    m.full_attention_interval = 4;

    // Attention at 3, 7, 11, … (i % 4 == 3)
    EXPECT_FALSE(m.is_full_attention_layer(0));
    EXPECT_FALSE(m.is_full_attention_layer(1));
    EXPECT_FALSE(m.is_full_attention_layer(2));
    EXPECT_TRUE (m.is_full_attention_layer(3));
    EXPECT_FALSE(m.is_full_attention_layer(4));
    EXPECT_FALSE(m.is_full_attention_layer(5));
    EXPECT_FALSE(m.is_full_attention_layer(6));
    EXPECT_TRUE (m.is_full_attention_layer(7));
    EXPECT_TRUE (m.is_full_attention_layer(39)); // last layer of 40-block model
}

// ---------------------------------------------------------------------------
// validate_qwen35moe_inventory — fail-loud error contract (no model file)
// ---------------------------------------------------------------------------

TEST(Qwen36Hparams, InventoryValidationPassesForCompleteInventory) {
    // n_blocks = 4 gives us 3 DeltaNet layers (0,1,2) + 1 attention layer (3)
    auto m = make_qwen35moe_meta(4);
    EXPECT_NO_THROW(validate_qwen35moe_inventory(m));
}

TEST(Qwen36Hparams, InventoryValidationFailsOnMissingRouter) {
    // Drop the MoE router for block 0 — error message must name the tensor
    auto m = make_qwen35moe_meta(4, "blk.0.ffn_gate_inp.weight");
    try {
        validate_qwen35moe_inventory(m);
        FAIL() << "Expected GGUFLoadError for missing router weight";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("ffn_gate_inp.weight"), std::string::npos)
            << "Error should name the missing tensor: " << e.what();
    }
}

TEST(Qwen36Hparams, InventoryValidationFailsOnMissingExpertGate) {
    auto m = make_qwen35moe_meta(4, "blk.1.ffn_gate_exps.weight");
    try {
        validate_qwen35moe_inventory(m);
        FAIL() << "Expected GGUFLoadError for missing expert gate weight";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("ffn_gate_exps.weight"), std::string::npos)
            << "Error should name the missing tensor: " << e.what();
    }
}

TEST(Qwen36Hparams, InventoryValidationFailsOnMissingSsmTensor) {
    // Block 2 is a DeltaNet layer (2 % 4 != 3)
    auto m = make_qwen35moe_meta(4, "blk.2.ssm_a");
    try {
        validate_qwen35moe_inventory(m);
        FAIL() << "Expected GGUFLoadError for missing ssm_a";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_a"), std::string::npos)
            << "Error should name the missing tensor: " << e.what();
    }
}

TEST(Qwen36Hparams, InventoryValidationFailsOnMissingAttentionWeight) {
    // Block 3 is an attention layer (3 % 4 == 3)
    auto m = make_qwen35moe_meta(4, "blk.3.attn_q.weight");
    try {
        validate_qwen35moe_inventory(m);
        FAIL() << "Expected GGUFLoadError for missing attn_q.weight";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("attn_q.weight"), std::string::npos)
            << "Error should name the missing tensor: " << e.what();
    }
}

// ---------------------------------------------------------------------------
// Model-file integration tests — skipped when QWEN_MODEL_PATH is unset
// ---------------------------------------------------------------------------

class Qwen36ModelFile : public ::testing::Test {
protected:
    void SetUp() override {
        if (model_path().empty())
            GTEST_SKIP() << "QWEN_MODEL_PATH not set — skipping model-file tests";
    }
};

TEST_F(Qwen36ModelFile, HparamsExtractedCorrectly) {
    Qwen3Model model;
    ASSERT_NO_THROW(model.load_metadata(model_path()));

    const auto& m = model.get_metadata();
    EXPECT_EQ(m.architecture, "qwen35moe");
    EXPECT_EQ(m.block_count, 40u);
    EXPECT_EQ(m.expert_count, 256u);
    EXPECT_EQ(m.expert_used_count, 8u);
    EXPECT_EQ(m.expert_feed_forward_length, 512u);
    EXPECT_EQ(m.expert_shared_feed_forward_length, 512u);
    EXPECT_EQ(m.full_attention_interval, 4u);
    EXPECT_EQ(m.ssm_conv_kernel, 4u);
    EXPECT_EQ(m.ssm_state_size, 128u);
    EXPECT_EQ(m.ssm_group_count, 16u);
    EXPECT_EQ(m.ssm_time_step_rank, 32u);
    EXPECT_EQ(m.ssm_inner_size, 4096u);
}

TEST_F(Qwen36ModelFile, TensorInventoryResolvesWithoutError) {
    Qwen3Model model;
    ASSERT_NO_THROW(model.load_metadata(model_path()));
    ASSERT_NO_THROW(model.load_tensors());
    EXPECT_TRUE(model.validate_architecture());
}
