#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "../../src/loader/gguf_loader.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/qwen36.h"
#include "../../src/core/model.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string model_path() {
    const char* p = std::getenv("QWEN_MODEL_PATH");
    return p ? std::string(p) : "";
}

// Build a minimal but complete qwen35moe ModelMetadata with a synthetic
// tensor inventory for n_blocks blocks (full_attention_interval = 4).
// Pass a non-empty drop_key to deliberately omit one tensor.
static ModelMetadata make_qwen35moe_meta(uint32_t n_blocks,
                                         const std::string& drop_key = "") {
    ModelMetadata m;
    m.architecture = "qwen35moe";
    m.block_count  = n_blocks;

    m.raw_kv.set("qwen35moe.full_attention_interval",   (uint32_t)4);
    m.raw_kv.set("qwen35moe.expert_count",              (uint32_t)256);
    m.raw_kv.set("qwen35moe.expert_used_count",         (uint32_t)8);
    m.raw_kv.set("qwen35moe.expert_feed_forward_length",(uint32_t)512);
    m.raw_kv.set("qwen35moe.ssm.conv_kernel",           (uint32_t)4);
    m.raw_kv.set("qwen35moe.ssm.state_size",            (uint32_t)128);
    m.raw_kv.set("qwen35moe.ssm.group_count",           (uint32_t)16);
    m.raw_kv.set("qwen35moe.ssm.time_step_rank",        (uint32_t)32);
    m.raw_kv.set("qwen35moe.ssm.inner_size",            (uint32_t)4096);
    m.raw_kv.set("qwen35moe.rope.dimension_count",      (uint32_t)64);

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

    const uint32_t fai = m.raw_kv.get_uint32("qwen35moe.full_attention_interval");
    for (uint32_t i = 0; i < n_blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        add(p + "attn_norm.weight");
        add(p + "post_attention_norm.weight");
        for (const auto& t : moe) add(p + t);
        const bool is_full = (fai > 0) && ((i % fai) == (fai - 1));
        if (is_full)
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
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m;
    m.architecture = "qwen35moe";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Qwen36Hparams, ArchValidationRejectsUnknown) {
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m;
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
// Tests Qwen35MoEConfig::is_full_attention_layer (the canonical location after
// the layer-kind migration). ModelMetadata::is_full_attention_layer is no
// longer called from the loader (model.cpp uses inline arithmetic now).
// ---------------------------------------------------------------------------

TEST(Qwen36Hparams, AttentionLayerPattern) {
    // Build a valid metadata for qwen35moe and derive the config from it.
    auto m = make_qwen35moe_meta(40);

    Qwen35MoEConfig cfg = Qwen35MoEConfig::from_metadata(m);

    // Attention at 3, 7, 11, … (i % 4 == 3)
    EXPECT_FALSE(cfg.is_full_attention_layer(0));
    EXPECT_FALSE(cfg.is_full_attention_layer(1));
    EXPECT_FALSE(cfg.is_full_attention_layer(2));
    EXPECT_TRUE (cfg.is_full_attention_layer(3));
    EXPECT_FALSE(cfg.is_full_attention_layer(4));
    EXPECT_FALSE(cfg.is_full_attention_layer(5));
    EXPECT_FALSE(cfg.is_full_attention_layer(6));
    EXPECT_TRUE (cfg.is_full_attention_layer(7));
    EXPECT_TRUE (cfg.is_full_attention_layer(39)); // last layer of 40-block model
}

// ---------------------------------------------------------------------------
// validate_inventory_for_architecture (qwen35moe) — fail-loud error contract
// ---------------------------------------------------------------------------

TEST(Qwen36Hparams, InventoryValidationPassesForCompleteInventory) {
    register_builtin_models();
    // n_blocks = 4 gives us 3 DeltaNet layers (0,1,2) + 1 attention layer (3)
    auto m = make_qwen35moe_meta(4);
    EXPECT_NO_THROW(validate_inventory_for_architecture(m));
}

TEST(Qwen36Hparams, InventoryValidationFailsOnMissingRouter) {
    register_builtin_models();
    // Drop the MoE router for block 0 — error message must name the tensor
    auto m = make_qwen35moe_meta(4, "blk.0.ffn_gate_inp.weight");
    try {
        validate_inventory_for_architecture(m);
        FAIL() << "Expected GGUFLoadError for missing router weight";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("ffn_gate_inp.weight"), std::string::npos)
            << "Error should name the missing tensor: " << e.what();
    }
}

TEST(Qwen36Hparams, InventoryValidationFailsOnMissingExpertGate) {
    register_builtin_models();
    auto m = make_qwen35moe_meta(4, "blk.1.ffn_gate_exps.weight");
    try {
        validate_inventory_for_architecture(m);
        FAIL() << "Expected GGUFLoadError for missing expert gate weight";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("ffn_gate_exps.weight"), std::string::npos)
            << "Error should name the missing tensor: " << e.what();
    }
}

TEST(Qwen36Hparams, InventoryValidationFailsOnMissingSsmTensor) {
    register_builtin_models();
    // Block 2 is a DeltaNet layer (2 % 4 != 3)
    auto m = make_qwen35moe_meta(4, "blk.2.ssm_a");
    try {
        validate_inventory_for_architecture(m);
        FAIL() << "Expected GGUFLoadError for missing ssm_a";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_a"), std::string::npos)
            << "Error should name the missing tensor: " << e.what();
    }
}

TEST(Qwen36Hparams, InventoryValidationFailsOnMissingAttentionWeight) {
    register_builtin_models();
    // Block 3 is an attention layer (3 % 4 == 3)
    auto m = make_qwen35moe_meta(4, "blk.3.attn_q.weight");
    try {
        validate_inventory_for_architecture(m);
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
    Model model;
    ASSERT_NO_THROW(model.load_metadata(model_path()));

    const auto& m = model.get_metadata();
    EXPECT_EQ(m.architecture, "qwen35moe");
    EXPECT_EQ(m.block_count, 40u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.expert_count"),                        256u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.expert_used_count"),                   8u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.expert_feed_forward_length"),          512u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.expert_shared_feed_forward_length"),   512u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.full_attention_interval"),             4u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.ssm.conv_kernel"),                 4u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.ssm.state_size"),                  128u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.ssm.group_count"),                 16u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.ssm.time_step_rank"),              32u);
    EXPECT_EQ(m.raw_kv.get_uint32("qwen35moe.ssm.inner_size"),                  4096u);
}

TEST_F(Qwen36ModelFile, TensorInventoryResolvesWithoutError) {
    Model model;
    ASSERT_NO_THROW(model.load_metadata(model_path()));
    ASSERT_NO_THROW(model.load_tensors());
    EXPECT_TRUE(model.validate_architecture());
}
