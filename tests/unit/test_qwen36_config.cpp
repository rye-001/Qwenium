#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "../../src/models/qwen36.h"
#include "../../src/core/model.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a minimal but complete ModelMetadata for qwen35moe with all
// family-specific fields populated to valid values.
static ModelMetadata make_valid_meta() {
    ModelMetadata m;
    m.architecture = "qwen35moe";
    m.block_count  = 4;

    m.raw_kv.set("qwen35moe.full_attention_interval",   (uint32_t)4);
    m.raw_kv.set("qwen35moe.ssm.conv_kernel",           (uint32_t)4);
    m.raw_kv.set("qwen35moe.ssm.state_size",            (uint32_t)128);
    m.raw_kv.set("qwen35moe.ssm.group_count",           (uint32_t)16);
    m.raw_kv.set("qwen35moe.ssm.time_step_rank",        (uint32_t)32);
    m.raw_kv.set("qwen35moe.ssm.inner_size",            (uint32_t)4096);
    m.raw_kv.set("qwen35moe.expert_count",              (uint32_t)256);
    m.raw_kv.set("qwen35moe.expert_used_count",         (uint32_t)8);
    m.raw_kv.set("qwen35moe.expert_feed_forward_length",(uint32_t)512);
    m.raw_kv.set("qwen35moe.rope.dimension_count",      (uint32_t)64);

    return m;
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

TEST(Qwen35MoEConfig, HappyPathPopulatesAllFields) {
    auto meta = make_valid_meta();
    Qwen35MoEConfig cfg;
    ASSERT_NO_THROW(cfg = Qwen35MoEConfig::from_metadata(meta));

    EXPECT_EQ(cfg.ssm_conv_kernel,             meta.raw_kv.get_uint32("qwen35moe.ssm.conv_kernel"));
    EXPECT_EQ(cfg.ssm_state_size,              meta.raw_kv.get_uint32("qwen35moe.ssm.state_size"));
    EXPECT_EQ(cfg.ssm_group_count,             meta.raw_kv.get_uint32("qwen35moe.ssm.group_count"));
    EXPECT_EQ(cfg.ssm_time_step_rank,          meta.raw_kv.get_uint32("qwen35moe.ssm.time_step_rank"));
    EXPECT_EQ(cfg.ssm_inner_size,              meta.raw_kv.get_uint32("qwen35moe.ssm.inner_size"));
    EXPECT_EQ(cfg.expert_count,                meta.raw_kv.get_uint32("qwen35moe.expert_count"));
    EXPECT_EQ(cfg.expert_used_count,           meta.raw_kv.get_uint32("qwen35moe.expert_used_count"));
    EXPECT_EQ(cfg.expert_feed_forward_length,  meta.raw_kv.get_uint32("qwen35moe.expert_feed_forward_length"));
    EXPECT_EQ(cfg.rope_dimension_count,        meta.raw_kv.get_uint32("qwen35moe.rope.dimension_count"));
    EXPECT_EQ(cfg.full_attention_interval,     meta.raw_kv.get_uint32("qwen35moe.full_attention_interval"));
}

TEST(Qwen35MoEConfig, ExpertUsedCountEqualToExpertCountIsAccepted) {
    auto meta = make_valid_meta();
    const uint32_t n = meta.raw_kv.get_uint32("qwen35moe.expert_count");
    meta.raw_kv.set("qwen35moe.expert_used_count", n);  // edge: equal is valid
    EXPECT_NO_THROW(Qwen35MoEConfig::from_metadata(meta));
}

// ---------------------------------------------------------------------------
// expert_used_count > expert_count — canonical fail-loud test from the prompt
// ---------------------------------------------------------------------------

TEST(Qwen35MoEConfig, ExpertUsedCountExceedsExpertCountThrows) {
    auto meta = make_valid_meta();
    const uint32_t n = meta.raw_kv.get_uint32("qwen35moe.expert_count");
    meta.raw_kv.set("qwen35moe.expert_used_count", n + 1);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for expert_used_count > expert_count";
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("expert_used_count"), std::string::npos)
            << "Error must name the failing field; got: " << msg;
        EXPECT_NE(msg.find("expert_count"), std::string::npos)
            << "Error must reference the bound; got: " << msg;
    }
}

// ---------------------------------------------------------------------------
// Each required field = 0 triggers fail-loud error naming that field
// ---------------------------------------------------------------------------

TEST(Qwen35MoEConfig, ZeroSsmStateSizeThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.ssm.state_size", (uint32_t)0);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_state_size = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_state_size"), std::string::npos)
            << "Error must name ssm_state_size; got: " << e.what();
    }
}

TEST(Qwen35MoEConfig, ZeroSsmInnerSizeThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.ssm.inner_size", (uint32_t)0);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_inner_size = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_inner_size"), std::string::npos)
            << "Error must name ssm_inner_size; got: " << e.what();
    }
}

TEST(Qwen35MoEConfig, ZeroSsmTimeStepRankThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.ssm.time_step_rank", (uint32_t)0);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_time_step_rank = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_time_step_rank"), std::string::npos)
            << "Error must name ssm_time_step_rank; got: " << e.what();
    }
}

TEST(Qwen35MoEConfig, ZeroSsmGroupCountThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.ssm.group_count", (uint32_t)0);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_group_count = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_group_count"), std::string::npos)
            << "Error must name ssm_group_count; got: " << e.what();
    }
}

TEST(Qwen35MoEConfig, ZeroSsmConvKernelThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.ssm.conv_kernel", (uint32_t)0);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_conv_kernel = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_conv_kernel"), std::string::npos)
            << "Error must name ssm_conv_kernel; got: " << e.what();
    }
}

TEST(Qwen35MoEConfig, ZeroExpertCountThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.expert_count",      (uint32_t)0);
    meta.raw_kv.set("qwen35moe.expert_used_count", (uint32_t)0);  // keep <= expert_count to isolate this check
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for expert_count = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("expert_count"), std::string::npos)
            << "Error must name expert_count; got: " << e.what();
    }
}

TEST(Qwen35MoEConfig, ZeroExpertFeedForwardLengthThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.expert_feed_forward_length", (uint32_t)0);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for expert_feed_forward_length = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("expert_feed_forward_length"), std::string::npos)
            << "Error must name expert_feed_forward_length; got: " << e.what();
    }
}

TEST(Qwen35MoEConfig, ZeroRopeDimensionCountThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.rope.dimension_count", (uint32_t)0);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for rope_dimension_count = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("rope_dimension_count"), std::string::npos)
            << "Error must name rope_dimension_count; got: " << e.what();
    }
}

TEST(Qwen35MoEConfig, ZeroFullAttentionIntervalThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35moe.full_attention_interval", (uint32_t)0);
    try {
        Qwen35MoEConfig::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for full_attention_interval = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("full_attention_interval"), std::string::npos)
            << "Error must name full_attention_interval; got: " << e.what();
    }
}

// ---------------------------------------------------------------------------
// Layer-kind helpers — correctness against inline arithmetic reference
// ---------------------------------------------------------------------------

TEST(Qwen35MoEConfig, IsFullAttentionLayerPatternInterval4) {
    // Canonical qwen35moe pattern: attention at il % 4 == 3 (layers 3,7,11,…,39)
    auto meta = make_valid_meta();
    meta.block_count = 40;
    meta.raw_kv.set("qwen35moe.full_attention_interval", (uint32_t)4);
    Qwen35MoEConfig cfg = Qwen35MoEConfig::from_metadata(meta);

    EXPECT_FALSE(cfg.is_full_attention_layer(0));
    EXPECT_FALSE(cfg.is_full_attention_layer(1));
    EXPECT_FALSE(cfg.is_full_attention_layer(2));
    EXPECT_TRUE (cfg.is_full_attention_layer(3));
    EXPECT_FALSE(cfg.is_full_attention_layer(4));
    EXPECT_FALSE(cfg.is_full_attention_layer(5));
    EXPECT_FALSE(cfg.is_full_attention_layer(6));
    EXPECT_TRUE (cfg.is_full_attention_layer(7));
    EXPECT_TRUE (cfg.is_full_attention_layer(39));  // last layer of 40-block model

    EXPECT_TRUE (cfg.is_ssm_layer(0));
    EXPECT_FALSE(cfg.is_ssm_layer(3));
}

TEST(Qwen35MoEConfig, LayerKindMethodsMatchInlineArithmeticForEveryLayer) {
    // Verify cfg.is_full_attention_layer(il) agrees with the reference inline math
    // for all il in [0, block_count), for two different interval values.
    for (uint32_t interval : {4u, 2u}) {
        auto meta = make_valid_meta();
        meta.block_count = 24;
        meta.raw_kv.set("qwen35moe.full_attention_interval", interval);
        Qwen35MoEConfig cfg = Qwen35MoEConfig::from_metadata(meta);

        for (uint32_t il = 0; il < meta.block_count; ++il) {
            const bool expected = (il % interval) == (interval - 1);
            EXPECT_EQ(cfg.is_full_attention_layer(il), expected)
                << "is_full_attention_layer(" << il << ") wrong for interval=" << interval;
            EXPECT_EQ(cfg.is_ssm_layer(il), !expected)
                << "is_ssm_layer(" << il << ") wrong for interval=" << interval;
        }
    }
}
