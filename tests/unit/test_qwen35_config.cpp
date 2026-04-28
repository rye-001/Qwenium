#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "../../src/models/qwen35.h"
#include "../../src/core/model.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a minimal but complete ModelMetadata for qwen35 with all
// family-specific fields populated to valid values.
static ModelMetadata make_valid_meta() {
    ModelMetadata m;
    m.architecture = "qwen35";
    m.block_count  = 8;

    m.raw_kv.set("qwen35.full_attention_interval", (uint32_t)4);
    m.raw_kv.set("qwen35.ssm.conv_kernel",         (uint32_t)4);
    m.raw_kv.set("qwen35.ssm.state_size",          (uint32_t)128);
    m.raw_kv.set("qwen35.ssm.group_count",         (uint32_t)16);
    m.raw_kv.set("qwen35.ssm.time_step_rank",      (uint32_t)16);
    m.raw_kv.set("qwen35.ssm.inner_size",          (uint32_t)2048);
    m.raw_kv.set("qwen35.rope.dimension_count",    (uint32_t)64);

    return m;
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

TEST(Qwen35Config, HappyPathPopulatesAllFields) {
    auto meta = make_valid_meta();
    Qwen35Config cfg;
    ASSERT_NO_THROW(cfg = Qwen35Config::from_metadata(meta));

    EXPECT_EQ(cfg.ssm_conv_kernel,         meta.raw_kv.get_uint32("qwen35.ssm.conv_kernel"));
    EXPECT_EQ(cfg.ssm_state_size,          meta.raw_kv.get_uint32("qwen35.ssm.state_size"));
    EXPECT_EQ(cfg.ssm_group_count,         meta.raw_kv.get_uint32("qwen35.ssm.group_count"));
    EXPECT_EQ(cfg.ssm_time_step_rank,      meta.raw_kv.get_uint32("qwen35.ssm.time_step_rank"));
    EXPECT_EQ(cfg.ssm_inner_size,          meta.raw_kv.get_uint32("qwen35.ssm.inner_size"));
    EXPECT_EQ(cfg.rope_dimension_count,    meta.raw_kv.get_uint32("qwen35.rope.dimension_count"));
    EXPECT_EQ(cfg.full_attention_interval, meta.raw_kv.get_uint32("qwen35.full_attention_interval"));
}

TEST(Qwen35Config, ZeroRopeDimensionCountIsAccepted) {
    // rope_dimension_count absent/0 means "use full head dimension" — valid for qwen35.
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35.rope.dimension_count", (uint32_t)0);
    EXPECT_NO_THROW(Qwen35Config::from_metadata(meta));
}

// ---------------------------------------------------------------------------
// Required-field validation — fail-loud error contract
// ---------------------------------------------------------------------------

TEST(Qwen35Config, ZeroSsmStateSizeThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35.ssm.state_size", (uint32_t)0);
    try {
        Qwen35Config::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_state_size = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_state_size"), std::string::npos)
            << "Error must name ssm_state_size; got: " << e.what();
    }
}

TEST(Qwen35Config, ZeroSsmInnerSizeThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35.ssm.inner_size", (uint32_t)0);
    try {
        Qwen35Config::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_inner_size = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_inner_size"), std::string::npos)
            << "Error must name ssm_inner_size; got: " << e.what();
    }
}

TEST(Qwen35Config, ZeroSsmTimeStepRankThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35.ssm.time_step_rank", (uint32_t)0);
    try {
        Qwen35Config::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_time_step_rank = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_time_step_rank"), std::string::npos)
            << "Error must name ssm_time_step_rank; got: " << e.what();
    }
}

TEST(Qwen35Config, ZeroSsmGroupCountThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35.ssm.group_count", (uint32_t)0);
    try {
        Qwen35Config::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_group_count = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_group_count"), std::string::npos)
            << "Error must name ssm_group_count; got: " << e.what();
    }
}

TEST(Qwen35Config, ZeroSsmConvKernelThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35.ssm.conv_kernel", (uint32_t)0);
    try {
        Qwen35Config::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for ssm_conv_kernel = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ssm_conv_kernel"), std::string::npos)
            << "Error must name ssm_conv_kernel; got: " << e.what();
    }
}

TEST(Qwen35Config, ZeroFullAttentionIntervalThrows) {
    auto meta = make_valid_meta();
    meta.raw_kv.set("qwen35.full_attention_interval", (uint32_t)0);
    try {
        Qwen35Config::from_metadata(meta);
        FAIL() << "Expected std::runtime_error for full_attention_interval = 0";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("full_attention_interval"), std::string::npos)
            << "Error must name full_attention_interval; got: " << e.what();
    }
}

// ---------------------------------------------------------------------------
// Layer-kind helpers agree with ModelMetadata equivalents
// ---------------------------------------------------------------------------

TEST(Qwen35Config, LayerKindMethodsMatchMetadataForEveryLayer) {
    // interval = 4 → attention at il % 4 == 3 (i.e. il = 3, 7, 11, …)
    ModelMetadata meta = make_valid_meta();
    meta.block_count = 24;
    meta.raw_kv.set("qwen35.full_attention_interval", (uint32_t)4);

    Qwen35Config cfg = Qwen35Config::from_metadata(meta);
    const uint32_t fai = meta.raw_kv.get_uint32("qwen35.full_attention_interval");

    for (uint32_t il = 0; il < meta.block_count; ++il) {
        const bool expected = (fai > 0) && ((il % fai) == (fai - 1));
        EXPECT_EQ(cfg.is_full_attention_layer(il), expected)
            << "is_full_attention_layer disagrees at layer " << il;
        EXPECT_EQ(cfg.is_ssm_layer(il), !expected)
            << "is_ssm_layer disagrees at layer " << il;
    }
}

TEST(Qwen35Config, LayerKindMethodsMatchMetadataInterval2) {
    // interval = 2 → attention at every odd layer
    ModelMetadata meta = make_valid_meta();
    meta.block_count = 12;
    meta.raw_kv.set("qwen35.full_attention_interval", (uint32_t)2);

    Qwen35Config cfg = Qwen35Config::from_metadata(meta);
    const uint32_t fai = meta.raw_kv.get_uint32("qwen35.full_attention_interval");

    for (uint32_t il = 0; il < meta.block_count; ++il) {
        const bool expected = (fai > 0) && ((il % fai) == (fai - 1));
        EXPECT_EQ(cfg.is_full_attention_layer(il), expected)
            << "is_full_attention_layer disagrees at layer " << il;
        EXPECT_EQ(cfg.is_ssm_layer(il), !expected)
            << "is_ssm_layer disagrees at layer " << il;
    }
}

TEST(Qwen35Config, IsFullAttentionLayerPattern) {
    // Spot-check concrete expected values for interval = 4.
    auto meta = make_valid_meta();  // full_attention_interval = 4
    Qwen35Config cfg = Qwen35Config::from_metadata(meta);

    EXPECT_FALSE(cfg.is_full_attention_layer(0));
    EXPECT_FALSE(cfg.is_full_attention_layer(1));
    EXPECT_FALSE(cfg.is_full_attention_layer(2));
    EXPECT_TRUE (cfg.is_full_attention_layer(3));
    EXPECT_FALSE(cfg.is_full_attention_layer(4));
    EXPECT_TRUE (cfg.is_full_attention_layer(7));

    EXPECT_TRUE (cfg.is_ssm_layer(0));
    EXPECT_FALSE(cfg.is_ssm_layer(3));
}
