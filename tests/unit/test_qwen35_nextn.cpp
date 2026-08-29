#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/models/qwen35.h"
#include "engine/model.h"

// ---------------------------------------------------------------------------
// The qwen35 recipe must recognise the optional NextN / MTP head that Qwen3.8
// ships and Qwen3.5 does not.  Qwen3.8-27B declares block_count = 65 with
// qwen35.nextn_predict_layers = 1: 64 main blocks plus one trailing NextN head
// block, which is attention-typed regardless of its position in the
// full-attention interval and carries four extra nextn.* tensors.
//
// Three outcomes, mirroring the qwen35moe tests in test_mtp_inventory.cpp:
//   absent  (nextn_predict_layers == 0)     → no head, load fine (Qwen3.5)
//   full    (nextn > 0, all four tensors)   → validates, head reported
//   partial (nextn > 0, any tensor missing) → fail-loud naming it
//
// Also pins ModelMetadata::nextn_predict_layers(), which composes the GGUF key
// from the architecture string rather than hard-coding one literal per family.
// Synthetic metadata only — no real GGUF needed.
// ---------------------------------------------------------------------------

namespace {

void add(ModelMetadata& m, const std::string& name) {
    m.tensor_inventory[name] = TensorMetadata{name, GGML_TYPE_F32, {1}, 0};
}

const std::vector<std::string> kShared = {
    "attn_norm.weight", "post_attention_norm.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
const std::vector<std::string> kAttn = {
    "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"};
const std::vector<std::string> kDn = {
    "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_alpha.weight",
    "ssm_beta.weight", "attn_qkv.weight", "attn_gate.weight",
    "ssm_norm.weight", "ssm_out.weight"};
const std::vector<std::string> kNextn = {
    "nextn.eh_proj.weight", "nextn.enorm.weight",
    "nextn.hnorm.weight", "nextn.shared_head_norm.weight"};

// Build a complete, valid qwen35 inventory: `block_count` blocks, of which the
// last `nextn` are the NextN head. fai = full-attention interval.
ModelMetadata make_meta(uint32_t block_count, uint32_t nextn, uint32_t fai = 4) {
    ModelMetadata m;
    m.architecture = "qwen35";
    m.block_count  = block_count;
    m.raw_kv.set("qwen35.full_attention_interval", fai);
    // Full ssm set so Qwen35Config::from_metadata also succeeds.
    m.raw_kv.set("qwen35.ssm.conv_kernel",    (uint32_t)4);
    m.raw_kv.set("qwen35.ssm.state_size",     (uint32_t)128);
    m.raw_kv.set("qwen35.ssm.group_count",    (uint32_t)16);
    m.raw_kv.set("qwen35.ssm.time_step_rank", (uint32_t)48);
    m.raw_kv.set("qwen35.ssm.inner_size",     (uint32_t)6144);
    m.raw_kv.set("qwen35.rope.dimension_count", (uint32_t)64);
    if (nextn > 0)
        m.raw_kv.set("qwen35.nextn_predict_layers", nextn);

    add(m, "token_embd.weight");
    add(m, "output_norm.weight");

    const uint32_t n_main = block_count - nextn;
    for (uint32_t i = 0; i < block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const bool is_nextn = (i >= n_main);
        for (auto& t : kShared) add(m, p + t);
        // NextN blocks are attention-typed regardless of position.
        const bool is_full = is_nextn || ((fai > 0) && ((i % fai) == (fai - 1)));
        for (auto& t : (is_full ? kAttn : kDn)) add(m, p + t);
        if (is_nextn)
            for (auto& t : kNextn) add(m, p + t);
    }
    return m;
}

} // namespace

// --- absent: standard Qwen3.5 GGUF, no head -------------------------------
TEST(Qwen35Nextn, AbsentHeadLoadsFine) {
    auto m = make_meta(/*block_count=*/4, /*nextn=*/0);
    EXPECT_NO_THROW(validate_qwen35_inventory(m));
    EXPECT_FALSE(Qwen35Config::from_metadata(m).has_mtp_head());
}

// --- full: head present, reported ------------------------------------------
TEST(Qwen35Nextn, FullHeadValidatesAndFlagsCapability) {
    auto m = make_meta(/*block_count=*/5, /*nextn=*/1);
    EXPECT_NO_THROW(validate_qwen35_inventory(m));
    auto cfg = Qwen35Config::from_metadata(m);
    EXPECT_TRUE(cfg.has_mtp_head());
    EXPECT_EQ(cfg.nextn_predict_layers, 1u);
}

// --- the Qwen3.8-27B shape: 65 blocks = 64 main + 1 NextN ------------------
TEST(Qwen35Nextn, Qwen38BlockCountShapeValidates) {
    auto m = make_meta(/*block_count=*/65, /*nextn=*/1);
    EXPECT_NO_THROW(validate_qwen35_inventory(m));
    auto cfg = Qwen35Config::from_metadata(m);
    EXPECT_EQ(cfg.nextn_predict_layers, 1u);
    // Block 64 is NOT a full-attention position (64 % 4 == 0, not 3), so it is
    // only attention-typed because it is the NextN head. That is exactly the
    // case that made the pre-fix loader look for blk.64.ssm_a.
    EXPECT_FALSE(cfg.is_full_attention_layer(64));
}

// --- partial: a missing NextN tensor is named ------------------------------
TEST(Qwen35Nextn, PartialHeadFailsLoudNamingTensor) {
    auto m = make_meta(5, 1);
    m.tensor_inventory.erase("blk.4.nextn.hnorm.weight");
    try {
        validate_qwen35_inventory(m);
        FAIL() << "expected fail-loud on missing nextn tensor";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("nextn.hnorm.weight"), std::string::npos)
            << "message must name the missing tensor: " << e.what();
    }
}

// --- NextN block is attention-typed: a missing attn tensor also fails -------
TEST(Qwen35Nextn, NextnBlockRequiresAttentionTensors) {
    auto m = make_meta(5, 1);
    m.tensor_inventory.erase("blk.4.attn_q.weight");
    EXPECT_THROW(validate_qwen35_inventory(m), std::runtime_error);
}

// --- a NextN block must NOT be validated as an SSM layer -------------------
// Regression pin for the original Qwen3.8 failure: block 4 of a 5-block model
// sits at 4 % 4 == 0, so position-only typing calls it SSM and demands
// blk.4.ssm_a. With NextN typing it must validate without any ssm_* tensor.
TEST(Qwen35Nextn, NextnBlockDoesNotDemandSsmTensors) {
    auto m = make_meta(5, 1);
    for (const auto& t : kDn)
        ASSERT_EQ(m.tensor_inventory.count("blk.4." + t), 0u)
            << "fixture should not place SSM tensors on the NextN block";
    EXPECT_NO_THROW(validate_qwen35_inventory(m));
}

// --- guard: nextn_predict_layers must be < block_count ---------------------
TEST(Qwen35Nextn, NextnNotLessThanBlockCountThrows) {
    auto m = make_meta(4, 0);
    m.raw_kv.set("qwen35.nextn_predict_layers", (uint32_t)4);  // == block_count
    EXPECT_THROW(validate_qwen35_inventory(m), std::runtime_error);
}

// --- the helper composes the key from the architecture string --------------
TEST(Qwen35Nextn, MetadataHelperReadsArchPrefixedKey) {
    ModelMetadata m;
    m.architecture = "qwen35";
    m.raw_kv.set("qwen35.nextn_predict_layers", (uint32_t)1);
    EXPECT_EQ(m.nextn_predict_layers(), 1u);

    // Same concept, different family, different key spelling.
    ModelMetadata moe;
    moe.architecture = "qwen35moe";
    moe.raw_kv.set("qwen35moe.nextn_predict_layers", (uint32_t)2);
    EXPECT_EQ(moe.nextn_predict_layers(), 2u);

    // A key belonging to another architecture must not be picked up.
    ModelMetadata mismatched;
    mismatched.architecture = "qwen35";
    mismatched.raw_kv.set("qwen35moe.nextn_predict_layers", (uint32_t)1);
    EXPECT_EQ(mismatched.nextn_predict_layers(), 0u);

    // Absent key ⇒ 0 ⇒ standard non-MTP GGUF.
    ModelMetadata bare;
    bare.architecture = "qwen35";
    EXPECT_EQ(bare.nextn_predict_layers(), 0u);
}
