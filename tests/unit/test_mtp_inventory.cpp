#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/models/qwen36.h"
#include "../../src/core/model.h"

// ---------------------------------------------------------------------------
// Phase 2 of docs/plan-mtp-decode.md: the qwen35moe inventory validator must
// recognise the optional NextN / MTP head tensor group with three outcomes —
//   absent  (nextn_predict_layers == 0)        → no head, load fine
//   full    (nextn > 0, all four tensors)      → capability on
//   partial (nextn > 0, any tensor missing)    → fail-loud naming it
// Synthetic metadata only — no real GGUF needed.
// ---------------------------------------------------------------------------

void validate_qwen36_inventory(const ModelMetadata& meta);  // from qwen36.cpp

namespace {

void add(ModelMetadata& m, const std::string& name) {
    m.tensor_inventory[name] = TensorMetadata{name, GGML_TYPE_F32, {1}, 0};
}

const std::vector<std::string> kMoe = {
    "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
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

// Build a complete, valid qwen35moe inventory: `block_count` blocks, of which
// the last `nextn` are the NextN head. fai = full-attention interval.
ModelMetadata make_meta(uint32_t block_count, uint32_t nextn, uint32_t fai = 4) {
    ModelMetadata m;
    m.architecture = "qwen35moe";
    m.block_count  = block_count;
    m.raw_kv.set("qwen35moe.full_attention_interval", fai);
    // Full ssm/expert set so Qwen35MoEConfig::from_metadata also succeeds.
    m.raw_kv.set("qwen35moe.ssm.conv_kernel",            (uint32_t)4);
    m.raw_kv.set("qwen35moe.ssm.state_size",             (uint32_t)128);
    m.raw_kv.set("qwen35moe.ssm.group_count",            (uint32_t)16);
    m.raw_kv.set("qwen35moe.ssm.time_step_rank",         (uint32_t)32);
    m.raw_kv.set("qwen35moe.ssm.inner_size",             (uint32_t)4096);
    m.raw_kv.set("qwen35moe.expert_count",               (uint32_t)256);
    m.raw_kv.set("qwen35moe.expert_used_count",          (uint32_t)8);
    m.raw_kv.set("qwen35moe.expert_feed_forward_length", (uint32_t)512);
    m.raw_kv.set("qwen35moe.rope.dimension_count",       (uint32_t)64);
    if (nextn > 0)
        m.raw_kv.set("qwen35moe.nextn_predict_layers", nextn);

    add(m, "token_embd.weight");
    add(m, "output_norm.weight");

    const uint32_t n_main = block_count - nextn;
    for (uint32_t i = 0; i < block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const bool is_nextn = (i >= n_main);
        add(m, p + "attn_norm.weight");
        add(m, p + "post_attention_norm.weight");
        for (auto& t : kMoe) add(m, p + t);
        const bool is_full = is_nextn || ((fai > 0) && ((i % fai) == (fai - 1)));
        for (auto& t : (is_full ? kAttn : kDn)) add(m, p + t);
        if (is_nextn)
            for (auto& t : kNextn) add(m, p + t);
    }
    return m;
}

} // namespace

// --- absent: standard GGUF, no head ---------------------------------------
TEST(MtpInventory, AbsentHeadLoadsFine) {
    auto m = make_meta(/*block_count=*/4, /*nextn=*/0);
    EXPECT_NO_THROW(validate_qwen36_inventory(m));
    EXPECT_FALSE(Qwen35MoEConfig::from_metadata(m).has_mtp_head());
}

// --- full: head present, capability on ------------------------------------
TEST(MtpInventory, FullHeadValidatesAndFlagsCapability) {
    auto m = make_meta(/*block_count=*/5, /*nextn=*/1);
    EXPECT_NO_THROW(validate_qwen36_inventory(m));
    auto cfg = Qwen35MoEConfig::from_metadata(m);
    EXPECT_TRUE(cfg.has_mtp_head());
    EXPECT_EQ(cfg.nextn_predict_layers, 1u);
}

// --- partial: a missing NextN tensor is named ------------------------------
TEST(MtpInventory, PartialHeadFailsLoudNamingTensor) {
    auto m = make_meta(5, 1);
    m.tensor_inventory.erase("blk.4.nextn.hnorm.weight");
    try {
        validate_qwen36_inventory(m);
        FAIL() << "expected fail-loud on missing nextn tensor";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("nextn.hnorm.weight"), std::string::npos)
            << "message must name the missing tensor: " << e.what();
    }
}

// --- NextN block is attention-typed: a missing attn tensor also fails -------
TEST(MtpInventory, NextnBlockRequiresAttentionTensors) {
    auto m = make_meta(5, 1);
    m.tensor_inventory.erase("blk.4.attn_q.weight");
    EXPECT_THROW(validate_qwen36_inventory(m), std::runtime_error);
}

// --- guard: nextn_predict_layers must be < block_count ---------------------
TEST(MtpInventory, NextnNotLessThanBlockCountThrows) {
    auto m = make_meta(4, 0);
    m.raw_kv.set("qwen35moe.nextn_predict_layers", (uint32_t)4);  // == block_count
    EXPECT_THROW(validate_qwen36_inventory(m), std::runtime_error);
}
