// test_gemma4_recipe.cpp — PR G4.8
//
// Pure-unit tests for Gemma4Config::from_metadata + validate_gemma4_inventory.
// No model file needed — synthesizes ModelMetadata + tensor inventory in
// memory.  The end-to-end logit-agreement gate against the 26B-A4B GGUF
// runs via tests/fixtures/gemma_ref_logits/g4_a4b/ once the canary harness
// is exercised; that integration is not in this file (see scripts/gen_gemma_ref_logits.py).
//
// Coverage:
//   1. Config26BA4B          — full A4B-shaped metadata produces the
//                              documented per-kind shapes and per-layer
//                              kind vector.
//   2. PerLayerKindFromInventory — toggling presence of attn_v.weight in
//                              the synthetic inventory flips a layer
//                              between sliding and global; layer-index
//                              maps follow.
//   3. SharedKvNonZeroRejected     — A4B invariant: shared_kv_layers > 0 is
//                              fail-loud (dense 31B follow-up promotes).
//   4. PleNonZeroRejected          — same for embedding_length_per_layer_input.
//   5. MissingKvHeadsForKindFails  — n_kv_heads derivation catches a
//                              malformed K-weight shape.
//   6. ValidatorAcceptsMinimalA4B  — validate_gemma4_inventory passes on
//                              the synthetic inventory we built.
//   7. ValidatorRejectsMissingPostAttnNorm — a single missing tensor is
//                              caught with the spec-mandated fail-loud
//                              error format.

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/core/model.h"
#include "../../src/models/gemma4.h"

// Build a synthetic 26B-A4B-shaped ModelMetadata with `n_blocks` layers,
// every (period_global)-th layer marked global (no attn_v.weight in inv).
// Defaults: 6 blocks, layer 5 global → 5 sliding + 1 global, mirrors the
// short-form pattern G3 uses.
static ModelMetadata make_gemma4_meta(uint32_t n_blocks       = 6,
                                       uint32_t period_global = 6)
{
    ModelMetadata m;
    m.architecture           = "gemma4";
    m.block_count            = n_blocks;
    m.attention_head_count   = 16;
    m.embedding_length       = 2816;
    m.context_length         = 8192;     // small for the test; not load-bearing
    m.rms_norm_eps           = 1e-6f;

    m.raw_kv.set("gemma4.attention.key_length",          (uint32_t)512);
    m.raw_kv.set("gemma4.attention.key_length_swa",      (uint32_t)256);
    m.raw_kv.set("gemma4.rope.dimension_count",          (uint32_t)512);
    m.raw_kv.set("gemma4.rope.dimension_count_swa",      (uint32_t)256);
    m.raw_kv.set("gemma4.rope.freq_base",                1.0e6f);
    m.raw_kv.set("gemma4.rope.freq_base_swa",            1.0e4f);
    m.raw_kv.set("gemma4.attention.sliding_window",      (uint32_t)1024);
    m.raw_kv.set("gemma4.feed_forward_length",           (uint32_t)2112);
    m.raw_kv.set("gemma4.expert_feed_forward_length",    (uint32_t)704);
    m.raw_kv.set("gemma4.expert_count",                  (uint32_t)128);
    m.raw_kv.set("gemma4.expert_used_count",             (uint32_t)8);
    m.raw_kv.set("gemma4.final_logit_softcapping",       30.0f);
    m.raw_kv.set("gemma4.attention.shared_kv_layers",    (uint32_t)0);
    m.raw_kv.set("gemma4.embedding_length_per_layer_input",(uint32_t)0);

    auto add = [&](const std::string& name,
                    const std::vector<uint64_t>& shape) {
        m.tensor_inventory[name] =
            TensorMetadata{ name, GGML_TYPE_F32, shape, 0 };
    };

    add("token_embd.weight",  {2816, 256000});
    add("output_norm.weight", {2816});

    const uint32_t head_dim_swa     = 256;
    const uint32_t head_dim_global  = 512;
    const uint32_t n_kv_swa         = 8;
    const uint32_t n_kv_global      = 2;

    for (uint32_t i = 0; i < n_blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const bool is_global = ((i % period_global) == (period_global - 1));

        add(p + "attn_norm.weight", {2816});
        add(p + "attn_q.weight",
            {2816, is_global ? 16u * head_dim_global : 16u * head_dim_swa});
        add(p + "attn_k.weight",
            {2816, is_global ? n_kv_global * head_dim_global
                              : n_kv_swa    * head_dim_swa});
        if (!is_global) {
            // V projection only present on sliding layers (V == K for global).
            add(p + "attn_v.weight",
                {2816, n_kv_swa * head_dim_swa});
        }
        add(p + "attn_q_norm.weight",
            {is_global ? head_dim_global : head_dim_swa});
        add(p + "attn_k_norm.weight",
            {is_global ? head_dim_global : head_dim_swa});
        add(p + "attn_output.weight",
            {is_global ? 16u * head_dim_global : 16u * head_dim_swa, 2816});
        add(p + "post_attention_norm.weight", {2816});

        // Dense FFN
        add(p + "ffn_norm.weight",     {2816});
        add(p + "ffn_gate.weight",     {2816, 2112});
        add(p + "ffn_up.weight",       {2816, 2112});
        add(p + "ffn_down.weight",     {2112, 2816});
        add(p + "post_ffw_norm.weight",{2816});

        // MoE FFN
        add(p + "pre_ffw_norm_2.weight",        {2816});
        add(p + "ffn_gate_inp.scale",            {2816});
        add(p + "ffn_gate_inp.weight",           {2816, 128});
        add(p + "ffn_gate_up_exps.weight",       {2816, 1408, 128});
        add(p + "ffn_down_exps.weight",          {704, 2816, 128});
        add(p + "layer_output_scale.weight",     {1});
    }
    return m;
}

// 1. Full A4B-shape metadata yields the documented per-kind shapes.
TEST(Gemma4Config, Config26BA4B) {
    const auto m = make_gemma4_meta(/*n_blocks=*/6, /*period_global=*/6);
    auto cfg = Gemma4Config::from_metadata(m);

    EXPECT_EQ(cfg.n_layers,          6u);
    EXPECT_EQ(cfg.n_head,            16u);
    EXPECT_EQ(cfg.hidden_dim,        2816u);
    EXPECT_FLOAT_EQ(cfg.rms_norm_eps, 1e-6f);

    EXPECT_EQ(cfg.head_dim_global,   512u);
    EXPECT_EQ(cfg.head_dim_swa,      256u);
    EXPECT_EQ(cfg.n_kv_heads_swa,    8u);
    EXPECT_EQ(cfg.n_kv_heads_global, 2u);
    EXPECT_EQ(cfg.rope_dim_global,   512u);
    EXPECT_EQ(cfg.rope_dim_swa,      256u);
    EXPECT_FLOAT_EQ(cfg.rope_base_global, 1.0e6f);
    EXPECT_FLOAT_EQ(cfg.rope_base_swa,    1.0e4f);
    EXPECT_EQ(cfg.sliding_window,    1024u);

    EXPECT_EQ(cfg.ffn_dim_dense,     2112u);
    EXPECT_EQ(cfg.ffn_dim_expert,    704u);
    EXPECT_EQ(cfg.n_experts,         128u);
    EXPECT_EQ(cfg.expert_top_k,      8u);
    EXPECT_FLOAT_EQ(cfg.final_softcap, 30.0f);

    // Per-layer pattern (period 6 → layer 5 is global).
    ASSERT_EQ(cfg.is_global.size(), 6u);
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_FALSE(cfg.is_global[i]) << "layer " << i << " misclassified as global";
    }
    EXPECT_TRUE(cfg.is_global[5]);

    EXPECT_EQ(cfg.n_swa_layers,    5u);
    EXPECT_EQ(cfg.n_global_layers, 1u);

    // Layer-index maps: contiguous within each kind, -1 outside.
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_EQ(cfg.swa_layer_idx[i],    static_cast<int>(i));
        EXPECT_EQ(cfg.global_layer_idx[i], -1);
    }
    EXPECT_EQ(cfg.swa_layer_idx[5],    -1);
    EXPECT_EQ(cfg.global_layer_idx[5], 0);
}

// 2. The A4B per-layer pattern (3 globals across 30 blocks at indices
//    {17, 23, 29}) round-trips when those layers omit attn_v.weight.
TEST(Gemma4Config, ActualA4BPattern) {
    auto m = make_gemma4_meta(/*n_blocks=*/30, /*period_global=*/30);  // none global
    // Manually flip three layers to global (delete attn_v, resize K/Q/Out to global shape).
    for (uint32_t glb : {17u, 23u, 29u}) {
        const std::string p = "blk." + std::to_string(glb) + ".";
        m.tensor_inventory.erase(p + "attn_v.weight");
        m.tensor_inventory[p + "attn_q.weight"] =
            TensorMetadata{p + "attn_q.weight", GGML_TYPE_F32, {2816, 16u * 512u}, 0};
        m.tensor_inventory[p + "attn_k.weight"] =
            TensorMetadata{p + "attn_k.weight", GGML_TYPE_F32, {2816, 2u * 512u}, 0};
        m.tensor_inventory[p + "attn_output.weight"] =
            TensorMetadata{p + "attn_output.weight", GGML_TYPE_F32, {16u * 512u, 2816}, 0};
        m.tensor_inventory[p + "attn_q_norm.weight"] =
            TensorMetadata{p + "attn_q_norm.weight", GGML_TYPE_F32, {512}, 0};
        m.tensor_inventory[p + "attn_k_norm.weight"] =
            TensorMetadata{p + "attn_k_norm.weight", GGML_TYPE_F32, {512}, 0};
    }

    auto cfg = Gemma4Config::from_metadata(m);
    EXPECT_EQ(cfg.n_swa_layers,    27u);
    EXPECT_EQ(cfg.n_global_layers, 3u);
    EXPECT_TRUE(cfg.is_global[17]);
    EXPECT_TRUE(cfg.is_global[23]);
    EXPECT_TRUE(cfg.is_global[29]);
    EXPECT_FALSE(cfg.is_global[16]);
    EXPECT_FALSE(cfg.is_global[18]);
    EXPECT_FALSE(cfg.is_global[24]);

    EXPECT_EQ(cfg.global_layer_idx[17], 0);
    EXPECT_EQ(cfg.global_layer_idx[23], 1);
    EXPECT_EQ(cfg.global_layer_idx[29], 2);

    EXPECT_EQ(cfg.swa_layer_idx[0], 0);
    EXPECT_EQ(cfg.swa_layer_idx[16], 16);
    EXPECT_EQ(cfg.swa_layer_idx[17], -1);  // it's global
    EXPECT_EQ(cfg.swa_layer_idx[18], 17);  // first sliding after the global gap
}

// 3. shared_kv_layers != 0 is fail-loud (A4B invariant; dense 31B promotes).
TEST(Gemma4Config, SharedKvNonZeroRejected) {
    auto m = make_gemma4_meta();
    m.raw_kv.set("gemma4.attention.shared_kv_layers", (uint32_t)2);
    EXPECT_THROW(Gemma4Config::from_metadata(m), std::runtime_error);
}

// 4. embedding_length_per_layer_input != 0 is fail-loud (PLE dormant in A4B).
TEST(Gemma4Config, PleNonZeroRejected) {
    auto m = make_gemma4_meta();
    m.raw_kv.set("gemma4.embedding_length_per_layer_input", (uint32_t)256);
    EXPECT_THROW(Gemma4Config::from_metadata(m), std::runtime_error);
}

// 5. A K-weight rank-deficient enough to break the n_kv_heads derivation
//    must be caught at config-build time, not silently produce 0.
TEST(Gemma4Config, MissingKvHeadsForKindFails) {
    auto m = make_gemma4_meta();
    // Mangle blk.0.attn_k to a 1D tensor — the derivation will try shape[1].
    m.tensor_inventory["blk.0.attn_k.weight"] =
        TensorMetadata{"blk.0.attn_k.weight", GGML_TYPE_F32, {2816}, 0};
    EXPECT_THROW(Gemma4Config::from_metadata(m), std::runtime_error);
}

// 6. The widened validator passes on the synthetic minimum.
TEST(Gemma4Inventory, AcceptsSyntheticMinimum) {
    const auto m = make_gemma4_meta();
    EXPECT_NO_THROW(validate_gemma4_inventory(m));
}

// 7. A single missing required tensor is caught with the fail-loud format.
TEST(Gemma4Inventory, RejectsMissingPostAttnNorm) {
    auto m = make_gemma4_meta();
    m.tensor_inventory.erase("blk.2.post_attention_norm.weight");
    try {
        validate_gemma4_inventory(m);
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("blk.2.post_attention_norm.weight"),
                  std::string::npos)
            << "error must name the missing tensor: " << what;
    }
}
