// test_qwen35_family.cpp — the recipe parts shared by qwen35 and qwen35moe.
//
// Scope here is the typed-input declarations, which are model-free and are
// exactly where the duplication used to hurt: the decode gather stride was
// wrong in qwen36 and right in qwen35 for months (architecture.md §12), because
// each recipe declared its own inputs. One declaration site now — these tests
// pin what it declares. The layer body is covered by the recipe-level bitwise
// gates, which is where a graph-building change actually shows up.

#include <gtest/gtest.h>

#include "../../src/models/qwen35_family.h"

namespace {

Qwen35Config dense_cfg() {
    Qwen35Config c{};
    c.ssm_conv_kernel = 4; c.ssm_state_size = 128; c.ssm_group_count = 16;
    c.ssm_time_step_rank = 32; c.ssm_inner_size = 4096;
    c.rope_dimension_count = 64;
    c.full_attention_interval = 4;
    c.nextn_predict_layers = 0;
    return c;                       // expert_* stay 0 ⇒ is_moe() == false
}

Qwen35Config moe_cfg() {
    Qwen35Config c = dense_cfg();
    c.expert_count = 256; c.expert_used_count = 8;
    c.expert_feed_forward_length = 512;
    return c;                       // is_moe() == true
}

}  // namespace

TEST(Qwen35Family, IsMoeIsDerivedFromExpertCount) {
    EXPECT_FALSE(dense_cfg().is_moe());
    EXPECT_TRUE(moe_cfg().is_moe());
}

TEST(Qwen35Family, CommonInputsDeclareTokensAndPositions) {
    GraphInputSet in;
    register_qwen35_common_inputs(in, dense_cfg());
    EXPECT_TRUE(in.has_slot("tokens"));
    EXPECT_TRUE(in.has_slot("inp_pos"));
}

// M-RoPE swaps the positions input for the 4-component variant. Both own the
// same slot name, so the observable contract is "inp_pos is declared exactly
// once" either way — a recipe must never declare both.
TEST(Qwen35Family, PositionsSlotDeclaredForBothRopeModes) {
    Qwen35Config mrope = dense_cfg();
    mrope.mrope_sections = MRopeSections::from_widths({11, 11, 10, 0},
                                                      "test.rope.dimension_sections", 64);
    ASSERT_TRUE(mrope.mrope_sections.active);

    GraphInputSet in;
    register_qwen35_common_inputs(in, mrope);
    EXPECT_TRUE(in.has_slot("tokens"));
    EXPECT_TRUE(in.has_slot("inp_pos"));
}

// Re-registering must not accumulate: common_inputs clears first, which is also
// the §7 ordering rule (graph_inputs_ cleared BEFORE the image splice, or the
// splice's uploaded embeddings are silently discarded).
TEST(Qwen35Family, CommonInputsClearsBeforeDeclaring) {
    GraphInputSet in;
    register_qwen35_common_inputs(in, dense_cfg());
    register_qwen35_common_inputs(in, dense_cfg());
    EXPECT_FALSE(in.empty());
    EXPECT_TRUE(in.has_slot("tokens"));
}

TEST(Qwen35Family, PrefillMasksOnePerFullAttentionLayer) {
    const Qwen35Config c = dense_cfg();          // interval 4 ⇒ layers 3,7,11 …
    GraphInputSet in;
    register_qwen35_prefill_masks(in, c, /*n_layers=*/12);

    for (uint32_t il = 0; il < 12; ++il) {
        const std::string slot = "kq_mask." + std::to_string(il);
        const bool expected = !c.is_ssm_layer(il);
        EXPECT_EQ(in.has_slot(slot.c_str()), expected)
            << "layer " << il << (expected ? " is attention — mask required"
                                           : " is SSM — no mask");
    }
}

// The decode set, including the two inputs the persistent-graph write path adds.
TEST(Qwen35Family, DecodeInputsDeclareMaskAndGather) {
    GraphInputSet in;
    register_qwen35_decode_inputs(in, dense_cfg(), /*n_ctx_max=*/4096,
                                  /*with_kv_write_indices=*/false);
    EXPECT_TRUE(in.has_slot("tokens"));
    EXPECT_TRUE(in.has_slot("inp_pos"));
    EXPECT_TRUE(in.has_slot("kq_mask_b"));
    EXPECT_TRUE(in.has_slot("gather_indices"));
    EXPECT_FALSE(in.has_slot("kv_write_indices"))
        << "the set_rows write index is opt-in (--persistent-graph)";
}

TEST(Qwen35Family, DecodeInputsAddWriteIndicesWhenArmed) {
    GraphInputSet in;
    register_qwen35_decode_inputs(in, moe_cfg(), /*n_ctx_max=*/4096,
                                  /*with_kv_write_indices=*/true);
    EXPECT_TRUE(in.has_slot("kv_write_indices"));
}

// Both hybrids declare the SAME decode inputs. This is the property whose
// absence produced the gather defect: qwen36 selected a different per-slot
// stride than qwen35 for the same cache layout.
TEST(Qwen35Family, DenseAndMoeDeclareIdenticalDecodeInputs) {
    for (bool armed : {false, true}) {
        GraphInputSet dense, moe;
        register_qwen35_decode_inputs(dense, dense_cfg(), 4096, armed);
        register_qwen35_decode_inputs(moe,   moe_cfg(),   4096, armed);
        for (const char* slot : {"tokens", "inp_pos", "kq_mask_b",
                                 "gather_indices", "kv_write_indices"}) {
            EXPECT_EQ(dense.has_slot(slot), moe.has_slot(slot))
                << "slot '" << slot << "' differs between the dense and MoE "
                << "hybrids (armed=" << armed << ") — they must not diverge";
        }
    }
}
