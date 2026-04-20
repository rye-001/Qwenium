// test_weight_binding.cpp — PR 2.6 / PR 2.7
//
// Tests the weight-binding infrastructure: WeightSlot, TensorDescriptor,
// TensorNameMap, and the four failure modes the spec requires.
//
// Failure modes (from docs/modular-layer-architecture.md § Weight Binding):
//   1. Unknown tensor name (not in TensorNameMap)
//   2. Missing slot (TensorNameMap declares a slot but no tensor resolves to it)
//   3. Shape mismatch (tensor resolves to a slot but ne[] differs)
//   4. Dtype mismatch (tensor resolves to a slot but dtype differs)
//
// Run: ./qwen3-weight-binding-tests --gtest_filter="WeightBinding*"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "../../src/loader/weight_binding.h"
#include "../../src/qinf_error.h"

// ── Helpers ───────────────────────────────────────────────────────────────────

static TensorDescriptor make_desc(const std::string& name,
                                  std::initializer_list<int64_t> shape,
                                  ggml_type dtype) {
    TensorDescriptor d;
    d.name  = name;
    d.dtype = dtype;
    int i = 0;
    for (int64_t s : shape) d.ne[i++] = s;
    d.n_dims = (int)shape.size();
    return d;
}

static WeightSlot make_slot(const std::string& name,
                             std::initializer_list<int64_t> shape,
                             ggml_type dtype) {
    WeightSlot s;
    s.name  = name;
    s.dtype = dtype;
    int i = 0;
    for (int64_t v : shape) s.expected_ne[i++] = v;
    s.n_dims = (int)shape.size();
    return s;
}

// ── Failure mode 1: unknown tensor name ───────────────────────────────────────

TEST(WeightBinding, UnknownTensorNameThrows) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    // tensor name does not appear in the map
    TensorDescriptor unknown = make_desc("blk.0.attn_TYPO.weight", {4096, 4096}, GGML_TYPE_F16);
    EXPECT_THROW(map.bind({unknown}), std::runtime_error);
}

TEST(WeightBinding, UnknownTensorNameMessageContainsName) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    TensorDescriptor unknown = make_desc("blk.0.attn_TYPO.weight", {4096, 4096}, GGML_TYPE_F16);
    try {
        map.bind({unknown});
        FAIL() << "Expected exception not thrown";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("blk.0.attn_TYPO.weight"), std::string::npos)
            << "error should name the unknown tensor";
    }
}

// ── Failure mode 2: missing slot ──────────────────────────────────────────────

TEST(WeightBinding, MissingSlotThrows) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    map.register_name("blk.0.attn_q.weight", "q_proj");
    // v_proj slot declared but no tensor maps to it
    map.add_slot("v_proj", make_slot("v_proj", {512, 4096}, GGML_TYPE_F16));

    TensorDescriptor q_tensor = make_desc("blk.0.attn_q.weight", {4096, 4096}, GGML_TYPE_F16);
    EXPECT_THROW(map.bind({q_tensor}), std::runtime_error);
}

TEST(WeightBinding, MissingSlotMessageContainsSlotName) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    map.register_name("blk.0.attn_q.weight", "q_proj");
    map.add_slot("v_proj", make_slot("v_proj", {512, 4096}, GGML_TYPE_F16));

    TensorDescriptor q_tensor = make_desc("blk.0.attn_q.weight", {4096, 4096}, GGML_TYPE_F16);
    try {
        map.bind({q_tensor});
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("v_proj"), std::string::npos)
            << "error should name the missing slot";
    }
}

// ── Failure mode 3: shape mismatch ────────────────────────────────────────────

TEST(WeightBinding, ShapeMismatchThrows) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    map.register_name("blk.0.attn_q.weight", "q_proj");

    // Provide wrong shape: [4096, 2048] instead of [4096, 4096]
    TensorDescriptor bad_shape = make_desc("blk.0.attn_q.weight", {4096, 2048}, GGML_TYPE_F16);
    EXPECT_THROW(map.bind({bad_shape}), std::runtime_error);
}

TEST(WeightBinding, ShapeMismatchMessageContainsSlotExpectedActual) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    map.register_name("blk.0.attn_q.weight", "q_proj");

    TensorDescriptor bad_shape = make_desc("blk.0.attn_q.weight", {4096, 2048}, GGML_TYPE_F16);
    try {
        map.bind({bad_shape});
        FAIL();
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("q_proj"),  std::string::npos) << "should name the slot";
        EXPECT_NE(msg.find("4096"),    std::string::npos) << "should show expected dim";
        EXPECT_NE(msg.find("2048"),    std::string::npos) << "should show actual dim";
    }
}

// ── Failure mode 4: dtype mismatch ────────────────────────────────────────────

TEST(WeightBinding, DtypeMismatchThrows) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    map.register_name("blk.0.attn_q.weight", "q_proj");

    TensorDescriptor bad_dtype = make_desc("blk.0.attn_q.weight", {4096, 4096}, GGML_TYPE_F32);
    EXPECT_THROW(map.bind({bad_dtype}), std::runtime_error);
}

TEST(WeightBinding, DtypeMismatchMessageContainsSlotExpectedActual) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    map.register_name("blk.0.attn_q.weight", "q_proj");

    TensorDescriptor bad_dtype = make_desc("blk.0.attn_q.weight", {4096, 4096}, GGML_TYPE_F32);
    try {
        map.bind({bad_dtype});
        FAIL();
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("q_proj"), std::string::npos) << "should name the slot";
        // ggml_type names should appear
        EXPECT_NE(msg.find("f16"),  std::string::npos) << "should show expected dtype";
        EXPECT_NE(msg.find("f32"),  std::string::npos) << "should show actual dtype";
    }
}

// ── Happy path ────────────────────────────────────────────────────────────────

TEST(WeightBinding, HappyPathBindsAllSlots) {
    TensorNameMap map;
    map.add_slot("q_proj", make_slot("q_proj", {4096, 4096}, GGML_TYPE_F16));
    map.add_slot("v_proj", make_slot("v_proj", {512,  4096}, GGML_TYPE_F16));
    map.register_name("blk.0.attn_q.weight", "q_proj");
    map.register_name("blk.0.attn_v.weight", "v_proj");

    std::vector<TensorDescriptor> tensors = {
        make_desc("blk.0.attn_q.weight", {4096, 4096}, GGML_TYPE_F16),
        make_desc("blk.0.attn_v.weight", {512,  4096}, GGML_TYPE_F16),
    };

    BindingResult result = map.bind(tensors);
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result.count("q_proj"), 1u);
    EXPECT_EQ(result.count("v_proj"), 1u);
    EXPECT_EQ(result.at("q_proj").name, "blk.0.attn_q.weight");
    EXPECT_EQ(result.at("v_proj").name, "blk.0.attn_v.weight");
}
