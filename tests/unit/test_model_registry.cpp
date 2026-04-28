// test_model_registry.cpp — PR G1.6
//
// Verifies the architecture → ForwardPass factory registry replaces the
// hard-coded if/else chains in chat.cpp / complete.cpp / main.cpp.
//
// Strategy: register a fake architecture with a fake factory, look it up,
// confirm dispatch returns that factory's product. Confirm that an unknown
// architecture fails loudly with "expected one of: ..., got '...'".
//
// We do NOT instantiate real Qwen3ForwardPass here — that requires loading a
// model. The registry must support fake factories so unit tests can exercise
// dispatch in isolation.

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "../../src/loader/gguf_loader.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"

// ── Fake ForwardPass for dispatch test ────────────────────────────────────────
//
// Implements only what the registry needs to construct it. The pure virtuals
// from ForwardPassBase are stubbed because dispatch tests never call them.

namespace {

class FakeForwardPass : public ForwardPassBase {
public:
    FakeForwardPass(const Model& m, const ModelMetadata* meta)
        : ForwardPassBase(m, meta) {}

    ggml_cgraph* build_prefill_graph(const std::vector<int32_t>&, int, uint32_t) override { return nullptr; }
    void set_inputs(ggml_cgraph*, const std::vector<int32_t>&, int) override {}
    void advance_cache(uint32_t, uint32_t) override {}
    void clear_slot(uint32_t) override {}
    void set_cache_pos(uint32_t, uint32_t) override {}
    uint32_t get_cache_pos(uint32_t) const override { return 0; }
    uint32_t get_physical_cache_pos(uint32_t) const override { return 0; }
    void clone_slot(uint32_t, uint32_t, uint32_t) override {}
    ggml_cgraph* build_decoding_graph(const std::vector<int32_t>&,
                                      const std::vector<uint32_t>&,
                                      const std::vector<int32_t>&) override { return nullptr; }
    void set_batched_inputs(ggml_cgraph*,
                            const std::vector<int32_t>&,
                            const std::vector<uint32_t>&,
                            const std::vector<int32_t>&) override {}

    static int construction_count;
};
int FakeForwardPass::construction_count = 0;

} // namespace

TEST(ModelRegistry, RegisterAndLookupFakeFactory) {
    FakeForwardPass::construction_count = 0;

    register_model("fake_arch_test_only",
        [](const Model& m, const ModelMetadata* meta, uint32_t, uint32_t, int) {
            ++FakeForwardPass::construction_count;
            return std::unique_ptr<ForwardPassBase>(new FakeForwardPass(m, meta));
        },
        [](const ModelMetadata&) {});

    EXPECT_TRUE(is_architecture_registered("fake_arch_test_only"));

    Model dummy;  // default-constructed; factory does not dereference it.
    ModelMetadata meta;
    meta.architecture = "fake_arch_test_only";

    auto fp = create_forward_pass(dummy, &meta, /*ctx=*/128, /*bs=*/1, /*kvb=*/0);
    EXPECT_NE(fp, nullptr);
    EXPECT_EQ(FakeForwardPass::construction_count, 1);

    unregister_model("fake_arch_test_only");
    EXPECT_FALSE(is_architecture_registered("fake_arch_test_only"));
}

TEST(ModelRegistry, UnknownArchitectureFailsLoud) {
    Model dummy;
    ModelMetadata meta;
    meta.architecture = "definitely_not_registered";

    try {
        (void)create_forward_pass(dummy, &meta, 128, 1, 0);
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("create_forward_pass"), std::string::npos) << msg;
        EXPECT_NE(msg.find("expected one of"), std::string::npos) << msg;
        EXPECT_NE(msg.find("got 'definitely_not_registered'"), std::string::npos) << msg;
    }
}

TEST(ModelRegistry, BuiltinModelsRegistersQwenFamily) {
    register_builtin_models();
    EXPECT_TRUE(is_architecture_registered("qwen2"));
    EXPECT_TRUE(is_architecture_registered("qwen3"));
    EXPECT_TRUE(is_architecture_registered("qwen35"));
    EXPECT_TRUE(is_architecture_registered("qwen35moe"));
}

TEST(ModelRegistry, BuiltinModelsIsIdempotent) {
    register_builtin_models();
    register_builtin_models();
    EXPECT_TRUE(is_architecture_registered("qwen3"));
}

// ── InventoryValidator registration and dispatch ───────────────────────────────

TEST(ModelRegistry, FakeValidatorIsCalledWithCorrectMetadata) {
    const std::string arch = "fake_validator_arch_test";
    int call_count = 0;
    std::string captured_arch;

    register_model(arch,
        [](const Model& m, const ModelMetadata* meta, uint32_t, uint32_t, int) {
            return std::unique_ptr<ForwardPassBase>(new FakeForwardPass(m, meta));
        },
        [&](const ModelMetadata& meta) {
            ++call_count;
            captured_arch = meta.architecture;
        });

    ModelMetadata meta;
    meta.architecture = arch;
    EXPECT_NO_THROW(validate_inventory_for_architecture(meta));
    EXPECT_EQ(call_count, 1);
    EXPECT_EQ(captured_arch, arch);

    // Verify that a validator error propagates verbatim as GGUFLoadError.
    register_model(arch,
        [](const Model& m, const ModelMetadata* meta, uint32_t, uint32_t, int) {
            return std::unique_ptr<ForwardPassBase>(new FakeForwardPass(m, meta));
        },
        [](const ModelMetadata&) {
            throw std::runtime_error("fake_validator_arch_test: missing tensor 'sentinel': "
                                     "expected in model weights, got absent");
        });
    try {
        validate_inventory_for_architecture(meta);
        FAIL() << "expected GGUFLoadError";
    } catch (const GGUFLoadError& e) {
        EXPECT_NE(std::string(e.what()).find("sentinel"), std::string::npos) << e.what();
    }

    unregister_model(arch);
}

TEST(ModelRegistry, ValidateInventoryUnknownArchFailsLoud) {
    ModelMetadata meta;
    meta.architecture = "not_a_real_arch_xyz";
    try {
        validate_inventory_for_architecture(meta);
        FAIL() << "expected GGUFLoadError for unregistered architecture";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("validate_inventory_for_architecture"), std::string::npos) << msg;
        EXPECT_NE(msg.find("got 'not_a_real_arch_xyz'"), std::string::npos) << msg;
    }
}
