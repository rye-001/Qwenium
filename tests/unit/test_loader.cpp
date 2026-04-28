// test_loader.cpp — PR 2.8
//
// Unit tests for src/loader/: GGUFLoader construction, Tokenizer build,
// and known-answer ASCII round-trip.  No model file required.
//
// Memory baseline: calculate_tensors_memory_size() for Qwen3-0.6B-Q8_0 ≈ 620 MB.
// This is recorded for Phase 4 regression detection, not enforced here.
//
// Run: ./qwen3-loader-tests

#include <gtest/gtest.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/loader/gguf_loader.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/models/model_registry.h"

// ── Fixture: minimal byte-level vocabulary ────────────────────────────────────
//
// Mirrors Tokenizer::initialize_byte_mapping() exactly so that
// encode(x) → decode() round-trips any ASCII text without a real model file.
// 256 byte tokens (indices 0–255) + one EOS control token at index 256.
static ModelMetadata make_byte_level_metadata() {
    std::map<int, int> b2u;
    std::set<int> printable;
    for (int i = 33;  i <= 126; ++i) { printable.insert(i); b2u[i] = i; }
    for (int i = 161; i <= 172; ++i) { printable.insert(i); b2u[i] = i; }
    for (int i = 174; i <= 255; ++i) { printable.insert(i); b2u[i] = i; }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!printable.count(b)) b2u[b] = 256 + n++;
    }

    auto to_utf8 = [](int u) -> std::string {
        std::string s;
        if (u < 128) {
            s += static_cast<char>(u);
        } else {
            s += static_cast<char>(0xC0 | (u >> 6));
            s += static_cast<char>(0x80 | (u & 0x3F));
        }
        return s;
    };

    ModelMetadata m;
    m.id_to_token.resize(257);
    m.token_types.resize(257, TokenType::NORMAL);
    for (int b = 0; b < 256; ++b) {
        m.id_to_token[b] = to_utf8(b2u[b]);
    }
    m.id_to_token[256]  = "<|endoftext|>";
    m.token_types[256]  = TokenType::CONTROL;
    m.eos_token_id      = 256;
    m.bos_token_id      = -1;
    m.padding_token_id  = -1;
    return m;
}

// ── Inventory schema tests (PR G1.2) ──────────────────────────────────────────

// Build a minimal ModelMetadata with a synthetic tensor inventory for tests.
// Tensor types and shapes are not exercised here — only required-key presence.
static TensorMetadata fake_tensor(const std::string& name) {
    TensorMetadata t;
    t.name = name;
    t.type = GGML_TYPE_F32;
    t.shape = {1};
    t.offset = 0;
    return t;
}

static ModelMetadata make_complete_gemma_inventory(uint32_t blocks = 2) {
    ModelMetadata m;
    m.architecture = "gemma";
    m.block_count = blocks;
    m.tensor_inventory["token_embd.weight"] = fake_tensor("token_embd.weight");
    m.tensor_inventory["output_norm.weight"] = fake_tensor("output_norm.weight");
    const std::vector<std::string> per_block = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "attn_norm.weight", "ffn_norm.weight"
    };
    for (uint32_t i = 0; i < blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block) {
            m.tensor_inventory[p + t] = fake_tensor(p + t);
        }
    }
    return m;
}

static ModelMetadata make_complete_qwen3_inventory(uint32_t blocks = 2) {
    ModelMetadata m;
    m.architecture = "qwen3";
    m.block_count = blocks;
    m.tensor_inventory["token_embd.weight"] = fake_tensor("token_embd.weight");
    m.tensor_inventory["output_norm.weight"] = fake_tensor("output_norm.weight");
    const std::vector<std::string> per_block = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
        "ffn_up.weight", "ffn_down.weight"
    };
    for (uint32_t i = 0; i < blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block) {
            m.tensor_inventory[p + t] = fake_tensor(p + t);
        }
    }
    return m;
}

TEST(InventorySchema, GemmaCompleteInventoryPasses) {
    register_builtin_models();
    auto m = make_complete_gemma_inventory();
    EXPECT_NO_THROW(validate_inventory_for_architecture(m));
}

TEST(InventorySchema, GemmaMissingGlobalTensorFailsLoudly) {
    register_builtin_models();
    auto m = make_complete_gemma_inventory();
    m.tensor_inventory.erase("output_norm.weight");
    try {
        validate_inventory_for_architecture(m);
        FAIL() << "expected GGUFLoadError";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("gemma"), std::string::npos) << msg;
        EXPECT_NE(msg.find("output_norm.weight"), std::string::npos) << msg;
    }
}

TEST(InventorySchema, GemmaMissingPerBlockTensorFailsLoudly) {
    register_builtin_models();
    auto m = make_complete_gemma_inventory(3);
    m.tensor_inventory.erase("blk.1.ffn_down.weight");
    try {
        validate_inventory_for_architecture(m);
        FAIL() << "expected GGUFLoadError";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("blk.1.ffn_down.weight"), std::string::npos) << msg;
    }
}

TEST(InventorySchema, GemmaTiedEmbeddingsAcceptedNoSeparateOutput) {
    register_builtin_models();
    // Gemma reuses token_embd.weight for the LM head — no `output.weight`.
    // Validator must NOT require output.weight.
    auto m = make_complete_gemma_inventory();
    EXPECT_TRUE(m.tensor_inventory.find("output.weight") == m.tensor_inventory.end());
    EXPECT_NO_THROW(validate_inventory_for_architecture(m));
}

TEST(InventorySchema, Qwen3InventoryUnaffectedByDispatch) {
    register_builtin_models();
    auto m = make_complete_qwen3_inventory();
    EXPECT_NO_THROW(validate_inventory_for_architecture(m));
}

TEST(InventorySchema, DispatchUnknownArchitectureFailsLoudly) {
    ModelMetadata m;
    m.architecture = "totally_unknown";
    try {
        validate_inventory_for_architecture(m);
        FAIL() << "expected GGUFLoadError";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("validate_inventory_for_architecture"), std::string::npos) << msg;
        EXPECT_NE(msg.find("got 'totally_unknown'"), std::string::npos) << msg;
    }
}

// ── Architecture allow-list tests (PR G1.1) ───────────────────────────────────

TEST(Loader, ArchitectureAllowListAcceptsQwen3) {
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m; m.architecture = "qwen3";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListAcceptsQwen2) {
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m; m.architecture = "qwen2";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListAcceptsQwen35) {
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m; m.architecture = "qwen35";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListAcceptsQwen35Moe) {
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m; m.architecture = "qwen35moe";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListAcceptsGemma) {
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m; m.architecture = "gemma";
    EXPECT_NO_THROW(loader.validate_architecture(m));
}

TEST(Loader, ArchitectureAllowListRejectsUnknownWithFailLoudFormat) {
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m; m.architecture = "unknown_arch_xyz";
    try {
        loader.validate_architecture(m);
        FAIL() << "expected GGUFLoadError for unknown architecture";
    } catch (const GGUFLoadError& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("validate_architecture"), std::string::npos) << msg;
        EXPECT_NE(msg.find("expected one of"), std::string::npos) << msg;
        EXPECT_NE(msg.find("got 'unknown_arch_xyz'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("'gemma'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("'qwen3'"), std::string::npos) << msg;
    }
}

TEST(Loader, ArchitectureAllowListRejectsEmpty) {
    register_builtin_models();
    GGUFLoader loader;
    ModelMetadata m; m.architecture = "";
    EXPECT_THROW(loader.validate_architecture(m), GGUFLoadError);
}

// ── GGUFLoader build tests ────────────────────────────────────────────────────

TEST(Loader, GGUFLoaderDefaultConstructs) {
    GGUFLoader loader;
    EXPECT_FALSE(loader.is_loaded());
}

TEST(Loader, CreateLoaderFactoryReturnsNonNull) {
    auto loader = create_gguf_loader();
    ASSERT_NE(loader, nullptr);
    EXPECT_FALSE(loader->is_loaded());
}

TEST(Loader, LoadNonexistentPathThrows) {
    GGUFLoader loader;
    EXPECT_THROW(
        loader.load_model("/tmp/qwen_inference_test_nonexistent_loader.gguf"),
        std::exception
    );
}

// ── Tokenizer build tests ─────────────────────────────────────────────────────

TEST(Loader, TokenizerVocabSizeMatchesInput) {
    ModelMetadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    EXPECT_EQ(tok.get_vocabulary().size(), 257u);
}

TEST(Loader, TokenizerReportsEOSTokenId) {
    ModelMetadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    EXPECT_EQ(tok.get_eos_token_id(), 256);
}

TEST(Loader, TokenizerDecodeEOSReturnsControlString) {
    ModelMetadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    // EOS is CONTROL type; decode returns the token string verbatim.
    EXPECT_EQ(tok.decode(256), "<|endoftext|>");
}

// ── Known-answer round-trip tests ─────────────────────────────────────────────
//
// No BPE merges in this vocabulary.  Every ASCII char encodes to exactly one
// byte token and decodes back via the GPT-2 byte mapping.

TEST(Loader, TokenizerRoundTripSingleWord) {
    ModelMetadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "Hello";
    auto ids = tok.encode(text);
    ASSERT_EQ(ids.size(), 5u);  // H e l l o → 5 byte tokens (no merges)
    EXPECT_EQ(tok.decode(ids), text);
}

TEST(Loader, TokenizerRoundTripPhraseWithSpace) {
    ModelMetadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "hello world";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST(Loader, TokenizerRoundTripDigitsAndPunctuation) {
    ModelMetadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "count 42!";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

// ── GGUFKVBag population tests (real model file) ──────────────────────────────
//
// These tests load the Qwen3.5-0.8B GGUF and verify that raw_kv contains a
// representative sample of scalar keys with values that match the corresponding
// typed fields on ModelMetadata.  Self-skip when QWEN35_MODEL_PATH is unset.

static std::string get_qwen35_model_path_for_loader() {
    const char* p = std::getenv("QWEN35_MODEL_PATH");
    return p ? std::string(p) : "";
}

#define SKIP_IF_NO_QWEN35()                                                \
    do {                                                                   \
        if (get_qwen35_model_path_for_loader().empty()) {                  \
            GTEST_SKIP() << "QWEN35_MODEL_PATH not set, skipping";        \
        }                                                                  \
    } while (0)

// Fixture: loads the model once for all bag tests.
class GGUFKVBagLoaderTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_qwen35_model_path_for_loader().empty()) return;
        loader_ = std::make_unique<GGUFLoader>();
        loader_->load_model(get_qwen35_model_path_for_loader());
        loader_->extract_metadata(meta_);
    }
    static void TearDownTestSuite() {
        loader_.reset();
    }

    static std::unique_ptr<GGUFLoader> loader_;
    static ModelMetadata meta_;
};
std::unique_ptr<GGUFLoader> GGUFKVBagLoaderTest::loader_ = nullptr;
ModelMetadata GGUFKVBagLoaderTest::meta_;

// uint32 — block_count agrees between typed field and bag.
TEST_F(GGUFKVBagLoaderTest, BlockCountInBagMatchesTypedField) {
    SKIP_IF_NO_QWEN35();
    const std::string key = meta_.architecture + ".block_count";
    ASSERT_TRUE(meta_.raw_kv.contains(key)) << "bag missing " << key;
    EXPECT_EQ(meta_.raw_kv.get_uint32(key), meta_.block_count);
}

// float — rope.freq_base agrees between typed field and bag.
TEST_F(GGUFKVBagLoaderTest, RopeFreqBaseInBagMatchesTypedField) {
    SKIP_IF_NO_QWEN35();
    const std::string key = meta_.architecture + ".rope.freq_base";
    ASSERT_TRUE(meta_.raw_kv.contains(key)) << "bag missing " << key;
    EXPECT_FLOAT_EQ(meta_.raw_kv.get_float(key), meta_.rope_freq_base);
}

// string — tokenizer.ggml.model agrees between typed field and bag.
TEST_F(GGUFKVBagLoaderTest, TokenizerModelInBagMatchesTypedField) {
    SKIP_IF_NO_QWEN35();
    ASSERT_TRUE(meta_.raw_kv.contains("tokenizer.ggml.model"))
        << "bag missing tokenizer.ggml.model";
    EXPECT_EQ(meta_.raw_kv.get_string("tokenizer.ggml.model"), meta_.tokenizer_type);
}

// string — tokenizer.ggml.pre agrees between typed field and bag.
TEST_F(GGUFKVBagLoaderTest, TokenizerPreInBagMatchesTypedField) {
    SKIP_IF_NO_QWEN35();
    ASSERT_TRUE(meta_.raw_kv.contains("tokenizer.ggml.pre"))
        << "bag missing tokenizer.ggml.pre";
    EXPECT_EQ(meta_.raw_kv.get_string("tokenizer.ggml.pre"), meta_.tokenizer_pre);
}

// uint32 (family-specific) — full_attention_interval is present and non-zero in the bag.
TEST_F(GGUFKVBagLoaderTest, FullAttentionIntervalInBag) {
    SKIP_IF_NO_QWEN35();
    const std::string key = meta_.architecture + ".full_attention_interval";
    ASSERT_TRUE(meta_.raw_kv.contains(key)) << "bag missing " << key;
    EXPECT_GT(meta_.raw_kv.get_uint32(key), 0u);
}

// bool (optional) — tokenizer.ggml.add_bos_token, if present, agrees.
TEST_F(GGUFKVBagLoaderTest, AddBosTokenInBagMatchesTypedFieldWhenPresent) {
    SKIP_IF_NO_QWEN35();
    auto v = meta_.raw_kv.get_bool_opt("tokenizer.ggml.add_bos_token");
    if (v.has_value()) {
        EXPECT_EQ(*v, meta_.add_bos_token);
    }
    // Not present in every GGUF — test passes either way.
}

// Tokenizer array keys are NOT in the bag (arrays are excluded by design).
TEST_F(GGUFKVBagLoaderTest, TokenizerArraysNotInBag) {
    SKIP_IF_NO_QWEN35();
    EXPECT_FALSE(meta_.raw_kv.contains("tokenizer.ggml.tokens"))
        << "tokenizer.ggml.tokens is an array and must not be in raw_kv";
    EXPECT_FALSE(meta_.raw_kv.contains("tokenizer.ggml.merges"))
        << "tokenizer.ggml.merges is an array and must not be in raw_kv";
}

// Wrong-type access on a bag entry throws with the correct message shape.
TEST_F(GGUFKVBagLoaderTest, WrongTypeAccessOnBagKeyThrows) {
    SKIP_IF_NO_QWEN35();
    const std::string key = meta_.architecture + ".block_count"; // uint32 in bag
    ASSERT_TRUE(meta_.raw_kv.contains(key));
    try {
        meta_.raw_kv.get_string(key);   // asks for string from a uint32 entry
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("string"), std::string::npos) << msg;
        EXPECT_NE(msg.find("uint32"), std::string::npos) << msg;
    }
}
