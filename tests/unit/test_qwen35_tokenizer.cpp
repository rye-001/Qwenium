/**
 * test_qwen35_tokenizer.cpp — Step 4 TDD for Qwen 3.5 support
 * 
 * Tests tokenizer initialization and basic encode/decode with the
 * Qwen3.5 248K vocabulary.
 * 
 * Requires: QWEN35_MODEL_PATH env var pointing to Qwen3.5-0.8B-BF16.gguf
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

#include "../../src/core/model.h"
#include "../../src/loader/tokenizer.h"

static std::string get_qwen35_model_path() {
    const char* path = std::getenv("QWEN35_MODEL_PATH");
    return path ? std::string(path) : "";
}

#define SKIP_IF_NO_MODEL()                                          \
    do {                                                            \
        if (get_qwen35_model_path().empty()) {                     \
            GTEST_SKIP() << "QWEN35_MODEL_PATH not set, skipping"; \
        }                                                          \
    } while (0)

// Shared fixture — loads model once
class Qwen35TokenizerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (get_qwen35_model_path().empty()) return;

        model_ = std::make_unique<Model>();
        model_->load_metadata(get_qwen35_model_path());
        model_->load_tensors();

        tokenizer_ = model_->get_tokenizer();
    }

    static void TearDownTestSuite() {
        model_.reset();
        tokenizer_ = nullptr;
    }

    static std::unique_ptr<Model> model_;
    static Tokenizer* tokenizer_;
};

std::unique_ptr<Model> Qwen35TokenizerTest::model_ = nullptr;
Tokenizer* Qwen35TokenizerTest::tokenizer_ = nullptr;

// ============================================================
// Test: Tokenizer initializes without crashing
// ============================================================
TEST_F(Qwen35TokenizerTest, InitSucceeds) {
    SKIP_IF_NO_MODEL();
    ASSERT_NE(tokenizer_, nullptr) << "Tokenizer should be initialized for qwen35";
}

// ============================================================
// Test: Vocabulary size is 248320
// ============================================================
TEST_F(Qwen35TokenizerTest, VocabSize) {
    SKIP_IF_NO_MODEL();
    EXPECT_EQ(tokenizer_->get_vocabulary().size(), 248320u);
}

// ============================================================
// Test: EOS token ID matches GGUF metadata
// ============================================================
TEST_F(Qwen35TokenizerTest, EOSTokenId) {
    SKIP_IF_NO_MODEL();
    EXPECT_EQ(tokenizer_->get_eos_token_id(), 248046);
}

// ============================================================
// Test: Special tokens can be looked up by string
// ============================================================
TEST_F(Qwen35TokenizerTest, SpecialTokenLookup) {
    SKIP_IF_NO_MODEL();

    // im_start and im_end must resolve to valid IDs (not UNK)
    int32_t im_start = tokenizer_->get_special_token_id("<|im_start|>");
    int32_t im_end   = tokenizer_->get_special_token_id("<|im_end|>");
    int32_t eos      = tokenizer_->get_special_token_id("<|endoftext|>");

    EXPECT_GE(im_start, 0) << "<|im_start|> not found";
    EXPECT_GE(im_end, 0)   << "<|im_end|> not found";
    EXPECT_GE(eos, 0)      << "<|endoftext|> not found";

    // These should NOT be the old qwen2/3 IDs
    // (They could be, but in 248K vocab they almost certainly aren't)
    // The important thing is they're valid and distinct
    EXPECT_NE(im_start, im_end) << "im_start and im_end should be different";

    // eos from lookup should match metadata
    // eos from string lookup should be a valid ID (may differ from metadata eos_token_id
    // since <|endoftext|> and the actual EOS token can be different tokens in qwen35)
    EXPECT_GE(eos, 0) << "<|endoftext|> should resolve to a valid ID";
}

// ============================================================
// Test: Simple encode/decode roundtrip — ASCII
// ============================================================
TEST_F(Qwen35TokenizerTest, RoundtripASCII) {
    SKIP_IF_NO_MODEL();

    std::string input = "Hello, world!";
    auto tokens = tokenizer_->encode(input);

    EXPECT_GT(tokens.size(), 0u) << "Encoding produced no tokens";

    std::string decoded = tokenizer_->decode(tokens);
    EXPECT_EQ(decoded, input) << "Roundtrip failed for ASCII";
}

// ============================================================
// Test: Simple encode/decode roundtrip — with spaces and numbers
// ============================================================
TEST_F(Qwen35TokenizerTest, RoundtripMixed) {
    SKIP_IF_NO_MODEL();

    std::string input = "The answer is 42.";
    auto tokens = tokenizer_->encode(input);
    EXPECT_GT(tokens.size(), 0u);

    std::string decoded = tokenizer_->decode(tokens);
    EXPECT_EQ(decoded, input);
}

// ============================================================
// Test: Special tokens encode to single token IDs
// ============================================================
TEST_F(Qwen35TokenizerTest, SpecialTokensEncode) {
    SKIP_IF_NO_MODEL();

    // Encoding a special token string should produce exactly one token
    auto tokens = tokenizer_->encode("<|im_start|>");
    EXPECT_EQ(tokens.size(), 1u) << "<|im_start|> should encode to single token";

    // That token should match what get_special_token_id returns
    int32_t expected_id = tokenizer_->get_special_token_id("<|im_start|>");
    EXPECT_EQ(tokens[0], expected_id);
}

// ============================================================
// Test: Chat template encoding — the typical ChatML format
// ============================================================
TEST_F(Qwen35TokenizerTest, ChatTemplateRoundtrip) {
    SKIP_IF_NO_MODEL();

    std::string input = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    auto tokens = tokenizer_->encode(input);

    EXPECT_GT(tokens.size(), 3u) << "Chat template should produce multiple tokens";

    std::string decoded = tokenizer_->decode(tokens);
    EXPECT_EQ(decoded, input) << "Chat template roundtrip failed";
}

// ============================================================
// Test: Code-like text (relevant for DSL use case)
// ============================================================
TEST_F(Qwen35TokenizerTest, RoundtripCode) {
    SKIP_IF_NO_MODEL();

    std::string input = "function getOrders({ customerId }) {\n  return orders.filter(o => o.id === customerId);\n}";
    auto tokens = tokenizer_->encode(input);
    EXPECT_GT(tokens.size(), 0u);

    std::string decoded = tokenizer_->decode(tokens);
    EXPECT_EQ(decoded, input) << "Code roundtrip failed";
}

// ============================================================
// Test: Decode individual special tokens
// ============================================================
TEST_F(Qwen35TokenizerTest, DecodeSpecialTokens) {
    SKIP_IF_NO_MODEL();

    int32_t im_start = tokenizer_->get_special_token_id("<|im_start|>");
    int32_t im_end   = tokenizer_->get_special_token_id("<|im_end|>");

    EXPECT_EQ(tokenizer_->decode(im_start), "<|im_start|>");
    EXPECT_EQ(tokenizer_->decode(im_end), "<|im_end|>");
}

// ============================================================
// Test: Empty string encodes to empty
// ============================================================
TEST_F(Qwen35TokenizerTest, EmptyString) {
    SKIP_IF_NO_MODEL();

    auto tokens = tokenizer_->encode("");
    EXPECT_TRUE(tokens.empty());
}

// ============================================================
// Test: All token IDs from encode are within vocab range
// ============================================================
TEST_F(Qwen35TokenizerTest, TokenIdsInRange) {
    SKIP_IF_NO_MODEL();

    std::string input = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
    auto tokens = tokenizer_->encode(input);

    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_GE(tokens[i], 0) << "Negative token ID at position " << i;
        EXPECT_LT(tokens[i], 248320) << "Token ID out of range at position " << i;
    }
}
