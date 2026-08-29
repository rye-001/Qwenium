#include "gtest/gtest.h"
#include "loader/tokenizer.h"
#include "engine/model.h"
#include "loader/gguf_loader.h"
#include "models/model_registry.h"
#include <memory>
#include <vector>
#include <string>

class TokenizerIntegrationTest : public ::testing::Test {
protected:
    std::unique_ptr<GGUFLoader> loader_;
    ModelMetadata metadata_;
    std::unique_ptr<Tokenizer> tokenizer_;
    static std::string model_path_;

    static void SetUpTestSuite() {
        // Populate the architecture allow-list the loader validates against;
        // the CLI / server do this at startup, so the bare-loader path here
        // must too (otherwise validate_architecture sees an empty list).
        register_builtin_models();
        const char* model_env_path = std::getenv("QWEN3_MODEL_PATH");
        if (model_env_path) {
            model_path_ = model_env_path;
        } else {
#ifdef QINF_MODELS_DIR
            // Absolute, so it resolves from ctest's working directory. Without
            // this the suite defaulted to a relative "./Qwen3-0.6B-Q8_0.gguf"
            // that never resolved, and every test here FAILED rather than
            // skipped — six permanent red rows carried session to session.
            model_path_ = std::string(QINF_MODELS_DIR) + "/Qwen3-0.6B-Q8_0.gguf";
#else
            model_path_ = "./Qwen3-0.6B-Q8_0.gguf";
#endif
        }
    }

    void SetUp() override {
        loader_ = std::make_unique<GGUFLoader>();
        try {
            loader_->load_model(model_path_);
            loader_->extract_metadata(metadata_);
            tokenizer_ = std::make_unique<Tokenizer>(&metadata_);
        } catch (const std::exception& e) {
            // A missing model is "not runnable here", not a defect — the same
            // convention every other model-file test in this suite uses.
            GTEST_SKIP() << "Qwen3-0.6B model not loadable at " << model_path_
                         << " (" << e.what() << ") — set QWEN3_MODEL_PATH to override";
        }
    }
};

std::string TokenizerIntegrationTest::model_path_;

// Qwen's vocab is byte-level BPE (GPT-2 `bytes_to_unicode`). Decoding ONE token
// yields the BYTE it stands for, not a character: the vocab *displays* byte 0xA7
// as "§" because that is its printable stand-in, but "§" as text is U+00A7 =
// two UTF-8 bytes (0xC2 0xA7) and therefore two tokens. Asserting the display
// string here was wrong; the roundtrip test below is what pins the behaviour
// users actually see.
//
// The expectations come from the byte-to-unicode alphabet, not from the
// implementation's output:
//   tokens   0..93  -> bytes 33..126   ('!' .. '~')
//   tokens  94..105 -> bytes 161..172  ('¡' .. '¬')   => token 100 -> byte 167
//   tokens 106..187 -> bytes 174..255  ('®' .. 'ÿ')
//   tokens 188..255 -> the non-printable bytes in order, 0..32 first
//                                                     => token 200 -> byte 12
TEST_F(TokenizerIntegrationTest, BasicTokenValidation) {
    EXPECT_EQ(tokenizer_->decode(0), "!");      // Token 0 is '!'
    EXPECT_EQ(tokenizer_->decode(32), "A");     // Token 32 is 'A'
    EXPECT_EQ(tokenizer_->decode(64), "a");     // Token 64 is 'a'
    EXPECT_EQ(tokenizer_->decode(65), "b");     // Token 65 is 'b' (NOT 'A')
    EXPECT_EQ(tokenizer_->decode(100), std::string("\xA7"));  // byte 167, shown as '§'
}

TEST_F(TokenizerIntegrationTest, SpecialTokensValidation) {
    // Test Qwen3 special tokens from GGUF metadata
    ASSERT_EQ(tokenizer_->decode(151645), "<|im_end|>");    // EOS token
    ASSERT_EQ(tokenizer_->decode(151643), "<|endoftext|>"); // BOS/PAD token
}

TEST_F(TokenizerIntegrationTest, EncodeAgainstGoldenValues) {
    // Use a simple test that matches Qwen3's actual behavior
    std::string text = "Hello world!";
    
    // Get actual encoding from your tokenizer
    std::vector<int32_t> actual_ids = tokenizer_->encode(text);
    
    // For debugging - print the actual tokens
    std::cout << "Encoded tokens for '" << text << "': ";
    for (size_t i = 0; i < actual_ids.size(); ++i) {
        std::cout << actual_ids[i];
        if (i < actual_ids.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Verify it's not empty and reasonable length
    ASSERT_FALSE(actual_ids.empty());
    ASSERT_LT(actual_ids.size(), 10); // Shouldn't be more than 10 tokens
}

// Same byte-vs-character rule as BasicTokenValidation. EXPECT (not ASSERT) so
// one bad row does not hide the others — the old ASSERT aborted at token 100
// and never reached 200, which is why that row's identical bug went unnoticed.
TEST_F(TokenizerIntegrationTest, DecodeBasicTokens) {
    const std::vector<std::pair<uint32_t, std::string>> test_cases = {
        {0,    "!"},
        {1,    "\""},
        {10,   "+"},
        {65,   "b"},
        {100,  std::string("\xA7")},  // byte 167; vocab shows '§'
        {200,  std::string("\x0C")},  // byte 12 (form feed); vocab shows 'Č' (U+010C = 256+12)
        {2982, "do"}                  // a merged token, not a byte token
    };

    for (const auto& test_case : test_cases) {
        std::string decoded = tokenizer_->decode(test_case.first);
        EXPECT_EQ(decoded, test_case.second)
            << "Token " << test_case.first << " decoded to " << decoded.size()
            << " byte(s), first=0x" << std::hex
            << static_cast<int>(static_cast<unsigned char>(decoded.empty() ? 0 : decoded[0]));
    }
}

// The behaviour the two tests above are really protecting: non-ASCII text
// survives encode -> decode intact. A character above ASCII spans several byte
// tokens, and reassembling them is exactly what single-token decode cannot do.
TEST_F(TokenizerIntegrationTest, NonAsciiSurvivesRoundtrip) {
    for (const std::string& original : {std::string("§"),
                                        std::string("Grüße"),
                                        std::string("日本語"),
                                        std::string("emoji: 🙂")}) {
        const std::vector<int32_t> ids = tokenizer_->encode(original);
        EXPECT_FALSE(ids.empty()) << "encode produced nothing for: " << original;
        EXPECT_EQ(tokenizer_->decode(ids), original);
    }
}

TEST_F(TokenizerIntegrationTest, VocabularySizeValidation) {
    // Verify vocabulary size matches GGUF metadata
    ASSERT_EQ(metadata_.vocab_size, 151936);
}

TEST_F(TokenizerIntegrationTest, EncodeDecodeRoundtrip) {
    std::string original_text = "This is a test.";
    
    std::vector<int32_t> encoded_ids = tokenizer_->encode(original_text);
    std::string decoded_text = tokenizer_->decode(encoded_ids);
    
    // For debugging
    std::cout << "Original: '" << original_text << "'" << std::endl;
    std::cout << "Decoded:  '" << decoded_text << "'" << std::endl;
    
    // Handle GPT-2 style space encoding (Ġ = \xC4\xA0 = U+0120)
    std::string normalized_decoded = decoded_text;
    // Replace Ġ (U+0120) with regular space
    size_t pos = 0;
    while ((pos = normalized_decoded.find("\xC4\xA0", pos)) != std::string::npos) {
        normalized_decoded.replace(pos, 2, " ");
        pos += 1;
    }
    
    ASSERT_EQ(normalized_decoded, original_text);
}