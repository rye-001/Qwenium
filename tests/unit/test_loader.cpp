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

// ── Fixture: minimal byte-level vocabulary ────────────────────────────────────
//
// Mirrors Tokenizer::initialize_byte_mapping() exactly so that
// encode(x) → decode() round-trips any ASCII text without a real model file.
// 256 byte tokens (indices 0–255) + one EOS control token at index 256.
static Qwen3Metadata make_byte_level_metadata() {
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

    Qwen3Metadata m;
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

// ── GGUFLoader build tests ────────────────────────────────────────────────────

TEST(Loader, GGUFLoaderDefaultConstructs) {
    QwenGGUFLoader loader;
    EXPECT_FALSE(loader.is_loaded());
}

TEST(Loader, CreateLoaderFactoryReturnsNonNull) {
    auto loader = create_gguf_loader();
    ASSERT_NE(loader, nullptr);
    EXPECT_FALSE(loader->is_loaded());
}

TEST(Loader, LoadNonexistentPathThrows) {
    QwenGGUFLoader loader;
    EXPECT_THROW(
        loader.load_model("/tmp/qwen_inference_test_nonexistent_loader.gguf"),
        std::exception
    );
}

// ── Tokenizer build tests ─────────────────────────────────────────────────────

TEST(Loader, TokenizerVocabSizeMatchesInput) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    EXPECT_EQ(tok.get_vocabulary().size(), 257u);
}

TEST(Loader, TokenizerReportsEOSTokenId) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    EXPECT_EQ(tok.get_eos_token_id(), 256);
}

TEST(Loader, TokenizerDecodeEOSReturnsControlString) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);
    // EOS is CONTROL type; decode returns the token string verbatim.
    EXPECT_EQ(tok.decode(256), "<|endoftext|>");
}

// ── Known-answer round-trip tests ─────────────────────────────────────────────
//
// No BPE merges in this vocabulary.  Every ASCII char encodes to exactly one
// byte token and decodes back via the GPT-2 byte mapping.

TEST(Loader, TokenizerRoundTripSingleWord) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "Hello";
    auto ids = tok.encode(text);
    ASSERT_EQ(ids.size(), 5u);  // H e l l o → 5 byte tokens (no merges)
    EXPECT_EQ(tok.decode(ids), text);
}

TEST(Loader, TokenizerRoundTripPhraseWithSpace) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "hello world";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}

TEST(Loader, TokenizerRoundTripDigitsAndPunctuation) {
    Qwen3Metadata meta = make_byte_level_metadata();
    Tokenizer tok(&meta);

    const std::string text = "count 42!";
    EXPECT_EQ(tok.decode(tok.encode(text)), text);
}
