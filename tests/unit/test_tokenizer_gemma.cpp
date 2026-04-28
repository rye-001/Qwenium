// test_tokenizer_gemma.cpp — PR G1.3
//
// Unit tests for the llama / SentencePiece-derived path inside Tokenizer
// (Gemma uses tokenizer.ggml.model = "llama"). No model file required —
// builds a minimal hand-crafted vocabulary that covers the failure modes
// listed in docs/plan-gemma-impl.md:
//   - ▁ (U+2581) ↔ space normalization
//   - byte-fallback for arbitrary UTF-8 inputs
//   - special tokens emitted as a single id (no decomposition)
//   - leading-space sensitivity
//
// The vocab here is a toy. Reference logits against
// transformers.AutoTokenizer.from_pretrained("google/gemma-2b") are
// validated separately by the integration test that consumes the harness
// from PR G1.0.

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../../src/loader/tokenizer.h"
#include "../../src/loader/tokenizer_config.h"

namespace {

constexpr const char* SPM_UNDER = "\xE2\x96\x81";  // U+2581 ▁

// Build a minimal SP-BPE-merge-compatible Gemma-like vocabulary.
//
// Unlike a Viterbi-over-substrings tokenizer, BPE merge can only build a
// composite token by repeatedly merging adjacent tokens whose concatenation
// is itself in the vocab. So to get from "hello" → ["hello"] we must include
// every intermediate step (h, e, l, o, he, hel, hell, hello). Scores are
// arranged so larger merges have *less negative* scores (= higher score),
// i.e. BPE prefers them — mirroring how real SentencePiece scores rank
// frequent words above rare letter pairs.
//
// IDs (in order):
//   0  <pad>             CONTROL
//   1  <eos>             CONTROL
//   2  <bos>             CONTROL
//   3  <unk>             UNKNOWN
//   4..259 <0x00>..<0xFF> BYTE   (score 0)
//   260 <start_of_turn>  USER_DEFINED
//   261 <end_of_turn>    USER_DEFINED
//   then NORMAL tokens listed below
// TokenizerConfig matching Gemma's SentencePiece BPE with byte fallback.
static TokenizerConfig gemma_test_config() {
    TokenizerConfig cfg;
    cfg.normalizer          = NormalizerKind::SpaceToUnderscore;
    cfg.byte_fallback       = true;
    cfg.extra_chat_specials = {"<start_of_turn>", "<end_of_turn>"};
    return cfg;
}

ModelMetadata make_minimal_gemma_vocab() {
    ModelMetadata m;
    m.bos_token_id     = 2;
    m.eos_token_id     = 1;
    m.padding_token_id = 0;
    m.unknown_token_id = 3;

    std::vector<std::string> ids   = {"<pad>", "<eos>", "<bos>", "<unk>"};
    std::vector<TokenType>   types = {TokenType::CONTROL, TokenType::CONTROL,
                                      TokenType::CONTROL, TokenType::UNKNOWN};
    std::vector<float>       scores = {0, 0, 0, 0};

    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        ids.emplace_back(buf);
        types.push_back(TokenType::BYTE);
        scores.push_back(0.0f);  // Real Gemma byte tokens also have score 0.
    }

    auto add = [&](const std::string& tok, TokenType tt, float score) {
        ids.push_back(tok);
        types.push_back(tt);
        scores.push_back(score);
    };

    add("<start_of_turn>", TokenType::USER_DEFINED, 0.0f);
    add("<end_of_turn>",   TokenType::USER_DEFINED, 0.0f);

    // Single-char NORMAL tokens (so initial chars can be looked up rather
    // than byte-fallback'd). Any not listed will byte-fallback.
    for (char c : std::string("hellowrd?")) add(std::string(1, c), TokenType::NORMAL, -10.0f);
    add(SPM_UNDER, TokenType::NORMAL, -10.0f);

    // Multi-char merges. Less negative = higher score = earlier merge.
    add("he",      TokenType::NORMAL, -8.0f);
    add("hel",     TokenType::NORMAL, -7.0f);
    add("hell",    TokenType::NORMAL, -6.0f);
    add("hello",   TokenType::NORMAL, -5.0f);
    add("wo",      TokenType::NORMAL, -8.0f);
    add("wor",     TokenType::NORMAL, -7.0f);
    add("worl",    TokenType::NORMAL, -6.0f);
    add("world",   TokenType::NORMAL, -5.0f);
    add(std::string(SPM_UNDER) + "h",     TokenType::NORMAL, -8.0f);
    add(std::string(SPM_UNDER) + "he",    TokenType::NORMAL, -7.0f);
    add(std::string(SPM_UNDER) + "hel",   TokenType::NORMAL, -6.0f);
    add(std::string(SPM_UNDER) + "hell",  TokenType::NORMAL, -5.5f);
    add(std::string(SPM_UNDER) + "hello", TokenType::NORMAL, -3.0f);  // strong bias
    add(std::string(SPM_UNDER) + "w",     TokenType::NORMAL, -8.0f);
    add(std::string(SPM_UNDER) + "wo",    TokenType::NORMAL, -7.0f);
    add(std::string(SPM_UNDER) + "wor",   TokenType::NORMAL, -6.0f);
    add(std::string(SPM_UNDER) + "worl",  TokenType::NORMAL, -5.5f);
    add(std::string(SPM_UNDER) + "world", TokenType::NORMAL, -3.0f);

    m.id_to_token = std::move(ids);
    m.token_types = std::move(types);
    m.scores      = std::move(scores);
    m.vocab_size  = m.id_to_token.size();
    return m;
}

int find_id(const ModelMetadata& m, const std::string& s) {
    for (size_t i = 0; i < m.id_to_token.size(); ++i) {
        if (m.id_to_token[i] == s) return static_cast<int>(i);
    }
    return -1;
}

} // namespace

TEST(GemmaTokenizer, ConstructsFromLlamaMetadata) {
    auto m = make_minimal_gemma_vocab();
    EXPECT_NO_THROW({ Tokenizer t(&m, gemma_test_config()); });
}

TEST(GemmaTokenizer, EncodeWordViaBPEMergeReachesFullToken) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    // No leading space. The BPE merge ladder he → hel → hell → hello must
    // collapse [h,e,l,l,o] into the single "hello" token. The toy vocab is
    // arranged so this is the highest-scoring fully-collapsed sequence.
    auto ids = t.encode("hello");
    ASSERT_EQ(ids.size(), 1u) << "expected single-token segmentation";
    EXPECT_EQ(ids[0], find_id(m, "hello"));
}

TEST(GemmaTokenizer, EncodeAddsUnderscoreForSpaceSeparatedWords) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    // "hello world" normalizes to "hello▁world". Without a dummy prefix, the
    // first word is plain "hello" and the second is "▁world".
    auto ids = t.encode("hello world");
    ASSERT_EQ(ids.size(), 2u) << "expected ['hello', '▁world']";
    EXPECT_EQ(ids[0], find_id(m, "hello"));
    EXPECT_EQ(ids[1], find_id(m, std::string(SPM_UNDER) + "world"));
}

TEST(GemmaTokenizer, EncodeUsesByteFallbackForUnknownByte) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    // '@' (0x40) is in no NORMAL token of the toy vocab — must byte-fallback.
    auto ids = t.encode("@");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 4 + 0x40);
}

TEST(GemmaTokenizer, EncodeSpecialTokenEmitsSingleId) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    auto ids = t.encode("<start_of_turn>");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], find_id(m, "<start_of_turn>"));
}

TEST(GemmaTokenizer, EncodeSpecialTokenSurroundedByText) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    // "hello<start_of_turn>world" → ["hello", <start_of_turn>, "world"].
    auto ids = t.encode("hello<start_of_turn>world");
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], find_id(m, "hello"));
    EXPECT_EQ(ids[1], find_id(m, "<start_of_turn>"));
    EXPECT_EQ(ids[2], find_id(m, "world"));
}

TEST(GemmaTokenizer, DecodeReplacesUnderscoreWithSpace) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    int32_t under_world = find_id(m, std::string(SPM_UNDER) + "world");
    ASSERT_GE(under_world, 0);
    EXPECT_EQ(t.decode(under_world), " world");
}

TEST(GemmaTokenizer, DecodeByteFallbackEmitsRawByte) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    // '?' → byte token at id 4 + 0x3F.
    EXPECT_EQ(t.decode(4 + 0x40), "@");
}

TEST(GemmaTokenizer, DecodeSpecialTokenReturnsLiteral) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    int32_t sot = find_id(m, "<start_of_turn>");
    ASSERT_GE(sot, 0);
    EXPECT_EQ(t.decode(sot), "<start_of_turn>");
}

TEST(GemmaTokenizer, RoundTripASCII) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    const std::string text = "hello world";
    auto ids = t.encode(text);
    EXPECT_EQ(t.decode(ids), text);
}

TEST(GemmaTokenizer, RoundTripWithBytesAndSpecial) {
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    // 'h' and 'l' are NORMAL chars in the toy vocab; '@' falls back to its
    // byte token. Combined with a special token in the middle, this
    // exercises the complete encode/decode round-trip path.
    const std::string text = "h@<start_of_turn>l";
    auto ids = t.encode(text);
    EXPECT_EQ(t.decode(ids), text);
}

TEST(GemmaTokenizer, BPEMergePicksHigherScoreWhenAvailable) {
    // " hello" normalizes to "▁hello". The merge ladder ▁h → ▁he → ... →
    // ▁hello reaches the full token, which has the strongest score among
    // achievable merges.
    auto m = make_minimal_gemma_vocab();
    Tokenizer t(&m, gemma_test_config());

    auto ids = t.encode(" hello");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], find_id(m, std::string(SPM_UNDER) + "hello"));
}
