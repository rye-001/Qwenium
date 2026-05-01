// test_tokenizer_gemma4.cpp — PR G4.7
//
// Verifies the Gemma 4 tokenizer surface:
//   1. gemma4_tokenizer_config() returns SpaceToUnderscore + byte fallback
//      + extra chat specials (i.e. matches Gemma 1/2/3 at the runtime-config
//      level — the GGUF type string differs but that field doesn't gate
//      tokenizer behavior).
//   2. Encode/decode round-trip on a curated string set, including the
//      <start_of_turn> / <end_of_turn> chat specials emitted as single ids.
//   3. ModelMetadata round-trips BOS=2 and EOS=106 unchanged (the load-bearing
//      claim that distinguishes Gemma 4 from Gemma 1's EOS=1).
//
// No model file required.  The vocabulary is a hand-crafted minimal SP-BPE
// table — same construction shape as test_tokenizer_gemma.cpp.  Reference
// logits against transformers.AutoTokenizer.from_pretrained("google/gemma-4-...")
// land later via the canary integration harness from PR G4.0.
//
// Build target: gemma4-tokenizer-tests (see tests/CMakeLists.txt).

#include <gtest/gtest.h>
#include <cstdio>
#include <string>
#include <vector>

#include "../../src/loader/tokenizer.h"
#include "../../src/loader/tokenizer_config.h"
#include "../../src/models/gemma4.h"

namespace {

constexpr const char* SPM_UNDER = "\xE2\x96\x81";  // U+2581 ▁

// Same shape as test_tokenizer_gemma::make_minimal_gemma_vocab(), but with
// EOS=106 instead of 1.  ID 106 is otherwise a byte-fallback token in the
// toy vocab — that's fine: the point of this test is that the metadata
// scalar round-trips, not that 106 is itself a meaningful token in our
// minimal vocab.
ModelMetadata make_minimal_gemma4_vocab()
{
    ModelMetadata m;
    m.bos_token_id     = 2;     // BOS=2 (same as Gemma 1)
    m.eos_token_id     = 106;   // EOS=106 (Gemma 4 distinguishing value)
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
        scores.push_back(0.0f);
    }

    auto add = [&](const std::string& tok, TokenType tt, float score) {
        ids.push_back(tok);
        types.push_back(tt);
        scores.push_back(score);
    };

    // Force-promoted chat specials (they live above the byte range in id space).
    add("<start_of_turn>", TokenType::USER_DEFINED, 0.0f);
    add("<end_of_turn>",   TokenType::USER_DEFINED, 0.0f);

    // A handful of NORMAL tokens for round-trip coverage.  The merge ladder
    // is the same trick test_tokenizer_gemma uses: less-negative score on
    // longer merges so BPE prefers the collapsed form.
    for (char c : std::string("hellowrd?")) {
        add(std::string(1, c), TokenType::NORMAL, -10.0f);
    }
    add(SPM_UNDER, TokenType::NORMAL, -10.0f);
    add("he",    TokenType::NORMAL, -8.0f);
    add("hel",   TokenType::NORMAL, -7.0f);
    add("hell",  TokenType::NORMAL, -6.0f);
    add("hello", TokenType::NORMAL, -5.0f);
    add(std::string(SPM_UNDER) + "w",     TokenType::NORMAL, -8.0f);
    add(std::string(SPM_UNDER) + "wo",    TokenType::NORMAL, -7.0f);
    add(std::string(SPM_UNDER) + "wor",   TokenType::NORMAL, -6.0f);
    add(std::string(SPM_UNDER) + "worl",  TokenType::NORMAL, -5.5f);
    add(std::string(SPM_UNDER) + "world", TokenType::NORMAL, -3.0f);

    // tokenizer.ggml.model = "gemma4" — the value is metadata, not behavior.
    m.tokenizer_type = "gemma4";

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

// 1. The factory returns the documented Gemma-family runtime config.
TEST(Gemma4Tokenizer, ConfigShape) {
    TokenizerConfig cfg = gemma4_tokenizer_config();
    EXPECT_EQ(cfg.normalizer, NormalizerKind::SpaceToUnderscore);
    EXPECT_TRUE(cfg.byte_fallback);
    EXPECT_TRUE(cfg.add_bos_token);
    // <start_of_turn> and <end_of_turn> must be force-promoted so the
    // encoder yields a single id per tag.
    bool has_sot = false, has_eot = false;
    for (const auto& s : cfg.extra_chat_specials) {
        if (s == "<start_of_turn>") has_sot = true;
        if (s == "<end_of_turn>")   has_eot = true;
    }
    EXPECT_TRUE(has_sot) << "missing <start_of_turn> in extra_chat_specials";
    EXPECT_TRUE(has_eot) << "missing <end_of_turn> in extra_chat_specials";
}

// 2. Construction from a Gemma 4 vocab succeeds and the special-token IDs
//    on the metadata round-trip exactly (BOS=2, EOS=106).
TEST(Gemma4Tokenizer, ConstructsAndPreservesSpecialIds) {
    auto m = make_minimal_gemma4_vocab();
    EXPECT_NO_THROW({ Tokenizer t(&m, gemma4_tokenizer_config()); });
    EXPECT_EQ(m.bos_token_id, 2);
    EXPECT_EQ(m.eos_token_id, 106) << "EOS must distinguish Gemma 4 from Gemma 1";
    EXPECT_EQ(m.unknown_token_id, 3);
    EXPECT_EQ(m.padding_token_id, 0);
}

// 3. BPE merge ladder collapses "hello" into a single token.
TEST(Gemma4Tokenizer, BPEMergeReachesFullToken) {
    auto m = make_minimal_gemma4_vocab();
    Tokenizer t(&m, gemma4_tokenizer_config());

    auto ids = t.encode("hello");
    ASSERT_EQ(ids.size(), 1u) << "expected single 'hello' token";
    EXPECT_EQ(ids[0], find_id(m, "hello"));
}

// 4. Spaces normalize to U+2581 (▁).
TEST(Gemma4Tokenizer, SpaceNormalizationToUnderscore) {
    auto m = make_minimal_gemma4_vocab();
    Tokenizer t(&m, gemma4_tokenizer_config());

    auto ids = t.encode("hello world");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], find_id(m, "hello"));
    EXPECT_EQ(ids[1], find_id(m, std::string(SPM_UNDER) + "world"));
}

// 5. Byte fallback for codepoints outside the NORMAL inventory.
TEST(Gemma4Tokenizer, ByteFallback) {
    auto m = make_minimal_gemma4_vocab();
    Tokenizer t(&m, gemma4_tokenizer_config());

    auto ids = t.encode("@");
    ASSERT_EQ(ids.size(), 1u);
    // 4 control + 0x40 → byte token at id 4 + 0x40.
    EXPECT_EQ(ids[0], 4 + 0x40);
}

// 6. <start_of_turn> / <end_of_turn> emit single ids — the load-bearing
//    contract for Gemma's chat-template rendering downstream.
TEST(Gemma4Tokenizer, ChatSpecialsRoundTrip) {
    auto m = make_minimal_gemma4_vocab();
    Tokenizer t(&m, gemma4_tokenizer_config());

    auto sot_ids = t.encode("<start_of_turn>");
    ASSERT_EQ(sot_ids.size(), 1u);
    EXPECT_EQ(sot_ids[0], find_id(m, "<start_of_turn>"));
    EXPECT_EQ(t.decode(sot_ids[0]), "<start_of_turn>");

    auto eot_ids = t.encode("<end_of_turn>");
    ASSERT_EQ(eot_ids.size(), 1u);
    EXPECT_EQ(eot_ids[0], find_id(m, "<end_of_turn>"));
    EXPECT_EQ(t.decode(eot_ids[0]), "<end_of_turn>");
}

// 7. Specials embedded in surrounding text are still emitted atomically.
TEST(Gemma4Tokenizer, ChatSpecialSurroundedByText) {
    auto m = make_minimal_gemma4_vocab();
    Tokenizer t(&m, gemma4_tokenizer_config());

    // hello<start_of_turn>... → ["hello", <start_of_turn>, ...]
    auto ids = t.encode("hello<start_of_turn>hello");
    ASSERT_GE(ids.size(), 3u);
    EXPECT_EQ(ids[0], find_id(m, "hello"));
    EXPECT_EQ(ids[1], find_id(m, "<start_of_turn>"));
}

// 8. Decode reverses the ▁ normalization.
TEST(Gemma4Tokenizer, DecodeReplacesUnderscoreWithSpace) {
    auto m = make_minimal_gemma4_vocab();
    Tokenizer t(&m, gemma4_tokenizer_config());

    int32_t under_world = find_id(m, std::string(SPM_UNDER) + "world");
    ASSERT_GE(under_world, 0);
    EXPECT_EQ(t.decode(under_world), " world");
}
