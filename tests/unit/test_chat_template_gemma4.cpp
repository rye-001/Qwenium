// test_chat_template_gemma4.cpp — PR G4.7 (patched after live-chat regression)
//
// Verifies the registry resolves "gemma4" to:
//   - Gemma4ChatTemplate — native <|turn>role\n...<turn|>\n format,
//     because Gemma 4 IT was trained on this and rendering with the
//     <start_of_turn>/<end_of_turn> markers shared by G1/2/3 makes the
//     model hallucinate <tool_call|> tokens (observed live).
//   - gemma4_tokenizer_config() (SpaceToUnderscore + byte fallback +
//     extra_chat_specials).
//
// Tool-calling Jinja and the thinking-channel content are out of scope;
// the renderer only emits the empty thinking-channel marker
// (<|channel>thought\n<channel|>) when opening an assistant turn so the
// IT model skips thinking emission and goes straight to user-facing
// content.

#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <vector>

#include "../../src/loader/chat_template.h"
#include "../../src/loader/tokenizer_config.h"
#include "../../src/models/model_registry.h"

// 1. Single-turn user prompt with assistant-prompt-open suffix renders the
//    native Gemma 4 IT format end-to-end.
TEST(Gemma4ChatTemplateRegistry, SingleTurnRendersNativeFormat) {
    register_builtin_models();
    const ChatTemplate* tmpl = lookup_chat_template("gemma4");
    ASSERT_NE(tmpl, nullptr) << "gemma4 not registered with a chat template";

    const std::string out = tmpl->render({{"user", "hi"}}, true);
    EXPECT_EQ(out,
              "<|turn>user\n"
              "hi<turn|>\n"
              "<|turn>model\n"
              "<|channel>thought\n<channel|>");
}

// 1b. Thinking defaults OFF for Gemma 4 (nullopt → false): the empty
//     thought-channel marker is emitted, byte-identical to the prior behaviour.
//     enable_thinking=true omits the marker so the model emits its channel.
TEST(Gemma4ChatTemplateRegistry, EnableThinkingTrueOmitsThoughtChannel) {
    register_builtin_models();
    const ChatTemplate* tmpl = lookup_chat_template("gemma4");
    ASSERT_NE(tmpl, nullptr);

    const std::string out = tmpl->render({{"user", "hi"}}, true, /*enable_thinking=*/true);
    EXPECT_EQ(out,
              "<|turn>user\n"
              "hi<turn|>\n"
              "<|turn>model\n");
}

// 2. Tokenizer config registry lookup — the specials promoted must be Gemma 4's
//    NATIVE markers. This test used to assert <start_of_turn>/<end_of_turn>,
//    the G1/2/3 pair, contradicting this file's own header: rendering Gemma 4
//    with those makes the model hallucinate <tool_call|> tokens (observed live),
//    which is why the recipe moved to <|turn>/<turn|> in the first place. The
//    assertion was never updated with the renderer, so it failed permanently.
TEST(Gemma4TokenizerConfigRegistry, GemmaFourLookupYieldsGemmaShape) {
    register_builtin_models();
    TokenizerConfig cfg = lookup_tokenizer_config("gemma4");
    EXPECT_EQ(cfg.normalizer, NormalizerKind::SpaceToUnderscore);
    EXPECT_TRUE(cfg.byte_fallback);
    EXPECT_TRUE(cfg.add_bos_token);

    // Turn delimiters and the thinking-channel delimiters must all be promoted
    // to single ids; a split <|turn> is what produced the live regression.
    for (const char* want : {"<|turn>", "<turn|>", "<|channel>", "<channel|>"}) {
        EXPECT_NE(std::find(cfg.extra_chat_specials.begin(),
                            cfg.extra_chat_specials.end(), want),
                  cfg.extra_chat_specials.end())
            << "gemma4 tokenizer config is missing special: " << want;
    }

    // And must NOT carry the Gemma 1/2/3 markers.
    for (const char* unwanted : {"<start_of_turn>", "<end_of_turn>"}) {
        EXPECT_EQ(std::find(cfg.extra_chat_specials.begin(),
                            cfg.extra_chat_specials.end(), unwanted),
                  cfg.extra_chat_specials.end())
            << "gemma4 must not promote the G1/2/3 marker: " << unwanted;
    }
}

// 3. Multi-turn rendering preserves role mapping (assistant → model) so
//    the IT model sees the same prompt shape it was trained on.
TEST(Gemma4ChatTemplateRegistry, MultiTurnAssistantMapsToModel) {
    register_builtin_models();
    const ChatTemplate* tmpl = lookup_chat_template("gemma4");
    ASSERT_NE(tmpl, nullptr);

    std::vector<ChatMessage> hist = {
        {"user",      "hi"},
        {"assistant", "hello"},
        {"user",      "and now?"},
    };
    const std::string out = tmpl->render(hist, true);
    const std::string expected =
        "<|turn>user\n"
        "hi<turn|>\n"
        "<|turn>model\n"
        "hello<turn|>\n"
        "<|turn>user\n"
        "and now?<turn|>\n"
        "<|turn>model\n"
        "<|channel>thought\n<channel|>";
    EXPECT_EQ(out, expected);
}

// 4. System messages get their own turn (not mapped to "user" like G1/2/3).
TEST(Gemma4ChatTemplateRegistry, SystemRendersAsSystemTurn) {
    register_builtin_models();
    const ChatTemplate* tmpl = lookup_chat_template("gemma4");
    ASSERT_NE(tmpl, nullptr);

    const std::string out = tmpl->render(
        {{"system", "be concise"}, {"user", "hi"}}, true);
    EXPECT_EQ(out,
              "<|turn>system\n"
              "be concise<turn|>\n"
              "<|turn>user\n"
              "hi<turn|>\n"
              "<|turn>model\n"
              "<|channel>thought\n<channel|>");
}

// 5. Without assistant prompt, the open turn (and thinking marker) are
//    omitted — useful for tokenizing finished conversations for eval.
TEST(Gemma4ChatTemplateRegistry, WithoutAssistantPromptOmitsOpenTurn) {
    register_builtin_models();
    const ChatTemplate* tmpl = lookup_chat_template("gemma4");
    ASSERT_NE(tmpl, nullptr);
    const std::string out = tmpl->render({{"user", "hi"}}, false);
    EXPECT_EQ(out, "<|turn>user\nhi<turn|>\n");
}

// 6. The turn-end suffix is the native Gemma 4 marker — confirms the
//    template the registry handed us is Gemma4ChatTemplate, not the
//    inherited G1/2/3 one.
TEST(Gemma4ChatTemplateRegistry, TurnEndSuffixIsNativeMarker) {
    register_builtin_models();
    const ChatTemplate* tmpl = lookup_chat_template("gemma4");
    ASSERT_NE(tmpl, nullptr);
    EXPECT_EQ(tmpl->turn_end_suffix(), "<turn|>\n");
}
