// test_chat_template.cpp
//
// Verifies the chat-template renderers produce:
//   - Qwen ChatML format  (<|im_start|> / <|im_end|>)
//   - Gemma format        (<start_of_turn> / <end_of_turn>)
// and that architecture → template dispatch via the registry is wired.

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../../src/loader/chat_template.h"
#include "../../src/models/model_registry.h"

// ── QwenChatTemplate ──────────────────────────────────────────────────────────

TEST(QwenChatTemplate, RendersIMTagsAndAssistantPrompt) {
    QwenChatTemplate tmpl;
    std::string out = tmpl.render({{"user", "hi"}}, true);
    EXPECT_EQ(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
}

TEST(QwenChatTemplate, RendersWithoutAssistantPrompt) {
    QwenChatTemplate tmpl;
    std::string out = tmpl.render({{"user", "hi"}}, false);
    EXPECT_EQ(out, "<|im_start|>user\nhi<|im_end|>\n");
}

TEST(QwenChatTemplate, TurnEndSuffix) {
    QwenChatTemplate tmpl;
    EXPECT_EQ(tmpl.turn_end_suffix(), "<|im_end|>\n");
}

TEST(QwenChatTemplate, MultiTurnDialog) {
    QwenChatTemplate tmpl;
    std::vector<ChatMessage> hist = {
        {"user", "hi"},
        {"assistant", "hello"},
        {"user", "and now?"},
    };
    std::string out = tmpl.render(hist, true);
    const std::string expected =
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nhello<|im_end|>\n"
        "<|im_start|>user\nand now?<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(out, expected);
}

// ── GemmaChatTemplate ─────────────────────────────────────────────────────────

TEST(GemmaChatTemplate, SingleUserTurnWithAssistantPrompt) {
    GemmaChatTemplate tmpl;
    std::string out = tmpl.render({{"user", "hi"}}, true);
    EXPECT_EQ(out, "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n");
}

TEST(GemmaChatTemplate, AssistantRoleMapsToModel) {
    GemmaChatTemplate tmpl;
    std::string out = tmpl.render({{"assistant", "yo"}}, false);
    EXPECT_EQ(out, "<start_of_turn>model\nyo<end_of_turn>\n");
}

TEST(GemmaChatTemplate, SystemRoleEmittedAsUser) {
    GemmaChatTemplate tmpl;
    std::string out = tmpl.render({{"system", "you are helpful"}}, false);
    EXPECT_EQ(out, "<start_of_turn>user\nyou are helpful<end_of_turn>\n");
}

TEST(GemmaChatTemplate, MultiTurnDialog) {
    GemmaChatTemplate tmpl;
    std::vector<ChatMessage> hist = {
        {"user", "hi"},
        {"assistant", "hello"},
        {"user", "and now?"},
    };
    std::string out = tmpl.render(hist, true);
    const std::string expected =
        "<start_of_turn>user\nhi<end_of_turn>\n"
        "<start_of_turn>model\nhello<end_of_turn>\n"
        "<start_of_turn>user\nand now?<end_of_turn>\n"
        "<start_of_turn>model\n";
    EXPECT_EQ(out, expected);
}

TEST(GemmaChatTemplate, WithoutAssistantPromptOmitsOpenTurn) {
    GemmaChatTemplate tmpl;
    std::string out = tmpl.render({{"user", "hi"}}, false);
    EXPECT_EQ(out, "<start_of_turn>user\nhi<end_of_turn>\n");
}

TEST(GemmaChatTemplate, TurnEndSuffix) {
    GemmaChatTemplate tmpl;
    EXPECT_EQ(tmpl.turn_end_suffix(), "<end_of_turn>\n");
}

// ── Registry dispatch ─────────────────────────────────────────────────────────

TEST(ChatTemplateRegistry, QwenFamilyLookupRendersChatML) {
    register_builtin_models();
    for (const auto& arch : {"qwen2", "qwen3", "qwen35", "qwen35moe"}) {
        const ChatTemplate* tmpl = lookup_chat_template(arch);
        ASSERT_NE(tmpl, nullptr) << "null template for " << arch;
        std::string out = tmpl->render({{"user", "x"}}, true);
        EXPECT_EQ(out, "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n")
            << "wrong template for " << arch;
    }
}

TEST(ChatTemplateRegistry, GemmaLookupRendersGemmaFormat) {
    register_builtin_models();
    const ChatTemplate* tmpl = lookup_chat_template("gemma");
    ASSERT_NE(tmpl, nullptr);
    std::string out = tmpl->render({{"user", "x"}}, true);
    EXPECT_EQ(out, "<start_of_turn>user\nx<end_of_turn>\n<start_of_turn>model\n");
}

TEST(ChatTemplateRegistry, UnknownArchThrows) {
    try {
        lookup_chat_template("definitely_not_registered_xyz");
        FAIL() << "expected runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("lookup_chat_template"),
                  std::string::npos) << e.what();
    }
}
