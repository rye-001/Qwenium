#include "chat_template.h"

#include <sstream>

// ── QwenChatTemplate ─────────────────────────────────────────────────────────

std::string QwenChatTemplate::render(const std::vector<ChatMessage>& history,
                                     bool add_assistant_prompt) const
{
    std::ostringstream out;
    for (const auto& m : history) {
        out << "<|im_start|>" << m.role << "\n"
            << m.content << "<|im_end|>\n";
    }
    if (add_assistant_prompt)
        out << "<|im_start|>assistant\n";
    return out.str();
}

std::string QwenChatTemplate::turn_end_suffix() const
{
    return "<|im_end|>\n";
}

// ── GemmaChatTemplate ─────────────────────────────────────────────────────────

static const char* gemma_role_for(const std::string& role)
{
    if (role == "assistant") return "model";
    if (role == "system")    return "user";
    return role.c_str();
}

std::string GemmaChatTemplate::render(const std::vector<ChatMessage>& history,
                                      bool add_assistant_prompt) const
{
    std::ostringstream out;
    for (const auto& m : history) {
        // if (m.content.empty()) continue;  // empty system/user turn = structural noise for Gemma
        out << "<start_of_turn>" << gemma_role_for(m.role) << "\n"
            << m.content << "<end_of_turn>\n";
    }
    if (add_assistant_prompt)
        out << "<start_of_turn>model\n";
    return out.str();
}

std::string GemmaChatTemplate::turn_end_suffix() const
{
    return "<end_of_turn>\n";
}

// ── Gemma4ChatTemplate (G4.7 patch) ──────────────────────────────────────────
//
// Native Gemma 4 IT format: <|turn>role\n...<turn|>\n.
// Differs from G1/2/3 in two ways:
//   1. Different turn markers (the IT model was trained on these — using
//      <start_of_turn>/<end_of_turn> here causes the model to drift into
//      <tool_call|> hallucinations because the prompt shape doesn't match).
//   2. "system" gets its own turn instead of being mapped to "user".
//
// Assistant → "model", as in G1/2/3.  When opening a generation prompt,
// also emit the empty thinking-channel marker so the IT model skips its
// thinking-channel emission and goes straight to user-facing content.

static const char* gemma4_role_for(const std::string& role)
{
    return (role == "assistant") ? "model" : role.c_str();
}

std::string Gemma4ChatTemplate::render(const std::vector<ChatMessage>& history,
                                        bool add_assistant_prompt) const
{
    std::ostringstream out;
    for (const auto& m : history) {
        out << "<|turn>" << gemma4_role_for(m.role) << "\n"
            << m.content << "<turn|>\n";
    }
    if (add_assistant_prompt) {
        out << "<|turn>model\n"
            << "<|channel>thought\n<channel|>";
    }
    return out.str();
}

std::string Gemma4ChatTemplate::turn_end_suffix() const
{
    return "<turn|>\n";
}
