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
