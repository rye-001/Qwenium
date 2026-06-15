#include "chat_template.h"

#include <sstream>
#include <string>

// ── QwenChatTemplate ─────────────────────────────────────────────────────────

// Remove every occurrence of `needle` from `s` (used to strip the Qwen
// "/no_think" / "/think" soft-switch tokens out of rendered content).
static void erase_all(std::string& s, const std::string& needle)
{
    std::string::size_type pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos)
        s.erase(pos, needle.size());
}

// Trim a single trailing space/tab left behind after stripping a soft-switch
// that sat at the end of a line (e.g. "translate this /no_think" → "translate
// this"). Keeps line breaks intact.
static std::string strip_soft_switches(std::string content)
{
    erase_all(content, "/no_think");
    erase_all(content, "/think");
    while (!content.empty() && (content.back() == ' ' || content.back() == '\t'))
        content.pop_back();
    return content;
}

std::string QwenChatTemplate::render(const std::vector<ChatMessage>& history,
                                     bool add_assistant_prompt,
                                     std::optional<bool> enable_thinking) const
{
    // Qwen "/no_think" is an input convention, not a contract knob: an explicit
    // enable_thinking wins; otherwise a "/no_think" in any message disables
    // thinking. Thinking is ON by default for the Qwen family.
    bool think;
    if (enable_thinking.has_value()) {
        think = *enable_thinking;
    } else {
        bool saw_no_think = false;
        for (const auto& m : history) {
            if (m.content.find("/no_think") != std::string::npos) { saw_no_think = true; break; }
        }
        think = !saw_no_think;
    }

    std::ostringstream out;
    for (const auto& m : history) {
        out << "<|im_start|>" << m.role << "\n"
            << strip_soft_switches(m.content) << "<|im_end|>\n";
    }
    if (add_assistant_prompt) {
        out << "<|im_start|>assistant\n";
        // The empty reasoning block is the hard signal Qwen3 is trained on to
        // skip thinking and go straight to the answer.
        if (!think)
            out << "<think>\n\n</think>\n\n";
    }
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
                                      bool add_assistant_prompt,
                                      std::optional<bool> /*enable_thinking*/) const
{
    // Gemma 1/2/3 have no reasoning channel — enable_thinking is inert here.
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
                                        bool add_assistant_prompt,
                                        std::optional<bool> enable_thinking) const
{
    // Gemma 4 thinking defaults OFF (nullopt → false): emitting the empty
    // thought-channel marker skips reasoning. enable_thinking=true omits it.
    const bool think = enable_thinking.value_or(false);

    std::ostringstream out;
    for (const auto& m : history) {
        out << "<|turn>" << gemma4_role_for(m.role) << "\n";
        // Thinking branch: the official template carries the thinking switch as
        // a <|think|> marker inside the system turn (HF add_generation_prompt).
        // The model needs this present to enter thinking mode; a forced empty
        // <|channel>thought\n<channel|> generation prompt WITHOUT it degenerates
        // on image input (docs/server-image-multirequest-bug.md §5).
        if (think && m.role == "system")
            out << "<|think|>\n";
        out << m.content << "<turn|>\n";
    }
    if (add_assistant_prompt) {
        out << "<|turn>model\n";
        // No-think branch: force an empty thought channel so the IT model skips
        // its reasoning and answers directly (the order-management default).
        // Think branch ends at "model\n" and lets the model open the channel.
        if (!think)
            out << "<|channel>thought\n<channel|>";
    }
    return out.str();
}

std::string Gemma4ChatTemplate::turn_end_suffix() const
{
    return "<turn|>\n";
}
