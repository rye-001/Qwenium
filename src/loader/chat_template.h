#pragma once

#include <string>
#include <vector>

struct ChatMessage {
    std::string role;
    std::string content;
};

// Abstract chat-template renderer.  Each model recipe provides a concrete
// subclass registered alongside its ForwardPass factory.
//
// render()         — format `history` into the prompt string the model
//                   expects.  If `add_assistant_prompt` is true, append the
//                   open-ended assistant turn header so the model samples
//                   directly into the response.
//
// turn_end_suffix() — the string to append after each assistant generation
//                   turn (e.g. "<|im_end|>\n" for ChatML,
//                   "<end_of_turn>\n" for Gemma).
class ChatTemplate {
public:
    virtual ~ChatTemplate() = default;

    virtual std::string render(const std::vector<ChatMessage>& history,
                               bool add_assistant_prompt) const = 0;

    virtual std::string turn_end_suffix() const = 0;
};

// ChatML chat template used by all Qwen family architectures.
//   <|im_start|>role\n…<|im_end|>\n
class QwenChatTemplate : public ChatTemplate {
public:
    std::string render(const std::vector<ChatMessage>& history,
                       bool add_assistant_prompt) const override;
    std::string turn_end_suffix() const override;
};

// Gemma chat template:  <start_of_turn>role\n…<end_of_turn>\n
// "system" has no first-class slot — content is emitted as a "user" turn
// matching the HF reference template behaviour.
class GemmaChatTemplate : public ChatTemplate {
public:
    std::string render(const std::vector<ChatMessage>& history,
                       bool add_assistant_prompt) const override;
    std::string turn_end_suffix() const override;
};

// Gemma 4 (IT) native chat template:  <|turn>role\n…<turn|>\n
//
// Distinct from Gemma 1/2/3's <start_of_turn>/<end_of_turn> markers — the
// 26B-A4B-It model was trained on this format (per the GGUF
// tokenizer.chat_template), and rendering with the pre-G4 markers makes
// the model hallucinate tool-call tokens because the prompt structure it
// sees doesn't match its training distribution.
//
// "assistant" role maps to "model" (HF convention).  "system" is rendered
// as its own turn (no role mapping to "user" like G1/2/3 — Gemma 4's
// native template gives system content a first-class system turn).
//
// When add_assistant_prompt is true, the template also emits the empty
// thinking-channel marker (<|channel>thought\n<channel|>) so the IT model
// goes straight to user-facing output instead of emitting thinking tokens.
// Tool-calling rendering and the full thinking-channel are out of scope
// for G4 (per docs/plan-gemma-impl.md → Phase G4 → "Out of scope").
//
// turn_end_suffix returns "<turn|>\n" — but EOS = 106 in the GGUF resolves
// to <turn|>, so generation typically stops on the EOS token before the
// suffix is consulted by the CLI.
class Gemma4ChatTemplate : public ChatTemplate {
public:
    std::string render(const std::vector<ChatMessage>& history,
                       bool add_assistant_prompt) const override;
    std::string turn_end_suffix() const override;
};
