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
