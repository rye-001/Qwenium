#pragma once
// tool_format.h — a tool request rendered in the model's own trained format
// (docs/plan-tool-calling.md, phase P1: rendering).
//
// Three dialects, one per text protocol (split, not parameterised — no call
// site uses two of them together):
//   HermesJson — Qwen3:            <tool_call>\n{"name": …, "arguments": …}\n</tool_call>
//   QwenXml    — Qwen3.5/3.6/3.8:  <tool_call>\n<function=N>\n<parameter=K>\nV\n</parameter>…
//   Gemma4     — Gemma 4:          <|tool_call>call:N{k:<|"|>v<|"|>}<tool_call|>
// The per-checkpoint differences inside a dialect (thinking default, whether
// past reasoning is re-rendered, the reasoning-effort sentence, how a scalar
// argument prints, Gemma's schema key filter) are parameters, set by the
// table of MEASURED templates: a GGUF's tokenizer.chat_template is identified
// by its sha256, and only a template whose renderings were captured and gated
// (tests/fixtures/chat_tools/<sha16>/) has a format. Anything else has none —
// the caller refuses the request, never guesses.
//
// The renderer is a hand transliteration of each template, gated byte for
// byte against Hugging Face's renderer and llama.cpp's Jinja engine on every
// measured template (gate G1). It reproduces the template, quirks included;
// what the template would silently drop or mangle is refused instead (throws
// std::invalid_argument naming the field, the expected and the actual value).
//
// Input normalisation (the server's job, assumed here): content null → "";
// tool_calls[].function.arguments parsed from its JSON string into an object,
// key order kept; content parts joined to text.
//
// Unit test: tests/unit/test_tool_format.cpp

#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace qinf::tools {

using ojson = nlohmann::ordered_json;

struct ToolCall {
    std::string id;
    std::string name;
    ojson       arguments;   // a JSON object, the client's key order
};

// One message of a tool request.
struct ToolTurn {
    std::string                role;          // "system" | "user" | "assistant" | "tool"
    std::string                content;       // null content → ""
    std::optional<std::string> reasoning;     // assistant reasoning_content
    std::vector<ToolCall>      tool_calls;    // assistant
    std::string                tool_call_id;  // tool
};

struct ToolChat {
    std::vector<ToolTurn>      messages;
    ojson                      tools;         // the request's tools array, as sent
    bool                       add_generation_prompt = true;
    std::optional<bool>        enable_thinking;    // absent = the template's default
    std::optional<bool>        preserve_thinking;  // only where the template has it
    std::optional<std::string> reasoning_effort;   // only where the template has it
};

enum class ToolDialect { HermesJson, QwenXml, Gemma4 };

const char* to_string(ToolDialect d);

class ToolFormat {
public:
    virtual ~ToolFormat() = default;
    virtual ToolDialect dialect() const = 0;

    // The whole prompt for a tool request, ending in the generation prompt when
    // chat.add_generation_prompt is set. Never includes the BOS token: the
    // tokenizer adds it (the Gemma 4 template's own leading bos_token is that
    // token). Throws std::invalid_argument on a request the template does not
    // render faithfully.
    virtual std::string render(const ToolChat& chat) const = 0;
};

// sha256 of the template text, the first 16 hex digits.
std::string template_sha16(const std::string& chat_template);

// The format of a measured template, by template_sha16; nullptr when the
// template was never measured.
const ToolFormat* lookup_tool_format(const std::string& sha16);

// The measured template hashes, in a fixed order (for the error message that
// lists them, and for the gate that walks every one).
std::vector<std::string> measured_tool_templates();

}  // namespace qinf::tools
