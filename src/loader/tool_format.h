#pragma once
// tool_format.h — a tool request rendered in the model's own trained format
// and the model's reply parsed back (docs/plan-tool-calling.md: P1 rendering,
// P2 parsing).
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
#include <stdexcept>
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

// One call the model made. The server assigns the OpenAI id.
struct ParsedCall {
    std::string name;
    ojson       arguments;   // a JSON object, in the order the model wrote it
};

// What the model generated after the generation prompt, split up.
struct ParsedReply {
    std::string             reasoning;   // the thinking text, trimmed; "" when none
    std::string             content;     // the text outside reasoning and calls, trimmed
    std::vector<ParsedCall> calls;       // empty: the model answered without a call
};

// The model's output does not parse as this format's tool calls, or a call
// does not fit its tool's schema (the checked subset: required parameters
// present; no unknown parameter when additionalProperties is false; each
// top-level value of its declared JSON type and, with an enum, one of it).
// The server answers 502 with the raw output; the text is never passed off as
// an answer. The message names the call, the field, the expected and the
// actual value.
class ToolCallError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The output constraint for tool_choice "required" (any of the tools) or a
// named function: a GBNF grammar for EXACTLY ONE call in this format — the
// server ends a grammar slot at its first accepting state, so parallel calls
// stay an "auto" feature. The call's form and the tool and parameter names are
// fixed; values are typed where the format and the schema allow (Qwen-XML and
// Gemma by the parameter's JSON type and enum, Hermes as any JSON object);
// required parameters and the rest of the schema subset are checked by parse.
// Reasoning is not in the grammar: the sampler's gate (Sampler::
// set_grammar_gate) leaves the model's reasoning block free and applies the
// grammar after it. gate_opener / gate_closer are the block's delimiter TOKENS
// as text ("" = no gate: thinking is off and the prompt closed the block).
struct ToolGrammar {
    std::string gbnf;
    std::string gate_opener;
    std::string gate_closer;
    bool        gate_start_inside = false;   // the generation prompt opened the block
};

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

    // The model's reply to render(chat): `generated` is the decoded text after
    // the generation prompt, the stop token excluded (a trailing end-of-turn or
    // Gemma's <|tool_response> marker is tolerated). Qwen-XML parameter values
    // are raw text and are typed by the tool's schema; the other dialects'
    // values carry their own types. Throws ToolCallError.
    virtual ParsedReply parse(const std::string& generated, const ToolChat& chat) const = 0;

    // The text that opens a call — one special token in every dialect, so
    // tool_choice "none" is enforced by banning that token.
    virtual const char* call_opener() const = 0;

    // Stop strings a tool request adds to the client's: Gemma 4 emits
    // <|tool_response> after a call and must stop there (Google's format page:
    // "an additional stop sequence"), or it invents the tool's answer.
    virtual std::vector<std::string> extra_stops() const = 0;

    // What closes the system turn every tool render opens with (it carries the
    // tools block): render(chat) up to and including the first occurrence is
    // the request's recurring, cacheable prefix.
    virtual const char* system_turn_end() const = 0;

    // The grammar for tool_choice "required" (only_tool null) or the named
    // function `*only_tool` (one of chat.tools; throws std::invalid_argument
    // otherwise).
    virtual ToolGrammar grammar(const ToolChat& chat, const std::string* only_tool) const = 0;
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
