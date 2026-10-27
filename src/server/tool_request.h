#pragma once
// tool_request.h — the tool-calling contract of /v1/chat/completions
// (docs/plan-tool-calling.md §3–§4; P0 refused every tool field, P3 serves
// them).
//
// A request is a TOOL REQUEST when it carries `tools`, or a message with
// `tool_calls` or role "tool". It is parsed into a qinf::tools::ToolChat — the
// shape loader/tool_format renders in the checkpoint's own trained format —
// and every field the server does not serve is refused by name (400), never
// ignored:
//   tools               1..128 {"type":"function","function":{name, description?,
//                       parameters?}}; name ^[A-Za-z0-9_-]{1,64}$, unique; "strict":
//                       true refused (only a top-level schema subset is checked)
//   tool_choice         "auto" (default) | "none" | "required" |
//                       {"type":"function","function":{"name":N}} — the last two
//                       constrain the reply to exactly one call by grammar (P4)
//   parallel_tool_calls true (default); false is refused (not enforced in v1)
//   messages            system (first only), user, assistant (content, tool_calls
//                       with arguments as a JSON-object string, reasoning_content),
//                       tool (tool_call_id, content); content a string or text parts
//   enable_thinking, chat_template_kwargs {enable_thinking, preserve_thinking,
//                       reasoning_effort}, reasoning_effort — passed to the template
//                       (which refuses a switch it lacks)
//   refused with tools: stream, conversation_id, grammar, image parts
//   refused always:     functions, function_call, role "function" (legacy)
//
// Unit test: tests/unit/test_tool_request.cpp

#include <functional>
#include <string>

#include "nlohmann/json.hpp"
#include "loader/tool_format.h"

namespace qinf::server {

// "" unless `body` carries a legacy function-calling field (functions,
// function_call, role "function", an assistant function_call) — refused on
// every request, tool request or not, naming its replacement.
std::string refuse_legacy_tool_fields(const nlohmann::json& body);

// True when `body` is a tool request (see the header comment).
bool is_tool_request(const nlohmann::json& body);

enum class ToolChoice { Auto, None, Required, Named };

struct ToolRequest {
    qinf::tools::ToolChat chat;
    ToolChoice            choice = ToolChoice::Auto;
    std::string           named;   // ToolChoice::Named: the function's name
};

// The validated tool request, from the raw request body: it is re-parsed in
// key order (nlohmann::ordered_json), because a template renders the tools as
// the client wrote them and the server's json sorts keys. Throws
// std::invalid_argument naming the field, the expected and the actual value
// (→ 400).
ToolRequest parse_tool_request(const std::string& request_body);

// "call_" + 24 hex digits, unique within this process.
std::string new_tool_call_id();

// The OpenAI assistant message for a parsed reply: content (null when the
// reply is only calls), reasoning_content when the model reasoned, tool_calls
// with arguments as a compact JSON string. `new_id` names each call.
nlohmann::json tool_reply_message(const qinf::tools::ParsedReply& reply,
                                  const std::function<std::string()>& new_id);

}  // namespace qinf::server
