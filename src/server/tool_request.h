#pragma once
// tool_request.h — the tool-calling fields of a /v1/chat/completions request.
//
// Phase P0 of docs/plan-tool-calling.md: the server does not call tools yet, so
// every tool field is REFUSED by name instead of being silently ignored (which
// served a tool request as plain chat — the model never saw the tools, a
// tool_calls-only assistant turn rendered empty, role "tool" rendered as a
// "tool" turn the model was never trained on). Later phases grow this file into
// the request validation and the tool_calls response mapping.
//
// Refused (the caller answers 400 with {"error": <message>}):
//   top level: tools, tool_choice, parallel_tool_calls, functions, function_call
//              (any value but null)
//   messages[i]: role "tool" or "function"; tool_calls (anything but null or
//              an empty array); function_call (any value but null)
//
// Unit test: tests/unit/test_tool_request.cpp

#include <string>

#include "nlohmann/json.hpp"

namespace qinf::server {

// "" when `body` carries no tool field; otherwise the refusal message, naming
// the field, the expected value and the actual value.
std::string refuse_tool_fields(const nlohmann::json& body);

}  // namespace qinf::server
