#include "tool_request.h"

using json = nlohmann::json;

namespace qinf::server {

namespace {

const char* kNotYet = "tool calling is not supported yet, docs/plan-tool-calling.md";

// What a client sent, short enough for an error message.
std::string describe(const json& v) {
    if (v.is_array())  return "an array of " + std::to_string(v.size());
    if (v.is_object()) return "an object";
    std::string s = v.dump();
    if (s.size() > 40) s = s.substr(0, 37) + "...";
    return s;
}

}  // namespace

std::string refuse_tool_fields(const json& body) {
    if (!body.is_object()) return "";
    for (const char* field : {"tools", "tool_choice", "parallel_tool_calls",
                              "functions", "function_call"}) {
        if (body.contains(field) && !body[field].is_null())
            return std::string("'") + field + "': expected absent (" + kNotYet +
                   "), actual: " + describe(body[field]);
    }
    if (!body.contains("messages") || !body["messages"].is_array()) return "";
    const json& messages = body["messages"];
    for (size_t i = 0; i < messages.size(); ++i) {
        const json& m = messages[i];
        if (!m.is_object()) continue;
        const std::string at = "'messages[" + std::to_string(i) + "]";
        const std::string role = m.contains("role") && m["role"].is_string()
                                     ? m["role"].get<std::string>() : "";
        if (role == "tool" || role == "function")
            return at + ".role': expected \"system\", \"user\" or \"assistant\" (" + kNotYet +
                   "), actual: \"" + role + "\"";
        if (m.contains("tool_calls") && !m["tool_calls"].is_null() &&
            !(m["tool_calls"].is_array() && m["tool_calls"].empty()))
            return at + ".tool_calls': expected absent (" + kNotYet + "), actual: " +
                   describe(m["tool_calls"]);
        if (m.contains("function_call") && !m["function_call"].is_null())
            return at + ".function_call': expected absent (" + kNotYet + "), actual: " +
                   describe(m["function_call"]);
    }
    return "";
}

}  // namespace qinf::server
