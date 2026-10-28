#include "tool_request.h"

#include <atomic>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>

using json = nlohmann::json;
using qinf::tools::ojson;

namespace qinf::server {

namespace {

[[noreturn]] void refuse(const std::string& field, const std::string& expected, const std::string& actual) {
    throw std::invalid_argument("'" + field + "': expected " + expected + ", actual: " + actual);
}

// What a client sent, short enough for an error message.
template <typename J>
std::string describe(const J& v) {
    if (v.is_array())  return "an array of " + std::to_string(v.size());
    if (v.is_object()) return "an object";
    std::string s = v.dump();
    if (s.size() > 40) s = s.substr(0, 37) + "...";
    return s;
}

bool present(const json& obj, const char* key) { return obj.contains(key) && !obj[key].is_null(); }
bool present(const ojson& obj, const char* key) { return obj.contains(key) && !obj[key].is_null(); }

// A message's content: a string, text parts joined, or null → "".
std::string content_text(const ojson& m, const std::string& at) {
    if (!m.contains("content") || m["content"].is_null()) return "";
    const ojson& c = m["content"];
    if (c.is_string()) return c.get<std::string>();
    if (!c.is_array()) refuse(at + ".content", "a string, an array of text parts or null", describe(c));
    std::string out;
    for (size_t i = 0; i < c.size(); ++i) {
        const ojson& p = c[i];
        const std::string pat = at + ".content[" + std::to_string(i) + "]";
        const std::string type = p.is_object() && p.contains("type") && p["type"].is_string()
                                     ? p["type"].get<std::string>() : "";
        if (type == "image_url" || type == "input_audio")
            refuse(pat + ".type", "\"text\" (images and audio are not served with tools in v1)", "\"" + type + "\"");
        if (type != "text" || !p.contains("text") || !p["text"].is_string())
            refuse(pat, "{\"type\":\"text\",\"text\":\"...\"}", describe(p));
        out += p["text"].get<std::string>();
    }
    return out;
}

bool valid_tool_name(const std::string& n) {
    if (n.empty() || n.size() > 64) return false;
    for (char c : n)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) return false;
    return true;
}

}  // namespace

std::string refuse_response_format(const json& body) {
    if (!body.is_object() || !present(body, "response_format")) return "";
    const json& f = body["response_format"];
    if (f.is_object() && f.size() == 1 && f.contains("type") && f["type"] == "text") return "";
    return "'response_format': expected absent or {\"type\":\"text\"} (json_object and json_schema are "
           "not served; constrain the output with 'grammar', GBNF), actual: " +
           (!f.is_object() || !f.contains("type") ? describe(f)
            : f["type"] == "text"                 ? std::string("type \"text\" with other keys")
                                                  : "type " + describe(f["type"]));
}

std::string refuse_legacy_tool_fields(const json& body) {
    if (!body.is_object()) return "";
    if (present(body, "functions"))
        return "'functions': expected absent (deprecated; send 'tools'), actual: " + describe(body["functions"]);
    if (present(body, "function_call"))
        return "'function_call': expected absent (deprecated; send 'tool_choice'), actual: " +
               describe(body["function_call"]);
    if (!body.contains("messages") || !body["messages"].is_array()) return "";
    const json& messages = body["messages"];
    for (size_t i = 0; i < messages.size(); ++i) {
        const json& m = messages[i];
        if (!m.is_object()) continue;
        const std::string at = "'messages[" + std::to_string(i) + "]";
        if (m.contains("role") && m["role"] == "function")
            return at + ".role': expected \"tool\" (role \"function\" is deprecated), actual: \"function\"";
        if (present(m, "function_call"))
            return at + ".function_call': expected absent (deprecated; send 'tool_calls'), actual: " +
                   describe(m["function_call"]);
    }
    return "";
}

bool is_tool_request(const json& body) {
    if (!body.is_object()) return false;
    if (present(body, "tools")) return true;
    if (!body.contains("messages") || !body["messages"].is_array()) return false;
    for (const json& m : body["messages"]) {
        if (!m.is_object()) continue;
        if (m.contains("role") && m["role"] == "tool") return true;
        if (present(m, "tool_calls") && !(m["tool_calls"].is_array() && m["tool_calls"].empty())) return true;
    }
    // tool_choice / parallel_tool_calls alone (no tools): a tool request
    // missing its tools, so parse_tool_request refuses it by name.
    return present(body, "tool_choice") || present(body, "parallel_tool_calls");
}

ToolRequest parse_tool_request(const std::string& request_body) {
    const ojson body = ojson::parse(request_body, nullptr, /*allow_exceptions=*/false);
    if (body.is_discarded() || !body.is_object()) refuse("body", "a JSON object", "invalid JSON");
    ToolRequest out;
    qinf::tools::ToolChat& chat = out.chat;

    // Refused with tools (v1).
    if (body.value("stream", false))
        refuse("stream", "false with tools (streamed tool calls are not served in v1)", "true");
    if (present(body, "conversation_id") && body["conversation_id"] != "")
        refuse("conversation_id", "absent with tools (not served in v1)", describe(body["conversation_id"]));
    if (present(body, "grammar") && body["grammar"] != "")
        refuse("grammar", "absent with tools (the tool format is the output constraint)", "a grammar");

    // tools
    if (!present(body, "tools"))
        refuse("tools", "the request's tool list (a request with tool calls, tool results or tool_choice "
                        "must send its tools)", "absent");
    const ojson& tools = body["tools"];
    if (!tools.is_array() || tools.empty() || tools.size() > 128)
        refuse("tools", "an array of 1 to 128 tools", describe(tools));
    std::set<std::string> names;
    for (size_t i = 0; i < tools.size(); ++i) {
        const ojson& t = tools[i];
        const std::string at = "tools[" + std::to_string(i) + "]";
        if (!t.is_object()) refuse(at, "an object", describe(t));
        if (!t.contains("type") || t["type"] != "function")
            refuse(at + ".type", "\"function\"", t.contains("type") ? describe(t["type"]) : "absent");
        if (!t.contains("function") || !t["function"].is_object())
            refuse(at + ".function", "an object", t.contains("function") ? describe(t["function"]) : "absent");
        const ojson& f = t["function"];
        if (!f.contains("name") || !f["name"].is_string() || !valid_tool_name(f["name"].get<std::string>()))
            refuse(at + ".function.name", "a name matching ^[A-Za-z0-9_-]{1,64}$",
                   f.contains("name") ? describe(f["name"]) : "absent");
        if (!names.insert(f["name"].get<std::string>()).second)
            refuse(at + ".function.name", "a name not used by another tool", describe(f["name"]));
        if (present(f, "description") && !f["description"].is_string())
            refuse(at + ".function.description", "a string", describe(f["description"]));
        if (present(f, "parameters")) {
            const ojson& p = f["parameters"];
            if (!p.is_object()) refuse(at + ".function.parameters", "a JSON Schema object", describe(p));
            if (p.contains("type") && p["type"] != "object")
                refuse(at + ".function.parameters.type", "\"object\"", describe(p["type"]));
        }
        if (present(f, "strict") && f["strict"] == true)
            refuse(at + ".function.strict", "false or absent (only a top-level schema subset is checked, "
                                            "docs/plan-tool-calling.md §5)", "true");
    }
    chat.tools = tools;

    // tool_choice, parallel_tool_calls
    if (present(body, "tool_choice")) {
        const ojson& c = body["tool_choice"];
        if (c == "auto") out.choice = ToolChoice::Auto;
        else if (c == "none") out.choice = ToolChoice::None;
        else if (c == "required") out.choice = ToolChoice::Required;
        else if (c.is_object()) {
            if (!c.contains("type") || c["type"] != "function")
                refuse("tool_choice.type", "\"function\"", c.contains("type") ? describe(c["type"]) : "absent");
            if (!c.contains("function") || !c["function"].is_object() || !c["function"].contains("name") ||
                !c["function"]["name"].is_string() || !names.count(c["function"]["name"].get<std::string>()))
                refuse("tool_choice.function.name", "the name of one of the request's tools",
                       c.contains("function") && c["function"].is_object() && c["function"].contains("name")
                           ? describe(c["function"]["name"]) : "absent");
            out.choice = ToolChoice::Named;
            out.named = c["function"]["name"].get<std::string>();
        } else {
            refuse("tool_choice", "\"auto\", \"none\", \"required\" or {\"type\":\"function\",\"function\":"
                                  "{\"name\":...}}", describe(c));
        }
    }
    // "none": the tools are not declared to the model at all (the template's
    // no-tools branch) — declaring them and banning the call opener
    // degenerates Gemma 4 (docs/plan-tool-calling.md P5). A call the model
    // makes anyway is a 502.
    chat.declare_tools = out.choice != ToolChoice::None;
    if (present(body, "parallel_tool_calls") && body["parallel_tool_calls"] != true)
        refuse("parallel_tool_calls", "true or absent (false is not enforced in v1)",
               describe(body["parallel_tool_calls"]));

    // messages
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
        refuse("messages", "a non-empty array", body.contains("messages") ? describe(body["messages"]) : "absent");
    std::set<std::string> call_ids;
    const ojson& messages = body["messages"];
    for (size_t i = 0; i < messages.size(); ++i) {
        const ojson& m = messages[i];
        const std::string at = "messages[" + std::to_string(i) + "]";
        if (!m.is_object()) refuse(at, "an object", describe(m));
        if (!m.contains("role") || !m["role"].is_string())
            refuse(at + ".role", "a string", m.contains("role") ? describe(m["role"]) : "absent");
        qinf::tools::ToolTurn t;
        t.role = m["role"].get<std::string>();
        if (t.role != "system" && t.role != "user" && t.role != "assistant" && t.role != "tool")
            refuse(at + ".role", "\"system\", \"user\", \"assistant\" or \"tool\"", "\"" + t.role + "\"");
        t.content = content_text(m, at);
        if (present(m, "reasoning_content")) {
            if (t.role != "assistant" || !m["reasoning_content"].is_string())
                refuse(at + ".reasoning_content", "a string on an assistant message", describe(m["reasoning_content"]));
            t.reasoning = m["reasoning_content"].get<std::string>();
        }
        if (present(m, "tool_calls")) {
            const ojson& calls = m["tool_calls"];
            if (t.role != "assistant" || !calls.is_array())
                refuse(at + ".tool_calls", "an array on an assistant message", describe(calls));
            for (size_t k = 0; k < calls.size(); ++k) {
                const ojson& c = calls[k];
                const std::string ck = at + ".tool_calls[" + std::to_string(k) + "]";
                if (!c.is_object() || !c.contains("function") || !c["function"].is_object())
                    refuse(ck + ".function", "an object", c.is_object() && c.contains("function")
                                                              ? describe(c["function"]) : "absent");
                if (c.contains("type") && c["type"] != "function")
                    refuse(ck + ".type", "\"function\"", describe(c["type"]));
                if (!c.contains("id") || !c["id"].is_string() || c["id"] == "")
                    refuse(ck + ".id", "a non-empty string", c.contains("id") ? describe(c["id"]) : "absent");
                const std::string id = c["id"].get<std::string>();
                if (!call_ids.insert(id).second) refuse(ck + ".id", "an id no other call uses", "\"" + id + "\"");
                const ojson& f = c["function"];
                if (!f.contains("name") || !f["name"].is_string() || !names.count(f["name"].get<std::string>()))
                    refuse(ck + ".function.name", "the name of one of the request's tools",
                           f.contains("name") ? describe(f["name"]) : "absent");
                ojson args = ojson::object();
                if (present(f, "arguments")) {
                    const ojson& a = f["arguments"];
                    if (a.is_string()) {
                        args = ojson::parse(a.get<std::string>(), nullptr, false);
                        if (args.is_discarded() || !args.is_object())
                            refuse(ck + ".function.arguments", "a JSON object, as a string", describe(a));
                    } else if (a.is_object()) {
                        args = a;
                    } else {
                        refuse(ck + ".function.arguments", "a JSON object, as a string", describe(a));
                    }
                }
                t.tool_calls.push_back({id, f["name"].get<std::string>(), std::move(args)});
            }
        }
        if (t.role == "tool") {
            if (!m.contains("tool_call_id") || !m["tool_call_id"].is_string())
                refuse(at + ".tool_call_id", "a string", m.contains("tool_call_id") ? describe(m["tool_call_id"]) : "absent");
            t.tool_call_id = m["tool_call_id"].get<std::string>();
        }
        chat.messages.push_back(std::move(t));
    }

    // Template switches. A top-level enable_thinking wins over the kwargs one,
    // as on the non-tool path.
    if (present(body, "chat_template_kwargs")) {
        const ojson& kw = body["chat_template_kwargs"];
        if (!kw.is_object()) refuse("chat_template_kwargs", "an object", describe(kw));
        for (auto it = kw.begin(); it != kw.end(); ++it) {
            const std::string k = it.key();
            const std::string at = "chat_template_kwargs." + k;
            if (k == "enable_thinking" || k == "preserve_thinking") {
                if (!it.value().is_boolean()) refuse(at, "a boolean", describe(it.value()));
                (k == "enable_thinking" ? chat.enable_thinking : chat.preserve_thinking) = it.value().get<bool>();
            } else if (k == "reasoning_effort") {
                if (!it.value().is_string()) refuse(at, "a string", describe(it.value()));
                chat.reasoning_effort = it.value().get<std::string>();
            } else {
                refuse(at, "one of enable_thinking, preserve_thinking, reasoning_effort", "present");
            }
        }
    }
    if (present(body, "enable_thinking")) {
        if (!body["enable_thinking"].is_boolean()) refuse("enable_thinking", "a boolean", describe(body["enable_thinking"]));
        chat.enable_thinking = body["enable_thinking"].get<bool>();
    }
    if (present(body, "reasoning_effort")) {
        if (!body["reasoning_effort"].is_string())
            refuse("reasoning_effort", "a string", describe(body["reasoning_effort"]));
        chat.reasoning_effort = body["reasoning_effort"].get<std::string>();
    }
    chat.add_generation_prompt = true;
    return out;
}

std::string new_tool_call_id() {
    static const uint64_t salt = [] {
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) ^ rd();
    }();
    static std::atomic<uint64_t> counter{0};
    const uint64_t n = counter.fetch_add(1);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "call_%016llx%08llx",
                  static_cast<unsigned long long>(salt ^ (n * 0x9e3779b97f4a7c15ull)),
                  static_cast<unsigned long long>(n & 0xffffffffull));
    return buf;
}

json tool_reply_message(const qinf::tools::ParsedReply& reply, const std::function<std::string()>& new_id) {
    json msg = {{"role", "assistant"}};
    if (reply.calls.empty()) msg["content"] = reply.content;
    else msg["content"] = reply.content.empty() ? json(nullptr) : json(reply.content);
    if (!reply.reasoning.empty()) msg["reasoning_content"] = reply.reasoning;
    if (!reply.calls.empty()) {
        json calls = json::array();
        for (const auto& c : reply.calls)
            calls.push_back({{"id", new_id()},
                             {"type", "function"},
                             {"function", {{"name", c.name}, {"arguments", c.arguments.dump()}}}});
        msg["tool_calls"] = std::move(calls);
    }
    return msg;
}

}  // namespace qinf::server
