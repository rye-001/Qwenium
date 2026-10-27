#include "tool_format.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace qinf::tools {

namespace {

[[noreturn]] void refuse(const std::string& field, const std::string& expected,
                         const std::string& actual) {
    throw std::invalid_argument("'" + field + "': expected " + expected + ", actual: " + actual);
}

// ── Python / Jinja value semantics ───────────────────────────────────────────
// The templates were written for Hugging Face's renderer (jinja2 on Python
// values); these reproduce the few behaviours the tool paths depend on.

// Python str.strip() whitespace: ASCII \t\n\v\f\r, \x1c-\x1f, space, and the
// Unicode White_Space code points.
bool py_space(uint32_t c) {
    return c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f) || c == 0x85 ||
           c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 ||
           c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

// The code point starting at s[i] and its byte length (invalid UTF-8 → the
// byte itself, length 1: never whitespace, so never stripped).
uint32_t cp_at(const std::string& s, size_t i, size_t& len) {
    const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char c = b(i);
    if (c < 0x80) { len = 1; return c; }
    const size_t n = (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 1;
    if (n == 1 || i + n > s.size()) { len = 1; return c; }
    uint32_t cp = c & (0xff >> (n + 1));
    for (size_t k = 1; k < n; ++k) {
        if ((b(i + k) & 0xc0) != 0x80) { len = 1; return c; }
        cp = (cp << 6) | (b(i + k) & 0x3f);
    }
    len = n;
    return cp;
}

// Jinja `|trim` = Python str.strip().
std::string py_strip(const std::string& s) {
    size_t a = 0, len = 0;
    while (a < s.size() && py_space(cp_at(s, a, len))) a += len;
    size_t e = s.size();
    while (e > a) {
        size_t start = e - 1;   // back up to the start of the last code point
        while (start > a && (static_cast<unsigned char>(s[start]) & 0xc0) == 0x80) --start;
        if (!py_space(cp_at(s, start, len)) || start + len != e) break;
        e = start;
    }
    return s.substr(a, e - a);
}

std::string lstrip_nl(const std::string& s) {
    const size_t a = s.find_first_not_of('\n');
    return a == std::string::npos ? "" : s.substr(a);
}
std::string rstrip_nl(const std::string& s) {
    const size_t e = s.find_last_not_of('\n');
    return e == std::string::npos ? "" : s.substr(0, e + 1);
}
std::string strip_nl(const std::string& s) { return lstrip_nl(rstrip_nl(s)); }

bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
bool ends_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Python repr(float): the shortest digits that round-trip, fixed notation for
// a decimal point position in (-4, 16], exponent notation (two-digit minimum)
// otherwise.
std::string py_float_repr(double x) {
    char buf[40];
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(buf, sizeof(buf), "%.*e", p - 1, x);
        if (std::strtod(buf, nullptr) == x) break;
    }
    std::string s = buf;
    std::string sign;
    if (s[0] == '-') { sign = "-"; s.erase(0, 1); }
    const size_t epos = s.find('e');
    const int exp10 = std::atoi(s.c_str() + epos + 1);
    std::string digits = s.substr(0, epos);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int decpt = exp10 + 1;
    const int nd = static_cast<int>(digits.size());
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) return sign + "0." + std::string(-decpt, '0') + digits;
        if (decpt >= nd) return sign + digits + std::string(decpt - nd, '0') + ".0";
        return sign + digits.substr(0, decpt) + "." + digits.substr(decpt);
    }
    char e[8];
    std::snprintf(e, sizeof(e), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    return sign + digits.substr(0, 1) + (nd > 1 ? "." + digits.substr(1) : "") + e;
}

// Python str() of a scalar — Jinja `{{ v }}` and `v|string`.
std::string py_str(const ojson& v, const std::string& field) {
    if (v.is_string())          return v.get<std::string>();
    if (v.is_boolean())         return v.get<bool>() ? "True" : "False";
    if (v.is_null())            return "None";
    if (v.is_number_unsigned()) return std::to_string(v.get<uint64_t>());
    if (v.is_number_integer())  return std::to_string(v.get<int64_t>());
    if (v.is_number_float())    return py_float_repr(v.get<double>());
    refuse(field, "a string, number, boolean or null", std::string("a ") + v.type_name());
}

// Python json.dumps(v, ensure_ascii=False) — the `tojson` filter Hugging Face
// installs: ", " and ": " separators, the given key order, only the JSON
// escapes (non-ASCII kept, control characters as lowercase \u00xx).
void py_dumps_into(const ojson& v, std::string& out) {
    switch (v.type()) {
    case ojson::value_t::object: {
        out += '{';
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            py_dumps_into(ojson(it.key()), out);
            out += ": ";
            py_dumps_into(it.value(), out);
        }
        out += '}';
        return;
    }
    case ojson::value_t::array: {
        out += '[';
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) out += ", ";
            py_dumps_into(v[i], out);
        }
        out += ']';
        return;
    }
    case ojson::value_t::string: {
        out += '"';
        for (const char ch : v.get_ref<const std::string&>()) {
            const unsigned char c = static_cast<unsigned char>(ch);
            switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) {
                    char e[8];
                    std::snprintf(e, sizeof(e), "\\u%04x", c);
                    out += e;
                } else {
                    out += ch;
                }
            }
        }
        out += '"';
        return;
    }
    case ojson::value_t::boolean: out += v.get<bool>() ? "true" : "false"; return;
    case ojson::value_t::null:    out += "null"; return;
    case ojson::value_t::number_float: out += py_float_repr(v.get<double>()); return;
    case ojson::value_t::number_unsigned: out += std::to_string(v.get<uint64_t>()); return;
    case ojson::value_t::number_integer:  out += std::to_string(v.get<int64_t>()); return;
    default: refuse("tojson", "a JSON value", v.type_name());
    }
}

std::string py_dumps(const ojson& v) {
    std::string out;
    py_dumps_into(v, out);
    return out;
}

// Jinja truthiness of an optional member: absent, null, false, 0, "" and empty
// containers are false.
bool truthy(const ojson& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key)) return false;
    const ojson& v = obj[key];
    if (v.is_null())    return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number())  return v.get<double>() != 0.0;
    if (v.is_string() || v.is_array() || v.is_object()) return !v.empty();
    return true;
}

std::string ascii_upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// Jinja `|dictsort`: the items sorted by key, case-insensitively, stable.
std::vector<std::pair<std::string, const ojson*>> dictsort(const ojson& obj) {
    std::vector<std::pair<std::string, const ojson*>> items;
    for (auto it = obj.begin(); it != obj.end(); ++it) items.emplace_back(it.key(), &it.value());
    std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        const std::string x = a.first, y = b.first;
        return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end(), [](char p, char q) {
            return std::tolower(static_cast<unsigned char>(p)) < std::tolower(static_cast<unsigned char>(q));
        });
    });
    return items;
}

// ── Shared request checks ────────────────────────────────────────────────────
// Every dialect renders the same conversation shapes; what one template would
// silently drop (a tool result with no matching call — Gemma renders tool
// results only through their call) is refused for all, so the contract does
// not depend on the family.

void check_chat(const ToolChat& chat) {
    if (!chat.tools.is_array() || chat.tools.empty())
        refuse("tools", "a non-empty array", chat.tools.is_array() ? "an empty array" : chat.tools.type_name());
    for (size_t i = 0; i < chat.tools.size(); ++i) {
        const ojson& t = chat.tools[i];
        const std::string at = "tools[" + std::to_string(i) + "]";
        if (!t.is_object() || !t.contains("function") || !t["function"].is_object())
            refuse(at + ".function", "an object", t.is_object() && t.contains("function")
                                                      ? t["function"].type_name() : "absent");
        if (!t["function"].contains("name") || !t["function"]["name"].is_string())
            refuse(at + ".function.name", "a string", "absent or not a string");
    }
    if (chat.messages.empty()) refuse("messages", "at least one message", "none");
    bool has_user = false;
    for (size_t i = 0; i < chat.messages.size(); ++i) {
        const ToolTurn& m = chat.messages[i];
        const std::string at = "messages[" + std::to_string(i) + "]";
        if (m.role == "user") has_user = true;
        if (m.role == "system") {
            if (i != 0) refuse(at + ".role", "\"system\" only as the first message", "\"system\" at " + std::to_string(i));
        } else if (m.role != "user" && m.role != "assistant" && m.role != "tool") {
            refuse(at + ".role", "\"system\", \"user\", \"assistant\" or \"tool\"", "\"" + m.role + "\"");
        }
        if (m.role != "assistant" && !m.tool_calls.empty())
            refuse(at + ".tool_calls", "absent on a \"" + m.role + "\" message", "an array of " +
                   std::to_string(m.tool_calls.size()));
        if (m.role != "assistant" && m.reasoning)
            refuse(at + ".reasoning_content", "absent on a \"" + m.role + "\" message", "a string");
        for (size_t k = 0; k < m.tool_calls.size(); ++k) {
            const ToolCall& c = m.tool_calls[k];
            const std::string ck = at + ".tool_calls[" + std::to_string(k) + "]";
            if (c.name.empty()) refuse(ck + ".function.name", "a non-empty name", "\"\"");
            if (!c.arguments.is_object())
                refuse(ck + ".function.arguments", "a JSON object", c.arguments.type_name());
        }
        if (m.role == "tool") {
            size_t j = i;
            while (j > 0 && chat.messages[j - 1].role == "tool") --j;
            if (j == 0 || chat.messages[j - 1].role != "assistant" || chat.messages[j - 1].tool_calls.empty())
                refuse(at, "a tool result following an assistant message with tool_calls",
                       j == 0 ? "the first message" : "after a \"" + chat.messages[j - 1].role + "\" message");
            const auto& calls = chat.messages[j - 1].tool_calls;
            if (std::none_of(calls.begin(), calls.end(), [&](const ToolCall& c) { return c.id == m.tool_call_id; }))
                refuse(at + ".tool_call_id", "the id of a call in messages[" + std::to_string(j - 1) + "]",
                       "\"" + m.tool_call_id + "\"");
        }
    }
    if (!has_user) refuse("messages", "at least one \"user\" message", "none");
}

void refuse_kwarg_absent_from_template(const std::optional<bool>& v, const char* name) {
    if (v) refuse(name, "absent (this checkpoint's template has no such switch)", *v ? "true" : "false");
}
void refuse_kwarg_absent_from_template(const std::optional<std::string>& v, const char* name) {
    if (v) refuse(name, "absent (this checkpoint's template has no such switch)", "\"" + *v + "\"");
}

// ── HermesJson: Qwen3 ────────────────────────────────────────────────────────

const char* kHermesToolsHead =
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const char* kHermesToolsTail =
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
    "\"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

class HermesJsonToolFormat : public ToolFormat {
public:
    // How the template tells a real user query from a tool result posted as a
    // user message: Qwen3-0.6B tests that both tags occur, Qwen3-1.7B that the
    // content starts and ends with them.
    enum class ToolResponseTest { Contains, StartsAndEnds };
    explicit HermesJsonToolFormat(ToolResponseTest t) : test_(t) {}
    ToolDialect dialect() const override { return ToolDialect::HermesJson; }

    std::string render(const ToolChat& chat) const override {
        check_chat(chat);
        refuse_kwarg_absent_from_template(chat.preserve_thinking, "preserve_thinking");
        refuse_kwarg_absent_from_template(chat.reasoning_effort, "reasoning_effort");
        const auto& msgs = chat.messages;
        const size_t n = msgs.size();

        std::string out = "<|im_start|>system\n";
        if (msgs[0].role == "system") out += msgs[0].content + "\n\n";
        out += kHermesToolsHead;
        for (const auto& t : chat.tools) out += "\n" + py_dumps(t);
        out += kHermesToolsTail;

        size_t last_query = n - 1;
        for (size_t k = n; k-- > 0;) {
            const std::string& c = msgs[k].content;
            const bool is_result = test_ == ToolResponseTest::Contains
                ? c.find("<tool_response>") != std::string::npos && c.find("</tool_response>") != std::string::npos
                : starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>");
            if (msgs[k].role == "user" && !is_result) { last_query = k; break; }
        }

        for (size_t i = 0; i < n; ++i) {
            const ToolTurn& m = msgs[i];
            if (m.role == "user") {
                out += "<|im_start|>user\n" + m.content + "<|im_end|>\n";
            } else if (m.role == "assistant") {
                std::string content = m.content, reasoning;
                if (m.reasoning) {
                    reasoning = *m.reasoning;
                } else if (content.find("</think>") != std::string::npos) {
                    const std::string before = rstrip_nl(content.substr(0, content.find("</think>")));
                    const size_t open = before.rfind("<think>");
                    reasoning = lstrip_nl(open == std::string::npos ? before : before.substr(open + 7));
                    content = lstrip_nl(content.substr(content.rfind("</think>") + 8));
                }
                if (i > last_query && (i == n - 1 || !reasoning.empty()))
                    out += "<|im_start|>assistant\n<think>\n" + strip_nl(reasoning) + "\n</think>\n\n" +
                           lstrip_nl(content);
                else
                    out += "<|im_start|>assistant\n" + content;
                for (size_t k = 0; k < m.tool_calls.size(); ++k) {
                    if (k > 0 || !content.empty()) out += "\n";
                    out += "<tool_call>\n{\"name\": \"" + m.tool_calls[k].name + "\", \"arguments\": " +
                           py_dumps(m.tool_calls[k].arguments) + "}\n</tool_call>";
                }
                out += "<|im_end|>\n";
            } else if (m.role == "tool") {
                if (i == 0 || msgs[i - 1].role != "tool") out += "<|im_start|>user";
                out += "\n<tool_response>\n" + m.content + "\n</tool_response>";
                if (i == n - 1 || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
            }
        }
        if (chat.add_generation_prompt) {
            out += "<|im_start|>assistant\n";
            if (chat.enable_thinking == false) out += "<think>\n\n</think>\n\n";
        }
        return out;
    }

private:
    ToolResponseTest test_;
};

// ── QwenXml: Qwen3.5 / 3.6 / 3.8 ─────────────────────────────────────────────

const char* kXmlToolsHead = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const char* kXmlToolsTail =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must "
    "be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n- If there is no function call available, answer the question like "
    "normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
const char* kEffortXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
const char* kEffortLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";

class QwenXmlToolFormat : public ToolFormat {
public:
    enum class Preserve { Absent, DefaultOff, DefaultOn };   // the preserve_thinking switch
    struct Params {
        bool     thinking_default_on;    // 3.5: off; 3.6, 3.8: on
        Preserve preserve;               // 3.6: off by default; 3.8-27B: on; others: no switch
        bool     reasoning_effort;       // 3.8-27B: the effort sentence in the system turn
        bool     split_think_in_content; // a "</think>" in an assistant content is split off (not 3.8-27B)
        bool     scalars_as_json;        // a non-string argument prints as JSON (3.8-27B) vs Python str()
    };
    explicit QwenXmlToolFormat(Params p) : p_(p) {}
    ToolDialect dialect() const override { return ToolDialect::QwenXml; }

    std::string render(const ToolChat& chat) const override {
        check_chat(chat);
        if (p_.preserve == Preserve::Absent)
            refuse_kwarg_absent_from_template(chat.preserve_thinking, "preserve_thinking");
        if (!p_.reasoning_effort)
            refuse_kwarg_absent_from_template(chat.reasoning_effort, "reasoning_effort");
        const auto& msgs = chat.messages;
        const size_t n = msgs.size();

        std::string effort;
        if (p_.reasoning_effort && chat.enable_thinking != false) {
            std::string e = chat.reasoning_effort.value_or("xhigh");
            if (e == "high") e = "xhigh";
            if (e != "xhigh" && e != "medium" && e != "low")
                refuse("reasoning_effort", "\"xhigh\", \"high\", \"medium\" or \"low\"", "\"" + e + "\"");
            effort = e == "xhigh" ? kEffortXhigh : e == "low" ? kEffortLow : "";
        } else if (p_.reasoning_effort && chat.reasoning_effort) {
            refuse("reasoning_effort", "absent when enable_thinking is false (the template ignores it)",
                   "\"" + *chat.reasoning_effort + "\"");
        }

        std::string out = "<|im_start|>system\n";
        if (!effort.empty()) out += effort + "\n\n";
        out += kXmlToolsHead;
        for (const auto& t : chat.tools) out += "\n" + py_dumps(t);
        out += kXmlToolsTail;
        if (msgs[0].role == "system") {
            const std::string sys = py_strip(msgs[0].content);
            if (!sys.empty()) out += "\n\n" + sys;
        }
        out += "<|im_end|>\n";

        size_t last_query = n;   // check_chat guarantees a user message
        for (size_t k = n; k-- > 0;) {
            if (msgs[k].role != "user") continue;
            const std::string c = py_strip(msgs[k].content);
            if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) { last_query = k; break; }
        }
        if (last_query == n)
            refuse("messages", "a user query that is not only a <tool_response>", "none");

        const bool preserve = p_.preserve == Preserve::DefaultOn ? chat.preserve_thinking != false
                            : p_.preserve == Preserve::DefaultOff ? chat.preserve_thinking == true
                            : false;
        for (size_t i = 0; i < n; ++i) {
            const ToolTurn& m = msgs[i];
            std::string content = py_strip(m.content);
            if (m.role == "user") {
                out += "<|im_start|>user\n" + content + "<|im_end|>\n";
            } else if (m.role == "assistant") {
                std::string reasoning;
                if (m.reasoning) {
                    reasoning = *m.reasoning;
                } else if (p_.split_think_in_content && content.find("</think>") != std::string::npos) {
                    const std::string before = rstrip_nl(content.substr(0, content.find("</think>")));
                    const size_t open = before.rfind("<think>");
                    reasoning = lstrip_nl(open == std::string::npos ? before : before.substr(open + 7));
                    content = lstrip_nl(content.substr(content.rfind("</think>") + 8));
                }
                reasoning = py_strip(reasoning);
                if (preserve || i > last_query)
                    out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
                else
                    out += "<|im_start|>assistant\n" + content;
                for (size_t k = 0; k < m.tool_calls.size(); ++k) {
                    const ToolCall& c = m.tool_calls[k];
                    if (k == 0)
                        out += (py_strip(content).empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=");
                    else
                        out += "\n<tool_call>\n<function=";
                    out += c.name + ">\n";
                    for (auto it = c.arguments.begin(); it != c.arguments.end(); ++it) {
                        const ojson& v = it.value();
                        std::string text;
                        if (p_.scalars_as_json) text = v.is_string() ? v.get<std::string>() : py_dumps(v);
                        else text = (v.is_object() || v.is_array()) ? py_dumps(v)
                                    : py_str(v, "tool_calls[].function.arguments." + it.key());
                        out += "<parameter=" + it.key() + ">\n" + text + "\n</parameter>\n";
                    }
                    out += "</function>\n</tool_call>";
                }
                out += "<|im_end|>\n";
            } else if (m.role == "tool") {
                if (i > 0 && msgs[i - 1].role != "tool") out += "<|im_start|>user";
                out += "\n<tool_response>\n" + content + "\n</tool_response>";
                if (i == n - 1 || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
            }
        }
        if (chat.add_generation_prompt) {
            out += "<|im_start|>assistant\n";
            const bool think = p_.thinking_default_on ? chat.enable_thinking != false
                                                      : chat.enable_thinking == true;
            out += think ? "<think>\n" : "<think>\n\n</think>\n\n";
        }
        return out;
    }

private:
    Params p_;
};

// ── Gemma4 ───────────────────────────────────────────────────────────────────

const std::string kQ = "<|\"|>";   // Gemma 4's string delimiter token

class Gemma4ToolFormat : public ToolFormat {
public:
    struct Params {
        // 26B-A4B's template filters the schema's own key names ("type",
        // "description", …) out of EVERY properties map — a parameter with one
        // of those names vanishes, so it is refused. 12B's filters only where
        // an object without "properties" is read as its own property map.
        bool filter_every_properties_map;
        // Whether a model turn whose tool results were spliced in ends with
        // <turn|> when its content is non-empty as SENT (26B) or as RENDERED
        // after strip_thinking (12B).
        bool turn_end_on_rendered_content;
    };
    explicit Gemma4ToolFormat(Params p) : p_(p) {}
    ToolDialect dialect() const override { return ToolDialect::Gemma4; }

    std::string render(const ToolChat& chat) const override {
        check_chat(chat);
        refuse_kwarg_absent_from_template(chat.preserve_thinking, "preserve_thinking");
        refuse_kwarg_absent_from_template(chat.reasoning_effort, "reasoning_effort");
        const bool think = chat.enable_thinking == true;

        std::string out = "<|turn>system\n";
        if (think) out += "<|think|>\n";
        size_t first = 0;
        if (chat.messages[0].role == "system") { out += py_strip(chat.messages[0].content); first = 1; }
        for (size_t i = 0; i < chat.tools.size(); ++i)
            out += "<|tool>" + py_strip(declaration(chat.tools[i], "tools[" + std::to_string(i) + "]")) + "<tool|>";
        out += "<turn|>\n";

        std::vector<const ToolTurn*> loop;
        for (size_t i = first; i < chat.messages.size(); ++i) loop.push_back(&chat.messages[i]);
        long last_user = -1;
        for (size_t i = 0; i < loop.size(); ++i) if (loop[i]->role == "user") last_user = static_cast<long>(i);

        std::string prev_type = "tool";
        for (size_t i = 0; i < loop.size(); ++i) {
            const ToolTurn& m = *loop[i];
            if (m.role == "tool") continue;
            prev_type.clear();
            const std::string role = m.role == "assistant" ? "model" : m.role;
            std::string prev_role;
            for (size_t j = i; j-- > 0;)
                if (loop[j]->role != "tool") { prev_role = loop[j]->role; break; }
            if (!(role == "model" && prev_role == "assistant")) out += "<|turn>" + role + "\n";
            if (m.reasoning && !m.reasoning->empty() && static_cast<long>(i) > last_user && !m.tool_calls.empty())
                out += "<|channel>thought\n" + *m.reasoning + "\n<channel|>";
            for (const ToolCall& c : m.tool_calls) {
                out += "<|tool_call>call:" + c.name + "{";
                bool first_arg = true;
                for (const auto& [k, v] : dictsort(c.arguments)) {
                    if (!first_arg) out += ",";
                    first_arg = false;
                    out += k + ":" + argument(*v, false, "tool_calls[].function.arguments." + k);
                }
                out += "}<tool_call|>";
                prev_type = "tool_call";
            }
            bool spliced = false;
            if (!m.tool_calls.empty()) {
                for (size_t k = i + 1; k < loop.size() && loop[k]->role == "tool"; ++k) {
                    std::string name = "unknown";
                    for (const ToolCall& c : m.tool_calls)
                        if (c.id == loop[k]->tool_call_id) name = c.name;
                    out += "<|tool_response>response:" + name + "{value:" + argument(ojson(loop[k]->content), false, "content") +
                           "}<tool_response|>";
                    spliced = true;
                    prev_type = "tool_response";
                }
            }
            const std::string rendered = role == "model" ? strip_thinking(m.content) : py_strip(m.content);
            out += rendered;
            const bool has_content = p_.turn_end_on_rendered_content ? !py_strip(rendered).empty()
                                                                     : !m.content.empty();
            if (prev_type == "tool_call" && !spliced) out += "<|tool_response>";
            else if (!(spliced && !has_content)) out += "<turn|>\n";
        }
        if (chat.add_generation_prompt && prev_type != "tool_response" && prev_type != "tool_call") {
            out += "<|turn>model\n";
            if (!think) out += "<|channel>thought\n<channel|>";
        }
        return out;
    }

private:
    // The template's strip_thinking macro: drop every <|channel>…<channel|>
    // span, then trim.
    static std::string strip_thinking(const std::string& text) {
        std::string result;
        size_t pos = 0;
        while (true) {
            const size_t end = text.find("<channel|>", pos);
            const std::string part = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            const size_t open = part.find("<|channel>");
            result += open == std::string::npos ? part : part.substr(0, open);
            if (end == std::string::npos) break;
            pos = end + 10;
        }
        return py_strip(result);
    }

    // format_argument: strings in <|"|>, booleans lower-case, keys sorted.
    static std::string argument(const ojson& v, bool escape_keys, const std::string& field) {
        if (v.is_string()) return kQ + v.get<std::string>() + kQ;
        if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
        if (v.is_object()) {
            std::string s = "{";
            bool first = true;
            for (const auto& [k, val] : dictsort(v)) {
                if (!first) s += ",";
                first = false;
                s += (escape_keys ? kQ + k + kQ : k) + ":" + argument(*val, escape_keys, field + "." + k);
            }
            return s + "}";
        }
        if (v.is_array()) {
            std::string s = "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) s += ",";
                s += argument(v[i], escape_keys, field + "[" + std::to_string(i) + "]");
            }
            return s + "]";
        }
        return py_str(v, field);
    }

    static std::string type_upper(const ojson& schema, const std::string& field) {
        if (!schema.is_object() || !schema.contains("type") || schema["type"].is_null()) return "";
        if (!schema["type"].is_string())
            refuse(field + ".type", "a string (a type list is not rendered by this template)",
                   schema["type"].type_name());
        return ascii_upper(schema["type"].get<std::string>());
    }

    static std::string required_list(const ojson& req, const std::string& field) {
        if (!req.is_array()) refuse(field, "an array of names", req.type_name());
        std::string s;
        for (size_t i = 0; i < req.size(); ++i) {
            if (i) s += ",";
            s += kQ + py_str(req[i], field + "[" + std::to_string(i) + "]") + kQ;
        }
        return s;
    }

    // format_parameters. `filter`: drop the schema's own key names; `refuse_filtered`:
    // a real property map would lose a parameter that way — refuse instead.
    std::string parameters(const ojson& props, bool filter, bool refuse_filtered, const std::string& field) const {
        static const char* kStandard[] = {"description", "type", "properties", "required", "nullable"};
        std::string out;
        bool found_first = false;
        for (const auto& [key, vp] : dictsort(props)) {
            const ojson& value = *vp;
            if (filter && std::find(std::begin(kStandard), std::end(kStandard), key) != std::end(kStandard)) {
                if (refuse_filtered)
                    refuse(field + "." + key, "a parameter name other than description, type, properties, "
                           "required or nullable (this checkpoint's template drops it)", "\"" + key + "\"");
                continue;
            }
            const std::string at = field + "." + key;
            if (!value.is_object()) refuse(at, "a schema object", value.type_name());
            bool add_comma = false;
            const auto comma = [&] { if (add_comma) out += ","; else add_comma = true; };
            if (found_first) out += ",";
            found_first = true;
            out += key + ":{";
            if (truthy(value, "description")) {
                out += "description:" + kQ + py_str(value["description"], at + ".description") + kQ;
                add_comma = true;
            }
            const std::string type = type_upper(value, at);
            if (type == "STRING") {
                if (truthy(value, "enum")) { comma(); out += "enum:" + argument(value["enum"], true, at + ".enum"); }
            } else if (type == "ARRAY") {
                if (value.contains("items") && value["items"].is_object() && !value["items"].empty()) {
                    const ojson& items = value["items"];
                    comma();
                    out += "items:{";
                    bool ff = false;
                    for (const auto& [ik, ivp] : dictsort(items)) {
                        const ojson& iv = *ivp;
                        if (iv.is_null()) continue;
                        if (ff) out += ",";
                        ff = true;
                        if (ik == "properties") {
                            out += "properties:{";
                            if (iv.is_object())
                                out += parameters(iv, p_.filter_every_properties_map, p_.filter_every_properties_map,
                                                  at + ".items.properties");
                            out += "}";
                        } else if (ik == "required") {
                            out += "required:[" + required_list(iv, at + ".items.required") + "]";
                        } else if (ik == "type") {
                            if (!iv.is_string())
                                refuse(at + ".items.type", "a string (a type list is not rendered by this template)",
                                       iv.type_name());
                            out += "type:" + argument(ojson(ascii_upper(iv.get<std::string>())), true, at + ".items.type");
                        } else {
                            out += ik + ":" + argument(iv, true, at + ".items." + ik);
                        }
                    }
                    out += "}";
                }
            }
            if (truthy(value, "nullable")) { comma(); out += "nullable:true"; }
            if (type == "OBJECT") {
                if (value.contains("properties") && value["properties"].is_object()) {
                    comma();
                    out += "properties:{" + parameters(value["properties"], p_.filter_every_properties_map,
                                                       p_.filter_every_properties_map, at + ".properties") + "}";
                } else {
                    // An object without "properties": the template reads the
                    // schema's own keys as its properties, minus the standard ones.
                    comma();
                    out += "properties:{" + parameters(value, true, false, at) + "}";
                }
                if (truthy(value, "required")) {
                    comma();
                    out += "required:[" + required_list(value["required"], at + ".required") + "]";
                }
            }
            comma();
            out += "type:" + kQ + type + kQ + "}";
        }
        return out;
    }

    // format_function_declaration.
    std::string declaration(const ojson& tool, const std::string& at) const {
        const ojson& f = tool["function"];
        if (f.contains("response"))
            refuse(at + ".function.response", "absent (not an OpenAI tool field)", "present");
        std::string s = "declaration:" + f["name"].get<std::string>() + "{description:" + kQ;
        if (f.contains("description") && !f["description"].is_null())
            s += py_str(f["description"], at + ".function.description");
        s += kQ;
        if (truthy(f, "parameters")) {
            const ojson& params = f["parameters"];
            if (!params.is_object()) refuse(at + ".function.parameters", "an object", params.type_name());
            s += ",parameters:{";
            if (truthy(params, "properties")) {
                if (!params["properties"].is_object())
                    refuse(at + ".function.parameters.properties", "an object", params["properties"].type_name());
                s += "properties:{" + parameters(params["properties"], p_.filter_every_properties_map,
                                                 p_.filter_every_properties_map, at + ".function.parameters.properties") +
                     "},";
            }
            if (truthy(params, "required"))
                s += "required:[" + required_list(params["required"], at + ".function.parameters.required") + "],";
            if (truthy(params, "type"))
                s += "type:" + kQ + type_upper(params, at + ".function.parameters") + kQ + "}";
        }
        return s + "}";
    }

    Params p_;
};

// ── The measured templates ───────────────────────────────────────────────────
// One row per template whose renderings are captured in
// tests/fixtures/chat_tools/<sha16>/ and gated byte for byte (G1).

struct Measured {
    const char*       sha16;
    const ToolFormat* format;
};

const std::vector<Measured>& measured() {
    using H = HermesJsonToolFormat;
    using X = QwenXmlToolFormat;
    using G = Gemma4ToolFormat;
    static const H qwen3_06b(H::ToolResponseTest::Contains);
    static const H qwen3_17b(H::ToolResponseTest::StartsAndEnds);
    //                       think on  preserve                effort split  json scalars
    static const X qwen35   ({false,   X::Preserve::Absent,     false, true,  false});
    static const X qwen36   ({true,    X::Preserve::DefaultOff, false, true,  false});
    static const X qwen38_9b({true,    X::Preserve::Absent,     false, true,  false});
    static const X qwen38_27({true,    X::Preserve::DefaultOn,  true,  false, true});
    static const G gemma4_26b({/*filter_every_properties_map=*/true,  /*turn_end_on_rendered_content=*/false});
    static const G gemma4_12b({/*filter_every_properties_map=*/false, /*turn_end_on_rendered_content=*/true});
    static const std::vector<Measured> rows = {
        {"57f1fd00f0013a2b", &qwen3_06b},   // Qwen3-0.6B
        {"8428c815ac94d820", &qwen3_17b},   // Qwen3-1.7B
        {"7f0e529032c25183", &qwen35},      // Qwen3.5-0.8B, Qwen3.5-9B
        {"55d4931433fe502b", &qwen36},      // Qwen3.6-27B, Qwen3.6-35B-A3B (Unsloth template)
        {"a4aee8afcf2e0711", &qwen38_9b},   // Qwen3.8-9B
        {"12827f24b742ea4e", &qwen38_27},   // Qwen3.8-27B (Unsloth template)
        {"85a08664d16d8f3b", &gemma4_26b},  // Gemma 4 26B-A4B-it
        {"36e3a42e5cf14cd0", &gemma4_12b},  // Gemma 4 12B-it
    };
    return rows;
}

}  // namespace

const char* to_string(ToolDialect d) {
    switch (d) {
    case ToolDialect::HermesJson: return "hermes-json";
    case ToolDialect::QwenXml:    return "qwen-xml";
    case ToolDialect::Gemma4:     return "gemma4";
    }
    return "?";
}

std::string template_sha16(const std::string& chat_template) {
    unsigned char md[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(chat_template.data(), static_cast<CC_LONG>(chat_template.size()), md);
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 8; ++i) {
        out += hex[md[i] >> 4];
        out += hex[md[i] & 15];
    }
    return out;
}

const ToolFormat* lookup_tool_format(const std::string& sha16) {
    for (const Measured& m : measured())
        if (sha16 == m.sha16) return m.format;
    return nullptr;
}

std::vector<std::string> measured_tool_templates() {
    std::vector<std::string> out;
    for (const Measured& m : measured()) out.push_back(m.sha16);
    return out;
}

}  // namespace qinf::tools
