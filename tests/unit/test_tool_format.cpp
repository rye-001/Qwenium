// test_tool_format.cpp — co-located unit test for src/loader/tool_format.cpp.
// Gate G1 of docs/plan-tool-calling.md: for every measured chat template, every
// case of tests/fixtures/chat_tools/cases.json renders byte for byte as Hugging
// Face's renderer renders it from the GGUF's own template (<case>.hf.txt,
// captured by py/capture_tool_chat_ref.py). llama.cpp's Jinja engine
// (<case>.llamacpp.txt, scripts/capture_tool_chat_ref_llamacpp.cpp) is the
// cross-check: it must agree with HF everywhere except the listed cases, where
// its Python emulation differs (recorded, not followed). Plus the refusals: what
// a template would silently drop or mangle throws, naming the field.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../../src/loader/tool_format.h"

namespace fs = std::filesystem;
using qinf::tools::ojson;
using qinf::tools::ToolCall;
using qinf::tools::ToolChat;
using qinf::tools::ToolTurn;

namespace {

const fs::path kFx = fs::path(QINF_TEST_FIXTURES_DIR) / "chat_tools";

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("open '" + p.string() + "': expected readable, actual: failed");
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

const ojson& cases() {
    static const ojson c = ojson::parse(slurp(kFx / "cases.json"));
    return c;
}

// cases.json is already normalised the way the server will normalise an OpenAI
// request (content strings, arguments objects).
ToolChat chat_of(const ojson& c) {
    ToolChat chat;
    chat.tools = cases()["tool_sets"][c["tools"].get<std::string>()];
    for (const ojson& m : c["messages"]) {
        ToolTurn t;
        t.role = m["role"].get<std::string>();
        t.content = m.value("content", std::string());
        if (m.contains("reasoning_content")) t.reasoning = m["reasoning_content"].get<std::string>();
        t.tool_call_id = m.value("tool_call_id", std::string());
        if (m.contains("tool_calls"))
            for (const ojson& tc : m["tool_calls"])
                t.tool_calls.push_back(ToolCall{tc["id"].get<std::string>(),
                                                tc["function"]["name"].get<std::string>(),
                                                tc["function"]["arguments"]});
        chat.messages.push_back(std::move(t));
    }
    const ojson kw = c.value("kwargs", ojson::object());
    if (kw.contains("enable_thinking"))   chat.enable_thinking = kw["enable_thinking"].get<bool>();
    if (kw.contains("preserve_thinking")) chat.preserve_thinking = kw["preserve_thinking"].get<bool>();
    if (kw.contains("reasoning_effort"))  chat.reasoning_effort = kw["reasoning_effort"].get<std::string>();
    return chat;
}

ToolChat case_named(const std::string& name) {
    for (const ojson& c : cases()["cases"])
        if (c["name"] == name) return chat_of(c);
    throw std::runtime_error("case '" + name + "': expected in cases.json, actual: absent");
}

std::string render(const std::string& sha16, const ToolChat& chat) {
    const auto* f = qinf::tools::lookup_tool_format(sha16);
    if (!f) throw std::runtime_error("template " + sha16 + ": expected measured, actual: unknown");
    return f->render(chat);
}

std::string refusal(const std::string& sha16, const ToolChat& chat) {
    try {
        render(sha16, chat);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "(rendered, not refused)";
}

// Which kwargs a template has no switch for: rendering would silently ignore
// them, so they are refused.
bool has_preserve(const std::string& h) { return h == "55d4931433fe502b" || h == "12827f24b742ea4e"; }
bool has_effort(const std::string& h)   { return h == "12827f24b742ea4e"; }

const std::string kQwen35 = "7f0e529032c25183", kQwen38_27 = "12827f24b742ea4e",
                  kGemma26 = "85a08664d16d8f3b", kGemma12 = "36e3a42e5cf14cd0",
                  kQwen3 = "57f1fd00f0013a2b";

}  // namespace

// Every measured template has a fixture directory, its template hashes to the
// directory's name, and nothing else is measured.
TEST(ToolFormat, MeasuredTemplatesAreTheFixtureDirectories) {
    std::set<std::string> dirs;
    for (const auto& e : fs::directory_iterator(kFx))
        if (e.is_directory()) dirs.insert(e.path().filename().string());
    const auto measured = qinf::tools::measured_tool_templates();
    EXPECT_EQ(std::set<std::string>(measured.begin(), measured.end()), dirs);
    for (const std::string& h : measured)
        EXPECT_EQ(qinf::tools::template_sha16(slurp(kFx / h / "template.jinja")), h) << h;
    EXPECT_EQ(qinf::tools::lookup_tool_format("0000000000000000"), nullptr);
}

TEST(ToolFormat, DialectPerTemplate) {
    using qinf::tools::ToolDialect;
    const auto d = [](const std::string& h) { return qinf::tools::lookup_tool_format(h)->dialect(); };
    EXPECT_EQ(d("57f1fd00f0013a2b"), ToolDialect::HermesJson);
    EXPECT_EQ(d("8428c815ac94d820"), ToolDialect::HermesJson);
    for (const char* h : {"7f0e529032c25183", "55d4931433fe502b", "a4aee8afcf2e0711", "12827f24b742ea4e"})
        EXPECT_EQ(d(h), ToolDialect::QwenXml) << h;
    EXPECT_EQ(d(kGemma26), ToolDialect::Gemma4);
    EXPECT_EQ(d(kGemma12), ToolDialect::Gemma4);
}

// G1: byte for byte against Hugging Face's rendering, every template × case.
TEST(ToolFormat, RendersLikeHuggingFaceOnEveryMeasuredTemplate) {
    int compared = 0, refused = 0;
    for (const std::string& h : qinf::tools::measured_tool_templates()) {
        const bool gemma = qinf::tools::lookup_tool_format(h)->dialect() == qinf::tools::ToolDialect::Gemma4;
        for (const ojson& c : cases()["cases"]) {
            const std::string name = c["name"].get<std::string>();
            const ToolChat chat = chat_of(c);
            const bool needs_preserve = chat.preserve_thinking.has_value();
            const bool needs_effort = chat.reasoning_effort.has_value();
            if ((needs_preserve && !has_preserve(h)) || (needs_effort && !has_effort(h))) {
                EXPECT_NE(refusal(h, chat).find("this checkpoint's template has no such switch"), std::string::npos)
                    << h << "/" << name << ": " << refusal(h, chat);
                ++refused;
                continue;
            }
            std::string expected = slurp(kFx / h / (name + ".hf.txt"));
            if (gemma) {   // the template's leading bos_token is the tokenizer's BOS
                ASSERT_EQ(expected.rfind("<bos>", 0), 0u) << h << "/" << name;
                expected.erase(0, 5);
            }
            EXPECT_EQ(render(h, chat), expected) << h << "/" << name;
            ++compared;
        }
    }
    EXPECT_EQ(compared, 8 * 19 - refused);
    EXPECT_EQ(refused, 6 * 1 /*preserve*/ + 7 * 2 /*effort*/);
}

// The cross-check: llama.cpp's Jinja engine agrees with HF on every rendering
// but the two where its Python emulation differs — Gemma's
// number_null_escape_forms: |dictsort is case-sensitive there ("Zeta" first),
// `{{ none }}` prints "" not "None", and 1e16 / 1e-05 print in fixed notation.
TEST(ToolFormat, LlamaCppAgreesWithHuggingFaceExceptTheRecordedCases) {
    const std::set<std::string> differ = {kGemma26 + "/number_null_escape_forms",
                                          kGemma12 + "/number_null_escape_forms"};
    for (const std::string& h : qinf::tools::measured_tool_templates())
        for (const ojson& c : cases()["cases"]) {
            const std::string name = c["name"].get<std::string>();
            const bool same = slurp(kFx / h / (name + ".hf.txt")) == slurp(kFx / h / (name + ".llamacpp.txt"));
            EXPECT_EQ(same, differ.count(h + "/" + name) == 0) << h << "/" << name;
        }
}

// ── Refusals: what a template would drop or mangle ──────────────────────────

TEST(ToolFormat, RefusesAToolResultWithoutItsCall) {
    ToolChat chat = case_named("one_call_result");
    chat.messages[2].tool_call_id = "call_9";
    for (const std::string& h : qinf::tools::measured_tool_templates())
        EXPECT_EQ(refusal(h, chat), "'messages[2].tool_call_id': expected the id of a call in messages[1], "
                                    "actual: \"call_9\"") << h;
    chat = case_named("one_call_result");
    chat.messages.erase(chat.messages.begin() + 1);   // the call itself
    EXPECT_EQ(refusal(kGemma12, chat), "'messages[1]': expected a tool result following an assistant message "
                                       "with tool_calls, actual: after a \"user\" message");
}

TEST(ToolFormat, RefusesASystemMessageAfterTheFirst) {
    ToolChat chat = case_named("tools_only");
    chat.messages.push_back(ToolTurn{"system", "late"});
    EXPECT_EQ(refusal(kQwen35, chat),
              "'messages[1].role': expected \"system\" only as the first message, actual: \"system\" at 1");
}

TEST(ToolFormat, RefusesAnEmptyToolListAndANonObjectArgument) {
    ToolChat chat = case_named("tools_only");
    chat.tools = ojson::array();
    EXPECT_EQ(refusal(kQwen3, chat), "'tools': expected a non-empty array, actual: an empty array");
    chat = case_named("one_call_result");
    chat.messages[1].tool_calls[0].arguments = "{\"city\": \"Paris\"}";   // still a JSON string
    EXPECT_EQ(refusal(kQwen38_27, chat),
              "'messages[1].tool_calls[0].function.arguments': expected a JSON object, actual: string");
}

// T6: Gemma 4 26B's template drops a parameter named like a schema keyword
// from the declaration the model sees; 12B's renders it.
TEST(ToolFormat, Gemma26RefusesAParameterTheTemplateWouldDrop) {
    ToolChat chat = case_named("tools_only");
    chat.tools[0]["function"]["parameters"]["properties"]["type"] = {{"type", "string"}};
    EXPECT_EQ(refusal(kGemma26, chat),
              "'tools[0].function.parameters.properties.type': expected a parameter name other than "
              "description, type, properties, required or nullable (this checkpoint's template drops it), "
              "actual: \"type\"");
    const std::string r = render(kGemma12, chat);
    EXPECT_NE(r.find("type:{type:<|\"|>STRING<|\"|>}"), std::string::npos) << r;
}

TEST(ToolFormat, RefusesAnUnknownReasoningEffortAndEffortWithThinkingOff) {
    ToolChat chat = case_named("tools_only");
    chat.reasoning_effort = "max";
    EXPECT_EQ(refusal(kQwen38_27, chat),
              "'reasoning_effort': expected \"xhigh\", \"high\", \"medium\" or \"low\", actual: \"max\"");
    chat.reasoning_effort = "low";
    chat.enable_thinking = false;
    EXPECT_EQ(refusal(kQwen38_27, chat),
              "'reasoning_effort': expected absent when enable_thinking is false (the template ignores it), "
              "actual: \"low\"");
}
