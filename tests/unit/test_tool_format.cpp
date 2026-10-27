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
#include <map>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../../src/loader/tool_format.h"
#include "../../src/sampling/grammar_vocab.h"

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

// ── P2: parsing the reply (gate G2) ──────────────────────────────────────────

namespace {

using qinf::tools::ParsedReply;
using qinf::tools::ToolCallError;

ParsedReply parse(const std::string& sha16, const std::string& text, const ToolChat& chat) {
    return qinf::tools::lookup_tool_format(sha16)->parse(text, chat);
}

std::string parse_error(const std::string& sha16, const std::string& text, const ToolChat& chat) {
    try {
        parse(sha16, text, chat);
    } catch (const ToolCallError& e) {
        return e.what();
    }
    return "(parsed, no error)";
}

// Key order aside (Gemma's template sorts keys), the same JSON value.
bool same_json(const ojson& a, const ojson& b) {
    return nlohmann::json::parse(a.dump()) == nlohmann::json::parse(b.dump());
}

std::string py_strip_ascii(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\n\r\f\v");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\n\r\f\v") - a + 1);
}

const std::string kGemma = kGemma26;   // the parsers do not differ between Gemma checkpoints

}  // namespace

// The round trip: an assistant turn with calls, rendered into history by the
// template, is exactly what the model was trained to emit after the
// generation prompt. So render(messages[:k]) + generation prompt must be a
// prefix of render(messages[:k+1]), and parsing the rest must give back the
// calls of messages[k] — on every measured template, every fixture case with
// calls. Two exclusions, each asserted: number_null_escape_forms breaks its
// own schema (all_day null for a boolean), so every dialect must REFUSE it;
// Gemma renders a calling turn's content AFTER the calls, which the parser
// refuses as text after a call (the model is expected to stop on
// <|tool_response>), so content_then_call is a refusal there too.
TEST(ToolFormat, ParsesBackWhatTheTemplateRendersAsAModelTurn) {
    std::map<std::string, int> round_trips;
    for (const std::string& h : qinf::tools::measured_tool_templates()) {
        const auto dialect = qinf::tools::lookup_tool_format(h)->dialect();
        for (const ojson& c : cases()["cases"]) {
            const std::string name = c["name"].get<std::string>();
            const ToolChat full = chat_of(c);
            for (size_t k = 0; k < full.messages.size(); ++k) {
                if (full.messages[k].tool_calls.empty()) continue;
                ToolChat before = full, upto = full;
                before.messages.resize(k);
                upto.messages.resize(k + 1);
                upto.add_generation_prompt = false;
                std::string P, Q;
                try { P = render(h, before); Q = render(h, upto); } catch (const std::invalid_argument&) { continue; }
                // Gemma with thinking off: the generation prompt ends in the empty
                // thought block, which history never re-renders — the reply
                // follows it, so it is what follows "<|turn>model\n" in history.
                const std::string empty_thought = "<|channel>thought\n<channel|>";
                size_t cut = P.size();
                if (Q.rfind(P, 0) != 0 && P.size() >= empty_thought.size() &&
                    P.compare(P.size() - empty_thought.size(), empty_thought.size(), empty_thought) == 0 &&
                    Q.rfind(P.substr(0, P.size() - empty_thought.size()), 0) == 0)
                    cut = P.size() - empty_thought.size();
                else
                    ASSERT_EQ(Q.rfind(P, 0), 0u) << h << "/" << name << ": the generation prompt is not a prefix";
                const std::string gen = Q.substr(cut);
                const std::string what = h + "/" + name + " k=" + std::to_string(k);
                if (name == "number_null_escape_forms") {
                    EXPECT_NE(parse_error(h, gen, before).find("tool_calls[0].arguments.all_day: expected boolean"),
                              std::string::npos) << what << ": " << parse_error(h, gen, before);
                    continue;
                }
                if (dialect == qinf::tools::ToolDialect::Gemma4 && name == "content_then_call") {
                    EXPECT_NE(parse_error(h, gen, before).find("after tool_calls[0]: expected another <|tool_call>"),
                              std::string::npos) << what << ": " << parse_error(h, gen, before);
                    continue;
                }
                const ParsedReply r = parse(h, gen, before);
                const ToolTurn& m = full.messages[k];
                ASSERT_EQ(r.calls.size(), m.tool_calls.size()) << what << "\n" << gen;
                for (size_t i = 0; i < r.calls.size(); ++i) {
                    EXPECT_EQ(r.calls[i].name, m.tool_calls[i].name) << what;
                    EXPECT_TRUE(same_json(r.calls[i].arguments, m.tool_calls[i].arguments))
                        << what << ": " << r.calls[i].arguments.dump() << " vs " << m.tool_calls[i].arguments.dump();
                }
                if (dialect != qinf::tools::ToolDialect::Gemma4)
                    EXPECT_EQ(r.content, py_strip_ascii(m.content)) << what;
                const bool shown = m.reasoning && gen.find(*m.reasoning) != std::string::npos;
                EXPECT_EQ(r.reasoning, shown ? *m.reasoning : "") << what;
                ++round_trips[h];
            }
        }
    }
    for (const std::string& h : qinf::tools::measured_tool_templates()) {
        EXPECT_GE(round_trips[h], 9) << h << ": too few round trips";
        RecordProperty("round_trips_" + h, round_trips[h]);
    }
}

TEST(ToolFormat, HermesParsesContentReasoningAndParallelCalls) {
    const ToolChat chat = case_named("tools_only");
    const ParsedReply r = parse(kQwen3,
        "<think>\nTwo lookups.\n</think>\n\nChecking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        "{\"city\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": "
        "{\"tz\": \"Europe/Paris\"}}\n</tool_call><|im_end|>", chat);
    EXPECT_EQ(r.reasoning, "Two lookups.");
    EXPECT_EQ(r.content, "Checking.");
    ASSERT_EQ(r.calls.size(), 2u);
    EXPECT_EQ(r.calls[1].name, "get_time");
    EXPECT_EQ(r.calls[1].arguments.dump(), "{\"tz\":\"Europe/Paris\"}");
    const ParsedReply plain = parse(kQwen3, "<think>\n\n</think>\n\nIt is sunny.", chat);
    EXPECT_TRUE(plain.calls.empty());
    EXPECT_EQ(plain.content, "It is sunny.");
}

TEST(ToolFormat, HermesNamesWhatIsMalformed) {
    const ToolChat chat = case_named("tools_only");
    EXPECT_EQ(parse_error(kQwen3, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par", chat),
              "tool_calls[0]: expected a call closed by </tool_call>, actual: unterminated, "
              "\"<tool_call>\\n{\\\"name\\\": \\\"get_weather\\\", \\\"arg\"... (offset 0)");
    EXPECT_EQ(parse_error(kQwen3, "<tool_call>\n{\"name\": get_weather}\n</tool_call>", chat),
              "tool_calls[0]: expected a JSON object {\"name\": ..., \"arguments\": {...}}, actual: "
              "\"\\n{\\\"name\\\": get_weather}\\n</tool_call>\" (offset 11)");
    EXPECT_EQ(parse_error(kQwen3, "<tool_call>\n{\"name\": \"get_wether\", \"arguments\": {}}\n</tool_call>", chat),
              "tool_calls[0].name: expected one of the request's tools (get_weather, get_time), actual: \"get_wether\"");
    EXPECT_EQ(parse_error(kQwen3, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n"
                                  "</tool_call>\nDone.", chat),
              "after tool_calls[0]: expected another <tool_call> or the end of the reply, actual: \"Done.\" (offset 81)");
    EXPECT_EQ(parse_error(kQwen3, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{}\"}\n</tool_call>", chat),
              "tool_calls[0].arguments: expected a JSON object, actual: \"{}\"");
}

// The checked schema subset (T5), the same for every dialect.
TEST(ToolFormat, ChecksArgumentsAgainstTheSchemaSubset) {
    ToolChat chat = case_named("tools_only");
    const auto call = [](const std::string& args) {
        return "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": " + args + "}\n</tool_call>";
    };
    EXPECT_EQ(parse_error(kQwen3, call("{}"), chat),
              "tool_calls[0].arguments.city: expected present (required), actual: absent");
    EXPECT_EQ(parse_error(kQwen3, call("{\"city\": 5}"), chat),
              "tool_calls[0].arguments.city: expected string, actual: a number 5");
    EXPECT_EQ(parse_error(kQwen3, call("{\"city\": \"Paris\", \"unit\": \"kelvin\"}"), chat),
              "tool_calls[0].arguments.unit: expected one of [\"celsius\",\"fahrenheit\"], actual: \"kelvin\"");
    EXPECT_EQ(parse_error(kQwen3, call("{\"city\": \"Paris\", \"days\": 3}"), chat), "(parsed, no error)");
    chat.tools[0]["function"]["parameters"]["additionalProperties"] = false;
    EXPECT_EQ(parse_error(kQwen3, call("{\"city\": \"Paris\", \"days\": 3}"), chat),
              "tool_calls[0].arguments.days: expected no such parameter (additionalProperties is false), actual: present");
}

TEST(ToolFormat, QwenXmlTypesValuesByTheSchema) {
    ToolChat chat = case_named("bool_true_arg");
    chat.messages.resize(1);
    const ParsedReply r = parse(kQwen38_27,
        "The user wants an event.\n</think>\n\n<tool_call>\n<function=create_event>\n"
        "<parameter=title>\n Planning \n</parameter>\n<parameter=duration_min>\n90\n</parameter>\n"
        "<parameter=all_day>\nTrue\n</parameter>\n<parameter=attendees>\n[\"a@x.org\"]\n</parameter>\n"
        "<parameter=location>\n{\"name\": \"München\", \"lat\": 48.1}\n</parameter>\n</function>\n</tool_call>", chat);
    EXPECT_EQ(r.reasoning, "The user wants an event.");
    ASSERT_EQ(r.calls.size(), 1u);
    EXPECT_EQ(r.calls[0].arguments.dump(),
              "{\"title\":\" Planning \",\"duration_min\":90,\"all_day\":true,\"attendees\":[\"a@x.org\"],"
              "\"location\":{\"name\":\"München\",\"lat\":48.1}}");
    // Thinking off: the reply starts after the empty block, no reasoning.
    chat.enable_thinking = false;
    const ParsedReply off = parse(kQwen35, "<tool_call>\n<function=create_event>\n<parameter=title>\nT\n"
                                           "</parameter>\n<parameter=duration_min>\n5\n</parameter>\n</function>\n"
                                           "</tool_call>", chat);
    EXPECT_EQ(off.reasoning, "");
    EXPECT_EQ(off.calls[0].arguments.dump(), "{\"title\":\"T\",\"duration_min\":5}");
}

TEST(ToolFormat, QwenXmlNamesWhatIsMalformed) {
    ToolChat chat = case_named("bool_true_arg");
    chat.messages.resize(1);
    chat.enable_thinking = false;
    const std::string head = "<tool_call>\n<function=create_event>\n<parameter=title>\nT\n</parameter>\n";
    EXPECT_EQ(parse_error(kQwen35, head + "<parameter=duration_min>\nninety\n</parameter>\n</function>\n</tool_call>", chat),
              "tool_calls[0].arguments.duration_min: expected integer, actual: \"ninety\"");
    EXPECT_EQ(parse_error(kQwen35, "<tool_call>\n<funtion=create_event>\n", chat),
              "tool_calls[0]: expected <function=NAME> after <tool_call>, actual: \"<funtion=create_event>\\n\" (offset 12)");
    EXPECT_EQ(parse_error(kQwen35, head + "<parameter=duration_min>\n9", chat),
              "tool_calls[0].arguments.duration_min: expected a value closed by </parameter>, actual: unterminated, "
              "\"9\" (offset 94)");
    EXPECT_EQ(parse_error(kQwen35, head + "<parameter=title>\nU\n</parameter>\n</function>\n</tool_call>", chat),
              "tool_calls[0].arguments.title: expected each parameter once, actual: a second value");
    EXPECT_EQ(parse_error(kQwen35, head + "<parameter=duration_min>\n5\n</parameter>\n</function>\n", chat),
              "tool_calls[0]: expected </tool_call> after </function>, actual: \"\" (offset 121)");
}

TEST(ToolFormat, Gemma4ParsesItsNotation) {
    ToolChat chat = case_named("tools_only");
    chat.enable_thinking = true;
    const ParsedReply r = parse(kGemma,
        "<|channel>thought\nTwo calls.<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,"
        "unit:<|\"|>celsius<|\"|>}<tool_call|><|tool_call>call:get_time{tz:<|\"|>Europe/Paris<|\"|>}<tool_call|>"
        "<|tool_response>", chat);
    EXPECT_EQ(r.reasoning, "Two calls.");
    EXPECT_EQ(r.content, "");
    ASSERT_EQ(r.calls.size(), 2u);
    EXPECT_EQ(r.calls[0].arguments.dump(), "{\"city\":\"Paris\",\"unit\":\"celsius\"}");
    ToolChat rich = case_named("bool_true_arg");
    rich.messages.resize(1);
    const ParsedReply n = parse(kGemma,
        "<|tool_call>call:create_event{all_day:false,attendees:[<|\"|>a@x.org<|\"|>,<|\"|>b@x.org<|\"|>],"
        "duration_min:90,location:{lat:-48.5e1,name:<|\"|>a, {b}: c<|\"|>},title:<|\"|>T<|\"|>}<tool_call|>", rich);
    EXPECT_EQ(n.calls[0].arguments.dump(),
              "{\"all_day\":false,\"attendees\":[\"a@x.org\",\"b@x.org\"],\"duration_min\":90,"
              "\"location\":{\"lat\":-485.0,\"name\":\"a, {b}: c\"},\"title\":\"T\"}");
}

TEST(ToolFormat, Gemma4NamesWhatIsMalformed) {
    const ToolChat chat = case_named("tools_only");
    EXPECT_EQ(parse_error(kGemma, "<|tool_call>call:get_weather{city:<|\"|>Par", chat),
              "tool_calls[0].arguments.city: expected a string closed by <|\"|>, actual: unterminated, "
              "\"<|\\\"|>Par\" (offset 34)");
    EXPECT_EQ(parse_error(kGemma, "<|tool_call>get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", chat),
              "tool_calls[0]: expected call:NAME{...} after <|tool_call>, actual: "
              "\"get_weather{city:<|\\\"|>Paris<|\\\"|>}<tool_c\"... (offset 12)");
    EXPECT_EQ(parse_error(kGemma, "<|tool_call>call:get_weather{city:Paris}<tool_call|>", chat),
              "tool_calls[0].arguments.city: expected a value (<|\"|>string<|\"|>, number, true, false, null, "
              "{...} or [...]), actual: \"Paris}<tool_call|>\" (offset 34)");
    EXPECT_EQ(parse_error(kGemma, "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}", chat),
              "tool_calls[0]: expected <tool_call|> after the arguments, actual: \"\" (offset 50)");
}

// ── P4: the grammar for tool_choice "required" / a named function (G4) ───────

namespace {

// A toy vocabulary: every byte as a token, plus the dialect's special tokens as
// whole tokens — so a text is fed the way the model emits it (special tokens
// atomic, everything else in pieces the grammar must accept piece by piece).
struct ToyVocab {
    std::vector<std::string> v;
    std::vector<std::string> specials;
    explicit ToyVocab(std::vector<std::string> sp) : specials(std::move(sp)) {
        for (int b = 0; b < 256; ++b) v.push_back(std::string(1, static_cast<char>(b)));
        for (const auto& s : specials) v.push_back(s);
    }
    // The next token id at `pos`: the longest special that starts there, else the byte.
    int32_t next(const std::string& text, size_t pos, size_t& len) const {
        int32_t best = static_cast<unsigned char>(text[pos]);
        len = 1;
        for (size_t i = 0; i < specials.size(); ++i)
            if (text.compare(pos, specials[i].size(), specials[i]) == 0 && specials[i].size() > len) {
                best = static_cast<int32_t>(256 + i);
                len = specials[i].size();
            }
        return best;
    }
};

ToyVocab toy_for(qinf::tools::ToolDialect d) {
    if (d == qinf::tools::ToolDialect::Gemma4)
        return ToyVocab({"<|tool_call>", "<tool_call|>", "<|\"|>", "<|channel>", "<channel|>", "<|tool_response>"});
    return ToyVocab({"<tool_call>", "</tool_call>", "<think>", "</think>"});
}

// "" when the grammar accepts `text` as a whole; else where and what it refused.
std::string grammar_verdict(const std::string& gbnf, const ToyVocab& vocab, const std::string& text) {
    auto g = qinf::GrammarVocab::parse_impl(gbnf);
    if (!g) return "grammar does not parse";
    size_t pos = 0;
    while (pos < text.size()) {
        size_t len = 0;
        const int32_t id = vocab.next(text, pos, len);
        const auto valid = g->get_valid_tokens(vocab.v);
        if (std::find(valid.begin(), valid.end(), id) == valid.end())
            return "refused " + ojson(vocab.v[id]).dump() + " at offset " + std::to_string(pos);
        g->accept_token(id, vocab.v);
        pos += len;
    }
    return g->is_accepting_state() ? "" : "not accepting at the end";
}

qinf::tools::ToolGrammar grammar_of(const std::string& h, const ToolChat& chat, const std::string* only = nullptr) {
    return qinf::tools::lookup_tool_format(h)->grammar(chat, only);
}

}  // namespace

// Every one-call model turn the template renders (what the model was trained
// to emit), with the reasoning block cut where the sampler's gate hands over,
// is accepted by the "required" grammar and by the grammar naming its tool —
// on every measured template, both tool sets.
TEST(ToolFormat, RequiredGrammarAcceptsWhatTheTemplateRendersAsACall) {
    std::map<std::string, int> accepted;
    for (const std::string& h : qinf::tools::measured_tool_templates()) {
        const auto* fmt = qinf::tools::lookup_tool_format(h);
        const ToyVocab vocab = toy_for(fmt->dialect());
        const std::string call_end = fmt->dialect() == qinf::tools::ToolDialect::Gemma4 ? "<tool_call|>" : "</tool_call>";
        for (const ojson& c : cases()["cases"]) {
            const std::string name = c["name"].get<std::string>();
            if (name == "number_null_escape_forms" || name == "content_then_call") continue;   // asserted below
            const ToolChat full = chat_of(c);
            for (size_t k = 0; k < full.messages.size(); ++k) {
                const ToolTurn& m = full.messages[k];
                if (m.tool_calls.size() != 1 || !py_strip_ascii(m.content).empty()) continue;
                ToolChat before = full, upto = full;
                before.messages.resize(k);
                upto.messages.resize(k + 1);
                upto.add_generation_prompt = false;
                std::string P, Q;
                try { P = render(h, before); Q = render(h, upto); } catch (const std::invalid_argument&) { continue; }
                const std::string empty_thought = "<|channel>thought\n<channel|>";
                size_t cut = P.size();
                if (Q.rfind(P, 0) != 0) cut = P.size() - empty_thought.size();   // Gemma, thinking off (P2)
                std::string gen = Q.substr(cut);
                const auto g = grammar_of(h, before);
                if (!g.gate_closer.empty()) {
                    const size_t close = gen.find(g.gate_closer);
                    if (close != std::string::npos) gen = gen.substr(close + g.gate_closer.size());
                }
                gen = gen.substr(0, gen.find(call_end) + call_end.size());
                const std::string what = h + "/" + name + " k=" + std::to_string(k);
                EXPECT_EQ(grammar_verdict(g.gbnf, vocab, gen), "") << what << "\n" << gen;
                EXPECT_EQ(grammar_verdict(grammar_of(h, before, &m.tool_calls[0].name).gbnf, vocab, gen), "") << what;
                ++accepted[h];
            }
        }
    }
    for (const std::string& h : qinf::tools::measured_tool_templates())
        EXPECT_GE(accepted[h], 8) << h << ": too few accepted calls";
}

TEST(ToolFormat, RequiredGrammarRefusesWhatIsNotACallOfTheTools) {
    const ToolChat weather = case_named("tools_only");
    ToolChat rich = case_named("bool_true_arg");
    rich.messages.resize(1);
    const std::string time_name = "get_time";
    // Hermes: another tool's name, text before the call, the named grammar.
    {
        const ToyVocab v = toy_for(qinf::tools::ToolDialect::HermesJson);
        ToolChat off = weather;
        off.enable_thinking = false;
        const std::string call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        EXPECT_EQ(grammar_verdict(grammar_of(kQwen3, off).gbnf, v, call), "");
        EXPECT_NE(grammar_verdict(grammar_of(kQwen3, off).gbnf, v,
                  "<tool_call>\n{\"name\": \"get_wether\", \"arguments\": {}}\n</tool_call>"), "");
        EXPECT_NE(grammar_verdict(grammar_of(kQwen3, off).gbnf, v, "Sure! " + call), "");
        EXPECT_NE(grammar_verdict(grammar_of(kQwen3, off, &time_name).gbnf, v, call), "");
        const std::string nope = "nope";
        EXPECT_THROW(grammar_of(kQwen3, off, &nope), std::invalid_argument);
    }
    // Qwen-XML: an unknown parameter, an integer that is not one, an enum miss.
    {
        const ToyVocab v = toy_for(qinf::tools::ToolDialect::QwenXml);
        const auto g = grammar_of(kQwen35, rich).gbnf;
        const std::string head = "<tool_call>\n<function=create_event>\n<parameter=title>\nT\n</parameter>\n";
        EXPECT_EQ(grammar_verdict(g, v, head + "<parameter=duration_min>\n90\n</parameter>\n</function>\n</tool_call>"), "");
        EXPECT_NE(grammar_verdict(g, v, head + "<parameter=venue>\nx\n</parameter>\n</function>\n</tool_call>"), "");
        EXPECT_NE(grammar_verdict(g, v, head + "<parameter=duration_min>\nninety\n</parameter>\n</function>\n</tool_call>"), "");
        ToolChat w = weather;
        w.enable_thinking = false;
        const auto gw = grammar_of(kQwen35, w).gbnf;
        const std::string wh = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n";
        EXPECT_EQ(grammar_verdict(gw, v, wh + "<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>"), "");
        EXPECT_NE(grammar_verdict(gw, v, wh + "<parameter=unit>\nkelvin\n</parameter>\n</function>\n</tool_call>"), "");
        // A multi-line string value is text, as the template renders it.
        EXPECT_EQ(grammar_verdict(gw, v, "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew\nYork\n"
                                         "</parameter>\n</function>\n</tool_call>"), "");
    }
    // Gemma: typed values, the string delimiter, an enum miss.
    {
        const ToyVocab v = toy_for(qinf::tools::ToolDialect::Gemma4);
        const auto g = grammar_of(kGemma, rich).gbnf;
        EXPECT_EQ(grammar_verdict(g, v, "<|tool_call>call:create_event{all_day:true,duration_min:90,title:<|\"|>a <b>, c<|\"|>}"
                                        "<tool_call|>"), "");
        EXPECT_NE(grammar_verdict(g, v, "<|tool_call>call:create_event{duration_min:<|\"|>90<|\"|>}<tool_call|>"), "");
        const auto gw = grammar_of(kGemma, weather).gbnf;
        EXPECT_NE(grammar_verdict(gw, v, "<|tool_call>call:get_weather{unit:<|\"|>kelvin<|\"|>}<tool_call|>"), "");
    }
}

// The gate tokens follow the template's thinking state.
TEST(ToolFormat, RequiredGrammarGatesTheReasoningBlock) {
    ToolChat chat = case_named("tools_only");
    auto g = grammar_of(kQwen3, chat);                 // Qwen3: thinks by default, opens <think> itself
    EXPECT_EQ(g.gate_opener + g.gate_closer, "<think></think>");
    EXPECT_FALSE(g.gate_start_inside);
    g = grammar_of(kQwen38_27, chat);                  // 3.8: the prompt opened <think>
    EXPECT_EQ(g.gate_closer, "</think>");
    EXPECT_TRUE(g.gate_start_inside);
    g = grammar_of(kQwen35, chat);                     // 3.5: thinking off by default — no gate
    EXPECT_EQ(g.gate_closer, "");
    g = grammar_of(kGemma, chat);
    EXPECT_EQ(g.gate_closer, "");
    chat.enable_thinking = true;
    g = grammar_of(kGemma, chat);
    EXPECT_EQ(g.gate_opener + g.gate_closer, "<|channel><channel|>");
    chat.enable_thinking = false;
    EXPECT_EQ(grammar_of(kQwen3, chat).gate_closer, "");
}

// The live failure of 2026-10-27: under "required", Qwen3-0.6B's first token
// was " Sure". GrammarVocab marks any token whose first character is in a
// standalone character class as valid, and the call's leading whitespace was
// one — the token passed, the grammar died on "Sure", the slot was forced to
// stop. Whitespace is now literals: a multi-character token that leaves the
// call's form is refused at the start, on every format.
TEST(ToolFormat, RequiredGrammarRefusesATextTokenAtTheStart) {
    ToolChat chat = case_named("tools_only");
    chat.enable_thinking = false;
    for (const std::string& h : qinf::tools::measured_tool_templates()) {
        const auto* fmt = qinf::tools::lookup_tool_format(h);
        ToyVocab vocab = toy_for(fmt->dialect());
        for (const char* t : {" Sure", "\nSure", "  Here", "\n\nI"}) vocab.v.push_back(t);
        auto g = qinf::GrammarVocab::parse_impl(fmt->grammar(chat, nullptr).gbnf);
        ASSERT_TRUE(g) << h;
        const auto valid = g->get_valid_tokens(vocab.v);
        for (size_t id = vocab.v.size() - 4; id < vocab.v.size(); ++id)
            EXPECT_EQ(std::count(valid.begin(), valid.end(), static_cast<int32_t>(id)), 0)
                << h << ": " << ojson(vocab.v[id]).dump() << " is valid at the start";
        EXPECT_NE(std::count(valid.begin(), valid.end(), static_cast<int32_t>(256)), 0)   // the call opener
            << h << ": the opener is not valid at the start";
    }
}
