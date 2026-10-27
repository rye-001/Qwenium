// test_tool_request.cpp — co-located unit test for src/server/tool_request.cpp.
// The tool contract of /v1/chat/completions (docs/plan-tool-calling.md §3–§4,
// P3): which requests are tool requests, how one parses into a ToolChat (the
// client's key order kept), every refusal by name (field, expected, actual),
// and the reply message. P0's refusals survive only for the legacy fields.

#include <gtest/gtest.h>

#include <regex>
#include <set>
#include <stdexcept>

#include "tool_request.h"

using nlohmann::json;
using qinf::server::is_tool_request;
using qinf::server::parse_tool_request;
using qinf::server::refuse_legacy_tool_fields;

namespace {

const char* kWeather =
    R"({"type":"function","function":{"name":"get_weather","description":"Weather.",)"
    R"("parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}})";

// A tool request body: `extra` is spliced in after messages (raw JSON text, so
// key order is exactly what the client sent).
std::string body(const std::string& messages, const std::string& extra = "") {
    return std::string(R"({"messages":)") + messages + R"(,"tools":[)" + kWeather + "]" +
           (extra.empty() ? "" : "," + extra) + "}";
}

const std::string kUser = R"([{"role":"user","content":"Weather in Paris?"}])";

std::string refusal(const std::string& b) {
    try {
        parse_tool_request(b);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "(parsed, not refused)";
}

}  // namespace

TEST(ToolRequest, LegacyFunctionCallingIsRefusedOnEveryRequest) {
    EXPECT_EQ(refuse_legacy_tool_fields(json::parse(R"({"messages":[],"functions":[{}]})")),
              "'functions': expected absent (deprecated; send 'tools'), actual: an array of 1");
    EXPECT_EQ(refuse_legacy_tool_fields(json::parse(R"({"messages":[],"function_call":"auto"})")),
              "'function_call': expected absent (deprecated; send 'tool_choice'), actual: \"auto\"");
    EXPECT_EQ(refuse_legacy_tool_fields(json::parse(
                  R"({"messages":[{"role":"user","content":"x"},{"role":"function","name":"f","content":"1"}]})")),
              "'messages[1].role': expected \"tool\" (role \"function\" is deprecated), actual: \"function\"");
    EXPECT_EQ(refuse_legacy_tool_fields(json::parse(
                  R"({"messages":[{"role":"assistant","content":null,"function_call":{"name":"f"}}]})")),
              "'messages[0].function_call': expected absent (deprecated; send 'tool_calls'), actual: an object");
    EXPECT_EQ(refuse_legacy_tool_fields(json::parse(R"({"messages":[{"role":"user","content":"x"}],"functions":null})")), "");
}

TEST(ToolRequest, WhatIsAToolRequest) {
    EXPECT_FALSE(is_tool_request(json::parse(R"({"messages":[{"role":"user","content":"hi"}]})")));
    EXPECT_FALSE(is_tool_request(json::parse(R"({"messages":[{"role":"user","content":"hi"}],"tools":null})")));
    EXPECT_FALSE(is_tool_request(json::parse(
        R"({"messages":[{"role":"assistant","content":"ok","tool_calls":[]}]})")));
    EXPECT_TRUE(is_tool_request(json::parse(body(kUser))));
    EXPECT_TRUE(is_tool_request(json::parse(R"({"messages":[{"role":"tool","tool_call_id":"c","content":"1"}]})")));
    EXPECT_TRUE(is_tool_request(json::parse(
        R"({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c"}]}]})")));
    EXPECT_TRUE(is_tool_request(json::parse(R"({"messages":[{"role":"user","content":"hi"}],"tool_choice":"none"})")));
}

TEST(ToolRequest, ParsesAFullRequestKeepingTheClientsKeyOrder) {
    const auto r = parse_tool_request(body(
        R"([{"role":"system","content":[{"type":"text","text":"Be "},{"type":"text","text":"terse."}]},)"
        R"({"role":"user","content":"Weather in Paris?"},)"
        R"({"role":"assistant","content":null,"reasoning_content":"Call it.","tool_calls":[)"
        R"({"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"z\":1,\"city\":\"Paris\"}"}}]},)"
        R"({"role":"tool","tool_call_id":"call_1","content":"15 C"}])",
        R"("tool_choice":"none","chat_template_kwargs":{"enable_thinking":true},"reasoning_effort":"low")"));
    EXPECT_EQ(r.choice, qinf::server::ToolChoice::None);
    const auto& c = r.chat;
    EXPECT_EQ(c.tools[0]["function"].begin().key(), "name");                // order kept, not sorted
    EXPECT_EQ(c.tools[0]["function"]["parameters"].begin().key(), "type");
    ASSERT_EQ(c.messages.size(), 4u);
    EXPECT_EQ(c.messages[0].content, "Be terse.");
    EXPECT_EQ(c.messages[2].content, "");
    EXPECT_EQ(*c.messages[2].reasoning, "Call it.");
    ASSERT_EQ(c.messages[2].tool_calls.size(), 1u);
    EXPECT_EQ(c.messages[2].tool_calls[0].arguments.dump(), "{\"z\":1,\"city\":\"Paris\"}");
    EXPECT_EQ(c.messages[3].tool_call_id, "call_1");
    EXPECT_EQ(c.enable_thinking, true);
    EXPECT_EQ(c.reasoning_effort, "low");
    EXPECT_FALSE(c.preserve_thinking.has_value());
    EXPECT_TRUE(c.add_generation_prompt);
}

TEST(ToolRequest, RefusesWhatV1DoesNotServeWithTools) {
    EXPECT_EQ(refusal(body(kUser, R"("stream":true)")),
              "'stream': expected false with tools (streamed tool calls are not served in v1), actual: true");
    EXPECT_EQ(refusal(body(kUser, R"("conversation_id":"new")")),
              "'conversation_id': expected absent with tools (not served in v1), actual: \"new\"");
    EXPECT_EQ(refusal(body(kUser, R"("grammar":"root ::= \"x\"")")),
              "'grammar': expected absent with tools (the tool format is the output constraint), actual: a grammar");
    EXPECT_EQ(refusal(body(R"([{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:,"}}]}])")),
              "'messages[0].content[0].type': expected \"text\" (images and audio are not served with tools in v1), "
              "actual: \"image_url\"");
    EXPECT_EQ(refusal(body(kUser, R"("tool_choice":"any")")),
              "'tool_choice': expected \"auto\", \"none\", \"required\" or {\"type\":\"function\",\"function\":"
              "{\"name\":...}}, actual: \"any\"");
    EXPECT_EQ(refusal(body(kUser, R"("tool_choice":{"type":"function","function":{"name":"get_time"}})")),
              "'tool_choice.function.name': expected the name of one of the request's tools, actual: \"get_time\"");
    EXPECT_EQ(refusal(body(kUser, R"("tool_choice":{"type":"allowed_tools"})")),
              "'tool_choice.type': expected \"function\", actual: \"allowed_tools\"");
    EXPECT_EQ(refusal(body(kUser, R"("parallel_tool_calls":false)")),
              "'parallel_tool_calls': expected true or absent (false is not enforced in v1), actual: false");
}

TEST(ToolRequest, RequiredAndNamedToolChoice) {
    EXPECT_EQ(parse_tool_request(body(kUser, R"("tool_choice":"required")")).choice,
              qinf::server::ToolChoice::Required);
    const auto r = parse_tool_request(body(kUser, R"("tool_choice":{"type":"function","function":{"name":"get_weather"}})"));
    EXPECT_EQ(r.choice, qinf::server::ToolChoice::Named);
    EXPECT_EQ(r.named, "get_weather");
}

TEST(ToolRequest, RefusesMalformedTools) {
    const auto with_tools = [](const std::string& tools) {
        return std::string(R"({"messages":[{"role":"user","content":"x"}],"tools":)") + tools + "}";
    };
    EXPECT_EQ(refusal(R"({"messages":[{"role":"tool","tool_call_id":"c","content":"1"}]})"),
              "'tools': expected the request's tool list (a request with tool calls, tool results or tool_choice "
              "must send its tools), actual: absent");
    EXPECT_EQ(refusal(with_tools("[]")), "'tools': expected an array of 1 to 128 tools, actual: an array of 0");
    EXPECT_EQ(refusal(with_tools(R"([{"type":"web_search"}])")),
              "'tools[0].type': expected \"function\", actual: \"web_search\"");
    EXPECT_EQ(refusal(with_tools(R"([{"type":"function","function":{"name":"get weather"}}])")),
              "'tools[0].function.name': expected a name matching ^[A-Za-z0-9_-]{1,64}$, actual: \"get weather\"");
    EXPECT_EQ(refusal(with_tools(std::string("[") + kWeather + "," + kWeather + "]")),
              "'tools[1].function.name': expected a name not used by another tool, actual: \"get_weather\"");
    EXPECT_EQ(refusal(with_tools(R"([{"type":"function","function":{"name":"f","parameters":{"type":"array"}}}])")),
              "'tools[0].function.parameters.type': expected \"object\", actual: \"array\"");
    EXPECT_EQ(refusal(with_tools(R"([{"type":"function","function":{"name":"f","strict":true}}])")),
              "'tools[0].function.strict': expected false or absent (only a top-level schema subset is checked, "
              "docs/plan-tool-calling.md §5), actual: true");
}

TEST(ToolRequest, RefusesMalformedMessages) {
    EXPECT_EQ(refusal(body(R"([{"role":"developer","content":"x"}])")),
              "'messages[0].role': expected \"system\", \"user\", \"assistant\" or \"tool\", actual: \"developer\"");
    EXPECT_EQ(refusal(body(R"([{"role":"user","content":"x","tool_calls":[]}])")),
              "'messages[0].tool_calls': expected an array on an assistant message, actual: an array of 0");
    const auto call = [](const std::string& fn, const std::string& id = R"("id":"c1",)") {
        return body(std::string(R"([{"role":"user","content":"x"},{"role":"assistant","content":null,"tool_calls":[{)") +
                    id + R"("type":"function","function":)" + fn + "}]}]");
    };
    EXPECT_EQ(refusal(call(R"({"name":"get_weather","arguments":"{}"})", "")),
              "'messages[1].tool_calls[0].id': expected a non-empty string, actual: absent");
    EXPECT_EQ(refusal(call(R"({"name":"get_time","arguments":"{}"})")),
              "'messages[1].tool_calls[0].function.name': expected the name of one of the request's tools, "
              "actual: \"get_time\"");
    EXPECT_EQ(refusal(call(R"({"name":"get_weather","arguments":"{\"city\": "})")),
              "'messages[1].tool_calls[0].function.arguments': expected a JSON object, as a string, "
              "actual: \"{\\\"city\\\": \"");
    EXPECT_EQ(refusal(call(R"({"name":"get_weather","arguments":"[1]"})")),
              "'messages[1].tool_calls[0].function.arguments': expected a JSON object, as a string, actual: \"[1]\"");
    EXPECT_EQ(refusal(body(R"([{"role":"user","content":"x"},{"role":"tool","content":"1"}])")),
              "'messages[1].tool_call_id': expected a string, actual: absent");
    EXPECT_EQ(refusal(body(kUser, R"("chat_template_kwargs":{"add_vision_id":true})")),
              "'chat_template_kwargs.add_vision_id': expected one of enable_thinking, preserve_thinking, "
              "reasoning_effort, actual: present");
    EXPECT_EQ(refusal(body(kUser, R"("enable_thinking":"yes")")),
              "'enable_thinking': expected a boolean, actual: \"yes\"");
}

TEST(ToolRequest, ReplyMessageShapes) {
    qinf::tools::ParsedReply calls;
    calls.reasoning = "Two lookups.";
    calls.calls.push_back({"get_weather", qinf::tools::ojson::parse(R"({"city":"München","n":1})")});
    int n = 0;
    const json m = qinf::server::tool_reply_message(calls, [&] { return "call_" + std::to_string(++n); });
    EXPECT_EQ(m.dump(), R"({"content":null,"reasoning_content":"Two lookups.","role":"assistant",)"
                        R"("tool_calls":[{"function":{"arguments":"{\"city\":\"München\",\"n\":1}","name":"get_weather"},)"
                        R"("id":"call_1","type":"function"}]})");
    qinf::tools::ParsedReply text;
    text.content = "It is sunny.";
    EXPECT_EQ(qinf::server::tool_reply_message(text, [] { return std::string("x"); }).dump(),
              R"({"content":"It is sunny.","role":"assistant"})");
}

TEST(ToolRequest, CallIdsAreUniqueAndOpenAIShaped) {
    std::set<std::string> ids;
    for (int i = 0; i < 1000; ++i) {
        const std::string id = qinf::server::new_tool_call_id();
        ASSERT_TRUE(std::regex_match(id, std::regex("call_[0-9a-f]{24}"))) << id;
        ids.insert(id);
    }
    EXPECT_EQ(ids.size(), 1000u);
}
