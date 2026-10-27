// test_tool_request.cpp — co-located unit test for src/server/tool_request.cpp.
// P0 of docs/plan-tool-calling.md: every tool field of a chat request is
// refused by name (field, expected, actual), and a plain chat request passes.

#include <gtest/gtest.h>

#include "tool_request.h"

using nlohmann::json;
using qinf::server::refuse_tool_fields;

namespace {

json chat(json extra = json::object()) {
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
    for (auto it = extra.begin(); it != extra.end(); ++it) body[it.key()] = it.value();
    return body;
}

const json kTool = {{"type", "function"},
                    {"function", {{"name", "get_weather"},
                                  {"parameters", {{"type", "object"}}}}}};

}  // namespace

TEST(ToolRequest, PlainChatPasses) {
    EXPECT_EQ(refuse_tool_fields(chat()), "");
    EXPECT_EQ(refuse_tool_fields(chat({{"temperature", 0}, {"stop", "x"}})), "");
}

TEST(ToolRequest, NullFieldsCountAsAbsent) {
    EXPECT_EQ(refuse_tool_fields(chat({{"tools", nullptr}, {"tool_choice", nullptr}})), "");
    json body = chat();
    body["messages"].push_back({{"role", "assistant"}, {"content", "ok"},
                                {"tool_calls", nullptr}, {"function_call", nullptr}});
    EXPECT_EQ(refuse_tool_fields(body), "");
}

TEST(ToolRequest, EmptyToolCallsOnAnAssistantMessagePasses) {
    json body = chat();
    body["messages"].push_back({{"role", "assistant"}, {"content", "ok"},
                                {"tool_calls", json::array()}});
    EXPECT_EQ(refuse_tool_fields(body), "");
}

TEST(ToolRequest, ToolsAreRefusedByName) {
    EXPECT_EQ(refuse_tool_fields(chat({{"tools", json::array({kTool, kTool})}})),
              "'tools': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: an array of 2");
    // An empty array is refused too: OpenAI rejects it, and it is not "no field".
    EXPECT_EQ(refuse_tool_fields(chat({{"tools", json::array()}})),
              "'tools': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: an array of 0");
}

TEST(ToolRequest, EveryTopLevelToolFieldIsRefused) {
    EXPECT_EQ(refuse_tool_fields(chat({{"tool_choice", "none"}})),
              "'tool_choice': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: \"none\"");
    EXPECT_EQ(refuse_tool_fields(chat({{"parallel_tool_calls", false}})),
              "'parallel_tool_calls': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: false");
    EXPECT_EQ(refuse_tool_fields(chat({{"functions", json::array({kTool["function"]})}})),
              "'functions': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: an array of 1");
    EXPECT_EQ(refuse_tool_fields(chat({{"function_call", {{"name", "get_weather"}}}})),
              "'function_call': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: an object");
}

TEST(ToolRequest, ToolAndFunctionRolesAreRefused) {
    json body = chat();
    body["messages"].push_back({{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "15C"}});
    EXPECT_EQ(refuse_tool_fields(body),
              "'messages[1].role': expected \"system\", \"user\" or \"assistant\" (tool calling "
              "is not supported yet, docs/plan-tool-calling.md), actual: \"tool\"");
    body["messages"][1]["role"] = "function";
    EXPECT_EQ(refuse_tool_fields(body),
              "'messages[1].role': expected \"system\", \"user\" or \"assistant\" (tool calling "
              "is not supported yet, docs/plan-tool-calling.md), actual: \"function\"");
}

TEST(ToolRequest, AssistantToolCallsAreRefused) {
    json body = chat();
    body["messages"].push_back(
        {{"role", "assistant"}, {"content", nullptr},
         {"tool_calls", json::array({{{"id", "call_1"}, {"type", "function"},
                                      {"function", {{"name", "get_weather"},
                                                    {"arguments", "{}"}}}}})}});
    EXPECT_EQ(refuse_tool_fields(body),
              "'messages[1].tool_calls': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: an array of 1");
    body["messages"][1].erase("tool_calls");
    body["messages"][1]["function_call"] = {{"name", "get_weather"}, {"arguments", "{}"}};
    EXPECT_EQ(refuse_tool_fields(body),
              "'messages[1].function_call': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: an object");
}

TEST(ToolRequest, LongScalarsAreShortened) {
    // 37 characters of the dump (the opening quote + 36 x), then "...".
    EXPECT_EQ(refuse_tool_fields(chat({{"tool_choice", std::string(100, 'x')}})),
              "'tool_choice': expected absent (tool calling is not supported yet, "
              "docs/plan-tool-calling.md), actual: \"" + std::string(36, 'x') + "...");
}
