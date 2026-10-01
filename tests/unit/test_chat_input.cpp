// test_chat_input.cpp — co-located unit test for src/cli/chat_input.cpp.
// The chat loop's input rules, without a model: turns end on an empty line,
// exit/quit end the conversation, end of input ends it after the last turn,
// and a bare empty line is never a turn (the stdin-EOF loop, 2026-09-30).

#include <gtest/gtest.h>

#include <sstream>

#include "../../src/cli/chat_input.h"

TEST(ChatInput, TurnEndsOnEmptyLine) {
    std::istringstream in("hello\nworld\n\nnext\n");
    const ChatInput t = read_user_turn(in);
    EXPECT_EQ(t.kind, ChatInputKind::Turn);
    EXPECT_EQ(t.text, "hello\nworld");
}

TEST(ChatInput, EndOfInputDeliversTheLastTurnThenEnds) {
    std::istringstream in("only turn\n");   // no terminating empty line
    const ChatInput t = read_user_turn(in);
    EXPECT_EQ(t.kind, ChatInputKind::Turn);
    EXPECT_EQ(t.text, "only turn");
    EXPECT_EQ(read_user_turn(in).kind, ChatInputKind::EndOfInput);
    EXPECT_EQ(read_user_turn(in).kind, ChatInputKind::EndOfInput);   // stays ended
}

TEST(ChatInput, EmptyInputIsEndOfInputNotAnEmptyTurn) {
    std::istringstream in("");
    EXPECT_EQ(read_user_turn(in).kind, ChatInputKind::EndOfInput);
}

TEST(ChatInput, BareEmptyLinesAreSkipped) {
    std::istringstream in("\n\n\nhi\n\n");
    const ChatInput t = read_user_turn(in);
    EXPECT_EQ(t.kind, ChatInputKind::Turn);
    EXPECT_EQ(t.text, "hi");
    EXPECT_EQ(read_user_turn(in).kind, ChatInputKind::EndOfInput);
}

TEST(ChatInput, ExitAndQuitEndTheConversation) {
    std::istringstream a("exit\n"), b("quit\n"), c("partly typed\nexit\n");
    EXPECT_EQ(read_user_turn(a).kind, ChatInputKind::Exit);
    EXPECT_EQ(read_user_turn(b).kind, ChatInputKind::Exit);
    const ChatInput t = read_user_turn(c);   // the partial turn is dropped, as before
    EXPECT_EQ(t.kind, ChatInputKind::Exit);
    EXPECT_EQ(t.text, "");
}

TEST(ChatInput, TwoTurnsThenExit) {
    std::istringstream in("first\n\nsecond\n\nexit\n");
    EXPECT_EQ(read_user_turn(in).text, "first");
    EXPECT_EQ(read_user_turn(in).text, "second");
    EXPECT_EQ(read_user_turn(in).kind, ChatInputKind::Exit);
}
