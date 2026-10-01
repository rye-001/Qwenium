#pragma once
// chat_input.h — reads one user turn of the interactive chat (run_chat).
//
// A turn is one or more lines ended by an empty line. `exit` or `quit` on a
// line of its own ends the conversation (a partly typed turn is dropped, as
// before). End of input ends it too, after delivering an unterminated last
// turn — so `printf 'prompt\n' | qwenium --chat` answers once and exits. A
// bare empty line with nothing typed is not a turn and is skipped.
//
// Pure stream logic, split out of chat.cpp so it is testable without a model
// (tests/unit/test_chat_input.cpp).

#include <istream>
#include <string>

enum class ChatInputKind { Turn, Exit, EndOfInput };

struct ChatInput {
    ChatInputKind kind;
    std::string   text;   // the turn's lines joined by '\n'; empty unless kind == Turn
};

ChatInput read_user_turn(std::istream& in);
