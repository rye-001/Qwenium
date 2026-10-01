#include "chat_input.h"

ChatInput read_user_turn(std::istream& in) {
    std::string text;
    std::string line;
    while (std::getline(in, line)) {
        if (line == "exit" || line == "quit") return {ChatInputKind::Exit, ""};
        if (line.empty()) {
            if (text.empty()) continue;   // a bare empty line is not a turn
            break;
        }
        if (!text.empty()) text += "\n";
        text += line;
    }
    if (text.empty()) return {ChatInputKind::EndOfInput, ""};
    return {ChatInputKind::Turn, text};
}
