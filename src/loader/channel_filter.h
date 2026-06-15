#pragma once
// channel_filter.h — Gemma 4 channel-stream filter (output text protocol).
//
// Gemma 4 frames its output in channels delimited by the <|channel> /
// <channel|> control specials: a "thought" channel (internal reasoning,
// suppressed) and a "model" channel (the user-facing answer, shown). The
// channel name can arrive split across several decoded tokens and <|channel>
// is its own token, so contiguous string-stripping is unreliable — this parses
// at the token level instead.
//
// ONE shared implementation, two entry points: the CLI streams tokens through
// feed() (a per-turn instance), the server holds the assembled response and
// runs the same state machine over it via the one-shot strip(). For non-Gemma-4
// models the control markers never appear, so both paths are inert.

#include <string>

class ChannelFilter {
public:
    // Feed one decoded token's text; returns the text to emit ("" == suppress).
    // Stateful across calls within a single generation; call reset() per turn.
    // Chunk-safe: a piece carrying both a channel name and following content
    // (e.g. "model\n<answer>") is split correctly.
    std::string feed(const std::string& token_str);

    // Reset to the Normal state (call between turns when reusing an instance).
    void reset();

    // One-shot: run the same state machine over a complete string. For callers
    // that already hold the full text (e.g. the server's assembled response).
    static std::string strip(const std::string& full);

private:
    // Process one piece — an exact control marker or a run of plain text —
    // advancing the channel state. feed() splits its input into such pieces.
    std::string consume(const std::string& s);

    // Emit text in the Normal/model channel, dropping the blank line(s) that
    // follow a channel header so the answer never starts with a leading newline.
    std::string emit_normal(const std::string& s);

    bool in_channel_header_     = false;  // saw <|channel>, reading the name
    bool in_suppressed_channel_ = false;  // inside the thought channel
    bool strip_leading_ws_      = false;  // drop blank lines just after a header
    std::string channel_name_buf_;
};
