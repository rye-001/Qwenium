// test_channel_filter.cpp — Gemma 4 channel-stream filter (loader/channel_filter).
//
// Pure string logic, no model file. Covers the streaming feed() path (used by
// the CLI) and the one-shot strip() path (used by the server), including the
// token-boundary fragmentation that defeated the earlier contiguous string
// strip (docs/server-image-multirequest-bug.md §5c).

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/loader/channel_filter.h"

namespace {

// Feed a sequence of decoded "tokens" through one filter, concatenate output.
std::string run(ChannelFilter& f, const std::vector<std::string>& toks) {
    std::string out;
    for (const auto& t : toks) out += f.feed(t);
    return out;
}

}  // namespace

// ── Normal text passes through untouched (non-Gemma-4 case) ──────────────────
TEST(ChannelFilter, NormalTextUnchanged) {
    ChannelFilter f;
    EXPECT_EQ(run(f, {"Hello", ", ", "world", "!"}), "Hello, world!");
}

// ── Thought channel is suppressed; model channel is shown ────────────────────
TEST(ChannelFilter, ThoughtSuppressedModelShown) {
    ChannelFilter f;
    // <|channel> thought <channel|> <|channel> model \n <answer> <channel|>
    std::string out = run(f, {
        "<|channel>", "thought", "let me reason", "<channel|>",
        "<|channel>", "model", "\n", "The answer is 42.", "<channel|>"});
    EXPECT_EQ(out, "The answer is 42.");
}

// ── Token-split markers: <|channel> and the name arrive separately ───────────
// This is exactly the fragmentation the contiguous string-strip could not catch.
TEST(ChannelFilter, SplitThoughtMarkerSuppressed) {
    ChannelFilter f;
    std::string out = run(f, {"<|channel>", "thought", "\n", "secret reasoning"});
    EXPECT_EQ(out, "") << "thought-channel content leaked";
}

// ── Empty thought channel then model answer (the observed G4 behavior) ───────
TEST(ChannelFilter, EmptyThoughtThenModelAnswer) {
    ChannelFilter f;
    std::string out = run(f, {
        "<|channel>", "thought", "<channel|>",
        "<|channel>", "model", "\n", "A spine X-ray."});
    EXPECT_EQ(out, "A spine X-ray.");
}

// ── turn markers are dropped ─────────────────────────────────────────────────
TEST(ChannelFilter, TurnMarkersDropped) {
    ChannelFilter f;
    EXPECT_EQ(run(f, {"hi", "<|turn>", "<turn|>", "bye"}), "hibye");
}

// ── Chunk-safe: a single piece carrying "model\n<answer>" is split correctly ──
TEST(ChannelFilter, ChunkCarriesNameAndContent) {
    ChannelFilter f;
    std::string out = run(f, {"<|channel>", "model\nHello there"});
    EXPECT_EQ(out, "Hello there") << "name+content chunk mishandled";
}

// ── No leaked control markers regardless of structure ────────────────────────
TEST(ChannelFilter, NoMarkersLeak) {
    ChannelFilter f;
    std::string out = run(f, {
        "<|channel>", "model", "\n", "Answer<channel|>", "<|turn>"});
    EXPECT_EQ(out.find("<|channel>"), std::string::npos);
    EXPECT_EQ(out.find("<channel|>"), std::string::npos);
    EXPECT_EQ(out.find("<|turn>"),    std::string::npos);
}

// ── reset() clears state between turns ───────────────────────────────────────
TEST(ChannelFilter, ResetClearsState) {
    ChannelFilter f;
    run(f, {"<|channel>", "thought", "reasoning"});  // leaves it suppressing
    f.reset();
    EXPECT_EQ(f.feed("fresh turn"), "fresh turn");
}

// ── show_thought=true: thought content is emitted alongside the answer ───────
TEST(ChannelFilter, ShowThoughtEmitsBoth) {
    ChannelFilter f(/*show_thought=*/true);
    std::string out = run(f, {
        "<|channel>", "thought", "let me reason\n", "<channel|>",
        "<|channel>", "model", "\n", "The answer is 42.", "<channel|>"});
    EXPECT_EQ(out, "let me reason\nThe answer is 42.");
}

// ── show_thought=true: still strips all control markers (only content leaks) ──
TEST(ChannelFilter, ShowThoughtNoMarkersLeak) {
    ChannelFilter f(/*show_thought=*/true);
    std::string out = run(f, {
        "<|channel>", "thought", "reasoning", "<channel|>", "<|turn>"});
    EXPECT_EQ(out, "reasoning");
    EXPECT_EQ(out.find("<|channel>"), std::string::npos);
    EXPECT_EQ(out.find("<channel|>"), std::string::npos);
    EXPECT_EQ(out.find("<|turn>"),    std::string::npos);
}

// ── show_thought=true: split thought marker still routes content correctly ────
TEST(ChannelFilter, ShowThoughtSplitMarker) {
    ChannelFilter f(/*show_thought=*/true);
    EXPECT_EQ(run(f, {"<|channel>", "thought", "\n", "secret reasoning"}),
              "\nsecret reasoning");
}

// ── in_thought() tracks the channel so the caller can style reasoning ─────────
TEST(ChannelFilter, InThoughtTracksChannel) {
    ChannelFilter f(/*show_thought=*/true);
    f.feed("<|channel>");
    f.feed("thought");
    f.feed("reasoning");
    EXPECT_TRUE(f.in_thought());
    f.feed("<channel|>");
    f.feed("<|channel>");
    f.feed("model");
    f.feed("answer");
    EXPECT_FALSE(f.in_thought());
}

// ── One-shot strip() over a complete string (server path) ────────────────────
TEST(ChannelFilter, StripWholeString) {
    EXPECT_EQ(
        ChannelFilter::strip(
            "<|channel>thought\ninternal<channel|><|channel>model\nFinal answer."),
        "Final answer.");
}

// ── strip(): thought content removed, model content kept, no markers left ────
TEST(ChannelFilter, StripDropsThoughtKeepsModel) {
    std::string out = ChannelFilter::strip(
        "<|channel>thought\nreason about it<channel|>"
        "<|channel>model\nVisible.<channel|><turn|>");
    EXPECT_EQ(out, "Visible.");
}

// ── strip(): plain text is unchanged ─────────────────────────────────────────
TEST(ChannelFilter, StripPlainTextUnchanged) {
    EXPECT_EQ(ChannelFilter::strip("just a normal answer"), "just a normal answer");
}
