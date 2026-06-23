#include "channel_filter.h"

#include <array>

void ChannelFilter::reset() {
    in_channel_header_   = false;
    in_thought_channel_  = false;
    strip_leading_ws_    = false;
    channel_name_buf_.clear();
}

std::string ChannelFilter::emit_normal(const std::string& s) {
    if (!strip_leading_ws_) return s;
    const size_t i = s.find_first_not_of("\n\r");
    if (i == std::string::npos) return "";  // all whitespace; keep stripping
    strip_leading_ws_ = false;
    return s.substr(i);
}

std::string ChannelFilter::feed(const std::string& s) {
    // Split the input on the control markers and route each piece (an exact
    // marker or a run of plain text) through consume(). In the streaming case
    // each registered special already arrives as its own piece, so this is a
    // no-op split; but it also keeps feed() robust if a marker ever arrives
    // glued to surrounding text, and lets strip() reuse exactly this logic.
    static const std::array<std::string, 4> markers = {
        "<|channel>", "<channel|>", "<|turn>", "<turn|>"};
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        size_t best = std::string::npos;
        const std::string* hit = nullptr;
        for (const auto& m : markers) {
            const size_t p = s.find(m, i);
            if (p < best) { best = p; hit = &m; }
        }
        if (hit == nullptr) { out += consume(s.substr(i)); break; }
        if (best > i) out += consume(s.substr(i, best - i));  // text run
        out += consume(*hit);                                 // the marker
        i = best + hit->size();
    }
    return out;
}

std::string ChannelFilter::consume(const std::string& s) {
    // One piece: either an exact control marker or a run of plain text.
    if (s == "<|turn>" || s == "<turn|>") return "";
    if (s == "<|channel>") {
        in_channel_header_ = true;
        channel_name_buf_.clear();
        return "";
    }
    if (s == "<channel|>") {
        in_channel_header_  = false;
        in_thought_channel_ = false;
        return "";
    }

    if (in_channel_header_) {
        // Accumulate the channel name across however many pieces it spans, then
        // classify once a known name appears. Chunk-safe: any text after the
        // name in this same piece is routed as channel content.
        channel_name_buf_ += s;
        const size_t tpos = channel_name_buf_.find("thought");
        if (tpos != std::string::npos) {
            in_channel_header_  = false;
            in_thought_channel_ = true;
            const std::string rest = channel_name_buf_.substr(tpos + 7);  // after "thought"
            channel_name_buf_.clear();
            return show_thought_ ? rest : "";  // remainder is thought content
        }
        const size_t pos = channel_name_buf_.find("model");
        if (pos != std::string::npos) {
            in_channel_header_  = false;
            in_thought_channel_ = false;
            strip_leading_ws_   = true;
            const std::string rest = channel_name_buf_.substr(pos + 5);  // after "model"
            channel_name_buf_.clear();
            return emit_normal(rest);
        }
        return "";  // name not yet complete
    }

    if (in_thought_channel_) return show_thought_ ? s : "";
    return emit_normal(s);
}

std::string ChannelFilter::strip(const std::string& full) {
    // A fresh streaming pass over the whole string — feed() already splits on
    // the markers, so strip() and feed() share one implementation.
    ChannelFilter f;
    return f.feed(full);
}
