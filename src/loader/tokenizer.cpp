#include "tokenizer.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <map>
#include <set>
#include <iostream>
#include <queue>       // std::priority_queue
#include <functional>  // std::greater
#include <vector>      // std::vector (likely already included)

Tokenizer::Tokenizer(const ModelMetadata* metadata, const TokenizerConfig& config)
    : metadata_(metadata), config_(config) {
    if (metadata_->id_to_token.empty()) {
        throw std::runtime_error("Invalid vocabulary: token list is empty.");
    }

    is_llama_tokenizer_ = (config_.normalizer == NormalizerKind::SpaceToUnderscore || metadata_->tokenizer_type == "gemma4");

    // Build token_to_id map
    token_to_id_.reserve(metadata_->id_to_token.size());
    for (size_t i = 0; i < metadata_->id_to_token.size(); ++i) {
        token_to_id_[metadata_->id_to_token[i]] = static_cast<int32_t>(i);
    }

    if (is_llama_tokenizer_) {
        // SentencePiece-derived BPE with byte fallback.
        // Uses score-based BPE merge for encode and ▁→space + byte-fallback decode.
        scores_ = metadata_->scores;
        initialize_special_tokens();

        unk_token_id_ = (metadata_->unknown_token_id >= 0)
                            ? metadata_->unknown_token_id
                            : -1;
        if (unk_token_id_ < 0) {
            for (size_t i = 0; i < metadata_->token_types.size(); ++i) {
                if (metadata_->token_types[i] == TokenType::UNKNOWN) {
                    unk_token_id_ = static_cast<int32_t>(i);
                    break;
                }
            }
        }
        initialize_llama_byte_fallback();
        return;
    }

    // Parse merge rules and populate the optimized map
    merges_.reserve(metadata_->merges.size());
    for (int i = 0; i < metadata_->merges.size(); ++i) {
        const std::string& merge_str = metadata_->merges[i];
        size_t space_pos = merge_str.find(' ');
        if (space_pos == std::string::npos) {
            throw std::runtime_error("Invalid merge format: " + merge_str);
        }
        
        std::string first_token_str = merge_str.substr(0, space_pos);
        std::string second_token_str = merge_str.substr(space_pos + 1);
        
        auto it1 = token_to_id_.find(first_token_str);
        auto it2 = token_to_id_.find(second_token_str);
        if (it1 == token_to_id_.end() || it2 == token_to_id_.end()) {
            // Skip merges with tokens not in the vocabulary
            continue;
        }
        
        std::string merged_token_str = first_token_str + second_token_str;
        auto merged_it = token_to_id_.find(merged_token_str);
        if (merged_it == token_to_id_.end()) {
            throw std::runtime_error("Merged token not found in vocab: " + merged_token_str);
        }
        
        merges_[{it1->second, it2->second}] = {i, merged_it->second};
    }
    
    // Initialize special tokens
    initialize_special_tokens();

    // Find and set the UNK token ID
    unk_token_id_ = -1;
    for (size_t i = 0; i < metadata_->token_types.size(); ++i) {
        if (metadata_->token_types[i] == TokenType::UNKNOWN) {
            unk_token_id_ = static_cast<int32_t>(i);
            break;
        }
    }
    
    // Initialize byte mapping for proper UTF-8 handling
    initialize_byte_mapping();
    
    // Build a comprehensive regex for pre-tokenization
    std::string special_token_pattern;
    for (const auto& [token_str, token_id] : special_tokens_) {
        if (!special_token_pattern.empty()) {
            special_token_pattern += "|";
        }
        // Escape special regex characters in the token string
        std::string escaped_token;
        for (char c : token_str) {
            if (std::string("()[]{}|?*+.").find(c) != std::string::npos) {
                escaped_token += '\\';
            }
            escaped_token += c;
        }
        special_token_pattern += escaped_token;
    }

    std::string base_pattern = R"('(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+)";

    pretokenization_regex_ = std::regex(
        special_token_pattern + "|" + base_pattern,
        std::regex_constants::ECMAScript
    );

    // Which pre-tokenizer this CHECKPOINT asks for. Unknown values keep the
    // GPT-2 pattern: a model nobody measured must not silently inherit
    // someone else's segmentation. The mapping mirrors llama.cpp's
    // llama-vocab.cpp so a GGUF tokenizes the same here as it does there.
    const std::string& pre = metadata_->tokenizer_pre;
    if (pre == "qwen35") {
        pre_kind_ = PreTokenizerKind::Qwen35;
    } else if (pre == "qwen2" || pre == "deepseek-r1-qwen" || pre == "megrez") {
        pre_kind_ = PreTokenizerKind::Qwen2;
    } else {
        pre_kind_ = PreTokenizerKind::Gpt2;
    }
}


void Tokenizer::initialize_special_tokens() {
    // Register BOS / EOS / PAD by ID so is_special_token() works on them
    // even if the caller never looks them up by name.
    special_token_ids_.insert(metadata_->eos_token_id);
    if (metadata_->bos_token_id >= 0)
        special_token_ids_.insert(metadata_->bos_token_id);
    if (metadata_->padding_token_id >= 0)
        special_token_ids_.insert(metadata_->padding_token_id);

    // All CONTROL and USER_DEFINED vocab entries are special by type.
    for (size_t i = 0; i < metadata_->token_types.size(); ++i) {
        if (metadata_->token_types[i] == TokenType::CONTROL ||
            metadata_->token_types[i] == TokenType::USER_DEFINED) {
            special_token_ids_.insert(static_cast<int32_t>(i));
            special_tokens_[metadata_->id_to_token[i]] = static_cast<int32_t>(i);
        }
    }

    // Force-promote extra tokens from config even if their TokenType is NORMAL.
    // Needed for GGUF exports that mislabel chat-control tokens.
    for (const auto& s : config_.extra_chat_specials) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            special_tokens_[s] = it->second;
            special_token_ids_.insert(it->second);
        }
    }
}


void Tokenizer::initialize_byte_mapping() {
    // Standard GPT-2 byte-level mapping to unique Unicode characters.
    std::map<int, int> byte_to_unicode_map;
    std::set<int> printable_bytes;

    // Populate with printable ASCII and some extended chars that are mapped 1-to-1
    for (int i = 33; i <= 126; ++i) { printable_bytes.insert(i); byte_to_unicode_map[i] = i; }
    for (int i = 161; i <= 172; ++i) { printable_bytes.insert(i); byte_to_unicode_map[i] = i; }
    for (int i = 174; i <= 255; ++i) { printable_bytes.insert(i); byte_to_unicode_map[i] = i; }

    // Assign remaining byte values to higher Unicode code points
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (printable_bytes.find(b) == printable_bytes.end()) {
            byte_to_unicode_map[b] = 256 + n;
            n++;
        }
    }

    byte_encoder_.resize(256);
    for (int b = 0; b < 256; ++b) {
        int unicode_val = byte_to_unicode_map[b];
        std::string& mapped_str = byte_encoder_[b];
        
        // Simple UTF-8 encoding for the Unicode code point
        if (unicode_val < 128) {
            mapped_str += static_cast<char>(unicode_val);
        } else {
            mapped_str += static_cast<char>(0xC0 | (unicode_val >> 6));
            mapped_str += static_cast<char>(0x80 | (unicode_val & 0x3F));
        }
        
        byte_decoder_[mapped_str] = b;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Qwen pre-tokenization
//
// Qwen's reference pattern (tokenizer.json, and llama.cpp's
// LLAMA_VOCAB_PRE_TYPE_QWEN35) is:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   | \p{N}
//   |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// This is NOT expressible in std::regex: ECMAScript has no \p{...} Unicode
// property classes, and std::regex matches bytes, so a UTF-8 'ü' is two
// non-ASCII bytes rather than one letter. Hence a hand-written scanner over
// codepoints. Alternatives are tried IN ORDER and the first that matches wins
// (leftmost-first, as ECMAScript and Python's `regex` do) — not longest-match.
//
// What the old GPT-2 pattern got wrong, measured 2026-09-11 against the
// reference tokenizer (docs/note-lens-mlx-swift-server.md §7):
//   · no \s*[\r\n]+ alternative     ⇒ "\n\n" became two tokens, not one
//   · letter runs only space-prefixed ⇒ ".example" became two tokens, not one
//   · [a-zA-Z] instead of \p{L}     ⇒ "Stückzahl" became St|ü|ckzahl
// German prompts ran 11–16% longer in tokens, and every umlauted word was fed
// to the model in a segmentation it was never trained on.

namespace {

// ── UTF-8 ───────────────────────────────────────────────────────────────────
// Decodes one codepoint at `i`, advancing it. Invalid bytes are returned as
// themselves and consume one byte, so a malformed input still terminates.
uint32_t utf8_next(const std::string& s, size_t& i) {
    const unsigned char c = (unsigned char)s[i];
    size_t len = 1;
    uint32_t cp = c;
    if      ((c & 0x80) == 0x00) { len = 1; cp = c; }
    else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
    else                         { i += 1; return c; }
    if (i + len > s.size()) { i += 1; return c; }
    for (size_t k = 1; k < len; ++k) {
        const unsigned char cc = (unsigned char)s[i + k];
        if ((cc & 0xC0) != 0x80) { i += 1; return c; }   // truncated sequence
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += len;
    return cp;
}

bool in_any(uint32_t cp, const std::pair<uint32_t, uint32_t>* r, size_t n) {
    for (size_t k = 0; k < n; ++k) if (cp >= r[k].first && cp <= r[k].second) return true;
    return false;
}
#define RANGES(name) name, sizeof(name) / sizeof(name[0])

// ── Character classes ───────────────────────────────────────────────────────
// ECMAScript/`regex` \s, including the Unicode spaces both treat as whitespace.
const std::pair<uint32_t, uint32_t> kSpace[] = {
    {0x0009, 0x000D}, {0x0020, 0x0020}, {0x0085, 0x0085}, {0x00A0, 0x00A0},
    {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029}, {0x202F, 0x202F},
    {0x205F, 0x205F}, {0x3000, 0x3000}, {0xFEFF, 0xFEFF},
};
bool is_space(uint32_t cp) { return in_any(cp, RANGES(kSpace)); }

// \p{M} — combining marks. The ranges that occur in the scripts this engine is
// used on, plus the general-purpose combining blocks.
const std::pair<uint32_t, uint32_t> kMark[] = {
    {0x0300, 0x036F}, {0x0483, 0x0489}, {0x0591, 0x05BD}, {0x05BF, 0x05BF},
    {0x05C1, 0x05C2}, {0x05C4, 0x05C5}, {0x05C7, 0x05C7}, {0x0610, 0x061A},
    {0x064B, 0x065F}, {0x0670, 0x0670}, {0x06D6, 0x06DC}, {0x06DF, 0x06E4},
    {0x06E7, 0x06E8}, {0x06EA, 0x06ED}, {0x0900, 0x0903}, {0x093A, 0x094F},
    {0x0951, 0x0957}, {0x0962, 0x0963}, {0x0E31, 0x0E31}, {0x0E34, 0x0E3A},
    {0x0E47, 0x0E4E}, {0x1AB0, 0x1AFF}, {0x1DC0, 0x1DFF}, {0x20D0, 0x20F0},
    {0x2CEF, 0x2CF1}, {0x302A, 0x302F}, {0x3099, 0x309A}, {0xFE00, 0xFE0F},
    {0xFE20, 0xFE2F},
};
bool is_mark(uint32_t cp) { return in_any(cp, RANGES(kMark)); }

// \p{N} — numbers. ASCII digits plus the decimal-digit blocks likely to appear
// in real documents. A digit outside these is classed as a letter by the
// fallback below, which is the same bucket the old pattern's punctuation class
// would NOT have used — see the scope note on is_letter().
const std::pair<uint32_t, uint32_t> kNumber[] = {
    {0x0030, 0x0039}, {0x00B2, 0x00B3}, {0x00B9, 0x00B9}, {0x00BC, 0x00BE},
    {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x0966, 0x096F}, {0x0E50, 0x0E59},
    {0x2070, 0x2070}, {0x2074, 0x2079}, {0x2080, 0x2089}, {0x2150, 0x218F},
    {0xFF10, 0xFF19},
};
bool is_number(uint32_t cp) { return in_any(cp, RANGES(kNumber)); }

// Non-letter non-ASCII: punctuation, symbols, currency, arrows, emoji, and the
// modifier/format blocks. Everything else non-ASCII is treated as a letter.
const std::pair<uint32_t, uint32_t> kNotLetter[] = {
    {0x00A1, 0x00A9}, {0x00AB, 0x00AD}, {0x00AE, 0x00B1}, {0x00B4, 0x00B4},
    {0x00B6, 0x00B8}, {0x00BB, 0x00BB}, {0x00BF, 0x00BF}, {0x00D7, 0x00D7},
    {0x00F7, 0x00F7}, {0x02B0, 0x02FF}, {0x0375, 0x0375}, {0x037E, 0x037E},
    {0x0384, 0x0385}, {0x0387, 0x0387}, {0x055A, 0x055F}, {0x0589, 0x058A},
    {0x05BE, 0x05BE}, {0x05C0, 0x05C0}, {0x05C3, 0x05C3}, {0x05C6, 0x05C6},
    {0x05F3, 0x05F4}, {0x0600, 0x0605}, {0x060C, 0x060D}, {0x061B, 0x061F},
    {0x066A, 0x066D}, {0x06D4, 0x06D4}, {0x0964, 0x0965}, {0x0970, 0x0970},
    {0x1360, 0x1368}, {0x1800, 0x180E}, {0x2010, 0x206F}, {0x20A0, 0x20CF},
    {0x2100, 0x214F}, {0x2190, 0x2BFF}, {0x2E00, 0x2E7F}, {0x3001, 0x3020},
    {0x3030, 0x303F}, {0xA670, 0xA67F}, {0xFB29, 0xFB29}, {0xFD3E, 0xFD3F},
    {0xFE10, 0xFE19}, {0xFE30, 0xFE6F}, {0xFF01, 0xFF0F}, {0xFF1A, 0xFF20},
    {0xFF3B, 0xFF40}, {0xFF5B, 0xFF65}, {0xFFE0, 0xFFEE}, {0xFFF9, 0xFFFF},
    {0x10100, 0x1013F}, {0x1D000, 0x1D7FF}, {0x1F000, 0x1FBFF},
};

// \p{L}. APPROXIMATION, and deliberately a conservative one: most assigned
// non-ASCII codepoints ARE letters, so the rule is "letter unless classed
// otherwise". The residual error is confined to unassigned codepoints and to
// punctuation/symbol blocks not listed above — where this over-merges, exactly
// as the old pattern under-merged. It is bounded and in the right direction;
// full \p{L} needs the Unicode tables, which is a dependency decision, not
// a bug fix.
bool is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    if (is_space(cp) || is_mark(cp) || is_number(cp)) return false;
    return !in_any(cp, RANGES(kNotLetter));
}
#undef RANGES

}  // namespace

// Scans one special-token-free stretch of text, appending pre-tokens to `out`.
// `marks_join_words` selects Qwen35 ([\p{L}\p{M}]+) over Qwen2 (\p{L}+).
static void qwen_scan(const std::string& s, bool marks_join_words,
                      std::vector<std::string>& out) {
    // Decode once: the scanner backtracks, and re-decoding UTF-8 per attempt
    // would make it quadratic.
    std::vector<uint32_t>  cp;     // codepoints
    std::vector<size_t>    at;     // byte offset of each codepoint
    for (size_t i = 0; i < s.size();) { at.push_back(i); cp.push_back(utf8_next(s, i)); }
    at.push_back(s.size());
    const size_t N = cp.size();

    auto word_cp = [&](size_t k) {
        return is_letter(cp[k]) || (marks_join_words && is_mark(cp[k]));
    };
    // The negated class of alternative 4: [^\s\p{L}\p{M}\p{N}] (Qwen35) or
    // [^\s\p{L}\p{N}] (Qwen2).
    auto punct_cp = [&](size_t k) {
        if (is_space(cp[k]) || is_letter(cp[k]) || is_number(cp[k])) return false;
        if (marks_join_words && is_mark(cp[k])) return false;
        return true;
    };
    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };

    size_t i = 0;
    while (i < N) {
        const size_t start = i;

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp[i] == '\'' && i + 1 < N) {
            const uint32_t a = lower(cp[i + 1]);
            size_t len = 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') len = 2;
            else if (i + 2 < N) {
                const uint32_t b = lower(cp[i + 2]);
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                    (a == 'l' && b == 'l')) len = 3;
            }
            if (len) {
                i += len;
                out.push_back(s.substr(at[start], at[i] - at[start]));
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+   (the optional prefix is one
        //    codepoint that is neither a newline, a letter, nor a number —
        //    which is what lets ".example" and " units" be single pre-tokens)
        {
            size_t j = i;
            if (cp[j] != '\r' && cp[j] != '\n' && !is_letter(cp[j]) && !is_number(cp[j]))
                ++j;
            if (j < N && word_cp(j)) {
                while (j < N && word_cp(j)) ++j;
                i = j;
                out.push_back(s.substr(at[start], at[i] - at[start]));
                continue;
            }
        }

        // 3. \p{N}  — ONE number codepoint, never a run
        if (is_number(cp[i])) {
            ++i;
            out.push_back(s.substr(at[start], at[i] - at[start]));
            continue;
        }

        // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        {
            size_t j = i;
            if (cp[j] == ' ') ++j;
            if (j < N && punct_cp(j)) {
                while (j < N && punct_cp(j)) ++j;
                while (j < N && (cp[j] == '\r' || cp[j] == '\n')) ++j;
                i = j;
                out.push_back(s.substr(at[start], at[i] - at[start]));
                continue;
            }
        }

        // 5. \s*[\r\n]+  — greedy \s* with backtracking, i.e. the whitespace
        //    run truncated at its LAST newline. This is the alternative the
        //    GPT-2 pattern lacks, and the reason "\n\n" used to split.
        {
            size_t j = i, last_nl = std::string::npos;
            while (j < N && is_space(cp[j])) {
                if (cp[j] == '\r' || cp[j] == '\n') last_nl = j;
                ++j;
            }
            if (last_nl != std::string::npos) {
                i = last_nl + 1;
                out.push_back(s.substr(at[start], at[i] - at[start]));
                continue;
            }
        }

        // 6. \s+(?!\S)  — a whitespace run only up to the last character when
        //    more text follows (the GPT-2 "keep one space for the next word"
        //    rule); the whole run at end of input.
        if (is_space(cp[i])) {
            size_t j = i;
            while (j < N && is_space(cp[j])) ++j;
            const size_t end = (j == N) ? j : j - 1;
            if (end > i) {
                i = end;
                out.push_back(s.substr(at[start], at[i] - at[start]));
                continue;
            }
            // 7. \s+
            i = j;
            out.push_back(s.substr(at[start], at[i] - at[start]));
            continue;
        }

        // No alternative matched. Unreachable for well-formed input; emit one
        // codepoint rather than spin, so a pathological byte cannot hang the
        // server.
        ++i;
        out.push_back(s.substr(at[start], at[i] - at[start]));
    }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    switch (pre_kind_) {
        case PreTokenizerKind::Qwen2:
        case PreTokenizerKind::Qwen35:
            return pretokenize_qwen(text);
        case PreTokenizerKind::Gpt2:
        default:
            return pretokenize_gpt2(text);
    }
}

std::vector<std::string> Tokenizer::pretokenize_gpt2(const std::string& text) const {
    std::vector<std::string> tokens;
    std::sregex_iterator iter(text.begin(), text.end(), pretokenization_regex_);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
        std::string match = iter->str();
        if (!match.empty()) {
            tokens.push_back(match);
        }
    }
    return tokens;
}

std::vector<std::string> Tokenizer::pretokenize_qwen(const std::string& text) const {
    const bool marks_join_words = (pre_kind_ == PreTokenizerKind::Qwen35);
    return split_on_special_tokens(
        text, [&](const std::string& chunk, std::vector<std::string>& out) {
            qwen_scan(chunk, marks_join_words, out);
        });
}

// Special tokens are matched before any pattern alternative and emitted whole —
// the same precedence the GPT-2 path gets from putting them first in its regex
// alternation. LONGEST match wins at a position, so a token that is a prefix of
// another (e.g. "<|im_start|>" vs a hypothetical "<|im_start|>x") cannot shadow
// it; the regex path took them in unordered-map order, which was arbitrary.
std::vector<std::string> Tokenizer::split_on_special_tokens(
    const std::string& text,
    const std::function<void(const std::string&, std::vector<std::string>&)>& scan) const {
    std::vector<std::string> out;
    size_t plain = 0;   // start of the current non-special stretch
    size_t i = 0;
    while (i < text.size()) {
        size_t best = 0;
        for (const auto& [tok, id] : special_tokens_) {
            (void)id;
            if (tok.size() > best && text.compare(i, tok.size(), tok) == 0) best = tok.size();
        }
        if (best == 0) { ++i; continue; }
        if (i > plain) scan(text.substr(plain, i - plain), out);
        out.push_back(text.substr(i, best));
        i += best;
        plain = i;
    }
    if (plain < text.size()) scan(text.substr(plain), out);
    return out;
}

// Gets all pairs of adjacent tokens.
std::vector<std::pair<int32_t, int32_t>> get_pairs(const std::vector<int32_t>& tokens) {
    std::vector<std::pair<int32_t, int32_t>> pairs;
    if (tokens.size() < 2) return pairs;
    pairs.reserve(tokens.size() - 1);
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        pairs.emplace_back(tokens[i], tokens[i+1]);
    }
    return pairs;
}

std::vector<int32_t> Tokenizer::apply_bpe(const std::vector<int32_t>& byte_tokens) const {
    if (byte_tokens.size() <= 1) return byte_tokens;

    const size_t n = byte_tokens.size();

    // Doubly-linked list for O(1) merge operations
    struct Node {
        int32_t token;
        int prev = -1;
        int next = -1;
        bool deleted = false;
    };
    
    std::vector<Node> nodes(n);
    for (size_t i = 0; i < n; ++i) {
        nodes[i].token = byte_tokens[i];
        nodes[i].prev = (i > 0) ? static_cast<int>(i - 1) : -1;
        nodes[i].next = (i < n - 1) ? static_cast<int>(i + 1) : -1;
    }
    
    // Min-heap: (rank, position) — lower rank = higher priority
    using Entry = std::pair<int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
    
    // Helper: get merge info for pair starting at position i
    auto get_merge = [&](int i) -> std::pair<int, int32_t> {
        if (i < 0 || nodes[i].next < 0) return {-1, -1};
        auto it = merges_.find({nodes[i].token, nodes[nodes[i].next].token});
        if (it == merges_.end()) return {-1, -1};
        return it->second;  // {rank, merged_token}
    };
    
    // Initialize queue with all mergeable pairs
    for (size_t i = 0; i + 1 < n; ++i) {
        auto [rank, merged] = get_merge(i);
        if (rank >= 0) {
            pq.push({rank, static_cast<int>(i)});
        }
    }
    
    // Process merges in rank order
    while (!pq.empty()) {
        auto [rank, pos] = pq.top();
        pq.pop();
        
        // Skip stale entries (node deleted or pair changed)
        if (nodes[pos].deleted) continue;
        int next_pos = nodes[pos].next;
        if (next_pos < 0 || nodes[next_pos].deleted) continue;
        
        // Verify merge is still valid (tokens may have changed)
        auto [current_rank, merged_token] = get_merge(pos);
        if (current_rank != rank) continue;
        
        // Perform merge: pos absorbs next_pos
        nodes[pos].token = merged_token;
        nodes[next_pos].deleted = true;
        
        // Update linked list
        nodes[pos].next = nodes[next_pos].next;
        if (nodes[next_pos].next >= 0) {
            nodes[nodes[next_pos].next].prev = pos;
        }
        
        // Enqueue new pairs formed by merge
        if (nodes[pos].prev >= 0) {
            auto [r, m] = get_merge(nodes[pos].prev);
            if (r >= 0) pq.push({r, nodes[pos].prev});
        }
        {
            auto [r, m] = get_merge(pos);
            if (r >= 0) pq.push({r, pos});
        }
    }
    
    // Collect result by traversing from head
    std::vector<int32_t> result;
    result.reserve(n);
    
    // Find first non-deleted node
    int head = 0;
    while (head < static_cast<int>(n) && nodes[head].deleted) ++head;
    
    // Traverse linked list
    for (int pos = head; pos >= 0; pos = nodes[pos].next) {
        result.push_back(nodes[pos].token);
    }
    
    return result;
}

std::vector<int32_t> Tokenizer::encode_with_special_tokens(const std::string& text) const {
    // Check if entire text is a special token
    auto special_it = special_tokens_.find(text);
    if (special_it != special_tokens_.end()) {
        return {special_it->second};
    }

    // Check cache (thread-safe read)
    {
        std::lock_guard<std::mutex> lock(bpe_cache_mutex_);
        auto cache_it = bpe_cache_.find(text);
        if (cache_it != bpe_cache_.end()) {
            return cache_it->second;
        }
    }
    
    // Cache miss - compute BPE
    if (text.empty()) return {};
    
    // Convert text to byte-level tokens
    std::vector<int32_t> byte_tokens = encode_single_token(text);
    
    // Apply BPE merges
    std::vector<int32_t> result = apply_bpe(byte_tokens);
    
    // Store in cache (thread-safe write)
    {
        std::lock_guard<std::mutex> lock(bpe_cache_mutex_);
        bpe_cache_[text] = result;
    }
    
    return result;
}

std::vector<int32_t> Tokenizer::encode_single_token(const std::string& text) const {
    std::vector<int32_t> byte_tokens;
    byte_tokens.reserve(text.length());
    for (char c : text) {
        const std::string& byte_str = byte_encoder_[static_cast<unsigned char>(c)];
        auto token_it = token_to_id_.find(byte_str);
        if (token_it != token_to_id_.end()) {
            byte_tokens.push_back(token_it->second);
        } else {
            byte_tokens.push_back(unk_token_id_);
        }
    }
    return byte_tokens;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};

    if (is_llama_tokenizer_) {
        return encode_llama(text);
    }

    // Step 1: Pre-tokenization (handles special tokens and regex splitting)
    std::vector<std::string> pretokens = pretokenize(text);

    // Step 2: Apply BPE to each pre-token
    std::vector<int32_t> result;
    for (const std::string& pretoken : pretokens) {
        std::vector<int32_t> encoded = encode_with_special_tokens(pretoken);
        result.insert(result.end(), encoded.begin(), encoded.end());
    }

    return result;
}

// ── llama / SentencePiece-derived tokenizer (Gemma) ───────────────────────────
//
// Byte-fallback table: for each byte 0..255, the vocab is expected to contain
// a token of the form "<0xNN>" (TokenType::BYTE). We index those here so the
// Viterbi encoder has a guaranteed fall-through for any input byte.
//
// Fail-loud: if a byte token is missing, encoding any text containing that
// byte would silently produce UNK and lose information. We accept partial
// tables only when the GGUF lacks BYTE-typed tokens (e.g. test fixtures);
// the encode path then falls back to the unknown_token_id for unmappable
// bytes, which the unit tests explicitly cover.
void Tokenizer::initialize_llama_byte_fallback() {
    byte_fallback_ids_.assign(256, -1);
    for (size_t i = 0; i < metadata_->token_types.size(); ++i) {
        if (metadata_->token_types[i] != TokenType::BYTE) continue;
        const std::string& s = metadata_->id_to_token[i];
        // Expected form: "<0xNN>".
        if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                return -1;
            };
            int hi = hex(s[3]);
            int lo = hex(s[4]);
            if (hi >= 0 && lo >= 0) {
                byte_fallback_ids_[(hi << 4) | lo] = static_cast<int32_t>(i);
            }
        }
    }
}

// Best-effort score lookup for fallthrough scoring.
// Tokens absent from `scores` are treated as having score 0.
static inline float score_or_zero(const std::vector<float>& scores, int32_t id) {
    if (id < 0 || static_cast<size_t>(id) >= scores.size()) return 0.0f;
    return scores[id];
}

// Length in bytes of a UTF-8 character starting at byte b. Returns 0 for
// invalid leading bytes (caller falls back to single-byte handling).
static inline int utf8_char_len(unsigned char b) {
    if ((b & 0x80) == 0x00) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}

// SentencePiece-style BPE merge by score (used by Gemma / Llama family).
// This matches what HF transformers does for `tokenizer.ggml.model = "llama"`:
//
//   1. Split the (already ▁-normalized) input into UTF-8 *characters*
//      and look each up in the vocab. Unknown chars decompose into
//      byte-fallback tokens (`<0xNN>`).
//   2. Repeatedly pick the highest-score adjacent (left, right) pair whose
//      *concatenation* is itself a vocab token, and merge it. Ties are
//      resolved by leftmost position (matching SentencePiece behavior).
//   3. Stop when no adjacent pair is a vocab token.
//
// This is *not* Viterbi over substring scores — that algorithm prefers
// shorter tokens with cheap byte-fallback whenever the longer tokens have
// large negative scores, which is exactly the failure mode that produced
// "user" → ['us','er'] and "Who" → ['▁W','ho'] before this rewrite.
std::vector<int32_t> Tokenizer::encode_llama(const std::string& text) const {
    // 1. Special-token splitting: walk the input, emitting any special-token
    //    match verbatim and Viterbi-encoding the gaps in between. Special
    //    tokens like <start_of_turn> must appear as a single id, not be
    //    decomposed into pieces.
    auto encode_segment = [&](const std::string& seg) -> std::vector<int32_t> {
        if (seg.empty()) return {};

        // 2. Normalize: replace each space with U+2581 (▁) per SentencePiece.
        static const std::string spm_under = "\xE2\x96\x81";
        std::string norm;
        norm.reserve(seg.size());
        for (char c : seg) {
            if (c == ' ') norm += spm_under;
            else norm += c;
        }

        // 3. Initial split: walk the normalized string as UTF-8 characters
        //    and emit one token per char. If the char isn't in the vocab,
        //    decompose into byte-fallback tokens (`<0xNN>`) so every byte
        //    has a starting token id.
        struct Node {
            int32_t id;       // -1 if deleted
            std::string text; // the raw (normalized) bytes this token covers
        };
        std::vector<Node> nodes;
        nodes.reserve(norm.size());

        for (size_t i = 0; i < norm.size();) {
            const unsigned char b = static_cast<unsigned char>(norm[i]);
            int len = utf8_char_len(b);
            if (len == 0 || i + len > norm.size()) {
                // Invalid leading byte → single-byte fallback path.
                int32_t bid = byte_fallback_ids_[b];
                nodes.push_back({bid >= 0 ? bid : unk_token_id_,
                                 std::string(1, norm[i])});
                i += 1;
                continue;
            }
            std::string ch = norm.substr(i, len);
            auto it = token_to_id_.find(ch);
            if (it != token_to_id_.end()) {
                nodes.push_back({it->second, ch});
            } else {
                // Char absent → emit each underlying byte as a byte-fallback.
                for (int k = 0; k < len; ++k) {
                    unsigned char bb = static_cast<unsigned char>(norm[i + k]);
                    int32_t bid = byte_fallback_ids_[bb];
                    nodes.push_back({bid >= 0 ? bid : unk_token_id_,
                                     std::string(1, norm[i + k])});
                }
            }
            i += len;
        }

        // 4. Iterative BPE merge by score. We use a doubly-linked list over
        //    `nodes` and a max-heap on score keyed by (score, position).
        //    Stale entries are filtered at pop time.
        const int N = static_cast<int>(nodes.size());
        if (N == 0) return {};

        std::vector<int> prev(N, -1), next(N, -1);
        for (int i = 0; i < N; ++i) {
            prev[i] = (i > 0)     ? i - 1 : -1;
            next[i] = (i + 1 < N) ? i + 1 : -1;
        }

        struct PQEntry {
            float       score;     // higher = better (max-heap)
            int         pos;       // left node index
            int         right_pos; // right node index — used to validate staleness
            int32_t     merged_id;
            std::string merged_text;
            bool operator<(const PQEntry& o) const { return score < o.score; }
        };
        std::priority_queue<PQEntry> pq;

        auto try_enqueue = [&](int left) {
            if (left < 0) return;
            int right = next[left];
            if (right < 0) return;
            std::string s = nodes[left].text + nodes[right].text;
            auto it = token_to_id_.find(s);
            if (it == token_to_id_.end()) return;
            // Skip special tokens — they're emitted by the outer loop only.
            if (special_token_ids_.count(it->second)) return;
            pq.push({score_or_zero(scores_, it->second),
                     left, right, it->second, std::move(s)});
        };

        for (int i = 0; i < N; ++i) try_enqueue(i);

        while (!pq.empty()) {
            PQEntry e = pq.top();
            pq.pop();
            // Staleness checks: either node deleted, or the right neighbor
            // changed since enqueue (which would mean the merged_text no
            // longer matches the live concatenation).
            if (nodes[e.pos].id < 0) continue;
            if (e.right_pos < 0 || next[e.pos] != e.right_pos) continue;
            if (nodes[e.right_pos].id < 0) continue;
            if (nodes[e.pos].text + nodes[e.right_pos].text != e.merged_text) continue;

            // Merge: absorb right into left.
            nodes[e.pos].id   = e.merged_id;
            nodes[e.pos].text = e.merged_text;
            int absorbed = e.right_pos;
            int new_next = next[absorbed];
            next[e.pos] = new_next;
            if (new_next >= 0) prev[new_next] = e.pos;
            nodes[absorbed].id = -1;

            // Enqueue new neighboring pairs.
            try_enqueue(prev[e.pos]);
            try_enqueue(e.pos);
        }

        // 5. Collect the surviving tokens left-to-right.
        std::vector<int32_t> out;
        out.reserve(N);
        for (int i = 0; i < N; i = next[i] < 0 ? -1 : next[i]) {
            if (i < 0) break;
            if (nodes[i].id < 0) continue;
            out.push_back(nodes[i].id);
            if (next[i] < 0) break;
        }
        return out;
    };

    // Build a list of special tokens sorted by length (longest first) so that
    // overlapping prefixes resolve to the longer match.
    std::vector<std::pair<std::string, int32_t>> specials(
        special_tokens_.begin(), special_tokens_.end());
    std::sort(specials.begin(), specials.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });

    std::vector<int32_t> result;
    size_t pos = 0;
    for (; pos < text.size(); ++pos) {
        for (const auto& [tok, id] : specials) {
            if (tok.empty()) continue;
            if (text.compare(pos, tok.size(), tok) == 0) {
                if (pos > 0) {
                    auto seg = encode_segment(text.substr(0, pos));
                    result.insert(result.end(), seg.begin(), seg.end());
                }
                result.push_back(id);
                // Recurse on the remainder so any further specials there are
                // emitted atomically as well.
                std::string rest = text.substr(pos + tok.size());
                std::vector<int32_t> tail = encode(rest);
                result.insert(result.end(), tail.begin(), tail.end());
                return result;
            }
        }
    }
    // No special token in `text` — encode the whole input as a single segment.
    auto seg = encode_segment(text);
    result.insert(result.end(), seg.begin(), seg.end());
    return result;
}

std::string Tokenizer::decode_llama(int32_t token_id) const {
    if (token_id < 0 ||
        token_id >= static_cast<int32_t>(metadata_->id_to_token.size())) {
        return "<ID_OOB>";
    }

    if (is_special_token(token_id)) {
        return metadata_->id_to_token[token_id];
    }

    const TokenType tt = (static_cast<size_t>(token_id) < metadata_->token_types.size())
                             ? metadata_->token_types[token_id]
                             : TokenType::NORMAL;

    if (tt == TokenType::BYTE) {
        const std::string& s = metadata_->id_to_token[token_id];
        if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                return -1;
            };
            int hi = hex(s[3]), lo = hex(s[4]);
            if (hi >= 0 && lo >= 0) {
                return std::string(1, static_cast<char>((hi << 4) | lo));
            }
        }
        return s;
    }

    // Replace U+2581 (▁, "\xE2\x96\x81") with space.
    std::string out;
    const std::string& tok = metadata_->id_to_token[token_id];
    out.reserve(tok.size());
    for (size_t i = 0; i < tok.size();) {
        if (i + 2 < tok.size() &&
            static_cast<unsigned char>(tok[i])     == 0xE2 &&
            static_cast<unsigned char>(tok[i + 1]) == 0x96 &&
            static_cast<unsigned char>(tok[i + 2]) == 0x81) {
            out += ' ';
            i += 3;
        } else {
            out += tok[i++];
        }
    }
    return out;
}

std::string Tokenizer::_decode_original_token(int32_t original_id) const {
    if (original_id < 0 || original_id >= static_cast<int32_t>(metadata_->id_to_token.size())) {
        return "<ID_OOB>";
    }

    const std::string& token = metadata_->id_to_token[original_id];

    // Special tokens are returned as-is
    if (is_special_token(original_id)) {
        return token;
    }

    // Regular tokens need byte decoding.
    // GPT-2 byte encoding maps byte values to Unicode characters, some of which
    // are multi-byte in UTF-8 (e.g., space 0x20 → Ġ U+0120 → \xC4\xA0).
    // We must try multi-byte lookups before single-byte.
    std::string result;
    size_t i = 0;
    while (i < token.size()) {
        // Try 2-byte UTF-8 sequence first (covers all GPT-2 byte mappings > 127)
        if (i + 1 < token.size()) {
            std::string two_byte = token.substr(i, 2);
            auto it = byte_decoder_.find(two_byte);
            if (it != byte_decoder_.end()) {
                result += static_cast<char>(it->second);
                i += 2;
                continue;
            }
        }
        // Fall back to single-byte lookup
        std::string one_byte(1, token[i]);
        auto it = byte_decoder_.find(one_byte);
        if (it != byte_decoder_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += token[i];
        }
        i++;
    }

    return result;
}

std::string Tokenizer::decode(int32_t token_id) const {
    if (is_llama_tokenizer_) {
        return decode_llama(token_id);
    }
    return _decode_original_token(token_id);
}

bool Tokenizer::is_special_token(int32_t token_id) const {
    return special_token_ids_.count(token_id) > 0;
}

std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string result;
    for (int32_t token_id : token_ids) {
        result += decode(token_id);
    }
    return result;
}

int32_t Tokenizer::get_special_token_id(const std::string& token_str) const {
    auto it = special_tokens_.find(token_str);
    if (it != special_tokens_.end()) {
        return it->second;
    }
    return unk_token_id_;
}

int32_t Tokenizer::get_eos_token_id() const {
    return metadata_->eos_token_id;
}

const std::vector<std::string>& Tokenizer::get_vocabulary() const {
    return metadata_->id_to_token;
}