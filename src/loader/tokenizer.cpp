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

TokenizerConfig tokenizer_config_from_gguf(const ModelMetadata& meta)
{
    TokenizerConfig cfg;
    cfg.add_bos_token = meta.add_bos_token;
    // tokenizer.ggml.model = "llama" means SentencePiece BPE with ▁-normalization
    // and byte-fallback.  This is a standard GGUF field value, not a family name.
    if (meta.tokenizer_type == "llama") {
        cfg.normalizer    = NormalizerKind::SpaceToUnderscore;
        cfg.byte_fallback = true;
        // Some GGUF exports mark chat-control tokens as NORMAL rather than
        // USER_DEFINED; force-promote them so the encoder handles them atomically.
        cfg.extra_chat_specials = {"<start_of_turn>", "<end_of_turn>"};
    }
    return cfg;
}

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

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
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