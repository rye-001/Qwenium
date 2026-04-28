// token-trie.cpp — TokenTrie implementation for grammar-constrained decoding.
// Built once from vocabulary at startup, immutable and read-only thereafter.

#include "token-trie.h"

namespace qwenium {

void TokenTrie::build(const std::vector<std::string>& vocab) {
    root_ = std::make_unique<TrieNode>();
    token_count_ = 0;

    // Clear precomputed first-char index
    for (int c = 0; c < 256; ++c)
        first_char_tokens_[c].clear();

    for (size_t i = 0; i < vocab.size(); ++i) {
        const std::string& token = vocab[i];
        if (token.empty()) continue;  // skip empty/special tokens

        // Build trie
        TrieNode* node = root_.get();
        for (char c : token) {
            auto idx = static_cast<unsigned char>(c);
            if (!node->children[idx]) {
                node->children[idx] = std::make_unique<TrieNode>();
            }
            node = node->children[idx].get();
        }
        node->token_ids.push_back(static_cast<int32_t>(i));

        // Build first-char index — contiguous arrays for cache-friendly access
        auto first = static_cast<unsigned char>(token[0]);
        first_char_tokens_[first].push_back(static_cast<int32_t>(i));

        token_count_++;
    }
}

void TokenTrie::collect_by_prefix(const std::string& prefix, size_t offset,
                                   std::vector<int32_t>& out) const {
    if (!root_) return;

    const TrieNode* node = root_.get();
    const size_t len = prefix.size();

    // Walk the trie along the prefix characters.
    // At each step, collect tokens that terminate here (they are prefixes of
    // the remaining literal — case (a): partial consumption).
    for (size_t i = offset; i < len; ++i) {
        auto idx = static_cast<unsigned char>(prefix[i]);
        const auto& child = node->children[idx];
        if (!child) return;  // no tokens match this prefix path at all
        node = child.get();

        // Tokens terminating at this node are prefixes of prefix[offset:]
        // with length (i - offset + 1). They handle partial consumption.
        out.insert(out.end(), node->token_ids.begin(), node->token_ids.end());
    }

    // Now `node` is at the end of the prefix path.
    // Collect all tokens in the subtree — these are tokens that start with
    // the full prefix and extend further (case (c): overshoot).
    // Note: tokens exactly at `node` were already added in the loop above
    // (the last iteration), so we only recurse into children.
    for (int c = 0; c < 256; ++c) {
        if (node->children[c]) {
            collect_subtree(node->children[c].get(), out);
        }
    }
}

void TokenTrie::collect_by_char_class(const std::set<char>& chars, bool negated,
                                       std::vector<int32_t>& out) const {
    if (!root_) return;

    for (int c = 0; c < 256; ++c) {
        if (!root_->children[c]) continue;
        bool in_set = chars.count(static_cast<char>(c)) > 0;
        bool allowed = negated ? !in_set : in_set;
        if (allowed) {
            collect_subtree(root_->children[c].get(), out);
        }
    }
}

void TokenTrie::collect_by_first_char(char c, std::vector<int32_t>& out) const {
    if (!root_) return;
    auto idx = static_cast<unsigned char>(c);
    if (!root_->children[idx]) return;
    collect_subtree(root_->children[idx].get(), out);
}

void TokenTrie::mark_char_class(const std::set<char>& chars, bool negated,
                                 std::vector<bool>& valid_mask) const {
    // Build a 256-byte lookup table from the char class.
    // This replaces std::set<char>::count() per token with a single array lookup.
    bool allowed[256] = {};
    if (negated) {
        // Start with all allowed, then exclude chars in the set
        for (int c = 0; c < 256; ++c) allowed[c] = true;
        for (char c : chars) allowed[static_cast<unsigned char>(c)] = false;
    } else {
        for (char c : chars) allowed[static_cast<unsigned char>(c)] = true;
    }

    // Iterate only the precomputed first-char buckets for allowed chars.
    // Each bucket is a contiguous vector<int32_t> — cache-friendly.
    // For negated [^"]: ~250 buckets × ~600 tokens/bucket ≈ 150K tokens,
    // but accessed sequentially instead of scattered string dereferences.
    for (int c = 0; c < 256; ++c) {
        if (!allowed[c]) continue;
        for (int32_t tok_id : first_char_tokens_[c]) {
            valid_mask[tok_id] = true;
        }
    }
}

void TokenTrie::collect_subtree(const TrieNode* node, std::vector<int32_t>& out) {
    if (!node) return;
    out.insert(out.end(), node->token_ids.begin(), node->token_ids.end());
    for (int c = 0; c < 256; ++c) {
        if (node->children[c]) {
            collect_subtree(node->children[c].get(), out);
        }
    }
}

} // namespace qwenium
