#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace qwen3 {

struct TrieNode {
    // Children indexed by byte value (0-255). nullptr = no child.
    std::unique_ptr<TrieNode> children[256] = {};
    // Token IDs whose decoded string exactly terminates at this node.
    std::vector<int32_t> token_ids;
};

class TokenTrie {
public:
    TokenTrie() = default;
    ~TokenTrie() = default;

    // Non-copyable (owns the trie tree), movable.
    TokenTrie(const TokenTrie&) = delete;
    TokenTrie& operator=(const TokenTrie&) = delete;
    TokenTrie(TokenTrie&&) = default;
    TokenTrie& operator=(TokenTrie&&) = default;

    // Build trie from vocabulary. vocab[i] is the decoded string of token i.
    // Empty strings and special tokens are skipped.
    // Called once at sampler init; the trie is immutable thereafter.
    void build(const std::vector<std::string>& vocab);

    // Collect all token IDs whose string starts with `prefix[offset:]`.
    // Also collects tokens on the path (tokens that are prefixes of the query).
    // This handles all three LITERAL cases:
    //   (a) token is a prefix of remaining literal → partial consumption
    //   (b) token exactly matches remaining literal → full consumption
    //   (c) token extends past remaining literal → overshoot
    void collect_by_prefix(const std::string& prefix, size_t offset,
                           std::vector<int32_t>& out) const;

    // Collect all token IDs whose first char is in the char set.
    // If negated=true, collect tokens whose first char is NOT in the set.
    void collect_by_char_class(const std::set<char>& chars, bool negated,
                               std::vector<int32_t>& out) const;

    // Fast path: collect all token IDs whose string starts with char c.
    void collect_by_first_char(char c, std::vector<int32_t>& out) const;

    // Mark valid tokens in a bitset directly for a char class.
    // Much faster than collect_by_char_class + insertion loop because it
    // uses precomputed first-char buckets (contiguous, cache-friendly).
    void mark_char_class(const std::set<char>& chars, bool negated,
                         std::vector<bool>& valid_mask) const;

    // Total number of non-empty tokens indexed by the trie.
    size_t token_count() const { return token_count_; }

private:
    std::unique_ptr<TrieNode> root_;
    size_t token_count_ = 0;

    // Precomputed index: first_char_tokens_[c] = all token IDs whose
    // decoded string starts with byte value c. Built once, contiguous
    // arrays for cache-friendly iteration.
    std::vector<int32_t> first_char_tokens_[256];

    // Recursively collect all token IDs in the subtree rooted at node.
    static void collect_subtree(const TrieNode* node, std::vector<int32_t>& out);
};

} // namespace qwen3
