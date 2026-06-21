#pragma once

#include <memory>
#include <string>
#include <vector>

namespace qinf::session {
class SnapshotWriter;
class SnapshotReader;
}  // namespace qinf::session

namespace qwenium {

class TokenTrie;  // forward declaration

// Option B: vocab-scan grammar — no pretokenized literals map.
// Every LITERAL state resolves valid tokens by scanning vocab at runtime.
// GrammarState tracks char_idx (chars consumed of current literal)
// instead of token_idx (index into pre-baked token sequence).
//
// When a TokenTrie is attached via set_token_trie(), LITERAL states use
// trie-based candidate narrowing instead of O(vocab_size) brute-force scan.
class GrammarVocab {
public:
    GrammarVocab();
    ~GrammarVocab();

    // Parse GBNF — no map, no tokenizer, no offline step.
    static std::unique_ptr<GrammarVocab> parse_impl(const std::string& gbnf_str);

    // Attach an externally-owned trie built from the vocabulary.
    // The trie must outlive this GrammarVocab instance.
    // When set, get_valid_tokens() uses trie lookups for LITERAL states.
    void set_token_trie(const TokenTrie* trie);

    // Vocab-scan: find all token IDs that are valid at current grammar position.
    // vocab[i] is the decoded string of token i.
    std::vector<int32_t> get_valid_tokens(const std::vector<std::string>& vocab) const;

    // Advance grammar state by accepting token_id.
    // vocab[token_id] tells us which string was consumed and how many chars.
    void accept_token(int32_t token_id, const std::vector<std::string>& vocab);

    void reset();
    bool is_accepting_state() const;
    void dump_expected() const;

    // Monotonic counter bumped whenever the active stacks change
    // (accept_token, reset). Sampler::peek_valid_set stamps this on its
    // cache; Sampler::apply_grammar_constraints asserts the stamp matches
    // before consuming the cache. A divergence means accept_token was
    // called between peek and apply — the cache would be stale.
    uint64_t state_version() const noexcept { return state_version_; }

    // --- L1 session snapshot: the grammar cursor ---
    // The ONLY persistent state after each accept_token is impl_->stacks — a
    // list of {pos, char_idx, continuations}, where pos/continuations are raw
    // pointers into impl_->rules. The consume_token/resolve_* recursion is
    // transient and persists nothing. So the cursor is fully externalizable:
    // each pointer serializes as a (rule_idx, elem_offset) coordinate, and the
    // rules themselves are rebuilt deterministically by re-parsing the same
    // GBNF. read_cursor REQUIRES this object was already constructed via
    // parse_impl(<same GBNF>) — it restores only the cursor, not the grammar.
    // Co-located section: src/sampling/sampling_snapshot.{h,cpp}.
    void write_cursor(qinf::session::SnapshotWriter&) const;
    void read_cursor(qinf::session::SnapshotReader&);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    const TokenTrie* trie_ = nullptr;
    uint64_t state_version_ = 0;
};

} // namespace qwenium