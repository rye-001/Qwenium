#pragma once

#include "grammar_vocab.h"
#include "token-trie.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>
#include <string>

namespace qwenium {

class Sampler {
public:
    virtual ~Sampler() = default;

    // Sample a token ID from full-vocab logits.
    // last_tokens: recent token history for repetition penalty.
    // token_strs:  vocab[i] = decoded string of token i (for grammar constraints).
    virtual int sample(std::vector<float>& logits,
                       const std::vector<int32_t>& last_tokens,
                       const std::vector<std::string>& token_strs) = 0;

    // Sample from sparse logits produced by the sparse LM head.
    // sparse_logits[j] is the logit for token valid_ids[j].
    // Grammar and pruning are already encoded in valid_ids (caller ran
    // peek_valid_set); this method only applies repetition penalty and
    // the subclass's own sampling strategy.
    // Returns the selected token ID (not the index into sparse_logits).
    virtual int32_t sample_sparse(std::vector<float>& sparse_logits,
                                  const std::vector<int32_t>& valid_ids,
                                  const std::vector<int32_t>& last_tokens) = 0;

    // Attach a grammar constraint (vocab-scan, Option B).
    void set_grammar(GrammarVocab* grammar) { grammar_ = grammar; }

    // Attach a vocab pruning set (optional).
    void set_pruned_vocab(const std::unordered_set<int32_t>* pruned_vocab) {
        pruned_vocab_ = pruned_vocab;
    }

    // Register a stop token ID. Call once per stop token at init;
    // duplicates are silently ignored.
    void add_eos_token_id(int32_t id) {
        if (id >= 0 && std::find(eos_token_ids_.begin(), eos_token_ids_.end(), id) == eos_token_ids_.end())
            eos_token_ids_.push_back(id);
    }

    // Build and attach a TokenTrie from the vocabulary for accelerated
    // grammar-constrained decoding. Called once; the trie is immutable.
    // Also caches the vocab so peek_valid_set needs no per-step parameter.
    void build_token_trie(const std::vector<std::string>& vocab);

    // Return the set of valid token IDs under the current grammar + pruning
    // constraints, without consuming any grammar state or touching logits.
    // Returns an empty vector when no constraints are active — caller treats
    // that as "use dense path."  Call this BEFORE the forward pass so the
    // result can drive sparse-head selection.
    // The result is cached: the next call to apply_grammar_constraints will
    // consume it instead of recomputing, eliminating the double-call on the
    // dense path.
    std::vector<int32_t> peek_valid_set();

    // Registered stop token IDs (for forced-run termination in decode_step).
    const std::vector<int32_t>& eos_token_ids() const { return eos_token_ids_; }

    // Advance grammar state after a token is committed.
    // No-op when no grammar is attached.
    void accept_token(int32_t token_id) {
        if (grammar_) grammar_->accept_token(token_id, vocab_);
    }

protected:
    void apply_vocab_pruning(std::vector<float>& logits);
    void apply_grammar_constraints(std::vector<float>& logits, const std::vector<int32_t>& last_tokens,
                                   const std::vector<std::string>& token_strs);
    static void apply_repetition_penalty_sparse(std::vector<float>& sparse_logits,
                                                const std::vector<int32_t>& valid_ids,
                                                const std::vector<int32_t>& last_tokens,
                                                float penalty, int lookback);

    GrammarVocab* grammar_     = nullptr;
    const std::unordered_set<int32_t>* pruned_vocab_ = nullptr;
    std::vector<int32_t> eos_token_ids_;
    std::unique_ptr<TokenTrie> token_trie_;
    std::vector<std::string> vocab_; // cached at build_token_trie time

    // Cache populated by peek_valid_set(); consumed (and cleared) by
    // apply_grammar_constraints() to avoid a redundant get_valid_tokens call
    // on the dense decode path.
    //
    // cached_valid_ids_version_ is stamped from GrammarVocab::state_version()
    // at write time. apply_grammar_constraints asserts the grammar's current
    // version matches before consuming — a mismatch means accept_token was
    // called between peek and apply and the cache is stale.
    std::vector<int32_t> cached_valid_ids_;
    bool                 cached_valid_ids_valid_   = false;
    uint64_t             cached_valid_ids_version_ = 0;
};

class GreedySampler : public Sampler {
public:
    explicit GreedySampler(float repetition_penalty = 1.2f,
                           int   repetition_lookback = 32)
        : repetition_penalty_(repetition_penalty),
          repetition_lookback_(repetition_lookback) {}

    int sample(std::vector<float>& logits,
               const std::vector<int32_t>& last_tokens,
               const std::vector<std::string>& token_strs = {}) override;

    int32_t sample_sparse(std::vector<float>& sparse_logits,
                          const std::vector<int32_t>& valid_ids,
                          const std::vector<int32_t>& last_tokens) override;

private:
    void apply_repetition_penalty(std::vector<float>& logits,
                                  const std::vector<int32_t>& last_tokens);
    float repetition_penalty_;
    int   repetition_lookback_;
};

class TemperatureSampler : public Sampler {
public:
    explicit TemperatureSampler(float temperature        = 0.7f,
                                float repetition_penalty = 1.1f,
                                int   repetition_lookback = 64,
                                int   top_k              = 40,
                                float top_p              = 0.95f);

    int sample(std::vector<float>& logits,
               const std::vector<int32_t>& last_tokens,
               const std::vector<std::string>& token_strs) override;

    int32_t sample_sparse(std::vector<float>& sparse_logits,
                          const std::vector<int32_t>& valid_ids,
                          const std::vector<int32_t>& last_tokens) override;

private:
    void apply_repetition_penalty(std::vector<float>& logits,
                                  const std::vector<int32_t>& last_tokens);

    float temperature_;
    float repetition_penalty_;
    int   repetition_lookback_;
    int   top_k_;
    float top_p_;
    std::mt19937 gen_;
};

} // namespace qwenium