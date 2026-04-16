#pragma once

#include "grammar_vocab.h"
#include "token-trie.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>
#include <string>

namespace qwen3 {

class Sampler {
public:
    virtual ~Sampler() = default;

    // Sample a token ID from logits.
    // last_tokens: recent token history for repetition penalty.
    // token_strs:  vocab[i] = decoded string of token i (for grammar constraints).
    virtual int sample(std::vector<float>& logits,
                       const std::vector<int32_t>& last_tokens,
                       const std::vector<std::string>& token_strs) = 0;

    // Attach a grammar constraint (vocab-scan, Option B).
    void set_grammar(GrammarVocab* grammar) { grammar_ = grammar; }

    // Attach a vocab pruning set (optional).
    void set_pruned_vocab(const std::unordered_set<int32_t>* pruned_vocab) {
        pruned_vocab_ = pruned_vocab;
    }

    void set_eos_token_id(int32_t id) {
        eos_token_id_ = id;
    }

    // Build and attach a TokenTrie from the vocabulary for accelerated
    // grammar-constrained decoding. Called once; the trie is immutable.
    void build_token_trie(const std::vector<std::string>& vocab);

protected:
    void apply_vocab_pruning(std::vector<float>& logits);
    void apply_grammar_constraints(std::vector<float>& logits, const std::vector<int32_t>& last_tokens,
                                   const std::vector<std::string>& token_strs);

    GrammarVocab* grammar_     = nullptr;
    const std::unordered_set<int32_t>* pruned_vocab_ = nullptr;
    int32_t eos_token_id_ = -1;
    std::unique_ptr<TokenTrie> token_trie_;
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

} // namespace qwen3