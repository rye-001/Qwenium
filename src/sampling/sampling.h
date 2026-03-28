#pragma once

#include <vector>
#include <random>
#include <unordered_set>
#include <cstdint>
#include <string>

namespace qwen3 {

// Forward declaration for grammar constraint
struct Grammar;

// Abstract base class for sampling strategies
class Sampler {
public:
    virtual ~Sampler() = default;
    
    // Sample from logits vector (CPU data)
    // token_strs: vocabulary strings indexed by token_id (for grammar)
    virtual int sample(std::vector<float>& logits,
                       const std::vector<int32_t>& last_tokens,
                       const std::vector<std::string>& token_strs = {}) = 0;
    
    // Set grammar constraint (optional, nullptr = no grammar)
    void set_grammar(Grammar* grammar) { grammar_ = grammar; }
    
    // Set pruned vocabulary (optional, nullptr = no pruning)
    void set_pruned_vocab(const std::unordered_set<int32_t>* vocab) { pruned_vocab_ = vocab; }
    
protected:
    // Apply vocab pruning to logits (mask tokens not in pruned set)
    void apply_vocab_pruning(std::vector<float>& logits);
    
    // Apply grammar constraints to logits (mask invalid tokens)
    void apply_grammar_constraints(std::vector<float>& logits,
                                   const std::vector<std::string>& token_strs);
    
    Grammar* grammar_ = nullptr;
    const std::unordered_set<int32_t>* pruned_vocab_ = nullptr;
};

// Greedy sampling implementation
class GreedySampler : public Sampler {
public:
    explicit GreedySampler() = default;
    int sample(std::vector<float>& logits,
               const std::vector<int32_t>& last_tokens,
               const std::vector<std::string>& token_strs = {}) override;
};

// Temperature sampling implementation
class TemperatureSampler : public Sampler {
public:
    explicit TemperatureSampler(float temperature, 
                                float repetition_penalty = 1.1f, 
                                int repetition_lookback = 64,
                                int top_k = 40,
                                float top_p = 0.95f);
    int sample(std::vector<float>& logits,
               const std::vector<int32_t>& last_tokens,
               const std::vector<std::string>& token_strs = {}) override;

private:
    void apply_repetition_penalty(std::vector<float>& logits, const std::vector<int32_t>& last_tokens);

    float temperature_;
    float repetition_penalty_;
    int repetition_lookback_;
    int top_k_;
    float top_p_;
    std::mt19937 gen_;
};

} // namespace qwen3