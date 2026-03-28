#include "sampling.h"
#include "grammar.h" 
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <numeric>
#include <vector>

#include <iostream>

namespace qwen3 {

void Sampler::apply_vocab_pruning(std::vector<float>& logits) {
    if (!pruned_vocab_) {
        return;  // No pruning constraint
    }
    
    for (size_t i = 0; i < logits.size(); ++i) {
        if (pruned_vocab_->find(static_cast<int32_t>(i)) == pruned_vocab_->end()) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

void Sampler::apply_grammar_constraints(std::vector<float>& logits,
                                       const std::vector<std::string>& token_strs) {
    if (!grammar_) {
        return;  // No grammar constraint
    }
    
    // Get valid tokens from grammar
    std::vector<int32_t> valid_tokens = grammar_->get_valid_tokens(token_strs);
    
    // if (valid_tokens.empty()) {
    //     return;  // Grammar allows all tokens (or error - be permissive)
    // }

if (valid_tokens.empty()) {
    std::cerr << "[Grammar] WARNING: No valid tokens — grammar is stuck!" << std::endl;
    // Force EOS instead of going permissive
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = -std::numeric_limits<float>::infinity();
    }
    // Allow only EOS
    logits[151643] = 0.0f; // Qwen2 EOS token ID — adjust if needed
    return;
}    
    
    // Build fast lookup set
    std::unordered_set<int32_t> valid_set(valid_tokens.begin(), valid_tokens.end());
    
    // Mask out invalid tokens
    for (size_t i = 0; i < logits.size(); ++i) {
        if (valid_set.find(static_cast<int32_t>(i)) == valid_set.end()) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

int GreedySampler::sample(std::vector<float>& logits,
                          const std::vector<int32_t>& last_tokens,
                          const std::vector<std::string>& token_strs) {
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits");
    }

    // 1. Pruning
    apply_vocab_pruning(logits);

    // 2. Grammar
    if (grammar_ && !token_strs.empty()) {
        apply_grammar_constraints(logits, token_strs);
    }
    
    // 3. Selection (argmax)
    auto max_it = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_it);
}

TemperatureSampler::TemperatureSampler(float temperature, float repetition_penalty, int repetition_lookback, int top_k, float top_p)
    : temperature_(temperature), 
      repetition_penalty_(repetition_penalty),
      repetition_lookback_(repetition_lookback),
      top_k_(top_k),
      top_p_(top_p) {
    std::random_device rd;
    gen_ = std::mt19937(rd());
}

void TemperatureSampler::apply_repetition_penalty(std::vector<float>& logits, const std::vector<int32_t>& last_tokens) {
    if (repetition_penalty_ == 1.0f || last_tokens.empty()) {
        return;
    }

    const size_t lookback = std::min(static_cast<size_t>(repetition_lookback_), last_tokens.size());
    std::unordered_set<int32_t> recent_tokens(last_tokens.end() - lookback, last_tokens.end());
    
    for (int32_t token_id : recent_tokens) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < logits.size()) {
            if (logits[token_id] < 0) {
                logits[token_id] *= repetition_penalty_;
            } else {
                logits[token_id] /= repetition_penalty_;
            }
        }
    }
}

int TemperatureSampler::sample(std::vector<float>& logits,
                               const std::vector<int32_t>& last_tokens,
                               const std::vector<std::string>& token_strs) {
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits");
    }

    // 1. Pruning
    apply_vocab_pruning(logits);

    // 2. Grammar
    if (grammar_ && !token_strs.empty()) {
        apply_grammar_constraints(logits, token_strs);
    }
    
    // 3. Repetition penalty
    apply_repetition_penalty(logits, last_tokens);

    // 4. Temperature (greedy fallback if <= 0)
    if (temperature_ <= 0.0f) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        return std::distance(logits.begin(), max_it);
    }

    const size_t vocab_size = logits.size();
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] /= temperature_;
    }

    // 5. Sort for Top-K / Top-P filtering
    std::vector<std::pair<int, float>> sorted_logits;
    sorted_logits.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        sorted_logits.emplace_back(static_cast<int>(i), logits[i]);
    }
    std::sort(sorted_logits.begin(), sorted_logits.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Apply Top-K
    if (top_k_ > 0 && top_k_ < static_cast<int>(vocab_size)) {
        sorted_logits.resize(top_k_);
    }

    // Apply Top-P
    if (top_p_ > 0.0f && top_p_ < 1.0f) {
        float max_logit = sorted_logits[0].second;
        float sum_exp = 0.0f;
        for (const auto& p : sorted_logits) {
            sum_exp += expf(p.second - max_logit);
        }

        float cum_prob = 0.0f;
        size_t cutoff_idx = sorted_logits.size();
        for (size_t i = 0; i < sorted_logits.size(); ++i) {
            cum_prob += expf(sorted_logits[i].second - max_logit) / sum_exp;
            if (cum_prob >= top_p_) {
                cutoff_idx = i + 1;
                break;
            }
        }
        sorted_logits.resize(cutoff_idx);
    }

    // 6. Softmax on filtered logits
    float max_l = sorted_logits[0].second;
    float sum_exp = 0.0f;
    std::vector<float> probabilities;
    probabilities.reserve(sorted_logits.size());
    for (const auto& p : sorted_logits) {
        float prob = expf(p.second - max_l);
        probabilities.push_back(prob);
        sum_exp += prob;
    }

    for (float& p : probabilities) {
        p /= sum_exp;
    }

    // 7. Sample
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    int sampled_idx = dist(gen_);
    
    return sorted_logits[sampled_idx].first;
}

} // namespace qwen3