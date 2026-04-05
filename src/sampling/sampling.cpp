#include "sampling.h"
// grammar_vocab.h is pulled in via sampling.h.
// grammar.h and pretokenized_literals.h are no longer needed.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <numeric>
#include <vector>
#include <iostream>

namespace qwen3 {

void Sampler::apply_vocab_pruning(std::vector<float>& logits) {
    if (!pruned_vocab_) return;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (pruned_vocab_->find(static_cast<int32_t>(i)) == pruned_vocab_->end())
            logits[i] = -std::numeric_limits<float>::infinity();
    }
}

void Sampler::apply_grammar_constraints(std::vector<float>& logits, const std::vector<int32_t>& last_tokens,
                                        const std::vector<std::string>& token_strs) {
    if (!grammar_) return;

    // vocab-scan, no pretokenized map.
    std::vector<int32_t> valid_tokens = grammar_->get_valid_tokens(token_strs);

    if (valid_tokens.empty()) {
        std::cerr << "[GrammarVocab] ERROR: Grammar is stuck! No valid tokens found in vocabulary.\n";
        std::cerr << "  Context (last 5 tokens):\n";
        int start = std::max(0, (int)last_tokens.size() - 5);
        for (size_t i = start; i < last_tokens.size(); ++i) {
            std::cerr << "'" << token_strs[last_tokens[i]] << "' ";
        }
        std::cerr << "\n";
        grammar_->dump_expected();
        std::cerr << "  Action: Forcing EOS to prevent crash.\n" << std::endl;

        for (size_t i = 0; i < logits.size(); ++i)
            logits[i] = -std::numeric_limits<float>::infinity();
        // Find EOS by scanning vocab strings — no hardcoded IDs.
        for (size_t i = 0; i < token_strs.size(); ++i) {
            if (token_strs[i] == "<|endoftext|>" || token_strs[i] == "<|im_end|>") {
                logits[i] = 0.0f;
                break;
            }
        }
        return;
    }

    std::unordered_set<int32_t> valid_set(valid_tokens.begin(), valid_tokens.end());

    // When grammar is accepting, also allow EOS — the model decides whether
    // to end or continue with another statement naturally.
    if (grammar_->is_accepting_state()) {
        
        for (size_t i = 0; i < token_strs.size(); ++i) {
            if (token_strs[i] == "<|endoftext|>" || token_strs[i] == "<|im_end|>") {
                valid_set.insert(static_cast<int32_t>(i));
                break;
            }
        }

        valid_set.insert(eos_token_id_);
    }
    

    for (size_t i = 0; i < logits.size(); ++i) {
        if (!valid_set.count(static_cast<int32_t>(i)))
            logits[i] = -std::numeric_limits<float>::infinity();
    }
}

// ---- GreedySampler ----

int GreedySampler::sample(std::vector<float>& logits,
                          const std::vector<int32_t>& last_tokens,
                          const std::vector<std::string>& token_strs) {
    if (logits.empty()) throw std::runtime_error("Cannot sample from empty logits");
    apply_vocab_pruning(logits);
    if (grammar_ && !token_strs.empty())
        apply_grammar_constraints(logits, last_tokens, token_strs);
    auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<int>(std::distance(logits.begin(), max_it));
}

// ---- TemperatureSampler ----

TemperatureSampler::TemperatureSampler(float temperature, float repetition_penalty,
                                       int repetition_lookback, int top_k, float top_p)
    : temperature_(temperature),
      repetition_penalty_(repetition_penalty),
      repetition_lookback_(repetition_lookback),
      top_k_(top_k),
      top_p_(top_p)
{
    std::random_device rd;
    gen_ = std::mt19937(rd());
}

void TemperatureSampler::apply_repetition_penalty(std::vector<float>& logits,
                                                   const std::vector<int32_t>& last_tokens) {
    if (repetition_penalty_ == 1.0f || last_tokens.empty()) return;
    const size_t lookback = std::min(static_cast<size_t>(repetition_lookback_), last_tokens.size());
    std::unordered_set<int32_t> recent(last_tokens.end() - lookback, last_tokens.end());
    for (int32_t id : recent) {
        if (id >= 0 && static_cast<size_t>(id) < logits.size()) {
            logits[id] = logits[id] < 0 ? logits[id] * repetition_penalty_
                                        : logits[id] / repetition_penalty_;
        }
    }
}

int TemperatureSampler::sample(std::vector<float>& logits,
                               const std::vector<int32_t>& last_tokens,
                               const std::vector<std::string>& token_strs) {
    if (logits.empty()) throw std::runtime_error("Cannot sample from empty logits");

    apply_vocab_pruning(logits);

    if (grammar_ && !token_strs.empty())
        apply_grammar_constraints(logits, last_tokens, token_strs);

    apply_repetition_penalty(logits, last_tokens);

    if (temperature_ <= 0.0f) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        return static_cast<int>(std::distance(logits.begin(), max_it));
    }

    const size_t vocab_size = logits.size();
    for (size_t i = 0; i < vocab_size; ++i)
        logits[i] /= temperature_;

    std::vector<std::pair<int, float>> sorted;
    sorted.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i)
        sorted.emplace_back(static_cast<int>(i), logits[i]);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (top_k_ > 0 && top_k_ < static_cast<int>(vocab_size))
        sorted.resize(top_k_);

    if (top_p_ > 0.0f && top_p_ < 1.0f) {
        const float max_l = sorted[0].second;
        float sum_exp = 0.0f;
        for (const auto& p : sorted) sum_exp += expf(p.second - max_l);
        float cum = 0.0f;
        size_t cutoff = sorted.size();
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += expf(sorted[i].second - max_l) / sum_exp;
            if (cum >= top_p_) { cutoff = i + 1; break; }
        }
        sorted.resize(cutoff);
    }

    const float max_l = sorted[0].second;
    float sum_exp = 0.0f;
    std::vector<float> probs;
    probs.reserve(sorted.size());
    for (const auto& p : sorted) {
        float e = expf(p.second - max_l);
        probs.push_back(e);
        sum_exp += e;
    }
    for (float& p : probs) p /= sum_exp;

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return sorted[dist(gen_)].first;
}

} // namespace qwen3