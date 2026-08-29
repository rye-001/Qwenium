#include "sampling.h"
#include "snapshot_io.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <numeric>
#include <vector>
#include <iostream>
#include <sstream>

namespace qinf {

void Sampler::apply_vocab_pruning(std::vector<float>& logits) {
    if (!pruned_vocab_) return;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (pruned_vocab_->find(static_cast<int32_t>(i)) == pruned_vocab_->end())
            logits[i] = -std::numeric_limits<float>::infinity();
    }
}

std::vector<int32_t> Sampler::peek_valid_set() {
    if (!grammar_ || vocab_.empty()) {
        cached_valid_ids_valid_ = false;
        return {};
    }

    std::vector<int32_t> valid = grammar_->get_valid_tokens(vocab_);

    // If the grammar is in an accepting state, ensure all registered stop
    // tokens are included in the valid set.
    if (grammar_->is_accepting_state()) {
        for (int32_t id : eos_token_ids_)
            valid.push_back(id);
    }

    if (pruned_vocab_) {
        valid.erase(
            std::remove_if(valid.begin(), valid.end(),
                [this](int32_t id) { return pruned_vocab_->count(id) == 0; }),
            valid.end());
    }

    // Cache the result so apply_grammar_constraints can consume it on the
    // dense path without calling get_valid_tokens a second time. Stamp the
    // grammar's state_version so apply can fail loud on a stale read.
    cached_valid_ids_         = valid;
    cached_valid_ids_valid_   = true;
    cached_valid_ids_version_ = grammar_->state_version();

    return valid;
}

void Sampler::build_token_trie(const std::vector<std::string>& vocab) {
    vocab_ = vocab;
    token_trie_ = std::make_unique<TokenTrie>();
    token_trie_->build(vocab);
}

void Sampler::apply_grammar_constraints(std::vector<float>& logits, const std::vector<int32_t>& last_tokens,
                                        const std::vector<std::string>& token_strs) {
    if (!grammar_) return;

    // Lazily build the trie on first grammar use if not yet built.
    if (!token_trie_ && !token_strs.empty()) {
        build_token_trie(token_strs);
    }
    // Attach trie to grammar for accelerated lookups.
    if (token_trie_) {
        grammar_->set_token_trie(token_trie_.get());
    }

    // Use the result cached by peek_valid_set() when available — avoids a
    // redundant get_valid_tokens call on the dense decode path. The cache
    // is stamped with the grammar's state_version at write time; assert it
    // still matches before consuming, else the cache is stale (caller ran
    // accept_token between peek and apply).
    std::vector<int32_t> valid_tokens;
    if (cached_valid_ids_valid_) {
        const uint64_t current_version = grammar_->state_version();
        if (current_version != cached_valid_ids_version_) {
            throw std::runtime_error(
                "Sampler::apply_grammar_constraints: stale cached_valid_ids_; "
                "expected: grammar->state_version() == cached_valid_ids_version_ ("
                + std::to_string(cached_valid_ids_version_) + "), "
                "actual: " + std::to_string(current_version));
        }
        valid_tokens            = std::move(cached_valid_ids_);
        cached_valid_ids_valid_ = false;
    } else {
        valid_tokens = grammar_->get_valid_tokens(token_strs);
    }

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
        for (int32_t id : eos_token_ids_) {
            if (id >= 0 && static_cast<size_t>(id) < logits.size())
                logits[id] = 0.0f;
        }
        return;
    }

    const size_t vocab_size = logits.size();
    std::vector<bool> valid_mask(vocab_size, false);
    for (int32_t id : valid_tokens) valid_mask[id] = true;

    if (grammar_->is_accepting_state()) {
        for (int32_t id : eos_token_ids_) {
            if (static_cast<size_t>(id) < vocab_size)
                valid_mask[id] = true;
        }
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        if (!valid_mask[i])
            logits[i] = -std::numeric_limits<float>::infinity();
    }
}

// ---- GreedySampler ----

void GreedySampler::apply_repetition_penalty(std::vector<float>& logits,
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

void Sampler::apply_repetition_penalty_sparse(std::vector<float>& sparse_logits,
                                               const std::vector<int32_t>& valid_ids,
                                               const std::vector<int32_t>& last_tokens,
                                               float penalty, int lookback) {
    if (penalty == 1.0f || last_tokens.empty()) return;
    const size_t window = std::min(static_cast<size_t>(lookback), last_tokens.size());
    std::unordered_set<int32_t> recent(last_tokens.end() - window, last_tokens.end());
    for (size_t j = 0; j < valid_ids.size(); ++j) {
        if (recent.count(valid_ids[j])) {
            sparse_logits[j] = sparse_logits[j] < 0 ? sparse_logits[j] * penalty
                                                     : sparse_logits[j] / penalty;
        }
    }
}

int GreedySampler::sample(std::vector<float>& logits,
                          const std::vector<int32_t>& last_tokens,
                          const std::vector<std::string>& token_strs) {
    if (logits.empty()) throw std::runtime_error("Cannot sample from empty logits");
    apply_vocab_pruning(logits);
    if (grammar_ && !token_strs.empty())
        apply_grammar_constraints(logits, last_tokens, token_strs);
    apply_repetition_penalty(logits, last_tokens);
    auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<int>(std::distance(logits.begin(), max_it));
}

int32_t GreedySampler::sample_sparse(std::vector<float>& sparse_logits,
                                      const std::vector<int32_t>& valid_ids,
                                      const std::vector<int32_t>& last_tokens) {
    if (sparse_logits.empty())
        throw std::runtime_error("sample_sparse: empty logits; expected: non-empty, actual: 0");
    if (sparse_logits.size() != valid_ids.size())
        throw std::runtime_error("sample_sparse: sparse_logits.size() != valid_ids.size(); expected: equal, actual: "
                                 + std::to_string(sparse_logits.size()) + " vs "
                                 + std::to_string(valid_ids.size()));
    Sampler::apply_repetition_penalty_sparse(sparse_logits, valid_ids, last_tokens,
                                              repetition_penalty_, repetition_lookback_);
    const size_t j = static_cast<size_t>(
        std::distance(sparse_logits.begin(),
                      std::max_element(sparse_logits.begin(), sparse_logits.end())));
    if (j >= valid_ids.size())
        throw std::runtime_error("sample_sparse: argmax index out of range; expected: < "
                                 + std::to_string(valid_ids.size()) + ", actual: "
                                 + std::to_string(j));
    const int32_t id = valid_ids[j];
    if (id < 0)
        throw std::runtime_error("sample_sparse: returned id from valid_ids is negative; expected: >= 0, actual: "
                                 + std::to_string(id));
    return id;
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

    // Total order: logit descending, token id ascending on exact ties. The tie
    // break is explicit because the sampler supports a caller-chosen seed
    // (OpenAI `seed`) and a reproducible draw stream needs a reproducible
    // candidate list; a logit-only comparator leaves ties unspecified.
    const auto by_logit = [](const std::pair<int, float>& a,
                             const std::pair<int, float>& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    };

    // Only the top-k candidates can survive, so rank only that many. Fully
    // sorting the vocabulary is O(V log V) every step, and at Qwen 3.5's
    // 248 320-token vocabulary that cost 15 ms/token -- about as much as the
    // entire forward pass. nth_element makes the selection O(V) and leaves
    // only the k survivors to sort.
    if (top_k_ > 0 && top_k_ < static_cast<int>(vocab_size)) {
        const size_t keep = static_cast<size_t>(top_k_);
        std::nth_element(sorted.begin(), sorted.begin() + keep, sorted.end(), by_logit);
        sorted.resize(keep);
    }
    std::sort(sorted.begin(), sorted.end(), by_logit);

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

int32_t TemperatureSampler::sample_sparse(std::vector<float>& sparse_logits,
                                           const std::vector<int32_t>& valid_ids,
                                           const std::vector<int32_t>& last_tokens) {
    if (sparse_logits.empty())
        throw std::runtime_error("sample_sparse: empty logits; expected: non-empty, actual: 0");
    if (sparse_logits.size() != valid_ids.size())
        throw std::runtime_error("sample_sparse: sparse_logits.size() != valid_ids.size(); expected: equal, actual: "
                                 + std::to_string(sparse_logits.size()) + " vs "
                                 + std::to_string(valid_ids.size()));

    Sampler::apply_repetition_penalty_sparse(sparse_logits, valid_ids, last_tokens,
                                              repetition_penalty_, repetition_lookback_);

    auto check_id = [&](size_t j) -> int32_t {
        if (j >= valid_ids.size())
            throw std::runtime_error("sample_sparse: argmax index out of range; expected: < "
                                     + std::to_string(valid_ids.size()) + ", actual: "
                                     + std::to_string(j));
        const int32_t id = valid_ids[j];
        if (id < 0)
            throw std::runtime_error("sample_sparse: returned id from valid_ids is negative; expected: >= 0, actual: "
                                     + std::to_string(id));
        return id;
    };

    if (temperature_ <= 0.0f) {
        const size_t j = static_cast<size_t>(
            std::distance(sparse_logits.begin(),
                          std::max_element(sparse_logits.begin(), sparse_logits.end())));
        return check_id(j);
    }

    const size_t k = sparse_logits.size();
    for (size_t j = 0; j < k; ++j)
        sparse_logits[j] /= temperature_;

    // sorted holds (position j in sparse_logits, logit value)
    std::vector<std::pair<size_t, float>> sorted;
    sorted.reserve(k);
    for (size_t j = 0; j < k; ++j)
        sorted.emplace_back(j, sparse_logits[j]);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (top_k_ > 0 && static_cast<size_t>(top_k_) < sorted.size())
        sorted.resize(static_cast<size_t>(top_k_));

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
    return check_id(sorted[static_cast<size_t>(dist(gen_))].first);
}

// --- L1 snapshot: mt19937 engine state ---
// std::mt19937 exposes no "draw counter"; its operator<< / operator>> serialize
// the full internal state (the 624-word array + position) as space-separated
// decimal text. That round-trips exactly and portably, so a restored sampler
// produces a bit-identical draw sequence. We carry it as a length-prefixed
// string in the snapshot stream.
void TemperatureSampler::write_rng_state(qinf::session::SnapshotWriter& w) const {
    std::ostringstream oss;
    oss << gen_;
    w.put_string(oss.str());
}

void TemperatureSampler::read_rng_state(qinf::session::SnapshotReader& r) {
    std::string state = r.get_string();
    std::istringstream iss(state);
    iss >> gen_;
    if (iss.fail()) {
        throw std::runtime_error(
            "sampler_rng: field \"mt19937_state\" expected a valid serialized "
            "std::mt19937 state, got an unparseable string of length " +
            std::to_string(state.size()));
    }
}

} // namespace qinf