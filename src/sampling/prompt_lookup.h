#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>

namespace qwen3 {

// ============================================================================
// PromptLookup: N-gram based draft token proposal
//
// Searches the prompt for a match of the last N generated tokens.
// If found, proposes the next K tokens from the prompt as draft candidates.
// This is "Prompt Lookup Decoding" (PLD) — no draft model needed.
// ============================================================================

struct PromptLookupConfig {
    int ngram_size = 3;    // N: how many recent tokens to match
    int max_draft  = 5;    // K: max draft tokens to propose
    int min_draft  = 1;    // Minimum useful draft length
};

class PromptLookup {
public:
    explicit PromptLookup(PromptLookupConfig config = {}) : config_(config) {}

    // Find draft tokens by searching prompt for n-gram match.
    //
    // prompt_tokens:    the original input prompt tokens (the haystack)
    // generated_tokens: all tokens generated so far (needle comes from tail)
    //
    // Returns: proposed draft tokens (empty if no match found)
    std::vector<int32_t> find_draft(
        const std::vector<int32_t>& prompt_tokens,
        const std::vector<int32_t>& generated_tokens) const
    {
        const int n = config_.ngram_size;

        if ((int)generated_tokens.size() < n) {
            return {};
        }
        if ((int)prompt_tokens.size() < n) {
            return {};
        }

        // Needle: last N generated tokens
        const int32_t* needle = generated_tokens.data() + generated_tokens.size() - n;

        // Search backwards through prompt (later matches more likely relevant)
        int best_pos = -1;
        int best_draft_len = 0;

        for (int i = (int)prompt_tokens.size() - n; i >= 0; --i) {
            bool match = true;
            for (int j = 0; j < n; ++j) {
                if (prompt_tokens[i + j] != needle[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                int draft_start = i + n;
                int draft_len = std::min(config_.max_draft,
                    (int)prompt_tokens.size() - draft_start);

                if (draft_len >= config_.min_draft) {
                    // Take first valid match (searching backwards = most recent context)
                    best_pos = draft_start;
                    best_draft_len = draft_len;
                    break;
                }
            }
        }

        if (best_pos < 0) {
            return {};
        }

        return std::vector<int32_t>(
            prompt_tokens.begin() + best_pos,
            prompt_tokens.begin() + best_pos + best_draft_len);
    }

    const PromptLookupConfig& config() const { return config_; }

private:
    PromptLookupConfig config_;
};

} // namespace qwen3