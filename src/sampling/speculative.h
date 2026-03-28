#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <algorithm>
#include <cassert>

#include "prompt_lookup.h"

namespace qwen3 {

// ============================================================================
// SpeculativeResult: What the caller gets back
// ============================================================================

struct SpeculativeResult {
    // Draft tokens that were verified correct (may be empty)
    std::vector<int32_t> accepted_tokens;

    // The correct token at the first mismatch position.
    // If all K draft tokens accepted, this is the token AFTER the last draft.
    // Always valid when draft was attempted (even if 0 accepted).
    int32_t bonus_token = -1;
    bool has_bonus = false;

    // How many tokens were proposed (for stats/logging)
    int draft_length = 0;

    // Total tokens produced this step: accepted.size() + (has_bonus ? 1 : 0)
    int total_tokens() const {
        return (int)accepted_tokens.size() + (has_bonus ? 1 : 0);
    }

    // Was a draft even attempted?
    bool attempted() const { return draft_length > 0; }
};

// ============================================================================
// SpeculativeStats: Running counters for monitoring
// ============================================================================

struct SpeculativeStats {
    int64_t drafts_attempted  = 0;
    int64_t drafts_found      = 0;   // n-gram match found
    int64_t tokens_drafted    = 0;   // total draft tokens proposed
    int64_t tokens_accepted   = 0;   // total draft tokens verified correct
    int64_t bonus_tokens      = 0;   // total bonus tokens from verification
    int64_t normal_decodes    = 0;   // steps where no draft was found

    double acceptance_rate() const {
        return tokens_drafted > 0 ? (double)tokens_accepted / tokens_drafted : 0.0;
    }

    // Average tokens per step (higher = better, 1.0 = no speedup)
    double tokens_per_step() const {
        int64_t total_steps = drafts_found + normal_decodes;
        int64_t total_tokens = tokens_accepted + bonus_tokens + normal_decodes;
        return total_steps > 0 ? (double)total_tokens / total_steps : 1.0;
    }

    void reset() { *this = SpeculativeStats{}; }
};

// ============================================================================
// SpeculativeDecoder: Orchestrates draft → verify → accept
//
// Usage (caller provides two lambdas):
//
//   VerifyFunc: runs K draft tokens through the FULL model for a slot.
//     - Input:  slot_id, draft tokens, start position
//     - Output: logits for ALL K positions [K * vocab_size]
//     - Side effect: KV cache advances by K positions
//
//   RewindCacheFunc: resets a slot's KV cache position.
//     - Input:  slot_id, new position
//     - Side effect: KV cache position set to given value
//
// Both main.cpp and inference_server.h can wrap their forward pass into these.
// ============================================================================

class SpeculativeDecoder {
public:
    // Run K draft tokens through model, return logits [K * vocab_size]
    using VerifyFunc = std::function<std::vector<float>(
        int slot_id,
        const std::vector<int32_t>& draft_tokens,
        int start_pos
    )>;

    // Set a slot's KV cache position (for rewind after partial acceptance)
    using RewindCacheFunc = std::function<void(int slot_id, int new_pos)>;

    explicit SpeculativeDecoder(
        PromptLookupConfig config = {},
        int vocab_size = 0)
        : lookup_(config)
        , vocab_size_(vocab_size)
    {}

    void set_vocab_size(int vs) { vocab_size_ = vs; }

    // ========================================================================
    // try_speculative_step: The main entry point
    //
    // Call this after each normal decode step. It will:
    //   1. Search for an n-gram match in the prompt
    //   2. If found, verify draft tokens with a single forward pass
    //   3. Return accepted tokens + bonus token
    //
    // The caller is responsible for:
    //   - Delivering accepted_tokens + bonus_token to output
    //   - NOT running another normal decode for the bonus token
    //     (it's already verified)
    //
    // Parameters:
    //   prompt_tokens:    original prompt (for n-gram search)
    //   generated_tokens: all tokens generated so far (including latest)
    //   slot_id:          which KV cache slot
    //   current_pos:      current KV cache position (after latest token)
    //   verify:           forward pass callable
    //   rewind:           cache rewind callable
    //   eos_token:        stop if draft/bonus is EOS
    //
    // Returns: SpeculativeResult (may be empty if no draft found)
    // ========================================================================
    SpeculativeResult try_speculative_step(
        const std::vector<int32_t>& prompt_tokens,
        const std::vector<int32_t>& generated_tokens,
        int slot_id,
        int current_pos,
        VerifyFunc verify,
        RewindCacheFunc rewind,
        int eos_token = -1)
    {
        assert(vocab_size_ > 0 && "Must set vocab_size before calling");

        SpeculativeResult result;
        stats_.drafts_attempted++;

        // 1. Try to find draft tokens via n-gram lookup
        std::vector<int32_t> draft = lookup_.find_draft(prompt_tokens, generated_tokens);

        if (draft.empty()) {
            stats_.normal_decodes++;
            return result;  // No draft found, caller does normal decode
        }

        stats_.drafts_found++;
        result.draft_length = (int)draft.size();
        stats_.tokens_drafted += draft.size();

        // 2. Run all draft tokens through full model in one forward pass
        //    This is equivalent to a prefill — causal attention, all positions
        //    KV cache advances by draft.size()
        std::vector<float> all_logits = verify(slot_id, draft, current_pos);

        // Sanity check: expect [K * vocab_size] logits
        int K = (int)draft.size();
        if ((int)all_logits.size() != K * vocab_size_) {
            // Verification returned unexpected shape — bail out safely
            rewind(slot_id, current_pos);
            stats_.normal_decodes++;
            return result;
        }

        // 3. Greedy verification: check argmax(logits[i]) == draft[i]
        //
        // Position 0 logits verify draft[0]: these are the logits from
        // running draft[0] at current_pos, predicting what comes next.
        //
        // Wait — careful! The logits at position i predict the NEXT token.
        // So logits[0] (from seeing draft[0]) predicts draft[1].
        // We need to verify: argmax(logits[i]) == draft[i+1] for i in 0..K-2
        // And the logits from the token BEFORE draft[0] already predicted draft[0]
        // (that was the normal decode step).
        //
        // Actually: the verify function runs draft[0..K-1] as a prefill.
        // - logits[0] = model output after seeing everything up to draft[0]
        //             → predicts what should come at position draft[1]
        // - logits[K-1] = model output after seeing everything up to draft[K-1]
        //               → predicts the token AFTER the draft
        //
        // So we verify: argmax(logits[i]) == draft[i+1] for i in 0..K-2
        // And bonus_token = argmax(logits[K-1])

        int accepted = 0;
        for (int i = 0; i < K - 1; ++i) {
            const float* logits_i = all_logits.data() + i * vocab_size_;
            int predicted = argmax(logits_i, vocab_size_);

            if (predicted != draft[i + 1]) {
                // Mismatch: draft[i+1] is wrong
                // accepted = i + 1 tokens (draft[0] through draft[i] are valid)
                // bonus = predicted (the correct token at position i+1)
                accepted = i + 1;
                result.bonus_token = predicted;
                result.has_bonus = (predicted != eos_token);
                break;
            }

            // Check EOS in accepted draft
            if (draft[i + 1] == eos_token) {
                accepted = i + 1;
                result.bonus_token = eos_token;
                result.has_bonus = false;
                break;
            }

            if (i == K - 2) {
                // All K-1 checks passed → all K draft tokens accepted
                accepted = K;
                // Bonus = what the model predicts after the entire draft
                const float* logits_last = all_logits.data() + (K - 1) * vocab_size_;
                result.bonus_token = argmax(logits_last, vocab_size_);
                result.has_bonus = (result.bonus_token != eos_token);
            }
        }

        // Handle K=1 case (single draft token, just get the bonus)
        if (K == 1) {
            accepted = 1;  // draft[0] was already verified by normal decode
            const float* logits_0 = all_logits.data();
            result.bonus_token = argmax(logits_0, vocab_size_);
            result.has_bonus = (result.bonus_token != eos_token);
        }

        // 4. Rewind KV cache to discard unverified positions
        //    Cache was advanced by K tokens. We keep: accepted + (has_bonus ? 1 : 0)
        int keep = accepted + (result.has_bonus ? 1 : 0);
        if (keep < K) {
            rewind(slot_id, current_pos + keep);
        }

        // 5. Build result
        result.accepted_tokens.assign(draft.begin(), draft.begin() + accepted);
        stats_.tokens_accepted += accepted;
        if (result.has_bonus) stats_.bonus_tokens++;

        return result;
    }

    const SpeculativeStats& stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }
    const PromptLookup& lookup() const { return lookup_; }

private:
    static int argmax(const float* data, int size) {
        int best = 0;
        float best_val = data[0];
        for (int i = 1; i < size; ++i) {
            if (data[i] > best_val) {
                best_val = data[i];
                best = i;
            }
        }
        return best;
    }

    PromptLookup lookup_;
    int vocab_size_ = 0;
    SpeculativeStats stats_;
};

} // namespace qwen3