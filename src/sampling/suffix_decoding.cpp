#include "suffix_decoding.h"

#include "../qinf_error.h"

#include <algorithm>
#include <string>

namespace qinf {

SuffixDecoding::SuffixDecoding(SuffixDecodingConfig config) : config_(config) {
    QINF_ASSERT(config_.min_match_len >= 1,
        "SuffixDecoding: min_match_len expected >= 1, got " +
        std::to_string(config_.min_match_len));
    QINF_ASSERT(config_.max_match_len >= config_.min_match_len,
        "SuffixDecoding: max_match_len expected >= min_match_len (" +
        std::to_string(config_.min_match_len) + "), got " +
        std::to_string(config_.max_match_len));
    QINF_ASSERT(config_.max_draft >= 1,
        "SuffixDecoding: max_draft expected >= 1, got " +
        std::to_string(config_.max_draft));
    QINF_ASSERT(config_.max_indexed_tokens >= (std::size_t)config_.max_match_len,
        "SuffixDecoding: max_indexed_tokens expected >= max_match_len (" +
        std::to_string(config_.max_match_len) + "), got " +
        std::to_string(config_.max_indexed_tokens));
}

namespace {

// haystack = the most recent `bound` tokens of prompt_tokens ++
// generated_tokens. O(bound) regardless of how large either input vector is
// -- the cap that keeps find_draft's cost independent of session length.
std::vector<int32_t> build_bounded_haystack(
    const std::vector<int32_t>& prompt_tokens,
    const std::vector<int32_t>& generated_tokens,
    std::size_t bound)
{
    const std::size_t gen_n = generated_tokens.size();

    if (gen_n >= bound) {
        // Generated tokens alone already fill the window; the prompt has
        // scrolled entirely out of the bounded index.
        return std::vector<int32_t>(generated_tokens.end() - (std::ptrdiff_t)bound,
                                     generated_tokens.end());
    }

    const std::size_t prompt_budget = bound - gen_n;
    const std::size_t prompt_take = std::min(prompt_budget, prompt_tokens.size());

    std::vector<int32_t> haystack;
    haystack.reserve(prompt_take + gen_n);
    haystack.insert(haystack.end(),
                     prompt_tokens.end() - (std::ptrdiff_t)prompt_take,
                     prompt_tokens.end());
    haystack.insert(haystack.end(), generated_tokens.begin(), generated_tokens.end());
    return haystack;
}

} // namespace

std::vector<int32_t> SuffixDecoding::find_draft(
    const std::vector<int32_t>& prompt_tokens,
    const std::vector<int32_t>& generated_tokens) const
{
    // The needle is drawn from generated_tokens' tail, same as PLD: there is
    // no "recent output" to match against before anything has been
    // generated, so a match length longer than what's been generated isn't
    // meaningful yet.
    const int max_n = std::min(config_.max_match_len, (int)generated_tokens.size());
    if (max_n < config_.min_match_len) {
        return {};
    }

    const std::vector<int32_t> haystack =
        build_bounded_haystack(prompt_tokens, generated_tokens, config_.max_indexed_tokens);
    const int hsize = (int)haystack.size();

    // Adaptive length: try the longest n-gram first, shrink toward the
    // floor. Longer match found first ⇒ we always prefer it (more
    // confident continuation) over a shorter one.
    for (int n = max_n; n >= config_.min_match_len; --n) {
        // Needle = the last n tokens of the haystack (== the last n tokens
        // of generated_tokens, since the haystack always ends with all of
        // generated_tokens or its bounded tail).
        const int needle_start = hsize - n;

        // Search earlier occurrences of the needle, most recent first (same
        // "search backwards" bias as PLD -- a nearer repeat is more likely
        // relevant than a far one). i + n <= needle_start - 1 + 1, i.e. the
        // loop bound below (hsize - n - 1) both keeps the match strictly
        // before the needle's own position (no trivial self-match) and
        // guarantees at least one continuation token exists after it.
        for (int i = needle_start - 1; i >= 0; --i) {
            bool match = true;
            for (int j = 0; j < n; ++j) {
                if (haystack[i + j] != haystack[needle_start + j]) {
                    match = false;
                    break;
                }
            }
            if (!match) continue;

            const int draft_start = i + n;
            const int draft_len = std::min(config_.max_draft, hsize - draft_start);
            if (draft_len < 1) continue;  // no continuation tokens available

            return std::vector<int32_t>(
                haystack.begin() + draft_start,
                haystack.begin() + draft_start + draft_len);
        }
    }

    return {};
}

} // namespace qinf
