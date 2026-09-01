#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace qinf {

// ============================================================================
// SuffixDecoding: session-scoped, adaptive-length draft proposal
//
// PLD (prompt_lookup.h) matches a FIXED n-gram against the PROMPT only. Two
// measured limits on an offline replay of 110 order-management-DSL sessions
// motivated this source: (1) once generation runs past the prompt, PLD's
// haystack goes stale -- repeated structure in what the model already
// produced (a common shape in DSL-style output) is invisible to it; (2) a
// fixed n-gram size finds a match only ~1 step in 4 (worse on some model
// families) -- draft AVAILABILITY, not quality, is what caps PLD.
//
// SuffixDecoding fixes both by generalizing the same n-gram-match idea:
//   - haystack = prompt_tokens ++ generated_tokens (the whole session so far,
//     not prompt-only) -- session-scoped, see the cap below.
//   - match length is ADAPTIVE: try the longest n-gram first (more context =
//     more confident continuation), shrinking toward a floor if nothing
//     matches, instead of PLD's single fixed n.
// Everything else -- verify, accept/reject, KV rewind -- is unchanged; this
// class only answers "what comes next", same contract as PromptLookup.
//
// Session-scoped only (load-bearing, not a placeholder): the replay also
// measured a cross-session persistent index at 2.24x vs this class's 1.87x.
// That extra 0.37x was judged not worth a persistent store for v1 -- it is a
// documented future option, not an oversight. Do not add cross-session
// persistence to this class; a persistent variant belongs in a new class
// (constructing it would change what "session-scoped" means for every
// existing caller).
// ============================================================================

struct SuffixDecodingConfig {
    // Adaptive match length: try the longest n-gram match against the
    // session's own history first, shrinking toward min_match_len until one
    // is found. Values are the measured setting (try n=12 down to 2) -- a
    // longer match is a more confident continuation, so the search always
    // prefers it over a shorter one.
    int max_match_len = 12;
    int min_match_len = 2;

    // Draft width B: continuation tokens proposed once a match is found.
    // Measured sweet spot on Gemma 4-12B: verify costs 1.41x a single step
    // at B=4 but 2.64x at B=8 -- wider is WORSE here (opposite of the CUDA
    // batch intuition). Do not raise this default.
    int max_draft = 4;

    // Bound on how many of the most recent session tokens (prompt tail +
    // all generated tokens) are searched. Session output grows without
    // limit; this cap keeps find_draft's cost O(max_indexed_tokens) on every
    // decode step, independent of how long the session has run.
    //
    // Chosen as 8192: the workload envelope (CLAUDE.md) ceilings a session's
    // live context at 10K tokens, so a session's prompt+output cannot
    // usefully exceed that inside one context window in the first place;
    // 8192 sits under that ceiling (headroom for the case a longer session
    // spans a context reset / warm-cache handoff and the tail end of the
    // index is what's still relevant) while keeping the per-step scan and
    // the index memory (8192 * 4 bytes = 32 KiB) trivial regardless of
    // --ctx-size. It is a fixed constant, not derived from --ctx-size, so it
    // stays meaningful if --ctx-size is raised or lowered independently.
    std::size_t max_indexed_tokens = 8192;
};

class SuffixDecoding {
public:
    explicit SuffixDecoding(SuffixDecodingConfig config = {});

    // Search the session's own token history for a repeated continuation of
    // what was just generated.
    //
    // prompt_tokens:    original input tokens (part of the haystack)
    // generated_tokens: everything sampled so far this session (the needle's
    //                   tail is drawn from here, same as PLD -- there is
    //                   nothing to match against before generation starts)
    //
    // Tries the needle at length max_match_len first, then max_match_len-1,
    // ... down to min_match_len, and returns the draft from the first
    // (longest, therefore most confident) match found. The haystack searched
    // is capped to the most recent config().max_indexed_tokens tokens of
    // prompt_tokens ++ generated_tokens.
    //
    // Returns: up to config().max_draft continuation tokens, or empty if no
    // match at any length in [min_match_len, max_match_len] was found (the
    // "no draft" contract IDraftSource::propose expects).
    std::vector<int32_t> find_draft(
        const std::vector<int32_t>& prompt_tokens,
        const std::vector<int32_t>& generated_tokens) const;

    const SuffixDecodingConfig& config() const { return config_; }

private:
    SuffixDecodingConfig config_;
};

} // namespace qinf
