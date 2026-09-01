#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

#include "suffix_decoding.h"
#include "draft_source.h"

using namespace qinf;

// ============================================================================
// Lightweight harness (matches test_speculative.cpp / test_draft_source.cpp:
// own main(), no gtest).
// ============================================================================
static int tests_passed = 0;
static int tests_total  = 0;
#define TEST(name) do { tests_total++; \
    std::cout << "  " << name << "... " << std::flush; } while(0)
#define PASS() do { tests_passed++; std::cout << "PASS" << std::endl; } while(0)

// All tests here are model-free: SuffixDecoding is a pure data structure over
// token vectors, so nothing loads a GGUF or a Tokenizer.

// ============================================================================
// Adaptive match length: try the longest n-gram first.
// ============================================================================
void test_longest_match_preferred() {
    TEST("Adaptive length: n=5 match found and preferred");

    SuffixDecodingConfig cfg;
    cfg.max_match_len = 5;
    cfg.min_match_len = 2;
    cfg.max_draft = 2;

    // segment1 (A B C D E -> X Y) at idx 0..6, filler at 7..9, a SHORTER
    // (D E -> Z) repeat at idx 10..12, then the tail A B C D E at 13..17 —
    // exactly reproducing segment1. The full 5-token tail only matches
    // segment1: the algorithm must find it at n=5 and must never even
    // consider the more recent but shorter n=2 match at segment "D E".
    std::vector<int32_t> generated = {
        1000, 1001, 1002, 1003, 1004, 2000, 2001,   // segment1: A B C D E X Y
        9000, 9001, 9002,                            // filler
        1003, 1004, 3000,                            // shorter repeat: D E Z
        1000, 1001, 1002, 1003, 1004,                // tail: A B C D E
    };
    std::vector<int32_t> prompt = {};

    SuffixDecoding idx(cfg);
    auto draft = idx.find_draft(prompt, generated);

    assert(draft == (std::vector<int32_t>{2000, 2001}));
    PASS();
}

void test_adaptive_falls_back_to_shorter() {
    TEST("Adaptive length: falls back n=5..3 -> n=2, most recent wins");

    SuffixDecodingConfig cfg;
    cfg.max_match_len = 5;
    cfg.min_match_len = 2;
    cfg.max_draft = 2;

    // idx 0..4 ends "... D(3) E(4)"; idx 7..8 is a SECOND, more recent
    // "D E" (=1003,1004) before the tail at idx 10..14. The tail's own
    // first 3 tokens (8000,8001,8002) are unique to the tail, so n=5, n=4,
    // n=3 all fail to match anything -- the search must fall through to
    // n=2, and among the two n=2 matches of "D E" (idx 3..4 and idx 7..8)
    // it must pick the more recent one (idx 7..8), same "search backwards"
    // recency bias as PLD.
    std::vector<int32_t> generated = {
        1000, 1001, 1002, 1003, 1004,   // idx 0..4: ... D(3) E(4)
        2000, 2001,                      // idx 5..6: filler
        1003, 1004, 3000,                // idx 7..9: D(7) E(8) Z(9) -- more recent n=2 match
        8000, 8001, 8002, 1003, 1004,    // idx 10..14: tail, ends D(13) E(14)
    };
    std::vector<int32_t> prompt = {};

    SuffixDecoding idx(cfg);
    auto draft = idx.find_draft(prompt, generated);

    // Most recent n=2 match of "D E" before the tail is at idx 7,8;
    // continuation = generated[9..10] = {3000, 8000}, capped to max_draft=2.
    assert(draft == (std::vector<int32_t>{3000, 8000}));
    PASS();
}

// ============================================================================
// Session-scoped: haystack is prompt + generated, unlike PLD's prompt-only.
// ============================================================================
void test_session_scoped_matches_generated_region() {
    TEST("Session-scoped: match found entirely within generated tokens");

    SuffixDecodingConfig cfg;
    cfg.max_match_len = 2;
    cfg.min_match_len = 2;
    cfg.max_draft = 2;

    // Empty prompt: PLD (prompt-only haystack) would have nothing to search
    // and could never find this. The repeat is entirely within what the
    // model has generated so far; SuffixDecoding's haystack includes it.
    std::vector<int32_t> prompt = {};
    std::vector<int32_t> generated = {
        42, 43,        // idx 0..1: earlier "A B" -> continuation 100, 101
        100, 101,       // idx 2..3: continuation
        42, 43,         // idx 4..5: tail repeats idx 0..1 exactly
    };

    SuffixDecoding idx(cfg);
    auto draft = idx.find_draft(prompt, generated);

    assert(draft == (std::vector<int32_t>{100, 101}));
    PASS();
}

// ============================================================================
// The bound: max_indexed_tokens caps the searched haystack.
// ============================================================================
void test_bound_excludes_old_tokens() {
    TEST("Bound: a match older than max_indexed_tokens is invisible");

    SuffixDecodingConfig cfg;
    cfg.max_match_len = 3;
    cfg.min_match_len = 2;
    cfg.max_draft = 2;
    cfg.max_indexed_tokens = 6;  // small bound for the test

    // The only match for the tail lives at the very front of `generated`,
    // 12 tokens back -- well outside a 6-token bound. With the bound
    // in effect no draft should be found.
    std::vector<int32_t> prompt = {};
    std::vector<int32_t> generated = {
        7, 8, 9, 100, 101,        // idx 0..4: the only earlier occurrence + continuation
        1, 2, 3, 4, 5, 6,          // idx 5..10: filler, pushes the match out of the window
        7, 8, 9,                   // idx 11..13: tail repeats idx 0..2
    };

    SuffixDecoding idx(cfg);
    auto draft = idx.find_draft(prompt, generated);
    assert(draft.empty());
    PASS();
}

void test_bound_includes_recent_match() {
    TEST("Bound: a match inside max_indexed_tokens is still found");

    SuffixDecodingConfig cfg;
    cfg.max_match_len = 3;
    cfg.min_match_len = 2;
    cfg.max_draft = 2;
    cfg.max_indexed_tokens = 100;  // generous bound: nothing excluded

    std::vector<int32_t> prompt = {};
    std::vector<int32_t> generated = {
        7, 8, 9, 100, 101,
        1, 2, 3, 4, 5, 6,
        7, 8, 9,
    };

    SuffixDecoding idx(cfg);
    auto draft = idx.find_draft(prompt, generated);
    assert(draft == (std::vector<int32_t>{100, 101}));
    PASS();
}

// ============================================================================
// Draft length is capped by what's actually available after the match, even
// under max_draft.
// ============================================================================
void test_draft_capped_by_availability() {
    TEST("Draft length capped by tokens available after the match");

    SuffixDecodingConfig cfg;
    cfg.max_match_len = 3;
    cfg.min_match_len = 2;
    cfg.max_draft = 5;  // ask for 5, but only 4 tokens exist after the match

    std::vector<int32_t> prompt = {};
    std::vector<int32_t> generated = {
        1, 2, 3,   // idx 0..2: earlier "1 2 3"
        99,         // idx 3: one filler token before the tail repeats it
        1, 2, 3,    // idx 4..6: tail repeats idx 0..2
    };
    // Match at i=0 (n=3): draft_start=3, and only 4 tokens (idx 3..6) exist
    // after it in a 7-token haystack -- less than max_draft=5, so the min()
    // cap in find_draft is what determines the returned length here.

    SuffixDecoding idx(cfg);
    auto draft = idx.find_draft(prompt, generated);
    assert(draft == (std::vector<int32_t>{99, 1, 2, 3}));
    PASS();
}

// ============================================================================
// Not enough generated tokens yet to even try the minimum n-gram.
// ============================================================================
void test_insufficient_generated_tokens() {
    TEST("Not enough generated tokens yet: no draft, regardless of prompt");

    SuffixDecodingConfig cfg;
    cfg.min_match_len = 2;

    // Prompt is full of repeats, but only 1 token has been generated --
    // below min_match_len -- so there is no meaningful needle yet.
    std::vector<int32_t> prompt    = {1, 2, 1, 2, 1, 2, 1, 2};
    std::vector<int32_t> generated = {1};

    SuffixDecoding idx(cfg);
    auto draft = idx.find_draft(prompt, generated);
    assert(draft.empty());
    PASS();
}

void test_no_match_anywhere() {
    TEST("No match at any length: empty draft");

    SuffixDecodingConfig cfg;
    cfg.max_match_len = 4;
    cfg.min_match_len = 2;

    std::vector<int32_t> prompt    = {1, 2, 3};
    std::vector<int32_t> generated = {10, 20, 30, 40, 50};  // all unique, no repeats

    SuffixDecoding idx(cfg);
    auto draft = idx.find_draft(prompt, generated);
    assert(draft.empty());
    PASS();
}

// ============================================================================
// Fail-loud construction guards.
// ============================================================================
template <typename Fn>
static bool throws_runtime_error(Fn&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void test_fail_loud_config_guards() {
    TEST("Fail-loud: invalid config throws std::runtime_error");

    assert(throws_runtime_error([] {
        SuffixDecodingConfig cfg; cfg.min_match_len = 0;
        SuffixDecoding idx(cfg);
    }));  // min_match_len < 1

    assert(throws_runtime_error([] {
        SuffixDecodingConfig cfg; cfg.max_match_len = 1; cfg.min_match_len = 2;
        SuffixDecoding idx(cfg);
    }));  // max_match_len < min_match_len

    assert(throws_runtime_error([] {
        SuffixDecodingConfig cfg; cfg.max_draft = 0;
        SuffixDecoding idx(cfg);
    }));  // max_draft < 1

    assert(throws_runtime_error([] {
        SuffixDecodingConfig cfg; cfg.max_match_len = 12; cfg.max_indexed_tokens = 4;
        SuffixDecoding idx(cfg);
    }));  // max_indexed_tokens < max_match_len

    // A valid config must NOT throw.
    bool valid_threw = throws_runtime_error([] {
        SuffixDecoding idx(SuffixDecodingConfig{});
    });
    assert(!valid_threw);
    PASS();
}

// ============================================================================
// Seam integration: SuffixDecodingDraft delegates to SuffixDecoding through
// IDraftSource, same as PromptLookupDraft does for PromptLookup
// (test_pld_through_seam_parity in test_draft_source.cpp is the precedent).
// ============================================================================
void test_seam_delegates_and_ignores_hidden_state() {
    TEST("SuffixDecodingDraft: delegates to SuffixDecoding, needs_hidden_state() == false");

    SuffixDecodingConfig cfg;
    cfg.max_match_len = 4;
    cfg.min_match_len = 2;
    cfg.max_draft = 3;

    SuffixDecoding    raw(cfg);
    SuffixDecodingDraft seam(cfg);
    assert(seam.needs_hidden_state() == false);

    std::vector<int32_t> prompt    = {};
    std::vector<int32_t> generated = {42, 43, 44, 45, 100, 101, 42, 43, 44, 45};
    std::vector<float>   no_hidden;

    auto raw_draft  = raw.find_draft(prompt, generated);
    auto seam_draft = seam.propose(
        DraftContext{prompt, generated, no_hidden, /*slot=*/0, /*pos=*/10});

    assert(seam_draft == raw_draft);
    assert(!seam_draft.empty());   // guard: the fixture actually matches
    PASS();
}

int main() {
    std::cout << "\n=== SuffixDecoding: adaptive length ===\n" << std::endl;
    test_longest_match_preferred();
    test_adaptive_falls_back_to_shorter();

    std::cout << "\n=== SuffixDecoding: session-scoped haystack ===\n" << std::endl;
    test_session_scoped_matches_generated_region();

    std::cout << "\n=== SuffixDecoding: the bound ===\n" << std::endl;
    test_bound_excludes_old_tokens();
    test_bound_includes_recent_match();

    std::cout << "\n=== SuffixDecoding: draft length + edge cases ===\n" << std::endl;
    test_draft_capped_by_availability();
    test_insufficient_generated_tokens();
    test_no_match_anywhere();

    std::cout << "\n=== SuffixDecoding: fail-loud ===\n" << std::endl;
    test_fail_loud_config_guards();

    std::cout << "\n=== SuffixDecodingDraft: seam integration ===\n" << std::endl;
    test_seam_delegates_and_ignores_hidden_state();

    std::cout << "\n" << tests_passed << "/" << tests_total << " passed\n" << std::endl;
    return tests_passed == tests_total ? 0 : 1;
}
