#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

#include "prompt_lookup.h"
#include "speculative.h"

using namespace qwenium;

// ============================================================================
// Test helpers
// ============================================================================

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) \
    do { \
        tests_total++; \
        std::cout << "  " << name << "... " << std::flush; \
    } while(0)

#define PASS() \
    do { \
        tests_passed++; \
        std::cout << "PASS" << std::endl; \
    } while(0)

#define FAIL(msg) \
    do { \
        std::cout << "FAIL: " << msg << std::endl; \
    } while(0)

// ============================================================================
// PromptLookup tests
// ============================================================================

void test_lookup_basic_match() {
    TEST("Basic n-gram match");

    PromptLookup lookup({.ngram_size = 3, .max_draft = 5});

    // Prompt: ... 10 20 30 40 50 60 ...
    std::vector<int32_t> prompt = {1, 2, 3, 10, 20, 30, 40, 50, 60, 7, 8, 9};

    // Generated ends with: ... 10 20 30  (matches prompt[3..5])
    std::vector<int32_t> generated = {99, 88, 10, 20, 30};

    auto draft = lookup.find_draft(prompt, generated);

    // Should propose: 40, 50, 60, 7, 8 (next 5 after match)
    assert(draft.size() == 5);
    assert(draft[0] == 40);
    assert(draft[1] == 50);
    assert(draft[2] == 60);
    assert(draft[3] == 7);
    assert(draft[4] == 8);
    PASS();
}

void test_lookup_no_match() {
    TEST("No n-gram match");

    PromptLookup lookup({.ngram_size = 3, .max_draft = 5});

    std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    std::vector<int32_t> generated = {99, 88, 77};

    auto draft = lookup.find_draft(prompt, generated);
    assert(draft.empty());
    PASS();
}

void test_lookup_match_at_end() {
    TEST("Match near end of prompt (short draft)");

    PromptLookup lookup({.ngram_size = 2, .max_draft = 10});

    // Match at position 3, only 1 token after
    std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    std::vector<int32_t> generated = {99, 3, 4};

    auto draft = lookup.find_draft(prompt, generated);
    assert(draft.size() == 1);
    assert(draft[0] == 5);
    PASS();
}

void test_lookup_match_at_very_end() {
    TEST("Match at very end of prompt (no tokens after)");

    PromptLookup lookup({.ngram_size = 2, .max_draft = 5, .min_draft = 1});

    std::vector<int32_t> prompt = {1, 2, 3, 4};
    std::vector<int32_t> generated = {99, 3, 4};

    auto draft = lookup.find_draft(prompt, generated);
    assert(draft.empty());  // No tokens after the match
    PASS();
}

void test_lookup_too_few_generated() {
    TEST("Too few generated tokens for n-gram");

    PromptLookup lookup({.ngram_size = 3, .max_draft = 5});

    std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    std::vector<int32_t> generated = {1, 2};  // Only 2, need 3

    auto draft = lookup.find_draft(prompt, generated);
    assert(draft.empty());
    PASS();
}

void test_lookup_multiple_matches_takes_last() {
    TEST("Multiple matches — takes last (most recent)");

    PromptLookup lookup({.ngram_size = 2, .max_draft = 3});

    // Pattern "10 20" appears twice, at positions 1 and 5
    std::vector<int32_t> prompt = {1, 10, 20, 30, 40, 10, 20, 50, 60};
    std::vector<int32_t> generated = {99, 10, 20};

    auto draft = lookup.find_draft(prompt, generated);
    // Should match the LATER occurrence (position 5) → draft = [50, 60]
    assert(draft.size() == 2);
    assert(draft[0] == 50);
    assert(draft[1] == 60);
    PASS();
}

// ============================================================================
// SpeculativeDecoder tests (with mock forward pass)
// ============================================================================

void test_speculative_all_accepted() {
    TEST("Speculative: all draft tokens accepted");

    const int VOCAB = 100;
    SpeculativeDecoder spec({.ngram_size = 2, .max_draft = 3}, VOCAB);

    // Prompt: ... 10 20 30 40 50 ...
    std::vector<int32_t> prompt = {1, 2, 10, 20, 30, 40, 50, 60};
    // Generated ends with: 10 20 → should draft [30, 40, 50]
    std::vector<int32_t> generated = {99, 10, 20};

    int rewind_called_with = -1;

    // Mock verify: return logits where argmax matches draft[i+1]
    // draft = [30, 40, 50]
    // logits[0] should predict 40 (draft[1])
    // logits[1] should predict 50 (draft[2])
    // logits[2] should predict 77 (bonus token after draft)
    auto verify = [&](int slot_id, const std::vector<int32_t>& draft, int pos) {
        std::vector<float> logits(draft.size() * VOCAB, 0.0f);
        // Position 0 → predict 40
        logits[0 * VOCAB + 40] = 10.0f;
        // Position 1 → predict 50
        logits[1 * VOCAB + 50] = 10.0f;
        // Position 2 → predict 77 (bonus)
        logits[2 * VOCAB + 77] = 10.0f;
        return logits;
    };

    auto rewind = [&](int slot_id, int pos) {
        rewind_called_with = pos;
    };

    auto result = spec.try_speculative_step(
        prompt, generated, /*slot_id=*/0, /*current_pos=*/3,
        verify, rewind, /*eos=*/-1);

    assert(result.attempted());
    assert(result.draft_length == 3);
    assert(result.accepted_tokens.size() == 3);
    assert(result.accepted_tokens[0] == 30);
    assert(result.accepted_tokens[1] == 40);
    assert(result.accepted_tokens[2] == 50);
    assert(result.has_bonus);
    assert(result.bonus_token == 77);
    assert(result.total_tokens() == 4);
    // All accepted + bonus = K+1 tokens kept, no rewind needed
    assert(rewind_called_with == -1);
    PASS();
}

void test_speculative_partial_accept() {
    TEST("Speculative: partial acceptance (reject at position 1)");

    const int VOCAB = 100;
    SpeculativeDecoder spec({.ngram_size = 2, .max_draft = 3}, VOCAB);

    std::vector<int32_t> prompt = {1, 2, 10, 20, 30, 40, 50, 60};
    std::vector<int32_t> generated = {99, 10, 20};
    // Draft will be [30, 40, 50]

    int rewind_pos = -1;

    auto verify = [&](int slot_id, const std::vector<int32_t>& draft, int pos) {
        std::vector<float> logits(draft.size() * VOCAB, 0.0f);
        // Position 0 → predict 40 (correct, matches draft[1])
        logits[0 * VOCAB + 40] = 10.0f;
        // Position 1 → predict 99 (WRONG, draft[2] is 50)
        logits[1 * VOCAB + 99] = 10.0f;
        // Position 2 → doesn't matter, won't be checked
        logits[2 * VOCAB + 60] = 10.0f;
        return logits;
    };

    auto rewind = [&](int slot_id, int pos) {
        rewind_pos = pos;
    };

    auto result = spec.try_speculative_step(
        prompt, generated, 0, /*current_pos=*/3,
        verify, rewind, -1);

    assert(result.attempted());
    assert(result.accepted_tokens.size() == 2);  // draft[0]=30 and draft[1]=40
    assert(result.accepted_tokens[0] == 30);
    assert(result.accepted_tokens[1] == 40);
    assert(result.has_bonus);
    assert(result.bonus_token == 99);  // The correct token at rejection point
    assert(result.total_tokens() == 3);
    // Rewind: current_pos(3) + accepted(2) + bonus(1) = 6
    // Cache was advanced by 3 (draft size), so rewind from 6 to 6... 
    // Actually: keep = 2 + 1 = 3 = K, no rewind needed
    // Wait: keep=3, K=3, so no rewind
    assert(rewind_pos == -1);
    PASS();
}

void test_speculative_reject_first() {
    TEST("Speculative: reject at first position");

    const int VOCAB = 100;
    SpeculativeDecoder spec({.ngram_size = 2, .max_draft = 3}, VOCAB);

    std::vector<int32_t> prompt = {1, 2, 10, 20, 30, 40, 50, 60};
    std::vector<int32_t> generated = {99, 10, 20};
    // Draft = [30, 40, 50]

    int rewind_pos = -1;

    auto verify = [&](int slot_id, const std::vector<int32_t>& draft, int pos) {
        std::vector<float> logits(draft.size() * VOCAB, 0.0f);
        // Position 0 → predict 88 (WRONG, draft[1] is 40)
        logits[0 * VOCAB + 88] = 10.0f;
        logits[1 * VOCAB + 50] = 10.0f;
        logits[2 * VOCAB + 60] = 10.0f;
        return logits;
    };

    auto rewind = [&](int slot_id, int pos) {
        rewind_pos = pos;
    };

    auto result = spec.try_speculative_step(
        prompt, generated, 0, 3, verify, rewind, -1);

    assert(result.attempted());
    assert(result.accepted_tokens.size() == 1);  // Only draft[0]=30
    assert(result.accepted_tokens[0] == 30);
    assert(result.has_bonus);
    assert(result.bonus_token == 88);
    assert(result.total_tokens() == 2);
    // keep = 1 + 1 = 2, K = 3, need rewind
    assert(rewind_pos == 3 + 2);  // current_pos + keep
    PASS();
}

void test_speculative_no_draft_found() {
    TEST("Speculative: no n-gram match");

    const int VOCAB = 100;
    SpeculativeDecoder spec({.ngram_size = 3, .max_draft = 5}, VOCAB);

    std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    std::vector<int32_t> generated = {99, 88, 77};

    bool verify_called = false;
    auto verify = [&](int, const std::vector<int32_t>&, int) {
        verify_called = true;
        return std::vector<float>();
    };
    auto rewind = [](int, int) {};

    auto result = spec.try_speculative_step(
        prompt, generated, 0, 3, verify, rewind, -1);

    assert(!result.attempted());
    assert(result.accepted_tokens.empty());
    assert(!result.has_bonus);
    assert(!verify_called);  // Should not call forward pass
    PASS();
}

void test_speculative_eos_in_draft() {
    TEST("Speculative: EOS in draft stops acceptance");

    const int VOCAB = 100;
    const int EOS = 42;
    SpeculativeDecoder spec({.ngram_size = 2, .max_draft = 4}, VOCAB);

    // Prompt has EOS token in the middle
    std::vector<int32_t> prompt = {1, 2, 10, 20, 30, EOS, 50, 60};
    std::vector<int32_t> generated = {99, 10, 20};
    // Draft = [30, 42(EOS), 50, 60]

    int rewind_pos = -1;

    auto verify = [&](int slot_id, const std::vector<int32_t>& draft, int pos) {
        std::vector<float> logits(draft.size() * VOCAB, 0.0f);
        // Position 0 → predict EOS (correct, matches draft[1])
        logits[0 * VOCAB + EOS] = 10.0f;
        logits[1 * VOCAB + 50] = 10.0f;
        logits[2 * VOCAB + 60] = 10.0f;
        logits[3 * VOCAB + 70] = 10.0f;
        return logits;
    };

    auto rewind = [&](int slot_id, int pos) {
        rewind_pos = pos;
    };

    auto result = spec.try_speculative_step(
        prompt, generated, 0, 3, verify, rewind, EOS);

    assert(result.attempted());
    assert(result.accepted_tokens.size() == 1);  // Only 30 accepted, then EOS stops
    assert(result.accepted_tokens[0] == 30);
    assert(!result.has_bonus);  // EOS is not a bonus
    assert(result.bonus_token == EOS);
    // keep = 1 + 0 = 1, K = 4, rewind
    assert(rewind_pos == 3 + 1);
    PASS();
}

void test_speculative_stats() {
    TEST("Speculative: stats tracking");

    const int VOCAB = 100;
    SpeculativeDecoder spec({.ngram_size = 2, .max_draft = 3}, VOCAB);

    std::vector<int32_t> prompt = {1, 2, 10, 20, 30, 40, 50};
    std::vector<int32_t> generated = {99, 10, 20};

    auto verify = [&](int, const std::vector<int32_t>& draft, int) {
        std::vector<float> logits(draft.size() * VOCAB, 0.0f);
        logits[0 * VOCAB + 40] = 10.0f;
        logits[1 * VOCAB + 50] = 10.0f;
        logits[2 * VOCAB + 77] = 10.0f;
        return logits;
    };
    auto rewind = [](int, int) {};

    spec.try_speculative_step(prompt, generated, 0, 3, verify, rewind, -1);

    auto& s = spec.stats();
    assert(s.drafts_attempted == 1);
    assert(s.drafts_found == 1);
    assert(s.tokens_drafted == 3);
    assert(s.tokens_accepted == 3);
    assert(s.bonus_tokens == 1);
    assert(s.acceptance_rate() == 1.0);
    PASS();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n=== PromptLookup Tests ===\n" << std::endl;
    test_lookup_basic_match();
    test_lookup_no_match();
    test_lookup_match_at_end();
    test_lookup_match_at_very_end();
    test_lookup_too_few_generated();
    test_lookup_multiple_matches_takes_last();

    std::cout << "\n=== SpeculativeDecoder Tests ===\n" << std::endl;
    test_speculative_all_accepted();
    test_speculative_partial_accept();
    test_speculative_reject_first();
    test_speculative_no_draft_found();
    test_speculative_eos_in_draft();
    test_speculative_stats();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_total
              << " passed ===" << std::endl;

    return (tests_passed == tests_total) ? 0 : 1;
}