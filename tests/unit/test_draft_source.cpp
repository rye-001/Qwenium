#include <iostream>
#include <vector>
#include <cassert>
#include <memory>

#include "draft_source.h"
#include "speculative.h"

using namespace qwenium;

// ============================================================================
// Lightweight harness (matches test_speculative.cpp — own main(), no gtest).
// ============================================================================
static int tests_passed = 0;
static int tests_total  = 0;
#define TEST(name) do { tests_total++; \
    std::cout << "  " << name << "... " << std::flush; } while(0)
#define PASS() do { tests_passed++; std::cout << "PASS" << std::endl; } while(0)

// ============================================================================
// A fake draft source: the second implementation of the seam in Phase 1 (the
// first is PromptLookupDraft, the real one is MtpDraft in Phase 4). It hands
// back a scripted draft and records what context it was given — enough to
// prove SpeculativeDecoder drives an arbitrary source, not just PLD.
// ============================================================================
class FakeDraftSource : public IDraftSource {
public:
    explicit FakeDraftSource(std::vector<int32_t> draft, bool needs_hidden = false)
        : draft_(std::move(draft)), needs_hidden_(needs_hidden) {}

    std::vector<int32_t> propose(const DraftContext& ctx) override {
        last_slot_        = ctx.slot;
        last_pos_         = ctx.current_pos;
        last_hidden_size_ = ctx.last_hidden.size();
        propose_calls_++;
        return draft_;
    }
    bool needs_hidden_state() const override { return needs_hidden_; }

    uint32_t last_slot_        = 0;
    int      last_pos_         = -1;
    size_t   last_hidden_size_ = 0;
    int      propose_calls_    = 0;

private:
    std::vector<int32_t> draft_;
    bool needs_hidden_;
};

// Verify mock: for a draft of length K, return [K*VOCAB] logits whose argmax at
// each position i is `predicts[i]` (predicts must have length K).
static SpeculativeDecoder::VerifyFunc make_verify(
    int vocab, const std::vector<int32_t>& predicts, int* rewind_seen = nullptr) {
    (void)rewind_seen;
    return [vocab, predicts](int, const std::vector<int32_t>& draft, int) {
        std::vector<float> logits(draft.size() * vocab, 0.0f);
        for (size_t i = 0; i < draft.size(); ++i)
            logits[i * vocab + predicts[i]] = 10.0f;
        return logits;
    };
}

// ============================================================================
// Tests: SpeculativeDecoder driven by an injected IDraftSource.
// ============================================================================

void test_seam_all_accepted() {
    TEST("Seam: fake source, all draft tokens accepted + bonus");
    const int VOCAB = 100;

    auto fake = std::make_unique<FakeDraftSource>(std::vector<int32_t>{30, 40, 50});
    FakeDraftSource* fake_raw = fake.get();
    SpeculativeDecoder spec(std::move(fake), VOCAB);

    std::vector<int32_t> prompt    = {1, 2, 10, 20};
    std::vector<int32_t> generated = {10, 20};

    // logits[0]->40 (=draft[1]), logits[1]->50 (=draft[2]), logits[2]->77 (bonus)
    auto verify = make_verify(VOCAB, {40, 50, 77});
    int rewind_called_with = -1;
    auto rewind = [&](int, int pos) { rewind_called_with = pos; };

    auto result = spec.try_speculative_step(
        prompt, generated, /*slot_id=*/0, /*current_pos=*/3, verify, rewind, -1);

    assert(result.attempted());
    assert(result.accepted_tokens.size() == 3);
    assert(result.accepted_tokens == (std::vector<int32_t>{30, 40, 50}));
    assert(result.has_bonus && result.bonus_token == 77);
    assert(result.total_tokens() == 4);
    assert(rewind_called_with == -1);           // nothing to discard
    assert(fake_raw->propose_calls_ == 1);
    assert(fake_raw->last_slot_ == 0);
    assert(fake_raw->last_pos_ == 3);           // context carried current_pos
    assert(fake_raw->last_hidden_size_ == 0);   // PLD-style: no hidden passed
    PASS();
}

void test_seam_partial_accept_rewinds() {
    TEST("Seam: partial acceptance rewinds cache to kept");
    const int VOCAB = 100;

    auto fake = std::make_unique<FakeDraftSource>(std::vector<int32_t>{30, 40, 50});
    SpeculativeDecoder spec(std::move(fake), VOCAB);

    std::vector<int32_t> prompt    = {1, 2, 10, 20};
    std::vector<int32_t> generated = {10, 20};

    // pos0->99 (WRONG, draft[1]=40) → reject at position 0. Positions 1,2 are
    // discarded, so keep < K and the cache must rewind.
    auto verify = make_verify(VOCAB, {99, 40, 50});
    int rewind_pos = -1;
    auto rewind = [&](int, int pos) { rewind_pos = pos; };

    auto result = spec.try_speculative_step(
        prompt, generated, /*slot_id=*/0, /*current_pos=*/3, verify, rewind, -1);

    // accepted = draft[0] = {30}; bonus = 99 (no KV entry — fed next step);
    // keep = accepted = 1 (< K=3) → rewind to current_pos + 1.
    assert(result.accepted_tokens == (std::vector<int32_t>{30}));
    assert(result.has_bonus && result.bonus_token == 99);
    assert(rewind_pos == 3 + 1);   // current_pos + accepted
    PASS();
}

void test_seam_needs_hidden_and_ctx_passthrough() {
    TEST("Seam: needs_hidden_state exposed + last_hidden carried to source");
    const int VOCAB = 100;

    auto fake = std::make_unique<FakeDraftSource>(std::vector<int32_t>{7},
                                                  /*needs_hidden=*/true);
    FakeDraftSource* fake_raw = fake.get();
    SpeculativeDecoder spec(std::move(fake), VOCAB);
    assert(spec.needs_hidden_state() == true);   // surfaced from the source

    std::vector<int32_t> prompt    = {1};
    std::vector<int32_t> generated = {1};
    std::vector<float>   hidden(2048, 0.5f);      // stand-in for a D3 output
    auto verify = make_verify(VOCAB, {77});       // K=1: draft accepted, 77=bonus
    auto rewind = [&](int, int) {};

    auto result = spec.try_speculative_step(
        prompt, generated, /*slot_id=*/2, /*current_pos=*/9, verify, rewind, -1,
        hidden);

    assert(result.accepted_tokens == (std::vector<int32_t>{7}));
    assert(result.has_bonus && result.bonus_token == 77);
    assert(fake_raw->last_slot_ == 2);
    assert(fake_raw->last_hidden_size_ == 2048);  // hidden reached the source
    PASS();
}

void test_seam_empty_draft_falls_back() {
    TEST("Seam: empty draft = no attempt, counts as normal decode");
    const int VOCAB = 100;

    auto fake = std::make_unique<FakeDraftSource>(std::vector<int32_t>{});  // empty
    SpeculativeDecoder spec(std::move(fake), VOCAB);

    std::vector<int32_t> prompt    = {1, 2, 3};
    std::vector<int32_t> generated = {1, 2, 3};
    bool verify_called = false;
    auto verify = [&](int, const std::vector<int32_t>&, int) {
        verify_called = true; return std::vector<float>{}; };
    auto rewind = [&](int, int) {};

    auto result = spec.try_speculative_step(
        prompt, generated, 0, 3, verify, rewind, -1);

    assert(!result.attempted());
    assert(!verify_called);                       // no draft ⇒ no forward pass
    assert(spec.stats().normal_decodes == 1);
    PASS();
}

// ============================================================================
// PLD-through-the-seam parity: PromptLookupDraft must yield exactly the draft
// the underlying PromptLookup does (the Phase 1 extraction gate, at unit level).
// ============================================================================
void test_pld_through_seam_parity() {
    TEST("PromptLookupDraft == raw PromptLookup (byte-identical draft)");

    PromptLookupConfig cfg{.ngram_size = 3, .max_draft = 5};
    PromptLookup       raw(cfg);
    PromptLookupDraft  seam(cfg);
    assert(seam.needs_hidden_state() == false);

    std::vector<int32_t> prompt    = {1, 2, 3, 10, 20, 30, 40, 50, 60, 7, 8, 9};
    std::vector<int32_t> generated = {99, 88, 10, 20, 30};
    std::vector<float>   no_hidden;

    auto raw_draft = raw.find_draft(prompt, generated);
    auto seam_draft = seam.propose(
        DraftContext{prompt, generated, no_hidden, /*slot=*/0, /*pos=*/5});

    assert(seam_draft == raw_draft);
    assert(!seam_draft.empty());   // guard: the fixture actually matches
    PASS();
}

// ============================================================================
// MtpDraft (Phase 4): the head behind a std::function bridge.
// ============================================================================
void test_mtp_draft_shape_and_seeding() {
    TEST("MtpDraft: [next_token, head tokens...], bridge seeded correctly");

    uint32_t seen_slot = 99; int32_t seen_tok = -1; int seen_pos = -1;
    uint32_t seen_k = 0; size_t seen_hidden = 0;
    MtpDraft mtp(
        [&](uint32_t slot, const std::vector<float>& h, int32_t t, int p,
            uint32_t k) {
            seen_slot = slot; seen_hidden = h.size(); seen_tok = t;
            seen_pos = p; seen_k = k;
            return std::vector<int32_t>{7, 8};
        },
        /*max_head_draft=*/2);
    assert(mtp.needs_hidden_state());

    std::vector<int32_t> prompt = {1}, generated = {1, 2};
    std::vector<float> hidden(16, 0.5f);
    DraftContext ctx{prompt, generated, hidden, /*slot=*/3, /*pos=*/10,
                     /*next_token=*/42};
    auto draft = mtp.propose(ctx);

    // Draft = sampled token (rides the verify batch) + the head's tokens.
    assert(draft == (std::vector<int32_t>{42, 7, 8}));
    assert(seen_slot == 3);
    assert(seen_hidden == 16);
    assert(seen_tok == 42);       // head seeded with the sampled token
    assert(seen_pos == 10);       // at its stream position
    assert(seen_k == 2);
    PASS();
}

void test_mtp_draft_guards() {
    TEST("MtpDraft: no next_token => empty; empty hidden => fail-loud");

    MtpDraft mtp([](uint32_t, const std::vector<float>&, int32_t, int,
                    uint32_t) { return std::vector<int32_t>{7}; }, 1);

    std::vector<int32_t> prompt = {1}, generated = {1};
    std::vector<float> hidden(4, 0.0f), no_hidden;

    // next_token unavailable → no draft (normal decode).
    DraftContext no_tok{prompt, generated, hidden, 0, 5, /*next_token=*/-1};
    assert(mtp.propose(no_tok).empty());

    // hidden missing while needs_hidden_state() → contract violation.
    DraftContext no_hid{prompt, generated, no_hidden, 0, 5, 42};
    bool threw = false;
    try { mtp.propose(no_hid); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    PASS();
}

int main() {
    std::cout << "\n=== Draft-source seam (Phase 1) ===\n" << std::endl;
    test_seam_all_accepted();
    test_seam_partial_accept_rewinds();
    test_seam_needs_hidden_and_ctx_passthrough();
    test_seam_empty_draft_falls_back();
    test_pld_through_seam_parity();

    std::cout << "\n=== MtpDraft (Phase 4) ===\n" << std::endl;
    test_mtp_draft_shape_and_seeding();
    test_mtp_draft_guards();

    std::cout << "\n" << tests_passed << "/" << tests_total << " passed\n" << std::endl;
    return tests_passed == tests_total ? 0 : 1;
}
