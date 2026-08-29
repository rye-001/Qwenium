#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "prompt_lookup.h"

namespace qinf {

// ============================================================================
// IDraftSource: where speculative draft tokens come from
//
// SpeculativeDecoder keeps everything that makes speculation *correct* —
// batched verify, greedy accept, bonus token, KV rewind. A draft source only
// answers "given where this slot is, what do you think comes next?".
//
// The seam exists because two genuinely different things answer that question:
//   - PromptLookupDraft — a pure function of token history (n-gram lookup).
//   - MtpDraft (Phase 4) — the model's trained NextN head, driven by the last
//     hidden state, not by token history.
// They read *disjoint* inputs, so the context carries a superset and each
// source reads only its slice. That disjointness is exactly why this is a real
// seam and not a PLD signature with a second implementation bolted on.
// ============================================================================

// Everything a draft source may need for one slot. Fields are const refs — the
// caller (the decode loop, via SpeculativeDecoder) owns the storage.
struct DraftContext {
    const std::vector<int32_t>& prompt_tokens;     // n-gram haystack (PLD reads)
    const std::vector<int32_t>& generated_tokens;  // all tokens so far

    // Pre-final-norm hidden of the last FED position. EMPTY unless an active
    // source reports needs_hidden_state().
    const std::vector<float>&   last_hidden;

    uint32_t slot;
    int      current_pos;  // KV pos after last fed token; draft[0] goes here

    // The sampled-but-not-yet-fed token: draft[0]'s only legal value under the
    // first-token guard, or -1 if unavailable. PLD ignores it; MTP seeds its
    // recursion with it.
    int32_t  next_token = -1;
};

class IDraftSource {
public:
    virtual ~IDraftSource() = default;

    // Propose up to max_draft continuation tokens for ctx.slot, or empty to
    // mean "no draft — fall back to a normal decode". A source is a pure
    // function of ctx across decode steps: verification produces the hidden
    // state of the accepted position, the loop feeds it back next call, so a
    // source holds no state between calls (any per-draft scratch — e.g. the
    // NextN head's private KV — lives and dies inside one propose()).
    virtual std::vector<int32_t> propose(const DraftContext& ctx) = 0;

    // Whether the decode loop must compute and pass ctx.last_hidden. This is
    // the switch behind D3's opt-in hidden-state graph output: false ⇒ the
    // forward pass is node-for-node what it is today (the PLD/byte-identity
    // path). Constant for the life of the source.
    virtual bool needs_hidden_state() const = 0;
};

// ============================================================================
// PromptLookupDraft: PLD behind the seam. Pure n-gram lookup over the prompt;
// last_hidden unused. Byte-for-byte the same draft PromptLookup produced when
// SpeculativeDecoder called it inline.
// ============================================================================
class PromptLookupDraft : public IDraftSource {
public:
    explicit PromptLookupDraft(PromptLookupConfig config = {})
        : lookup_(config) {}

    std::vector<int32_t> propose(const DraftContext& ctx) override {
        return lookup_.find_draft(ctx.prompt_tokens, ctx.generated_tokens);
    }

    bool needs_hidden_state() const override { return false; }

    const PromptLookup& lookup() const { return lookup_; }

private:
    PromptLookup lookup_;
};

// ============================================================================
// MtpDraft: the model's trained NextN head behind the seam.
//
// Layering: sampling/ must not depend on models/, so the head is reached
// through a std::function bridge wired in cli/ (the SpeculativeBridge
// precedent). The bridge signature mirrors IMtpDraftable::mtp_draft with the
// scheduler already bound.
//
// Draft shape: [next_token, head_draft...]. The sampled-but-unfed token rides
// the verify batch as draft[0] — it must be fed through the model anyway, it
// trivially passes the first-token guard, and the verify logits it produces
// are what check the head's first real draft. The head then recurses from
// (last_hidden, next_token) for up to max_head_draft more tokens.
// ============================================================================
class MtpDraft : public IDraftSource {
public:
    // bridge(slot, hidden, last_token, pos, k) -> k head-drafted tokens.
    using DraftFn = std::function<std::vector<int32_t>(
        uint32_t, const std::vector<float>&, int32_t, int, uint32_t)>;

    MtpDraft(DraftFn bridge, int max_head_draft)
        : bridge_(std::move(bridge)), max_head_draft_(max_head_draft) {
        if (!bridge_)
            throw std::runtime_error(
                "MtpDraft: bridge expected callable, got empty function");
        if (max_head_draft_ < 1)
            throw std::runtime_error(
                "MtpDraft: max_head_draft expected >= 1, got " +
                std::to_string(max_head_draft_));
    }

    std::vector<int32_t> propose(const DraftContext& ctx) override {
        if (ctx.next_token < 0)
            return {};  // nothing sampled yet — nothing to extend
        if (ctx.last_hidden.empty())
            throw std::runtime_error(
                "MtpDraft: last_hidden expected non-empty (needs_hidden_state "
                "is true), got empty — decode loop did not capture hidden");

        // next_token sits at ctx.current_pos; the head extends it.
        std::vector<int32_t> draft;
        draft.reserve(1 + max_head_draft_);
        draft.push_back(ctx.next_token);
        std::vector<int32_t> head = bridge_(
            ctx.slot, ctx.last_hidden, ctx.next_token, ctx.current_pos,
            static_cast<uint32_t>(max_head_draft_));
        draft.insert(draft.end(), head.begin(), head.end());
        return draft;
    }

    bool needs_hidden_state() const override { return true; }

private:
    DraftFn bridge_;
    int     max_head_draft_;
};

} // namespace qinf
