#pragma once

#include <vector>
#include <cstdint>

#include "ggml-backend.h"

// ============================================================================
// IMtpDraftable — the capability a recipe offers to host MTP (NextN) draft
// generation. Mirrors i_image_embeddable (Seam B): a narrow, recipe-side
// interface a model-agnostic consumer (MtpDraft, an IDraftSource, Phase 4)
// drives without knowing which family's head it is. Present only on recipes
// whose GGUF carries a trained NextN head — qwen36 today, qwen35 later.
// See docs/plan-mtp-decode.md §4 (head spec) and §5 D2.
//
// Layering: MtpDraft lives in sampling/ and reaches this interface through a
// std::function bridge wired in cli/ (the SpeculativeBridge precedent), so
// sampling/ never depends on models/.
// ============================================================================
class IMtpDraftable {
public:
    virtual ~IMtpDraftable() = default;

    // Does this loaded instance carry a usable MTP head? A recipe that
    // implements the interface but was loaded from a non-MTP GGUF returns
    // false (nextn_predict_layers == 0). `--speculative mtp` on such a model
    // fails loud upstream, naming the absent capability.
    virtual bool mtp_supported() const = 0;

    // Draft up to `k` continuation tokens for `slot` by running the NextN head
    // recursively (the recipe owns the loop — the Phase-3 decision). Seeds from
    // `hidden` (the main model's pre-final-norm hidden at the last accepted
    // position, produced by the D3 "hidden_out" output) and `last_token`; each
    // step feeds the drafted token and the head's own hidden back into the next
    // step. Greedy (argmax) — draft tokens are always model-verified downstream,
    // so sampling here would only lower acceptance. Uses a private head KV,
    // reset at the start of every call (stateless across decode steps). `pos`
    // is the KV position of `last_token`, for RoPE. Returns the k drafted
    // tokens (empty if the head is unsupported).
    virtual std::vector<int32_t> mtp_draft(
        uint32_t                  slot,
        const std::vector<float>& hidden,
        int32_t                   last_token,
        int                       pos,
        uint32_t                  k,
        ggml_backend_sched_t      sched) = 0;
};
