#pragma once

#include "graph_input.h"

#include <cstdint>
#include <string>

// Owns one KQ mask slot. Parameterize-vs-split: the sliding window is a *parameter* here,
// not a separate LocalMaskInput/GlobalMaskInput. window == 0 => global
// (pure causal); window > 0 => causal AND within-window cutoff (Gemma 2/3/4
// local layers).
//
// Bidirectional span (Gemma vision): an optional contiguous run of
// ABSOLUTE positions [bidi_start, bidi_start + bidi_len) that attend to each
// other in BOTH directions, bypassing causal order AND the sliding window.
// This is the image-token span: image soft-tokens see the whole image, not
// just earlier image tokens. Like `window`, it is a *parameter* the recipe
// sets — gemma3 passes a non-zero span when image embeddings are armed; every
// other recipe (and gemma3 text-only) leaves it disabled (bidi_len == 0),
// recovering the pure causal+window mask bit-for-bit. The bidi relaxation is
// scoped to within-span query/key pairs only: a query inside the span still
// attends causally+windowed to text *before* the span, and text *after* the
// span attends causally+windowed back to the image (the span is "in the past"
// once it closes). Recipe-bound shape decision; nothing lifts to ForwardPassBase.
//
// Slot name is the full tensor name as built by the recipe:
//   - per-layer prefill mask: "kq_mask.{il}"  (shape [n_kv, n_tokens])
//   - shared batched-decode mask: "kq_mask_b" (shape [n_kv, 1, 1, n_batch])
// Both layouts are row-major mask[row*n_kv + j]; only the per-row query
// position differs (StepContext::row_pos), so one body serves both.
class AttnMaskInput : public GraphInput {
public:
    AttnMaskInput(std::string slot, uint32_t window,
                  int32_t bidi_start = -1, uint32_t bidi_len = 0)
        : slot_(std::move(slot)), window_(window),
          bidi_start_(bidi_start), bidi_len_(bidi_len) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_.c_str(); }

private:
    std::string slot_;
    uint32_t    window_;       // 0 = global / no window
    int32_t     bidi_start_;   // first absolute pos of the bidi span; <0 = none
    uint32_t    bidi_len_;     // span length; 0 = no bidi span
};
