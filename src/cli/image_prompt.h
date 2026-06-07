#pragma once
// image_prompt.h — Gemma 3 image-marker expansion / chunk-list step (Phase 6).
//
// The chat template renders the user turn with a single `<start_of_image>`
// marker where the image goes (string interface preserved — markers live in
// the rendered string). This step expands that marker at the TOKEN level into
// the Gemma 3 image block and reports the soft-token span the orchestrator must
// fill with encoded image embeddings:
//
//   ... <start_of_image> <image_soft_token>×N <end_of_image> ...
//                        └────────── span ──────────┘
//
// One image per turn in v1 (single-tile). Pure function over token ids — no
// model, no tokenizer, no bitmap — so it is unit-gated directly.

#include <cstdint>
#include <vector>

namespace qinf::cli {

struct ExpandedImagePrompt {
    std::vector<int32_t> tokens;       // stream with the image block expanded
    int32_t              span_start;   // index of the first soft-token
    uint32_t             span_len;     // == n_image_tokens
};

// Expand the single `start_of_image_id` token in `tokens` into
//   start_of_image, soft_token_id×n_image_tokens, end_of_image_id
// and report the soft-token span [span_start, span_start+span_len).
//
// Fail-loud (CLAUDE.md): throws std::runtime_error if `tokens` does not contain
// exactly one start_of_image_id (zero → no image marker; >1 → multi-image is
// Phase 7), naming the parameter and the actual count.
ExpandedImagePrompt expand_image_markers(
    const std::vector<int32_t>& tokens,
    int32_t  start_of_image_id,
    int32_t  soft_token_id,
    int32_t  end_of_image_id,
    uint32_t n_image_tokens);

}  // namespace qinf::cli
