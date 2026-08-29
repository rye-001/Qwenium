#pragma once
// image_prompt.h — image-marker expansion at the token level.
//
// The chat template renders the user turn with a single begin-image marker
// where the image goes (string interface preserved — markers live in the
// rendered string). This step expands that ONE marker into the family's image
// block and reports the soft-token span the orchestrator must fill with encoded
// image embeddings:
//
//   ... <begin_image> <image_soft_token>×N <end_image> ...
//                     └────────── span ──────────┘
//
// FAMILY-GENERIC: the three marker ids are parameters, resolved once per
// projector by vision/vision_profile (Gemma 3's <start_of_image>, Gemma 4's and
// Qwen-VL's own). This file knows nothing about which family it is serving —
// which is why it is not in cli/ and carries no family name.
//
// One image per turn in v1 (single-tile). Pure function over token ids — no
// model, no tokenizer, no bitmap — so it is unit-gated directly.
// Unit test: tests/unit/test_image_prompt.cpp

#include <cstdint>
#include <vector>

namespace qinf::image {

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

}  // namespace qinf::image
