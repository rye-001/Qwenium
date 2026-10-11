#pragma once

#include "graph_input.h"

#include <cstdint>
#include <string>

// Owns one encoder KQ mask: every query attends in BOTH directions. Its own
// class, not a mode of AttnMaskInput, because the two masks do not share a
// meaning — AttnMaskInput's window is a causal look-back count with an optional
// image span; this one is a distance in either direction and has neither.
//
//   open(slot)          — every key is visible (full-attention layers);
//   band(slot, radius)  — key j is visible from query row q iff |q - j| <= radius
//                         (sliding layers). radius > 0, fail-loud otherwise.
//
// EmbeddingGemma-2's radius is HF's `sliding_window` (512): its transformers
// config documents it as an INCLUSIVE radius and masks with
// abs(q_idx - kv_idx) <= sliding_window (masking_utils
// sliding_window_bidirectional_overlay, read 2026-10-07). The GGUF stores twice
// that (llama.cpp's converter doubles it); the recipe converts.
//
// Slot shape: [n_kv, n_rows] F32, row-major mask[row*n_kv + j], query row r at
// position StepContext::row_kv(r). One mask serves every layer of its kind —
// the recipe builds it once per graph, not once per layer.
class BidirectionalMaskInput : public GraphInput {
public:
    static BidirectionalMaskInput open(std::string slot);
    static BidirectionalMaskInput band(std::string slot, uint32_t radius);

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_.c_str(); }

    bool     is_band() const { return radius_ > 0; }
    uint32_t radius()  const { return radius_; }

private:
    BidirectionalMaskInput(std::string slot, uint32_t radius)
        : slot_(std::move(slot)), radius_(radius) {}

    std::string slot_;
    uint32_t    radius_;   // 0 only via open(): no band
};
