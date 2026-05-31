#pragma once

#include "graph_input.h"

#include <cstdint>
#include <string>

// Owns one KQ mask slot. Parameterize-vs-split: the sliding window is a *parameter* here,
// not a separate LocalMaskInput/GlobalMaskInput. window == 0 => global
// (pure causal); window > 0 => causal AND within-window cutoff (Gemma 2/3/4
// local layers).
//
// Slot name is the full tensor name as built by the recipe:
//   - per-layer prefill mask: "kq_mask.{il}"  (shape [n_kv, n_tokens])
//   - shared batched-decode mask: "kq_mask_b" (shape [n_kv, 1, 1, n_batch])
// Both layouts are row-major mask[row*n_kv + j]; only the per-row query
// position differs (StepContext::row_pos), so one body serves both.
class AttnMaskInput : public GraphInput {
public:
    AttnMaskInput(std::string slot, uint32_t window)
        : slot_(std::move(slot)), window_(window) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_.c_str(); }

private:
    std::string slot_;
    uint32_t    window_;  // 0 = global / no window
};
