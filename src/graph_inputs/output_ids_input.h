#pragma once

#include "graph_input.h"

// Owns the "out_ids" slot: the int32 token-position row-selection tensor for
// the prefill LM head. During prefill every layer runs on all N prompt tokens
// (context ingestion), but the ~150k-wide LM head only needs the position(s)
// that actually produce logits. A ggml_get_rows on the hidden state immediately
// before the head, indexed by this slot, elides the discarded first N-1 rows.
//
// Default index set: last token only (position n_rows-1). This is the
// token-position axis slice; it is orthogonal to the vocabulary-axis slice
// owned by SparseHeadInput ("valid_indices", a get_rows on the head WEIGHT).
// Both can be active in the same graph and compose order-independently — they
// gather different tensors on different axes.
//
// Fail-loud (CLAUDE.md): the slot must be present, non-empty when n_rows>0,
// and every selected position must be in [0, n_rows). Violations throw at the
// module boundary naming the slot, the expected range, and the actual value.
// No silent fallback to a dense head.
class OutputIdsInput : public GraphInput {
public:
    explicit OutputIdsInput(const char* slot = "out_ids") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const char* slot_;
};
