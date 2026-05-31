#pragma once

#include "graph_input.h"

// Owns the "out_ids" slot: the int32 token-position row-selection tensor for
// the prefill LM head. During prefill every layer runs on all N prompt tokens
// (context ingestion), but the ~150k-wide LM head only needs the position(s)
// that actually produce logits. A ggml_get_rows on the hidden state immediately
// before the head, indexed by this slot, elides the discarded first N-1 rows.
class OutputIdsInput : public GraphInput {
public:
    explicit OutputIdsInput(const char* slot = "out_ids") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const char* slot_;
};
