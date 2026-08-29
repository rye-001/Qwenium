#pragma once

#include "graph_input.h"

// Owns the "valid_indices" slot: the I32 row selection for the sparse LM head
// (ggml_get_rows over the output weight), so a grammar-narrowed step computes
// logits for only the legal rows of a ~150k-row head.
//
// Registered by build_output_head ONLY when sparse_decode_ids_ is non-empty, so
// unlike the other inputs this slot may be absent from the graph entirely —
// that absence is the dense path, not an error.
// Unit test: tests/unit/test_sparse_head_input.cpp
class SparseHeadInput : public GraphInput {
public:
    explicit SparseHeadInput(const char* slot = "valid_indices") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const char* slot_;
};
