#pragma once

#include "graph_input.h"

// Owns the "valid_indices" slot: the int32 row-selection tensor for the
// sparse LM head (ggml_get_rows over the output weight). 
// (the tensor exists in the graph only on the sparse path).
class SparseHeadInput : public GraphInput {
public:
    explicit SparseHeadInput(const char* slot = "valid_indices") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const char* slot_;
};
