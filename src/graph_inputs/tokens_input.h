#pragma once

#include "graph_input.h"

// Owns the "tokens" slot: the I32 input feeding ggml_get_rows for the embedding
// lookup — one id per row of the batch, in build order.
//
// Filled by ForwardPassBase::set_{prefill,decode}_inputs, which fans set_input
// over the recipe's GraphInputSet. A recipe declares this input while building;
// it never writes the tensor itself.
// Unit test: tests/unit/test_tokens_input.cpp
class TokensInput : public GraphInput {
public:
    explicit TokensInput(const char* slot = "tokens") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const char* slot_;
};
