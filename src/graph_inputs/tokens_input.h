#pragma once

#include "graph_input.h"

// Owns the "tokens" slot (the I32 input feeding ggml_get_rows for the
// embedding lookup).
class TokensInput : public GraphInput {
public:
    explicit TokensInput(const char* slot = "tokens") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const char* slot_;
};
