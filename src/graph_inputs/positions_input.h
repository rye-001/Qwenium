#pragma once

#include "graph_input.h"

// Owns the "inp_pos" slot: ONE position per token, for recipes using plain
// NEOX RoPE. The M-RoPE recipes use MRopePositionsInput instead (four
// components per token) — a recipe declares exactly one of the two.
//
// The value is the ROPE position, which is not the KV row index once an image
// span is involved (see ForwardPassBase::get_rope_pos).
// Unit test: tests/unit/test_positions_input.cpp
class PositionsInput : public GraphInput {
public:
    explicit PositionsInput(const char* slot = "inp_pos") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const char* slot_;
};
