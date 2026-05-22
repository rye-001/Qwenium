#pragma once

#include "graph_input.h"

// Owns the "inp_pos" slot.
class PositionsInput : public GraphInput {
public:
    explicit PositionsInput(const char* slot = "inp_pos") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const char* slot_;
};
