#pragma once

#include "graph_input.h"
#include "../layers/routing_trace.h"

// Owns every "moe_routing.<il>" slot: the pinned top-k expert selection that
// MoELayer built as a graph input instead of computing with argsort.
//
// ONE input for ALL MoE layers rather than one per layer, because the trace is
// one object and a per-layer split would give it many writers. It scans the
// graph for the slots MoELayer actually created, so a truncated prefill (the
// lens verify path builds only the first N blocks) is handled by construction:
// the slots that do not exist are not filled, and none are missing.
//
// Absent unless DecodePolicy::routing_replay is set, and its absence is the
// router path, not an error — same contract as SparseHeadInput.
// Unit test: tests/unit/test_routing_replay_input.cpp
class RoutingReplayInput : public GraphInput {
public:
    // `trace` is borrowed and must outlive the graph. Null is a programming
    // error, not a silent no-op: this input is only ever registered on the
    // replay path.
    explicit RoutingReplayInput(const RoutingTrace* trace,
                                const char* slot = "moe_routing")
        : trace_(trace), slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    const RoutingTrace* trace_;
    const char*         slot_;
};
