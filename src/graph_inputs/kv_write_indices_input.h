#pragma once
// kv_write_indices_input.h — typed graph input: KV-cache write row indices.
//
// Responsibility: populate the I64 "kv_write_indices" tensor consumed by the
//   ggml_set_rows KV write in the batched decode path (one destination row
//   per batch element: slot_idx * n_ctx_max + position).
// This is the tensor that makes the decode KV write POSITION a run-time
//   value instead of a graph-baked view offset — the enabling move for the
//   persistent decode graph (docs/plan-persistent-decode-graph.md §2.1).
// Invariants: fail-loud when a position falls outside [0, n_ctx_max) or the
//   tensor row count disagrees with the step's batch size.
// Unit test: tests/unit/test_kv_write_indices_input.cpp

#include "graph_input.h"

#include <cstdint>

class KvWriteIndicesInput : public GraphInput {
public:
    explicit KvWriteIndicesInput(uint32_t n_ctx_max) : n_ctx_max_(n_ctx_max) {}

    void set_input(const StepContext& step) override;

    const char* slot_name() const override { return slot_; }

    static constexpr const char* slot_ = "kv_write_indices";

private:
    uint32_t n_ctx_max_;
};
