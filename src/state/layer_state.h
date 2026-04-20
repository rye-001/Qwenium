#pragma once
// LayerState — abstract interface for all per-sequence state.
//
// Responsibility: minimal contract shared by KVCache and RecurrentState.
//   Both implement reset_sequence and memory_bytes; no other shared methods
//   exist (KV and recurrent state have fundamentally different update semantics
//   and forcing a common advance/evict/checkpoint method produces no-ops on
//   one side — see docs/modular-layer-architecture.md § State Management).
// Public surface: reset_sequence, memory_bytes, virtual destructor.
// State owned: none — concrete subclasses own their backing storage.
// Invariants: after reset_sequence(seq_id) the state for seq_id is zeroed
//   and ready for a new sequence starting at position 0.
// Reference: docs/modular-layer-architecture.md § State Management Lifecycles.
// Unit test: tests/unit/test_layer_state.cpp

#include <cstddef>

class LayerState {
public:
    virtual ~LayerState() = default;

    // Zero the state for seq_id so it is ready for a new sequence.
    virtual void reset_sequence(int seq_id) = 0;

    // Total bytes allocated for this state object (constant after construction).
    virtual size_t memory_bytes() const = 0;
};
