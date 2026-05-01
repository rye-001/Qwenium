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

// ── KV cache sharing spec ──────────────────────────────────────────
//
// Per-layer descriptor controlling whether a KV-cache slot is owned by the
// layer or aliased to another layer's slot.  Used by Gemma 4 to express the
// "shared KV across layers" feature: the last `num_kv_shared_layers` layers
// reuse the K and V tensors from the last non-shared layer of the same
// attention type.  Each shared layer still computes its own Q.
//
// Status in 26B-A4B: INACTIVE (gemma4.attention.shared_kv_layers == 0).  The
// interface ships in G4 and is exercised by a synthetic state-isolation unit
// test (test_kv_reference_mode.cpp).  The dense 31B follow-up promotes this
// to a canary gate.
//
// is_reference  — true: this layer's slot aliases another layer's slot; no
//                 storage is allocated for this layer.  false (default):
//                 the layer owns its own slot.
// source_layer  — the physical layer index whose K/V tensors this layer
//                 references when is_reference == true.  Must be < the
//                 current layer index AND must point to an owning layer
//                 (transitive aliasing is not permitted; sources resolve
//                 in one hop, by construction of the simple_kv_cache
//                 constructor).
struct KvSharingSpec {
    bool is_reference = false;
    int  source_layer = -1;
};
