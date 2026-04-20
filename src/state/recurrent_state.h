#pragma once
// recurrent_state.h — fixed-size per-slot recurrent state with checkpoint/restore.
//
// Responsibility: manage the fixed-size state matrix for recurrent layer types
//   (DeltaNet, SSM). Implements LayerState; provides checkpoint/restore for
//   grammar backtracking and speculative-decoding rejection.
// Public surface: LayerState interface + checkpoint / restore / release +
//   state_data accessor for graph building.
// State owned: a flat float buffer per slot, allocated on the CPU in Phase 2.
//   Phase 3 will add a Metal-backed variant when DeltaNet is wired in.
// Invariants:
//   - Each CheckpointId is valid exactly once. release() on an invalid id
//     throws std::runtime_error naming the id, per the fail-loud contract.
//   - Checkpoint ids are recycled (free-list) so the id space stays bounded.
//   - restore() does NOT release the checkpoint — caller must call release()
//     explicitly when done (supports multiple restore attempts per checkpoint).
// Reference: docs/modular-layer-architecture.md § State Management Lifecycles.
// Unit test: tests/unit/test_recurrent_state.cpp

#include "layer_state.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using CheckpointId = uint32_t;
static constexpr CheckpointId kInvalidCheckpoint = std::numeric_limits<uint32_t>::max();

class RecurrentState : public LayerState {
public:
    // state_floats: number of float elements in one slot's state.
    // n_slots: maximum parallel sequences.
    RecurrentState(size_t state_floats, int n_slots);

    // LayerState interface.
    void   reset_sequence(int seq_id) override;
    size_t memory_bytes() const override;

    // Snapshot the current state for seq_id.
    // Returns a CheckpointId that can be passed to restore() or release().
    CheckpointId checkpoint(int seq_id);

    // Overwrite the live state for the original seq_id with the snapshot.
    // Does NOT release the checkpoint — caller decides when to call release().
    void restore(CheckpointId id);

    // Free the snapshot. Throws std::runtime_error on double-release or invalid id.
    void release(CheckpointId id);

    // Direct access to the live state floats for seq_id.
    // Used by the graph builder to create ggml tensor views over the state.
    float*       state_data(int seq_id);
    const float* state_data(int seq_id) const;

    size_t state_floats() const { return state_floats_; }
    int    n_slots()      const { return n_slots_; }

private:
    struct Checkpoint {
        int               seq_id = -1;
        std::vector<float> data;
        bool              valid  = false;
    };

    size_t state_floats_;
    int    n_slots_;

    std::vector<std::vector<float>> state_;       // [n_slots][state_floats]
    std::vector<Checkpoint>         checkpoints_; // indexed by CheckpointId
    std::vector<uint32_t>           free_ids_;    // recycled checkpoint slots

    void validate_seq_id(int seq_id) const;
    void validate_checkpoint_id(CheckpointId id, const char* caller) const;
};
