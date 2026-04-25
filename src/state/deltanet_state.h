#pragma once
// deltanet_state.h — per-slot recurrent + conv state for DeltaNet layers.
//
// Responsibility: manage the fixed-size state (S matrix + causal-conv window)
//   for Gated DeltaNet layers. Implements LayerState. Provides
//   checkpoint/restore for grammar backtracking and speculative-decoding
//   rejection. Backend-backed so state lives on the same device (Metal/CPU)
//   as the inference compute.
// Public surface:
//   LayerState interface (reset_sequence, memory_bytes)
//   checkpoint / restore / release — full-state snapshots per slot
//   recurrent_tensor / conv_tensor — ggml tensor handles for graph building
//   get/set_recurrent, get/set_conv — direct CPU↔backend I/O for management
//   clear_slot / clone_slot — lifecycle helpers
// State owned: per-layer backend tensors [state_floats, n_slots] (recurrent)
//   and [(conv_kernel-1)*conv_channels, n_slots] (conv window). CPU-side
//   snapshot vectors for checkpoint storage.
// Invariants:
//   - Each CheckpointId is valid exactly once. release() on an already-released
//     id throws std::runtime_error naming the id (fail-loud contract).
//   - Checkpoint ids are recycled via a free-list so the id space is bounded.
//   - restore() does NOT release the checkpoint — caller must release() after.
//   - state is always zeroed on construction and after reset_sequence().
// Reference: docs/modular-layer-architecture.md § State Management Lifecycles
//   and § DeltaNet (Gated Delta Rule).
// Unit test: tests/unit/test_deltanet_state.cpp

#include "layer_state.h"
#include "recurrent_state.h"  // for CheckpointId / kInvalidCheckpoint

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class DeltaNetState : public LayerState {
public:
    struct Hparams {
        uint32_t       n_dn_layers;    // number of DeltaNet layers (logical index 0..n-1)
        uint32_t       n_slots;        // maximum parallel sequences
        uint32_t       head_v_dim;     // value head dimension (d_inner / num_v_heads)
        uint32_t       head_k_dim;     // key head dimension (ssm_state_size)
        uint32_t       num_v_heads;    // number of value heads (ssm_group_count)
        uint32_t       conv_channels;  // full conv channel width
        uint32_t       conv_kernel;    // causal conv1d kernel size (e.g. 4)
        ggml_backend_t backend;        // nullptr → CPU backend
    };

    explicit DeltaNetState(const Hparams& hp);

    // LayerState interface.
    void   reset_sequence(int seq_id) override;
    size_t memory_bytes() const override;

    // Snapshot the full DeltaNet state (recurrent + conv, all layers) for seq_id.
    // Returns a CheckpointId usable for restore() or release().
    CheckpointId checkpoint(int seq_id);

    // Overwrite the live state for the original seq_id from the snapshot.
    // Does NOT release the checkpoint; caller must call release() when done.
    void restore(CheckpointId id);

    // Free the snapshot. Throws std::runtime_error on double-release or kInvalidCheckpoint.
    void release(CheckpointId id);

    // Return the ggml tensor for layer dn_idx suitable for view-based graph building.
    // Shape: [rec_slot_floats, n_slots], type F32.
    ggml_tensor* recurrent_tensor(uint32_t dn_idx);

    // Return the ggml tensor for the conv window of layer dn_idx.
    // Shape: [(conv_kernel-1)*conv_channels, n_slots], type F32.
    ggml_tensor* conv_tensor(uint32_t dn_idx);

    // Direct CPU↔backend read/write (for clear_slot, clone_slot, checkpoint/restore).
    void get_recurrent(uint32_t dn_idx, uint32_t slot, float* dst) const;
    void set_recurrent(uint32_t dn_idx, uint32_t slot, const float* src);
    void get_conv(uint32_t dn_idx, uint32_t slot, float* dst) const;
    void set_conv(uint32_t dn_idx, uint32_t slot, const float* src);

    // Zero all state across all layers for the given slot.
    void clear_slot(uint32_t slot);

    // Copy all state from src_slot to dst_slot across all layers.
    void clone_slot(uint32_t src_slot, uint32_t dst_slot);

    uint32_t n_dn_layers()     const { return hp_.n_dn_layers; }
    uint32_t n_slots_count()   const { return hp_.n_slots; }
    size_t   rec_slot_floats() const { return rec_slot_floats_; }
    size_t   conv_slot_floats() const { return conv_slot_floats_; }
    uint32_t head_k_dim()      const { return hp_.head_k_dim; }
    uint32_t num_v_heads()     const { return hp_.num_v_heads; }
    uint32_t conv_kernel()     const { return hp_.conv_kernel; }

private:
    Hparams  hp_;
    size_t   rec_slot_floats_;    // head_v_dim * head_k_dim * num_v_heads
    size_t   conv_slot_floats_;   // (conv_kernel - 1) * conv_channels

    std::unique_ptr<ggml_context,       void(*)(ggml_context*)>          ctx_;
    std::unique_ptr<ggml_backend_buffer, void(*)(ggml_backend_buffer*)>  buf_;

    std::vector<ggml_tensor*> rec_tensors_;   // [n_dn_layers]: [rec_slot_floats, n_slots]
    std::vector<ggml_tensor*> conv_tensors_;  // [n_dn_layers]: [conv_slot_floats, n_slots]

    struct Checkpoint {
        int   seq_id = -1;
        bool  valid  = false;
        // CPU-side snapshots of all-layers state for one slot
        std::vector<std::vector<float>> rec;   // [n_dn_layers][rec_slot_floats]
        std::vector<std::vector<float>> conv;  // [n_dn_layers][conv_slot_floats]
    };
    std::vector<Checkpoint>   checkpoints_;
    std::vector<CheckpointId> free_ids_;

    void init_storage();
    void validate_slot(uint32_t slot, const char* caller) const;
    void validate_checkpoint(CheckpointId id, const char* caller) const;
};
