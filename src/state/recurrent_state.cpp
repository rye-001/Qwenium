#include "recurrent_state.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

RecurrentState::RecurrentState(size_t state_floats, int n_slots)
    : state_floats_(state_floats), n_slots_(n_slots) {
    state_.assign(n_slots, std::vector<float>(state_floats, 0.0f));
}

void RecurrentState::reset_sequence(int seq_id) {
    validate_seq_id(seq_id);
    std::fill(state_[seq_id].begin(), state_[seq_id].end(), 0.0f);
}

size_t RecurrentState::memory_bytes() const {
    // Live state only; checkpoints are transient and not counted.
    return static_cast<size_t>(n_slots_) * state_floats_ * sizeof(float);
}

CheckpointId RecurrentState::checkpoint(int seq_id) {
    validate_seq_id(seq_id);

    uint32_t id;
    if (!free_ids_.empty()) {
        id = free_ids_.back();
        free_ids_.pop_back();
        checkpoints_[id].seq_id = seq_id;
        checkpoints_[id].data   = state_[seq_id];
        checkpoints_[id].valid  = true;
    } else {
        id = static_cast<uint32_t>(checkpoints_.size());
        checkpoints_.push_back({seq_id, state_[seq_id], true});
    }
    return id;
}

void RecurrentState::restore(CheckpointId id) {
    validate_checkpoint_id(id, "restore");
    const Checkpoint& cp = checkpoints_[id];
    state_[cp.seq_id] = cp.data;
}

void RecurrentState::release(CheckpointId id) {
    validate_checkpoint_id(id, "release");
    checkpoints_[id].valid = false;
    checkpoints_[id].data.clear();
    free_ids_.push_back(id);
}

float* RecurrentState::state_data(int seq_id) {
    validate_seq_id(seq_id);
    return state_[seq_id].data();
}

const float* RecurrentState::state_data(int seq_id) const {
    validate_seq_id(seq_id);
    return state_[seq_id].data();
}

void RecurrentState::validate_seq_id(int seq_id) const {
    if (seq_id < 0 || seq_id >= n_slots_) {
        throw std::runtime_error(
            std::string("RecurrentState: seq_id ") + std::to_string(seq_id) +
            " out of range, expected [0, " + std::to_string(n_slots_ - 1) + "]");
    }
}

void RecurrentState::validate_checkpoint_id(CheckpointId id, const char* caller) const {
    if (id == kInvalidCheckpoint
        || id >= static_cast<uint32_t>(checkpoints_.size())
        || !checkpoints_[id].valid) {
        throw std::runtime_error(
            std::string("RecurrentState::") + caller + ": checkpoint id " +
            std::to_string(id) + " is invalid or already released");
    }
}
