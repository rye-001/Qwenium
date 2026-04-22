#include "deltanet_state.h"

#include "ggml-backend.h"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

// ── Construction ──────────────────────────────────────────────────────────────

DeltaNetState::DeltaNetState(const Hparams& hp)
    : hp_(hp)
    , rec_slot_floats_(static_cast<size_t>(hp.head_v_dim) * hp.head_k_dim * hp.num_v_heads)
    , conv_slot_floats_(static_cast<size_t>(hp.conv_kernel > 0 ? hp.conv_kernel - 1 : 0) * hp.conv_channels)
    , ctx_(nullptr, ggml_free)
    , buf_(nullptr, ggml_backend_buffer_free)
{
    if (hp_.n_dn_layers == 0)
        throw std::runtime_error("deltanet_state: n_dn_layers must be > 0");
    if (hp_.n_slots == 0)
        throw std::runtime_error("deltanet_state: n_slots must be > 0");

    init_storage();
}

void DeltaNetState::init_storage() {
    // 2 tensors per DeltaNet layer (recurrent + conv) + small headroom
    const size_t ctx_size = (static_cast<size_t>(hp_.n_dn_layers) * 2 + 64)
                             * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_.reset(ggml_init(params));
    if (!ctx_)
        throw std::runtime_error("deltanet_state: ggml_init failed");

    rec_tensors_.resize(hp_.n_dn_layers);
    conv_tensors_.resize(hp_.n_dn_layers);

    for (uint32_t il = 0; il < hp_.n_dn_layers; ++il) {
        rec_tensors_[il] = ggml_new_tensor_2d(
            ctx_.get(), GGML_TYPE_F32,
            static_cast<int64_t>(rec_slot_floats_),
            static_cast<int64_t>(hp_.n_slots));

        conv_tensors_[il] = ggml_new_tensor_2d(
            ctx_.get(), GGML_TYPE_F32,
            static_cast<int64_t>(conv_slot_floats_),
            static_cast<int64_t>(hp_.n_slots));
    }

    if (!hp_.backend)
        throw std::runtime_error("deltanet_state: backend must be non-null");
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(hp_.backend);

    buf_.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_.get(), buft));
    if (!buf_)
        throw std::runtime_error("deltanet_state: failed to allocate backend buffer");

    // Zero-initialise all tensors
    {
        std::vector<uint8_t> zr(rec_slot_floats_  * hp_.n_slots * sizeof(float), 0);
        std::vector<uint8_t> zc(conv_slot_floats_ * hp_.n_slots * sizeof(float), 0);
        for (uint32_t il = 0; il < hp_.n_dn_layers; ++il) {
            ggml_backend_tensor_set(rec_tensors_[il],  zr.data(), 0, zr.size());
            if (conv_slot_floats_ > 0)
                ggml_backend_tensor_set(conv_tensors_[il], zc.data(), 0, zc.size());
        }
    }
}

// ── LayerState ────────────────────────────────────────────────────────────────

void DeltaNetState::reset_sequence(int seq_id) {
    validate_slot(static_cast<uint32_t>(seq_id), "reset_sequence");
    clear_slot(static_cast<uint32_t>(seq_id));
}

size_t DeltaNetState::memory_bytes() const {
    if (!buf_) return 0;
    return ggml_backend_buffer_get_size(buf_.get());
}

// ── Tensor accessors ──────────────────────────────────────────────────────────

ggml_tensor* DeltaNetState::recurrent_tensor(uint32_t dn_idx) {
    if (dn_idx >= hp_.n_dn_layers) {
        std::ostringstream ss;
        ss << "deltanet_state: recurrent_tensor: dn_idx=" << dn_idx
           << " >= n_dn_layers=" << hp_.n_dn_layers;
        throw std::runtime_error(ss.str());
    }
    return rec_tensors_[dn_idx];
}

ggml_tensor* DeltaNetState::conv_tensor(uint32_t dn_idx) {
    if (dn_idx >= hp_.n_dn_layers) {
        std::ostringstream ss;
        ss << "deltanet_state: conv_tensor: dn_idx=" << dn_idx
           << " >= n_dn_layers=" << hp_.n_dn_layers;
        throw std::runtime_error(ss.str());
    }
    return conv_tensors_[dn_idx];
}

// ── Direct read / write ───────────────────────────────────────────────────────

void DeltaNetState::get_recurrent(uint32_t dn_idx, uint32_t slot, float* dst) const {
    GGML_ASSERT(dn_idx < hp_.n_dn_layers);
    GGML_ASSERT(slot   < hp_.n_slots);
    ggml_backend_tensor_get(rec_tensors_[dn_idx], dst,
        slot * rec_slot_floats_ * sizeof(float),
        rec_slot_floats_ * sizeof(float));
}

void DeltaNetState::set_recurrent(uint32_t dn_idx, uint32_t slot, const float* src) {
    GGML_ASSERT(dn_idx < hp_.n_dn_layers);
    GGML_ASSERT(slot   < hp_.n_slots);
    ggml_backend_tensor_set(rec_tensors_[dn_idx], src,
        slot * rec_slot_floats_ * sizeof(float),
        rec_slot_floats_ * sizeof(float));
}

void DeltaNetState::get_conv(uint32_t dn_idx, uint32_t slot, float* dst) const {
    GGML_ASSERT(dn_idx < hp_.n_dn_layers);
    GGML_ASSERT(slot   < hp_.n_slots);
    if (conv_slot_floats_ == 0) return;
    ggml_backend_tensor_get(conv_tensors_[dn_idx], dst,
        slot * conv_slot_floats_ * sizeof(float),
        conv_slot_floats_ * sizeof(float));
}

void DeltaNetState::set_conv(uint32_t dn_idx, uint32_t slot, const float* src) {
    GGML_ASSERT(dn_idx < hp_.n_dn_layers);
    GGML_ASSERT(slot   < hp_.n_slots);
    if (conv_slot_floats_ == 0) return;
    ggml_backend_tensor_set(conv_tensors_[dn_idx], src,
        slot * conv_slot_floats_ * sizeof(float),
        conv_slot_floats_ * sizeof(float));
}

// ── clear_slot / clone_slot ───────────────────────────────────────────────────

void DeltaNetState::clear_slot(uint32_t slot) {
    validate_slot(slot, "clear_slot");
    const std::vector<float> zr(rec_slot_floats_,  0.0f);
    const std::vector<float> zc(conv_slot_floats_, 0.0f);
    for (uint32_t il = 0; il < hp_.n_dn_layers; ++il) {
        set_recurrent(il, slot, zr.data());
        if (conv_slot_floats_ > 0) set_conv(il, slot, zc.data());
    }
}

void DeltaNetState::clone_slot(uint32_t src_slot, uint32_t dst_slot) {
    validate_slot(src_slot, "clone_slot src");
    validate_slot(dst_slot, "clone_slot dst");
    std::vector<float> buf_r(rec_slot_floats_);
    std::vector<float> buf_c(conv_slot_floats_);
    for (uint32_t il = 0; il < hp_.n_dn_layers; ++il) {
        get_recurrent(il, src_slot, buf_r.data());
        set_recurrent(il, dst_slot, buf_r.data());
        if (conv_slot_floats_ > 0) {
            get_conv(il, src_slot, buf_c.data());
            set_conv(il, dst_slot, buf_c.data());
        }
    }
}

// ── Checkpoint / restore / release ───────────────────────────────────────────

CheckpointId DeltaNetState::checkpoint(int seq_id) {
    validate_slot(static_cast<uint32_t>(seq_id), "checkpoint");

    CheckpointId id;
    if (!free_ids_.empty()) {
        id = free_ids_.back();
        free_ids_.pop_back();
    } else {
        id = static_cast<CheckpointId>(checkpoints_.size());
        checkpoints_.emplace_back();
        checkpoints_.back().rec.resize(hp_.n_dn_layers,
            std::vector<float>(rec_slot_floats_));
        checkpoints_.back().conv.resize(hp_.n_dn_layers,
            std::vector<float>(conv_slot_floats_));
    }

    Checkpoint& cp = checkpoints_[id];
    cp.seq_id = seq_id;
    cp.valid  = true;

    const uint32_t slot = static_cast<uint32_t>(seq_id);
    for (uint32_t il = 0; il < hp_.n_dn_layers; ++il) {
        get_recurrent(il, slot, cp.rec[il].data());
        if (conv_slot_floats_ > 0)
            get_conv(il, slot, cp.conv[il].data());
    }

    return id;
}

void DeltaNetState::restore(CheckpointId id) {
    validate_checkpoint(id, "restore");
    const Checkpoint& cp = checkpoints_[id];
    const uint32_t slot = static_cast<uint32_t>(cp.seq_id);
    for (uint32_t il = 0; il < hp_.n_dn_layers; ++il) {
        set_recurrent(il, slot, cp.rec[il].data());
        if (conv_slot_floats_ > 0)
            set_conv(il, slot, cp.conv[il].data());
    }
}

void DeltaNetState::release(CheckpointId id) {
    validate_checkpoint(id, "release");
    checkpoints_[id].valid = false;
    free_ids_.push_back(id);
}

// ── Internal validation ───────────────────────────────────────────────────────

void DeltaNetState::validate_slot(uint32_t slot, const char* caller) const {
    if (slot >= hp_.n_slots) {
        std::ostringstream ss;
        ss << "deltanet_state::" << caller << ": slot=" << slot
           << " >= n_slots=" << hp_.n_slots;
        throw std::runtime_error(ss.str());
    }
}

void DeltaNetState::validate_checkpoint(CheckpointId id, const char* caller) const {
    if (id == kInvalidCheckpoint || id >= checkpoints_.size() || !checkpoints_[id].valid) {
        std::ostringstream ss;
        ss << "deltanet_state::" << caller << ": invalid or already-released CheckpointId=" << id;
        throw std::runtime_error(ss.str());
    }
}
