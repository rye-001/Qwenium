#pragma once

#include "layer_state.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <memory>
#include <cstdint>

class simple_kv_cache : public LayerState {
public:
    simple_kv_cache(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t n_batch_max,
        uint32_t n_embd_k,
        uint32_t n_embd_v,
        ggml_type type_k = GGML_TYPE_F16,
        ggml_type type_v = GGML_TYPE_F16,
        ggml_backend_t backend = nullptr);  // Optional backend parameter

    // Reference-mode constructor.  Same shape as the primary
    // constructor, plus a per-layer sharing vector.  When sharing[il] is
    // a reference, layer `il` allocates no K/V storage; its slots point at
    // sharing[il].source_layer's tensors.  Pre-conditions enforced
    // fail-loud at construction time:
    //   - sharing.size() == n_layers
    //   - for every reference layer `il`: source_layer is in [0, il) and
    //     is itself a non-reference (owning) layer.
    // When `sharing` is empty the behavior is identical to the primary
    // constructor.
    simple_kv_cache(
        uint32_t n_layers,
        uint32_t n_ctx_max,
        uint32_t n_batch_max,
        uint32_t n_embd_k,
        uint32_t n_embd_v,
        ggml_type type_k,
        ggml_type type_v,
        ggml_backend_t backend,
        const std::vector<KvSharingSpec>& sharing);

    ~simple_kv_cache() = default;

    // True iff layer `il` aliases another layer's K/V storage.  Recipes that
    // route Q-only attention computations through the shared slot use this
    // to skip the (illegal) cpy_k / cpy_v call for reference layers.
    bool is_reference_layer(uint32_t il) const {
        return il < sharing_.size() && sharing_[il].is_reference;
    }
    int  reference_source(uint32_t il) const {
        return is_reference_layer(il) ? sharing_[il].source_layer : -1;
    }

    // Get a full view of cached K for layer il and slot_idx: [n_embd_k, n_kv]
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, uint32_t slot_idx = 0);

    // Get a full view of cached V for layer il and slot_idx: [n_embd_v, n_kv]
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, uint32_t slot_idx = 0);

    // Copy k_cur into cache for slot_idx at current position
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il, uint32_t slot_idx = 0);

    // Copy v_cur into cache for slot_idx at current position
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il, uint32_t slot_idx = 0);

    // LayerState interface.
    void   reset_sequence(int seq_id) override { clear_slot(static_cast<uint32_t>(seq_id)); }
    size_t memory_bytes() const override;

    void advance(uint32_t n_tokens, uint32_t slot_idx = 0);
    void clear_slot(uint32_t slot_idx);
    void clear_all();
    void set_pos(uint32_t p, uint32_t slot_idx = 0);
    uint32_t get_pos(uint32_t slot_idx = 0) const { return positions[slot_idx]; }

    // O(1) head-pointer truncation: discard everything after pos.
    // KV data is not zeroed; next writes will overwrite it.
    // Used by grammar backtracking and speculative-decoding rejection.
    void truncate_to_position(int pos, uint32_t slot_idx = 0) {
        positions[slot_idx] = static_cast<uint32_t>(pos);
    }

    // SnapKV: compact all layers for a slot, keeping only the listed positions.
    // `retained_positions` must be sorted and unique.
    // Updates the position counter to retained_positions.size().
    void compact(uint32_t slot_idx, const std::vector<uint32_t>& retained_positions);

    // Direct memory copy between slots using backend copy
    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens);

    // Gather KV data for specific slots into a batch tensor
    // Returns a tensor of shape [n_embd, n_ctx, n_slots]
    // The returned tensor is a new tensor in the compute context, populated via cpy
    // Requires graph gf to append copy operations
    ggml_tensor* gather_k(ggml_context* ctx, ggml_cgraph* gf, int32_t il, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv);
    ggml_tensor* gather_v(ggml_context* ctx, ggml_cgraph* gf, int32_t il, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv);

    uint32_t get_n_ctx_max() const { return n_ctx_max; }

    ggml_tensor* get_k_cache_tensor(int layer) { return k_cache[layer]; }
    ggml_tensor* get_v_cache_tensor(int layer) { return v_cache[layer]; }


private:
    const uint32_t n_layers;
    const uint32_t n_ctx_max;
    const uint32_t n_batch_max;
    const uint32_t n_embd_k;
    const uint32_t n_embd_v;
    const ggml_type type_k;
    const ggml_type type_v;
    ggml_backend_t backend;  // Store backend reference

    std::vector<uint32_t> positions;

    // Backend buffer and context
    std::unique_ptr<ggml_backend_buffer, void(*)(ggml_backend_buffer*)> buf;
    std::unique_ptr<ggml_context, void(*)(ggml_context*)> ctx;

    // Persistent cache tensors (allocated in buf)
    std::vector<ggml_tensor*> k_cache;  // One per layer
    std::vector<ggml_tensor*> v_cache;  // One per layer

    // per-layer sharing spec.  Empty when no layer references any
    // other (the standard path).  init_cache() honors this vector when
    // populated.
    std::vector<KvSharingSpec> sharing_;

    void init_cache();
};