#pragma once

#include "layer_state.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "session/section_ids.h"
#include "session/snapshot_section.h"
#include <vector>
#include <memory>
#include <cstdint>

namespace qinf::session {
class SnapshotWriter;
class SnapshotReader;
}  // namespace qinf::session

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

    // --- L2 session snapshot (AppendKV lane) ---
    // Serialize / restore the [0, positions[slot]) K/V span for one slot. The
    // span is a contiguous block per layer (slot rows are contiguous), so this
    // is a per-layer ggml_backend_tensor_get/set — byte-exact on CPU and Metal.
    // Reference (aliased) layers are skipped; their source carries the data and
    // the alias is rebuilt at construction. deserialize validates shape/type
    // fail-loud against this cache and sets the slot cursor. Co-located section:
    // KvCacheSection below.
    void serialize_slot(qinf::session::SnapshotWriter& w, uint32_t slot) const;
    void deserialize_slot(qinf::session::SnapshotReader& r, uint32_t slot);

    // Kernel-path identity of this cache: a hash of the backend (CPU / Metal —
    // the decode kernel path), the K/V dtypes, and the context width. These are
    // exactly the determinants of whether a frozen KV blob is valid to memcpy
    // into and resume here. Feeds CompatHeader::build_path_tag so an L2 blob
    // built under a different path is refused fail-loud rather than producing a
    // divergent resume. See docs/plan-session-snapshot.md (build_path_tag).
    uint64_t path_tag() const;

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

// L2 AppendKV section: serializes one slot's KV span. Borrows the cache.
class KvCacheSection : public qinf::session::SnapshotSection {
public:
    KvCacheSection(simple_kv_cache& kv, uint32_t slot) : kv_(kv), slot_(slot) {}
    qinf::session::SectionId id() const override {
        return qinf::session::kKvCacheSectionId;
    }
    qinf::session::StateLane lane() const override {
        return qinf::session::StateLane::AppendKV;
    }
    void write(qinf::session::SnapshotWriter& w) const override {
        kv_.serialize_slot(w, slot_);
    }
    void read(qinf::session::SnapshotReader& r) override {
        kv_.deserialize_slot(r, slot_);
    }

private:
    simple_kv_cache& kv_;
    uint32_t         slot_;
};