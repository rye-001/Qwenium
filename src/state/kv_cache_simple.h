#pragma once

#include "layer_state.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "session/section_ids.h"
#include "session/snapshot_section.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <string>

namespace qinf::session {
class SnapshotWriter;
class SnapshotReader;
}  // namespace qinf::session

// ── KV element type vocabulary (--kv-type) ───────────────────────────────────
// Which ggml types an attention KV cache may be stored as, and which of those
// are quantized. This lives here rather than in a front end because the cache
// is what the type is a property OF: `path_tag()` folds type_k/type_v and the
// snapshot header round-trips them, so a blob captured under one element type
// is already refused fail-loud under another. Adding a type here is therefore
// the whole change on the state side.
//
// F32 is the default and the historical, byte-identical behaviour. F16 halves
// KV bytes. Q8_0/Q4_0 quarter and eighth them, and are the lever on the
// ctx x slots axis that the workload envelope is written in.
//
// THE QUANTIZED TYPES ARE FLASH-ONLY, and that is a hard structural fact, not a
// policy preference: the materialized attention path transposes V with
// ggml_permute(v, 1,2,0,3) followed by ggml_cont (src/layers/attention.cpp),
// which moves ne[0] -- the block dimension of every quantized type -- out of
// position. Metal's CPY/CONT accepts only F32 and F16 sources, so a quantized
// cont is not supported and would take ggml's SILENT CPU fallback. ggml's
// flash_attn_ext consumes Q8_0/Q4_0 K/V natively (K and V must be the SAME
// type) and needs no transpose at all. Callers must refuse a quantized KV
// without flash attention rather than let it degrade quietly.
enum class KvTypeSupport { Ok, Unknown };

// Parse a --kv-type spelling. Returns Unknown (and leaves `out` untouched) for
// anything not listed; the caller owns the error text so it can name its own
// flag. Accepted: f32, f16, q8_0, q4_0.
inline KvTypeSupport kv_type_from_string(const std::string& s, ggml_type* out) {
    if (s == "f32")  { *out = GGML_TYPE_F32;  return KvTypeSupport::Ok; }
    if (s == "f16")  { *out = GGML_TYPE_F16;  return KvTypeSupport::Ok; }
    if (s == "q8_0") { *out = GGML_TYPE_Q8_0; return KvTypeSupport::Ok; }
    if (s == "q4_0") { *out = GGML_TYPE_Q4_0; return KvTypeSupport::Ok; }
    return KvTypeSupport::Unknown;
}

// The accepted spellings, for error messages and --help. One source, so a new
// type cannot be added to the parser and forgotten in the diagnostics.
inline const char* kv_type_choices() { return "f32|f16|q8_0|q4_0"; }

// True for the element types the materialized attention path cannot read.
inline bool kv_type_is_quantized(ggml_type t) {
    return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
}

inline const char* kv_type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_0: return "q4_0";
        default:             return "unknown";
    }
}

// The refusal for a quantized KV cache without flash attention. Fail-loud
// contract order: parameter, expected, actual.
inline std::string kv_type_requires_flash_refusal(ggml_type t) {
    return std::string("--kv-type ") + kv_type_name(t) +
           ": expected --flash-attn to be enabled as well (a quantized KV cache "
           "is readable only by the flash attention kernel; the materialized "
           "path transposes V, which a quantized type cannot represent), "
           "actual: --flash-attn is off";
}

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

    // Value-driven batched write (decode path): scatter k_cur's rows into the
    // cache via ggml_set_rows. Destination row indices arrive as a graph INPUT
    // (I64, one entry per batch row, row = slot_idx * n_ctx_max + position) so
    // the write position is a run-time value, not a baked view offset — unlike
    // cpy_k/cpy_v this write survives graph reuse (persistent decode graph,
    // docs/plan-persistent-decode-graph.md §2.1). Range enforcement lives in
    // the index setter (KvWriteIndicesInput), fail-loud per step.
    // k_cur/v_cur: F32 [n_embd, n_batch]. row_indices: I64 [n_batch].
    ggml_tensor * set_rows_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il, ggml_tensor * row_indices);
    ggml_tensor * set_rows_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il, ggml_tensor * row_indices);

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

    // Gather from an EXPLICIT rows-view source instead of the raw cache tensor.
    // The set_rows write returns a view of the whole cache (rows shape); passing
    // that view here makes the gather a data-DEPENDENT of the write, so the
    // scheduler orders write-before-read (read-after-write edge). Used by the
    // set_rows decode write path when MORE THAN ONE slot is active; the
    // single-slot case takes gather_k_single instead (2026-08-30).
    // src_rows: [n_embd, n_ctx_max*n_batch_max].
    //
    // This comment used to add that reading the raw cache "is safe only when
    // node insertion order happens to serialize them, which bucketed decode
    // graphs break." That is stronger than what is observed: the persistent
    // (set_rows + bucketed) shape reading through a plain view is token-identical
    // to this path with the Metal graph-optimize pass BOTH enabled and disabled,
    // on qwen35 and gemma3, and over 300 steps of real graph reuse across a
    // bucket boundary — a reordering hazard would have shown up as a difference
    // between those. The explicit edge is kept here anyway: it costs the
    // multi-slot path nothing, and an edge the compiler can see beats an
    // ordering argument that depends on one backend's optimizer.
    ggml_tensor* gather_k_from(ggml_context* ctx, ggml_tensor* src_rows, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv);
    ggml_tensor* gather_v_from(ggml_context* ctx, ggml_tensor* src_rows, ggml_tensor* indices, uint32_t n_active, uint32_t n_kv);

    // Single-slot KV read: [n_embd, n_kv, 1] — the identity-gather fast path.
    //
    // gather_k builds indices[t] = slot * n_ctx_max + t. With exactly ONE
    // active slot that is a single contiguous run, so the "gather" copies the
    // whole K/V history into scratch only to reproduce the layout it already
    // had. ggml_view_2d expresses the same rows for free: same values, same
    // contiguous strides, so the consuming mul_mat sees a bit-identical
    // operand and the GET_ROWS dispatch disappears. Worth ~0.97 ms/step on
    // Qwen3.5-0.8B at n_kv 756 — see docs/decode-gap-status.md §4.
    //
    // Usable on BOTH write paths, including set_rows. This carries no data edge
    // to the write, because ggml_view of a view re-points view_src at the
    // underlying cache — write-before-read rests instead on the write being
    // build_forward_expand'd first PLUS the Metal backend's memory-range
    // analysis, which sees the view and the write aliasing the same cache
    // bytes. Both backend passes honour that: the encode-time concurrency check
    // emits a barrier for overlapping ranges, and the reorder pass refuses to
    // hoist a node past unprocessed nodes it overlaps (and does not treat
    // SET_ROWS as reorderable at all). llama.cpp relies on exactly this — its
    // get_k is a bare ggml_view_4d of the cache sitting next to a set_rows
    // write. Gated by KvWriteSetRows.IdentityGatherOrdersAfterTheWrite, which
    // runs the persistent shape with graph-optimize enabled AND disabled: a
    // reordering bug would show up as a difference between the two.
    ggml_tensor* gather_k_single(ggml_context* ctx, int32_t il, uint32_t slot, uint32_t n_kv);
    ggml_tensor* gather_v_single(ggml_context* ctx, int32_t il, uint32_t slot, uint32_t n_kv);

    uint32_t get_n_ctx_max() const { return n_ctx_max; }
    uint32_t get_n_layers() const { return n_layers; }

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
    // the kernel path), the K/V dtypes, the context width, and any salt the
    // owner has stamped in. These are the determinants of whether a frozen KV
    // blob is valid to memcpy into and resume here. Feeds
    // CompatHeader::build_path_tag so an L2 blob built under a different path is
    // refused fail-loud rather than producing a divergent resume. See
    // docs/plan-session-snapshot.md (build_path_tag).
    uint64_t path_tag() const;

    // Fold an owner-defined value into path_tag(). The cache does not interpret
    // it: it exists because the bytes in a KV cache depend on more than the
    // cache's own configuration. --flash-attn is the case that forced it —
    // flash changes the attention output, hence the residual stream, hence
    // every later layer's K/V, so a prefill done under flash is NOT
    // interchangeable with one done materialized. Stamped by
    // ForwardPassBase::set_attn_impl so no snapshot call site can forget it.
    void set_path_salt(uint64_t salt) { path_salt_ = salt; }

private:
    uint64_t path_salt_ = 0;

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