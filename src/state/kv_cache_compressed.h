#pragma once
// kv_cache_compressed.h — TurboQuant compressed backing store for KV cache.
//
// Pure C++, no ggml dependency. Manages compressed byte buffers indexed by
// [layer][slot][position]. Compress/decompress is done per-head using the
// turboquant math library.
//
// Usage:
//   1. After ggml graph compute, read F32 KV from cache tensors
//   2. Call compress_token_k/v() to store compressed copies
//   3. For clone_slot(), memcpy compressed data (fast)
//   4. (Phase 2) Before graph compute, call decompress to restore F32

#include "layer_state.h"
#include <cstddef>
#include <cstdint>
#include <vector>

class CompressedKVStore : public LayerState {
public:
    // head_dim must be a power of 2 (for WHT). n_embd_k = n_heads_kv * head_dim.
    CompressedKVStore(uint32_t n_layers, uint32_t n_slots, uint32_t n_ctx_max,
                      uint32_t n_embd_k, uint32_t n_embd_v,
                      uint32_t head_dim, int bits, int block_size = 32);

    // LayerState interface.
    void   reset_sequence(int seq_id) override { clear_slot(static_cast<uint32_t>(seq_id)); }
    size_t memory_bytes() const override       { return total_compressed_bytes(); }

    // ── Per-token compress/decompress ────────────────────────────────────────
    // src/dst: float[n_embd_k] or float[n_embd_v] for a single token.
    // Applies per-head WHT + Lloyd-Max quantization internally.

    void compress_token_k(uint32_t layer, uint32_t slot, uint32_t pos, const float* src);
    void compress_token_v(uint32_t layer, uint32_t slot, uint32_t pos, const float* src);

    void decompress_token_k(uint32_t layer, uint32_t slot, uint32_t pos, float* dst) const;
    void decompress_token_v(uint32_t layer, uint32_t slot, uint32_t pos, float* dst) const;

    // ── Bulk operations ──────────────────────────────────────────────────────

    // Compact: keep only the positions listed in `retained` (sorted, unique).
    // Moves compressed data so retained[i] becomes position i.
    // Does NOT update the position counter — caller must call set_pos() after.
    void compact(uint32_t layer, uint32_t slot,
                 const std::vector<uint32_t>& retained_positions);

    // Clone all layers' compressed data from src_slot to dst_slot.
    // Also copies the source slot's position counter.
    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens);

    // Zero out a slot's compressed data and reset its position to 0.
    void clear_slot(uint32_t slot);

    // ── Position tracking ─────────────────────────────────────────────────────
    // Replaces simple_kv_cache position management when TQ is the only cache.
    // One counter per slot; independent of compressed data storage.

    void     advance(uint32_t slot, uint32_t n_tokens);
    void     set_pos(uint32_t slot, uint32_t pos);
    uint32_t get_pos(uint32_t slot) const;

    // ── Accessors ────────────────────────────────────────────────────────────

    uint32_t n_layers()  const { return n_layers_; }
    uint32_t n_slots()   const { return n_slots_; }
    uint32_t n_ctx_max() const { return n_ctx_max_; }
    int      bits()      const { return bits_; }

    size_t compressed_bytes_per_token_k() const { return bytes_per_token_k_; }
    size_t compressed_bytes_per_token_v() const { return bytes_per_token_v_; }
    size_t total_compressed_bytes()       const;

private:
    uint32_t n_layers_;
    uint32_t n_slots_;
    uint32_t n_ctx_max_;
    uint32_t n_embd_k_;
    uint32_t n_embd_v_;
    uint32_t head_dim_;
    uint32_t n_heads_k_;
    uint32_t n_heads_v_;
    int      bits_;
    int      block_size_;

    size_t bytes_per_token_k_;  // compressed bytes for one token's K
    size_t bytes_per_token_v_;  // compressed bytes for one token's V

    // Storage: compressed_k_[layer] holds all slots × ctx contiguously.
    // Layout: [slot0_pos0, slot0_pos1, ..., slot0_posN, slot1_pos0, ...]
    std::vector<std::vector<uint8_t>> compressed_k_;  // [n_layers]
    std::vector<std::vector<uint8_t>> compressed_v_;  // [n_layers]

    // Position counters: one uint32_t per slot.
    std::vector<uint32_t> positions_;                 // [n_slots]

    size_t token_offset_k(uint32_t slot, uint32_t pos) const;
    size_t token_offset_v(uint32_t slot, uint32_t pos) const;

    void compress_token(const float* src, uint8_t* dst,
                        uint32_t n_embd, uint32_t n_heads) const;
    void decompress_token(const uint8_t* src, float* dst,
                          uint32_t n_embd, uint32_t n_heads) const;
};
