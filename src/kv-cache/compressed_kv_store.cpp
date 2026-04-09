#include "compressed_kv_store.h"
#include "turboquant.h"

#include <algorithm>
#include <cassert>
#include <cstring>

CompressedKVStore::CompressedKVStore(
    uint32_t n_layers, uint32_t n_slots, uint32_t n_ctx_max,
    uint32_t n_embd_k, uint32_t n_embd_v,
    uint32_t head_dim, int bits, int block_size)
    : n_layers_(n_layers), n_slots_(n_slots), n_ctx_max_(n_ctx_max),
      n_embd_k_(n_embd_k), n_embd_v_(n_embd_v),
      head_dim_(head_dim), bits_(bits), block_size_(block_size)
{
    assert(head_dim > 0 && (head_dim & (head_dim - 1)) == 0
           && "head_dim must be a power of 2");
    assert(n_embd_k % head_dim == 0);
    assert(n_embd_v % head_dim == 0);
    assert(bits >= 2 && bits <= 4);

    n_heads_k_ = n_embd_k / head_dim;
    n_heads_v_ = n_embd_v / head_dim;

    // Compressed size per head, then per token
    size_t bytes_per_head = turboquant::compressed_size(head_dim, bits, block_size);
    bytes_per_token_k_ = bytes_per_head * n_heads_k_;
    bytes_per_token_v_ = bytes_per_head * n_heads_v_;

    // Allocate storage: one contiguous buffer per layer
    size_t layer_k_bytes = static_cast<size_t>(n_slots) * n_ctx_max * bytes_per_token_k_;
    size_t layer_v_bytes = static_cast<size_t>(n_slots) * n_ctx_max * bytes_per_token_v_;

    compressed_k_.resize(n_layers);
    compressed_v_.resize(n_layers);
    for (uint32_t l = 0; l < n_layers; ++l) {
        compressed_k_[l].assign(layer_k_bytes, 0);
        compressed_v_[l].assign(layer_v_bytes, 0);
    }

    positions_.assign(n_slots, 0);
}

// ── Offset calculation ───────────────────────────────────────────────────────

size_t CompressedKVStore::token_offset_k(uint32_t slot, uint32_t pos) const {
    return (static_cast<size_t>(slot) * n_ctx_max_ + pos) * bytes_per_token_k_;
}

size_t CompressedKVStore::token_offset_v(uint32_t slot, uint32_t pos) const {
    return (static_cast<size_t>(slot) * n_ctx_max_ + pos) * bytes_per_token_v_;
}

// ── Per-head compress/decompress ─────────────────────────────────────────────

void CompressedKVStore::compress_token(
    const float* src, uint8_t* dst,
    uint32_t n_embd, uint32_t n_heads) const
{
    size_t bytes_per_head = turboquant::compressed_size(head_dim_, bits_, block_size_);

    // Scratch buffer for one head (WHT modifies in-place)
    std::vector<float> scratch(head_dim_);

    for (uint32_t h = 0; h < n_heads; ++h) {
        std::memcpy(scratch.data(), src + h * head_dim_,
                    head_dim_ * sizeof(float));
        turboquant::compress(scratch.data(), dst + h * bytes_per_head,
                             head_dim_, bits_, block_size_);
    }
}

void CompressedKVStore::decompress_token(
    const uint8_t* src, float* dst,
    uint32_t n_embd, uint32_t n_heads) const
{
    size_t bytes_per_head = turboquant::compressed_size(head_dim_, bits_, block_size_);

    for (uint32_t h = 0; h < n_heads; ++h) {
        turboquant::decompress(src + h * bytes_per_head,
                               dst + h * head_dim_,
                               head_dim_, bits_, block_size_);
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

void CompressedKVStore::compress_token_k(
    uint32_t layer, uint32_t slot, uint32_t pos, const float* src)
{
    assert(layer < n_layers_ && slot < n_slots_ && pos < n_ctx_max_);
    size_t off = token_offset_k(slot, pos);
    compress_token(src, compressed_k_[layer].data() + off, n_embd_k_, n_heads_k_);
}

void CompressedKVStore::compress_token_v(
    uint32_t layer, uint32_t slot, uint32_t pos, const float* src)
{
    assert(layer < n_layers_ && slot < n_slots_ && pos < n_ctx_max_);
    size_t off = token_offset_v(slot, pos);
    compress_token(src, compressed_v_[layer].data() + off, n_embd_v_, n_heads_v_);
}

void CompressedKVStore::decompress_token_k(
    uint32_t layer, uint32_t slot, uint32_t pos, float* dst) const
{
    assert(layer < n_layers_ && slot < n_slots_ && pos < n_ctx_max_);
    size_t off = token_offset_k(slot, pos);
    decompress_token(compressed_k_[layer].data() + off, dst, n_embd_k_, n_heads_k_);
}

void CompressedKVStore::decompress_token_v(
    uint32_t layer, uint32_t slot, uint32_t pos, float* dst) const
{
    assert(layer < n_layers_ && slot < n_slots_ && pos < n_ctx_max_);
    size_t off = token_offset_v(slot, pos);
    decompress_token(compressed_v_[layer].data() + off, dst, n_embd_v_, n_heads_v_);
}

void CompressedKVStore::clone_slot(
    uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens)
{
    assert(src_slot < n_slots_ && dst_slot < n_slots_ && n_tokens <= n_ctx_max_);

    for (uint32_t l = 0; l < n_layers_; ++l) {
        // K
        size_t src_off_k = token_offset_k(src_slot, 0);
        size_t dst_off_k = token_offset_k(dst_slot, 0);
        size_t n_bytes_k = static_cast<size_t>(n_tokens) * bytes_per_token_k_;
        std::memcpy(compressed_k_[l].data() + dst_off_k,
                    compressed_k_[l].data() + src_off_k, n_bytes_k);

        // V
        size_t src_off_v = token_offset_v(src_slot, 0);
        size_t dst_off_v = token_offset_v(dst_slot, 0);
        size_t n_bytes_v = static_cast<size_t>(n_tokens) * bytes_per_token_v_;
        std::memcpy(compressed_v_[l].data() + dst_off_v,
                    compressed_v_[l].data() + src_off_v, n_bytes_v);
    }

    // Copy position counter so the destination slot is ready for decode.
    positions_[dst_slot] = positions_[src_slot];
}

void CompressedKVStore::clear_slot(uint32_t slot) {
    assert(slot < n_slots_);

    for (uint32_t l = 0; l < n_layers_; ++l) {
        size_t off_k = token_offset_k(slot, 0);
        size_t n_bytes_k = static_cast<size_t>(n_ctx_max_) * bytes_per_token_k_;
        std::memset(compressed_k_[l].data() + off_k, 0, n_bytes_k);

        size_t off_v = token_offset_v(slot, 0);
        size_t n_bytes_v = static_cast<size_t>(n_ctx_max_) * bytes_per_token_v_;
        std::memset(compressed_v_[l].data() + off_v, 0, n_bytes_v);
    }

    positions_[slot] = 0;
}

// ── Position tracking ────────────────────────────────────────────────────────

void CompressedKVStore::advance(uint32_t slot, uint32_t n_tokens) {
    assert(slot < n_slots_);
    positions_[slot] += n_tokens;
}

void CompressedKVStore::set_pos(uint32_t slot, uint32_t pos) {
    assert(slot < n_slots_);
    positions_[slot] = pos;
}

uint32_t CompressedKVStore::get_pos(uint32_t slot) const {
    assert(slot < n_slots_);
    return positions_[slot];
}

size_t CompressedKVStore::total_compressed_bytes() const {
    size_t per_layer = static_cast<size_t>(n_slots_) * n_ctx_max_
                       * (bytes_per_token_k_ + bytes_per_token_v_);
    return per_layer * n_layers_;
}
