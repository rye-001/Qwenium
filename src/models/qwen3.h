#pragma once

#include "forward_pass_base.h"
#include "../state/kv_cache_simple.h"
#include "../state/kv_cache_compressed.h"

/**
 * Forward pass for Qwen2 and Qwen3 architectures.
 * 
 * All layers are uniform transformer blocks with standard softmax attention.
 * KV cache spans all layers with identity index mapping.
 * 
 * This class preserves the exact same public API as the original
 * Qwen3ForwardPass — existing call sites (CLI, server, tests) need
 * no changes.
 */
class Qwen3ForwardPass : public ForwardPassBase {
    // Grant the unit test access to private members for validation.
    friend class RopeCorrectnessTest_ApplyRopeMatchesGoldenValues_Test;
    friend class GQAAttentionCorrectnessTest_GQAAttentionMatchesGoldenValues_Test;
public:
    explicit Qwen3ForwardPass(const Qwen3Model& model, const Qwen3Metadata* metadata, uint32_t context_len, uint32_t max_batch_size = 1, int kv_quant_bits = 0);
    ~Qwen3ForwardPass() override = default;

    // --- Graph building ---
    struct ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0) override;

    struct ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions
    ) override;

    // --- Input setting (after graph allocation) ---
    void set_inputs(ggml_cgraph* gf,
                    const std::vector<int32_t>& tokens,
                    int pos) override;

    void set_batched_inputs(ggml_cgraph* gf,
                    const std::vector<int32_t>& tokens,
                    const std::vector<uint32_t>& slots,
                    const std::vector<int32_t>& positions) override;

    // --- Cache management ---
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        if (tq_store_) tq_store_->advance(slot_idx, n_tokens);
        else if (kv_cache_) kv_cache_->advance(n_tokens, slot_idx);
        snapkv_advance_seq_pos(slot_idx, n_tokens);
    }

    void clear_slot(uint32_t slot_idx) override {
        if (tq_store_) tq_store_->clear_slot(slot_idx);  // resets position + compressed data
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
        _tq_invalidate_watermarks(slot_idx);
        snapkv_clear_seq_pos(slot_idx);
    }

    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        if (tq_store_) tq_store_->set_pos(slot_idx, pos);
        else if (kv_cache_) kv_cache_->set_pos(pos, slot_idx);
        _tq_invalidate_watermarks(slot_idx);
    }

    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        // After SnapKV, return logical sequence position (for RoPE)
        uint32_t seq = snapkv_get_seq_pos(slot_idx);
        if (seq > 0) return seq;
        if (tq_store_) return tq_store_->get_pos(slot_idx);
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    uint32_t get_physical_cache_pos(uint32_t slot_idx) const override {
        if (tq_store_) return tq_store_->get_pos(slot_idx);
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override {
        if (tq_store_) tq_store_->clone_slot(src_slot, dst_slot, n_tokens);  // copies pos too
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
        _tq_invalidate_watermarks(dst_slot);  // scratch for dst is stale
    }

    simple_kv_cache* get_kv_cache_ptr() { return kv_cache_.get(); }

    // ── TurboQuant ──────────────────────────────────────────────
    CompressedKVStore* get_tq_store() { return tq_store_.get(); }

    std::vector<float> run_prefill(
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx,
        ggml_backend_sched_t scheduler) override;

private:
    std::unique_ptr<simple_kv_cache> kv_cache_;
    std::unique_ptr<CompressedKVStore> tq_store_;
    std::unique_ptr<simple_kv_cache> tq_scratch_cache_;  // B-layer F32 scratch (one slot per attention layer in a batch)

    // Pre-computed RoPE tables
    std::vector<float> rope_cos_cached_;
    std::vector<float> rope_sin_cached_;

    // Attention subgraph construction is in src/layers/attention.cpp.

    // ── Per-layer / batched graph builders (TurboQuant Phase 2) ───
    // Batch builder: builds [il_start, il_end) layers in one ggml context.
    // Each attention layer in the batch uses scratch cache slot scratch_idx (0-based).
    ggml_cgraph* _build_layer_batch_graph(
        uint32_t il_start, uint32_t il_end,
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx, uint32_t n_tokens);

    ggml_cgraph* _build_output_head_graph();

    // Decompress full KV history for one model layer → scratch cache layer scratch_layer.
    void _tq_decompress_layer(uint32_t layer, uint32_t slot_idx,
                              uint32_t scratch_layer = 0);

    // Compress newly written KV tokens from scratch cache layer scratch_layer → store.
    void _tq_compress_new(uint32_t layer, uint32_t slot_idx,
                          uint32_t pos, uint32_t n_tokens,
                          uint32_t scratch_layer = 0);

    // Scratch buffer for inter-layer data transfer
    std::vector<float> tq_layer_scratch_;

    // Per-layer watermark: tq_scratch_valid_pos_[layer][slot] = number of tokens
    // already decompressed into the persistent scratch for this (layer, slot).
    // Enables incremental (delta) decompression: only [watermark..pos) is uploaded.
    std::vector<std::vector<uint32_t>> tq_scratch_valid_pos_;

    void _tq_invalidate_watermarks(uint32_t slot_idx) {
        for (auto& per_slot : tq_scratch_valid_pos_)
            if (slot_idx < per_slot.size()) per_slot[slot_idx] = 0;
    }

};