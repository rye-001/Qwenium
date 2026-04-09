#pragma once

#include "forward-pass-base.h"
#include "../kv-cache/simple-kv-cache.h"
#include "../kv-cache/compressed_kv_store.h"

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
    }

    void clear_slot(uint32_t slot_idx) override {
        if (tq_store_) tq_store_->clear_slot(slot_idx);  // resets position + compressed data
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
    }

    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        if (tq_store_) tq_store_->set_pos(slot_idx, pos);
        else if (kv_cache_) kv_cache_->set_pos(pos, slot_idx);
    }

    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        if (tq_store_) return tq_store_->get_pos(slot_idx);
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override {
        if (tq_store_) tq_store_->clone_slot(src_slot, dst_slot, n_tokens);  // copies pos too
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
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
    std::unique_ptr<simple_kv_cache> tq_scratch_cache_;  // 1-layer F32 for per-layer compute

    // Pre-computed RoPE tables
    std::vector<float> rope_cos_cached_;
    std::vector<float> rope_sin_cached_;

    // --- Qwen2/3-specific attention builders ---
    ggml_tensor* _build_attention_layer(
        ggml_cgraph* gf,
        simple_kv_cache* kv_cache,
        ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
        int layer_idx, float kq_scale,
        uint32_t n_tokens, uint32_t slot_idx, int il);

    // ── Per-layer graph builders (TurboQuant Phase 2) ──────────
    ggml_cgraph* _build_single_layer_graph(
        uint32_t il, const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx, uint32_t n_tokens);

    ggml_cgraph* _build_output_head_graph();

    // Decompress full KV history for one layer from store → F32 cache (layer 0)
    void _tq_decompress_layer(uint32_t layer, uint32_t slot_idx);

    // Compress newly written KV tokens from F32 cache (layer 0) → store
    void _tq_compress_new(uint32_t layer, uint32_t slot_idx, uint32_t pos, uint32_t n_tokens);

    // Scratch buffer for inter-layer data transfer
    std::vector<float> tq_layer_scratch_;

    ggml_tensor* _build_batched_attention_layer(
        ggml_cgraph* gf,
        simple_kv_cache* kv_cache,
        ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
        int layer_idx, float kq_scale,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions,
        ggml_tensor* kq_mask,
        ggml_tensor* gather_indices, int il);
};