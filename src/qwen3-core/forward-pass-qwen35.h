#pragma once

#include "forward-pass-base.h"
#include "../kv-cache/simple-kv-cache.h"
#include "../kv-cache/ssm-state-cache.h"
#include "../kv-cache/compressed_kv_store.h"

/**
 * Forward pass for Qwen3.5 hybrid SSM/attention architecture.
 *
 * 24 layers: 18 GatedDeltaNet (SSM) + 6 full softmax attention.
 * Full attention at every 4th layer (3, 7, 11, 15, 19, 23).
 *
 * Key differences from Qwen2/3:
 *   - Joint Q+Gate projection in attention layers (strided view extraction)
 *   - Gate sigmoid gating after attention output
 *   - Partial RoPE (64 of 256 dims)
 *   - KV cache only for 6 attention layers (not 24)
 *   - Fixed-size SSM recurrent state + conv state for 18 SSM layers
 *   - Weight tying (no output.weight — LM head reuses token_embd)
 *   - post_attention_norm instead of ffn_norm
 */
class Qwen35ForwardPass : public ForwardPassBase {
public:
    Qwen35ForwardPass(const Qwen3Model& model, const Qwen3Metadata* metadata,
                      uint32_t context_len, uint32_t max_batch_size = 1,
                      int kv_quant_bits = 0);
    ~Qwen35ForwardPass() override = default;

    // --- Graph building ---
    struct ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0) override;

    // --- Input setting ---
    void set_inputs(ggml_cgraph* gf,
                    const std::vector<int32_t>& tokens,
                    int pos) override;

    // --- Cache management (delegates to KV, SSM, and TQ caches) ---
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        if (tq_store_) tq_store_->advance(slot_idx, n_tokens);
        else if (kv_cache_) kv_cache_->advance(n_tokens, slot_idx);
        snapkv_advance_seq_pos(slot_idx, n_tokens);
        // SSM state is updated in-graph, no manual advance needed
    }

    void clear_slot(uint32_t slot_idx) override {
        if (tq_store_) tq_store_->clear_slot(slot_idx);  // resets position + data
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
        if (ssm_cache_) ssm_cache_->clear_slot(slot_idx);
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
        if (ssm_cache_) ssm_cache_->clone_slot(src_slot, dst_slot);
        _tq_invalidate_watermarks(dst_slot);  // scratch for dst is stale
    }


    ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) override;

    void set_batched_inputs(ggml_cgraph* gf,
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) override;

    // --- Accessors ---
    simple_kv_cache* get_kv_cache_ptr() { return kv_cache_.get(); }
    ssm_state_cache* get_ssm_cache_ptr() { return ssm_cache_.get(); }
    CompressedKVStore* get_tq_store() { return tq_store_.get(); }

    // ── TurboQuant per-layer compute ────────────────────────────
    std::vector<float> run_prefill(
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx,
        ggml_backend_sched_t scheduler) override;

    // Physical layer → cache index mappings
    int32_t get_kv_layer_index(uint32_t physical_layer) const {
        return (physical_layer < kv_layer_map_.size()) ? kv_layer_map_[physical_layer] : -1;
    }
    int32_t get_ssm_layer_index(uint32_t physical_layer) const {
        return (physical_layer < ssm_layer_map_.size()) ? ssm_layer_map_[physical_layer] : -1;
    }

private:
    std::unique_ptr<simple_kv_cache> kv_cache_;       // attention layers (null when TQ enabled)
    std::unique_ptr<ssm_state_cache> ssm_cache_;      // SSM layers (always present)
    std::unique_ptr<CompressedKVStore> tq_store_;     // TurboQuant compressed store (optional)
    std::unique_ptr<simple_kv_cache> tq_scratch_cache_; // persistent F32 scratch for TQ (n_kv_layers)

    // Per-KV-layer watermark: tq_scratch_valid_pos_[kv_idx][slot] = number of tokens
    // already decompressed into the persistent scratch for this (kv_layer, slot).
    std::vector<std::vector<uint32_t>> tq_scratch_valid_pos_;

    void _tq_invalidate_watermarks(uint32_t slot_idx) {
        for (auto& per_slot : tq_scratch_valid_pos_)
            if (slot_idx < per_slot.size()) per_slot[slot_idx] = 0;
    }

    // Physical layer index → cache layer index (-1 = not this cache type)
    std::vector<int32_t> kv_layer_map_;    // [24] → 0..5 or -1
    std::vector<int32_t> ssm_layer_map_;   // [24] → 0..17 or -1

    // --- Qwen35-specific attention builder (single-slot prefill) ---
    ggml_tensor* build_attention_layer(
        ggml_cgraph* gf,
        simple_kv_cache* kv_cache,
        ggml_tensor* cur,         // [n_embd, n_tokens] — normed input
        ggml_tensor* inp_pos,     // [n_tokens] — position IDs
        int kv_cache_layer,       // mapped KV cache index (0..5)
        uint32_t n_tokens,
        uint32_t slot_idx,
        int il);                  // physical layer index (for tensor names)

    // --- SSM layer builder (single-slot prefill) ---
    ggml_tensor* build_ssm_layer(
        ggml_cgraph* gf,
        ggml_tensor* cur,         // [n_embd, n_tokens] — normed input
        int ssm_cache_layer,      // mapped SSM cache index (0..17)
        uint32_t n_tokens,
        uint32_t slot_idx,
        int il);

    // Delta net: single token recurrence step (used by prefill path)
    // S:[d,d,H,1] q/k/v:[d,H,1,1] gate/beta:[1,H,1,1]
    // Returns output [d,H,1,1], *S_out = new state [d,d,H,1]
    ggml_tensor* build_delta_net_step(ggml_cgraph* gf,
        ggml_tensor* S, ggml_tensor* q_t, ggml_tensor* k_t, ggml_tensor* v_t,
        ggml_tensor* gate_t, ggml_tensor* beta_t,
        ggml_tensor** S_out, int il);

    // ── TurboQuant per-layer / batch helpers ─────────────────────
    // Single-layer graph (kept for reference; run_prefill uses the batch variant).
    ggml_cgraph* _build_qwen35_layer_graph(uint32_t il, uint32_t n_tokens, uint32_t slot_idx,
                                            uint32_t scratch_layer = 0);

    // Batch graph: builds [il_start, il_end) layers in one ggml context.
    ggml_cgraph* _build_qwen35_batch_graph(uint32_t il_start, uint32_t il_end,
                                            uint32_t n_tokens, uint32_t slot_idx);

    // Decompress KV history for one attention layer from tq_store_ → scratch slot scratch_layer.
    void _tq_decompress_attn_layer(uint32_t kv_idx, uint32_t slot_idx,
                                   uint32_t scratch_layer = 0);

    // Compress newly written KV tokens from scratch slot scratch_layer → tq_store_.
    void _tq_compress_attn_layer(uint32_t kv_idx, uint32_t slot_idx, uint32_t pos,
                                  uint32_t n_tokens, uint32_t scratch_layer = 0);

    // --- Batched decode builders (multi-slot, 1 token per slot) ---

    // Batched attention: joint Q+Gate, sigmoid gating, gathered KV
    ggml_tensor* build_batched_attention_layer(
        ggml_cgraph* gf,
        ggml_tensor* cur,         // [n_embd, n_batch]
        ggml_tensor* inp_pos,     // [n_batch]
        ggml_tensor* kq_mask,     // [n_kv_len, 1, 1, n_batch]
        ggml_tensor* gather_indices, // [n_batch * n_kv_len]
        int kv_cache_layer,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions,
        int il);

    // Batched SSM: batched projections, per-slot conv + recurrence
    ggml_tensor* build_batched_ssm_layer(
        ggml_cgraph* gf,
        ggml_tensor* cur,         // [n_embd, n_batch]
        int ssm_cache_layer,
        const std::vector<uint32_t>& slots,
        int il);
};