#pragma once

#include "forward_pass_base.h"
#include "../layers/deltanet.h"
#include "../state/deltanet_state.h"
#include "../state/kv_cache_simple.h"
#include "../state/kv_cache_compressed.h"

// Validates the tensor inventory for qwen35 architecture.
// Throws std::runtime_error naming the missing tensor on failure.
void validate_qwen35_inventory(const ModelMetadata& meta);

// Typed config for the qwen35 recipe.  Holds only the fields that are
// family-specific to qwen35 (SSM / DeltaNet, partial-RoPE dimension, and
// the hybrid-attention interval that backs the layer-kind helpers).
// Universal fields (block_count, embedding_length, head counts, RoPE base,
// RMS eps) stay on ModelMetadata and are read directly by the recipe.
//
// The layer-kind methods (is_full_attention_layer / is_ssm_layer) live here
// so the recipe calls cfg_.* rather than touching ModelMetadata.
// ModelMetadata no longer carries these methods; all load-time call sites
// (model.cpp, qwen35.cpp validators) use the inline arithmetic directly.
struct Qwen35Config {
    // SSM / DeltaNet block
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_time_step_rank;
    uint32_t ssm_inner_size;

    // Partial-RoPE dimension (0 → fall back to full head dimension at use sites)
    uint32_t rope_dimension_count;

    // Hybrid-attention scheduling
    uint32_t full_attention_interval;

    // Layer-kind helpers — identical semantics to ModelMetadata::is_*_layer
    // but self-contained so the recipe does not need to touch the metadata.
    bool is_full_attention_layer(uint32_t il) const {
        if (full_attention_interval == 0) return true;
        return (il % full_attention_interval) == (full_attention_interval - 1);
    }
    bool is_ssm_layer(uint32_t il) const { return !is_full_attention_layer(il); }

    // Factory: copies family-specific fields from meta and validates
    // qwen35-specific invariants.  Throws std::runtime_error on violation
    // following the fail-loud contract: field name, expected, actual.
    static Qwen35Config from_metadata(const ModelMetadata& meta);
};

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
    Qwen35ForwardPass(const Model& model, const ModelMetadata* metadata,
                      uint32_t context_len, uint32_t max_batch_size = 1,
                      int kv_quant_bits = 0);
    ~Qwen35ForwardPass() override = default;

    // --- Graph building ---
    struct ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0) override;

    // --- Input setting ---
    void set_inputs(ggml_cgraph* gf,
                    const std::vector<int32_t>& tokens,
                    int pos) override;

    ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) override;

    void set_batched_inputs(ggml_cgraph* gf,
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) override;

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
        if (dn_state_) dn_state_->clear_slot(slot_idx);
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
        if (dn_state_) dn_state_->clone_slot(src_slot, dst_slot);
        _tq_invalidate_watermarks(dst_slot);  // scratch for dst is stale
    }

    // --- Accessors ---
    simple_kv_cache* get_kv_cache_ptr() { return kv_cache_.get(); }
    DeltaNetState*   get_dn_state_ptr() { return dn_state_.get(); }
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
    Qwen35Config cfg_;  // family-specific config, derived from ModelMetadata at construction

    std::unique_ptr<simple_kv_cache> kv_cache_;       // attention layers (null when TQ enabled)
    std::unique_ptr<DeltaNetState>   dn_state_;       // DeltaNet recurrent state (always present)
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

};