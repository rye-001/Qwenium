#pragma once

#include "forward_pass_base.h"
#include "../layers/deltanet.h"
#include "../state/deltanet_state.h"
#include "../state/kv_cache_simple.h"

// Validates the tensor inventory for qwen35 architecture.
// Throws std::runtime_error naming the missing tensor on failure.
void validate_qwen35_inventory(const ModelMetadata& meta);

// Typed config for the qwen35 recipe.  Holds only the fields that are
// family-specific to qwen35 (SSM / DeltaNet, partial-RoPE dimension, and
// the hybrid-attention interval that backs the layer-kind helpers).
// Universal fields (block_count, embedding_length, head counts, RoPE base,
// RMS eps) stay on ModelMetadata and are read directly by the recipe.
struct Qwen35Config {
    // SSM / DeltaNet block
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_time_step_rank;
    uint32_t ssm_inner_size;

    // Partial-RoPE dimension (0 -> fall back to full head dimension at use sites)
    uint32_t rope_dimension_count;

    // Hybrid-attention scheduling
    uint32_t full_attention_interval;

    bool is_full_attention_layer(uint32_t il) const {
        if (full_attention_interval == 0) return true;
        return (il % full_attention_interval) == (full_attention_interval - 1);
    }
    bool is_ssm_layer(uint32_t il) const { return !is_full_attention_layer(il); }

    // Factory: copies family-specific fields from meta and validates
    // qwen35-specific invariants.  Throws std::runtime_error on violation
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
                      uint32_t context_len, uint32_t max_batch_size = 1);
    ~Qwen35ForwardPass() override = default;

    // --- Graph building ---
    struct ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0, bool want_logits = true) override;

    ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) override;

    // qwen35 honors want_logits=false with one head-guard site in
    // build_prefill_graph.
    bool feed_tokens_supported() const override { return true; }

    // --- Cache management (delegates to KV and SSM caches) ---
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->advance(n_tokens, slot_idx);
    }

    void clear_slot(uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
        if (dn_state_) dn_state_->clear_slot(slot_idx);
    }

    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->set_pos(pos, slot_idx);
    }

    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override {
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
        if (dn_state_) dn_state_->clone_slot(src_slot, dst_slot);
    }

    // --- Accessors ---
    simple_kv_cache* get_kv_cache_ptr() { return kv_cache_.get(); }
    DeltaNetState*   get_dn_state_ptr() { return dn_state_.get(); }

    // Physical layer → cache index mappings
    int32_t get_kv_layer_index(uint32_t physical_layer) const {
        return (physical_layer < kv_layer_map_.size()) ? kv_layer_map_[physical_layer] : -1;
    }
    int32_t get_ssm_layer_index(uint32_t physical_layer) const {
        return (physical_layer < ssm_layer_map_.size()) ? ssm_layer_map_[physical_layer] : -1;
    }

private:
    Qwen35Config cfg_;  // family-specific config, derived from ModelMetadata at construction

    std::unique_ptr<simple_kv_cache> kv_cache_;       // attention layers
    std::unique_ptr<DeltaNetState>   dn_state_;       // DeltaNet recurrent state (always present)

    // Physical layer index → cache layer index (-1 = not this cache type)
    std::vector<int32_t> kv_layer_map_;    // [24] → 0..5 or -1
    std::vector<int32_t> ssm_layer_map_;   // [24] → 0..17 or -1
};