#pragma once

#include "forward_pass_base.h"
#include "../state/kv_cache_simple.h"
#include "../loader/tokenizer_config.h"

// Validates the tensor inventory for qwen2/qwen3 architectures.
// Throws std::runtime_error naming the missing tensor on failure.
void validate_qwen3_inventory(const ModelMetadata& meta);

// TokenizerConfig shared by all Qwen family architectures (qwen2/3/35/35moe).
TokenizerConfig qwen_tokenizer_config();

/**
 * Forward pass for Qwen2 and Qwen3 architectures.
 * 
 * All layers are uniform transformer blocks with standard softmax attention.
 * KV cache spans all layers with identity index mapping.
 */
class Qwen3ForwardPass : public ForwardPassBase {
    // Grant the unit test access to private members for validation.
    friend class RopeCorrectnessTest_ApplyRopeMatchesGoldenValues_Test;
    friend class GQAAttentionCorrectnessTest_GQAAttentionMatchesGoldenValues_Test;
public:
    explicit Qwen3ForwardPass(const Model& model, const ModelMetadata* metadata, uint32_t context_len, uint32_t max_batch_size = 1,
                              ggml_type kv_type = GGML_TYPE_F32);
    ~Qwen3ForwardPass() override = default;

    // --- Graph building ---
    struct ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0, bool want_logits = true) override;

    struct ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions
    ) override;

    // Flash attention on decode: one causal mask, cast to F16 per graph, and
    // no attention softcap on this recipe — so build_attn_mha's two refusals
    // are both satisfied. Covers both architectures this recipe hosts (qwen2,
    // qwen3); the QKV-bias and QK-norm forks are upstream of attention.
    bool supports_flash_attn() const override { return true; }

    // --- Cache management ---
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->advance(n_tokens, slot_idx);
    }

    void clear_slot(uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
    }

    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->set_pos(pos, slot_idx);
    }

    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }

    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override {
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
    }

    simple_kv_cache* get_kv_cache_ptr() { return kv_cache_.get(); }

private:
    std::unique_ptr<simple_kv_cache> kv_cache_;

    // Pre-computed RoPE tables
    std::vector<float> rope_cos_cached_;
    std::vector<float> rope_sin_cached_;
};