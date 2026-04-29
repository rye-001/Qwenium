#pragma once
// gemma2.h — Gemma 2 (2B / 9B / 27B) model recipe.
//
// Structural additions over Gemma 1 (each retires one post-G1 assumption):
//   1. Sandwich norm  — post_attention_norm + post_ffw_norm per block.
//   2. Alternating local/global attention — even layers use a 4096-token
//      sliding window; odd layers use full (global) attention.
//   3. Attention logit soft-capping — cap * tanh(QK / cap) before softmax.
//   4. Final logit soft-capping   — cap * tanh(logits / cap) after LM head.
//
// Tokenizer and chat template are inherited from Gemma 1 (same family).
// Inventory validator is Gemma-2-specific (post_attention_norm + post_ffw_norm).
//
// Scope (PR G2.5): prefill path producing logits — sufficient for the canary
// logit-agreement gate.  Decode is stubbed with a clear error until a later PR.

#include "forward_pass_base.h"
#include "../state/kv_cache_simple.h"
#include "../loader/tokenizer_config.h"
#include <vector>

struct ModelMetadata;

// ── Gemma 2 architecture config ───────────────────────────────────────────────
//
// Populated by from_metadata(); no architecture string literals outside this file.
struct Gemma2Config {
    uint32_t n_layers;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_embd_head;   // key / value length per head
    uint32_t hidden_dim;    // embedding_length
    uint32_t context_len;
    float    rms_norm_eps;
    float    freq_base;
    float    attn_softcap;      // gemma2.attn_logit_softcapping  (50.0 for all variants)
    float    final_softcap;     // gemma2.final_logit_softcapping (30.0 for all variants)
    uint32_t sliding_window;    // gemma2.attention.sliding_window (4096 for 2B/9B/27B)

    // Per-layer window size: 0 = global (full attention), > 0 = local (sliding).
    // Gemma 2 alternates even=local / odd=global starting from layer 0.
    std::vector<uint32_t> layer_window;

    // Factory: reads all required fields from GGUF metadata.
    // Throws std::runtime_error (fail-loud contract) for missing/wrong-type keys.
    static Gemma2Config from_metadata(const ModelMetadata& meta);
};

// Validates the tensor inventory for gemma2 architecture.
// Expects the same per-block tensors as Gemma 1 plus:
//   blk.N.post_attention_norm.weight
//   blk.N.post_ffw_norm.weight
// Throws std::runtime_error naming the missing tensor on failure.
void validate_gemma2_inventory(const ModelMetadata& meta);

class Gemma2ForwardPass : public ForwardPassBase {
public:
    Gemma2ForwardPass(const Model& model, const ModelMetadata* metadata,
                      uint32_t context_len, uint32_t max_batch_size = 1,
                      int kv_quant_bits = 0);
    ~Gemma2ForwardPass() override = default;

    ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens,
                                      int pos, uint32_t slot_idx = 0) override;

    ggml_cgraph* build_decoding_graph(const std::vector<int32_t>& tokens,
                                      const std::vector<uint32_t>& slots,
                                      const std::vector<int32_t>& positions) override;

    void set_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens,
                    int pos) override;

    void set_batched_inputs(ggml_cgraph* gf,
                            const std::vector<int32_t>& tokens,
                            const std::vector<uint32_t>& slots,
                            const std::vector<int32_t>& positions) override;

    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->advance(n_tokens, slot_idx);
        snapkv_advance_seq_pos(slot_idx, n_tokens);
    }
    void clear_slot(uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->clear_slot(slot_idx);
        snapkv_clear_seq_pos(slot_idx);
    }
    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override {
        if (kv_cache_) kv_cache_->set_pos(pos, slot_idx);
    }
    uint32_t get_cache_pos(uint32_t slot_idx) const override {
        uint32_t seq = snapkv_get_seq_pos(slot_idx);
        if (seq > 0) return seq;
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }
    uint32_t get_physical_cache_pos(uint32_t slot_idx) const override {
        return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    }
    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override {
        if (kv_cache_) kv_cache_->clone_slot(src_slot, dst_slot, n_tokens);
    }

private:
    Gemma2Config              config_;
    std::unique_ptr<simple_kv_cache> kv_cache_;

    // Per-block post-norm weight pointers (loaded from model context by name).
    std::vector<ggml_tensor*> post_attn_norm_;  // blk.N.post_attention_norm.weight
    std::vector<ggml_tensor*> post_ffn_norm_;   // blk.N.post_ffw_norm.weight

    // Retrieve a named tensor from the model context; throws on missing.
    ggml_tensor* require_tensor(uint32_t il, const char* suffix) const;
};
