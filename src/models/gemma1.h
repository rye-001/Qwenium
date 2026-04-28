#pragma once
// gemma1.h — Gemma 1 (2B / 7B) model recipe.
//
// Also defines gemma1_tokenizer_config() so the registry can wire it in
// without architecture string literals in the CLI layer.
// (GemmaChatTemplate lives in src/loader/chat_template.h.)
//
// Composes existing layer modules + Gemma-specific ops:
//   - build_rms_norm_gemma:  (x / rms(x)) * (1 + w)
//   - build_embed_scale:     embedding * sqrt(hidden_size)  (in fp32)
//   - build_ffn_geglu_tanh:  down(geglu_tanh(gate(x), up(x)))
//   - tied embeddings:       LM head reuses token_embd.weight
//
// No QK-norm, no biases, GQA via standard ggml ops, RoPE base 10K.
//
// Scope (PR G1.5): prefill path producing logits — sufficient for the
// canary logit-agreement gate against the HF reference harness. Decode
// (single-token, batched) is stubbed; chat / streaming generation lives
// behind a `not yet supported` error until PR G2 generalizes the
// attention-kind dispatch.

#include "forward_pass_base.h"
#include "../state/kv_cache_simple.h"
#include "../loader/tokenizer_config.h"

// Validates the tensor inventory for gemma architecture.
// Throws std::runtime_error naming the missing tensor on failure.
void validate_gemma1_inventory(const ModelMetadata& meta);

// TokenizerConfig for Gemma 1 (SentencePiece BPE with byte fallback).
TokenizerConfig gemma1_tokenizer_config();

class Gemma1ForwardPass : public ForwardPassBase {
public:
    Gemma1ForwardPass(const Model& model, const ModelMetadata* metadata,
                      uint32_t context_len, uint32_t max_batch_size = 1,
                      int kv_quant_bits = 0);
    ~Gemma1ForwardPass() override = default;

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
    std::unique_ptr<simple_kv_cache> kv_cache_;
};
