#pragma once
// gemma3.h — Gemma 3 (1B / 4B / 12B / 27B) model recipe.
//
// Structural additions over Gemma 2 (each retires one post-G2 assumption):
//   1. QK-norm: Q and K are RMS-normed per head before the attention dot product.
//      Weight shape is [head_dim], broadcast across all heads.
//      Wired via the existing w.q_norm / w.k_norm slots in TransformerBlockWeights —
//      no new attention module variant.
//   2. Dual RoPE base: local (sliding-window) layers use base 10K; global layers
//      use the base from GGUF (gemma3.rope.freq_base; typically 1e6). Encoded as a
//      per-layer vector; the block loop updates blk_hp.freq_base before each call.
//   3. 5:1 alternation: five local layers then one global, repeating.
//      Reuses the per-layer layer_window infrastructure from Gemma 2.
//   4. No soft-cap: attn_softcap and final_softcap are both absent/zero.
//
// Gemma 3 otherwise inherits from Gemma 2: sandwich norm, GeGLU-tanh FFN,
// Gemma RMS norm (1+w), tied embeddings, no QKV biases.
//
// Scope (PR G3.4): prefill path producing logits — sufficient for the canary
// logit-agreement gate. Decode is stubbed with a clear error, matching G2's
// stub boundary.

#include "forward_pass_base.h"
#include "../state/kv_cache_simple.h"
#include "../loader/tokenizer_config.h"
#include <vector>

struct ModelMetadata;

// ── Gemma 3 architecture config ───────────────────────────────────────────────
//
// Populated by from_metadata(); no architecture string literals outside this file.
struct Gemma3Config {
    uint32_t n_layers;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_embd_head;   // key / value length per head
    uint32_t hidden_dim;    // embedding_length
    uint32_t context_len;
    float    rms_norm_eps;
    float    global_rope_base;  // from GGUF (gemma3.rope.freq_base; 1e6 for all known sizes)
    float    local_rope_base;   // hardcoded 10000.0f per Gemma3 architecture spec
    uint32_t sliding_window;    // gemma3.attention.sliding_window (512 for 1B)

    // Per-layer window size: 0 = global (full attention), >0 = local (sliding).
    // Gemma 3 uses a 5:1 pattern: layer i is global iff (i % 6 == 5).
    std::vector<uint32_t> layer_window;

    // Per-layer RoPE base frequency.
    // local_rope_base for sliding layers, global_rope_base for global layers.
    std::vector<float> layer_rope_base;

    // Factory: reads all required fields from GGUF metadata.
    // Throws std::runtime_error (fail-loud contract) for missing/wrong-type keys.
    static Gemma3Config from_metadata(const ModelMetadata& meta);
};

// Validates the tensor inventory for gemma3 architecture.
// Requires the same per-block tensors as Gemma 2, plus QK-norm weights:
//   blk.N.attn_q_norm.weight
//   blk.N.attn_k_norm.weight
// Unknown tensors (e.g. multimodal vision / mm.* prefixed) are silently ignored.
// Throws std::runtime_error naming the missing tensor on failure.
void validate_gemma3_inventory(const ModelMetadata& meta);

class Gemma3ForwardPass : public ForwardPassBase {
public:
    Gemma3ForwardPass(const Model& model, const ModelMetadata* metadata,
                      uint32_t context_len, uint32_t max_batch_size = 1,
                      int kv_quant_bits = 0);
    ~Gemma3ForwardPass() override = default;

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
    Gemma3Config                    config_;
    std::unique_ptr<simple_kv_cache> kv_cache_;

    // Per-block sandwich-norm weight pointers (G2-inherited).
    std::vector<ggml_tensor*> post_attn_norm_;  // blk.N.post_attention_norm.weight
    std::vector<ggml_tensor*> post_ffn_norm_;   // blk.N.post_ffw_norm.weight

    // Per-block QK-norm weight pointers (G3-new).
    std::vector<ggml_tensor*> q_norm_;  // blk.N.attn_q_norm.weight
    std::vector<ggml_tensor*> k_norm_;  // blk.N.attn_k_norm.weight

    // Retrieve a named tensor from the model context; throws on missing.
    ggml_tensor* require_tensor(uint32_t il, const char* suffix) const;
};
