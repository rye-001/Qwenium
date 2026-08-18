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
// Prefill and batched decode are both implemented; decode is gated Tier-1
// bitwise against single-token run_prefill (tests/unit/test_gemma_batched_decode.cpp).

#include "forward_pass_base.h"
#include "i_image_embeddable.h"
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

    // Linear RoPE frequency scale applied to GLOBAL layers (gemma3.rope.scaling
    // .type=linear; freq_scale = 1/scaling.factor, e.g. 0.125 for factor 8).
    // Local/SWA layers use 1.0 (freq_scale_swa). Defaults to 1.0 when the GGUF
    // declares no scaling. Matches llama.cpp get_rope_freq_scale per layer.
    std::vector<float> layer_rope_scale;

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

class Gemma3ForwardPass : public ForwardPassBase, public IImageEmbeddable {
public:
    Gemma3ForwardPass(const Model& model, const ModelMetadata* metadata,
                      uint32_t context_len, uint32_t max_batch_size = 1);
    ~Gemma3ForwardPass() override = default;

    ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens,
                                      int pos, uint32_t slot_idx = 0,
                                      bool want_logits = true) override;

    ggml_cgraph* build_decoding_graph(const std::vector<int32_t>& tokens,
                                      const std::vector<uint32_t>& slots,
                                      const std::vector<int32_t>& positions) override;

    // ── Vision: image-token embedding substitution (Phase 3) ──────────────────
    // Arm precomputed image-token embeddings before the next prefill build.
    // `embd` is [hidden_dim * n_tokens] row-major (hidden_dim fastest) — exactly
    // the layout VisionEncoder::encode returns. `span_start` is the position of
    // the first image soft-token; the n_tokens embeddings replace the scaled
    // text embeddings at [span_start, span_start + n_tokens). Consumed (moved
    // out) by the next build_prefill_graph; re-arm to repeat.
    //
    // This is the single C3 coupling point between the text recipe and the
    // vision subsystem: the recipe owns the residual stream and performs the
    // substitution; it does NOT know how the embeddings were produced. Vision
    // stays out of ForwardPassBase and StepContext.
    void set_image_embeddings(std::vector<float> embd,
                              int32_t span_start,
                              uint32_t n_tokens) override {
        image_embd_       = std::move(embd);
        image_span_start_ = span_start;
        image_n_tokens_   = n_tokens;
    }

    // Phase 3 of docs/plan-feed-tokens.md: gemma3 honors want_logits=false
    // with one head-guard site. Attention-only; still owes its own
    // KV-append mid-stream differential.
    bool feed_tokens_supported() const override { return true; }

    // Decode graph is persistent-capable: every step-varying quantity is a
    // graph-input value (P1, docs/plan-persistent-decode-graph.md).
    bool supports_persistent_decode() const override { return true; }

    // has_decode_graph() inherits the default (true): build_decoding_graph is
    // implemented (Phase 3 of docs/plan-gemma-batched-decode.md).

    // Inputs are populated via the typed graph_inputs_ set built in
    // build_prefill_graph / build_decoding_graph (no set_inputs override).

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

    // L2 snapshot reach-through (pure attention: AppendKV only, no recurrent).
    simple_kv_cache* snapshot_kv_cache() override { return kv_cache_.get(); }

private:
    Gemma3Config                    config_;
    std::unique_ptr<simple_kv_cache> kv_cache_;

    // Per-block sandwich-norm weight pointers (G2-inherited).
    std::vector<ggml_tensor*> post_attn_norm_;  // blk.N.post_attention_norm.weight
    std::vector<ggml_tensor*> post_ffn_norm_;   // blk.N.post_ffw_norm.weight

    // Per-block QK-norm weight pointers (G3-new).
    std::vector<ggml_tensor*> q_norm_;  // blk.N.attn_q_norm.weight
    std::vector<ggml_tensor*> k_norm_;  // blk.N.attn_k_norm.weight

    // Armed image-token embeddings for the next prefill (empty => text-only).
    // Consumed (moved) by build_prefill_graph. See set_image_embeddings.
    std::vector<float> image_embd_;
    int32_t            image_span_start_ = -1;
    uint32_t           image_n_tokens_   = 0;

    // Retrieve a named tensor from the model context; throws on missing.
    ggml_tensor* require_tensor(uint32_t il, const char* suffix) const;
};
