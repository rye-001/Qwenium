#pragma once
// gemma_embedding2.h — EmbeddingGemma-2 (GGUF arch "gemma-embedding2"), the
// first encoder recipe: one bidirectional pass over a whole text, producing
// one vector per token. Added 2026-10-07.
//
// What it shares with gemma4 (same layer modules, same order):
//   - per-layer sliding/global attention with per-kind head size, KV heads and
//     RoPE base (sliding: head 256, base 1e4; global: head 512, base 1e6),
//     plain NEOX RoPE on both (no proportional freq factors);
//   - QK-norm, weightless V-norm, kq_scale 1.0, sandwich norms, GeGLU-tanh FFN,
//     sqrt(d_model) embed scale, per-layer layer_output_scale.
// What is new:
//   - NO KV cache and NO decode: attention is build_uncached_attention over a
//     BIDIRECTIONAL mask (BidirectionalMaskInput) — a band |q - k| <= radius on
//     sliding layers, open on global layers. Two masks per graph, shared by
//     every layer of their kind;
//   - Per-Layer Embeddings (layers/ple.h), injected after each layer's FFN;
//   - global layers carry their own attn_v (gemma4's global layers reuse K),
//     so the sliding/global split is read from the GGUF's
//     attention.sliding_window_pattern, not inferred from attn_v's absence;
//   - the head is output_norm → output.weight (512 → 768) per token, the graph
//     output "embd_tokens"; pooling and L2 normalisation are engine/embed's.
//
// It derives from ForwardPassBase for the shared scaffolding (arena, typed
// inputs, embedding lookup, the registry factory). Every cache and decode
// method of that interface refuses fail-loud: an encoder has neither. One such
// recipe is tolerable; a second encoder is the trigger to give encoders their
// own base (architecture.md §12).
//
// Reference: llama.cpp src/models/gemma-embedding2.cpp at f498f864f.
// Gate: tests/unit/test_embed.cpp (cosine vs llama.cpp's embeddings).

#include "forward_pass_base.h"
#include "i_embedding_output.h"
#include "../loader/tokenizer_config.h"

#include <cstdint>
#include <vector>

struct ModelMetadata;

struct GemmaEmbedding2Config {
    uint32_t n_layers        = 0;   // block_count (24)
    uint32_t n_head          = 0;   // attention.head_count (4)
    uint32_t hidden_dim      = 0;   // embedding_length (512)
    uint32_t ffn_dim         = 0;   // feed_forward_length (2048)
    uint32_t context_len     = 0;   // context_length — RoPE's original context
    float    rms_norm_eps    = 0.f;

    uint32_t head_dim_global = 0;   // attention.key_length (512)
    uint32_t head_dim_swa    = 0;   // attention.key_length_swa (256)
    uint32_t rope_dim_global = 0;   // rope.dimension_count (512)
    uint32_t rope_dim_swa    = 0;   // rope.dimension_count_swa (256)
    float    rope_base_global= 0.f; // rope.freq_base (1e6)
    float    rope_base_swa   = 0.f; // rope.freq_base_swa (1e4)
    // attention.sliding_window is the TOTAL band width (1024): llama.cpp's
    // converter doubles HF's sliding_window, which EmbeddingGemma-2's own
    // transformers config defines as an inclusive radius (512). The recipe
    // uses the radius; from_metadata halves the GGUF value (fail-loud if odd).
    uint32_t sliding_radius  = 0;

    uint32_t ple_dim         = 0;   // embedding_length_per_layer_input (512)
    uint32_t embd_out        = 0;   // embedding_length_out (768)
    IEmbeddingOutput::Pooling pooling = IEmbeddingOutput::Pooling::Mean;  // pooling_type

    std::vector<bool>     is_global;   // from attention.sliding_window_pattern (true = sliding)
    std::vector<uint32_t> n_kv_heads;  // per layer, from blk.N.attn_k.weight's shape

    // Reads the gemma-embedding2.* keys, fail-loud on any that are missing or
    // contradict what the recipe builds (causal attention, shared KV, unequal
    // K/V head sizes, an unknown pooling type).
    static GemmaEmbedding2Config from_metadata(const ModelMetadata& meta);
};

// Fail-loud tensor inventory for "gemma-embedding2".
void validate_gemma_embedding2_inventory(const ModelMetadata& meta);

// The Gemma 4 vocabulary and normalisation (tokenizer.ggml.model "gemma4").
TokenizerConfig gemma_embedding2_tokenizer_config();

class GemmaEmbedding2ForwardPass : public ForwardPassBase, public IEmbeddingOutput {
public:
    // context_len bounds the sequence length one pass accepts (no cache is
    // allocated). The registry factory's batch size and KV type have nothing
    // to size here — no slots, no cache — so the factory does not pass them.
    GemmaEmbedding2ForwardPass(const Model& model, const ModelMetadata* metadata,
                               uint32_t context_len);
    ~GemmaEmbedding2ForwardPass() override = default;

    // ── IEmbeddingOutput ────────────────────────────────────────────────────
    ggml_cgraph* build_embedding_graph(const std::vector<int32_t>& tokens) override;
    uint32_t embedding_dim() const override { return config_.embd_out; }
    Pooling  pooling() const override { return config_.pooling; }

    // ── ForwardPassBase: an encoder has no logits, cache or decode ──────────
    ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos,
                                     uint32_t slot_idx = 0, bool want_logits = true) override;
    ggml_cgraph* build_decoding_graph(const std::vector<int32_t>& tokens,
                                      const std::vector<uint32_t>& slots,
                                      const std::vector<int32_t>& positions) override;
    void advance_cache(uint32_t n_tokens, uint32_t slot_idx) override;
    void clear_slot(uint32_t slot_idx) override;
    void set_cache_pos(uint32_t pos, uint32_t slot_idx) override;
    uint32_t get_cache_pos(uint32_t slot_idx) const override;
    void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) override;

    // Flash applies to the one pass (DecodePolicy's prefill implementation).
    bool supports_flash_attn() const override { return true; }

    const GemmaEmbedding2Config& config() const { return config_; }

private:
    GemmaEmbedding2Config config_;
    uint32_t              max_tokens_;

    struct BlockWeights {
        ggml_tensor* attn_norm       = nullptr;
        ggml_tensor* attn_q          = nullptr;
        ggml_tensor* attn_k          = nullptr;
        ggml_tensor* attn_v          = nullptr;
        ggml_tensor* attn_q_norm     = nullptr;
        ggml_tensor* attn_k_norm     = nullptr;
        ggml_tensor* attn_output     = nullptr;
        ggml_tensor* post_attn_norm  = nullptr;
        ggml_tensor* ffn_norm        = nullptr;
        ggml_tensor* ffn_gate        = nullptr;
        ggml_tensor* ffn_up          = nullptr;
        ggml_tensor* ffn_down        = nullptr;
        ggml_tensor* post_ffn_norm   = nullptr;
        ggml_tensor* ple_inp_gate    = nullptr;   // blk.N.inp_gate.weight
        ggml_tensor* ple_proj        = nullptr;   // blk.N.proj.weight
        ggml_tensor* ple_post_norm   = nullptr;   // blk.N.post_norm.weight
        ggml_tensor* layer_out_scale = nullptr;
    };
    std::vector<BlockWeights> block_w_;
    ggml_tensor* ple_model_proj_ = nullptr;   // per_layer_model_proj.weight
    ggml_tensor* ple_proj_norm_  = nullptr;   // per_layer_proj_norm.weight
    ggml_tensor* output_proj_    = nullptr;   // output.weight [hidden, embd_out]

    ggml_tensor* require_tensor(const std::string& name) const;

    ggml_tensor* build_block(ggml_cgraph* gf, ggml_tensor* cur, ggml_tensor* inp_pos,
                             ggml_tensor* ple_inputs, ggml_tensor* kq_mask, uint32_t il,
                             uint32_t n_tokens);

    [[noreturn]] void refuse(const char* what) const;
};
