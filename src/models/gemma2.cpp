#include "gemma2.h"

#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"
#include "../layers/transformer_block.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>

// ── Gemma2Config::from_metadata ───────────────────────────────────────────────

Gemma2Config Gemma2Config::from_metadata(const ModelMetadata& meta)
{
    Gemma2Config cfg;
    cfg.n_layers     = meta.block_count;
    cfg.n_head       = meta.attention_head_count;
    cfg.n_head_kv    = meta.attention_head_count_kv;
    cfg.n_embd_head  = meta.attention_key_length;
    cfg.hidden_dim   = meta.embedding_length;
    cfg.context_len  = meta.context_length;
    cfg.rms_norm_eps = meta.rms_norm_eps;
    cfg.freq_base    = (meta.rope_freq_base > 0.0f) ? meta.rope_freq_base : 10000.0f;

    // Gemma-2-specific scalars live in raw_kv (stored by the generic loader).
    cfg.attn_softcap   = meta.raw_kv.get_float("gemma2.attn_logit_softcapping");
    cfg.final_softcap  = meta.raw_kv.get_float("gemma2.final_logit_softcapping");
    cfg.sliding_window = meta.raw_kv.get_uint32("gemma2.attention.sliding_window");

    // Per-layer attention kind: Gemma 2 alternates even=local / odd=global.
    // There is no per-layer array in the GGUF — the pattern is structural.
    cfg.layer_window.resize(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; ++i) {
        cfg.layer_window[i] = (i % 2 == 0) ? cfg.sliding_window : 0u;
    }

    return cfg;
}

// ── Inventory validator ───────────────────────────────────────────────────────

void validate_gemma2_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                "gemma2: missing tensor '" + name +
                "': expected in model weights, got absent");
    };

    require("token_embd.weight");
    require("output_norm.weight");
    // Gemma 2 uses tied embeddings — no separate output.weight.

    static const std::vector<std::string> per_block = {
        "attn_norm.weight",
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        // Sandwich norm (G2-specific):
        "post_attention_norm.weight",
        "post_ffw_norm.weight",
    };
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block) require(p + t);
    }
}

// ── ForwardPass constructor ───────────────────────────────────────────────────

constexpr size_t GEMMA2_GRAPH_SIZE = 16384;

Gemma2ForwardPass::Gemma2ForwardPass(
    const Model& model, const ModelMetadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, int kv_quant_bits)
    : ForwardPassBase(model, metadata),
      config_(Gemma2Config::from_metadata(*metadata))
{
    if (kv_quant_bits != 0) {
        throw std::runtime_error(
            "Gemma2ForwardPass: kv_quant_bits != 0 not supported "
            "(architecture='gemma2'); expected 0, got " +
            std::to_string(kv_quant_bits));
    }

    ggml_backend_t cache_backend = model_.has_metal_backend()
        ? model_.get_backend_metal()
        : model_.get_backend_cpu();

    const uint32_t n_embd_k = config_.n_embd_head * config_.n_head_kv;
    const uint32_t n_embd_v = config_.n_embd_head * config_.n_head_kv;

    kv_cache_ = std::make_unique<simple_kv_cache>(
        config_.n_layers,
        context_len,
        max_batch_size,
        n_embd_k,
        n_embd_v,
        GGML_TYPE_F32,
        GGML_TYPE_F32,
        cache_backend);

    // Pre-load the G2-specific post-norm weight pointers.
    // These are in the ggml context but not in the generic TransformerBlock struct.
    post_attn_norm_.resize(config_.n_layers, nullptr);
    post_ffn_norm_.resize(config_.n_layers, nullptr);
    for (uint32_t il = 0; il < config_.n_layers; ++il) {
        post_attn_norm_[il] = require_tensor(il, "post_attention_norm.weight");
        post_ffn_norm_[il]  = require_tensor(il, "post_ffw_norm.weight");
    }
}

ggml_tensor* Gemma2ForwardPass::require_tensor(uint32_t il, const char* suffix) const
{
    char name[128];
    std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
    ggml_tensor* t = ggml_get_tensor(model_.get_context(), name);
    if (!t) {
        throw std::runtime_error(
            std::string("Gemma2ForwardPass: tensor '") + name +
            "': expected in model context, got absent");
    }
    return t;
}

// ── build_prefill_graph ───────────────────────────────────────────────────────

ggml_cgraph* Gemma2ForwardPass::build_prefill_graph(
    const std::vector<int32_t>& tokens, int /*pos*/, uint32_t slot_idx)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const int n_layers    = static_cast<int>(config_.n_layers);
    const int hidden_dim  = static_cast<int>(config_.hidden_dim);
    const int n_head      = static_cast<int>(config_.n_head);
    const int n_head_kv   = static_cast<int>(config_.n_head_kv);
    const int n_embd_head = static_cast<int>(config_.n_embd_head);
    const size_t n_tokens = tokens.size();

    // 1. Token embedding + sqrt(d_model) scale (same as Gemma 1).
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");
    inpL = build_embed_scale(ctx_, inpL, std::sqrt(static_cast<float>(hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);

    // Position tensor.
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // Typed inputs (replaces set_inputs). Gemma 2 interleaves local/global
    // attention: the per-layer sliding window is a *parameter* on
    // AttnMaskInput (config_.layer_window[il]; 0 = global). No mask-body
    // edits — the interface hosts Gemma without bending.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    for (uint32_t il = 0; il < config_.n_layers; ++il)
        graph_inputs_.add(std::make_unique<AttnMaskInput>(
            "kq_mask." + std::to_string(il), config_.layer_window[il]));

    // 2. Per-layer hparams (shared across layers; sandwich norm weights differ per layer).
    TransformerBlockHparams blk_hp;
    blk_hp.is_qwen2       = false;
    blk_hp.n_head         = n_head;
    blk_hp.n_head_kv      = n_head_kv;
    blk_hp.n_embd_head    = n_embd_head;
    blk_hp.freq_base      = config_.freq_base;
    blk_hp.context_length = static_cast<int>(config_.context_len);
    blk_hp.rms_norm_eps   = config_.rms_norm_eps;
    // GGUF Gemma 2 exports pre-shift the norm weights to (1+w) at conversion
    // time (llama.cpp convert_hf_to_gguf.py adds 1 to *.norm.weight).
    // At runtime the op is standard x * w_gguf — gemma_rms_norm stays false.
    blk_hp.gemma_rms_norm = false;
    blk_hp.gemma_geglu    = true;   // Gemma 2 uses GeGLU-tanh (same as Gemma 1)
    blk_hp.attn_softcap   = config_.attn_softcap;

    // 3. Transformer stack.
    for (uint32_t il = 0; il < static_cast<uint32_t>(n_layers); ++il) {
        const auto& block = model_.get_block(il);

        TransformerBlockWeights w{};
        w.attn_norm = block.attn_norm_weight;
        w.q         = block.attn_q_weight;
        w.k         = block.attn_k_weight;
        w.v         = block.attn_v_weight;
        w.q_bias    = nullptr;
        w.k_bias    = nullptr;
        w.v_bias    = nullptr;
        w.q_norm    = nullptr;  // Gemma 2 has no QK norm (that's G3).
        w.k_norm    = nullptr;
        w.out       = block.attn_output_weight;
        w.ffn_norm  = block.ffn_norm_weight;
        w.ffn_gate  = block.ffn_gate_weight;
        w.ffn_up    = block.ffn_up_weight;
        w.ffn_down  = block.ffn_down_weight;
        // G2 sandwich norm weights:
        w.post_attn_norm = post_attn_norm_[il];
        w.post_ffn_norm  = post_ffn_norm_[il];

        inpL = build_transformer_layer(ctx_, gf, kv_cache_.get(), inpL, inp_pos,
                                       w, blk_hp, il, slot_idx,
                                       static_cast<uint32_t>(n_tokens));

        char dbg[64];
        std::snprintf(dbg, sizeof(dbg), "layer_out.%u", il);
        set_tensor_name(gf, inpL, dbg);
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    // 4. Final norm (Gemma GGUF pre-shifts weights, so standard build_rms_norm).
    ggml_tensor* cur = build_rms_norm(
        ctx_, inpL, model_.get_output_norm_weight(),
        config_.rms_norm_eps, /*il=*/-1);
    set_tensor_name(gf, cur, "final_norm");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // 5. LM head (tied embeddings — no separate output.weight).
    if (model_.get_output_weight() != nullptr) {
        cur = ggml_mul_mat(ctx_, model_.get_output_weight(), cur);
    } else {
        cur = ggml_mul_mat(ctx_, model_.get_token_embedding_weight(), cur);
    }

    // 6. Final logit soft-capping (Gemma 2 only).
    if (config_.final_softcap > 0.0f) {
        cur = build_softcap(ctx_, cur, config_.final_softcap);
    }

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ── build_decoding_graph ──────────────────────────────────────────────────────

ggml_cgraph* Gemma2ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& /*tokens*/,
    const std::vector<uint32_t>& /*slots*/,
    const std::vector<int32_t>& /*positions*/)
{
    // TODO(sparse): when single-token decode lands here, build the LM head
    // via build_output_head(gf, inpL) — NOT a hand-rolled ggml_mul_mat like
    // build_prefill_graph does. decode_step arms sparse_decode_ids_ for
    // grammar-constrained decode; a hand-rolled head ignores them and returns
    // full-vocab logits, causing sample_sparse size-mismatch / bad-access
    // (the class of bug fixed in qwen3.cpp / qwen35.cpp).
    throw std::runtime_error(
        "Gemma2ForwardPass::build_decoding_graph: batched decode not "
        "implemented in PR G2.5; expected: prefill-only path, got: batched call");
}

