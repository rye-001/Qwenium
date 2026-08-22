#include "gemma2.h"

#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"
#include "../layers/transformer_block.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/gather_indices_input.h"

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
    uint32_t context_len, uint32_t max_batch_size, ggml_type kv_type)
    : ForwardPassBase(model, metadata),
      config_(Gemma2Config::from_metadata(*metadata))
{
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
        kv_type,
        kv_type,
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
    const std::vector<int32_t>& tokens, int /*pos*/, uint32_t slot_idx,
    bool want_logits)
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

    // 4. Output head. THE single per-recipe head-presence guard site for
    // gemma2 (docs/plan-feed-tokens.md → Head-presence locality constraint:
    // exactly one site, not scattered want_logits conditionals).
    // want_logits=false (feed_tokens) prunes final norm → LM head → softcap.
    // The head-less anchor is the per-layer ggml_set_output(inpL) already
    // emitted in the layer loop (scheduler backend-propagation root), so no
    // separate else is needed here — same invariant as the qwen35/qwen36
    // else-anchor, different mechanism because Gemma already keeps layer
    // outputs alive. KV cpy_k/v state-write roots are independently
    // ggml_build_forward_expand'd in build_transformer_layer; attention-only,
    // but still owes its own KV-append mid-stream differential.
    if (want_logits) {
        // Final norm (Gemma GGUF pre-shifts weights → standard build_rms_norm).
        ggml_tensor* cur = build_rms_norm(
            ctx_, build_out_ids_slice(gf, inpL), model_.get_output_norm_weight(),
            config_.rms_norm_eps, /*il=*/-1);
        set_tensor_name(gf, cur, "final_norm");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        // LM head (tied embeddings — no separate output.weight).
        if (model_.get_output_weight() != nullptr) {
            cur = ggml_mul_mat(ctx_, model_.get_output_weight(), cur);
        } else {
            cur = ggml_mul_mat(ctx_, model_.get_token_embedding_weight(), cur);
        }

        // Final logit soft-capping (Gemma 2 only).
        if (config_.final_softcap > 0.0f) {
            cur = build_softcap(ctx_, cur, config_.final_softcap);
        }

        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
    }

    return gf;
}

// ── build_decoding_graph ──────────────────────────────────────────────────────

ggml_cgraph* Gemma2ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const int    n_layers    = static_cast<int>(config_.n_layers);
    const int    hidden_dim  = static_cast<int>(config_.hidden_dim);
    const int    n_head      = static_cast<int>(config_.n_head);
    const int    n_head_kv   = static_cast<int>(config_.n_head_kv);
    const int    n_embd_head = static_cast<int>(config_.n_embd_head);
    const int    n_rot       = n_embd_head;
    const size_t n_tokens    = tokens.size();   // total tokens across all slots
    const float  freq_base   = config_.freq_base;

    // 1. Token embedding + sqrt(d_model) scale (same head-of-graph delta from
    //    Qwen decode as Gemma 1).
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");
    inpL = build_embed_scale(ctx_, inpL, std::sqrt(static_cast<float>(hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");

    // Position tensor (one per token across all slots).
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // KV gather window, sized from the deepest slot.
    uint32_t max_pos = 0;
    for (uint32_t s : slots) {
        uint32_t p = get_cache_pos(s);
        if (p > max_pos) max_pos = p;
    }
    uint32_t n_kv_len = max_pos + 1;

    // Per-layer attention windows: Gemma 2 interleaves local (even) / global
    // (odd) via config_.layer_window[il] (0 = global). The SAME window seam
    // prefill uses — the interleave is a parameter on the mask, no mask-body
    // fork. Masks are deduplicated by window value below (global + local ⇒ 2
    // tensors), so the decode graph-input count stays O(distinct windows).
    std::vector<uint32_t> layer_windows(n_layers);
    for (int il = 0; il < n_layers; ++il)
        layer_windows[il] = config_.layer_window[il];

    // KV gather indices, shared across layers.
    uint32_t n_total_indices = n_tokens * n_kv_len;
    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_total_indices);
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // Typed inputs: window-deduplicated masks (one AttnMaskInput per distinct
    // window) plus the shared gather indices.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    std::vector<ggml_tensor*> layer_masks = build_decode_layer_masks(
        gf, layer_windows, n_kv_len, static_cast<uint32_t>(n_tokens));
    graph_inputs_.add(std::make_unique<GatherIndicesInput>(kv_cache_->get_n_ctx_max()));

    // 2. Transformer stack — hand-rolled batched decode (mirrors Gemma 1 decode).
    //    Gemma 2 deltas: per-layer SWA mask (layer_masks[il]), attention softcap
    //    (config_.attn_softcap), and sandwich norm (post_attn_norm_/post_ffn_norm_
    //    applied to the attn/FFN output before each residual add).
    ggml_tensor* cur;
    for (uint32_t il = 0; il < static_cast<uint32_t>(n_layers); ++il) {
        ggml_tensor* inpSA = inpL;
        const auto& block = model_.get_block(il);
        cur = build_norm(gf, inpL, block.attn_norm_weight, il);
        ggml_tensor* Qcur = ggml_mul_mat(ctx_, block.attn_q_weight, cur);
        ggml_tensor* Kcur = ggml_mul_mat(ctx_, block.attn_k_weight, cur);
        ggml_tensor* Vcur = ggml_mul_mat(ctx_, block.attn_v_weight, cur);
        Qcur = ggml_reshape_3d(ctx_, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head_kv, n_tokens);
        Qcur = ggml_rope_ext(ctx_, Qcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = ggml_rope_ext(ctx_, Kcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        float kq_scale = 1.0f / sqrtf(static_cast<float>(n_embd_head));
        cur = build_batched_attention(ctx_, gf, kv_cache_.get(), Qcur, Kcur, Vcur, il, kq_scale, slots, positions, layer_masks[il], gather_indices, il, config_.attn_softcap);
        cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);
        cur = build_norm(gf, cur, post_attn_norm_[il], il);
        ggml_tensor* ffn_inp = ggml_add(ctx_, cur, inpSA);
        cur = build_norm(gf, ffn_inp, block.ffn_norm_weight, il);
        cur = build_ffn_geglu_tanh(ctx_, gf, cur, block.ffn_gate_weight, block.ffn_up_weight, block.ffn_down_weight, il);
        cur = build_norm(gf, cur, post_ffn_norm_[il], il);
        cur = ggml_add(ctx_, cur, ffn_inp);
        inpL = cur;
    }

    // 3. Output head — final norm + tied LM head + final logit soft-capping,
    //    routed through build_output_head so the sparse decode path is honored.
    //    NOTE: unlike Gemma 1 (which uses build_rms_norm_gemma), Gemma 2's
    //    prefill final norm is STANDARD build_rms_norm — its GGUF pre-shifts the
    //    output_norm weight to (1+w) (gemma2.cpp build_prefill_graph). So
    //    gemma_final_norm=FALSE here matches prefill bit-for-bit; passing true
    //    double-applies (+1) and grossly diverges. final_softcap matches the
    //    prefill build_softcap.
    build_output_head(gf, inpL, /*valid_idx=*/nullptr, /*gemma_final_norm=*/false,
                      /*final_softcap=*/config_.final_softcap);

    return gf;
}

