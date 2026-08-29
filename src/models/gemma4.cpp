#include "gemma4.h"

#include "engine/model.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/gather_indices_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ── gemma4_tokenizer_config (unchanged from G4.7) ────────────────────────────

TokenizerConfig gemma4_tokenizer_config()
{
    TokenizerConfig cfg;
    cfg.normalizer    = NormalizerKind::SpaceToUnderscore;
    cfg.byte_fallback = true;
    cfg.add_bos_token = true;
    cfg.extra_chat_specials = {"<|turn>", "<turn|>", "<|channel>", "<channel|>"};
    return cfg;
}

// ── Gemma4Config::from_metadata ──────────────────────────────────────────────

Gemma4Config Gemma4Config::from_metadata(const ModelMetadata& meta)
{
    Gemma4Config cfg;
    cfg.n_layers          = meta.block_count;
    cfg.n_head            = meta.attention_head_count;
    cfg.hidden_dim        = meta.embedding_length;
    cfg.context_len       = meta.context_length;
    cfg.rms_norm_eps      = meta.rms_norm_eps;

    // Per-kind shapes — read from raw_kv (the loader doesn't know about
    // gemma4.* keys; the recipe is the architecture-aware boundary).
    cfg.head_dim_global   = meta.raw_kv.get_uint32("gemma4.attention.key_length");
    cfg.head_dim_swa      = meta.raw_kv.get_uint32("gemma4.attention.key_length_swa");
    cfg.rope_dim_global   = meta.raw_kv.get_uint32("gemma4.rope.dimension_count");
    cfg.rope_dim_swa      = meta.raw_kv.get_uint32("gemma4.rope.dimension_count_swa");
    cfg.rope_base_global  = meta.raw_kv.get_float ("gemma4.rope.freq_base");
    cfg.rope_base_swa     = meta.raw_kv.get_float ("gemma4.rope.freq_base_swa");
    cfg.sliding_window    = meta.raw_kv.get_uint32("gemma4.attention.sliding_window");

    cfg.ffn_dim_dense     = meta.raw_kv.get_uint32("gemma4.feed_forward_length");
    cfg.final_softcap     = meta.raw_kv.get_float ("gemma4.final_logit_softcapping");

    // Expert keys are absent on the dense 12B-it variant; present on
    // 26B-A4B / 31B.  Optional reads default to 0 ⇒ is_moe = false, which
    // selects the dense FFN branch in build_block and relaxes the inventory
    // validator's MoE-tensor requirements.
    cfg.ffn_dim_expert    = meta.raw_kv.get_uint32_opt("gemma4.expert_feed_forward_length").value_or(0);
    cfg.n_experts         = meta.raw_kv.get_uint32_opt("gemma4.expert_count").value_or(0);
    cfg.expert_top_k      = meta.raw_kv.get_uint32_opt("gemma4.expert_used_count").value_or(0);
    cfg.is_moe            = cfg.n_experts > 0;

    // Optional "dormant" keys — present in 26B-A4B GGUF as 0; we accept
    // missing too so synthetic / cut-down test fixtures don't trip.
    cfg.shared_kv_layers           =
        meta.raw_kv.get_uint32_opt("gemma4.attention.shared_kv_layers").value_or(0);
    cfg.embedding_length_per_layer =
        meta.raw_kv.get_uint32_opt("gemma4.embedding_length_per_layer_input").value_or(0);

    // Per-layer attention kind — derived from tensor inventory because
    // gemma4.attention.sliding_window_pattern is a GGUF array and
    // GGUFKVBag stores only scalars (and the loader is forbidden territory).
    // A layer is global iff its attn_v.weight is absent (V == K for global).
    const auto& inv = meta.tensor_inventory;
    cfg.is_global.assign(cfg.n_layers, false);
    cfg.swa_layer_idx.assign(cfg.n_layers, -1);
    cfg.global_layer_idx.assign(cfg.n_layers, -1);

    int swa_idx = 0, glb_idx = 0;
    for (uint32_t il = 0; il < cfg.n_layers; ++il) {
        const std::string vk = "blk." + std::to_string(il) + ".attn_v.weight";
        const bool has_v = inv.find(vk) != inv.end();
        cfg.is_global[il] = !has_v;
        if (has_v) cfg.swa_layer_idx[il]    = swa_idx++;
        else       cfg.global_layer_idx[il] = glb_idx++;
    }
    cfg.n_swa_layers    = static_cast<uint32_t>(swa_idx);
    cfg.n_global_layers = static_cast<uint32_t>(glb_idx);

    // Derive per-kind n_kv_heads from the first matching layer's K-weight
    // shape (shape[1] = n_kv_heads * head_dim_kind).
    auto k_out_dim = [&](uint32_t il) -> uint64_t {
        const std::string kk = "blk." + std::to_string(il) + ".attn_k.weight";
        auto it = inv.find(kk);
        if (it == inv.end() || it->second.shape.size() < 2) {
            throw std::runtime_error(
                "Gemma4Config::from_metadata: tensor '" + kk +
                "' missing or rank-deficient; expected shape [n_embd, n_kv_heads*head_dim]");
        }
        return it->second.shape[1];
    };
    bool seen_swa = false, seen_glb = false;
    for (uint32_t il = 0; il < cfg.n_layers && !(seen_swa && seen_glb); ++il) {
        if (!seen_swa && !cfg.is_global[il]) {
            cfg.n_kv_heads_swa = static_cast<uint32_t>(k_out_dim(il) / cfg.head_dim_swa);
            seen_swa = true;
        }
        if (!seen_glb &&  cfg.is_global[il]) {
            cfg.n_kv_heads_global = static_cast<uint32_t>(k_out_dim(il) / cfg.head_dim_global);
            seen_glb = true;
        }
    }

    if (cfg.n_swa_layers > 0 && cfg.n_kv_heads_swa == 0) {
        throw std::runtime_error(
            "Gemma4Config::from_metadata: field 'n_kv_heads_swa' "
            "expected > 0 (derived from blk.<sliding>.attn_k.weight shape), got 0");
    }
    if (cfg.n_global_layers > 0 && cfg.n_kv_heads_global == 0) {
        throw std::runtime_error(
            "Gemma4Config::from_metadata: field 'n_kv_heads_global' "
            "expected > 0 (derived from blk.<global>.attn_k.weight shape), got 0");
    }

    // Dormant-feature guards: bodies for PLE / shared-KV ship with G4.4 /
    // G4.3 but the 26B-A4B GGUF declares both as zero.  Refuse to run the
    // recipe if a future GGUF flips them on without us also having
    // promoted the path off the synthetic-test gate.
    if (cfg.shared_kv_layers != 0) {
        throw std::runtime_error(
            "Gemma4Config: field 'shared_kv_layers' expected 0 in 26B-A4B "
            "(end-to-end shared-KV is on the deferred dense 31B follow-up), "
            "got " + std::to_string(cfg.shared_kv_layers));
    }
    if (cfg.embedding_length_per_layer != 0) {
        throw std::runtime_error(
            "Gemma4Config: field 'embedding_length_per_layer_input' expected "
            "0 in 26B-A4B (end-to-end PLE is on the deferred dense 31B "
            "follow-up), got " + std::to_string(cfg.embedding_length_per_layer));
    }

    return cfg;
}

// ── validate_gemma4_inventory ────────────────────────────────────────────────

void validate_gemma4_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                "gemma4: missing tensor '" + name +
                "': expected in model weights, got absent");
    };

    require("token_embd.weight");
    require("output_norm.weight");
    // Tied embeddings — no separate output.weight.

    // expert_count absent (or 0) ⇒ dense variant (12B-it): single GeGLU FFN,
    // no expert tensors and no parallel-branch post-norms.  expert_count > 0
    // ⇒ MoE variant (26B-A4B / 31B): parallel dense+MoE FFN per layer.
    const bool is_moe =
        meta.raw_kv.get_uint32_opt("gemma4.expert_count").value_or(0) > 0;

    // Per-block tensors common to BOTH variants.
    static const std::vector<std::string> per_block_common = {
        "attn_norm.weight",
        "attn_q.weight",  "attn_k.weight",
        "attn_q_norm.weight", "attn_k_norm.weight",
        "attn_output.weight",
        "post_attention_norm.weight",
        // FFN entry norm + GeGLU weights (dense FFN, or MoE's shared dense MLP).
        "ffn_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        // Outer post-norm applied to the FFN output before the residual add
        // (common to dense and MoE; for MoE it follows the dense+moe sum).
        "post_ffw_norm.weight",
        // Per-layer output scale.
        "layer_output_scale.weight",
    };
    // MoE-only per-block tensors: the parallel-branch post-norms and the
    // expert dispatch weights.  Required only when expert_count > 0.
    static const std::vector<std::string> per_block_moe = {
        // Dense-branch post-norm (post_ffw_norm_1) + MoE-branch sandwich
        // (pre_ffw_norm_2 → moe → post_ffw_norm_2).
        "post_ffw_norm_1.weight",
        "pre_ffw_norm_2.weight",
        "post_ffw_norm_2.weight",
        "ffn_gate_inp.scale",   "ffn_gate_inp.weight",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.weight",
    };
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block_common) require(p + t);
        if (is_moe)
            for (const auto& t : per_block_moe) require(p + t);
        // attn_v.weight is conditional: present iff the layer is sliding
        // (global layers reuse K as V).  We do not enforce it either way
        // here — Gemma4Config::from_metadata derives the per-layer pattern
        // from this exact tensor's presence; an inventory inconsistent
        // with that derivation will surface as a shape mismatch at
        // graph-build time, not silently produce wrong logits.
    }
}

// ── Gemma4ForwardPass ────────────────────────────────────────────────────────

constexpr size_t GEMMA4_GRAPH_SIZE = 32768;  // larger than G3 — dual FFN + MoE

Gemma4ForwardPass::Gemma4ForwardPass(
    const Model& model, const ModelMetadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, ggml_type kv_type)
    : ForwardPassBase(model, metadata),
      config_(Gemma4Config::from_metadata(*metadata))
{
    ggml_backend_t cache_backend = model_.has_metal_backend()
        ? model_.get_backend_metal()
        : model_.get_backend_cpu();

    // Two KV caches with per-kind shapes.  shared_kv_layers == 0 (enforced
    // in from_metadata), so neither cache uses the G4.4 sharing vector.
    if (config_.n_swa_layers > 0) {
        const uint32_t n_embd_k = config_.n_kv_heads_swa * config_.head_dim_swa;
        const uint32_t n_embd_v = n_embd_k;  // sliding has separate V same shape
        kv_cache_swa_ = std::make_unique<simple_kv_cache>(
            config_.n_swa_layers, context_len, max_batch_size,
            n_embd_k, n_embd_v, kv_type, kv_type, cache_backend);
    }
    if (config_.n_global_layers > 0) {
        const uint32_t n_embd_k = config_.n_kv_heads_global * config_.head_dim_global;
        const uint32_t n_embd_v = n_embd_k;  // V == K for global; same shape
        kv_cache_global_ = std::make_unique<simple_kv_cache>(
            config_.n_global_layers, context_len, max_batch_size,
            n_embd_k, n_embd_v, kv_type, kv_type, cache_backend);
    }

    // Pre-resolve all per-block tensor pointers — string lookups on the
    // hot path are wasteful and the inventory is fixed at load time.
    block_w_.resize(config_.n_layers);
    for (uint32_t il = 0; il < config_.n_layers; ++il) {
        BlockWeights& w = block_w_[il];
        const auto& blk = model_.get_block(il);
        // The Model class already binds the universal tensors per block;
        // the Gemma-4-specific ones we resolve by name.
        w.attn_norm   = blk.attn_norm_weight;
        w.attn_q      = blk.attn_q_weight;
        w.attn_k      = blk.attn_k_weight;
        w.attn_v      = config_.is_global[il] ? nullptr : blk.attn_v_weight;
        w.attn_q_norm = blk.attn_q_norm_weight;
        w.attn_k_norm = blk.attn_k_norm_weight;
        w.attn_output = blk.attn_output_weight;
        w.ffn_norm    = blk.ffn_norm_weight;
        w.ffn_gate    = blk.ffn_gate_weight;
        w.ffn_up      = blk.ffn_up_weight;
        w.ffn_down    = blk.ffn_down_weight;

        w.post_attn_norm   = require_tensor(il, "post_attention_norm.weight");
        w.post_ffn_norm    = require_tensor(il, "post_ffw_norm.weight");
        w.layer_out_scale  = require_tensor(il, "layer_output_scale.weight");

        // MoE-only handles stay null on the dense variant (build_block
        // branches on config_.is_moe and never dereferences them).
        if (config_.is_moe) {
            w.post_ffn_norm_1  = require_tensor(il, "post_ffw_norm_1.weight");
            w.pre_moe_norm     = require_tensor(il, "pre_ffw_norm_2.weight");
            w.post_ffn_norm_2  = require_tensor(il, "post_ffw_norm_2.weight");
            w.moe_router_scale = require_tensor(il, "ffn_gate_inp.scale");
            w.moe_router       = require_tensor(il, "ffn_gate_inp.weight");
            w.moe_gate_up_exps = require_tensor(il, "ffn_gate_up_exps.weight");
            w.moe_down_exps    = require_tensor(il, "ffn_down_exps.weight");
        }
    }

}

ggml_tensor* Gemma4ForwardPass::require_tensor(uint32_t il, const char* suffix) const
{
    char name[128];
    std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
    ggml_tensor* t = ggml_get_tensor(model_.get_context(), name);
    if (!t) {
        throw std::runtime_error(
            std::string("Gemma4ForwardPass: tensor '") + name +
            "' missing from model context (architecture='gemma4'); "
            "expected after validate_gemma4_inventory pass, got absent");
    }
    return t;
}

ggml_tensor* Gemma4ForwardPass::maybe_tensor(uint32_t il, const char* suffix) const
{
    char name[128];
    std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
    return ggml_get_tensor(model_.get_context(), name);
}

// ── build_moe_geglu ──────────────────────────────────────────────────────────
//
// Same dispatch shape as MoELayer (top-k + ggml_mul_mat_id) but with
// GeGLU-tanh activation and a per-channel input scale applied before the
// router matmul.  No shared expert (Gemma 4 A4B has none — the
// MoEConfig::has_shared_expert nullable shape from the modular plan stays
// off).
ggml_tensor* Gemma4ForwardPass::build_moe_geglu(
    ggml_cgraph* gf,
    ggml_tensor* expert_in, ggml_tensor* router_in,
    const BlockWeights& w, uint32_t il, uint32_t n_tokens)
{
    const int64_t n_embd   = expert_in->ne[0];
    const int     n_exp    = static_cast<int>(config_.n_experts);
    const int     top_k    = static_cast<int>(config_.expert_top_k);
    const int64_t ffn_dim  = static_cast<int64_t>(config_.ffn_dim_expert);

    // Routing logits + top-k indices + softmaxed routing weights.
    // router_in is computed by the caller from attn_out (pre-norm input):
    //   router_in = ggml_mul(rms_norm(attn_out) * 1/sqrt(n_embd), ffn_gate_inp.scale)
    ggml_tensor* logits = ggml_mul_mat(ctx_, w.moe_router, router_in);
    set_tensor_name(gf, logits, "moe_logits", static_cast<int>(il));

    ggml_tensor* sorted_idx = ggml_argsort(ctx_, logits, GGML_SORT_ORDER_DESC);
    ggml_tensor* expert_idx = ggml_view_2d(ctx_, sorted_idx,
        top_k, n_tokens, sorted_idx->nb[1], 0);
    set_tensor_name(gf, expert_idx, "moe_idx", static_cast<int>(il));

    // Reshape logits to [1, n_experts, n_tokens] so ggml_get_rows picks
    // from the n_experts dim.
    ggml_tensor* logits_3d = ggml_reshape_3d(ctx_, logits, 1, n_exp, n_tokens);
    ggml_tensor* expert_logits = ggml_get_rows(ctx_, logits_3d, expert_idx);
    expert_logits = ggml_reshape_2d(ctx_, expert_logits, top_k, n_tokens);
    ggml_tensor* expert_weights = ggml_soft_max(ctx_, expert_logits);
    set_tensor_name(gf, expert_weights, "moe_weights", static_cast<int>(il));

    // 3. Split fused gate+up tensor into halves.
    //    moe_gate_up_exps: [n_embd, 2*ffn_dim, n_experts].
    //    Gate = first ffn_dim along ne[1]; up = next ffn_dim along ne[1].
    //    The expert stride (nb[2]) is unchanged in the views — each
    //    expert's gate-half and up-half remain at the same in-buffer
    //    offsets the original tensor places them at, so ggml_mul_mat_id
    //    can resolve them via the existing per-expert nb[2] stride.
    ggml_tensor* w_gate = ggml_view_3d(ctx_, w.moe_gate_up_exps,
        n_embd, ffn_dim, n_exp,
        w.moe_gate_up_exps->nb[1], w.moe_gate_up_exps->nb[2], 0);
    ggml_tensor* w_up   = ggml_view_3d(ctx_, w.moe_gate_up_exps,
        n_embd, ffn_dim, n_exp,
        w.moe_gate_up_exps->nb[1], w.moe_gate_up_exps->nb[2],
        static_cast<size_t>(ffn_dim) * w.moe_gate_up_exps->nb[1]);

    // 4. Expert dispatch: for each token, run its top-k experts.
    //    input_3d: [n_embd, 1, n_tokens] aligns with mul_mat_id's expected
    //    "b" tensor shape (n_embd, ?, n_tokens).
    ggml_tensor* input_3d = ggml_reshape_3d(ctx_, expert_in, n_embd, 1, n_tokens);

    ggml_tensor* gate_out = ggml_mul_mat_id(ctx_, w_gate, input_3d, expert_idx);
    ggml_tensor* up_out   = ggml_mul_mat_id(ctx_, w_up,   input_3d, expert_idx);
    set_tensor_name(gf, gate_out, "moe_exp_gate", static_cast<int>(il));
    set_tensor_name(gf, up_out,   "moe_exp_up",   static_cast<int>(il));

    // 5. GeGLU-tanh activation: gelu(gate) * up.  ggml_gelu uses the
    //    tanh approximation by default (matches gelu_pytorch_tanh).
    ggml_tensor* exp_act = ggml_mul(ctx_, ggml_gelu(ctx_, gate_out), up_out);

    // 6. Down projection: [n_embd, top_k, n_tokens]
    ggml_tensor* exp_down = ggml_mul_mat_id(ctx_, w.moe_down_exps,
                                             exp_act, expert_idx);
    set_tensor_name(gf, exp_down, "moe_exp_down", static_cast<int>(il));

    // 7. Weighted sum across the top_k axis → [n_embd, n_tokens].
    ggml_tensor* w_expanded = ggml_reshape_3d(ctx_, expert_weights,
                                               1, top_k, n_tokens);
    ggml_tensor* weighted = ggml_mul(ctx_, exp_down, w_expanded);

    ggml_tensor* routed = ggml_view_2d(ctx_, weighted,
        n_embd, n_tokens, weighted->nb[2], 0);
    for (int k = 1; k < top_k; ++k) {
        ggml_tensor* slice = ggml_view_2d(ctx_, weighted,
            n_embd, n_tokens, weighted->nb[2],
            static_cast<size_t>(k) * weighted->nb[1]);
        routed = ggml_add(ctx_, routed, slice);
    }
    set_tensor_name(gf, routed, "moe_routed", static_cast<int>(il));
    return routed;
}

// ── build_block ──────────────────────────────────────────────────────────────

ggml_tensor* Gemma4ForwardPass::build_block(
    ggml_cgraph* gf, ggml_tensor* cur, ggml_tensor* inp_pos,
    uint32_t il, uint32_t n_tokens, const AttnPhase& phase)
{
    const BlockWeights& w = block_w_[il];
    const bool is_global  = config_.is_global[il];

    // Per-kind attention shape.
    const int   head_dim   = is_global ? config_.head_dim_global   : config_.head_dim_swa;
    const int   n_kv_heads = is_global ? config_.n_kv_heads_global : config_.n_kv_heads_swa;
    const int   rope_dim   = is_global ? config_.rope_dim_global   : config_.rope_dim_swa;
    const float rope_base  = is_global ? config_.rope_base_global  : config_.rope_base_swa;

    simple_kv_cache* cache = is_global ? kv_cache_global_.get() : kv_cache_swa_.get();
    const int cache_il = is_global ? config_.global_layer_idx[il]
                                   : config_.swa_layer_idx[il];

    ggml_tensor* inpSA = cur;

    // ── A. Attention ─────────────────────────────────────────────────────
    cur = build_rms_norm(ctx_, cur, w.attn_norm, config_.rms_norm_eps,
                         static_cast<int>(il));
    set_tensor_name(gf, cur, "attn_pre_norm", static_cast<int>(il));

    ggml_tensor* Qcur = ggml_mul_mat(ctx_, w.attn_q, cur);
    ggml_tensor* Kcur = ggml_mul_mat(ctx_, w.attn_k, cur);

    // V == K for global layers (no attn_v.weight in the GGUF; the
    // attention kernel still wants a separately-shaped V tensor, so we
    // alias the projection result rather than the raw K-cache.  The
    // separate KV-cache write paths still go through cpy_k / cpy_v with
    // the same source data — that wastes V-cache bytes on global layers
    // but keeps the call site uniform.  G4.8 follow-up could special-case
    // this if memory matters.)
    ggml_tensor* Vcur = is_global ? Kcur : ggml_mul_mat(ctx_, w.attn_v, cur);

    Qcur = ggml_reshape_3d(ctx_, Qcur, head_dim, config_.n_head, n_tokens);
    Kcur = ggml_reshape_3d(ctx_, Kcur, head_dim, n_kv_heads,     n_tokens);
    Vcur = ggml_reshape_3d(ctx_, Vcur, head_dim, n_kv_heads,     n_tokens);

    // QK-norm: per-head, [head_dim]-shaped weight broadcast across heads
    // (Gemma 3 / 4 share this shape; the wider [head_dim, n_head] qwen3
    // variant is a different op).
    Qcur = build_rms_norm(ctx_, Qcur, w.attn_q_norm, config_.rms_norm_eps,
                          static_cast<int>(il));
    Kcur = build_rms_norm(ctx_, Kcur, w.attn_k_norm, config_.rms_norm_eps,
                          static_cast<int>(il));
    // V is RMS-normed without a weight (Gemma 4 spec; matches llama.cpp's
    // gemma4-iswa reference: `Vcur = ggml_rms_norm(ctx0, Vcur, eps)`).
    Vcur = ggml_rms_norm(ctx_, Vcur, config_.rms_norm_eps);
    set_tensor_name(gf, Vcur, "Vcur_normed", static_cast<int>(il));

    // RoPE — full or pruned.  When rope_dim == head_dim (the case in
    // 26B-A4B), Pruned and Standard produce bit-identical output.  The
    // op is selected by build_rope_pruned regardless; the kernel's
    // pass-through tail handles the (rope_dim < head_dim) case cleanly.
    Qcur = build_rope_pruned(ctx_, Qcur, inp_pos, rope_dim,
                              static_cast<int>(config_.context_len), rope_base);
    Kcur = build_rope_pruned(ctx_, Kcur, inp_pos, rope_dim,
                              static_cast<int>(config_.context_len), rope_base);

    // const float kq_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Gemma 3/4 with QK-norm typically use a fixed kq_scale of 1.0f
    // because the standard 1/sqrt(d) scaling is either omitted or baked
    // into the q_norm / k_norm weights.
    const float kq_scale = 1.0f;

    // Attention — the ONE fork between prefill and batched decode. The K/V
    // tensors are reshaped identically above; build_batched_attention infers
    // head_dim / n_kv_heads from Kcur->ne[0]/ne[1], so the per-kind shape
    // (global vs swa) flows through without explicit dims. softcap off for G4
    // (QK-norm replaces it).
    if (phase.is_decode()) {
        cur = build_batched_attention(ctx_, gf, cache, Qcur, Kcur, Vcur,
                                      /*layer_idx=*/cache_il, kq_scale,
                                      *phase.slots, *phase.positions,
                                      (*phase.layer_masks)[il],
                                      phase.gather_indices,
                                      /*il=*/static_cast<int>(il),
                                      /*softcap=*/0.0f);
    } else {
        cur = build_attention(ctx_, gf, cache, Qcur, Kcur, Vcur,
                              /*layer_idx=*/cache_il,
                              kq_scale, n_tokens, phase.slot_idx,
                              /*il=*/static_cast<int>(il),
                              head_dim, head_dim, n_kv_heads,
                              /*softcap=*/0.0f);
    }

    cur = ggml_mul_mat(ctx_, w.attn_output, cur);
    set_tensor_name(gf, cur, "attn_out", static_cast<int>(il));

    cur = build_rms_norm(ctx_, cur, w.post_attn_norm, config_.rms_norm_eps,
                         static_cast<int>(il));
    set_tensor_name(gf, cur, "post_attn_normed", static_cast<int>(il));

    ggml_tensor* attn_out = ggml_add(ctx_, cur, inpSA);
    set_tensor_name(gf, attn_out, "attn_residual", static_cast<int>(il));

    // ── B. Feed-forward (Gemma 4 topology) ───────────────────────────────
    // Two variants, selected by config_.is_moe (= expert_count > 0):
    //   - Dense (12B-it): a single GeGLU-tanh FFN.
    //   - MoE (26B-A4B / 31B): parallel dense + MoE branches, summed.
    // Both feed off attn_out; the result `ffn_inner` then passes through the
    // common outer post-norm (post_ffw_norm), is added to attn_out as the
    // single block residual, and scaled by layer_output_scale.  The MoE path
    // is byte-identical to the pre-dense recipe (same ops, same order).
    ggml_tensor* ffn_inner;
    if (config_.is_moe) {
        // B.1 Dense (shared) MLP sandwich: ffn_norm → GeGLU-tanh → post_ffw_norm_1
        ggml_tensor* cur_mlp = build_rms_norm(ctx_, attn_out, w.ffn_norm,
                                              config_.rms_norm_eps,
                                              static_cast<int>(il));
        cur_mlp = build_ffn_geglu_tanh(ctx_, gf, cur_mlp,
                                       w.ffn_gate, w.ffn_up, w.ffn_down,
                                       static_cast<int>(il));
        cur_mlp = build_rms_norm(ctx_, cur_mlp, w.post_ffn_norm_1,
                                 config_.rms_norm_eps, static_cast<int>(il));
        set_tensor_name(gf, cur_mlp, "ffn_mlp", static_cast<int>(il));

        // B.2 MoE branch — expert input uses pre_ffw_norm_2(attn_out); the
        //     router uses an unweighted rms_norm of attn_out scaled by
        //     1/sqrt(n_embd) and the per-channel ffn_gate_inp.scale.  This
        //     mirrors llama.cpp's gemma4-iswa reference exactly.
        ggml_tensor* expert_in = build_rms_norm(ctx_, attn_out, w.pre_moe_norm,
                                                config_.rms_norm_eps,
                                                static_cast<int>(il));
        set_tensor_name(gf, expert_in, "ffn_norm_2", static_cast<int>(il));

        ggml_tensor* router_in = ggml_rms_norm(ctx_, attn_out, config_.rms_norm_eps);
        router_in = ggml_scale(ctx_, router_in,
                               1.0f / std::sqrt(static_cast<float>(config_.hidden_dim)));
        router_in = ggml_mul(ctx_, router_in, w.moe_router_scale);
        set_tensor_name(gf, router_in, "moe_router_in", static_cast<int>(il));

        ggml_tensor* cur_moe = build_moe_geglu(gf, expert_in, router_in, w, il, n_tokens);
        cur_moe = build_rms_norm(ctx_, cur_moe, w.post_ffn_norm_2,
                                 config_.rms_norm_eps, static_cast<int>(il));
        set_tensor_name(gf, cur_moe, "ffn_moe", static_cast<int>(il));

        // B.3 Sum dense + MoE.
        ffn_inner = ggml_add(ctx_, cur_mlp, cur_moe);
        set_tensor_name(gf, ffn_inner, "ffn_moe_combined", static_cast<int>(il));
    } else {
        // Dense variant: a single GeGLU-tanh FFN (no parallel MoE branch and
        // no post_ffw_norm_1; the common post_ffw_norm below is the only
        // FFN post-norm).  Matches llama.cpp gemma4.cpp non-MoE branch.
        ffn_inner = build_rms_norm(ctx_, attn_out, w.ffn_norm,
                                   config_.rms_norm_eps, static_cast<int>(il));
        ffn_inner = build_ffn_geglu_tanh(ctx_, gf, ffn_inner,
                                         w.ffn_gate, w.ffn_up, w.ffn_down,
                                         static_cast<int>(il));
        set_tensor_name(gf, ffn_inner, "ffn_out", static_cast<int>(il));
    }

    // B.4 Common outer post-norm (post_ffw_norm), single residual, and
    //     layer_output_scale on the whole layer output.  Matches llama.cpp's
    //     `ffn_post_norm` (applied to either branch result) + `out_scale`.
    cur = build_rms_norm(ctx_, ffn_inner, w.post_ffn_norm,
                         config_.rms_norm_eps, static_cast<int>(il));
    set_tensor_name(gf, cur, "ffn_post_norm", static_cast<int>(il));

    ggml_tensor* layer_out = ggml_add(ctx_, attn_out, cur);
    layer_out = ggml_mul(ctx_, layer_out, w.layer_out_scale);
    set_tensor_name(gf, layer_out, "layer_out_scaled", static_cast<int>(il));
    return layer_out;
}

// ── build_prefill_graph ──────────────────────────────────────────────────────

ggml_cgraph* Gemma4ForwardPass::build_prefill_graph(
    const std::vector<int32_t>& tokens, int /*pos*/, uint32_t slot_idx,
    bool want_logits)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const uint32_t n_tokens = static_cast<uint32_t>(tokens.size());

    // 1. Embedding + sqrt(d_model) scale (fp32; consistent with G1/G2/G3).
    ggml_tensor* inpL = embedding(gf, tokens);
    inpL = build_embed_scale(ctx_, inpL,
                             std::sqrt(static_cast<float>(config_.hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");
    ggml_build_forward_expand(gf, inpL);

    // 2. Position tensor (one per token; per-layer rope_base differs but
    //    they all consume the same int32 positions).
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // Image attention mask: CAUSAL (bidi span UNWIRED 2026-06-16).
    //
    // The AttnMaskInput bidi span (bidi_start/bidi_len) is the mechanism llama
    // uses for Gemma image attention (mtmd_decode_use_non_causal==true for
    // PROJECTOR_TYPE_GEMMA4UV). It is RETAINED and unit-tested (Gemma 3 arms it;
    // see test_attn_mask_input.cpp) but DELIBERATELY NOT ARMED here: bidi is not
    // the bug. Proven via `LLAMA_FORCE_IMAGE_CAUSAL` — llama with CAUSAL image
    // attention is fully correct ("…rib cage and spine area"), so causal is
    // sufficient. Our image-causal forward is what diverges from llama-causal;
    // arming bidi only makes it worse (token-soup). Fix the causal forward first,
    // then revisit bidi. To re-arm: pass `pos + image_span_start_` and
    // `image_n_tokens_` as the bidi_start/bidi_len args below (image-armed only).
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    for (uint32_t il = 0; il < config_.n_layers; ++il)
        graph_inputs_.add(std::make_unique<AttnMaskInput>(
            "kq_mask." + std::to_string(il),
            config_.is_global[il] ? 0u : config_.sliding_window));

    // 2b. Image-token embedding substitution (IImageEmbeddable, Seam B — the one
    //     C3 boundary site between this recipe and the vision subsystem). The
    //     residual-stream overwrite is the shared
    //     ForwardPassBase::build_image_substitution (image rows enter unscaled,
    //     hence AFTER build_embed_scale above). The recipe-specific image concern
    //     — the bidirectional image-span mask — is handled in the AttnMaskInput
    //     wiring above; this step is only the substitution.
    //     Consumed (moved) here.
    if (!image_embd_.empty()) {
        inpL = build_image_substitution(
            gf, inpL, std::move(image_embd_), image_span_start_,
            image_n_tokens_, static_cast<int>(config_.hidden_dim), n_tokens);
        image_span_start_ = -1;  // consume-on-use (image_embd_ moved out)
        image_n_tokens_   = 0;
    }

    // 3. Transformer stack (manual composition; build_transformer_layer
    //    can't host this — see the gemma4.h scope comment).
    const AttnPhase prefill_phase{ /*slot_idx=*/slot_idx };
    for (uint32_t il = 0; il < config_.n_layers; ++il) {
        inpL = build_block(gf, inpL, inp_pos, il, n_tokens, prefill_phase);
        char dbg[64];
        std::snprintf(dbg, sizeof(dbg), "layer_out.%u", il);
        set_tensor_name(gf, inpL, dbg);
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    // 4. Output head. THE single per-recipe head-presence guard site for
    // gemma4 (docs/plan-feed-tokens.md → Head-presence locality constraint:
    // exactly one site, not scattered want_logits conditionals).
    // want_logits=false (feed_tokens) prunes final norm → LM head → softcap.
    // Head-less anchor is the per-layer ggml_set_output(inpL) from the layer
    // loop above (same invariant as the qwen35/qwen36 else-anchor, different
    // mechanism). KV cpy_k/v state-write roots are independently expanded in
    // build_block; attention-only (dense + MoE FFN, no recurrent state),
    // still owes its own KV-append mid-stream differential.
    if (want_logits) {
        // Final norm + LM head + final logit soft-cap (G4.6).
        ggml_tensor* cur = build_rms_norm(
            ctx_, build_out_ids_slice(gf, inpL), model_.get_output_norm_weight(),
            config_.rms_norm_eps, /*il=*/-1);
        set_tensor_name(gf, cur, "final_norm");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        // Tied embeddings: prefer output.weight if explicitly present, else
        // reuse token_embd.weight (the Gemma 1/2/3/4 path).
        if (model_.get_output_weight() != nullptr) {
            cur = ggml_mul_mat(ctx_, model_.get_output_weight(), cur);
        } else {
            cur = ggml_mul_mat(ctx_, model_.get_token_embedding_weight(), cur);
        }

        if (config_.final_softcap > 0.0f) {
            cur = build_softcap(ctx_, cur, config_.final_softcap);
        }

        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
    }
    return gf;
}

// ── Decoding (batched) ────────────────────────────────────────────────────────
//
// Mirrors the proven Gemma 3 decode transform (gemma3.cpp): same head-of-graph
// embedding+scale delta, per-layer AttnMaskInput windows, one shared
// gather_indices, and build_output_head for the sparse-aware LM head. The block
// body is reused via build_block(phase=decode) — only the attention call differs
// from prefill. 
// The two physical KV caches share context_len (n_ctx_max) and advance in
// lockstep, so a single gather_indices addresses both and n_kv_len is common.

ggml_cgraph* Gemma4ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const int    n_layers   = static_cast<int>(config_.n_layers);
    const int    hidden_dim = static_cast<int>(config_.hidden_dim);
    const size_t n_tokens   = tokens.size();   // total tokens across all slots

    // 1. Token embedding + sqrt(d_model) scale (fp32).
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");
    inpL = build_embed_scale(ctx_, inpL, std::sqrt(static_cast<float>(hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");

    // 2. Position tensor (one per token across all slots; per-layer rope_base
    //    differs but all layers consume the same int32 positions).
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. KV gather window, sized from the deepest slot (both caches lockstep).
    uint32_t max_pos = 0;
    for (uint32_t s : slots) {
        uint32_t p = get_cache_pos(s);
        if (p > max_pos) max_pos = p;
    }
    const uint32_t n_kv_len = max_pos + 1;

    // 4. Per-layer attention windows: global layers see the full history
    //    (window 0), sliding layers see config_.sliding_window. The same window
    //    seam as prefill, no mask-body fork.
    std::vector<uint32_t> layer_windows(n_layers);
    for (int il = 0; il < n_layers; ++il)
        layer_windows[il] = config_.is_global[il] ? 0u : config_.sliding_window;

    // 5. KV gather indices, shared across all layers AND both caches (identical
    //    n_ctx_max + lockstep positions ⇒ one index tensor addresses both).
    ggml_tensor* gather_indices =
        ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens * n_kv_len);
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // 6. Typed inputs. Masks are deduplicated by window (global + sliding ⇒ 2
    //    tensors, not n_layers) so the decode graph-input count stays under the
    //    backend scheduler's split-input cap on Metal. Shared gather indices are
    //    sized by the (common) cache n_ctx_max. Text-only decode — no image span.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    std::vector<ggml_tensor*> layer_masks = build_decode_layer_masks(
        gf, layer_windows, n_kv_len, static_cast<uint32_t>(n_tokens));
    graph_inputs_.add(std::make_unique<GatherIndicesInput>(
        kv_cache_global_->get_n_ctx_max()));

    // 7. Transformer stack — reuse build_block with the decode attention path.
    const AttnPhase decode_phase{
        /*slot_idx=*/0, &slots, &positions, &layer_masks, gather_indices };
    for (uint32_t il = 0; il < config_.n_layers; ++il) {
        inpL = build_block(gf, inpL, inp_pos, il, n_tokens, decode_phase);
    }

    // 8. Output head — final norm + tied LM head, routed through
    //    build_output_head so the grammar→sparse decode path is honored
    //    (decode_step arms sparse_decode_ids_; a hand-rolled ggml_mul_mat would
    //    ignore them and break sample_sparse). gemma_final_norm=false: Gemma 4
    //    prefill uses standard build_rms_norm on the output-norm weight, so the
    //    decode head matches with the standard form. final_softcap from config, matching prefill.
    build_output_head(gf, inpL, /*valid_idx=*/nullptr, /*gemma_final_norm=*/false,
                      /*final_softcap=*/config_.final_softcap);

    return gf;
}

// ── Cache routing ────────────────────────────────────────────────────────────
// Both caches advance / clear / reposition in lockstep so the per-layer
// position counters stay synchronized.  The single get_cache_pos return
// is the SWA cache when present, else the global cache — the prefill /
// decode path treats both as "current sequence position" and Gemma 4
// guarantees they advance together.

void Gemma4ForwardPass::advance_cache(uint32_t n_tokens, uint32_t slot_idx)
{
    if (kv_cache_swa_)    kv_cache_swa_->advance(n_tokens, slot_idx);
    if (kv_cache_global_) kv_cache_global_->advance(n_tokens, slot_idx);
}

void Gemma4ForwardPass::clear_slot(uint32_t slot_idx)
{
    if (kv_cache_swa_)    kv_cache_swa_->clear_slot(slot_idx);
    if (kv_cache_global_) kv_cache_global_->clear_slot(slot_idx);
}

void Gemma4ForwardPass::set_cache_pos(uint32_t pos, uint32_t slot_idx)
{
    if (kv_cache_swa_)    kv_cache_swa_->set_pos(pos, slot_idx);
    if (kv_cache_global_) kv_cache_global_->set_pos(pos, slot_idx);
}

uint32_t Gemma4ForwardPass::get_cache_pos(uint32_t slot_idx) const
{
    if (kv_cache_swa_)    return kv_cache_swa_->get_pos(slot_idx);
    if (kv_cache_global_) return kv_cache_global_->get_pos(slot_idx);
    return 0;
}

void Gemma4ForwardPass::clone_slot(uint32_t src_slot, uint32_t dst_slot,
                                    uint32_t n_tokens)
{
    if (kv_cache_swa_)    kv_cache_swa_->clone_slot(src_slot, dst_slot, n_tokens);
    if (kv_cache_global_) kv_cache_global_->clone_slot(src_slot, dst_slot, n_tokens);
}
