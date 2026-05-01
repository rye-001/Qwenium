#include "gemma4.h"

#include "../core/model.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ── DEBUG (G4): runtime diagnostics for the <tool_call|> hallucination ───────
namespace {
struct TStats { float min, max, mean, absmean; size_t nan; size_t n; };
static TStats tensor_stats_f32(ggml_tensor* t) {
    TStats s{0,0,0,0,0,0};
    if (!t) return s;
    const size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    s.n = n;
    s.min = std::numeric_limits<float>::infinity();
    s.max = -std::numeric_limits<float>::infinity();
    double sum = 0, asum = 0;
    for (float v : buf) {
        if (std::isnan(v)) { s.nan++; continue; }
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        sum  += v;
        asum += std::fabs(v);
    }
    const size_t live = n - s.nan;
    s.mean    = live ? float(sum  / live) : 0.f;
    s.absmean = live ? float(asum / live) : 0.f;
    return s;
}
static void print_stats(const char* label, ggml_tensor* t) {
    if (!t) { std::fprintf(stderr, "[G4DBG] %-32s = (null)\n", label); return; }
    if (t->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[G4DBG] %-32s type=%s (skipped, non-F32)  shape=[%lld,%lld,%lld,%lld]\n",
                     label, ggml_type_name(t->type),
                     (long long)t->ne[0], (long long)t->ne[1],
                     (long long)t->ne[2], (long long)t->ne[3]);
        return;
    }
    TStats s = tensor_stats_f32(t);
    std::fprintf(stderr,
        "[G4DBG] %-32s n=%zu  min=%.6f  max=%.6f  mean=%.6f  |mean|=%.6f  nan=%zu\n",
        label, s.n, s.min, s.max, s.mean, s.absmean, s.nan);
}
} // namespace


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
    cfg.ffn_dim_expert    = meta.raw_kv.get_uint32("gemma4.expert_feed_forward_length");
    cfg.n_experts         = meta.raw_kv.get_uint32("gemma4.expert_count");
    cfg.expert_top_k      = meta.raw_kv.get_uint32("gemma4.expert_used_count");
    cfg.final_softcap     = meta.raw_kv.get_float ("gemma4.final_logit_softcapping");

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

    // Per-block tensors common to every layer kind.
    static const std::vector<std::string> per_block_common = {
        "attn_norm.weight",
        "attn_q.weight",  "attn_k.weight",
        "attn_q_norm.weight", "attn_k_norm.weight",
        "attn_output.weight",
        "post_attention_norm.weight",
        // Dense FFN (sandwich: ffn_norm → ffn → post_ffw_norm_1)
        "ffn_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "post_ffw_norm_1.weight",
        // MoE FFN (sandwich: pre_ffw_norm_2 → moe → post_ffw_norm_2)
        "pre_ffw_norm_2.weight",
        "post_ffw_norm_2.weight",
        "ffn_gate_inp.scale",   "ffn_gate_inp.weight",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.weight",
        // Outer post-norm applied after (dense + moe) sum.
        "post_ffw_norm.weight",
        // Output scale
        "layer_output_scale.weight",
    };
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block_common) require(p + t);
        // attn_v.weight is conditional: present iff the layer is sliding
        // (global layers reuse K as V).  We do not enforce it either way
        // here — Gemma4Config::from_metadata derives the per-layer pattern
        // from this exact tensor's presence; an inventory inconsistent
        // with that derivation will surface as a shape mismatch at
        // graph-build time, not silently produce wrong logits.
    }

    // (post_ffw_norm_1 / post_ffw_norm_2 are required above — they are the
    // dense-branch and MoE-branch post-norms in Gemma 4's parallel-FFN
    // topology, not Altup artifacts as an earlier draft assumed.)
}

// ── Gemma4ForwardPass ────────────────────────────────────────────────────────

constexpr size_t GEMMA4_GRAPH_SIZE = 32768;  // larger than G3 — dual FFN + MoE

Gemma4ForwardPass::Gemma4ForwardPass(
    const Model& model, const ModelMetadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, int kv_quant_bits)
    : ForwardPassBase(model, metadata),
      config_(Gemma4Config::from_metadata(*metadata))
{
    if (kv_quant_bits != 0) {
        throw std::runtime_error(
            "Gemma4ForwardPass: kv_quant_bits != 0 not supported "
            "(architecture='gemma4'); expected 0, got " +
            std::to_string(kv_quant_bits));
    }

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
            n_embd_k, n_embd_v, GGML_TYPE_F32, GGML_TYPE_F32, cache_backend);
    }
    if (config_.n_global_layers > 0) {
        const uint32_t n_embd_k = config_.n_kv_heads_global * config_.head_dim_global;
        const uint32_t n_embd_v = n_embd_k;  // V == K for global; same shape
        kv_cache_global_ = std::make_unique<simple_kv_cache>(
            config_.n_global_layers, context_len, max_batch_size,
            n_embd_k, n_embd_v, GGML_TYPE_F32, GGML_TYPE_F32, cache_backend);
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
        w.post_ffn_norm_1  = require_tensor(il, "post_ffw_norm_1.weight");
        w.post_ffn_norm    = require_tensor(il, "post_ffw_norm.weight");
        w.pre_moe_norm     = require_tensor(il, "pre_ffw_norm_2.weight");
        w.post_ffn_norm_2  = require_tensor(il, "post_ffw_norm_2.weight");
        w.moe_router_scale = require_tensor(il, "ffn_gate_inp.scale");
        w.moe_router       = require_tensor(il, "ffn_gate_inp.weight");
        w.moe_gate_up_exps = require_tensor(il, "ffn_gate_up_exps.weight");
        w.moe_down_exps    = require_tensor(il, "ffn_down_exps.weight");
        w.layer_out_scale  = require_tensor(il, "layer_output_scale.weight");
    }

    // ── DEBUG: dump resolved hparams + weight stats for layer 0 ─────────────
    // Goal: settle the (1+w) pre-bake question empirically.
    //   - If attn_norm.weight mean ≈ 1.0  → GGUF IS pre-baked → use build_rms_norm.
    //   - If attn_norm.weight mean ≈ 0.0  → GGUF is RAW       → use build_rms_norm_gemma.
    std::fprintf(stderr, "[G4DBG] ── Gemma4Config ─────────────────────────\n");
    std::fprintf(stderr, "[G4DBG] n_layers=%u  n_head=%u  hidden=%u  ctx=%u\n",
                 config_.n_layers, config_.n_head, config_.hidden_dim, config_.context_len);
    std::fprintf(stderr, "[G4DBG] head_dim_swa=%u  head_dim_global=%u  rope_base_swa=%.1f  rope_base_global=%.1f\n",
                 config_.head_dim_swa, config_.head_dim_global,
                 config_.rope_base_swa, config_.rope_base_global);
    std::fprintf(stderr, "[G4DBG] n_kv_heads_swa=%u  n_kv_heads_global=%u  sliding_window=%u  final_softcap=%.1f\n",
                 config_.n_kv_heads_swa, config_.n_kv_heads_global,
                 config_.sliding_window, config_.final_softcap);
    std::fprintf(stderr, "[G4DBG] n_swa_layers=%u  n_global_layers=%u  layer0_kind=%s\n",
                 config_.n_swa_layers, config_.n_global_layers,
                 config_.is_global[0] ? "GLOBAL" : "SWA");
    std::fprintf(stderr, "[G4DBG] ── Norm weights (raw GGUF values) ────────\n");
    print_stats("blk.0.attn_norm",         block_w_[0].attn_norm);
    print_stats("blk.0.attn_q_norm",       block_w_[0].attn_q_norm);
    print_stats("blk.0.attn_k_norm",       block_w_[0].attn_k_norm);
    print_stats("blk.0.post_attention_norm", block_w_[0].post_attn_norm);
    print_stats("blk.0.ffn_norm",          block_w_[0].ffn_norm);
    print_stats("blk.0.post_ffw_norm_1",   block_w_[0].post_ffn_norm_1);
    print_stats("blk.0.post_ffw_norm",     block_w_[0].post_ffn_norm);
    print_stats("blk.0.pre_ffw_norm_2",    block_w_[0].pre_moe_norm);
    print_stats("blk.0.post_ffw_norm_2",   block_w_[0].post_ffn_norm_2);
    print_stats("blk.0.layer_output_scale",block_w_[0].layer_out_scale);
    print_stats("output_norm",             model_.get_output_norm_weight());
    print_stats("token_embd",              model_.get_token_embedding_weight());
    std::fprintf(stderr, "[G4DBG] ── (mean ≈ 1 ⇒ pre-baked  |  mean ≈ 0 ⇒ raw) ─────\n");
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
    uint32_t il, uint32_t slot_idx, uint32_t n_tokens)
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

    cur = build_attention(ctx_, gf, cache, Qcur, Kcur, Vcur,
                          /*layer_idx=*/cache_il,
                          kq_scale, n_tokens, slot_idx,
                          /*il=*/static_cast<int>(il),
                          head_dim, head_dim, n_kv_heads,
                          /*softcap=*/0.0f);  // attn softcap off for G4 (QK-norm replaces it)

    cur = ggml_mul_mat(ctx_, w.attn_output, cur);
    set_tensor_name(gf, cur, "attn_out", static_cast<int>(il));

    cur = build_rms_norm(ctx_, cur, w.post_attn_norm, config_.rms_norm_eps,
                         static_cast<int>(il));
    set_tensor_name(gf, cur, "post_attn_normed", static_cast<int>(il));

    ggml_tensor* attn_out = ggml_add(ctx_, cur, inpSA);
    set_tensor_name(gf, attn_out, "attn_residual", static_cast<int>(il));
    if (il == 0) { ggml_set_output(attn_out); ggml_build_forward_expand(gf, attn_out); }

    // ── B. Parallel dense + MoE feed-forward (Gemma 4 topology) ──────────
    // Both branches feed off attn_out (NOT off each other's residuals).
    // Their outputs are summed, the sum is post-normed, the scaled result
    // is added to attn_out as the single block residual.

    // B.1 Dense FFN sandwich: ffn_norm → GeGLU-tanh → post_ffw_norm_1
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

    // B.3 Sum dense + MoE → outer post-norm.
    ggml_tensor* ffn_combined = ggml_add(ctx_, cur_mlp, cur_moe);
    set_tensor_name(gf, ffn_combined, "ffn_moe_combined", static_cast<int>(il));
    if (il == 0) { ggml_set_output(ffn_combined); ggml_build_forward_expand(gf, ffn_combined); }

    cur = build_rms_norm(ctx_, ffn_combined, w.post_ffn_norm,
                         config_.rms_norm_eps, static_cast<int>(il));
    set_tensor_name(gf, cur, "ffn_post_norm", static_cast<int>(il));

    // B.4 Single residual; layer_output_scale applied to the whole layer
    //     output (not to MoE alone).  Matches llama.cpp's `out_scale` hook.
    ggml_tensor* layer_out = ggml_add(ctx_, attn_out, cur);
    layer_out = ggml_mul(ctx_, layer_out, w.layer_out_scale);
    set_tensor_name(gf, layer_out, "layer_out_scaled", static_cast<int>(il));
    return layer_out;
}

// ── build_prefill_graph ──────────────────────────────────────────────────────

ggml_cgraph* Gemma4ForwardPass::build_prefill_graph(
    const std::vector<int32_t>& tokens, int /*pos*/, uint32_t slot_idx)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const uint32_t n_tokens = static_cast<uint32_t>(tokens.size());

    // 1. Embedding + sqrt(d_model) scale (fp32; consistent with G1/G2/G3).
    ggml_tensor* inpL = embedding(gf, tokens);
    inpL = build_embed_scale(ctx_, inpL,
                             std::sqrt(static_cast<float>(config_.hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);

    // 2. Position tensor (one per token; per-layer rope_base differs but
    //    they all consume the same int32 positions).
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. Transformer stack (manual composition; build_transformer_layer
    //    can't host this — see the gemma4.h scope comment).
    for (uint32_t il = 0; il < config_.n_layers; ++il) {
        inpL = build_block(gf, inpL, inp_pos, il, slot_idx, n_tokens);
        char dbg[64];
        std::snprintf(dbg, sizeof(dbg), "layer_out.%u", il);
        set_tensor_name(gf, inpL, dbg);
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    // 4. Final norm + LM head + final logit soft-cap (G4.6).
    ggml_tensor* cur = build_rms_norm(
        ctx_, inpL, model_.get_output_norm_weight(),
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
    return gf;
}

// ── set_inputs ───────────────────────────────────────────────────────────────

void Gemma4ForwardPass::set_inputs(ggml_cgraph* gf,
                                    const std::vector<int32_t>& tokens,
                                    int pos)
{
    const uint32_t n_tokens = static_cast<uint32_t>(tokens.size());

    // ── DEBUG: dump first ~16 tokens & positions of first prefill call ──────
    static int dbg_call = 0;
    if (dbg_call < 3) {
        std::fprintf(stderr, "[G4DBG] set_inputs call #%d: n_tokens=%u  pos=%d\n",
                     dbg_call, n_tokens, pos);
        const size_t lim = std::min<size_t>(tokens.size(), 16);
        std::fprintf(stderr, "[G4DBG]   tokens[0..%zu]:", lim);
        for (size_t i = 0; i < lim; ++i) std::fprintf(stderr, " %d", tokens[i]);
        std::fprintf(stderr, "%s\n", tokens.size() > lim ? " …" : "");
        dbg_call++;
    }

    ggml_tensor* tokens_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tokens_t) {
        throw std::runtime_error(
            "Gemma4ForwardPass::set_inputs: 'tokens' tensor not found in graph");
    }
    ggml_backend_tensor_set(tokens_t, tokens.data(), 0,
                            tokens.size() * sizeof(int32_t));

    ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (!inp_pos) {
        throw std::runtime_error(
            "Gemma4ForwardPass::set_inputs: 'inp_pos' tensor not found");
    }
    std::vector<int32_t> positions(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        positions[i] = pos + static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(inp_pos, positions.data(), 0,
                            positions.size() * sizeof(int32_t));

    // Per-layer KQ mask: causal for global; causal + window cutoff for sliding.
    for (uint32_t il = 0; il < config_.n_layers; ++il) {
        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, name);
        if (!kq_mask) {
            throw std::runtime_error(
                "Gemma4ForwardPass::set_inputs: kq_mask tensor not found "
                "for layer " + std::to_string(il));
        }
        const uint32_t n_kv = kq_mask->ne[0];
        const uint32_t window = config_.is_global[il] ? 0u : config_.sliding_window;

        std::vector<float> mask(n_kv * n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const uint32_t q_pos = static_cast<uint32_t>(pos) + i;
            for (uint32_t j = 0; j < n_kv; ++j) {
                const bool causal = (j <= q_pos);
                const bool in_win = (window == 0) || (q_pos - j < window);
                mask[i * n_kv + j] = (causal && in_win) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0,
                                n_kv * n_tokens * sizeof(float));
    }
}

// ── DEBUG run_prefill: read back intermediates after compute ─────────────────
std::vector<float> Gemma4ForwardPass::run_prefill(
    const std::vector<int32_t>& tokens,
    int pos, uint32_t slot_idx,
    ggml_backend_sched_t scheduler)
{
    static int dbg_call = 0;
    ggml_backend_sched_reset(scheduler);
    ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
    ggml_backend_sched_alloc_graph(scheduler, gf);
    set_inputs(gf, tokens, pos);
    ggml_backend_sched_graph_compute(scheduler, gf);

    if (dbg_call < 2) {
        std::fprintf(stderr, "[G4DBG] ── post-compute intermediates (call #%d) ─\n", dbg_call);
        print_stats("inpL_scaled (post-embed)", ggml_graph_get_tensor(gf, "inpL_scaled"));
        // Layer 0 sub-components
        print_stats("L0 attn_residual", ggml_graph_get_tensor(gf, "attn_residual.0"));
        print_stats("L0 dense_ffn_residual", ggml_graph_get_tensor(gf, "dense_ffn_residual.0"));
        // First, mid, last layer outputs
        char buf[64];
        std::snprintf(buf, sizeof(buf), "layer_out.%u", 0u);
        print_stats("layer_out.0",   ggml_graph_get_tensor(gf, buf));
        if (config_.n_layers > 2) {
            std::snprintf(buf, sizeof(buf), "layer_out.%u", config_.n_layers/2);
            print_stats("layer_out.mid", ggml_graph_get_tensor(gf, buf));
        }
        std::snprintf(buf, sizeof(buf), "layer_out.%u", config_.n_layers - 1);
        print_stats("layer_out.last", ggml_graph_get_tensor(gf, buf));
        print_stats("final_norm",   ggml_graph_get_tensor(gf, "final_norm"));
        // Logits stats + top-K
        ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
        if (logits) {
            print_stats("logits", logits);
            const size_t vocab  = static_cast<size_t>(logits->ne[0]);
            const size_t ntok   = static_cast<size_t>(logits->ne[1]);
            std::vector<float> last(vocab);
            const size_t off = (ntok - 1) * vocab * sizeof(float);
            ggml_backend_tensor_get(logits, last.data(), off, vocab * sizeof(float));
            // top-10
            std::vector<int> idx(vocab);
            for (size_t i = 0; i < vocab; ++i) idx[i] = (int)i;
            std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                [&](int a, int b){ return last[a] > last[b]; });
            std::fprintf(stderr, "[G4DBG] last-token logits top-10:\n");
            for (int k = 0; k < 10; ++k) {
                std::fprintf(stderr, "[G4DBG]   #%d  id=%6d  logit=%.4f\n",
                             k, idx[k], last[idx[k]]);
            }
        }
        std::fprintf(stderr, "[G4DBG] ──────────────────────────────────────────\n");
        dbg_call++;
    }

    advance_cache(tokens.size(), slot_idx);
    return get_output_logits(gf);
}

// ── Decoding (batched) — stubbed in G4.8, mirrors G2/G3 ──────────────────────

ggml_cgraph* Gemma4ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& /*tokens*/,
    const std::vector<uint32_t>& /*slots*/,
    const std::vector<int32_t>& /*positions*/)
{
    throw std::runtime_error(
        "Gemma4ForwardPass::build_decoding_graph: batched decode not "
        "implemented in PR G4.8 (architecture='gemma4'); expected: "
        "prefill-only path, got: batched call");
}

void Gemma4ForwardPass::set_batched_inputs(
    ggml_cgraph* /*gf*/,
    const std::vector<int32_t>& /*tokens*/,
    const std::vector<uint32_t>& /*slots*/,
    const std::vector<int32_t>& /*positions*/)
{
    throw std::runtime_error(
        "Gemma4ForwardPass::set_batched_inputs: batched decode not "
        "implemented in PR G4.8 (architecture='gemma4'); expected: "
        "prefill-only path, got: batched call");
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
    snapkv_advance_seq_pos(slot_idx, n_tokens);
}

void Gemma4ForwardPass::clear_slot(uint32_t slot_idx)
{
    if (kv_cache_swa_)    kv_cache_swa_->clear_slot(slot_idx);
    if (kv_cache_global_) kv_cache_global_->clear_slot(slot_idx);
    snapkv_clear_seq_pos(slot_idx);
}

void Gemma4ForwardPass::set_cache_pos(uint32_t pos, uint32_t slot_idx)
{
    if (kv_cache_swa_)    kv_cache_swa_->set_pos(pos, slot_idx);
    if (kv_cache_global_) kv_cache_global_->set_pos(pos, slot_idx);
}

uint32_t Gemma4ForwardPass::get_cache_pos(uint32_t slot_idx) const
{
    const uint32_t seq = snapkv_get_seq_pos(slot_idx);
    if (seq > 0) return seq;
    if (kv_cache_swa_)    return kv_cache_swa_->get_pos(slot_idx);
    if (kv_cache_global_) return kv_cache_global_->get_pos(slot_idx);
    return 0;
}

uint32_t Gemma4ForwardPass::get_physical_cache_pos(uint32_t slot_idx) const
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
