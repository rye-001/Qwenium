#include "gemma3.h"

#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"
#include "../layers/transformer_block.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/gather_indices_input.h"
#include "../graph_inputs/kv_write_indices_input.h"

#include <cstdlib>
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// ── Gemma3Config::from_metadata ───────────────────────────────────────────────

Gemma3Config Gemma3Config::from_metadata(const ModelMetadata& meta)
{
    Gemma3Config cfg;
    cfg.n_layers     = meta.block_count;
    cfg.n_head       = meta.attention_head_count;
    cfg.n_head_kv    = meta.attention_head_count_kv;
    cfg.n_embd_head  = meta.attention_key_length;
    cfg.hidden_dim   = meta.embedding_length;
    cfg.context_len  = meta.context_length;
    cfg.rms_norm_eps = meta.rms_norm_eps;

    // Gemma 3 GGUF stores the global RoPE base (1e6) in the standard key.
    // The local RoPE base (10K) is a hard-coded architectural constant — it is
    // not present in the GGUF.
    cfg.global_rope_base = (meta.rope_freq_base > 0.0f) ? meta.rope_freq_base : 1000000.0f;
    cfg.local_rope_base  = 10000.0f;

    cfg.sliding_window = meta.raw_kv.get_uint32("gemma3.attention.sliding_window");

    // Linear RoPE scaling on global layers (gemma3.rope.scaling.type=linear;
    // factor 8 for the 131K-context models). freq_scale = 1/factor; absent key
    // => 1.0 (no scaling). Local/SWA layers always use 1.0.
    const std::optional<float> rope_scale_factor =
        meta.raw_kv.get_float_opt("gemma3.rope.scaling.factor");
    const float global_rope_scale =
        (rope_scale_factor && *rope_scale_factor > 0.0f)
            ? 1.0f / *rope_scale_factor
            : 1.0f;

    // 5:1 alternation: five local (sliding-window) layers then one global,
    // repeating from layer 0. Layer i is global iff (i % 6 == 5).
    cfg.layer_window.resize(cfg.n_layers);
    cfg.layer_rope_base.resize(cfg.n_layers);
    cfg.layer_rope_scale.resize(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; ++i) {
        const bool is_global = (i % 6 == 5);
        cfg.layer_window[i]     = is_global ? 0u : cfg.sliding_window;
        cfg.layer_rope_base[i]  = is_global ? cfg.global_rope_base : cfg.local_rope_base;
        cfg.layer_rope_scale[i] = is_global ? global_rope_scale : 1.0f;
    }

    return cfg;
}

// ── Inventory validator ───────────────────────────────────────────────────────

void validate_gemma3_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                "gemma3: missing tensor '" + name +
                "': expected in model weights, got absent");
    };

    require("token_embd.weight");
    require("output_norm.weight");
    // Gemma 3 uses tied embeddings — no separate output.weight.

    static const std::vector<std::string> per_block = {
        "attn_norm.weight",
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q_norm.weight", "attn_k_norm.weight",   // QK-norm (G3-new)
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        // Sandwich norm (G2-inherited):
        "post_attention_norm.weight",
        "post_ffw_norm.weight",
    };
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block) require(p + t);
    }
    // Multimodal tensors (mm.* / vision.*) are intentionally not validated —
    // they are out of scope for text-only inference and ignored when present.
}

// ── ForwardPass constructor ───────────────────────────────────────────────────

constexpr size_t GEMMA3_GRAPH_SIZE = 16384;

Gemma3ForwardPass::Gemma3ForwardPass(
    const Model& model, const ModelMetadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, ggml_type kv_type)
    : ForwardPassBase(model, metadata),
      config_(Gemma3Config::from_metadata(*metadata))
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

    // Sandwich norm weights (G2-inherited).
    post_attn_norm_.resize(config_.n_layers, nullptr);
    post_ffn_norm_.resize(config_.n_layers, nullptr);
    // QK-norm weights (G3-new).
    q_norm_.resize(config_.n_layers, nullptr);
    k_norm_.resize(config_.n_layers, nullptr);

    for (uint32_t il = 0; il < config_.n_layers; ++il) {
        post_attn_norm_[il] = require_tensor(il, "post_attention_norm.weight");
        post_ffn_norm_[il]  = require_tensor(il, "post_ffw_norm.weight");
        q_norm_[il]         = require_tensor(il, "attn_q_norm.weight");
        k_norm_[il]         = require_tensor(il, "attn_k_norm.weight");
    }
}

ggml_tensor* Gemma3ForwardPass::require_tensor(uint32_t il, const char* suffix) const
{
    char name[128];
    std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
    ggml_tensor* t = ggml_get_tensor(model_.get_context(), name);
    if (!t) {
        throw std::runtime_error(
            std::string("Gemma3ForwardPass: tensor '") + name +
            "': expected in model context, got absent");
    }
    return t;
}

// ── build_prefill_graph ───────────────────────────────────────────────────────

ggml_cgraph* Gemma3ForwardPass::build_prefill_graph(
    const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx,
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

    // 1. Token embedding + sqrt(d_model) scale.
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");
    inpL = build_embed_scale(arena_.ctx(), inpL, std::sqrt(static_cast<float>(hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);

    // Position tensor.
    ggml_tensor* inp_pos = ggml_new_tensor_1d(arena_.ctx(), GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // Image-span bidirectional mask (Phase 4). When image embeddings are armed,
    // the soft-token span attends bidirectionally (recipe property — Gemma 3
    // specific). image_span_start_ is a LOCAL index into this batch (it is a
    // column offset into inpL in the substitution below); the mask works in
    // ABSOLUTE positions (q_pos = pos + r, key index j is absolute), so convert:
    // absolute span start = pos + image_span_start_. Disabled (len 0) for
    // text-only prefill, recovering the pure causal+window mask. This is the
    // single mask site — the bidi span is a *parameter* on AttnMaskInput, the
    // same parameterize-don't-split discipline as the sliding window itself.
    const bool img_armed = !image_embd_.empty();
    // Image-span attention is CAUSAL by default for Gemma 3.
    //
    // Bidirectional image attention is the nominal reference behavior (llama.cpp
    // mtmd: mtmd_decode_use_non_causal == true for PROJECTOR_TYPE_GEMMA3), and a
    // prior revision enabled it by default after wiring the global-layer linear
    // RoPE scaling (gemma3.rope.scaling.factor=8 → freq_scale=0.125, see
    // Gemma3Config::layer_rope_scale). However, measured on the real target model
    // (medgemma gemma3-siglip + mmproj-BF16), bidi STILL corrupts generation:
    // with bidi armed the model degenerates into token-soup / immediate <eos> for
    // almost any prompt phrasing (e.g. "Describe this image." → "://://://…"),
    // surviving only for a few near-exact phrasings. Flipping the SAME mask to
    // causal — the only difference being the image↔image sub-block — produces
    // coherent, image-grounded answers across prompts (verified A/B, identical
    // model/image/prompt, env var the only change). The bidi'd image-span hidden
    // states are evidently brittle in a way the rope-scale fix did not resolve;
    // root cause is still open (candidate: image-embedding magnitude exposed only
    // under full-span aggregation). Causal is correct-enough and robust, so it is
    // the shipping default until bidi is genuinely fixed.
    //
    // Set QINF_GEMMA3_IMAGE_BIDI=1 to re-enable the bidirectional span for that
    // investigation. See docs/gemma-vision-handoff.md and plan-gemma-vision-impl.md §P4.
    const bool     use_bidi      = img_armed &&
        (std::getenv("QINF_GEMMA3_IMAGE_BIDI") != nullptr);
    const int32_t  bidi_start    = use_bidi ? pos + image_span_start_ : -1;
    const uint32_t bidi_len      = use_bidi ? image_n_tokens_ : 0u;

    // Typed inputs (replaces set_inputs). Gemma 3 uses a 5:1 local:global
    // pattern; the per-layer sliding window is a parameter on AttnMaskInput
    // (config_.layer_window[il]; 0 = global). Per-layer RoPE base is a graph
    // op, not an input — out of scope (plan scope fence).
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    for (uint32_t il = 0; il < config_.n_layers; ++il)
        graph_inputs_.add(std::make_unique<AttnMaskInput>(
            "kq_mask." + std::to_string(il), config_.layer_window[il],
            bidi_start, bidi_len));

    // 1b. Image-token embedding substitution (Phase 3 — the one C3 boundary
    // site between this recipe and the vision subsystem). The residual-stream
    // overwrite is the shared ForwardPassBase::build_image_substitution (image
    // rows enter unscaled, hence AFTER build_embed_scale above). Gemma 3's
    // recipe-specific image concern — the bidirectional image-span mask — is
    // handled separately in the AttnMaskInput wiring above; this step is only
    // the substitution. Consumed (moved) here; re-arm to repeat.
    if (!image_embd_.empty()) {
        inpL = build_image_substitution(
            gf, inpL, std::move(image_embd_), image_span_start_,
            image_n_tokens_, hidden_dim, n_tokens);
        image_span_start_ = -1;  // consume-on-use (image_embd_ moved out)
        image_n_tokens_   = 0;
    }

    // 2. Base block hyperparameters (per-layer freq_base is overwritten in the loop).
    TransformerBlockHparams blk_hp;
    blk_hp.is_qwen2       = false;
    blk_hp.n_head         = n_head;
    blk_hp.n_head_kv      = n_head_kv;
    blk_hp.n_embd_head    = n_embd_head;
    blk_hp.freq_base      = config_.local_rope_base;  // overwritten per layer below
    blk_hp.context_length = static_cast<int>(config_.context_len);
    blk_hp.rms_norm_eps   = config_.rms_norm_eps;
    // Gemma GGUF pre-shifts norm weights to (1+w) at conversion time, so the
    // standard x*w_gguf op is correct at runtime — gemma_rms_norm stays false.
    blk_hp.gemma_rms_norm = false;
    blk_hp.gemma_geglu    = true;  // GeGLU-tanh (same as Gemma 1 and 2)
    blk_hp.attn_softcap   = 0.0f; // No soft-cap in Gemma 3

    // 3. Transformer stack.
    for (uint32_t il = 0; il < static_cast<uint32_t>(n_layers); ++il) {
        const auto& block = model_.get_block(il);

        // Update per-layer RoPE base: local layers 10K, global layers 1M.
        blk_hp.freq_base  = config_.layer_rope_base[il];
        blk_hp.freq_scale = config_.layer_rope_scale[il];

        TransformerBlockWeights w{};
        w.attn_norm = block.attn_norm_weight;
        w.q         = block.attn_q_weight;
        w.k         = block.attn_k_weight;
        w.v         = block.attn_v_weight;
        w.q_bias    = nullptr;
        w.k_bias    = nullptr;
        w.v_bias    = nullptr;
        // QK-norm (G3): loaded from blk.N.attn_q_norm.weight / attn_k_norm.weight.
        // These are [head_dim]-shaped weights broadcast across all heads.
        w.q_norm    = q_norm_[il];
        w.k_norm    = k_norm_[il];
        w.out       = block.attn_output_weight;
        w.ffn_norm  = block.ffn_norm_weight;
        w.ffn_gate  = block.ffn_gate_weight;
        w.ffn_up    = block.ffn_up_weight;
        w.ffn_down  = block.ffn_down_weight;
        // Sandwich norm (G2-inherited).
        w.post_attn_norm = post_attn_norm_[il];
        w.post_ffn_norm  = post_ffn_norm_[il];

        inpL = build_transformer_layer(arena_.ctx(), gf, kv_cache_.get(), inpL, inp_pos,
                                       w, blk_hp, il, slot_idx,
                                       static_cast<uint32_t>(n_tokens),
                                       /*ple_residual=*/nullptr,
                                       /*use_flash=*/use_flash_attn());

        char dbg[64];
        std::snprintf(dbg, sizeof(dbg), "layer_out.%u", il);
        set_tensor_name(gf, inpL, dbg);
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    // 4. Output head. THE single per-recipe head-presence guard site for
    // gemma3 (docs/plan-feed-tokens.md → Head-presence locality constraint:
    // exactly one site, not scattered want_logits conditionals).
    // want_logits=false (feed_tokens) prunes final norm → LM head. Head-less
    // anchor is the per-layer ggml_set_output(inpL) from the layer loop
    // (same invariant as the qwen35/qwen36 else-anchor, different mechanism).
    // KV cpy_k/v state-write roots are independently expanded in
    // build_transformer_layer; attention-only, still owes its KV-append
    // mid-stream differential ("attention-only" is not a skip).
    if (want_logits) {
        // Final norm.
        ggml_tensor* cur = build_rms_norm(
            arena_.ctx(), build_out_ids_slice(gf, inpL), model_.get_output_norm_weight(),
            config_.rms_norm_eps, /*il=*/-1);
        set_tensor_name(gf, cur, "final_norm");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        // LM head (tied embeddings — no separate output.weight in Gemma 3).
        if (model_.get_output_weight() != nullptr) {
            cur = ggml_mul_mat(arena_.ctx(), model_.get_output_weight(), cur);
        } else {
            cur = ggml_mul_mat(arena_.ctx(), model_.get_token_embedding_weight(), cur);
        }

        // No final soft-cap for Gemma 3 (removed in G3 vs. G2).

        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
    }

    return gf;
}

// ── build_decoding_graph ──────────────────────────────────────────────────────

ggml_cgraph* Gemma3ForwardPass::build_decoding_graph(
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

    // 1. Token embedding + sqrt(d_model) scale (same head-of-graph delta from
    //    Qwen decode as Gemma 1/2).
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");
    inpL = build_embed_scale(arena_.ctx(), inpL, std::sqrt(static_cast<float>(hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");

    // Position tensor (one per token across all slots).
    ggml_tensor* inp_pos = ggml_new_tensor_1d(arena_.ctx(), GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // KV gather window, sized from the deepest slot, then bucketed (padded
    // tail −inf-masked, zero-init rows) so one decode graph shape survives a
    // whole bucket of steps — plan-persistent-decode-graph.md.
    uint32_t max_pos = 0;
    for (uint32_t s : slots) {
        uint32_t p = get_cache_pos(s);
        if (p > max_pos) max_pos = p;
    }
    uint32_t n_kv_len = decode_kv_len(max_pos + 1, kv_cache_->get_n_ctx_max());

    // Per-layer attention windows: Gemma 3 runs five local (sliding-window)
    // layers per one global (config_.layer_window[il]; 0 = global). The 5:1
    // pattern lives entirely in the per-layer window values, no mask-body fork.
    // Masks are deduplicated by window value below (global + sliding ⇒ 2
    // tensors), so the decode graph-input count stays O(distinct windows).
    std::vector<uint32_t> layer_windows(n_layers);
    for (int il = 0; il < n_layers; ++il)
        layer_windows[il] = config_.layer_window[il];

    // KV gather indices, shared across layers.
    uint32_t n_total_indices = n_tokens * n_kv_len;
    ggml_tensor* gather_indices = ggml_new_tensor_1d(arena_.ctx(), GGML_TYPE_I32, n_total_indices);
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // KV write rows as input VALUES (persistent-graph write path); Cpy mode
    // is the byte-gate reference and builds no such tensor.
    ggml_tensor* kv_write_idx = nullptr;
    if (kv_write_mode() == KvWriteMode::SetRows) {
        kv_write_idx = ggml_new_tensor_1d(arena_.ctx(), GGML_TYPE_I64,
                                          static_cast<int64_t>(n_tokens));
        ggml_set_input(kv_write_idx);
        ggml_set_name(kv_write_idx, KvWriteIndicesInput::slot_);
    }

    // Typed inputs: window-deduplicated masks (one AttnMaskInput per distinct
    // window) plus the shared gather indices. Text-only decode — the image bidi
    // span stays disabled (window-only masks), same as the text-only prefill path.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    std::vector<ggml_tensor*> layer_masks = build_decode_layer_masks(
        gf, layer_windows, n_kv_len, static_cast<uint32_t>(n_tokens));

    // Flash attention (opt-in --flash-attn) needs F16 masks —
    // ggml_flash_attn_ext hard-asserts it. build_decode_layer_masks already
    // deduplicates by window, so layer_masks holds only a couple of DISTINCT
    // tensors repeated across layers; cast each distinct one once rather than
    // once per layer. The F32 tensors stay the graph inputs AttnMaskInput
    // fills — only what attention reads changes.
    if (use_flash_attn()) {
        std::unordered_map<ggml_tensor*, ggml_tensor*> mask_f16;
        for (ggml_tensor*& m : layer_masks) {
            auto it = mask_f16.find(m);
            if (it == mask_f16.end()) {
                ggml_tensor* c = ggml_cast(arena_.ctx(), m, GGML_TYPE_F16);
                ggml_set_name(c, "kq_mask_f16");
                it = mask_f16.emplace(m, c).first;
            }
            m = it->second;
        }
    }

    graph_inputs_.add(std::make_unique<GatherIndicesInput>(kv_cache_->get_n_ctx_max()));
    if (kv_write_idx)
        graph_inputs_.add(std::make_unique<KvWriteIndicesInput>(
            kv_cache_->get_n_ctx_max()));

    // 2. Transformer stack — hand-rolled batched decode (mirrors Gemma 2
    //    decode). Gemma 3 deltas: per-layer RoPE base/scale (local 10K/1.0,
    //    global 1M/linear-scaled — config_.layer_rope_base/layer_rope_scale),
    //    QK-norm on Q and K after reshape (q_norm_/k_norm_, before RoPE, same
    //    order as build_transformer_layer), and no attention softcap.
    ggml_tensor* cur;
    for (uint32_t il = 0; il < static_cast<uint32_t>(n_layers); ++il) {
        ggml_tensor* inpSA = inpL;
        const auto& block = model_.get_block(il);
        cur = build_norm(gf, inpL, block.attn_norm_weight, il);
        ggml_tensor* Qcur = ggml_mul_mat(arena_.ctx(), block.attn_q_weight, cur);
        ggml_tensor* Kcur = ggml_mul_mat(arena_.ctx(), block.attn_k_weight, cur);
        ggml_tensor* Vcur = ggml_mul_mat(arena_.ctx(), block.attn_v_weight, cur);
        Qcur = ggml_reshape_3d(arena_.ctx(), Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(arena_.ctx(), Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(arena_.ctx(), Vcur, n_embd_head, n_head_kv, n_tokens);
        Qcur = build_norm(gf, Qcur, q_norm_[il], il);
        Kcur = build_norm(gf, Kcur, k_norm_[il], il);
        const float freq_base  = config_.layer_rope_base[il];
        const float freq_scale = config_.layer_rope_scale[il];
        Qcur = ggml_rope_ext(arena_.ctx(), Qcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base, freq_scale, 0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = ggml_rope_ext(arena_.ctx(), Kcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base, freq_scale, 0.0f, 1.0f, 32.0f, 1.0f);
        float kq_scale = 1.0f / sqrtf(static_cast<float>(n_embd_head));
        cur = build_batched_attention(arena_.ctx(), gf, kv_cache_.get(), Qcur, Kcur, Vcur, il, kq_scale, slots, positions, layer_masks[il], gather_indices, il, 0.0f, kv_write_idx,
                                      use_flash_attn());
        cur = ggml_mul_mat(arena_.ctx(), block.attn_output_weight, cur);
        cur = build_norm(gf, cur, post_attn_norm_[il], il);
        ggml_tensor* ffn_inp = ggml_add(arena_.ctx(), cur, inpSA);
        cur = build_norm(gf, ffn_inp, block.ffn_norm_weight, il);
        cur = build_ffn_geglu_tanh(arena_.ctx(), gf, cur, block.ffn_gate_weight, block.ffn_up_weight, block.ffn_down_weight, il);
        cur = build_norm(gf, cur, post_ffn_norm_[il], il);
        cur = ggml_add(arena_.ctx(), cur, ffn_inp);
        inpL = cur;
    }

    // 3. Output head — final norm + tied LM head, routed through
    //    build_output_head so the sparse decode path is honored.
    //    Gemma 3's prefill final norm is STANDARD build_rms_norm (its GGUF
    //    pre-shifts the output_norm weight to (1+w), same as Gemma 2) ⇒
    //    gemma_final_norm=false. No final logit soft-cap in G3 ⇒ 0.0f.
    build_output_head(gf, inpL, /*valid_idx=*/nullptr, /*gemma_final_norm=*/false,
                      /*final_softcap=*/0.0f);

    return gf;
}

