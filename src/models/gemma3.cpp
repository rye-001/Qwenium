#include "gemma3.h"

#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"
#include "../layers/transformer_block.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/image_embeddings_input.h"

#include <cstdlib>
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>

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
    uint32_t context_len, uint32_t max_batch_size)
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
        GGML_TYPE_F32,
        GGML_TYPE_F32,
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
    inpL = build_embed_scale(ctx_, inpL, std::sqrt(static_cast<float>(hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);

    // Position tensor.
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
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
    // Bidirectional image attention is the reference-correct behavior for
    // Gemma 3 (llama.cpp mtmd: mtmd_decode_use_non_causal == true for
    // PROJECTOR_TYPE_GEMMA3 — the image chunk is decoded with causal_attn=false).
    // It is therefore enabled by default whenever image embeddings are armed.
    //
    // This was previously gated off because bidi corrupted generation (the model
    // repeated <start_of_image>). Root cause: global layers ignored the GGUF's
    // linear RoPE scaling (gemma3.rope.scaling.factor=8 → freq_scale=0.125), so
    // their relative positions were 8x too large. Causal masking hides this
    // (attention is local-dominated) but the bidi span forces image tokens to
    // attend across the full ±n_img span, where the position error is fatal.
    // Fixed by Gemma3Config::layer_rope_scale (wired into blk_hp.freq_scale).
    // Set QINF_GEMMA3_IMAGE_CAUSAL=1 to force the legacy causal span for A/B
    // investigation. See docs/gemma-vision-handoff.md and plan-gemma-vision-impl.md §P4.
    const bool     use_bidi      = img_armed &&
        (std::getenv("QINF_GEMMA3_IMAGE_CAUSAL") == nullptr);
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
    // site between this recipe and the vision subsystem). When image
    // embeddings are armed (set_image_embeddings), overwrite the SCALED text
    // embeddings at [span_start, span_start + n_img) with the precomputed
    // image embeddings. Gemma 3 scales token embeddings by sqrt(d_model) but
    // image embeddings enter the residual stream UNSCALED — llama.cpp gemma3:
    //   inpL = ggml_scale(inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    // Substituting AFTER build_embed_scale and overwriting the rows wholesale
    // is exactly that semantics (the scaled values at those positions are
    // discarded). Single parameterized site, mirroring the want_logits head
    // guard's locality discipline. Consumed (moved) here; re-arm to repeat.
    if (!image_embd_.empty()) {
        const uint32_t n_img = image_n_tokens_;
        const size_t   want  = static_cast<size_t>(hidden_dim) * n_img;
        if (image_embd_.size() != want)
            throw std::runtime_error(
                "Gemma3ForwardPass: slot 'image_embeddings': expected " +
                std::to_string(want) + " floats (hidden_dim=" +
                std::to_string(hidden_dim) + " * n_img=" +
                std::to_string(n_img) + "), got: " +
                std::to_string(image_embd_.size()));
        if (image_span_start_ < 0 ||
            static_cast<size_t>(image_span_start_) + n_img > n_tokens)
            throw std::runtime_error(
                "Gemma3ForwardPass: slot 'image_span': expected span within "
                "[0, " + std::to_string(n_tokens) + "), got: start=" +
                std::to_string(image_span_start_) + " n_img=" +
                std::to_string(n_img));

        ggml_tensor* img_in = ggml_new_tensor_2d(
            ctx_, GGML_TYPE_F32, hidden_dim, static_cast<int64_t>(n_img));
        ggml_set_input(img_in);
        set_tensor_name(gf, img_in, "image_embeddings");
        ggml_build_forward_expand(gf, img_in);

        // Overwrite the image-span columns of the residual stream in place:
        //   inpL[:, span : span+n_img] = img_in
        // inpL is [hidden_dim, n_tokens] and contiguous, so the image span is a
        // clean 2-D sub-block; ggml_set_2d copies inpL and writes img_in at the
        // column offset (one op, one node). The image columns enter UNSCALED
        // while the surviving text columns keep their sqrt(d_model) scale —
        // exactly llama.cpp gemma3.cpp:93 (scale = ubatch.token ? sqrt : 1.0),
        // since the scaled text values at those positions are discarded.
        //
        // NOTE: an earlier revision used a 3-way ggml_concat splice with a
        // comment claiming ggml_set_2d was a "silent no-op." That was backwards:
        // the differential test (SubstitutedEmbeddingsChangeLogits) showed the
        // CONCAT path never propagated img_in into the residual stream (output
        // bit-identical to the text-only baseline, even with img_in scaled
        // 1000x), while ggml_set_2d substitutes correctly. set_2d is the right
        // primitive for an in-place sub-block overwrite.
        const int64_t span = image_span_start_;
        inpL = ggml_set_2d(ctx_, inpL, img_in, inpL->nb[1],
                           static_cast<size_t>(span) * inpL->nb[1]);
        set_tensor_name(gf, inpL, "inpL_image_subst");
        ggml_build_forward_expand(gf, inpL);

        graph_inputs_.add(
            std::make_unique<ImageEmbeddingsInput>(std::move(image_embd_)));
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
            ctx_, build_out_ids_slice(gf, inpL), model_.get_output_norm_weight(),
            config_.rms_norm_eps, /*il=*/-1);
        set_tensor_name(gf, cur, "final_norm");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        // LM head (tied embeddings — no separate output.weight in Gemma 3).
        if (model_.get_output_weight() != nullptr) {
            cur = ggml_mul_mat(ctx_, model_.get_output_weight(), cur);
        } else {
            cur = ggml_mul_mat(ctx_, model_.get_token_embedding_weight(), cur);
        }

        // No final soft-cap for Gemma 3 (removed in G3 vs. G2).

        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
    }

    return gf;
}

// ── build_decoding_graph ──────────────────────────────────────────────────────

ggml_cgraph* Gemma3ForwardPass::build_decoding_graph(
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
        "Gemma3ForwardPass::build_decoding_graph: batched decode not "
        "implemented in PR G3.4; expected: prefill-only path, got: batched call");
}

