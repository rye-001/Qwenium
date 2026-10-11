#include "gemma_embedding2.h"

#include "engine/model.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"
#include "../layers/ple.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/bidirectional_mask_input.h"

#include "ggml.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const std::string kArch = "gemma-embedding2";
std::string key(const char* suffix) { return kArch + "." + suffix; }
}  // namespace

TokenizerConfig gemma_embedding2_tokenizer_config()
{
    TokenizerConfig cfg;
    cfg.normalizer    = NormalizerKind::SpaceToUnderscore;
    cfg.byte_fallback = true;
    cfg.add_bos_token = true;   // the architecture decides, not the GGUF flag
    cfg.add_eos_token = true;   // EmbeddingGemma reads [BOS] text [EOS]
    return cfg;
}

// ── GemmaEmbedding2Config::from_metadata ─────────────────────────────────────

GemmaEmbedding2Config GemmaEmbedding2Config::from_metadata(const ModelMetadata& meta)
{
    GemmaEmbedding2Config cfg;
    cfg.n_layers     = meta.block_count;
    cfg.n_head       = meta.attention_head_count;
    cfg.hidden_dim   = meta.embedding_length;
    cfg.ffn_dim      = meta.feed_forward_length;
    cfg.context_len  = meta.context_length;
    cfg.rms_norm_eps = meta.rms_norm_eps;

    const auto& kv = meta.raw_kv;
    cfg.head_dim_global  = kv.get_uint32(key("attention.key_length"));
    cfg.head_dim_swa     = kv.get_uint32(key("attention.key_length_swa"));
    cfg.rope_dim_global  = kv.get_uint32(key("rope.dimension_count"));
    cfg.rope_dim_swa     = kv.get_uint32(key("rope.dimension_count_swa"));
    cfg.rope_base_global = kv.get_float (key("rope.freq_base"));
    cfg.rope_base_swa    = kv.get_float (key("rope.freq_base_swa"));
    const uint32_t window = kv.get_uint32(key("attention.sliding_window"));
    if (window == 0 || window % 2 != 0)
        throw std::runtime_error(
            "GemmaEmbedding2Config: 'attention.sliding_window': expected an even total width > 0 "
            "(twice HF's radius), got: " + std::to_string(window));
    cfg.sliding_radius = window / 2;
    cfg.ple_dim          = kv.get_uint32(key("embedding_length_per_layer_input"));
    cfg.embd_out         = kv.get_uint32(key("embedding_length_out"));

    // What this recipe builds, checked against what the file declares.
    const uint32_t v_global = kv.get_uint32(key("attention.value_length"));
    const uint32_t v_swa    = kv.get_uint32(key("attention.value_length_swa"));
    if (v_global != cfg.head_dim_global || v_swa != cfg.head_dim_swa)
        throw std::runtime_error(
            "GemmaEmbedding2Config: 'attention.value_length[_swa]': expected equal to key_length[_swa] (" +
            std::to_string(cfg.head_dim_global) + ", " + std::to_string(cfg.head_dim_swa) + "), got: " +
            std::to_string(v_global) + ", " + std::to_string(v_swa));
    if (kv.get_bool(key("attention.causal")))
        throw std::runtime_error(
            "GemmaEmbedding2Config: 'attention.causal': expected false (an encoder), got: true");
    const uint32_t shared_kv = kv.get_uint32_opt(key("attention.shared_kv_layers")).value_or(0);
    if (shared_kv != 0)
        throw std::runtime_error(
            "GemmaEmbedding2Config: 'attention.shared_kv_layers': expected 0, got: " +
            std::to_string(shared_kv));
    if (cfg.ple_dim == 0)
        throw std::runtime_error(
            "GemmaEmbedding2Config: 'embedding_length_per_layer_input': expected > 0, got: 0");

    // llama.cpp pooling_type: 1 = mean, 3 = last (2 = CLS, 4 = rank: not built).
    const uint32_t pooling = kv.get_uint32(key("pooling_type"));
    if (pooling == 1)      cfg.pooling = IEmbeddingOutput::Pooling::Mean;
    else if (pooling == 3) cfg.pooling = IEmbeddingOutput::Pooling::Last;
    else
        throw std::runtime_error(
            "GemmaEmbedding2Config: 'pooling_type': expected 1 (mean) or 3 (last), got: " +
            std::to_string(pooling));

    // Sliding vs global, per layer, from the file — gemma4's "global iff no
    // attn_v" does not hold here (these global layers carry attn_v).
    const std::vector<bool>& pattern = kv.get_bool_array(key("attention.sliding_window_pattern"));
    if (pattern.size() != cfg.n_layers)
        throw std::runtime_error(
            "GemmaEmbedding2Config: 'attention.sliding_window_pattern' length: expected block_count=" +
            std::to_string(cfg.n_layers) + ", got: " + std::to_string(pattern.size()));
    cfg.is_global.resize(cfg.n_layers);
    cfg.n_kv_heads.resize(cfg.n_layers);
    for (uint32_t il = 0; il < cfg.n_layers; ++il) {
        cfg.is_global[il] = !pattern[il];
        const uint32_t head_dim = cfg.is_global[il] ? cfg.head_dim_global : cfg.head_dim_swa;
        const std::string kk = "blk." + std::to_string(il) + ".attn_k.weight";
        const auto it = meta.tensor_inventory.find(kk);
        if (it == meta.tensor_inventory.end() || it->second.shape.size() < 2)
            throw std::runtime_error(
                "GemmaEmbedding2Config: tensor '" + kk +
                "': expected shape [n_embd, n_kv_heads * head_dim], got: absent or rank < 2");
        const uint64_t width = it->second.shape[1];
        if (width == 0 || width % head_dim != 0)
            throw std::runtime_error(
                "GemmaEmbedding2Config: tensor '" + kk + "' width: expected a multiple of head_dim=" +
                std::to_string(head_dim) + ", got: " + std::to_string(width));
        cfg.n_kv_heads[il] = static_cast<uint32_t>(width / head_dim);
        if (cfg.n_head % cfg.n_kv_heads[il] != 0)
            throw std::runtime_error(
                "GemmaEmbedding2Config: layer " + std::to_string(il) +
                " 'n_kv_heads': expected a divisor of n_head=" + std::to_string(cfg.n_head) +
                ", got: " + std::to_string(cfg.n_kv_heads[il]));
    }
    return cfg;
}

// ── validate_gemma_embedding2_inventory ──────────────────────────────────────

void validate_gemma_embedding2_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                "gemma-embedding2: missing tensor '" + name +
                "': expected in model weights, got absent");
    };
    require("token_embd.weight");
    require("output_norm.weight");
    require("output.weight");                 // 512 → 768 projection, not an LM head
    require("per_layer_model_proj.weight");
    require("per_layer_proj_norm.weight");

    static const char* per_block[] = {
        "attn_norm.weight",
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q_norm.weight", "attn_k_norm.weight",
        "attn_output.weight", "post_attention_norm.weight",
        "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "post_ffw_norm.weight",
        "inp_gate.weight", "proj.weight", "post_norm.weight",
        "layer_output_scale.weight",
    };
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const char* t : per_block) require(p + t);
    }
}

// ── GemmaEmbedding2ForwardPass ───────────────────────────────────────────────

GemmaEmbedding2ForwardPass::GemmaEmbedding2ForwardPass(
    const Model& model, const ModelMetadata* metadata, uint32_t context_len)
    : ForwardPassBase(model, metadata),
      config_(GemmaEmbedding2Config::from_metadata(*metadata)),
      max_tokens_(context_len)
{
    if (max_tokens_ == 0)
        throw std::runtime_error(
            "GemmaEmbedding2ForwardPass: 'context_len': expected > 0, got: 0");

    ple_model_proj_ = require_tensor("per_layer_model_proj.weight");
    ple_proj_norm_  = require_tensor("per_layer_proj_norm.weight");
    output_proj_    = require_tensor("output.weight");
    if (output_proj_->ne[0] != (int64_t)config_.hidden_dim ||
        output_proj_->ne[1] != (int64_t)config_.embd_out)
        throw std::runtime_error(
            "GemmaEmbedding2ForwardPass: tensor 'output.weight' shape: expected [" +
            std::to_string(config_.hidden_dim) + ", " + std::to_string(config_.embd_out) +
            "], got: [" + std::to_string(output_proj_->ne[0]) + ", " +
            std::to_string(output_proj_->ne[1]) + "]");

    block_w_.resize(config_.n_layers);
    for (uint32_t il = 0; il < config_.n_layers; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        BlockWeights& w = block_w_[il];
        w.attn_norm       = require_tensor(p + "attn_norm.weight");
        w.attn_q          = require_tensor(p + "attn_q.weight");
        w.attn_k          = require_tensor(p + "attn_k.weight");
        w.attn_v          = require_tensor(p + "attn_v.weight");
        w.attn_q_norm     = require_tensor(p + "attn_q_norm.weight");
        w.attn_k_norm     = require_tensor(p + "attn_k_norm.weight");
        w.attn_output     = require_tensor(p + "attn_output.weight");
        w.post_attn_norm  = require_tensor(p + "post_attention_norm.weight");
        w.ffn_norm        = require_tensor(p + "ffn_norm.weight");
        w.ffn_gate        = require_tensor(p + "ffn_gate.weight");
        w.ffn_up          = require_tensor(p + "ffn_up.weight");
        w.ffn_down        = require_tensor(p + "ffn_down.weight");
        w.post_ffn_norm   = require_tensor(p + "post_ffw_norm.weight");
        w.ple_inp_gate    = require_tensor(p + "inp_gate.weight");
        w.ple_proj        = require_tensor(p + "proj.weight");
        w.ple_post_norm   = require_tensor(p + "post_norm.weight");
        w.layer_out_scale = require_tensor(p + "layer_output_scale.weight");
    }
}

ggml_tensor* GemmaEmbedding2ForwardPass::require_tensor(const std::string& name) const
{
    ggml_tensor* t = ggml_get_tensor(model_.get_context(), name.c_str());
    if (!t)
        throw std::runtime_error(
            "GemmaEmbedding2ForwardPass: tensor '" + name +
            "': expected in the model context (architecture 'gemma-embedding2'), got absent");
    return t;
}

void GemmaEmbedding2ForwardPass::refuse(const char* what) const
{
    throw std::runtime_error(
        std::string("gemma-embedding2: '") + what +
        "': expected a call this recipe supports, got one it does not — it is an "
        "encoder (no logits, no KV cache, no decode); embed through engine/embed "
        "(CLI: --embed)");
}

ggml_cgraph* GemmaEmbedding2ForwardPass::build_prefill_graph(
    const std::vector<int32_t>&, int, uint32_t, bool) { refuse("build_prefill_graph"); }
ggml_cgraph* GemmaEmbedding2ForwardPass::build_decoding_graph(
    const std::vector<int32_t>&, const std::vector<uint32_t>&,
    const std::vector<int32_t>&) { refuse("build_decoding_graph"); }
void GemmaEmbedding2ForwardPass::advance_cache(uint32_t, uint32_t) { refuse("advance_cache"); }
void GemmaEmbedding2ForwardPass::clear_slot(uint32_t) { refuse("clear_slot"); }
void GemmaEmbedding2ForwardPass::set_cache_pos(uint32_t, uint32_t) { refuse("set_cache_pos"); }
uint32_t GemmaEmbedding2ForwardPass::get_cache_pos(uint32_t) const { refuse("get_cache_pos"); }
void GemmaEmbedding2ForwardPass::clone_slot(uint32_t, uint32_t, uint32_t) { refuse("clone_slot"); }

// ── build_block ──────────────────────────────────────────────────────────────
// Order follows llama.cpp's gemma-embedding2 graph: attention sandwich →
// residual → FFN sandwich → residual → PLE injection → layer_output_scale.

ggml_tensor* GemmaEmbedding2ForwardPass::build_block(
    ggml_cgraph* gf, ggml_tensor* cur, ggml_tensor* inp_pos,
    ggml_tensor* ple_inputs, ggml_tensor* kq_mask, uint32_t il, uint32_t n_tokens)
{
    ggml_context* ctx = arena_.ctx();
    const BlockWeights& w = block_w_[il];
    const bool  is_global  = config_.is_global[il];
    const int   head_dim   = is_global ? config_.head_dim_global  : config_.head_dim_swa;
    const int   rope_dim   = is_global ? config_.rope_dim_global  : config_.rope_dim_swa;
    const float rope_base  = is_global ? config_.rope_base_global : config_.rope_base_swa;
    const int   n_kv_heads = static_cast<int>(config_.n_kv_heads[il]);
    const float eps        = config_.rms_norm_eps;
    const int   iil        = static_cast<int>(il);

    ggml_tensor* inpSA = cur;

    // ── Attention ──
    cur = build_rms_norm(ctx, cur, w.attn_norm, eps, iil);
    ggml_tensor* Qcur = ggml_mul_mat(ctx, w.attn_q, cur);
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w.attn_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w.attn_v, cur);
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, config_.n_head, n_tokens);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_kv_heads,     n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_kv_heads,     n_tokens);
    Qcur = build_rms_norm(ctx, Qcur, w.attn_q_norm, eps, iil);
    Kcur = build_rms_norm(ctx, Kcur, w.attn_k_norm, eps, iil);
    Vcur = ggml_rms_norm(ctx, Vcur, eps);
    Qcur = build_rope_pruned(ctx, Qcur, inp_pos, rope_dim,
                             static_cast<int>(config_.context_len), rope_base);
    Kcur = build_rope_pruned(ctx, Kcur, inp_pos, rope_dim,
                             static_cast<int>(config_.context_len), rope_base);
    // kq_scale 1.0: QK-norm replaces 1/sqrt(d) (same as gemma4).
    cur = build_uncached_attention(ctx, gf, Qcur, Kcur, Vcur, kq_mask, /*kq_scale=*/1.0f,
                                   iil, /*softcap=*/0.0f, use_flash_attn_prefill());
    cur = ggml_mul_mat(ctx, w.attn_output, cur);
    cur = build_rms_norm(ctx, cur, w.post_attn_norm, eps, iil);
    ggml_tensor* attn_out = ggml_add(ctx, cur, inpSA);
    set_tensor_name(gf, attn_out, "attn_residual", iil);

    // ── FFN ──
    cur = build_rms_norm(ctx, attn_out, w.ffn_norm, eps, iil);
    cur = build_ffn_geglu_tanh(ctx, gf, cur, w.ffn_gate, w.ffn_up, w.ffn_down, iil);
    cur = build_rms_norm(ctx, cur, w.post_ffn_norm, eps, iil);
    cur = ggml_add(ctx, cur, attn_out);

    // ── Per-layer embedding, then the layer's output scale ──
    cur = build_ple_inject(ctx, gf, cur, ple_inputs, w.ple_inp_gate, w.ple_proj,
                           w.ple_post_norm, eps, iil);
    cur = ggml_mul(ctx, cur, w.layer_out_scale);
    set_tensor_name(gf, cur, "layer_out", iil);
    return cur;
}

// ── build_embedding_graph ────────────────────────────────────────────────────

ggml_cgraph* GemmaEmbedding2ForwardPass::build_embedding_graph(const std::vector<int32_t>& tokens)
{
    const uint32_t n_tokens = static_cast<uint32_t>(tokens.size());
    if (n_tokens == 0 || n_tokens > max_tokens_)
        throw std::runtime_error(
            "gemma-embedding2: 'tokens': expected 1.." + std::to_string(max_tokens_) +
            " (the pass's context_len), got: " + std::to_string(n_tokens));

    reset_context();
    ggml_cgraph* gf = new_graph();
    ggml_context* ctx = arena_.ctx();

    ggml_tensor* inpL = embedding(gf, tokens);
    inpL = build_embed_scale(ctx, inpL, std::sqrt(static_cast<float>(config_.hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");
    ggml_build_forward_expand(gf, inpL);

    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());

    // Two masks for the whole graph, one per layer kind, shared by every layer
    // of that kind (cast to F16 once under flash). A kind no layer uses gets
    // no mask.
    ggml_tensor* mask_swa    = nullptr;
    ggml_tensor* mask_global = nullptr;
    auto make_mask = [&](const char* name) {
        ggml_tensor* m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
        ggml_set_input(m);
        set_tensor_name(gf, m, name);
        ggml_build_forward_expand(gf, m);
        if (!use_flash_attn_prefill()) return m;
        ggml_tensor* m16 = ggml_cast(ctx, m, GGML_TYPE_F16);
        set_tensor_name(gf, m16, (std::string(name) + "_f16").c_str());
        return m16;
    };
    bool any_swa = false, any_global = false;
    for (bool g : config_.is_global) (g ? any_global : any_swa) = true;
    if (any_swa) {
        mask_swa = make_mask("kq_mask_swa");
        graph_inputs_.add(std::make_unique<BidirectionalMaskInput>(
            BidirectionalMaskInput::band("kq_mask_swa", config_.sliding_radius)));
    }
    if (any_global) {
        mask_global = make_mask("kq_mask_global");
        graph_inputs_.add(std::make_unique<BidirectionalMaskInput>(
            BidirectionalMaskInput::open("kq_mask_global")));
    }

    // Per-layer inputs are a projection of the SCALED embeddings — computed
    // once, sliced per layer.
    ggml_tensor* ple_inputs = build_ple_inputs(ctx, gf, inpL, ple_model_proj_, ple_proj_norm_,
                                               config_.ple_dim, config_.n_layers,
                                               config_.rms_norm_eps);

    for (uint32_t il = 0; il < config_.n_layers; ++il)
        inpL = build_block(gf, inpL, inp_pos, ple_inputs,
                           config_.is_global[il] ? mask_global : mask_swa, il, n_tokens);

    ggml_tensor* cur = build_rms_norm(ctx, inpL, model_.get_output_norm_weight(),
                                      config_.rms_norm_eps, /*il=*/-1);
    cur = ggml_mul_mat(ctx, output_proj_, cur);     // [embd_out, n_tokens]
    ggml_set_name(cur, "embd_tokens");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
