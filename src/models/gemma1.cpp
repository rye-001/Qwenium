#include "gemma1.h"

#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/norm.h"
#include "../layers/transformer_block.h"

#include "ggml.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

// ── Gemma tokenizer config ────────────────────────────────────────────────────

TokenizerConfig gemma1_tokenizer_config()
{
    TokenizerConfig cfg;
    cfg.normalizer    = NormalizerKind::SpaceToUnderscore;
    cfg.byte_fallback = true;
    cfg.add_bos_token = true;
    // Some GGUF exports label these as NORMAL rather than USER_DEFINED.
    cfg.extra_chat_specials = {"<start_of_turn>", "<end_of_turn>"};
    return cfg;
}

constexpr size_t GEMMA1_GRAPH_SIZE = 16384;

Gemma1ForwardPass::Gemma1ForwardPass(
    const Model& model, const ModelMetadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, int kv_quant_bits)
    : ForwardPassBase(model, metadata)
{
    if (kv_quant_bits != 0) {
        // TurboQuant for Gemma is not in scope for PR G1.5. The fail-loud
        // contract: refuse rather than silently produce wrong outputs.
        throw std::runtime_error(
            "Gemma1ForwardPass: kv_quant_bits != 0 not supported in this phase "
            "(architecture='gemma'); expected 0, got " +
            std::to_string(kv_quant_bits));
    }

    ggml_backend_t cache_backend = model_.has_metal_backend()
        ? model_.get_backend_metal()
        : model_.get_backend_cpu();

    const uint32_t n_embd_k = meta_.attention_key_length   * meta_.attention_head_count_kv;
    const uint32_t n_embd_v = meta_.attention_value_length * meta_.attention_head_count_kv;

    kv_cache_ = std::make_unique<simple_kv_cache>(
        meta_.block_count,
        context_len,
        max_batch_size,
        n_embd_k,
        n_embd_v,
        GGML_TYPE_F32,
        GGML_TYPE_F32,
        cache_backend
    );
}

ggml_cgraph* Gemma1ForwardPass::build_prefill_graph(
    const std::vector<int32_t>& tokens, int /*pos*/, uint32_t slot_idx)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const int n_layers     = meta_.block_count;
    const int hidden_dim   = meta_.embedding_length;
    const int n_head       = meta_.attention_head_count;
    const int n_head_kv    = meta_.attention_head_count_kv;
    const int n_embd_head  = meta_.attention_key_length;
    const size_t n_tokens  = tokens.size();

    // 1. Token embedding lookup followed by sqrt(d_model) scaling (Gemma).
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");
    inpL = build_embed_scale(ctx_, inpL, std::sqrt(static_cast<float>(hidden_dim)));
    set_tensor_name(gf, inpL, "inpL_scaled");
    // Diagnostic-only: keep this tensor's buffer alive so DumpInpL* can read it.
    // Removing this in steady state is a 1-line revert; cheap to leave for now.
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);

    // Position tensor.
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 2. Transformer stack.
    TransformerBlockHparams blk_hp;
    blk_hp.is_qwen2        = false;       // GQA without bias.
    blk_hp.n_head          = n_head;
    blk_hp.n_head_kv       = n_head_kv;
    blk_hp.n_embd_head     = n_embd_head;
    blk_hp.freq_base       = (meta_.rope_freq_base > 0) ? meta_.rope_freq_base : 10000.0f;
    blk_hp.context_length  = static_cast<int>(meta_.context_length);
    blk_hp.rms_norm_eps    = meta_.rms_norm_eps;
    // GGUF Gemma exports pre-shift the norm weights to (1+w) at conversion
    // time (llama.cpp convert_hf_to_gguf.py: `data_torch + 1` for *.norm.weight).
    // So at runtime the norm is just `_norm(x) * w_gguf` — same as Qwen.
    // Setting gemma_rms_norm=false keeps the existing build_rms_norm path.
    blk_hp.gemma_rms_norm  = false;
    blk_hp.gemma_geglu     = true;

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
        w.q_norm    = nullptr;  // Gemma 1 has no QK norm.
        w.k_norm    = nullptr;
        w.out       = block.attn_output_weight;
        w.ffn_norm  = block.ffn_norm_weight;
        w.ffn_gate  = block.ffn_gate_weight;
        w.ffn_up    = block.ffn_up_weight;
        w.ffn_down  = block.ffn_down_weight;

        inpL = build_transformer_layer(ctx_, gf, kv_cache_.get(), inpL, inp_pos,
                                       w, blk_hp, il, slot_idx,
                                       static_cast<uint32_t>(n_tokens));
        // Diagnostic: keep each layer's output alive so we can compare per-layer.
        char dbg[64];
        std::snprintf(dbg, sizeof(dbg), "layer_out.%u", il);
        set_tensor_name(gf, inpL, dbg);
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    // 3. Output head: Gemma final norm (1+w) → tied LM head matmul.
    // Also keep the post-block residual stream alive for diagnostics.
    ggml_set_output(inpL);
    ggml_build_forward_expand(gf, inpL);
    set_tensor_name(gf, inpL, "post_layers");

    ggml_tensor* cur = build_rms_norm_gemma(
        ctx_, inpL, model_.get_output_norm_weight(),
        meta_.rms_norm_eps, /*il=*/-1);
    set_tensor_name(gf, cur, "final_norm");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (model_.get_output_weight() != nullptr) {
        cur = ggml_mul_mat(ctx_, model_.get_output_weight(), cur);
    } else {
        cur = ggml_mul_mat(ctx_, model_.get_token_embedding_weight(), cur);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph* Gemma1ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& /*tokens*/,
    const std::vector<uint32_t>& /*slots*/,
    const std::vector<int32_t>& /*positions*/)
{
    throw std::runtime_error(
        "Gemma1ForwardPass::build_decoding_graph: batched decode not "
        "implemented in PR G1.5; expected: prefill-only path, got: batched call");
}

void Gemma1ForwardPass::set_inputs(ggml_cgraph* gf,
                                   const std::vector<int32_t>& tokens,
                                   int pos)
{
    const uint32_t n_tokens = static_cast<uint32_t>(tokens.size());

    ggml_tensor* tokens_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tokens_t) {
        throw std::runtime_error("Gemma1ForwardPass::set_inputs: 'tokens' tensor not found in graph");
    }
    ggml_backend_tensor_set(tokens_t, tokens.data(), 0,
                            tokens.size() * sizeof(int32_t));

    ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (!inp_pos) {
        throw std::runtime_error("Gemma1ForwardPass::set_inputs: 'inp_pos' tensor not found in graph");
    }
    std::vector<int32_t> positions(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        positions[i] = pos + static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(inp_pos, positions.data(), 0,
                            positions.size() * sizeof(int32_t));

    // Per-layer causal kq_mask. build_attention allocates one mask tensor per
    // layer, named "kq_mask.{il}". Without populating these the attention
    // softmax sees uninitialized memory and the model produces noise — this
    // is the same wiring Qwen3ForwardPass::set_inputs does.
    for (uint32_t il = 0; il < meta_.block_count; ++il) {
        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, name);
        if (!kq_mask) {
            throw std::runtime_error(
                "Gemma1ForwardPass::set_inputs: kq_mask tensor not found "
                "for layer " + std::to_string(il));
        }
        const uint32_t n_kv = kq_mask->ne[0];
        std::vector<float> mask(n_kv * n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const uint32_t q_pos = pos + i;
            for (uint32_t j = 0; j < n_kv; ++j) {
                mask[i * n_kv + j] = (j <= q_pos) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0,
                                n_kv * n_tokens * sizeof(float));
    }
}

void Gemma1ForwardPass::set_batched_inputs(
    ggml_cgraph* /*gf*/,
    const std::vector<int32_t>& /*tokens*/,
    const std::vector<uint32_t>& /*slots*/,
    const std::vector<int32_t>& /*positions*/)
{
    throw std::runtime_error(
        "Gemma1ForwardPass::set_batched_inputs: batched decode not implemented "
        "in PR G1.5; expected: prefill-only path, got: batched call");
}

// ── Inventory validator ──────────────────────────────────────────────────────

void validate_gemma1_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                "gemma: missing tensor '" + name +
                "': expected in model weights, got absent");
    };

    // Tied embeddings: no separate output.weight.
    require("token_embd.weight");
    require("output_norm.weight");

    static const std::vector<std::string> per_block = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
        "ffn_up.weight", "ffn_down.weight"
    };
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block) require(p + t);
    }
}
