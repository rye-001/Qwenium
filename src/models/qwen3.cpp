#include "qwen3.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/transformer_block.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/gather_indices_input.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>
#include <cinttypes>
#include <memory>

// ── Qwen tokenizer config ─────────────────────────────────────────────────────

TokenizerConfig qwen_tokenizer_config()
{
    TokenizerConfig cfg;
    cfg.normalizer    = NormalizerKind::None;
    cfg.byte_fallback = false;
    cfg.add_bos_token = false;
    // <|im_start|> and <|im_end|> are CONTROL-typed in real Qwen GGUFs and
    // are caught by the generic CONTROL scan.  List them explicitly so the
    // intent is self-documenting and any GGUF that mislabels them still works.
    cfg.extra_chat_specials = {"<|im_start|>", "<|im_end|>"};
    return cfg;
}

constexpr size_t GRAPH_SIZE = 16384;

Qwen3ForwardPass::Qwen3ForwardPass(
    const Model& model, const ModelMetadata* metadata,
    uint32_t context_len, uint32_t max_batch_size)
    : ForwardPassBase(model, metadata) {
        ggml_backend_t cache_backend = model_.has_metal_backend()
            ? model_.get_backend_metal()
            : model_.get_backend_cpu();

            uint32_t n_embd_k, n_embd_v;
            if (meta_.architecture == "qwen2") {
                n_embd_k = meta_.embedding_length / meta_.attention_head_count * meta_.attention_head_count_kv;
                n_embd_v = meta_.embedding_length / meta_.attention_head_count * meta_.attention_head_count_kv;
            } else { // qwen3
                n_embd_k = meta_.attention_key_length * meta_.attention_head_count_kv;
                n_embd_v = meta_.attention_value_length * meta_.attention_head_count_kv;
            }

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

struct ggml_cgraph* Qwen3ForwardPass::build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx, [[maybe_unused]] bool want_logits) {
    reset_context();
    ggml_cgraph* gf = new_graph();
    int n_layers = meta_.block_count;        // Layers 0-27
    int hidden_dim = meta_.embedding_length;    // Model dimension
    int n_head = meta_.attention_head_count;       // Query heads
    int n_head_kv = meta_.attention_head_count_kv;       // KV heads (GQA: 2:1 ratio)
    
    int n_embd_head;
    if (meta_.architecture == "qwen3") {
        n_embd_head = meta_.attention_key_length;
    } else { // qwen2
        n_embd_head = hidden_dim / n_head;
    }

    // 1. Token embedding lookup
    const size_t n_tokens = tokens.size();

    // Token embedding lookup
    ggml_tensor * inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // Position tensor
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);  // Mark as input tensor
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // Typed inputs for this graph (replaces set_inputs). Cleared here so
    // build_output_head can append SparseHeadInput on the sparse path; one
    // uniform causal mask per (all-attention) layer, no sliding window.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    for (uint32_t il = 0; il < static_cast<uint32_t>(n_layers); ++il)
        graph_inputs_.add(std::make_unique<AttnMaskInput>(
            "kq_mask." + std::to_string(il), 0u));

    // memset(inp_pos->data, 0, ggml_nbytes(inp_pos));
    // for (int i = 0; i < n_tokens; ++i) {
    //     ggml_set_i32_1d(inp_pos, i, pos + i);
    // }

    // 2. Loop through each transformer block
    TransformerBlockHparams blk_hp;
    blk_hp.is_qwen2      = (meta_.architecture == "qwen2");
    blk_hp.n_head        = n_head;
    blk_hp.n_head_kv     = n_head_kv;
    blk_hp.n_embd_head   = n_embd_head;
    blk_hp.freq_base     = meta_.rope_freq_base;
    blk_hp.context_length = static_cast<int>(meta_.context_length);
    blk_hp.rms_norm_eps  = meta_.rms_norm_eps;

    for (uint32_t il = 0; il < static_cast<uint32_t>(n_layers); ++il) {
        auto& block = model_.get_block(il);
        TransformerBlockWeights w;
        w.attn_norm = block.attn_norm_weight;
        w.q         = block.attn_q_weight;
        w.k         = block.attn_k_weight;
        w.v         = block.attn_v_weight;
        w.q_bias    = block.attn_q_bias;
        w.k_bias    = block.attn_k_bias;
        w.v_bias    = block.attn_v_bias;
        w.q_norm    = block.attn_q_norm_weight;
        w.k_norm    = block.attn_k_norm_weight;
        w.out       = block.attn_output_weight;
        w.ffn_norm  = block.ffn_norm_weight;
        w.ffn_gate  = block.ffn_gate_weight;
        w.ffn_up    = block.ffn_up_weight;
        w.ffn_down  = block.ffn_down_weight;

        inpL = build_transformer_layer(ctx_, gf, kv_cache_.get(), inpL, inp_pos,
                                       w, blk_hp, il, slot_idx,
                                       static_cast<uint32_t>(n_tokens));
    }

    // 3. Final normalization and output projection
    build_output_head(gf, inpL);

    return gf;
}

struct ggml_cgraph* Qwen3ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens, 
    const std::vector<uint32_t>& slots, 
    const std::vector<int32_t>& positions
) {
    // Reset context
    if (ctx_) {
        ggml_free(ctx_);
    }
    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true, 
    };
    ctx_ = ggml_init(params);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx_, GRAPH_SIZE, false);
    
    // Core parameters
    int n_layers = meta_.block_count;
    int hidden_dim = meta_.embedding_length;
    int n_head = meta_.attention_head_count;
    int n_head_kv = meta_.attention_head_count_kv;
    
    int n_embd_head;
    if (meta_.architecture == "qwen3") {
        n_embd_head = meta_.attention_key_length;
    } else { // qwen2
        n_embd_head = hidden_dim / n_head;
    }

    const int n_rot = n_embd_head;
    const size_t n_tokens = tokens.size(); // Total tokens across all slots

    // 1. Embeddings (batched)
    ggml_tensor * inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // Position tensor (vector of positions, one per token)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    uint32_t max_pos = 0;
    for (uint32_t s : slots) {
        uint32_t p = get_cache_pos(s);
        if (p > max_pos) max_pos = p;
    }
    uint32_t n_kv_len = max_pos + 1;

    // Attention Mask (shared across layers)
    // Shape: [n_kv_len, n_tokens, 1]
    // ggml_tensor* kq_mask = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, n_kv_len, n_tokens, 1);
    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, n_kv_len, 1, 1, n_tokens);
    
    ggml_set_input(kq_mask);
    ggml_set_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    // KV Gather Indices (shared across layers)
    // Shape: [n_tokens * n_kv_len] (1D tensor of indices)
    uint32_t n_total_indices = n_tokens * n_kv_len;
    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_total_indices);
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // Typed inputs for the decode graph (replaces set_batched_inputs).
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    graph_inputs_.add(std::make_unique<AttnMaskInput>("kq_mask_b", 0u));
    graph_inputs_.add(std::make_unique<GatherIndicesInput>(
        kv_cache_->get_n_ctx_max()));

    // 2. Transformer Blocks
    ggml_tensor * cur;
    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor * inpSA = inpL;
        auto& block = model_.get_block(il);

        // A. Norm
        cur = build_norm(gf, inpL, block.attn_norm_weight, il);
        
        // B. Q, K, V
        ggml_tensor* Qcur = ggml_mul_mat(ctx_, block.attn_q_weight, cur);
        if (meta_.architecture == "qwen2") Qcur = ggml_add(ctx_, Qcur, block.attn_q_bias);
        
        ggml_tensor* Kcur = ggml_mul_mat(ctx_, block.attn_k_weight, cur);
        if (meta_.architecture == "qwen2") Kcur = ggml_add(ctx_, Kcur, block.attn_k_bias);
        
        ggml_tensor* Vcur = ggml_mul_mat(ctx_, block.attn_v_weight, cur);
        if (meta_.architecture == "qwen2") Vcur = ggml_add(ctx_, Vcur, block.attn_v_bias);

        // Reshape [n_head, n_embd_head, n_tokens]
        Qcur = ggml_reshape_3d(ctx_, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head_kv, n_tokens);

        // Qwen3 Norm
        if (meta_.architecture == "qwen3") {
            Qcur = build_norm(gf, Qcur, block.attn_q_norm_weight, il);
            Kcur = build_norm(gf, Kcur, block.attn_k_norm_weight, il);
        }

        // RoPE (using vector positions)
        float freq_base = meta_.rope_freq_base;
        Qcur = ggml_rope_ext(ctx_, Qcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = ggml_rope_ext(ctx_, Kcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // D. Attention (Batched)
        float kq_scale = 1.0f/sqrtf(float(n_embd_head));
        cur = build_batched_attention(ctx_, gf, kv_cache_.get(), Qcur, Kcur, Vcur, il, kq_scale, slots, positions, kq_mask, gather_indices, il);

        // E. Output Projection & Residual
        cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);
        ggml_tensor * ffn_inp = ggml_add(ctx_, cur, inpSA);

        // F. FFN
        cur = build_norm(gf, ffn_inp, block.ffn_norm_weight, il);
        cur = build_ffn_swiglu(ctx_, gf, cur, block.ffn_gate_weight, block.ffn_up_weight, block.ffn_down_weight, il);
        
        // H. FFN Residual
        cur = ggml_add(ctx_, cur, ffn_inp);
        inpL = cur;
    }
    
    // 3. Output Head — shared helper (final norm + LM head). Routes through
    //    build_output_head so the sparse decode path (sparse_decode_ids_ →
    //    ggml_get_rows on the output weight) is honored, matching the prefill
    //    path (line ~182). Dense behavior is unchanged: with no sparse ids
    //    armed this is build_norm + ggml_mul_mat over the full output weight
    //    (or token-embedding fallback), identical to the prior hand-rolled
    //    code. Without this, grammar-constrained decode arms a 29-element
    //    sparse set but the graph returns full-vocab logits → sample_sparse
    //    size-mismatch / out-of-bounds (the Qwen3.5 bad-access class).
    build_output_head(gf, inpL);

    return gf;
}

// _build_attention_layer and _build_batched_attention_layer have been
// extracted to src/layers/attention.cpp as build_attention() and
// build_batched_attention(). Call sites updated to use the free functions.


// set_inputs / set_batched_inputs removed: inputs are now populated by the
// typed GraphInputSet (graph_inputs_) built in build_prefill_graph and
// build_decoding_graph. See docs/plan-typed-graph-inputs.md.


// ── Inventory validator ──────────────────────────────────────────────────────

void validate_qwen3_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name, const std::string& ctx) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                meta.architecture + ": missing tensor '" + name +
                "': expected in " + ctx + ", got absent");
    };
    require("token_embd.weight", "model weights");
    require("output_norm.weight", "model weights");

    static const std::vector<std::string> per_block = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
        "ffn_up.weight", "ffn_down.weight"
    };
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block)
            require(p + t, "block " + std::to_string(i));
    }
}

