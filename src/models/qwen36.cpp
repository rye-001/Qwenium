#include "qwen36.h"

#include "../layers/attention.h"
#include "../layers/deltanet.h"
#include "../layers/ffn.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

// ── Constructor ──────────────────────────────────────────────────────────────

Qwen36ForwardPass::Qwen36ForwardPass(
    const Qwen3Model&    model,
    const Qwen3Metadata* metadata,
    uint32_t             context_len,
    uint32_t             max_batch_size,
    int                  /*kv_quant_bits*/)
    : ForwardPassBase(model, metadata)
{
    const auto& m = *metadata;

    // Cache attention hparams — these never change per token.
    n_embd_head_ = static_cast<int>(m.attention_key_length);   // 256
    n_head_       = static_cast<int>(m.attention_head_count);   // 16
    n_head_kv_    = static_cast<int>(m.attention_head_count_kv);// 2
    n_rot_        = static_cast<int>(m.rope_dimension_count);   // 64 (partial)

    // MoE hparams — same for every layer.
    moe_hp_ = MoELayer::Hparams{
        static_cast<int>(m.expert_count),                    // 256
        static_cast<int>(m.expert_used_count),               // 8
        static_cast<int>(m.expert_feed_forward_length),      // 512
        true                                                 // has_shared_expert
    };

    // Build physical→logical layer index maps.
    kv_layer_map_.assign(m.block_count, -1);
    dn_layer_map_.assign(m.block_count, -1);
    int kv_idx = 0, dn_idx = 0;
    for (uint32_t il = 0; il < m.block_count; ++il) {
        if (m.is_full_attention_layer(il))
            kv_layer_map_[il] = kv_idx++;
        else
            dn_layer_map_[il] = dn_idx++;
    }
    const int n_kv_layers = kv_idx;  // 10
    const int n_dn_layers = dn_idx;  // 30

    // KV cache — 10 attention layers, F32, on Metal if available.
    ggml_backend_t cache_backend = model_.has_metal_backend()
        ? model_.get_backend_metal()
        : model_.get_backend_cpu();

    const uint32_t n_embd_k = static_cast<uint32_t>(n_head_kv_ * n_embd_head_);
    const uint32_t n_embd_v = n_embd_k;

    kv_cache_ = std::make_unique<simple_kv_cache>(
        static_cast<uint32_t>(n_kv_layers),
        context_len,
        max_batch_size,
        n_embd_k, n_embd_v,
        GGML_TYPE_F32, GGML_TYPE_F32,
        cache_backend);

    // DeltaNet state — 30 DeltaNet layers, backend-backed.
    const uint32_t d_inner       = m.ssm_inner_size;     // 4096
    const uint32_t num_v_heads   = m.ssm_time_step_rank; // 32
    const uint32_t num_k_heads   = m.ssm_group_count;    // 16
    const uint32_t head_v_dim    = d_inner / num_v_heads; // 128
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.ssm_state_size; // 8192

    DeltaNetState::Hparams dn_state_hp{
        static_cast<uint32_t>(n_dn_layers),
        max_batch_size,
        head_v_dim,
        m.ssm_state_size,  // head_k_dim = 128
        num_v_heads,
        conv_channels,
        m.ssm_conv_kernel, // 4
        cache_backend
    };
    dn_state_ = std::make_unique<DeltaNetState>(dn_state_hp);
}

// ── Private helpers ──────────────────────────────────────────────────────────

// Inline MoE FFN for one physical layer, after the pre-FFN norm has been applied.
// Returns the FFN output (before residual). il is the physical layer index.
static ggml_tensor* build_moe_layer(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  cur,
    const TransformerBlock& blk,
    const MoELayer::Hparams& hp,
    int il)
{
    MoELayer moe(
        blk.moe_router_weight,
        blk.moe_exp_gate_weight,
        blk.moe_exp_up_weight,
        blk.moe_exp_down_weight,
        blk.moe_shexp_gate_w,
        blk.moe_shexp_up_weight,
        blk.moe_shexp_down_weight,
        blk.moe_shexp_gate,
        hp);
    return moe.build(ctx, gf, cur, Phase::Prefill, il);
}

// Build the DeltaNet subgraph for physical layer il (DeltaNet index dn_idx).
static ggml_tensor* build_dn_layer(
    ggml_context*   ctx,
    ggml_cgraph*    gf,
    ggml_tensor*    cur,
    const TransformerBlock& blk,
    DeltaNetState*  dn_state,
    const DeltaNetState::Hparams& state_hp,
    uint32_t num_k_heads,
    uint32_t n_embd,
    uint32_t dn_idx,
    uint32_t n_tokens,
    uint32_t slot_idx,
    float    rms_norm_eps,
    int      il)
{
    return build_deltanet_layer(
        ctx, gf, cur,
        dn_state,
        dn_idx, slot_idx, n_tokens,
        blk.attn_qkv_weight,
        blk.attn_gate_weight,
        blk.ssm_beta_weight,
        blk.ssm_alpha_weight,
        blk.ssm_dt_bias,
        blk.ssm_a,
        blk.ssm_conv1d_weight,
        blk.ssm_norm_weight,
        blk.ssm_out_weight,
        static_cast<int>(n_embd),                                          // n_embd
        static_cast<int>(state_hp.head_v_dim * state_hp.num_v_heads),      // d_inner
        static_cast<int>(state_hp.head_k_dim),
        static_cast<int>(num_k_heads),
        static_cast<int>(state_hp.num_v_heads),
        static_cast<int>(state_hp.head_v_dim),
        static_cast<int>(state_hp.conv_channels),
        static_cast<int>(state_hp.conv_kernel),
        rms_norm_eps,
        il);
}

// ── build_prefill_graph ──────────────────────────────────────────────────────

ggml_cgraph* Qwen36ForwardPass::build_prefill_graph(
    const std::vector<int32_t>& tokens,
    int pos, uint32_t slot_idx)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const auto& m        = meta_;
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());

    // Derive DeltaNet state hparams from metadata for the helper.
    const uint32_t d_inner       = m.ssm_inner_size;
    const uint32_t num_v_heads   = m.ssm_time_step_rank;
    const uint32_t num_k_heads   = m.ssm_group_count;
    const uint32_t head_v_dim    = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.ssm_state_size;
    DeltaNetState::Hparams dn_hp{
        0, 0,             // n_dn_layers / n_slots not used in helper
        head_v_dim,
        m.ssm_state_size,
        num_v_heads,
        conv_channels,
        m.ssm_conv_kernel,
        nullptr
    };

    // 1. Token embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // 2. Position tensor (shared by all attention layers)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. Transformer loop
    for (uint32_t il = 0; il < m.block_count; ++il) {
        const auto& blk = model_.get_block(il);
        ggml_tensor* inpSA = inpL;

        // ── Pre-attention norm ──────────────────────────────────────────────
        ggml_tensor* cur = build_norm(gf, inpL, blk.attn_norm_weight, il);

        // ── Attention or DeltaNet ───────────────────────────────────────────
        if (m.is_full_attention_layer(il)) {
            int kv_idx = kv_layer_map_[il];
            // Qwen3.6 uses a joint Q+Gate projection: Q weight outputs
            // [(n_embd_head*2)*n_head, n_tokens]. build_qwen35_attention
            // handles the strided view split, sigmoid gating, and out-proj.
            cur = build_qwen35_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos,
                kv_idx, n_tok, slot_idx, il,
                blk.attn_q_weight, blk.attn_q_norm_weight,
                blk.attn_k_weight, blk.attn_k_norm_weight,
                blk.attn_v_weight, blk.attn_output_weight,
                n_embd_head_, n_head_, n_head_kv_,
                n_rot_, m.rope_freq_base,
                static_cast<int>(m.context_length),
                m.rms_norm_eps);
        } else {
            // DeltaNet layer
            uint32_t dn_idx = static_cast<uint32_t>(dn_layer_map_[il]);
            cur = build_dn_layer(ctx_, gf, cur, blk, dn_state_.get(),
                                 dn_hp, num_k_heads, m.embedding_length, dn_idx, n_tok, slot_idx,
                                 m.rms_norm_eps, il);
        }

        // ── Residual 1 (attention / DeltaNet) ──────────────────────────────
        cur = ggml_add(ctx_, cur, inpSA);

        // ── Pre-FFN norm ────────────────────────────────────────────────────
        ggml_tensor* ffn_inp = cur;
        cur = build_norm(gf, cur, blk.ffn_norm_weight, il);

        // ── MoE FFN ─────────────────────────────────────────────────────────
        cur = build_moe_layer(ctx_, gf, cur, blk, moe_hp_, il);

        // ── Residual 2 (FFN) ─────────────────────────────────────────────────
        cur = ggml_add(ctx_, cur, ffn_inp);
        set_tensor_name(gf, cur, "layer_out", il);

        inpL = cur;
    }

    // 4. Final norm + LM head
    build_output_head(gf, inpL);

    return gf;
}

// ── build_decoding_graph ─────────────────────────────────────────────────────

ggml_cgraph* Qwen36ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>&  positions)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const auto&    m      = meta_;
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());

    // Derive DeltaNet state hparams from metadata for the helper.
    const uint32_t d_inner       = m.ssm_inner_size;
    const uint32_t num_v_heads   = m.ssm_time_step_rank;
    const uint32_t num_k_heads   = m.ssm_group_count;
    const uint32_t head_v_dim    = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.ssm_state_size;
    DeltaNetState::Hparams dn_hp{
        0, 0, head_v_dim, m.ssm_state_size,
        num_v_heads, conv_channels, m.ssm_conv_kernel, nullptr
    };

    // 1. Token embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // 2. Position tensor (one per batch element)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_batch);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. KV gather mask — shared across all attention layers.
    uint32_t max_physical = 0;
    for (uint32_t s : slots) {
        uint32_t phys = get_physical_cache_pos(s);
        if (phys > max_physical) max_physical = phys;
    }
    const uint32_t n_kv_len = max_physical + 1;  // +1 for token being written

    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32,
                                               n_kv_len, 1, 1, n_batch);
    ggml_set_input(kq_mask);
    ggml_set_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    ggml_tensor* gather_indices = ggml_new_tensor_1d(
        ctx_, GGML_TYPE_I32, static_cast<int64_t>(n_batch * n_kv_len));
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // 4. Transformer loop
    for (uint32_t il = 0; il < m.block_count; ++il) {
        const auto& blk = model_.get_block(il);
        ggml_tensor* inpSA = inpL;

        // Pre-attention norm
        ggml_tensor* cur = build_norm(gf, inpL, blk.attn_norm_weight, il);

        if (m.is_full_attention_layer(il)) {
            int kv_idx = kv_layer_map_[il];
            cur = build_qwen35_batched_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos,
                kq_mask, gather_indices,
                kv_idx, slots, positions, il,
                blk.attn_q_weight, blk.attn_q_norm_weight,
                blk.attn_k_weight, blk.attn_k_norm_weight,
                blk.attn_v_weight, blk.attn_output_weight,
                n_embd_head_, n_head_, n_head_kv_,
                n_rot_, m.rope_freq_base,
                static_cast<int>(m.context_length),
                m.rms_norm_eps);
        } else {
            uint32_t dn_idx = static_cast<uint32_t>(dn_layer_map_[il]);
            // One token per slot: pass slots vector to DeltaNet decode path.
            DeltaNetLayer::DecodeArgs da{slots};
            DeltaNetLayer::PrefillArgs pa_unused{1, 0};

            const auto& sm = dn_hp;
            DeltaNetLayer dn_layer(
                blk.attn_qkv_weight,
                blk.attn_gate_weight,
                blk.ssm_beta_weight,
                blk.ssm_alpha_weight,
                blk.ssm_dt_bias,
                blk.ssm_a,
                blk.ssm_conv1d_weight,
                blk.ssm_norm_weight,
                blk.ssm_out_weight,
                dn_state_.get(),
                DeltaNetLayer::Hparams{
                    static_cast<int>(m.embedding_length),
                    static_cast<int>(sm.head_v_dim * sm.num_v_heads),
                    static_cast<int>(sm.head_k_dim),
                    static_cast<int>(num_k_heads),
                    static_cast<int>(sm.num_v_heads),
                    static_cast<int>(sm.head_v_dim),
                    static_cast<int>(sm.conv_channels),
                    static_cast<int>(sm.conv_kernel),
                    meta_.rms_norm_eps
                });
            cur = dn_layer.build(ctx_, gf, cur, dn_idx,
                                 Phase::Decode, pa_unused, &da);
        }

        // Residual 1
        cur = ggml_add(ctx_, cur, inpSA);

        // Pre-FFN norm + MoE
        ggml_tensor* ffn_inp = cur;
        cur = build_norm(gf, cur, blk.ffn_norm_weight, il);
        cur = build_moe_layer(ctx_, gf, cur, blk, moe_hp_, il);

        // Residual 2
        cur = ggml_add(ctx_, cur, ffn_inp);
        inpL = cur;
    }

    build_output_head(gf, inpL);
    return gf;
}

// ── set_inputs ────────────────────────────────────────────────────────────────

void Qwen36ForwardPass::set_inputs(ggml_cgraph* gf,
                                   const std::vector<int32_t>& tokens,
                                   int pos)
{
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());

    // Tokens
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) throw std::runtime_error("qwen36: 'tokens' tensor missing from graph");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_tok * sizeof(int32_t));

    // Position IDs
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) throw std::runtime_error("qwen36: 'inp_pos' tensor missing from graph");
    std::vector<int32_t> pos_data(n_tok);
    for (uint32_t i = 0; i < n_tok; ++i) pos_data[i] = pos + static_cast<int>(i);
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, n_tok * sizeof(int32_t));

    // Causal masks — only for attention layers.
    // build_attention() names each layer's mask "kq_mask.{physical_il}".
    for (uint32_t il = 0; il < meta_.block_count; ++il) {
        if (!meta_.is_full_attention_layer(il)) continue;

        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, name);
        if (!kq_mask) continue;  // mask may not exist if kv_cache was empty

        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        std::vector<float> mask(n_kv * n_tok);
        for (uint32_t t = 0; t < n_tok; ++t) {
            const uint32_t q_pos = static_cast<uint32_t>(pos) + t;
            for (uint32_t j = 0; j < n_kv; ++j)
                mask[t * n_kv + j] = (j <= q_pos) ? 0.0f : -INFINITY;
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
    }
}

// ── set_batched_inputs ────────────────────────────────────────────────────────

void Qwen36ForwardPass::set_batched_inputs(
    ggml_cgraph* gf,
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>&  positions)
{
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());

    // Tokens
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) throw std::runtime_error("qwen36: 'tokens' tensor missing from graph");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_batch * sizeof(int32_t));

    // Position IDs (one per batch slot)
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) throw std::runtime_error("qwen36: 'inp_pos' tensor missing from graph");
    ggml_backend_tensor_set(pos_t, positions.data(), 0, n_batch * sizeof(int32_t));

    // Shared KV mask [n_kv_len, 1, 1, n_batch]
    ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, "kq_mask_b");
    if (kq_mask) {
        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        std::vector<float> mask(n_kv * n_batch, -INFINITY);
        for (uint32_t b = 0; b < n_batch; ++b) {
            const uint32_t q_pos = static_cast<uint32_t>(positions[b]);
            for (uint32_t j = 0; j <= q_pos && j < n_kv; ++j)
                mask[b * n_kv + j] = 0.0f;
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
    }

    // Gather indices [n_batch * n_kv_len]
    ggml_tensor* gi = ggml_graph_get_tensor(gf, "gather_indices");
    if (gi) {
        const uint32_t n_kv   = static_cast<uint32_t>(gi->ne[0]) / n_batch;
        std::vector<int32_t> idx(n_batch * n_kv);
        for (uint32_t b = 0; b < n_batch; ++b) {
            const uint32_t slot = slots[b];
            for (uint32_t j = 0; j < n_kv; ++j)
                idx[b * n_kv + j] = static_cast<int32_t>(slot * n_kv + j);
        }
        ggml_backend_tensor_set(gi, idx.data(), 0, idx.size() * sizeof(int32_t));
    }
}
