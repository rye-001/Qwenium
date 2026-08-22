#include "qwen36.h"

#include "../layers/attention.h"
#include "../layers/deltanet.h"
#include "../layers/ffn.h"
#include "../qinf_error.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/gather_indices_input.h"
#include "../graph_inputs/kv_write_indices_input.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <memory>

// ── Qwen35MoEConfig::from_metadata ───────────────────────────────────────────

Qwen35MoEConfig Qwen35MoEConfig::from_metadata(const ModelMetadata& meta) {
    const uint32_t ssm_state_size          = meta.raw_kv.get_uint32("qwen35moe.ssm.state_size");
    const uint32_t ssm_inner_size          = meta.raw_kv.get_uint32("qwen35moe.ssm.inner_size");
    const uint32_t ssm_time_step_rank      = meta.raw_kv.get_uint32("qwen35moe.ssm.time_step_rank");
    const uint32_t ssm_group_count         = meta.raw_kv.get_uint32("qwen35moe.ssm.group_count");
    const uint32_t ssm_conv_kernel         = meta.raw_kv.get_uint32("qwen35moe.ssm.conv_kernel");
    const uint32_t expert_count            = meta.raw_kv.get_uint32("qwen35moe.expert_count");
    const uint32_t expert_used_count       = meta.raw_kv.get_uint32("qwen35moe.expert_used_count");
    const uint32_t expert_feed_forward_length = meta.raw_kv.get_uint32("qwen35moe.expert_feed_forward_length");
    const uint32_t rope_dimension_count    = meta.raw_kv.get_uint32("qwen35moe.rope.dimension_count");
    const uint32_t full_attention_interval = meta.raw_kv.get_uint32("qwen35moe.full_attention_interval");
    // Optional: absent on standard GGUFs ⇒ 0 ⇒ no MTP head, n_main == block_count.
    const uint32_t nextn_predict_layers    = meta.nextn_predict_layers();

    QINF_ASSERT(ssm_state_size > 0,
        "Qwen35MoEConfig: field \"ssm_state_size\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.ssm.state_size)");
    QINF_ASSERT(ssm_inner_size > 0,
        "Qwen35MoEConfig: field \"ssm_inner_size\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.ssm.inner_size)");
    QINF_ASSERT(ssm_time_step_rank > 0,
        "Qwen35MoEConfig: field \"ssm_time_step_rank\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.ssm.time_step_rank)");
    QINF_ASSERT(ssm_group_count > 0,
        "Qwen35MoEConfig: field \"ssm_group_count\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.ssm.group_count)");
    QINF_ASSERT(ssm_conv_kernel > 0,
        "Qwen35MoEConfig: field \"ssm_conv_kernel\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.ssm.conv_kernel)");
    QINF_ASSERT(expert_count > 0,
        "Qwen35MoEConfig: field \"expert_count\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.expert_count)");
    QINF_ASSERT(expert_feed_forward_length > 0,
        "Qwen35MoEConfig: field \"expert_feed_forward_length\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.expert_feed_forward_length)");
    QINF_ASSERT(expert_used_count <= expert_count,
        "Qwen35MoEConfig: field \"expert_used_count\" expected <= expert_count (" +
        std::to_string(expert_count) + "), got " +
        std::to_string(expert_used_count));
    QINF_ASSERT(rope_dimension_count > 0,
        "Qwen35MoEConfig: field \"rope_dimension_count\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.rope.dimension_count)");
    QINF_ASSERT(full_attention_interval > 0,
        "Qwen35MoEConfig: field \"full_attention_interval\" expected > 0, got 0 "
        "(GGUF key: qwen35moe.full_attention_interval)");
    QINF_ASSERT(nextn_predict_layers < meta.block_count,
        "Qwen35MoEConfig: field \"nextn_predict_layers\" expected < block_count (" +
        std::to_string(meta.block_count) + "), got " +
        std::to_string(nextn_predict_layers) +
        " (GGUF key: qwen35moe.nextn_predict_layers)");

    return Qwen35MoEConfig{
        ssm_conv_kernel,
        ssm_state_size,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_inner_size,
        expert_count,
        expert_used_count,
        expert_feed_forward_length,
        rope_dimension_count,
        full_attention_interval,
        nextn_predict_layers,
    };
}

// ── Constructor ──────────────────────────────────────────────────────────────

Qwen36ForwardPass::Qwen36ForwardPass(
    const Model&    model,
    const ModelMetadata* metadata,
    uint32_t             context_len,
    uint32_t             max_batch_size,
    ggml_type            kv_type)
    : ForwardPassBase(model, metadata)
{
    const auto& m = *metadata;

    // Construct the typed config; validates qwen35moe-specific invariants.
    cfg_ = Qwen35MoEConfig::from_metadata(m);

    // Cache attention hparams — these never change per token.
    n_embd_head_ = static_cast<int>(m.attention_key_length);   // 256
    n_head_       = static_cast<int>(m.attention_head_count);   // 16
    n_head_kv_    = static_cast<int>(m.attention_head_count_kv);// 2
    n_rot_        = static_cast<int>(cfg_.rope_dimension_count); // 64 (partial)

    // MoE hparams — same for every layer.
    moe_hp_ = MoELayer::Hparams{
        static_cast<int>(cfg_.expert_count),             // 256
        static_cast<int>(cfg_.expert_used_count),        // 8
        static_cast<int>(cfg_.expert_feed_forward_length), // 512
        true                                             // has_shared_expert
    };

    // Main decode stack excludes the trailing NextN/MTP head block(s), which
    // are bound (model.cpp) but not executed here (docs/plan-mtp-decode.md §4).
    n_main_layers_ = m.block_count - cfg_.nextn_predict_layers;

    // Build physical→logical layer index maps over the main stack only.
    kv_layer_map_.assign(n_main_layers_, -1);
    dn_layer_map_.assign(n_main_layers_, -1);
    int kv_idx = 0, dn_idx = 0;
    for (uint32_t il = 0; il < n_main_layers_; ++il) {
        if (cfg_.is_full_attention_layer(il))
            kv_layer_map_[il] = kv_idx++;
        else
            dn_layer_map_[il] = dn_idx++;
    }
    const int n_kv_layers = kv_idx;  // 10
    const int n_dn_layers = dn_idx;  // 30

    std::cout << "[qwen36] Hybrid cache: " << n_kv_layers
              << " attention layers (KV), " << n_dn_layers
              << " DeltaNet layers (recurrent state)" << std::endl;

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
        kv_type, kv_type,
        cache_backend);

    // DeltaNet state — 30 DeltaNet layers, backend-backed.
    const uint32_t d_inner       = cfg_.ssm_inner_size;     // 4096
    const uint32_t num_v_heads   = cfg_.ssm_time_step_rank; // 32
    const uint32_t num_k_heads   = cfg_.ssm_group_count;    // 16
    const uint32_t head_v_dim    = d_inner / num_v_heads; // 128
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * cfg_.ssm_state_size; // 8192

    DeltaNetState::Hparams dn_state_hp{
        static_cast<uint32_t>(n_dn_layers),
        max_batch_size,
        head_v_dim,
        cfg_.ssm_state_size,  // head_k_dim = 128
        num_v_heads,
        conv_channels,
        cfg_.ssm_conv_kernel, // 4
        cache_backend
    };
    dn_state_ = std::make_unique<DeltaNetState>(dn_state_hp);

    // NextN/MTP head's private KV: 1 layer, 1 slot, draft-window context only
    // (reset per mtp_draft call — §4.4). ~128 KB; allocated only when the GGUF
    // carries the head.
    if (cfg_.has_mtp_head()) {
        mtp_kv_ = std::make_unique<simple_kv_cache>(
            /*n_layers=*/1, /*n_ctx_max=*/32, /*n_batch_max=*/1,
            n_embd_k, n_embd_v,
            kv_type, kv_type,
            cache_backend);
        std::cout << "[qwen36] MTP/NextN head present (block "
                  << n_main_layers_ << ") — draft capability on" << std::endl;
    }
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
    int pos, uint32_t slot_idx, bool want_logits)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const auto& m        = meta_;
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());

    // Derive DeltaNet state hparams from the typed config for the helper.
    const uint32_t d_inner       = cfg_.ssm_inner_size;
    const uint32_t num_v_heads   = cfg_.ssm_time_step_rank;
    const uint32_t num_k_heads   = cfg_.ssm_group_count;
    const uint32_t head_v_dim    = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * cfg_.ssm_state_size;
    DeltaNetState::Hparams dn_hp{
        0, 0,             // n_dn_layers / n_slots not used in helper
        head_v_dim,
        cfg_.ssm_state_size,
        num_v_heads,
        conv_channels,
        cfg_.ssm_conv_kernel,
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

    // Typed inputs (replaces set_inputs). One uniform causal mask per
    // attention layer; DeltaNet layers have none. build_output_head appends
    // SparseHeadInput on the sparse path.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    for (uint32_t il = 0; il < n_main_layers_; ++il)
        if (cfg_.is_full_attention_layer(il))
            graph_inputs_.add(std::make_unique<AttnMaskInput>(
                "kq_mask." + std::to_string(il), 0u));

    // 3. Transformer loop (main stack only; NextN head held out — §4)
    for (uint32_t il = 0; il < n_main_layers_; ++il) {
        const auto& blk = model_.get_block(il);
        ggml_tensor* inpSA = inpL;

        // ── Pre-attention norm ──────────────────────────────────────────────
        ggml_tensor* cur = build_norm(gf, inpL, blk.attn_norm_weight, il);

        // ── Attention or DeltaNet ───────────────────────────────────────────
        if (cfg_.is_full_attention_layer(il)) {
            int kv_idx = kv_layer_map_[il];
            // Gated attention: joint Q+Gate projection, Q weight outputs
            // [(n_embd_head*2)*n_head, n_tokens]. build_gated_attention
            // handles the strided view split, sigmoid gating, and out-proj.
            cur = build_gated_attention(
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

    // 4. Final norm + LM head.
    // THE single per-recipe head-presence guard site (docs/plan-feed-tokens.md
    // → Head-presence locality constraint: exactly one site, identical in
    // shape across recipes — not scattered want_logits conditionals). When
    // want_logits=false (feed_tokens), the head is pruned: state-write roots
    // — KV cpy_k/v and DeltaNet conv + recurrent ggml_cpy — are independently
    // ggml_build_forward_expand'd inside the layer builders, so the head-less
    // graph still advances both state types. KV-append vs recurrent-overwrite
    // stays at the cache-object level; the recurrence kernel is not forked.
    // The else-branch anchors the residual tip as a graph output (the pruned
    // logits node used to be the scheduler's backend-propagation anchor;
    // without it ggml_gallocr aborts on buffer_id < 0). Numerically inert.
    // D3: expose the pre-final-norm hidden (all positions) so the MTP head can
    // condition on it. Marking an existing node as output adds no compute ⇒
    // off-path is byte-identical; the verify pass needs all K positions, which
    // is exactly inpL before the head slice. Named before the head builds.
    if (output_hidden_) {
        set_tensor_name(gf, inpL, "hidden_out");
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    if (want_logits) {
        build_output_head(gf, build_out_ids_slice(gf, inpL));
    } else {
        ggml_build_forward_expand(gf, inpL);
        ggml_set_output(inpL);
    }

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

    // Derive DeltaNet state hparams from the typed config for the helper.
    const uint32_t d_inner       = cfg_.ssm_inner_size;
    const uint32_t num_v_heads   = cfg_.ssm_time_step_rank;
    const uint32_t num_k_heads   = cfg_.ssm_group_count;
    const uint32_t head_v_dim    = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * cfg_.ssm_state_size;
    DeltaNetState::Hparams dn_hp{
        0, 0, head_v_dim, cfg_.ssm_state_size,
        num_v_heads, conv_channels, cfg_.ssm_conv_kernel, nullptr
    };

    // 1. Token embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // 2. Position tensor (one per batch element)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_batch);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. KV gather mask — shared across all attention layers. Width is
    // bucketed (padded tail −inf-masked, zero-init rows) so one decode graph
    // shape survives a whole bucket of steps — plan-persistent-decode-graph.md.
    uint32_t max_pos = 0;
    for (uint32_t s : slots) {
        uint32_t p = get_cache_pos(s);
        if (p > max_pos) max_pos = p;
    }
    const uint32_t n_kv_len =
        decode_kv_len(max_pos + 1, kv_cache_->get_n_ctx_max());

    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32,
                                               n_kv_len, 1, 1, n_batch);
    ggml_set_input(kq_mask);
    ggml_set_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    ggml_tensor* gather_indices = ggml_new_tensor_1d(
        ctx_, GGML_TYPE_I32, static_cast<int64_t>(n_batch * n_kv_len));
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // KV write rows as input VALUES (persistent-graph write path); Cpy mode
    // is the byte-gate reference and builds no such tensor.
    ggml_tensor* kv_write_idx = nullptr;
    if (kv_write_mode_ == KvWriteMode::SetRows) {
        kv_write_idx = ggml_new_tensor_1d(ctx_, GGML_TYPE_I64, n_batch);
        ggml_set_input(kv_write_idx);
        ggml_set_name(kv_write_idx, KvWriteIndicesInput::slot_);
    }

    // Typed inputs (replaces set_batched_inputs). qwen36 KV gather uses an
    // n_kv_len per-slot stride (not n_ctx_max).
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    graph_inputs_.add(std::make_unique<AttnMaskInput>("kq_mask_b", 0u));
    graph_inputs_.add(std::make_unique<GatherIndicesInput>(
        GatherIndicesInput::Stride::NKvLen));
    if (kv_write_idx)
        graph_inputs_.add(std::make_unique<KvWriteIndicesInput>(
            kv_cache_->get_n_ctx_max()));

    // 4. Transformer loop (main stack only; NextN head held out — §4)
    for (uint32_t il = 0; il < n_main_layers_; ++il) {
        const auto& blk = model_.get_block(il);
        ggml_tensor* inpSA = inpL;

        // Pre-attention norm
        ggml_tensor* cur = build_norm(gf, inpL, blk.attn_norm_weight, il);

        if (cfg_.is_full_attention_layer(il)) {
            int kv_idx = kv_layer_map_[il];
            cur = build_gated_batched_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos,
                kq_mask, gather_indices,
                kv_idx, slots, positions, il,
                blk.attn_q_weight, blk.attn_q_norm_weight,
                blk.attn_k_weight, blk.attn_k_norm_weight,
                blk.attn_v_weight, blk.attn_output_weight,
                n_embd_head_, n_head_, n_head_kv_,
                n_rot_, m.rope_freq_base,
                static_cast<int>(m.context_length),
                m.rms_norm_eps,
                kv_write_idx);
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

    // D3: expose the pre-final-norm hidden (all active slots) on the decode
    // graph too (Phase-3 "prefill + batched decode" scope). Off ⇒ byte-identical.
    if (output_hidden_) {
        set_tensor_name(gf, inpL, "hidden_out");
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    build_output_head(gf, inpL);
    return gf;
}

// ── MTP / NextN head (docs/plan-mtp-decode.md §4, Phase 3) ───────────────────
// One draft step. Mirrors llama.cpp qwen35moe graph_mtp node-for-node:
//   concat( enorm(embed(tok)), hnorm(h) ) → eh_proj → gated-attn(private KV)
//   → +residual → post_attention_norm → MoE(+shared expert) → +residual
//   → shared_head_norm → { "mtp_h_next" (chained hidden), "mtp_logits" }.
// enorm/hnorm are plain RMS·w per the reference (NOT the Gemma (1+w) form —
// Phase 0's inference from the weight means was corrected by the reference
// impl; the differential fixture is the arbiter if this ever mismatches).
// Inputs are bespoke named tensors filled manually by mtp_draft — this graph
// deliberately does not use the typed GraphInputSet machinery, because it is
// private to the recipe and never flows through run_prefill/decode_step.
ggml_cgraph* Qwen36ForwardPass::build_mtp_graph(uint32_t n_past)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const auto&    m  = meta_;
    const uint32_t il = n_main_layers_;           // 40 — the NextN block
    const auto&   blk = model_.get_block(il);

    // Inputs: the token being extended (1), and the hidden it rides on.
    ggml_tensor* t_tok = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, 1);
    ggml_set_input(t_tok);
    set_tensor_name(gf, t_tok, "mtp_tokens");
    ggml_build_forward_expand(gf, t_tok);

    ggml_tensor* t_h = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, m.embedding_length, 1);
    ggml_set_input(t_h);
    set_tensor_name(gf, t_h, "mtp_h");
    ggml_build_forward_expand(gf, t_h);

    ggml_tensor* t_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, 1);
    ggml_set_input(t_pos);
    set_tensor_name(gf, t_pos, "mtp_pos");
    ggml_build_forward_expand(gf, t_pos);

    // enorm(embed) ‖ hnorm(hidden) → eh_proj. Concat order [embed; hidden]
    // matches the reference ggml_concat(e_norm, h_norm, 0) — reversing it is
    // the classic silent bug (§4.3).
    ggml_tensor* e  = ggml_get_rows(ctx_, model_.get_token_embedding_weight(), t_tok);
    e               = build_norm(gf, e, blk.nextn_enorm, il);
    ggml_tensor* hn = build_norm(gf, t_h, blk.nextn_hnorm, il);
    ggml_tensor* cur = ggml_concat(ctx_, e, hn, 0);            // [2*n_embd, 1]
    cur = ggml_mul_mat(ctx_, blk.nextn_eh_proj, cur);          // [n_embd, 1]
    set_tensor_name(gf, cur, "mtp_eh_proj");

    // The NextN block proper — same gated attention + MoE as a main attention
    // layer, on the private single-slot KV (layer 0, slot 0).
    ggml_tensor* inpSA = cur;
    cur = build_norm(gf, cur, blk.attn_norm_weight, il);
    cur = build_gated_attention(
        ctx_, gf, mtp_kv_.get(), cur, t_pos,
        /*kv_cache_layer=*/0, /*n_tokens=*/1, /*slot_idx=*/0, static_cast<int>(il),
        blk.attn_q_weight, blk.attn_q_norm_weight,
        blk.attn_k_weight, blk.attn_k_norm_weight,
        blk.attn_v_weight, blk.attn_output_weight,
        n_embd_head_, n_head_, n_head_kv_,
        n_rot_, m.rope_freq_base,
        static_cast<int>(m.context_length),
        m.rms_norm_eps);
    cur = ggml_add(ctx_, cur, inpSA);

    ggml_tensor* ffn_inp = cur;
    cur = build_norm(gf, cur, blk.ffn_norm_weight, il);
    cur = build_moe_layer(ctx_, gf, cur, blk, moe_hp_, static_cast<int>(il));
    cur = ggml_add(ctx_, cur, ffn_inp);

    // shared_head_norm → chained hidden out; then the SHARED output head.
    cur = build_norm(gf, cur, blk.nextn_shared_head_norm, -1);
    set_tensor_name(gf, cur, "mtp_h_next");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_tensor* head_w = model_.get_output_weight()
        ? model_.get_output_weight()
        : model_.get_token_embedding_weight();   // tied-embeddings fallback (reference does the same)
    ggml_tensor* logits = ggml_mul_mat(ctx_, head_w, cur);     // [n_vocab, 1]
    set_tensor_name(gf, logits, "mtp_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    (void)n_past;  // shape is n_past-independent; the KV views read get_pos()
    return gf;
}

std::vector<int32_t> Qwen36ForwardPass::mtp_draft(
    uint32_t                  slot,
    const std::vector<float>& hidden,
    int32_t                   last_token,
    int                       pos,
    uint32_t                  k,
    ggml_backend_sched_t      sched)
{
    (void)slot;  // private KV is single-slot: reset per call, so slots never mix
    if (!mtp_supported())
        throw std::runtime_error(
            "mtp_draft: mtp_supported expected=true, actual=false — "
            "this GGUF carries no NextN head (qwen35moe.nextn_predict_layers=0)");
    if (hidden.size() != meta_.embedding_length)
        throw std::runtime_error(
            "mtp_draft: hidden size expected=" +
            std::to_string(meta_.embedding_length) + ", actual=" +
            std::to_string(hidden.size()));
    if (k == 0) return {};
    if (k >= 32)
        throw std::runtime_error(
            "mtp_draft: k expected < 32 (private head KV window), actual=" +
            std::to_string(k));

    const int n_vocab = static_cast<int>(meta_.vocab_size);
    const std::string mask_name = "kq_mask." + std::to_string(n_main_layers_);

    // Stateless across decode steps (§4.6): every draft attempt starts from an
    // empty head KV and re-seeds from (hidden, last_token).
    mtp_kv_->set_pos(0, 0);

    std::vector<int32_t> drafted;
    std::vector<float>   h = hidden;
    std::vector<float>   logits(n_vocab);
    int32_t              tok = last_token;

    for (uint32_t i = 0; i < k; ++i) {
        ggml_backend_sched_reset(sched);
        ggml_cgraph* gf = build_mtp_graph(/*n_past=*/i);
        ggml_backend_sched_alloc_graph(sched, gf);

        ggml_tensor* t_tok  = ggml_graph_get_tensor(gf, "mtp_tokens");
        ggml_tensor* t_h    = ggml_graph_get_tensor(gf, "mtp_h");
        ggml_tensor* t_pos  = ggml_graph_get_tensor(gf, "mtp_pos");
        ggml_tensor* t_mask = ggml_graph_get_tensor(gf, mask_name.c_str());
        if (!t_tok || !t_h || !t_pos || !t_mask)
            throw std::runtime_error(
                "mtp_draft: graph input expected present, actual missing: " +
                std::string(!t_tok ? "mtp_tokens" : !t_h ? "mtp_h"
                          : !t_pos ? "mtp_pos" : mask_name));

        const int32_t p = pos + static_cast<int32_t>(i);
        ggml_backend_tensor_set(t_tok, &tok, 0, sizeof(int32_t));
        ggml_backend_tensor_set(t_h,   h.data(), 0, ggml_nbytes(t_h));
        ggml_backend_tensor_set(t_pos, &p,   0, sizeof(int32_t));
        // Single query token attending to all i+1 KV entries: no masking.
        const std::vector<float> mask_zeros(i + 1, 0.0f);
        ggml_backend_tensor_set(t_mask, mask_zeros.data(), 0,
                                mask_zeros.size() * sizeof(float));

        ggml_backend_sched_graph_compute(sched, gf);

        ggml_tensor* t_logits = ggml_graph_get_tensor(gf, "mtp_logits");
        ggml_tensor* t_hn     = ggml_graph_get_tensor(gf, "mtp_h_next");
        if (!t_logits || !t_hn)
            throw std::runtime_error(
                "mtp_draft: graph output expected present, actual missing: " +
                std::string(!t_logits ? "mtp_logits" : "mtp_h_next"));
        ggml_backend_tensor_get(t_logits, logits.data(), 0,
                                n_vocab * sizeof(float));
        ggml_backend_tensor_get(t_hn, h.data(), 0, h.size() * sizeof(float));

        mtp_kv_->advance(1, 0);

        // Greedy: draft tokens are model-verified downstream (§5 D4).
        int best = 0;
        for (int j = 1; j < n_vocab; ++j)
            if (logits[j] > logits[best]) best = j;
        drafted.push_back(best);
        tok = best;
    }

    return drafted;
}

// set_inputs / set_batched_inputs removed: inputs are now populated by the
// typed GraphInputSet (graph_inputs_) built in build_prefill_graph and
// build_decoding_graph. See docs/plan-typed-graph-inputs.md.

// ── Inventory validator ──────────────────────────────────────────────────────

void validate_qwen36_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                "qwen35moe: missing tensor '" + name +
                "': expected in model weights, got absent");
    };

    require("token_embd.weight");
    require("output_norm.weight");

    static const std::vector<std::string> moe_tensors = {
        "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"
    };
    static const std::vector<std::string> attn_tensors = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"
    };
    static const std::vector<std::string> dn_tensors = {
        "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias",
        "ssm_alpha.weight", "ssm_beta.weight",
        "attn_qkv.weight", "attn_gate.weight",
        "ssm_norm.weight", "ssm_out.weight"
    };

    // NextN / MTP head: the four tensors that turn the trailing block(s) into a
    // multi-token-prediction head (docs/plan-mtp-decode.md §4). Optional group:
    //   nextn_predict_layers == 0            → no head, capability off (fine).
    //   nextn_predict_layers  > 0, all found → capability on.
    //   nextn_predict_layers  > 0, any missing → fail-loud naming it (below).
    static const std::vector<std::string> nextn_tensors = {
        "nextn.eh_proj.weight", "nextn.enorm.weight",
        "nextn.hnorm.weight",   "nextn.shared_head_norm.weight"
    };

    const uint32_t fai   = meta.raw_kv.get_uint32("qwen35moe.full_attention_interval");
    const uint32_t nextn = meta.nextn_predict_layers();
    if (nextn >= meta.block_count)
        throw std::runtime_error(
            "qwen35moe: field 'nextn_predict_layers' expected < block_count (" +
            std::to_string(meta.block_count) + "), got " + std::to_string(nextn));
    const uint32_t n_main = meta.block_count - nextn;

    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const bool is_nextn = (i >= n_main);
        require(p + "attn_norm.weight");
        require(p + "post_attention_norm.weight");
        for (const auto& t : moe_tensors) require(p + t);
        // Inline arithmetic: validator runs at load time, before Qwen35MoEConfig is constructed.
        // NextN blocks are attention-typed regardless of position.
        const bool is_full = is_nextn || ((fai > 0) && ((i % fai) == (fai - 1)));
        if (is_full)
            for (const auto& t : attn_tensors) require(p + t);
        else
            for (const auto& t : dn_tensors)   require(p + t);
        if (is_nextn)
            for (const auto& t : nextn_tensors) require(p + t);
    }
}
