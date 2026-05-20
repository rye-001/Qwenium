#include "qwen35.h"
#include "../layers/attention.h"
#include "../layers/deltanet.h"
#include "../layers/ffn.h"
#include "../state/snapkv.h"
#include "../qinf_error.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/gather_indices_input.h"

#include <memory>

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>
#include <future>
#include <thread>

// ── Qwen35Config::from_metadata ──────────────────────────────────────────────

Qwen35Config Qwen35Config::from_metadata(const ModelMetadata& meta) {
    const uint32_t ssm_state_size          = meta.raw_kv.get_uint32("qwen35.ssm.state_size");
    const uint32_t ssm_inner_size          = meta.raw_kv.get_uint32("qwen35.ssm.inner_size");
    const uint32_t ssm_time_step_rank      = meta.raw_kv.get_uint32("qwen35.ssm.time_step_rank");
    const uint32_t ssm_group_count         = meta.raw_kv.get_uint32("qwen35.ssm.group_count");
    const uint32_t ssm_conv_kernel         = meta.raw_kv.get_uint32("qwen35.ssm.conv_kernel");
    const uint32_t full_attention_interval = meta.raw_kv.get_uint32("qwen35.full_attention_interval");
    const uint32_t rope_dimension_count    = meta.raw_kv.get_uint32_opt("qwen35.rope.dimension_count").value_or(0u);

    QINF_ASSERT(ssm_state_size > 0,
        "Qwen35Config: field \"ssm_state_size\" expected > 0, got 0 "
        "(GGUF key: qwen35.ssm.state_size)");
    QINF_ASSERT(ssm_inner_size > 0,
        "Qwen35Config: field \"ssm_inner_size\" expected > 0, got 0 "
        "(GGUF key: qwen35.ssm.inner_size)");
    QINF_ASSERT(ssm_time_step_rank > 0,
        "Qwen35Config: field \"ssm_time_step_rank\" expected > 0, got 0 "
        "(GGUF key: qwen35.ssm.time_step_rank)");
    QINF_ASSERT(ssm_group_count > 0,
        "Qwen35Config: field \"ssm_group_count\" expected > 0, got 0 "
        "(GGUF key: qwen35.ssm.group_count)");
    QINF_ASSERT(ssm_conv_kernel > 0,
        "Qwen35Config: field \"ssm_conv_kernel\" expected > 0, got 0 "
        "(GGUF key: qwen35.ssm.conv_kernel)");
    QINF_ASSERT(full_attention_interval > 0,
        "Qwen35Config: field \"full_attention_interval\" expected > 0, got 0 "
        "(GGUF key: qwen35.full_attention_interval)");

    return Qwen35Config{
        ssm_conv_kernel,
        ssm_state_size,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_inner_size,
        rope_dimension_count,
        full_attention_interval,
    };
}

// Layers per fused graph_compute call (same value as forward-pass.cpp).
// Each attention layer has its own persistent scratch slot for incremental decompression.
static constexpr uint32_t TQ_LAYER_BATCH = 4;

// ============================================================
// Constructor: hybrid cache setup
// ============================================================

Qwen35ForwardPass::Qwen35ForwardPass(
    const Model& model, const ModelMetadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, int kv_quant_bits)
    : ForwardPassBase(model, metadata)
{
    // Construct the typed config; validates qwen35-specific invariants.
    cfg_ = Qwen35Config::from_metadata(*metadata);

    ggml_backend_t cache_backend = model_.has_metal_backend()
        ? model_.get_backend_metal()
        : model_.get_backend_cpu();

    // Count layers and build index mappings
    uint32_t n_attn_layers = 0;
    uint32_t n_ssm_layers = 0;

    kv_layer_map_.resize(meta_.block_count, -1);
    ssm_layer_map_.resize(meta_.block_count, -1);

    for (uint32_t il = 0; il < meta_.block_count; ++il) {
        if (cfg_.is_full_attention_layer(il)) {
            kv_layer_map_[il] = static_cast<int32_t>(n_attn_layers++);
        } else {
            ssm_layer_map_[il] = static_cast<int32_t>(n_ssm_layers++);
        }
    }

    std::cout << "[qwen35] Hybrid cache: " << n_attn_layers
              << " attention layers (KV), " << n_ssm_layers
              << " SSM layers (recurrent state)" << std::endl;

    // KV cache — attention layers only.
    // TQ enabled: skip full F32 KV cache; only the 1-layer scratch is needed.
    uint32_t n_embd_k = meta_.attention_key_length * meta_.attention_head_count_kv;
    uint32_t n_embd_v = meta_.attention_value_length * meta_.attention_head_count_kv;

    if (kv_quant_bits < 2) {
        kv_cache_ = std::make_unique<simple_kv_cache>(
            n_attn_layers, context_len, max_batch_size,
            n_embd_k, n_embd_v,
            GGML_TYPE_F32, GGML_TYPE_F32,
            cache_backend
        );
    }

    // DeltaNet recurrent state — GatedDeltaNet layers only
    // conv_channels = d_inner + 2 * n_group * d_state = 2048 + 2*16*128 = 6144
    const uint32_t d_inner_dn     = cfg_.ssm_inner_size;
    const uint32_t num_v_heads_dn = cfg_.ssm_time_step_rank;
    const uint32_t num_k_heads_dn = cfg_.ssm_group_count;
    const uint32_t head_v_dim_dn  = d_inner_dn / num_v_heads_dn;
    const uint32_t conv_channels_dn =
        d_inner_dn + 2 * num_k_heads_dn * cfg_.ssm_state_size;

    DeltaNetState::Hparams dn_hp{
        n_ssm_layers,
        max_batch_size,
        head_v_dim_dn,
        cfg_.ssm_state_size,  // head_k_dim
        num_v_heads_dn,
        conv_channels_dn,
        cfg_.ssm_conv_kernel,
        cache_backend
    };
    dn_state_ = std::make_unique<DeltaNetState>(dn_hp);

    // TurboQuant compressed backing store + persistent scratch cache
    if (kv_quant_bits >= 2 && kv_quant_bits <= 4) {
        uint32_t head_dim = meta_.attention_key_length;
        tq_store_ = std::make_unique<CompressedKVStore>(
            n_attn_layers,
            max_batch_size,
            context_len,
            n_embd_k, n_embd_v,
            head_dim, kv_quant_bits);

        // Persistent scratch: one layer per attention layer for incremental decompress.
        tq_scratch_cache_ = std::make_unique<simple_kv_cache>(
            n_attn_layers, context_len, max_batch_size,
            n_embd_k, n_embd_v,
            GGML_TYPE_F32, GGML_TYPE_F32,
            cache_backend);

        // Watermarks: [n_attn_layers][n_slots], all zero initially.
        tq_scratch_valid_pos_.assign(n_attn_layers,
            std::vector<uint32_t>(max_batch_size, 0));

        size_t full_kv_mb = static_cast<size_t>(n_attn_layers) * max_batch_size
            * context_len * (n_embd_k + n_embd_v) * 4 / (1024 * 1024);
        size_t tq_mb = tq_store_->total_compressed_bytes() / (1024 * 1024);
        float scratch_mb = static_cast<float>(n_attn_layers) * max_batch_size
            * context_len * (n_embd_k + n_embd_v) * 4 / (1024.0f * 1024.0f);
        printf("TurboQuant KV compression: %d-bit (incremental decompress)\n", kv_quant_bits);
        printf("  F32 KV (without TQ): %zu MB\n", full_kv_mb);
        printf("  Compressed store:    %zu MB\n", tq_mb);
        printf("  Scratch (%u layers): %.1f MB\n", n_attn_layers, scratch_mb);
        printf("  Compression ratio:   %.1fx\n",
               static_cast<float>(full_kv_mb) / tq_mb);
    }
}

// ============================================================
// build_prefill_graph — hybrid layer loop
// ============================================================

struct ggml_cgraph* Qwen35ForwardPass::build_prefill_graph(
    const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx,
    bool want_logits)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const uint32_t n_layers = meta_.block_count;
    const size_t n_tokens   = tokens.size();

    // Typed inputs for this graph (replaces set_inputs). Cleared here so
    // build_output_head can append SparseHeadInput when the sparse path is
    // armed; per-attention-layer masks are added in the layer loop.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());

    // 1. Token embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // Position tensor (for attention layers with RoPE)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    ggml_tensor* cur;

    // 2. Layer loop
    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor* inpSA = inpL;
        auto& block = model_.get_block(il);

        // Pre-attention norm (shared by both layer types)
        cur = build_norm(gf, inpL, block.attn_norm_weight, il);
        set_tensor_name(gf, cur, "attn_norm", il);

        if (cfg_.is_ssm_layer(il)) {
            uint32_t dn_idx = static_cast<uint32_t>(ssm_layer_map_[il]);
            cur = build_deltanet_layer(
                ctx_, gf, cur, dn_state_.get(), dn_idx, slot_idx, n_tokens,
                block.attn_qkv_weight, block.attn_gate_weight,
                block.ssm_beta_weight, block.ssm_alpha_weight,
                block.ssm_dt_bias, block.ssm_a, block.ssm_conv1d_weight,
                block.ssm_norm_weight, block.ssm_out_weight,
                static_cast<int>(meta_.embedding_length),
                static_cast<int>(cfg_.ssm_inner_size),
                static_cast<int>(cfg_.ssm_state_size),
                static_cast<int>(cfg_.ssm_group_count),
                static_cast<int>(cfg_.ssm_time_step_rank),
                static_cast<int>(cfg_.ssm_inner_size / cfg_.ssm_time_step_rank),
                static_cast<int>(cfg_.ssm_inner_size + 2 * cfg_.ssm_group_count * cfg_.ssm_state_size),
                static_cast<int>(cfg_.ssm_conv_kernel),
                meta_.rms_norm_eps,
                il);
        } else {
            // TQ: use 1-layer scratch (layer 0) for all attention layers.
            // kv_idx maps to scratch layer 0 always; physical separation is
            // handled by decompress-before / compress-after in run_prefill().
            simple_kv_cache* attn_cache = tq_scratch_cache_ ? tq_scratch_cache_.get()
                                                             : kv_cache_.get();
            int32_t kv_idx = tq_scratch_cache_ ? 0 : kv_layer_map_[il];
            const int n_embd_head = meta_.attention_key_length;
            const int n_rot = (cfg_.rope_dimension_count > 0)
                ? cfg_.rope_dimension_count : n_embd_head;
            cur = ::build_gated_attention(
                ctx_, gf, attn_cache, cur, inp_pos, kv_idx, n_tokens, slot_idx, il,
                block.attn_q_weight, block.attn_q_norm_weight,
                block.attn_k_weight, block.attn_k_norm_weight,
                block.attn_v_weight, block.attn_output_weight,
                n_embd_head,
                meta_.attention_head_count,
                meta_.attention_head_count_kv,
                n_rot,
                meta_.rope_freq_base,
                static_cast<int>(meta_.context_length),
                meta_.rms_norm_eps);

            // build_gated_attention names this layer's mask "kq_mask.{il}".
            // Qwen3.5 uses one uniform causal mask (no sliding window).
            graph_inputs_.add(std::make_unique<AttnMaskInput>(
                "kq_mask." + std::to_string(il), 0u));
        }

        // Residual connection after attention/SSM
        cur = ggml_add(ctx_, cur, inpSA);
        set_tensor_name(gf, cur, "attn_residual", il);

        // Save for FFN residual
        ggml_tensor* ffn_residual = cur;

        // Post-attention norm
        cur = build_norm(gf, cur, block.ffn_norm_weight, il);
        set_tensor_name(gf, cur, "post_attn_norm", il);

        // FFN
        cur = build_ffn_swiglu(ctx_, gf, cur, block.ffn_gate_weight, block.ffn_up_weight,
                         block.ffn_down_weight, il);
        set_tensor_name(gf, cur, "ffn_out", il);

        // FFN residual
        cur = ggml_add(ctx_, cur, ffn_residual);
        set_tensor_name(gf, cur, "post_ffn", il);

        inpL = cur;
    }

    // 3. Output head.
    // THE single per-recipe head-presence guard site for qwen35
    // (docs/plan-feed-tokens.md → Head-presence locality constraint: exactly
    // one site, identical in shape across recipes — not scattered want_logits
    // conditionals). want_logits=false (feed_tokens) prunes the head; KV
    // cpy_k/v and DeltaNet conv + recurrent ggml_cpy state-write roots are
    // independently ggml_build_forward_expand'd in the layer builders, so the
    // head-less graph still advances both state types. The else-branch
    // anchors the residual tip as a graph output: the scheduler propagates
    // backend assignment from outputs, and the pruned logits node used to be
    // that anchor (without it ggml_gallocr aborts on buffer_id < 0). This
    // anchor is numerically inert — it forces compute, not different state.
    // The TQ run_prefill path builds its head in a separate gf_out graph and
    // is unreachable from feed_tokens (scoped out, fails loud via
    // tq_active()) — so this remains exactly one reachable guard site.
    if (want_logits) {
        build_output_head(gf, build_out_ids_slice(gf, inpL));
    } else {
        ggml_build_forward_expand(gf, inpL);
        ggml_set_output(inpL);
    }
    return gf;
}

// set_inputs / set_batched_inputs removed: inputs are now populated by the
// typed GraphInputSet (graph_inputs_) built in build_prefill_graph and
// build_decoding_graph. See docs/plan-typed-graph-inputs.md.

// ============================================================
// TurboQuant Phase 2: per-layer compute (Qwen35 hybrid)
// ============================================================

// Build a standalone graph for one physical layer (SSM or attention).
// Input tensor "inpL": F32 [hidden_dim, n_tokens] — hidden state from prev layer.
// Output tensor "layer_out": F32 [hidden_dim, n_tokens].
ggml_cgraph* Qwen35ForwardPass::_build_qwen35_layer_graph(
    uint32_t il, uint32_t n_tokens, uint32_t slot_idx, uint32_t scratch_layer)
{
    reset_context();
    ggml_cgraph* gf = new_graph();
    auto& block = model_.get_block(il);

    ggml_tensor* inpL = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32,
        meta_.embedding_length, n_tokens);
    ggml_set_input(inpL);
    ggml_set_name(inpL, "inpL");
    ggml_build_forward_expand(gf, inpL);

    ggml_tensor* inpSA = inpL;
    ggml_tensor* cur = build_norm(gf, inpL, block.attn_norm_weight, il);

    if (cfg_.is_ssm_layer(il)) {
        uint32_t dn_idx = static_cast<uint32_t>(ssm_layer_map_[il]);
        cur = build_deltanet_layer(
            ctx_, gf, cur, dn_state_.get(), dn_idx, slot_idx, n_tokens,
            block.attn_qkv_weight, block.attn_gate_weight,
            block.ssm_beta_weight, block.ssm_alpha_weight,
            block.ssm_dt_bias, block.ssm_a, block.ssm_conv1d_weight,
            block.ssm_norm_weight, block.ssm_out_weight,
            static_cast<int>(meta_.embedding_length),
            static_cast<int>(cfg_.ssm_inner_size),
            static_cast<int>(cfg_.ssm_state_size),
            static_cast<int>(cfg_.ssm_group_count),
            static_cast<int>(cfg_.ssm_time_step_rank),
            static_cast<int>(cfg_.ssm_inner_size / cfg_.ssm_time_step_rank),
            static_cast<int>(cfg_.ssm_inner_size + 2 * cfg_.ssm_group_count * cfg_.ssm_state_size),
            static_cast<int>(cfg_.ssm_conv_kernel),
            meta_.rms_norm_eps,
            il);
    } else {
        ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp_pos);
        ggml_set_name(inp_pos, "inp_pos");
        ggml_build_forward_expand(gf, inp_pos);

        {
            const int n_embd_head = meta_.attention_key_length;
            const int n_rot = (cfg_.rope_dimension_count > 0)
                ? cfg_.rope_dimension_count : n_embd_head;
            cur = ::build_gated_attention(
                ctx_, gf, tq_scratch_cache_.get(), cur, inp_pos,
                scratch_layer, n_tokens, slot_idx, il,
                block.attn_q_weight, block.attn_q_norm_weight,
                block.attn_k_weight, block.attn_k_norm_weight,
                block.attn_v_weight, block.attn_output_weight,
                n_embd_head,
                meta_.attention_head_count,
                meta_.attention_head_count_kv,
                n_rot,
                meta_.rope_freq_base,
                static_cast<int>(meta_.context_length),
                meta_.rms_norm_eps);
        }
    }

    cur = ggml_add(ctx_, cur, inpSA);
    ggml_tensor* ffn_residual = cur;
    cur = build_norm(gf, cur, block.ffn_norm_weight, il);
    cur = build_ffn_swiglu(ctx_, gf, cur, block.ffn_gate_weight, block.ffn_up_weight,
                     block.ffn_down_weight, il);
    cur = ggml_add(ctx_, cur, ffn_residual);

    ggml_set_name(cur, "layer_out");
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Build a fused graph for layers [il_start, il_end) in one ggml context.
// SSM and attention layers are interleaved exactly as in the real architecture.
// Each attention layer uses its persistent scratch slot (kv_layer_map_[il]).
ggml_cgraph* Qwen35ForwardPass::_build_qwen35_batch_graph(
    uint32_t il_start, uint32_t il_end, uint32_t n_tokens, uint32_t slot_idx)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    ggml_tensor* inpL = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32,
        meta_.embedding_length, n_tokens);
    ggml_set_input(inpL);
    ggml_set_name(inpL, "inpL");
    ggml_build_forward_expand(gf, inpL);

    // inp_pos is shared across all attention layers in the batch
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    ggml_tensor* cur = inpL;

    for (uint32_t il = il_start; il < il_end; ++il) {
        auto& block     = model_.get_block(il);
        ggml_tensor* inpSA = cur;

        cur = build_norm(gf, cur, block.attn_norm_weight, il);

        if (cfg_.is_ssm_layer(il)) {
            uint32_t dn_idx = static_cast<uint32_t>(ssm_layer_map_[il]);
            cur = build_deltanet_layer(
                ctx_, gf, cur, dn_state_.get(), dn_idx, slot_idx, n_tokens,
                block.attn_qkv_weight, block.attn_gate_weight,
                block.ssm_beta_weight, block.ssm_alpha_weight,
                block.ssm_dt_bias, block.ssm_a, block.ssm_conv1d_weight,
                block.ssm_norm_weight, block.ssm_out_weight,
                static_cast<int>(meta_.embedding_length),
                static_cast<int>(cfg_.ssm_inner_size),
                static_cast<int>(cfg_.ssm_state_size),
                static_cast<int>(cfg_.ssm_group_count),
                static_cast<int>(cfg_.ssm_time_step_rank),
                static_cast<int>(cfg_.ssm_inner_size / cfg_.ssm_time_step_rank),
                static_cast<int>(cfg_.ssm_inner_size + 2 * cfg_.ssm_group_count * cfg_.ssm_state_size),
                static_cast<int>(cfg_.ssm_conv_kernel),
                meta_.rms_norm_eps,
                il);
        } else {
            // Each attention layer uses its persistent scratch slot (kv_layer_map_[il])
            {
                const int n_embd_head = meta_.attention_key_length;
                const int n_rot = (cfg_.rope_dimension_count > 0)
                    ? cfg_.rope_dimension_count : n_embd_head;
                cur = ::build_gated_attention(
                    ctx_, gf, tq_scratch_cache_.get(), cur, inp_pos,
                    kv_layer_map_[il], n_tokens, slot_idx, il,
                    block.attn_q_weight, block.attn_q_norm_weight,
                    block.attn_k_weight, block.attn_k_norm_weight,
                    block.attn_v_weight, block.attn_output_weight,
                    n_embd_head,
                    meta_.attention_head_count,
                    meta_.attention_head_count_kv,
                    n_rot,
                    meta_.rope_freq_base,
                    static_cast<int>(meta_.context_length),
                    meta_.rms_norm_eps);
            }
        }

        cur = ggml_add(ctx_, cur, inpSA);
        ggml_tensor* ffn_residual = cur;
        cur = build_norm(gf, cur, block.ffn_norm_weight, il);
        cur = build_ffn_swiglu(ctx_, gf, cur, block.ffn_gate_weight, block.ffn_up_weight,
                         block.ffn_down_weight, il);
        cur = ggml_add(ctx_, cur, ffn_residual);
    }

    ggml_set_name(cur, "layer_out");
    ggml_build_forward_expand(gf, cur);
    return gf;
}

void Qwen35ForwardPass::_tq_decompress_attn_layer(uint32_t kv_idx, uint32_t slot_idx,
                                                   uint32_t scratch_layer) {
    const uint32_t pos = tq_store_->get_pos(slot_idx);
    if (pos == 0) return;

    // Incremental: only decompress tokens [watermark..pos).
    const uint32_t watermark = tq_scratch_valid_pos_[kv_idx][slot_idx];
    if (watermark >= pos) return;  // scratch is already up-to-date
    const uint32_t n_delta = pos - watermark;

    const uint32_t n_embd_k = meta_.attention_key_length * meta_.attention_head_count_kv;
    const uint32_t n_embd_v = meta_.attention_value_length * meta_.attention_head_count_kv;

    ggml_tensor* k_tensor = tq_scratch_cache_->get_k_cache_tensor(scratch_layer);
    ggml_tensor* v_tensor = tq_scratch_cache_->get_v_cache_tensor(scratch_layer);

    std::vector<float> k_buf(static_cast<size_t>(n_delta) * n_embd_k);
    std::vector<float> v_buf(static_cast<size_t>(n_delta) * n_embd_v);

    // Each token decompresses into a non-overlapping slice — safe to parallelise.
    static constexpr uint32_t MIN_PAR_TOKENS = 4;
    const uint32_t hw_threads =
        std::max(1u, static_cast<uint32_t>(std::thread::hardware_concurrency()));
    const uint32_t n_threads = (n_delta >= MIN_PAR_TOKENS)
        ? std::min(n_delta, hw_threads) : 1u;

    auto decompress_range = [&](uint32_t t0, uint32_t t1) {
        for (uint32_t t = t0; t < t1; ++t) {
            const uint32_t abs_pos = watermark + t;
            tq_store_->decompress_token_k(kv_idx, slot_idx, abs_pos,
                k_buf.data() + t * n_embd_k);
            tq_store_->decompress_token_v(kv_idx, slot_idx, abs_pos,
                v_buf.data() + t * n_embd_v);
        }
    };

    if (n_threads == 1) {
        decompress_range(0, n_delta);
    } else {
        const uint32_t chunk = (n_delta + n_threads - 1) / n_threads;
        std::vector<std::future<void>> futures;
        futures.reserve(n_threads);
        for (uint32_t tid = 0; tid < n_threads; ++tid) {
            const uint32_t t0 = tid * chunk;
            const uint32_t t1 = std::min(t0 + chunk, n_delta);
            if (t0 >= t1) break;
            futures.push_back(std::async(std::launch::async, decompress_range, t0, t1));
        }
        for (auto& f : futures) f.get();
    }

    // Upload only the delta range into the Metal buffer at the correct offset.
    const size_t k_offset = slot_idx * k_tensor->nb[2]
                          + static_cast<size_t>(watermark) * n_embd_k * sizeof(float);
    const size_t v_offset = slot_idx * v_tensor->nb[2]
                          + static_cast<size_t>(watermark) * n_embd_v * sizeof(float);

    ggml_backend_tensor_set(k_tensor, k_buf.data(), k_offset,
        static_cast<size_t>(n_delta) * n_embd_k * sizeof(float));
    ggml_backend_tensor_set(v_tensor, v_buf.data(), v_offset,
        static_cast<size_t>(n_delta) * n_embd_v * sizeof(float));

    tq_scratch_valid_pos_[kv_idx][slot_idx] = pos;
}

void Qwen35ForwardPass::_tq_compress_attn_layer(
    uint32_t kv_idx, uint32_t slot_idx, uint32_t pos, uint32_t n_tokens,
    uint32_t scratch_layer)
{
    const uint32_t n_embd_k = meta_.attention_key_length * meta_.attention_head_count_kv;
    const uint32_t n_embd_v = meta_.attention_value_length * meta_.attention_head_count_kv;

    ggml_tensor* k_tensor = tq_scratch_cache_->get_k_cache_tensor(scratch_layer);
    ggml_tensor* v_tensor = tq_scratch_cache_->get_v_cache_tensor(scratch_layer);

    // Bulk read: one ggml_backend_tensor_get per K/V instead of one per token.
    // Tokens pos..pos+n_tokens-1 are contiguous within a slot (nb[1] == n_embd*sizeof(float)).
    std::vector<float> k_buf(static_cast<size_t>(n_tokens) * n_embd_k);
    std::vector<float> v_buf(static_cast<size_t>(n_tokens) * n_embd_v);

    ggml_backend_tensor_get(k_tensor, k_buf.data(),
        slot_idx * k_tensor->nb[2] + pos * k_tensor->nb[1],
        static_cast<size_t>(n_tokens) * n_embd_k * sizeof(float));
    ggml_backend_tensor_get(v_tensor, v_buf.data(),
        slot_idx * v_tensor->nb[2] + pos * v_tensor->nb[1],
        static_cast<size_t>(n_tokens) * n_embd_v * sizeof(float));

    // compress_token_k/v copies to an internal scratch before calling turboquant::compress,
    // so each call only reads its own slice of k_buf/v_buf and writes to a distinct
    // offset in the compressed store — safe to parallelise across tokens.
    static constexpr uint32_t MIN_PAR_TOKENS = 4;
    const uint32_t hw_threads =
        std::max(1u, static_cast<uint32_t>(std::thread::hardware_concurrency()));
    const uint32_t n_threads = (n_tokens >= MIN_PAR_TOKENS)
        ? std::min(n_tokens, hw_threads) : 1u;

    auto compress_range = [&](uint32_t t0, uint32_t t1) {
        for (uint32_t t = t0; t < t1; ++t) {
            tq_store_->compress_token_k(kv_idx, slot_idx, pos + t,
                k_buf.data() + t * n_embd_k);
            tq_store_->compress_token_v(kv_idx, slot_idx, pos + t,
                v_buf.data() + t * n_embd_v);
        }
    };

    if (n_threads == 1) {
        compress_range(0, n_tokens);
    } else {
        const uint32_t chunk = (n_tokens + n_threads - 1) / n_threads;
        std::vector<std::future<void>> futures;
        futures.reserve(n_threads);
        for (uint32_t tid = 0; tid < n_threads; ++tid) {
            const uint32_t t0 = tid * chunk;
            const uint32_t t1 = std::min(t0 + chunk, n_tokens);
            if (t0 >= t1) break;
            futures.push_back(std::async(std::launch::async, compress_range, t0, t1));
        }
        for (auto& f : futures) f.get();
    }

    // Graph compute wrote new KV into the scratch; it's now valid up to pos + n_tokens.
    tq_scratch_valid_pos_[kv_idx][slot_idx] = pos + n_tokens;
}

// Find the last attention layer (for SnapKV scoring)
static uint32_t _find_last_attn_layer(const ModelMetadata& meta) {
    const uint32_t fai = meta.raw_kv.get_uint32("qwen35.full_attention_interval");
    for (int32_t il = meta.block_count - 1; il >= 0; --il) {
        const bool is_full = (fai > 0) && ((il % fai) == (fai - 1));
        if (is_full) return il;
    }
    return meta.block_count - 1;  // fallback
}

std::vector<float> Qwen35ForwardPass::run_prefill(
    const std::vector<int32_t>& tokens,
    int pos, uint32_t slot_idx,
    ggml_backend_sched_t scheduler)
{
    const bool do_snapkv = snapkv_budget_ > 0 && tokens.size() > snapkv_window_;

    // ── Non-TQ path (monolithic graph) ──────────────────────────────────
    if (!tq_store_) {
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);

        const uint32_t scoring_layer = _find_last_attn_layer(meta_);
        if (do_snapkv) {
            char name[32];
            snprintf(name, sizeof(name), "kq_soft.%d", scoring_layer);
            ggml_tensor* kq = ggml_graph_get_tensor(gf, name);
            if (kq) ggml_set_output(kq);
        }

        ggml_backend_sched_alloc_graph(scheduler, gf);
        set_prefill_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);

        advance_cache(tokens.size(), slot_idx);
        auto logits = get_output_logits(gf);

        if (do_snapkv) {
            // n_kv_layers for compaction (only 6 attention layers have KV)
            uint32_t n_kv_layers = 0;
            for (uint32_t il = 0; il < meta_.block_count; ++il)
                if (cfg_.is_full_attention_layer(il)) ++n_kv_layers;
            uint32_t original_seq_len = pos + tokens.size();
            apply_snapkv_from_graph(gf, scoring_layer, tokens.size(),
                n_kv_layers, snapkv_budget_, snapkv_window_,
                slot_idx, kv_cache_.get(), nullptr, nullptr);
            snapkv_set_seq_pos(slot_idx, original_seq_len);
        }

        return logits;
    }

    // ── TQ path ─────────────────────────────────────────────────────────
    const uint32_t n_layers  = meta_.block_count;
    const uint32_t n_tokens  = tokens.size();
    const int      hidden_dim = meta_.embedding_length;

    std::vector<float> hidden(static_cast<size_t>(hidden_dim) * n_tokens);

    // Compute embeddings via ggml (handles quantized weights).
    {
        reset_context();
        ggml_cgraph* gf_emb = new_graph();
        ggml_tensor* emb = embedding(gf_emb, tokens);
        ggml_build_forward_expand(gf_emb, emb);

        ggml_backend_sched_reset(scheduler);
        ggml_backend_sched_alloc_graph(scheduler, gf_emb);
        {
            StepContext step;
            step.gf = gf_emb;
            step.tokens = &tokens;
            TokensInput().set_input(step);
        }
        ggml_backend_sched_graph_compute(scheduler, gf_emb);

        ggml_tensor* emb_out = ggml_graph_get_tensor(gf_emb, "embed_lookup");
        ggml_backend_tensor_get(emb_out, hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));
    }

    tq_scratch_cache_->set_pos(tq_store_->get_pos(slot_idx), slot_idx);

    // ── Batched layer compute ────────────────────────────────────────────
    // TQ_LAYER_BATCH layers share one graph_compute call.
    // Each attention layer uses its persistent scratch slot (kv_layer_map_[il]).
    bool tq_advanced = false;
    for (uint32_t il0 = 0; il0 < n_layers; il0 += TQ_LAYER_BATCH) {
        const uint32_t il1 = std::min(il0 + TQ_LAYER_BATCH, n_layers);

        // 1. Decompress KV delta for attention layers into their persistent scratch slots
        for (uint32_t il = il0; il < il1; ++il) {
            if (!cfg_.is_ssm_layer(il)) {
                const uint32_t kv_idx = static_cast<uint32_t>(kv_layer_map_[il]);
                _tq_decompress_attn_layer(kv_idx, slot_idx, kv_idx);
            }
        }

        // 2. Build fused graph for [il0, il1)
        ggml_cgraph* gf = _build_qwen35_batch_graph(il0, il1, n_tokens, slot_idx);

        // SnapKV: if this batch contains the scoring layer, mark kq_soft as output
        const uint32_t scoring_layer = _find_last_attn_layer(meta_);
        const bool batch_has_scoring = do_snapkv && (scoring_layer >= il0 && scoring_layer < il1);
        if (batch_has_scoring) {
            char sname[32];
            snprintf(sname, sizeof(sname), "kq_soft.%d", scoring_layer);
            ggml_tensor* kq = ggml_graph_get_tensor(gf, sname);
            if (kq) ggml_set_output(kq);
        }

        // 3. Allocate and set inputs
        ggml_backend_sched_reset(scheduler);
        ggml_backend_sched_alloc_graph(scheduler, gf);

        // inpL is a hidden-state carrier (precomputed embeddings / prev-batch
        // output), not a typed input — stays a direct set (plan scope fence).
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inpL"),
            hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));

        // Positions + per-attention-layer causal masks: same typed inputs as
        // the non-TQ path. Collapses the 4th duplicated poke copy.
        {
            StepContext step;
            step.gf = gf;
            step.tokens = &tokens;
            step.pos = pos;
            PositionsInput().set_input(step);
            for (uint32_t il = il0; il < il1; ++il) {
                if (cfg_.is_ssm_layer(il)) continue;
                AttnMaskInput("kq_mask." + std::to_string(il), 0u)
                    .set_input(step);
            }
        }

        // 4. One compute call for all layers in batch
        ggml_backend_sched_graph_compute(scheduler, gf);

        // 5. Read back hidden state once per batch
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer_out"),
            hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));

        // 6. Compress attention layers from their persistent scratch slots
        for (uint32_t il = il0; il < il1; ++il) {
            if (!cfg_.is_ssm_layer(il)) {
                const uint32_t kv_idx = static_cast<uint32_t>(kv_layer_map_[il]);
                _tq_compress_attn_layer(kv_idx, slot_idx,
                    tq_store_->get_pos(slot_idx), n_tokens, kv_idx);
            }
        }

        // 7. SnapKV: evict from compressed store using this batch's kq_soft
        if (batch_has_scoring) {
            tq_store_->advance(slot_idx, n_tokens);
            uint32_t original_seq_len = pos + n_tokens;

            uint32_t n_kv_layers = 0;
            for (uint32_t il = 0; il < meta_.block_count; ++il)
                if (cfg_.is_full_attention_layer(il)) ++n_kv_layers;

            apply_snapkv_from_graph(gf, scoring_layer, n_tokens,
                n_kv_layers, snapkv_budget_, snapkv_window_,
                slot_idx, nullptr, tq_store_.get(),
                [this](uint32_t s) { _tq_invalidate_watermarks(s); });

            tq_scratch_cache_->set_pos(tq_store_->get_pos(slot_idx), slot_idx);
            snapkv_set_seq_pos(slot_idx, original_seq_len);
            tq_advanced = true;
        }
    }

    if (!tq_advanced)
        tq_store_->advance(slot_idx, n_tokens);

    // ── Output head ──────────────────────────────────────────────────────
    reset_context();
    ggml_cgraph* gf_out = new_graph();
    ggml_tensor* final_in = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hidden_dim, n_tokens);
    ggml_set_input(final_in);
    ggml_set_name(final_in, "final_in");
    ggml_build_forward_expand(gf_out, final_in);
    build_output_head(gf_out, final_in);

    ggml_backend_sched_reset(scheduler);
    ggml_backend_sched_alloc_graph(scheduler, gf_out);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_out, "final_in"), hidden.data(), 0,
        static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));
    ggml_backend_sched_graph_compute(scheduler, gf_out);

    return get_output_logits(gf_out);
}

// ============================================================
// build_decoding_graph — multi-slot single-token decode
// ============================================================

ggml_cgraph* Qwen35ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions)
{
    reset_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);

    const uint32_t n_layers = meta_.block_count;
    const size_t n_batch    = tokens.size();
    const int64_t n_embd    = meta_.embedding_length;

    // 1. Token embedding (batched)
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // 2. Position tensor
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_batch);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. Attention mask + gather indices (used by attention layers only)
    // Use physical cache positions for KV sizing (after SnapKV, logical pos > physical pos).
    uint32_t max_physical = 0;
    for (uint32_t s : slots) {
        uint32_t phys = get_physical_cache_pos(s);
        if (phys > max_physical) max_physical = phys;
    }
    uint32_t n_kv_len = max_physical + 1;

    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32,
        n_kv_len, 1, 1, n_batch);
    ggml_set_input(kq_mask);
    ggml_set_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    uint32_t n_total_indices = n_batch * n_kv_len;
    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_total_indices);
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // Typed inputs for the decode graph (replaces set_batched_inputs).
    // Single shared causal mask + KV gather; build_output_head appends
    // SparseHeadInput when the grammar sparse path is armed.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    graph_inputs_.add(std::make_unique<AttnMaskInput>("kq_mask_b", 0u));
    graph_inputs_.add(std::make_unique<GatherIndicesInput>(
        kv_cache_->get_n_ctx_max()));

    // 4. Layer loop
    ggml_tensor* cur;

    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor* inpSA = inpL;
        auto& block = model_.get_block(il);

        // Pre-attention/SSM norm
        cur = build_norm(gf, inpL, block.attn_norm_weight, il);

        if (cfg_.is_ssm_layer(il)) {
            uint32_t dn_idx = static_cast<uint32_t>(ssm_layer_map_[il]);
            DeltaNetLayer::PrefillArgs pa_unused{1, 0};
            DeltaNetLayer::DecodeArgs da{slots};
            DeltaNetLayer dn_layer(
                block.attn_qkv_weight, block.attn_gate_weight,
                block.ssm_beta_weight, block.ssm_alpha_weight,
                block.ssm_dt_bias, block.ssm_a, block.ssm_conv1d_weight,
                block.ssm_norm_weight, block.ssm_out_weight,
                dn_state_.get(),
                DeltaNetLayer::Hparams{
                    static_cast<int>(meta_.embedding_length),
                    static_cast<int>(cfg_.ssm_inner_size),
                    static_cast<int>(cfg_.ssm_state_size),
                    static_cast<int>(cfg_.ssm_group_count),
                    static_cast<int>(cfg_.ssm_time_step_rank),
                    static_cast<int>(cfg_.ssm_inner_size / cfg_.ssm_time_step_rank),
                    static_cast<int>(cfg_.ssm_inner_size + 2 * cfg_.ssm_group_count * cfg_.ssm_state_size),
                    static_cast<int>(cfg_.ssm_conv_kernel),
                    meta_.rms_norm_eps
                });
            cur = dn_layer.build(ctx_, gf, cur, dn_idx, Phase::Decode, pa_unused, &da);
        } else {
            {
                int32_t kv_idx = kv_layer_map_[il];
                const int n_embd_head = meta_.attention_key_length;
                const int n_rot = (cfg_.rope_dimension_count > 0)
                    ? cfg_.rope_dimension_count : n_embd_head;
                cur = ::build_gated_batched_attention(
                    ctx_, gf, kv_cache_.get(), cur, inp_pos,
                    kq_mask, gather_indices, kv_idx, slots, positions, il,
                    block.attn_q_weight, block.attn_q_norm_weight,
                    block.attn_k_weight, block.attn_k_norm_weight,
                    block.attn_v_weight, block.attn_output_weight,
                    n_embd_head,
                    meta_.attention_head_count,
                    meta_.attention_head_count_kv,
                    n_rot,
                    meta_.rope_freq_base,
                    static_cast<int>(meta_.context_length),
                    meta_.rms_norm_eps);
            }
        }

        // Residual after attention/SSM
        cur = ggml_add(ctx_, cur, inpSA);

        ggml_tensor* ffn_residual = cur;

        // Post-attention norm + FFN + residual
        cur = build_norm(gf, cur, block.ffn_norm_weight, il);
        cur = build_ffn_swiglu(ctx_, gf, cur, block.ffn_gate_weight,
                         block.ffn_up_weight, block.ffn_down_weight, il);
        cur = ggml_add(ctx_, cur, ffn_residual);

        inpL = cur;
    }

    // 5. Output head — shared helper (final norm + LM head). Routes through
    //    build_output_head so the sparse decode path (sparse_decode_ids_ →
    //    ggml_get_rows on the output weight) is honored, matching the prefill
    //    path (line ~265) and Qwen3.6. Dense behavior is unchanged: with no
    //    sparse ids armed, this is build_norm + ggml_mul_mat over the full
    //    output weight (or token-embedding fallback), identical to the prior
    //    hand-rolled code.
    build_output_head(gf, inpL);

    return gf;
}


// ── Inventory validator ──────────────────────────────────────────────────────

void validate_qwen35_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name, const std::string& ctx) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                "qwen35: missing tensor '" + name +
                "': expected in " + ctx + ", got absent");
    };
    require("token_embd.weight", "model weights");
    require("output_norm.weight", "model weights");

    static const std::vector<std::string> shared = {
        "attn_norm.weight", "post_attention_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
    };
    static const std::vector<std::string> attn_tensors = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"
    };
    static const std::vector<std::string> ssm_tensors = {
        "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias",
        "ssm_alpha.weight", "ssm_beta.weight",
        "attn_qkv.weight", "attn_gate.weight",
        "ssm_norm.weight", "ssm_out.weight"
    };
    const uint32_t fai = meta.raw_kv.get_uint32("qwen35.full_attention_interval");
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : shared) require(p + t, "block " + std::to_string(i));
        const bool is_full = (fai > 0) && ((i % fai) == (fai - 1));
        const auto& chosen = is_full ? attn_tensors : ssm_tensors;
        const std::string kind = is_full ? "attention" : "SSM";
        for (const auto& t : chosen) {
            if (inv.find(p + t) == inv.end())
                throw std::runtime_error(
                    "qwen35: missing tensor '" + p + t +
                    "': expected in " + kind + " layer " + std::to_string(i) +
                    ", got absent");
        }
    }
}