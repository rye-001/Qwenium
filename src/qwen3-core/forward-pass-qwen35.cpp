#include "forward-pass-qwen35.h"
#include "../kv-cache/snapkv-eviction.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>
#include <future>
#include <thread>

// Layers per fused graph_compute call (same value as forward-pass.cpp).
// Each attention layer has its own persistent scratch slot for incremental decompression.
static constexpr uint32_t TQ_LAYER_BATCH = 4;

// ============================================================
// Constructor: hybrid cache setup
// ============================================================

Qwen35ForwardPass::Qwen35ForwardPass(
    const Qwen3Model& model, const Qwen3Metadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, int kv_quant_bits)
    : ForwardPassBase(model, metadata)
{
    ggml_backend_t cache_backend = model_.has_metal_backend()
        ? model_.get_backend_metal()
        : model_.get_backend_cpu();

    // Count layers and build index mappings
    uint32_t n_attn_layers = 0;
    uint32_t n_ssm_layers = 0;

    kv_layer_map_.resize(meta_.block_count, -1);
    ssm_layer_map_.resize(meta_.block_count, -1);

    for (uint32_t il = 0; il < meta_.block_count; ++il) {
        if (meta_.is_full_attention_layer(il)) {
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

    // SSM state cache — GatedDeltaNet layers only
    // conv_channels = d_inner + 2 * n_group * d_state = 2048 + 2*16*128 = 6144
    uint32_t conv_dim = meta_.ssm_inner_size + 2 * meta_.ssm_group_count * meta_.ssm_state_size;

    ssm_cache_ = std::make_unique<ssm_state_cache>(
        n_ssm_layers, max_batch_size,
        meta_.ssm_state_size, meta_.ssm_time_step_rank,
        conv_dim, meta_.ssm_conv_kernel,
        cache_backend
    );

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
    const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const uint32_t n_layers = meta_.block_count;
    const size_t n_tokens   = tokens.size();

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

        if (meta_.is_ssm_layer(il)) {
            int32_t ssm_idx = ssm_layer_map_[il];
            cur = build_ssm_layer(gf, cur, ssm_idx, n_tokens, slot_idx, il);
        } else {
            // TQ: use 1-layer scratch (layer 0) for all attention layers.
            // kv_idx maps to scratch layer 0 always; physical separation is
            // handled by decompress-before / compress-after in run_prefill().
            simple_kv_cache* attn_cache = tq_scratch_cache_ ? tq_scratch_cache_.get()
                                                             : kv_cache_.get();
            int32_t kv_idx = tq_scratch_cache_ ? 0 : kv_layer_map_[il];
            cur = build_attention_layer(gf, attn_cache, cur, inp_pos,
                                         kv_idx, n_tokens, slot_idx, il);
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
        cur = ffn_swiglu(gf, cur, block.ffn_gate_weight, block.ffn_up_weight,
                         block.ffn_down_weight, il);
        set_tensor_name(gf, cur, "ffn_out", il);

        // FFN residual
        cur = ggml_add(ctx_, cur, ffn_residual);
        set_tensor_name(gf, cur, "post_ffn", il);

        inpL = cur;
    }

    // 3. Output head
    build_output_head(gf, inpL);
    return gf;
}

// ============================================================
// build_attention_layer — joint Q+Gate, gated attention, partial RoPE
// ============================================================

ggml_tensor* Qwen35ForwardPass::build_attention_layer(
    ggml_cgraph* gf,
    simple_kv_cache* kv_cache,
    ggml_tensor* cur,
    ggml_tensor* inp_pos,
    int kv_cache_layer,
    uint32_t n_tokens,
    uint32_t slot_idx,
    int il)
{
    const int n_embd_head = meta_.attention_key_length;
    const int n_head      = meta_.attention_head_count;
    const int n_head_kv   = meta_.attention_head_count_kv;
    auto& block = model_.get_block(il);

    const int n_rot = (meta_.rope_dimension_count > 0)
        ? meta_.rope_dimension_count : n_embd_head;

    // A. Joint Q+Gate projection
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx_, block.attn_q_weight, cur);
    set_tensor_name(gf, Qcur_full, "Qcur_full", il);

    // B. Extract Q via strided view
    ggml_tensor* Qcur = ggml_view_3d(ctx_, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    set_tensor_name(gf, Qcur, "Qcur", il);

    Qcur = build_norm(gf, Qcur, block.attn_q_norm_weight, il);
    set_tensor_name(gf, Qcur, "Qcur_normed", il);

    // C. K and V projections
    ggml_tensor* Kcur = ggml_mul_mat(ctx_, block.attn_k_weight, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx_, block.attn_v_weight, cur);

    Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head_kv, n_tokens);

    Kcur = build_norm(gf, Kcur, block.attn_k_norm_weight, il);
    set_tensor_name(gf, Kcur, "Kcur_normed", il);

    // D. Extract Gate
    ggml_tensor* gate = ggml_view_3d(ctx_, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx_, gate, n_embd_head * n_head, n_tokens);
    set_tensor_name(gf, gate, "gate", il);

    // E. RoPE
    const float freq_base = meta_.rope_freq_base;
    Qcur = ggml_rope_ext(ctx_, Qcur, inp_pos, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX,
                          meta_.context_length, freq_base,
                          1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(ctx_, Kcur, inp_pos, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX,
                          meta_.context_length, freq_base,
                          1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // F. KV cache + attention
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    const uint32_t cache_pos = kv_cache->get_pos(slot_idx);
    const uint32_t n_kv = cache_pos + n_tokens;

    ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx_, Kcur, kv_cache_layer, slot_idx));
    ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx_, Vcur, kv_cache_layer, slot_idx));

    ggml_tensor* k_full = kv_cache->get_k(ctx_, kv_cache_layer, n_kv, slot_idx);
    ggml_tensor* v_full = kv_cache->get_v(ctx_, kv_cache_layer, n_kv, slot_idx);

    const int n_embd_kv = n_head_kv * n_embd_head;
    ggml_tensor* k_view = ggml_view_3d(ctx_, k_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * sizeof(float), n_embd_kv * sizeof(float), 0);
    ggml_tensor* v_view = ggml_view_3d(ctx_, v_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * sizeof(float), n_embd_kv * sizeof(float), 0);

    ggml_tensor* kq_mask = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, n_kv, n_tokens);
    set_tensor_name(gf, kq_mask, "kq_mask", il);
    ggml_build_forward_expand(gf, kq_mask);

    cur = build_attn_mha(gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, cache_pos, il);

    // G. Gate sigmoid
    cur = ggml_mul(ctx_, cur, ggml_sigmoid(ctx_, gate));
    set_tensor_name(gf, cur, "attn_gated", il);

    // H. Output projection
    cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);
    set_tensor_name(gf, cur, "attn_output", il);

    return cur;
}

// ============================================================
// build_ssm_layer — GatedDeltaNet pipeline
//   Projections → Conv1D → QKV split → L2 norm → [delta net STUB] →
//   Gated norm with z → Output projection
// ============================================================

ggml_tensor* Qwen35ForwardPass::build_ssm_layer(
    ggml_cgraph* gf,
    ggml_tensor* cur,
    int ssm_cache_layer,
    uint32_t n_tokens,
    uint32_t slot_idx,
    int il)
{
    auto& block = model_.get_block(il);

    // Dimensions (confirmed via llama.cpp debug output)
    const int64_t d_inner      = meta_.ssm_inner_size;     // 2048
    const int64_t head_k_dim   = meta_.ssm_state_size;     // 128
    const int64_t num_k_heads  = meta_.ssm_group_count;    // 16
    const int64_t num_v_heads  = meta_.ssm_time_step_rank; // 16
    const int64_t head_v_dim   = d_inner / num_v_heads;    // 128
    const int64_t n_embd       = meta_.embedding_length;   // 1024

    const int64_t conv_kernel_size = meta_.ssm_conv_kernel; // 4
    // conv_channels = d_inner + 2 * n_group * d_state = 2048 + 2*16*128 = 6144
    const int64_t conv_channels = d_inner + 2 * num_k_heads * head_k_dim;

    const int64_t n_seqs       = 1;
    const int64_t n_seq_tokens = n_tokens;

    // ========================================================
    // 1. Input projections
    // ========================================================

    // QKV mixed: attn_qkv [1024, 6144] @ cur → [6144, n_tokens]
    ggml_tensor* qkv_mixed = ggml_mul_mat(ctx_, block.attn_qkv_weight, cur);
    qkv_mixed = ggml_reshape_3d(ctx_, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    set_tensor_name(gf, qkv_mixed, "ssm_qkv", il);

    // Z (output gate): attn_gate [1024, 2048] @ cur → [2048, n_tokens]
    ggml_tensor* z = ggml_mul_mat(ctx_, block.attn_gate_weight, cur);
    set_tensor_name(gf, z, "ssm_z", il);

    // ========================================================
    // 2. Gate projections
    // ========================================================

    // Beta: sigmoid(ssm_beta @ cur) → [1, num_v_heads, n_tokens, 1]
    ggml_tensor* beta = ggml_mul_mat(ctx_, block.ssm_beta_weight, cur);
    beta = ggml_reshape_4d(ctx_, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx_, beta);
    set_tensor_name(gf, beta, "ssm_beta", il);

    // Alpha → decay gate: softplus(alpha + dt_bias) * A_log
    ggml_tensor* alpha = ggml_mul_mat(ctx_, block.ssm_alpha_weight, cur);
    alpha = ggml_reshape_3d(ctx_, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* alpha_biased = ggml_add(ctx_, alpha, block.ssm_dt_bias);
    ggml_tensor* alpha_sp = ggml_softplus(ctx_, alpha_biased);
    ggml_tensor* decay_gate = ggml_mul(ctx_, alpha_sp, block.ssm_a);
    decay_gate = ggml_reshape_4d(ctx_, decay_gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    set_tensor_name(gf, decay_gate, "ssm_gate", il);

    // ========================================================
    // 3. Causal Conv1D
    // ========================================================

    // Read conv state from cache for this slot
    ggml_tensor* conv_all = ssm_cache_->get_conv_tensor(ssm_cache_layer);
    const int64_t conv_state_elems = (conv_kernel_size - 1) * conv_channels;

    ggml_tensor* conv_states = ggml_view_1d(ctx_, conv_all,
        conv_state_elems, slot_idx * conv_all->nb[1]);
    conv_states = ggml_reshape_3d(ctx_, conv_states,
        conv_kernel_size - 1, conv_channels, n_seqs);
    set_tensor_name(gf, conv_states, "ssm_conv_st", il);

    ggml_tensor* qkv_t = ggml_transpose(ctx_, qkv_mixed);
    ggml_tensor* conv_input = ggml_concat(ctx_, conv_states, qkv_t, 0);
    set_tensor_name(gf, conv_input, "ssm_conv_in", il);

    // Update conv state cache
    ggml_tensor* last_conv = ggml_view_3d(ctx_, conv_input,
        conv_kernel_size - 1, conv_channels, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        (conv_input->ne[0] - (conv_kernel_size - 1)) * ggml_element_size(conv_input));
    ggml_tensor* conv_dst = ggml_view_1d(ctx_, conv_all,
        conv_state_elems, slot_idx * conv_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx_, last_conv, conv_dst));

    // Depthwise conv1d + SiLU
    ggml_tensor* conv_out = ggml_ssm_conv(ctx_, conv_input, block.ssm_conv1d_weight);
    conv_out = ggml_silu(ctx_, conv_out);
    set_tensor_name(gf, conv_out, "ssm_conv_out", il);

    // ========================================================
    // 4. Split conv output into Q, K, V
    // ========================================================

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor* q_conv = ggml_view_4d(ctx_, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens, 0);

    ggml_tensor* k_conv = ggml_view_4d(ctx_, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        head_k_dim * num_k_heads * ggml_element_size(conv_out));

    ggml_tensor* v_conv = ggml_view_4d(ctx_, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_v_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    // ========================================================
    // 5. L2 normalize Q and K
    // ========================================================

    q_conv = ggml_l2_norm(ctx_, q_conv, meta_.rms_norm_eps);
    k_conv = ggml_l2_norm(ctx_, k_conv, meta_.rms_norm_eps);

    if (num_k_heads != num_v_heads) {
        q_conv = ggml_repeat_4d(ctx_, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx_, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }    

    // ========================================================
    // 6. Delta net recurrence (fused op)
    // ========================================================

    ggml_tensor* rec_all = ssm_cache_->get_recurrent_tensor(ssm_cache_layer);
    const int64_t rec_slot_floats = head_v_dim * head_k_dim * num_v_heads;

    ggml_tensor* S = ggml_view_1d(ctx_, rec_all,
        rec_slot_floats, slot_idx * rec_all->nb[1]);
    S = ggml_reshape_4d(ctx_, S, head_v_dim, head_k_dim, num_v_heads, n_seqs);
    set_tensor_name(gf, S, "ssm_state_in", il);

    ggml_tensor* result = ggml_gated_delta_net(ctx_,
        q_conv, k_conv, v_conv, decay_gate, beta, S);
    set_tensor_name(gf, result, "ssm_gdn_result", il);

    // Output: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    ggml_tensor* output = ggml_view_4d(ctx_, result,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens), 0);
    set_tensor_name(gf, output, "ssm_delta_out", il);

    // New state: [head_v_dim, head_k_dim, num_v_heads, n_seqs]
    ggml_tensor* new_state = ggml_view_4d(ctx_, result,
        head_v_dim, head_k_dim, num_v_heads, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens * n_seqs));

    // Write new state back to cache
    ggml_tensor* rec_dst = ggml_view_1d(ctx_, rec_all,
        rec_slot_floats, slot_idx * rec_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx_,
        ggml_reshape_1d(ctx_, new_state, rec_slot_floats), rec_dst));

    // ========================================================
    // 7. Gated normalization
    // ========================================================

    ggml_tensor* z_4d = ggml_reshape_4d(ctx_, z,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* out_normed = build_norm(gf, output, block.ssm_norm_weight, il);
    ggml_tensor* z_silu = ggml_silu(ctx_, z_4d);
    ggml_tensor* gated = ggml_mul(ctx_, out_normed, z_silu);
    set_tensor_name(gf, gated, "ssm_gated", il);

    // ========================================================
    // 8. Output projection
    // ========================================================

    ggml_tensor* flat = ggml_reshape_3d(ctx_, gated,
        head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cur = ggml_mul_mat(ctx_, block.ssm_out_weight, flat);
    set_tensor_name(gf, cur, "ssm_output", il);
    cur = ggml_reshape_2d(ctx_, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

// ============================================================
// set_inputs
// ============================================================

void Qwen35ForwardPass::set_inputs(
    ggml_cgraph* gf,
    const std::vector<int32_t>& tokens,
    int pos)
{
    const size_t n_tokens = tokens.size();

    ggml_tensor* tokens_tensor = ggml_graph_get_tensor(gf, "tokens");
    if (!tokens_tensor) throw std::runtime_error("tokens tensor not found");
    ggml_backend_tensor_set(tokens_tensor, tokens.data(), 0, n_tokens * sizeof(int32_t));

    ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (!inp_pos) throw std::runtime_error("inp_pos tensor not found");
    std::vector<int32_t> pos_data(n_tokens);
    for (size_t i = 0; i < n_tokens; i++) pos_data[i] = pos + i;
    ggml_backend_tensor_set(inp_pos, pos_data.data(), 0, n_tokens * sizeof(int32_t));

    for (uint32_t il = 0; il < meta_.block_count; ++il) {
        char mask_name[32];
        snprintf(mask_name, sizeof(mask_name), "kq_mask.%d", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, mask_name);
        if (!kq_mask) continue;

        const uint32_t n_kv = kq_mask->ne[0];
        std::vector<float> mask_data(n_kv * n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const uint32_t q_pos = pos + i;
            for (uint32_t j = 0; j < n_kv; ++j) {
                mask_data[i * n_kv + j] = (j <= q_pos) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(kq_mask, mask_data.data(), 0, n_kv * n_tokens * sizeof(float));
    }
}

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

    if (meta_.is_ssm_layer(il)) {
        cur = build_ssm_layer(gf, cur, ssm_layer_map_[il], n_tokens, slot_idx, il);
    } else {
        ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp_pos);
        ggml_set_name(inp_pos, "inp_pos");
        ggml_build_forward_expand(gf, inp_pos);

        cur = build_attention_layer(gf, tq_scratch_cache_.get(), cur, inp_pos,
                                    scratch_layer, n_tokens, slot_idx, il);
    }

    cur = ggml_add(ctx_, cur, inpSA);
    ggml_tensor* ffn_residual = cur;
    cur = build_norm(gf, cur, block.ffn_norm_weight, il);
    cur = ffn_swiglu(gf, cur, block.ffn_gate_weight, block.ffn_up_weight,
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

        if (meta_.is_ssm_layer(il)) {
            cur = build_ssm_layer(gf, cur, ssm_layer_map_[il], n_tokens, slot_idx, il);
        } else {
            // Each attention layer uses its persistent scratch slot (kv_layer_map_[il])
            cur = build_attention_layer(gf, tq_scratch_cache_.get(), cur, inp_pos,
                                        kv_layer_map_[il], n_tokens, slot_idx, il);
        }

        cur = ggml_add(ctx_, cur, inpSA);
        ggml_tensor* ffn_residual = cur;
        cur = build_norm(gf, cur, block.ffn_norm_weight, il);
        cur = ffn_swiglu(gf, cur, block.ffn_gate_weight, block.ffn_up_weight,
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
static uint32_t _find_last_attn_layer(const Qwen3Metadata& meta) {
    for (int32_t il = meta.block_count - 1; il >= 0; --il) {
        if (meta.is_full_attention_layer(il)) return il;
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
        set_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);

        advance_cache(tokens.size(), slot_idx);
        auto logits = get_output_logits(gf);

        if (do_snapkv) {
            // n_kv_layers for compaction (only 6 attention layers have KV)
            uint32_t n_kv_layers = 0;
            for (uint32_t il = 0; il < meta_.block_count; ++il)
                if (meta_.is_full_attention_layer(il)) ++n_kv_layers;
            apply_snapkv_from_graph(gf, scoring_layer, tokens.size(),
                n_kv_layers, snapkv_budget_, snapkv_window_,
                slot_idx, kv_cache_.get(), nullptr, nullptr);
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
        ggml_tensor* tok_tensor = ggml_graph_get_tensor(gf_emb, "tokens");
        ggml_backend_tensor_set(tok_tensor, tokens.data(), 0, n_tokens * sizeof(int32_t));
        ggml_backend_sched_graph_compute(scheduler, gf_emb);

        ggml_tensor* emb_out = ggml_graph_get_tensor(gf_emb, "embed_lookup");
        ggml_backend_tensor_get(emb_out, hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));
    }

    tq_scratch_cache_->set_pos(tq_store_->get_pos(slot_idx), slot_idx);

    // Pre-compute position IDs — same for every batch
    std::vector<int32_t> pos_data(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) pos_data[i] = pos + i;

    // ── Batched layer compute ────────────────────────────────────────────
    // TQ_LAYER_BATCH layers share one graph_compute call.
    // Each attention layer uses its persistent scratch slot (kv_layer_map_[il]).
    bool tq_advanced = false;
    for (uint32_t il0 = 0; il0 < n_layers; il0 += TQ_LAYER_BATCH) {
        const uint32_t il1 = std::min(il0 + TQ_LAYER_BATCH, n_layers);

        // 1. Decompress KV delta for attention layers into their persistent scratch slots
        for (uint32_t il = il0; il < il1; ++il) {
            if (!meta_.is_ssm_layer(il)) {
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

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inpL"),
            hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp_pos"),
            pos_data.data(), 0, n_tokens * sizeof(int32_t));

        // Set causal mask for each attention layer in batch
        for (uint32_t il = il0; il < il1; ++il) {
            if (meta_.is_ssm_layer(il)) continue;
            char mask_name[32];
            snprintf(mask_name, sizeof(mask_name), "kq_mask.%d", il);
            ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, mask_name);
            if (!kq_mask) continue;
            const uint32_t n_kv = kq_mask->ne[0];
            std::vector<float> mask_data(n_kv * n_tokens);
            for (uint32_t i = 0; i < n_tokens; ++i) {
                const uint32_t q_pos = pos + i;
                for (uint32_t j = 0; j < n_kv; ++j)
                    mask_data[i * n_kv + j] = (j <= q_pos) ? 0.0f : -INFINITY;
            }
            ggml_backend_tensor_set(kq_mask, mask_data.data(), 0,
                n_kv * n_tokens * sizeof(float));
        }

        // 4. One compute call for all layers in batch
        ggml_backend_sched_graph_compute(scheduler, gf);

        // 5. Read back hidden state once per batch
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer_out"),
            hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));

        // 6. Compress attention layers from their persistent scratch slots
        for (uint32_t il = il0; il < il1; ++il) {
            if (!meta_.is_ssm_layer(il)) {
                const uint32_t kv_idx = static_cast<uint32_t>(kv_layer_map_[il]);
                _tq_compress_attn_layer(kv_idx, slot_idx,
                    tq_store_->get_pos(slot_idx), n_tokens, kv_idx);
            }
        }

        // 7. SnapKV: evict from compressed store using this batch's kq_soft
        if (batch_has_scoring) {
            tq_store_->advance(slot_idx, n_tokens);

            uint32_t n_kv_layers = 0;
            for (uint32_t il = 0; il < meta_.block_count; ++il)
                if (meta_.is_full_attention_layer(il)) ++n_kv_layers;

            apply_snapkv_from_graph(gf, scoring_layer, n_tokens,
                n_kv_layers, snapkv_budget_, snapkv_window_,
                slot_idx, nullptr, tq_store_.get(),
                [this](uint32_t s) { _tq_invalidate_watermarks(s); });

            tq_scratch_cache_->set_pos(tq_store_->get_pos(slot_idx), slot_idx);
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
// build_delta_net_step — single token recurrence
//
// The 6-step GatedDeltaNet algorithm:
//   1. g = exp(gate)                    // decay ∈ (0,1]
//   2. S = g * S                        // decay old state
//   3. retrieved = S^T @ k              // what state predicts
//   4. delta = beta * (v - retrieved)   // error correction
//   5. S = S + k ⊗ delta               // outer product update
//   6. o = S^T @ q / sqrt(d_k)         // read output
// ============================================================

ggml_tensor* Qwen35ForwardPass::build_delta_net_step(
    ggml_cgraph* gf,
    ggml_tensor* S,        // [d, d, H, 1]
    ggml_tensor* q_t,      // [d, H, 1, 1]
    ggml_tensor* k_t,      // [d, H, 1, 1]
    ggml_tensor* v_t,      // [d, H, 1, 1]
    ggml_tensor* gate_t,   // [1, H, 1, 1]
    ggml_tensor* beta_t,   // [1, H, 1, 1]
    ggml_tensor** S_out,
    int il)
{
    const int64_t d = S->ne[0];  // head_v_dim = head_k_dim = 128
    const int64_t H = S->ne[2];  // num_heads = 16

    // Step 1: g = exp(gate_t)
    // gate_t is already -exp(A_log) * softplus(...), so exp(gate_t) ∈ (0, 1]
    ggml_tensor* g = ggml_exp(ctx_, gate_t);  // [1, H, 1, 1]

    // Step 2: S = g * S  (broadcast g across dims 0 and 1)
    // Reshape g to [1, 1, H, 1] for broadcasting with S [d, d, H, 1]
    ggml_tensor* g_bc = ggml_reshape_4d(ctx_, g, 1, 1, H, 1);
    ggml_tensor* S_decayed = ggml_mul(ctx_, S, g_bc);  // [d, d, H, 1]

    // Step 3: retrieved = S^T @ k per head
    // Reshape k_t to [d, 1, H, 1] for batched matmul
    ggml_tensor* k_col = ggml_reshape_4d(ctx_, k_t, d, 1, H, 1);
    // ggml_mul_mat(a, b) = a^T @ b, so mul_mat(S_decayed, k_col):
    //   S_decayed^T [d, d, H, 1]^T @ k_col [d, 1, H, 1] = [d, 1, H, 1]
    ggml_tensor* retrieved = ggml_mul_mat(ctx_, S_decayed, k_col);  // [d, 1, H, 1]

    // Step 4: delta = beta * (v - retrieved)
    ggml_tensor* v_col = ggml_reshape_4d(ctx_, v_t, d, 1, H, 1);
    ggml_tensor* diff = ggml_sub(ctx_, v_col, retrieved);  // [d, 1, H, 1]
    // beta_t [1, H, 1, 1] → reshape to [1, 1, H, 1] for broadcasting
    ggml_tensor* beta_bc = ggml_reshape_4d(ctx_, beta_t, 1, 1, H, 1);
    ggml_tensor* delta = ggml_mul(ctx_, diff, beta_bc);    // [d, 1, H, 1]

    // Step 5: S = S_decayed + k ⊗ delta (outer product per head)
    // k_col: [d, 1, H, 1], delta: [d, 1, H, 1]
    // Need: result[i][j][h] = k[i][h] * delta[j][h]  →  [d, d, H, 1]
    //
    // Approach: transpose delta to [1, d, H, 1], then repeat both to [d, d, H, 1]
    ggml_tensor* delta_row = ggml_permute(ctx_, delta, 1, 0, 2, 3);  // [1, d, H, 1]
    delta_row = ggml_cont(ctx_, delta_row);  // contiguous after permute

    // Repeat to S's shape [d, d, H, 1]
    ggml_tensor* k_rep = ggml_repeat(ctx_, k_col, S_decayed);       // [d, d, H, 1]
    ggml_tensor* delta_rep = ggml_repeat(ctx_, delta_row, S_decayed); // [d, d, H, 1]

    ggml_tensor* outer = ggml_mul(ctx_, k_rep, delta_rep);  // [d, d, H, 1]
    *S_out = ggml_add(ctx_, S_decayed, outer);              // [d, d, H, 1]

    // Step 6: o = S_new^T @ q / sqrt(d)
    ggml_tensor* q_col = ggml_reshape_4d(ctx_, q_t, d, 1, H, 1);
    ggml_tensor* o = ggml_mul_mat(ctx_, *S_out, q_col);  // [d, 1, H, 1]
    o = ggml_scale(ctx_, o, 1.0f / sqrtf(float(d)));

    // Reshape to [d, H, 1, 1] to match expected output shape
    o = ggml_reshape_4d(ctx_, o, d, H, 1, 1);

    return o;
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
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > (int32_t)max_pos) max_pos = (uint32_t)p;
    }
    uint32_t n_kv_len = max_pos + 1;

    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32,
        n_kv_len, 1, 1, n_batch);
    ggml_set_input(kq_mask);
    ggml_set_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    uint32_t n_total_indices = n_batch * n_kv_len;
    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_total_indices);
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // 4. Layer loop
    ggml_tensor* cur;

    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor* inpSA = inpL;
        auto& block = model_.get_block(il);

        // Pre-attention/SSM norm
        cur = build_norm(gf, inpL, block.attn_norm_weight, il);

        if (meta_.is_ssm_layer(il)) {
            int32_t ssm_idx = ssm_layer_map_[il];
            cur = build_batched_ssm_layer(gf, cur, ssm_idx, slots, il);
        } else {
            int32_t kv_idx = kv_layer_map_[il];
            cur = build_batched_attention_layer(gf, cur, inp_pos,
                kq_mask, gather_indices, kv_idx, slots, positions, il);
        }

        // Residual after attention/SSM
        cur = ggml_add(ctx_, cur, inpSA);

        ggml_tensor* ffn_residual = cur;

        // Post-attention norm + FFN + residual
        cur = build_norm(gf, cur, block.ffn_norm_weight, il);
        cur = ffn_swiglu(gf, cur, block.ffn_gate_weight,
                         block.ffn_up_weight, block.ffn_down_weight, il);
        cur = ggml_add(ctx_, cur, ffn_residual);

        inpL = cur;
    }

    // 5. Output head
    cur = build_norm(gf, inpL, model_.get_output_norm_weight(), -1);
    if (model_.get_output_weight() != nullptr) {
        cur = ggml_mul_mat(ctx_, model_.get_output_weight(), cur);
    } else {
        cur = ggml_mul_mat(ctx_, model_.get_token_embedding_weight(), cur);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ============================================================
// build_batched_ssm_layer — GatedDeltaNet for multi-slot decode
//
// Batches the initial projections (matmuls on full [n_embd, n_batch]),
// then per-slot loop for state-dependent conv + fused recurrence,
// then collects outputs into [n_embd, n_batch].
// ============================================================

ggml_tensor* Qwen35ForwardPass::build_batched_ssm_layer(
    ggml_cgraph* gf,
    ggml_tensor* cur,         // [n_embd, n_batch]
    int ssm_cache_layer,
    const std::vector<uint32_t>& slots,
    int il)
{
    auto& block = model_.get_block(il);
    const size_t n_batch = slots.size();

    const int64_t d_inner      = meta_.ssm_inner_size;     // 2048
    const int64_t head_k_dim   = meta_.ssm_state_size;     // 128
    const int64_t num_k_heads  = meta_.ssm_group_count;    // 16
    const int64_t num_v_heads  = meta_.ssm_time_step_rank; // 16
    const int64_t head_v_dim   = d_inner / num_v_heads;    // 128
    const int64_t n_embd       = meta_.embedding_length;   // 1024

    const int64_t conv_kernel_size = meta_.ssm_conv_kernel; // 4
    const int64_t conv_channels = d_inner + 2 * num_k_heads * head_k_dim; // 6144

    // ========================================================
    // A. Batched projections (all slots at once)
    // ========================================================

    // [6144, n_batch]
    ggml_tensor* qkv_mixed = ggml_mul_mat(ctx_, block.attn_qkv_weight, cur);

    // [2048, n_batch]
    ggml_tensor* z_all = ggml_mul_mat(ctx_, block.attn_gate_weight, cur);

    // Beta: sigmoid([num_v_heads, n_batch])
    ggml_tensor* beta_all = ggml_mul_mat(ctx_, block.ssm_beta_weight, cur);
    beta_all = ggml_sigmoid(ctx_, beta_all);

    // Decay gate: softplus(alpha + dt_bias) * A → [num_v_heads, n_batch]
    ggml_tensor* alpha_all = ggml_mul_mat(ctx_, block.ssm_alpha_weight, cur);
    ggml_tensor* alpha_biased = ggml_add(ctx_, alpha_all, block.ssm_dt_bias);
    ggml_tensor* alpha_sp = ggml_softplus(ctx_, alpha_biased);
    ggml_tensor* gate_all = ggml_mul(ctx_, alpha_sp, block.ssm_a);

    // ========================================================
    // B. Per-slot loop: conv + recurrence + gated norm + output proj
    // ========================================================

    ggml_tensor* conv_all_tensor = ssm_cache_->get_conv_tensor(ssm_cache_layer);
    ggml_tensor* rec_all_tensor  = ssm_cache_->get_recurrent_tensor(ssm_cache_layer);

    const int64_t conv_state_elems = (conv_kernel_size - 1) * conv_channels;
    const int64_t rec_slot_floats  = head_v_dim * head_k_dim * num_v_heads;
    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    std::vector<ggml_tensor*> slot_outputs;
    slot_outputs.reserve(n_batch);

    for (size_t b = 0; b < n_batch; ++b) {
        uint32_t slot_idx = slots[b];

        // --- Slice this slot's projections ---

        // qkv_b: [conv_channels, 1]
        ggml_tensor* qkv_b = ggml_view_2d(ctx_, qkv_mixed,
            conv_channels, 1,
            qkv_mixed->nb[1], b * qkv_mixed->nb[1]);

        // z_b: [d_inner, 1]
        ggml_tensor* z_b = ggml_view_2d(ctx_, z_all,
            d_inner, 1,
            z_all->nb[1], b * z_all->nb[1]);

        // beta_b: [num_v_heads, 1] → [1, num_v_heads, 1, 1]
        ggml_tensor* beta_b = ggml_view_2d(ctx_, beta_all,
            num_v_heads, 1,
            beta_all->nb[1], b * beta_all->nb[1]);
        beta_b = ggml_reshape_4d(ctx_, beta_b, 1, num_v_heads, 1, 1);

        // gate_b: [num_v_heads, 1] → [1, num_v_heads, 1, 1]
        ggml_tensor* gate_b = ggml_view_2d(ctx_, gate_all,
            num_v_heads, 1,
            gate_all->nb[1], b * gate_all->nb[1]);
        gate_b = ggml_reshape_4d(ctx_, gate_b, 1, num_v_heads, 1, 1);

        // --- Conv1D with per-slot state ---

        ggml_tensor* conv_state = ggml_view_1d(ctx_, conv_all_tensor,
            conv_state_elems, slot_idx * conv_all_tensor->nb[1]);
        conv_state = ggml_reshape_3d(ctx_, conv_state,
            conv_kernel_size - 1, conv_channels, 1);

        // qkv_b → [conv_channels, 1, 1] → transpose → [1, conv_channels, 1]
        ggml_tensor* qkv_b_3d = ggml_reshape_3d(ctx_, qkv_b, conv_channels, 1, 1);
        ggml_tensor* qkv_b_t = ggml_transpose(ctx_, qkv_b_3d);

        // Concat: [conv_kernel-1 + 1, conv_channels, 1] = [conv_kernel, conv_channels, 1]
        ggml_tensor* conv_input = ggml_concat(ctx_, conv_state, qkv_b_t, 0);

        // Update conv state cache: last (conv_kernel-1) entries
        ggml_tensor* last_conv = ggml_view_3d(ctx_, conv_input,
            conv_kernel_size - 1, conv_channels, 1,
            conv_input->nb[1], conv_input->nb[2],
            (conv_input->ne[0] - (conv_kernel_size - 1)) * ggml_element_size(conv_input));
        ggml_tensor* conv_dst = ggml_view_1d(ctx_, conv_all_tensor,
            conv_state_elems, slot_idx * conv_all_tensor->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx_, last_conv, conv_dst));

        // Depthwise conv1d + SiLU
        ggml_tensor* conv_out = ggml_ssm_conv(ctx_, conv_input, block.ssm_conv1d_weight);
        conv_out = ggml_silu(ctx_, conv_out);

        // --- QKV split from conv output ---

        const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

        ggml_tensor* q_conv = ggml_view_4d(ctx_, conv_out,
            head_k_dim, num_k_heads, 1, 1,
            ggml_row_size(conv_out->type, head_k_dim),
            nb1_qkv, nb1_qkv, 0);

        ggml_tensor* k_conv = ggml_view_4d(ctx_, conv_out,
            head_k_dim, num_k_heads, 1, 1,
            ggml_row_size(conv_out->type, head_k_dim),
            nb1_qkv, nb1_qkv,
            head_k_dim * num_k_heads * ggml_element_size(conv_out));

        ggml_tensor* v_conv = ggml_view_4d(ctx_, conv_out,
            head_v_dim, num_v_heads, 1, 1,
            ggml_row_size(conv_out->type, head_v_dim),
            nb1_qkv, nb1_qkv,
            ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

        // L2 normalize Q and K
        q_conv = ggml_l2_norm(ctx_, q_conv, meta_.rms_norm_eps);
        k_conv = ggml_l2_norm(ctx_, k_conv, meta_.rms_norm_eps);

        if (num_k_heads != num_v_heads) {
            q_conv = ggml_repeat_4d(ctx_, q_conv, head_k_dim, num_v_heads, 1, 1);
            k_conv = ggml_repeat_4d(ctx_, k_conv, head_k_dim, num_v_heads, 1, 1);
        }    

        // --- Fused delta net recurrence ---

        ggml_tensor* S = ggml_view_1d(ctx_, rec_all_tensor,
            rec_slot_floats, slot_idx * rec_all_tensor->nb[1]);
        S = ggml_reshape_4d(ctx_, S, head_v_dim, head_k_dim, num_v_heads, 1);

        ggml_tensor* gdn_result = ggml_gated_delta_net(ctx_,
            q_conv, k_conv, v_conv, gate_b, beta_b, S);

        // Output: [head_v_dim, num_v_heads, 1, 1]
        ggml_tensor* output = ggml_view_4d(ctx_, gdn_result,
            head_v_dim, num_v_heads, 1, 1,
            ggml_row_size(gdn_result->type, head_v_dim),
            ggml_row_size(gdn_result->type, head_v_dim * num_v_heads),
            ggml_row_size(gdn_result->type, head_v_dim * num_v_heads), 0);

        // New state: [head_v_dim, head_k_dim, num_v_heads, 1]
        ggml_tensor* new_state = ggml_view_4d(ctx_, gdn_result,
            head_v_dim, head_k_dim, num_v_heads, 1,
            ggml_row_size(gdn_result->type, head_v_dim),
            ggml_row_size(gdn_result->type, head_v_dim * head_k_dim),
            ggml_row_size(gdn_result->type, head_v_dim * head_k_dim * num_v_heads),
            ggml_row_size(gdn_result->type, head_v_dim * num_v_heads * 1));

        // Write new state back to cache
        ggml_tensor* rec_dst = ggml_view_1d(ctx_, rec_all_tensor,
            rec_slot_floats, slot_idx * rec_all_tensor->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx_,
            ggml_reshape_1d(ctx_, new_state, rec_slot_floats), rec_dst));

        // --- Gated normalization ---

        ggml_tensor* z_b_4d = ggml_reshape_4d(ctx_, z_b,
            head_v_dim, num_v_heads, 1, 1);
        ggml_tensor* out_normed = build_norm(gf, output, block.ssm_norm_weight, il);
        ggml_tensor* gated = ggml_mul(ctx_, out_normed, ggml_silu(ctx_, z_b_4d));

        // --- Output projection → [n_embd, 1] ---

        ggml_tensor* flat = ggml_reshape_2d(ctx_, gated, d_inner, 1);
        ggml_tensor* out_b = ggml_mul_mat(ctx_, block.ssm_out_weight, flat);

        slot_outputs.push_back(out_b);
    }

    // ========================================================
    // C. Concat slot outputs → [n_embd, n_batch]
    // ========================================================

    ggml_tensor* result;
    if (n_batch == 1) {
        result = slot_outputs[0];
    } else {
        result = slot_outputs[0];
        for (size_t b = 1; b < n_batch; ++b) {
            result = ggml_concat(ctx_, result, slot_outputs[b], 1);
        }
    }
    result = ggml_reshape_2d(ctx_, result, n_embd, n_batch);

    return result;
}

// ============================================================
// build_batched_attention_layer — Qwen35 attention for multi-slot decode
//
// Joint Q+Gate projection, Q/K norms, partial RoPE, sigmoid gating.
// KV cache: per-slot write, gathered read, shared mask.
// ============================================================

ggml_tensor* Qwen35ForwardPass::build_batched_attention_layer(
    ggml_cgraph* gf,
    ggml_tensor* cur,            // [n_embd, n_batch]
    ggml_tensor* inp_pos,        // [n_batch]
    ggml_tensor* kq_mask,        // [n_kv_len, 1, 1, n_batch]
    ggml_tensor* gather_indices, // [n_batch * n_kv_len]
    int kv_cache_layer,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions,
    int il)
{
    const int n_embd_head = meta_.attention_key_length;
    const int n_head      = meta_.attention_head_count;
    const int n_head_kv   = meta_.attention_head_count_kv;
    const size_t n_batch  = slots.size();
    auto& block = model_.get_block(il);

    const int n_rot = (meta_.rope_dimension_count > 0)
        ? meta_.rope_dimension_count : n_embd_head;

    // A. Joint Q+Gate projection → [(n_embd_head*2)*n_head, n_batch]
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx_, block.attn_q_weight, cur);

    // B. Extract Q via strided view → [n_embd_head, n_head, n_batch]
    ggml_tensor* Qcur = ggml_view_3d(ctx_, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);

    Qcur = build_norm(gf, Qcur, block.attn_q_norm_weight, il);

    // C. K and V projections
    ggml_tensor* Kcur = ggml_mul_mat(ctx_, block.attn_k_weight, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx_, block.attn_v_weight, cur);

    Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head_kv, n_batch);
    Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head_kv, n_batch);

    Kcur = build_norm(gf, Kcur, block.attn_k_norm_weight, il);

    // D. Extract Gate → [n_embd_head*n_head, n_batch]
    ggml_tensor* gate = ggml_view_3d(ctx_, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx_, gate, n_embd_head * n_head, n_batch);

    // E. RoPE
    const float freq_base = meta_.rope_freq_base;
    Qcur = ggml_rope_ext(ctx_, Qcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        meta_.context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(ctx_, Kcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        meta_.context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // F. Per-slot KV cache write
    const int n_embd_k = n_head_kv * n_embd_head;
    const int n_embd_v = n_head_kv * n_embd_head;

    ggml_tensor* k_storage_fmt = ggml_reshape_3d(ctx_, Kcur, n_embd_k, 1, n_batch);
    ggml_tensor* v_storage_fmt = ggml_reshape_3d(ctx_, Vcur, n_embd_v, 1, n_batch);

    for (size_t b = 0; b < n_batch; ++b) {
        ggml_tensor* k_slice = ggml_view_2d(ctx_, k_storage_fmt,
            n_embd_k, 1, k_storage_fmt->nb[1], b * k_storage_fmt->nb[2]);
        ggml_tensor* v_slice = ggml_view_2d(ctx_, v_storage_fmt,
            n_embd_v, 1, v_storage_fmt->nb[1], b * v_storage_fmt->nb[2]);

        ggml_build_forward_expand(gf, kv_cache_->cpy_k(ctx_, k_slice, kv_cache_layer, slots[b]));
        ggml_build_forward_expand(gf, kv_cache_->cpy_v(ctx_, v_slice, kv_cache_layer, slots[b]));
    }

    // G. Gather KV for all slots
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > (int32_t)max_pos) max_pos = (uint32_t)p;
    }
    uint32_t n_kv_len = max_pos + 1;

    ggml_tensor* k_gathered = kv_cache_->gather_k(ctx_, gf, kv_cache_layer,
        gather_indices, n_batch, n_kv_len);
    ggml_tensor* v_gathered = kv_cache_->gather_v(ctx_, gf, kv_cache_layer,
        gather_indices, n_batch, n_kv_len);

    // Reshape to 4D: [n_embd_head, n_head_kv, n_kv_len, n_batch]
    ggml_tensor* k_view = ggml_view_4d(ctx_, k_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_k * sizeof(float),
        n_embd_k * n_kv_len * sizeof(float), 0);
    ggml_tensor* v_view = ggml_view_4d(ctx_, v_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_v * sizeof(float),
        n_embd_v * n_kv_len * sizeof(float), 0);

    // H. Attention
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    cur = build_attn_mha(gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il);

    // I. Gate sigmoid
    cur = ggml_mul(ctx_, cur, ggml_sigmoid(ctx_, gate));

    // J. Output projection
    cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);

    return cur;
}

// ============================================================
// set_batched_inputs — fill input tensors for batched decode
// ============================================================

void Qwen35ForwardPass::set_batched_inputs(
    ggml_cgraph* gf,
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions)
{
    const size_t n_batch = tokens.size();
    if (slots.size() != n_batch || positions.size() != n_batch) {
        throw std::runtime_error("Input vector size mismatch in set_batched_inputs");
    }

    // 1. Token IDs
    ggml_tensor* tokens_tensor = ggml_graph_get_tensor(gf, "tokens");
    if (!tokens_tensor) throw std::runtime_error("tokens tensor not found");
    ggml_backend_tensor_set(tokens_tensor, tokens.data(), 0, n_batch * sizeof(int32_t));

    // 2. Position IDs
    ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (!inp_pos) throw std::runtime_error("inp_pos tensor not found");
    ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_batch * sizeof(int32_t));

    // 3. Attention mask (causal, per-slot positions)
    ggml_tensor* kq_mask_t = ggml_graph_get_tensor(gf, "kq_mask_b");
    if (!kq_mask_t) throw std::runtime_error("kq_mask_b tensor not found");

    const uint32_t n_kv_dim = kq_mask_t->ne[0];
    std::vector<float> mask_data(n_kv_dim * n_batch);
    for (uint32_t b = 0; b < n_batch; ++b) {
        const int32_t current_pos = positions[b];
        for (uint32_t j = 0; j < n_kv_dim; ++j) {
            mask_data[b * n_kv_dim + j] = ((int32_t)j <= current_pos) ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(kq_mask_t, mask_data.data(), 0, n_kv_dim * n_batch * sizeof(float));

    // 4. KV gather indices
    ggml_tensor* gather_indices = ggml_graph_get_tensor(gf, "gather_indices");
    if (gather_indices) {
        uint32_t n_total_indices = gather_indices->ne[0];
        uint32_t n_kv_len = n_total_indices / n_batch;
        std::vector<int32_t> indices(n_total_indices);

        for (uint32_t b = 0; b < n_batch; ++b) {
            uint32_t slot_idx = slots[b];
            for (uint32_t t = 0; t < n_kv_len; ++t) {
                indices[b * n_kv_len + t] = slot_idx * kv_cache_->get_n_ctx_max() + t;
            }
        }
        ggml_backend_tensor_set(gather_indices, indices.data(), 0, indices.size() * sizeof(int32_t));
    }
}