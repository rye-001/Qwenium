#include "forward-pass-qwen35.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>

// ============================================================
// Constructor: hybrid cache setup
// ============================================================

Qwen35ForwardPass::Qwen35ForwardPass(
    const Qwen3Model& model, const Qwen3Metadata* metadata,
    uint32_t context_len, uint32_t max_batch_size)
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

    // KV cache — attention layers only
    uint32_t n_embd_k = meta_.attention_key_length * meta_.attention_head_count_kv;
    uint32_t n_embd_v = meta_.attention_value_length * meta_.attention_head_count_kv;

    kv_cache_ = std::make_unique<simple_kv_cache>(
        n_attn_layers, context_len, max_batch_size,
        n_embd_k, n_embd_v,
        GGML_TYPE_F32, GGML_TYPE_F32,
        cache_backend
    );

    // SSM state cache — GatedDeltaNet layers only
    // conv_channels = d_inner + 2 * n_group * d_state = 2048 + 2*16*128 = 6144
    uint32_t conv_dim = meta_.ssm_inner_size + 2 * meta_.ssm_group_count * meta_.ssm_state_size;

    ssm_cache_ = std::make_unique<ssm_state_cache>(
        n_ssm_layers, max_batch_size,
        meta_.ssm_state_size, meta_.ssm_group_count,
        conv_dim, meta_.ssm_conv_kernel,
        cache_backend
    );
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
            int32_t kv_idx = kv_layer_map_[il];
            cur = build_attention_layer(gf, kv_cache_.get(), cur, inp_pos,
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

ggml_cgraph* Qwen35ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions)
{
    throw std::runtime_error("qwen35: batched decoding not yet implemented");
}

void Qwen35ForwardPass::set_batched_inputs(ggml_cgraph* gf,
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions)
{
    throw std::runtime_error("qwen35: batched inputs not yet implemented");
}