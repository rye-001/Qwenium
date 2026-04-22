#include "deltanet.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

// ── Internal helpers ──────────────────────────────────────────────────────────

static void dn_set_name(ggml_cgraph* gf, ggml_tensor* t, const char* base, int il) {
    char name[64];
    snprintf(name, sizeof(name), "%s.%d", base, il);
    ggml_set_name(t, name);
    ggml_build_forward_expand(gf, t);
}

// ── Free function ─────────────────────────────────────────────────────────────

ggml_tensor* build_deltanet_layer(
    ggml_context*   ctx,
    ggml_cgraph*    gf,
    ggml_tensor*    cur,
    DeltaNetState*  dn_state,
    uint32_t        dn_idx,
    uint32_t        slot_idx,
    uint32_t        n_tokens,
    ggml_tensor*    w_qkv,
    ggml_tensor*    w_gate,
    ggml_tensor*    w_beta,
    ggml_tensor*    w_a,
    ggml_tensor*    w_dt_bias,
    ggml_tensor*    w_a_log,
    ggml_tensor*    w_conv,
    ggml_tensor*    w_norm,
    ggml_tensor*    w_out,
    int             n_embd,
    int             d_inner,
    int             head_k_dim,
    int             num_k_heads,
    int             num_v_heads,
    int             head_v_dim,
    int             conv_channels,
    int             conv_kernel,
    float           rms_norm_eps,
    int             il)
{
    const int64_t n_seq_tokens = static_cast<int64_t>(n_tokens);
    const int64_t n_seqs       = 1;

    // ── 1. Input projections ─────────────────────────────────────────────────

    // QKV mixed: [n_embd, conv_channels] @ [n_embd, n_tokens]^T → [conv_channels, n_tokens]
    ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, w_qkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    dn_set_name(gf, qkv_mixed, "dn_qkv", il);

    // Z (output gate): [n_embd, d_inner] @ cur → [d_inner, n_tokens]
    ggml_tensor* z = ggml_mul_mat(ctx, w_gate, cur);
    dn_set_name(gf, z, "dn_z", il);

    // ── 2. Beta and decay-gate projections ───────────────────────────────────

    // Beta: sigmoid([n_embd, num_v_heads] @ cur) → [1, num_v_heads, n_tokens, 1]
    ggml_tensor* beta = ggml_mul_mat(ctx, w_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);
    dn_set_name(gf, beta, "dn_beta", il);

    // Alpha → decay gate: softplus(alpha @ cur + dt_bias) * A_log
    ggml_tensor* alpha = ggml_mul_mat(ctx, w_a, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* alpha_biased = ggml_add(ctx, alpha, w_dt_bias);
    ggml_tensor* alpha_sp     = ggml_softplus(ctx, alpha_biased);
    ggml_tensor* decay_gate   = ggml_mul(ctx, alpha_sp, w_a_log);
    decay_gate = ggml_reshape_4d(ctx, decay_gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    dn_set_name(gf, decay_gate, "dn_decay", il);

    // ── 3. Causal conv1d ─────────────────────────────────────────────────────

    ggml_tensor* conv_all = dn_state->conv_tensor(dn_idx);
    const int64_t conv_state_elems = static_cast<int64_t>(conv_kernel - 1) * conv_channels;

    // Extract the sliding window for this slot
    ggml_tensor* conv_states = ggml_view_1d(ctx, conv_all,
        conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    conv_states = ggml_reshape_3d(ctx, conv_states, conv_kernel - 1, conv_channels, n_seqs);
    dn_set_name(gf, conv_states, "dn_conv_st", il);

    // Concatenate conv window with new QKV tokens (along time axis = dim 0)
    ggml_tensor* qkv_t = ggml_transpose(ctx, qkv_mixed);
    ggml_tensor* conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);
    dn_set_name(gf, conv_input, "dn_conv_in", il);

    // Update conv state: keep last (conv_kernel-1) tokens
    ggml_tensor* last_conv = ggml_view_3d(ctx, conv_input,
        conv_kernel - 1, conv_channels, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        static_cast<size_t>(conv_input->ne[0] - (conv_kernel - 1)) * ggml_element_size(conv_input));
    ggml_tensor* conv_dst = ggml_view_1d(ctx, conv_all,
        conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_dst));

    // Depthwise conv1d + SiLU activation
    ggml_tensor* conv_out = ggml_ssm_conv(ctx, conv_input, w_conv);
    conv_out = ggml_silu(ctx, conv_out);
    dn_set_name(gf, conv_out, "dn_conv_out", il);

    // ── 4. Split conv output into Q, K, V ────────────────────────────────────

    const int64_t qkv_dim  = static_cast<int64_t>(head_k_dim) * num_k_heads * 2
                             + static_cast<int64_t>(head_v_dim) * num_v_heads;
    const int64_t nb1_qkv  = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor* q_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens, 0);

    ggml_tensor* k_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        static_cast<size_t>(head_k_dim) * num_k_heads * ggml_element_size(conv_out));

    ggml_tensor* v_conv = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_v_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        ggml_row_size(conv_out->type, 2LL * head_k_dim * num_k_heads));

    // ── 5. L2-normalise Q and K ───────────────────────────────────────────────

    q_conv = ggml_l2_norm(ctx, q_conv, rms_norm_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, rms_norm_eps);

    // Repeat K heads to match V heads if needed (GQA-style grouping)
    if (num_k_heads != num_v_heads) {
        q_conv = ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // ── 6. Gated delta-net recurrence (fused op) ──────────────────────────────

    ggml_tensor* rec_all = dn_state->recurrent_tensor(dn_idx);
    const int64_t rec_slot_floats = static_cast<int64_t>(head_v_dim) * head_k_dim * num_v_heads;

    ggml_tensor* S = ggml_view_1d(ctx, rec_all,
        rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    S = ggml_reshape_4d(ctx, S, head_v_dim, head_k_dim, num_v_heads, n_seqs);
    dn_set_name(gf, S, "dn_state_in", il);

    // ggml_gated_delta_net: fused gated-delta-rule forward pass.
    // Returns a tensor that packs both the per-token output AND the final state:
    //   output:    [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]   (first part)
    //   new_state: [head_v_dim, head_k_dim,  num_v_heads,  n_seqs]  (second part)

    ggml_tensor* result = ggml_gated_delta_net(ctx,
        q_conv, k_conv, v_conv, decay_gate, beta, S);
    dn_set_name(gf, result, "dn_gdn_result", il);

    // Extract per-token output view
    ggml_tensor* output = ggml_view_4d(ctx, result,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens),
        0);
    dn_set_name(gf, output, "dn_delta_out", il);

    // Extract and write back the new recurrent state
    ggml_tensor* new_state = ggml_view_4d(ctx, result,
        head_v_dim, head_k_dim, num_v_heads, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim * num_v_heads),
        ggml_row_size(result->type,
            static_cast<int64_t>(head_v_dim) * num_v_heads * n_seq_tokens * n_seqs));

    ggml_tensor* rec_dst = ggml_view_1d(ctx, rec_all,
        rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx,
        ggml_reshape_1d(ctx, new_state, rec_slot_floats), rec_dst));

    // ── 7. Gated RMSNorm ─────────────────────────────────────────────────────

    // output is [head_v_dim, num_v_heads, n_seq_tokens, n_seqs].
    // Normalize each head_v_dim vector independently (per-head, not d_inner-wide).
    // w_norm is [head_v_dim]; ggml broadcasts it over all heads and tokens.
    ggml_tensor* z_4d   = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* normed = ggml_rms_norm(ctx, output, rms_norm_eps);
    normed = ggml_mul(ctx, normed, w_norm);
    ggml_tensor* z_silu = ggml_silu(ctx, z_4d);
    ggml_tensor* gated  = ggml_mul(ctx, normed, z_silu);
    dn_set_name(gf, gated, "dn_gated", il);

    // ── 8. Output projection ─────────────────────────────────────────────────

    ggml_tensor* flat = ggml_reshape_3d(ctx, gated,
        static_cast<int64_t>(head_v_dim) * num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* out_proj = ggml_mul_mat(ctx, w_out, flat);
    dn_set_name(gf, out_proj, "dn_output", il);
    out_proj = ggml_reshape_2d(ctx, out_proj, n_embd, n_seq_tokens * n_seqs);

    return out_proj;
}

// ── DeltaNetLayer class ───────────────────────────────────────────────────────

DeltaNetLayer::DeltaNetLayer(
    ggml_tensor*   w_qkv,
    ggml_tensor*   w_gate,
    ggml_tensor*   w_beta,
    ggml_tensor*   w_a,
    ggml_tensor*   w_dt_bias,
    ggml_tensor*   w_a_log,
    ggml_tensor*   w_conv,
    ggml_tensor*   w_norm,
    ggml_tensor*   w_out,
    DeltaNetState* dn_state,
    const Hparams& hp)
    : w_qkv_(w_qkv), w_gate_(w_gate), w_beta_(w_beta), w_a_(w_a)
    , w_dt_bias_(w_dt_bias), w_a_log_(w_a_log)
    , w_conv_(w_conv), w_norm_(w_norm), w_out_(w_out)
    , dn_state_(dn_state), hp_(hp)
{}

ggml_tensor* DeltaNetLayer::build(
    ggml_context*      ctx,
    ggml_cgraph*       gf,
    ggml_tensor*       input,
    uint32_t           dn_idx,
    Phase              phase,
    const PrefillArgs& prefill_args,
    const DecodeArgs*  decode_args)
{
    if (phase == Phase::Prefill) {
        return build_prefill(ctx, gf, input, dn_idx, prefill_args);
    } else {
        if (!decode_args) {
            throw std::runtime_error(
                "DeltaNetLayer::build: decode_args must be non-null for Phase::Decode");
        }
        return build_decode(ctx, gf, input, dn_idx, *decode_args);
    }
}

ggml_tensor* DeltaNetLayer::build_prefill(
    ggml_context*      ctx,
    ggml_cgraph*       gf,
    ggml_tensor*       input,
    uint32_t           dn_idx,
    const PrefillArgs& args)
{
    // Derive n_embd from input shape
    const int n_embd = static_cast<int>(input->ne[0]);

    return build_deltanet_layer(
        ctx, gf, input, dn_state_, dn_idx, args.slot_idx, args.n_tokens,
        w_qkv_, w_gate_, w_beta_, w_a_, w_dt_bias_, w_a_log_,
        w_conv_, w_norm_, w_out_,
        n_embd, hp_.d_inner,
        hp_.head_k_dim, hp_.num_k_heads, hp_.num_v_heads, hp_.head_v_dim,
        hp_.conv_channels, hp_.conv_kernel, hp_.rms_norm_eps,
        static_cast<int>(dn_idx));
}

ggml_tensor* DeltaNetLayer::build_decode(
    ggml_context*     ctx,
    ggml_cgraph*      gf,
    ggml_tensor*      input,
    uint32_t          dn_idx,
    const DecodeArgs& args)
{
    // Decode: batch of single tokens, one per slot.
    // We process each slot individually to keep state writes correct, then
    // concatenate the outputs. For n_batch == 1 this reduces to one prefill call.
    const int n_embd  = static_cast<int>(input->ne[0]);
    const int n_batch = static_cast<int>(args.slots.size());

    if (n_batch == 0) {
        throw std::runtime_error("DeltaNetLayer::build_decode: slots must be non-empty");
    }

    if (n_batch == 1) {
        PrefillArgs pa;
        pa.n_tokens = 1;
        pa.slot_idx = args.slots[0];
        return build_prefill(ctx, gf, input, dn_idx, pa);
    }

    // Multi-slot decode: slice the batch input per slot, process independently,
    // then concatenate. This is correct but not fused; Phase 4 can optimize.
    std::vector<ggml_tensor*> slot_outs;
    slot_outs.reserve(n_batch);

    for (int b = 0; b < n_batch; ++b) {
        // input[:, b] — one token for slot b
        ggml_tensor* token_in = ggml_view_2d(ctx, input,
            n_embd, 1,
            input->nb[1],
            static_cast<size_t>(b) * input->nb[1]);

        PrefillArgs pa;
        pa.n_tokens = 1;
        pa.slot_idx = args.slots[b];
        ggml_tensor* out_b = build_prefill(ctx, gf, token_in, dn_idx, pa);
        slot_outs.push_back(out_b);
    }

    // Concatenate along token dimension
    ggml_tensor* combined = slot_outs[0];
    for (int b = 1; b < n_batch; ++b) {
        combined = ggml_concat(ctx, combined, slot_outs[b], 1);
    }
    return combined;
}
