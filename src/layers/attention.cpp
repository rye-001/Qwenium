#include "attention.h"
#include "norm.h"
#include "../state/kv_cache_simple.h"

#include "ggml.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cstring>

// ── Internal helper ──────────────────────────────────────────────────────────
// Mirrors ForwardPassBase::set_tensor_name — names a tensor with an optional
// layer-index suffix (".N").
static void set_name(ggml_tensor* t, const char* base, int il = -1) {
    if (il >= 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s.%d", base, il);
        ggml_set_name(t, buf);
    } else {
        ggml_set_name(t, base);
    }
}

// ── build_attn_mha ───────────────────────────────────────────────────────────
// Extracted from ForwardPassBase::build_attn_mha (src/models/forward_pass_base.cpp).
// Logic is identical — only ctx_ → ctx parameter.
ggml_tensor* build_attn_mha(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  q,
    ggml_tensor*  k,
    ggml_tensor*  v,
    ggml_tensor*  kq_mask,
    ggml_tensor*  sinks,
    float         kq_scale,
    uint32_t      pos,
    int           il)
{
    (void)gf; (void)pos; // gf/pos unused directly; kept for API symmetry with callers

    const bool v_trans = v->nb[1] > v->nb[2];
    (void)v_trans;

    const auto n_stream = k->ne[3];

    q = ggml_reshape_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream);
    set_name(q, "q_reshaped", il);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    set_name(q, "q_permuted", il);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    set_name(k, "k_permuted", il);
    v = ggml_permute(ctx, v, 1, 2, 0, 3);
    set_name(v, "v_permuted", il);
    v = ggml_cont(ctx, v);
    set_name(v, "v_cont", il);

    ggml_tensor* cur;
    {
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        set_name(kq, "kq", il);

        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0);
        set_name(kq, "kq_soft", il);

        ggml_soft_max_add_sinks(kq, sinks);

        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
        set_name(kqv, "kqv", il);

        cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        set_name(cur, "kqv_permuted", il);

        cur = ggml_cont_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        set_name(cur, "attn_recombined", il);
    }

    return cur;
}

// ── build_attention ──────────────────────────────────────────────────────────
// Extracted from Qwen3ForwardPass::_build_attention_layer (forward-pass.cpp).
// Logic is identical — ctx_ → ctx, meta_ fields → n_embd_head/n_head_kv params.
ggml_tensor* build_attention(
    ggml_context*    ctx,
    ggml_cgraph*     gf,
    simple_kv_cache* kv_cache,
    ggml_tensor*     q,
    ggml_tensor*     k,
    ggml_tensor*     v,
    int              layer_idx,
    float            kq_scale,
    uint32_t         n_tokens,
    uint32_t         slot_idx,
    int              il,
    int              n_embd_head,
    int              n_head_kv)
{
    const uint32_t pos  = kv_cache->get_pos(slot_idx);
    const uint32_t n_kv = pos + n_tokens;

    // 1. Cache the new K and V values
    ggml_tensor* k_cached = kv_cache->cpy_k(ctx, k, layer_idx, slot_idx);
    set_name(k_cached, "k_cached", il);
    ggml_build_forward_expand(gf, k_cached);

    ggml_tensor* v_cached = kv_cache->cpy_v(ctx, v, layer_idx, slot_idx);
    set_name(v_cached, "v_cached", il);
    ggml_build_forward_expand(gf, v_cached);

    // 2. Retrieve the full K and V sequences from the cache
    ggml_tensor* k_full = kv_cache->get_k(ctx, layer_idx, n_kv, slot_idx);
    ggml_tensor* v_full = kv_cache->get_v(ctx, layer_idx, n_kv, slot_idx);

    // 3. Create views for the attention calculation
    const int n_embd_k = n_head_kv * n_embd_head;
    const int n_embd_v = n_head_kv * n_embd_head;

    k = ggml_view_3d(ctx, k_full,
        n_embd_head,
        n_head_kv,
        n_kv,
        n_embd_head * sizeof(float),
        n_embd_k   * sizeof(float),
        0);

    v = ggml_view_3d(ctx, v_full,
        n_embd_head,
        n_head_kv,
        n_kv,
        n_embd_head * sizeof(float),
        n_embd_v   * sizeof(float),
        0);

    set_name(k, "k_view", il);
    set_name(v, "v_view", il);

    // 4. Build the causal mask for the full sequence
    ggml_tensor* kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    set_name(kq_mask, "kq_mask", il);
    ggml_build_forward_expand(gf, kq_mask);

    // 5. Run MHA
    return build_attn_mha(ctx, gf, q, k, v, kq_mask, nullptr, kq_scale, pos, il);
}

// ── build_batched_attention ──────────────────────────────────────────────────
// Extracted from Qwen3ForwardPass::_build_batched_attention_layer (forward-pass.cpp).
// Logic is identical — ctx_ → ctx, kv_cache_ → kv_cache parameter.
ggml_tensor* build_batched_attention(
    ggml_context*                ctx,
    ggml_cgraph*                 gf,
    simple_kv_cache*             kv_cache,
    ggml_tensor*                 q,
    ggml_tensor*                 k,
    ggml_tensor*                 v,
    int                          layer_idx,
    float                        kq_scale,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>&  positions,
    ggml_tensor*                 kq_mask,
    ggml_tensor*                 gather_indices,
    int                          il)
{
    const size_t n_batch    = slots.size();
    const int    n_embd_head = k->ne[0];
    const int    n_head_kv   = k->ne[1];
    const int    n_embd_k    = n_embd_head * n_head_kv;
    const int    n_embd_v    = n_embd_head * n_head_kv;

    // 1. Write new K/V per slot
    ggml_tensor* k_storage_fmt = ggml_reshape_3d(ctx, k, n_embd_k, 1, n_batch);
    ggml_tensor* v_storage_fmt = ggml_reshape_3d(ctx, v, n_embd_v, 1, n_batch);

    for (size_t i = 0; i < n_batch; ++i) {
        size_t k_offset = i * k_storage_fmt->nb[2];
        ggml_tensor* k_slice = ggml_view_2d(ctx, k_storage_fmt, n_embd_k, 1,
            k_storage_fmt->nb[1], k_offset);

        size_t v_offset = i * v_storage_fmt->nb[2];
        ggml_tensor* v_slice = ggml_view_2d(ctx, v_storage_fmt, n_embd_v, 1,
            v_storage_fmt->nb[1], v_offset);

        ggml_tensor* k_stored = kv_cache->cpy_k(ctx, k_slice, layer_idx, slots[i]);
        set_name(k_stored, "k_stored_b", il);
        ggml_build_forward_expand(gf, k_stored);

        ggml_tensor* v_stored = kv_cache->cpy_v(ctx, v_slice, layer_idx, slots[i]);
        set_name(v_stored, "v_stored_b", il);
        ggml_build_forward_expand(gf, v_stored);
    }

    // 2. Gather KV cache for attention
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > (int32_t)max_pos) max_pos = (uint32_t)p;
    }
    uint32_t n_kv_len = max_pos + 1;

    ggml_tensor* k_gathered = kv_cache->gather_k(ctx, gf, layer_idx, gather_indices,
        n_batch, n_kv_len);
    ggml_tensor* v_gathered = kv_cache->gather_v(ctx, gf, layer_idx, gather_indices,
        n_batch, n_kv_len);

    // 3. Reshape for attention
    ggml_tensor* k_view = ggml_view_4d(ctx, k_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_k    * sizeof(float),
        n_embd_k    * n_kv_len * sizeof(float),
        0);

    ggml_tensor* v_view = ggml_view_4d(ctx, v_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_v    * sizeof(float),
        n_embd_v    * n_kv_len * sizeof(float),
        0);

    // 4. Run MHA
    return build_attn_mha(ctx, gf, q, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il);
}

// ── build_gated_attention ─────────────────────────────────────────────────────
// Gated attention variant used by Qwen3.5 and Qwen3.6: joint Q+Gate projection,
// Q/K RMS norms, partial RoPE, sigmoid gating on the output.
ggml_tensor* build_gated_attention(
    ggml_context*    ctx,
    ggml_cgraph*     gf,
    simple_kv_cache* kv_cache,
    ggml_tensor*     cur,
    ggml_tensor*     inp_pos,
    int              kv_cache_layer,
    uint32_t         n_tokens,
    uint32_t         slot_idx,
    int              il,
    ggml_tensor*     w_q,
    ggml_tensor*     w_q_norm,
    ggml_tensor*     w_k,
    ggml_tensor*     w_k_norm,
    ggml_tensor*     w_v,
    ggml_tensor*     w_out,
    int              n_embd_head,
    int              n_head,
    int              n_head_kv,
    int              n_rot,
    float            freq_base,
    int              context_length,
    float            rms_norm_eps)
{
    // A. Joint Q+Gate projection
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx, w_q, cur);
    set_name(Qcur_full, "Qcur_full", il);

    // B. Extract Q via strided view (every other n_embd_head block)
    ggml_tensor* Qcur = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    set_name(Qcur, "Qcur", il);

    Qcur = build_rms_norm(ctx, Qcur, w_q_norm, rms_norm_eps, il);
    set_name(Qcur, "Qcur_normed", il);

    // C. K and V projections
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    Kcur = build_rms_norm(ctx, Kcur, w_k_norm, rms_norm_eps, il);
    set_name(Kcur, "Kcur_normed", il);

    // D. Extract Gate (offset by n_embd_head within each interleaved pair)
    ggml_tensor* gate = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
    set_name(gate, "gate", il);

    // E. Partial RoPE
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // F. KV cache write + full-history read
    const float    kq_scale  = 1.0f / sqrtf(float(n_embd_head));
    const uint32_t cache_pos = kv_cache->get_pos(slot_idx);
    const uint32_t n_kv      = cache_pos + n_tokens;

    ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx, Kcur, kv_cache_layer, slot_idx));
    ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx, Vcur, kv_cache_layer, slot_idx));

    ggml_tensor* k_full = kv_cache->get_k(ctx, kv_cache_layer, n_kv, slot_idx);
    ggml_tensor* v_full = kv_cache->get_v(ctx, kv_cache_layer, n_kv, slot_idx);

    const int n_embd_kv = n_head_kv * n_embd_head;
    ggml_tensor* k_view = ggml_view_3d(ctx, k_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * sizeof(float), n_embd_kv * sizeof(float), 0);
    ggml_tensor* v_view = ggml_view_3d(ctx, v_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * sizeof(float), n_embd_kv * sizeof(float), 0);

    ggml_tensor* kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    set_name(kq_mask, "kq_mask", il);
    ggml_build_forward_expand(gf, kq_mask);

    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, cache_pos, il);

    // G. Sigmoid gating
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));
    set_name(cur, "attn_gated", il);

    // H. Output projection
    cur = ggml_mul_mat(ctx, w_out, cur);
    set_name(cur, "attn_output", il);

    return cur;
}

// ── build_gated_batched_attention ─────────────────────────────────────────────
// Batched decode variant of build_gated_attention: same projections/norms/gating,
// operates on a batch of slots with pre-built kq_mask and gather_indices.
ggml_tensor* build_gated_batched_attention(
    ggml_context*                ctx,
    ggml_cgraph*                 gf,
    simple_kv_cache*             kv_cache,
    ggml_tensor*                 cur,
    ggml_tensor*                 inp_pos,
    ggml_tensor*                 kq_mask,
    ggml_tensor*                 gather_indices,
    int                          kv_cache_layer,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>&  positions,
    int                          il,
    ggml_tensor*                 w_q,
    ggml_tensor*                 w_q_norm,
    ggml_tensor*                 w_k,
    ggml_tensor*                 w_k_norm,
    ggml_tensor*                 w_v,
    ggml_tensor*                 w_out,
    int                          n_embd_head,
    int                          n_head,
    int                          n_head_kv,
    int                          n_rot,
    float                        freq_base,
    int                          context_length,
    float                        rms_norm_eps)
{
    const size_t n_batch = slots.size();

    // A. Joint Q+Gate projection → [(n_embd_head*2)*n_head, n_batch]
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx, w_q, cur);

    // B. Extract Q via strided view → [n_embd_head, n_head, n_batch]
    ggml_tensor* Qcur = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);

    Qcur = build_rms_norm(ctx, Qcur, w_q_norm, rms_norm_eps, il);

    // C. K and V projections
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_batch);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_batch);

    Kcur = build_rms_norm(ctx, Kcur, w_k_norm, rms_norm_eps, il);

    // D. Extract Gate → [n_embd_head*n_head, n_batch]
    ggml_tensor* gate = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_batch);

    // E. Partial RoPE
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // F. Per-slot KV cache write
    const int n_embd_k = n_head_kv * n_embd_head;
    const int n_embd_v = n_head_kv * n_embd_head;

    ggml_tensor* k_storage_fmt = ggml_reshape_3d(ctx, Kcur, n_embd_k, 1, n_batch);
    ggml_tensor* v_storage_fmt = ggml_reshape_3d(ctx, Vcur, n_embd_v, 1, n_batch);

    for (size_t b = 0; b < n_batch; ++b) {
        ggml_tensor* k_slice = ggml_view_2d(ctx, k_storage_fmt,
            n_embd_k, 1, k_storage_fmt->nb[1], b * k_storage_fmt->nb[2]);
        ggml_tensor* v_slice = ggml_view_2d(ctx, v_storage_fmt,
            n_embd_v, 1, v_storage_fmt->nb[1], b * v_storage_fmt->nb[2]);

        ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx, k_slice, kv_cache_layer, slots[b]));
        ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx, v_slice, kv_cache_layer, slots[b]));
    }

    // G. Gather KV for all slots
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > (int32_t)max_pos) max_pos = (uint32_t)p;
    }
    uint32_t n_kv_len = max_pos + 1;

    ggml_tensor* k_gathered = kv_cache->gather_k(ctx, gf, kv_cache_layer,
        gather_indices, n_batch, n_kv_len);
    ggml_tensor* v_gathered = kv_cache->gather_v(ctx, gf, kv_cache_layer,
        gather_indices, n_batch, n_kv_len);

    ggml_tensor* k_view = ggml_view_4d(ctx, k_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_k    * sizeof(float),
        n_embd_k    * n_kv_len * sizeof(float), 0);
    ggml_tensor* v_view = ggml_view_4d(ctx, v_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_v    * sizeof(float),
        n_embd_v    * n_kv_len * sizeof(float), 0);

    // H. Attention
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il);

    // I. Sigmoid gating
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));

    // J. Output projection
    cur = ggml_mul_mat(ctx, w_out, cur);

    return cur;
}

// ── AttentionLayer ────────────────────────────────────────────────────────────

AttentionLayer::AttentionLayer(
    ggml_tensor* w_q, ggml_tensor* w_k, ggml_tensor* w_v, ggml_tensor* w_out,
    ggml_tensor* w_q_norm, ggml_tensor* w_k_norm,
    ggml_tensor* q_bias, ggml_tensor* k_bias, ggml_tensor* v_bias,
    simple_kv_cache* kv_cache,
    const Hparams& hp)
    : w_q_(w_q), w_k_(w_k), w_v_(w_v), w_out_(w_out),
      w_q_norm_(w_q_norm), w_k_norm_(w_k_norm),
      q_bias_(q_bias), k_bias_(k_bias), v_bias_(v_bias),
      kv_cache_(kv_cache), hp_(hp) {}

ggml_tensor* AttentionLayer::build(
    ggml_context*      ctx,
    ggml_cgraph*       gf,
    ggml_tensor*       input,
    int                layer_idx,
    Phase              phase,
    const PrefillArgs& prefill_args,
    const DecodeArgs*  decode_args)
{
    if (phase == Phase::Prefill)
        return build_prefill(ctx, gf, input, layer_idx, prefill_args);
    return build_decode(ctx, gf, input, layer_idx, *decode_args);
}

AttentionLayer::QKV AttentionLayer::project_qkv(
    ggml_context* ctx, ggml_cgraph* /*gf*/,
    ggml_tensor* input, ggml_tensor* inp_pos,
    int il, uint32_t n_tokens)
{
    const int n_embd = (int)input->ne[0];

    ggml_tensor* Qcur = ggml_mul_mat(ctx, w_q_, input);
    set_name(Qcur, "Qcur", il);
    if (hp_.has_bias) {
        Qcur = ggml_add(ctx, Qcur, q_bias_);
    }

    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k_, input);
    set_name(Kcur, "Kcur", il);
    if (hp_.has_bias) {
        Kcur = ggml_add(ctx, Kcur, k_bias_);
    }

    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v_, input);
    set_name(Vcur, "Vcur", il);
    if (hp_.has_bias) {
        Vcur = ggml_add(ctx, Vcur, v_bias_);
    }

    Qcur = ggml_reshape_3d(ctx, Qcur, hp_.n_embd_head, hp_.n_head,    n_tokens);
    Kcur = ggml_reshape_3d(ctx, Kcur, hp_.n_embd_head, hp_.n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, hp_.n_embd_head, hp_.n_head_kv, n_tokens);

    if (hp_.has_q_norm) {
        Qcur = ggml_rms_norm(ctx, Qcur, hp_.rms_norm_eps);
        Qcur = ggml_mul(ctx, Qcur, ggml_repeat(ctx, w_q_norm_,
                        ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                           hp_.n_embd_head, hp_.n_head, n_tokens)));
        set_name(Qcur, "Qcur_normed", il);

        Kcur = ggml_rms_norm(ctx, Kcur, hp_.rms_norm_eps);
        Kcur = ggml_mul(ctx, Kcur, ggml_repeat(ctx, w_k_norm_,
                        ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                           hp_.n_embd_head, hp_.n_head_kv, n_tokens)));
        set_name(Kcur, "Kcur_normed", il);
    }

    // RoPE.
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
                         hp_.n_rot, GGML_ROPE_TYPE_NEOX, hp_.context_len,
                         hp_.freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    set_name(Qcur, "Qcur_roped", il);

    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
                         hp_.n_rot, GGML_ROPE_TYPE_NEOX, hp_.context_len,
                         hp_.freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    set_name(Kcur, "Kcur_roped", il);

    (void)n_embd;
    return {Qcur, Kcur, Vcur};
}

ggml_tensor* AttentionLayer::build_prefill(
    ggml_context* ctx, ggml_cgraph* gf,
    ggml_tensor* input, int layer_idx,
    const PrefillArgs& args)
{
    QKV qkv = project_qkv(ctx, gf, input, args.inp_pos, layer_idx, args.n_tokens);

    ggml_tensor* attn_out = build_attention(
        ctx, gf, kv_cache_,
        qkv.q, qkv.k, qkv.v,
        layer_idx, hp_.kq_scale,
        args.n_tokens, args.slot_idx,
        layer_idx, hp_.n_embd_head, hp_.n_head_kv);
    set_name(attn_out, "attn_out", layer_idx);

    ggml_tensor* out = ggml_mul_mat(ctx, w_out_, attn_out);
    set_name(out, "attn_proj", layer_idx);
    return out;
}

ggml_tensor* AttentionLayer::build_decode(
    ggml_context* ctx, ggml_cgraph* gf,
    ggml_tensor* input, int layer_idx,
    const DecodeArgs& args)
{
    uint32_t n_batch = (uint32_t)input->ne[1];
    QKV qkv = project_qkv(ctx, gf, input, args.inp_pos, layer_idx, n_batch);

    ggml_tensor* attn_out = build_batched_attention(
        ctx, gf, kv_cache_,
        qkv.q, qkv.k, qkv.v,
        layer_idx, hp_.kq_scale,
        *args.slots, *args.positions,
        args.kq_mask, args.gather_indices,
        layer_idx);
    set_name(attn_out, "attn_out_dec", layer_idx);

    ggml_tensor* out = ggml_mul_mat(ctx, w_out_, attn_out);
    set_name(out, "attn_proj_dec", layer_idx);
    return out;
}
