#include "attention.h"
#include <stdexcept>
#include <string>
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

// ── build_softcap ────────────────────────────────────────────────────────────
// Gemma 2 logit soft-capping: cap * tanh(x / cap).
// Applied on raw attention QK scores (before scaled softmax) and on final
// logits.  cap must be > 0; callers gate the call on that condition.
ggml_tensor* build_softcap(ggml_context* ctx, ggml_tensor* x, float cap)
{
    x = ggml_scale(ctx, x, 1.0f / cap);
    x = ggml_tanh(ctx, x);
    x = ggml_scale(ctx, x, cap);
    return x;
}

// ── build_rope_pruned ────────────────────────────────────────────────────────
// p-RoPE (Gemma 4 global layers): rotate only the first n_rot dimensions of
// each head; the remaining (head_dim - n_rot) dimensions pass through unchanged.
//
// ggml_rope_ext with `n_dims = n_rot` already implements this exact behavior:
// the kernel rotates dims [0, n_rot) and copies dims [n_rot, head_dim) verbatim.
// We expose it as a named function so recipes that select RopeKind::Pruned can
// document the intent explicitly at the call site, and so the unit test has a
// stable surface to oracle-check against.
ggml_tensor* build_rope_pruned(
    ggml_context* ctx,
    ggml_tensor*  x,
    ggml_tensor*  inp_pos,
    int           n_rot,
    int           context_len,
    float         freq_base)
{
    // Argument order matches the existing call in transformer_block.cpp:
    //   freq_base, freq_scale=1, ext_factor=0, attn_factor=1, beta_fast=32, beta_slow=1
    return ggml_rope_ext(ctx, x, inp_pos, /*freq_factors=*/nullptr,
                         n_rot, GGML_ROPE_TYPE_NEOX, context_len,
                         freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

// ── MRopeSections::from_widths ───────────────────────────────────────────────
MRopeSections MRopeSections::from_widths(const std::vector<int32_t>& widths,
                                         const char* key, int n_rot)
{
    if (widths.size() != 4)
        throw std::runtime_error(
            std::string("MRopeSections: key '") + key +
            "' expected 4 section widths, actual: " +
            std::to_string(widths.size()));

    int64_t sum = 0;
    for (int32_t w : widths) {
        if (w < 0)
            throw std::runtime_error(
                std::string("MRopeSections: key '") + key +
                "' expected non-negative section widths, actual: " +
                std::to_string(w));
        sum += w;
    }

    if (sum != n_rot / 2)
        throw std::runtime_error(
            std::string("MRopeSections: key '") + key +
            "' expected widths summing to n_rot/2 (" +
            std::to_string(n_rot / 2) + "), actual: " + std::to_string(sum));

    if (widths[0] <= 0 && widths[1] <= 0 && widths[2] <= 0)
        throw std::runtime_error(
            std::string("MRopeSections: key '") + key +
            "' expected at least one of the first three widths > 0 "
            "(ggml_rope_multi asserts this), actual: all zero");

    MRopeSections m;
    for (int i = 0; i < 4; ++i) m.widths[i] = widths[i];
    m.active = true;
    return m;
}

// ── build_rope_gated ─────────────────────────────────────────────────────────
// The RoPE used by the gated (Qwen 3.5-family) attention paths.
//
// Two kernels, one operation. ggml applies the SAME rotation layout to MROPE
// as to NEOX — rotate_pairs(n_dims, n_dims/2) in both cases — and differs only
// in where each dimension's theta comes from: NEOX uses the single position,
// MROPE picks one of four position components per section. When the four
// components are equal (every text-only step) ggml_mrope_cache_init walks all
// four thetas in lockstep, so whichever section is selected yields the same
// theta NEOX would have produced, and the results coincide exactly.
//
// That is why P2 can switch the Qwen 3.5 family onto ggml_rope_multi and still
// gate on byte-identical text output.
static ggml_tensor* build_rope_gated(
    ggml_context*        ctx,
    ggml_tensor*         x,
    ggml_tensor*         inp_pos,
    int                  n_rot,
    int                  context_length,
    float                freq_base,
    const MRopeSections& mrope)
{
    if (!mrope.active) {
        return ggml_rope_ext(ctx, x, inp_pos, /*freq_factors=*/nullptr,
                             n_rot, GGML_ROPE_TYPE_NEOX,
                             context_length, freq_base,
                             1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }
    // ggml takes a mutable int[4]; MRopeSections holds int32_t. Copy rather
    // than cast so the widths cannot be modified through our config.
    int sections[GGML_MROPE_SECTIONS] = {
        static_cast<int>(mrope.widths[0]), static_cast<int>(mrope.widths[1]),
        static_cast<int>(mrope.widths[2]), static_cast<int>(mrope.widths[3]),
    };
    // Every scaling parameter is identical to the NEOX branch above; the ONLY
    // difference is the kernel and the shape of inp_pos.
    return ggml_rope_multi(ctx, x, inp_pos, /*freq_factors=*/nullptr,
                           n_rot, sections, GGML_ROPE_TYPE_MROPE,
                           context_length, freq_base,
                           1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

// ── build_attn_mha ───────────────────────────────────────────────────────────
// Extracted from ForwardPassBase::build_attn_mha (src/models/forward_pass_base.cpp).
// Logic is identical — only ctx_ → ctx parameter.
// softcap: when > 0, applies cap * tanh(kq / cap) before the scaled softmax.
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
    int           il,
    float         softcap,
    bool          use_flash)
{
    (void)gf; (void)pos; // gf/pos unused directly; kept for API symmetry with callers

    const auto n_stream = k->ne[3];

    q = ggml_reshape_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream);
    set_name(q, "q_reshaped", il);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    set_name(q, "q_permuted", il);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    set_name(k, "k_permuted", il);

    // ── Flash attention (opt-in, --flash-attn) ───────────────────────────────
    // One ggml_flash_attn_ext replaces kq -> soft_max -> kqv AND the V
    // transpose: the op consumes V in the same [d, n_kv, n_head_kv] layout as
    // K, so the ggml_cont below is not needed here. Four dispatches per
    // attention layer become one.
    //
    // NOT byte-identical to the materialized path (the softmax is reduced in
    // registers, in a different order), and kq_soft never exists — which is
    // why DecodePolicy pairs this with an empty attention_taps set and the
    // front ends refuse --flash-attn together with --attention-lens.
    if (use_flash) {
        // ggml_flash_attn_ext hard-asserts an F16 mask (ggml.c). The recipe
        // casts once per graph rather than once per layer; a recipe that
        // forgets gets this message instead of an abort inside ggml.
        if (kq_mask->type != GGML_TYPE_F16) {
            throw std::runtime_error(
                std::string("build_attn_mha: layer ") + std::to_string(il) +
                " slot 'kq_mask': flash attention expected type F16, got: " +
                ggml_type_name(kq_mask->type));
        }
        // Softcap forwards directly, because ggml implements OUR convention.
        // Verified 2026-08-30 in both backends rather than assumed: the host
        // pre-divides (`scale /= logit_softcap`) and the kernel then computes
        // `s*scale` followed by `logit_softcap*tanh(s)` — i.e.
        // cap * tanh(QK·scale / cap), the scale applied BEFORE the clamp, and
        // the mask added after. That is exactly build_softcap composed after
        // ggml_scale(kq, kq_scale) on the materialized path below, and matches
        // HF Gemma 2. (This was refused until the convention was checked; the
        // worry was that llama applies the clamp to the raw QK product, which
        // it can do because it folds the scale into Q beforehand.)
        ggml_tensor* v_flash = ggml_permute(ctx, v, 0, 2, 1, 3);
        set_name(v_flash, "v_permuted", il);

        ggml_tensor* cur = ggml_flash_attn_ext(ctx, q, k, v_flash, kq_mask,
                                               kq_scale, 0.0f /* max_bias */,
                                               softcap);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        if (sinks) ggml_flash_attn_ext_add_sinks(cur, sinks);
        set_name(cur, "kqv_flash", il);

        // FA already emits [d, n_head, n_q, n_stream] — the layout the
        // materialized path reaches only after its kqv permute — so this is
        // the same recombination, one step shorter.
        cur = ggml_reshape_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        set_name(cur, "attn_recombined", il);
        return cur;
    }

    v = ggml_permute(ctx, v, 1, 2, 0, 3);
    set_name(v, "v_permuted", il);
    v = ggml_cont(ctx, v);
    set_name(v, "v_cont", il);

    ggml_tensor* cur;
    {
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        set_name(kq, "kq", il);

        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        // Gemma 2 attention logit soft-capping (softcap == 0 → off).
        // The scale (1/√d) must be applied BEFORE the tanh clamp:
        //   cap · tanh( QKᵀ/√d  /  cap )
        // Passing kq_scale to soft_max_ext instead would multiply after the
        // tanh, compressing the logits into ±(cap·kq_scale) and making
        // attention near-uniform.
        if (softcap > 0.0f) {
            kq = ggml_scale(ctx, kq, kq_scale);
            set_name(kq, "kq_scaled", il);
            kq = build_softcap(ctx, kq, softcap);
            set_name(kq, "kq_softcapped", il);
            kq = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f, 0);
        } else {
            kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0);
        }
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
// G4.1: n_embd_head split into head_dim_k + head_dim_v so Gemma 4 can use
// different K/V head dims than Q head dim.  All existing callers pass the
// same value for head_dim_k and head_dim_v — behavior is bit-identical.
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
    int              head_dim_k,
    int              head_dim_v,
    int              n_head_kv,
    float            softcap,
    bool             use_flash)
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
    const int n_embd_k = n_head_kv * head_dim_k;
    const int n_embd_v = n_head_kv * head_dim_v;

    k = ggml_view_3d(ctx, k_full,
        head_dim_k,
        n_head_kv,
        n_kv,
        ggml_row_size(k_full->type, head_dim_k),
        ggml_row_size(k_full->type, n_embd_k),
        0);

    v = ggml_view_3d(ctx, v_full,
        head_dim_v,
        n_head_kv,
        n_kv,
        ggml_row_size(v_full->type, head_dim_v),
        ggml_row_size(v_full->type, n_embd_v),
        0);

    set_name(k, "k_view", il);
    set_name(v, "v_view", il);

    // 4. Build the causal mask for the full sequence
    ggml_tensor* kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    set_name(kq_mask, "kq_mask", il);
    ggml_build_forward_expand(gf, kq_mask);

    // Flash attention needs an F16 mask. Unlike the decode path — where the
    // recipe owns one mask for the whole graph and casts it once — the prefill
    // mask is built HERE, per layer, at [n_kv x n_tokens]. So the cast is per
    // layer too. It is O(n^2) bytes, but so is the kq it replaces, and flash
    // never materializes that: see docs/decode-gap-status.md §18 for the
    // measurement rather than the argument.
    ggml_tensor* mha_mask = kq_mask;
    if (use_flash) {
        mha_mask = ggml_cast(ctx, kq_mask, GGML_TYPE_F16);
        set_name(mha_mask, "kq_mask_f16", il);
    }

    // 5. Run MHA
    return build_attn_mha(ctx, gf, q, k, v, mha_mask, nullptr, kq_scale, pos, il, softcap, use_flash);
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
    int                          il,
    float                        softcap,
    ggml_tensor*                 kv_write_indices,
    bool                         use_flash)
{
    const size_t n_batch    = slots.size();
    const int    n_embd_head = k->ne[0];
    const int    n_head_kv   = k->ne[1];
    const int    n_embd_k    = n_embd_head * n_head_kv;
    const int    n_embd_v    = n_embd_head * n_head_kv;

    // KV read width follows the mask the recipe built (single source of truth —
    // re-deriving it from positions would silently diverge when the recipe
    // buckets its decode n_kv, plan-persistent-decode-graph.md §2.2; padded
    // columns are −inf-masked and read zero-initialized rows).
    uint32_t n_kv_len = static_cast<uint32_t>(kq_mask->ne[0]);

    // 1. Write new K/V per slot, then 2. gather for attention. In the set_rows
    // path the gather reads THROUGH the write view, so the graph itself orders
    // the read after the write; the cpy path keeps its separate
    // write-then-gather-from-raw-cache shape (byte-identical).
    ggml_tensor* k_gathered;
    ggml_tensor* v_gathered;
    if (kv_write_indices) {
        // Value-driven write: destination rows are graph-input values, so the
        // node survives graph reuse (persistent decode graph). One set_rows
        // per cache replaces the n_batch cpy nodes.
        ggml_tensor* k_rows = ggml_reshape_2d(ctx, k, n_embd_k, n_batch);
        ggml_tensor* v_rows = ggml_reshape_2d(ctx, v, n_embd_v, n_batch);

        ggml_tensor* k_stored = kv_cache->set_rows_k(ctx, k_rows, layer_idx, kv_write_indices);
        set_name(k_stored, "k_stored_b", il);
        ggml_build_forward_expand(gf, k_stored);

        ggml_tensor* v_stored = kv_cache->set_rows_v(ctx, v_rows, layer_idx, kv_write_indices);
        set_name(v_stored, "v_stored_b", il);
        ggml_build_forward_expand(gf, v_stored);

        if (n_batch == 1) {
            // Identity gather on the set_rows path too — see gather_k_single.
            // The read is a plain view of the cache, so it carries no data edge
            // to the set_rows write above; ordering rests on the write being
            // expanded first PLUS the Metal backend's memory-range analysis,
            // which sees the view and the write aliasing the same cache bytes.
            // Both of the backend's passes honour that: the encode-time
            // concurrency check inserts a barrier for overlapping ranges, and
            // the reorder pass refuses to hoist a node past unprocessed nodes it
            // overlaps (SET_ROWS is not in its reorderable set at all). This is
            // exactly what llama.cpp does — its get_k is a bare ggml_view_4d of
            // the cache next to a set_rows write. Gated by the ordering test in
            // test_kv_write_setrows.cpp, which runs the persistent shape with
            // graph-optimize both enabled and disabled.
            ggml_build_forward_expand(gf, gather_indices);
            k_gathered = kv_cache->gather_k_single(ctx, layer_idx, slots[0], n_kv_len);
            v_gathered = kv_cache->gather_v_single(ctx, layer_idx, slots[0], n_kv_len);
        } else {
            k_gathered = kv_cache->gather_k_from(ctx, k_stored, gather_indices, n_batch, n_kv_len);
            v_gathered = kv_cache->gather_v_from(ctx, v_stored, gather_indices, n_batch, n_kv_len);
        }
    } else {
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

        if (n_batch == 1) {
            // Identity gather: one active slot means the index run is one
            // contiguous span, so the read is a free view of the cache instead
            // of two materializing GET_ROWS (docs/decode-gap-status.md §4).
            // Values, strides and layout are unchanged, so the consuming
            // mul_mat is bit-identical to the gathered path.
            //
            // gather_indices is no longer consumed here, so expand it to keep
            // it in the graph: GatherIndicesInput still owns the slot and its
            // require_tensor lookup is fail-loud on an absent tensor. That
            // costs an n_kv-int32 upload per step, immaterial against the
            // 2 x n_kv x n_embd copy it removes.
            ggml_build_forward_expand(gf, gather_indices);
            k_gathered = kv_cache->gather_k_single(ctx, layer_idx, slots[0], n_kv_len);
            v_gathered = kv_cache->gather_v_single(ctx, layer_idx, slots[0], n_kv_len);
        } else {
            k_gathered = kv_cache->gather_k(ctx, gf, layer_idx, gather_indices, n_batch, n_kv_len);
            v_gathered = kv_cache->gather_v(ctx, gf, layer_idx, gather_indices, n_batch, n_kv_len);
        }
    }

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
    return build_attn_mha(ctx, gf, q, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il, softcap, use_flash);
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
    float            rms_norm_eps,
    const MRopeSections& mrope,
    bool             use_flash)
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

    // E. Partial RoPE (M-RoPE when the recipe declares sections)
    Qcur = build_rope_gated(ctx, Qcur, inp_pos, n_rot, context_length, freq_base, mrope);
    Kcur = build_rope_gated(ctx, Kcur, inp_pos, n_rot, context_length, freq_base, mrope);

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
        ggml_row_size(k_full->type, n_embd_head), ggml_row_size(k_full->type, n_embd_kv), 0);
    ggml_tensor* v_view = ggml_view_3d(ctx, v_full,
        n_embd_head, n_head_kv, n_kv,
        ggml_row_size(v_full->type, n_embd_head), ggml_row_size(v_full->type, n_embd_kv), 0);

    ggml_tensor* kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    set_name(kq_mask, "kq_mask", il);
    ggml_build_forward_expand(gf, kq_mask);

    // Flash attention needs an F16 mask. Unlike the decode path — where the
    // recipe owns one mask for the whole graph and casts it once — the prefill
    // mask is built HERE, per layer, at [n_kv x n_tokens]. So the cast is per
    // layer too. It is O(n^2) bytes, but so is the kq it replaces, and flash
    // never materializes that: see docs/decode-gap-status.md §18 for the
    // measurement rather than the argument.
    ggml_tensor* mha_mask = kq_mask;
    if (use_flash) {
        mha_mask = ggml_cast(ctx, kq_mask, GGML_TYPE_F16);
        set_name(mha_mask, "kq_mask_f16", il);
    }

    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, mha_mask, nullptr, kq_scale, cache_pos, il, 0.0f, use_flash);

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
    float                        rms_norm_eps,
    ggml_tensor*                 kv_write_indices,
    const MRopeSections&         mrope,
    bool                         use_flash)
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

    // E. Partial RoPE (M-RoPE when the recipe declares sections)
    Qcur = build_rope_gated(ctx, Qcur, inp_pos, n_rot, context_length, freq_base, mrope);
    Kcur = build_rope_gated(ctx, Kcur, inp_pos, n_rot, context_length, freq_base, mrope);

    // F. Per-slot KV cache write + G. gather. Read width follows the recipe-
    // built mask (keeps bucketed decode n_kv coherent). In the set_rows path
    // the gather reads THROUGH the write view, so the graph itself orders the
    // read after the write; the cpy path keeps its
    // write-then-gather-from-raw-cache shape.
    const int n_embd_k = n_head_kv * n_embd_head;
    const int n_embd_v = n_head_kv * n_embd_head;
    uint32_t n_kv_len = static_cast<uint32_t>(kq_mask->ne[0]);

    ggml_tensor* k_gathered;
    ggml_tensor* v_gathered;
    if (kv_write_indices) {
        // Value-driven write (see build_batched_attention): one set_rows per
        // cache, destination rows supplied per step — survives graph reuse.
        ggml_tensor* k_rows = ggml_reshape_2d(ctx, Kcur, n_embd_k, n_batch);
        ggml_tensor* v_rows = ggml_reshape_2d(ctx, Vcur, n_embd_v, n_batch);
        ggml_tensor* k_stored = kv_cache->set_rows_k(ctx, k_rows, kv_cache_layer, kv_write_indices);
        ggml_tensor* v_stored = kv_cache->set_rows_v(ctx, v_rows, kv_cache_layer, kv_write_indices);
        ggml_build_forward_expand(gf, k_stored);
        ggml_build_forward_expand(gf, v_stored);
        if (n_batch == 1) {
            // Identity gather on the set_rows path too — see gather_k_single.
            // The read is a plain view of the cache, so it carries no data edge
            // to the set_rows write above; ordering rests on the write being
            // expanded first PLUS the Metal backend's memory-range analysis,
            // which sees the view and the write aliasing the same cache bytes.
            // Both of the backend's passes honour that: the encode-time
            // concurrency check inserts a barrier for overlapping ranges, and
            // the reorder pass refuses to hoist a node past unprocessed nodes it
            // overlaps (SET_ROWS is not in its reorderable set at all). This is
            // exactly what llama.cpp does — its get_k is a bare ggml_view_4d of
            // the cache next to a set_rows write. Gated by the ordering test in
            // test_kv_write_setrows.cpp, which runs the persistent shape with
            // graph-optimize both enabled and disabled.
            ggml_build_forward_expand(gf, gather_indices);
            k_gathered = kv_cache->gather_k_single(ctx, kv_cache_layer, slots[0], n_kv_len);
            v_gathered = kv_cache->gather_v_single(ctx, kv_cache_layer, slots[0], n_kv_len);
        } else {
            k_gathered = kv_cache->gather_k_from(ctx, k_stored, gather_indices, n_batch, n_kv_len);
            v_gathered = kv_cache->gather_v_from(ctx, v_stored, gather_indices, n_batch, n_kv_len);
        }
    } else {
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

        if (n_batch == 1) {
            // Identity gather: one active slot means the index run is one
            // contiguous span, so the read is a free view of the cache instead
            // of two materializing GET_ROWS (docs/decode-gap-status.md §4).
            // Values, strides and layout are unchanged, so the consuming
            // mul_mat is bit-identical to the gathered path.
            //
            // gather_indices is no longer consumed here, so expand it to keep
            // it in the graph: GatherIndicesInput still owns the slot and its
            // require_tensor lookup is fail-loud on an absent tensor. That
            // costs an n_kv-int32 upload per step, immaterial against the
            // 2 x n_kv x n_embd copy it removes.
            ggml_build_forward_expand(gf, gather_indices);
            k_gathered = kv_cache->gather_k_single(ctx, kv_cache_layer, slots[0], n_kv_len);
            v_gathered = kv_cache->gather_v_single(ctx, kv_cache_layer, slots[0], n_kv_len);
        } else {
            k_gathered = kv_cache->gather_k(ctx, gf, kv_cache_layer, gather_indices, n_batch, n_kv_len);
            v_gathered = kv_cache->gather_v(ctx, gf, kv_cache_layer, gather_indices, n_batch, n_kv_len);
        }
    }

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
    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il, 0.0f, use_flash);

    // I. Sigmoid gating
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));

    // J. Output projection
    cur = ggml_mul_mat(ctx, w_out, cur);

    return cur;
}
