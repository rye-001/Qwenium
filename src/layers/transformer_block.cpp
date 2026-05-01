#include "transformer_block.h"
#include "attention.h"
#include "ffn.h"
#include "norm.h"

#include "ggml.h"
#include <cmath>
#include <cstdio>

// Names a tensor with a layer-index suffix, matching ForwardPassBase::set_tensor_name.
static void tblk_name(ggml_tensor* t, const char* base, uint32_t il) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s.%d", base, static_cast<int>(il));
    ggml_set_name(t, buf);
}

ggml_tensor* build_transformer_layer(
    ggml_context*                  ctx,
    ggml_cgraph*                   gf,
    simple_kv_cache*               kv_cache,
    ggml_tensor*                   cur,
    ggml_tensor*                   inp_pos,
    const TransformerBlockWeights& w,
    const TransformerBlockHparams& hp,
    uint32_t                       il,
    uint32_t                       slot_idx,
    uint32_t                       n_tokens,
    ggml_tensor*                   ple_residual)
{
    // G4.2: PLE second-residual injection.
    // Injection point: block entry, before the pre-attention norm.  The
    // injected tensor becomes part of the residual stream — subsequent
    // norms, attention, and FFN all see the post-injection value, and the
    // final residual reuses it via inpSA below.
    // ple_residual == nullptr → degenerate to the Qwen / Gemma 1/2/3 path
    //   (no ggml_add node emitted; bit-identical to before this PR).
    // ple_residual all-zeros  → identity (ggml_add of zeros): the path is
    //   bit-identical at the value level, by ggml's add semantics.
    if (ple_residual != nullptr) {
        cur = ggml_add(ctx, cur, ple_residual);
        tblk_name(cur, "cur_with_ple", il);
    }

    ggml_tensor* inpSA = cur;

    // Resolve per-layer shape overrides (G4.1).
    // Zero-defaults degrade to the existing uniform-shape behavior for all
    // recipes that do not set these fields (Qwen, Gemma 1/2/3).
    const int head_dim_k = hp.head_dim_k ? hp.head_dim_k : hp.n_embd_head;
    const int head_dim_v = hp.head_dim_v ? hp.head_dim_v : hp.n_embd_head;
    const int n_rot      = hp.rope_dim   ? hp.rope_dim   : hp.n_embd_head;

    // Pick the norm builder once. Gemma's (1+w) form is the same shape but
    // a different scaling — selecting it here keeps the rest of the block
    // identical between families.
    auto rms = [&](ggml_tensor* t, ggml_tensor* weight, int idx) {
        return hp.gemma_rms_norm
                   ? build_rms_norm_gemma(ctx, t, weight, hp.rms_norm_eps, idx)
                   : build_rms_norm(ctx, t, weight, hp.rms_norm_eps, idx);
    };

    // A. Pre-attention RMS norm
    cur = rms(cur, w.attn_norm, static_cast<int>(il));
    tblk_name(cur, "cur_normed", il);

    // B. QKV projections
    ggml_tensor* Qcur = ggml_mul_mat(ctx, w.q, cur);
    tblk_name(Qcur, "Qcur", il);
    if (hp.is_qwen2) {
        Qcur = ggml_add(ctx, Qcur, w.q_bias);
        tblk_name(Qcur, "Qcur_biased", il);
    }

    ggml_tensor* Kcur = ggml_mul_mat(ctx, w.k, cur);
    tblk_name(Kcur, "Kcur", il);
    if (hp.is_qwen2) {
        Kcur = ggml_add(ctx, Kcur, w.k_bias);
        tblk_name(Kcur, "Kcur_biased", il);
    }

    ggml_tensor* Vcur = ggml_mul_mat(ctx, w.v, cur);
    tblk_name(Vcur, "Vcur", il);
    if (hp.is_qwen2) {
        Vcur = ggml_add(ctx, Vcur, w.v_bias);
        tblk_name(Vcur, "Vcur_biased", il);
    }

    // Q is always shaped by n_embd_head (Q head dim).
    // K and V use head_dim_k/v which may differ from n_embd_head when
    // head_dim_k/v are explicitly set (e.g. Gemma 4 global layers).
    Qcur = ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    n_tokens);
    tblk_name(Qcur, "Qcur_3d", il);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim_k, hp.n_head_kv, n_tokens);
    tblk_name(Kcur, "Kcur_3d", il);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim_v, hp.n_head_kv, n_tokens);
    tblk_name(Vcur, "Vcur_3d", il);

    // C. Optional QK RMS norms (qwen3 only)
    if (!hp.is_qwen2 && w.q_norm != nullptr) {
        // Gemma 1 has no QK norm (q_norm is null); Qwen 3 has it.
        Qcur = build_rms_norm(ctx, Qcur, w.q_norm, hp.rms_norm_eps, static_cast<int>(il));
        tblk_name(Qcur, "Qcur_normed", il);
    }

    // D. RoPE — n_rot may be < head_dim_k (p-RoPE lands in G4.5).
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX, hp.context_length,
                          hp.freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    tblk_name(Qcur, "Qcur_roped", il);

    if (!hp.is_qwen2 && w.k_norm != nullptr) {
        Kcur = build_rms_norm(ctx, Kcur, w.k_norm, hp.rms_norm_eps, static_cast<int>(il));
        tblk_name(Kcur, "Kcur_normed", il);
    }
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX, hp.context_length,
                          hp.freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    tblk_name(Kcur, "Kcur_roped", il);

    // E. Grouped-Query Attention
    // kq_scale uses Q head dim (hp.n_embd_head) — correct regardless of K/V head dims.
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(hp.n_embd_head));
    cur = build_attention(ctx, gf, kv_cache, Qcur, Kcur, Vcur,
                          static_cast<int>(il), kq_scale, n_tokens, slot_idx,
                          static_cast<int>(il),
                          head_dim_k, head_dim_v, hp.n_head_kv,
                          hp.attn_softcap);

    // F. Output projection
    cur = ggml_mul_mat(ctx, w.out, cur);
    tblk_name(cur, "attn_output", il);

    // F2. Post-attention norm (sandwich norm — Gemma 2).
    // nullptr = Qwen / Gemma-1 path; behavior is bit-identical when absent.
    if (w.post_attn_norm != nullptr) {
        cur = rms(cur, w.post_attn_norm, static_cast<int>(il));
        tblk_name(cur, "post_attn_normed", il);
    }

    // G. Post-attention residual
    ggml_tensor* ffn_inp = ggml_add(ctx, cur, inpSA);
    tblk_name(ffn_inp, "ffn_inp", il);

    // H. Pre-FFN RMS norm
    cur = rms(ffn_inp, w.ffn_norm, static_cast<int>(il));
    tblk_name(cur, "ffn_norm", il);

    // I. FFN: SwiGLU (default) or GeGLU-tanh (Gemma)
    if (hp.gemma_geglu) {
        cur = build_ffn_geglu_tanh(ctx, gf, cur, w.ffn_gate, w.ffn_up, w.ffn_down,
                                    static_cast<int>(il));
    } else {
        cur = build_ffn_swiglu(ctx, gf, cur, w.ffn_gate, w.ffn_up, w.ffn_down,
                                static_cast<int>(il));
    }
    tblk_name(cur, "ffn_output", il);

    // I2. Post-FFN norm (sandwich norm — Gemma 2).
    if (w.post_ffn_norm != nullptr) {
        cur = rms(cur, w.post_ffn_norm, static_cast<int>(il));
        tblk_name(cur, "post_ffn_normed", il);
    }

    // J. Post-FFN residual
    cur = ggml_add(ctx, cur, ffn_inp);
    tblk_name(cur, "cur_ffn_inp", il);

    return cur;
}
