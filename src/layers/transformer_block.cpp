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
    uint32_t                       n_tokens)
{
    ggml_tensor* inpSA = cur;

    // A. Pre-attention RMS norm
    cur = build_rms_norm(ctx, cur, w.attn_norm, hp.rms_norm_eps, static_cast<int>(il));
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

    Qcur = ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    n_tokens);
    tblk_name(Qcur, "Qcur_3d", il);
    Kcur = ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, n_tokens);
    tblk_name(Kcur, "Kcur_3d", il);
    Vcur = ggml_reshape_3d(ctx, Vcur, hp.n_embd_head, hp.n_head_kv, n_tokens);
    tblk_name(Vcur, "Vcur_3d", il);

    // C. Optional QK RMS norms (qwen3 only)
    if (!hp.is_qwen2) {
        Qcur = build_rms_norm(ctx, Qcur, w.q_norm, hp.rms_norm_eps, static_cast<int>(il));
        tblk_name(Qcur, "Qcur_normed", il);
    }

    // D. RoPE
    const int n_rot = hp.n_embd_head;
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX, hp.context_length,
                          hp.freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    tblk_name(Qcur, "Qcur_roped", il);

    if (!hp.is_qwen2) {
        Kcur = build_rms_norm(ctx, Kcur, w.k_norm, hp.rms_norm_eps, static_cast<int>(il));
        tblk_name(Kcur, "Kcur_normed", il);
    }
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX, hp.context_length,
                          hp.freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    tblk_name(Kcur, "Kcur_roped", il);

    // E. Grouped-Query Attention
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(hp.n_embd_head));
    cur = build_attention(ctx, gf, kv_cache, Qcur, Kcur, Vcur,
                          static_cast<int>(il), kq_scale, n_tokens, slot_idx,
                          static_cast<int>(il), hp.n_embd_head, hp.n_head_kv);

    // F. Output projection
    cur = ggml_mul_mat(ctx, w.out, cur);
    tblk_name(cur, "attn_output", il);

    // G. Post-attention residual
    ggml_tensor* ffn_inp = ggml_add(ctx, cur, inpSA);
    tblk_name(ffn_inp, "ffn_inp", il);

    // H. Pre-FFN RMS norm
    cur = build_rms_norm(ctx, ffn_inp, w.ffn_norm, hp.rms_norm_eps, static_cast<int>(il));
    tblk_name(cur, "ffn_norm", il);

    // I. SwiGLU FFN
    cur = build_ffn_swiglu(ctx, gf, cur, w.ffn_gate, w.ffn_up, w.ffn_down,
                            static_cast<int>(il));
    tblk_name(cur, "ffn_output", il);

    // J. Post-FFN residual
    cur = ggml_add(ctx, cur, ffn_inp);
    tblk_name(cur, "cur_ffn_inp", il);

    return cur;
}
