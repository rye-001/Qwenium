#include "qwen35_layer.h"

#include <stdexcept>
#include <string>

#include "ggml.h"

#include "../engine/model.h"            // ModelMetadata, TransformerBlock
#include "../layers/attention.h"        // build_gated_{,batched_}attention
#include "../layers/deltanet.h"         // build_deltanet_layer, DeltaNetLayer
#include "../layers/ffn.h"              // build_ffn_swiglu
#include "../layers/norm.h"             // build_rms_norm
#include "../state/deltanet_state.h"
#include "../state/kv_cache_simple.h"

namespace {

void name_layer_tensor(ggml_tensor* t, const char* base, uint32_t il) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s.%u", base, il);
    ggml_set_name(t, buf);
}

// The FFN seam. moe_hp non-null ⇒ routed experts; null ⇒ dense SwiGLU.
// Checked against cfg->is_moe() so a mis-wired recipe fails loudly here rather
// than silently building the wrong feed-forward.
ggml_tensor* build_family_ffn(const Qwen35LayerCommon& c,
                              ggml_tensor* cur,
                              const TransformerBlock& blk,
                              Phase phase,
                              uint32_t il) {
    const bool want_moe = c.cfg->is_moe();
    if (want_moe != (c.moe_hp != nullptr))
        throw std::runtime_error(
            std::string("build_qwen35_layer: slot \"moe_hp\" expected ") +
            (want_moe ? "non-null (cfg.is_moe()==true)"
                      : "null (cfg.is_moe()==false)") +
            ", actual: " + (c.moe_hp ? "non-null" : "null") +
            " — layer " + std::to_string(il));

    if (!want_moe)
        return build_ffn_swiglu(c.ctx, c.gf, cur, blk.ffn_gate_weight,
                                blk.ffn_up_weight, blk.ffn_down_weight,
                                static_cast<int>(il));

    MoELayer moe(blk.moe_router_weight,
                 blk.moe_exp_gate_weight,
                 blk.moe_exp_up_weight,
                 blk.moe_exp_down_weight,
                 blk.moe_shexp_gate_w,
                 blk.moe_shexp_up_weight,
                 blk.moe_shexp_down_weight,
                 blk.moe_shexp_gate,
                 *c.moe_hp);
    return moe.build(c.ctx, c.gf, cur, phase, static_cast<int>(il));
}

// Rotated width: the declared partial-RoPE dimension when present, else the
// full head dimension. qwen35moe always declares it (asserted > 0 in the
// config factory), so this reduces to the declared value there.
int effective_n_rot(const Qwen35LayerCommon& c) {
    return c.cfg->rope_dimension_count > 0
        ? static_cast<int>(c.cfg->rope_dimension_count)
        : static_cast<int>(c.meta->attention_key_length);
}

DeltaNetLayer::Hparams deltanet_hparams(const Qwen35LayerCommon& c) {
    const Qwen35Config&  cfg = *c.cfg;
    const ModelMetadata& m   = *c.meta;
    return DeltaNetLayer::Hparams{
        static_cast<int>(m.embedding_length),
        static_cast<int>(cfg.ssm_inner_size),
        static_cast<int>(cfg.ssm_state_size),
        static_cast<int>(cfg.ssm_group_count),
        static_cast<int>(cfg.ssm_time_step_rank),
        static_cast<int>(cfg.ssm_inner_size / cfg.ssm_time_step_rank),
        static_cast<int>(cfg.ssm_inner_size +
                         2 * cfg.ssm_group_count * cfg.ssm_state_size),
        static_cast<int>(cfg.ssm_conv_kernel),
        m.rms_norm_eps
    };
}

}  // namespace

ggml_tensor* build_qwen35_layer_prefill(
    const Qwen35LayerCommon& c,
    ggml_tensor*             inpL,
    const TransformerBlock&  blk,
    ggml_tensor*             inp_pos,
    uint32_t                 n_tokens,
    uint32_t                 slot_idx,
    uint32_t                 dn_idx,
    int                      kv_idx,
    uint32_t                 il)
{
    const Qwen35Config&  cfg = *c.cfg;
    const ModelMetadata& m   = *c.meta;
    ggml_tensor* const inpSA = inpL;

    ggml_tensor* cur = build_rms_norm(c.ctx, inpL, blk.attn_norm_weight,
                                      m.rms_norm_eps, static_cast<int>(il));
    name_layer_tensor(cur, "attn_norm", il);

    if (cfg.is_ssm_layer(il)) {
        cur = build_deltanet_layer(
            c.ctx, c.gf, cur, c.dn_state, dn_idx, slot_idx, n_tokens,
            blk.attn_qkv_weight, blk.attn_gate_weight,
            blk.ssm_beta_weight, blk.ssm_alpha_weight,
            blk.ssm_dt_bias, blk.ssm_a, blk.ssm_conv1d_weight,
            blk.ssm_norm_weight, blk.ssm_out_weight,
            static_cast<int>(m.embedding_length),
            static_cast<int>(cfg.ssm_inner_size),
            static_cast<int>(cfg.ssm_state_size),
            static_cast<int>(cfg.ssm_group_count),
            static_cast<int>(cfg.ssm_time_step_rank),
            static_cast<int>(cfg.ssm_inner_size / cfg.ssm_time_step_rank),
            static_cast<int>(cfg.ssm_inner_size +
                             2 * cfg.ssm_group_count * cfg.ssm_state_size),
            static_cast<int>(cfg.ssm_conv_kernel),
            m.rms_norm_eps,
            static_cast<int>(il));
    } else {
        cur = build_gated_attention(
            c.ctx, c.gf, c.kv_cache, cur, inp_pos, kv_idx, n_tokens, slot_idx,
            static_cast<int>(il),
            blk.attn_q_weight, blk.attn_q_norm_weight,
            blk.attn_k_weight, blk.attn_k_norm_weight,
            blk.attn_v_weight, blk.attn_output_weight,
            static_cast<int>(m.attention_key_length),
            static_cast<int>(m.attention_head_count),
            static_cast<int>(m.attention_head_count_kv),
            effective_n_rot(c),
            m.rope_freq_base,
            static_cast<int>(m.context_length),
            m.rms_norm_eps,
            cfg.mrope_sections);
    }

    cur = ggml_add(c.ctx, cur, inpSA);
    name_layer_tensor(cur, "attn_residual", il);

    ggml_tensor* const ffn_residual = cur;
    cur = build_rms_norm(c.ctx, cur, blk.ffn_norm_weight, m.rms_norm_eps,
                         static_cast<int>(il));
    name_layer_tensor(cur, "post_attn_norm", il);

    cur = build_family_ffn(c, cur, blk, Phase::Prefill, il);
    name_layer_tensor(cur, "ffn_out", il);

    cur = ggml_add(c.ctx, cur, ffn_residual);
    name_layer_tensor(cur, "layer_out", il);
    return cur;
}

ggml_tensor* build_qwen35_layer_decode(
    const Qwen35LayerCommon&     c,
    ggml_tensor*                 inpL,
    const TransformerBlock&      blk,
    ggml_tensor*                 inp_pos,
    ggml_tensor*                 kq_mask,
    ggml_tensor*                 gather_indices,
    ggml_tensor*                 kv_write_idx,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>&  positions,
    uint32_t                     dn_idx,
    int                          kv_idx,
    uint32_t                     il)
{
    const Qwen35Config&  cfg = *c.cfg;
    const ModelMetadata& m   = *c.meta;
    ggml_tensor* const inpSA = inpL;

    ggml_tensor* cur = build_rms_norm(c.ctx, inpL, blk.attn_norm_weight,
                                      m.rms_norm_eps, static_cast<int>(il));

    if (cfg.is_ssm_layer(il)) {
        // One token per slot: the DeltaNet decode path takes the slot vector.
        DeltaNetLayer::PrefillArgs pa_unused{1, 0};
        DeltaNetLayer::DecodeArgs  da{slots};
        DeltaNetLayer dn_layer(
            blk.attn_qkv_weight, blk.attn_gate_weight,
            blk.ssm_beta_weight, blk.ssm_alpha_weight,
            blk.ssm_dt_bias, blk.ssm_a, blk.ssm_conv1d_weight,
            blk.ssm_norm_weight, blk.ssm_out_weight,
            c.dn_state,
            deltanet_hparams(c));
        cur = dn_layer.build(c.ctx, c.gf, cur, dn_idx, Phase::Decode,
                             pa_unused, &da);
    } else {
        cur = build_gated_batched_attention(
            c.ctx, c.gf, c.kv_cache, cur, inp_pos,
            kq_mask, gather_indices, kv_idx, slots, positions,
            static_cast<int>(il),
            blk.attn_q_weight, blk.attn_q_norm_weight,
            blk.attn_k_weight, blk.attn_k_norm_weight,
            blk.attn_v_weight, blk.attn_output_weight,
            static_cast<int>(m.attention_key_length),
            static_cast<int>(m.attention_head_count),
            static_cast<int>(m.attention_head_count_kv),
            effective_n_rot(c),
            m.rope_freq_base,
            static_cast<int>(m.context_length),
            m.rms_norm_eps,
            kv_write_idx,
            cfg.mrope_sections);
    }

    cur = ggml_add(c.ctx, cur, inpSA);

    ggml_tensor* const ffn_residual = cur;
    cur = build_rms_norm(c.ctx, cur, blk.ffn_norm_weight, m.rms_norm_eps,
                         static_cast<int>(il));
    cur = build_family_ffn(c, cur, blk, Phase::Decode, il);
    cur = ggml_add(c.ctx, cur, ffn_residual);
    return cur;
}
