#include "qwen35.h"
#include "../layers/attention.h"
#include "../layers/deltanet.h"
#include "../layers/ffn.h"
#include "../qinf_error.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/gather_indices_input.h"
#include "../graph_inputs/kv_write_indices_input.h"

#include <memory>

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>

// ── Qwen35Config::from_metadata ──────────────────────────────────────────────

Qwen35Config Qwen35Config::from_metadata(const ModelMetadata& meta) {
    const uint32_t ssm_state_size          = meta.raw_kv.get_uint32("qwen35.ssm.state_size");
    const uint32_t ssm_inner_size          = meta.raw_kv.get_uint32("qwen35.ssm.inner_size");
    const uint32_t ssm_time_step_rank      = meta.raw_kv.get_uint32("qwen35.ssm.time_step_rank");
    const uint32_t ssm_group_count         = meta.raw_kv.get_uint32("qwen35.ssm.group_count");
    const uint32_t ssm_conv_kernel         = meta.raw_kv.get_uint32("qwen35.ssm.conv_kernel");
    const uint32_t full_attention_interval = meta.raw_kv.get_uint32("qwen35.full_attention_interval");
    const uint32_t rope_dimension_count    = meta.raw_kv.get_uint32_opt("qwen35.rope.dimension_count").value_or(0u);
    // M-RoPE sections (P2). Effective n_rot mirrors the use sites below:
    // rope_dimension_count when declared, else the full head dimension.
    MRopeSections mrope_sections;
    {
        static constexpr const char* kSectionsKey = "qwen35.rope.dimension_sections";
        if (auto widths = meta.raw_kv.get_int32_array_opt(kSectionsKey)) {
            const int n_rot = rope_dimension_count > 0
                ? static_cast<int>(rope_dimension_count)
                : static_cast<int>(meta.attention_key_length);
            mrope_sections = MRopeSections::from_widths(*widths, kSectionsKey, n_rot);
        }
    }
    // Optional: absent on standard GGUFs ⇒ 0 ⇒ no MTP head, n_main == block_count.
    const uint32_t nextn_predict_layers    = meta.nextn_predict_layers();

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
    QINF_ASSERT(nextn_predict_layers < meta.block_count,
        "Qwen35Config: field \"nextn_predict_layers\" expected < block_count (" +
        std::to_string(meta.block_count) + "), got " +
        std::to_string(nextn_predict_layers) +
        " (GGUF key: qwen35.nextn_predict_layers)");

    return Qwen35Config{
        ssm_conv_kernel,
        ssm_state_size,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_inner_size,
        rope_dimension_count,
        mrope_sections,
        full_attention_interval,
        nextn_predict_layers,
    };
}

// ============================================================
// Constructor: hybrid cache setup
// ============================================================

Qwen35ForwardPass::Qwen35ForwardPass(
    const Model& model, const ModelMetadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, ggml_type kv_type)
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

    // Trailing NextN / MTP head blocks are loaded but are not part of the
    // residual stream; the decode stack is n_main_layers_ deep.
    n_main_layers_ = meta_.block_count - cfg_.nextn_predict_layers;

    kv_layer_map_.resize(n_main_layers_, -1);
    ssm_layer_map_.resize(n_main_layers_, -1);

    for (uint32_t il = 0; il < n_main_layers_; ++il) {
        if (cfg_.is_full_attention_layer(il)) {
            kv_layer_map_[il] = static_cast<int32_t>(n_attn_layers++);
        } else {
            ssm_layer_map_[il] = static_cast<int32_t>(n_ssm_layers++);
        }
    }

    std::cout << "[qwen35] Hybrid cache: " << n_attn_layers
              << " attention layers (KV), " << n_ssm_layers
              << " SSM layers (recurrent state)";
    if (cfg_.has_mtp_head())
        std::cout << " (+" << cfg_.nextn_predict_layers
                  << " NextN head block, excluded from the decode stack)";
    std::cout << std::endl;

    // KV cache — attention layers only.
    uint32_t n_embd_k = meta_.attention_key_length * meta_.attention_head_count_kv;
    uint32_t n_embd_v = meta_.attention_value_length * meta_.attention_head_count_kv;

    kv_cache_ = std::make_unique<simple_kv_cache>(
        n_attn_layers, context_len, max_batch_size,
        n_embd_k, n_embd_v,
        kv_type, kv_type,
        cache_backend
    );

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

    const uint32_t n_layers = n_main_layers_;   // excludes any NextN head block
    const size_t n_tokens   = tokens.size();

    // Typed inputs for this graph (replaces set_inputs). Cleared here so
    // build_output_head can append SparseHeadInput when the sparse path is
    // armed; per-attention-layer masks are added in the layer loop.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    add_positions_input();

    // 1. Token embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // 1b. Image-token substitution (Seam B — the one boundary site between this
    //     recipe and the vision subsystem). Qwen 3.5-family models apply no
    //     embedding scale, so the image rows go in directly. Image-span
    //     attention stays CAUSAL: Qwen-VL is causal over image tokens, unlike
    //     Gemma 3's bidi — so there is no mask parameter here at all.
    //
    //     The grid width is handed to set_prefill_inputs via the base's
    //     mrope_img_grid_w_, where MRopePositionsInput turns it into per-token
    //     (t, h, w, e) components. Refuse a spatially-unstructured image rather
    //     than encode it as a 1-D run.
    if (!image_embd_.empty()) {
        if (cfg_.mrope_sections.active && image_grid_w_ == 0)
            throw std::runtime_error(
                "Qwen35ForwardPass: slot \"set_image_embeddings.grid_w\" "
                "expected > 0 (this recipe uses M-RoPE and needs the image's "
                "soft-token grid), actual: 0");
        if (image_grid_w_ != 0 && image_n_tokens_ % image_grid_w_ != 0)
            throw std::runtime_error(
                "Qwen35ForwardPass: slot \"set_image_embeddings\" expected "
                "n_tokens divisible by grid_w, actual: " +
                std::to_string(image_n_tokens_) + " % " +
                std::to_string(image_grid_w_));

        inpL = build_image_substitution(
            gf, inpL, std::move(image_embd_), image_span_start_,
            image_n_tokens_, static_cast<int>(meta_.embedding_length), n_tokens);
        mrope_img_grid_w_ = cfg_.mrope_sections.active ? image_grid_w_ : 0u;
        image_span_start_ = -1;   // consume-on-use (image_embd_ moved out)
        image_n_tokens_   = 0;
        image_grid_w_     = 0;
        image_grid_h_     = 0;
    }

    // Under M-RoPE the rope position and the KV row count diverge once an image
    // has been written (an image writes nx·ny rows but advances the position by
    // max(nx, ny)). The mask indexes KV rows, so tell it where this batch's rows
    // start; without it the causal test silently hides most of the image.
    if (cfg_.mrope_sections.active)
        mrope_kv_base_ = static_cast<int>(get_cache_pos(slot_idx));

    // Position tensor (for attention layers with RoPE). M-RoPE needs four
    // position components per token — see MRopePositionsInput.
    ggml_tensor* inp_pos = ggml_new_tensor_1d(
        ctx_, GGML_TYPE_I32, n_tokens * n_pos_per_token());
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
            int32_t kv_idx = kv_layer_map_[il];
            const int n_embd_head = meta_.attention_key_length;
            const int n_rot = (cfg_.rope_dimension_count > 0)
                ? cfg_.rope_dimension_count : n_embd_head;
            cur = ::build_gated_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos, kv_idx, n_tokens, slot_idx, il,
                block.attn_q_weight, block.attn_q_norm_weight,
                block.attn_k_weight, block.attn_k_norm_weight,
                block.attn_v_weight, block.attn_output_weight,
                n_embd_head,
                meta_.attention_head_count,
                meta_.attention_head_count_kv,
                n_rot,
                meta_.rope_freq_base,
                static_cast<int>(meta_.context_length),
                meta_.rms_norm_eps,
                cfg_.mrope_sections);

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
// build_decoding_graph — multi-slot single-token decode
// ============================================================

ggml_cgraph* Qwen35ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions)
{
    reset_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);

    const uint32_t n_layers = n_main_layers_;   // excludes any NextN head block
    const size_t n_batch    = tokens.size();
    const int64_t n_embd    = meta_.embedding_length;

    // 1. Token embedding (batched)
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // 2. Position tensor (four components per row under M-RoPE)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(
        ctx_, GGML_TYPE_I32, n_batch * n_pos_per_token());
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. Attention mask + gather indices (used by attention layers only).
    // Width is bucketed (padded tail −inf-masked, zero-init rows) so one
    // decode graph shape survives a whole bucket of steps —
    // plan-persistent-decode-graph.md.
    uint32_t max_pos = 0;
    for (uint32_t s : slots) {
        uint32_t p = get_cache_pos(s);
        if (p > max_pos) max_pos = p;
    }
    uint32_t n_kv_len = decode_kv_len(max_pos + 1, kv_cache_->get_n_ctx_max());

    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32,
        n_kv_len, 1, 1, n_batch);
    ggml_set_input(kq_mask);
    ggml_set_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    uint32_t n_total_indices = n_batch * n_kv_len;
    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_total_indices);
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

    // Typed inputs for the decode graph (replaces set_batched_inputs).
    // Single shared causal mask + KV gather; build_output_head appends
    // SparseHeadInput when the grammar sparse path is armed.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    add_positions_input();
    graph_inputs_.add(std::make_unique<AttnMaskInput>("kq_mask_b", 0u));
    graph_inputs_.add(std::make_unique<GatherIndicesInput>(
        kv_cache_->get_n_ctx_max()));
    if (kv_write_idx)
        graph_inputs_.add(std::make_unique<KvWriteIndicesInput>(
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
                    meta_.rms_norm_eps,
                    kv_write_idx,
                    cfg_.mrope_sections);
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
    // NextN / MTP head: the four tensors that turn the trailing block(s) into a
    // multi-token-prediction head (docs/plan-mtp-decode.md §4).  Optional group:
    //   nextn_predict_layers == 0            → no head, capability off (fine).
    //   nextn_predict_layers  > 0, all found → capability on.
    //   nextn_predict_layers  > 0, any missing → fail-loud naming it (below).
    static const std::vector<std::string> nextn_tensors = {
        "nextn.eh_proj.weight", "nextn.enorm.weight",
        "nextn.hnorm.weight",   "nextn.shared_head_norm.weight"
    };

    const uint32_t fai   = meta.raw_kv.get_uint32("qwen35.full_attention_interval");
    const uint32_t nextn = meta.nextn_predict_layers();
    if (nextn >= meta.block_count)
        throw std::runtime_error(
            "qwen35: field 'nextn_predict_layers' expected < block_count (" +
            std::to_string(meta.block_count) + "), got " + std::to_string(nextn));
    const uint32_t n_main = meta.block_count - nextn;

    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const bool is_nextn = (i >= n_main);
        for (const auto& t : shared) require(p + t, "block " + std::to_string(i));
        // Inline arithmetic: the validator runs at load time, before Qwen35Config
        // is constructed.  NextN blocks are attention-typed regardless of position.
        const bool is_full = is_nextn || ((fai > 0) && ((i % fai) == (fai - 1)));
        const auto& chosen = is_full ? attn_tensors : ssm_tensors;
        const std::string kind = is_nextn ? "NextN" : (is_full ? "attention" : "SSM");
        for (const auto& t : chosen) {
            if (inv.find(p + t) == inv.end())
                throw std::runtime_error(
                    "qwen35: missing tensor '" + p + t +
                    "': expected in " + kind + " layer " + std::to_string(i) +
                    ", got absent");
        }
        if (is_nextn)
            for (const auto& t : nextn_tensors)
                require(p + t, "NextN layer " + std::to_string(i));
    }
}