#include "forward-pass.h"
#include "../kv-cache/turboquant.h"
#include "../kv-cache/snapkv-eviction.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>
#include <cinttypes>
#include <future>
#include <thread>

constexpr size_t GRAPH_SIZE = 16384;

// Number of consecutive transformer layers fused into one ggml graph compute call.
// Cuts Metal scheduler overhead and hidden-state CPU↔Metal round-trips by this factor.
// Each layer has its own persistent scratch slot for incremental decompression.
static constexpr uint32_t TQ_LAYER_BATCH = 4;

Qwen3ForwardPass::Qwen3ForwardPass(
    const Qwen3Model& model, const Qwen3Metadata* metadata,
    uint32_t context_len, uint32_t max_batch_size, int kv_quant_bits)
    : ForwardPassBase(model, metadata) {
        kv_cache_ = nullptr;
        ggml_backend_t cache_backend = model_.has_metal_backend()
            ? model_.get_backend_metal()
            : model_.get_backend_cpu();

            // === Qwen2/Qwen3 standard cache ===
            uint32_t n_embd_k, n_embd_v;
            if (meta_.architecture == "qwen2") {
                n_embd_k = meta_.embedding_length / meta_.attention_head_count * meta_.attention_head_count_kv;
                n_embd_v = meta_.embedding_length / meta_.attention_head_count * meta_.attention_head_count_kv;
            } else { // qwen3
                n_embd_k = meta_.attention_key_length * meta_.attention_head_count_kv;
                n_embd_v = meta_.attention_value_length * meta_.attention_head_count_kv;
            }

            // TQ enabled: skip full F32 KV cache — only the 1-layer scratch is needed.
            // Without TQ: allocate the standard full-size cache.
            if (kv_quant_bits < 2) {
                kv_cache_ = std::make_unique<simple_kv_cache>(
                    meta_.block_count,
                    context_len,
                    max_batch_size,
                    n_embd_k,
                    n_embd_v,
                    GGML_TYPE_F32,
                    GGML_TYPE_F32,
                    cache_backend
                );
            }

            // TurboQuant compressed backing store + scratch cache (Phase 1 + 2)
            if (kv_quant_bits >= 2 && kv_quant_bits <= 4) {
                uint32_t head_dim = (meta_.architecture == "qwen2")
                    ? meta_.embedding_length / meta_.attention_head_count
                    : meta_.attention_key_length;
                tq_store_ = std::make_unique<CompressedKVStore>(
                    meta_.block_count,
                    max_batch_size,
                    context_len,
                    n_embd_k, n_embd_v,
                    head_dim, kv_quant_bits);

                // Persistent scratch cache: one layer per model layer for incremental
                // (delta) decompression. Only new tokens are decompressed + uploaded
                // each decode step, turning O(context_length) into O(1).
                const uint32_t scratch_layers = meta_.block_count;
                tq_scratch_cache_ = std::make_unique<simple_kv_cache>(
                    scratch_layers,
                    context_len,
                    max_batch_size,
                    n_embd_k, n_embd_v,
                    GGML_TYPE_F32, GGML_TYPE_F32,
                    cache_backend);

                // Watermarks: [n_layers][n_slots], all zero initially.
                tq_scratch_valid_pos_.assign(scratch_layers,
                    std::vector<uint32_t>(max_batch_size, 0));

                size_t full_kv_mb = static_cast<size_t>(meta_.block_count) * max_batch_size
                    * context_len * (n_embd_k + n_embd_v) * 4 / (1024 * 1024);
                size_t tq_mb = tq_store_->total_compressed_bytes() / (1024 * 1024);
                float scratch_mb = static_cast<float>(scratch_layers) * max_batch_size
                    * context_len * (n_embd_k + n_embd_v) * 4 / (1024.0f * 1024.0f);
                printf("TurboQuant KV compression: %d-bit (incremental decompress)\n", kv_quant_bits);
                printf("  F32 KV (without TQ): %zu MB\n", full_kv_mb);
                printf("  Compressed store:    %zu MB\n", tq_mb);
                printf("  Scratch (%u layers): %.1f MB\n", scratch_layers, scratch_mb);
                printf("  Compression ratio:   %.1fx\n",
                       static_cast<float>(full_kv_mb) / tq_mb);
            }
        }

struct ggml_cgraph* Qwen3ForwardPass::build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx) {
    reset_context();
    ggml_cgraph* gf = new_graph();
    int n_layers = meta_.block_count;        // Layers 0-27
    int hidden_dim = meta_.embedding_length;    // Model dimension
    int n_head = meta_.attention_head_count;       // Query heads
    int n_head_kv = meta_.attention_head_count_kv;       // KV heads (GQA: 2:1 ratio)
    
    int n_embd_head;
    if (meta_.architecture == "qwen3") {
        n_embd_head = meta_.attention_key_length;
    } else { // qwen2
        n_embd_head = hidden_dim / n_head;
    }

    int vocab_size = meta_.vocab_size;  // From final output projection
    const int n_rot = n_embd_head;
    int n_embd_k = n_head_kv * n_embd_head;
    int n_embd_v = n_head_kv * n_embd_head;
    
    // 1. Token embedding lookup
    const size_t n_tokens = tokens.size();
    ggml_tensor * cur;

    // Token embedding lookup
    ggml_tensor * inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // Position tensor
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);  // Mark as input tensor
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // memset(inp_pos->data, 0, ggml_nbytes(inp_pos));
    // for (int i = 0; i < n_tokens; ++i) {
    //     ggml_set_i32_1d(inp_pos, i, pos + i);
    // }

    // 2. Loop through each transformer block
    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor * inpSA = inpL;
        auto& block = model_.get_block(il);

        // A. RMS Normalization before attention
        cur = build_norm(gf, inpL, block.attn_norm_weight, il);
        set_tensor_name(gf, cur, "cur_normed", il);

        // B. Self-Attention Q, K, V Projections
        ggml_tensor* Qcur = ggml_mul_mat(ctx_, block.attn_q_weight, cur);
        set_tensor_name(gf, Qcur, "Qcur", il);
        if (meta_.architecture == "qwen2") {
            Qcur = ggml_add(ctx_, Qcur, block.attn_q_bias);
            set_tensor_name(gf, Qcur, "Qcur_biased", il);
        }
        ggml_tensor* Kcur = ggml_mul_mat(ctx_, block.attn_k_weight, cur);
        set_tensor_name(gf, Kcur, "Kcur", il);
        if (meta_.architecture == "qwen2") {
            Kcur = ggml_add(ctx_, Kcur, block.attn_k_bias);
            set_tensor_name(gf, Kcur, "Kcur_biased", il);
        }
        ggml_tensor* Vcur = ggml_mul_mat(ctx_, block.attn_v_weight, cur);
        set_tensor_name(gf, Vcur, "Vcur", il);
        if (meta_.architecture == "qwen2") {
            Vcur = ggml_add(ctx_, Vcur, block.attn_v_bias);
            set_tensor_name(gf, Vcur, "Vcur_biased", il);
        }


        Qcur = ggml_reshape_3d(ctx_, Qcur, n_embd_head, n_head,    n_tokens);
        set_tensor_name(gf, Qcur, "Qcur_3d", il);
        Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head_kv, n_tokens);
        set_tensor_name(gf, Kcur, "Kcur_3d", il);
        Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head_kv, n_tokens);
        set_tensor_name(gf, Vcur, "Vcur_3d", il);

        if (meta_.architecture == "qwen3") {
            Qcur = build_norm(gf, Qcur, block.attn_q_norm_weight, il);
            set_tensor_name(gf, Qcur, "Qcur_normed", il);
        }
        // float freq_base = 10000.0f;
        float freq_base = meta_.rope_freq_base;
        Qcur = ggml_rope_ext(ctx_, Qcur, inp_pos, nullptr, 
                              n_rot,                           // rotation dims
                              GGML_ROPE_TYPE_NEOX,           // mode = 2
                              meta_.context_length,        // n_ctx_orig = 40960
                              freq_base,        // freq_base = 1000000
                              1.0f,                           // freq_scale = 1
                              0.0f,                           // ext_factor = 0
                              1.0f,                           // attn_factor = 1
                              32.0f,                          // beta_fast = 32
                              1.0f);                          // beta_slow = 1


        set_tensor_name(gf, Qcur, "Qcur_roped", il);
        if (meta_.architecture == "qwen3") {
            Kcur = build_norm(gf, Kcur, block.attn_k_norm_weight, il);
            set_tensor_name(gf, Kcur, "Kcur_normed", il);
        }
        Kcur = ggml_rope_ext(ctx_, Kcur, inp_pos, nullptr, 
                              n_rot,                           // rotation dims
                              GGML_ROPE_TYPE_NEOX,           // mode = 2
                              meta_.context_length,        // n_ctx_orig = 40960
                              freq_base,        // freq_base = 1000000
                              1.0f,                           // freq_scale = 1
                              0.0f,                           // ext_factor = 0
                              1.0f,                           // attn_factor = 1
                              32.0f,                          // beta_fast = 32
                              1.0f);

        set_tensor_name(gf, Kcur, "Kcur_roped", il);

        // Build attention

        // D. Grouped-Query Attention
        float kq_scale = 1.0f/sqrtf(float(n_embd_head));
        cur = _build_attention_layer(gf, kv_cache_.get(), Qcur, Kcur, Vcur, il, kq_scale, n_tokens, slot_idx, il);

        cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);
        set_tensor_name(gf, cur, "attn_output", il);

        // Only need logits for the last token
        // if (il == n_layer - 1 && inp_out_ids) {
        //     cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
        //     inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        // }

        

        // E. Residual connection after attention
        ggml_tensor * ffn_inp = ggml_add(ctx_, cur, inpSA);
        set_tensor_name(gf, ffn_inp, "ffn_inp", il);

        // F. RMS Normalization before FFN
        cur = build_norm(gf, ffn_inp, block.ffn_norm_weight, il);
        set_tensor_name(gf, cur, "ffn_norm", il);

        // G. Feed-Forward Network (SwiGLU)
        cur = ffn_swiglu(gf, cur, block.ffn_gate_weight, block.ffn_up_weight, block.ffn_down_weight, il);
        set_tensor_name(gf, cur, "ffn_output", il);

        // H. Residual connection after FFN
        cur = ggml_add(ctx_, cur, ffn_inp);
        set_tensor_name(gf, cur, "cur_ffn_inp", il);

        // cur = build_cvec(cur, il);
        inpL = cur;
    }

    // 3. Final normalization and output projection
    build_output_head(gf, inpL);

    return gf;
}

struct ggml_cgraph* Qwen3ForwardPass::build_decoding_graph(
    const std::vector<int32_t>& tokens, 
    const std::vector<uint32_t>& slots, 
    const std::vector<int32_t>& positions
) {
    // Reset context
    if (ctx_) {
        ggml_free(ctx_);
    }
    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true, 
    };
    ctx_ = ggml_init(params);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx_, GRAPH_SIZE, false);
    
    // Core parameters
    int n_layers = meta_.block_count;
    int hidden_dim = meta_.embedding_length;
    int n_head = meta_.attention_head_count;
    int n_head_kv = meta_.attention_head_count_kv;
    
    int n_embd_head;
    if (meta_.architecture == "qwen3") {
        n_embd_head = meta_.attention_key_length;
    } else { // qwen2
        n_embd_head = hidden_dim / n_head;
    }

    const int n_rot = n_embd_head;
    const size_t n_tokens = tokens.size(); // Total tokens across all slots

    // 1. Embeddings (batched)
    ggml_tensor * inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // Position tensor (vector of positions, one per token)
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // Find max position to determine Mask/Gather length
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > (int32_t)max_pos) max_pos = (uint32_t)p;
    }
    uint32_t n_kv_len = max_pos + 1;

    // Attention Mask (shared across layers)
    // Shape: [n_kv_len, n_tokens, 1]
    // ggml_tensor* kq_mask = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, n_kv_len, n_tokens, 1);
    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, n_kv_len, 1, 1, n_tokens);
    
    ggml_set_input(kq_mask);
    ggml_set_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    // KV Gather Indices (shared across layers)
    // Shape: [n_tokens * n_kv_len] (1D tensor of indices)
    uint32_t n_total_indices = n_tokens * n_kv_len;
    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_total_indices);
    ggml_set_input(gather_indices);
    ggml_set_name(gather_indices, "gather_indices");

    // 2. Transformer Blocks
    ggml_tensor * cur;
    for (uint32_t il = 0; il < n_layers; ++il) {
        ggml_tensor * inpSA = inpL;
        auto& block = model_.get_block(il);

        // A. Norm
        cur = build_norm(gf, inpL, block.attn_norm_weight, il);
        
        // B. Q, K, V
        ggml_tensor* Qcur = ggml_mul_mat(ctx_, block.attn_q_weight, cur);
        if (meta_.architecture == "qwen2") Qcur = ggml_add(ctx_, Qcur, block.attn_q_bias);
        
        ggml_tensor* Kcur = ggml_mul_mat(ctx_, block.attn_k_weight, cur);
        if (meta_.architecture == "qwen2") Kcur = ggml_add(ctx_, Kcur, block.attn_k_bias);
        
        ggml_tensor* Vcur = ggml_mul_mat(ctx_, block.attn_v_weight, cur);
        if (meta_.architecture == "qwen2") Vcur = ggml_add(ctx_, Vcur, block.attn_v_bias);

        // Reshape [n_head, n_embd_head, n_tokens]
        Qcur = ggml_reshape_3d(ctx_, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head_kv, n_tokens);

        // Qwen3 Norm
        if (meta_.architecture == "qwen3") {
            Qcur = build_norm(gf, Qcur, block.attn_q_norm_weight, il);
            Kcur = build_norm(gf, Kcur, block.attn_k_norm_weight, il);
        }

        // RoPE (using vector positions)
        float freq_base = meta_.rope_freq_base;
        Qcur = ggml_rope_ext(ctx_, Qcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = ggml_rope_ext(ctx_, Kcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // D. Attention (Batched)
        float kq_scale = 1.0f/sqrtf(float(n_embd_head));
        cur = _build_batched_attention_layer(gf, kv_cache_.get(), Qcur, Kcur, Vcur, il, kq_scale, slots, positions, kq_mask, gather_indices, il);

        // E. Output Projection & Residual
        cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);
        ggml_tensor * ffn_inp = ggml_add(ctx_, cur, inpSA);

        // F. FFN
        cur = build_norm(gf, ffn_inp, block.ffn_norm_weight, il);
        cur = ffn_swiglu(gf, cur, block.ffn_gate_weight, block.ffn_up_weight, block.ffn_down_weight, il);
        
        // H. FFN Residual
        cur = ggml_add(ctx_, cur, ffn_inp);
        inpL = cur;
    }
    
    // 3. Output Head
    cur = inpL;
    cur = build_norm(gf, cur, model_.get_output_norm_weight(), -1);
    if (model_.get_output_weight() != nullptr) {
        cur = ggml_mul_mat(ctx_, model_.get_output_weight(), cur);
    } else {
        cur = ggml_mul_mat(ctx_, model_.get_token_embedding_weight(), cur);
    }
    ggml_set_name(cur, "logits"); // Name output for retrieval
    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_tensor * Qwen3ForwardPass::_build_batched_attention_layer(
    ggml_cgraph * gf,
    simple_kv_cache * kv_cache,
    ggml_tensor * q, 
    ggml_tensor * k, 
    ggml_tensor * v,
    int layer_idx, 
    float kq_scale,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>& positions,
    ggml_tensor * kq_mask,
    ggml_tensor * gather_indices,
    int il) {

    // Assumptions for Phase 3 Step 1 (Uniform Decoding):
    // - Each slot contributes exactly 1 token.
    // - 'positions' contains the sequence position for that 1 token for each slot.
    // - 'n_tokens' (batch size) = slots.size()
    
    size_t n_batch = slots.size();
    
    // 1. Update KV Cache (Write Step)
    // We iterate through each slot in the batch and write its new K/V to storage.
    // k and v input tensors are [n_embd, 1, n_batch] conceptually, but likely reshaped.
    // k is [n_embd_head, n_head_kv, n_batch] from caller.
    
    // We need to reshape k/v back to [n_embd_k, n_batch] to slice them for storage copy
    int n_embd_head = k->ne[0];
    int n_head_kv   = k->ne[1];
    int n_embd_k    = n_embd_head * n_head_kv;
    int n_embd_v    = n_embd_head * n_head_kv;
    
    ggml_tensor* k_storage_fmt = ggml_reshape_3d(ctx_, k, n_embd_k, 1, n_batch);
    ggml_tensor* v_storage_fmt = ggml_reshape_3d(ctx_, v, n_embd_v, 1, n_batch);

    for (size_t i = 0; i < n_batch; ++i) {
        // Slice the current batch item
        // k_slice: [n_embd_k, 1]
        size_t k_offset = i * k_storage_fmt->nb[2];
        ggml_tensor* k_slice = ggml_view_2d(ctx_, k_storage_fmt, n_embd_k, 1, k_storage_fmt->nb[1], k_offset);
        
        size_t v_offset = i * v_storage_fmt->nb[2];
        ggml_tensor* v_slice = ggml_view_2d(ctx_, v_storage_fmt, n_embd_v, 1, v_storage_fmt->nb[1], v_offset);
        
        // Copy to storage at correct slot and position
        ggml_tensor* k_stored = kv_cache->cpy_k(ctx_, k_slice, layer_idx, slots[i]);
        set_tensor_name(gf, k_stored, "k_stored_b", il);
        ggml_build_forward_expand(gf, k_stored);
        
        ggml_tensor* v_stored = kv_cache->cpy_v(ctx_, v_slice, layer_idx, slots[i]);
        set_tensor_name(gf, v_stored, "v_stored_b", il);
        ggml_build_forward_expand(gf, v_stored);
    }
    
    // 2. Gather KV Cache for Attention (Read Step)
    // We need the full history for each slot.
    // For now, assume all slots have same history length (max_pos) or we take the max.
    // The mask will handle the raggedness.
    
    // Find max position to determine Gather length
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > (int32_t)max_pos) max_pos = (uint32_t)p;
    }
    uint32_t n_kv_len = max_pos + 1; // inclusive of current token
    
    // Gather returns [n_embd, n_kv_len, n_batch]
    ggml_tensor* k_gathered = kv_cache->gather_k(ctx_, gf, layer_idx, gather_indices, n_batch, n_kv_len);
    ggml_tensor* v_gathered = kv_cache->gather_v(ctx_, gf, layer_idx, gather_indices, n_batch, n_kv_len);
    
    // 3. Prepare for Attention
    // k_gathered is [n_embd, n_kv_len, n_batch]
    // We need [n_embd_head, n_head_kv, n_kv_len, n_batch]
    
    ggml_tensor* k_view = ggml_view_4d(ctx_, k_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_k * sizeof(float),
        n_embd_k * n_kv_len * sizeof(float),
        0);
        
    ggml_tensor* v_view = ggml_view_4d(ctx_, v_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * sizeof(float),
        n_embd_v * sizeof(float),
        n_embd_v * n_kv_len * sizeof(float),
        0);
        
    // 4. Compute Attention
    return build_attn_mha(gf, q, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il);
}

ggml_tensor * Qwen3ForwardPass::_build_attention_layer(
    ggml_cgraph * gf,
    simple_kv_cache * kv_cache,
    ggml_tensor * q, 
    ggml_tensor * k, 
    ggml_tensor * v,
    int layer_idx, 
    float kq_scale,
    uint32_t n_tokens,
    uint32_t slot_idx,
    int il) {
    const uint32_t pos = kv_cache->get_pos(slot_idx);
    const uint32_t n_kv = pos + n_tokens;

    // 1. Cache the new K and V values
    ggml_tensor * k_cached = kv_cache->cpy_k(ctx_, k, layer_idx, slot_idx);
    set_tensor_name(gf, k_cached, "k_cached", il);
    ggml_build_forward_expand(gf, k_cached);

    ggml_tensor * v_cached = kv_cache->cpy_v(ctx_, v, layer_idx, slot_idx);
    set_tensor_name(gf, v_cached, "v_cached", il);
    ggml_build_forward_expand(gf, v_cached);

    // 2. Retrieve the full K and V sequences from the cache (up to current position)
    ggml_tensor * k_full = kv_cache->get_k(ctx_, layer_idx, n_kv, slot_idx);
    ggml_tensor * v_full = kv_cache->get_v(ctx_, layer_idx, n_kv, slot_idx);

    // 3. Create views for the attention calculation
    // k_full is [n_embd_k=1024, n_kv], need [128, 8, n_kv, 1]

    int hidden_dim = meta_.embedding_length;
    int n_head = meta_.attention_head_count;
    int n_embd_head;
    if (meta_.architecture == "qwen3") {
        n_embd_head = meta_.attention_key_length;
    } else { // qwen2
        n_embd_head = hidden_dim / n_head;
    }
    int n_head_kv = meta_.attention_head_count_kv;
    int n_embd_k = n_head_kv * n_embd_head;
    int n_embd_v = n_head_kv * n_embd_head;

    k = ggml_view_3d(ctx_, k_full, 
        n_embd_head,      // head dimension: n_embd_k / n_head_kv
        n_head_kv,        // n_head_kv
        n_kv,             // cached sequence length
        n_embd_head * sizeof(float),   // nb[1]: stride between heads
        n_embd_k * sizeof(float),      // nb[2]: stride between tokens
        0);

    // v_full is [n_embd_v, n_kv], need [n_embd_head, n_head_kv, n_kv]
    v = ggml_view_3d(ctx_, v_full,
        n_embd_head,      // head dimension: n_embd_v / n_head_kv
        n_head_kv,
        n_kv,             // cached sequence length
        n_embd_head * sizeof(float),   // nb[1]: stride between heads
        n_embd_v * sizeof(float),      // nb[2]: stride between tokens
        0);


    set_tensor_name(gf, k, "k_view", il);
    set_tensor_name(gf, v, "v_view", il);

    // 4. Build the causal mask for the full sequence
    // char mask_name[32];
    // snprintf(mask_name, sizeof(mask_name), "kq_mask_%d", layer_idx);

    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, n_kv, n_tokens);
    // ggml_set_input(kq_mask);
    set_tensor_name(gf, kq_mask, "kq_mask", il);
    ggml_build_forward_expand(gf, kq_mask);

    // 5. Execute attention with the cached values
    ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_mask, nullptr, kq_scale, pos, il);
    return cur;
}


// Set input data AFTER allocation
void Qwen3ForwardPass::set_inputs(ggml_cgraph* gf, 
                                   const std::vector<int32_t>& tokens, 
                                   int pos) {
    const size_t n_tokens = tokens.size();
    
    // 1. Set token IDs
    ggml_tensor* tokens_tensor = ggml_graph_get_tensor(gf, "tokens");
    if (!tokens_tensor) {
        throw std::runtime_error("tokens tensor not found in graph");
    }
    ggml_backend_tensor_set(tokens_tensor, tokens.data(), 0, n_tokens * sizeof(int32_t));

    // DEBUG: Verify token data transfer
    std::vector<int32_t> tokens_readback(n_tokens);
    ggml_backend_tensor_get(tokens_tensor, tokens_readback.data(), 0, n_tokens * sizeof(int32_t));
    if (tokens != tokens_readback) {
        fprintf(stderr, "FATAL: Token data mismatch after transfer to GPU!\n");
        for(size_t i = 0; i < n_tokens; ++i) {
            if (tokens[i] != tokens_readback[i]) {
                fprintf(stderr, "Mismatch at index %zu: Original=%d, Readback=%d\n", i, tokens[i], tokens_readback[i]);
            }
        }
        exit(1); // Abort on critical error
    }
    
    // 2. Set position IDs
    ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (!inp_pos) {
        throw std::runtime_error("inp_pos tensor not found in graph");
    }
    std::vector<int32_t> pos_data(n_tokens);
    for (size_t i = 0; i < n_tokens; i++) {
        pos_data[i] = pos + i;
    }
    ggml_backend_tensor_set(inp_pos, pos_data.data(), 0, n_tokens * sizeof(int32_t));

    // DEBUG: Verify position data transfer
    std::vector<int32_t> pos_readback(n_tokens);
    ggml_backend_tensor_get(inp_pos, pos_readback.data(), 0, n_tokens * sizeof(int32_t));
    if (pos_data != pos_readback) {
        fprintf(stderr, "FATAL: Position data mismatch after transfer to GPU!\n");
        exit(1); // Abort on critical error
    }

    // 3. Set causal attention mask for each layer
    for (uint32_t il = 0; il < meta_.block_count; ++il) {
        char mask_name[32];
        snprintf(mask_name, sizeof(mask_name), "kq_mask.%d", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, mask_name);
        if (!kq_mask) {
            throw std::runtime_error("kq_mask tensor not found in graph for layer " + std::to_string(il));
        }

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


// Set input data for batched graph
void Qwen3ForwardPass::set_batched_inputs(ggml_cgraph* gf, 
                                   const std::vector<int32_t>& tokens, 
                                   const std::vector<uint32_t>& slots,
                                   const std::vector<int32_t>& positions) {
    const size_t n_tokens = tokens.size();
    if (slots.size() != n_tokens || positions.size() != n_tokens) {
        throw std::runtime_error("Input vector size mismatch in set_batched_inputs");
    }
    
    // 1. Set token IDs
    ggml_tensor* tokens_tensor = ggml_graph_get_tensor(gf, "tokens");
    if (!tokens_tensor) {
        throw std::runtime_error("tokens tensor not found in graph");
    }
    ggml_backend_tensor_set(tokens_tensor, tokens.data(), 0, n_tokens * sizeof(int32_t));
    
    // 2. Set position IDs
    ggml_tensor* inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (!inp_pos) {
        throw std::runtime_error("inp_pos tensor not found in graph");
    }
    ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));

    // 3. Set shared causal attention mask
    ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, "kq_mask_b");
    if (!kq_mask) {
        throw std::runtime_error("kq_mask_b tensor not found in graph");
    }

    const uint32_t n_kv_dim = kq_mask->ne[0]; 

std::vector<float> mask_data(n_kv_dim * n_tokens);
for (uint32_t b = 0; b < n_tokens; ++b) {
    const int32_t current_pos = positions[b];
    for (uint32_t j = 0; j < n_kv_dim; ++j) {
        // Layout: [n_kv, 1, 1, n_batch] — batch is slowest dimension
        mask_data[b * n_kv_dim + j] = ((int32_t)j <= current_pos) ? 0.0f : -INFINITY;
    }
}


    ggml_backend_tensor_set(kq_mask, mask_data.data(), 0, n_kv_dim * n_tokens * sizeof(float));

    // 4. Set KV Gather Indices (shared across layers)
    ggml_tensor* gather_indices = ggml_graph_get_tensor(gf, "gather_indices");
    if (gather_indices) {
        // gather_indices->ne[0] is n_tokens * n_kv_len
        uint32_t n_total_indices = gather_indices->ne[0];
        uint32_t n_kv_len = n_total_indices / n_tokens;
        std::vector<int32_t> indices(n_total_indices);
        
        for (uint32_t b = 0; b < n_tokens; ++b) {
            uint32_t slot_idx = slots[b];
            for (uint32_t t = 0; t < n_kv_len; ++t) {
                // Flat index = slot_idx * n_ctx_max + t
                indices[b * n_kv_len + t] = slot_idx * kv_cache_->get_n_ctx_max() + t;
            }
        }
        ggml_backend_tensor_set(gather_indices, indices.data(), 0, indices.size() * sizeof(int32_t));
    }
}

// ============================================================================
// TurboQuant Phase 2: Per-layer compute with compressed KV
// ============================================================================

void Qwen3ForwardPass::_tq_decompress_layer(uint32_t layer, uint32_t slot_idx,
                                             uint32_t scratch_layer) {
    const uint32_t pos = tq_store_->get_pos(slot_idx);
    if (pos == 0) return;

    // Incremental: only decompress tokens [watermark..pos).
    const uint32_t watermark = tq_scratch_valid_pos_[layer][slot_idx];
    if (watermark >= pos) return;  // scratch is already up-to-date
    const uint32_t n_delta = pos - watermark;

    const int n_embd_head = (meta_.architecture == "qwen3")
        ? meta_.attention_key_length
        : meta_.embedding_length / meta_.attention_head_count;
    const int n_head_kv = meta_.attention_head_count_kv;
    const int n_embd_k  = n_head_kv * n_embd_head;
    const int n_embd_v  = n_head_kv * n_embd_head;

    ggml_tensor* k_tensor = tq_scratch_cache_->get_k_cache_tensor(scratch_layer);
    ggml_tensor* v_tensor = tq_scratch_cache_->get_v_cache_tensor(scratch_layer);

    std::vector<float> k_buf(static_cast<size_t>(n_delta) * n_embd_k);
    std::vector<float> v_buf(static_cast<size_t>(n_delta) * n_embd_v);

    // Each token writes to a non-overlapping slice; reads from tq_store_ are
    // at distinct offsets — safe to parallelise. Serial fast-path for n_delta < 4.
    static constexpr uint32_t MIN_PAR_TOKENS = 4;
    const uint32_t hw_threads =
        std::max(1u, static_cast<uint32_t>(std::thread::hardware_concurrency()));
    const uint32_t n_threads = (n_delta >= MIN_PAR_TOKENS)
        ? std::min(n_delta, hw_threads) : 1u;

    auto decompress_range = [&](uint32_t t0, uint32_t t1) {
        for (uint32_t t = t0; t < t1; ++t) {
            const uint32_t abs_pos = watermark + t;
            tq_store_->decompress_token_k(layer, slot_idx, abs_pos,
                k_buf.data() + t * n_embd_k);
            tq_store_->decompress_token_v(layer, slot_idx, abs_pos,
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

    tq_scratch_valid_pos_[layer][slot_idx] = pos;
}

void Qwen3ForwardPass::_tq_compress_new(uint32_t layer, uint32_t slot_idx,
                                         uint32_t pos, uint32_t n_tokens,
                                         uint32_t scratch_layer) {
    const int n_embd_head = (meta_.architecture == "qwen3")
        ? meta_.attention_key_length
        : meta_.embedding_length / meta_.attention_head_count;
    const int n_head_kv = meta_.attention_head_count_kv;
    const int n_embd_k  = n_head_kv * n_embd_head;
    const int n_embd_v  = n_head_kv * n_embd_head;

    ggml_tensor* k_tensor = tq_scratch_cache_->get_k_cache_tensor(scratch_layer);
    ggml_tensor* v_tensor = tq_scratch_cache_->get_v_cache_tensor(scratch_layer);

    // Bulk read: two tensor_get calls total instead of 2*n_tokens.
    // Tokens pos..pos+n_tokens-1 are contiguous within a slot.
    std::vector<float> k_buf(static_cast<size_t>(n_tokens) * n_embd_k);
    std::vector<float> v_buf(static_cast<size_t>(n_tokens) * n_embd_v);

    ggml_backend_tensor_get(k_tensor, k_buf.data(),
        slot_idx * k_tensor->nb[2] + pos * k_tensor->nb[1],
        static_cast<size_t>(n_tokens) * n_embd_k * sizeof(float));
    ggml_backend_tensor_get(v_tensor, v_buf.data(),
        slot_idx * v_tensor->nb[2] + pos * v_tensor->nb[1],
        static_cast<size_t>(n_tokens) * n_embd_v * sizeof(float));

    // compress_token_k/v copies to an internal scratch before turboquant::compress,
    // so each call only touches its own slice — safe to parallelise across tokens.
    static constexpr uint32_t MIN_PAR_TOKENS = 4;
    const uint32_t hw_threads =
        std::max(1u, static_cast<uint32_t>(std::thread::hardware_concurrency()));
    const uint32_t n_threads = (n_tokens >= MIN_PAR_TOKENS)
        ? std::min(n_tokens, hw_threads) : 1u;

    auto compress_range = [&](uint32_t t0, uint32_t t1) {
        for (uint32_t t = t0; t < t1; ++t) {
            tq_store_->compress_token_k(layer, slot_idx, pos + t,
                k_buf.data() + t * n_embd_k);
            tq_store_->compress_token_v(layer, slot_idx, pos + t,
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
    tq_scratch_valid_pos_[layer][slot_idx] = pos + n_tokens;
}

ggml_cgraph* Qwen3ForwardPass::_build_single_layer_graph(
    uint32_t il, const std::vector<int32_t>& tokens,
    int pos, uint32_t slot_idx, uint32_t n_tokens,
    uint32_t scratch_layer)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    int n_embd_head = (meta_.architecture == "qwen3")
        ? meta_.attention_key_length
        : meta_.embedding_length / meta_.attention_head_count;
    int n_head = meta_.attention_head_count;
    int n_head_kv = meta_.attention_head_count_kv;
    int n_embd_k = n_head_kv * n_embd_head;
    const int n_rot = n_embd_head;

    auto& block = model_.get_block(il);

    // Input hidden state — will be set via ggml_backend_tensor_set
    ggml_tensor* inpL = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32,
        meta_.embedding_length, n_tokens);
    ggml_set_input(inpL);
    ggml_set_name(inpL, "inpL");
    ggml_build_forward_expand(gf, inpL);

    // Position tensor
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    ggml_tensor* inpSA = inpL;
    ggml_tensor* cur;

    // A. RMS Normalization before attention
    cur = build_norm(gf, inpL, block.attn_norm_weight, il);

    // B. Q, K, V projections
    ggml_tensor* Qcur = ggml_mul_mat(ctx_, block.attn_q_weight, cur);
    if (meta_.architecture == "qwen2") Qcur = ggml_add(ctx_, Qcur, block.attn_q_bias);
    ggml_tensor* Kcur = ggml_mul_mat(ctx_, block.attn_k_weight, cur);
    if (meta_.architecture == "qwen2") Kcur = ggml_add(ctx_, Kcur, block.attn_k_bias);
    ggml_tensor* Vcur = ggml_mul_mat(ctx_, block.attn_v_weight, cur);
    if (meta_.architecture == "qwen2") Vcur = ggml_add(ctx_, Vcur, block.attn_v_bias);

    Qcur = ggml_reshape_3d(ctx_, Qcur, n_embd_head, n_head, n_tokens);
    Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head_kv, n_tokens);

    if (meta_.architecture == "qwen3") {
        Qcur = build_norm(gf, Qcur, block.attn_q_norm_weight, il);
        Kcur = build_norm(gf, Kcur, block.attn_k_norm_weight, il);
    }

    // RoPE
    float freq_base = meta_.rope_freq_base;
    Qcur = ggml_rope_ext(ctx_, Qcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
        meta_.context_length, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(ctx_, Kcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
        meta_.context_length, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // C. Attention
    float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    cur = _build_attention_layer(gf, tq_scratch_cache_.get(), Qcur, Kcur, Vcur,
        scratch_layer, kq_scale, n_tokens, slot_idx, il);

    // D. Output projection + residual
    cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);
    ggml_tensor* ffn_inp = ggml_add(ctx_, cur, inpSA);

    // E. FFN
    cur = build_norm(gf, ffn_inp, block.ffn_norm_weight, il);
    cur = ffn_swiglu(gf, cur, block.ffn_gate_weight, block.ffn_up_weight,
        block.ffn_down_weight, il);

    // F. FFN residual
    cur = ggml_add(ctx_, cur, ffn_inp);
    ggml_set_name(cur, "layer_out");
    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph* Qwen3ForwardPass::_build_output_head_graph() {
    reset_context();
    ggml_cgraph* gf = new_graph();
    // We'll handle this inline in run_prefill instead.
    return gf;
}

// Build a fused graph for layers [il_start, il_end) in one ggml context.
// Each attention layer in the batch is assigned an incrementing scratch_layer index
// so it reads/writes a distinct slot of tq_scratch_cache_.
// Input tensor: "inpL"  (hidden state before il_start)
// Output tensor: "layer_out" (hidden state after il_end - 1)
ggml_cgraph* Qwen3ForwardPass::_build_layer_batch_graph(
    uint32_t il_start, uint32_t il_end,
    const std::vector<int32_t>& /*tokens*/,
    int /*pos*/, uint32_t slot_idx, uint32_t n_tokens)
{
    reset_context();
    ggml_cgraph* gf = new_graph();

    const int n_embd_head = (meta_.architecture == "qwen3")
        ? static_cast<int>(meta_.attention_key_length)
        : static_cast<int>(meta_.embedding_length / meta_.attention_head_count);
    const int n_head    = static_cast<int>(meta_.attention_head_count);
    const int n_head_kv = static_cast<int>(meta_.attention_head_count_kv);
    const int n_rot     = n_embd_head;
    const float freq_base = meta_.rope_freq_base;

    // Shared inputs — set once, used by every layer in the batch
    ggml_tensor* inpL = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32,
        meta_.embedding_length, n_tokens);
    ggml_set_input(inpL);
    ggml_set_name(inpL, "inpL");
    ggml_build_forward_expand(gf, inpL);

    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    ggml_tensor* cur = inpL;

    for (uint32_t il = il_start; il < il_end; ++il) {
        auto& block  = model_.get_block(il);
        ggml_tensor* inpSA = cur;

        // Pre-attention norm
        cur = build_norm(gf, cur, block.attn_norm_weight, il);

        // Q, K, V projections
        ggml_tensor* Qcur = ggml_mul_mat(ctx_, block.attn_q_weight, cur);
        ggml_tensor* Kcur = ggml_mul_mat(ctx_, block.attn_k_weight, cur);
        ggml_tensor* Vcur = ggml_mul_mat(ctx_, block.attn_v_weight, cur);
        if (meta_.architecture == "qwen2") {
            Qcur = ggml_add(ctx_, Qcur, block.attn_q_bias);
            Kcur = ggml_add(ctx_, Kcur, block.attn_k_bias);
            Vcur = ggml_add(ctx_, Vcur, block.attn_v_bias);
        }

        Qcur = ggml_reshape_3d(ctx_, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head_kv, n_tokens);

        if (meta_.architecture == "qwen3") {
            Qcur = build_norm(gf, Qcur, block.attn_q_norm_weight, il);
            Kcur = build_norm(gf, Kcur, block.attn_k_norm_weight, il);
        }

        // RoPE
        Qcur = ggml_rope_ext(ctx_, Qcur, inp_pos, nullptr, n_rot,
            GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base,
            1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = ggml_rope_ext(ctx_, Kcur, inp_pos, nullptr, n_rot,
            GGML_ROPE_TYPE_NEOX, meta_.context_length, freq_base,
            1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Attention — each layer uses its persistent scratch cache slot (il)
        const float kq_scale = 1.0f / sqrtf(static_cast<float>(n_embd_head));
        cur = _build_attention_layer(gf, tq_scratch_cache_.get(), Qcur, Kcur, Vcur,
            il, kq_scale, n_tokens, slot_idx, il);

        // Output projection + residuals
        cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);
        ggml_tensor* ffn_inp = ggml_add(ctx_, cur, inpSA);

        cur = build_norm(gf, ffn_inp, block.ffn_norm_weight, il);
        cur = ffn_swiglu(gf, cur, block.ffn_gate_weight, block.ffn_up_weight,
            block.ffn_down_weight, il);
        cur = ggml_add(ctx_, cur, ffn_inp);
    }

    ggml_set_name(cur, "layer_out");
    ggml_build_forward_expand(gf, cur);
    return gf;
}

std::vector<float> Qwen3ForwardPass::run_prefill(
    const std::vector<int32_t>& tokens,
    int pos, uint32_t slot_idx,
    ggml_backend_sched_t scheduler)
{
    const bool do_snapkv = snapkv_budget_ > 0 && tokens.size() > snapkv_window_;

    // ── Non-TQ path (monolithic graph) ──────────────────────────────────
    if (!tq_store_) {
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);

        // Mark kq_soft at last layer as output so scheduler keeps it
        const uint32_t scoring_layer = meta_.block_count - 1;
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
            apply_snapkv_from_graph(gf, scoring_layer, tokens.size(),
                meta_.block_count, snapkv_budget_, snapkv_window_,
                slot_idx, kv_cache_.get(), nullptr, nullptr);
        }

        return logits;
    }

    // ── TQ path ─────────────────────────────────────────────────────────
    const uint32_t n_layers = meta_.block_count;
    const uint32_t n_tokens = tokens.size();
    const int hidden_dim = meta_.embedding_length;

    // Scratch for hidden state between layers
    std::vector<float> hidden(static_cast<size_t>(hidden_dim) * n_tokens);

    // ── Token embeddings via ggml (handles quantized weights correctly) ──
    // Direct ggml_backend_tensor_get on the embedding weight would compute
    // wrong byte offsets for quantized (Q4_K etc.) tensors. Use ggml_get_rows.
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

    // Sync scratch cache position with TQ store's position tracker
    tq_scratch_cache_->set_pos(tq_store_->get_pos(slot_idx), slot_idx);

    // Pre-compute position IDs (same for every batch)
    std::vector<int32_t> pos_data(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) pos_data[i] = pos + i;

    // ── Batched layer compute ────────────────────────────────────────────
    // TQ_LAYER_BATCH layers share one graph_compute call → cuts scheduler
    // overhead and hidden-state CPU↔Metal round-trips by TQ_LAYER_BATCH×.
    bool tq_advanced = false;
    for (uint32_t il0 = 0; il0 < n_layers; il0 += TQ_LAYER_BATCH) {
        const uint32_t il1 = std::min(il0 + TQ_LAYER_BATCH, n_layers);

        // 1. Decompress KV delta for each layer into its persistent scratch slot
        for (uint32_t il = il0; il < il1; ++il)
            _tq_decompress_layer(il, slot_idx, il);

        // 2. Build fused graph for [il0, il1)
        ggml_cgraph* gf = _build_layer_batch_graph(il0, il1, tokens, pos, slot_idx, n_tokens);

        // SnapKV: if this batch contains the scoring layer, mark kq_soft as output
        const uint32_t scoring_layer = n_layers - 1;
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

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp_pos"),
            pos_data.data(), 0, n_tokens * sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inpL"),
            hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));

        // Set one causal mask per layer in the batch
        for (uint32_t il = il0; il < il1; ++il) {
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

        // 5. Read back hidden state once per batch (not once per layer)
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer_out"),
            hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));

        // 6. Compress KV for each layer from its persistent scratch slot
        for (uint32_t il = il0; il < il1; ++il)
            _tq_compress_new(il, slot_idx, tq_store_->get_pos(slot_idx), n_tokens, il);

        // 7. SnapKV: evict from compressed store using this batch's kq_soft
        if (batch_has_scoring) {
            tq_store_->advance(slot_idx, n_tokens);

            apply_snapkv_from_graph(gf, scoring_layer, n_tokens,
                n_layers, snapkv_budget_, snapkv_window_,
                slot_idx, nullptr, tq_store_.get(),
                [this](uint32_t s) { _tq_invalidate_watermarks(s); });

            tq_scratch_cache_->set_pos(tq_store_->get_pos(slot_idx), slot_idx);
            tq_advanced = true;
        }
    }

    if (!tq_advanced)
        tq_store_->advance(slot_idx, n_tokens);

    // ── Output head ──────────────────────────────────────────────────────
    // Build a minimal graph for final norm + LM head
    reset_context();
    ggml_cgraph* gf_out = new_graph();

    ggml_tensor* final_in = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32,
        hidden_dim, n_tokens);
    ggml_set_input(final_in);
    ggml_set_name(final_in, "final_in");
    ggml_build_forward_expand(gf_out, final_in);

    build_output_head(gf_out, final_in);

    ggml_backend_sched_reset(scheduler);
    ggml_backend_sched_alloc_graph(scheduler, gf_out);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_out, "final_in"),
        hidden.data(), 0,
        static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));
    ggml_backend_sched_graph_compute(scheduler, gf_out);

    return get_output_logits(gf_out);
}

