#include "qwen3.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../layers/transformer_block.h"
#include "../state/turboquant.h"
#include "../state/snapkv.h"
#include "../graph_inputs/tokens_input.h"
#include "../graph_inputs/positions_input.h"
#include "../graph_inputs/attn_mask_input.h"
#include "../graph_inputs/gather_indices_input.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <iostream>
#include <cmath>
#include <cinttypes>
#include <memory>
#include <future>
#include <sstream>
#include <thread>

// ── Qwen tokenizer config ─────────────────────────────────────────────────────

TokenizerConfig qwen_tokenizer_config()
{
    TokenizerConfig cfg;
    cfg.normalizer    = NormalizerKind::None;
    cfg.byte_fallback = false;
    cfg.add_bos_token = false;
    // <|im_start|> and <|im_end|> are CONTROL-typed in real Qwen GGUFs and
    // are caught by the generic CONTROL scan.  List them explicitly so the
    // intent is self-documenting and any GGUF that mislabels them still works.
    cfg.extra_chat_specials = {"<|im_start|>", "<|im_end|>"};
    return cfg;
}

constexpr size_t GRAPH_SIZE = 16384;

// Number of consecutive transformer layers fused into one ggml graph compute call.
// Cuts Metal scheduler overhead and hidden-state CPU↔Metal round-trips by this factor.
// Each layer has its own persistent scratch slot for incremental decompression.
static constexpr uint32_t TQ_LAYER_BATCH = 4;

Qwen3ForwardPass::Qwen3ForwardPass(
    const Model& model, const ModelMetadata* metadata,
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

struct ggml_cgraph* Qwen3ForwardPass::build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx, [[maybe_unused]] bool want_logits) {
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

    // 1. Token embedding lookup
    const size_t n_tokens = tokens.size();

    // Token embedding lookup
    ggml_tensor * inpL = embedding(gf, tokens);
    set_tensor_name(gf, inpL, "inpL");

    // Position tensor
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos);  // Mark as input tensor
    set_tensor_name(gf, inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // Typed inputs for this graph (replaces set_inputs). Cleared here so
    // build_output_head can append SparseHeadInput on the sparse path; one
    // uniform causal mask per (all-attention) layer, no sliding window.
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    for (uint32_t il = 0; il < static_cast<uint32_t>(n_layers); ++il)
        graph_inputs_.add(std::make_unique<AttnMaskInput>(
            "kq_mask." + std::to_string(il), 0u));

    // memset(inp_pos->data, 0, ggml_nbytes(inp_pos));
    // for (int i = 0; i < n_tokens; ++i) {
    //     ggml_set_i32_1d(inp_pos, i, pos + i);
    // }

    // 2. Loop through each transformer block
    TransformerBlockHparams blk_hp;
    blk_hp.is_qwen2      = (meta_.architecture == "qwen2");
    blk_hp.n_head        = n_head;
    blk_hp.n_head_kv     = n_head_kv;
    blk_hp.n_embd_head   = n_embd_head;
    blk_hp.freq_base     = meta_.rope_freq_base;
    blk_hp.context_length = static_cast<int>(meta_.context_length);
    blk_hp.rms_norm_eps  = meta_.rms_norm_eps;

    for (uint32_t il = 0; il < static_cast<uint32_t>(n_layers); ++il) {
        auto& block = model_.get_block(il);
        TransformerBlockWeights w;
        w.attn_norm = block.attn_norm_weight;
        w.q         = block.attn_q_weight;
        w.k         = block.attn_k_weight;
        w.v         = block.attn_v_weight;
        w.q_bias    = block.attn_q_bias;
        w.k_bias    = block.attn_k_bias;
        w.v_bias    = block.attn_v_bias;
        w.q_norm    = block.attn_q_norm_weight;
        w.k_norm    = block.attn_k_norm_weight;
        w.out       = block.attn_output_weight;
        w.ffn_norm  = block.ffn_norm_weight;
        w.ffn_gate  = block.ffn_gate_weight;
        w.ffn_up    = block.ffn_up_weight;
        w.ffn_down  = block.ffn_down_weight;

        inpL = build_transformer_layer(ctx_, gf, kv_cache_.get(), inpL, inp_pos,
                                       w, blk_hp, il, slot_idx,
                                       static_cast<uint32_t>(n_tokens));
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

    // Find max physical cache position to determine KV gather length.
    // After SnapKV, positions[] contains logical (RoPE) positions which may be
    // much larger than the compacted cache. Use the physical cache pos instead.
    uint32_t max_physical = 0;
    for (uint32_t s : slots) {
        uint32_t phys = get_physical_cache_pos(s);
        if (phys > max_physical) max_physical = phys;
    }
    uint32_t n_kv_len = max_physical + 1;  // +1 for the new token being written

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

    // Typed inputs for the decode graph (replaces set_batched_inputs).
    graph_inputs_.clear();
    graph_inputs_.add(std::make_unique<TokensInput>());
    graph_inputs_.add(std::make_unique<PositionsInput>());
    graph_inputs_.add(std::make_unique<AttnMaskInput>("kq_mask_b", 0u));
    graph_inputs_.add(std::make_unique<GatherIndicesInput>(
        kv_cache_->get_n_ctx_max()));

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
        cur = build_batched_attention(ctx_, gf, kv_cache_.get(), Qcur, Kcur, Vcur, il, kq_scale, slots, positions, kq_mask, gather_indices, il);

        // E. Output Projection & Residual
        cur = ggml_mul_mat(ctx_, block.attn_output_weight, cur);
        ggml_tensor * ffn_inp = ggml_add(ctx_, cur, inpSA);

        // F. FFN
        cur = build_norm(gf, ffn_inp, block.ffn_norm_weight, il);
        cur = build_ffn_swiglu(ctx_, gf, cur, block.ffn_gate_weight, block.ffn_up_weight, block.ffn_down_weight, il);
        
        // H. FFN Residual
        cur = ggml_add(ctx_, cur, ffn_inp);
        inpL = cur;
    }
    
    // 3. Output Head — shared helper (final norm + LM head). Routes through
    //    build_output_head so the sparse decode path (sparse_decode_ids_ →
    //    ggml_get_rows on the output weight) is honored, matching the prefill
    //    path (line ~182). Dense behavior is unchanged: with no sparse ids
    //    armed this is build_norm + ggml_mul_mat over the full output weight
    //    (or token-embedding fallback), identical to the prior hand-rolled
    //    code. Without this, grammar-constrained decode arms a 29-element
    //    sparse set but the graph returns full-vocab logits → sample_sparse
    //    size-mismatch / out-of-bounds (the Qwen3.5 bad-access class).
    build_output_head(gf, inpL);

    return gf;
}

// _build_attention_layer and _build_batched_attention_layer have been
// extracted to src/layers/attention.cpp as build_attention() and
// build_batched_attention(). Call sites updated to use the free functions.


// set_inputs / set_batched_inputs removed: inputs are now populated by the
// typed GraphInputSet (graph_inputs_) built in build_prefill_graph and
// build_decoding_graph. See docs/plan-typed-graph-inputs.md.

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

    TransformerBlockHparams blk_hp;
    blk_hp.is_qwen2       = (meta_.architecture == "qwen2");
    blk_hp.n_head         = n_head;
    blk_hp.n_head_kv      = n_head_kv;
    blk_hp.n_embd_head    = n_embd_head;
    blk_hp.freq_base      = freq_base;
    blk_hp.context_length = static_cast<int>(meta_.context_length);
    blk_hp.rms_norm_eps   = meta_.rms_norm_eps;

    ggml_tensor* cur = inpL;

    for (uint32_t il = il_start; il < il_end; ++il) {
        auto& block = model_.get_block(il);
        TransformerBlockWeights w;
        w.attn_norm = block.attn_norm_weight;
        w.q         = block.attn_q_weight;
        w.k         = block.attn_k_weight;
        w.v         = block.attn_v_weight;
        w.q_bias    = block.attn_q_bias;
        w.k_bias    = block.attn_k_bias;
        w.v_bias    = block.attn_v_bias;
        w.q_norm    = block.attn_q_norm_weight;
        w.k_norm    = block.attn_k_norm_weight;
        w.out       = block.attn_output_weight;
        w.ffn_norm  = block.ffn_norm_weight;
        w.ffn_gate  = block.ffn_gate_weight;
        w.ffn_up    = block.ffn_up_weight;
        w.ffn_down  = block.ffn_down_weight;

        cur = build_transformer_layer(ctx_, gf, tq_scratch_cache_.get(), cur, inp_pos,
                                      w, blk_hp, il, slot_idx, n_tokens);
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
        set_prefill_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);

        advance_cache(tokens.size(), slot_idx);
        auto logits = get_output_logits(gf);

        if (do_snapkv) {
            // Record the original sequence length BEFORE compaction
            uint32_t original_seq_len = pos + tokens.size();
            apply_snapkv_from_graph(gf, scoring_layer, tokens.size(),
                meta_.block_count, snapkv_budget_, snapkv_window_,
                slot_idx, kv_cache_.get(), nullptr, nullptr);
            // Set logical seq_pos so get_cache_pos() returns the correct
            // RoPE position for subsequent decode tokens.
            snapkv_set_seq_pos(slot_idx, original_seq_len);
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

        {
            StepContext step;
            step.gf = gf_emb;
            step.tokens = &tokens;
            TokensInput().set_input(step);
        }

        ggml_backend_sched_graph_compute(scheduler, gf_emb);

        ggml_tensor* emb_out = ggml_graph_get_tensor(gf_emb, "embed_lookup");
        ggml_backend_tensor_get(emb_out, hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));
    }

    // Sync scratch cache position with TQ store's position tracker
    tq_scratch_cache_->set_pos(tq_store_->get_pos(slot_idx), slot_idx);


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

        // inpL is a hidden-state carrier, not a typed input — direct set
        // (plan scope fence).
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inpL"),
            hidden.data(), 0,
            static_cast<size_t>(hidden_dim) * n_tokens * sizeof(float));

        // Positions + per-layer causal masks: same typed inputs as the
        // non-TQ path. Collapses the 4th duplicated poke copy.
        {
            StepContext step;
            step.gf = gf;
            step.tokens = &tokens;
            step.pos = pos;
            PositionsInput().set_input(step);
            for (uint32_t il = il0; il < il1; ++il)
                AttnMaskInput("kq_mask." + std::to_string(il), 0u)
                    .set_input(step);
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
            uint32_t original_seq_len = pos + n_tokens;

            apply_snapkv_from_graph(gf, scoring_layer, n_tokens,
                n_layers, snapkv_budget_, snapkv_window_,
                slot_idx, nullptr, tq_store_.get(),
                [this](uint32_t s) { _tq_invalidate_watermarks(s); });

            tq_scratch_cache_->set_pos(tq_store_->get_pos(slot_idx), slot_idx);
            snapkv_set_seq_pos(slot_idx, original_seq_len);
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

// ── Inventory validator ──────────────────────────────────────────────────────

void validate_qwen3_inventory(const ModelMetadata& meta)
{
    const auto& inv = meta.tensor_inventory;
    auto require = [&](const std::string& name, const std::string& ctx) {
        if (inv.find(name) == inv.end())
            throw std::runtime_error(
                meta.architecture + ": missing tensor '" + name +
                "': expected in " + ctx + ", got absent");
    };
    require("token_embd.weight", "model weights");
    require("output_norm.weight", "model weights");

    static const std::vector<std::string> per_block = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
        "ffn_up.weight", "ffn_down.weight"
    };
    for (uint32_t i = 0; i < meta.block_count; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        for (const auto& t : per_block)
            require(p + t, "block " + std::to_string(i));
    }
}

