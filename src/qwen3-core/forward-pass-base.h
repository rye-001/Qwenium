#pragma once

#include <vector>
#include <cstdint>
#include <memory>

#include "qwen3-model.h"
#include "../kv-cache/simple-kv-cache.h"
#include "ggml-backend.h"

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

constexpr size_t FP_GRAPH_SIZE_METADATA = 128 * 1024 * 1024;
constexpr size_t FP_GRAPH_SIZE = 16384;

/**
 * Base class for forward pass implementations.
 * 
 * Owns the shared ggml context and buffer. Provides utility methods
 * used by all architectures: embedding lookup, RMS norm, SwiGLU FFN,
 * multi-head attention kernel, output logits extraction.
 *
 * Each architecture subclass owns its own cache(s) and implements
 * its own graph building logic.
 */
class ForwardPassBase {
public:
    ForwardPassBase(const Qwen3Model& model, const Qwen3Metadata* metadata);
    virtual ~ForwardPassBase();

    virtual ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0) = 0;
    virtual void set_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens, int pos) = 0;
    virtual void advance_cache(uint32_t n_tokens, uint32_t slot_idx) = 0;
    virtual void clear_slot(uint32_t slot_idx) = 0;
    virtual void set_cache_pos(uint32_t pos, uint32_t slot_idx) = 0;
    virtual uint32_t get_cache_pos(uint32_t slot_idx) const = 0;
    virtual void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) = 0;
    virtual ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) = 0;

    // ── TurboQuant per-layer compute ────────────────────────────
    // Encapsulates the full prefill pipeline: build → alloc → set → compute → advance.
    // When TQ is enabled, processes one layer at a time with compressed KV.
    // When TQ is disabled, delegates to the existing monolithic path.
    // Returns output logits.
    virtual std::vector<float> run_prefill(
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx,
        ggml_backend_sched_t scheduler) {
        // Default: monolithic path (subclasses override for TQ)
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
        ggml_backend_sched_alloc_graph(scheduler, gf);
        set_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);
        advance_cache(tokens.size(), slot_idx);
        return get_output_logits(gf);
    }

    virtual void set_batched_inputs(ggml_cgraph* gf,
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) = 0;
    
    // --- Output extraction (shared by all architectures) ---
    std::vector<float> get_output_logits(ggml_cgraph* gf);
    std::vector<float> get_output_logits_for_slot(ggml_cgraph* gf, uint32_t slot_index);

    // ── SnapKV configuration ─────────────────────────────────────
    void set_snapkv_config(uint32_t budget, uint32_t window) {
        snapkv_budget_ = budget;
        snapkv_window_ = window;
    }
    uint32_t snapkv_budget() const { return snapkv_budget_; }
    uint32_t snapkv_window() const { return snapkv_window_; }

    // ── SnapKV dual position tracking ──────────────────────────────
    // After eviction, seq_pos tracks the logical sequence position (for RoPE)
    // while the cache's physical position tracks the compacted length.
    // When seq_pos > 0, get_cache_pos() should return seq_pos instead of
    // the physical cache position.

    // Called after SnapKV compaction to set the logical sequence position.
    void snapkv_set_seq_pos(uint32_t slot_idx, uint32_t original_length) {
        if (slot_idx >= snapkv_seq_pos_.size())
            snapkv_seq_pos_.resize(slot_idx + 1, 0);
        snapkv_seq_pos_[slot_idx] = original_length;
    }

    // Advance the logical sequence position (call alongside advance_cache).
    void snapkv_advance_seq_pos(uint32_t slot_idx, uint32_t n_tokens) {
        if (slot_idx < snapkv_seq_pos_.size() && snapkv_seq_pos_[slot_idx] > 0)
            snapkv_seq_pos_[slot_idx] += n_tokens;
    }

    // Reset seq_pos tracking for a slot (call on clear_slot).
    void snapkv_clear_seq_pos(uint32_t slot_idx) {
        if (slot_idx < snapkv_seq_pos_.size())
            snapkv_seq_pos_[slot_idx] = 0;
    }

    // Get the logical sequence position (0 = SnapKV not active for this slot).
    uint32_t snapkv_get_seq_pos(uint32_t slot_idx) const {
        return (slot_idx < snapkv_seq_pos_.size()) ? snapkv_seq_pos_[slot_idx] : 0;
    }

protected:
    const Qwen3Metadata& meta_;
    const Qwen3Model& model_;
    struct ggml_context* ctx_;
    std::vector<uint8_t> ctx_buffer_;

    // SnapKV: post-prefill KV eviction (0 = disabled)
    uint32_t snapkv_budget_ = 0;
    uint32_t snapkv_window_ = 32;

    // Per-slot logical sequence position after SnapKV compaction.
    // 0 = SnapKV not active (use physical cache position).
    std::vector<uint32_t> snapkv_seq_pos_;

    // --- Context management ---
    // Reset the ggml context (call at the start of every graph build)
    void reset_context();

    // Create a new graph
    ggml_cgraph* new_graph();

    // --- Shared graph-building primitives ---

    ggml_tensor* embedding(ggml_cgraph* gf, const std::vector<int32_t>& tokens);

    ggml_tensor* build_norm(
        ggml_cgraph* gf,
        ggml_tensor* cur,
        ggml_tensor* mw,
        int il) const;

    ggml_tensor* ffn_swiglu(
        ggml_cgraph* gf,
        ggml_tensor* cur,
        ggml_tensor* gate,
        ggml_tensor* up,
        ggml_tensor* down,
        int il);

    // Core multi-head attention: Q @ K^T → softmax → @ V
    // Handles GQA, permutations, stream splitting, and recombination.
    ggml_tensor* build_attn_mha(
        ggml_cgraph* gf,
        ggml_tensor* q,
        ggml_tensor* k,
        ggml_tensor* v,
        ggml_tensor* kq_mask,
        ggml_tensor* sinks,
        float kq_scale,
        uint32_t pos,
        int il) const;

    // Build the output head: final norm → LM head matmul → "logits" tensor
    void build_output_head(ggml_cgraph* gf, ggml_tensor* cur);

    void set_tensor_name(ggml_cgraph* gf, ggml_tensor* tensor, const char* name, int il = -1) const;
};