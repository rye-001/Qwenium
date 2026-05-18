#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "../core/model.h"
#include "../state/kv_cache_simple.h"
#include "../graph_inputs/graph_input.h"
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
    ForwardPassBase(const Model& model, const ModelMetadata* metadata);
    virtual ~ForwardPassBase();

    // want_logits=false prunes the LM head (the single per-recipe head-guard
    // site) so the graph advances state only — see feed_tokens and
    // docs/plan-feed-tokens.md. State-write roots are independent graph roots;
    // pruning the head leaves them intact.
    virtual ggml_cgraph* build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx = 0, bool want_logits = true) = 0;
    virtual void advance_cache(uint32_t n_tokens, uint32_t slot_idx) = 0;
    virtual void clear_slot(uint32_t slot_idx) = 0;
    virtual void set_cache_pos(uint32_t pos, uint32_t slot_idx) = 0;
    virtual uint32_t get_cache_pos(uint32_t slot_idx) const = 0;

    // Physical cache position (KV write offset / attention entry count).
    // Unlike get_cache_pos(), this always returns the physical position even
    // when SnapKV seq_pos tracking is active. Used for KV gather sizing.
    virtual uint32_t get_physical_cache_pos(uint32_t slot_idx) const = 0;
    virtual void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) = 0;
    virtual ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) = 0;

    // True when TurboQuant KV compression is active for this forward pass.
    // BRIDGE (Option 1 pending): build_decoding_graph has no TurboQuant
    // support — under TQ the full kv_cache_ is intentionally null. decode_step
    // queries this to refuse the TQ-unaware decode graph and route to the
    // legacy per-recipe single-token run_prefill path (which IS TQ-aware).
    // Default false; recipes that allocate a compressed store override it.
    virtual bool tq_active() const { return false; }

    // False ⇒ this recipe has no build_decoding_graph (it throws); decode_step
    // must route through the legacy single-token run_prefill bridge instead.
    // Default true; recipes whose build_decoding_graph is unimplemented
    // (Gemma 1–4) override to false.
    virtual bool has_decode_graph() const { return true; }

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
        set_prefill_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);
        advance_cache(tokens.size(), slot_idx);
        return get_output_logits(gf);
    }

    // ── feed_tokens ──────────────────────────────────────────────
    // Advance this slot's model state — attention KV-append AND recurrent
    // overwrite — over a span of already-known tokens WITHOUT building the
    // LM head or producing logits. The inverse of run_prefill-with-logits:
    // consume tokens to condition future predictions; predict nothing.
    // State mutation, no return value. See docs/plan-feed-tokens.md.
    //
    // Standalone at the API; thin parameterization underneath: the internal
    // impl is the existing prefill builder with want_logits=false (exactly
    // one head-guard site per recipe). NOT a forked KV/recurrent write.
    //
    // TurboQuant is scoped OUT:
    // feed_tokens onto a non-empty mid-decode cache under TQ is a
    // compressed-store write path that is deliberately NOT folded into this
    // primitive. It fails loud rather than silently corrupting compressed
    // state, mirroring the existing decode bridge in decode_step.cpp.
    void feed_tokens(const std::vector<int32_t>& tokens, uint32_t slot,
                     ggml_backend_sched_t scheduler) {
        if (tq_active())
            throw std::runtime_error(
                "feed_tokens: tq_active expected=false actual=true — "
                "TurboQuant is scoped out (docs/plan-feed-tokens.md TQ "
                "clause, option 2). feed_tokens onto a non-empty mid-decode "
                "cache under TQ is a compressed-store write path not folded "
                "into this primitive; refusing rather than corrupting state.");
        if (!feed_tokens_supported())
            throw std::runtime_error(
                "feed_tokens: feed_tokens_supported expected=true "
                "actual=false — this recipe has no want_logits=false head "
                "guard yet. docs/plan-feed-tokens.md is phased (qwen36 "
                "first); refusing rather than silently building the head.");

        const int pos = static_cast<int>(get_cache_pos(slot));

        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf =
            build_prefill_graph(tokens, pos, slot, /*want_logits=*/false);
        ggml_backend_sched_alloc_graph(scheduler, gf);
        set_prefill_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);
        advance_cache(static_cast<uint32_t>(tokens.size()), slot);
        // No get_output_logits: head-less by contract.
    }

    // True only for recipes whose build_prefill_graph honors want_logits=false
    // with exactly one head-guard site. Phased per docs/plan-feed-tokens.md
    // (qwen36 first). Default false so feed_tokens fails loud for recipes
    // that have not implemented the guard, rather than silently building the
    // head (wasteful) or shipping an unverified state-advance path.
    virtual bool feed_tokens_supported() const { return false; }

    // Typed inputs the most-recently-built graph owns. Populated by the
    // recipe's build_*_graph; consumed by run_prefill / decode_step via
    // StepContext.
    GraphInputSet& graph_inputs() { return graph_inputs_; }

    // Populate the current prefill graph's typed inputs (contiguous
    // positions: row r -> pos + r). sparse_decode_ids_ is consumed on use
    // (clear mirrors the former upload_sparse_indices semantics).
    void set_prefill_inputs(ggml_cgraph* gf,
        const std::vector<int32_t>& tokens, int pos) {
        StepContext step;
        step.gf         = gf;
        step.tokens     = &tokens;
        step.pos        = pos;
        step.sparse_ids = sparse_decode_ids_.empty()
            ? nullptr : &sparse_decode_ids_;
        graph_inputs_.set_input(step);
        sparse_decode_ids_.clear();
    }

    // Populate the current decode graph's typed inputs for a batched step.
    void set_decode_inputs(ggml_cgraph* gf,
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) {
        StepContext step;
        step.gf         = gf;
        step.tokens     = &tokens;
        step.positions  = &positions;
        step.slots      = &slots;
        step.sparse_ids = sparse_decode_ids_.empty()
            ? nullptr : &sparse_decode_ids_;
        graph_inputs_.set_input(step);
        sparse_decode_ids_.clear();
    }

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
    const ModelMetadata& meta_;
    const Model& model_;
    struct ggml_context* ctx_;
    std::vector<uint8_t> ctx_buffer_;

    // Typed inputs for the current graph. Each recipe rebuilds this in its
    // build_*_graph; run_prefill / decode_step fan set_input over it.
    GraphInputSet graph_inputs_;

    // Sparse decode: host-side valid token ids armed before graph build.
    // build_output_head registers a SparseHeadInput when this is non-empty;
    // set_prefill_inputs / set_decode_inputs upload it via StepContext and
    // clear it (consume-on-use).
    std::vector<int32_t> sparse_decode_ids_;

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
    // valid_idx: [k] int32 input tensor selecting which vocab rows to compute;
    // nullptr = dense (full vocab). Use when the caller knows the candidate set
    // ahead of the forward pass and wants to avoid materializing the full vocab matmul.
    // If valid_idx is nullptr and sparse_decode_ids_ is non-empty, the tensor is
    // created automatically from sparse_decode_ids_ (set by set_sparse_decode_ids).
    void build_output_head(ggml_cgraph* gf, ggml_tensor* cur, ggml_tensor* valid_idx = nullptr);

    void set_tensor_name(ggml_cgraph* gf, ggml_tensor* tensor, const char* name, int il = -1) const;

public:
    // ── Sparse decode support ────────────────────────────────────────────────
    // Set the host-side valid token indices for the next decode step.
    // Pass an empty vector to use the dense path (default).
    // Must be called before build_decoding_graph.
    void set_sparse_decode_ids(std::vector<int32_t> ids) {
        sparse_decode_ids_ = std::move(ids);
    }
};