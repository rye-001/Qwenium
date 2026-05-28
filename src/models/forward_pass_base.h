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

    virtual void clone_slot(uint32_t src_slot, uint32_t dst_slot, uint32_t n_tokens) = 0;
    virtual ggml_cgraph* build_decoding_graph(
        const std::vector<int32_t>& tokens,
        const std::vector<uint32_t>& slots,
        const std::vector<int32_t>& positions) = 0;

    // False ⇒ this recipe has no build_decoding_graph (it throws); decode_step
    // must route through the legacy single-token run_prefill bridge instead.
    // Default true; recipes whose build_decoding_graph is unimplemented
    // (Gemma 1–4) override to false.
    virtual bool has_decode_graph() const { return true; }

    // Encapsulates the full prefill pipeline: build → alloc → set → compute → advance.
    // Returns output logits.
    virtual std::vector<float> run_prefill(
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx,
        ggml_backend_sched_t scheduler) {
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
    // State mutation, no return value.
    //
    // Contract: token-stable across span vs. sequential decode — the sampled
    // token never flips, low FP bits may differ. See plan-feed-tokens.md.
    //
    // Standalone at the API; thin parameterization underneath: the internal
    // impl is the existing prefill builder with want_logits=false (exactly
    // one head-guard site per recipe). NOT a forked KV/recurrent write.
    void feed_tokens(const std::vector<int32_t>& tokens, uint32_t slot,
                     ggml_backend_sched_t scheduler) {
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

    // Prefill-only token-position slice. Inserts a ggml_get_rows on the hidden
    // state immediately before the LM head so the ~150k-wide head runs only on
    // the position(s) that produce logits (default: last token). Registers an
    // OutputIdsInput owning the "out_ids" slot. Returns `cur` unchanged when
    // the slice is disabled (the differential dense-reference seam) — this is
    // an explicit, caller-selected path, NOT a silent fallback on error.
    //
    // Orthogonal to the vocab-axis sparse slice in build_output_head: this
    // gathers the hidden state, that gathers the head weight. Both compose
    // order-independently in one graph. Called at the single per-recipe
    // prefill head site only — never from build_decoding_graph (decode with
    // n_tokens=1 is already trivially sliced).
    ggml_tensor* build_out_ids_slice(ggml_cgraph* gf, ggml_tensor* cur);

    void set_tensor_name(ggml_cgraph* gf, ggml_tensor* tensor, const char* name, int il = -1) const;

public:
    // ── Sparse decode support ────────────────────────────────────────────────
    // Set the host-side valid token indices for the next decode step.
    // Pass an empty vector to use the dense path (default).
    // Must be called before build_decoding_graph.
    void set_sparse_decode_ids(std::vector<int32_t> ids) {
        sparse_decode_ids_ = std::move(ids);
    }

    // Differential seam for the prefill output-position slice. Default true:
    // prefill builds the LM head only on the last token (the optimization).
    // Set false to build the dense head over all N positions — the bit-for-bit
    // reference the slice differential compares against. Explicit and
    // caller-selected; not an error fallback (CLAUDE.md fail-loud contract).
    void set_slice_prefill_head(bool on) { slice_prefill_head_ = on; }
    bool slice_prefill_head() const { return slice_prefill_head_; }

protected:
    bool slice_prefill_head_ = true;
};