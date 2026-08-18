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
class DeltaNetState;  // L2 snapshot reach-through (OverwriteRecurrent lane)

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
        return run_prefill(tokens, pos, slot_idx, scheduler, nullptr);
    }

    // Same pipeline, additionally capturing the D3 "hidden_out" output (all
    // positions) when the caller wants it. hidden_out non-null requires
    // set_output_hidden(true) before the call — get_output_hidden fails loud
    // otherwise. The MTP drafting loop is the consumer.
    std::vector<float> run_prefill(
        const std::vector<int32_t>& tokens,
        int pos, uint32_t slot_idx,
        ggml_backend_sched_t scheduler,
        std::vector<float>* hidden_out) {
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
        ggml_backend_sched_alloc_graph(scheduler, gf);
        set_prefill_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);
        advance_cache(tokens.size(), slot_idx);
        if (hidden_out) *hidden_out = get_output_hidden(gf);
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

    // ── L2 session snapshot reach-through ────────────────────────────────────
    // Accessors (not logic) exposing the per-sequence state an L2 snapshot
    // serializes: the AppendKV cache and the OverwriteRecurrent state. A recipe
    // returns its own member, or nullptr if it has none (e.g. pure-attention
    // recipes have no recurrent state). The L2 manifest borrows these to build a
    // KvCacheSection / DeltaNetStateSection. Default nullptr so a recipe without
    // a snapshot accessor is treated fail-loud (no silent empty section) by the
    // caller rather than silently snapshotting nothing.
    virtual simple_kv_cache* snapshot_kv_cache() { return nullptr; }
    virtual DeltaNetState*   snapshot_recurrent() { return nullptr; }

    // Multi-cache reach-through: the ordered list of AppendKV caches an L2
    // snapshot serializes. Single-cache recipes (Qwen35, Gemma3) inherit this
    // default, which wraps snapshot_kv_cache(). Recipes with more than one
    // physical KV cache (Gemma4 — disjoint sliding + global caches) OVERRIDE it
    // to return all of them IN A FIXED ORDER; the manifest adds one
    // KvCacheSection per cache, matched positionally on restore. The order MUST
    // be identical between capture and restore. Empty when the recipe has no L2
    // support. This is the load-bearing generalization that lets Gemma4's dual
    // cache host L2/V2 without bending the single-cache recipes.
    virtual std::vector<simple_kv_cache*> snapshot_kv_caches() {
        simple_kv_cache* one = snapshot_kv_cache();
        if (one) return {one};
        return {};
    }

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

    // ── D3: opt-in hidden-state output (docs/plan-mtp-decode.md §5 D3) ────────
    // When on, the recipe marks the pre-final-norm residual tip (all positions)
    // as a graph output named "hidden_out" — the input the MTP head conditions
    // on. Default OFF ⇒ the graph is node-for-node what it is today (marking an
    // existing node as an output adds no compute), so the MTP-off byte-identity
    // gate holds. Turned on only when the active draft source
    // needs_hidden_state(). A no-op on recipes that don't read it (Gemma), so
    // their byte gate is unaffected. Honoured by qwen36's prefill AND decode
    // graphs (the Phase-3 "prefill + batched decode" scope decision).
    void set_output_hidden(bool on) { output_hidden_ = on; }
    bool output_hidden() const { return output_hidden_; }
    std::vector<float> get_output_hidden(ggml_cgraph* gf);

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
    //
    // gemma_final_norm: false (default) → standard final norm (x * w), which is
    //   byte-identical for Qwen and every non-Gemma recipe. true → Gemma's
    //   (x / rms(x)) * (1 + w) form (build_rms_norm_gemma). Gemma stores the
    //   final-norm weight as a delta-from-unity, so the (1+w) form is required
    //   for the decode head to match the Gemma recipe's prefill final norm
    //   bit-for-bit. This is a knob on the head, mirroring TransformerBlockHparams
    //   ::gemma_rms_norm on the per-layer norm — not a forked head.
    void build_output_head(ggml_cgraph* gf, ggml_tensor* cur,
                           ggml_tensor* valid_idx = nullptr,
                           bool gemma_final_norm = false,
                           float final_softcap = 0.0f);

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

    // Substitute a span of precomputed image-token embeddings into the scaled
    // residual stream at columns [span_start, span_start + n_img). Shared by
    // every vision recipe (Gemma 3, Gemma 4): one graph shape, one fail-loud
    // validation, and the galloc-output pin that keeps multi-request image
    // prefill deterministic (docs/server-image-multirequest-bug.md). Operates on
    // a generic float span — it does NOT know how the embeddings were produced;
    // vision stays out of the base (see i_image_embeddable.h). Consumes `embd`
    // (moved into the registered ImageEmbeddingsInput). Returns the new residual
    // root, named "inpL_image_subst" and pinned as a graph output.
    //
    // Image rows enter UNSCALED: the recipe must call this AFTER its
    // sqrt(d_model) embed scale, so ggml_set_2d overwrites the scaled text
    // columns wholesale (llama.cpp gemma3: scale = ubatch.token ? sqrt : 1.0).
    // Recipe-specific concerns (e.g. Gemma 3's bidirectional image mask) stay in
    // the recipe; this is only the residual-stream substitution.
    ggml_tensor* build_image_substitution(
        ggml_cgraph* gf,
        ggml_tensor* inpL,
        std::vector<float>&& embd,
        int32_t span_start,
        uint32_t n_img,
        int hidden_dim,
        size_t n_tokens);

    // Build the batched-decode KQ masks, deduplicated by window value. Qwen and Gemma 1
    // already use a single mask, so they are unaffected; this is the Gemma
    // 2/3/4 sliding-window
    std::vector<ggml_tensor*> build_decode_layer_masks(
        ggml_cgraph* gf,
        const std::vector<uint32_t>& layer_windows,
        uint32_t n_kv_len, uint32_t n_tokens);

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

    // ── Decode KV write mode ─────────────────────────────────────────────────
    // Differential seam for the decode KV write, mirroring slice_prefill_head:
    // Cpy (default) → the legacy baked-offset ggml_cpy write — today's decode,
    // byte-reproducible; SetRows → the value-driven ggml_set_rows write whose
    // position is a graph input; persistent-graph capable — see
    // docs/plan-persistent-decode-graph.md §2.1. Byte-identical to Cpy at exact
    // n_kv (P1 gate); the persistent path turns it ON together with bucketing.
    // Only recipes that pass kv_write_indices into the batched attention
    // helpers honor it; others are Cpy-only regardless. Explicit and
    // caller-selected; not a fallback.
    enum class KvWriteMode { SetRows, Cpy };
    void set_kv_write_mode(KvWriteMode m) { kv_write_mode_ = m; }
    KvWriteMode kv_write_mode() const { return kv_write_mode_; }

    // True ⇒ this recipe's build_decoding_graph is persistent-graph capable:
    // every step-varying quantity is a graph-input VALUE (tokens, positions,
    // mask, gather indices, KV write rows), so a built+allocated decode graph
    // can be recomputed across steps without rebuild. Consumed by the P3
    // DecodeGraphCache; false keeps the recipe on the rebuild-per-step path.
    virtual bool supports_persistent_decode() const { return false; }

    // ── Decode n_kv bucketing ────────────────────────────────────────────────
    // Bucket B ⇒ converted recipes size the decode graph's KV read width
    // (mask / gather / gathered views) at the next multiple of B instead of
    // exactly max_pos+1, capped at the cache's n_ctx_max — so one graph shape
    // (hence one allocation) stays valid across a whole bucket of steps: the
    // persistent-graph precondition (plan-persistent-decode-graph.md §2.2).
    // Padded columns are −inf-masked and read zero-initialized cache rows.
    // DEFAULT 0 = exact sizing (today's decode; byte-reproducible). Bucketing
    // is NOT byte-identical to exact: widening n_kv re-blocks the softmax /
    // scores·V reduction, so it is token-stable-modulo-ties, not bit-identical
    // (test_decode_kv_bucket; same fork class as architecture.md §11). The
    // persistent path opts IN to bucket 256; the default decode path stays
    // exact. Only recipes that call decode_kv_len() honor it.
    void set_decode_kv_bucket(uint32_t b) { decode_kv_bucket_ = b; }
    uint32_t decode_kv_bucket() const { return decode_kv_bucket_; }

protected:
    // Bucketed decode KV width: max_pos_plus_1 rounded up to the bucket,
    // capped at n_ctx_max. Bucket 0 ⇒ exact (max_pos_plus_1 unchanged).
    uint32_t decode_kv_len(uint32_t max_pos_plus_1, uint32_t n_ctx_max) const {
        if (decode_kv_bucket_ == 0) return max_pos_plus_1;
        const uint64_t up =
            (static_cast<uint64_t>(max_pos_plus_1) + decode_kv_bucket_ - 1) /
            decode_kv_bucket_ * decode_kv_bucket_;
        return up < n_ctx_max ? static_cast<uint32_t>(up) : n_ctx_max;
    }

    bool slice_prefill_head_ = true;
    bool output_hidden_ = false;   // D3 opt-in; see set_output_hidden
    // Defaults = today's decode: baked-offset cpy write, exact per-step n_kv.
    // The opt-in persistent path (--persistent-graph) sets both to
    // {SetRows, 256} together; nothing else changes them.
    KvWriteMode kv_write_mode_ = KvWriteMode::Cpy;
    uint32_t decode_kv_bucket_ = 0;
};