#pragma once

#include <functional>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "engine/model.h"
#include "../state/kv_cache_simple.h"
#include "../graph_inputs/graph_input.h"
#include "../engine/graph_compute.h"
#include "graph_arena.h"
#include "decode_policy.h"
#include "ggml-backend.h"

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
class DeltaNetState;  // L2 snapshot reach-through (OverwriteRecurrent lane)

// forward_pass_base.h — the base every model recipe derives from.
//
// Responsibility: own the per-forward-pass ggml context and buffer, provide the
//   graph-building scaffolding every recipe shares (embedding lookup, output
//   head, the Seam B image splice), and declare the interface the engine drives
//   a recipe through. Recipes own their own state (KV cache, recurrent state)
//   and their own layer composition; this class owns neither.
// Public surface, in four groups:
//   (1) the recipe interface — build_prefill_graph / build_decoding_graph /
//       advance_cache / clear_slot / set_cache_pos / clone_slot, plus the
//       run_prefill and feed_tokens drivers built on them;
//   (2) per-slot rope coordinate (get_rope_pos) — NOT the KV row count; an
//       M-RoPE image span consumes nx*ny rows while advancing position by
//       max(nx, ny), so the two diverge and every decode site wants the former;
//   (3) opt-in, default-off output seams — hidden state (MTP, plan-mtp-decode.md
//       §5 D3), attention rows (the lens tap, plan-qemmi-lens.md P1/A1), sparse
//       LM head ids, and the L2 snapshot reach-through;
//   (4) decode-path policy the engine sets — KV write mode (baked-offset cpy vs
//       value-driven set_rows) and n_kv bucketing, which move together and only
//       under --persistent-graph.
// Invariants:
//   - The opt-in seams are byte-inert when disarmed: an empty tap set marks no
//     node, output_hidden_ adds no graph output. Gated by
//     tests/unit/test_forward_pass_base.cpp (TapOffByteIdentical).
//   - graph_inputs_ must be cleared BEFORE build_image_substitution, never
//     after — clearing after discards the ImageEmbeddingsInput that uploads the
//     encoder output while leaving the splice in place, so the image span
//     carries stale buffer contents and the model describes noise. That was the
//     qwen36 vision bug; set_prefill_inputs now refuses it fail-loud
//     (GraphInputSet::has_slot).
//   - Architecture direction: this is a shared base class, and the blueprint
//     wants composition over inheritance — a known eventual deletion target
//     (architecture.md §12).
// Unit test: tests/unit/test_forward_pass_base.cpp
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
        qinf::engine::require_compute_success(
            ggml_backend_sched_graph_compute(scheduler, gf), "run_prefill");
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
    // `pos_override` (P4): the rope position this batch starts at. Default -1
    // keeps the historical behaviour of deriving it from the KV row count.
    //
    // Those two were always the same number until M-RoPE: an image span
    // occupies nx·ny KV rows but advances the position by only max(nx, ny), so
    // after an image the cache length and the rope coordinate DIVERGE and
    // get_cache_pos is no longer a valid position source. Callers that prefill
    // across an image must pass the position they are tracking.
    // ── Per-slot rope coordinate ─────────────────────────────────────────────
    //
    // The KV row count and the rope position were the same number until M-RoPE.
    // An image span occupies nx·ny rows but advances the position by only
    // max(nx, ny), so from the first image onward `get_cache_pos()` answers
    // "how many rows" and NOT "what position comes next". Anything feeding a
    // rotation angle must ask here instead.
    //
    // Stored as a per-slot ROW-MINUS-POSITION DELTA rather than an absolute
    // counter, because the delta is self-maintaining: every ordinary token
    // advances rows and positions by exactly one, so only an image span ever
    // changes it. Absent/zero delta ⇒ get_rope_pos() == get_cache_pos(), which
    // is every recipe and every text path — byte-identical, no new bookkeeping.
    //
    // KNOWN LIMIT: a slot truncated back *into* or *before* an image span keeps
    // the stale delta. The prefix-cache / snapshot paths must carry the rope
    // coordinate to be VL-safe (docs/plan-qwen35-vision-impl.md §3.2, §4
    // decision 3); until they do, VL sessions are single-turn-safe only.
    int32_t get_rope_pos(uint32_t slot) const {
        const int32_t rows = static_cast<int32_t>(get_cache_pos(slot));
        const RopeDivergence* rec = live_rope_record(slot);
        return rec ? rows - rec->delta : rows;
    }

    // Record that a span just written to `slot` occupied `n_rows` KV rows while
    // advancing the sequence position by only `n_pos`. Called by the multimodal
    // orchestrator, the only place that knows the two differ.
    void note_span_rows_vs_positions(uint32_t slot, uint32_t n_rows,
                                     uint32_t n_pos) {
        if (n_pos > n_rows)
            throw std::runtime_error(
                "note_span_rows_vs_positions: slot 'n_pos': expected <= n_rows ("
                + std::to_string(n_rows) + "), got: " + std::to_string(n_pos));
        if (n_rows == n_pos) return;  // scalar-position recipe: nothing to track
        // Drop a record the slot has already outlived BEFORE accumulating onto
        // it. The writer must apply the same staleness test as the readers: a
        // server slot is cleared between image requests, so without this the
        // second image's delta lands on top of the first image's and every
        // decode position after it is ~n_rows too low — coherent request #1,
        // token soup from request #2 on. `+=` is still right within one live
        // history (a future multi-image turn accumulates span by span).
        RopeDivergence& rec = mutable_rope_record(slot);
        rec.delta += static_cast<int32_t>(n_rows) - static_cast<int32_t>(n_pos);
        rec.rows_after = static_cast<int32_t>(get_cache_pos(slot)) +
                         static_cast<int32_t>(n_rows);
    }

    // True when this slot's rows and rope positions have diverged — i.e. it has
    // hosted an image span. Snapshot / prefix-cache paths ask before persisting
    // or restoring: the blob format carries a row count and no rope coordinate,
    // so a diverged slot cannot be round-tripped faithfully (plan §4 decision 3).
    bool has_rope_divergence(uint32_t slot) const {
        return live_rope_record(slot) != nullptr;
    }

    // Drop a slot's divergence record explicitly. get_rope_pos also self-heals
    // (see above), so this is belt-and-braces for callers that clear a slot.
    void reset_rope_pos(uint32_t slot) { rope_row_delta_.erase(slot); }

    void feed_tokens(const std::vector<int32_t>& tokens, uint32_t slot,
                     ggml_backend_sched_t scheduler, int pos_override = -1) {
        if (!feed_tokens_supported())
            throw std::runtime_error(
                "feed_tokens: feed_tokens_supported expected=true "
                "actual=false — this recipe has no want_logits=false head "
                "guard yet. docs/plan-feed-tokens.md is phased (qwen36 "
                "first); refusing rather than silently building the head.");

        const int pos = pos_override >= 0
            ? pos_override
            : static_cast<int>(get_cache_pos(slot));

        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf =
            build_prefill_graph(tokens, pos, slot, /*want_logits=*/false);
        ggml_backend_sched_alloc_graph(scheduler, gf);
        set_prefill_inputs(gf, tokens, pos);
        qinf::engine::require_compute_success(
            ggml_backend_sched_graph_compute(scheduler, gf), "feed_tokens");
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
        // build_image_substitution registers the ImageEmbeddingsInput that
        // uploads the encoder output into the "image_embeddings" slot. A recipe
        // that calls graph_inputs_.clear() AFTER the splice silently discards
        // it: the graph still has the tensor and the splice still overwrites the
        // residual stream with it, but nothing ever fills it, so the image span
        // carries whatever the buffer held. The model then reads noise and says
        // so confidently, with no error anywhere in the engine.
        //
        // That was the qwen36 P4 bug (docs/plan-qwen35-vision-impl.md §9) and it
        // survived weeks of investigation precisely because every component was
        // correct in isolation. Refuse it instead.
        if (image_spliced_ && !graph_inputs_.has_slot("image_embeddings"))
            throw std::runtime_error(
                "set_prefill_inputs: slot 'image_embeddings': expected the "
                "ImageEmbeddingsInput registered by build_image_substitution to "
                "still be present at set_input time, got: absent (the recipe "
                "called graph_inputs_.clear() AFTER the image splice; clear "
                "before it)");
        image_spliced_ = false;

        StepContext step;
        step.gf         = gf;
        step.tokens     = &tokens;
        step.pos        = pos;
        step.img_grid_w = mrope_img_grid_w_;
        step.kv_base    = mrope_kv_base_;
        step.sparse_ids = sparse_decode_ids_.empty()
            ? nullptr : &sparse_decode_ids_;
        graph_inputs_.set_input(step);
        sparse_decode_ids_.clear();
        mrope_img_grid_w_ = 0;  // consume-on-use, like sparse_decode_ids_
        mrope_kv_base_    = -1;
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
    void set_output_hidden(bool on) { policy_.output_hidden = on; }
    bool output_hidden() const { return policy_.output_hidden; }
    std::vector<float> get_output_hidden(ggml_cgraph* gf);

    // ── Lens tap: opt-in attention-row output (docs/plan-qemmi-lens.md P1/A1) ─
    // The Qemmi-Lens trust layer reads each tapped attention layer's
    // post-softmax row `kq_soft.<il>` (the distribution over KV positions for
    // the query token). Those tensors are already NAMED by layers/attention.cpp
    // on every recipe, so this seam is recipe-agnostic (Gemma hosts it inert)
    // and requires zero layer/recipe edits — the same "mark an existing node as
    // an output" trick as D3 set_output_hidden above. Default (empty layer set)
    // ⇒ nothing is marked ⇒ the graph is node-for-node what it is today, so the
    // tap-off byte-identity gate holds.
    //
    // Usage mirrors the QDOCS probe sequence, caller-driven so no recipe method
    // changes: build_decoding_graph(...) → mark_attention_taps(gf) → alloc →
    // set_decode_inputs → compute → get_attention_taps(gf). Marking must happen
    // after build and before graph alloc (galloc would otherwise reuse the
    // buffer). Single query token per call (decode); the row is [n_kv, n_head].
    struct AttentionTap {
        int layer;                 // attention layer index (il), as requested
        int n_kv;                  // KV positions in the row (= tensor ne[0])
        int n_head;                // attention heads (= tensor ne[2])
        std::vector<float> rows;   // row-major [n_head][n_kv]; row h sums to ~1
    };
    void set_attention_taps(std::vector<int> layers) { policy_.attention_taps = std::move(layers); }
    const std::vector<int>& attention_taps() const { return policy_.attention_taps; }
    // Mark each armed layer's `kq_soft.<il>` as a graph output on `gf`. No-op
    // when the layer set is empty (byte-inert). Fail-loud if an armed layer's
    // tap tensor is absent from the graph (names the layer, expected, actual).
    void mark_attention_taps(ggml_cgraph* gf);
    // Read the marked rows back after compute. Result[i] corresponds to
    // attention_taps()[i]. Fail-loud if a tap tensor is missing (the caller
    // forgot mark_attention_taps before alloc).
    std::vector<AttentionTap> get_attention_taps(ggml_cgraph* gf);

protected:
    const ModelMetadata& meta_;
    const Model& model_;
    // Composition, not inheritance: the context lifecycle is a value this class
    // HOLDS. First extraction toward retiring this base class entirely
    // (architecture.md §12). Recipes reach the context as arena_.ctx().
    GraphArena arena_;

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
    // P4 (docs/plan-qwen35-vision-impl.md) — set by a recipe's
    // build_prefill_graph when the batch it just built IS an image chunk and
    // the recipe uses M-RoPE; the value is the image's soft-token grid width.
    // Read and cleared by set_prefill_inputs. 0 = ordinary text batch, which is
    // every batch for every recipe that does not set it.
    // Per-slot rows-minus-positions divergence; see get_rope_pos. Empty for
    // every text-only session, which is why the scalar path is untouched.
    // `rows_after` is the slot's row count just after the span was recorded, so
    // a later rewind past it can invalidate the record instead of corrupting
    // every subsequent position. Mutable: get_rope_pos is const and self-heals.
    struct RopeDivergence {
        int32_t delta      = 0;  // rows - positions accumulated on this slot
        int32_t rows_after = 0;  // row count the delta was recorded at
    };
    mutable std::unordered_map<uint32_t, RopeDivergence> rope_row_delta_;

    // The ONE staleness test, so no reader or writer can apply a record the
    // slot has already outlived. A record describes a specific stretch of
    // history; if the slot has since been cleared or rewound to before
    // `rows_after`, that history is gone. Applying the delta anyway hands back
    // a position BELOW where the slot actually is (negative, after a clear),
    // and accumulating onto it doubles the divergence. Returns null once the
    // record is dropped — rows == positions, exactly right for a freshly
    // cleared slot and the pre-M-RoPE behaviour otherwise.
    const RopeDivergence* live_rope_record(uint32_t slot) const {
        const auto it = rope_row_delta_.find(slot);
        if (it == rope_row_delta_.end()) return nullptr;
        if (static_cast<int32_t>(get_cache_pos(slot)) < it->second.rows_after) {
            rope_row_delta_.erase(it);
            return nullptr;
        }
        return &it->second;
    }

    // Writer-side counterpart: the live record for this slot, or a fresh zeroed
    // one if there is none (or the previous one was stale and just dropped).
    RopeDivergence& mutable_rope_record(uint32_t slot) {
        live_rope_record(slot);          // drops a stale record if present
        return rope_row_delta_[slot];    // live one, or a fresh zeroed one
    }

    // Set by build_image_substitution, consumed by set_prefill_inputs, cleared
    // by reset_context. Exists only so the guard above can tell "this graph
    // spliced an image" from "this is a text graph".
    bool image_spliced_ = false;

    uint32_t mrope_img_grid_w_ = 0;

    // KV rows already written for the slot this prefill targets, when that
    // differs from the rope position (i.e. after an image span under M-RoPE).
    // -1 = they coincide, which is the case for every other recipe and every
    // text-only batch. Set by build_prefill_graph, consumed by
    // set_prefill_inputs alongside mrope_img_grid_w_.
    int mrope_kv_base_ = -1;

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
    void set_slice_prefill_head(bool on) { policy_.slice_prefill_head = on; }
    bool slice_prefill_head() const { return policy_.slice_prefill_head; }

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
    // The enum lives on DecodePolicy; aliased here so existing call sites keep
    // spelling it ForwardPassBase::KvWriteMode.
    using KvWriteMode = DecodePolicy::KvWriteMode;
    void set_kv_write_mode(KvWriteMode m) { policy_.kv_write_mode = m; }
    KvWriteMode kv_write_mode() const { return policy_.kv_write_mode; }

    // ── Attention implementation ─────────────────────────────────────────────
    // Materialized (default) → kq / kq_soft / kqv are real tensors, the
    // byte-reproducible path the receipts identity is defined on. Flash → one
    // ggml_flash_attn_ext per attention layer; drops 3 dispatches (2 of them
    // matmuls) and the V transpose, at the cost of byte-identity and of
    // kq_soft ever existing. Opt-in via --flash-attn; only recipes that thread
    // it into the attention helpers honor it, others stay Materialized.
    using AttnImpl = DecodePolicy::AttnImpl;

    // Also stamps every KV cache's path salt. Flash changes the attention
    // output, so it changes the residual stream, so it changes every later
    // layer's K/V — a prefix or session blob frozen under one implementation is
    // not valid to resume under the other. Stamping here rather than at each
    // make_snapshot_header call site means none of the ~10 of them can forget.
    void set_attn_impl(AttnImpl a) {
        policy_.attn_impl = a;
        const uint64_t salt =
            (a == AttnImpl::Flash)
                ? static_cast<uint64_t>(std::hash<std::string>{}("attn=flash"))
                : 0ull;
        for (simple_kv_cache* c : snapshot_kv_caches())
            if (c) c->set_path_salt(salt);
    }
    AttnImpl attn_impl() const { return policy_.attn_impl; }
    bool use_flash_attn() const { return policy_.attn_impl == AttnImpl::Flash; }

    // True ⇒ this recipe's build_decoding_graph is persistent-graph capable:
    // every step-varying quantity is a graph-input VALUE (tokens, positions,
    // mask, gather indices, KV write rows), so a built+allocated decode graph
    // can be recomputed across steps without rebuild. Consumed by the P3
    // DecodeGraphCache; false keeps the recipe on the rebuild-per-step path.
    virtual bool supports_persistent_decode() const { return false; }

    // True ⇒ this recipe threads use_flash into its decode attention builder
    // AND casts its kq_mask to F16 (ggml_flash_attn_ext hard-asserts an F16
    // mask). A recipe that does neither would silently keep the materialized
    // path, so --flash-attn refuses rather than pretending it applied.
    // Prefill is materialized on every recipe — see attention.h.
    virtual bool supports_flash_attn() const { return false; }

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
    void set_decode_kv_bucket(uint32_t b) { policy_.decode_kv_bucket = b; }
    uint32_t decode_kv_bucket() const { return policy_.decode_kv_bucket; }

    // The whole run-time policy as one value, for a caller that wants to read or
    // assert on the pass's mode rather than poll five getters.
    const DecodePolicy& decode_policy() const { return policy_; }

protected:
    uint32_t decode_kv_len(uint32_t max_pos_plus_1, uint32_t n_ctx_max) const {
        return policy_.decode_kv_len(max_pos_plus_1, n_ctx_max);
    }

    // Run-time policy, held not inherited (see decode_policy.h). Defaults are
    // the byte-reproducible path; --persistent-graph is the only thing that sets
    // kv_write_mode and decode_kv_bucket, and it sets them together.
    DecodePolicy policy_;
};