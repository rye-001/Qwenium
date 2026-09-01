#pragma once
// qwen35_family.h — the transformer layer body shared by the Qwen 3.5-family
// hybrids: qwen35 (dense SwiGLU FFN) and qwen35moe / Qwen 3.6 (routed experts).
//
// Responsibility: build ONE layer of the family's residual stream —
//   norm → (DeltaNet | gated attention) → residual → norm → FFN → residual —
//   and return the new residual. The two recipes ran character-for-character
//   equivalent copies of this until 2026-08-29, differing in exactly one call
//   (build_ffn_swiglu vs the MoE layer) plus incidental naming.
//
// WHY THIS EXISTS. The duplication was not free: 11 of the 20 most recent
//   commits touching either recipe had to touch BOTH — vision/Seam B, --kv-f16,
//   the NextN head, the persistent decode graph, feed_tokens, typed graph
//   inputs, forced-token elision. Each was implemented twice by hand. And it
//   produced at least one real defect: the decode KV gather used the wrong
//   stride in qwen36 while qwen35 had it right, latent for months
//   (architecture.md §12, fixed the same day). One body, one place to fix.
//
// The FFN is a PARAMETER, not a fork: `moe_hp` non-null selects routed experts,
//   null selects dense SwiGLU. That is the same judgment Gemma 4 makes for its
//   own dense/MoE split, and settles the inconsistency §12 recorded.
//
// Free functions, not a base class: the recipes COMPOSE this, they do not
//   inherit it (docs/modular-layer-architecture.md). It touches no recipe state
//   — every input is an argument, and it registers no graph inputs, which is why
//   both recipes hoist their mask registration out of the layer loop.
//
// Unit test: tests/unit/test_qwen35_family.cpp (shape/dispatch, model-free).

#include <cstdint>
#include <vector>

#include "qwen35.h"                 // Qwen35Config (serves both hybrids)
#include "../layers/moe.h"          // MoELayer::Hparams
#include "../graph_inputs/graph_input.h"  // GraphInputSet

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct ModelMetadata;
struct TransformerBlock;
class  simple_kv_cache;
class  DeltaNetState;

// Everything both phases need that does not vary per layer.
struct Qwen35LayerCommon {
    ggml_context*            ctx;
    ggml_cgraph*             gf;
    const Qwen35Config*      cfg;
    const ModelMetadata*     meta;
    simple_kv_cache*         kv_cache;
    DeltaNetState*           dn_state;
    // The FFN seam: non-null ⇒ MoE, null ⇒ dense SwiGLU. Must agree with
    // cfg->is_moe(); the builders check and refuse fail-loud if it does not.
    const MoELayer::Hparams* moe_hp;
    // Opt-in flash attention (--flash-attn), decode path only. The recipe sets
    // it from DecodePolicy and casts kq_mask to F16 once per graph, because
    // ggml_flash_attn_ext requires an F16 mask. Default false keeps the
    // byte-reproducible materialized path.
    bool                     use_flash = false;
};

// ── Typed graph inputs ──────────────────────────────────────────────────────
//
// Kept here for the same reason as the layer body: these blocks were duplicated
// verbatim, and that duplication is where the Stride::NKvLen gather defect came
// from — qwen36 selected the wrong per-slot stride while qwen35 had it right
// (architecture.md §12). One declaration site, one stride, one place to be
// wrong. The prefill masks are a SEPARATE call because the two recipes register
// them at different points relative to the image splice, and that ordering is
// load-bearing (§7: graph_inputs_ must be cleared before build_image_substitution).

// clear + tokens + positions. The positions input is M-RoPE-aware: a recipe
// whose GGUF declares rope.dimension_sections needs four components per token.
void register_qwen35_common_inputs(GraphInputSet& inputs, const Qwen35Config& cfg);

// One causal mask per full-attention layer, named "kq_mask.{il}" to match what
// build_gated_attention emits. Qwen 3.5-family uses one uniform causal mask —
// no sliding window.
void register_qwen35_prefill_masks(GraphInputSet& inputs, const Qwen35Config& cfg,
                                   uint32_t n_layers);

// The whole decode input set: common inputs, the batched mask, the KV gather
// indices, and (when the persistent-graph write path is armed) the set_rows
// write indices. n_ctx_max is the cache's per-slot stride — see
// GatherIndicesInput, which has exactly one stride policy for this reason.
void register_qwen35_decode_inputs(GraphInputSet& inputs, const Qwen35Config& cfg,
                                   uint32_t n_ctx_max, bool with_kv_write_indices);

// Prefill: all prompt tokens in one pass, single slot.
ggml_tensor* build_qwen35_layer_prefill(
    const Qwen35LayerCommon& c,
    ggml_tensor*             inpL,
    const TransformerBlock&  blk,
    ggml_tensor*             inp_pos,
    uint32_t                 n_tokens,
    uint32_t                 slot_idx,
    uint32_t                 dn_idx,     // index into recurrent state (SSM layers)
    int                      kv_idx,     // index into the KV cache (attention layers)
    uint32_t                 il);

// ── Decode graph node-count guard ───────────────────────────────────────────
//
// DeltaNetLayer::build_decode (src/layers/deltanet.cpp) builds a full
// per-slot chain and concatenates — O(n_slots) graph nodes, not O(1)
// (docs/plan-deltanet-batched-decode.md, PARKED: the batching fix is scoped
// but not built). That makes the decode graph's node count grow linearly in
// the batch size B, and it can exceed FP_GRAPH_SIZE (graph_arena.h), which
// previously surfaced as a raw GGML_ASSERT(cgraph->n_nodes < cgraph->size)
// abort deep inside graph building instead of a named, fail-loud refusal.
//
// This computes the same limit ahead of time, from FP_GRAPH_SIZE and the
// checkpoint's own DeltaNet layer count, so construction refuses an
// over-limit max_batch_size before any graph is built or state is allocated.
//
// Confirmed by direct GGML_METAL_GRAPH_DEBUG=1 node-count census (exact
// integers, not a curve fit — docs/note-batch-scaling-cross-family.md):
// build_deltanet_layer's per-slot chain contributes exactly 44 graph nodes
// per DeltaNet layer per slot, identically on both shipped checkpoints
// (qwen35 9B: 24 layers, 44*24=1056; qwen35moe/Qwen 3.6: 30 layers,
// 44*30=1320). That per-layer count is a property of the C++ code that
// builds one DeltaNet layer for one slot, so it is expected to hold for any
// DeltaNet layer count, not just the two measured.
//
// The B-independent remainder of the decode graph (embedding, output head,
// attention layers, and — for qwen35moe — the MoE routers/experts, which are
// O(1) in B per CLAUDE.md's MoE dispatch invariant) was measured at 596
// nodes on the dense checkpoint (8 attention layers) and 2144 on the MoE one
// (10 attention layers, 40 MoE layers). Two data points cannot separate how
// much of that remainder tracks attention-layer count vs. MoE-router count
// vs. a fixed base, so rather than guess at a further decomposition this
// keys the remainder on is_moe() alone and uses the measured constant for
// that bucket. For a future Qwen 3.5-family checkpoint with a different
// attention-layer count than either measured config, this may refuse a
// max_batch_size that would technically still fit — that is the safe
// direction. The alternative is an under-estimated remainder that lets a
// batch size through that overflows the graph, which is the exact bug this
// guard exists to close.
//
// Fail-loud, no clamping: throws std::runtime_error naming the parameter,
// the expected limit, and the actual value, per qinf_error.h's contract.
// n_dn_layers == 0 is not this family's failure mode (no per-slot DeltaNet
// chain to overflow) and is accepted unconditionally.
void validate_deltanet_decode_batch_size(uint32_t n_dn_layers,
                                         bool     is_moe,
                                         uint32_t max_batch_size);

// Decode: one token per active slot, batched.
ggml_tensor* build_qwen35_layer_decode(
    const Qwen35LayerCommon&     c,
    ggml_tensor*                 inpL,
    const TransformerBlock&      blk,
    ggml_tensor*                 inp_pos,
    ggml_tensor*                 kq_mask,
    ggml_tensor*                 gather_indices,
    ggml_tensor*                 kv_write_idx,   // nullable (set_rows write path)
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>&  positions,
    uint32_t                     dn_idx,
    int                          kv_idx,
    uint32_t                     il);
