#pragma once
// attention.h — attention layer graph-building: free functions + AttentionLayer class.
//
// Responsibility: construct the KV-cache-backed attention subgraph for one
//   transformer layer.
// Public surface — the free functions, which are what every recipe actually
// calls. They sit at TWO DIFFERENT ALTITUDES, and the names hide it:
//
//   ATTENTION CORE — caller has already projected; pass Q/K/V in.
//     build_attn_mha          — Q@K^T → softmax → @V (GQA-aware)
//     build_attention         — prefill: cache write + full-history MHA
//     build_batched_attention — decode: per-slot scatter + gather MHA
//   WHOLE ATTENTION LAYER — caller passes the normed residual and the weights;
//   projections, Q/K norms, RoPE and the output projection happen inside.
//     build_gated_attention          — prefill (Qwen 3.5/3.6)
//     build_gated_batched_attention  — decode   (Qwen 3.5/3.6)
//
// So `build_attention` and `build_gated_attention` differ by far more than the
// adjective suggests: one is a kernel, the other is a layer. Resolving that is a
// deliberate redesign, not a rename — see architecture.md §12.
//
// DEAD: class AttentionLayer, below, has NO production caller (only
// tests/unit/test_attention.cpp). This header used to call it "canonical going
// forward"; it never became canonical, and its project_qkv DUPLICATES the
// projection logic recipes do inline, so the two can drift. Deleting it is
// proposed in architecture.md §12 — do not build on it.
// State owned: none — KV cache and weight tensors are passed in by the caller.
// Invariants: all tensors are appended to the caller's ggml_cgraph;
//   no ggml_context is created inside this module.
// Reference: Qwen3ForwardPass::_build_attention_layer and
//   ForwardPassBase::build_attn_mha in src/models/forward_pass_base.cpp.
// Unit test: tests/unit/test_attention.cpp

#include "layer.h"
#include "ggml.h"
#include <cstdint>
#include <vector>

class simple_kv_cache;
struct ggml_context;
struct ggml_cgraph;

// Logit soft-capping op (Gemma 2): cap * tanh(x / cap).
// When cap == 0 the caller should skip the call; this function never returns
// the input unchanged — it always applies the op regardless of cap value.
// Used on raw attention scores (before scaled softmax) and on final logits.
ggml_tensor* build_softcap(
    ggml_context* ctx,
    ggml_tensor*  x,
    float         cap);

// Pruned RoPE (p-RoPE — Gemma 4 global layers).
//
// Applies NEOX-style RoPE rotation to only the first n_rot dimensions of each
// head; the remaining (head_dim - n_rot) dimensions pass through unchanged.
// When n_rot == head_dim this is bit-identical to standard full-head RoPE.
//
// x:           [head_dim, n_head, n_tokens] (or [head_dim, n_kv, n_tokens])
// inp_pos:     [n_tokens] I32 token-position indices.
// n_rot:       number of leading head dimensions to rotate (1 ≤ n_rot ≤ head_dim).
// context_len: original training context length (RoPE extension parameter).
// freq_base:   RoPE frequency base.
//
// The implementation delegates to ggml_rope_ext with NEOX type; the trailing
// (head_dim - n_rot) elements are left untouched by the kernel.
ggml_tensor* build_rope_pruned(
    ggml_context* ctx,
    ggml_tensor*  x,
    ggml_tensor*  inp_pos,
    int           n_rot,
    int           context_len,
    float         freq_base);

// Core MHA kernel: Q@K^T → [optional softcap] → softmax(scale, mask) → @V.
// Handles GQA, head permutation, stream splitting, and contiguous recombination.
// softcap: attention logit soft-capping value (0.0 = off — Qwen/Gemma-1 path
//   produces bit-identical output). When > 0: cap * tanh(x/cap) is applied to
//   the raw QK product before the scaled softmax.
// Extracted from ForwardPassBase::build_attn_mha — identical logic.
ggml_tensor* build_attn_mha(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* q,
    ggml_tensor* k,
    ggml_tensor* v,
    ggml_tensor* kq_mask,
    ggml_tensor* sinks,
    float        kq_scale,
    uint32_t     pos,
    int          il,
    float        softcap = 0.0f);

// Prefill / single-slot attention.
// Writes K/V to the cache at the current slot position, reads the full
// cached sequence, builds a per-layer causal mask, then runs MHA.
//
// head_dim_k: K per-head dimension (used for both K-cache view dimensions).
// head_dim_v: V per-head dimension.  For almost all models head_dim_k ==
//   head_dim_v; they are kept separate to support future asymmetric cases.
// n_head_kv: number of KV heads (GQA).
// softcap: forwarded to build_attn_mha (0.0 = off; Qwen/Gemma-1 unchanged).
//
// Extracted from Qwen3ForwardPass::_build_attention_layer — identical logic.
ggml_tensor* build_attention(
    ggml_context*     ctx,
    ggml_cgraph*      gf,
    simple_kv_cache*  kv_cache,
    ggml_tensor*      q,
    ggml_tensor*      k,
    ggml_tensor*      v,
    int               layer_idx,
    float             kq_scale,
    uint32_t          n_tokens,
    uint32_t          slot_idx,
    int               il,
    int               head_dim_k,
    int               head_dim_v,
    int               n_head_kv,
    float             softcap = 0.0f);

// Decode / batched multi-slot attention.
// Scatters K/V into each slot, gathers the full KV history via indices,
// applies a shared causal mask, then runs MHA.
// Reads n_embd_head and n_head_kv from k->ne[0]/ne[1].
// Extracted from Qwen3ForwardPass::_build_batched_attention_layer — identical logic.
//
// kv_write_indices (nullable): I64 [n_batch] graph input carrying the KV
//   write destination rows (slot * n_ctx_max + pos). Non-null ⇒ the K/V write
//   is one ggml_set_rows per cache (write position is a run-time VALUE — the
//   persistent-decode-graph write path, docs/plan-persistent-decode-graph.md
//   §2.1). Null ⇒ the legacy per-slot ggml_cpy at a build-time-baked offset,
//   byte-identical to the pre-P1 path.
ggml_tensor* build_batched_attention(
    ggml_context*                   ctx,
    ggml_cgraph*                    gf,
    simple_kv_cache*                kv_cache,
    ggml_tensor*                    q,
    ggml_tensor*                    k,
    ggml_tensor*                    v,
    int                             layer_idx,
    float                           kq_scale,
    const std::vector<uint32_t>&    slots,
    const std::vector<int32_t>&     positions,
    ggml_tensor*                    kq_mask,
    ggml_tensor*                    gather_indices,
    int                             il,
    float                           softcap = 0.0f,
    ggml_tensor*                    kv_write_indices = nullptr);

// ── M-RoPE section widths ────────────────────────────────────────────────────
// P2 of docs/plan-qwen35-vision-impl.md. `<arch>.rope.dimension_sections`
// splits the rotated dimensions into four sections, each reading its own
// position component. When active, RoPE is ggml_rope_multi instead of
// ggml_rope_ext and `inp_pos` carries 4 components per token (component-major
// — see MRopePositionsInput).
//
// This is a PARAMETER on RoPE, not a separate attention variant: the rotation
// layout is identical (ggml uses the same rotate_pairs(n_dims, n_dims/2) for
// MROPE as for NEOX), only the per-dimension theta source differs. With all
// four components equal — every text-only step — the two are numerically the
// same operation.
struct MRopeSections {
    // Widths summing to n_rot/2. Qwen 3.5-family GGUFs all declare
    // [11, 11, 10, 0] against rope.dimension_count 64.
    int32_t widths[4] = {0, 0, 0, 0};

    // False ⇒ the recipe uses plain NEOX RoPE and a 1-component inp_pos.
    // Absence of the GGUF key is a legitimate, well-defined state (a text-only
    // checkpoint), not a missing required input — so it is a flag, not a throw.
    bool active = false;

    // Build from a GGUF `rope.dimension_sections` array, validating what ggml
    // and the rotation maths actually require. Shared by every recipe that
    // reads the key so the checks cannot drift between them.
    //
    // Fail-loud on a PRESENT but malformed key — that is a real contradiction
    // in the file, unlike simple absence. `key` names the slot in the message,
    // `n_rot` is the recipe's effective rotated width.
    //   - exactly 4 widths
    //   - widths sum to n_rot/2 (each rotated pair belongs to exactly one
    //     section; a short sum makes ggml's `sector % sect_dims` wrap and
    //     silently rotate against the wrong component)
    //   - at least one of the first three is > 0 (ggml asserts this)
    static MRopeSections from_widths(const std::vector<int32_t>& widths,
                                     const char* key,
                                     int n_rot);
};

// ── Gated attention variants (Qwen3.5, Qwen3.6) ─────────────────────────────
// These models use a joint Q+Gate projection, Q/K RMS norms, partial RoPE, and
// sigmoid gating after the attention output. They differ structurally from the
// Qwen2/3 attention above, so they live as separate free functions.

// Prefill / single-slot gated attention.
// Takes raw normed input cur; performs Q/K/V projections, Q/K norms, partial
// RoPE, KV cache write + full-history MHA, then sigmoid gating + output proj.
ggml_tensor* build_gated_attention(
    ggml_context*    ctx,
    ggml_cgraph*     gf,
    simple_kv_cache* kv_cache,
    ggml_tensor*     cur,          // normed input [n_embd, n_tokens]
    ggml_tensor*     inp_pos,      // [n_tokens]
    int              kv_cache_layer,
    uint32_t         n_tokens,
    uint32_t         slot_idx,
    int              il,
    ggml_tensor*     w_q,          // attn_q_weight (joint Q+Gate)
    ggml_tensor*     w_q_norm,     // attn_q_norm_weight
    ggml_tensor*     w_k,          // attn_k_weight
    ggml_tensor*     w_k_norm,     // attn_k_norm_weight
    ggml_tensor*     w_v,          // attn_v_weight
    ggml_tensor*     w_out,        // attn_output_weight
    int              n_embd_head,
    int              n_head,
    int              n_head_kv,
    int              n_rot,
    float            freq_base,
    int              context_length,
    float            rms_norm_eps,
    // Default-inactive so existing call sites keep NEOX behaviour verbatim.
    const MRopeSections& mrope = MRopeSections{});

// Decode / batched multi-slot gated attention.
// Same gated projections/norms/gating as above, but operates on a batch of
// slots with pre-built kq_mask and gather_indices.
// kv_write_indices: same contract as build_batched_attention above (nullable;
//   non-null ⇒ set_rows write, null ⇒ legacy baked-offset cpy write).
ggml_tensor* build_gated_batched_attention(
    ggml_context*                   ctx,
    ggml_cgraph*                    gf,
    simple_kv_cache*                kv_cache,
    ggml_tensor*                    cur,           // normed input [n_embd, n_batch]
    ggml_tensor*                    inp_pos,       // [n_batch]
    ggml_tensor*                    kq_mask,       // [n_kv_len, 1, 1, n_batch]
    ggml_tensor*                    gather_indices,// [n_batch * n_kv_len]
    int                             kv_cache_layer,
    const std::vector<uint32_t>&    slots,
    const std::vector<int32_t>&     positions,
    int                             il,
    ggml_tensor*                    w_q,
    ggml_tensor*                    w_q_norm,
    ggml_tensor*                    w_k,
    ggml_tensor*                    w_k_norm,
    ggml_tensor*                    w_v,
    ggml_tensor*                    w_out,
    int                             n_embd_head,
    int                             n_head,
    int                             n_head_kv,
    int                             n_rot,
    float                           freq_base,
    int                             context_length,
    float                           rms_norm_eps,
    ggml_tensor*                    kv_write_indices = nullptr,
    const MRopeSections&            mrope = MRopeSections{});

// ── AttentionLayer class (Phase 2 canonical interface) ────────────────────────
//
// Wraps weight refs + hyperparams; exposes a single build() that dispatches on
// Phase.  input is the normed residual [n_embd, n_tokens/n_batch]; this class
// performs Q/K/V projections, optional RMS norms, RoPE, KV cache write or
// scatter/gather, MHA, and output projection internally.
//
// The existing free functions (build_attention, build_batched_attention) remain
// as the internal implementation and for legacy call sites in model recipes
// until Phase 3 migrates them.

class AttentionLayer {
public:
    struct Hparams {
        int   n_head;
        int   n_head_kv;
        int   n_embd_head;
        float kq_scale;
        int   n_rot;
        float freq_base;
        int   context_len;
        float rms_norm_eps;
        bool  has_q_norm;  // true for Qwen3 (QK RMS norms applied post-projection)
        bool  has_bias;    // true for Qwen2 (QKV additive biases)
    };

    // All weight tensors are borrowed references — AttentionLayer does not own them.
    // w_q_norm / w_k_norm: required when hp.has_q_norm == true; nullptr otherwise.
    // q_bias / k_bias / v_bias: required when hp.has_bias == true; nullptr otherwise.
    AttentionLayer(
        ggml_tensor* w_q,
        ggml_tensor* w_k,
        ggml_tensor* w_v,
        ggml_tensor* w_out,
        ggml_tensor* w_q_norm,
        ggml_tensor* w_k_norm,
        ggml_tensor* q_bias,
        ggml_tensor* k_bias,
        ggml_tensor* v_bias,
        simple_kv_cache* kv_cache,
        const Hparams& hp);

    // Args for Phase::Prefill.
    struct PrefillArgs {
        ggml_tensor* inp_pos;  // [n_tokens] token position indices
        uint32_t     n_tokens;
        uint32_t     slot_idx;
    };

    // Args for Phase::Decode.
    struct DecodeArgs {
        ggml_tensor*                 inp_pos;         // [n_batch]
        const std::vector<uint32_t>* slots;           // per-batch slot index
        const std::vector<int32_t>*  positions;       // per-batch token position
        ggml_tensor*                 kq_mask;         // causal mask [n_kv, 1, 1, n_batch]
        ggml_tensor*                 gather_indices;  // KV gather indices
    };

    // Unified build entry point.
    // phase == Prefill: uses prefill_args; decode_args may be nullptr.
    // phase == Decode:  uses *decode_args; prefill_args fields are ignored.
    // Returns: output tensor [n_embd, n_tokens/n_batch] after output projection.
    ggml_tensor* build(
        ggml_context*      ctx,
        ggml_cgraph*       gf,
        ggml_tensor*       input,
        int                layer_idx,
        Phase              phase,
        const PrefillArgs& prefill_args,
        const DecodeArgs*  decode_args = nullptr);

private:
    ggml_tensor*     w_q_;
    ggml_tensor*     w_k_;
    ggml_tensor*     w_v_;
    ggml_tensor*     w_out_;
    ggml_tensor*     w_q_norm_;
    ggml_tensor*     w_k_norm_;
    ggml_tensor*     q_bias_;
    ggml_tensor*     k_bias_;
    ggml_tensor*     v_bias_;
    simple_kv_cache* kv_cache_;
    Hparams          hp_;

    ggml_tensor* build_prefill(ggml_context*, ggml_cgraph*, ggml_tensor* input,
                                int layer_idx, const PrefillArgs&);
    ggml_tensor* build_decode(ggml_context*, ggml_cgraph*, ggml_tensor* input,
                               int layer_idx, const DecodeArgs&);
    // Project input → Q/K/V, apply optional RMS norm and RoPE.
    // Returns {Qcur, Kcur, Vcur} reshaped to [n_embd_head, n_head(_kv), n_tokens].
    struct QKV { ggml_tensor* q; ggml_tensor* k; ggml_tensor* v; };
    QKV project_qkv(ggml_context*, ggml_cgraph*, ggml_tensor* input,
                    ggml_tensor* inp_pos, int layer_idx, uint32_t n_tokens);
};
