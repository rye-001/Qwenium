#include "moe.h"

#include <cstdlib>

#include "ggml.h"

#include <cstdio>
#include <stdexcept>
#include <string>

// Expert dispatch is three batched ggml_mul_mat_id calls per layer (gate / up /
// down), each backed by a native Metal MUL_MAT_ID kernel, so the C-level launch
// count is O(1) in n_experts. This was once guarded by a QINF_MOE_FALLBACK
// option reserving the "OFF" branch for a fused Phase 4 kernel. That kernel was
// never written — docs/phase4-investigation.md measured its ceiling at ≤1.13x on
// Qwen 3.6 and dropped it — and the option could not actually be switched off
// (OFF hit an #error). Both were deleted 2026-08-29. If MoE dispatch is ever
// revisited, the candidate is patching ggml-metal's MUL_MAT_ID kernel for sparse
// routing, not a graph-level fusion (CLAUDE.md, ggml constraints).

// ── Internal helpers ──────────────────────────────────────────────────────────

static void moe_set_name(ggml_cgraph* gf, ggml_tensor* t, const char* base, int il) {
    char name[64];
    snprintf(name, sizeof(name), "%s.%d", base, il);
    ggml_set_name(t, name);
    ggml_build_forward_expand(gf, t);
}

// ── MoELayer ──────────────────────────────────────────────────────────────────

ggml_tensor* moe_build_expert_idx(ggml_context* ctx,
                                  ggml_cgraph*  gf,
                                  ggml_tensor*  logits,
                                  int           top_k,
                                  int64_t       n_tokens,
                                  RoutingSource routing,
                                  int           il)
{
    // The ONLY branch routing replay introduces, and it is a choice of
    // PRODUCER, not a change to the math: ggml_mul_mat_id already takes the
    // top-k index list as a tensor operand, so supplying it is the same shape
    // of thing as computing it. Every consumer downstream is identical on both
    // paths and does not know which one ran.
    if (routing == RoutingSource::Replay) {
        // Supplied: RoutingReplayInput fills this before compute.
        // Deliberately named differently from `moe_idx` — a replayed selection
        // is not a computed one, and anything that CAPTURES routing (which
        // looks for `moe_idx.<il>`) must not silently re-capture what it just
        // replayed.
        ggml_tensor* idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, n_tokens);
        ggml_set_input(idx);
        moe_set_name(gf, idx, "moe_routing", il);

        // The router logits must OUTLIVE this branch, and on the Replay path
        // nothing else keeps them alive. On the Router path the argsort is a
        // second consumer of `logits`; remove it and the only remaining
        // reference is a reshape VIEW, after which gallocr reuses the logits
        // buffer before ggml_get_rows gathers the gating weights from it. The
        // selection is then perfectly correct and the WEIGHTS are garbage —
        // which is worse than not pinning at all, because it looks like a
        // working replay.
        //
        // Measured on Gemma 4-26B-A4B: replayed ids verified identical for all
        // 30 layers and all 285 rows, and the last-position logits still moved
        // by 22 — an order of magnitude larger than the drift under study. Two
        // independent probes agree on the mechanism and produce bitwise-equal
        // output: pinning the logits (this line) and restoring the argsort
        // purely as a consumer.
        //
        // Qwen's MoELayer never showed it, so this cost a cross-family run to
        // find — which is what that rule is for.
        //
        // Costs n_experts * n_tokens * 4 bytes per MoE layer that the Router
        // path does not pay (~4.4 MB on Gemma 4 at 285 tokens): the price of
        // replay, not a leak.
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        return idx;
    }
    // Computed: sorted_idx is [n_experts, n_tokens] I32, and the result is the
    // top_k prefix of each token's ranking. It carries the PARENT's row stride,
    // so a reader must pin and slice sorted_idx rather than copy these bytes.
    ggml_tensor* sorted_idx = ggml_argsort(ctx, logits, GGML_SORT_ORDER_DESC);
    ggml_tensor* idx = ggml_view_2d(ctx, sorted_idx, top_k, n_tokens,
                                    sorted_idx->nb[1], 0);
    moe_set_name(gf, idx, "moe_idx", il);
    return idx;
}

MoELayer::MoELayer(
    ggml_tensor* w_router,
    ggml_tensor* w_exp_gate,
    ggml_tensor* w_exp_up,
    ggml_tensor* w_exp_down,
    ggml_tensor* w_sh_gate,
    ggml_tensor* w_sh_up,
    ggml_tensor* w_sh_down,
    ggml_tensor* w_sh_norm,
    const Hparams& hp,
    RoutingSource routing)
    : w_router_(w_router)
    , w_exp_gate_(w_exp_gate)
    , w_exp_up_(w_exp_up)
    , w_exp_down_(w_exp_down)
    , w_sh_gate_(w_sh_gate)
    , w_sh_up_(w_sh_up)
    , w_sh_down_(w_sh_down)
    , w_sh_norm_(w_sh_norm)
    , hp_(hp)
    , routing_(routing)
{
    if (hp_.has_shared_expert) {
        if (!w_sh_gate_ || !w_sh_up_ || !w_sh_down_) {
            throw std::runtime_error(
                "moe_layer: has_shared_expert=true but shared expert weights are null");
        }
    }
}

ggml_tensor* MoELayer::build(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  input,
    Phase         /*phase*/,
    int           il)
{
    // input: [n_embd, n_tokens]
    const int64_t n_embd   = input->ne[0];
    const int64_t n_tokens = input->ne[1];
    const int     n_exp    = hp_.n_experts;
    const int     top_k    = hp_.top_k;
    const int64_t ffn_dim  = hp_.ffn_dim;

    // ── 1. Routing logits and top-k gating ────────────────────────────────────

    // logits: [n_experts, n_tokens]
    ggml_tensor* logits = ggml_mul_mat(ctx, w_router_, input);
    moe_set_name(gf, logits, "moe_logits", il);

    // The ONLY branch routing replay introduces, and it is a choice of
    // PRODUCER, not a change to the math: ggml_mul_mat_id already takes the
    // top-k index list as a tensor operand, so supplying it is the same shape
    // of thing as computing it. Everything below this point is identical on
    // both paths and does not know which one ran.
    //
    // Note what stays on BOTH paths: the router matmul above. Replay pins the
    // discrete selection only — the gating weights are still gathered from the
    // real logits, at the replayed indices. Freezing the gate as well would be
    // a different and much stronger claim than the one measured.
    // Replay pins the discrete selection only — the gating weights below are
    // still gathered from the real router logits, at the replayed indices.
    ggml_tensor* expert_idx =
        moe_build_expert_idx(ctx, gf, logits, top_k, n_tokens, routing_, il);

    // Gather the actual logit values for the top-k experts
    // To use ggml_get_rows per token, we reshape logits to [1, n_experts, n_tokens]
    // so that ggml_get_rows picks from the n_experts dimension (ne[1]).
    ggml_tensor* logits_3d = ggml_reshape_3d(ctx, logits, 1, n_exp, n_tokens);
    ggml_tensor* expert_logits = ggml_get_rows(ctx, logits_3d, expert_idx);
    // expert_logits is [1, top_k, n_tokens], reshape back to 2D for softmax
    expert_logits = ggml_reshape_2d(ctx, expert_logits, top_k, n_tokens);

    // Apply softmax over top-k weights per token to normalize routing weights
    ggml_tensor* expert_weights = ggml_soft_max(ctx, expert_logits);
    moe_set_name(gf, expert_weights, "moe_weights", il);

    // ── 2. Expert dispatch via ggml_mul_mat_id ────────────────────────────────
    //
    // ggml_mul_mat_id: batched matmul where each token uses a different expert
    // weight matrix. Signature: (W [in, out, n_exp], x [in, n_tok], idx [top_k, n_tok])
    // Returns: [out, top_k, n_tok]

    // Reshape input for ggml_mul_mat_id: [in, n_tok] -> [in, 1, n_tok]
    // This aligns b->ne[2] with ids->ne[1] (n_tokens).
    ggml_tensor* input_3d = ggml_reshape_3d(ctx, input, n_embd, 1, n_tokens);

    // Gate projection: each token × its top_k expert gate weights
    ggml_tensor* exp_gate_out = ggml_mul_mat_id(ctx, w_exp_gate_, input_3d, expert_idx);
    moe_set_name(gf, exp_gate_out, "moe_exp_gate", il);
    // [ffn_dim, top_k, n_tokens]

    // Up projection
    ggml_tensor* exp_up_out = ggml_mul_mat_id(ctx, w_exp_up_, input_3d, expert_idx);
    moe_set_name(gf, exp_up_out, "moe_exp_up", il);

    // SwiGLU activation: silu(gate) * up
    ggml_tensor* exp_act = ggml_mul(ctx, ggml_silu(ctx, exp_gate_out), exp_up_out);
    moe_set_name(gf, exp_act, "moe_exp_act", il);
    // [ffn_dim, top_k, n_tokens]

    // Down projection
    // exp_act: [ffn_dim, top_k, n_tokens] — need to reshape for ggml_mul_mat_id
    // which expects x as [in_dim, n_tokens] with index [top_k, n_tokens].
    // We reshape exp_act to treat each (token, topk) independently:
    // ggml_mul_mat_id with w_exp_down [ffn_dim, n_embd, n_exp], input [ffn_dim, top_k, n_tokens]
    // This correctly picks the down-weight for each expert per token.
    ggml_tensor* exp_down_out = ggml_mul_mat_id(ctx, w_exp_down_, exp_act, expert_idx);
    moe_set_name(gf, exp_down_out, "moe_exp_down", il);
    // [n_embd, top_k, n_tokens]

    // ── 3. Weighted sum of expert outputs ─────────────────────────────────────

    // expert_weights: [top_k, n_tokens] — reshape to [1, top_k, n_tokens] for broadcast
    ggml_tensor* w_expanded = ggml_reshape_3d(ctx, expert_weights,
        1, top_k, n_tokens);

    // exp_down_out: [n_embd, top_k, n_tokens]
    // Multiply each expert output by its routing weight
    ggml_tensor* weighted = ggml_mul(ctx, exp_down_out, w_expanded);
    moe_set_name(gf, weighted, "moe_weighted", il);

    // Sum over top_k dimension → [n_embd, n_tokens].
    // weighted: [n_embd, top_k, n_tokens]; slice each expert via view and accumulate.
    ggml_tensor* routed_out = ggml_view_2d(ctx, weighted,
        n_embd, n_tokens, weighted->nb[2], 0);
    for (int k = 1; k < top_k; ++k) {
        ggml_tensor* expert_k = ggml_view_2d(ctx, weighted,
            n_embd, n_tokens, weighted->nb[2],
            static_cast<size_t>(k) * weighted->nb[1]);
        routed_out = ggml_add(ctx, routed_out, expert_k);
    }
    moe_set_name(gf, routed_out, "moe_routed_out", il);

    // ── 4. Shared expert (optional) ───────────────────────────────────────────

    if (!hp_.has_shared_expert) {
        return routed_out;
    }

    // Shared expert: standard SwiGLU FFN on all tokens
    ggml_tensor* sh_gate_out = ggml_mul_mat(ctx, w_sh_gate_, input);
    ggml_tensor* sh_up_out   = ggml_mul_mat(ctx, w_sh_up_,   input);
    ggml_tensor* sh_act      = ggml_mul(ctx, ggml_silu(ctx, sh_gate_out), sh_up_out);
    ggml_tensor* sh_down_out = ggml_mul_mat(ctx, w_sh_down_, sh_act);
    moe_set_name(gf, sh_down_out, "moe_shared_out", il);

    // Per-token scalar gate: w_sh_norm_ is [n_embd], mul_mat gives [1, n_tokens],
    // sigmoid maps to (0,1), ggml_mul broadcasts over [n_embd, n_tokens].
    ggml_tensor* sh_gate_logit = ggml_mul_mat(ctx, w_sh_norm_, input);
    ggml_tensor* sh_gate       = ggml_sigmoid(ctx, sh_gate_logit);
    ggml_tensor* sh_contribution = ggml_mul(ctx, sh_down_out, sh_gate);
    moe_set_name(gf, sh_contribution, "moe_shared_contrib", il);

    ggml_tensor* combined = ggml_add(ctx, routed_out, sh_contribution);
    moe_set_name(gf, combined, "moe_combined", il);

    return combined;
}
