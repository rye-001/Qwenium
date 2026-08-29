#pragma once

#include "../models/forward_pass_base.h"

// DecodePlan — the resolved, immutable decision for ONE decode step.
//
// decode_step used to read several decode-path flags ad hoc at point of use
// (force_dense, slice_prefill_head, forced_run, want_logits,
// feed_tokens_supported). Each was individually clean;
// collectively they were an implicit, scattered, untested combinatorial
// surface — the "path zoo" smell. This makes that surface first-class:
// resolve_decode_plan() computes the decision ONCE, rejects illegal
// combinations fail-loud at resolution, and every site reads plan.* (nothing
// recomputes). Pure consolidation — zero behavior change.

// The two former diagnostic seams collapsed into one. decode_step's
// force_dense and ForwardPassBase::slice_prefill_head are the SAME kind of
// thing — "build the dense reference vs the optimized path" — not user
// config. One enum, decided once. force_dense_param is the sole diagnostic
// input;
enum class DecodeDiagnostic {
    Optimized,   // sparse LM head + out_ids prefill slice as available
    ForceDense,  // dense reference everywhere (sparse head off; no slicing)
};

// There is exactly one decode route: build_decoding_graph. The legacy
// `run_prefill` bridge (DecodeRoute::Bridge) was deleted on 2026-08-29 once
// every recipe had a decode graph — Gemma 1-4 were the last holdouts and the
// Gemma batched-decode work removed their override. It was unreachable, not
// merely unused: has_decode_graph() returned true for every registered recipe.

struct DecodePlan {
    DecodeDiagnostic diagnostic;

    // Phase B forced-token elision permitted this step. Still gated at use
    // by the grammar actually yielding exactly one token (data, not a flag).
    // Encodes feed_tokens_supported.
    //
    // Invariant (used at the decode_step forced-block use site):
    //     plan.allow_forced_elision  ⇒  forced_run_enabled  (caller passed a
    //                                                        non-null sink)
    // It can only be true when the caller opted in, so the forced block may
    // dereference forced_run unconditionally.
    bool allow_forced_elision;

    // Sparse LM head may fire. The final decision also needs the runtime
    // valid-set size — that is data, not a flag, and stays at the use site.
    bool sparse_head_allowed;
};

// Pure-logic resolver (no ForwardPassBase dependency). The truth table lives
// here so it is enumerable from a unit test without a model. Throws on R3
// (forced_run requested on a recipe with no feed_tokens support — a genuine
// upstream contract violation). Self-tautology checks against the resolver's
// own assignments are NOT here; they live in the unit test as assertions on
// the resolved values.
DecodePlan resolve_decode_plan_inputs(bool feed_tokens_supported,
                                      bool slice_prefill_head,
                                      bool force_dense,
                                      bool forced_run_enabled);

// Thin shim that reads the recipe predicates and delegates to
// resolve_decode_plan_inputs. decode_step calls this.
DecodePlan resolve_decode_plan(const ForwardPassBase* fp,
                               bool forced_run_enabled,
                               bool force_dense);
