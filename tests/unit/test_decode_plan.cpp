// test_decode_plan.cpp — the truth table is the contract.
//
// resolve_decode_plan's value is making the decode-path combinatorial surface
// *enumerable*. This test walks every input combination of the pure-logic
// resolver and asserts the resolved DecodePlan, so the truth table no longer
// lives only in the resolver's body. If a future edit changes a single
// assignment, exactly one row of this table flips and the test names it.
//
// It also pins:
//  - R3 (module-boundary fail-loud): forced_run on a non-feed recipe throws.
//  - The invariant the use site relies on:
//        allow_forced_elision ⇒ forced_run_enabled
//  - The collapsed diagnostic seam: ForceDense iff
//        force_dense_param || !slice_prefill_head.
//    (force_dense_param is the sole diagnostic input; the QWENIUM_FORCE_DENSE
//    env var was removed — one seam, not two.)
//  - Self-tautologies that were dropped from the resolver's rejection table
//    (R5) — asserted here on resolved values, where they belong.
//
// The has_decode_graph axis (and with it DecodeRoute, R1 and R2) was removed on
// 2026-08-29: every registered recipe has a decode graph, so the Bridge route
// was unreachable. Four input axes remain, hence 16 combinations.

#include <gtest/gtest.h>

#include "engine/decode_plan.h"

namespace {

struct Inputs {
    bool feed_ok;
    bool slice_pref;
    bool force_dense_param;
    bool forced_run_enabled;
};

// Independent re-derivation of what the resolver MUST return. Written so a
// reader can match it against the resolver body line-for-line; any drift
// fires a named test.
struct Expected {
    DecodeDiagnostic diagnostic;
    bool             allow_forced_elision;
    bool             sparse_head_allowed;
};

Expected expect(const Inputs& in) {
    Expected e;
    const bool diag_dense  = in.force_dense_param || !in.slice_pref;
    e.diagnostic           = diag_dense ? DecodeDiagnostic::ForceDense
                                        : DecodeDiagnostic::Optimized;
    e.allow_forced_elision = in.forced_run_enabled && in.feed_ok;
    e.sparse_head_allowed  = (e.diagnostic == DecodeDiagnostic::Optimized);
    return e;
}

std::string label(const Inputs& in) {
    std::string s;
    s += "feed=";    s += in.feed_ok           ? '1' : '0';
    s += " slice=";   s += in.slice_pref        ? '1' : '0';
    s += " fdense=";  s += in.force_dense_param ? '1' : '0';
    s += " forced=";  s += in.forced_run_enabled? '1' : '0';
    return s;
}

Inputs decode_bits(int bits) {
    return Inputs {
        ((bits >> 0) & 1) != 0, ((bits >> 1) & 1) != 0,
        ((bits >> 2) & 1) != 0, ((bits >> 3) & 1) != 0,
    };
}

DecodePlan resolve(const Inputs& in) {
    return resolve_decode_plan_inputs(
        in.feed_ok, in.slice_pref,
        in.force_dense_param, in.forced_run_enabled);
}

}  // namespace

TEST(DecodePlanTruthTable, AllResolvedValuesMatchIndependentDerivation) {
    int rows = 0;
    for (int bits = 0; bits < (1 << 4); ++bits) {
        Inputs in = decode_bits(bits);
        // R3 rows throw — covered in a dedicated test; skip them here.
        if (in.forced_run_enabled && !in.feed_ok) continue;
        ++rows;

        DecodePlan p = resolve(in);
        Expected   e = expect(in);

        SCOPED_TRACE(label(in));
        EXPECT_EQ(static_cast<int>(p.diagnostic), static_cast<int>(e.diagnostic));
        EXPECT_EQ(p.allow_forced_elision,         e.allow_forced_elision);
        EXPECT_EQ(p.sparse_head_allowed,          e.sparse_head_allowed);
    }
    // 16 total combos − 4 R3-throwing rows (forced=1, feed=0, other 2 axes
    // free) = 12 non-throwing rows.
    EXPECT_EQ(rows, 16 - 4);
}

// Dropped self-tautologies, re-asserted on resolved values where they
// actually belong (CLAUDE.md: fail-loud is for module boundaries, not for
// the line above).

TEST(DecodePlanTruthTable, SparseHeadAllowedOnlyWhenOptimized) {  // ex-R5
    for (int bits = 0; bits < (1 << 4); ++bits) {
        Inputs in = decode_bits(bits);
        if (in.forced_run_enabled && !in.feed_ok) continue;
        DecodePlan p = resolve(in);
        if (p.sparse_head_allowed) {
            EXPECT_EQ(static_cast<int>(p.diagnostic),
                      static_cast<int>(DecodeDiagnostic::Optimized)) << label(in);
        }
    }
}

// The invariant the decode_step forced-block use site dereferences on.
TEST(DecodePlanTruthTable, AllowForcedElisionImpliesForcedRunEnabled) {
    for (int bits = 0; bits < (1 << 4); ++bits) {
        Inputs in = decode_bits(bits);
        if (in.forced_run_enabled && !in.feed_ok) continue;
        DecodePlan p = resolve(in);
        if (p.allow_forced_elision)
            EXPECT_TRUE(in.forced_run_enabled) << label(in);
    }
}

// R3 — the only remaining module-boundary rejection: forced_run on a
// non-feed recipe must throw. The 2 free axes other than feed/forced give
// 4 throwing rows.
TEST(DecodePlanBoundary, R3_ForcedRunWithoutFeedTokensSupportThrows) {
    for (int bits = 0; bits < (1 << 2); ++bits) {
        const bool slice_pref   = ((bits >> 0) & 1) != 0;
        const bool force_dense  = ((bits >> 1) & 1) != 0;
        EXPECT_THROW(
            resolve_decode_plan_inputs(/*feed_ok=*/false,
                                       slice_pref, force_dense,
                                       /*forced_run_enabled=*/true),
            std::runtime_error);
    }
}
