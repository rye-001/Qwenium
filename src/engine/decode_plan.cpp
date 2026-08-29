#include "decode_plan.h"

#include <stdexcept>

DecodePlan resolve_decode_plan_inputs(bool feed_tokens_supported,
                                      bool slice_prefill_head,
                                      bool force_dense,
                                      bool forced_run_enabled) {
    if (forced_run_enabled && !feed_tokens_supported)
        throw std::runtime_error(
            "resolve_decode_plan: forced_run requested but recipe has no "
            "feed_tokens support (forced elision requires feed_tokens); "
            "expected feed_tokens_supported=true, actual=false");

    // The two diagnostic seams, folded: either one asking for the dense
    // reference makes the whole step dense. force_dense is the sole
    // diagnostic input (the QWENIUM_FORCE_DENSE env var was removed —
    // one seam, not two).
    const bool diag_dense = force_dense || !slice_prefill_head;

    DecodePlan p;
    p.diagnostic           = diag_dense ? DecodeDiagnostic::ForceDense
                                        : DecodeDiagnostic::Optimized;
    p.allow_forced_elision = forced_run_enabled
                          && feed_tokens_supported;
    p.sparse_head_allowed  = (p.diagnostic == DecodeDiagnostic::Optimized);
    return p;
}

DecodePlan resolve_decode_plan(const ForwardPassBase* fp,
                               bool forced_run_enabled,
                               bool force_dense) {
    return resolve_decode_plan_inputs(
        fp->feed_tokens_supported(),
        fp->slice_prefill_head(),
        force_dense,
        forced_run_enabled);
}
