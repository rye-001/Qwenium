#pragma once
// graph_compute.h — the one place a graph-compute result is checked.
//
// Responsibility: turn a ggml_status from ggml_backend_sched_graph_compute into
//   either "carry on" or a fail-loud throw. One function, so the message is
//   uniform and a new compute site cannot invent its own (or forget).
//
// WHY THIS EXISTS. Every text-path compute site used to discard the status.
//   ggml-metal returns GGML_STATUS_FAILED when a command buffer fails — most
//   often GPU out-of-memory, which a full-budget image prefill next to a
//   13-17 GB model reaches on a 32 GB host — and then LATCHES it:
//     "once set, graph_compute will return GGML_STATUS_FAILED until the backend
//      is recreated"                       (ggml-metal-context.m, ctx->has_error)
//   Discarding that meant the engine read whatever was in the output buffer and
//   kept decoding, emitting fluent, plausible, wrong text. It has already caused
//   one misdiagnosis. Silent degradation at a module boundary is exactly what
//   the fail-loud contract forbids (CLAUDE.md).
//
// The three vision encoders already checked their status; this generalizes their
//   pattern to the text path rather than inventing a new one.
//
// NOT RECOVERABLE. The Metal error flag is sticky by design — the backend tells
//   you to recreate it, which we cannot do mid-run. So every later compute also
//   throws. That is the honest state: a caller gets a named error every time
//   instead of one wrong answer followed by more wrong answers. Callers are
//   responsible for containing the throw (the server fails the request; the CLI
//   exits non-zero) rather than letting it terminate the process.
//
// Unit test: tests/unit/test_graph_compute.cpp

#include <stdexcept>
#include <string>

#include "ggml-backend.h"

namespace qinf::engine {

// Human-readable ggml_status, so the error names the actual value rather than
// an integer the reader has to look up.
inline const char* ggml_status_name(ggml_status st) {
    switch (st) {
        case GGML_STATUS_ALLOC_FAILED: return "GGML_STATUS_ALLOC_FAILED";
        case GGML_STATUS_FAILED:       return "GGML_STATUS_FAILED";
        case GGML_STATUS_SUCCESS:      return "GGML_STATUS_SUCCESS";
        case GGML_STATUS_ABORTED:      return "GGML_STATUS_ABORTED";
        default:                       return "GGML_STATUS_<unknown>";
    }
}

// Throw unless the compute succeeded. `site` names the caller (e.g.
// "decode_step", "run_prefill") so the message says which pass died.
inline void require_compute_success(ggml_status st, const char* site) {
    if (st == GGML_STATUS_SUCCESS) return;

    std::string msg = std::string(site) +
        ": slot \"graph_compute\" expected GGML_STATUS_SUCCESS, actual: " +
        ggml_status_name(st);

    if (st == GGML_STATUS_ALLOC_FAILED || st == GGML_STATUS_FAILED) {
        msg += " — the backend command buffer failed, most often GPU "
               "out-of-memory. On Metal this state is latched until the backend "
               "is recreated, so every later compute will fail too. Reduce "
               "--ctx-size, the image resolution, or the number of slots. "
               "Refusing rather than decoding from an uncomputed buffer.";
    }
    throw std::runtime_error(msg);
}

}  // namespace qinf::engine
