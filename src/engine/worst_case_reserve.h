#pragma once
// worst_case_reserve.h — size the scheduler once, for the largest step a server
// will ever run.
//
// Responsibility: before the first request, reserve the scheduler's memory plan
//   for (1) the deepest batched decode — every slot at the context ceiling — and
//   (2) the largest prefill chunk, then leave every slot empty again.
//
// WHY THIS EXISTS. build_decoding_graph sizes its growing per-step inputs from
//   the CURRENT KV depth, so a reserve on a fresh cache sizes the smallest decode
//   graph and galloc reallocates as a conversation deepens — and that mid-
//   generation reallocation corrupts decode scratch (non-deterministic logits,
//   premature mid-word stops). Reserving at maximum depth sizes galloc once.
//
// Extracted 2026-10-05 from two byte-identical copies — the engine's
//   http_server.cpp and qemmi-lens-server's lens_server.cpp — so a client server
//   no longer builds graphs or calls ggml_backend_sched_reserve itself
//   (docs/plan-readout-api.md, the A3 open item). Behaviour, including the
//   stdout / stderr lines, is unchanged.
//
// Unit test: tests/unit/test_worst_case_reserve.cpp

#include "ggml-backend.h"
#include "../models/forward_pass_base.h"

namespace qinf::engine {

// Reserve for `max_slots` slots decoding at depth `max_ctx_per_slot - 1`, and for
// a prefill of min(max_ctx_per_slot, 2048) tokens. Every slot is cleared before
// return. A failed reserve prints a WARNING (as both servers always did) and
// does not throw.
void reserve_worst_case(ForwardPassBase& fp, ggml_backend_sched_t sched,
                        int max_slots, int max_ctx_per_slot);

}  // namespace qinf::engine
