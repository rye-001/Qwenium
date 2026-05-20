#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "../models/forward_pass_base.h"
#include "../sampling/sampling.h"

#include "ggml-backend.h"

// Perform one decode step for a single slot.
//
// Calls peek_valid_set before the forward pass; selects the sparse LM head
// when the grammar narrows the candidate set below vocab_size / 8, and falls
// back to the dense path otherwise.  Grammar state is advanced inside via
// sampler->accept_token so the caller does not need to touch the grammar.
//
// Phase B — forced-token elision. When `forced_run` is non-null and the
// grammar leaves exactly one valid token, the next token(s) are determined
// without any forward pass: the deterministic run is rolled, model state is
// batch-advanced via feed_tokens (KV append + recurrent overwrite, no LM
// head), and the run's tokens are emitted. On return, `forced_run` holds the
// tokens emitted *before* the returned one (the caller must drain it, in
// order, with the same stop/print/history handling it gives the return
// value; grammar state for all of them is already advanced here). Empty on a
// normal (branching) step. Opt-in: pass nullptr to disable (unchanged
// behavior). Requires feed_tokens (supported recipe, not under TurboQuant);
// otherwise the forced path is skipped and the normal path runs.
//
// Returns the sampled (or last forced) token ID.
int32_t decode_step(
    ForwardPassBase*       fp,
    ggml_backend_sched_t   scheduler,
    qwenium::Sampler*      sampler,
    int32_t                token,
    uint32_t               slot,
    const std::vector<int32_t>&    history,
    const std::vector<std::string>& vocab,
    uint32_t               vocab_size,
    bool                   force_dense = false,  // diagnostic: skip sparse head selection
    std::vector<int32_t>*  forced_run = nullptr  // Phase B: forced-token elision (opt-in)
);
