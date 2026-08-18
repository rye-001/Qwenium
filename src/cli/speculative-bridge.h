#pragma once

#include <vector>
#include <cstdint>

#include "ggml-backend.h"
#include "../models/forward_pass_base.h"
#include "../sampling/speculative.h"

struct SpeculativeBridge {
    ForwardPassBase* forward_pass;
    ggml_backend_sched_t scheduler;

    // Verify: run draft tokens as a mini-prefill, return [K * vocab_size] logits.
    // The prefill head slice (last-position-only, the default) must be OFF for
    // the verify pass: verification needs logits at ALL K draft positions.
    // With the slice on, run_prefill returns [1 * vocab] and the decoder's
    // shape check rejects every draft — speculation silently degenerates to
    // normal decode (the bug that masked this path for months).
    qwenium::SpeculativeDecoder::VerifyFunc make_verify(uint32_t slot) {
        return [this, slot](int /*slot_id*/, const std::vector<int32_t>& draft, int start_pos)
            -> std::vector<float>
        {
            const bool prev = forward_pass->slice_prefill_head();
            forward_pass->set_slice_prefill_head(false);
            std::vector<float> logits =
                forward_pass->run_prefill(draft, start_pos, slot, scheduler);
            forward_pass->set_slice_prefill_head(prev);
            return logits;
        };
    }

    // Rewind: set cache position back (discards unverified KV entries).
    // Argument order is (pos, slot_idx) — this call site once had them swapped,
    // which made every partial-reject write positions[new_pos] on a 1-slot
    // cache (out-of-bounds) instead of rewinding.
    qwenium::SpeculativeDecoder::RewindCacheFunc make_rewind(uint32_t slot) {
        return [this, slot](int /*slot_id*/, int new_pos) {
            forward_pass->set_cache_pos(static_cast<uint32_t>(new_pos), slot);
        };
    }
};