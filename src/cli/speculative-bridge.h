#pragma once

#include <vector>
#include <cstdint>

#include "ggml-backend.h"
#include "../models/forward_pass_base.h"
#include "../sampling/speculative.h"

struct SpeculativeBridge {
    ForwardPassBase* forward_pass;
    ggml_backend_sched_t scheduler;

    // Verify: run draft tokens as a mini-prefill, return [K * vocab_size] logits
    qwenium::SpeculativeDecoder::VerifyFunc make_verify(uint32_t slot) {
        return [this, slot](int /*slot_id*/, const std::vector<int32_t>& draft, int start_pos)
            -> std::vector<float>
        {
            return forward_pass->run_prefill(draft, start_pos, slot, scheduler);
        };
    }

    // Rewind: set cache position back (discards unverified KV entries)
    qwenium::SpeculativeDecoder::RewindCacheFunc make_rewind(uint32_t slot) {
        return [this, slot](int /*slot_id*/, int new_pos) {
            forward_pass->set_cache_pos(slot, new_pos);
        };
    }
};