#pragma once

#include <vector>
#include <cstdint>

#include "ggml-backend.h"
#include "../qwen3-core/forward-pass-base.h"
#include "../sampling/speculative.h"

struct SpeculativeBridge {
    ForwardPassBase* forward_pass;
    ggml_backend_sched_t scheduler;

    // Verify: run draft tokens as a mini-prefill, return [K * vocab_size] logits
    qwen3::SpeculativeDecoder::VerifyFunc make_verify(uint32_t slot) {
        return [this, slot](int /*slot_id*/, const std::vector<int32_t>& draft, int start_pos)
            -> std::vector<float>
        {
            ggml_backend_sched_reset(scheduler);
            ggml_cgraph* gf = forward_pass->build_prefill_graph(
                const_cast<std::vector<int32_t>&>(draft), start_pos, slot);
            ggml_backend_sched_alloc_graph(scheduler, gf);
            forward_pass->set_inputs(gf, const_cast<std::vector<int32_t>&>(draft), start_pos);
            ggml_backend_sched_graph_compute(scheduler, gf);
            forward_pass->advance_cache(draft.size(), slot);
            return forward_pass->get_output_logits(gf);
        };
    }

    // Rewind: set cache position back (discards unverified KV entries)
    qwen3::SpeculativeDecoder::RewindCacheFunc make_rewind(uint32_t slot) {
        return [this, slot](int /*slot_id*/, int new_pos) {
            forward_pass->set_cache_pos(slot, new_pos);
        };
    }
};