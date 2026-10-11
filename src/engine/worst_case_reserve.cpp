// worst_case_reserve.cpp — see worst_case_reserve.h. The body is the one both
// servers carried, moved verbatim (renamed members only).

#include "worst_case_reserve.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace qinf::engine {

void reserve_worst_case(ForwardPassBase& fp, ggml_backend_sched_t sched,
                        int max_slots, int max_ctx_per_slot) {
    std::cout << "Reserving memory for max batch size: " << max_slots << " and max ctx: " << max_ctx_per_slot << std::endl;

    // 1. Reserve for max decode batch — at MAXIMUM KV depth.
    //
    // build_decoding_graph sizes its growing per-step inputs (gather_indices
    // and the per-window attention masks) from get_cache_pos(slot), i.e. the
    // CURRENT KV depth — NOT the `positions` argument. Reserving with a fresh
    // cache (depth 0 → n_kv_len 1) therefore reserves the SMALLEST decode
    // graph, so galloc must reallocate as the real conversation deepens. That
    // mid-generation reallocation corrupts decode scratch → non-deterministic
    // logits and premature mid-word stops on long generations. Advance every
    // slot's cache to the context ceiling first so the reserved graph is the
    // largest one decode will ever build; galloc is then sized once and never
    // grows during a request. Positions/values are irrelevant here — only the
    // graph topology/sizes matter for reservation — so the dummy 0s are fine.
    {
        const int32_t max_depth =
            max_ctx_per_slot > 0 ? max_ctx_per_slot - 1 : 0;

        std::vector<int32_t>  tokens(max_slots, 0);             // Dummy tokens
        std::vector<uint32_t> slot_ids(max_slots);             // 0..max_slots-1
        // Positions MUST equal the cache depth: build_decoding_graph sizes
        // gather_indices from get_cache_pos, while build_batched_attention
        // derives n_kv_len from this `positions` argument. Real decode keeps
        // them equal (positions = get_cache_pos); the reserve must too, or the
        // gather and the attention reshape disagree (GGML_ASSERT in reshape).
        std::vector<int32_t>  positions(max_slots, max_depth);

        for (int i = 0; i < max_slots; ++i) {
            slot_ids[i] = i;
        }

        if (max_depth > 0) {
            for (int i = 0; i < max_slots; ++i) {
                fp.advance_cache(static_cast<uint32_t>(max_depth), i);
            }
        }

        ggml_backend_sched_reset(sched);

        // Build the graph for the maximum possible workload (deepest KV)
        ggml_cgraph* gf = fp.build_decoding_graph(tokens, slot_ids, positions);

        // Reserve memory in the scheduler
        if (!ggml_backend_sched_reserve(sched, gf)) {
            std::cerr << "WARNING: Failed to reserve memory for max decode batch!" << std::endl;
        }

        // Restore the caches to empty — the reserve only needed the topology.
        for (int i = 0; i < max_slots; ++i) {
            fp.clear_slot(i);
        }
    }

    // 2. Reserve for max prefill batch
    // We reserve for the maximum number of tokens we might process in a single prefill step.
    // Assuming we might process up to max_ctx_per_slot tokens at once (full prompt).
    {
        // Limit reservation to a reasonable batch size if max_ctx is huge, 
        // but here we use max_ctx_per_slot as it's the theoretical max.
        // In practice, this should be the configured 'batch_size' for prompt processing.
        // Since we don't have a separate batch_size config here, we use max_ctx_per_slot.
        int n_tokens = max_ctx_per_slot;
        
        // Use a safe upper bound if max_ctx is very large to avoid OOM during reservation if not needed
        if (n_tokens > 2048) n_tokens = 2048; // Cap at 2048 for reservation safety if not specified otherwise

        std::vector<int32_t> tokens(n_tokens, 0);
        
        ggml_backend_sched_reset(sched);
        ggml_cgraph* gf = fp.build_prefill_graph(tokens, 0, 0);
         if (!ggml_backend_sched_reserve(sched, gf)) {
            std::cerr << "WARNING: Failed to reserve memory for max prefill batch!" << std::endl;
        }
    }
    
    std::cout << "Memory reservation complete." << std::endl;
}

}  // namespace qinf::engine
