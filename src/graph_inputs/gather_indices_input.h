#pragma once

#include "graph_input.h"

#include <cstdint>

// Owns the "gather_indices" slot used by batched decode to gather per-slot KV
// rows out of the shared cache:
//
//   index = slot * n_ctx_max + t
//
// The stride is n_ctx_max because that is the cache's actual layout, not a
// policy choice: simple_kv_cache::gather_k reshapes the cache to a flat
// [n_embd, n_ctx_max * n_batch_max] and ggml_get_rows indexes into THAT.
//
// This class used to offer a second `Stride::NKvLen` policy (slot * n_kv_len +
// t) which qwen36 selected. It is correct only for slot 0, where both reduce to
// t, and reads rows out of slot 0's region for every other slot. It was a latent
// wrong-rows gather, not a live one, because qwen36 has only ever run
// single-slot. Both the policy and the enum were deleted 2026-08-29 along with
// the fix: with one stride there is no wrong one to pick.
class GatherIndicesInput : public GraphInput {
public:
    // Caller supplies the cache's per-slot stride (kv_cache->get_n_ctx_max()).
    explicit GatherIndicesInput(uint32_t n_ctx_max,
                                const char* slot = "gather_indices")
        : n_ctx_max_(n_ctx_max), slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    uint32_t    n_ctx_max_;
    const char* slot_;
};
