#pragma once

#include "graph_input.h"

#include <cstdint>

// Owns the "gather_indices" slot used by batched decode to gather per-slot KV
// rows out of the shared cache. The per-slot row stride differs by recipe KV
// layout (parameterize-vs-split: a stride *policy* parameter, not a separate
// class — same reasoning as the sliding-window parameter on AttnMaskInput):
//
//   - NCtxMax: index = slot * kv_cache->get_n_ctx_max() + t   (qwen3, qwen35)
//   - NKvLen : index = slot * n_kv_len + t                    (qwen36)
//
// where n_kv_len = gather_indices->ne[0] / n_batch, derived at set time.
class GatherIndicesInput : public GraphInput {
public:
    enum class Stride { NCtxMax, NKvLen };

    // NCtxMax policy: caller supplies the fixed per-slot stride.
    explicit GatherIndicesInput(uint32_t n_ctx_max,
                                const char* slot = "gather_indices")
        : stride_(Stride::NCtxMax), n_ctx_max_(n_ctx_max), slot_(slot) {}

    // NKvLen policy: stride is the per-batch gather length (self-derived).
    explicit GatherIndicesInput(Stride policy,
                                const char* slot = "gather_indices")
        : stride_(policy), n_ctx_max_(0), slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    Stride      stride_;
    uint32_t    n_ctx_max_;
    const char* slot_;
};
