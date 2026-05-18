#include "gather_indices_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>
#include <vector>

void GatherIndicesInput::set_input(const StepContext& step) {
    if (!step.slots)
        throw std::runtime_error(
            "GatherIndicesInput: slot 'gather_indices': expected batched slot "
            "list, got: null StepContext::slots");

    ggml_tensor* t = require_tensor(step, slot_, GGML_TYPE_I32);

    const uint32_t n_total = static_cast<uint32_t>(t->ne[0]);
    const uint32_t n_batch = static_cast<uint32_t>(step.n_rows());
    const uint32_t n_kv_len = n_total / n_batch;
    const uint32_t stride = (stride_ == Stride::NCtxMax) ? n_ctx_max_ : n_kv_len;

    std::vector<int32_t> indices(n_total);
    for (uint32_t b = 0; b < n_batch; ++b) {
        const uint32_t slot_idx = (*step.slots)[b];
        for (uint32_t tt = 0; tt < n_kv_len; ++tt)
            indices[b * n_kv_len + tt] = slot_idx * stride + tt;
    }
    ggml_backend_tensor_set(t, indices.data(), 0,
                            indices.size() * sizeof(int32_t));
}
