#include "attn_mask_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <vector>

void AttnMaskInput::set_input(const StepContext& step) {
    ggml_tensor* t = require_tensor(step, slot_.c_str(), GGML_TYPE_F32);

    const uint32_t n_kv   = static_cast<uint32_t>(t->ne[0]);
    const size_t   n_rows = step.n_rows();

    std::vector<float> mask(static_cast<size_t>(n_kv) * n_rows);
    for (size_t r = 0; r < n_rows; ++r) {
        const int64_t q_pos = step.row_pos(r);
        for (uint32_t j = 0; j < n_kv; ++j) {
            const bool causal = (static_cast<int64_t>(j) <= q_pos);
            const bool in_win = (window_ == 0) ||
                                (q_pos - static_cast<int64_t>(j) <
                                 static_cast<int64_t>(window_));
            mask[r * n_kv + j] = (causal && in_win) ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(t, mask.data(), 0,
                            static_cast<size_t>(n_kv) * n_rows * sizeof(float));
}
