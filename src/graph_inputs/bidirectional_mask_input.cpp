#include "bidirectional_mask_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <stdexcept>
#include <vector>

BidirectionalMaskInput BidirectionalMaskInput::open(std::string slot) {
    return BidirectionalMaskInput(std::move(slot), 0);
}

BidirectionalMaskInput BidirectionalMaskInput::band(std::string slot, uint32_t radius) {
    if (radius == 0)
        throw std::runtime_error(
            "BidirectionalMaskInput::band: slot '" + slot +
            "' radius: expected > 0 (use open() for no band), got: 0");
    return BidirectionalMaskInput(std::move(slot), radius);
}

void BidirectionalMaskInput::set_input(const StepContext& step) {
    ggml_tensor* t = require_tensor(step, slot_.c_str(), GGML_TYPE_F32);
    const uint32_t n_kv   = static_cast<uint32_t>(t->ne[0]);
    const size_t   n_rows = step.n_rows();
    const int64_t  r      = static_cast<int64_t>(radius_);

    std::vector<float> mask(static_cast<size_t>(n_kv) * n_rows, 0.0f);
    if (radius_ > 0) {
        for (size_t row = 0; row < n_rows; ++row) {
            const int64_t q = step.row_kv(row);
            for (uint32_t j = 0; j < n_kv; ++j) {
                const int64_t d = q - static_cast<int64_t>(j);
                if (d < -r || d > r) mask[row * n_kv + j] = -INFINITY;
            }
        }
    }
    ggml_backend_tensor_set(t, mask.data(), 0, mask.size() * sizeof(float));
}
