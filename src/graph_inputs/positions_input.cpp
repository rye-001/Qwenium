#include "positions_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

void PositionsInput::set_input(const StepContext& step) {
    const size_t n = step.n_rows();
    ggml_tensor* t = require_tensor(step, slot_, GGML_TYPE_I32);

    std::vector<int32_t> pos_data(n);
    for (size_t i = 0; i < n; ++i)
        pos_data[i] = step.row_pos(i);

    ggml_backend_tensor_set(t, pos_data.data(), 0, n * sizeof(int32_t));
}
