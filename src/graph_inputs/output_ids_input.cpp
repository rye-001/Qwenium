#include "output_ids_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>
#include <string>
#include <vector>

void OutputIdsInput::set_input(const StepContext& step) {
    const size_t n_rows = step.n_rows();

    // absent: require_tensor throws naming the slot if the tensor is missing.
    ggml_tensor* t = require_tensor(step, slot_, GGML_TYPE_I32);

    const int64_t k = t->ne[0];

    // empty when N>0: the slot exists but selects zero positions while there
    // are rows to choose from. Name slot, expected range, actual value.
    if (k <= 0 && n_rows > 0)
        throw std::runtime_error(
            std::string("OutputIdsInput: slot '") + slot_ +
            "': expected at least 1 selected position in [0, " +
            std::to_string(n_rows) + "), got: 0 (empty selection with n_rows=" +
            std::to_string(n_rows) + ")");

    if (n_rows == 0)
        throw std::runtime_error(
            std::string("OutputIdsInput: slot '") + slot_ +
            "': expected n_rows >= 1, got: 0 (no tokens to select a head "
            "position from)");

    // Default index set: last token only. The discarded first n_rows-1 head
    // rows are exactly what this slice elides.
    if (k != 1)
        throw std::runtime_error(
            std::string("OutputIdsInput: slot '") + slot_ +
            "': expected slot width 1 (last-token-only default), got: " +
            std::to_string(k));

    const int32_t last = static_cast<int32_t>(n_rows) - 1;

    // out-of-range: defensive against a future caller-supplied set. Name slot,
    // expected range, actual value.
    if (last < 0 || static_cast<size_t>(last) >= n_rows)
        throw std::runtime_error(
            std::string("OutputIdsInput: slot '") + slot_ +
            "': expected position in [0, " + std::to_string(n_rows) +
            "), got: " + std::to_string(last));

    ggml_backend_tensor_set(t, &last, 0, sizeof(int32_t));
}
