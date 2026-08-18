#include "kv_write_indices_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdexcept>
#include <string>
#include <vector>

void KvWriteIndicesInput::set_input(const StepContext& step) {
    if (!step.slots)
        throw std::runtime_error(
            "KvWriteIndicesInput: slot 'kv_write_indices': expected batched "
            "slot list, got: null StepContext::slots");

    ggml_tensor* t = require_tensor(step, slot_, GGML_TYPE_I64);

    const size_t n_rows = step.n_rows();
    if (static_cast<size_t>(t->ne[0]) != n_rows)
        throw std::runtime_error(
            "KvWriteIndicesInput: slot 'kv_write_indices': expected " +
            std::to_string(n_rows) + " rows, got: " +
            std::to_string(t->ne[0]));

    std::vector<int64_t> indices(n_rows);
    for (size_t r = 0; r < n_rows; ++r) {
        const int32_t pos = step.row_pos(r);
        if (pos < 0 || static_cast<uint32_t>(pos) >= n_ctx_max_)
            throw std::runtime_error(
                "KvWriteIndicesInput: slot 'kv_write_indices': expected "
                "position in [0, " + std::to_string(n_ctx_max_) +
                "), got: " + std::to_string(pos) + " (batch row " +
                std::to_string(r) + ")");
        indices[r] = static_cast<int64_t>((*step.slots)[r]) * n_ctx_max_ + pos;
    }
    ggml_backend_tensor_set(t, indices.data(), 0,
                            indices.size() * sizeof(int64_t));
}
