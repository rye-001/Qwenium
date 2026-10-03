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
        // The KV ROW, not the rope position: after an M-RoPE image they differ,
        // and writing at the position overwrites a row of the image.
        const int64_t row = step.row_kv(r);
        if (row < 0 || row >= static_cast<int64_t>(n_ctx_max_))
            throw std::runtime_error(
                "KvWriteIndicesInput: slot 'kv_write_indices': expected "
                "KV row in [0, " + std::to_string(n_ctx_max_) +
                "), got: " + std::to_string(row) + " (batch row " +
                std::to_string(r) + ")");
        indices[r] = static_cast<int64_t>((*step.slots)[r]) * n_ctx_max_ + row;
    }
    ggml_backend_tensor_set(t, indices.data(), 0,
                            indices.size() * sizeof(int64_t));
}
