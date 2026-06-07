#include "attn_mask_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <vector>

void AttnMaskInput::set_input(const StepContext& step) {
    ggml_tensor* t = require_tensor(step, slot_.c_str(), GGML_TYPE_F32);

    const uint32_t n_kv   = static_cast<uint32_t>(t->ne[0]);
    const size_t   n_rows = step.n_rows();

    // Bidirectional image span (absolute positions), disabled when bidi_len_==0.
    const bool    has_bidi  = (bidi_len_ > 0) && (bidi_start_ >= 0);
    const int64_t span_lo   = bidi_start_;
    const int64_t span_hi   = bidi_start_ + static_cast<int64_t>(bidi_len_);  // exclusive

    std::vector<float> mask(static_cast<size_t>(n_kv) * n_rows);
    for (size_t r = 0; r < n_rows; ++r) {
        const int64_t q_pos = step.row_pos(r);
        const bool q_in_span = has_bidi && q_pos >= span_lo && q_pos < span_hi;
        for (uint32_t j = 0; j < n_kv; ++j) {
            const bool causal = (static_cast<int64_t>(j) <= q_pos);
            const bool in_win = (window_ == 0) ||
                                (q_pos - static_cast<int64_t>(j) <
                                 static_cast<int64_t>(window_));
            // Both query and key inside the image span => attend regardless of
            // causal order or window (image tokens see the whole image). Every
            // other pair keeps the standard causal+window rule.
            const bool j_in_span = has_bidi &&
                                   static_cast<int64_t>(j) >= span_lo &&
                                   static_cast<int64_t>(j) < span_hi;
            const bool allow = (q_in_span && j_in_span) || (causal && in_win);
            mask[r * n_kv + j] = allow ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(t, mask.data(), 0,
                            static_cast<size_t>(n_kv) * n_rows * sizeof(float));
}
