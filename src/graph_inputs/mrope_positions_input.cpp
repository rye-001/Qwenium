#include "mrope_positions_input.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

void MRopePositionsInput::set_input(const StepContext& step) {
    const size_t n = step.n_rows();
    ggml_tensor* t = require_tensor(step, slot_, GGML_TYPE_I32);

    // Four contiguous blocks of n: [t | h | w | e]. See the header — the
    // kernel indexes pos[i + k*n], so this is component-major.
    std::vector<int32_t> pos_data(n * kComponents);

    if (step.img_grid_w > 0) {
        // Image chunk. Ported from llama.cpp's mtmd_image_tokens_get_decoder_pos
        // (MTMD_POS_TYPE_MROPE) plus set_position_mrope_2d, which together fix
        // both the values and which component each lands in:
        //   t = pos0            (constant across the image)
        //   h = pos0 + row      (.y  -> component 1)
        //   w = pos0 + col      (.x  -> component 2)
        //   e = 0               (.z, unused for images)
        // Swapping h and w compiles, runs, and produces a transposed image —
        // the quiet failure this layout comment exists to prevent.
        const int32_t  pos0 = step.pos;
        const uint32_t nx   = step.img_grid_w;
        for (size_t i = 0; i < n; ++i) {
            const int32_t row = static_cast<int32_t>(i / nx);
            const int32_t col = static_cast<int32_t>(i % nx);
            pos_data[0 * n + i] = pos0;
            pos_data[1 * n + i] = pos0 + row;
            pos_data[2 * n + i] = pos0 + col;
            pos_data[3 * n + i] = 0;
        }
    } else {
        // Text: all four components carry the ordinary sequence position, which
        // is what makes ggml_rope_multi reduce to the NEOX rotation (P2).
        for (size_t i = 0; i < n; ++i) {
            const int32_t p = step.row_pos(i);
            for (int k = 0; k < kComponents; ++k)
                pos_data[k * n + i] = p;
        }
    }

    ggml_backend_tensor_set(t, pos_data.data(),
                            0, pos_data.size() * sizeof(int32_t));
}
