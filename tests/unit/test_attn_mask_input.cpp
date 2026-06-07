// test_attn_mask_input.cpp — AttnMaskInput produces the exact causal /
// sliding-window mask the recipes' hand-rolled loops did (bit-identical).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include "../../src/graph_inputs/attn_mask_input.h"

namespace {

struct Harness {
    ggml_context*        ctx  = nullptr;
    ggml_backend_t       be   = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph*         gf   = nullptr;
    ggml_tensor*         mask = nullptr;

    Harness(uint32_t n_kv, uint32_t n_rows, const char* name) {
        ggml_init_params p{ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
                            nullptr, true };
        ctx  = ggml_init(p);
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, 1, 1, n_rows);
        ggml_set_input(mask);
        ggml_set_name(mask, name);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, mask);
        be  = ggml_backend_cpu_init();
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    }
    ~Harness() {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(be);
        ggml_free(ctx);
    }
};

float ref(int64_t q_pos, uint32_t j, uint32_t window) {
    bool causal = (int64_t)j <= q_pos;
    bool in_win = (window == 0) ||
                  (q_pos - (int64_t)j < (int64_t)window);
    return (causal && in_win) ? 0.0f : -INFINITY;
}

// Reference with a bidirectional span [lo, hi): within-span query/key pairs
// attend both ways, bypassing causal+window; everything else is causal+window.
float ref_bidi(int64_t q_pos, uint32_t j, uint32_t window,
               int64_t lo, int64_t hi) {
    bool causal = (int64_t)j <= q_pos;
    bool in_win = (window == 0) ||
                  (q_pos - (int64_t)j < (int64_t)window);
    bool q_in = q_pos >= lo && q_pos < hi;
    bool j_in = (int64_t)j >= lo && (int64_t)j < hi;
    bool allow = (q_in && j_in) || (causal && in_win);
    return allow ? 0.0f : -INFINITY;
}

}  // namespace

TEST(AttnMaskInput, PrefillCausalNoWindow) {
    const uint32_t n_kv = 6, n_rows = 4;
    Harness h(n_kv, n_rows, "kq_mask.0");

    std::vector<int32_t> toks(n_rows, 0);
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;
    step.pos = 0;  // contiguous: row r -> q_pos = r

    AttnMaskInput in("kq_mask.0", 0u);
    in.set_input(step);

    std::vector<float> got(n_kv * n_rows);
    ggml_backend_tensor_get(h.mask, got.data(), 0, got.size() * sizeof(float));
    for (uint32_t r = 0; r < n_rows; ++r)
        for (uint32_t j = 0; j < n_kv; ++j)
            EXPECT_FLOAT_EQ(got[r * n_kv + j], ref(r, j, 0))
                << "r=" << r << " j=" << j;
}

TEST(AttnMaskInput, SlidingWindowMatchesReference) {
    const uint32_t n_kv = 12, n_rows = 8, window = 3;
    Harness h(n_kv, n_rows, "kq_mask.1");

    std::vector<int32_t> toks(n_rows, 0);
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;
    step.pos = 0;

    AttnMaskInput in("kq_mask.1", window);
    in.set_input(step);

    std::vector<float> got(n_kv * n_rows);
    ggml_backend_tensor_get(h.mask, got.data(), 0, got.size() * sizeof(float));
    for (uint32_t r = 0; r < n_rows; ++r)
        for (uint32_t j = 0; j < n_kv; ++j)
            EXPECT_FLOAT_EQ(got[r * n_kv + j], ref(r, j, window));
}

TEST(AttnMaskInput, BatchedExplicitPositions) {
    const uint32_t n_kv = 10, n_rows = 3;
    Harness h(n_kv, n_rows, "kq_mask_b");

    std::vector<int32_t> toks(n_rows, 0);
    std::vector<int32_t> positions{7, 2, 9};
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;
    step.positions = &positions;  // batched: q_pos = positions[r]

    AttnMaskInput in("kq_mask_b", 0u);
    in.set_input(step);

    std::vector<float> got(n_kv * n_rows);
    ggml_backend_tensor_get(h.mask, got.data(), 0, got.size() * sizeof(float));
    for (uint32_t r = 0; r < n_rows; ++r)
        for (uint32_t j = 0; j < n_kv; ++j)
            EXPECT_FLOAT_EQ(got[r * n_kv + j], ref(positions[r], j, 0));
}

// ── Phase 4 (Gemma 3 vision): image-span bidirectional mask ──────────────────
//
// Sequence of 12 positions, image span [4, 8) (positions 4,5,6,7). Three
// configs from docs/plan-gemma-vision-impl.md Phase 4, asserted on BOTH a
// global layer (window 0) and a local layer (window 6, tight enough that the
// span fits within a window of the immediately-following text token but text
// pairs further apart are cut off — exactly the 5:1 sliding-window composition
// the plan flags as the sharp edge). Full-grid equality vs ref_bidi pins the
// composed rule; explicit hand-computed cells below document the three configs.
namespace {
constexpr uint32_t kNkv = 12, kNrows = 12;
constexpr int64_t  kLo = 4, kHi = 8;  // image span [4,8), len 4

void run_bidi(uint32_t window, std::vector<float>& got) {
    Harness h(kNkv, kNrows, "kq_mask.0");
    std::vector<int32_t> toks(kNrows, 0);
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;
    step.pos = 0;  // contiguous: q_pos = r, so positions == indices
    AttnMaskInput in("kq_mask.0", window, /*bidi_start=*/(int32_t)kLo,
                     /*bidi_len=*/(uint32_t)(kHi - kLo));
    in.set_input(step);
    got.assign(kNkv * kNrows, 0.0f);
    ggml_backend_tensor_get(h.mask, got.data(), 0, got.size() * sizeof(float));
}
auto cell = [](const std::vector<float>& g, uint32_t q, uint32_t j) {
    return g[q * kNkv + j];
};
}  // namespace

TEST(AttnMaskInput, ImageSpanBidiMatchesReference_GlobalAndLocal) {
    for (uint32_t window : {0u, 6u}) {
        std::vector<float> got;
        run_bidi(window, got);
        for (uint32_t q = 0; q < kNrows; ++q)
            for (uint32_t j = 0; j < kNkv; ++j)
                EXPECT_FLOAT_EQ(cell(got, q, j),
                                ref_bidi(q, j, window, kLo, kHi))
                    << "window=" << window << " q=" << q << " j=" << j;
    }
}

TEST(AttnMaskInput, ImageSpanBidi_ThreeConfigs) {
    for (uint32_t window : {0u, 6u}) {
        std::vector<float> g;
        run_bidi(window, g);
        const float OK = 0.0f, NO = -INFINITY;

        // (b) image-span tokens attend bidirectionally to each other — including
        //     the "future" direction a causal mask would forbid.
        EXPECT_FLOAT_EQ(cell(g, 4, 7), OK) << "w=" << window;  // q before key, both in span
        EXPECT_FLOAT_EQ(cell(g, 7, 4), OK) << "w=" << window;  // reverse direction
        EXPECT_FLOAT_EQ(cell(g, 5, 6), OK) << "w=" << window;

        // (c) text after the span attends causally to the ENTIRE span (the span
        //     fits inside the window from the immediately-following token).
        EXPECT_FLOAT_EQ(cell(g, 8, 4), OK) << "w=" << window;  // 8-4=4 < 6
        EXPECT_FLOAT_EQ(cell(g, 8, 7), OK) << "w=" << window;

        // (a) text prefix is causal+window, untouched by the bidi span: a text
        //     query never attends to a "future" key (image or text).
        EXPECT_FLOAT_EQ(cell(g, 2, 5), NO) << "w=" << window;  // text q, future key
        EXPECT_FLOAT_EQ(cell(g, 3, 4), NO) << "w=" << window;  // text q just before span, future

        // Bidi does NOT leak past the span boundary: an image query does not see
        // future text, and attends to earlier text only causally+windowed.
        EXPECT_FLOAT_EQ(cell(g, 7, 8), NO) << "w=" << window;  // image q, future text
        EXPECT_FLOAT_EQ(cell(g, 4, 2), OK) << "w=" << window;  // image q, past text (4-2=2<6)
    }

    // Window genuinely bites text–text pairs on the local layer (proves the
    // bidi relaxation didn't disable the sliding window elsewhere).
    std::vector<float> g6;
    run_bidi(6u, g6);
    EXPECT_FLOAT_EQ(cell(g6, 11, 2), -INFINITY);  // 11-2=9 >= 6 → cut off
    std::vector<float> g0;
    run_bidi(0u, g0);
    EXPECT_FLOAT_EQ(cell(g0, 11, 2), 0.0f);       // global: causal, no cutoff
}

TEST(AttnMaskInput, FailLoudOnAbsentSlot) {
    Harness h(4, 2, "kq_mask.0");
    std::vector<int32_t> toks(2, 0);
    StepContext step;
    step.gf = h.gf;
    step.tokens = &toks;

    AttnMaskInput in("kq_mask.99", 0u);  // not in graph
    EXPECT_THROW(in.set_input(step), std::runtime_error);
}
