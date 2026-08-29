#pragma once

#include "graph_input.h"

// Owns the "inp_pos" slot for a recipe that uses M-RoPE (ggml_rope_multi).
//
// P2 of docs/plan-qwen35-vision-impl.md. The Qwen 3.5 family declares
// `<arch>.rope.dimension_sections` in every GGUF, which splits the rotated
// dimensions into four sections that each read their OWN position component
// (t/h/w/e — time, height, width, extra). ggml therefore expects `inp_pos` to
// hold FOUR position components per token instead of one.
//
// Layout is COMPONENT-MAJOR, not interleaved. The kernel reads
//   p_t = pos[i],  p_h = pos[i + n],  p_w = pos[i + 2n],  p_e = pos[i + 3n]
// (ggml-cpu/ops.cpp, ggml_compute_forward_rope), so the buffer is four
// contiguous blocks of n_tokens, in that order. Interleaving would compile,
// run, and produce silently wrong rotations.
//
// TEXT-ONLY BEHAVIOUR (all this class does today): every component is set to
// the row's ordinary sequence position. With all four equal, ggml's
// mrope cache init reduces exactly to the NEOX cache init — same
// rotate_pairs(n_dims, n_dims/2) layout, same thetas — so the model's output
// is unchanged. That equality is the P2 gate.
//
// Images are what make the components diverge (an nx×ny merged-patch span
// occupies nx·ny rows but advances position by only max(nx, ny)). Nothing
// produces divergent components yet; the producer arrives with Seam B in P4.
class MRopePositionsInput : public GraphInput {
public:
    explicit MRopePositionsInput(const char* slot = "inp_pos") : slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

    // Number of position components ggml_rope_multi expects per token. Public
    // so the recipe can size its inp_pos tensor from the same constant that
    // fills it, rather than a hardcoded 4 in two places.
    static constexpr int kComponents = 4;

private:
    const char* slot_;
};
