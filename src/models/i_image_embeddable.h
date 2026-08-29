#pragma once
// i_image_embeddable.h — Seam B (docs/plan-gemma4-vision-impl.md §3).
//
// The text-recipe side of the vision boundary, extracted when Gemma 4 became
// the second multimodal recipe. A recipe that implements this can have a span
// of precomputed image-token embeddings armed into its next prefill; the recipe
// owns the residual stream and performs the substitution (after its embed
// scale, so image rows enter unscaled). It does NOT know how the embeddings
// were produced — vision stays out of ForwardPassBase and StepContext.
//
// `prefill_multimodal` (the one orchestrator) targets IImageEmbeddable +
// IVisionEncoder instead of the concrete Gemma 3 types. Both Gemma3ForwardPass
// and Gemma4ForwardPass implement this; the method shape was already present on
// Gemma 3 before the extraction, so this is the existing contract named, not a
// new one invented to anticipate a hypothetical recipe.

#include <cstdint>
#include <vector>

class IImageEmbeddable {
public:
    virtual ~IImageEmbeddable() = default;

    // Arm precomputed image-token embeddings before the next prefill build.
    // `embd` is [hidden_dim * n_tokens] row-major (hidden_dim fastest) — exactly
    // the layout IVisionEncoder::encode returns. `span_start` is the LOCAL batch
    // column index of the first image soft-token; the n_tokens embeddings replace
    // the scaled text embeddings at [span_start, span_start + n_tokens). Consumed
    // (moved out) by the next build_prefill_graph; re-arm to repeat.
    //
    // `grid_w` / `grid_h` are the image's soft-token grid shape (nx by ny,
    // nx*ny == n_tokens), added for M-RoPE (P4). A recipe with 2-D positions
    // needs the row/column of each soft token; one with scalar positions
    // ignores them, which is why they default to 0 = "no spatial structure
    // supplied". Optional parameters rather than a widened struct, so hosting
    // M-RoPE required no logic edits in the recipes that don't use it.
    virtual void set_image_embeddings(std::vector<float> embd,
                                      int32_t span_start,
                                      uint32_t n_tokens,
                                      uint32_t grid_w = 0,
                                      uint32_t grid_h = 0) = 0;

    // Does an image span occupy a 2-D block of POSITIONS in this recipe, or a
    // 1-D run? (P4.) It decides how far the sequence position advances across
    // the span: max(nx, ny) when 2-D, n_tokens when not.
    //
    // False for every scalar-position recipe, where one soft token is one
    // position and the two answers coincide. True only under M-RoPE, where the
    // image's rows and columns share positions, so the span advances the
    // position far less than it advances the KV.
    //
    // The orchestrator asks; the recipe answers. Neither hardcodes a family.
    virtual bool image_span_is_2d() const { return false; }
};
