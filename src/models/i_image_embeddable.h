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
    virtual void set_image_embeddings(std::vector<float> embd,
                                      int32_t span_start,
                                      uint32_t n_tokens) = 0;
};
