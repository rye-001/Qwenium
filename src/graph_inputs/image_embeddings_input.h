#pragma once

#include "graph_input.h"

#include <cstdint>
#include <vector>

// Owns the "image_embeddings" slot: a [n_embd, n_image_tokens] F32 tensor
// holding precomputed image-token embeddings that a multimodal recipe
// substitutes into its residual stream at image-placeholder positions
// (see docs/plan-gemma-vision-impl.md Phase 3).
//
// Unlike SparseHeadInput / GatherIndicesInput — which read per-step data out
// of StepContext — this input carries its payload directly. The embeddings
// are bulk per-image data produced upstream by the vision encoder (Phase 2)
// or, in tests, a synthetic block; threading 2.6 MB through the shared
// StepContext (read by every text recipe) would leak a vision concept into
// architecture-agnostic infrastructure. Holding it here keeps the multimodal
// surface localized to the recipes that opt in. The recipe moves the payload
// in at build time; set_input uploads it to the graph slot.
//
// Layout contract: `data` is row-major with n_embd fastest — element (e, t)
// at index t*n_embd + e — exactly the layout VisionEncoder::encode returns
// ([projection_dim, mm_tokens_per_image]) and the layout ggml expects for the
// [n_embd, n_image_tokens] slot tensor.
class ImageEmbeddingsInput : public GraphInput {
public:
    ImageEmbeddingsInput(std::vector<float> data, const char* slot = "image_embeddings")
        : data_(std::move(data)), slot_(slot) {}

    void set_input(const StepContext& step) override;
    const char* slot_name() const override { return slot_; }

private:
    std::vector<float> data_;
    const char*        slot_;
};
