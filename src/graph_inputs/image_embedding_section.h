#pragma once
// image_embedding_section.h — V1 image-embedding artifact (vision track).
//
// Serializes the vision encoder's host-side output — the context-free image
// embedding ([n_embd, n_tokens] row-major, n_embd fastest, exactly
// ImageEmbeddingsInput's layout / IVisionEncoder::encode's return) — so it can
// be cached to disk and reused to SKIP the ViT pass on a later turn / process /
// session for the same image. See docs/plan-session-snapshot.md (Vision-side
// artifacts, V1) and the cache-boundary rule: SigLIP (bidirectional) caches the
// ENCODED form; gemma4uv (causal) the RAW pre-LM form — both are exactly the
// encoder output, so one section type serves both.
//
// Lane Control: this is input data, not append-KV or overwrite-recurrent state.
// Co-located with image_embeddings_input (graph_inputs) per the plan — snapshot-
// ting an image IS serializing that input.

#include <cstdint>
#include <vector>

#include "session/section_ids.h"
#include "session/snapshot_section.h"

namespace qinf::session {
class SnapshotWriter;
class SnapshotReader;
}  // namespace qinf::session

namespace qinf::vision {

// The cacheable encoder output plus the identity needed to validate reuse.
struct ImageEmbedding {
    uint64_t content_id = 0;   // FNV hash of the normalized pixels (Bitmap::content_id)
    uint32_t n_embd = 0;       // projection_dim (== host model embedding_length)
    uint32_t n_tokens = 0;     // image soft-tokens (SigLIP: 256; gemma4uv: per-image)
    std::vector<float> data;   // size == n_embd * n_tokens, row-major n_embd fastest
};

// SnapshotSection wrapper for one ImageEmbedding. Borrows the struct.
class ImageEmbeddingSection : public qinf::session::SnapshotSection {
public:
    explicit ImageEmbeddingSection(ImageEmbedding& e) : e_(e) {}
    qinf::session::SectionId id() const override {
        return qinf::session::kImageEmbeddingSectionId;
    }
    qinf::session::StateLane lane() const override {
        return qinf::session::StateLane::Control;
    }
    void write(qinf::session::SnapshotWriter& w) const override;
    void read(qinf::session::SnapshotReader& r) override;

private:
    ImageEmbedding& e_;
};

}  // namespace qinf::vision
