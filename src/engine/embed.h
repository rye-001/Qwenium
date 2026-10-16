#pragma once
// embed.h — one text embedding from a recipe that offers IEmbeddingOutput.
//
// Responsibility: the model-agnostic half of an embedding. The recipe builds
//   the per-token vectors (models/i_embedding_output.h); this file frames the
//   token sequence the way the model was trained to read it, runs the pass,
//   pools (mean or last token, as the recipe reports) and L2-normalises. One
//   copy of that arithmetic for every family, so a Gemma and a Qwen embedding
//   cannot disagree on how a vector was reduced.
//
// Public surface:
//   frame              — [BOS] ids [EOS], as the ARCHITECTURE's TokenizerConfig says
//   pool_and_normalize — pure: per-token vectors → one unit vector (no model)
//   run                — build → alloc → inputs → compute → read → pool
//   image_input        — one image as an EmbedInput: [BOS] <boi> soft×n <eoi> [EOS]
//   run(EmbedInput)    — the same pass with image soft tokens armed (Seam B)
//
// This is where a client gets an embedding (architecture.md §6): the lens
// server, and the CLI's --embed (S2). Retrieval, ranking and indexes are the
// client's — the engine stops at a normalised vector.
//
// Unit test: tests/unit/test_embed.cpp (pure tests always; the model gate —
//   cosine against llama.cpp's embeddings — when EMBEDDINGGEMMA2_MODEL_PATH is set).

#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "../loader/tokenizer_config.h"
#include "../models/forward_pass_base.h"
#include "../models/i_embedding_output.h"
#include "../models/i_image_embeddable.h"

namespace qinf::engine::embed {

// Prepend BOS when cfg.add_bos_token, append EOS when cfg.add_eos_token. `cfg`
// is the registry's config for the architecture (lookup_tokenizer_config) —
// the architecture decides, not the GGUF's flags, which an export can get
// wrong (model.cpp, the Unsloth Gemma case). Fail-loud when cfg asks for a
// token whose id the file does not carry.
std::vector<int32_t> frame(const TokenizerConfig& cfg, const ModelMetadata& meta,
                           std::vector<int32_t> ids);

// per_token: [dim × n_tokens], dim fastest (the "embd_tokens" layout).
// Mean = the average over all n_tokens rows (BOS and EOS included, as llama.cpp
// and sentence-transformers do); Last = the final row. The result is scaled to
// unit L2 norm. Accumulates in double. Fail-loud on a size mismatch, zero
// tokens, or a zero vector (which has no direction to normalise).
std::vector<float> pool_and_normalize(const std::vector<float>& per_token,
                                      uint32_t dim, uint32_t n_tokens,
                                      IEmbeddingOutput::Pooling pooling);

// One embedding of an already-framed sequence. `fp` must implement
// IEmbeddingOutput (fail-loud, naming the architecture, otherwise). Uses
// `sched` for this one graph and leaves it reset-able by the next caller.
std::vector<float> run(ForwardPassBase& fp, ggml_backend_sched_t sched,
                       const std::vector<int32_t>& tokens,
                       const std::string& architecture);

// Image soft tokens placed in a framed sequence (docs/plan-embed-image.md G4):
// `embd` is [hidden_dim × n_tokens] from an IVisionEncoder (the encoder's
// projection_dim equals the text model's width); it replaces the token
// embeddings at [span_start, span_start + n_tokens) of EmbedInput::tokens.
struct EmbedImageSpan {
    int32_t            span_start = -1;
    uint32_t           n_tokens   = 0;
    std::vector<float> embd;
};

// An already-framed sequence plus its image spans. At most ONE image per input
// today (Seam B arms one span per build); more is refused fail-loud.
struct EmbedInput {
    std::vector<int32_t>        tokens;
    std::vector<EmbedImageSpan> images;
};

// One image as the model reads it (llama.cpp mtmd framing, model card: no
// prefix on images): frame(<boi>, soft × n, <eoi>), i.e. [BOS] <|image>
// <|image|>×n <image|> [EOS] for EmbeddingGemma-2. The soft ids are
// placeholders the splice overwrites. Marker ids come from the vision profile.
EmbedInput image_input(const TokenizerConfig& cfg, const ModelMetadata& meta,
                       int32_t boi_id, int32_t soft_id, int32_t eoi_id,
                       std::vector<float> embd, uint32_t n_tokens);

// The pass with images: arms each span on the recipe's IImageEmbeddable (fail-
// loud, naming the architecture, when it has none), then exactly run(tokens).
// No images ⇒ identical to run(fp, sched, input.tokens, architecture).
std::vector<float> run(ForwardPassBase& fp, ggml_backend_sched_t sched,
                       EmbedInput input, const std::string& architecture);

}  // namespace qinf::engine::embed
