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

}  // namespace qinf::engine::embed
