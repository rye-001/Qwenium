#pragma once
// chat.h — the interactive multi-turn chat loop.
//
// Responsibility: drive one terminal conversation over a loaded model — render
//   each turn through the family's chat template, prefill, decode to a stop
//   condition, print, repeat. Owns the conversation history and the per-turn
//   grammar/sampler wiring; owns no model state (the recipe does).
// Seams it drives: decode_step (per token), the speculative bridge, vision via
//   vision_profile + image/image_loader, and the opt-in warm-KV caches
//   (--prefix-cache, --image-prefix-cache) through session/.
// Known behaviour debt (architecture.md §12): the loop does not terminate on
//   stdin EOF — feed it `printf 'prompt\n\nexit\n' | qwenium ...` in scripts.
// No unit test: an interactive loop over a real model. Covered end-to-end by
//   tests/smoke/ (image coherence, conversational mode).

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ggml.h"
#include "ggml-backend.h"

#include "cli-args.h"
#include "speculative-bridge.h"

#include "engine/model.h"
#include "../loader/tokenizer.h"
#include "../sampling/sampling.h"
#include "../sampling/grammar_vocab.h"
#include "../sampling/speculative.h"
#include "../sampling/vocab_utils.h"



class Model;
struct ModelMetadata;
class ForwardPassBase;
class Tokenizer;
struct ChatMessage;

namespace qwenium {
    class Sampler;
    class SpeculativeDecoder;
}

/// Run the interactive multi-turn chat loop.
/// Returns process exit code (0 on clean exit).
int run_chat(
    Model& model,
    const CliArgs& args,
    std::unique_ptr<qwenium::GrammarVocab>& grammar,        // nullable, may be reset per turn
    qwenium::SpeculativeDecoder* spec,                  // nullable
    bool use_speculative,
    std::function<void(int32_t)> log_token,
    std::function<void(const std::vector<int32_t>&)> log_tokens
);