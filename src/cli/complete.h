#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ggml.h"
#include "ggml-backend.h"

#include "cli-args.h"
#include "speculative-bridge.h"

#include "../core/model.h"
#include "../loader/tokenizer.h"
#include "../sampling/sampling.h"
#include "../sampling/grammar_vocab.h"
#include "../sampling/speculative.h"
#include "../sampling/vocab_utils.h"

class Model;

namespace qwenium {
    class SpeculativeDecoder;
}

/// Run single-prompt (non-interactive) generation.
/// Returns process exit code.
int run_complete(
    Model& model,
    const CliArgs& args,
    std::unique_ptr<qwenium::GrammarVocab>& grammar,        // nullable
    qwenium::SpeculativeDecoder* spec,                  // nullable
    bool use_speculative,
    std::function<void(int32_t)> log_token,
    std::function<void(const std::vector<int32_t>&)> log_tokens
);