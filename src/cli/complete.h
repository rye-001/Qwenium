#pragma once
// complete.h — single-prompt, non-interactive generation.
//
// Responsibility: the -p/--prompt path. One prefill, decode to a token budget or
//   stop condition, print, exit. Same engine seams as chat.h with no history and
//   no turn rendering, which makes it the path the byte-identical text gates use
//   (greedy -t 0 + --log-tokens-to; see docs/architecture.md §11).
// No unit test: exercised by the integration tests under tests/integration/ and
//   by the release text gates.

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ggml.h"
#include "ggml-backend.h"

#include "cli_args.h"
#include "speculative_bridge.h"

#include "engine/model.h"
#include "../loader/tokenizer.h"
#include "../sampling/sampling.h"
#include "../sampling/grammar_vocab.h"
#include "../sampling/speculative.h"
#include "../sampling/vocab_utils.h"

class Model;

namespace qinf {
    class SpeculativeDecoder;
}

/// Run single-prompt (non-interactive) generation.
/// Returns process exit code.
int run_complete(
    Model& model,
    const CliArgs& args,
    std::unique_ptr<qinf::GrammarVocab>& grammar,        // nullable
    qinf::SpeculativeDecoder* spec,                  // nullable
    bool use_speculative,
    std::function<void(int32_t)> log_token,
    std::function<void(const std::vector<int32_t>&)> log_tokens
);