#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ggml.h"
#include "ggml-backend.h"

#include "cli-args.h"
#include "speculative-bridge.h"

#include "../qwen3-core/qwen3-model.h"
#include "../qwen3-core/forward-pass-factory.h"
#include "../qwen3-core/tokenizer.h"
#include "../sampling/sampling.h"
#include "../sampling/grammar.h"
#include "../sampling/speculative.h"
#include "../sampling/vocab_utils.h"



class Qwen3Model;
struct Qwen3Metadata;
class ForwardPassBase;
class Tokenizer;
struct ChatMessage;

namespace qwen3 {
    class Sampler;
    class Grammar;
    class SpeculativeDecoder;
}

/// Run the interactive multi-turn chat loop.
/// Returns process exit code (0 on clean exit).
int run_chat(
    Qwen3Model& model,
    const CliArgs& args,
    std::unique_ptr<qwen3::Grammar>& grammar,        // nullable, may be reset per turn
    qwen3::SpeculativeDecoder* spec,                  // nullable
    bool use_speculative,
    std::function<void(int32_t)> log_token,
    std::function<void(const std::vector<int32_t>&)> log_tokens
);