#pragma once
// session_mode.h — CLI surface for the L1 portable session snapshot
// (docs/plan-session-snapshot.md, Phase 2). The CLI is L1's first host: it owns
// a token vector + sampler + grammar in one generation loop, exactly the control
// state L1 captures.
//
//   --save-session <path>     snapshot the live session at --save-session-at and
//                             keep generating (so the same run yields the
//                             continuous tail to compare against).
//   --load-session <path>     resume a saved session: validate the header against
//                             this model (fail-loud on mismatch), feed_tokens the
//                             stored span, then continue decoding.
//
// The snapshot file carries control state only (token span + sampler RNG +
// grammar cursor) — NOT the grammar definition, which is request config: a
// --load-session run with a grammar must pass the same --grammar-file.

#include <cstdint>
#include <functional>
#include <memory>

#include "cli_args.h"

class Model;
namespace qinf {
class GrammarVocab;
}

// Runs the save/load session flow. Returns a process exit code (0 = success).
int run_session(Model& model, const CliArgs& args,
                std::unique_ptr<qinf::GrammarVocab>& grammar,
                const std::function<void(int32_t)>& log_token);
