#pragma once
// embed_mode.h — the CLI's --embed path: text in, one embedding per text out.
//
// Responsibility: the front end over engine/embed. Read the texts (-p TEXT, or
//   --embed-in FILE with one text per line) — or one image (--image PATH with
//   --mmproj, the embedding model's gemma4v image tower; docs/plan-embed-image.md
//   G6) — tokenize, frame, embed, and write
//   one JSON object per text — JSONL — to --embed-out FILE or to stdout:
//     {"index":0,"n_tokens":18,"dim":768,"embedding":[0.0287212,...]}
//   Floats are printed with 9 significant digits, enough to round-trip a float.
//   Only models whose recipe implements IEmbeddingOutput embed; any other model
//   is refused by name.
// Flags that mean nothing for an embedding (chat, grammar, speculative,
//   sessions, caches, persistent graph, a KV type) are refused before the model
//   loads — a flag the run would silently ignore is a misread run waiting to
//   happen. Allowed alongside: --flash-attn (the one pass), --ctx-size (the
//   longest text accepted, in tokens), --mmap-weights, -v. Sampling flags
//   (-t, -n, --top-k, ...) have no effect.
//
// Unit test: tests/unit/test_embed_mode.cpp (pure helpers always; the end-to-end
//   path against the llama.cpp fixture when EMBEDDINGGEMMA2_MODEL_PATH is set).

#include <cstddef>
#include <istream>
#include <string>
#include <vector>

#include "cli_args.h"

class Model;

// Empty when `args` is a valid --embed invocation; otherwise the refusal, in
// the fail-loud form (parameter, expected, actual). Checked before load.
std::string embed_mode_refusal(const CliArgs& args);

// One text per line. A trailing '\r' is dropped (CRLF files). An empty line is
// refused, naming `source` and the 1-based line number — an accidental blank
// line would otherwise yield a vector for [BOS][EOS] and shift every index
// after it. An input with no lines at all is refused too.
std::vector<std::string> read_embed_inputs(std::istream& in, const std::string& source);

// One JSONL record (no trailing newline).
std::string format_embedding_line(size_t index, size_t n_tokens,
                                  const std::vector<float>& embedding);

// Run --embed on a loaded model. Returns the process exit code; errors are
// reported on stderr.
int run_embed(Model& model, const CliArgs& args);
