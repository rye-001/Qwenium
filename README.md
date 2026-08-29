# Qwenium

**The inference engine that shows its work.**

A self-contained C++ engine for the Qwen and Gemma families on
[ggml](https://github.com/ggerganov/llama.cpp/tree/master/ggml), serving 10
concurrent users from a single 32 GB Mac. Two model families on one
cross-family architecture. Vision. Speculative decoding with a trained MTP
head. Grammar-leashed output. Portable session snapshots. Warm conversational
serving. Metal throughout, with custom fused DeltaNet kernels.

Every answer comes with the working notes behind it: which parts of your input
it used, which parts it never touched, and a byte-identical re-run whenever you
want to check. No other local engine does this.

Prefill throughput and memory footprint match llama.cpp on the same file, and
the forward pass agrees token-for-token with HuggingFace `transformers` — the
model's own definition — not merely with another engine.

Three things it is built to be:

**Fast where it counts.** Metal throughout, batched multi-slot decode, three
tiers of KV reuse. See [Performance](#performance) for the numbers, ours and
theirs.

**Readable and changeable.** One directory per layer type, one file per model
recipe, tests co-located with their module. Adding a model is a recipe, not a
refactor; adding a layer type is a file, not a factory. The architecture is a
written contract ([`CLAUDE.md`](CLAUDE.md),
[`docs/architecture.md`](docs/architecture.md)), enforced by a cross-family
rule: no forward-pass interface is done until it hosts both a Qwen *and* a
Gemma recipe without bending.

**Auditable.** The part below.

## Answers, with receipts

Hand a contract to an assistant and ask: *"What's the penalty if we cancel
late?"*

A normal engine gives you an answer. It sounds confident. You have no idea
whether it actually read page 7, where the exception is buried.

Qwenium gives you the answer **plus the three things a careful assistant would
volunteer:**

1. **"Here's where I got it."** The exact spans the answer drew on, as
   per-value source citations.
2. **"Here's what I never read."** Page 7 was never consulted. *This is the one
   nobody else hands you, and it is the difference between trusting an answer
   and trusting a guess.*
3. **"Check me if you like."** Ask again and get the identical answer, byte for
   byte. It cannot quietly change its story.

### How that is actually done

- **Where it looked — attention.** Every decode step's attention is
  materialized in the graph (not fused away) and tappable at zero extra
  compute. Calibrated, it becomes receipts: per-value **source citations**, a
  whole-document **coverage audit** ("this line was never consulted"), and an
  **ungrounded-value flag** — served via `/v1/extract` with
  `--attention-lens`.
- **What decided it — determinism.** Greedy decode is byte-deterministic, with
  forkable state (warm prefix restore + recurrent-state checkpoints). Remove
  one line of the input, re-run, and the diff is fact, not sampling noise —
  counterfactual "prove it" experiments per field.
- **Proof it happened — integrity.** A weights hash computed at load,
  version-gated snapshots, kernel-path tags, and fail-loud replay refusal: a
  generation can carry a kilobyte witness and re-run byte-identical on demand.


## What it does

- **Multi-family support** — Qwen 2/2.5/3/3.5/3.6 and Gemma 1/2/3/4, sharing the same layer modules and serving stack.
- **Hybrid architectures** — Pure transformer (Qwen3, Gemma), attention + SSM/GatedDeltaNet (Qwen3.5), DeltaNet + attention + MoE (Qwen3.6), and per-layer embeddings + sliding-window attention (Gemma 3/4).
- **Vision (multimodal)** — Image input for Gemma 3 (SigLIP encoder) and Gemma 4 (`gemma4uv` projector) via a `--mmproj` GGUF, in both the CLI (`--image`) and the server (OpenAI `image_url` content).
- **Batched inference** — Slot-based KV cache with batched decode. ~4× throughput over sequential processing for 10 concurrent users.
- **KV caching, three opt-in tiers** — `--prefix-cache` (disk-backed warm KV for a recurring system prompt, survives restarts), `--chat-prefix-cache` (transparent reuse of a slot's KV when a chat history re-arrives as a strict prefix; measured 2.4× per-turn at 4K context), and `--conversational` (explicit `conversation_id` handle: the server keeps the conversation's KV warm and clients send only the new turn). Image variants: `--image-embed-cache`, `--image-prefix-cache`.
- **OpenAI-compatible API** — `/v1/completions` and `/v1/chat/completions` with SSE streaming, per-request sampling params, and a `grammar` field for constrained output. Drop-in for existing tools.
- **Attention Lens (`/v1/extract`, `--attention-lens`)** — extraction with receipts: per-value source citations, a document coverage/omission audit, and an ungrounded-value flag, all read from the same forward pass at no extra compute.
- **Grammar-constrained generation** — GBNF grammars with a precomputed token-trie for fast constrained decoding; the valid-token set also drives a sparse LM head (skip logits for illegal tokens).
- **Session snapshots** — Save a mid-generation session to a portable file and resume it (`--save-session` / `--load-session`), byte-faithful across processes.
- **Speculative decoding** — Prompt-lookup based, no draft model needed.
- **Metal acceleration** — Apple Silicon via ggml's Metal backend, with custom fused DeltaNet kernels (`patches/`) for Qwen 3.5 / 3.6 decode.

## Supported Models

| Model | Architecture | Quant | Memory | Status |
|-------|--------------|-------|--------|--------|
| Qwen 2.5 Coder 14B Instruct | Transformer | Q4_0 | ~9 GB | ✅ Primary target |
| Qwen 3 1.7B / 0.6B | Transformer | BF16 / Q8_0 | ~3.4 GB / ~0.6 GB | ✅ Tested |
| Qwen 3.5 27B | Attention + SSM | Q4 / Q8 | ~16 GB | ✅ |
| Qwen 3.6 35B-A3B | DeltaNet + Attention + MoE | Q4 / Q8 | ~20 GB | ✅ |
| Gemma 1 (2B / 7B) | Transformer | Q4 / Q8 | ~1.5 GB / ~5 GB | ✅ |
| Gemma 2 (2B / 9B) | Transformer + logit soft-cap | Q4 / Q8 | ~1.5 GB / ~6 GB | ✅ |
| Gemma 3 (4B) | Transformer + sliding-window; vision via SigLIP mmproj | Q4 / Q8 | ~3 GB | ✅ |
| Gemma 4 (4B / 12B) | Transformer + PLE + pruned RoPE (dense & MoE); vision via `gemma4uv` mmproj | Q4 / Q8 | ~3–8 GB | ✅ |

## Quick Start

```bash
# Build (macOS)
mkdir build && cd build
cmake -DGGML_USE_METAL=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)

# Build (Linux)
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Start the server

```bash
./bin/http_server --model path/to/model.gguf --port 8080
# optional: --mmproj vision.gguf (image input) · --slots N · --ctx N
#           --prefix-cache DIR | --chat-prefix-cache | --conversational
```

```bash
# Generate
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write hello world in Python:", "max_tokens": 100}'

# Stream
curl -N http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quicksort:", "max_tokens": 200, "stream": true}'

# Chat (OpenAI messages format; add "grammar" for GBNF-constrained output,
# or an image_url content part when the server was started with --mmproj)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Fix this bug: ..."}], "max_tokens": 200}'

# Warm conversation (server started with --conversational): pass
# "conversation_id": "new" on the first turn, then reuse the returned id and
# send only the new user turn. DELETE /v1/conversations/{id} to clear.
```

### Or use the CLI

```bash
./bin/qwenium model.gguf --chat                  # interactive
./bin/qwenium model.gguf --chat --speculative    # with speculative decoding
./bin/qwenium model.gguf -p "Hello, world!"      # single prompt
./bin/qwenium model.gguf --chat \
    --mmproj vision.gguf --image photo.jpg       # ask about an image (Gemma 3/4, Qwen 3.5/3.6)
```

## Architecture

Qwenium is organized by concept, not by model. Each layer type is an independent module that builds into a shared `ggml_cgraph`. Models are recipes that compose layer modules.

```
src/
├── layers/                      # Layer-type modules (build graphs, share one ggml_context)
│   ├── attention.*              #   Plain + gated attention (build_attention, build_gated_attention)
│   ├── deltanet.*               #   Gated DeltaNet (Qwen3.5/3.6 recurrent path)
│   ├── ffn.*                    #   SwiGLU FFN
│   ├── moe.*                    #   Mixture-of-Experts dispatch (3x mul_mat_id)
│   ├── norm.*                   #   Shared RMS-norm helper (+ Gemma (1+w) variant)
│   ├── ple.*                    #   Per-layer embeddings (Gemma 4)
│   ├── layer.h                  #   Shared vocabulary: Phase, AttentionKind, LayerSpec
│   └── transformer_block.*      #   Per-layer composer (norm → attn → FFN → residual)
├── state/                       # Per-sequence state (KV + recurrent)
│   ├── kv_cache_simple.*        #   ggml-backed slot KV cache (append semantics)
│   ├── deltanet_state.*         #   Recurrent state for GatedDeltaNet
│   └── recurrent_state.*        #   Base for checkpoint/restore semantics
├── models/                      # Model recipes (compose layer modules)
│   ├── forward_pass_base.*      #   Shared graph scaffolding + the recipe interface
│   ├── model_registry.*         #   GGUF arch string → factory + inventory validator
│   ├── qwen3.*                  #   Qwen2/2.5/3 (pure transformer)
│   ├── qwen35.*                 #   Qwen3.5 (attention + SSM)
│   ├── qwen36.*                 #   Qwen3.6 (DeltaNet + attention + MoE)
│   └── gemma1-4.*               #   Gemma family (logit soft-cap, sliding-window, PLE, pruned RoPE)
├── graph_inputs/                # Typed graph inputs (tokens, positions, masks, image embeddings)
├── vision/                      # Image → soft-token encoders (SigLIP, gemma4uv, qwen3vl)
├── engine/                      # The loaded model + orchestration of one step over it
├── session/                     # Session persistence: snapshot format + slot/prefix/image caches
├── loader/                      # GGUF parsing, mmap tensor loading, tokenizer, chat templates
├── sampling/                    # Sampling, grammar, token-trie, speculative decoding
├── server/                      # Inference server + OpenAI-compatible HTTP API
├── image/                       # Host-side image pipeline (decode/resample/normalize, marker expansion)
└── cli/                         # Chat + single-prompt CLI
```

One inference thread owns the model. HTTP threads push requests to a queue and block on per-request token queues. All active slots decode together in a single batched forward pass.

For the full as-built map (codemap, dataflows, invariants), see [`docs/architecture.md`](docs/architecture.md); for the blueprint and rationale, see [`docs/modular-layer-architecture.md`](docs/modular-layer-architecture.md).

## Research & Experimentation

The small, modular codebase makes this useful for research that's painful in larger frameworks:

- **Layer skipping** — Comment out a `build_*` call in the model recipe, run the test suite, see what breaks.
- **Logit lens** — Project hidden states through the LM head after each layer to watch the model "decide".
- **Attention head pruning** — Zero out heads and measure impact on downstream tasks.
- **Activation patching** — Swap activations between runs to isolate what drives specific outputs.
- **New layer types** — Drop a file in `src/layers/` and a unit test alongside it. No factory plumbing, no inheritance hierarchies.
- **Custom sampling** — The sampler is ~180 lines; easy to add new strategies.

## Performance

Head-to-head against llama.cpp, both Release + Metal, same GGUF file, same
token IDs, one engine resident at a time. M1 Pro (32 GB), Qwen3.5-0.8B BF16,
756-token prompt, 128 decode steps, 3–6 runs per leg, run-to-run spread ~8%
(2026-08-23; llama.cpp `e85caa81e`, ours on the same ggml revision):

| Metric | Qwenium | llama.cpp |
|---|---|---|
| Prefill | ~2096 tok/s | ~2157 tok/s |
| Decode (batch 1) | ~55.8 tok/s | ~73.7 tok/s |
| Steady-state RSS | 1.75 GB | 1.72 GB |


Engine-level numbers, M1 Pro (32 GB), Qwen 2.5 Coder 14B Q4:

| Metric | Value |
|--------|-------|
| Throughput (10 concurrent) | ~4× vs sequential |
| Warm chat prefix hit (`--chat-prefix-cache`) | 2.4× per turn at 4K context (flat vs quadratic re-prefill) |
| Persistent decode graph (`--persistent-graph`) | 1.32× on Qwen 3.6 35B-A3B; ~1.0× on small models — the win is model-specific |
| DSL code generation | 100% pass rate (10 concurrent) |

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `GGML_USE_METAL` | OFF | Metal GPU acceleration (macOS) |
| `QWEN3_BUILD_TESTS` | OFF | Build unit + integration tests |
| `QWEN3_ENABLE_OPENMP` | ON | OpenMP parallelism |

## Tests

```bash
cmake -DQWEN3_BUILD_TESTS=ON -DGGML_USE_METAL=ON -DCMAKE_BUILD_TYPE=Debug ..
make -j8

./bin/unit-tests                     # per-module unit tests (no model needed)
./bin/integration-tests              # e2e generation (needs model files)
./bin/parallel-order-mngmt-test      # 10-concurrent DSL benchmark
```

Each module under `src/<dir>/<module>.cpp` has its unit test at `tests/unit/test_<module>.cpp`. Python benchmarks in `py/` for load testing and DSL validation.

## Dependencies

Fetched automatically via CMake:

- [ggml](https://github.com/ggerganov/llama.cpp) — tensor ops + Metal backend
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) — HTTP server
- [nlohmann/json](https://github.com/nlohmann/json) — JSON
- [Google Test](https://github.com/google/googletest) — testing

## Acknowledgements

Inspired by and built on concepts from [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml). This project wouldn't exist without the foundational work by Georgi Gerganov and the ggml community.

## Contributing

Issues and PRs welcome. The codebase is intentionally small and modular — please keep it that way. Locality-of-reasoning, test co-location, and concept-named directories are load-bearing rules; see `CLAUDE.md` for the short list.

## License

[MIT](LICENSE)
