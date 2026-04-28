# Qwenium

A from-scratch LLM inference engine in C++ for the Qwen family, using [ggml](https://github.com/ggerganov/llama.cpp/tree/master/ggml). Serve 10 concurrent users from a single 32GB Mac.

**Small codebase, modular architecture, clear boundaries.** Pure transformers, hybrid SSM+attention, and MoE, all in a codebase you can read in an afternoon.

## Why not just use llama.cpp?

You probably should! [llama.cpp](https://github.com/ggerganov/llama.cpp) supports more models, more hardware, and more features. Qwenium exists because:

**If you need a small, self-contained inference server** — for an internal tool, a small team, a startup that wants to stop paying per-token API costs, or an on-prem deployment where data can't leave the building — this gives you exactly that with nothing extra.

**If you want to understand how LLM inference works** — llama.cpp is ~300K lines. Qwenium is an order of magnitude smaller, organized by concept with clear module boundaries (one directory per layer type, one file per model recipe). You can trace a request from HTTP to tensor op without getting lost.

**If you want to hack on the model itself** — adding a new layer type means one new file under `src/layers/`. Adding a new model means one new recipe under `src/models/`. Swap layers, add probes, prune attention heads, tap intermediate hidden states for [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) experiments. The kind of model surgery that's painful in a 300K-line codebase is straightforward here.

## What it does

- **Hybrid architectures** — Pure transformer (Qwen3), attention + SSM/GatedDeltaNet (Qwen3.5), and DeltaNet + attention + MoE (Qwen3.6).
- **Batched inference** — Slot-based KV cache with batched decode. ~4× throughput over sequential processing for 10 concurrent users.
- **Prefix caching** — System prompt KV state cloned per slot. 2.25× faster wall time, 5.3× faster time-to-first-token.
- **TurboQuant KV** — Compressed KV cache backed by Walsh–Hadamard rotation + Lloyd–Max quantization for long contexts.
- **SnapKV** — Importance-based KV eviction to keep cache bounded on long multi-turn conversations.
- **OpenAI-compatible API** — `/v1/completions` with SSE streaming. Drop-in for existing tools.
- **Grammar-constrained generation** — GBNF grammars with a precomputed token-trie for fast constrained decoding.
- **Speculative decoding** — Prompt-lookup based, no draft model needed.
- **Metal acceleration** — Apple Silicon via ggml's Metal backend.

## Supported Models

| Model | Architecture | Quant | Memory | Status |
|-------|--------------|-------|--------|--------|
| Qwen 2.5 Coder 14B Instruct | Transformer | Q4_0 | ~9 GB | ✅ Primary target |
| Qwen 3 1.7B / 0.6B | Transformer | BF16 / Q8_0 | ~3.4 GB / ~0.6 GB | ✅ Tested |
| Qwen 3.5 27B | Attention + SSM | Q4 / Q8 | ~16 GB | ✅ |
| Qwen 3.6 35B-A3B | DeltaNet + Attention + MoE | Q4 / Q8 | ~20 GB | ✅ |

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

# With cached system prompt (subsequent requests skip system prompt prefill)
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are a coding assistant.", "prompt": "Fix this bug:", "max_tokens": 200}'
```

### Or use the CLI

```bash
./bin/qwenium model.gguf --chat                  # interactive
./bin/qwenium model.gguf --chat --speculative    # with speculative decoding
./bin/qwenium model.gguf -p "Hello, world!"      # single prompt
```

## Architecture

Qwenium is organized by concept, not by model. Each layer type is an independent module that builds into a shared `ggml_cgraph`. Models are recipes that compose layer modules.

```
src/
├── layers/                      # Layer-type modules (build graphs, share one ggml_context)
│   ├── attention.*              #   Plain + gated attention (build_attention, build_gated_attention)
│   ├── deltanet.*               #   Gated DeltaNet (Qwen3.5/3.6 recurrent path)
│   ├── ffn.*                    #   SwiGLU FFN
│   ├── moe.*                    #   Mixture-of-Experts dispatch + residency
│   ├── norm.*                   #   Shared RMS-norm helper
│   └── transformer_block.*      #   Per-layer composer (norm → attn → FFN → residual)
├── state/                       # Per-sequence state (KV + recurrent)
│   ├── kv_cache_simple.*        #   ggml-backed slot KV cache
│   ├── kv_cache_compressed.*    #   TurboQuant CPU-side compressed store
│   ├── snapkv.*                 #   Importance-based KV eviction
│   ├── deltanet_state.*         #   Recurrent state for GatedDeltaNet
│   └── recurrent_state.*        #   Base for checkpoint/restore semantics
├── models/                      # Model recipes (compose layer modules)
│   ├── qwen3.*                  #   Qwen2/2.5/3 (pure transformer)
│   ├── qwen35.*                 #   Qwen3.5 (attention + SSM)
│   └── qwen36.*                 #   Qwen3.6 (DeltaNet + attention + MoE)
├── loader/                      # GGUF parsing, mmap tensor loading, tokenizer
├── sampling/                    # Sampling, grammar, token-trie, speculative decoding
├── server/                      # Inference server + OpenAI-compatible HTTP API
├── cli/                         # Chat + single-prompt CLI
└── metal/, quant/, telemetry/   # Backend dispatch, weight quant, observability
```

One inference thread owns the model. HTTP threads push requests to a queue and block on per-request token queues. All active slots decode together in a single batched forward pass.

For the architectural blueprint and rationale, see [`docs/modular-layer-architecture.md`](docs/modular-layer-architecture.md).

## Research & Experimentation

The small, modular codebase makes this useful for research that's painful in larger frameworks:

- **Layer skipping** — Comment out a `build_*` call in the model recipe, run the test suite, see what breaks.
- **Logit lens** — Project hidden states through the LM head after each layer to watch the model "decide".
- **Attention head pruning** — Zero out heads and measure impact on downstream tasks.
- **Activation patching** — Swap activations between runs to isolate what drives specific outputs.
- **New layer types** — Drop a file in `src/layers/` and a unit test alongside it. No factory plumbing, no inheritance hierarchies.
- **Custom sampling** — The sampler is ~180 lines; easy to add new strategies.

## Performance

Measured on M1 MacBook Pro (32GB) with Qwen 2.5 Coder 14B Q4:

| Metric | Value |
|--------|-------|
| Throughput (10 concurrent) | ~4× vs sequential |
| System prompt cache hit | 2.25× wall time, 5.3× TTFT |
| DSL code generation | 100% pass rate (10 concurrent) |

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `GGML_USE_METAL` | OFF | Metal GPU acceleration (macOS) |
| `QWEN3_BUILD_TESTS` | OFF | Build unit + integration tests |
| `QWEN3_ENABLE_OPENMP` | ON | OpenMP parallelism |
| `QINF_MOE_FALLBACK` | OFF | Use unfused MoE dispatch (reference / debug build) |

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
