# Qwenium Architecture

This is the **as-built map** of Qwenium: what exists, where it lives, and how a
request flows through it. It is the companion to two other documents:

- [`CLAUDE.md`](../CLAUDE.md) — the *rules* (collaboration invariants, workload
  envelope, ggml constraints). Short, load-bearing, always current.
- [`modular-layer-architecture.md`](modular-layer-architecture.md) — the
  *blueprint* (why the layer-module design was chosen, the refactor phasing).

This document deliberately stays at the map level. Detail that changes often —
exact signatures, line numbers, flag defaults — lives in the code and its
co-located tests; this doc names the file and the concept so you know where to
look. If a section here contradicts the code, the code is right and this doc
has rotted: fix it in the same PR.

---

## 1. What Qwenium is

A C++ inference engine for Qwen- and Gemma-family LLMs (GGUF format, K-quant
weights) on Apple Silicon, using [ggml](https://github.com/ggml-org/ggml) as
the tensor library and Metal as the GPU backend. It ships two front ends over
one engine:

- a **CLI** (`qwen3` binary): single-user chat/completion, with vision,
  grammar-constrained output, speculative decoding, and session snapshots;
- an **HTTP server**: OpenAI-compatible `/v1/completions` and
  `/v1/chat/completions`, serving up to ~10 concurrent requests by batching
  them into one forward pass.

The **workload envelope** is load-bearing and explains many "missing" features:
≤10 concurrent slots, ≤4K-context order-management-style prompts, ~12 GB
quantized models on unified memory. KV-cache *memory* is not the constraint in
this regime — which is why KV compression (TurboQuant) and KV eviction (SnapKV)
were built, measured, and then deleted. Decode-time *latency* is the constraint,
so the optimization surface is kernel launches, grammar-step cost, and prefill
reuse (caching).

Supported architectures (registered in
[`src/models/model_registry.cpp`](../src/models/model_registry.cpp)):
`qwen2`, `qwen3` (pure transformer), `qwen35` (attention + SSM hybrid),
`qwen35moe` (Qwen 3.6: DeltaNet + attention + MoE hybrid), `gemma`, `gemma2`,
`gemma3`, `gemma4` (dense and MoE variants; Gemma 3 and 4 with vision).
Gemma is not a courtesy port — it is the designated **cross-family forcing
function**: every forward-pass interface must be proven against at least one
Qwen and one Gemma recipe before it counts as done.

---

## 2. Bird's-eye view

```
                       ┌──────────────────────────────────────────────┐
                       │                 front ends                   │
                       │  src/cli/ (chat, complete, session_mode)     │
                       │  src/server/ (http_server → inference_server)│
                       └──────────────┬───────────────────────────────┘
                                      │ callbacks: prefill / batched_decode /
                                      │ clear_slot / tokenize / ... (slot_id + tokens)
                       ┌──────────────▼───────────────────────────────┐
                       │              engine core                     │
                       │  src/core/model.{h,cpp}   — owns weights,    │
                       │    backend, scheduler, the forward pass      │
                       │  src/core/decode_plan / decode_step          │
                       │  src/core/multimodal_prefill                 │
                       └──────┬───────────────┬───────────────────────┘
                              │               │
              ┌───────────────▼──┐   ┌────────▼─────────────────────┐
              │ recipes           │   │ decode-time algorithms       │
              │ src/models/       │   │ src/sampling/                │
              │ qwen3/35/36,      │   │ samplers, GBNF grammar +     │
              │ gemma1–4          │   │ token-trie, speculative (PLD)│
              └──────┬────────────┘   └──────────────────────────────┘
                     │ composes
              ┌──────▼────────────┐   ┌──────────────────────────────┐
              │ layer modules      │   │ typed graph inputs           │
              │ src/layers/        │   │ src/graph_inputs/            │
              │ attention, ffn,    │   │ tokens, positions, attn mask,│
              │ moe, deltanet,     │   │ sparse head, image embeds    │
              │ norm, ple, block   │   └──────────────────────────────┘
              └──────┬────────────┘
                     │ reads/writes
              ┌──────▼────────────────────────────────────────────────┐
              │ state: src/state/                                     │
              │ KV cache (append) ≠ recurrent state (overwrite)       │
              └───────────────────────────────────────────────────────┘

  side pipelines:  src/vision/  (image → soft tokens, joins at one seam)
                   src/session/ + src/core/{slot_snapshot,prefix_library}
                                (portable snapshots, warm-KV caches)
  foundation:      ggml (pinned, patched via patches/), Metal + CPU backends
```

A **model is a recipe**: a file in `src/models/` that owns the residual stream
and composes layer modules in order. Layer modules are graph-building
functions — they append nodes to a shared `ggml_cgraph`, they don't execute
anything. Family differences are parameters at the same call site (sliding
window, softcap, GEGLU-vs-SwiGLU, partial RoPE), not forked module copies. A
genuinely different signal flow (per-layer embeddings, DeltaNet) gets its own
module. That parameterize-vs-split judgment is made case by case and flagged
explicitly when it comes up — both "model zoo of near-identical modules" and
"one fat function of orthogonal knobs" are failure modes.

---

## 3. The ggml foundation (read this before touching any graph code)

ggml is **lazy and two-phase**. Phase one builds a graph: every `ggml_*` call
creates a *node* describing an operation; nothing computes. Phase two hands the
graph to a `ggml_backend_sched`, which allocates scratch memory, assigns each
node to Metal or CPU, and runs it. A `ggml_tensor*` is a plan node, not a
buffer of numbers.

Rules that follow from this (violations are the classic bug sources):

- **One context, one graph.** All layer modules build into the same
  `ggml_context`/`ggml_cgraph` per forward pass. Prefill and decode use
  *separate* graphs so the allocator sees one consistent shape each time.
- **Views are free; `ggml_cont` is not.** Reshape/permute/view just relabel
  strides. `ggml_cont`/`ggml_cpy` materialize data and *pin scratch memory* —
  only call them when an op genuinely requires contiguous input.
- **Side-effect nodes need explicit roots.** KV-cache writes (`ggml_cpy` into a
  view of the cache) feed nothing downstream; register them with
  `ggml_build_forward_expand` or the allocator prunes them.
- **The scheduler's CPU fallback is silent.** If Metal's `supports_op` table
  rejects a node (wrong type, non-contiguous, custom op), that node quietly
  runs on CPU, splitting the graph. This is why `ggml_custom_op` is banned
  (Metal has no case for it at all) and why a stray BF16 tensor can silently
  cost an order of magnitude (see the vision im2col story, §7).
- **Quantized matmul is transparent.** `ggml_mul_mat` accepts a quantized
  weight and dequantizes inside the kernel; there is no dequantize node.

ggml itself is pinned to a fixed revision and extended via **build-time
patches** ([`patches/`](../patches/), applied idempotently by `apply-all.sh`):
currently the two fused DeltaNet kernels, each landed as a CPU + Metal pair so
they can be differentially tested against each other. Patch-not-fork was a
deliberate call: reproducible builds, no merge treadmill, upstream optional.

---

## 4. Codemap

Every directory in `src/` is concept-named; each module's unit test lives at
`tests/unit/test_<module>.cpp`, always.

| Directory | Concept | Key files |
|---|---|---|
| `src/layers/` | Layer modules — graph-building ML primitives | `attention` (GQA, QK-norm, RoPE/p-RoPE, softcap, sliding-window mask), `ffn` (SwiGLU/GEGLU), `moe` (top-k routing, 3× `mul_mat_id`, shared expert), `deltanet` (gated delta rule), `norm` (RMSNorm + Gemma `(1+w)` variant), `ple` (per-layer embeddings), `transformer_block` (standard block assembly), `moe_residency` (routing-skew telemetry) |
| `src/models/` | Recipes + registry | one file per family (`qwen3`, `qwen35`, `qwen36`, `gemma1`–`gemma4`), `model_registry` (GGUF arch string → factory + tensor-inventory validator), `forward_pass_base` (shared graph scaffolding: embed, output head, sparse decode ids), `i_image_embeddable` (Seam B, §7) |
| `src/graph_inputs/` | Typed graph inputs — named tensors a recipe declares and a setter fills at run time | `tokens`, `positions`, `attn_mask` (causal/sliding/bidi-span), `sparse_head`, `output_ids`, `image_embeddings`, `gather_indices` |
| `src/state/` | What persists across tokens | `kv_cache_simple` (append semantics, O(1) truncate, per-slot batch axis, cross-layer KV sharing), `recurrent_state` + `deltanet_state` + `ssm_state_cache` (overwrite semantics, checkpoint/restore), `token_sequence_section` |
| `src/sampling/` | Decode-time algorithms | `sampling` (greedy/temperature+top-k/top-p/rep-penalty, sparse variants), `grammar_vocab` (GBNF engine, §8), `token-trie` (candidate narrowing), `speculative` + `prompt_lookup` (draft-free speculative decoding), `sampling_snapshot` |
| `src/loader/` | GGUF → live model | `gguf_loader` (mmap, metadata), `weight_binding` (fail-loud tensor inventory), `tokenizer`, `chat_template` (per-family prompt rendering), `channel_filter` (Gemma 4 thought/answer channel split), `multimodal_check` |
| `src/core/` | Engine orchestration + persistence primitives | `model` (owns weights/backend/scheduler; the load path), `decode_plan`/`decode_step` (batched decode orchestration), `multimodal_prefill`, `prefix_library` (disk warm-KV blobs, hash-keyed, version-gated), `slot_snapshot`, `image_embedding_cache` + `persistent_image_embedding_store`, `platform` (mmap wrapper) |
| `src/vision/` | Image → soft tokens (§7) | `i_vision_encoder` (Seam A), `siglip_encoder` (Gemma 3, 27-layer ViT), `gemma4uv_encoder` (Gemma 4, blockless), `vision_loader`, `vision_model`, `bitmap` |
| `src/session/` | Portable snapshot file format | `snapshot_io`, `session_manifest`, `compat_header`, `section_ids` — versioned, sectioned, fail-loud on mismatch |
| `src/server/` | HTTP serving (§6) | `inference_server.h` (slots, queues, batching, warm paths — the engine-agnostic core), `http_server.cpp` (endpoints, SSE, OpenAI mapping), `server_vision`, `image_data_uri` |
| `src/cli/` | Terminal front end | `main` (flag parsing, wiring), `chat`/`complete`, `session_mode`, `image_loader` (preprocessing lives here — the encoder is content-blind), `image_prompt`, `speculative-bridge` |
| `src/telemetry/` | Metrics | `metrics.h` |
| `src/qinf_error.h` | The fail-loud error contract: errors name the slot/parameter, expected, then actual | |

Test tiers under `tests/`: `unit/` (co-located per module, includes bitwise
recipe gates), `integration/`, `smoke/` (end-to-end shell gates against real
models — server caching, conversational mode, image coherence), `perf/`,
`grammar/`, `diff/` (differential fixtures, e.g. captured llama.cpp tensors).

> **Flagged smell — `src/core/`.** The directory postdates the CLAUDE.md
> "no dumping grounds" list and its name is concept-free; today it hosts the
> engine owner (`model`), decode orchestration, and four persistence/caching
> facilities that mostly serve the snapshot/caching system. It has not yet been
> deliberately admitted to the allowlist or split (e.g. `engine/` + merging the
> cache pieces into `session/`). Until that decision is made, treat additions
> to `core/` with suspicion. Same status applies to `session/`, `telemetry/`,
> `vision/` (all reasonable concepts, none yet in the CLAUDE.md list), and to
> `metal/`/`quant/` (listed in CLAUDE.md but not existing as directories —
> Metal work lives in `patches/`).

---

## 5. Dataflow 1 — a text generation, end to end

**Load.** `gguf_loader` mmaps the file and reads metadata; `model_registry`
maps the GGUF `general.architecture` string to a recipe factory and a tensor
*inventory validator* — the load fails loudly if a tensor the recipe needs is
missing or mis-shaped, before any graph is built. `core/model` then allocates
one backend buffer for all weights and copies them in (the copy is the SSD
read; unified memory removes the PCIe hop, not the disk).

**Prefill.** The prompt is tokenized and rendered through the family's
`chat_template`. The recipe builds the *prefill graph* — all prompt tokens in
one pass — writing K/V into the slot's region of the KV cache (and, for
hybrids, advancing the recurrent state). Output: logits for the last position.

**Decode loop.** Each step builds the (much smaller) *decode graph* for one
token per active slot, runs it, and hands logits to sampling. The sampler
applies repetition penalty, temperature/top-k/top-p (or argmax), constrained
by the grammar mask if one is active. The chosen token is fed back as the next
step's input. Exit on EOS, stop sequence, token budget, or (server) timeout.

**Speculative decoding** (CLI `--speculative`, Prompt Lookup Decoding): no
draft model — the draft is an n-gram match of recent output against the prompt.
Drafted tokens are verified in one batched pass; on mismatch the KV cache
truncates (O(1) pointer move) and the recurrent state restores a checkpoint
(a copy — overwrite semantics can't rewind, see §9).

---

## 6. Dataflow 2 — a server request

The server is two classes of thread with exactly two lock boundaries:

- **N HTTP threads** (httplib): parse the request, push it onto a
  `RequestQueue`, then block streaming tokens back off a per-request
  `TokenQueue` as SSE. A failed `sink.write` (client disconnect) sets the
  request's atomic `cancelled` flag.
- **One inference thread**: the only thread that touches the model. It assigns
  queued requests to **slots** (a fixed pool; `slot_id` is both the pool index
  and the KV cache's batch-axis index), prefills them, then loops on
  **batched decode** — one forward pass computes the next token for *all*
  active slots. Concurrency is batching, not threading.

The engine seam is dependency-inverted: `InferenceServer` holds no reference
to the model or ggml — it calls `std::function` callbacks (`prefill`,
`batched_decode`, `clear_slot`, `tokenize`, …) wired at startup. The shared
vocabulary across the seam is `slot_id` + token vectors, which is what makes
the queueing/slot logic unit-testable with fake engines.

Production edges, all converging on the same slot-release path: full queue →
503; cooperative cancellation checked once per step; per-request timeout;
fail-loud rejection of oversized prompts. Per-slot sampler state (temperature,
seed) and per-slot GBNF grammar are honored on the server path.

**Warm paths.** Prefill is the dominant repeated cost in this workload, so the
server has three opt-in, mutually-layered KV-reuse mechanisms — kept separate
on purpose (revisit before adding a fourth):

| Flag | What it reuses | Contract |
|---|---|---|
| `--prefix-cache <dir>` | A fixed system-prompt block, from disk (`prefix_library`), across restarts | transparent; hash- and version-gated |
| `--chat-prefix-cache` | The longest strict-prefix of a slot's retained KV, in RAM, across requests | transparent; append-only, hybrid-safe; ~0 hits on thinking models (scaffold stripped on re-render breaks the prefix) |
| `--conversational` | The whole conversation's KV via an explicit `conversation_id` handle | **a semantics change, not a cache**: retains the reasoning scaffold like `chat.cpp`, so warm ≠ cold by construction; create/continue-delta/recover protocol; `DELETE /v1/conversations[/{id}]` to clear |

The third exists because the second measurably can't help thinking models:
reusing an answer's KV is inseparable from retaining its scaffold's KV, so any
warm reuse there is honest only as an explicit opt-in handle. Details:
[`plan-warm-conversational-server.md`](plan-warm-conversational-server.md).

Endpoints: `/health`, `/v1/models`, `/v1/completions`,
`/v1/chat/completions` (text + OpenAI `image_url` when `--mmproj` is loaded),
`DELETE /v1/conversations/{id}` and `/v1/conversations`.

---

## 7. Dataflow 3 — an image request

Vision is a **separate subsystem joined at one seam**, not a sixth layer type.
The encoder owns its own graph and scheduler (sharing the device backend),
runs once per image *before* text prefill, and its entire deliverable is a
`std::vector<float>` of embeddings in the text model's embedding space —
"soft tokens" the decoder cannot distinguish from text.

Two interfaces carry the whole boundary, and both have two implementations
(which is the evidence they're real seams, not Gemma-3-shaped code):

- **Seam A** — `vision/i_vision_encoder.h`: what an encoder *is* to the text
  side (`encode(bitmap)`, `mm_tokens_for(bitmap)`, `projection_dim()`).
  `SiglipEncoder` (Gemma 3): 896×896 fixed input → 27-layer ViT
  (bidirectional attention, LayerNorm+biases, plain GELU — same ggml ops,
  different recipe than the decoder) → 4×4 pool → project → always 256 tokens.
  `Gemma4UvEncoder`: *blockless* — im2col patchify, LayerNorms, linear
  projection, no attention at all → 40–280 tokens, count decided by
  `smart_resize` in preprocessing and known before prefill.
- **Seam B** — `models/i_image_embeddable.h`: what a recipe must offer to host
  an image. The splice is `ggml_set_2d` overwriting the residual stream at the
  reserved placeholder span, *after* Gemma's √d embedding scale (image rows
  enter unscaled). Gemma 3's span attends bidirectionally (a mask parameter);
  Gemma 4's is plain causal — hosting it *removed* an interface parameter,
  which is the pressure test passing.

Preprocessing (resize/normalize) lives in `cli/image_loader`, parameterized
per family and byte-gated against captured llama.cpp references — the encoder
must see exactly what it saw in training (aspect-preserving letterbox,
align-corners bilinear, uint8 intermediate).

Server-side, `ServerVision` routes `image_url` content through the same seams,
with an embedding cache (`--image-embed-cache`) and an image-prefix KV cache
(`--image-prefix-cache`) so a recurring image skips the encode and/or the
image-span prefill. Image prefill uses a dedicated scheduler — reusing the
text scheduler's allocation for image-prefill graphs corrupts every request
after the first ([`server-image-multirequest-bug.md`](server-image-multirequest-bug.md)).

---

## 8. Dataflow 4 — grammar-constrained decoding

`sampling/grammar_vocab` is a from-scratch GBNF engine. The hard problem is
that **the model emits tokens but grammars are defined over characters** — a
token can partially fill, exactly fill, or overshoot a literal. The engine is
a nondeterministic pushdown automaton (many live states; an explicit
continuation stack for rule recursion) that tracks position at *character*
granularity inside the current literal.

Per decode step: `get_valid_tokens` (peek — which tokens are legal now) runs
*before* the forward pass, so its result both masks the sampler and drives the
**sparse output head** (the recipe computes logits only for legal rows of the
~150k-row head — the grammar makes the forward pass cheaper, not just the
sampling). After a token is chosen, `accept_token` advances the automaton. A
`state_version` counter makes the peek result cacheable across the two entry
points, fail-loud on staleness.

Speed comes from three layers: a byte-**TokenTrie** narrows literal candidates
(prefix walk = partial matches, subtree = overshoots), precomputed first-char
buckets handle char-classes, and **resolve-once** groups states by
`(literal, char_idx)` so the expensive grammar expansion runs once per group
instead of once per candidate token.

A fourth mechanism skips the forward pass entirely: **forced-token elision**
(`core/decode_step`, opt-in per call). When the peek collapses to exactly one
legal token, the next token needs no model — the run of determined tokens
(capped at 64) is chained through the automaton and model state is advanced
over the whole run in a single `feed_tokens` dispatch: no decode graph, no
head, no sampling. Two bounds: elision is *token*-level (a forced string with
multiple tokenizations still branches at the token level, so it falls back to
a normal step), and it is **CLI-only today** — the server decode loop needs a
batch-aware `decode_step` variant that doesn't exist yet (TODO at the top of
the server's decode path in `http_server.cpp`).

Two known bugs are documented in the code and bounded **safe-by-direction**
(only ever too permissive, never blocking a legal token); the correct fix for
one was measured at 10× decode cost and consciously rejected. A downstream
validator can reject the rare over-permissive token.

---

## 9. The state model

The single most load-bearing distinction in the engine:

| | KV cache (attention) | Recurrent state (DeltaNet/SSM) |
|---|---|---|
| Update | **append** a column per token | **overwrite** one fixed-size matrix |
| Size | grows with context | constant |
| Rollback | move the position pointer (O(1) truncate) | restore a checkpoint (copy) |

They are **never unified** behind a shared base class — a common interface
would force no-ops on one side and hide the rewind asymmetry that matters for
speculative decoding and every warm-KV feature. Hybrids (Qwen 3.5/3.6) carry
both kinds simultaneously, which is why every prefix-reuse feature in §6 is
**strict-append only**: an append is safe for both state kinds; a rewind is
safe only for pure-attention models.

Both state kinds, plus sampler state and token history, serialize into the
**portable session snapshot** (`src/session/` format, `core/slot_snapshot`
extraction): versioned sections, a compatibility header (backend/kernel-path
tag — cross-backend restore refuses fail-loud rather than silently degrading),
byte-fidelity gated on Metal and CPU for both families. The same machinery
backs the disk prefix library and the image-prefix cache.

---

## 10. Performance doctrine

All headline numbers from one setup — **Apple M1 Pro, Metal, Qwen 3.6 35B-A3B
Q2_K, baseline 19.8 tok/s (~50.5 ms/token)** — and recorded in
[`phase4-investigation.md`](phase4-investigation.md) and the plan docs. The
doctrine, each rule earned by a measurement:

- **Amdahl before code.** The proposed MoE fusion had a measured ceiling of
  1.13× (the fusable slice is ~5.7 ms of 50.5) against a claimed 3×; dropped
  without writing a kernel. The 3× came from a codebase that launches one
  matmul per expert — ours already dispatches any expert count in three
  `ggml_mul_mat_id` calls per layer.
- **Ask launch-bound or math-bound first.** DeltaNet's cost (~29 ms/token,
  the biggest slice) was dozens of tiny op launches, not its matmul (8.4% of
  the layer). So the fusions target the small ops; the matmul stays native.
- **Two signals or it didn't happen.** The two shipped fused kernels
  (`deltanet_post_state`, `deltanet_pre_state`, ~+7% tok/s combined) had to
  show up in both per-step timing and end-to-end tok/s — Metal per-step
  numbers swing ±25%.
- **Deleting is optimizing.** TurboQuant/SnapKV removed (wrong constraint for
  the envelope); norm fusion never attempted (1.7% ceiling); conv fusion
  deferred below the agreed µs bar.
- **The head is skippable.** The sparse output head (§8) and prefill-head
  slicing are explicit caller switches, never silent engine choices.

The scoped-but-unbuilt frontier is the ANE output head
([`plan-ane-lm-head.md`](plan-ane-lm-head.md)): a new backend beside the GPU,
used only for the output-head matmul, overlapping with the next token's body.

---

## 11. Correctness doctrine

- **Byte-identical extraction gates.** Any refactor of the forward pass must
  produce bit-for-bit identical logits before/after, on a Qwen *and* a Gemma
  model. Extraction and optimization are never combined in one step.
- **…except where the hardware forbids it.** Metal selects different kernels
  by matmul batch size (matrix×matrix vs matrix×vector), and float addition
  isn't associative — so any transform that changes batch shape (batching,
  head slicing, warm-vs-cold prefill) cannot promise bit-identity. The
  standard gate there is **token-stable + loose logit ceiling**, with the
  strict bitwise test kept but `DISABLED_` and documented. The spec bent, not
  the code. Surfacing this conflict is a standing decision rule, not a
  one-off.
- **Fail-loud at module boundaries.** Errors name the slot/parameter, the
  expected value, and the actual value, in that order (`qinf_error.h`). No
  silent fallbacks, no best-effort recovery: a missing tensor kills the load,
  a wrong-dim vision projector refuses to encode, a version-mismatched
  snapshot refuses to restore, an unknown `conversation_id` tells the client
  to resend history.
- **The cross-family rule keeps abstractions honest.** An interface validated
  only on Qwen is presumed Qwen-shaped until a Gemma recipe proves otherwise.
  The success metric for hosting a new variant: zero logic edits to modules
  other recipes depend on — if it needed them, that's an interface defect to
  fix, not a feature to celebrate.

---

## 12. Known soft spots (honest ledger)

Current, verified against the tree at time of writing:

- `src/core/` naming/admission is unsettled (§4 flag).
- `forward_pass_base` remains a shared base class; the blueprint's direction
  is composition-over-inheritance and it is a known eventual deletion target.
- The chat endpoint flattens engine finish reasons (`timeout`, `cancelled`,
  `error`) to OpenAI's `"stop"` — the completions endpoint reports honestly;
  the chat path lies by enum-compat (`chat_finish_reason` in
  `http_server.cpp`).
- Thinking-model token budgets: the thought channel spends `max_tokens`, so a
  visible answer can be cut with a clean `"length"` — no
  `max_completion_tokens` split yet.
- Grammar engine: two documented too-permissive bugs (§8).
- Vision: the strict numeric encoder differentials vs llama.cpp are
  `DISABLED_` (coarse gates + coherence smokes stand in); Gemma 4 image turns
  can emit a short degenerate prefix before recovering.
- The conversational-server gate lacks its Gemma 4 (pure-attention thinking)
  leg; recover responses reuse generic HTTP statuses rather than the
  documented 409.
- No CORS/auth on the server — it is local-oriented by design, but that makes
  browser front ends a P2.

---

## 13. Keeping this document alive

This document is governed by the **Architecture Doc Protocol** in
[`CLAUDE.md`](../CLAUDE.md): read it at the start of every session, and any
change that alters what it describes must be surfaced to the user for approval
*before* it lands — the doc update then travels in the same change.

Update triggers — if your PR does any of these, touch this file in the same PR:

- adds/removes a directory under `src/`, a model recipe, a server endpoint or
  flag, or a state kind;
- changes a seam named here (the server callback set, Seam A/B, the snapshot
  header contract);
- settles a §12 item (delete the bullet) or adds a new known soft spot.

Numbers in §10 are point-in-time measurements with named provenance; don't
update them casually — re-measure or leave them, never interpolate.
