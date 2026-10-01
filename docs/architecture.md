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

- a **CLI** (`qwenium` binary, CMake target `qwenium-cli`): single-user
  chat/completion, with vision,
  grammar-constrained output, speculative decoding, an opt-in persistent
  decode graph (`--persistent-graph`, §5), opt-in flash attention
  (`--flash-attn`, §5 — on the CLI this is both phases, which the lens
  excludes; the server scopes it to prefill when the lens is on), a
  selectable KV element type (`--kv-type f32|f16|q8_0|q4_0`, §9), opt-in
  mmap-backed weights (`--mmap-weights`, §5), and session snapshots;
- an **HTTP server** (`qwenium-server` binary, CMake target `qwenium-server`;
  source `src/server/http_server.cpp`, which keeps the generic file name):
  OpenAI-compatible `/v1/completions` and
  `/v1/chat/completions`, serving up to ~10 concurrent requests by batching
  them into one forward pass.

**"~10 concurrent" is a slot count, not a throughput promise, and what it buys
is family-dependent (measured 2026-08-30).** Batching B requests into one pass
returns between ~1.6× and ~9.6× the tokens/wall of running them serially,
depending entirely on the recipe:

| recipe | cost of one extra lane, as a fraction of a B=1 step | what batching buys |
|---|---|---|
| `gemma4` dense 12B-it Q8_0 | **0.25** (and only **~0.02** above B=8) | **9.60× at B=32, still flat** |
| `qwen35` 9B Q4_K_M | **0.52** | ceiling **1.91×** |
| `qwen35moe` 35B-A3B Q2_K_XL | **0.63** | ceiling **1.59×** *as currently built* |

*As currently built* is load-bearing: the hybrid figures are set by two pieces
of software, not by the hardware — `DeltaNetLayer::build_decode` builds a full
chain per slot (`src/layers/deltanet.cpp`), and ggml-metal's `MUL_MAT_ID` has
no small-batch kernel and does not reach `mul_mm_id` below `ne21 = 32`, so the
**entire ≤10-slot envelope sits below the MoE batched-kernel threshold**.
A further consequence worth knowing before tuning the server: **8 slots is in
the trough between two ggml-metal kernel regimes** — past the `mul_mv_ext`
small-batch path (B≤8, and B≤5 before its rows-per-threadgroup table halves)
and below the `mul_mm` crossover (`ne11_mm_min = 8`, engaged only *above* 8).
On Gemma 4 both **B=5 and B=16 are materially better than B=8**. Provenance:
[`note-batch-scaling-cross-family.md`](note-batch-scaling-cross-family.md).

> **CORRECTED 2026-08-31 — replication does not support acting on the B=5
> claim.** Replicated on the same Gemma 4-12B-it Q8_0 leg, speedup vs serial:
>
> | B | plain | server-realistic (per-lane logits readback + argmax) |
> |---|---:|---:|
> | 5 | 2.89× | 2.64× |
> | 8 | 2.96× | 2.48× |
> | 16 | 5.16× | 4.54× |
>
> In **plain** mode B=5 is *not* materially better than B=8 — a wash, arguably
> a slight edge to B=8. Under **server-realistic** per-lane work B=8 does drop
> below B=5, so B=8 is a genuine local bad spot there — but B=5 is still not an
> actionable improvement over it, it's just less bad. B=16 robustly beats B=8
> in both modes, and always did, but it sits outside the declared ≤10-slot
> envelope and past the qwen36 DeltaNet node-count abort (§12: confirmed abort
> at B=11). **Net: do not act on slot-count guidance from this paragraph.**
> Full replication, including the structural confirmations behind the
> above-B=8 marginal-cost collapse and the caveat that none of this is
> established for any Qwen or K-quant model:
> [`note-batch-scaling-cross-family.md`](note-batch-scaling-cross-family.md).

The **workload envelope** is load-bearing and explains many "missing" features:
≤10 concurrent slots, ≤10K context, ~12 GB quantized models on unified memory.
KV-cache *memory* is not the constraint in this regime — which is why KV
compression (TurboQuant) and KV eviction (SnapKV) were built, measured, and
then deleted. Decode-time *latency* is the constraint, so the optimization
surface is kernel launches, grammar-step cost, and prefill reuse (caching).

The context ceiling was **raised from 4K to 10K on 2026-08-24**. The old figure
described order-management text prompts and predated any image whose token
count scales with resolution: Gemma 3 vision is always 256 soft tokens and
Gemma 4 is 40–280, but a Qwen-VL-class encoder emits 2K–8K for a document page,
which did not fit. 10K is the smallest ceiling that holds one high-resolution
document plus its prompt and answer. Two consequences worth stating plainly:
KV bytes scale as **ctx × slots**, so this is a 2.4× multiplier on the very
axis the TurboQuant/SnapKV deletion declared non-binding — that rationale now
rests on a measurement taken at 4K and should be re-checked rather than assumed
if a 10-slot host runs tight (`--kv-f16` is the first lever). And the measured
figures elsewhere in this document that name "ctx 4096" (§KV element type) are
historical measurements, not envelope statements; they stay as recorded until
re-measured.

**The receipts identity (2026-07-19).** Beyond serving answers, the engine
treats its own computation as a product surface — *"where it looked, what
decided it, and proof it happened"*: **attention** (materialized decode rows,
tapped and calibrated into citations/coverage — the lens, §6), **determinism**
(byte-reproducible greedy decode with forkable state — counterfactual re-runs),
and **integrity** (weights-hash, version-gated snapshots, fail-loud replay).
This is a doctrine-level commitment with named engineering constraints — see
§11's *receipts constraints* bullet before optimizing anything on these paths.
Receipts are **per-model calibrated capabilities** (like vision or MTP), not a
blanket property; the claim boundary is fixed by measurement: receipts show
what the model *consulted*, never why it *chose* (consideration, not
commitment — the non-claims contract in [`lens-format.md`](lens-format.md)).

Supported architectures (registered in
[`src/models/model_registry.cpp`](../src/models/model_registry.cpp)):
`qwen2`, `qwen3` (pure transformer), `qwen35` (DeltaNet + attention hybrid —
hosts both the Qwen 3.5 and Qwen 3.8 releases; layer counts and the DeltaNet
V:K head ratio come from metadata, and a trailing NextN/MTP head block is held
out of the decode stack when present),
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
                       │  src/engine/model.{h,cpp} — owns weights,    │
                       │    backend, scheduler, the forward pass      │
                       │  src/engine/decode_plan / decode_step        │
                       │  src/engine/multimodal_prefill               │
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
                   src/session/ (portable snapshots, warm-KV caches:
                                format + slot_snapshot/prefix_library/
                                image-embedding caches)
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
| `src/layers/` | Layer modules — graph-building ML primitives | `attention` (GQA, QK-norm, RoPE/p-RoPE, softcap, sliding-window mask), `ffn` (SwiGLU/GEGLU), `moe` (top-k routing, 3× `mul_mat_id`, shared expert), `deltanet` (gated delta rule), `norm` (RMSNorm + Gemma `(1+w)` variant), `ple` (per-layer embeddings), `transformer_block` (standard block assembly) |
| `src/models/` | Recipes + registry | one file per family (`qwen3`, `qwen35`, `qwen36`, `gemma1`–`gemma4`), `model_registry` (GGUF arch string → factory + tensor-inventory validator), `forward_pass_base` (the recipe interface plus the graph scaffolding recipes share: embed, output head, the Seam B image splice, decode masks — each of which builds nodes *and* declares the typed input they consume; per-step arming lives here too. Context and run-time policy were extracted out to `graph_arena` / `decode_policy`), `i_image_embeddable` (Seam B, §7 — implemented by `gemma3`, `gemma4`, `qwen36`, `qwen35`), `graph_arena` (the per-pass ggml context + metadata buffer, held not inherited), `decode_policy` (the pass's run-time policy as one value; its defaults are the byte-reproducible path), `qwen35_family` (what the two Qwen 3.5-family hybrids share: typed-input declarations and the layer body, with the FFN as a parameter), `i_mtp_draftable` (MTP/NextN draft capability — qwen36 only; see §5; qwen35 binds NextN weights when the GGUF carries a head, e.g. Qwen 3.8, but does not yet draft from it) |
| `src/graph_inputs/` | Typed graph inputs — named tensors a recipe declares and a setter fills at run time | `tokens`, `positions`, `mrope_positions` (4 components/token, component-major — Qwen 3.5 family), `attn_mask` (causal/sliding/bidi-span), `sparse_head`, `output_ids`, `image_embeddings`, `gather_indices` |
| `src/state/` | What persists across tokens | `kv_cache_simple` (append semantics, O(1) truncate, per-slot batch axis, cross-layer KV sharing), `recurrent_state` + `deltanet_state` (overwrite semantics, checkpoint/restore), `token_sequence_section` |
| `src/sampling/` | Decode-time algorithms | `sampling` (greedy/temperature+top-k/top-p/rep-penalty, sparse variants), `grammar_vocab` (GBNF engine, §8), `token-trie` (candidate narrowing), `speculative` + `draft_source` (draft-source seam: `IDraftSource`) + `prompt_lookup` (PLD) + `suffix_decoding` (SuffixDecoding: session-scoped, adaptive-length lookup, §5), `sampling_snapshot` |
| `src/loader/` | GGUF → live model | `gguf_loader` (mmap + metadata, and the three fail-loud gates: tensor **type** — is every ggml type id one this build knows, checked at parse because `ggml_type_size`'s own bounds check is a plain `assert` that Release removes; **architecture** — is `general.architecture` in the registry allow-list; **inventory** — does the file carry every tensor the recipe needs, correctly shaped), `tokenizer`, `chat_template` (per-family prompt rendering), `channel_filter` (Gemma 4 thought/answer channel split), `multimodal_check`, `gguf_value` (generic GGUF scalar/array KV bag), `platform` (mmap wrapper) |
| `src/engine/` | The loaded model, and the orchestration of one step over it | `model` (owns weights/backend/scheduler; the load path), `decode_plan`/`decode_step` (batched decode orchestration), `decode_graph_cache` (opt-in persistent decode graph — reuse one built+allocated graph across steps on a dedicated scheduler, §5), `multimodal_prefill`, `graph_compute` (the one place a compute status is checked — fail-loud on backend failure) |
| `src/vision/` | Image → soft tokens (§7) | `i_vision_encoder` (Seam A), `siglip_encoder` (Gemma 3, 27-layer ViT), `gemma4uv_encoder` (Gemma 4, blockless), `qwen3vl_encoder` (Qwen 3.5 family, ViT + 2×2 merger, in-ViT M-RoPE), `vision_profile` (projector → encoder+recipe dispatch), `image_preprocess` (preprocessing recipes), `vision_loader` (3 projectors: `gemma3`, `gemma4uv`, `qwen3vl_merger`), `vision_model`, `bitmap` |
| `src/session/` | Persisting and reusing session state | The **format**: `snapshot_io`, `session_manifest`, `compat_header`, `section_ids` — versioned, sectioned, fail-loud on mismatch (built as `qinf-session`, deliberately dependency-free so it unit-tests in isolation). The **services** on top of it: `slot_snapshot` (extract/restore a slot), `prefix_library` (disk warm-KV blobs, hash-keyed, version-gated), `image_embedding_cache` + `persistent_image_embedding_store`. The services that need `models/`/`graph_inputs/` build into `qinf-engine` or `qinf-snapshot` rather than into `qinf-session` — directory is the concept, target is the layering (see `session/CMakeLists.txt`). As of 2026-08-30 every one of them has exactly one home: `image_embedding_cache` is pure std so it joined `qinf-session`; `slot_snapshot` needs the model, so it is `qinf-snapshot`. |
| `src/server/` | HTTP serving (§6) | `inference_server.h` (slots, queues, batching, warm paths — the engine-agnostic core), `http_server.cpp` (endpoints, SSE, OpenAI mapping), `server_vision`, `image_verdict` (`/v1/verdict` with an **image** — yes / no / unclear about visible marks from the answer logits after one image pass, questions resumed from a snapshot; its own calibration table and gates: `--mmproj`, a full model, a measured row; NOT the lens, reads no attention), `server_lens` (opt-in `--attention-lens`: `/v1/extract` — document → audited key-value JSON on the attention trust layer; `/v1/verify` — teacher-forces a known extraction to reproduce the same report without generating; `/v1/locate` — document + keys → byte ranges, nothing generated and nothing audited, on its own calibrated head; `/v1/verdict` — document + yes/no questions → yes / no / unclear read off the prefill's last row, full lens server or verify-only with `--lens-verdict`; `/v1/compare` — an original's units + a second version → which units are missing, every lens server; pure lens computation + single-slot tapped-decode/tapped-prefill drivers), `image_data_uri` |
| `src/image/` | Host-side image pipeline (IO, not encoding) | `image_loader` (decode/resample/normalize → `Bitmap`; the encoder is content-blind, and the preprocessing *recipe* it applies lives in `vision/image_preprocess`), `image_prompt` (token-level marker expansion → the soft-token span). Both front ends consume these, which is why they are not in `cli/`. |
| `src/cli/` | Terminal front end | `main` (flag parsing, wiring), `chat`/`complete`, `session_mode`, `speculative-bridge` |
| `src/qinf_error.h` | The fail-loud error contract: errors name the slot/parameter, expected, then actual | `QINF_ASSERT`. The format is the rule, not the macro — most errors are written by hand, e.g. `assign_tensor_pointers`' `require()` |

Test tiers under `tests/`: `unit/` (co-located per module, includes bitwise
recipe gates), `integration/`, `smoke/` (end-to-end shell gates against real
models — server caching, conversational mode, image coherence), `perf/`,
`grammar/`, `diff/` (differential fixtures, e.g. captured llama.cpp tensors).

Read the co-location invariant (`src/<m>.cpp` ⇒ `tests/unit/test_<m>.cpp`) as
the rule for *modules*, not recipes. Recipes and the front ends are covered by
**aspect** tests instead — `test_qwen35_forward_attn`, `test_gemma3_config`,
`test_qwen36_hparams`, `test_gemma_batched_decode` — which is better testing
than one file per recipe would be, but it means "no `test_qwen35.cpp`" does not
mean "qwen35 is untested". Model-file tests self-skip when their model is
absent (`QWEN3_MODEL_PATH` and friends), so a green run with skips is normal;
check the reported total, not only the failure list.

> **Directory admission — settled and still open.** `src/core/` was a flagged
> smell: a concept-free name hosting the engine owner, decode orchestration and
> four persistence facilities, with `loader/` depending on it while it *was* the
> load path. **Settled 2026-08-29:** `gguf_value` + `platform` → `loader/`, the
> four persistence services → `session/`, the remainder renamed `engine/` (build
> target `core` → `qinf-engine`), and the host-side image pipeline moved out of
> `cli/` into its own `image/` (both front ends consume it — the server compiles
> it directly). `engine/` and `image/` are in the CLAUDE.md allowlist.
>
> Also settled the same day: `session/` and `vision/` joined the allowlist, and
> three directories left the tree entirely. `metal/` and `quant/` held nothing but
> a comment-only CMakeLists reserving space for a Phase 4 whose measured ceiling
> (≤1.13×, [`phase4-investigation.md`](phase4-investigation.md)) killed it — a
> directory is admitted when it holds code, not in advance. `telemetry/` was one
> 17-line header with a single-field struct, feeding `layers/moe_residency`,
> which had **no production caller at all**; both went. The allowlist is now what
> the tree is.

---

## 5. Dataflow 1 — a text generation, end to end

**Load.** `gguf_loader` mmaps the file and reads metadata; `model_registry`
maps the GGUF `general.architecture` string to a recipe factory and a tensor
*inventory validator* — the load fails loudly if a tensor the recipe needs is
missing or mis-shaped, before any graph is built. `engine/model` then allocates
one backend buffer for all weights and copies them in (the copy is the SSD
read; unified memory removes the PCIe hop, not the disk), and **then releases
the mapping** (`release_file_mapping`). That release is load-bearing, not
tidiness: the copy faults in every page, so holding the mapping open keeps a
second full copy of the weights resident for the process lifetime — measured
5.31 → 1.75 GB steady-state RSS on Qwen3.5-0.8B (9B: ~10.4 → 5.83 GB), and
~13 GB of avoidable residency on a 27B. After it, the loader's tensor-data
accessors throw rather than dereference a released mapping. `vision_loader`
has always done the equivalent (`unload_model`) for the projector.

This fixes *steady-state* residency, not the **load-time peak**: the copy
still needs the source mapping and the destination buffer live at the same
instant, so peak stays ≈ 2× model size for any model large enough that the
copy dominates (measured: 9B peak unchanged at ~10.6 GB). Capacity-plan
loading against 2× and serving against 1× on the default path.

**`--mmap-weights` removes the copy, and with it the peak (opt-in, 2026-09-10).**
`Model::set_mmap_weights(true)` before `load_tensors()` backs the weights buffer
with the GGUF's own mapped pages via `ggml_backend_dev_buffer_from_host_ptr`
(Metal declares `buffer_from_host_ptr` and implements it with
`newBufferWithBytesNoCopy`, page-aligning internally) and assigns each tensor's
`data` into the mapping. There is no copy, so the mapping is deliberately NOT
released; a per-tensor bounds check fails loud rather than letting a tensor point
outside the mapping. Measured on Qwen 3.6 35B-A3B Q2_K:

| | copy (default) | `--mmap-weights` |
|---|---|---|
| peak RSS during load | 15894 MB | **104 MB** |
| phys_footprint after a forward | 12349 MB | **361 MB** |
| load wall time | 26.7 s | **4.1 s** |
| warm prefill | 338.6 ms | 339.9 ms |

**Byte-identical, gated cross-family**: the full logits vector hashes equal on
`qwen35` (0.8B), `qwen35moe` (35B) and `gemma2` (10 GB), and the CLI's greedy
output is identical with and without the flag — against a baseline-vs-baseline
control, since the engine's default temperature is 0.7 and a greedy comparison
needs `-t 0`. Load is **6.5× faster** because a 12 GB memcpy is gone; steady-state
decode is unchanged.

**What the footprint number does and does not mean.** The weights still occupy
physical memory while hot — they must, for the GPU to read them. What changed is
the *category*: dirty anonymous pages the kernel cannot reclaim become clean
file-backed pages it can. `phys_footprint` stops attributing them to the process,
which is exactly the accounting that makes third-party "peak active memory, page
cache excluded" claims flattering — do not read 361 MB as "the model needs
361 MB". What is really bought: the 2× load peak is gone, load is 6.5× faster,
and the OS can reclaim weight pages under pressure instead of swapping or OOMing.
**Degradation under actual memory pressure was NOT measured** — that is the
mechanism's prediction, not a result. The default stays copy-and-release.

**Prefill.** The prompt is tokenized and rendered through the family's
`chat_template`. The recipe builds the *prefill graph* — all prompt tokens in
one pass — writing K/V into the slot's region of the KV cache (and, for
hybrids, advancing the recurrent state). Output: logits for the last position.

**Decode loop.** Each step builds the (much smaller) *decode graph* for one
token per active slot, runs it, and hands logits to sampling. The sampler
applies repetition penalty, temperature/top-k/top-p (or argmax), constrained
by the grammar mask if one is active. **Greedy means argmax**: `GreedySampler`
applies no repetition penalty unless a caller passes one explicitly, so
temperature 0 is the model's actual argmax — the precondition for §1's
byte-reproducible-greedy-decode claim. It defaulted to a 1.2 penalty until
2026-08, which silently steered every temperature-0 generation and was the
sole cause of an apparent forward-pass divergence from llama.cpp and HF that
did not exist
([`engine-divergence-probe-results.md`](engine-divergence-probe-results.md)). The chosen token is fed back as the next
step's input. Exit on EOS, stop sequence, token budget, or (server) timeout.

Rebuilding + reallocating that graph every step costs ~12 ms of galloc replan
(26% of a step on M1 Pro). The opt-in **persistent decode graph** (CLI
`--persistent-graph`, `engine/decode_graph_cache`) removes it: the graph is
built + allocated once per KV-width *bucket* on a dedicated scheduler and
reused across steps — only the typed inputs (tokens, positions, mask, gather
and set_rows write indices) are refilled and recomputed. Measured **1.32×**
decode on Qwen 3.6 (§10). Two P1/P2 changes make this possible and are inert
by default: the decode KV write became value-driven (`ggml_set_rows`, write
row an input) instead of a build-time-baked `ggml_cpy` offset, and n_kv is
padded to a bucket so one allocation stays valid across a run of steps.
Bucketing re-blocks the attention reduction, so this path is **token-stable
modulo ties, not byte-identical** to the default exact-n_kv decode — the same
status as speculative decoding (§11) — which is why it is opt-in; the default
decode path is unchanged. Sparse-head (grammar) steps and non-persistent-
capable recipes fall back to the per-step rebuild. Qwen 3.5/3.6 + Gemma 3 are
persistent-capable today; the write-mode/bucket are a differential seam gated
byte-for-byte at exact width (`test_kv_write_setrows`, `test_decode_kv_bucket`,
`test_decode_graph_cache`).

**Flash attention** (CLI/server `--flash-attn`, `DecodePolicy::AttnImpl` —
one field per phase, `attn_impl` for decode and `prefill_attn_impl` for
prefill): on **both prefill and decode** unless a caller scopes it (the lens
does, below), one `ggml_flash_attn_ext` replaces the whole
`kq` → `soft_max` → `kqv` chain *and* the V transpose — four Metal dispatches
per attention layer become one. On decode the recipe casts its mask to F16 once
per graph (Gemma 2/3/4 dedupe by window first); on prefill the mask is built per
layer inside the attention helper, so the cast lives there. `build_attn_mha`
refuses an F32 mask, naming the layer, and forwards softcap — ggml applies the
scale before the tanh clamp, our convention.

> **CORRECTED 2026-09-02 — "both prefill and decode" was not true of the CLI.**
> `cli/complete.cpp` called `run_prefill` ~40 lines BEFORE
> `set_attn_impl(AttnImpl::Flash)`, so on the `-p` completion path flash applied
> to decode only and prefill silently ran materialized attention, with no
> diagnostic. `cli/chat.cpp` had the same ordering for its system-prompt prefill
> alone (per-turn prefills come after, hence were always fine); **the server was
> always correct** (`enable_flash_attn()` runs at startup). Fixed by moving the
> arming to immediately after `create_forward_pass`. Measured on Qwen3-0.6B-Q8_0
> with a 2601-token prompt, interleaved A/B, cold run discarded: prefill
> **3546–3747 ms flash-off vs 3683/3636 ms flash-on before the fix — no win at
> all** — and **1940/2033 ms after**, ~1.83×. It surfaced only because a
> quantized KV cache, which only the flash kernel can read, aborted inside
> `build_prefill_graph`'s `ggml_mul_mat`. **Consequence for the numbers below:**
> any prefill figure taken through the CLI `-p` path before this date measured a
> no-op. Whether the 7%→55% range came from that path or from a bench binary is
> unverified — treat it as an inherited number until re-measured.

**Prefill is where it pays most**:
materialized attention is O(n²) in the prompt length, so the win grows from ~7%
at 756 tokens to ~55% at 3000 on attention-heavy recipes, and it is what keeps
prefill competitive at the 10K envelope. Token-stable, **not byte-identical**
(the softmax reduces in registers, in a different order), so
it is opt-in like `--persistent-graph`, and **every recipe supports it**
(`supports_flash_attn()`). Gemma 2's attention softcap forwards to the kernel:
ggml pre-divides `scale /= logit_softcap` and computes
`logit_softcap*tanh(s*scale)` — the scale applied before the clamp, which is
our convention and HF's. **What flash is worth varies by an order of magnitude
across models — 4% to 33% of a decode step — and is not predictable from family
or layer count**; see §16–§17 of the gap ledger for the seven-model measurement
and why the obvious generalizations are wrong. Qwen 3.6 came almost free — it shares
`qwen35_family`'s layer body, so the recipe-side change was the F16 mask cast
and one flag on `Qwen35LayerCommon`. On the MoE hybrid only the 9 attention
layers change; the 36 MoE routers keep their own `SOFT_MAX` and the experts
their `MUL_MAT_ID`, untouched.

**Flash attention and the receipts identity are mutually exclusive IN A
PHASE**, and this is enforced, not documented-and-hoped: the flash kernel never
materializes `kq_soft`, which is precisely the tensor the attention lens taps
(§1, §11). `DecodePolicy::is_attn_impl_coherent()` states the pairing for a
tapped **decode** (`/v1/extract`); a tapped **prefill** (`/v1/verify`, §6) is
protected structurally instead — `run_lens_verify` sets its own tapped pass
materialized, and `mark_attention_taps` fails loud on the missing `kq_soft`
node if that is ever lost.

**Whether the flag may join `--attention-lens` is a PER-MODEL question**, and
since 2026-09-13 the calibration table answers it
(`LensConstants::flash_prefill_ok`, `server_lens.h`). `DecodePolicy` scopes the
implementation per phase, so the lens can run a flash **prefill** (which
`extract` never taps) over a materialized **decode** (which it does) — on a
model that has earned it through the drift gate
(`tests/perf/attn_provenance.cpp`, `BANDDRIFT=1 DRIFT_ARM=flash DRIFT_LANG=all`;
`tests/smoke/lens_drift_gate.sh`). The calibrated models answer differently, and
the split is **dense vs MoE** — not size, not family:

| model | decision margin | max \|Δpeak\| under flash prefill | verdict |
|---|---|---|---|
| Qwen3.8-9B-Q8_0 (dense) | 0.00126 (DE-bound) | 0.00084 | **PASS** — enabled, 8.4% off the prefill |
| Qwen3.8-27B-Q3_K_M (dense) | 0.00099 (EN-bound) | 0.00054 | **PASS** — enabled, 1.8× headroom (thin) |
| **Qwen3.6-35B-A3B (MoE)** | 0.0237 | **0.0242** | **FAIL** — refused |

Both dense models are **15/15 token-identical** under a prefill-shape change;
both MoE models change one extraction in fifteen. Note also that **which
language binds is per-model** — German on the 9B, English on the 27B — so both
halves must be run on every entry rather than assuming German is always the
tighter one.

On the 35B the sibling arm fails harder: a chunked prefill changes **one
extraction in fifteen outright** (82 generated tokens against 109). That is MoE
routing, demonstrated rather than inferred (`MOEROUTE=1`): **2.32% of routing
decisions on the 35B and 6.37% on Gemma 4-26B-A4B select a different expert
set**, first flip in both landing in the last slot of the top-8 — the expert
nearest a tie. Two dense models under the same perturbation change nothing.
**Expert selection is a top-k argmax**, so on an MoE the perturbation is not
small, it is *discrete*, and no decision-margin argument applies to it at all.
A `flash_prefill_ok` of `true` on an MoE entry would therefore be unsound even
with a small measured drift, and the field **defaults to false** so an
unmeasured model is refused rather than inheriting another model's permission.

Three consequences worth naming. (a) The pairing check moved **after** the model
load, which the rest of that block deliberately avoids — the answer depends on
which model is loaded, and that is the price of it being per-model.
(b) `--attention-lens` with a quantized `--kv-type` is now refused **directly**
(`http_server.cpp`): it used to fall out of the blanket lens/flash refusal, and
phase-scoping opened the hole, since the KV cache is read by the materialized
decode where a quantized V would take ggml's silent CPU fallback (§9).
(c) Every lens report now carries a `config` stamp — weights hash, attention
implementation, KV type — because a licensed configuration is only safe if a
report says which one it ran under (`lens-format.md`).

This was the first place where a speed lever and the receipts doctrine were in
direct conflict; the resolution is a parameter on one operation with two
implementations (llama's own `-fa on/off` split), scoped per phase, not two
attention modules.

**Speculative decoding** (CLI `--speculative [pld|mtp|suffix]`): drafts come
from an `IDraftSource` (`sampling/draft_source.h`) and are verified in one
batched pass (head slice off — verification needs logits at every draft
position); a first-token guard ensures draft[0] matches the token the sampler
actually chose. On mismatch the KV cache truncates (O(1) pointer move) and,
on hybrids, the recurrent state restores a pre-verify checkpoint and the
accepted prefix is re-fed (`feed_tokens`) — overwrite semantics can't rewind
(§9). Three draft sources, all sharing that one verify/rewind path:

- **PLD** (bare `--speculative` or `--speculative pld`; no model — the draft
  is a fixed-length n-gram match of recent output against the *prompt only*).
- **SuffixDecoding** (`--speculative suffix`, `sampling/suffix_decoding.h`;
  no model — generalizes PLD along the two axes an offline 110-session replay
  of the order-management DSL corpus showed PLD losing draft *availability*
  on: the haystack is the whole session (prompt + everything generated so
  far, capped to the most recent 8192 tokens — session output grows
  without bound, the cap keeps the per-step scan O(1) in session length,
  not persisted across sessions, see suffix_decoding.h for the rationale),
  and the match length is adaptive — tried longest first from 12 down to a
  floor of 2, a longer match standing in for higher confidence. Measured:
  62% hit rate vs PLD's 26%, 2.64 tokens/step vs 1.61. Draft width defaults
  to 4 (`--suffix-max-draft`) — measured worse at 8 on Gemma 4-12B, the
  opposite of the usual wider-batch-is-cheaper intuition.
- **MTP head** (`--speculative mtp`, depth `--mtp-max-draft`, default 1; Qwen 3.6 NextN:
  an extra trained attention+MoE block held out of the main stack, drafting
  recursively from the last position's hidden state via
  `models/i_mtp_draftable.h` on a private KV + dedicated scheduler; the
  hidden is exposed by an opt-in, default-off graph output). MTP is a
  capability of MTP-converted GGUFs, mirroring how vision is a capability of
  `--mmproj` — Qwen-only, as vision is Gemma-only. Status: **experimental, and
  measured a net loss** — 74–92% acceptance, ~2.7 tokens/step, **0.92×
  baseline** (28 vs 31 tok/s, Qwen 3.6 A3B Q2_K, M-series, 2026-09-11). The
  cause is *not* per-step dispatch overhead: that hypothesis was measured and
  refuted — graph build and CPU bookkeeping are 0–1% of the round, and the time
  is in real forward passes. Two structural reasons, both outside the head's
  control: (1) the MoE — verify costs ≈29 ms fixed + ≈9 ms/token, the fixed
  part being one full decode, because top-8 routing re-reads expert weights per
  token (`mul_mv_id` cannot share those loads; `mul_mm_id` needs `ne21 ≥ 32`);
  (2) the hybrid — DeltaNet's overwrite semantics cannot rewind, so a partial
  reject re-feeds the accepted prefix, 4.0 ms/token at depth 1. Ceiling with a
  free, perfect rollback: 1.07×. Deeper drafting is self-defeating (acceptance
  88/69/45% at depth 1/2/4 against rollback on 24/54/89% of rounds), hence the
  default of 1. Round breakdown in `note-mtp-step-breakdown.md`; measurements
  and the five speculative-machinery bugs fixed en route live in
  `plan-mtp-decode.md` §7/§9.

Emitted tokens are model-verified under the kernel path that computed them;
batch-shape numerical forks (§11) mean speculative-on is token-stable, not
byte-identical, vs speculative-off — true for all three draft sources, since
they share verification.

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
`batched_decode`, `clear_slot`, `tokenize`, `speculative_decode`,
`speculative_eligible`, …) wired at startup. The shared vocabulary across the
seam is `slot_id` + token vectors, which is what makes the queueing/slot logic
unit-testable with fake engines. The two speculative callbacks
(`InferenceServer::set_speculative_decode` / `set_speculative_eligible`) are
new to this seam — see below.

Production edges, all converging on the same slot-release path: full queue →
503; cooperative cancellation checked once per step; per-request timeout;
fail-loud rejection of oversized prompts. Per-slot sampler state (temperature,
seed) and per-slot GBNF grammar are honored on the server path.

**Warm paths.** Prefill is the dominant repeated cost in this workload, so the
server has three opt-in, mutually-layered KV-reuse mechanisms — kept separate
on purpose (revisit before adding a fourth):

| Flag | What it reuses | Contract |
|---|---|---|
| `--prefix-cache <dir>` | A fixed system-prompt block, from disk (`prefix_library`), across restarts | transparent **on dense models only** (see below); hash- and version-gated |
| `--chat-prefix-cache` | The longest strict-prefix of a slot's retained KV, in RAM, across requests | transparent **on dense models only** (see below); append-only, hybrid-safe; ~0 hits on thinking models (scaffold stripped on re-render breaks the prefix) |
| `--conversational` | The whole conversation's KV via an explicit `conversation_id` handle | **a semantics change, not a cache**: retains the reasoning scaffold like `chat.cpp`, so warm ≠ cold by construction; create/continue-delta/recover protocol; `DELETE /v1/conversations[/{id}]` to clear |

**"Transparent" is a DENSE result, and the qualifier is measured, not
cautionary.** A warm hit is numerically one prefill boundary: the leading tokens
were written to KV by an earlier request at that request's batch width, the
suffix is prefilled now. On a dense stack that perturbation is ~1e-4 and moves
nothing — 0 of 15 documents, token-identical. On an MoE stack it lands on
`top_k`, which is an `argmax` with no margin, so it changes *which expert runs*.
Measured on Qwen3.6-35B-A3B (2026-09-16): 6.08% of routing decisions flip at a
system-prompt-sized boundary, and turning `--prefix-cache` on changed the answer
for **2 of 3** documents on the shipped server, against a clean control (two
flag-off processes agree byte-for-byte). `tests/smoke/server_text_cache_smoke.sh`
passes on that model because its gate compares a cache hit against a cache miss
*with the flag on* — snapshot fidelity, which holds — and never compares flag-on
against flag-off. Full measurement and the three options:
[`note-moe-cache-transparency.md`](note-moe-cache-transparency.md). Until one is
chosen, treat both caches as dense-only.

The third exists because the second measurably can't help thinking models:
reusing an answer's KV is inseparable from retaining its scaffold's KV, so any
warm reuse there is honest only as an explicit opt-in handle. Details:
[`plan-warm-conversational-server.md`](plan-warm-conversational-server.md).

**Token-id request log (`--token-log <path>`, off by default).** One JSON object
appended per completed request: prompt ids, generated ids, counts, finish reason,
whether a grammar was active. Written from `log_slot_end`, which every completion
path already converges on, so no exit route silently skips it; fail-loud if the
path cannot be appended to, because a logging flag that quietly does nothing is
worse than no flag.

It records **ids, not text, and that is the point**: re-tokenizing logged text
yields a different sequence than the model saw (chat-template rendering, special
tokens, thinking scaffold), while every analysis this log exists to serve —
draft hit rate and accepted length for a speculative source, strict-prefix reuse
for `--chat-prefix-cache`, how much of `max_tokens` the thought channel spends —
turns on exact token boundaries. It converts those from experiments that need a
model and a machine into scripts that replay a file. Two caveats: `n_generated`
counts the stop token where the OpenAI `usage.completion_tokens` does not, and
the ids are losslessly reversible to the prompt text, which is why it is opt-in.

Endpoints: `/health`, `/v1/models`, `/v1/completions`,
`/v1/chat/completions` (text + OpenAI `image_url` when `--mmproj` is loaded),
`DELETE /v1/conversations/{id}` and `/v1/conversations`.

**Speculative decoding (opt-in `--speculative [pld|mtp|suffix]`, off by
default).** The draft sources, the verify/rewind mechanics, and the
per-source measurements are §5's — this covers only what's server-specific:
how it engages here, why it's restricted the way it is, and how it sits
inside the request lifecycle above.

`decode_step()` takes the speculative branch only when **exactly one slot is
active**; two or more active slots fall back to the unchanged batched path.
This is a deliberate ceiling, not an oversight: the server's batch axis is
*slots* — one forward pass advances every active slot — while speculation's
batch axis is *draft positions* — one verify pass checks every drafted
token. Running both at once would multiply the two axes, and that
multiplication is unaffordable on two independent, measured grounds:

- **Verify cost grows faster than draft width helps.** Batching more draft
  positions into one verify pass is, mechanically, the same batch-axis
  operation §10 already measured across slot counts: on Gemma 4-12B-it Q8_0,
  a batched pass costs 1.41× a single step at width 4, 2.64× at width 8, and
  2.98× at width 16 (`note-batch-scaling-cross-family.md` §3 — 126.3/235.7/
  266.2 ms against a 89.4 ms B=1 step). Cost keeps climbing past the point
  where a wider draft stops finding more correct tokens to accept, which is
  also why §5's suffix decoder defaults its own draft width to 4 and measured
  8 as worse, not better.
- **The Qwen hybrids' decode graph is O(batch) in node count and now aborts
  above the envelope, fail-loud.** §12 records `n_nodes = 1320·B + 2144` on
  `qwen35moe`, confirmed to build at B=10 (94% of the graph-size limit) and
  abort at B=11 — one slot of margin above the declared ≤10-slot envelope —
  and the fail-loud guard (`validate_deltanet_decode_batch_size`,
  `models/qwen35_family.{h,cpp}`) that now refuses an over-limit batch before
  any graph is built. 5 concurrent slots × a 4-wide speculative draft is 20
  lanes on the exact axis that guard is watching; letting speculation run
  across slots would walk straight into it.

Single-slot is therefore the considered ceiling, not a placeholder: lifting
it needs the O(B) node-growth defect fixed first (§12's `qwen35moe`/`qwen35`
bullet — the batching fix is scoped but **PARKED** by user decision), and
even then the Gemma 4 cost curve above says a wider verify batch is not free
just because it fits in the graph. Someone revisiting this restriction should
read both grounds before lifting it, not just the one that happens to be
fixed.

The fallback is **edge-triggered and logged once**, not per step:
`log_speculative_fallback` prints the reason the moment the engine leaves a
speculating stretch — a second slot activated, or the one active slot is now
running a grammar — then stays silent for as long as the stretch continues,
however many steps that is, and re-arms the instant speculation resumes.
This is a policy log, not an error: the request is served correctly by the
batched path either way.

**A per-request GBNF grammar disables speculation on that slot.** The server
mechanism is `speculative_eligible_`
(`InferenceServer::SpeculativeEligibleFunc`, wired via
`set_speculative_eligible`), which the integration implements as `slot_id`
having no grammar (`slot_grammars_[slot_id] == nullptr`) — the same rule as
the CLI's `bool use_speculative = args.speculative && !grammar;`
(`src/cli/main.cpp:322`). Draft tokens are never checked against a grammar,
so accepting one could emit something the grammar would have masked out;
disabling speculation outright avoids that rather than trying to validate
drafts against the grammar mask.

**Two new callbacks on the server engine seam:**
`InferenceServer::set_speculative_decode` takes a `SpeculativeDecodeFunc`
that runs one speculative step for the single active slot — feed the last
token, draft, verify in one batched pass, and return every token the step
produced (tokens already model-verified, plus optionally a still-pending
"bonus" token carried to the following step as `Slot::last_token_pending`);
`set_speculative_eligible` takes the grammar gate above. Both are unset by
default (`speculative_decode_` / `speculative_eligible_` default-null), which
is what keeps a server started without `--speculative` on the exact
pre-existing batched path — `decode_step()` never evaluates the branch, so
the byte-identical-with-speculation-off gate holds structurally, not by
convention.

Cancellation and the per-request timeout are still checked **once per decode
step**, unchanged from the batched path — but a speculative step can now
discard up to `draft_width` already-computed, undelivered tokens on a
cancel/timeout instead of at most one. The compute already happened either
way (the batched path also discards one computed-but-undelivered token on
the same check); a speculative step just widens how much can be thrown away
per check, not how often the check runs.

**`--speculative` is refused together with `--attention-lens` at server
startup**, which used to mirror the `--flash-attn`/`--attention-lens` refusal.
That one is now phase-scoped (§5) and this one is **not**, deliberately: flash
moved because prefill is a phase the lens does not tap, whereas speculation
changes the decode the lens reads. Same class of reason as before (§11's
receipts constraints): receipts-grade
determinism is single-slot and byte-identical, and speculative decoding is
token-stable, **not** byte-identical — and, today, only ever engages on a
single slot in the first place, which would be whichever slot the lens is
mid-extraction on.

**Attention Lens (opt-in `--attention-lens`, `POST /v1/extract`).** A dedicated
endpoint — separate from the OpenAI surface by design (its inputs are a
document + a complete key vocabulary of `{key, gloss}` concepts, not chat
messages; its output is the lens format, not a completion).

**The unit is a document OR an ordered thread** (additive, 2026-09-05). A request
carries **exactly one** of `document` or `messages` — both or neither is a
fail-loud 400, because boundaries cannot be recovered from concatenated text and
a precedence rule would be a silent fallback. With `messages`, the server joins
them with one blank line (the text a caller would otherwise have concatenated, so
the validated prompt regime is unchanged) and every citation reports the message
index it landed in; fields carry `citation_messages` and the report `n_messages`.
This buys **attribution** — *"read from message 23 of 24"* — and deliberately
**not** a staleness alarm. The alarm was built and measured first: the predicate
a server can actually compute fired on 7 of 9 correctly-handled corrections and
stayed silent on the real failure, because a later message routinely restates an
old value, so **turn order does not identify supersession**
([`note-ss3-matched-pairs.md`](note-ss3-matched-pairs.md)). The receipts are
additive, so the format stays `qemmi-lens/v2`. It runs **one free
tapped decode, in one prefill** (`run_lens_extract` → `run_lens_tapped_decode`):
argmax over the full vocabulary with the P1 attention tap armed on the two frozen
lens layers, then `compute_lens_report` derives citations (L3H13, N3), the
grounded/ungrounded badge (body_mass, N3b), the tier (A5.3) and the coverage
report (COV1). All signals are document-relative. A pure `apply_absent_by_omission`
orders fields by the hinted concepts and marks the ones the model did not state
`value:null`/`badge:"absent"`.

**No grammar.** The lens once constrained this decode with ONE fixed KV grammar,
and ran a two-pass grounded presence gate on top of it. Both are **gone** (Stage 2,
2026-07-17). The grammar was refuted by measurement: on the Leg C corpus it lost
on every axis *including* the guaranteed parse it existed for (14/15 vs free's
15/15), and its `value ::= (…)+` forced a non-empty value for every hinted key —
the sole cause of the absent-concept collapse, and therefore of the presence gate
built to contain it. Freed of it the model declines natively (absent handled 30/30
vs the grammar's 10/30) — so the gate had nothing left to do, and its **N+1
prefills collapsed to 1**. `lens_grammar_gbnf()` survives *only* as the QDOCS_S1
probe's control arm (`run_lens_extract`'s `control_arm_grammar`, a probe-only
seam), so the comparison stays reproducible on shipped code; it is unreachable
from the endpoint. **This says nothing about the engine's GBNF machinery or the
per-request `grammar` field on `/v1/completions` and `/v1/chat/completions`** —
that is a separate, shipped, unaffected feature.

**The shape contract** replaces the grammar's (false) parse guarantee: *tolerant*
on shape — `lens_find_json_object` skips a ``` fence and takes the outermost
object by string-aware brace depth — and *loud* on failure —
`LensUnparseableError` ⇒ **`422 unparseable_extraction`**, split from `400
bad_request` and carrying the model's `raw`. Never a partial extraction: a refusal
and "the document has none of these concepts" are different facts.

**Repeated keys are all reported** (v3, 2026-09-05). A flat schema over a
repeating document — an invoice with three line items — makes the model emit the
key several times. v2 read the first occurrence and kept one field, dropping the
rest with no error, no badge, and **no coverage flag**: measured, the
un-extracted lines were *not* reported as un-consulted, because the model does
read them. Each occurrence now carries its own value, byte span and citations
(`LensField::occurrence`), which the per-value-span trust math supports
unchanged. Structural change ⇒ format bump to `qemmi-lens/v3`; the first entry
per key keeps its v2 value and position. It does **not** group repeated keys into
records — that is the leaf-path design, gated on measuring a repeating-group hint.

**Values are scalars or arrays of scalars; deeper nesting is refused**
(2026-09-05). An array of scalars expands to one occurrence per element, since an
element is locatable and gets its own citations — the same treatment as a
repeated key. The two are one answer in two encodings: on a three-line invoice
Qwen 3.8-9B repeats the key and Qwen 3.6-35B emits `[7, 19, 43]`, so refusing
either would make the contract depend on the loaded model. `lens_value_of` reads
scalars only and `lens_keys` is depth-blind, so before this guard a nested value
did not fail — it produced a truncated fragment shipped as a real field, and a
nested key could answer a top-level lookup with the wrong value and citations
pointing at a span the model never read (both measured, `LensNestedOutput`).
Refused rather than summarized, because a faithful record cannot be claimed for a
value that was not read. Supporting arrays properly is a *leaf-path* design —
the per-value-span trust math already generalizes to any depth — gated on a
Leg-B-style measurement of a repeating-group hint, since `line_items[0..N]`
cannot be enumerated in a complete hint up front.

**The candidate set** (v4, 2026-09-06, architect-approved,
[`plan-candidate-set.md`](plan-candidate-set.md)). Alongside the value `fields`
already returns, a request that opts into the (currently internal-only,
`LensExtractOptions::want_candidates`) candidate-set producer also gets **every
span in the document that answers a key** — not just the one the model
returned — via a **second, cold, taps-disarmed decode** over the same document
with a different instruction (`run_cand_pass2`); pass 1 is byte-untouched.
`lens_report_to_json` emits this as a new top-level `key_candidates`: an object
keyed by the **complete request vocabulary** (every key present, `[]` included
— `[]` and absent-from-the-map are different facts), each candidate a
byte-exact `{value, byte_lo, byte_hi, anchor, returned_as}`, array order
**document order (`byte_lo` ascending) and load-bearing** — not mass, not a
ranking (CF1 forbids a verdict). `anchor` and `returned_as` are derived at
serialize time from `document` and the sibling `fields` entries, not stored on
`LensCandidate` — `returned_as` links a candidate to the `fields` occurrence it
answers by **bidirectional containment** on whitespace-normalized text
(tiebreak: tightest containment, then earliest `byte_lo`), because pass 2
legitimately returns spans wider or narrower than the field's own value.
**Three distinguishable top-level states**, the reason the format bumps rather
than riding in place: `key_candidates` present ⇒ the finder ran; absent +
`candidates_error` ⇒ the finder **failed** on this document (never rendered as
an empty set — the exact confusion this feature exists to prevent, one level
down from `fields`' own `422`); absent + no error ⇒ candidates were not
requested. `fields` — taps, calibration, coverage — is completely unchanged.
Format bump: `qemmi-lens/v3` → **`qemmi-lens/v4`**, additive
([`lens-format.md`](lens-format.md)). No server/CLI flag exposes
`want_candidates` yet; that remains a separate, unapproved decision. **Action
item on `../qemmi-lens`: its `ACCEPTED_FORMAT_VERSIONS` must add
`"qemmi-lens/v4"` or every extract call fails its own fail-loud version gate.**

**Single-slot and exclusive**: `extract_lens_json` holds the model lock for the
whole tapped decode and uses slot 0 (the only correct qwen36 decode KV gather,
§12); do not drive concurrent OpenAI traffic on slot 0 while extracting.
Fail-loud on empty concepts or an oversized document.
Off ⇒ the route 404s and no lens code runs.

**Per-model calibration, keyed `{architecture, block_count, file_type}`**
(decided 2026-09-05; `file_type` added 2026-09-18, see the collision note below). `LensConstants` is not one frozen struct: `kLensCalibrations` in
`server_lens.h` is a table of calibration entries, and `--attention-lens` is
**refused at startup on any model with no entry** (`lens_calibration_for` /
`lens_calibration_refusal`, resolved once in `enable_attention_lens()` and
carried on the integration; `run_lens_extract` takes the constants with **no
default argument**, so no call site can silently fall back to someone else's
coordinates). Three models are calibrated today, all Qwen: **Qwen 3.6-35B-A3B**
(`qwen35moe`/40 and /41, citation **L3H13**), **Qwen 3.8-9B** (`qwen35`/33,
citation **L27H13** — 98% vs 84% top-3 and 0% vs 7% ungrounded false alarm on
the same messy corpus, `note-lens-qwen38-probe.md` §5.3) and **Qwen 3.8-27B**
(`qwen35`/65, citation **L19H20**, coverage **L11 @ 0.705** — added 2026-09-15,
`note-lens-qwen38-27b-probe.md`). A fourth entry, **Ternary-Bonsai-27B** (prism-ml, ternary Q2_0
group-64), was added 2026-09-18 and **reverted 2026-09-19** when the model was
de-scoped; its measurements are kept in `note-lens-bonsai-27b-probe.md` and the
`file_type` key field it motivated stayed (see below). The 9B's locate pair
(**L11 h=6**, 96.0% top-3 / 88.0% top-1 EN 95.0 / DE 97.1 **on Q8_0**, and
93.3 / 84.0 on Q4_K_M — the row is unpinned so it serves both, and its
provenance names both rates) was added
2026-09-19 and is *free*: it equals the coverage layer, so the cut stays 28 of
33 blocks. The 27B's locate pair (**L27 h=10**, 94.7% top-3 / 81.3% top-1,
EN 97.5 / DE 91.4) was added the same day and is **not** free — it is deeper
than both other constants and moves that model's cut from 20 to **28 of 65**.
The +8 blocks were accepted because the free zone is empty: the best candidate
at or below the citation layer, L15 h=17, reads 90.7% pooled and **fails German
at 88.6%**. The best head overall, L35 h=16, scores a perfect 100.0/100.0/100.0
and was declined on depth (36 of 65). `note-lens-qwen38-27b-probe.md` §8.

The `--lens-verify-only` cut is `max(citation_layer, coverage_layer,
locate_layer) + 1`. **Locate joined that max on 2026-09-18**, when an entry
first carried both a locate pair and a citation layer deeper than it: the
process serves `/v1/locate`, locate taps its own layer, and a calibration whose
locate layer sat deeper than the other two would tap a block the cut never
loaded. It was written down before it could bite, while both calibrated locate
pairs still sat at L11 equal to coverage. **It bites as of 2026-09-19**:
Qwen3.8-27B's locate 27 against citation 19 and coverage 11 is what makes that
model's cut 28 of 65 rather than 20, and it is the only entry where removing
locate from the max would still leave every other model working while silently
serving `/v1/locate` a block this mode never loaded.

*Three constants, three different provenances.* The 27B is the first model to
clear **every** arm of leg C (citation 91% top-3, coverage 97% used-clear, 0/75
ungrounded false alarms, bilingual) — the 9B that ships has never passed it, its
coverage arm scoring 87% and 83% on German. It is also the model that falsified
the *selection method*: the N3 leg picks a citation head on three synthetic
prompts, and on the 27B that head (L11H22) ranks **14 of 384** on real
documents. Heads are now selected on the corpus that judges them (`LEGCSEARCH`)
and confirmed on prompts they did not see. Two caveats travel with the entry:
German citation is 90.4% against a 90% bar, and every number is measured at
Q3_K_M where the 9B's are Q8_0 — which is what the report's `config.weights`
stamp exists to record.

*Why the key is the model and not the architecture.* An architecture string is a
family: `qwen35` alone hosts Qwen3.5-0.8B (24), Qwen3.5-9B (32), Qwen3.8-9B (33),
Qwen3.6-27B (64) and Qwen3.8-27B (65). An arch-keyed allowlist would admit all
five under coordinates measured on one of them — the exact false receipt the
refusal exists to prevent. The key is the **raw GGUF `block_count`**, not the
decode-stack depth: depth collides Qwen3.5-9B with Qwen3.8-9B at 32 and
Qwen3.6-27B with Qwen3.8-27B at 64, each pairing a calibrated model with an
uncalibrated one. Calibrating Qwen3.8-27B (2026-09-15) sharpened
this rather than easing it: two of the five `qwen35` models are calibrated now,
with **different** coordinates (L27H13 and L19H20), so an arch-keyed allowlist
has no winner to pick; and the depth collision at 64 now pairs a genuinely
calibrated model (Qwen3.8-27B) with an uncalibrated one (Qwen3.6-27B), whose
coordinates score 7.1% on the model that owns them. A future collision is
resolved by **adding a field to the key, never by widening an entry to a model
nobody measured** — and on 2026-09-18 that future arrived. Ternary-Bonsai-27B
was `qwen35` with `block_count` 64, the *same key* as the uncalibrated
Qwen3.6-27B, so admitting one would have admitted both. The added field is GGUF
`general.file_type`, and it is the principled field rather than a convenient
one: that run measured the calibration to **be** quant-sensitive —
Qwen3.8-27B's own L19H20 survives ternary at rank 5 of 384 but falls from
DE 90.4% to **89.8%**, crossing the 90% bar. **The field outlived the row that
motivated it**, deliberately: the Bonsai entry was reverted on 2026-09-19, but
a plain non-MTP build of Qwen3.8-27B also keys as `{qwen35, 64}` (65 = 64
decode + 1 NextN) and would silently inherit the MTP row's coordinates without
it. No row pins a quantization today; a row may set `kLensAnyFileType` to
accept any, which is what all four carry so the field cannot refuse a model
that worked before it landed, and a row that **pins** one beats a row that does
not. That "any" is a preserved looseness, not an endorsement: `models/` holds
Qwen3.8-9B at both Q8_0 and Q4_K_M under the one `{qwen35, 33}` entry, and only
the Q8_0 was ever measured. Tightening that is a separate decision. `coverage_used_peak`
stays 0.705 on every entry, and since 2026-09-15 that is measured rather than
inherited. The old "weak arm at 87%/84%" reading came from a control group that
was itself wrong: `OMISSION1` ablated all 133 leg C spans per model and found
53% (9B) / 38% (27B) of spans labelled "filler" to be **causally used**. Scored
against causal labels, coverage separates used from unused at **AUC 0.912 /
0.955**. 0.705 is kept because `skipped[]` is a recall-first screen (97% / 90%
recall) and the accuracy-optimal ~0.30 would trade that away; `COVCAUSAL` found
better-precision layers (L7 on the 9B, L15 on the 27B) at identical recall, but
they disagree across models and buy a shorter list, not a stronger claim. See
`plan-lens-only-engine.md` §4.

*The lens is a Qwen-family capability by measurement, not by neglect.* CLAUDE.md
makes Gemma the falsifier for anything touching the forward pass, and Gemma was
run properly: **0 of 768 candidate heads** clear even a 70% bar against the 90%
requirement, topping out at 63% (L7H13), and the norm-weighted metric is *inert
by construction* there — `gemma4.cpp:447` RMS-norms V with no learned weight, so
Spearman(rankA, rankB) = 1.0000 over all 768 candidates. The refusal is the
mechanism that keeps that honest; it is not an untested-family placeholder.
[`note-lens-gemma4-probe.md`](note-lens-gemma4-probe.md),
[`note-lens-gemma-norm-weighted.md`](note-lens-gemma-norm-weighted.md),
[`note-lens-norm-weighted-metric.md`](note-lens-norm-weighted-metric.md).
[`plan-qemmi-lens.md`](plan-qemmi-lens.md), [`lens-format.md`](lens-format.md),
[`note-nogrammar-refutation.md`](note-nogrammar-refutation.md),
[`note-lens-absent-attempt.md`](note-lens-absent-attempt.md); gates
`tests/smoke/server_extract_smoke.sh` + `QDOCS_S1=1 bin/attn-provenance`.

**Verify (opt-in `--attention-lens`, same flag; `POST /v1/verify`; Movement 1 of
[`plan-lens-server-shape.md`](plan-lens-server-shape.md)).** The other half of
the lens surface: `verify(document, key_vocabulary, extraction)` teacher-forces
a KNOWN extraction — the exact text a prior `/v1/extract` report's `raw` field
carried — instead of generating one, and reproduces the same report without a
decode loop or sampling. Request shape mirrors `/v1/extract`'s
(`document`|`messages`, `key_vocabulary`) plus the required `extraction`
string; same 404-when-off, 400-bad-request, and 422-`unparseable_extraction`
(the shape contract applied to the teacher-forced text). Driver:
`run_lens_verify` (`server_lens.{h,cpp}`), wired via
`QweniumServerIntegration::verify_lens_json`.

**Locate (opt-in `--attention-lens`, same flag; `POST /v1/locate`; approved
2026-09-18, [`plan-lens-only-engine.md`](plan-lens-only-engine.md) §5 and
`../qemmi-lens/docs/plan-locate-and-cut.md`).** The THIRD verb and the weakest
of the three claims. `locate(document, key_vocabulary, top_k)` returns, per key,
up to `top_k` DOCUMENT byte ranges — where those key tokens look. Nothing is
generated and nothing is audited, so the response deliberately carries **no
`extraction_origin`**: it is not an extraction of either origin, and a consumer
that reads it as one is reading a claim never made. Driver: `run_lens_locate`,
wired via `QweniumServerIntegration::locate_lens_json`. One head-less tapped
prefill over the ordinary lens prompt; each key owns a token span in the
instruction, and that span's rows over the document are the readout (causally
available because the document precedes the instruction).

Two facts about it are load-bearing:

- **It reads its OWN calibrated pair, `LensConstants::{locate_layer,
  locate_head}`, not the citation pair.** The route shipped for one day reading
  the citation head on the assumption that a retrieval head is a retrieval head;
  the LOCHEAD sweep put that head at rank 107 of 160 for this regime. The
  generated→source head is not the key→source head. `locate_provenance` names
  the run, same discipline as `flash_prefill_provenance`. **Default -1 = not
  measured, and such a model is REFUSED** rather than served another model's
  coordinates.
- **It runs on the main scheduler, like every verb (2026-09-27).** Every
  read-back pass allocates through `ForwardPassBase::alloc_readback_graph`
  (§12, the tap seam), which plans memory from the graph being run, so verbs
  share one scheduler whatever layers they tap. Locate, `/v1/verdict` and
  `/v1/compare` ran on a dedicated `locate_scheduler()` until then; merging it
  saved **1.39 GB** on a verify-only server and **1.83 GB** on a full one
  (physical footprint after a ~9.6K-token verify + locate: 9,342 → 7,954 MB and
  11,264 → 9,434 MB), responses byte-identical, chat byte-identical with lens
  traffic in between. Safe because every model operation holds
  `model_mutex_`, so no two passes ever use the scheduler at once. What
  follows is the 2026-09-18 defect the dedicated scheduler once fixed — found
  by a client. Locate taps
  ONE layer; extract and verify tap TWO. `ggml_gallocr_needs_realloc` keys on
  node count and node SIZES, **never on `GGML_TENSOR_FLAG_OUTPUT`**, and
  locate's graph has the same node count as verify's tapped pass with strictly
  larger tensors. So after a locate over a longer document, verify's smaller
  graph "fits" the cached plan, galloc skips re-planning, and verify inherits a
  plan in which `kq_soft.<citation_layer>` is not protected — DeltaNet layers 4
  and 6 write over the tap and every later `/v1/verify` returns a confident
  FALSE receipt (negative `body_mass`, badges flipped, no error) for the life of
  the process. It reproduced only under `--lens-verify-only`, where
  `reserve_max_batch` is skipped and no large stable plan is planted first.
  The rule it rested on until 2026-09-27 — every verb sharing a scheduler
  marks the same tap set — is retired: `alloc_readback_graph` removes the need.
  `ForwardPassBase::get_attention_taps` still range-checks every tap against
  `[0, 1]` — a post-softmax weight cannot fall outside it, so bytes that do are
  not this pass's attention, and the lens refuses rather than reports — but
  that check cannot see another layer's softmax in the slot, which is why the
  fix is the plan, not the check. Gate: `tests/smoke/server_locate_smoke.sh`
  gate 8, `VERIFY_ONLY=1` (the gate cannot run under `--lens-locate-only`,
  which refuses `/v1/verify`; that leg runs gates 1–7 plus an explicit
  refusal check and says so rather than reporting a full pass). Precedents:
  the MTP head's dedicated scheduler and `server-image-multirequest-bug.md`.
- **It takes an optional `key_aggregation`** (`"max"` default, `"mean"` opt-in;
  2026-09-20). Not a tuning knob — a property of the KEY KIND. `max` is the
  reduction LOCHEAD measured and is correct for the 1-4 token field names it
  swept; on SENTENCE-length keys it inverts into a defect, letting one filler
  token spike so the wordiest key wins (`an` against ` agreement`, 0.507, with
  the same key's next span at 0.047). `mean` was worth **+17.5 points** on a
  4-way routing task at identical latency (DECIDE1,
  `note-lens-qwen38-probe.md`). Default is byte-identical to the behaviour that
  predates the field, and `mean` sets `uncalibrated` because the shipped
  provenance rates were all measured under `max`.
- **It takes an optional `head`** (`"locate"` default, plus `"choice"`,
  `"absent"` and `"score"` opt-in; 2026-09-20; `"inject"` 2026-09-24), selecting which calibrated
  job's pair to read. **Four jobs now read four different heads** on the 9B —
  locate L11 h=6, choice L11 h=3, absent L19 h=10, score L19 h=11 — two pairs
  sharing layer 11 and two sharing layer 19. A layer is not a job: on Qwen3.8-9B choice sits at the **same layer** as locate (11) on a
  different head (3 vs 6), measured 92.5% on 4-way routing — identical on Q8_0
  and Q4_K_M — so choice is free on a `--lens-locate-only` server. Refused
  fail-loud where DECIDEHEAD has not run; reading choice off the locate pair
  measured 87.5%, and off the citation pair far worse. The two jobs have
  opposite recipes (`max` for locate, `mean` for choice) and `uncalibrated`
  follows whichever applies. A third value, `"absent"` (2026-09-20), reads the
  `noul` pair — **L19 h=10** on the 9B, AUC 0.9948/0.9953 across quants, rank 1
  of 128 on both. It is the first head that is **not** free: absence has no
  quant-stable shallow alternative, so a `--lens-locate-only` server's cut
  became `max(locate, choice, absent) + 1` and moved from 12 to **20 of 33
  blocks** (2055 MB → 3038 MB on Q4_K_M). Verify-only is unchanged at 28, where
  citation still dominates. A fourth value, `"score"` (2026-09-20), reads the
  ordinal pair — **L19 h=11**, the head *next to* absence's h=10 on the layer
  absence already paid for, so the cut is unmoved at 20 and score is free. It
  is also the first pair not selected by accuracy: an ordinal's argmax flips
  between the two adjacent levels a document sits between, so the sweep ranked
  heads by ordinal concordance (1.0000, rank 1 of 128 and unique on **both**
  quants) and tie-broke on how far apart adjacent levels are pushed. Its
  fractional output is deliberately **uncalibrated in scale** — four true
  levels read 0.62/1.02/1.49/2.13 — so callers read the fraction and the
  masses; an affine correction would be the first *fitted* constant in the
  table and none is landed. Adjacency is not interchangeability: h=10 reads
  the ordinal job at 0.9861 and separates at 1.97 SD against h=11's 3.48.
  **The 27B carries three** — locate L27 h=10, absent L31 h=23, choice L39 h=7
  — and they put its locate-only cut at **40 of 65** against the 9B's 20 of 33.
  Which pairs are free is a property of the model, not of the job: choice
  shares locate's layer on the 9B and costs 8 blocks over absence here, which
  is why the cut is computed from the constants rather than written per mode.
  The 27B's choice pair was also landed *against* the better pooled head
  (L47 h=13, 100% pooled but 90% worst-case held out, +8 blocks) — a pooled
  maximum is not a rate. Its score pair was swept and **refused**: no held-out
  agreement on any of three axes. A `-1` therefore means one of two different
  things, and the provenance string is what distinguishes "measured and
  refused" from "never measured" — both are refused at the route either way.
  A fifth value, `"inject"` (2026-09-24, `note-lens-injection-probe.md`), is
  the first role that **reads no key**: it averages the *template-tail* rows
  (everything after the instruction) of **L11 h=0** over the document, after
  Attention Tracker's "distraction effect", and returns ONE `hits` entry named
  `instruction_like` instead of one per key; `key_aggregation` does not apply
  and only a key-mode prompt sets `uncalibrated`. It is a **highlighter, not a
  detector**: the injected sentence is the top segment 90.3% of the time
  (chance 1.8%), but a single threshold across documents reads only
  0.83–0.90 AUC. It always points somewhere, so a clean document still gets
  a span. Free on the 9B (locate's layer; the cut stays 20) and folded into
  both cut expressions so a model where it sits deeper pays honestly.
- **It truncates after `locate_layer` alone**, so the constant IS the route's
  cost. On Qwen 3.6-35B that layer is 11, which is exactly
  `max(citation_layer, coverage_layer)` — so `/v1/locate` costs a
  `--lens-verify-only` server **not one additional block**, which is why it is
  served there and `/v1/extract` is not. That equality is a property of this
  model's calibration, not a general fact: a model whose sweep lands deeper
  makes the locate slice bigger than the verify slice — which Qwen3.8-27B then
  did (locate 27 vs citation 19). Because the constant IS the cost, it is also
  what `--lens-locate-only` loads to (see below): 12 of 33 blocks on the 9B.

Two things had to generalize, both **recipe-agnostic and byte-inert when
unarmed** — the same discipline the P1 tap seam already keeps:

- **Truncated prefill** (`DecodePolicy::truncate_after_layer` /
  `effective_layer_count()`, `models/decode_policy.h`; setter
  `ForwardPassBase::set_truncate_after_layer`). Causality is what makes this
  *exact*, not an approximation: an attention layer cannot depend on a layer
  above it, so a prefill stopped after `max(citation_layer, coverage_layer)`
  produces identical tapped rows to the untruncated pass, cheaper because the
  omitted layers' matmuls (and any MoE/DeltaNet dispatch inside them) never
  run. Default `-1` (full stack) keeps `is_default_byte_reproducible()` true
  and every recipe's default path node-for-node unchanged. **Every recipe
  bounds its prefill layer loop(s) with `effective_layer_count()`** —
  `qwen3`, `qwen35`, `qwen36`, `gemma1`, `gemma2`, `gemma3`, `gemma4` — which
  is the cross-family proof this is a plain iteration-count parameter, not a
  Qwen-shaped kernel capability requiring a `supports_*` gate the way
  `--flash-attn`/`--persistent-graph` do. Verify is the only caller; decode
  graphs never read the field, and Gemma carries no lens *claim* (still no
  calibration entry, §12), only the interface proof.
- **`get_attention_taps` learned the prefill shape** (`forward_pass_base.{h,cpp}`).
  `kq_soft.<il>` is `[n_kv, 1, n_head]` at decode (one query row per step) and
  `[n_kv, n_q, n_head]` at a tapped prefill (every query position the graph
  processed in one pass). `AttentionTap` gained an `n_q` field (default 1,
  decode's shape byte-for-byte unchanged) and the reader now sizes off the
  tensor's own `ne[1]` instead of assuming 1 — no new method, since the tap
  tensor is named identically either way. `server_lens.cpp`'s
  `slice_prefill_tap_row` translates one query row of that block into the
  same decode-shaped `[n_head][n_kv]` layout a `LensStep` already carries, so
  `compute_lens_report` needed **zero changes** — it never learns there are
  two shapes; the translation lives entirely in the new driver.

**Correctness gate (§3.4.5 of the plan): not bit-for-bit, decision-for-decision.**
The batch-vs-single-token numerical fork (§11, "…except where the hardware
forbids it") applies here too — a teacher-forced multi-row prefill and a
token-by-token decode take different Metal kernels. Measured drift 5.6e-4
against a 0.019 decision margin (plan §2.1). The gate is self-checking (no
corpus): `tests/smoke/server_verify_smoke.sh` extracts a document once, feeds
that exact extraction back through `/v1/verify`, and diffs the two reports
field-for-field — same values, same badges, same tiers, same top-1 citation by
real-source membership (not exact mass), same `skipped[]` membership. **Passed
on both calibrated entries** (Qwen 3.8-9B-Q8_0 and Qwen 3.6-35B-A3B-UD-Q3_K_XL).

**Verify-only server mode (`--lens-verify-only`, same flag family, 2026-09-15).**
Verify's own truncated-prefill argument — causality means a pass stopped after
`max(citation_layer, coverage_layer)` is *exact*, not approximate, because a
layer cannot depend on one above it — applies just as well to *loading* as to
*computing*: a block above that cutoff is never read by any `/v1/verify` call
this process will ever serve, so there is no reason to load its weights either.
**It also serves `/v1/locate`** (2026-09-18). Sharing a scheduler with
`/v1/verify` once corrupted every later verify here (§6), because
`reserve_max_batch` is skipped in this mode; locate ran on its own scheduler
until 2026-09-27 and shares the main one again now that each read-back pass
gets its own plan (`server_locate_smoke.sh` gate 8, `VERIFY_ONLY=1`, is the
regression test on exactly this sharing). The block
count is safe **by intent, not by arithmetic** — and that changed. It was once
safe by coincidence: locate truncates after `locate_layer`, which on every model
swept up to 2026-09-18 was 11, exactly the `max(citation_layer, coverage_layer)`
cutoff this mode already loaded to. Qwen3.8-27B's sweep (2026-09-19) landed
locate at **27**, deeper than that cutoff, which is the case this paragraph was
written to anticipate. The resolution was the first of the two options it named:
`needed` is `max(citation_layer, coverage_layer, locate_layer) + 1`, so the 27B
loads 28 of 65 instead of 20. `/v1/locate` was **not** refused here. Any future
row is covered by the same arithmetic; no third option is needed.

`--lens-verify-only` (requires `--attention-lens`; refused fail-loud alone)
resolves the calibration entry from GGUF **metadata** (`{architecture,
block_count}`, readable before any tensor is touched) at the same point
`enable_attention_lens()` would, computes `needed = max(citation_layer,
coverage_layer) + 1`, and calls `Model::load_tensors(needed)` instead of the
default full load. `GGUFLoader::load_tensor_metadata` gained an optional
`max_blocks` filter (default: every block, byte-identical to before the
parameter existed) that skips `output.weight`, `output_norm.weight`, and every
`blk.<i>.*` for `i >= max_blocks` — the exact tensor set a forward pass
truncated at `max_blocks - 1` can never read — so the SSD read and the
Metal-buffer copy both shrink with it; `Model::assign_tensor_pointers` leaves
those blocks default-constructed (every pointer null) instead of `require()`-
ing tensors that were never asked for. Measured on Qwen 3.8-9B-Q8_0 (calibration
L27H13, cutoff layer 27 ⇒ 28 of 33 blocks loaded, 372 of 442 tensors): Metal
weights buffer **9322 MB → 7168 MB** (−23.1%, 2154 MB). KV cache size is
UNCHANGED (256 MB either way at this ctx) — the recurrent/attention state is
sized off `n_main_layers_` from GGUF metadata, not off which blocks actually
loaded, so this flag saves weight bytes only; sizing the state to match is a
further optimization this change does not attempt. This is the smallest
saving of the three calibrated entries (9B skips 5/33 blocks); the 35B-A3B
entry (12 of 40) and the 27B entry (20 of 65) skip a much larger fraction.

**Split locate prefill — a per-model licence (2026-09-25).** `run_lens_locate`
takes a `LensPrefillShape`: `OneShot` (the pass every constant was measured
on), `Split` (the document prefilled untapped, then the instruction + keys +
template tail tapped at their real positions) or `SplitFlash` (the same with
flash on the document pass). Every row a mode reads sits after the document
and attention is causal, so only chunk-boundary and flash rounding differ;
the route fails loud if a read row would fall inside the document pass.
**Which shape a model uses is a field of its calibration row**
(`LensConstants::locate_prefill_shape` + `locate_prefill_provenance`, default
`OneShot`), the same pattern as `flash_prefill_ok`: licensed per model by the
LOCSPLIT drift gate (`tests/perf/attn_provenance.cpp`), never inherited — and
per QUANTIZATION: Qwen3.8-9B at Q4_K_M passed and Q8_0 failed (absent on long
documents changed 2 of 12 top-1 spans), so the licence lives on the table's
first `file_type`-pinned row (`kGgufFileTypeQ4_K_M`, 15). Both 9B rows are built
by one function (`qwen38_9b_constants()`), so the pinned row cannot drift from
the any-quant row in anything but the licence; the any-quant row, and so Q8_0,
stays `OneShot`. The report names the shape (`prefill`)
and a split+flash locate is stamped `config.attention = "flash-prefill"` by
route (`RoutePrefill::DocumentPassFlash`), because its document pass ran flash
by licence rather than by server flag. Measured at a 10K prompt: locate
27.5 → 23.4 s, absent 45.5 → 37.4 s, GPU compute buffer 8.2 → 2.9 GB; split
without flash is slightly slower and exists as the base for keeping a
document between requests ([`note-lens-prefill-only-engine.md`](note-lens-prefill-only-engine.md)).

**The kept document (`document_id` on `/v1/locate`, 2026-09-26).** Built on the
split: pass 1 depends on the document alone, so `run_lens_locate` can store slot
0 right after it (`qinf::snapshot::capture_slot` — KV **and** DeltaNet state,
which a position rewind cannot restore) and later restore it and run pass 2
only. The store is `LensDocumentStore` (`server_lens.h`), owned by the server
beside the slot it restores into and guarded by `model_mutex_`; `qinf-server`
now links `qinf-snapshot` for it. The only hit test is exact equality of pass
1's tokens; the document hash only turns an id reused for other text into a
400. A pass 1 serves any read at or below the layer it was computed to
(truncation is causal in depth). One-shot rows refuse an id. RAM only: at
most **`--lens-kept-documents`** entries (default 4, 2026-09-27) keyed (route, id) — extract, locate (verdict shares it), compare — so
one document kept on all three fills 3; LRU across routes; 15-minute idle TTL
checked on every extract/locate/verdict/compare call; ids are 1–256 bytes on
every route. The TTL is a fixed default, not a flag (`kLensDocumentStoreTtl`);
the size flag exists because the store, not the GPU, sets how many users stay
warm: users who take turns past the size all go cold (0/15 warm at 5 users on
4 entries, 24/24 at 8 users on 8 — docs/note-lens-concurrency.md). An entry is
~64 KB per document token + ~48 MB DeltaNet state on the 9B (F32 KV). Gated
by LOCWARM: warm bit-identical to cold on 115/115 reports; 10K locate 22.4 s →
0.30 s. **`/v1/extract` uses the same store** (2026-09-26): its older warm path
(`LensWarmDocument` — keep slot 0, rewind the position) is deleted, because a
position rewind restores KV but not DeltaNet state, and EXTWARM measured it
NOT warm == cold on the hybrid 9B (edit loop 0/6; with a locate in between the
output changed 6/6). Extract's document pass is now kept and restored as a
snapshot (`LensDocPass`), and the candidates pass resumes from the same
snapshot instead of rewinding. Entries are keyed per route
(`LensKeptRoute`), because the two routes compute the document pass
differently (truncated tapped graph vs full `run_prefill`); one store of 4
serves both. EXTWARM: 15/15 identical to cold, both sequences, candidates on.

**The verdict (`POST /v1/verdict`, 2026-09-26, docs/plan-lens-verdict.md).** A
new lens route that reads **logits**, not attention: per question, one answer
(yes / no / unclear) off the prefill's last row, after `verdict_layer` (logit
lens; 27 → 28 of 33 blocks on the 9B), plus a receipt from the locate head. A
new driver, `run_lens_verdict` — locate, extract and verify are untouched
(gate G1: 90 reports byte-identical before and after). One document pass per
request, truncated at the verdict layer and computed as locate's pass 1 (the
row's `locate_prefill_shape`); each question resumes from its snapshot. With a
`document_id` that pass is kept under `LensKeptRoute::Locate`, so it serves a
later locate too. Served on the full lens server, and on a verify-only server
started with **`--lens-verdict`** (2026-09-27; requires `--lens-verify-only`,
refused anywhere else and on a row without `verdict_layer`): truncated servers
load `token_embd` and the blocks but not the output head, which on the 9B is
untied (`output.weight` 4096 × 248320 Q6_K, ~834 MB = +796 MiB measured), and
the flag loads it — `Model::load_tensors(max_blocks, keep_output_head)` /
`GGUFLoader::load_tensor_metadata(..., keep_output_head)` — and folds
`verdict_layer` into the verify-only cut (no change on the 9B, 28/33). Answers
are byte-identical to the full server's (7/7 responses, 22 questions). Licensed per model and quant by
`LensConstants::verdict_layer` / `verdict_provenance` / `verdict_envelope_tokens`
(appended last; −1 = refused) — set only on the 9B Q4_K_M row. Every pass is a
truncated prefill, on the main scheduler like every verb.

**The image verdict (`POST /v1/verdict` with `"image"`, 2026-10-01,
docs/plan-image-verdict.md).** A different concern that only shares the URL:
`handle_image_verdict` (http_server.cpp) claims a body carrying `"image"` and
returns before any lens gate; a document body reaches the lens handler
unchanged (G0: 11 responses, valid and refused, byte-identical with and
without the dispatch). The logic is `src/server/image_verdict` (in
`qinf-server`): encode + prefill up to the end of the image span **once**,
`capture_slot` (the RPOS section carries the M-RoPE position, §12), then per
question `restore_slot` and a prefill of the question alone from
`get_rope_pos`; P(yes) vs P(no) with `run_lens_verdict`'s token sets; the LLM
prefill pinned **materialized** (how the cuts were measured). Questions are a
calibrated **mark** (the server words it exactly as measured: signature,
stamp, date) or a free question (0.5, `calibrated: false`); answer = yes if
p ≥ cut_yes, no if p < cut_no, else unclear (stamp 1.0 / 0.5: never yes —
lookalikes score like real stamps, docs/note-stamp-lures.md). Gates of its
own — `--mmproj`, a full model (a truncated lens server is refused), and an
`ImageVerdictCalibration` row keyed `{arch, block_count, file_type,
projector, projection_dim}` (one row: Qwen3.6-35B-A3B UD-Q3_K_XL + its
mmproj) — reported at startup. `ServerVision` gained only plain accessors and
`prepare_image`; `server_lens` is untouched. Slot 0, exclusive, under the
model lock like the lens verbs; the slot is left clean (a chat request after
it answers normally). Reproduces the probe bit for bit (G2, 166/166). Format
`qemmi-verdict-image/v1`. **`image_id`** (1..256 bytes) keeps the post-image
snapshot in the image verdict's own `ImageVerdictStore` (not the lens's): a hit
needs the same id, the same image (preprocessed pixels' content id) and the
same image-inclusive tokens; the same id for another image is a 400; 4 entries,
LRU, 15-minute TTL, fixed (no flag). Warm answers are the cold ones exactly on
the 35B-A3B (G4), 3 questions 11.1 s → 0.7 s.

**Compare (`POST /v1/compare`, 2026-09-26, docs/plan-lens-compare.md).** The
seventh mode, the first read ACROSS two documents: the original (the caller's
units, one per line) and a second version (free text — a translation, a
rewrite); per unit, the compare head's attention from the second version's
rows (max over rows, mean over the unit's tokens, relative to the mean of the
best-covered quarter of units — `lens_compare_baseline`; the median until
2026-09-27, which missed 35–49% of drops when most units were missing),
and `missing` below the row's `compare_threshold`. A new driver,
`run_lens_compare`; split prefill (pass 1 = instruction + original, pass 2 =
the second version, compare head tapped), truncated after `compare_layer` (15
→ 16 of 33 blocks). The compare layer is folded into every lens server's cut
(it changes no cut on the 9B), and the readout is attention, not logits, so
**every** lens server serves it — locate-only included. With a `document_id`
the original's pass is kept under `LensKeptRoute::Compare`, so many revisions
resume from one original. Licensed by `compare_layer` / `compare_head` /
`compare_threshold` / `compare_provenance` / `compare_min_units` /
`compare_envelope_tokens`, appended last, set only on the 9B Q4_K_M row.
The original's pass is always materialized: unlike locate and verdict it does
not follow the row's split+flash licence (flash changed 6 of 31,807 flags at
COMPAREGATE G3).

**Locate-only server mode (`--lens-locate-only`, 2026-09-19).** The same
argument taken to its floor. `/v1/locate` reads ONE layer and generates
nothing, so a process that serves locate *alone* needs `locate_layer + 1`
blocks — citation and coverage are deliberately **not** in this cut, because
this mode refuses `/v1/verify`, and folding them in would load blocks solely to
keep a route that is switched off. Requires `--attention-lens`, is **mutually
exclusive** with `--lens-verify-only` (they set different cuts and serve
different route sets, so silently preferring one would either waste weights or
withdraw a route the operator asked for), and is **refused fail-loud on a model
whose calibration row carries no measured locate head** rather than falling
back to the verify-only cut.

Measured on Qwen3.8-9B-Q8_0 (`locate_layer` 11 ⇒ **12 of 33 blocks**, 160 of 442
tensors): Metal weights buffer **3661 MB**, against verify-only's 7168 MB on the
same model — a **1.96×** reduction. Latency is unchanged by the mode and always
was: `/v1/locate` sets `truncate_after_layer` per request, so it computed 12
blocks even on a full server. What the flag buys is **residency, not speed** —
which is exactly what makes it the replication unit. One process still serves
one request at a time (every lens route holds `model_mutex_` exclusively, and
the lens's global engine state — `attention_taps`, `truncate_after_layer`,
`prefill_attn_impl` — lives on the one shared `ForwardPassBase`), so concurrency
for span-only comes from running N of these, not from batching. Measured
locate latency on the 9B: 367 ms at 326 chars, 733 ms at 1430, 1750 ms at 4006,
4116 ms at 8606 — slightly superlinear, and nearly flat in key count (+6.7%
from 1 key to 15). Against 1810 ms for `/v1/verify` on the same document,
locate is 2.59× cheaper.

The three modes are one `LensServerMode` enum (`Full` / `VerifyOnly` /
`LocateOnly`), not a pair of booleans: the cut, the block counters and the
refusal text all derive from that single value, so a server cannot disagree
with itself about which blocks it holds. Route gating reads `lens_truncated()`
(any cut) except for `/v1/verify`, which is legal under `VerifyOnly` and
refused under `LocateOnly` — a per-route fact, not a property of being cut.

Three things keep this from becoming a second, cheaper way to lie:

- **The output head and final norm are never needed.** `run_lens_verify`
  already calls `build_prefill_graph(..., want_logits=false)` on both its
  passes — verify never decodes, so `output_norm.weight`/`output.weight` were
  dead weight (literally) even before this flag existed; skipping them is not
  a new omission, just a load that finally matches the read pattern.
- **`config.weights` is unaffected.** `ModelMetadata::weights_hash` is computed
  once in `GGUFLoader::load_model`, from the *full* parsed tensor inventory,
  before `load_tensors`/`load_tensor_metadata` (partial or not) ever runs — so
  a verify-only report and a full-server report of the same file stamp the
  identical hash and stay comparable (`lens-format.md`: "two reports are only
  comparable when this matches"). Hashing the loaded subset instead would have
  made every verify-only report incomparable with every full-server one —
  exactly the failure mode the stamp exists to prevent. Pinned by
  `tests/unit/test_model.cpp`.
- **Every other route is refused fail-loud, not silently degraded.**
  `/v1/extract`, `/v1/completions` and `/v1/chat/completions` all 404 by name
  (endpoint, mode, alternative) before touching the engine — a truncated model
  cannot generate, and there is no partial-credit response to give. The
  startup banner states blocks loaded vs total and which calibration constants
  set the cut, so an operator reading the log sees the same fact the refusal
  text states. Reservation (`reserve_max_batch`, §10) is skipped in this mode:
  it exists to pre-size the largest graph *decode* will ever build so galloc
  never reallocates mid-generation, and this mode never decodes — every
  `/v1/verify` call already resets the scheduler and allocates its own graph
  fresh, same as `/v1/extract` does today with no reservation help. Reserving
  anyway would additionally be unsafe: reservation builds a full-depth decode
  graph, and this process's blocks past `needed` are unloaded.

Metal-only: the copy-avoiding filter lives in the backend-buffer load path:
Path A (CPU-only) has no filter, so `Model::load_tensors` refuses a partial
request fail-loud rather than silently loading (and paying to copy) every
tensor while claiming a saving that never happened.

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
  which is the pressure test passing. Four implementations now: `gemma3`,
  `gemma4`, `qwen36` (`qwen35moe`) and `qwen35`. The two Qwen recipes add 2-D
  positions via `image_span_is_2d()` and the optional `grid_w`/`grid_h`
  parameters — additive, so the Gemma recipes ignore them unchanged.

  **Ordering contract (load-bearing).** `build_image_substitution` both splices
  the span *and* registers the `ImageEmbeddingsInput` that uploads the encoder
  output, so a recipe MUST call `graph_inputs_.clear()` **before** the splice.
  Clearing after it discards the upload silently: the graph keeps the tensor and
  the splice still overwrites the residual stream with it, but nothing fills it,
  so the image span carries stale buffer contents and the model confidently
  describes noise. That was the qwen36 vision bug, and it survived weeks of
  investigation because every component was correct in isolation.
  `ForwardPassBase::set_prefill_inputs` now refuses it fail-loud
  (`GraphInputSet::has_slot`, pinned by `test_graph_input`).

**Encoder attention is flash attention, always (2026-09-30, user decision; no
flag).** Both ViTs (`siglip_encoder`, `qwen3vl_encoder`) use one
`ggml_flash_attn_ext` per layer — K/V cast to F16, F32 accumulation — instead
of `kq` → `soft_max` → `kqv`. The materialized form wrote an `n_pos × n_pos`
score matrix per head per layer (a 1024 × 1440 page = 5760 patches: ~2.1 GB
per layer) and was 64% of the Qwen tower's time: **10.0 → 6.1 s per page**
on the 35B-A3B's mmproj (`docs/note-verdict-img-probe.md` §11). Unlike the
text side's opt-in `--flash-attn` (§5), there is no switch: the encoder is
never tapped (the lens reads the LLM, and image attention readouts are closed),
so nothing needs the materialized scores. **Not bit-identical to the old
form:** Qwen page embeddings moved by rel-L2 ~1% (the same with F32 K/V — it is
the reduction order, not F16); SigLIP moved *closer* to the captured llama.cpp
reference (rel-L2 2.94e-3 → 2.59e-3, min cosine 0.999977 → 0.999984), and
MedGemma's greedy description of the e2e X-ray is word-for-word unchanged.

Preprocessing splits along the same grain as the seams. The **recipe** —
`vision/image_preprocess.h`, `ImagePreprocess` plus one factory per projector —
is projector knowledge and lives with the encoders. The **pipeline** —
`image/image_loader` — is IO: decode, resample, normalize, emit a Bitmap. It
lives in its own directory rather than in `vision/` (which is the encoder
subsystem, not the image pipeline) and rather than in `cli/` (both front ends
consume it — the server compiles it directly). It is
byte-gated against captured llama.cpp references — both sizing modes, the
gemma3 fixed square and the qwen3vl dyn-size canvas (one fixture per branch of
smart_resize) — because the encoder must see exactly what it saw in training
(aspect-preserving letterbox, align-corners bilinear, uint8 intermediate).

Which recipe and which encoder a given mmproj gets is **one** decision, made in
`vision/vision_profile`: projector type → `{encoder, cache tag, marker token
ids, framing string, thinking flag, preprocessing recipe}`, as an exhaustive
switch that throws on an unregistered projector. Both front ends (CLI and
server) consume that profile rather than branching themselves, so a new
projector is taught to the system in exactly one place.

Server-side, `ServerVision` routes `image_url` content through the same seams,
with an embedding cache (`--image-embed-cache`) and an image-prefix KV cache
(`--image-prefix-cache`) so a recurring image skips the encode and/or the
image-span prefill. `--image-prefix-cache` is **refused at setup on an M-RoPE
recipe** (both front ends): the snapshot blob carries a row count and no rope
coordinate, so a VL slot cannot be round-tripped (§12).

Two distinct defects have made "every image request after the first" degenerate
into token soup, and both fixes live in `ForwardPassBase` — the shared owner —
so no recipe can miss one:

- **Stale galloc buffer** ([`server-image-multirequest-bug.md`](server-image-multirequest-bug.md)).
  The image-prefill graph runs on the SAME scheduler as text prefill and decode;
  galloc re-plans across those alternating shapes and used to hand the
  substituted residual a reused buffer. Fixed by pinning that node as a graph
  output (`ggml_set_output` in `build_image_substitution`). The dedicated-image
  scheduler was the leading candidate and was **tried and reverted** — it did not
  fix the bug. Do not re-propose it without new evidence.
- **Accumulated rope divergence** (P6, Qwen-only). The per-slot rows-minus-
  positions record survived the slot clear between requests, so the second
  image's delta landed on top of the first's and every decode position after it
  went negative. Fixed by making the record's staleness test single-sourced
  (`live_rope_record`), so the writer drops an outlived record exactly as the
  readers do. Scalar recipes never write a record, so Gemma could not reach it.

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
(`engine/decode_step`, opt-in per call). When the peek collapses to exactly one
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
| Element type | F32 default; F16/Q8_0/Q4_0 via `--kv-type` | always F32 |

**KV element type.** `simple_kv_cache` takes `type_k`/`type_v`; every recipe
passes what `create_forward_pass` was given, defaulting to **F32** (the
historical, byte-identical behaviour). **`--kv-type f32|f16|q8_0|q4_0`** selects
it on both front ends; `--kv-f16` remains as an alias. Each step down is
token-stable but *not* byte-identical, so it carries the same status as
`--persistent-graph` (§10). Recurrent state is unaffected — always F32.

**The quantized types are flash-only, and that is structural, not policy.** The
materialized path transposes V (`ggml_permute(v,1,2,0,3)` then `ggml_cont`),
which moves `ne[0]` — the block dimension of every quantized type — out of
position, and Metal's `CPY`/`CONT` has no quantized *source* case, so the node
would take ggml's silent CPU fallback (§3) rather than fail. Both front ends
therefore refuse `q8_0`/`q4_0` without `--flash-attn`, fail-loud, before any
weights load (`kv_type_requires_flash_refusal`, `state/kv_cache_simple.h`).
**The attention lens is excluded by a second, direct refusal** (`http_server.cpp`,
2026-09-13). It used to fall out transitively from the flash/lens refusal; once
that became phase-scoped, a quantized cache could have slipped through on
`--attention-lens --flash-attn`. It must not: the cache is written by prefill
and **read by decode**, and the lens decode is materialized by construction, so
the V transpose above would meet a quantized source on exactly the path the
receipts are read from. Phase-scoped flash therefore does **not** unblock a
quantized KV cache for the lens, and cannot while the tap needs a materialized
decode — the one place `plan-lens-server-shape.md` §4.2 guessed wrong. Measured KV bytes on Qwen3-0.6B at ctx 2048: **896 / 448 / 238 /
126 MB** for f32 / f16 / q8_0 / q4_0. This is a **capacity** lever on the
`ctx × slots` axis, not a speed one: KV is ~1% of decode bandwidth against the
weights, so the decode ceiling is ~1% (§10, Amdahl). Note the ggml b10582
dequant-to-F16 flash pre-pass is **prefill-only** — it gates on
`src[0]->ne[1] < 32`, i.e. n_batch < 32, above our slot count.
Measured on both families at ctx 4096: Qwen3.5-0.8B 96 -> 48 MB, Gemma 3 1B
208 -> 104 MB, Qwen 3.6 160 -> 80 MB, with the greedy token sequence and the
top-5 ordering unchanged. Attention reads the cache through views whose
strides are derived from the tensor's own type, never `sizeof(float)`;
hardcoding the stride silently mis-reads a non-F32 cache instead of failing.
**That sentence described an intent the decode side did not honour until
2026-09-02**, and the gap was invisible because the two gather branches return
different types: `gather_k`/`gather_v` (multi-slot) go through `ggml_get_rows`,
which always emits F32, so a float stride is correct there; `gather_k_single`
(the B==1 fast path) returns a *view* of the cache and keeps its element type,
where the same literal mis-reads it. `build_batched_attention` and
`build_gated_batched_attention` hardcoded `sizeof(float)` in both, so `--kv-f16`
had been silently corrupting single-slot decode since it shipped — degenerate
output, not a failure. Both now use `ggml_row_size(k_gathered->type, …)`, which
is exact-identity for F32 (`4n`) and therefore byte-identical on the default
path; gated before/after on `qwen35` + `gemma3` + `qwen3` (identical output) and
by the existing `Gemma3`/`Gemma4 Tier1SingleSlotBitwise` logit memcmps. The type
contract itself is now pinned model-free by
`test_kv_cache_simple.cpp::KVCacheGatherTypeTest`.
The dtype is part of `path_tag()` and of each slot's serialized header, so a
snapshot or prefix blob captured under one KV dtype is refused fail-loud
rather than resumed under another.

They are **never unified** behind a shared base class — a common interface
would force no-ops on one side and hide the rewind asymmetry that matters for
speculative decoding and every warm-KV feature. Hybrids (Qwen 3.5/3.6) carry
both kinds simultaneously, which is why every prefix-reuse feature in §6 is
**strict-append only**: an append is safe for both state kinds; a rewind is
safe only for pure-attention models.

Both state kinds, plus sampler state and token history, serialize into the
**portable session snapshot** (`src/session/` format, `session/slot_snapshot`
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

### MoE routing replay (`RoutingSource`, 2026-09-16)

A MoE router's top-k is an `argmax`: a discrete choice with no margin. Perturb
the arithmetic by ~1e-4 — a different prefill shape, batch width, driver or
ggml build — and a different expert runs. That is why MoE receipts are
reproducible *per configuration* and not *across* them, while dense stacks under
the same perturbation are token-identical 15/15.

**The seam.** `moe_build_expert_idx()` (`src/layers/moe.h`) produces the top-k
index operand `ggml_mul_mat_id` already takes, from one of two sources:

- `RoutingSource::Router` — argsort the router logits, name it `moe_idx.<il>`.
  The default, and byte-identical to the behaviour that predates this.
- `RoutingSource::Replay` — an I32 graph input `moe_routing.<il>`, filled by
  `RoutingReplayInput` from a `RoutingTrace`.

Selected once by `DecodePolicy::routing_source()`; capture mirrors the attention
taps (`mark_moe_routing` → `alloc_readback_graph` → compute →
`read_moe_routing`, which refuses any other allocation). **Fixed 2026-09-27:**
capture used a plain alloc, and `mark_moe_routing` only flags nodes the graph
already has, so a captured prefill after a longer uncaptured one of the same
shape inherited a plan in which `moe_idx` was scratch — the recorded experts
were wrong in 40/40 layers (Qwen3.6-35B-A3B) and 29/30 (gemma-4-26B-A4B), with
no error (`RoutingCaptureAfterSameShapePassGetsItsOwnPlan`). On a server this
was the long-standing "first extract after start routes differently": the
start-up plan made its recorded trace wrong while its output was right (A/B:
digest `05841bf8…` vs `4bd3c2dc…`, identical after the fix;
`server_routing_replay_smoke.sh` gate 0).
**Only the discrete choice is pinned** — the router matmul still runs and the
gating weights are still gathered from the real logits at the replayed indices.

It is a FREE function, not a method, because Qwen's `MoELayer` and Gemma 4's
`build_moe_geglu` are structurally different layers (dual-FFN vs plain MoE,
GeGLU vs SwiGLU, shared expert vs none) that select experts with identical
lines. One place decides for both families.

**Measured** (warm prefill split, 15 documents EN+DE, last-position logits;
replay control bitwise-identical 15/15 on both, determinism control likewise):

| | unpinned worst | pinned worst | that family's dense band |
|---|---|---|---|
| Qwen 3.6-35B-A3B | 1.167e+00 | **1.447e-02** | 1.384e-02 (9B) |
| Gemma 4-26B-A4B | 2.620e+00 | **7.141e-01** | 1.054e+00 (12B) |

Both land in their own family's dense band — the regime that passes the drift
gate. Raw |Δlogit| is **not** comparable across families: Gemma's dense stack
drifts ~76× more than Qwen's under the identical perturbation, so a pinned arm
is judged against its own family's dense reference and never against a fixed
threshold or a reduction ratio.

**One non-obvious invariant.** On the Replay path `ggml_set_output(logits)` is
required. Removing the argsort removes the router logits' only node consumer —
the sole remaining reference is a reshape *view* — after which the allocator
reuses that buffer before the gating weights are gathered from it. The selection
is then perfectly correct and the weights are garbage, which is worse than not
pinning because it looks like it works. Qwen never exposed it; Gemma 4 did, at
22 logits with provably identical replayed ids. Cost: `n_experts × n_tokens × 4`
bytes per MoE layer that the Router path does not pay.

- **Ask launch-bound or math-bound first.** DeltaNet's cost (~29 ms/token,
  the biggest slice) was dozens of tiny op launches, not its matmul (8.4% of
  the layer). So the fusions target the small ops; the matmul stays native.
- **Two signals or it didn't happen.** The two shipped fused kernels
  (`deltanet_post_state`, `deltanet_pre_state`, ~+7% tok/s combined) had to
  show up in both per-step timing and end-to-end tok/s — Metal per-step
  numbers swing ±25%.
- **Read the dispatch source before the stopwatch — it predicts, and a
  wall-clock number only describes.** The 2026-08-30 batch-scaling sweep is the
  doctrine's own counterexample and its best instrument at once. *Amdahl before
  code* and *two signals* both held — nothing below was called on a single
  reading. What was new is that **every discontinuity in the curves was located
  in `ggml-metal-ops.cpp` and predicted before it was measured**, then found:
  the `mul_mv_ext` gate at `ne11 ∈ [4,8]` for K-quants predicted that **B=4
  would be absolutely cheaper than B=3** on Q4_K_M (measured 137.4 vs 143.8 ms,
  3/3 passes, against 2.2% spread); the kernel's rows-per-threadgroup table
  (`…5→5, 6→3…`) predicted a free 3rd lane and a cliff at B=5→6 (measured
  108.2→106.7, then **+46.3 ms for one lane**); `ne11_mm_min = 8` predicted the
  marginal collapse above B=8 (measured ~2 ms/lane from B=10). A source-read
  that names *where* the step will be is a stronger instrument than a
  wall-clock number that only says *how big* — and it is what turned an
  ambiguous null into a diagnosis: the July `QINF_BATCH_IDENTICAL` control read
  as "MoE decode is not weight-read bound" was in fact `mul_mv_id`'s
  per-*(expert, token)* grid being structurally unable to share a weight load,
  confirmed by the same control running as a clean no-op on a dense recipe.
  **The corollary is the inherited-number rule:** `ms/step ≈ 20 + 25.6·B` and
  its "1.76× ceiling" were a Qwen 3.6 measurement propagated into six documents
  as a cross-family constant — by a probe that was hard-gated to `qwen35moe`
  and *could not have taken the comparison*. Re-measured, the same recipe fits
  `12.3 + 21.0·B` and its ceiling **fell** to 1.59×, while a dense recipe
  reaches 9.60× at B=32. Point-in-time numbers get a named model, not just a
  named setup. Provenance:
  [`note-batch-scaling-cross-family.md`](note-batch-scaling-cross-family.md).
- **Deleting is optimizing.** TurboQuant/SnapKV removed (wrong constraint for
  the envelope); norm fusion never attempted (1.7% ceiling); conv fusion
  deferred below the agreed µs bar.
- **The head is skippable.** The sparse output head (§8) and prefill-head
  slicing are explicit caller switches, never silent engine choices.
- **Kill per-step overhead, not per-step math.** The persistent decode graph
  (§5) attacks the ~12 ms/step galloc replan `decode_breakdown` localized —
  not the compute. Measured **1.32× decode on Qwen 3.6 35B-A3B Q2_K (20 → 27
  tok/s), stable across 3 runs**, matching the standalone probe's 1.28×
  per-step prediction (two signals). Provenance:
  [`plan-persistent-decode-graph.md`](plan-persistent-decode-graph.md),
  [`note-decode-overhead-probes.md`](note-decode-overhead-probes.md). It is
  opt-in because the enabling bucketing is token-stable-not-byte-identical
  (§11) — a case where a measured win deliberately did NOT become the default.

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
  to resend history, and **a failed graph compute stops the pass** rather than
  letting the caller read an uncomputed buffer (`engine/graph_compute.h`).
  That last one was absent from the text path until 2026-08-29 — ggml-metal
  returns `GGML_STATUS_FAILED` on a command-buffer failure (usually GPU OOM)
  and latches it, and every text-path site discarded the status, so the engine
  decoded fluent nonsense and once caused a misdiagnosis. The vision encoders
  had always checked. Detection belongs to the engine; **containment belongs to
  the caller** — the server fails that batch's requests and keeps serving (its
  inference loop is a bare `std::thread`, so an escaping throw would kill the
  process), the CLI reports and exits non-zero.
- **The cross-family rule keeps abstractions honest.** An interface validated
  only on Qwen is presumed Qwen-shaped until a Gemma recipe proves otherwise.
  The success metric for hosting a new variant: zero logic edits to modules
  other recipes depend on — if it needed them, that's an interface defect to
  fix, not a feature to celebrate.
- **The receipts constraints (§1 identity — check before optimizing these
  paths).** (a) The attention module's **`kq_soft.<il>` tensor names are a
  public seam** — the lens tap locates rows by name; renaming is a breaking
  change (§13 trigger). (b) **Materialized decode attention is load-bearing on
  tapped layers**: any future fused/flash-style attention must keep tapped
  layers materialized or export their rows (decode is one query row × ≤10K
  keys, so fusion buys nothing there — the conflict is theoretical inside the
  envelope, named so it stays theoretical). (c) **Receipts-grade determinism
  is per-config AND single-slot**: the batch-shape fork (above) means a
  generation that ran batched cannot be byte-replayed without its batch;
  byte-replay claims (witnesses, counterfactual diffs) hold at B=1 — the lens
  path is single-slot for this reason too, not only the qwen36 gather bug
  (§12). **What that leaves for a customer is a DECISION claim, and since
  2026-09-13 it is measured rather than implied** (`lens-format.md`, honest
  limits): identical bits within a config, and across configs a report whose
  decisions are stable with measured room — **on the 9B**. The room there is
  1.5× (margin 0.00126 against 0.00084 of observed movement, EN+DE; 25× on the
  flattering English-only arm). **On the 35B the same gate FAILS**: drift 0.0242
  against a 0.0237 margin, and a chunked-prefill control changes one extraction
  in fifteen. Expert routing is a top-k argmax, so an MoE hybrid does not
  perturb smoothly and the cross-configuration claim is not currently available
  on the model the lens was calibrated on (`lens-format.md`, honest limits).
  Which language binds is model-dependent too: German on the 9B, English on the
  35B. Byte-identity was always both too strong (it forbids changes that
  provably move no decision — phase-scoped flash, §5) and too weak (it does not
  survive a driver, a GPU or a ggml bump, none of which this repo pins). The
  drift gate is where a candidate change earns the claim:
  `tests/perf/attn_provenance.cpp`, `BANDDRIFT=1 DRIFT_ARM=<arm>`, exit code 0
  = PASS. **KV element type is part of "config"**: an F16-cache generation
  replays byte-identically only under F16, and the lens calibration numbers
  were measured under F32, so F16 is not a calibrated receipts path until
  re-measured. This is why `--kv-f16` is opt-in and F32 stays the default. (d) **Nondeterministic kernels are inadmissible on the receipts
  path** — a Metal kernel using atomics/async reduction ordering may be fast,
  but it forfeits every replay claim; it needs an explicit decision, not a
  benchmark win.

---

## 12. Known soft spots (honest ledger)

Current, verified against the tree at time of writing:

- **`qwen35moe` aborts at B=16 on a node-count assert.**
  `GGML_ASSERT(cgraph->n_nodes < cgraph->size)` fires in `build_deltanet_layer`,
  reached from `Qwen36ForwardPass::build_decoding_graph`. Cause is structural
  and known: `DeltaNetLayer::build_decode` builds a **full DeltaNet chain per
  slot** and concatenates, so the decode graph's node count is **O(B)** —
  it overflows the preallocated graph somewhere in `8 < B < 16`. Pre-existing
  and **just outside the declared ≤10-slot envelope**, which is why it has never
  been hit in production; found 2026-08-30 while sweeping batch sizes past the
  envelope
  ([`note-batch-scaling-cross-family.md`](note-batch-scaling-cross-family.md)
  §7). It is a hard abort, not a degradation, so the failure mode is at least
  loud. Two things to know before relying on it staying benign: the envelope's
  own ceiling (10) is close enough to the failure band that a modest raise
  crosses it, and the fix and the batching work are the same fix — batching the
  per-slot loop removes the O(B) node growth along with the O(B) dispatches.

  > **CORRECTED 2026-08-31 — the crossing point above is wrong, and it was
  > wrong in the more dangerous direction (too optimistic).** Node-count
  > census (`GGML_METAL_GRAPH_DEBUG=1`, exact integers, not wall-clock):
  > `n_nodes = 1320·B + 2144` on `qwen35moe` (30 DeltaNet + 10 attention
  > layers), exact for B≥2. **Confirmed directly: B=10 builds (15344 of 16384
  > nodes — 94% of the limit), B=11 aborts.** That is **one slot of margin**
  > against the declared ≤10-slot envelope, not "just outside" it. The
  > mechanism is unchanged: each DeltaNet layer contributes exactly 44 graph
  > nodes per slot (≈14.3 `VIEW`, 10 `RESHAPE`, 5 `MUL_MAT`, 2 `CONCAT`, ~2.7
  > `CPY`, plus one each of `DELTANET_PRE_STATE`, `DELTANET_POST_STATE`,
  > `GATED_DELTA_NET`, `SSM_CONV`, `TRANSPOSE`, `ADD`, `MUL`, `SIGMOID`,
  > `SILU`, `SOFTPLUS`); `MUL_MAT_ID` stays exactly constant in B (120 nodes),
  > confirming the CLAUDE.md claim that MoE dispatch is O(1) in expert count
  > and batch.
  >
  > **`qwen35` (the dense hybrid) has the same defect, previously
  > unrecorded.** `n_nodes = 1056·B + 596` (24 DeltaNet + 8 attention layers),
  > exact for B≥2. **Confirmed directly: B=14 builds (15380 nodes), B=15
  > aborts.** Five slots of margin above the ≤10 envelope, unlike qwen36's one.
  >
  > **The crash itself is fixed, the node-growth defect is not.** As of
  > 2026-08-31, `create_forward_pass` refuses an over-limit `max_batch_size`
  > fail-loud (`validate_deltanet_decode_batch_size`,
  > `src/models/qwen35_family.{h,cpp}`, called from both `Qwen35ForwardPass`
  > and `Qwen36ForwardPass`'s constructors) — before any graph is built or
  > state allocated, naming the parameter, the derived limit, and the actual
  > value. The underlying O(B) growth this bullet describes is untouched: the
  > batching fix that removes it is scoped in
  > [`plan-deltanet-batched-decode.md`](plan-deltanet-batched-decode.md), which
  > is **PARKED by user decision (2026-08-31)** — the remaining work is ggml
  > kernel engineering needing a dedicated measurement bench, for a payoff
  > (concurrent-user throughput) with no present demand. Provenance for both
  > formulas:
  > [`note-batch-scaling-cross-family.md`](note-batch-scaling-cross-family.md).
- **`--kv-f16` on Gemma 4 MoE is unexplained and ungated.** F32→F16 shifts the
  step-0 top-1 logit by 0.93 on `gemma-4-26B-A4B-it-Q2_K` (later steps drift
  0.06–0.3), against 0.0007–0.0387 on every other recipe including Gemma 4
  *dense* at Q8_0 and Qwen 3.6 MoE. Ruled out by measurement: it is not
  nondeterminism (F32-vs-F32 and F16-vs-F16 are both bit-identical), not the
  Gemma 4 recipe's KV plumbing (dense is 0.0387), not MoE as such (Qwen 3.6 is
  0.0021), and not nominal quant level (both are `file_type=10`, and the Gemma
  checkpoint carries *more* bits/param). Greedy tokens still matched at 4
  steps, but that is thin evidence on a checkpoint whose output is already
  incoherent on the probe prompt. Treat Gemma 4 MoE + `--kv-f16` as unvalidated
  until the amplification is explained.
  **A candidate mechanism appeared 2026-09-02 and has NOT been tested against
  this bullet:** single-slot decode was reading a non-F32 cache with a hardcoded
  float stride (§9), so every `--kv-f16` figure taken before that fix — the ones
  above included — was measured on a corrupted read. Re-measure before treating
  any of them as real; the anomaly may simply dissolve. A hypothesis, not a
  finding.
- **`forward_pass_base` is being shrunk to a cohesive core — not deleted.** The
  blueprint's direction is composition-over-inheritance, and this was recorded as
  an eventual deletion target until the primitives were actually measured
  (below), which does not support that. The **first extraction landed
  2026-08-29**: the ggml context and its metadata
  buffer are now a `GraphArena` the base *holds* rather than *is*
  (`models/graph_arena.h`, unit-tested without a model or backend). Recipes
  reach it as `arena_.ctx()`. The **second extraction landed the same day**: the
  run-time policy flags — prefill head slice, hidden-state output, attention
  taps, KV write mode, decode n_kv bucket — are now a `DecodePolicy` value the
  base holds (`models/decode_policy.h`), the base's accessors delegating so no
  caller changed. Its defaults ARE the byte-reproducible path, which is the
  precondition §11's receipts claims rest on; `is_default_byte_reproducible()`
  makes that assertable, and `decode_kv_len`'s bucketing — including its
  cap-at-`n_ctx_max` edge — is now unit-tested without a model.
  **The graph primitives were then measured, and the "delete the base class"
  framing needs revising.** They are three different things, not one:
  (a) four already-pure helpers (`set_tensor_name`, the three `get_output_*`) —
  extractable, but they are the caller-facing interface and moving them would be
  ~46 call sites of churn for no structural gain;
  (b) thin wrappers over layer modules — `build_attn_mha` was one and had
  **zero callers** despite a comment claiming qwen35 used it (deleted
  2026-08-29); `build_norm`/`embedding` stay, they save 22 and 14 call sites
  from repeating `meta_.rms_norm_eps` and the token-embedding lookup;
  (c) `build_output_head`, `build_out_ids_slice`, `build_image_substitution`,
  `build_decode_layer_masks` — each builds graph nodes AND registers the typed
  input those nodes consume (`SparseHeadInput`, `OutputIdsInput`,
  `ImageEmbeddingsInput`, `AttnMaskInput`).
  That coupling in (c) is **correct, not accidental**: creating the node and its
  input together is what prevents "node built but input never filled" — which is
  precisely the qwen36 vision bug (§7). Pulling them out as free functions would
  mean threading `meta_`, `model_`, the arena, `graph_inputs_`, `policy_`,
  `sparse_decode_ids_` and `image_spliced_` through 5-6 parameters each, i.e.
  trading a cohesive class for the fat-parameter smell CLAUDE.md warns about.
  So the honest end state is a SMALL base class, not none. Per-step arming
  (`sparse_decode_ids_`, the rope-divergence record) is the remaining candidate
  to move; the (c) group should stay.
- **The attention free functions sit at two altitudes under one naming scheme.**
  `build_attention`/`build_batched_attention` take already-projected Q/K/V (an
  attention *core*); `build_gated_attention`/`build_gated_batched_attention` take
  the normed residual plus six weight tensors and project internally (a whole
  *layer*, 24 parameters at the decode variant). The prefill/decode split is
  legitimate — different graph topology, forced by §3's one-topology-per-graph
  rule. The plain/gated split is not the same kind of thing. Deliberately NOT
  renamed: a rename would make the confusion less visible without resolving it,
  and the fix is to unify the altitude, which is a redesign needing its own plan.
  The header now states the split explicitly.
- **Qwen 3.5 and 3.6 share one config and one layer body (2026-08-29), but are
  still two recipe classes.** They differ in exactly one call — dense SwiGLU vs
  routed experts — so the FFN is now a PARAMETER (`Qwen35Config::is_moe()`,
  `Qwen35LayerCommon::moe_hp`), settling the inconsistency with Gemma 4, which
  had always parameterized its own dense/MoE split. `models/qwen35_family.h`
  holds the shared body; `Qwen35MoEConfig` is an alias of `Qwen35Config`. The
  duplication was not theoretical: 11 of the 20 most recent commits touching
  either recipe had to touch both, and it produced the `Stride::NKvLen` gather
  defect (wrong in qwen36, right in qwen35, latent for months).
  The typed-input declarations are shared too (`register_qwen35_*_inputs`):
  neither recipe calls `graph_inputs_.add` any more. That is the block the
  gather defect actually lived in, so the defect class is now structurally
  impossible here — one declaration site, one stride, pinned by
  `test_qwen35_family` including a test that the dense and MoE hybrids declare
  identical decode inputs.
  **Collapsing the two recipe classes was considered and deliberately rejected.**
  What is left is not duplication: the image splice, the MoE hparam wiring and
  the NextN head-out genuinely differ, so a merged class would branch internally
  rather than share — trading CLAUDE.md's "model zoo" failure mode for its "fat
  function of orthogonal knobs" one. The MTP head (`IMtpDraftable`, qwen36 only,
  ~240 lines) would also make a merged class implement a capability
  conditionally. Two clearly-named classes over a shared config, layer body and
  input set is the better side of that judgment.

- **`server/inference_server.h` is a 1210-line header-only class.** Past what a
  header should carry, but header-only on purpose: it is what lets the slot and
  queue logic be unit-tested against fake engines with no model, which is a real
  design win. Recorded as a known shape, not a defect.
- The chat endpoint flattens engine finish reasons (`timeout`, `cancelled`,
  `error`) to OpenAI's `"stop"` — the completions endpoint reports honestly;
  the chat path lies by enum-compat (`chat_finish_reason` in
  `http_server.cpp`).
- Thinking-model token budgets: the thought channel spends `max_tokens`, so a
  visible answer can be cut with a clean `"length"` — no
  `max_completion_tokens` split yet.
- Grammar engine: two documented too-permissive bugs (§8).
- **Namespacing is half-unified.** The `qwenium` root was collapsed into `qinf`
  on 2026-08-29, so the two-root split is gone (`qwenium` now survives only as
  the product name — the binary, and the `qwenium_version` field serialized into
  every snapshot header, which must NOT be renamed). What remains is `qinf` with
  per-subsystem sub-namespaces where a subsystem is self-contained
  (`qinf::vision`, `qinf::session`, `qinf::image`, `qinf::engine`) alongside a
  large body of core code — `layers/`, `models/`, `graph_inputs/`, `loader/`,
  most of `state/` — still in the GLOBAL namespace. The blueprint asks for
  `qinf::layers` / `qinf::models` / `qinf::state`; getting there is a whole-tree
  mechanical change with little functional payoff, so it is recorded rather than
  scheduled.
- Vision: the strict numeric encoder differentials vs llama.cpp are
  `DISABLED_` (coarse gates + coherence smokes stand in); Gemma 4 image turns
  can emit a short degenerate prefix before recovering.
- The conversational-server gate lacks its Gemma 4 (pure-attention thinking)
  leg; recover responses reuse generic HTTP statuses rather than the
  documented 409.
- No CORS/auth on the server — it is local-oriented by design, but that makes
  browser front ends a P2.
- The KV cache has two write paths: baked-offset `ggml_cpy` (prefill, and
  default decode) and value-driven `ggml_set_rows` (opt-in `--persistent-graph`
  decode only — the write row an input, so the graph can be reused;
  [`plan-persistent-decode-graph.md`](plan-persistent-decode-graph.md)).
  Byte-identical at exact width by gate (`test_kv_write_setrows`). The set_rows
  path is exercised only under the flag; unify (retire cpy on the decode side)
  once the persistent path is the default — which awaits a decision, since
  bucketing makes it token-stable-not-identical (a deliberate opt-in, §5/§11),
  not a soft spot to silently fix.
- The Qemmi-Lens attention tap (`forward_pass_base`
  `set_attention_taps`/`mark_attention_taps`/`get_attention_taps`,
  [`plan-qemmi-lens.md`](plan-qemmi-lens.md) P1/A1) reads the frozen
  `kq_soft.<il>` rows on the qwen36 decode path. V1 serves single-slot — now by
  choice (receipts-grade determinism is B=1, §11) rather than because the gather
  was broken: qwen36's decode KV gather used `Stride::NKvLen`
  (`slot*n_kv_len + t`) against `gather_k`'s `n_ctx_max`-strided flat layout,
  correct only for slot 0. **Fixed 2026-08-29** — it now uses the cache's
  `n_ctx_max` stride like qwen35 and gemma3, and the second stride policy was
  deleted outright so there is no wrong one left to select
  (`test_gather_indices_input` pins the multi-slot rows, and asserts the slot-0
  identity that let the defect stay latent). The tap seam remains opt-in and
  byte-inert when disarmed (default empty layer set marks no node — same
  liveness-only argument as `set_output_hidden`; gated by
  `test_forward_pass_base` `TapOffByteIdentical`) and recipe-agnostic (the tensor
  name is the seam, so any recipe naming `kq_soft` hosts it). No lens *claims*
  for Gemma — and as of 2026-09-04 that is a **settled measurement, not an
  unprobed gap**: 0 of 768 candidate heads clear a 70% bar (§6). The seam hosts
  the probe on any recipe; only Qwen models have a calibration entry.
  **As of Movement 1 (`/v1/verify`, §6) `get_attention_taps` also reads a
  tapped PREFILL block** — `kq_soft.<il>` at `[n_kv, n_q, n_head]`, one row per
  query position processed in that pass, vs decode's `n_q==1` — via the same
  reader (`AttentionTap::n_q`, sized off the tensor's own `ne[1]`), so decode's
  shape and byte layout are unchanged. Paired with prefill truncation
  (`DecodePolicy::truncate_after_layer`), also opt-in and byte-inert (default
  `-1` = full stack) and honored by every recipe (Qwen and Gemma alike) as a
  plain layer-loop bound, not a per-recipe kernel capability.
  **Head-selected tap (2026-09-25).** `set_attention_taps(layers, heads)` —
  one call, so arming taps without heads resets the list rather than
  inheriting it (`DecodePolicy::attention_tap_heads`, default empty = every
  head, today's tap). With a list, `mark_attention_taps` leaves `kq_soft.<il>`
  an ordinary intermediate and copies each selected head's contiguous
  `[n_kv, n_q]` block into its own output `kq_tap.<il>.<h>`
  (`ggml_view_3d` + `ggml_cont`); `get_attention_taps` returns them as
  consecutive blocks with `AttentionTap::heads` (and `block_of(h)`) saying
  which model head each block is. Why: a lens job reads one head, and the full
  tap is `16 x P x P` floats — 6.2 GB of host copy and 5.8 s of readback at a
  10K prompt ([`note-lens-locate-baseline.md`](note-lens-locate-baseline.md)).
  The copies are the same softmax, so values are byte-identical to the full
  tap's (`HeadSelectedTapEqualsFullTap{Decode,Prefill}`, run per recipe, Qwen
  and Gemma); `/v1/locate` is the one caller that sets it (`{use_head}`);
  extract, verify and the probes keep the full tap. It does NOT remove the
  materialized `kq_soft` itself — that transient still costs the GPU compute
  buffer (8–9 GB at 10K); only a split pass with flash on the document does
  (note-lens-prefill-only-engine.md).
  **Fresh plan per read-back pass (2026-09-27).** The seam's sequence is now
  build → `mark_attention_taps(gf)` → **`alloc_readback_graph(sched, gf)`** →
  inputs → compute → `get_attention_taps(gf)`, and it is enforced:
  `get_attention_taps` refuses a graph that was not marked and allocated
  through `alloc_readback_graph` (a small state machine per reader, Marked →
  Planned → consumed by the one read). MoE routing capture uses the same
  helper and the same refusal (§10, the routing seam). What was MARKED on the
  graph decides — a prefill with taps armed but unmarked (run_prefill before a
  tapped decode) stays on the plain alloc. Why: `ggml_gallocr_needs_realloc` reuses the
  previous plan whenever node count matches and every node fits, and never
  compares output flags, so a tapped graph shaped like an earlier one on the
  same scheduler inherits a plan in which its tap is ordinary scratch. With
  head-selected taps two graphs of the same truncation tapping DIFFERENT
  layers have equal node counts; the recycled slot then holds a later layer's
  `kq_soft` — same size, also in `[0, 1]` — so the range check cannot see it:
  a silent wrong receipt. Reproduced 2026-09-27 on all four recipes
  (`SameShapeDifferentTapLayerGetsItsOwnPlan`: qwen35/qwen3 silently wrong,
  gemma3/gemma4 caught by the range check). How: the helper first reserves a
  one-node graph (built and freed per call), which replaces the cached plan
  without shrinking any buffer, so the tapped graph's single alloc plans
  afresh from its own output flags. Not `ggml_backend_sched_reserve(gf)` +
  alloc(gf): each splits `gf`, and the split rewrites `node->src[j]` to
  per-backend input copies that the second split frees (measured:
  `TapOffByteIdentical` logits off by up to 23). Cost within noise on every
  lens verb (9B Q4_K_M, full server); every lens report byte-identical
  (LENSDUMP, 95 reports). With nothing marked the helper is the plain
  reset + alloc. The `[0, 1]` range check stays as a backstop.

- **Qwen 3.5-family vision is gated end-to-end by coherence smokes, not by an
  automated test** — but the two links most likely to fail quietly are now
  pinned separately. **Preprocessing is in `tests/`** as of P5:
  `image-loader-tests::MatchesLlamaCppQwen3VlReference{,Upscaled}` compares the
  whole `Bitmap` against `mtmd_image_preprocessor_dyn_size`, bit-exact, with one
  fixture per branch of smart_resize. The ViT (`qwen3vl_encoder`) has a numeric
  reference — captured encoder-only via `clip_init`/`clip_image_encode` against
  the vendored mtmd source, cosine 0.999875 whole-block and 0.9999 per-token at
  two sizes and both grid parities — but **that** differential still lives in a
  scratch harness, not in `tests/`. End to end, both Qwen recipes are verified
  only by manual smokes on single images — CLI, and (P6) three consecutive
  `/v1/chat/completions` image requests that must come back grounded AND
  byte-identical to each other. The per-slot rope bookkeeping those smokes
  exercise *is* gated automatically and model-free
  (`tests/unit/test_rope_divergence.cpp`). See `plan-qwen35-vision-impl.md`
  §6 (P5, P6) and §8.6.
- **VL sessions are snapshottable; the image-prefix caches are not M-RoPE-safe
  yet.** An M-RoPE image span occupies nx·ny KV rows while advancing the
  sequence position by only max(nx, ny). Since 2026-10-01 a snapshot carries
  that rope coordinate: `capture_slot` appends an **`RPOS` section** (the slot's
  `RopeDivergence{delta, rows_after}`, read through
  `ForwardPassBase::rope_record` / `set_rope_record`) **only for a diverged
  slot**, as the blob's last section, and `restore_slot` re-installs it when the
  blob's section count says it is there — so text blobs and Gemma image blobs
  are byte-identical to before (measured), and a restored Qwen image slot
  resumes bit-identically (`test-image-prefix-roundtrip` on the 35B-A3B: 389
  rows, rope position 29, logits max diff 0). The CLI `--image-prefix-cache`
  and the server's V2 cache still refuse an M-RoPE recipe at setup: they place
  the question by KV rows; porting them to `get_rope_pos` is a separate
  decision (`docs/plan-image-verdict.md` §4).

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
