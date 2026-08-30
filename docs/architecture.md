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
  decode graph (`--persistent-graph`, §5), opt-in flash attention on decode
  (`--flash-attn`, §5 — mutually exclusive with the attention lens), and
  session snapshots;
- an **HTTP server**: OpenAI-compatible `/v1/completions` and
  `/v1/chat/completions`, serving up to ~10 concurrent requests by batching
  them into one forward pass.

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
| `src/sampling/` | Decode-time algorithms | `sampling` (greedy/temperature+top-k/top-p/rep-penalty, sparse variants), `grammar_vocab` (GBNF engine, §8), `token-trie` (candidate narrowing), `speculative` + `draft_source` (draft-source seam: `IDraftSource`) + `prompt_lookup` (PLD), `sampling_snapshot` |
| `src/loader/` | GGUF → live model | `gguf_loader` (mmap + metadata, and the fail-loud architecture/tensor-inventory validators), `tokenizer`, `chat_template` (per-family prompt rendering), `channel_filter` (Gemma 4 thought/answer channel split), `multimodal_check`, `gguf_value` (generic GGUF scalar/array KV bag), `platform` (mmap wrapper) |
| `src/engine/` | The loaded model, and the orchestration of one step over it | `model` (owns weights/backend/scheduler; the load path), `decode_plan`/`decode_step` (batched decode orchestration), `decode_graph_cache` (opt-in persistent decode graph — reuse one built+allocated graph across steps on a dedicated scheduler, §5), `multimodal_prefill`, `graph_compute` (the one place a compute status is checked — fail-loud on backend failure) |
| `src/vision/` | Image → soft tokens (§7) | `i_vision_encoder` (Seam A), `siglip_encoder` (Gemma 3, 27-layer ViT), `gemma4uv_encoder` (Gemma 4, blockless), `qwen3vl_encoder` (Qwen 3.5 family, ViT + 2×2 merger, in-ViT M-RoPE), `vision_profile` (projector → encoder+recipe dispatch), `image_preprocess` (preprocessing recipes), `vision_loader` (3 projectors: `gemma3`, `gemma4uv`, `qwen3vl_merger`), `vision_model`, `bitmap` |
| `src/session/` | Persisting and reusing session state | The **format**: `snapshot_io`, `session_manifest`, `compat_header`, `section_ids` — versioned, sectioned, fail-loud on mismatch (built as `qinf-session`, deliberately dependency-free so it unit-tests in isolation). The **services** on top of it: `slot_snapshot` (extract/restore a slot), `prefix_library` (disk warm-KV blobs, hash-keyed, version-gated), `image_embedding_cache` + `persistent_image_embedding_store`. The two services that need `models/`/`graph_inputs/` build into `qinf-engine` or directly into consumers rather than into `qinf-session` — directory is the concept, target is the layering (see `session/CMakeLists.txt`). |
| `src/server/` | HTTP serving (§6) | `inference_server.h` (slots, queues, batching, warm paths — the engine-agnostic core), `http_server.cpp` (endpoints, SSE, OpenAI mapping), `server_vision`, `server_lens` (opt-in `--attention-lens` `/v1/extract`: document → audited key-value JSON on the attention trust layer; pure lens computation + single-slot tapped-decode driver), `image_data_uri` |
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
loading against 2× and serving against 1×. Removing the peak means removing
the copy — backing the backend buffer with the mmap'd pages — which is not
done here.

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

**Flash attention** (CLI/server `--flash-attn`, `DecodePolicy::AttnImpl`): on
**both prefill and decode**, one `ggml_flash_attn_ext` replaces the whole
`kq` → `soft_max` → `kqv` chain *and* the V transpose — four Metal dispatches
per attention layer become one. On decode the recipe casts its mask to F16 once
per graph (Gemma 2/3/4 dedupe by window first); on prefill the mask is built per
layer inside the attention helper, so the cast lives there. `build_attn_mha`
refuses an F32 mask, naming the layer, and forwards softcap — ggml applies the
scale before the tanh clamp, our convention. **Prefill is where it pays most**:
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

**Flash attention and the receipts identity are mutually exclusive**, and this
is enforced, not documented-and-hoped: the flash kernel never materializes
`kq_soft`, which is precisely the tensor the attention lens taps (§1, §11).
`DecodePolicy::is_attn_impl_coherent()` states the pairing, and the server
refuses `--flash-attn` together with `--attention-lens` at startup. This is the
first place where a speed lever and the receipts doctrine are in direct
conflict; the resolution is a parameter on one operation with two
implementations (llama's own `-fa on/off` split), not two attention modules.

**Speculative decoding** (CLI `--speculative [pld|mtp]`): drafts come from an
`IDraftSource` (`sampling/draft_source.h`) and are verified in one batched
pass (head slice off — verification needs logits at every draft position); a
first-token guard ensures draft[0] matches the token the sampler actually
chose. On mismatch the KV cache truncates (O(1) pointer move) and, on
hybrids, the recurrent state restores a pre-verify checkpoint and the
accepted prefix is re-fed (`feed_tokens`) — overwrite semantics can't rewind
(§9). Two draft sources: **PLD** (bare `--speculative`; no model — the draft
is an n-gram match of recent output against the prompt) and the **MTP head**
(`--speculative mtp`, depth `--mtp-max-draft`; Qwen 3.6 NextN: an extra
trained attention+MoE block held out of the main stack, drafting recursively
from the last position's hidden state via `models/i_mtp_draftable.h` on a
private KV + dedicated scheduler; the hidden is exposed by an opt-in,
default-off graph output). MTP is a capability of MTP-converted GGUFs,
mirroring how vision is a capability of `--mmproj` — Qwen-only, as vision is
Gemma-only. Status: **experimental** — 74–92% acceptance, ~3.3 tokens/step,
but end-to-end ≈ baseline on M1 Pro until the per-head-step dispatch overhead
is attacked; measurements and the five speculative-machinery bugs fixed en
route live in `plan-mtp-decode.md` §7/§9. Emitted tokens are model-verified
under the kernel path that computed them; batch-shape numerical forks (§11)
mean speculative-on is token-stable, not byte-identical, vs speculative-off.

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

**Attention Lens (opt-in `--attention-lens`, `POST /v1/extract`).** A dedicated
endpoint — separate from the OpenAI surface by design (its inputs are a
document + a complete key vocabulary of `{key, gloss}` concepts, not chat
messages; its output is the lens format, not a completion). It runs **one free
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

**Single-slot and exclusive**: `extract_lens_json` holds the model lock for the
whole tapped decode and uses slot 0 (the only correct qwen36 decode KV gather,
§12); do not drive concurrent OpenAI traffic on slot 0 while extracting.
Qwen3.6-pinned constants; fail-loud on empty concepts or an oversized document.
Off ⇒ the route 404s and no lens code runs.
[`plan-qemmi-lens.md`](plan-qemmi-lens.md), [`lens-format.md`](lens-format.md),
[`note-nogrammar-refutation.md`](note-nogrammar-refutation.md),
[`note-lens-absent-attempt.md`](note-lens-absent-attempt.md); gates
`tests/smoke/server_extract_smoke.sh` + `QDOCS_S1=1 bin/attn-provenance`.

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
| Element type | F32, or F16 via `--kv-f16` | always F32 |

**KV element type.** `simple_kv_cache` takes `type_k`/`type_v`; every recipe
passes what `create_forward_pass` was given, defaulting to **F32** (the
historical, byte-identical behaviour). `--kv-f16` halves KV bytes and is
token-stable but *not* byte-identical, so it carries the same status as
`--persistent-graph` (§10). Recurrent state is unaffected — always F32.
Measured on both families at ctx 4096: Qwen3.5-0.8B 96 -> 48 MB, Gemma 3 1B
208 -> 104 MB, Qwen 3.6 160 -> 80 MB, with the greedy token sequence and the
top-5 ordering unchanged. Attention reads the cache through views whose
strides are derived from the tensor's own type, never `sizeof(float)`;
hardcoding the stride silently mis-reads a non-F32 cache instead of failing.
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
  (§12). **KV element type is part of "config"**: an F16-cache generation
  replays byte-identically only under F16, and the lens calibration numbers
  were measured under F32, so F16 is not a calibrated receipts path until
  re-measured. This is why `--kv-f16` is opt-in and F32 stays the default. (d) **Nondeterministic kernels are inadmissible on the receipts
  path** — a Metal kernel using atomics/async reduction ordering may be fast,
  but it forfeits every replay claim; it needs an explicit decision, not a
  benchmark win.

---

## 12. Known soft spots (honest ledger)

Current, verified against the tree at time of writing:

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
  name is the seam, so any recipe naming `kq_soft` hosts it — no lens *claims*
  for Gemma yet, its constants are unprobed).

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
- **VL sessions are not snapshottable or prefix-cacheable.** An M-RoPE image
  span occupies nx·ny KV rows while advancing the sequence position by only
  max(nx, ny), and the snapshot blob records a row count with no rope
  coordinate — so such a slot cannot be round-tripped. `capture_slot` refuses a
  slot with a recorded divergence, and the CLI refuses `--image-prefix-cache` on
  an M-RoPE recipe at setup. Gemma (one position per row) is unaffected.
  Lifting this needs the snapshot header bump described in
  `plan-qwen35-vision-impl.md` §4 decision 3.

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
