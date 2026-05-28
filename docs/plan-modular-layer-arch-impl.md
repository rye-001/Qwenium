# Implementation Plan

Executable plan for the refactor specified in
[`modular-layer-architecture.md`](modular-layer-architecture.md).

This document does **not** re-litigate architecture. The spec is locked.
The job of this plan is to make each phase of the spec executable: PR
breakdown, test gates, branch strategy, migration mechanics, and the
TDD discipline that binds them.

---

## Working Discipline

### TDD cycle (applies to every PR in Phases 2–5)

For each unit of work:

1. **Red.** Write the test against the spec's oracle hierarchy (see
   `modular-layer-architecture.md` → Testing Strategy). The test fails
   because the code doesn't exist or doesn't conform yet.
2. **Green.** Write the minimum code needed to make the test pass. No
   extra fields, no speculative abstractions, no "while I'm here"
   cleanup.
3. **Refactor.** With the test green, improve structure. Tests stay
   green throughout.
4. **Commit.** Red → Green → Refactor is one commit boundary at most,
   often three. Whichever gives a reviewer the cleanest history.

**Phase 1 is the one exception.** Mechanical extraction is not TDD — no
new tests are written, because the behavior is unchanged by construction.
The "test" is the existing integration suite passing bit-for-bit. Phase 1
discipline is revert-safe extraction, not red/green/refactor.

### Test gates per phase

| Phase | Gate |
|---|---|
| 1 | Bit-for-bit identical logits on canary prompts. Full existing test suite passes (unit, integration, CLI, server). |
| 2 | Phase 1 gate still holds. New per-module unit tests pass against their declared oracle with declared tolerance. No output change on canary prompts. |
| 3 | Qwen 3.6 produces logit-level agreement with llama.cpp on canary prompts (top-8 rank preservation, top-1 divergence index ≥128). Qwen3 tests from Phases 1–2 unchanged. |
| 4 | Identical output to Phase 3 on canary prompts. Measured speedup on targeted kernels. No regression on other layer types. |
| 5 | Perplexity within declared bounds on a held-out text corpus. Quality regression budget is explicit per PR. |

### Branch strategy

- One long-lived branch per phase: `phase-1-extraction`, `phase-2-interface`,
  `phase-3-new-layers`, `phase-4-kernels`, `phase-5-quant`.
- PRs land into the phase branch, not into `main`.
- The phase branch merges into `main` only when the phase gate is met.
- `main` stays shippable throughout. Phase-branch work is visible but
  isolated.

### Rollback

- Every PR must be revertable as a single `git revert` without collateral.
- Phase branches are squash-merged to `main` only after the gate is met,
  giving a single revert point per phase if a latent issue surfaces later.
- No PR is allowed to land that requires a follow-up PR to be revertable.

### Canary prompt set

- Location: `tests/fixtures/canary_prompts.txt` (see spec's Testing
  Strategy section).
- The canary set is versioned. Changes to the set require:
  - A deprecation window where both old and new prompts pass.
  - Justification in the PR description.
- Canary prompts cover: short prompt + long generation; long prompt +
  short generation; grammar-constrained output; multi-turn dialogue;
  code generation.

---

## Phase 1 — Mechanical Extraction

**Goal:** move code into the new tree without changing behavior.
**Duration estimate:** 1–2 weeks, one engineer.
**Branch:** `phase-1-extraction`.
**Gate:** bit-for-bit identical logits on Qwen3-32B against canary prompts;
full existing test suite passes.

### PRs

**PR 1.1 — Directory scaffolding.** Create empty `src/layers/`, `src/state/`,
`src/models/`, `src/loader/`, `src/quant/` directories with `.gitkeep`.
Update `CMakeLists.txt` to recognize them (sources still live in old
paths). Zero logic change. Merges green. `src/quant/` stays empty
through all of Phase 1 — K-Quants weight quantization does not move
yet; it migrates only when a concrete Phase 5 need drives it.

**PR 1.2 — Move KV cache + quantization into `src/state/`.** `git mv`
`src/kv-cache/compressed_kv_store` → `src/state/kv_cache_compressed`,
`src/kv-cache/simple-kv-cache` → `src/state/kv_cache_simple`,
Update `#include`
paths only. No code changes inside the moved files. **The compressed
and simple KV caches stay as two sibling files** — do not collapse or
absorb them during the move. Whether they should be unified behind a
single `KVCache` class with a runtime config, or kept as two concrete
implementations of a common interface, is a Phase 2 interface decision
driven by the test gate, not a Phase 1 extraction choice. Build system
updated accordingly. Gate: all existing tests pass unchanged.

**PR 1.3 — Move loader + tokenizer into `src/loader/`.** Same pattern
as 1.2 for `src/core/gguf-loader` and `src/core/tokenizer`.
Include-path updates only. Gate: build clean, tests pass.

**PR 1.4 — Extract attention block into `AttentionLayer`.** Create
`src/layers/attention.h` + `.cpp`. Move the attention subgraph construction
from `src/core/forward-pass.cpp` into a free function
`build_attention(ggml_context*, ggml_tensor* input, int layer_idx, ...)`.
The forward pass calls it in place of the inlined code. **No logic
changes — literal cut-paste with include adjustment.** Gate: bit-for-bit
logits unchanged on canary prompts.

**PR 1.5 — Extract FFN block into `FFNLayer`.** Same pattern as 1.4 for
the SwiGLU block. Gate: bit-for-bit.

**PR 1.6 — Extract model into `src/models/qwen3.cpp` Also extract model into `src/models/qwen35.cpp`.** Move the
per-layer orchestration loop (norm → attention → residual → norm →
FFN → residual) from the forward pass into a recipe file. The existing
entry point dispatches to it directly (no factory indirection). Gate:
bit-for-bit.

**PR 1.7 — Delete the old dispatch scaffolding.** Remove
`src/core/forward-pass.cpp`, `src/core/forward-pass-base.*`,
and `src/core/forward-pass-factory.*` — these are the old
dispatch abstraction that the model recipe (PR 1.6) and Phase 2's
`REGISTER_MODEL` replace. Do not preserve them as shims or carry them
into the new tree; that defeats the extraction's purpose. Gate:
bit-for-bit logits, server and CLI integration tests unchanged.

### Phase 1 exit criteria

- [ ] Canary prompts produce bit-for-bit identical logits.
- [ ] Unit, integration, CLI, and server test suites pass unchanged.
- [ ] No kernel changes. No state-management changes. No quantization
      changes. No new layer types.

---

## Phase 2 — Interface Stabilization

**Goal:** formalize the spec's interfaces (`LayerState`, `Phase` enum,
weight binding, error contract); add per-module unit tests.
**Duration estimate:** 1–2 weeks.
**Branch:** `phase-2-interface`.
**Gate:** Phase 1 gate still holds. New unit tests pass against declared
oracles. No logit change on canary prompts.

### TDD applies from here forward

Each PR below writes tests first. Implementation follows until green.

### PRs

**PR 2.1 — `LayerState` interface.** Test first: a mock `LayerState`
implements the minimal interface (`reset_sequence`, `memory_bytes`,
destructor) and is exercised via polymorphic calls. Implement
`src/state/layer_state.h` with the interface. Gate: unit test green.

**PR 2.2 — `KVCache` conforms to `LayerState`.** Tests first:
existing KV cache behavior preserved, plus a new test for
`truncate_to_position(pos)` (O(1) head adjustment). Refactor
`src/state/kv_cache.cpp` to inherit from `LayerState`. Gate: Phase 1
integration gate + new unit tests.

**PR 2.3 — `RecurrentState` skeleton with `checkpoint/restore/release`.**
Tests first: create a RecurrentState with a synthetic state tensor,
checkpoint it, mutate, restore, verify state equality. Verify
`CheckpointId` double-release is a fail-loud error. Implement the
skeleton (no specific model wiring yet — this is interface-only).
Gate: unit tests green.

**PR 2.4 — `Phase` enum + `build()` signature refactor.** Tests first:
for the extracted `build_attention`, update unit tests (which exist
after Phase 1 as integration-only) to exercise both `Phase::Prefill`
and `Phase::Decode` paths separately. Refactor the signature to
`build(ctx, input, layer_idx, phase)`. Attention internally branches
on phase. Gate: unit tests green, Phase 1 integration gate holds.

**PR 2.5 — `FFNLayer::build()` signature refactor.** Same as 2.4; Dense
FFN has one shape so the phase argument is ignored but accepted for
uniformity. Gate: green.

**PR 2.6 — Weight binding infrastructure.** Tests first: declare a
synthetic layer with two weight slots, provide a fake GGUF tensor
descriptor list, exercise the four failure modes from the spec
(unknown tensor name; missing slot; shape mismatch; dtype mismatch).
Implement `WeightSlot`, `TensorNameMap`, and the loader binding pass.
Refactor the Qwen3 loader path to use it. Gate: existing integration
tests unchanged; four failure-mode tests green.

**PR 2.7 — Error-contract macros.** Introduce `QINF_ASSERT` and the
standard error-formatting macros for slot/expected/actual messaging.
Tests first: verify message format. Migrate existing error sites in
the refactored code to the new macros. Gate: green, error messages
match the contract.

**PR 2.8 — Per-module unit test scaffolding.** For each extracted
module (`attention`, `ffn`, `kv_cache`, `loader`), ensure a
`tests/unit/test_<module>.cpp` exists with at least a build test,
a known-answer test against its declared oracle, and a scratch-budget
test recording `ggml_gallocr_get_buffer_size`. Gate: all unit tests
green.

### Phase 2 exit criteria

- [ ] `LayerState` interface and both concrete classes exist with full
      checkpoint/restore semantics on recurrent state.
- [ ] `build(ctx, input, layer_idx, phase)` is the canonical signature
      for all layer modules.
- [ ] Weight binding rejects the four spec failure modes with
      contract-conformant error messages.
- [ ] Every extracted module has a unit test file at the spec-mandated
      path.
- [ ] No logit change on canary prompts.

---

## Phase 3 — New Layer Types (DeltaNet, MoE)

**Goal:** implement DeltaNet and MoE modules; compose Qwen 3.6.
**Duration estimate:** 3–5 weeks. This is the largest phase.
**Branch:** `phase-3-new-layers`.
**Gate:** Qwen 3.6 logits agree with llama.cpp on canary prompts
(top-8 rank preservation, top-1 divergence index ≥128).

### Sequencing note

DeltaNet and MoE can be developed in parallel on sub-branches, but
neither integrates into a runnable Qwen 3.6 until both land. Plan
with this in mind: one engineer per module is the natural split.

### DeltaNet sub-phase

**PR 3.D.1 — `DeltaNetState` (RecurrentState implementation).**
Tests first: initialize to zeros, apply a synthetic delta update,
verify exponential-moving-average arithmetic via a PyTorch oracle.
Test checkpoint/restore on a non-trivial state. Implement
`src/state/deltanet_state.cpp`. Gate: unit tests green.

**PR 3.D.2 — DeltaNet autoregressive path (1-token decode).**
Tests first: PyTorch reference implementation of the gated delta rule
produces expected output on synthetic inputs at synthetic weights.
Port the autoregressive kernel (non-fused, straightforward ggml
subgraph). Match PyTorch output within declared tolerance. Gate:
unit test green, scratch budget recorded.

**PR 3.D.3 — DeltaNet chunked prefill path.**
Tests first: chunked version produces the same output as
autoregressive run on the same input sequence (consistency test).
Also test against PyTorch reference on long sequences where
chunked/autoregressive should diverge only in roundoff. Implement
chunked path. Gate: consistency test + oracle test green.

**PR 3.D.4 — DeltaNet fused path (single ggml op combining chunked +
autoregressive).** Tests first: fused output matches chunked/autoreg
output on all previously-tested inputs. Implement fused graph construction.
Gate: all prior DeltaNet tests green against the fused path.

### MoE sub-phase

**PR 3.M.1 — Gating network.** Tests first: top-k softmax over 256
experts produces correct indices and weights on synthetic inputs
(compared to a reference implementation in the test file itself).
Implement `src/layers/moe_gating.cpp`. Gate: unit test green.

**PR 3.M.2 — MoE expert dispatch (fallback path).** Tests first:
given a routing table and expert weights, compute the weighted sum
of top-k SwiGLU FFNs. Use the fallback `ggml_mul_mat_id` approach
— the spec allows it, the fused kernel is Phase 4 territory.
**Gate the fallback behind a CMake option `QINF_MOE_FALLBACK`**, not
a runtime branch. Phase 3 builds with `-DQINF_MOE_FALLBACK=ON`;
Phase 4 introduces the fused build. The option is a real escape
hatch, not a dead codepath carried alongside production. CI
compiles both configurations from Phase 4 onward so neither rots
silently. Gate: unit test green against PyTorch oracle on the
fallback build.

**PR 3.M.3 — Shared expert + output blend.** Tests first: verify
the sigmoid-gated shared expert contribution is added correctly.
Gate: green.

**PR 3.M.4 — MoE residency skeleton.** Tests first: simulate a
sequence of routing decisions, verify the residency set tracks the
top-K by frequency, verify cold-path dispatch counter increments on
non-resident dispatch. **Do not wire Metal residency APIs yet — this
is pure bookkeeping in C++.** Gate: unit tests green. Metal wiring
lands in Phase 4.

### Composition and integration

**PR 3.I.1 — `RecurrentState` hparams validation + GGUF binding for
Qwen 3.6 hybrid.** Tests first: load the Qwen3.6 GGUF, verify
hparams are extracted correctly (layer count, which layers are
attention vs DeltaNet, expert count, etc.), verify tensor name map
resolves every tensor. Gate: all four weight-binding failure modes
exercised with the real model file.

**PR 3.I.2 — `Qwen36Model` recipe.** Tests first: logit-level
comparison against llama.cpp on a single short prompt. Implement
the recipe (`src/models/qwen36.cpp`) composing DeltaNet + Attention
+ MoE with the correct `layer_index % 4 == 3` branch. Gate: single-
prompt logit agreement within tolerance.

**PR 3.I.3 — Qwen 3.6 full canary set.** Tests first: extend the
canary-prompt oracle to cover the full canary set for Qwen 3.6.
Fix any tolerance overshoots. Gate: top-8 rank preservation across
all canary prompts, top-1 divergence index ≥128 on generation
prompts.

**PR 3.I.4 — CLI + server integration for Qwen 3.6.** Tests first:
CLI runs Qwen 3.6 to completion on a known prompt; server handles
a /chat request. Gate: integration tests green.

### Phase 3 exit criteria

- [ ] DeltaNet autoregressive, chunked, and fused paths all produce
      matching output within tolerance.
- [ ] MoE gating + dispatch (fallback path) produces output matching
      PyTorch reference.
- [ ] MoE residency bookkeeping tracks top-K correctly; cold-path
      counter exposed as telemetry.
- [ ] Qwen 3.6 runs end-to-end through CLI and server.
- [ ] Logit agreement with llama.cpp meets the spec's gate.
- [ ] Qwen3-32B path unchanged throughout.

### Out of scope for Phase 3

- Fused MoE Metal kernel (Phase 4).
- Metal residency APIs (`MTLResidencySet`, purgeable state) (Phase 4).
- DeltaNet state quantization (Phase 5).
- Attention kernel optimization (Phase 4).
- Speculative decoding (post-Phase-3, separate initiative).

---

## Phase 4 — Metal Kernel Surgery

**Goal:** land all Metal kernel fusions (MoE custom op, residency,
DeltaNet, attention) against an interface that has been validated by
real cross-family pressure.
**Sequencing:** Phase 4 follows the Gemma plan (G1 → G4). See
[`plan-gemma-impl.md`](plan-gemma-impl.md) for the rationale; in short,
fusing kernels before the block interface has hosted a non-Qwen family
risks baking in Qwen-shaped assumptions, and even the "mandatory" MoE
fusion benefits from being designed against two MoE families (Qwen 3.6
A3B and Gemma 4 A4B) rather than one. During the Gemma window, MoE runs
in `QINF_MOE_FALLBACK=ON` mode.
**Duration estimate:** 3–5 weeks (was previously split as 2–4 weeks for
mandatory + open-ended for discretionary; the merged phase is bounded
because the interface is now stable).
**Branch:** `phase-4-kernels`.
**Gate:** identical output to Phase 3 (and to G1–G4 recipes) on the
combined canary prompt set; measured speedup on targeted kernels; no
regression on unchanged layer types.

### Mandatory vs. discretionary as a thinking lens

The mandatory/discretionary distinction is *not* a phasing mechanism —
post-Gemma there is no decision to delay. It is, however, a useful lens
to apply to any individual fusion proposal, then and in the future:

- **Mandatory** = forced by hardware/runtime constraints, family-agnostic.
  Fused MoE custom op (chained `mul_mat_id` per expert is not viable on
  Metal regardless of family) and Metal residency wiring are the standing
  examples. These survive any interface refactor because the constraint
  driving them lives below the interface.
- **Discretionary** = pays off only if the interface being fused against
  is the right one. DeltaNet and attention kernel surgery are the
  standing examples. Pre-Gemma, the assumptions a discretionary fusion
  would have baked in (single residual stream, no soft-cap, no PLE, no
  cross-layer KV sharing) were all latent risks. Post-Gemma, the interface
  exposes hooks for each — fusion now wraps a proven shape.

When a future kernel proposal arrives, ask the lens question before
implementing: *is the constraint driving this fusion above or below the
interface?* Above-the-interface fusions wait for interface stability;
below-the-interface fusions ship whenever they are ready. That is the
durable lesson; the 4a/4b operational split was a pre-Gemma hedge that
expires the moment Gemma lands.

### Characterization first

**PR 4.0 — Performance baseline.** Run all canary prompts through the
post-G4 build with Metal GPU capture. Record per-layer-type timing,
kernel launch counts, scratch budget, and tokens/sec across both Qwen
3.6 (MoE A3B) and Gemma 4 (MoE A4B). This is the baseline that Phase 4
PRs must beat (or match, for kernels not changed). Lives in
`tests/fixtures/perf_baseline_post_gemma.json`.

### PRs

**PR 4.1 — Fused MoE custom op. DROPPED.** Investigated, both premises
falsified. See [docs/phase4-investigation.md](phase4-investigation.md):
(1) `ggml_custom_op` does not reach the Metal backend — the
`supports_op` table has no case for `GGML_OP_CUSTOM`, so any such node
is scheduled to CPU; (2) our `src/layers/moe.cpp` issues three batched
`ggml_mul_mat_id` calls per layer (not chained per-expert as
`ik_llama.cpp` does), so the "kernel launch overhead per expert"
premise was carried over from a different graph shape than ours. Even
if (1) were fixed, fusable MoE-side dispatch totals ~5.7 ms/token on
Qwen 3.6, capping end-to-end speedup at 1.13×. PR slot reused for
DeltaNet sub-stage breakdown below.

**PR 4.1 (replaces dropped) — DeltaNet sub-stage breakdown.**
Tests first: write `tests/perf/deltanet_substage_breakdown.cpp` that
cuts the decode graph at QKV proj / β,α proj / conv update / state
update / output RMSNorm / output proj. Run on Qwen 3.6 layer 0,
record per-stage μs / layer to
`tests/fixtures/deltanet_substage_breakdown_qwen36.json`. Gate: data
distinguishes "launch-bound" (small ops × dispatch overhead dominate)
from "matmul-bound" (output projection GEMV is irreducible). The
choice between PR 4.2 (a fusion PR) and "no action — baseline
preserved" depends on this measurement. Coarse total per-layer cost
is already known: 0.969 ms × 30 layers ≈ 29.1 ms / token (~58% of
50.5 ms decode budget) — this is the largest single chunk of the
budget we have measured.

**PR 4.2 — DeltaNet kernel optimization (conditional).** Tests
first: output identical to Phase 3 fused path within tolerance.
Performance test: measurable speedup vs Phase 3 implementation on
the canary set. Two scopes, decided by PR 4.1's data:
- *Launch-bound regime:* implement a fused Metal kernel for the
  DeltaNet decode step, replacing the current ggml-decomposed
  ~10–15-op-per-step path. Reachable ceiling, conditional on the
  hypothesis: ~1.4–1.7× end-to-end on Qwen 3.6.
- *Matmul-bound regime:* close with "no action needed, baseline
  preserved." Do not optimize for its own sake.
Gate: correctness + performance tests green; OR explicit no-action
close with the breakdown data attached as rationale.

**PR 4.3 — Attention measurement + kernel review.** First:
re-do the attention breakdown via integration with a real
`Qwen36ForwardPass` (the standalone-`simple_kv_cache` approach
attempted in `tests/perf/attention_breakdown.cpp` hits a
cross-context gallocr issue — see investigation doc, "Attention —
deferred"). Record per-layer μs at n_kv ∈ {128, 512, 2048}. Then
review the existing kernel against that data. Primary optimization
candidate: switch to `ggml_flash_attn_ext`. The win is asymmetric —
meaningful for prefill (long prompts, large N×N score matrix never
materialized) and small for single-token decode. Evaluate prefill
throughput specifically. Additional benefit: `ggml_flash_attn_ext`
has native parameters for sliding-window, QK-norm, logit soft-cap,
and scale, which would let us delete the custom mask-building and
score-scaling logic currently in `attention.cpp`. This PR explicitly
accounts for the soft-cap (G2), sandwich-norm (G2), sliding-window
(G2/G3/G4), QK-norm (G3), p-RoPE (G4), and shared-KV (G4) hooks added
during the Gemma phases — the fused kernel exposes them rather than
hard-coding their absence. Gate: measurement landed; either
prefill-throughput speedup demonstrated or explicit "no action
needed, baseline preserved" with data.

**PR 4.4 — Metal residency APIs for MoE.** Tests first: on a build
with simulated memory pressure, verify pinned experts stay resident
and cold-path counter matches expected routing misses, on both Qwen
3.6 and Gemma 4 A4B. Wire `MTLResidencySet` (or
`setPurgeableState:NonVolatile` fallback) to the residency bookkeeping
from PR 3.M.4. Gate: residency test green on M-series hardware; no
kernel panics under pressure scenarios that previously tripped
llama.cpp. (Reordered after the kernel PRs above — residency is a
correctness-under-pressure feature, not a throughput PR, so it does
not compete for measurement-driven prioritization.)

### Phase 4 exit criteria

- [ ] DeltaNet sub-stage breakdown landed; PR 4.2 either ships a
      measured kernel speedup or signs off "baseline preserved" with
      data justifying the no-action close.
- [ ] Attention measurement landed via ForwardPass integration;
      PR 4.3 either optimizes prefill (e.g. `ggml_flash_attn_ext`)
      with measured speedup or signs off "baseline preserved."
- [ ] MoE expert residency is wired to Metal APIs and demonstrates
      no kernel panics under pressure.
- [ ] Output identical to post-G4 baseline on canary prompts.
- [ ] No regression on Qwen3-32B, Qwen 3.6, or any Gemma generation.

---

## Phase 5 — State Optimizations

**Goal:** prototype DeltaNet state quantization; explore per-layer-type
precision budgeting. (TurboQuant was removed — outside workload envelope.)
**Duration estimate:** ongoing / research-driven.
**Branch:** `phase-5-quant` (and sub-branches for experiments).
**Gate:** output quality within declared perplexity bounds on a held-out
corpus.

This phase is explicitly research-shaped. The PR breakdown is sketched,
not prescribed — findings from early PRs will reshape later ones.

### PRs (indicative)

**PR 5.0 — Perplexity harness.** Tests first: the harness produces
stable perplexity numbers on a fixed corpus across repeated runs.
Implement `tests/perf/perplexity.cpp` measuring perplexity on a held-
out text (e.g. WikiText subset, code subset, dialogue subset).
Gate: harness is deterministic and reproducible.

**PR 5.1 — DROPPED (TurboQuant removed).** KV-precision compression is
outside the workload envelope (≤10 slots, ≤4K context, 12GB models).

**PR 5.1b — DROPPED (TurboQuant removed).** Hadamard-transform K/V
comparison no longer has a baseline to compare against.

**PR 5.2 — DeltaNet state quantization prototype.** Tests first:
the perplexity harness accepts a quantized-state build and reports
numbers for multiple bit-widths. Prototype INT8 recurrent state.
Gate: this is a research PR. The "gate" is a reported finding
(perplexity delta, memory savings), not a ship decision. If the
finding is positive, a follow-up PR ships it; if negative, the
branch is archived with the findings documented.

**PR 5.3 — Per-layer-type precision budgeting.** Tests first:
perplexity harness runs against multiple per-layer-type quant
configurations (e.g. Q6_K attention + Q2_K DeltaNet + Q3_K MoE).
Gate: findings reported, decisions documented. No silent
commitments to specific bit-widths in the absence of measurement.

### Phase 5 exit criteria

Phase 5 does not have a fixed exit. It proceeds PR by PR as research
findings accumulate. A Phase 6 is called when a cohesive new capability
emerges (e.g. "shipping a role-quantized Qwen 3.6 preset").

### Phase 5 minimum viable slice

If Phase 5 must be deferred or compressed, **PR 5.0 (perplexity harness)
is load-bearing** and should not be skipped. PR 5.1/5.1b are dropped
(TurboQuant removed). The rest of Phase 5 (5.2, 5.3) is genuinely
research-shaped and can be deferred without blocking shipping.

---

## Prior Art References

Short pointers to external projects whose ideas we want to compare
against or borrow from. Not commitments — scoping notes for the PRs
above.

- **`ik_llama.cpp`** (Iwan Kawrakow's llama.cpp fork). CPU/CUDA focus,
  no Metal. Worth watching:
  - **Fused MoE graph shape** — reference for PR 4.1.
  - **Hadamard transform on K/V + Q8_KV** — PR 5.1b dropped
    (TurboQuant removed; no baseline to compare against).
  - **Self-speculative + ngram decoding** — reference when speculative
    decoding resumes as a separate initiative (see Cross-Cutting →
    Speculative Decoding).
  - **Trellis quants (IQ*_KT)** — novel weight-quant family; watch
    quality numbers, do not port speculatively.
  - Not relevant to us: FlashMLA (wrong model family — Qwen doesn't use
    MLA), tensor overrides for hybrid CPU/GPU storage (wrong hardware —
    unified memory on Apple Silicon).

Full scrutiny lives in conversation / commit history; this section
exists so future-you remembers the pointer without re-deriving it.

---

## Cross-Cutting Concerns

### Concurrency model

Decided now so the impl doesn't paint itself into a corner:

- **Model is immutable after load.** `ggml_context`, weight tensors,
  `MTLResidencySet` configuration — all frozen after `Qwen36Model`
  construction completes.
- **Per-sequence state is passed in, not owned.** `KVCache` and
  `RecurrentState` instances are owned by the caller (CLI or server),
  not by the model. The model's `forward()` takes them as arguments.
- **A single model instance can serve multiple sequences sequentially.**
  Concurrent forward passes against the same model instance require
  external synchronization. Multi-sequence batching is a server-layer
  concern handled in Phase 6+.

### Observability bus

One metrics surface, introduced in Phase 3 at the same time as MoE
residency:

- `src/telemetry/metrics.h` exposes counters and histograms.
- MoE cold-path dispatch counter is the first citizen.
- Per-layer timing, KV cache occupancy, and residency hit rate are
  added as they become relevant.
- **No metrics framework dependency.** Plain C++ structs; consumers
  can hook a Prometheus / OpenTelemetry exporter at the server layer
  if needed. Core engine stays free of transport concerns.

### Migration mechanics

- Phase 1's `git mv` PRs are kept minimal to avoid merge conflicts
  with concurrent work. Schedule them as the first thing on a quiet
  day; merge quickly.
- Include-path updates may touch many files. Use a scripted rename
  (`rg --files-with-matches 'old/path' | xargs sed -i 's|old|new|g'`)
  committed as a single mechanical commit, reviewable by diff-checking
  the script.
- CI stays green throughout. If a phase-branch PR breaks CI on the
  phase branch, it is reverted within 24h, not left broken.

### Checkpoint format versioning

Deferred until Phase 5 ships state quantization or Phase 6 ships state
serialization for persistent memory. When needed: a 16-byte header with
magic number + version + content length, followed by the raw tensor
data. Do not pre-design this before a concrete serialization use lands.

### Speculative decoding

Listed as "in progress" in `CLAUDE.md` before this refactor. Status
after the refactor:

- The rewind primitives it needs (`RecurrentState::checkpoint/restore`)
  ship in Phase 2.
- The draft / verify / reject loop lives above the model layer and is
  orthogonal to the modular-layer refactor.
- It is **not** a gate for any refactor phase. When the refactor
  reaches Phase 3, speculative decoding can resume as an independent
  initiative using the primitives the refactor provides.

---

## Out of Scope

Captured here so nobody has to re-ask:

- Anything in `docs/future-work.md` (self-speculative decoding,
  persistent DeltaNet memory, MoE-as-semantic-hash, etc.). Those are
  not commitments of this refactor.
- CUDA / Vulkan backend support. The spec and this plan are Metal-only.
- Multi-GPU tensor parallelism.
- A custom GGUF variant. We consume standard GGUFs produced upstream.
- A replacement tokenizer or chat-template engine.

---

## Timeline Sketch

Rough, not a commitment:

| Phase | Estimate | Parallelizable? |
|---|---|---|
| 1 | 1–2 weeks | No (serial extraction) |
| 2 | 1–2 weeks | Partial (interface PRs are sequential; unit tests can parallelize) |
| 3 | 3–5 weeks | Yes (DeltaNet and MoE tracks run in parallel) |
| 4 | 2–4 weeks | Yes (per-kernel PRs parallelize) |
| 5 | Ongoing | Yes (research-shaped) |

Total from locked spec to Phase 3 exit (Qwen 3.6 runnable): **5–9
weeks** with one or two engineers. Phase 4 adds the performance story;
Phase 5 is continuous.

---

## Success Looks Like

At the end of Phase 3:

- Qwen 3.6 runs end-to-end on the refactored engine.
- Adding a hypothetical new hybrid model (say, Jamba) requires one new
  recipe file and zero changes to existing modules.
- Every layer module has its own test file, its own Metal kernel, and
  its own declared oracle.
- The spec's collaboration invariants hold: no file exceeds ~1500 lines,
  no dumping-ground directories, errors at module boundaries name slot
  + expected + actual.

At the end of Phase 4:

- MoE performance is within spec on Apple Silicon, no kernel panics
  under pressure.
- Per-layer performance baseline is documented and protected by the
  perf regression gate.

At the end of Phase 5 (open-ended):

- The engine ships per-layer-type precision presets validated by the
  perplexity harness.
- DeltaNet state quantization findings documented, shipped if positive.
