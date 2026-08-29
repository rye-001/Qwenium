## Project Overview

A high-performance inference engine for Qwen-family models, running on Apple
Silicon via Metal. Written in C++ using ggml as the tensor computation backend.

## Architecture Doc Protocol (load-bearing)

`docs/architecture.md` is the **as-built map** of the system — the context
bootstrap for every session.

- **Read it at the start of every session**, before proposing or making
  changes. It orients faster and more accurately than re-deriving the system
  from the tree.
- **Architecture changes require user approval before they land.** Anything
  that would alter what the doc describes — adding/removing a `src/`
  directory, model recipe, server endpoint or flag, state kind; changing a
  named seam (server callbacks, vision Seam A/B, snapshot header); settling a
  §12 soft-spot item — must be surfaced to the user as an explicit decision
  first. Once approved, update `architecture.md` in the same change (§13 of
  the doc lists the triggers). An unapproved architecture drift or a stale doc
  is a review-rejectable defect, same as a failing gate.

## Architecture Direction

The engine is being refactored from a monolithic, model-specific design toward
a composable, layer-type-driven architecture. See `docs/modular-layer-architecture.md`
for the full blueprint.

Key principles:
- **Layer-type modules** (Attention, SSM, DeltaNet, MoE, Dense FFN) are independent,
  testable units that build into a shared `ggml_cgraph`. They are graph-building
  functions, not standalone executables.
- **Models are recipes** — a model file composes layer modules and owns the residual
  stream. Adding a new model should require zero changes to existing modules.
- **Architecture-agnostic systems** (sampling/grammar, speculative decoding, weight
  quantization, Metal dispatch) sit above or below the layer modules.
- The refactor follows strict phasing: extract first (bit-for-bit identical),
  stabilize interfaces second, add new layer types third, optimize kernels fourth.
  Never combine extraction with optimization in the same step.

## Current Capabilities (Decode Phase)

1. Weights: K-Quants
2. Output Leash: Grammar / TokenTrie
3. Decode Speed: Speculative Decoding (in progress)

## Workload Envelope (load-bearing)

This engine targets ≤10 concurrent slots, ≤10 K context, 12 GB-class
quantized models on Apple Silicon unified memory.

**Raised from 4 K to 10 K (2026-08-24)** so our own assumption stops
blocking us. The 4 K figure was written for order-management text prompts
and never contemplated an image: a Qwen-VL-class encoder sizes its token
count to the image, so one A4 page at usable scan resolution costs
2 K–8 K soft tokens and did not fit. 10 K is the smallest ceiling under
which a single high-resolution document plus its prompt and answer fit.

KV memory is still not expected to be the binding constraint, but the
margin is thinner: KV bytes scale as **ctx x slots**, so this is a 2.4x
multiplier on the axis that was previously declared non-binding. The
TurboQuant / SnapKV deletion rationale ("no payoff inside this envelope")
now rests on a weaker measurement and should be re-checked, not assumed,
if a 10-slot host ever runs tight. `--kv-f16` halves KV bytes and is the
first lever to reach for.

## Target Models

Qwen family:
- Qwen2/2.5
- Qwen3-32B (pure transformer — attention + dense FFN)
- Qwen3.5 (attention + SSM)
- Qwen 3.6-35B-A3B (hybrid — DeltaNet + attention + MoE, arch: `qwen35moe` in GGUF)

Gemma family (text-only) — the designated **cross-family forcing function**;
see `docs/plan-gemma-impl.md`:
- Gemma 1 (pure transformer)
- Gemma 2 (interleaved local/global attention — even=local, odd=global)
- Gemma 3 (5:1 local:global, per-layer RoPE base)
- Gemma 4 (parallel dual-FFN — dense FFN **and** MoE FFN summed per layer;
  structurally distinct from Qwen 3.6 A3B's pure per-layer MoE)

**Cross-family rule (load-bearing).** Any plan, interface, or refactor that
touches the forward pass MUST be designed and grounded against at least one
Qwen recipe AND at least one Gemma recipe before it is considered complete.
An interface validated only on Qwen is presumed Qwen-shaped until a Gemma
recipe proves otherwise — Gemma is the falsifier, not an afterthought. The
byte-identical gate for any extraction includes a Gemma model, not just Qwen.

## ggml Constraints

- All layer modules must share one `ggml_context` and build into one `ggml_cgraph`.
- Avoid premature `ggml_cont()` / `ggml_cpy()` — they pin scratch buffer memory.
- MoE dispatch is built on three batched `ggml_mul_mat_id` calls per layer
  (gate / up / down), each backed by a native Metal `MUL_MAT_ID` kernel. C-level
  launch count is O(1) in `n_experts`. Do **not** introduce `ggml_custom_op`
  paths — the Metal backend's `supports_op` table has no case for
  `GGML_OP_CUSTOM` / `GGML_OP_MAP_CUSTOM*`, so any such node is scheduled to
  CPU. If MoE-side optimization is ever revisited, the candidate is patching
  ggml-metal's `MUL_MAT_ID` kernel for sparse routing, not graph-level fusion.
  See [docs/phase4-investigation.md](docs/phase4-investigation.md) — measured
  ceiling on Qwen 3.6 ≤ 1.13×.
- KV cache (append semantics) and recurrent state (overwrite semantics) are
  fundamentally different — never unify their implementations behind a shared base
  that assumes one update pattern.

## Collaboration Invariants

Load-bearing rules that protect the codebase from drifting into shapes that
are hostile to both human engineers under time pressure and LLM agents under
context limits. If any rule ever conflicts with good engineering, good
engineering wins and the rule is revised. Full list in
`docs/modular-layer-architecture.md` → Collaboration Invariants.

- **Locality of reasoning.** A change to one layer type should not require
  reading more than two other files. Reviewers reject violations regardless
  of whether the change compiles.
- **Test co-location.** `src/<path>/<module>.cpp` has its unit test at
  `tests/unit/test_<module>.cpp`. Always. No nesting variants.
- **No dumping grounds.** `src/` contains only concept-named directories
  (`layers/`, `state/`, `graph_inputs/`, `models/`, `metal/`, `sampling/`,
  `quant/`, `loader/`, `cli/`, `server/`). No `util/`, `common/`, `misc/`,
  `helpers/`.
- **Fail-loud error contract.** Errors at module boundaries name the slot
  or parameter, the expected value, and the actual value, in that order.
  Silent fallbacks and best-effort recovery are forbidden at module
  boundaries.
- **No cleverness that hides the public surface.** Prefer verbose and
  explicit over clever and terse. No template metaprogramming, runtime
  reflection, or multi-level indirection. Generated code lives under
  `src/generated/` and is never hand-edited.

## Architecture Pressure Test

How we judge whether a proposed change is healthy or quietly corrosive:

- **Architecture earns its keep by surviving real pressure, not by looking clean in isolation.** A design that has only ever hosted Qwen variants has not been falsified yet. New requirements (a new model family, a new attention variant, a new state type) are *forcing functions* that validate or invalidate the abstraction. We adopt them when they sharpen module boundaries; we resist them when they would require special cases.
- **Parameterize or split — pick the conceptual seam, case by case.** A sliding-window mask is a parameter on attention; p-RoPE is a parameter on RoPE; PLE is its own module because it is a different signal flow, not a knob on embeddings. Both failure modes are real and equally bad: a model zoo of parallel modules, *and* a fat function whose params are orthogonal silos that no single call site uses together. Neither default ("always parameterize" or "always split") is safe. This is a judgment call that needs on-demand attention at design time — flag it explicitly when it comes up, do not let it ride.
- **No model zoo.** We do not add support for a model because it exists. We add it when (a) it is a forcing function for clarifying the architecture, or (b) it is a credible production target. Both bars are high. The failure mode we are explicitly avoiding is the `llama.cpp` trap: support every model, accumulate special cases, end up with a giant switch statement and no real abstraction.
- **Success metric for any cross-family or cross-variant change: "the architecture hosted X without bending."** If implementing X required non-trivial edits to existing layer modules (not new optional parameters — actual logic edits to code other recipes depend on), treat it as an interface defect, not a feature win. Pause and fix the interface.

## Collaboration Style

Proactively flag architectural smells, redundant abstractions, wrong abstraction levels, or design improvements — don't wait to be asked. If a completed task reveals something worth questioning about the broader design, say so immediately.
