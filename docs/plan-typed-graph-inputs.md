# Typed Graph Inputs

> **STATUS (2026-05-17): PHASE 1 COMPLETE — byte-identical gate GREEN.
> Extraction-only refactor. No behavior change, no perf claim. This is
> interface infrastructure that unblocks graph reuse (#2), zero-copy input
> writes, and hybrid-memory composition (Qwen3.5 / Qwen3.6).**
>
> **Zero-diff gate (passed):** deterministic differential harnesses,
> compiled identically on pre-refactor `main` @7fac6f7 and this branch,
> `cmp`/SHA-256-compared. Rows 1–6: `tests/diff/logits_dump.cpp` (full
> prefill logits + 8 greedy decode tokens). Row 7: the extended
> `tests/grammar/test_sparse_differential.cpp` Run A (real grammar,
> `force_dense=false`) dumping full prefill logits + first token + 8
> grammar-constrained decode steps, each with the `valid_indices` graph-node
> readback + pre-mapping sparse `[k]` logits (the direct tripwire for the
> SparseHeadInput set/clear ORDERING bug — invisible in post-mapped vocab
> logits). Tree-portable via `__has_include` (`main`: legacy
> `upload_sparse_indices`+`set_batched_inputs`; branch: `set_decode_inputs`).
>
> | # | recipe | config | result |
> |---|---|---|---|
> | 1 | qwen35 | kv=0 dense | ZERO-DIFF (SHA-256) |
> | 2 | qwen35 | kv=2 TQ | ZERO-DIFF |
> | 3 | qwen3 | kv=0 dense | ZERO-DIFF |
> | 4 | qwen3 | kv=2 TQ | ZERO-DIFF |
> | 5 | gemma2 | dense (interleaved local/global mask) | ZERO-DIFF |
> | 6 | gemma3 | dense (5:1 local/global mask) | ZERO-DIFF |
> | 7 | qwen35 | **sparse + grammar** (`valid_indices` node + `[k]` logits) | ZERO-DIFF |
>
> Dense + TQ + the cross-family Gemma falsifier + the highest-risk sparse
> path are all proven byte-identical, not merely wired. Row 7 exercised a
> genuine sparse step (29 valid ids; `valid_indices` node and 29 `[k]`
> logits read back and compared).
>
> **Four-copies fully collapsed:** the TQ per-layer batch graph and the
> embedding sub-graph now flow through the typed inputs (TokensInput /
> PositionsInput / AttnMaskInput) for qwen3 and qwen35. Only `inpL` /
> `final_in` hidden-state carriers remain direct `ggml_backend_tensor_set`
> — explicitly out of scope per the plan's scope fence (typed inputs cover
> tokens/positions/masks/state, not intermediate hidden states).
>
> **Phase 1 done so far:** `src/graph_inputs/` (GraphInput contract,
> StepContext, GraphInputSet + 5 concrete inputs) built; all 7 recipes
> (qwen3/35/36, gemma1-4) migrated; per-recipe `set_inputs`/
> `set_batched_inputs` and the `upload_sparse_indices`/`valid_indices_input_`
> one-off deleted; callers (decode_step, http_server) updated; 6 unit-test
> modules added (16 tests passing). qwen35/qwen3 **TQ-batch loop left as-is**
> (out of scope per owner decision — its inpL/final_in are hidden-state
> carriers behind the plan's scope fence, not typed inputs).
>
> **Parameterize-vs-split decision recorded:** qwen36's KV gather uses an
> `n_kv_len` per-slot stride vs qwen3/qwen35's `n_ctx_max` — handled as an
> explicit `GatherIndicesInput::Stride` policy *parameter*, not a new class
> (same reasoning as the sliding-window parameter).
>
> **Outstanding gate (owner-run, needs weights+GPU):** byte-identical logits
> vs `main` on a fixed prompt — dense and sparse — for Qwen3-32B, Qwen3.6
> (TQ on/off), and **Gemma 2 + Gemma 3** (per-layer interleaved local/global
> mask path). Diff must be exactly zero. Qwen-only green is not Phase 1 done.

---

**Goal:** Replace the scattered, every-step, name-lookup tensor population
(`ggml_graph_get_tensor(gf,"name")` + `ggml_backend_tensor_set`) with a set
of typed input objects, each of which knows (a) how to populate its own
tensor (`set_input`) and (b) whether it is still valid versus the previous
step (`can_reuse`). Same data on the wire. Same numerical output. The only
change is *who owns the poke*.

**Non-goals (explicitly deferred to follow-up plans):**
- Graph reuse across decode steps (#2). This plan makes it *possible*, not
  done.
- Zero-copy in-place input writes (Apple Silicon unified memory). Falls out
  once tensor addresses are stable under reuse; not in scope here.
- Hybrid-memory composition (KV + recurrent state in one step). This plan is
  *designed against* that case but does not implement it.
- In-graph sampling (#3). Out of scope, and stays out — sampling remains a
  CPU-side `sampling/` module. Typed inputs cover tokens/positions/masks/
  state only.

---

## Why now

The current state, grounded in code:

- `src/models/qwen35.cpp:273` `set_inputs` and `:920` `set_batched_inputs`
  each do string lookup + memcpy per tensor, per step.
- The *same* pattern is duplicated in the TQ per-layer path
  (`qwen35.cpp:619–736`), the embedding sub-graph (`:661`), and the final-norm
  sub-graph (`:783`). Four independent copies of "find tensor by name, set
  bytes."
- Sparse decode already invented a *one-off* typed input by hand:
  `set_sparse_decode_ids` / `upload_sparse_indices`
  (`forward_pass_base.h:188,195`). That is the pattern, applied once,
  ad hoc. This plan generalizes it instead of letting each input reinvent it.
- `decode_step.cpp:74–79` rebuilds + reallocs the graph every step. You
  cannot reason about "what changed since last step" because the answer is
  smeared across four call sites and string keys.

The recipes in scope are not just Qwen: `src/models/` has `gemma1–4`,
`qwen3`, `qwen35`, `qwen36`, each with its own hand-rolled `set_inputs` /
`set_batched_inputs` (e.g. `gemma2.cpp:258`, `gemma3.cpp:272`,
`qwen36.cpp:442`). Per `docs/plan-gemma-impl.md`, Gemma *is* the designated
cross-family forcing function — the typed-input interface must host Gemma's
per-layer interleaved local/global masks without bending, or it is a Qwen-
shaped interface wearing a generic name.

The forcing function is **hybrid memory**. Qwen3.5 (attention + SSM) and
Qwen3.6-35B-A3B (DeltaNet + attention + MoE) need a KV input (append) and a
recurrent-state input (overwrite) live in the same graph, same step. The
correct shape for that is *composition of two typed inputs*, not a unified
base (CLAUDE.md ggml-constraints invariant). If the typed-input interface is
designed without that consumer in mind, it will bend when those models land.
So this interface is designed against the hardest consumer first, even
though that consumer is not built here.

---

## Interface

One contract. Verbose and explicit on purpose (no template metaprogramming,
no CRTP — CLAUDE.md cleverness rule).

```cpp
// src/graph_inputs/graph_input.h
class GraphInput {
public:
    virtual ~GraphInput() = default;

    // Populate this input's tensor(s) for the given step. Called after
    // ggml_backend_sched_alloc_graph, before compute.
    virtual void set_input(const StepContext& step) = 0;

    // True iff nothing this input depends on changed since the last
    // set_input. Conservative: default false. Only an input that can
    // *prove* invariance returns true.
    virtual bool can_reuse(const StepContext& step) const { return false; }

    // Name of the tensor slot this input owns, for fail-loud diagnostics.
    virtual const char* slot_name() const = 0;
};
```

Concrete inputs, one per current named tensor:

| Class | Owns slot | Depends on |
|---|---|---|
| `TokensInput` | `tokens` / `inpL` | token id list |
| `PositionsInput` | `inp_pos` | positions |
| `AttnMaskInput` | `kq_mask*` / `kq_mask_b` | n_kv, n_tokens, slot causal extent, **per-layer window** (`layer_window[il]`) |
| `GatherIndicesInput` | `gather_indices` | batched slot layout |
| `SparseHeadInput` | `valid_indices` | grammar valid set (folds in the existing `sparse_decode_ids_` one-off) |

`StepContext` is a plain struct carrying what the inputs read (tokens, slots,
positions, sparse ids, cache positions). It replaces the four overlapping
parameter lists (`set_inputs`, `set_batched_inputs`, the TQ loop, the
sub-graphs) with one.

**Parameterize-vs-split (DECIDED): sliding window is a parameter on
`AttnMaskInput`, not a separate input class.** Gemma 2 (`gemma2.cpp:36`,
even=local / odd=global) and Gemma 3 (`gemma3.cpp:38`, 5:1 local:global)
build a *different* `kq_mask` per layer depending on
`config_.layer_window[il]`. Qwen recipes use one uniform causal mask. These
are the same concept — a causal mask with an optional window cutoff — so
`AttnMaskInput` takes the per-layer window as a construction parameter and
emits the right mask. We do **not** create `LocalMaskInput` /
`GlobalMaskInput`; that would be the model-zoo failure mode, and CLAUDE.md's
canonical parameterize example is precisely "a sliding-window mask is a
parameter on attention." Recorded here so it is not re-litigated per recipe.

**Explicitly NOT inputs (scope fence):** Gemma 3 per-layer RoPE base
(local 10K / global 1M, `gemma3.cpp:41–43`) and Gemma 2 attention/final
logit soft-cap are *graph ops*, not input tensors. They do not add
`GraphInput` types and are out of scope for this refactor. Listed so a
later reader does not pull them in.

**Fail-loud contract (CLAUDE.md):** if `set_input` cannot find its tensor in
the graph, it throws naming the slot, the expected dtype/shape, and the
actual (or "absent"), in that order. No silent skip. This is strictly
stronger than today's `ggml_graph_get_tensor` returning null and segfaulting
downstream.

**Composition (designed-for, not built here):** a future
`HybridMemoryInput` holds `unique_ptr<AttnKvInput>` + `unique_ptr<RsInput>`
side by side and fans `set_input`/`can_reuse` out to both — each keeps its
own semantics. This is the llama.cpp `llm_graph_input_mem_hybrid` shape,
which is itself composition not unification, and is therefore consistent
with our KV≠RS invariant. The interface above must make that wrapper
expressible with zero edits to `AttnKvInput`/`RsInput`. That is the
acceptance test for the interface, even though the wrapper is out of scope.

---

## Directory & test placement (DECIDED 2026-05-16)

**Decision: `src/graph_inputs/` — its own top-level concept dir.**

Rationale: "graph input" is a distinct concept (a self-describing,
reusable contract over a graph slot), not a `models/`-private detail. It is
the seam that hybrid memory, graph reuse, and zero-copy writes all build on;
those consumers are not all inside `models/`. A top-level dir keeps the
concept addressable and prevents `models/` from accreting an unrelated
sub-system. This satisfies the CLAUDE.md "concept-named directories / no
dumping grounds" rule (`graph_inputs/` is added to the allowed concept-dir
list alongside `layers/`, `state/`, etc.).

Layout:
- `src/graph_inputs/graph_input.{h,cpp}` — the `GraphInput` contract +
  `GraphInputSet`.
- `src/graph_inputs/<name>_input.{h,cpp}` — one pair per concrete input
  (`tokens_input`, `positions_input`, `attn_mask_input`,
  `gather_indices_input`, `sparse_head_input`).
- Tests co-locate as `tests/unit/test_<module>.cpp` (CLAUDE.md test
  co-location, no nesting variants): `tests/unit/test_graph_input.cpp`,
  `tests/unit/test_tokens_input.cpp`, …

> **CLAUDE.md follow-up (do as part of Phase 1):** add `graph_inputs/` to
> the "concept-named directories" enumeration in CLAUDE.md and in
> `docs/modular-layer-architecture.md` so the allowed-dir list stays the
> single source of truth and a future reviewer does not flag it as a
> dumping ground.

---

## Phasing — strict, extraction before anything new

### Phase 1 — Extract, bit-for-bit identical
Introduce `GraphInput` + the five concrete classes. Each `set_input` body is
**moved, not rewritten**, from the corresponding lines in `qwen35.cpp`. No
`can_reuse` returns true yet (all default false). `decode_step` and
`run_prefill` build a `std::vector<GraphInput*>` and loop `set_input`
instead of calling `set_inputs`/`set_batched_inputs`. The four duplicated
copies (decode, batched, TQ loop, sub-graphs) collapse to one list.

**Done criteria:**
- Byte-identical logits vs. `main` on a fixed prompt, dense and sparse,
  Qwen3-32B and Qwen3.6 (TQ on and off). Diff must be exactly zero, not
  "close."
- Byte-identical logits on **Gemma 2 and Gemma 3** as well — these exercise
  the per-layer interleaved local/global mask path (`gemma2.cpp:258`,
  `gemma3.cpp:272`) that Qwen does not. Gemma passing is the proof the
  interface is cross-family, not Qwen-shaped. Qwen-only green is **not**
  Phase 1 done.
- `set_inputs` / `set_batched_inputs` virtuals deleted from
  `forward_pass_base.h`; no remaining `ggml_graph_get_tensor` in the
  forward-pass hot path.
- The `set_sparse_decode_ids`/`upload_sparse_indices` one-off is replaced by
  `SparseHeadInput` (behavior identical).
- Unit tests per `tests/unit/test_<module>.cpp`.
- **Pause gate:** if extraction requires *logic* edits to any module other
  recipes depend on (not new optional params — actual behavior edits), stop.
  That means the seam is wrong; redo the interface, do not paper over it.

### Phase 2 — Stabilize `can_reuse` (still no behavior change)
Implement honest `can_reuse` per input (e.g. `PositionsInput` reusable iff
the position delta is the trivial +1; `AttnMaskInput` reusable iff n_kv and
causal extent unchanged). **Sliding-window stressor (mandatory test case):**
a saturated Gemma local layer has a shape-stable mask whose *content* shifts
every step — the naive "n_kv unchanged ⇒ reuse" rule is a false positive
here and would silently corrupt output. `AttnMaskInput::can_reuse` must
return false for a windowed layer unless it can prove the window contents are
unchanged. Phase 2 is not done until the assertion path is exercised on a
Gemma 3 run long enough to saturate the sliding window. Add a
`GraphInputSet::can_reuse()` that ANDs the
inputs (llama.cpp `llm_graph_result::can_reuse` shape). **Wire nothing to
it yet** — add an assertion-only path that logs reuse-eligibility vs. the
existing always-rebuild path and verifies they would agree. This validates
the predicate before any rebuild is skipped.

**Done criteria:** reuse-eligibility log matches a hand-derived expectation
on a constrained-decode run; zero behavior change; no graph actually reused.

### Phase 3 (separate plan, not scheduled here)
Consume `can_reuse` in `decode_step` to skip rebuild/realloc (#2). Then
zero-copy in-place writes (stable tensor address under reuse removes the
portability memcpy). Then `HybridMemoryInput`. Each is its own plan with its
own pause gate. **Listed for direction only — do not start from this doc.**

---

## Risks / drift watch

- **Drift into llama.cpp's class tree.** Take the *pattern* (self-describing
  input, `set_input`/`can_reuse`). Do **not** import its hierarchy depth,
  and do **not** let sampling become a graph input (their
  `llm_graph_input_sampling` is the seam where they fuse concerns we keep
  apart). Five concrete classes, flat, no base-of-base.
- **`can_reuse` false-positive = silent corruption.** A wrongly-true
  `can_reuse` reuses a stale tensor and produces wrong tokens with no error.
  Hence Phase 2 is assertion-only and Phase 3 is gated separately. Default
  must always be false; reusability is opt-in per input with a proof.
- **Interface defect via hybrid.** Success metric: the (out-of-scope)
  `HybridMemoryInput` wrapper is expressible with zero edits to
  `AttnKvInput`/`RsInput`. If a reviewer can't convince themselves of that
  from the Phase 1 interface, Phase 1 is not done.

---

## Out of scope, on record so it is not retried

- In-graph sampling (#3): rejected for this backend (unified memory → no
  readback cost; `ggml_pad` shape trick violates the cleverness rule). See
  prior analysis. Sampling stays CPU-side in `sampling/`.
- Prefill last-token hidden-state slicing (llama.cpp `inp_out_ids`, #1):
  real but orthogonal; a perf task for when prefill is the bottleneck, not
  part of this interface refactor.
