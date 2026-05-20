# Grammar-Guided Decode: Sparse LM Head + Forced-Token Elision

> **STATUS (2026-05-16): Phase A landed as correctness infrastructure (0% perf
> on this workload, accepted). Phase B NOT justified — do not start. The real
> performance win came from outside this plan.**
>
> What this investigation actually established:
>
> - **Phase A (sparse LM head): landed, byte-identical, fail-loud guards in
>   place. Measured 0% tok/s** on order-management + Qwen3.6. The plan's
>   `<3%` pause-gate tripped. Kept as cheap correct infra and a precondition
>   for any future candidate-narrowing; *not* a perf feature. Inert when TQ is
>   on (decode routes through the TQ-aware `run_prefill` bridge).
> - **The bottleneck was never the forward pass.** It is grammar-side
>   `get_valid_tokens` (~54% of decode). Phase B elides the forward pass and
>   therefore does **not** address it — Phase B is unjustified on this workload
>   and is shelved, not scheduled.
> - **The real win was a grammar fix, not an engine fix:** closing the DSL's
>   func/field identifier sets to literal alternations (a *grammar-authoring*
>   change, no engine code) cut grammar cost −66%, +39% tok/s on a
>   spec-following model. See [Outcomes](#outcomes-2026-05-16).
> - **Two killed hypotheses (by measurement, before code):** prompt-trie value
>   slots (value slots were 0.36% of decode), and a per-parse variable-name
>   symbol table (variable-site scan <1%). Recorded so they are not retried.
> - **Remaining optimization work** is tracked in
>   [plan-resolve-once.md → Superseding Work](plan-resolve-once.md#superseding-work-the-real-todo),
>   not here.
> - **Two open items that are NOT optimization and need human owners:**
>   (1) the CHAR_CLASS too-permissive *correctness* bug from the Phase A commit
>   (still parked as an inline comment); (2) the Q2-vs-9B product decision —
>   closed-schema grammars convert the weak shipping model's soft failures into
>   hard no-output. Neither is a prototype task.
>
> Everything below is the original plan, retained for context. Phase A's
> "Done criteria" are annotated with actual results. Phase B is retained as
> analysis only — **not** a scheduled next step.

---

**Goal:** Exploit grammar-determined token sets to (a) shrink the LM-head matmul when only k tokens are valid, and (b) skip the forward pass entirely when only one token is valid.
**Impact (rough, on Qwen 3.6-A3B with a constrained DSL):**
- Sparse LM head alone: ~5–10% more tok/s.
- Combined with forced-token elision: ~1.5–2× tok/s on grammar-heavy workloads (e.g. [tests/system_prompt_order_mngmt.txt](../tests/system_prompt_order_mngmt.txt)).

This plan is the consequence of a "do we need the whole LLM for this DSL?" stress test. The answer: no — and the existing `sampling/` ↔ `models/` seam can host the optimization without bending any layer module. That makes it a healthy architectural pressure test, not just a feature.

---

## Core idea

Today: forward pass produces full `[vocab]` logits → grammar masks invalid tokens → sample.
Proposed: grammar tells us the valid set *before* the forward pass → we use that knowledge to shrink (or skip) the forward pass.

Two distinct surgeries, with very different leverage:

| Trie state at step | Today | Proposed |
|---|---|---|
| 1 valid token (forced) | full forward + mask + sample | **emit token, batched-append to KV, no forward** |
| k valid tokens, k small | full forward + mask + sample | **forward with sparse LM head (k rows only)** |
| unrestricted | full forward + sample | unchanged |

The forced-token case is where the big wins live — JS DSLs like the order-mgmt prompt are 40–60% boilerplate (`await `, `const x = `, `({ `, `});`). The sparse-head case is smaller but is the right thing to land first because the wiring it requires (`grammar → recipe valid_indices`) is also what forced-token elision needs.

---

## Architectural framing

**This is not "model surgery."** The model is unchanged. Both tricks live at the **sampling ↔ recipe** seam:

- `sampling/` already owns the TokenTrie, which already maintains the valid set per slot.
- `models/` (the recipe) already builds the LM head; it just builds it dense today.

What changes is the **direction of the dependency at decode time**: today it's forward → grammar; we add a backward edge grammar → forward(valid_indices). No layer module touched. No new ggml op. Generalizes to any grammar-constrained workload (not just this DSL).

**Constraints we will respect:**

- No `if (domain == "order_mgmt")` anywhere. The grammar is data; the engine handles "constrained decoding" as a general capability.
- KV append vs recurrent overwrite stays distinct (per CLAUDE.md). Forced-token elision touches KV append only; recurrent state still runs its update on every step regardless.
- No layer module gains a sampling-aware code path. The optimization is contained to recipe + sampling + decode loop.

---

## Phase A — Sparse LM head (the warm-up)

**What changes mathematically:**

```cpp
// dense (today, forward_pass_base.cpp:104-108)
cur = ggml_mul_mat(ctx_, output_weight, cur);            // [vocab]

// sparse (new path)
W_k     = ggml_get_rows(ctx_, output_weight, valid_idx); // [k, hidden]
logits_k = ggml_mul_mat(ctx_, W_k, cur);                 // [k]
```

`valid_idx` is a `[k]` int32 input tensor populated host-side before `ggml_graph_compute`. Its size changes per step — that's the dynamic-shape wrinkle ggml dislikes inside a single persistent graph.

**Build per step, with a branch in the head builder.** The decode pipeline already does `reset_context()` and rebuilds the decode graph on every step, so we don't need (and can't easily have) two persistent graphs co-resident in one context. Two pre-built graphs in two separate contexts is structurally heavier (doubled scratch buffers, doubled scheduler bindings) and buys nothing here, because no other part of the engine wants a persistent decode graph. A `k_max`-padded static graph wastes memory on every step and hides the conditional inside the graph — the kind of cleverness CLAUDE.md tells us to refuse.

The clean shape:

- The **decode loop** computes `valid_indices` if the grammar narrows below threshold (e.g. `k < vocab/8`); passes `nullptr` otherwise. Policy lives here, in one place.
- `build_output_head(gf, cur, valid_indices)` takes the optional tensor. **Mechanism** lives here: if non-null, build the sparse path (`get_rows` then `mul_mat`); if null, build today's dense path.
- One context, one graph at a time, one builder function. No two-graph problem.

The per-step `ggml_backend_sched_alloc_graph` cost is real but small — O(graph walk), single-digit ms at most, against ~50 ms per Qwen 3.6-A3B decode token. Noise.

**Future escape hatch.** If the ANE plan ([plan-ane-lm-head.md](plan-ane-lm-head.md)) ever lands and wants pipelining-friendly persistent graphs, we promote to a two-context design *then*, driven by ANE's actual needs. Not speculatively now.

**Why this matters more on MoE.** On dense Qwen3-32B the LM head is ~1% of forward — barely worth touching. On Qwen3.6-A3B only ~3B of 35B params activate per token, so the body shrinks ~10× and the LM head climbs to ~10–15% of forward. The headline target model is exactly where this earns its keep.

### Files touched (Phase A)

Core math:
- [src/models/forward_pass_base.cpp:100](../src/models/forward_pass_base.cpp:100) — extend `build_output_head(gf, cur)` to `build_output_head(gf, cur, ggml_tensor* valid_indices = nullptr)`. Internally: if `valid_indices != nullptr`, `W_k = ggml_get_rows(...)` then `mul_mat`; else today's dense path.
- [src/models/forward_pass_base.h:161](../src/models/forward_pass_base.h:161) — update the declaration; no new graph-handle state since the graph is rebuilt per step.

Recipe call sites (no per-recipe change required):
- All recipes that call `build_output_head(gf, inpL)` keep working unchanged because `valid_indices` defaults to `nullptr` (dense path).
- The opt-in surface is at the **decode loop**, not the recipe — see below.
- [src/models/qwen36.cpp:311](../src/models/qwen36.cpp:311), [:436](../src/models/qwen36.cpp:436) — primary target for benchmarking (MoE, biggest payoff), but no edit needed.
- [src/models/qwen3.cpp:182](../src/models/qwen3.cpp:182), [src/models/qwen35.cpp:265](../src/models/qwen35.cpp:265), [:779](../src/models/qwen35.cpp:779) — secondary, also no edit needed.

Sampling (produce `valid_indices` *before* the forward pass):
- [src/sampling/token-trie.h](../src/sampling/token-trie.h) / [.cpp](../src/sampling/token-trie.cpp) — add `current_valid_token_ids(slot) -> std::vector<int32_t>`. The trie's bitset already has this; just an enumeration helper.
- [src/sampling/sampling.cpp](../src/sampling/sampling.cpp) — split today's `sample(dense_logits, …)` into:
  - `peek_valid_set(context) -> vector<int32_t>` — called before forward.
  - `sample_sparse(sparse_logits, valid_idx, context)` — maps k-logits back to token IDs.

Decode loop (the selector):
- [src/cli/chat.cpp:142](../src/cli/chat.cpp:142), [:190](../src/cli/chat.cpp:190)
- [src/cli/complete.cpp:89](../src/cli/complete.cpp:89), [:136](../src/cli/complete.cpp:136), [:185](../src/cli/complete.cpp:185)
- [src/server/http_server.cpp:267](../src/server/http_server.cpp:267), [:301](../src/server/http_server.cpp:301)

All three currently do the same three-step dance (run_prefill → tail logits → sample). The selector lifts cleanly into a helper in `src/core/` so the three callers each become a one-liner.

**Note.** The three-way duplication in the decode loop is pre-existing — this plan surfaces it, doesn't cause it. Worth fixing while wiring this through.

### Done criteria (Phase A) — *actual results annotated*

1. ✅ `build_output_head` accepts optional `valid_indices`; sparse path activates when non-null.
2. ✅ Decode loop selects sparse vs dense per step; policy in `decode_step` (one place).
3. ✅ Bit-for-bit identical sampling — verified via `tests/grammar/test_sparse_differential.cpp`.
4. ❌ **Not met: measured 0%, not +5–10%.** Forward pass was not the bottleneck on this workload; `get_valid_tokens` (grammar) dominates. Accepted: Phase A is correctness infra, not a perf feature.
5. ✅ No new graph node on the dense path; no `valid_indices` through layer modules.

> Also surfaced and fixed during Phase A: an uninitialized-sparse-indices
> ordering bug (caught by the differential test). Still **open** from Phase A:
> the CHAR_CLASS too-permissive *correctness* bug — needs a human owner, not
> a perf pass.

---

## Phase B — Forced-token elision (the main event)

> **NOT SCHEDULED (2026-05-16).** Phase B elides the forward pass; the measured
> bottleneck is grammar-side `get_valid_tokens`, which Phase B does not touch.
> Its economics were derived assuming forward dominated — false on this
> workload. Retained below as analysis only. Revisit only if a future workload
> demonstrates forward-pass dominance *and* the grammar cost
> ([plan-resolve-once.md](plan-resolve-once.md#superseding-work-the-real-todo))
> is already addressed.

**The mechanism.** When the TokenTrie state has exactly one valid token, the next token is determined. We:
1. Emit that token to the output stream.
2. Append it to the KV cache via a batched mini-prefill (potentially across multiple consecutive forced tokens).
3. Update the trie state.
4. Resume normal decode at the next *branching* state.

No forward pass for forced positions. The recurrent state (DeltaNet) still needs to advance for those positions because its state mutates — so a forced run becomes a "prefill of length r" on the recurrent layers, with the LM head skipped entirely (we're not predicting; we already know the tokens).

**Why this is harder than Phase A.** Phase A is one matmul. Phase B is a control-flow change in the decode loop and a new interaction with `state/`:
- The recurrent state must support "advance by r tokens without producing logits."
- The KV cache must support "append r tokens in one batched dispatch."
- Both probably exist for prefill already — Phase B's job is to reach into prefill-mode for runs of length r mid-decode.

**Why it composes with Phase A.** At a forced position, Phase B skips the work. At a branching position with small k, Phase A makes the work cheap. Different stretches, different optimizations, same seam.

### Files likely touched (Phase B, sketch only)

- `src/sampling/token-trie.{h,cpp}` — add `forced_run() -> vector<int32_t>` returning all consecutive forced tokens until the next branching state.
- `src/state/` — verify the KV cache and DeltaNet state both expose a "prefill r tokens" entry point usable mid-decode; if not, that's the load-bearing refactor.
- `src/core/` decode helper — orchestrate: while trie says forced, batch-advance state, emit tokens, no LM head.
- `src/cli/*`, `src/server/*` — call the new helper; no change to their structure beyond Phase A.

This phase is genuinely architectural — it's the stress test for whether the sampling↔state seam is as clean as it looks today. If `state/` cannot host "prefill r tokens during decode" without leaking into layer modules, that's an interface defect worth fixing before adding the feature.

### Done criteria (Phase B)

1. Forced-token elision verified bit-for-bit against a non-elided run (same output sequence, just produced faster).
2. No `if (grammar)` branches inside any layer module.
3. `state/` exposes its prefill entry points cleanly; KV append vs recurrent overwrite stay distinct.

---

## What we are explicitly *not* doing

- **Not distilling a specialist model.** Out of scope for an inference engine.
- **Not baking any specific grammar into the engine.** The order-mgmt DSL is a benchmark, not a hard-coded path.
- **Not unifying KV cache and recurrent state.** CLAUDE.md is explicit; this plan respects it.
- **Not optimizing layer modules for constrained decoding.** Layer code stays unaware.

---

## Order of operations

> **What actually happened (2026-05-16):** Phase A landed (steps 1–2 done,
> incl. the `src/core/decode_step` lift). The `<3%` gate at step 4 tripped
> (measured 0%). Per this section's own instruction, we paused and rethought
> the seam instead of starting Phase B — that investigation found the
> bottleneck is grammar-side, not forward-side, and the leverage was a grammar
> rewrite. Phase B (step 3) was correctly **not** started. Remaining work
> moved to [plan-resolve-once.md](plan-resolve-once.md#superseding-work-the-real-todo).

1. ✅ **Phase A first.** Done. Wiring proven; perf gate failed (accepted).
2. ✅ **Lift the decode-loop duplication** into `src/core/decode_step`. Done.
3. ⛔ **Phase B second** — NOT started. Gate at step 4 tripped; bottleneck is
   not the forward pass. Shelved as analysis.
4. ✅ Re-measured at the boundary. Result: 0% → pause-gate honored.

The pause-gate below did its job: Phase A's win landed at 0%, we paused and
rethought the seam rather than committing to Phase B. That is the intended
behavior, working as designed.

---

## Outcomes (2026-05-16)

Final accounting for this investigation.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Phase A sparse LM head | **Landed, 0% perf** — correctness infra only | Differential test byte-identical; tok/s flat (forward not the bottleneck) |
| Phase B forced-token elision | **Not started** | Bottleneck is grammar-side; Phase B elides forward |
| Prompt-trie value slots | **Killed by measurement** | Value slots = 0.36% of decode |
| Per-parse variable symbol table | **Killed by measurement** | Variable-site identifier scan <1% of grammar cost |
| **Close func/field identifier sets in GBNF** | **✅ The real win** | −66% grammar cost, +39% tok/s (spec-following model); grammar-only, no engine code |

**Where the cost actually is:** ~54% of decode is `get_valid_tokens`. Residual
after schema-close splits into cross-call recompute (50–75%, the resolve-once
*cache* — distinct from the already-implemented per-call pre-resolve) and
per-call vocab-sized bitmask churn (~20–40%). Both tracked in
[plan-resolve-once.md → Superseding Work](plan-resolve-once.md#superseding-work-the-real-todo).

**Open, needs a human owner (not optimization):**
1. CHAR_CLASS too-permissive **correctness** bug from the Phase A commit — still
   parked as an inline comment in `grammar_vocab.cpp`.
2. **Q2-vs-9B product decision** — closing the schema set converts the weak
   shipping model's soft failures (slightly-wrong code) into hard failures
   (forced-EOS, no output). Product call, separate owner.

**Method note.** Five hypotheses, each decided by one measurement before
committing code. The two that would have been the most engineering work
(prompt-trie, symbol table) were both killed by a one-afternoon profile. The
recurring failure mode throughout: trusting an *aggregated* cost bucket. Every
correct decision came from splitting the bucket at the right granularity first.

---

## Related

- [docs/plan-grammar-trie.md](plan-grammar-trie.md) — the TokenTrie this plan builds on.
- [docs/plan-resolve-once.md](plan-resolve-once.md) — orthogonal grammar-side optimization; both target the same workload but optimize different layers of the stack.
- [docs/phase4-investigation.md](phase4-investigation.md) — measured cost breakdown that contextualizes "what fraction of forward is the LM head."
- [docs/future-work.md](future-work.md) — conversation branching entry; unrelated but adjacent (also a sampling/state-seam feature).
