# feed_tokens — Advance State Over Known Tokens, No Prediction

**Contract (one line):** `feed_tokens(span, slot)` advances a slot's model state
— attention KV-append **and** recurrent overwrite (DeltaNet conv + recurrent) —
over a span of already-known tokens, **without building the LM head or
producing logits**. It is the inverse of `run_prefill`-with-logits: consume
tokens to condition future predictions; predict nothing.

This is a **core engine state primitive, not a grammar feature.** Its name,
location, documentation, and tests stand alone. If grammar-guided decode were
deleted tomorrow, `feed_tokens` still belongs. Phase B of grammar-guided decode
is merely its *first consumer*, not its reason to exist.

This doc pins a **contract and a verification obligation**. It makes **no
performance claim** — performance is each consumer's concern. Avoiding
optimistic single-number targets is deliberate; that has been this codebase's
recurring failure mode.

---

## Consumers (first consumer ≠ reason to exist)

| Consumer | Use | Reproducibility need |
|---|---|---|
| Grammar-guided decode, Phase B (first) | Skip forward pass at forced positions; advance state over the forced run | Token-stable sufficient (re-enters normal decode at next branch) |
| Speculative decoding (in progress) | Advance state over accepted draft tokens without re-predicting | Token-stable sufficient (re-verifies) |
| Conversation branching (future-work) | Replay a token span onto a slot's state | **Stronger** — a subtly divergent branch is a user-visible correctness bug |

The consumer table is also the evidence for the per-consumer decision fork
below: these three do not have the same reproducibility requirement.

---

## API and semantics

```
void feed_tokens(const std::span<const int32_t> tokens, uint32_t slot);
```

- Advances state for `tokens` at positions `[cache_pos(slot), cache_pos(slot) + n)`.
- Advances **both** state types per token across the span: attention KV-append
  and recurrent overwrite. The KV-append vs recurrent-overwrite distinction
  stays at the *cache-object* level (CLAUDE.md invariant) — the recurrence
  kernel is **not** forked.
- Builds the graph **head-less**: no `build_output_head`, no `"logits"` node,
  `get_output_logits` is never called.
- Produces no return value. It is a state mutation, not a prediction.

### Seam decision — RESOLVED (stated, not left to the implementer)

This is the CLAUDE.md parameterize-vs-split judgment call, decided here:

- **Public surface = a new, named, independently documented and tested method**
  (`feed_tokens`). Standalone identity.
- **Internal implementation = a `want_logits = false` parameter on the existing
  prefill graph builder.** A thin adapter over the existing prefill path with
  the head pruned. **Not** a separate path. **Not** a forked KV/recurrent
  write. Reimplementing state writes is a defect, not the feature.

Standalone at the API; thin parameterization underneath. The doc states this so
the implementer does not relitigate the seam.

### Head-presence locality constraint

Head pruning must be **one expressible choice per recipe, identical in shape
across all recipes**: exactly one guard point in each of `qwen35`, `qwen36`,
`gemma1`, `gemma2`, `gemma3`, `gemma4` — six recipes, one site each. **Not**
scattered `if (want_logits)` conditionals within a recipe. If the adapter would
introduce multiple guard sites in any recipe, that is a locality-of-reasoning
violation (CLAUDE.md: a change to one recipe must not smear across many sites)
and a design defect to resolve **before** implementation, not after.

Established facts (cited, not re-derived): the head is a prunable tail call in
qwen35/qwen36 (one `build_output_head` call site each) and a short inline block
in gemma1–4 (one site each). State writes are independent graph roots; pruning
the head leaves them intact. `get_output_logits` is the only head consumer.
Position offset is structurally supported (`build_prefill_graph` takes a base
position; multi-turn chat already exercises mid-stream prefill at nonzero
offset). On qwen36, decode *is* `build_prefill(n_tokens=1)` through one
recurrence kernel — there is no separate chunked-vs-sequential path at the
architecture level.

---

## The correctness contract (spine)

For **every recipe** that implements `feed_tokens`: feeding a span onto a
**non-empty mid-decode cache** must produce KV + recurrent + conv state
**bit-for-bit identical** to feeding those same tokens as N sequential
single-token decode steps from the same starting state.

- This is a **standing per-recipe obligation**, not a one-time check. It is
  verified by a mid-decode differential test (reuse the existing
  sparse-differential harness pattern; one parameterized harness, not six
  copies).
- A fresh-prefill-from-position-0 comparison is **insufficient and proves
  nothing**. Mid-stream append onto already-populated state is the entire risk
  surface; the test must start from a non-empty cache.
- "Obviously fine because it's attention-only" (Gemma) is **not** a waiver.
  That reasoning has burned this project before. Each recipe owes its own
  passing differential, Gemma included.

### TurboQuant clause (do not omit)

Under TurboQuant, the decode path that *defines* this contract is the
`run_prefill` compress/decompress bridge — `decode_step.cpp` already routes TQ
decode there because the unified graph has no TQ path. `feed_tokens` onto a
non-empty mid-decode cache under TQ is a **compressed-store write path**.

The doc-as-implemented must do **one** of:

1. **Fold TQ into the contract:** `feed_tokens` reuses the TQ bridge (or proves
   equivalence to it), and the per-recipe differential is run **with TQ
   active** as well as without.
2. **Scope TQ out explicitly:** `feed_tokens` fails loud — asserts / refuses —
   when `tq_active()`, mirroring the existing TQ decode bridge, with the reason
   stated in code and here.

Silent omission is **forbidden**. This is the project's known segfault-class
gap (uninitialized state under TQ); it must be a stated decision, not an
oversight.

---

## The open decision fork (present; owner resolves)

If the recurrent kernel chunks a span internally, `feed_tokens(N)` may diverge
from N × `feed_tokens(1)` in low FP bits **even though it is one kernel**. This
is not yet measured (it is the next probe — the mid-decode differential on
qwen36). If it diverges:

- **(a) Make span-advance bitwise-reproducible** — deltanet-kernel work,
  unbounded relative to the ~85-LOC adapter estimate.
- **(b) Accept divergence if it never flips a sampled token on the workload** —
  relax the gate to token-stable.
- **(c) Bounded tolerance**: state within ε, sampled output byte-identical.

**The answer may not be global — it may be per-consumer.** Speculative decoding
re-verifies and tolerates (b); conversation branching *replays* a span where
subtle divergence is a user-visible bug, pushing toward (a)/(c). A single
engine-wide reproducibility knob may itself be the wrong abstraction. The
owner must explicitly decide whether this is **one guarantee or N
consumer-keyed policies** — assuming the question is singular smuggles in a
default. Do not let "(a) by default" or "(c) by default" ride.

This decision has implications beyond Phase B (speculative decoding,
conversation branching both lean on state reproducibility). It is an owner
decision, documented here when made.

### Measurement (Phases 1–3 complete — fork consciously kept deferred)

Mid-decode differentials run, feeding a 6-token span onto a non-empty
mid-decode cache vs. 6 sequential single-token steps:

| Recipe | Phase | bitwise (a) | `max_abs_diff` | token-stable (b) |
|---|---|---|---|---|
| qwen36 | 1 (`test_qwen36_feed_tokens.cpp`) | FAILS | ≈3.34e-6 | HOLDS (198==198) |
| qwen35 | 2 (`test_qwen35_feed_tokens.cpp`) | FAILS | ≈2.86e-6 | HOLDS (128==128) |
| gemma1 | 3 (`test_gemma_feed_tokens.cpp`) | FAILS | low-bit | HOLDS |
| gemma2 | 3 (`test_gemma_feed_tokens.cpp`) | FAILS | low-bit | HOLDS |
| gemma3 | 3 (`test_gemma_feed_tokens.cpp`) | FAILS | low-bit | HOLDS |
| gemma4 | 3 (`test_gemma_feed_tokens.cpp`) | FAILS | **≈0.126** | HOLDS (135==135) |

qwen35/36: predicted chunked-vs-sequential FP reduction-order divergence in
the DeltaNet/SSM recurrence. gemma1–3: same class via attention-only
KV-append. **Not** adapter defects (state-write path is the unmodified
prefill builder). "Attention-only so it's fine" was not a skip — each
gemma owed and got its own KV-append mid-stream differential.

**gemma4 is the falsifier (CLAUDE.md's designed role).** It is attention-
only for *state* but has a parallel **dense + MoE FFN**: chunk-vs-sequential
FP-order differences cross a top-k expert-selection boundary, producing a
discrete **≈0.126** raw-logit shift that is *still token-stable*. qwen36
(also MoE) stayed at 3e-6 only because its workload didn't flip an expert;
gemma4's did. This invalidated the original tight ε (sized for recurrent
low-bit noise) — see the gate-design decision below.

**Head-guard shape (uniform across recipes — one site each).** Exactly one
`if (want_logits) { …head… }` site per recipe (qwen35, qwen36, gemma1–4).
The head-less graph still needs a scheduler backend-propagation anchor (the
pruned logits node used to be it; without one `ggml_gallocr` aborts on
`buffer_id < 0` — caught on qwen35, qwen36 had survived by luck). qwen35/36
add an explicit `else { build_forward_expand(inpL); set_output(inpL); }`;
gemma1–4 already emit per-layer `ggml_set_output(inpL)`, which serves the
same invariant — so no separate else there. The anchor is numerically inert
(forces compute, not different state) so the differential stays valid.

**Gate-design decision (owner, recorded): token-stable + a COARSE universal
ceiling.** The original tight ε (1e-4, sized for recurrent low-bit noise)
was falsified by gemma4's ≈0.126 token-stable MoE-expert-flip divergence.
The contract that actually holds across all 6 recipes — and the one the
consumer table needs — is **token-stable (option b)**. The per-recipe
`MidDecodeDifferentialTokenStable` now asserts token-stable AND
`max_abs_diff < 1.0` (one constant, all recipes, no recipe taxonomy): a
gross-regression sanity net (wrong positions / corrupt state → O(1)
divergence and/or a token flip), explicitly **not** a precision claim.

**Owner decision (recorded): keep the fork deferred — quarantined, not red.**
The spine differential is split so the deferred fork is a *documented
quarantined signal* instead of a permanently-red suite that would mask
future regressions:

- `MidDecodeDifferentialTokenStable` — **runs and passes** on all 6
  recipes. Asserts token-stable AND the coarse `max_abs_diff < 1.0` net.
- `DISABLED_MidDecodeDifferentialBitwise` — the strict option (a) contract,
  gtest-**disabled** (not red in CI). Carries the per-recipe measurement;
  enable with `--gtest_also_run_disabled_tests` only once the fork is
  consciously resolved toward (a).

**Phases 1–3 complete.** All six recipes (qwen36, qwen35, gemma1–4) ship
the one-site head guard + the split differential; every
`MidDecodeDifferentialTokenStable` passes, every bitwise variant stays
quarantined. The phasing gate ("no recipe proceeds until the prior
differential passes **or the fork is consciously resolved**") was satisfied
at each step by the conscious deferral. Still explicitly unresolved — do
not default: strict bitwise (a); per-consumer vs. global; whether
conversation branching (stronger repro need) forces (a)/(c) when it lands.

---

## Phasing (risk-ordered; each gated on its own passing differential)

1. **qwen36 first** — highest risk. Proves the recurrent path and the
   differential; resolves (or triggers) the decision fork. No further recipe
   proceeds until qwen36's mid-decode differential passes (or the fork is
   consciously resolved).
2. **qwen35** — second; attention + SSM/recurrent, lower risk than the MoE
   hybrid.
3. **gemma1–4** — last. Attention-only: the recurrent questions are moot, but
   each still owes its KV-append mid-stream differential. One recipe at a time;
   "attention-only so it's fine" is not a skip.

Each phase: implement the one-site head guard + the thin driver, run the
recipe's mid-decode differential (with and without TQ per the TQ clause), gate
the next recipe on it passing.

---

## Non-goals

- **No performance target or claim.** Correctness primitive; perf is the
  consumer's concern. Phase B's economics remain separately gated on the
  grammar-cost fix ([plan-resolve-once.md](plan-resolve-once.md#superseding-work-the-real-todo)).
- **Not bound to grammar.** Grammar-guided decode is one consumer.
- **No new state-advancing code path.** Thin adapter over existing prefill,
  head pruned. A forked KV/recurrent write is a defect.
- **No kernel work** unless the decision fork explicitly lands on option (a).
- **No model-zoo expansion.** This primitive is added to recipes already
  supported; it does not justify new model support.

---

## Done criteria

1. `feed_tokens(span, slot)` exists as a named public method; internal impl is
   `want_logits=false` on the existing prefill builder, one guard site per
   recipe.
2. Per-recipe mid-decode differential passes (bitwise, from non-empty cache),
   for every recipe shipped, with TQ handled per the TQ clause.
3. The decision fork is consciously resolved and documented (global or
   per-consumer), not defaulted.
4. No layer-module changes. KV-append vs recurrent-overwrite stay distinct.
   No forked state-write path. Head guard is one site per recipe.

---

## Related

- [plan-grammar-guided-decode.md](plan-grammar-guided-decode.md) — Phase B, the
  first consumer (itself gated on the grammar-cost fix).
- [plan-resolve-once.md](plan-resolve-once.md) — the grammar-cost work that
  gates Phase B's *economics* (independent of this primitive's correctness).
- [future-work.md](future-work.md) — conversation branching, a consumer with a
  stronger reproducibility need.
