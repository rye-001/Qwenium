# Does norm-weighted attention reorder Gemma 4's candidate space? — INERT (2026-09-04)

**Verdict: INERT.** Spearman(rankA, rankB) over all **768** (layer, head)
candidates = **1.0000**, exactly — not merely rounded: 0 of 768 candidates
differ on top1 count, 0 of 768 differ on top3 count, under a scoring pass
that searched the full candidate space directly on the 15-doc messy corpus
(closing the Prompt-A-selection defect at the same time). Metric B does not
reorder Gemma 4's candidate space either, and this null is *stronger* than
Qwen's: on Qwen (`note-lens-norm-weighted-metric.md`) top-1 moved by
0.8pp out of 413; here it moves by **zero** tokens out of 397, on every one
of 768 candidates. The reason is structural, not statistical: Gemma 4
RMS-normalizes every V vector with **no learned scale** before it is written
to the KV cache (`gemma4.cpp`: `Vcur = ggml_rms_norm(ctx0, Vcur, eps)`), and
an unweighted RMS-norm forces `‖V_j‖ = sqrt(head_dim)` **exactly**, for every
position, every head, every layer, by construction. Verified directly:
`cv=0.0000%` on both a sliding-layer head (mean 16.0000 = √256) and a
global-layer head (mean 22.6274 = √512). Metric B's `alpha_j · ‖V_j‖`,
renormalized, is therefore **bit-identical** to Metric A's `alpha_j` on this
architecture — not approximately, provably. `note-lens-norm-weighted-metric.md`'s
open question — "does the correction do anything in the diffuse regime it
was designed for?" — is answered here as decisively as a probe can answer it:
no, and the mechanism is now known, not just the number.

**No usable head either way**, which is the leg's other, expected finding
(pre-committed framing: this leg's job was to characterize Metric B, not to
find a citation head). Scored directly on the messy corpus, the best
candidate under **both** metrics is **L7H13** (top1 44.1%, top3 63.0%) — a
**different** head from L4H7, the head `note-lens-gemma4-probe.md` selected
via Prompt A and only then confirmed against this same corpus. L4H7 now
ranks **#3** (161/397 top1, 204/397 top3 — reproducing that note's §6 numbers
almost exactly, 40.6%/51.4% vs its reported 41%/51%, confirming the plumbing
here is correct, not merely similar). **0 of 768 candidates clear 70%, 80%,
or 90% top3 under either metric.** The two-signal defect this leg was built
to close (§2 of that note: Prompt-A selection is a four-way tie at 75.0% with
almost no discriminating power) is closed, and closing it changes *which*
head looks best without changing the conclusion: Gemma 4 has no citation
head under raw attention, and none under norm-weighted attention either.

**Date** 2026-09-04 · **Status** measurement note; no shipped code touched;
`LensConstants` (`src/server/server_lens.h`) and the architecture refusal
(already refuses `gemma4`) are unchanged, as instructed. `src/` is untouched —
everything below lives in `tests/perf/attn_provenance.cpp`, additive only.

## 0. The question

`note-lens-norm-weighted-metric.md` tested Metric B (`score_j = alpha_j ·
‖V_j‖`, renormalized) against Metric A (`score_j = alpha_j`) on Qwen and came
out NEUTRAL — but Qwen's citation head is near one-hot (median peak ~0.93),
and norm-weighting is theoretically most useful when attention is *diffuse*.
Gemma 4's best head tops out at 63% top3 mass and never had a >90% candidate
at all (`note-lens-gemma4-probe.md`), i.e. it lives in the diffuse regime the
Qwen null could not rule out. This leg re-runs the same paired A/B citation
scoring on Gemma 4, over the full candidate space, scored directly on the
messy corpus rather than selected on short prompts first.

## 1. Provenance

| | |
|---|---|
| Model | `models/gemma-4-12B-it-Q8_0.gguf`, arch `gemma4`, 48 blocks, Q8_0 dense |
| Attention layers | all 48 materialize `kq_soft` (graph-scan discovered, not assumed) |
| Heads | `n_head`=16 (uniform query-head count across both layer kinds) |
| **Candidates** | **48 x 16 = 768**, every one scored under both metrics in one pass |
| KV cache split | **two disjoint caches** — sliding: 40 layers, 8 KV heads, head_dim 256 (group 2); global: 8 layers, **1 KV head** (MQA), head_dim 512 (group 16) |
| BOS | `add_bos_token=true`, id 2, text `"<bos>"` (5 bytes) — applied, confirmed in the run log |
| Chat template | `Gemma4ChatTemplate` (selected by `g_arch`, `qdocs_chat_prompt`) |
| `--flash-attn` | off throughout (required — `kq_soft` never materializes under flash) |
| Corpus | Leg C messy corpus, 15 docs EN+DE, same grammar/task string as `run_qdocs_leg_c` — 397 scored value tokens (EN 209, DE 188) |
| Driver | `tests/perf/attn_provenance.cpp` → `build-release/bin/attn-provenance`, env `GEMMA4_SEARCH_DUAL=1` (new leg, additive) |
| Raw log | `.session-results/gemma4_search_dual.log` |
| Process hygiene | model process exited cleanly; confirmed via `ps aux` — no `attn-provenance` process remained after either run |

## 2. Two traps this leg found that the brief did not fully anticipate

The brief warned "on gemma4 all 48 blocks are attention so the map is
probably identity — derive it, do not assume it." That warning undersold the
actual shape of the problem:

- **Gemma 4 does not have one KV cache — it has two, with different shapes.**
  `gemma4.h`/`gemma4.cpp`: sliding layers keep a separate V projection
  (head_dim 256, 8 KV heads); global layers have **no `attn_v.weight` in the
  GGUF at all** — V is aliased from K (`Vcur = is_global ? Kcur : ...`), and
  reuse **1 KV head** (true MQA) at head_dim 512. `ForwardPassBase::snapshot_kv_cache()`
  (the accessor every other leg in this file calls) returns **nullptr** on
  gemma4 — the recipe overrides the multi-cache `snapshot_kv_caches()`
  instead. The per-layer kind (sliding vs global), each layer's index within
  its own cache, and each cache's own `(n_kv_heads, head_dim)` are not exposed
  by any generic accessor; they are re-derived here read-only from
  `meta.tensor_inventory` / `meta.raw_kv` (presence of `blk.<il>.attn_v.weight`
  decides the kind), reproducing `Gemma4Config::from_metadata`'s own
  derivation without including `gemma4.h`. The two caches
  `snapshot_kv_caches()` returns are matched to (global, swa) by **shape**
  (layer count + V `ne[0]`), not by the order the header documents, since
  trusting an undocumented-to-this-file return order is exactly the kind of
  assumption the cache-layer trap warns against.
- **GQA group size is not uniform across the model.** Sliding layers group
  16 query heads into 8 KV heads (group 2); global layers group all 16 into
  1 (group 16, true MQA). A single global `group = n_head / n_head_kv`
  constant — the pattern every prior leg in this file used, correctly, on
  single-cache Qwen models — would have been silently wrong for 8 of the 48
  layers here. Group size is derived per-slot from the resolved layout.

Neither trap changes the verdict (see §4), but either one, gotten wrong,
would have produced a plausible-looking wrong number under Metric B without
any crash — the exact failure mode the brief called the most dangerous part
of the leg.

## 3. Why Metric B is a structural no-op here (not just an empirical one)

`gemma4.cpp`'s attention block RMS-normalizes V with **no learned weight**
right before it is written to either cache:

```
Vcur = ggml_rms_norm(arena_.ctx(), Vcur, config_.rms_norm_eps);
```

An unweighted RMS-norm divides every vector by its own RMS, which by
definition forces the output's L2 norm to `sqrt(head_dim)` **exactly** (up to
floating-point rounding) — not approximately, for every position, because
that identity falls straight out of the RMS-norm formula and does not depend
on what the input values were. Read directly off the cache for the first
document (two representative layers — the search's #1 candidate's sliding
layer, and the first global layer):

| slot | kind | kv_head | n | min | mean | max | sd | cv |
|---|---|---|---|---|---|---|---|---|
| 7 (L7, winner's layer) | sliding, head_dim 256 | 6 | 442 | 16.0000 | 16.0000 | 16.0000 | 0.0000 | **0.0000%** |
| 5 (first global layer) | global, head_dim 512 | 0 | 442 | 22.6274 | 22.6274 | 22.6274 | 0.0000 | **0.0000%** |

`16.0000 = sqrt(256)`, `22.6274 = sqrt(512)`, to four decimal places, with
zero measured spread. Since Metric B is `alpha_j · ‖V_j‖` renormalized over
`j`, and `‖V_j‖` is the *same* constant for every `j` in a row, the
renormalized quantity is `alpha_j · c / (c · sum_j alpha_j) = alpha_j`
exactly — Metric B **cannot** differ from Metric A on this architecture, for
any head, any layer, any corpus. The empirical zero-differences result (§4)
is the necessary consequence of this, not an independent confirmation of it —
but it is confirmed anyway, from real generations rather than from the
formula alone, which is what closes the loop.

This also settles the Qwen-vs-Gemma "regime" question the brief raised: it
is not that Gemma's diffuse attention makes norm-weighting matter more, as
theory alone would suggest. Whether V is renormalized before caching is an
**architecture** property (Gemma 4 does it, unweighted; Qwen's attention
build has no equivalent op on V), and that property alone determines whether
Metric B can possibly say anything Metric A doesn't — independent of how
peaked or diffuse the attention itself is.

## 4. The search — all 768 candidates, scored directly on the messy corpus

No Prompt A/B/C anywhere in this leg — every candidate is scored against the
same 15-doc EN+DE messy corpus and grammar/task `run_qdocs_leg_c` uses, 397
value tokens (EN 209, DE 188).

| | Metric A (raw alpha) | Metric B (alpha·‖V‖) |
|---|---|---|
| **best candidate** | **L7H13** | **L7H13** (identical) |
| top1 | 175/397 (44.1%) | 175/397 (44.1%) |
| top3 | 250/397 (**63.0%**) | 250/397 (**63.0%**) |
| best candidate's rank under the *other* metric | rank 1 | rank 1 |
| candidates clearing top3 ≥70% | **0** / 768 | **0** / 768 |
| candidates clearing top3 ≥80% | **0** / 768 | **0** / 768 |
| candidates clearing top3 ≥90% | **0** / 768 | **0** / 768 |
| top1 differs from the other metric, of 768 | 0 | 0 |
| top3 differs from the other metric, of 768 | 0 | 0 |
| **Spearman(rankA, rankB), all 768** | **1.0000** | (symmetric) |

Top 10 under Metric A (identical, position for position, to top 10 under
Metric B — reproduced here once):

| rank | layer | head | top1/397 | top3/397 |
|---|---|---|---|---|
| 1 | 7 | 13 | 175 (44.1%) | 250 (**63.0%**) |
| 2 | 6 | 5 | 175 (44.1%) | 205 (51.6%) |
| 3 | **4** | **7** | 161 (40.6%) | 204 (51.4%) ← the note-lens-gemma4-probe.md head |
| 4 | 7 | 12 | 149 (37.5%) | 220 (55.4%) |
| 5 | 39 | 0 | 144 (36.3%) | 178 (44.8%) |
| 6 | 5 | 3 | 132 (33.2%) | 191 (48.1%) |
| 7 | 16 | 9 | 131 (33.0%) | 171 (43.1%) |
| 8 | 41 | 14 | 130 (32.7%) | 188 (47.4%) |
| 9 | 45 | 12 | 130 (32.7%) | 173 (43.6%) |
| 10 | 47 | 0 | 129 (32.5%) | 213 (53.7%) |

**The L4H7 reproduction (the BROKEN check).** L4H7 was the head
`note-lens-gemma4-probe.md` selected via Prompt A and reported at
161/397 (41%) top1, 204/397 (51%) top3 on this same messy corpus. This run's
Metric A, same corpus, same grammar, same task string, reproduces it as
**161/397 (40.6%) top1, 204/397 (51.4%) top3** — the raw counts match
exactly (161 and 204); the percentages differ only in rounding (41%→40.6%,
51%→51.4%, both `round()` to the note's reported integers). That confirms
this leg's citation plumbing is correct, not merely close, so the metric
comparison above is trustworthy: **verdict is INERT, not BROKEN.**

L7H13 outranks L4H7 here because L4H7 was selected on a procedure this note
already documented as weak (§2 of `note-lens-gemma4-probe.md`: a four-way tie
at 75.0% on Prompt A). Scoring directly on the messy corpus surfaces a better
candidate — L7H13 at 63.0% top3 vs L4H7's 51.4% — but "better" here still
means "0 of 768 candidates clear even the 70% floor," so the practical
conclusion is unchanged: no head, either constant.

## 5. What this leg does NOT claim

- **Not a new Gemma 4 lens candidate.** L7H13's 63.0% top3 is the best number
  in the entire 768-candidate space and is still 27 points under the 90% bar
  `note-lens-gemma4-probe.md` set (and 34 points under Qwen 3.8's 98%). This
  does not reopen the "pursue a Gemma 4 lens" question that note closed.
- **Not a coverage result.** Per the brief, coverage is explicitly out of
  scope for this leg (Gemma's coverage constants are uncalibrated — see that
  note's §5.3 caveat). Only citation was scored.
- **Not evidence that norm-weighting is worthless in general.** The null here
  is specific to architectures that RMS-normalize V with no learned scale
  before caching. An architecture whose V is *not* renormalized (Qwen's
  attention build has no such op) is where the Qwen leg's — separately
  neutral, for a different, purely empirical reason — result actually applies.
  The mechanism identified in §3 is falsifiable and architecture-specific, not
  a general claim about `‖V_j‖`.

## 6. What was NOT done

- **No coverage arm** — out of scope per the brief; Gemma's coverage
  constants are uncalibrated (`note-lens-gemma4-probe.md` §9).
- **No images.** Both taps are closed
  (`docs/note-image-lens-probe.md`, `docs/note-image-prefill-tap-probe.md`).
- **No 26B A4B run** — known-incoherent on probe prompts
  (`architecture.md` §12), would confound the generation quality this leg's
  citation scoring depends on. Only the dense 12B-it was used.
- **No `‖W_o(alpha_j V_j)‖`** — the output-projection-weighted refinement
  named in the Qwen note as the next rung. Not attempted here either; the
  §3 mechanism suggests it would face the same `‖V_j‖`-is-constant obstacle
  at the value-norm stage, though `W_o` could in principle reintroduce
  variation `‖V_j‖` alone cannot — not tested.
- **The `‖V_j‖` = constant spot-check (§3) was measured on 2 of 48 layers,
  1 of 15 docs.** This is deliberate, not a coverage gap: the invariant
  follows from the RMS-norm formula itself (every input vector is divided by
  its own RMS, which is definitionally true regardless of the vector's
  values), so it holds for every layer, head, position, and document by
  construction — the spot-check confirms the *implementation* matches that
  formula, not that the invariant might only sometimes hold. The zero-diff
  result over all 768 candidates (§4) is the exhaustive confirmation.
- **`LensConstants` unchanged; the server architecture refusal untouched.**
- **No `src/` changes.** Everything in this leg is additive to
  `tests/perf/attn_provenance.cpp` (two new small readers, two new no-alloc
  top-3 scans, the Gemma4 KV-layout/cache resolvers, and the search driver
  itself, gated behind `GEMMA4_SEARCH_DUAL=1`) — every existing leg in that
  file, on every architecture, is untouched and byte-inert to this change.
