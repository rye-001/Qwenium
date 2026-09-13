# Plan — coverage layer × threshold search (COVSEARCH)

**Status: SPEC.** Probe only. No engine change, no format change, no constant
moves until the gate in §7 passes.

## 1. What is stale — and the incumbent's real provenance

`LensConstants::coverage_used_peak = 0.705` and `coverage_layer = 11` were frozen
from **COV1 on Qwen 3.6** and carried whole onto Qwen 3.8
(`src/server/server_lens.h:152`). They are the lens's weak arm on **both**
models:

| model | used spans clearing 0.705 | bar |
|---|---|---|
| Qwen 3.6-35B-A3B | 84% (63/75) | ≥90% |
| Qwen 3.8-9B | 87% (65/75) | ≥90% |

**Correction to the standing note.** `server_lens.h:158` and `architecture.md`
both say the constant is carried "until a real coverage-layer search runs",
implying none has. One did. `run_coverage_probe`
(`tests/perf/attn_provenance.cpp:1581`) sweeps **every source × every scalar** —
`for (s = 0..11) for (sc = 0..3)` — and takes the argmax of calibration
accuracy. `note-attn-coverage-probe.md` §3 records the result: layer 11 × peak
scored **100% on 12 calibration spans**, beating L3H13's 92%, and then went
11/11 on the 11 held-out spans.

So the defect is not a missing search. It is **a 48-candidate argmax scored on
12 spans**, which is exactly the shape that produces a perfect calibration score
and does not survive contact with a larger corpus — and on Leg C's 75 spans it
does not: 84% / 87%. Several other candidates sat at 92% on those same 12 spans,
one point of separation being a single span. Nothing in that record establishes
that layer 11 is better than its neighbours; it establishes that layer 11 won a
coin-flip-scale margin on a sample too small to resolve one.

That reframes this probe. It is not "run the search nobody ran." It is **re-run
the selection with the three things the original lacked**: an evaluation corpus
that took no part in choosing (Leg C, 6× larger), a threshold-free selection
criterion (AUC, not calibration accuracy), and a stability requirement that a
lone spike cannot pass.

It is cheap for a specific reason: `span_scalars()`
(`tests/perf/attn_provenance.cpp:1422`) **already computes the span peak for
every tapped attention layer in one pass**. The shipped coverage metric reads
`sc[1 + L11_SLOT][0]` and throws the other 7–9 layers away. Recording the whole
vector instead costs no extra forward passes.

The search space is small and fully enumerable. Both calibrated models put
attention every 4th block (`qwen35.h:62`), so the candidate layers are:

- Qwen 3.8-9B (33 blocks, 32 main): 8 attention layers — 3, 7, 11, …, 31.
- Qwen 3.6-35B-A3B (40 main): 10 attention layers — 3, 7, 11, …, 39.

Ten candidates, not a hyperparameter space.

## 2. The metric under search

Unchanged in shape, deliberately. For a span `[lo,hi]` and a tapped layer:

    span_peak(layer) = max over decode steps
                       ( max over heads ( sum of attention mass on [lo,hi] ) )

This is `span_scalars` scalar index 0 (`peak`). The other three scalars (`mean`,
`maxsingle`, `hitrate`) are **reported as free columns but are not part of the
selection**. Searching 10 layers × 4 shapes against 23 hard-labelled spans is
precisely the 48-candidate argmax that produced the incumbent, and repeating it
with a different random tiebreak would teach us nothing. Holding the shape fixed
cuts the candidate count by 4× at no measurement cost, since `span_scalars`
computes all four regardless. If a non-peak shape dominates at *every* layer —
a pattern, not an argmax — that is a finding to schedule separately.

## 3. The two-sided framing the shipped constant never got

`coverage_used_peak` has only ever been scored on one side — the fraction of
**used** spans that clear it. That rate is trivially maximised by lowering the
threshold, so it cannot select anything on its own. The omission report makes
two errors, and they move in opposite directions:

- **False omission alarm** (`peak < thr` on a span the model *did* use): we tell
  the user a line was skipped when it was consulted. This is the harmful one —
  it is a confident wrong claim — and the 13%/16% miss above is exactly this.
- **Silent miss** (`peak ≥ thr` on a span the model ignored): a genuinely skipped
  line is absent from the omission report. Costless-looking, and it is what a
  lowered threshold buys the first error with.

So selection is on **AUC**, which is threshold-free and two-sided. Thresholds are
picked only after a layer is fixed.

## 4. Corpora — and why two of them

**COV1** (`cov_calib()` + `cov_held()`) is the only corpus with an explicit
USED-vs-DROPPED label on a span the model was actually asked about. It is also
tiny: **12 + 11 = 23 TARGET spans**. Its negatives are *hard* — a span in play
that the model declined to use.

**Leg C messy corpus** (`qdocs_messy_corpus()`, 15 docs / 75 labelled fields)
has no DROPPED label, but it supplies the two populations the product actually
sees:

- positives — value spans of fields the model emitted (**the exact 75-span
  population the ≥90% bar is quoted on**);
- negatives — **filler** body spans carrying no labelled value, i.e. the ambient
  bulk of a document, which is what the omission report is mostly ruling on.

Filler is an *easier* negative than COV1's DROPPED, and the two numbers must
never be pooled or compared to each other. Reporting both is the point: COV1
says whether the layer separates the hard case, Leg C says whether it survives
at product scale on a corpus that took no part in choosing it.

## 5. Arms

**Arm 1 — separation curve (COV1).** One pass over `cov_calib()` + `cov_held()`,
recording the full per-layer peak vector for every TARGET span. Offline: AUC per
layer, plus per-class medians (FILLER / DROPPED / USED / VALUE anchor).

**Arm 2 — ambient curve (Leg C).** One pass over the 15 messy documents,
recording the per-layer peak vector for every emitted field value span and for a
matched filler span per document. Offline: AUC per layer, and the used-clear and
filler-clear rates.

**Arm 3 — out-of-sample gate.** Fix the layer on Arm 1's AUC. Fix the threshold
on Arm 1's calib split only. *Then* report Arm 2's rates at that pair. Arm 2 took
no part in either choice, so this number is honest in a way the ~4-point
recalibration was not. The frozen `(layer 11, 0.705)` runs as the control in the
same table.

## 6. Selection rule, declared before the run

1. Choose the layer with the highest **held-split AUC** on Arm 1.
2. **Require a plateau.** The chosen layer and at least one adjacent attention
   layer must both beat the median layer's AUC. A lone spike on 23 spans is
   noise, and the honest report of that outcome is *"no layer is distinguishable
   at this sample size"* — which keeps 0.705 and is a perfectly good result.

   *Amended after the first run (§10.1):* this rule also fails when the metric
   **saturates** — if half the layers tie at AUC 1.000, nothing can strictly beat
   the median. That is the same underlying verdict (the corpus cannot select a
   layer) reached by a different route, and the report must say which of the two
   occurred rather than calling a saturated field "a lone spike".
3. Only then pick the threshold, on the **calib split alone**, by best accuracy.
4. The gate in §7 is scored on Arm 2, which was not consulted in 1–3.

## 7. Gate

A new `(layer, threshold)` ships only if **all** hold on Arm 2, on both
calibrated Qwen models:

- **G1** used-clear ≥ **90%** (the standing bar; 84% / 87% today).
- **G2** filler-clear no worse than the frozen constant's on the same run —
  the used-clear gain must not be bought entirely with silent misses.
- **G3** the layer passed the §6.2 plateau test.
- **G4** the citation arm is **unchanged**. Coverage and citation read different
  layers; this probe must not move `citation_layer`, and the run reproduces the
  known citation numbers as its propagation check.

Failing any of these keeps 0.705 and layer 11, and the finding is recorded as a
negative result.

## 8. Run

Head selection is not automatic. `ATTN_FROZEN_SLOT`/`ATTN_FROZEN_HEAD` default to
**L3H13**, the Qwen 3.6 constants; on Qwen 3.8 those are the known-failing case
(84% top-3, `note-lens-qwen38-probe.md` §5.2). Qwen 3.8 needs slot 6.

```
cmake --build build-metal --target attn-provenance
# Qwen 3.8-9B  (attention layers 3,7,…,31 — citation slot 6 = L27)
COVSEARCH=1 ATTN_FROZEN_SLOT=6 ATTN_FROZEN_HEAD=13 \
  QWEN36_MODEL_PATH=models/Qwen3.8-9B-Q8_0.gguf build-metal/bin/attn-provenance
# Qwen 3.6-35B-A3B (attention layers 3,7,…,39 — citation slot 0 = L3, default)
COVSEARCH=1 \
  QWEN36_MODEL_PATH=models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf build-metal/bin/attn-provenance
```

`build-metal` only (Release + `GGML_METAL=ON`); `build/` is the CPU-only debug
tree. `pgrep attn-provenance` before launching — a leftover run contends for the
GPU and interleaves into the same log.

## 9. What this probe cannot decide

The labelled population is still 23 hard spans and 15 self-authored documents. A
plateau here is evidence that a layer is better; it is **not** evidence that the
threshold value generalises to documents nobody in this repo wrote. If Arm 1
shows no plateau, that is the expected outcome at this sample size and the
constant stays put.

## 10. Results — Qwen 3.8-9B

| | |
|---|---|
| Model | `models/Qwen3.8-9B-Q8_0.gguf`, arch `qwen35`, 33 blocks, Q8_0 |
| Driver | `COVSEARCH=1 ATTN_FROZEN_SLOT=6 ATTN_FROZEN_HEAD=13`, `build-metal/bin/attn-provenance` |
| Candidates | 8 attention layers — 3, 7, 11, …, 31 (as §1 predicted; incumbent = slot 2) |
| Populations | COV1 calib 12 spans (10 USED) / held 11 (8 USED); Leg C 149 spans (75 positive, 74 length-matched filler) |
| Propagation | used-clear at the frozen constant **65/75 = 87%**, citation **89.3% / 97.8% over 413 value tokens** — both reproduce `note-lens-qwen38-probe.md` exactly, so the instrument is measuring the shipped quantity |

### 10.1 COV1 cannot select a layer — it is saturated

| layer | AUC calib | AUC held | AUC pool | med USED | med DROPPED |
|---|--:|--:|--:|--:|--:|
| L3 | 0.950 | **1.000** | 0.978 | 0.913 | 0.294 |
| L7 | 0.950 | **1.000** | 0.989 | 0.905 | 0.391 |
| **L11** *(incumbent)* | 1.000 | **1.000** | 0.989 | 0.929 | 0.411 |
| L15 | 1.000 | **1.000** | 1.000 | 0.970 | 0.342 |
| L19 | 1.000 | **1.000** | 1.000 | 0.971 | 0.735 |
| L23 | 1.000 | **1.000** | 1.000 | 0.946 | 0.727 |
| L27 | 0.850 | **1.000** | 0.933 | 0.993 | 0.734 |
| L31 | 0.950 | 0.833 | 0.889 | 0.789 | 0.390 |

**Seven of eight layers score a perfect held AUC.** Four also score a perfect
calib AUC. This is the §1 diagnosis confirmed directly: layer 11's original
"100% vs 92%" win was not a signal, it was a tie broken by one span on a corpus
that separates USED from DROPPED at *every depth in the model*. Any of six
layers would have won that argmax on a different draw.

The §6.2 plateau test consequently failed — the median layer AUC is itself
1.000, so no layer can strictly beat it. That is the rule firing correctly (its
job is to refuse a selection the data cannot support) but by degeneracy rather
than by design; the honest statement is not "L3 is a lone spike" but **"COV1 has
no discriminating power over layers at n=23."** §6.2 should say so directly.

### 10.2 Leg C does discriminate — and ranks the incumbent 6th of 8

| layer | AUC Leg C | med value | med filler |
|---|--:|--:|--:|
| L3 | **0.893** | 0.903 | 0.253 |
| L7 | 0.863 | 0.938 | 0.296 |
| **L11** *(incumbent)* | 0.843 | 0.913 | 0.323 |
| L15 | 0.814 | 0.949 | 0.386 |
| L19 | 0.843 | 0.950 | 0.483 |
| L23 | 0.863 | 0.924 | 0.371 |
| L27 | 0.854 | 0.975 | 0.306 |
| L31 | 0.808 | 0.645 | 0.225 |

Layer 11 is not the best coverage layer on the larger corpus; it is mid-pack,
and the spread (0.808–0.893) is narrow enough that no layer is clearly right.

### 10.3 The gate — and why two-sidedness was the load-bearing choice

| | used-clear | filler-clear |
|---|--:|--:|
| incumbent L11 @ 0.705 | 65/75 = **87%** | 18/74 = **24%** |
| candidate L3 @ 0.321 | 74/75 = **99%** | 30/74 = **41%** |

- **G1 used-clear ≥90% — PASS (99%).**
- **G2 filler-clear ≤ incumbent — FAIL (41% vs 24%).**
- **G3 plateau — FAIL** (see §10.1).
- **G4 citation unchanged — PASS** (89.3% / 97.8%, n=413).

**Verdict: keep layer 11 @ 0.705.**

The G2 line is the result worth keeping. A one-sided search — the kind the
incumbent got, and the kind `note-lens-norm-weighted-metric.md`'s "recalibration
is worth ~4 points" also was — would have reported a **12-point** used-clear win
here and shipped it. What actually happened is that the threshold dropped from
0.705 to 0.321 and bought those points by calling 41% of filler spans consulted.
The omission report would have reported fewer skipped lines, most of the
reduction being lines that really were skipped. That is a worse product on a
better-looking number.

So the honest reading of the standing "~4 points are sitting there" claim is:
**they are not sitting there.** They were the same trade, measured on one side.

### 10.4 Matched FPR — the comparison that actually decides, and it inverts §10.3

AUC ranks a layer over every operating point at once; the omission report runs at
exactly one. So pin every layer to the **incumbent's own false-positive rate** —
admit exactly the 18/74 filler spans layer 11 @ 0.705 admits — and ask which
layer then reports the most genuinely-consulted spans. The operating point is
pinned, not fitted, so no layer can buy used-clear with silent misses.

| layer | threshold | filler-clear | used-clear |
|---|--:|--:|--:|
| L3 | 0.691 | 18/74 | 59/75 = 79% |
| L7 | 0.841 | 18/74 | 62/75 = 83% |
| **L11** *(incumbent)* | 0.724 | 18/74 | 65/75 = **87%** |
| L15 | 0.908 | 18/74 | 47/75 = 63% |
| L19 | 0.872 | 18/74 | 57/75 = 76% |
| L23 | 0.740 | 18/74 | 62/75 = 83% |
| **L27** | 0.853 | 18/74 | 69/75 = **92%** |
| L31 | 0.541 | 18/74 | 51/75 = 68% |

**Layer 27 clears the ≥90% bar at zero false-positive cost** — same filler-clear
as the incumbent, five more consulted spans recovered.

**§6.1 was the wrong selection rule, and this table is why.** L3 won Arm 2's AUC
(0.893) and lands second-worst here (79%); L27 was mid-pack on AUC (0.854) and
wins. A criterion that averages over operating points the product never uses
cannot select for the product. The rule should be: **select on used-clear at the
incumbent's pinned FPR.** §6 is superseded by this paragraph.

**L27 is Qwen 3.8's citation layer** (`citation_layer = 27`, L27H13). The
original COV1 note concluded the opposite — that the best coverage source is a
*different* layer from the citation head — and that conclusion came from the same
12 spans §10.1 shows cannot rank layers at all.

**This is a candidate, not a result.** L27 was selected on Leg C and scored on Leg
C. Eight candidates against 75 positives beats the incumbent's 48-against-12 by a
wide margin, but it is still one corpus doing both jobs. Two falsifiers, both
cheap, decide it:

1. **Split-half.** EN (8 docs) and DE (7 docs) are disjoint. A winner that holds
   on ~40 and on ~35 spans independently is not an artefact of which spans we had.
2. **Cross-model.** Qwen 3.6-35B-A3B is the other calibrated entry and the model
   layer 11 was *originally measured on*. The prediction is sharp: if coverage
   wants the citation layer, **L3** wins there — and layer 11 loses on its home
   model.

### 10.5 Split-half — the Qwen 3.8 candidate collapses

| layer | EN used-clear | DE used-clear |
|---|--:|--:|
| L3 | 31/40 = 78% | 26/35 = 74% |
| L7 | 29/40 = 72% | 33/35 = **94%** |
| **L11** *(incumbent)* | 36/40 = **90%** | 29/35 = 83% |
| L15 | 20/40 = 50% | 23/35 = 66% |
| L19 | 28/40 = 70% | 31/35 = 89% |
| L23 | 33/40 = 82% | 25/35 = 71% |
| **L27** | 25/40 = 62% | 30/35 = 86% |
| L31 | 26/40 = 65% | 30/35 = 86% |

**L27's 92% was a pooling artefact.** Re-pinned per half it is 25+30 = 55/75 =
**73%**, nineteen points below its pooled figure, and it wins neither half.
Pooled pinning admits its 18 filler spans *wherever they fall*; for L27 they fall
disproportionately in one language, leaving the threshold lenient for the other.

**The split code proves itself on the incumbent.** L11 scores 36+29 = 65/75 =
87%, identical to its pooled 87% — as it must, since pinning layer 11 to layer
11's own FPR returns 0.705 on any subset. A layer that reproduces exactly under
both poolings while the candidate moves 19 points is evidence the difference is
in the data, not the arithmetic.

**No layer clears ≥90% on both halves on Qwen 3.8.** 75 positives against 8
candidates still cannot select a layer — the incumbent's disease, one order of
magnitude milder.

## 11. Results — Qwen 3.6-35B-A3B (cross-model)

`models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf`, 10 candidate layers (3, 7, …, 39),
default citation slot. Propagation: coverage **63/75 = 84%**, citation **83.7%
over 418 value tokens** — both reproduce `note-lens-qwen38-probe.md`'s Qwen 3.6
reference column (84%, 350/418) exactly.

**COV1 saturation is worse here: all TEN layers score a held AUC of 1.000.**
§10.1 was not a Qwen 3.8 accident.

| layer | matched-FPR used-clear | EN | DE | both halves ≥90%? |
|---|--:|--:|--:|:--:|
| **L3** *(this model's citation layer)* | 71/75 = **95%** | **98%** | **91%** | **yes** |
| L7 | 87% | 82% | 86% | no |
| **L11** *(incumbent)* | 63/75 = **84%** | 88% | 77% | no |
| L15 | 75% | 60% | 91% | no |
| L19 | 67% | 50% | 74% | no |
| L23 | 92% | 95% | 83% | no |
| L27 | 61% | 65% | 60% | no |
| L31 | 67% | 65% | 66% | no |
| **L35** | 74/75 = **99%** | **90%** | **100%** | **yes** |
| L39 | 93% | 98% | 86% | no |

### 11.1 My stability rule was also wrong — twice now

The probe prints DISAGREE because EN's argmax (L3) and DE's argmax (L35) differ.
That verdict is too brittle: **two layers clear the bar on both halves
independently**, which is the thing actually worth knowing. With ten near-tied
candidates the argmax is decided by noise even when the ranking is not. The
criterion should be *"clears ≥90% on both disjoint halves"*, not *"is the
per-half winner"*.

That is the second rule defect this probe has found in itself (§6.1 selected on
AUC, which §10.4 showed is the wrong operating point; §6.2's plateau test
degenerates under saturation, §10.1). Recorded rather than quietly patched: a
selection rule written before seeing data is worth exactly as much as its
falsifiers, and these were caught only because both falsifiers ran.

### 11.2 L3 on Qwen 3.6 is the only candidate that survived anything

Its split sum is 39+32 = 71/75 = **95%**, *exactly* its pooled 95% — **zero
pooling inflation**, against L27's 19-point collapse. It beats the incumbent by
+10 points on English and **+14 on German**, at the incumbent's own FPR.

And it is this model's **citation layer** (`citation_layer = 3`). But that is not
a law: on Qwen 3.8 the citation layer L27 scores 62%/86% and is among the worst.
So "coverage wants the citation layer" holds on one model and fails on the other
— which is what per-model calibration already assumes, and is not evidence for a
mechanism.

**Status: a candidate on Qwen 3.6 only, and not enough to move a constant.** L3
was chosen by reading this table and scored on the corpus that produced it. Ten
candidates against 75 positives, with the per-half check as the only guard. The
honest next step is a document set nobody in this repo wrote — the same
bottleneck §9 named before the run.

### 11.3 The finding that did survive both models: it is a language split

| model | incumbent EN | incumbent DE |
|---|--:|--:|
| Qwen 3.6-35B-A3B | 88% | **77%** |
| Qwen 3.8-9B | 90% | **83%** |

Both models, same direction, and on Qwen 3.8 English **already clears the ≥90%
bar**. Every number that has driven this investigation — 84%, 87%, "the weak arm
on both models" — is a pooled average hiding a German deficit of 11 and 7 points.

This is a different hypothesis from "wrong layer", it is the only one this corpus
supports across both models, and nothing about it points at a constant. It is the
live thread this probe produced.

## 12. The layer and the threshold do not have the same evidentiary status

Worth separating before anything is proposed for landing, because the two halves
of `(layer, threshold)` are not equally supported and it would be easy to ship
them as one decision:

- **The layer is a discrete choice among 8**, and it is falsifiable by exactly the
  two checks above. If L27 wins on EN and on DE and on a second model, that is a
  real finding about where this family's coverage signal lives.
- **The threshold is a continuous value fit on 75 spans of a corpus we wrote.**
  0.853 is whatever pinned layer 27 to the incumbent's FPR *on this corpus*.
  That is the same class of object as the 0.453 that
  `note-lens-norm-weighted-metric.md` produced and this repo declined to ship —
  better supported (75 spans, not 23) but not different in kind.

So a passing result argues for **moving `coverage_layer`, and deriving the
threshold separately and conservatively** — not for landing the pair as measured.
Concretely, the conservative reading of a passing result is: keep the used-clear
gain that comes from the better layer, and do not also spend the FPR budget the
pinning happened to leave. A threshold that holds filler-clear *below* the
incumbent's while still clearing 90% would be the shippable version.

### 12.1 If this lands — the files it touches

Not a change list to act on, a scope check. The constant is asserted in more
places than it is defined:

- `src/server/server_lens.h:87` (the Qwen 3.6 defaults) and `:152` (the Qwen 3.8
  entry's explicit `coverage_used_peak`, plus the comment block recording the
  2026-09-05 "carry 0.705" decision, which this probe discharges).
- `tests/unit/test_server_lens.cpp:710` asserts `coverage_used_peak == 0.705` on
  **every** calibration entry, deliberately, as the guard on that decision.
  Changing a constant means changing the test that encodes the decision — those
  land together or not at all.
- `docs/architecture.md:750` repeats the "until a real coverage-layer search
  runs" line that §1 corrects.

Per CLAUDE.md this is a user decision before it lands, not a cleanup.

## 13. Verdict

**Keep `coverage_layer = 11`, `coverage_used_peak = 0.705`, on both models.**
Nothing measured here is strong enough to move a shipped constant.

What the probe actually produced:

1. **The incumbent's provenance is now known and it is weak** — a 48-candidate
   argmax on 12 spans, on a corpus that §10.1/§11 show cannot rank layers at all
   (7 of 8 and 10 of 10 layers at a perfect held AUC). Layer 11 won a coin flip.
   The "never searched" line in `server_lens.h:158` and `architecture.md:750`
   is wrong and should be corrected to say this instead.
2. **"Recalibration is worth ~4 points" is retired.** §10.3 reproduces the effect
   at *twelve* points and shows it is bought by moving filler-clear 24% → 41%.
   Measured on one side, it is not a win; it is an unpriced trade.
3. **The right operating point for any future coverage work is matched-FPR**
   (§10.4), not AUC and not a bare used-clear rate.
4. **One candidate survives on one model** — L3 on Qwen 3.6, +10 EN / +14 DE at
   the incumbent's own FPR with zero pooling inflation (§11.2) — and it is not
   enough on its own.
5. **The live thread is the language split** (§11.3), which holds on both models
   and is not about layers.

The bottleneck is unchanged and now measured twice: 23 hard spans and 15
self-authored documents cannot select among 8–10 candidates. A corpus this repo
did not write is the prerequisite for any further work on this constant.
