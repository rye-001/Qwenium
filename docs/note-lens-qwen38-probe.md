# Does the Qemmi-Lens citation head exist on Qwen 3.8? — yes (2026-09-01)

**Verdict: GO on the citation head. NO-GO / not-yet-measured on coverage.**

Qwen 3.8 has a retrieval head, it is **L27H13**, and it is *stronger* than the
head the shipped constants pin on Qwen 3.6 — 98% vs 84% top-3-in-span on the same
messy bilingual corpus, with a 0% ungrounded false-alarm rate vs 7%. It was
selected on one prompt and confirmed on an independent corpus, so it is not
overfit. A separate and more surprising result: the **unmodified Qwen 3.6
constants** (L3H13 + layer 11 @ 0.705 + body_mass 0.538) applied whole to
Qwen 3.8 reproduce Qwen 3.6's own numbers to within one point on every metric.

This is a go/no-go probe, **not a calibration**, and it changes no shipped code.
The one arm that does *not* clear its bar on either model is coverage
(used-spans clearing 0.705: 87% on Qwen 3.8, 84% on Qwen 3.6, bar 90%) — the
coverage constants were not searched here and should not be assumed.

**Date** 2026-09-01 · **Status** measurement note; `LensConstants` unchanged,
architecture refusal unchanged

A go/no-go probe, not a calibration. The shipped `LensConstants`
(`src/server/server_lens.h`) were measured on **one** model, Qwen 3.6, and
`--attention-lens` is now refused on any other architecture. This note asks the
prior question for a second model: **does Qwen 3.8 have a citation head at all?**
It does not propose constants, and it does not change any.

## 1. Provenance

| | |
|---|---|
| Model | `models/Qwen3.8-9B-Q8_0.gguf`, arch `qwen35`, 33 blocks, Q8_0 |
| Attention layers | 8, at `il` = 3, 7, 11, 15, 19, 23, 27, 31 (24 SSM layers + 1 NextN head block, excluded) |
| Heads | 16 (`n_head_q`) ⇒ **128 (layer, head) candidates**, all captured per decode |
| Driver | `tests/perf/attn_provenance.cpp` → `build-release/bin/attn-provenance` |
| Tap | `ForwardPassBase::set_attention_taps` → `kq_soft.<il>`, `--flash-attn` off |
| Reference model | Qwen 3.6 (`qwen35moe`), where the frozen head is **L3H13** |
| Raw logs | `.session-results/qwen38_n3_search.log`, `.session-results/qwen38_legC_L27H13.log` |

**Scoring** (unchanged from the N3 leg that produced the Qwen 3.6 constants):
teacher-forced decode over labeled prompts with known field values and known
prompt-token source spans; for each scored value token, does the **argmax** of
that head's `kq_soft` row land inside the field's source span (±2 tokens)?
`top1/N` is the in-span hit rate over value tokens.

## 2. Probe defect fixed first

`attn_provenance.cpp`'s `main()` derived its attention-layer list from
`meta.raw_kv.get_uint32("qwen35moe.full_attention_interval")` — a key only a
`qwen35moe` GGUF carries. The probe therefore **could not load any other
architecture**, and was structurally incapable of asking whether the constants
transfer. This is the same defect, with the same fix, as the P1 tap gate
(`tests/unit/test_forward_pass_base.cpp`): attention layers are now discovered by
**scanning the built decode graph** for `kq_soft.<il>`, which is the seam's own
definition and needs no per-family knowledge.

**Qwen 3.6 regression, re-run on the changed code**
(`.session-results/qwen36_regression_after.log`, `ATTN_TAP_SELFTEST=1`): the
graph scan yields `3 7 11 15 19 23 27 31 35 39` — exactly the old `fai`-derived
list (`block_count=41`, `nextn=1`, `fai=4`) — and the tapped rows are live
softmax (`kq_soft.3` dims `[116,1,16,1]`, every head sum = 1.000000). The qwen36
path is behaviourally unchanged.

`ATTN_FROZEN_SLOT` / `ATTN_FROZEN_HEAD` were added so the confirmation legs can
be pointed at a candidate found on another model without editing the file; unset,
behaviour is identical to the committed qwen36 path.

## 3. The search — Prompt A, all 128 candidates

Top 10 of 128, ranked by in-span top1 (N = 28 scored value tokens):

| rank | layer | head | top1/N | in-span | top3/N | bos_mass |
|---|---|---|---|---|---|---|
| 1 | **27** | **13** | 28/28 | **100.0%** | 28/28 | 0.008 |
| 2 | **3** | **13** | 26/28 | **92.9%** | 27/28 | 0.020 |
| 3 | 19 | 12 | 24/28 | 85.7% | 26/28 | 0.041 |
| 4 | 3 | 15 | 22/28 | 78.6% | 26/28 | 0.051 |
| 5 | 7 | 5 | 22/28 | 78.6% | 25/28 | 0.083 |
| 6 | 19 | 10 | 21/28 | 75.0% | 24/28 | 0.025 |
| 7 | 19 | 11 | 21/28 | 75.0% | 21/28 | 0.010 |
| 8 | 31 | 3 | 20/28 | 71.4% | 25/28 | 0.015 |
| 9 | 15 | 13 | 20/28 | 71.4% | 25/28 | 0.049 |
| 10 | 19 | 14 | 20/28 | 71.4% | 23/28 | 0.009 |

**The floor matters more than the peak.** Per-layer *mean* top1 across all 16
heads is 0.23–0.54:

| layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
|---|---|---|---|---|---|---|---|---|
| head-mean top1 | 0.26 | 0.40 | 0.23 | 0.35 | 0.51 | 0.54 | 0.40 | 0.27 |

So 100% is not a lucky draw from a crowd of near-ties — the top of the ranking
stands well clear of the field, which is the signature of an actual retrieval
head rather than noise with a fortunate argmax.

**Two results, not one:**

1. **L27H13 is Qwen 3.8's own best citation head** (28/28 on A).
2. **L3H13 — the shipped Qwen 3.6 coordinates — scores 92.9% on Qwen 3.8, rank
   2 of 128.** That is a surprise and is *not* what the architecture refusal
   assumed. Head index 13 wins on both models; the layer differs.

Treat #2 as a measurement, not a licence: it is one prompt on one model, and
the citation head is only one of the three constants (`coverage_layer`,
`coverage_used_peak`, `ungrounded_body_mass` were **not** probed here).

## 4. Held-out prompts, frozen on L27H13

| leg | corpus | top1 | top3 |
|---|---|---|---|
| Prompt B | held-out, same shape | 25/29 (86%) | 29/29 (100%) |
| Prompt C | reformatted values + date conflict | 27/29 (93%) | 27/29 (93%) |

Prompt C is the informative one: 15/16 in-span on the three **reformatted**
fields (`1.250`←`1.25`, `8,75`←`EUR 8,75`, `10.937,50`), i.e. the head points at
the source even when the emitted token is not a byte copy. The date-conflict
readout also reproduces the Qwen 3.6 behaviour — mass 0.821 on the ORDER span vs
0.048 on the DELIVERY span. The probe's own N3 gate reports **PASS**
(top1 86% ≥ 66.7%, top3 100% ≥ 80%).

Prompts A/B/C are **one corpus family** — teacher-forced, structurally similar
order emails. They are not two signals. §5 is.

## 5. Second signal — the messy corpus (Leg C)

15 messy real-shape documents, EN + DE, free-running extraction, 413 scored value
tokens. This is an independent corpus from A/B/C and is the one that decides the
question.

### 5.1 A probe trap, found and fixed

The first attempt at this leg reported a "L27H13" result that was **not L27H13**.
`ATTN_FROZEN_SLOT` moved `FROZEN_SLOT`, but `qdocs_eval_field` — the function the
QDOCS legs actually call — takes its citation **layer** from a *separate*
constant, `L3H13_SLOT`, and only its **head** from `FROZEN_HEAD`. The override
was a no-op on the layer, the run silently scored layer 3, and the printf label
said "frozen L3H13" regardless. Both defects are fixed (`L3H13_SLOT` now follows
the override, and the label prints the layer/head actually scored).

The mis-run is not wasted — it is the shipped-constants measurement in §5.2.
Recorded because a probe that reports a confirmation it never ran is the exact
failure mode this note exists to avoid.

### 5.2 The shipped Qwen 3.6 constants, applied whole to Qwen 3.8

`.session-results/qwen38_legC_L3H13_ACTUAL.log` — citation **L3H13**, coverage
**layer 11 @ 0.705**, ungrounded **body_mass 0.538**: the entire shipped
`LensConstants` set, unmodified, on Qwen 3.8.

| metric | Qwen 3.6 L3H13 (reference, `note-qemmi-docs-p0.md`) | **Qwen 3.8, same constants** |
|---|---|---|
| citation top3-in-span, EN | 84% (185/219) | 84% (184/218) |
| citation top3-in-span, DE | 83% (165/199) | 83% (162/195) |
| **citation top3-in-span, combined** | **84% (350/418)** | **84% (346/413)** |
| coverage used-spans clearing 0.705 | 84% (63/75) | 87% (65/75) |
| coverage median used-peak | 0.928 | 0.913 |
| ungrounded false-alarm on grounded fields | 7% (5/75) | 4% (3/74) |

**The probe prints `LEG C VERDICT: FAIL`. Read that carefully: the bar it fails
is top3 ≥ 90%, and Qwen 3.6 — the model the constants were calibrated on —
fails the same bar on the same corpus at the same 84%.** The published Qwen 3.6
note says so in as many words. So this is not a Qwen 3.8 failure; it is the two
models landing on top of each other, EN and DE alike, to within one percentage
point on every metric. Qwen 3.8 is slightly *better* on coverage-clearing and on
the false-alarm rate.

### 5.3 Qwen 3.8's own best head on the messy corpus — the confirmation

`.session-results/qwen38_legC_L27H13_real.log` — citation **L27H13** (tap-slot 6),
coverage layer 11 @ 0.705, same 15 documents, same 413 value tokens.

| metric | Qwen 3.6 **L3H13** (its own pinned head) | Qwen 3.8 **L27H13** (its own best head) |
|---|---|---|
| citation top1-in-span, combined | — | 369/413 (**89%**) |
| citation top3-in-span, EN | 84% (185/219) | **98%** (213/218) |
| citation top3-in-span, DE | 83% (165/199) | **98%** (191/195) |
| **citation top3-in-span, combined** | **84% (350/418)** | **98% (404/413)** ✓ bar ≥90 |
| ungrounded false-alarm | 7% (5/75) | **0% (0/74)** ✓ bar <10 |
| coverage used-spans clearing 0.705 | 84% (63/75) | 87% (65/75) ✗ bar ≥90 |

**L27H13 is not overfit to Prompts A/B/C.** It was selected on Prompt A (28/28)
and it holds on an independent, messy, bilingual corpus at 98% top3 — *fourteen
points above* what the pinned Qwen 3.6 head scores on that same corpus. EN and DE
are identical (98%/98%), so this is not a language artifact. That is the two
independent signals the go/no-go asked for.

**About the printed `LEG C VERDICT: FAIL`.** The composite predicate is
`top3 ≥ 90 && false_alarm < 10 && used_clear ≥ 90`. Qwen 3.8/L27H13 passes the
first two decisively and misses only the third (87%). Qwen 3.6/L3H13 misses the
*first and third* (84%, 84%). The word FAIL is carrying the **coverage**
threshold, not the citation result — and it fails on the pinned model too. Do
not quote it as "the lens does not work on Qwen 3.8"; the citation arm is the
strongest measurement in this note.

## 5.4 Summary of the two signals

| | Prompt A (selection) | B | C | Leg C messy (independent) |
|---|---|---|---|---|
| L27H13 top1 | 28/28 (100%) | 25/29 (86%) | 27/29 (93%) | 369/413 (89%) |
| L27H13 top3 | 28/28 (100%) | 29/29 (100%) | 27/29 (93%) | 404/413 (98%) |

Confirmed on a second corpus, not overfit.

## 6. Decisions this raises — for the user, not for the probe

None of these were acted on. Each is an architecture decision.

1. **Should the architecture refusal become an allowlist?** `--attention-lens` is
   currently refused on everything but `qwen35moe`. §5.2 is evidence that the
   shipped constants are not as Qwen-3.6-specific as the pin assumes. The refusal
   is still *correct today* — it is a receipts path and "measured on two models"
   is not "calibrated for two models" — but the pin is now a decision rather than
   a necessity.
2. **Should `LensConstants` become per-architecture?** If Qwen 3.8 is ever a lens
   target, L27H13 (not L3H13) is its head, and the constants stop being a single
   frozen struct. That is a shape change to a shipped receipts type.
3. **The coverage constant `0.705` is the weak link on BOTH models** (87% / 84%
   used-clear, bar 90%). It was never searched — it was frozen from COV1 on one
   model. A coverage-layer search is the obvious next probe and is the arm most
   likely to be genuinely miscalibrated.

## 7. What was NOT done

- **No coverage-layer SEARCH.** Leg C *evaluates* layer 11 @ 0.705 on Qwen 3.8
  (87% used-clear, median peak 0.913), but no other layer was scored, so "is
  there a better coverage layer on Qwen 3.8?" is open. The dedicated COVERAGE leg
  (consulted-vs-skipped span separation) was not run. `ATTN_COV_SLOT` was added
  to make that search a one-liner for whoever picks it up.
- **`ungrounded_body_mass=0.538` was evaluated, not searched** — 0% false alarm
  on Qwen 3.8 via Leg C, but the ATTN_UNGROUNDED leg (which is the probe that
  would justify the threshold) was not run.
- **No images.** `docs/note-image-lens-probe.md` closes that question for the
  decode tap; out of scope by construction.
- **`LensConstants` was not changed**, and the architecture refusal added in the
  same change was not relaxed. Making the constants per-model is an architecture
  decision, not a probe outcome.

## LEGCSEARCH on the messy bilingual corpus (2026-09-19) — L3H13 does NOT hold

§3's table above is Prompt A, **N = 28 English value tokens, 128 candidates**.
It ranked `L3 H13` second at 92.9% top1 / 96.4% top3, one place behind
`L27 H13`. Because `L3` sits below the coverage layer (`L11`), shipping it
would have cut `--lens-verify-only` from 28 of 33 blocks to 12 — **48% of the
stack** — so it was worth scoring properly.

Scored on the Leg C corpus (15 docs EN+DE, the same corpus that judges every
other model's citation head), the shallow head collapses:

| rank | layer head | top1 | top3 | EN | DE | bar | verify blocks |
|---|---|---|---|---|---|---|---|
| 1 | **L27 H13** | 88.6% | **97.8%** | 97.7% | 97.9% | PASS | 28/33 |
| 2 | L31 H3 | 78.2% | 93.0% | 92.7% | 93.3% | PASS | 32/33 |
| 3 | L31 H0 | 67.8% | 91.0% | 90.4% | 91.8% | PASS | 32/33 |
| 4 | L31 H1 | 69.2% | 90.8% | 90.4% | 91.2% | PASS | 32/33 |
| 7 | **L3 H13** | 74.8% | **83.8%** | 84.5% | 83.0% | **fail** | 12/33 |

`L3 H13` misses the 90% bar by **6.2 points pooled, and on BOTH halves** — so
this is not the German-weakness failure mode that disqualified `L19 H20` on
Bonsai. The shallow head simply does not carry the citation signal once the
corpus is messy. The N=28 English probe overestimated it by ~13 points.

Per-layer best head, top3 — the signal has a sharp onset at L27, not a gradient:

| layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
|---|---|---|---|---|---|---|---|---|
| best top3 | 83.8 | 75.3 | 70.2 | 82.3 | 82.8 | 79.2 | **97.8** | 93.0 |

**No head at or below L23 clears the bar.** The sweep's own summary line reads
"shallowest passer: the same candidate — no depth/quality trade here".
Cross-language selection holds both directions (EN→L27H13→DE 97.9%,
DE→L27H13→EN 97.7%).

`L27 H13` is confirmed: best AND shallowest passer. The row is unchanged, and
the 48% cut does not exist on this model.

## LOCHEAD (2026-09-19) — the 9B has a locate head, at L11, and it costs nothing

`/v1/locate` was refused on this model: the row carried `locate_layer = -1`.
Swept all 128 (layer, head) pairs on the Leg C messy corpus, 75 scored keys
(EN 40 / DE 35).

| rank | layer head | top1 | top3 | EN | DE |
|---|---|---|---|---|---|
| 1 | **L11 h=6** | 88.0% | **96.0%** | 95.0% | 97.1% |
| 2 | L15 h=11 | 88.0% | 96.0% | 95.0% | 97.1% |
| 3 | L15 h=1 | 81.3% | 94.7% | 97.5% | 91.4% |
| 13 | L7 h=3 | 68.0% | 92.0% | 92.5% | 91.4% |
| 107 | L3 h=13 (the citation coordinate) | 33.3% | 49.3% | — | — |

Ranks 1 and 2 tie exactly. **L11 wins on depth** — the same tiebreak the 35B's
row already documents ("L11 was chosen over the higher-scoring L23" because it
equals `max(citation_layer, coverage_layer)`).

**It is free.** `max(citation 27, coverage 11, locate 11) + 1 = 28` — the
`--lens-verify-only` cut is unchanged at 28/33. A hypothetical locate-only
server would need 12/33, i.e. 64% of the stack skipped.

### Locate and citation run in OPPOSITE directions on the same model

Per-layer best head, top3, both swept on the same corpus:

| layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
|---|---|---|---|---|---|---|---|---|
| **locate** (key→source) | 50.7 | 92.0 | **96.0** | 96.0 | 93.3 | 88.0 | 86.7 | 84.0 |
| **citation** (generated→source) | 83.8 | 75.3 | 70.2 | 82.3 | 82.8 | 79.2 | **97.8** | 93.0 |

Locate peaks early and decays with depth; citation is near-flat until a sharp
onset at L27. This is the fourth model on which the generated→source head is
not the key→source head, and the cleanest demonstration of why: they are not
merely different coordinates, they have inverted depth profiles. Reading
locate off the citation layer scores 49.3% here (rank 107 of 128).

## Addendum 2026-09-19 — LOCABSENT: can locate tell a key is NOT there?

**Yes, the signal is real and survives a position control: AUC 0.927 bilingual
(EN 0.930 / DE 0.922). At ZERO false accusations it catches 62–71% of genuine
absences; at ~5–10% false-alarm, 79–83%. It is a HINT, not a verdict, and no
constant is landed.**

This is the gap that made the omission claim hollow: `/v1/locate` always returns
spans, so a field the document does not contain came back as three confident
byte ranges with no signal.

### Setup

| | |
|---|---|
| Model | `models/Qwen3.8-9B-Q8_0.gguf`, shipped pair **L11 h=6** read from the calibration table (never env overrides) |
| Present | each document's own labelled fields — **75** (EN 40 / DE 35) |
| Absent | **6 concepts × 15 docs = 90**: `warranty_period`, `incoterms`, `vat_number`, `discount_rate`, `payment_terms`, `contract_number` — every one verified absent from all 15 documents in both languages |
| Scores | `raw` = peak attention on the best document position (what the route emits); `conc` = `raw × doc_tokens` (length-free) |
| Driver | `LOCABSENT=1 LOCABSENT_SHUFFLE=1`, `tests/perf/attn_provenance.cpp` |
| Logs | `.session-results/9b_locabsent{,2,3,_rev,_shuf,_n2}.log` |

### The confound this leg found in itself — read this before the numbers

The first three runs appended the absent keys after the present ones, so every
present key sat at positions 1..N and every absent key after them. **The score
is dominated by a key's POSITION in the request.** Reversing the absent list,
same documents and same model, moved `contract_number`'s median from 8.40 to
**28.20** and `warranty_period`'s from 10.14 to **2.17**:

    position in request |  1st    2nd    3rd    4th    5th    6th
    forward  (conc p50) | 10.14   8.07   2.64   2.75   2.56   8.40
    reversed (conc p50) | 28.20   9.26   4.39   2.47   2.00   2.17

So "absent scores lower" in the append-order arm was partly "absent was listed
later", and **the AUC 0.9615 it produced is not a measurement of absence.**
`LOCABSENT_SHUFFLE=1` interleaves the classes with a fixed seed and is the only
arm whose AUC means what the header says. Every number below is from that arm.
A tell that the control worked: EN and DE AUC came into agreement (0.930 /
0.922) where the confounded arm had them 0.976 / 0.944.

### Result — position-balanced, cross-language

Threshold chosen on one language at a false-alarm target, scored on the other.
The budgeted error is a **false absent** — telling a caller a field is missing
when it is on the page — because that is the error the omission claim cannot
survive.

    score=raw
      target | chose T on | T       | scored on | false-alarm | absent-detected
          5% | EN         | 0.0850  | DE        |      0.0%   |     71.4%
          5% | DE         | 0.0908  | EN        |      5.0%   |     62.5%
         10% | EN         | 0.1296  | DE        |      5.7%   |     78.6%
         10% | DE         | 0.1307  | EN        |     10.0%   |     83.3%

`raw` beats `conc` at every operating point. **That is not a finding about long
documents**: this corpus spans only ~84–198 document tokens, so the length
confound `conc` exists to remove is untested, and raw must not be assumed to
transfer to a 4 KB CV.

### Two dependencies that stop a bare constant from shipping

**1. Recall varies by concept; safety does not.** At one threshold
(`raw` T=0.0908), detection per held-out concept:

    discount_rate 86.7% | vat_number 73.3% | warranty_period 80.0%
    payment_terms 80.0% | incoterms    53.3% | contract_number 26.7%

`contract_number` is the floor, and the reason is structural: **every document
carries an order number / PO**, so the missing field has a strong wrong place to
look. The generalizable statement is *absence detection degrades when a
structurally similar field IS present* — which is also the realistic case.

**2. The score is a property of the REQUEST, not of (document, key).** Going
from 2 absent keys to 6 moved the absent median from 0.0922 to 0.0646 (1.43×)
while the present median held at 0.294 → 0.291 (1.2%). This asymmetry is the
good news: a fixed threshold's **false-alarm rate is stable** (it is set from
present fields, which barely move) while its **recall drifts with schema size**.
Safe, variable yield — the better of the two failure modes.

### Verdict

- **Not another scalar-confidence kill.** AUC 0.927 under a position control is
  a real signal, unlike the three margin/confidence probes this repo has closed.
- **Ship as a soft badge, never a boolean.** "weak evidence for this key" at the
  0-false-alarm threshold is defensible and catches ~2 in 3 real absences.
  A hard `"absent": true` at these rates would be a new way to lie.
- **No constant landed.** Moving/adding a lens calibration constant is the
  user's decision, and two things should precede it: a **long-document leg**
  (the length range here cannot choose raw vs conc) and a **key-count leg** to
  size the recall drift at the schema size the product actually sends.

## Scouting 2026-09-20 — decisions from span-only: what failed, and what worked

**SCOUTING, NOT A PROBE.** N=4–8 hand-written cases, one model (9B Q8_0), no
held-out split. Enough to choose what to build; not enough to claim a rate.

### Dead ends (do not re-propose)

**1. Options planted in the prompt, winning span read as the decision — 0/6.**
The span lands on the *evidence*, never the option: "is the invoice overdue"
returned `ueberfaellig` and `Rechnung`; "seniority level" returned `Senior` and
`Staff Engineer`. Where it did touch an option it was lexical overlap (question
"over**due**" → option text "**due**"), not reasoning. This re-confirms the
standing law — *attention marks consideration, not commitment* — in a new
regime. PROOF1's "commitment is the SLOT" does not rescue it; a planted option
is not that kind of slot.

**2. Negated option keys — negation is invisible.** "the contract can be
terminated" vs "cannot be terminated" both land on `term`, and the winner is
whichever has better lexical overlap. Attention has no NOT.

**3. Polarity yes/no — 2/4.** "payment is overdue and unpaid" scored HIGHER on
the document saying payment was made on time (0.395 vs 0.226), because that
document is still *about payment*. Absence works for **structural** presence
("is there a termination clause", 7.3x; "warranty period", 11.1x) and fails for
**state** ("is the thing true"). That boundary is the rule to remember.

### What worked: choice as argmax over evidence presence

Do not ask the model to pick. Score each option's *content description* as a
locate key and take the argmax — one key per category, the span as the receipt.
4-way routing (finance / legal / engineering / people), 8 documents EN+DE:

| arm | prefills | latency | score |
|---|---|---|---|
| all 4 keys in **ONE** call | 1 | **232 ms** | **7/8** |
| 4 rotations, averaged | 4 | 1338 ms | **8/8** |

Chance is 2/8. For reference Jev answers in 70–500 ms and scores 67.8% on its
own benchmark — different task and data, so not a comparison, only a scale.

**Position bias perturbs magnitudes but does not flip a confident argmax.**
Rotating the key order on one document: engineering won every time at 0.72 /
0.80 / 0.87 / 0.71. That is what makes the one-call arm viable despite the
position dependence measured in LOCABSENT.

**Long keys need care.** The single one-call failure was `contract -> finance`,
caused by one stray token pairing (`an` ↔ ` agreement`) spiking to 0.507 while
the same key's next span was 0.047. The server takes `max` over the key's
tokens, which is right for short field names and wrong for sentences. Two fixes
both work: decompose the key into content words and average (fixes it), or
rotate (fixes it).

**Margin is NOT a usable confidence.** The miss had margin 0.13; a hit had 0.09.
Escalating to rotation on a small margin is a defensible heuristic, not a
calibrated gate — the fourth time scalar confidence has failed in this repo.

### Score on a scale — weakest of the three

Ordered levels as content descriptions, 4 CVs: **exact 2/4, within-1 4/4**. The
Jev-style expected value is monotone but compressed (0.82 / 1.36 / 1.99 / 1.93)
and the top two levels collapse. Do not claim `score`.

### Where this leaves the three Jev types

| Jev | span-only route | status |
|---|---|---|
| `choice` | argmax over evidence keys | **strong** — 7/8 one-call, 8/8 rotated |
| `noul` | structural presence (LOCABSENT) | **measured** — AUC 0.927 |
| `score` | ordinal argmax | **weak** — within-1 only |
| (polarity / judgement) | — | **does not work** |

Next, if pursued: a real `DECIDE1` probe — a labelled routing corpus, bilingual,
both halves, option order rotated, with the one-call and rotated arms scored
separately. Client-side this needs **no server change**: it is `/v1/locate` with
category descriptions as `key_vocabulary`.

## DECIDE1 2026-09-20 — choice as argmax over evidence presence, measured

**75.0% on a 4-way routing task in ONE 231 ms prefill (chance 25%), which
decomposes into 95.8% on clean documents and 43.8% on documents deliberately
seeded with another category's vocabulary. The scouting figure of 87.5% was
optimistic by 12 points — this is why the probe was built.**

Driver: `py/lens_decide1.py`, run against the shipped `POST /v1/locate` on a
`--lens-locate-only` 9B (not a probe-local tap — the product question is whether
a client can do this with no server change, so it goes through the real route).
Corpus: **40 documents, 20 EN / 20 DE, 4 categories, 16 carrying lures.**
Log: `.session-results/decide1_9b.log`.

### Arms

| arm | keys | prefills | latency | overall | EN | DE |
|---|---|---|---|---|---|---|
| A | sentence descriptions, canonical order | 1 | 234 ms | 50.0% | 40.0% | 60.0% |
| B | sentence descriptions, 4 rotations | 4 | 936 ms | 62.5% | 55.0% | 70.0% |
| **C** | **content-word parts, one call** | **1** | **231 ms** | **75.0%** | 70.0% | 80.0% |
| D | content-word parts, rotated | 4 | — | 75.0% | 75.0% | 75.0% |

### What arm A's failure diagnosed

Arm A collapsed into `finance`: legal→finance 6 times, people→finance 4,
engineering→finance 2. The cause is ours, not the model's. **The server takes
`max` over a key's tokens**, so a description carrying more filler — *"**an**
invoice, **a** payment, **a** budget, or **an** amount **of** money owed"* —
gets more chances for one stray token to spike. Scouting had already seen the
exact mechanism (`an` ↔ ` agreement` at 0.507 while the same key's next span was
0.047); this corpus shows it dominating a whole task.

Removing the filler (arm C) recovers **+25 points at identical latency** and the
`finance` row becomes 10/10. Two consequences worth keeping:

- **Position stops mattering once the filler is gone.** Arms C and D tie at
  75.0%, where the sentence arms gained 12.5 points from rotation. The position
  sensitivity LOCABSENT measured is largely a stopword artifact in this regime.
  (Arm A's canonical order was also the worst of the four — 50.0% against a
  50.0–65.0% spread — because `finance` led the list and its bias compounded.)
- **`max`-over-key-tokens is the wrong aggregation for sentence-length keys.**
  It is correct for the short field names locate was calibrated on. If decision
  routing is ever pursued server-side, a mean-over-key-tokens option is the
  change to make, and it would remove the client's need to decompose by hand.

### The real boundary: clean 95.8% vs lure 43.8%

| | arm A | arm C |
|---|---|---|
| clean documents (24) | 62.5% | **95.8%** |
| lure documents (16) | 31.2% | **43.8%** |

The lures are documents whose *primary* intent is one category while their
vocabulary is another's: a termination notice that mentions outstanding fees, an
HR note about salary bands, a migration ticket about the database bill. Evidence
argmax detects **presence**, and in a lure both categories are genuinely
present — so it cannot weigh primary intent. This is the same boundary LOCABSENT
found for absence (structural presence works, state does not), and it should be
stated in any product claim rather than averaged away.

### Cross-lingual

On the 20 German documents, an **English** schema scored 60.0% and a **German**
schema 70.0%. A client sending one schema to every language pays ~10 points;
localize the category descriptions.

### Honesty caveats

- **`PARTS_EN` was written AFTER seeing arm A's confusion matrix.** The
  *mechanism* (strip filler) was identified in scouting on different data, so
  this is not pure fitting — but the specific word choices are not out-of-sample
  the way arm A is. Arm C's 75.0% should be read as an upper estimate until it
  is re-run on a corpus written after the parts were frozen.
- **The corpus is synthetic and self-authored**, the same limitation the Leg C
  corpus carries. It measures the mechanism, not the market.
- One model, one task shape. No claim beyond 4-way topical routing.

### Verdict

Span-only can serve Jev's `choice` for **topical routing/triage where the
categories are separable by content** — 95.8% there, at 231 ms, with the
evidence span attached. It should NOT be sold for mixed-intent documents, where
it is 43.8% and a coin-flip-plus. Combined with `noul` (LOCABSENT, AUC 0.927)
that is two of Jev's three types, honestly bounded.

## Quant comparison 2026-09-20 — Q4_K_M vs Q8_0 on the 9B

**Q4_K_M costs 2.7 points of locate top-3 and 4.0 of top-1, saves 1.78x memory,
and leaves the derived signals unchanged. The important finding is not the
trade — it is that a Q4_K_M report currently CLAIMS the Q8_0 rate.**

Motivation: span-only loads 12 of 33 blocks, so the question was whether the
freed budget should buy *more precision*. It should not — measured below, the
opposite direction is where the interesting trade is.

| | Q8_0 | Q4_K_M | delta |
|---|---|---|---|
| file | 9.11 GB | 5.38 GB | |
| **Metal, `--lens-locate-only`** | **3661 MB** | **2055 MB** | **1.78x smaller** |
| locate L11 h=6, top-3 | 96.0% | **93.3%** | **-2.7** |
| locate L11 h=6, top-1 | 88.0% | **84.0%** | **-4.0** |
| L11 h=6 rank of 128 | 1 (tied with L15 h=11) | **2** | |
| LOCABSENT AUC (raw, position-balanced) | 0.9270 | 0.9209 | -0.006, noise |
| DECIDE1 arm C (1 prefill) | 75.0% | 80.0% | +5.0, noise at N=40 |
| locate smoke, locate-only | 9/9 | 9/9 | unchanged |

### The degradation is real; the two derived numbers are not a difference

L11 h=6 losing 2.7 points of top-3 and 4.0 of top-1 is a consistent, directional
shift on the shipped claim. By contrast DECIDE1 moved *up* 5 points and
LOCABSENT's AUC moved down 0.006 — at N=40 and N=165 those are noise, and the
fact that they moved in opposite directions is itself the evidence. **Anything
this corpus reports at the 5-point level is not a finding.** That retires one
claim from the DECIDE1 write-up above: the "localize the schema, worth ~10
points" result does not replicate here (Q4_K_M gives EN 65.0% / DE 60.0%, the
reverse ordering), so treat schema language as unmeasured.

### The head selection is quant-dependent

On Q8_0, L11 h=6 and L15 h=11 tied exactly (both 88.0 / 96.0) and L11 was taken
on depth. **At Q4_K_M the tie breaks**: L15 h=11 leads at 85.3 / 96.0 while
L11 h=6 falls to 84.0 / 93.3. So the calibrated coordinate is not optimal on a
file it is currently served for.

### The receipt problem this exposes

The 9B row is `kLensAnyFileType`, so a Q4_K_M 9B is **admitted today under
constants measured on Q8_0** — and its report carries

    locate_provenance: "... L11 h=6 = 88.0% top1 / 96.0% top3 ..."

which is **false for that file**; the measured rate there is 84.0 / 93.3. The
spans are still good and both halves still clear the 90% bar, so this is not a
broken route — it is a receipt overstating itself, which is the failure class
this table exists to prevent. `config.weights` does differ between the two
files, so the reports are already marked incomparable; the provenance string is
the part that lies.

Three options, all of them the user's call because each moves or pins a
calibration constant:

1. **Pin the 9B row to Q8_0's `file_type`** — a Q4_K_M 9B is then refused
   fail-loud instead of served with another file's numbers. Strictest, and it
   is exactly what the `file_type` key was kept for.
2. **Add a Q4_K_M row** with its own measured pair (L15 h=11 = 85.3 / 96.0, or
   L11 h=6 = 84.0 / 93.3 to keep the shallower cut) and its own provenance.
3. **Accept and document** — cheapest, but leaves a provenance string that is
   wrong for one admitted file.

### Recommendation on the original question

Do **not** spend the freed budget on a higher quant. Q8_0 is already near
lossless here, our errors are mechanism errors rather than precision errors, and
the binding constraint is throughput (~1.4 req/s per process), not accuracy.
Q4_K_M is the interesting direction: **1.78x more replicas for 2.7 points of
top-3**, which is a trade worth making for a routing/triage deployment and worth
declining for one where a caller acts on a single span (top-1 84.0%).

## key_aggregation landed 2026-09-20 — +17.5 points on sentence keys

DECIDE1 diagnosed the defect: the locate driver reduced a key's query rows with
**MAX**, which is correct for the 1–4 token field names LOCHEAD swept and wrong
for sentence-length keys, where one filler token spikes and the wordiest key
wins. `POST /v1/locate` now takes `key_aggregation: "max" | "mean"`.

Measured on the DECIDE1 corpus (9B Q8_0, 40 docs, same run, only the flag changed):

| arm | `max` | `mean` | delta |
|---|---|---|---|
| A — sentence descriptions, 1 call | 50.0% | **67.5%** | **+17.5** |
| B — sentence descriptions, rotated | 62.5% | **80.0%** | **+17.5** |
| C — hand-decomposed content words, 1 call | 75.0% | 80.0% | +5.0 (noise) |
| D — decomposed, rotated | 75.0% | 77.5% | +2.5 (noise) |

The shape is the point: **mean helps exactly where max was broken (sentence
keys, +17.5) and does nothing where max was already right (short decomposed
keys, within noise).** That is what a correct fix looks like, as opposed to a
tuning knob that moves every number a bit.

It also **removes most of the reason for a client to hand-decompose**: arm A
under `mean` (67.5%, one prefill) closes most of the gap to hand-built parts
under `max` (75.0%), and arm B under `mean` (80.0%) matches arm C outright.

### Contract

- Default `"max"`, byte-identical to the behaviour that predates the field.
- The value is echoed in the report as `key_aggregation`, emitted
  unconditionally so an absent member cannot be confused with an old server.
- `"mean"` sets **`uncalibrated: true`**, for the same reason a question
  vocabulary does: every shipped `locate_provenance` rate was measured under
  `max`, so quoting 96.0% for a `mean` request would be a receipt claiming a
  number nobody measured.
- An unknown value is a fail-loud **400** naming both accepted values.
- Verified live: `max`→`uncalibrated:false`, `mean`→`uncalibrated:true`,
  omitted→`max`, `"median"`→400.

980/980 unit tests; locate smoke green on both the locate-only and verify-only
legs (gate 8 included).

## DECIDEHEAD 2026-09-20 — the locate head was the wrong head for decisions

**Held-out: CHOICE 85–95%, SCORE 50–83.3% exact with 91–95% within-1. The
incumbent locate pair ranks 20 of 128 on choice and 53 of 128 on score. Every
decision number this repo had before today was read off a head calibrated for a
different task.**

Driver: `DECIDEHEAD=1`, `tests/perf/attn_provenance.cpp`. One tapped prefill per
(document × instruction shape) yields all 128 candidates, LOCHEAD's trick.
Corpora: CHOICE 40 docs EN+DE 4-way (mirrors `py/lens_decide1.py`), SCORE 24
docs EN+DE over 4 ordered urgency levels. Log: `.session-results/9b_decidehead2.log`.

### Why this leg existed

DECIDE1's 80% and the 2/4 ordinal scouting were both read off **L11 h=6**, the
pair LOCHEAD calibrated for *"given a field name, find its value's span"*. There
was precedent for that being wrong: LOCHEAD found the **citation** head ranks
372 of 384 at locate. Declaring "span-only cannot rate" from one head chosen for
another job would have repeated that mistake exactly.

It was the right call to check. **On score the incumbent scores 45.8%; the best
head scores 87.5%.** The earlier conclusion was a measurement of the wrong head.

### Four levers, one pass

| task | best config (pooled) | head | exact | EN / DE |
|---|---|---|---|---|
| CHOICE | question / mean-mass | **L19 h=10** | 97.5% | 95.0 / 100.0 |
| SCORE | extract / max-mass | **L15 h=11** | 87.5% | 91.7 / 83.3 |
| SCORE (within-1) | question / mean-peak | L15 h=1 | 87.5% | **100% within-1** |

- **The head is the biggest lever.** Incumbent → best is +10 points on choice
  (87.5 → 97.5) and **+41.7 on score** (45.8 → 87.5).
- **`mass` beats `peak`** in most winning configs — total attention over the
  document, not one best position. That is the opposite of what locate wants,
  and it makes sense: a decision is about how much of the document a description
  covers, not where its single best match sits.
- **The question instruction shape wins** both pooled bests and is structurally
  immune to the comma-collision defect (options go on their own lines).
- **`mean` key-aggregation** appears in both pooled bests, consistent with
  DECIDE1's +17.5.

### Held out — select on one language, score on the other

The pooled table above chooses the best of 128 × 8 = **1024 candidates on the
data it reports**. That maximum is inflated even when the signal is real, so
these are the numbers to quote:

| task | selected on | config | head | scored on | exact |
|---|---|---|---|---|---|
| CHOICE | EN | extract / max-peak | L7 h=1 | DE | **85.0%** |
| CHOICE | DE | question / mean-mass | L19 h=10 | EN | **95.0%** |
| SCORE | EN | extract / max-mass | L15 h=11 | DE | **83.3%** |
| SCORE | DE | question / max-peak | L11 h=14 | EN | **50.0%** |

**CHOICE transfers. SCORE does not, reliably.** Choice holds 85–95% in both
directions even though the two halves select *different* configs. Score swings
83.3 → 50.0 depending on which language chose the head, and the selected head
and variant differ too — that instability is the finding, not a detail.
Score's **within-1 is the stable part at 91–95%**.

### Depth

Choice's per-layer curve rises to L19 and falls after: L7 90.0%, L11 92.5%,
L15 92.5%, **L19 97.5%**, L23 95.0%, L31 80.0%. So a decision head costs
**20 of 33 blocks** — more than locate's 12, less than the verify cut's 28.
L11 at 92.5% is available for free on a locate-only server if the 5 points are
not worth 8 blocks.

### What this changes

- **"Span-only cannot rate" is withdrawn.** It was a claim about L11 h=6.
- **Choice is strong and held-out-validated**: 85–95% bilingual on 4-way
  routing, against Jev's 67.8% on its own benchmark (different data — a scale,
  not a comparison).
- **Score should be sold as within-1, not exact.** 91–95% never-more-than-one-
  level-out is defensible; the exact rate is not stable across languages yet.
- A decision head would be a **separate constant** from the locate pair, on a
  separate layer, with a different score variant. That is a calibration-table
  change and therefore the user's call; nothing is landed.

### Caveats

- N=40 (choice) and N=24 (score) are small; score especially.
- Corpora are synthetic and self-authored, same limitation as Leg C.
- One model, one task shape per type.

### DECIDEHEAD on Q4_K_M (2026-09-20) — the useful heads are quant-stable

Re-run because quantization was already shown to reorder heads (the locate
L11 h=6 / L15 h=11 tie breaks at Q4). It does move the top of the table — but
not the two candidates that matter.

| | Q8_0 | Q4_K_M |
|---|---|---|
| CHOICE pooled best | 97.5% (L19 h=10) | 95.0% (L15 h=9) |
| CHOICE held out EN→DE / DE→EN | 85.0 / 95.0 | 80.0 / 85.0 |
| **CHOICE at L11 h=3 (free, 12 blocks)** | **92.5%** | **92.5%** |
| CHOICE incumbent L11 h=6 | 87.5% (rank 20) | 82.5% (rank 30) |
| SCORE pooled best | 87.5% (**L15 h=11**) | 87.5% (**L15 h=11**) |
| SCORE held out EN→DE / DE→EN | 83.3 / 50.0 | 83.3 / 66.7 |
| SCORE incumbent L11 h=6 | 45.8% (rank 53) | 37.5% (rank 77) |

Three things this settles:

- **`L11 h=3` gives exactly 92.5% on both quants.** The free routing head — same
  twelve blocks a locate-only server already loads — is the most stable number
  in either sweep. That is the candidate to build on.
- **`L15 h=11` tops the ordinal task on both quants.** The score head is not a
  quant artifact either, even though its *held-out* behaviour still is not
  trustworthy.
- **The score held-out asymmetry is NOT a quant effect.** EN→DE holds at 83.3%
  on both while DE→EN reads 50.0% / 66.7%. German consistently selects a worse
  configuration, which is a property of the task and corpus, not of precision.
  It stays the reason not to land an exact-score constant.

Also consistent with the quant comparison above: the incumbent locate pair
degrades on Q4 for both tasks (87.5→82.5, 45.8→37.5), which is the same
directional loss LOCHEAD measured for locate itself (96.0→93.3).
