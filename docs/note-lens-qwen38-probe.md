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
