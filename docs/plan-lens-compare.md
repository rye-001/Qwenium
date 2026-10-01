# Plan — the compare route (`POST /v1/compare`): what is missing from a second version

> **Baseline changed 2026-09-27 (COMPARE3, user-approved).** Coverage is now
> relative to the mean of the best-covered quarter of units, threshold
> **0.35** (was the median unit, 0.50). The median failed when most of an
> original was missing (a summary): 51–65% of drops caught. COMPARE3 chose the
> new pair on one set of translation trials (normal 0–2 drops + heavy 50/75%)
> and confirmed it, pre-registered, on fresh ones: normal 87.5–95.8% flagged /
> 0% false on complete copies, heavy 95.6–98.1% / 0% on kept units. It also
> exposed that the median at 0.50 sat on an edge for EN→DE copies (18.8% and
> 12.5% false alarms on two fresh samples, 6.2% on the gate's). Re-gated:
> COMPAREGATE translation 95.8 / 87.5% flagged, 0% false; AbsenceBench numbers
> 83.2, poetry 76.5; flash still changes flags (4 of 31,807, stays
> materialized); warm = cold 96/96; other reports byte-identical (LENSDUMP);
> live 2 / 5 / 6 of 8 dropped all exact on the three server kinds. Numbers in
> the tables below are the 2026-09-26 median-baseline ones.

Status: **LANDED 2026-09-26 (uncommitted).** The user chose option (a) —
expose "compare", the seventh prefill-only mode — after AbsenceBench; the gate
probe that had to come first (COMPARE2) passed both bars, and every build gate
below passed (COMPAREGATE in `tests/perf/attn_provenance.cpp`):

| gate | result |
|---|---|
| G1 other features unchanged | LENSDUMP 95 reports (locate, extract, verify, verdict) byte-identical; full suite green |
| G2 shipped driver, fixed threshold 0.50 | translation EN→DE **91.7%** flagged / **6.2%** false; DE→EN **97.9% / 0.0%**; AbsenceBench EVAL numbers **82.9**, poetry **77.0** (COMPARE2 82.9 / 78.0) |
| G3 flash on the original's pass | **failed** — 6 of 31,807 flags changed (max Δcoverage 0.0084) ⇒ the pass stays materialized |
| G4 kept original, successive revisions | 96/96 identical to cold |
| G5 live | full, verify-only and locate-only servers: 8-sentence ferry notice, 2 sentences dropped in the German version → exactly units 2 and 6 flagged (coverage 0.10 / 0.17, the rest 0.89–1.21), identical on all three; second request with the same `document_id` `"prefix":"warm"`, same answer, 0.64 → 0.30 s; Q8_0 refuses (400); no document text in any log |

Envelope: smallest original gated 8 units, longest prompt 9,762 tokens.

## 1. What it is

An **original** (as a list of units — lines, sentences, clauses, as the caller
cuts it) and a **second version** (free text: a translation, a rewrite, a
summary, a new draft) in; per original unit, whether the second version still
covers it. One prefill, no text generated. It works where `diff` cannot — the
second version need not be word-for-word — which is the product: translation QA
(dropped sentences), contract redlines (a clause that quietly disappeared),
summary coverage, notes against an agenda.

## 2. Evidence (Qwen3.8-9B Q4_K_M)

| probe | what | result |
|---|---|---|
| COMPAREHEAD/HARD (2026-09-24) | EN↔DE translations, shuffled, anchors stripped | omission 100% at message level, ~80% at sentence level |
| ABSBENCH (2026-09-26) | AbsenceBench, external, head picked on a dev split | poetry **76.5**, numbers **78.6** (GPT-4.1 54.3 / 57.5; Claude-3.7-Sonnet-thinking 72.7 / 96.0); code diffs **8.3** (fail) |
| **COMPARE2** (2026-09-26) — the served shape: one generic prompt, caller's units, a threshold | translation, 0/1/2 sentences dropped, threshold chosen on one direction and scored on the other | EN→DE **91.7%** of drops flagged, **6.2%** of complete copies falsely flagged; DE→EN **97.9% / 0%** — bar (≥ 80% / ≤ 10%) **passed** |
| | AbsenceBench through the generic prompt | poetry **78.0**, numbers **82.9** — bar (within 10 of 76.5 / 78.6) **passed** |

The head is **L15 h1** (the compare probe's sentence-level best): the only one
that holds on every set. L15 h13 scores 92.9 on poetry but 38.8 on numbers and
false-alarms on 31% of complete translations; the landed absent head (L19 h10)
catches two thirds. The threshold chosen independently on translation (0.47 /
0.52) and on AbsenceBench (0.43 / 0.50) agrees.

## 3. Wire format (additive; lens-format.md gets a section)

```json
POST /v1/compare
{ "original_units": ["The ferry leaves at seven.", "Bring your ticket.", "..."],
  "revised": "Die Fähre fährt um sieben ab. ...",
  "document_id": "contract-v1" }        // optional: keep the original, compare many revisions
```

```json
{ "model": "...", "config": {...}, "prefill": "split", "prefix": "cold",
  "validated_envelope": true, "threshold": 0.50,
  "compare": {"layer": 15, "head": 1, "provenance": "..."},
  "units": [ { "index": 2, "coverage": 0.08, "missing": true,
               "restated_at": null },
             { "index": 0, "coverage": 1.00, "missing": false,
               "restated_at": {"byte_lo": 0, "byte_hi": 29} } ] }
```

* `coverage` = the unit's attention from the revised text, relative to the
  document's median unit (1.0 = typical). **Not a confidence** — a ranking.
* `missing` = `coverage < threshold` (the calibrated value, per model).
* `restated_at` = where in the revised text the unit is most attended from — a
  receipt for "covered", `null` when missing.
* Refusals (400): fewer than 2 units, an empty unit, an empty revision, a row
  without a compare head.

## 4. Engine and serving — the "no other feature gets worse" rule, as for /v1/verdict

* **`run_lens_compare`**, a new driver; every existing route untouched (gate G1).
* Prompt = COMPARE2's generic text; units one per line. Split prefill: pass 1 =
  instruction + original (untapped), pass 2 = the revision and the question,
  tapped on L15 h1 only; truncated after layer 15 (**16 of 33 blocks**).
* **Served on every lens server, with no load change**: 16 blocks sits inside
  the locate-only cut (20), and the readout is attention, not logits — no
  output head needed.
* **Kept original** with `document_id`, under a new `LensKeptRoute::Compare`
  (its pass 1 starts with the compare instruction, so it never matches a locate
  or verdict entry anyway).
* Calibration: `compare_layer`, `compare_head`, `compare_threshold`,
  `compare_provenance`, **appended last**, −1 = refused; set only on the 9B
  **Q4_K_M** row (every measurement above is Q4_K_M).
* `validated_envelope` false when the prompt exceeds the longest gated one or
  the original has fewer units than the gated minimum (the median needs units
  to be a baseline; COMPARE2's smallest original had 8).

## 5. Gates (all before landing)

| # | gate | bar |
|---|---|---|
| G1 | other features unchanged — LENSDUMP before vs after, full suite | byte-identical; green |
| G2 | the shipped driver at the FIXED calibrated threshold reproduces COMPARE2: translation flag rates and false alarms both directions, AbsenceBench poetry / numbers F1 | within noise of COMPARE2; bars still met |
| G3 | flash on pass 1 vs materialized | no `missing` flag changes — else pass 1 stays materialized |
| G4 | warm == cold with `document_id` (several revisions against one kept original) | bit-identical |
| G5 | live on full, verify-only and locate-only servers; Q8_0 refuses; nothing logged | pass |

## 6. Not in v1

* **Additions** (parts of the revision the original lacks) — probed at message
  level only, never gated with a threshold.
* **Repetitive text** — code diffs, tables of repeated values: AbsenceBench
  diffs 8.3. Documented; for verbatim copies `diff` is the right tool.
* Other models / quants — refused until gated.

## 7. Cost

About a day, mostly gates (G2 reuses COMPARE2; G1/G4 reuse LENSDUMP / LOCWARM
patterns).
