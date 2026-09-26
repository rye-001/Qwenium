# Bundle A — a set of heads vs one head, and asking across languages

**STATUS: MEASURED 2026-09-25 on Qwen3.8-9B Q4_K_M. Two answers, no engine
change.** (1) One head per job holds everywhere **except bind**, where a sum of
five heads beats the score head by ~7 points held out in both directions — so
a head-only tap should keep a *list* of heads, not exactly one. (2) Asking in
the other language works: every cross-language cell stays within ~6 points of
same-language — and **English questions beat German ones even over German
documents** (requirements AUC 0.995 vs 0.928).

## 1. The questions

* **Top-k.** QRHead (EMNLP 2025) sums a handful of heads, AT2 (2025) learns a
  weight per head, Expert Heads (ICLR 2026) takes a vote; the lens reads one
  head per job. Is one head leaving accuracy on the table?
* **Cross-language asking** (backlog #9). A German question over an English
  document and the reverse — does it work, and how close to same-language?

## 2. The probe

`BUNDLEA=1` in `tests/perf/attn_provenance.cpp`. One tapped prefill per prompt
gives all 8 attention layers x 16 heads, so a head set is a re-scoring of the
same pass: no extra prefills for Q1.

* **Jobs**, each with the signal its own note settled on, same seeds and
  prompts as those legs:

  | job | readout | landed head | corpus |
  |---|---|---|---|
  | search A | chunk mass, 16 chunks in one prompt | absent L19 h=10 | SEARCHHEAD setup A |
  | bind | max over question rows, per value | score L19 h=11 | BINDHEAD orders |
  | requirements, met vs unmet | mean-row mass over the CV | score L19 h=11 | REQHEAD CVs |
  | requirements, evidence line | segment argmax | locate L11 h=6 | REQHEAD CVs |
  | compare, dropped sentence | A-unit coverage density, argmin | score L19 h=11 | COMPAREHARD sentence leg |

* **Q1 — head sets.** Rank single heads on one language (one direction, for
  compare), take the top k (1, 3, 5, 10), sum their per-candidate scores —
  raw, and scaled by each head's mean on the selection half — and score on the
  other language. Bar: a set replaces the landed head only if it beats it held
  out in **both** directions by more than noise (~5 points at ~30 items).
* **Q2 — cross-language.** Search: the question of the asked document's
  translation twin (20 parallel routing notes + 6 parallel order e-mails).
  Bind: question language swapped, product name translated, order text
  unchanged. Requirements: all 72 questions translated before the run, labels
  and CVs unchanged. Bar: within ~10 points of same-language, far above chance.
* 580 prefills, 43 minutes. The landed-head rows reproduce every earlier note
  (search 100 / 96.9, bind 80.7 / 85.8, requirements 0.954 / 0.928, evidence
  100 / 97.2, compare 87.5 / 75.0) — the built-in check passed.

## 3. Q1 — one head vs a set (held out: selected on one half, scored on the other)

Columns: scored on EN (selected on DE) / scored on DE (selected on EN); for
compare, scored on DE>EN / EN>DE.

| job | landed head | best single | k=3 raw | k=5 raw | k=10 raw |
|---|---|---|---|---|---|
| search A, top-1 | 100 / 96.9 | 100 / 96.9 | 100 / 96.9 | 100 / 96.9 | 100 / 96.9 |
| **bind, right** | **80.7 / 85.8** | 80.7 / 85.8 | 90.3 / 89.8 | **88.1 / 93.2** | 87.5 / 92.6 |
| requirements, AUC | 0.954 / 0.928 | 0.980 / 0.956 (L23) | 0.991 / 0.959 | 0.985 / 0.957 | 0.971 / 0.948 |
| requirements, evidence | 100 / 97.2 | 88.9 / 97.2 | 100 / 100 | 94.4 / 100 | 100 / 100 |
| compare, dropped sentence | 87.5 / 75.0 | 87.5 / 81.2 | 93.8 / 81.2 | 87.5 / 87.5 | 81.2 / 93.8 |

Scaled sums match raw within a point or two everywhere (bind k=5 scaled: 89.2 /
91.5).

**Reading.**

* **Bind is the one job a set helps**: k=5 = +7.4 / +7.4 over the score head,
  both directions past the bar. The set is L19 h=11 + L15 h=1 + L19 h=15 +
  L15 h=11 + L15 h=3 (selected on EN; on DE the set shares L19 h=11, L15 h=1,
  L15 h=3) — all inside the 20-block locate-only cut.
* **Requirements:** the gain comes from **depth, not summing** — one L23 head
  alone (L23 h=9 / h=0) reaches 0.980 / 0.956; sets add ≤ 0.01. L23 costs 4
  blocks beyond the cut (REQHEAD found the same).
* **Search** is saturated; **evidence** is at the ceiling (one item of 36 =
  2.8 points); **compare** moves by one trial of 16 (6.2 points) — noise.
* So: **one head per job stays**, with bind as the exception. The head-only
  tap (note-lens-prefill-only-engine.md §9, step 1) should therefore keep **a
  list of heads per job**, which costs nothing and leaves bind room.

## 4. Q2 — asking across languages (landed head)

| job | EN documents: EN questions → DE questions | DE documents: DE questions → EN questions |
|---|---|---|
| search A, top-1 (none-AUC) | 100 (0.994) → **100 (0.987)** | 96.9 (0.979) → **100 (0.996)** |
| bind, right | 80.7 → 75.0 | 85.8 → 81.2 |
| requirements, AUC | 0.954 → 0.898 | 0.928 → **0.995** |
| requirements, evidence | 100 → 94.4 | 97.2 → 94.4 |

**Reading.**

* **It works**: every cell within ~6 points of same-language, all far above
  chance (search 6.2%, bind 13.6%, evidence ~5%).
* **The question's language matters more than matching the document.** English
  questions over German documents *beat* German questions over the same
  documents (search 100 vs 96.9, requirements 0.995 vs 0.928); German
  questions are the weaker arm over English documents too. Consistent with the
  standing finding that German weakness is the model's, not the corpus's.
* **Recipe:** ask in English, whatever the document's language. The spans and
  evidence lines come back in the document's own language regardless.

## 5. What this is and is not

* **Small corpora**, self-written, reused from the earlier legs. Bind's 176
  questions per language come from 24 generated orders, so answers within an
  order are correlated — its +7.4 is less certain than the question count
  suggests.
* **Translations are mine**, written before the run; a professional
  translation, or questions written natively in each language, may differ.
* **One aggregation family** (sums). Votes and learned weights (Expert Heads,
  AT2) were not tried; a learned weight would break training-free.
* **Cosmetic:** the leg prints search chance as 3.6% / 3.7% (none-trials
  included in the average); the true chance is 1/16 = 6.2%. No result depends
  on it; fix pending.

## Reproduce

```
BUNDLEA=1 QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf ./build-metal/bin/attn-provenance   # ~43 min
```
