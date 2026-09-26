# Verdict readout — step 5 ("bundle B")

**STATUS: PROBED 2026-09-26, nothing landed.** Qwen3.8-9B Q4_K_M. One prefill
per item, no decode step: P(yes) against P(no) read off the prefill's own last
row, plus the attention receipt from the same pass. The verdict covers most of
attention's blind spots — "not", "at least", "the latest value", "is the claim
supported" — at 90–100% in EN and DE. It fails on hypotheticals and on sums,
and it needs **28 of 33 blocks**, so it does not fit the 20-block locate-only
server. Whether a verdict route is built is the user's decision.

## 1. The question

Attention marks presence. It has no NOT, no comparison, no "latest", no sum
(docs/note-lens-req-probe.md: comparisons failed; the laws in
docs/note-lens-prefill-only-engine.md). Can a one-token verdict, read in the
same prefill as the receipt, cover those — and how shallow can it be read?

## 2. Setup

* **Items.** Five families, minimal pairs (one flipped detail flips the
  answer), EN and DE written in parallel, 10 items per family per language:

  | family | variants | example |
  |---|---|---|
  | NEG | stated (yes) / negated / hypothetical | "We want express shipping" / "do not want" / "would want it if the surcharge were lower" |
  | CMP | question flips | CV "five years": "at least three?" / "at least seven?" |
  | LATEST | question flips, old value = lure | booking for 12, then 18: "is it 18?" / "is it 12?" |
  | XCHECK | stated total = sum / = sum minus one line | two-line invoices: the lure equals a row's value |
  | CLAIM | supported / contradicted / not mentioned | "Revenue rose 8%" / "fell 8%" / an unrelated sentence |

* **Arms.** EN (EN document, EN question), DE (DE document, DE question), DEx
  (DE document, EN question — BUNDLEA's "ask in English" recipe).
* **Prompt.** Document, then `Question: … / Answer with yes or no only.`
  (DE: `Frage: … / Antworte nur mit Ja oder Nein.`), chat template, thinking off.
* **Readouts per item, one full-depth pass (want_logits + head taps):**
  * *verdict* — softmax mass of the yes tokens over yes+no tokens, threshold
    0.5, no calibration; *compliance* = is the top token of the whole
    vocabulary a yes/no token.
  * *receipt* — locate head L11 h6, question rows, argmax sentence/line of the
    document: does it land on the decisive sentence?
  * *attention baseline* — score head L19 h11, question-row mass on the
    document.
  * *logit lens* — the same verdict with the prefill stopped after 20 / 24 / 28
    blocks and the output head applied there (EN and DE arms only).
* **Bar, set before the run:** verdict ≥ 90% per family per language at 0.5;
  logit lens = the shallowest depth within 5 points of full depth.

## 3. Results (full depth, verdict accuracy at 0.5)

| family | EN | DE | DEx | attention: yes-variant ranked above no-variant (chance 50%) | verdict: same ranking |
|---|---|---|---|---|---|
| NEG | 83.3 ✗ | 93.3 | 90.0 | 45% | 100% |
| CMP | 100 | 90.0 | 95.0 | 80–90% | 100% |
| LATEST | 95.0 | 90.0 | 95.0 | 70–90% | 90% |
| XCHECK | 60.0 ✗ | 70.0 ✗ | 70.0 ✗ | 30% | 70–90% |
| CLAIM | 100 | 100 | 100 | 85–95% | 100% |

Compliance was 100% in every cell. (CMP and LATEST flip the question, not the
document, so their attention ranking compares two different questions and is
not a clean baseline.)

* **NEG.** Stated and negated are 100% right in all three arms. The whole
  shortfall is the **hypothetical**: 50% EN, 80% DE, 70% DEx — "would want it
  if…" reads as half a yes. The verdict still ranks stated above hypothetical
  every time, so the ordering is right and the fixed threshold is what fails.
* **XCHECK.** A lure total that equals a row's value (the two-line invoices) is
  caught 4 of 4 in EN and DE. A total that leaves one line out of three or four
  is a coin flip — P(yes) sits near 0.5 on both variants. One token does not
  add.
* **CLAIM.** 100% in every arm, contradicted *and* not-mentioned — the verdict
  separates "says the opposite" from "does not say" without help.
* **DEx vs DE.** No consistent difference (NEG 90.0 vs 93.3, CMP 95 vs 90,
  XCHECK 70 vs 70). BUNDLEA's "ask in English" gain does not show here.

## 4. Depth (logit lens)

| blocks | top token is yes/no, EN / DE | NEG EN / DE | CMP EN / DE | LATEST EN / DE | XCHECK EN / DE | CLAIM EN / DE |
|---|---|---|---|---|---|---|
| 20 | 40.8% / 19.2% | 36.7 / 70.0 | 80 / 85 | 50 / 65 | 50 / 50 | 53.3 / 70.0 |
| 24 | 52.5% / 48.3% | 66.7 / 76.7 | 80 / 85 | 80 / 95 | 50 / 65 | 96.7 / 100 |
| 28 | 100% / 100% | 83.3 / 93.3 | 95 / 90 | 95 / 90 | 75 / 65 | 100 / 100 |
| 33 (full) | 100% / 100% | 83.3 / 93.3 | 100 / 90 | 95 / 90 | 60 / 70 | 100 / 100 |

At 20 blocks the model has not yet formed an answer (the top token is yes/no
for 41% / 19% of items). **28 blocks is the shallowest depth within 5 points of
full depth** in every family. A verdict needs a 28-block server, not the
20-block locate-only one.

## 5. Trust flag: receipt vs verdict

Verdict error rate by whether the receipt landed on the decisive sentence:

| arm | receipt hit: n, errors | receipt missed: n, errors |
|---|---|---|
| EN | 109, 12.8% | 11, 0.0% |
| DE | 116, 10.3% | 4, 0.0% |
| DEx | 113, 9.7% | 7, 0.0% |

Every wrong verdict had its receipt on the right sentence. The model looked in
the right place and concluded wrongly. **The receipt says where the model
looked, not whether the verdict is right** — a receipt/verdict disagreement is
not a trust flag.

## 6. Probe defect caught

The first run kept only single-token spellings of yes/no. A bare "Nein" is two
tokens, so a German "no" scored as neither and every DE no-variant read as yes
(DE compliance 33–70%, no-accuracy 0–40%). The leg now takes the **first token**
of every spelling (bare and space-led) and fails loud if the yes and no sets
share a token. All numbers above are from the corrected run.

## 7. What this is and is not

* **A real mode for four jobs**: negation (stated vs negated), comparison,
  latest value, claim support — the exact places attention scores at or below
  chance.
* **Not for hypotheticals or sums.** Both would need either a different
  question shape or generation (thinking), which is outside prefill-only.
* **P(yes) is a verdict, not a confidence.** Four earlier scalar-confidence
  kills stand; nothing here calibrates it.
* **Depth is the price**: 28 of 33 blocks vs 20 for the locate-only cut.
* **Small corpus**: 10 items per family per language, synthetic, short
  documents. Long documents and real mail are not tested.
* **Only the 9B Q4_K_M** was run.

## Reproduce

```
BUNDLEB=1 QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf ./build-metal/bin/attn-provenance   # ~6 min
```
