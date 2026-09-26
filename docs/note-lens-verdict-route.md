# The verdict route — VERDICT2, then `POST /v1/verdict`

**STATUS: LANDED (uncommitted) 2026-09-26.** Bundle B
(docs/note-lens-verdict-probe.md) showed that a one-token yes/no, read off the
prefill's own last row, covers what attention cannot. This note records the
probe that checked it on real documents (VERDICT2) and the route built from it
(docs/plan-lens-verdict.md). The user's condition for building it was: **"it
must not make any other feature worse"** — and it did not: 90 locate / extract
/ verify reports came out byte-identical before and after.

## 1. VERDICT2 — before building (9B Q4_K_M)

Three instructions × Bundle B's items and REQHEAD's 6 real CVs (72
requirements) — short, and buried in ~4K / ~8K tokens of unrelated e-mail — ×
full depth vs 28 blocks. Each document read once; every question resumes from
its snapshot. Bars set before the run, EN and DE separately.

| bar | result | verdict |
|---|---|---|
| 1 an instruction fixes hypotheticals | "fact-only" (I1): EN 50 → 90, DE 80 → 100, but it misses paraphrased facts on real CVs (present 100 → 94.4 short, EN 88.9 → 77.8 at 8K); three-way (I3): 70 / 70 | trade, no clean win |
| 2 three-way separates cases | claims: contradicted → no **100 / 100**, not mentioned → unclear **100 / 100**; CV requirements not mentioned → unclear only EN 75, DE 67 (the model says "no") | claims yes, CVs no |
| 3 real CVs, short | yes vs not-yes **100 / 100**, incl. all 12 comparisons attention failed | pass |
| 4 buried in 4K / 8K | EN 97.2 / 94.4, DE 94.4 / 94.4; German comparisons 4 of 6 ("Spanish C1?" vs B2 → false yes) | marginal fail |
| 5 28 vs 33 blocks | within 5 points on every set but sums | pass |

Also: sums stay a coin flip under every instruction; when three-way misses a
paraphrase in a long document it says "unclear", not "no".

**Decision (user):** build it anyway — a user who gets poor answers on long
documents simply won't use it there — but it must not harm anything else. My
one objection, adopted: users cannot *see* when it is wrong (a false "yes" at
0.94, receipt on the right line), so the route discloses its envelope.

## 2. What was built

* **`POST /v1/verdict`** — `{document, questions: [{id, question}], language?:
  "en"|"de", document_id?}` → per question `answer` (yes / no / unclear), `p`
  (the three-way softmax — labelled not a confidence), `receipt` (the locate
  head's top spans over the question's rows — where it looked, not a check),
  plus `validated_envelope` (false above 519 prompt tokens), `prefill`,
  `prefix`, `verdict.blocks`, provenance.
* **`run_lens_verdict`** (`server_lens.cpp`), a new driver; locate, extract and
  verify untouched. The instruction is VERDICT2's three-way text
  (`lens_verdict_instruction`). One document pass per request, truncated after
  layer 27 and computed as locate's pass 1 (the row's split+flash licence); each
  question restores its snapshot and runs its own rows with `want_logits` and
  the locate head tapped.
* **The kept document is shared with locate** (`LensKeptRoute::Locate`): the
  verdict's pass is locate's computed deeper, and a deeper pass serves a
  shallower read, so "where" and "whether" on one document pay for it once and
  the store's 4 slots are not split.
* **Licence, per model and quant:** `LensConstants::verdict_layer` (27),
  `verdict_provenance`, `verdict_envelope_tokens` (519), appended last, −1 =
  refused. Set only on the 9B **Q4_K_M** row.
* **Served on the full `--attention-lens` server only.** Truncated servers load
  `token_embd` and the blocks but no output head, and the 9B's is untied
  (`output.weight` 4096 × 248320 Q6_K, ~834 MB) — so `--lens-verify-only` and
  `--lens-locate-only` refuse (404) and keep their size. Runs on
  `locate_scheduler()` (every pass is a truncated prefill).
* Docs: architecture.md (module map + paragraph), lens-format.md (verdict
  section with its non-uses), plan-lens-verdict.md (marked landed).

## 3. Gates

| # | gate | result |
|---|---|---|
| G1 | other features unchanged: LENSDUMP (every shipped locate head with and without a kept document, extract with candidates with and without an id, verify) before vs after | **90 reports byte-identical** (the dump itself deterministic: two baseline runs identical); suite **1028/1028** (+6 verdict tests) + 18 HTTP tests serial |
| G2 | the shipped driver vs a reference of VERDICT2's I3 path at 28 blocks | **576/576** answers, max \|Δp\| **0** |
| G3 | licensed split+flash document pass vs materialized | **576/576** answers unchanged, max \|Δp\| 0.0055 ⇒ sharing with locate allowed |
| G4 | warm == cold with a `document_id`; locate ↔ verdict sharing (locate kept → verdict deepens it → locate warm == cold → verdict warm) | **18/18** and **18/18**, bit-identical |
| G5 | live | full server answers; verify-only / locate-only 404; Q8_0 400; bad `language` 400; nothing logged |

Shipped-route three-way accuracy (a "not met" requirement accepts no or
unclear): REQ short EN 97.2 / DE 75.0 (German absent requirements say "no"
where the strict score wants "unclear"), 4K EN 94.4 / DE 72.2, 8K EN 88.9 / DE
69.4; Bundle B EN 87.5 / DE 85.8 / DE-doc-EN-question 91.7.

Live example, a short CV ("five years of Java, English B2, does not hold a
forklift licence"): "at least three years?" → yes; "English C1 or better?" →
unclear; "forklift licence?" → no; "Python?" → unclear. Second call with the
same id: `prefix: warm`, same answers; a `/v1/locate` on that id afterwards was
warm too.

## 4. What this is and is not

* **Validated on short real documents only** (≤ 519 tokens). Longer prompts are
  answered and marked `validated_envelope: false`.
* **Not for sums, hypotheticals, or long German comparisons** — measured
  failures, documented rather than guarded.
* **On a checklist, read `no` and `unclear` together** — a requirement the
  document never mentions mostly comes back `no`.
* **`p` is not a confidence and the receipt is not a check** — a wrong answer
  can be sure and look in the right place.
* **Gated on the 9B Q4_K_M only**; every other row refuses until a VERDICT2 run
  licenses it. Serving on the verify-only server (whose 28 blocks match the
  verdict exactly) needs its output head loaded — a separate, opt-in decision.
* The stronger version of this readout is a trained reader over hidden states
  (training ladder L1) — not done.

## Reproduce

```
M=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf
VERDICT2=1    QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance                               # ~25 min
VERDICTGATE=1 QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance                               # ~67 min
LENSDUMP=1 LENSDUMP_OUT=before.txt QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance          # G1: run before and after, cmp
```
