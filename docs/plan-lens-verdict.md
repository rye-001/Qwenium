# Plan — the verdict route (`POST /v1/verdict`)

Status: **LANDED (uncommitted) 2026-09-26 — all gates pass.** G1: 90 locate /
extract / verify reports byte-identical before and after (LENSDUMP), suite
1028/1028 + 18 HTTP. G2: the shipped driver reproduces VERDICT2's path 576/576
(max |dp| 0). G3: split+flash document pass changes no answer 576/576 (max |dp|
0.0055) ⇒ shared with locate's kept documents. G4: warm == cold 18/18; locate ↔
verdict sharing 18/18. G5: full server answers, verify-only / locate-only refuse
(404), Q8_0 refuses (400), nothing logged. Envelope: 519 tokens. Was:
**PROPOSED 2026-09-26, not built.** The user decided to build it ("even
if the doc is big and the result is not good, the user wouldn't use it") on one
condition: **it must not make any other feature worse.** This plan is shaped by
that condition. Evidence: docs/note-lens-verdict-probe.md (BUNDLEB) and the
VERDICT2 leg (`tests/perf/attn_provenance.cpp`, results below).

## 1. What it is

A document and yes/no questions in; per question, one answer read off the
prefill's own last row — **yes / no / unclear** — with no decode step. It covers
what attention cannot: negation, comparison ("at least five years", "German C1
or better"), the latest value, and whether a claim is supported, contradicted or
not mentioned. Every answer carries a receipt: where the locate head looked.

## 2. What the measurements license (9B Q4_K_M)

| | result | consequence |
|---|---|---|
| real CVs, short (~0.5K tokens), plain yes/no | **100 / 100** EN / DE, incl. 12/12 comparisons attention failed | the core job |
| claims, three-way | contradicted → no **100 / 100**, not mentioned → unclear **100 / 100** | three-way is worth shipping |
| CVs buried in 4K / 8K | EN 97 / 94, DE 94 / 94; **DE comparisons 4/6** (false "yes" at 0.87); paraphrases missed | outside the validated envelope |
| hypotheticals ("would … if") | EN 50–70, DE 70–80 (the fact-only instruction fixes them but costs paraphrases) | documented non-use |
| sums (does the total add up) | coin flip under every instruction | documented non-use: locate the numbers, add them in the client |
| 28 vs 33 blocks | within 5 points on every set but sums | `verdict_depth` = layer 27 |

The instruction is **I3, three-way**: when I3 misses a paraphrase in a long
document it says "unclear" rather than a wrong "no". Where a verdict is wrong it
looks as confident as a right one, and its receipt sits on the right line anyway
(BUNDLEB trust flag) — so the route **discloses its envelope**; it cannot
detect its own errors.

## 3. Wire format (additive; lens-format.md gets a section)

```json
POST /v1/verdict
{ "document": "...",
  "questions": [ {"id": "r1", "question": "Does the candidate have at least five years of experience?"} ],
  "language": "en" | "de",          // instruction language, default "en"; both measured
  "document_id": "cv-17" }           // optional — the kept document (step 6)
```

```json
{ "model": "...", "config": {...}, "prefill": "...", "prefix": "cold"|"warm",
  "validated_envelope": true,        // false above the longest prompt the gate passed at
  "verdict_depth": 28, "verdict_provenance": "...",
  "answers": [ { "id": "r1", "answer": "yes"|"no"|"unclear",
                 "p": {"yes": 0.98, "no": 0.01, "unclear": 0.01},
                 "receipt": [ {"byte_lo": ..., "byte_hi": ..., "peak": ...} ] } ] }
```

* `p` is the softmax over the three answer-token sets. **Not a confidence** —
  four scalar-confidence kills say so, and the German false "yes" at 0.87 is a
  fifth; the format doc says it in so many words.
* `receipt` = the locate head's top span(s) over the question's rows: where the
  model looked, **not** a check of the answer.
* Refusals (400): a model row without `verdict_depth`; an empty question; more
  than 64 questions; an id reused for other text (the store's existing rule).

## 4. Engine

* **`run_lens_verdict`** in `server_lens.cpp` — a new function. It calls no
  modified code path: `run_lens_locate`, `run_lens_extract` and
  `run_lens_verify` are untouched.
* **One document pass per request**, then one short pass per question that
  resumes from its snapshot (the VERDICT2 shape): truncated after
  `verdict_depth`, `want_logits = true`, locate head tapped for the receipt.
  A request with 12 questions reads the document once even without an id.
* **The kept document is shared with locate.** The verdict's document pass is
  computed exactly as locate's pass 1 — same graph builder, same attention
  implementation (the row's `locate_prefill_shape`), deeper cut — and is stored
  under `LensKeptRoute::Locate`. A deeper pass serves a shallower read (LOCWARM
  G2), so a UI that asks "where" and "whether" of one document pays for it once,
  and the store's 4 slots are not split three ways. **If gate G3 fails**, the
  verdict's pass stays materialized and gets its own route key
  (`LensKeptRoute::Verdict`) with its own cap, so it can never evict a locate or
  extract entry.
* Answer tokens: the first token of every spelling of yes/ja, no/nein,
  unclear/unklar, bare and space-led, disjointness checked at startup (BUNDLEB's
  "Nein is two tokens" defect).

## 5. Where it is served — the "no other feature gets worse" rule

| server | today | with the verdict |
|---|---|---|
| `--attention-lens` (full) | 33/33 blocks + output head | **serves `/v1/verdict`** — no load change |
| `--lens-verify-only` | 28/33 blocks, **no output head** | refuses — unchanged |
| `--lens-locate-only` | 20/33 blocks, no output head | refuses — unchanged |

The truncated servers load only `token_embd` and the blocks. The 9B's output
head is **not tied** to its embedding: `output.weight` is 4096 × 248320 Q6_K,
**~834 MB**. Serving the verdict on the verify-only server (whose 28 blocks
match `verdict_depth` exactly) would need that head loaded — a separate,
opt-in decision (phase 2), never a silent change to an existing mode.

> **Phase 2 LANDED 2026-09-27: `--lens-verdict`** (requires
> `--lens-verify-only`; refused on a full or locate-only server and on a row
> without `verdict_layer`). `Model::load_tensors(max_blocks, keep_output_head)`
> keeps the final norm and output weight on a partial load; `verdict_layer` is
> folded into the verify-only cut (no change on the 9B: 28/33). Gates: 7/7
> verdict responses (22 questions EN/DE, plus a kept document cold → warm)
> byte-identical to the full server; plain verify-only unchanged (footprint
> 5,025 MB at `-c 11264`, `/v1/verdict` still 404); the head costs +796 MB
> (4,577 → 5,373 MB at `-c 4096`); suite 1050/1050, HTTP 18/18, locate smokes
> on all three server kinds.

## 6. Calibration

`LensConstants` gains `verdict_layer` (27 → 28 blocks) and `verdict_provenance`,
**appended last** (the rows are positional aggregates — a field in the middle
silently shifts every pair after it), default −1 = refused. Only the **9B
Q4_K_M-pinned row** sets it: VERDICT2 ran on Q4_K_M only, and the licence rule
is per model *and* per quant. Q8_0, the 27B and the 35B refuse until measured.
`validated_envelope` threshold = the longest prompt the gate passed at, stored in
the provenance, not guessed.

## 7. Gates (all before landing)

| # | gate | bar |
|---|---|---|
| G1 | **other features unchanged** — LOCWARM, EXTWARM, LOCSPLIT and the full suite, compared against their outputs from before the change | byte-identical reports; suite green |
| G2 | the shipped `run_lens_verdict` reproduces VERDICT2 (I3, 28 blocks) on the same items | same answers (argmax) on every item |
| G3 | the verdict under the licensed split+flash document pass vs materialized | no answer changes; else materialized + own route key (§4) |
| G4 | warm == cold with `document_id`, including a locate kept first and a verdict served from it, and the reverse | bit-identical |
| G5 | live: full lens server answers; verify-only and locate-only refuse with a reason; Q8_0 refuses; nothing logged | pass |

## 8. Architecture triggers (§13)

A new endpoint, two new calibration fields, a new reader of the kept-document
store. architecture.md and lens-format.md are updated in the same change.

## 9. Out of scope

* Serving on `--lens-verify-only` (needs the output head; phase 2, opt-in).
* Hypotheticals and sums — documented non-uses, not fixed.
* A trained reader over hidden states (training ladder L1) — the stronger
  version of this readout; a separate decision.
* Other models and quants — refused until a VERDICT2 run licenses them.

## 10. Cost

About a day: the function and route (~half), the gates (G1–G4 reuse existing
legs; G2/G3 extend VERDICT2), docs.
