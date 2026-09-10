# OCR + Lens — probe spec

**Status: specified, not run.** No engine change is proposed and none is
needed. Everything below is a measurement.

## 1. The question

Does the calibrated text lens hold when the `document` is machine-extracted
from a page — a PDF text layer or OCR output — rather than clean typed text?

If yes, `/v1/extract` serves scanned and PDF documents today, with real
citations that map back to **pixel regions on the page**, and with no image
model in the loop at all.

## 2. What this is not

This is **not** a revival of the image lens. That is closed on both taps
(`note-image-lens-probe.md`, decode, 2026-08-31; `note-image-prefill-tap-probe.md`,
prefill, 2026-09-03) and stays closed. Nothing here needs a new mechanism for
attention over patches, because **the model never sees the image.** It reads
text, which is the case the lens is calibrated for.

Nor is it model transcription. "Transcribe this page" is a generation, not a
reading: the model chooses reading order, table representation, what counts as
content, whether to normalize. A `found_in_document` check against a text the
same model wrote moments earlier is close to circular. A PDF text layer or an
OCR engine is a **measurement** — it does not invent a line that was not there,
and it carries per-glyph geometry. That geometry is the whole point.

## 3. The two variables, separated

Two things could break, and lumping them makes the result unattributable.

| variable | isolated by |
|---|---|
| Does the lens survive **layout-broken text**? (hard line breaks at column boundaries, hyphenation, irregular spacing) | **Arm T** |
| Is the **extraction accurate enough** to start from? (character errors, dropped regions, reading order) | **Arm O** |

**Arm T — text layer.** Digital PDF → embedded text + bounding boxes
(`pdftotext -bbox` or equivalent). Character error rate is **zero by
construction**. Arm T therefore measures exactly one thing: the lens on text
whose line structure came from a page layout rather than from prose.

**Arm O — OCR.** Rendered or scanned page image → OCR text + boxes (Apple
Vision; on this platform it is free and on-device). Arm O carries Arm T's
question *plus* extraction error.

Run T first. If T fails, O cannot pass and the idea dies for under a day.

Note what Arm T is on its own merits: for digital PDFs there is **no OCR step
at all**, and that is likely the largest and lowest-risk share of real
documents. Arm T passing is a shippable result even if Arm O never does.

## 4. Corpus

**Paired, against the existing baseline.** `qdocs_messy_corpus()`
(`tests/perf/attn_provenance.cpp`) is 15 documents, EN+DE, 75 labelled fields,
and the clean-text numbers on it are already published. Render each document
to a page, extract it back through T and O, and re-run the same fields. Every
delta is attributable to the round trip, because nothing else moved.

- **T1** — 15 corpus docs → PDF → text layer + boxes. `n = 75` fields.
- **O1** — the same 15 rendered pages → image → OCR. `n = 75`. Optimistic:
  clean fonts, no skew, no scan noise. This is a **ceiling**, not a forecast.
- **O2** — 5 real scans or photographs, hand-labelled (`n ≈ 25`). The floor.
  Only run if O1 clears bar 5.
- **T2/O2-col** — one deliberately multi-column page, added to both. Reading
  order is the failure mode a single-column business-document corpus is blind
  to, and the existing corpus is blind to it.

**Fixtures** land under `tests/fixtures/ocr/` as `<doc>.txt`,
`<doc>.boxes.json`, `<doc>.png`, generated offline by a script and committed.
The harness reads fixtures; it does not run OCR. This keeps the engine test
free of a platform dependency and makes the arm reproducible.

## 5. Bars

Model: **Qwen3.8-9B** (`qwen35`/33, L27H13) — the better-measured head.
Harness: `tests/perf/attn_provenance.cpp`, env-gated arm `OCRT=1`, built in
`build-metal` (Release + Metal). Not `build/`, which is CPU-only Debug.

**Selecting the head is not automatic.** `run_qdocs_leg_c` (the arm this clones)
defaults to `FROZEN_SLOT=0, FROZEN_HEAD=13` — that is **L3H13**, the Qwen 3.6
constants, which on Qwen 3.8 are the known-failing case (84% top-3,
`note-lens-qwen38-probe.md` §5.2). L27H13 requires
`ATTN_FROZEN_SLOT=6 ATTN_FROZEN_HEAD=13` on **both** the baseline and Arm T.
Reproducing §5.3's **98% top-3 (404/413)** on the baseline is what proves the
override actually reached the citation read — an earlier version of this code
accepted the override and silently kept reading L3H13 while printing the L27H13
label. The published number is the propagation check; do not skip it.

### Bar 1 — citation survives layout-broken text  *(Arm T, then O)*

Top-3 citation-in-span on the **`distinctive`** tier, the class the format
vouches for. Report top-1 alongside.

- **Baseline**: the same fields in clean text (98% top-3, `note-lens-qwen38-probe.md`
  §5.3; 89%/98% at thread scale, `note-ss2-thread-alarm.md` Gate 0).
- **Pass**: ≥ 0.90 top-3, the standing gate-1 bar.
- **Report the paired delta**, not just the absolute — the delta is the finding.
- **Kill**: < 0.90 on Arm T. Layout-broken text defeats the head, and no OCR
  quality rescues that.

`short_numeric` fields are measured and reported but **not barred** — they are
~65% in clean text and the format already says so.

### Bar 2 — zero confident false receipts  *(both arms)*

The standing fidelity invariant, unchanged, applied to a new input shape: no
`grounded`, `distinctive`, verbatim field whose top citation lands on no
occurrence of its value.

- **Pass**: exactly 0.
- **Kill**: any non-zero. This is not a rate.

### Bar 3 — byte span → pixel box is exact  *(both arms)*

Not a statistical bar. For every citation, the mapped rectangle set must cover
the pixels of the cited bytes.

The mapping must be an **index built while the document string is assembled**,
never a re-search of the text. A re-search reintroduces exactly the silent-drop
class the candidate finder was carrying until it was fixed — and here the
failure is invisible: highlights drift onto neighbouring pixels and nothing
surfaces it. A span crossing a line or column break maps to **several**
rectangles; a mapping that returns the first one is wrong.

- **Pass**: 100% by construction, plus 20 rendered crops checked by eye.
- **Kill**: any drift.

### Bar 4 — the false-absent bar  *(the dangerous one)*

Every labelled field **is** on the page. So every `absent` returned is a false
absent — and the omission report is the lens's one non-commodity asset. A
dropped OCR line produces a confident "not on this page" that nothing
downstream can catch.

Split the causes, because they have different owners:

- **extraction loss** — the ground-truth value string does not appear in the
  document at all. OCR's fault.
- **model loss** — it appears, and the model still returned `absent`. The
  lens's fault.

- **Pass**: model loss matches the clean-text baseline. Extraction loss ≤ 2%
  (≤ 1 of 75) on Arm O, **every instance root-caused**, none on Arm T.
- **Kill**: extraction loss above that, unless a page-coverage heuristic
  (box area vs non-whitespace page area) flags every instance — in which case
  the finding is "shippable with a coverage warning," which is a different and
  weaker product. Record it as such; do not quietly count it as a pass.

### Bar 5 — is OCR good enough to start from?  *(Arm O only, gates O2)*

Fraction of the 75 ground-truth value strings present verbatim in the OCR text.
Report CER alongside.

- **Pass**: ≥ 0.95. **Kill**: below it — OCR is the bottleneck and lens
  quality is irrelevant.

## 6. What this does not measure

- **Whether the extraction matches the page.** Bar 5 measures value recall,
  not fidelity of everything else on the page. A misread in an unlabelled
  region is invisible here. The product answer is that the extracted text is
  shown to the user; the probe answer is that it is out of scope.
- **Coverage / `consulted` on page-derived text.** `coverage_used_peak = 0.705`
  is the weak arm on both calibrated models (87%/84% against a ≥90% bar) and
  was never searched. Adding a bar it cannot clear in the clean case would tell
  us nothing about the OCR case.
- **Handwriting.** Out of scope for both arms.
- **Anything about the image lens.** Still closed.

## 7. Cost and kill order

| stage | work | kills on |
|---|---|---|
| Fixture script (render → PDF/PNG → text+boxes) | ~0.5 day, offline, no engine | — |
| Arm T + bars 1–4 | ~0.5 day, harness arm | bar 1 or 2 |
| Arm O1 + bar 5 | ~0.25 day | bar 5 |
| Arm O2 (real scans) | ~0.25 day | bar 4 |

~1.5 days total, front-loaded so the cheapest arm kills first. **No engine
change at any stage** — this measures shipped code on a new input shape.

## 8. Results — Arm T, 2026-09-08

**Bar 1 PASS, bar 2 PASS, bar 4 PASS.** Qwen3.8-9B-Q8_0, L27H13
(`ATTN_FROZEN_SLOT=6 ATTN_FROZEN_HEAD=13`), `build-metal`, arm `OCRT`.

| | top-1 | top-3 | n (value tokens) |
|---|---|---|---|
| baseline, clean text | 369/413 (89%) | **404/413 (97.8%)** | 413 |
| Arm T, PDF round-trip | 348/384 (91%) | **377/384 (98.2%)** | 384 |

The baseline reproduces `note-lens-qwen38-probe.md` §5.3 (404/413) exactly,
which is the propagation check §5 requires — the override reached the citation
read rather than silently scoring L3H13.

- **Bar 1**: paired delta on top-3 is **+0.4 points**. Flat. Pass (≥0.90).
- **Bar 2**: 0 confident false receipts on both arms (0/74 baseline, 0/70 Arm T).
- **Bar 4**: 1/75 false-absent (`m_de2.customer` — the model writes
  `"A. Schmitt"` for `"Nordlicht GmbH"`). **The identical field fails identically
  in the baseline**, so model loss is unchanged by the round-trip, which is the
  pass condition. It is a pre-existing corpus failure, not layout-induced.
- Coverage `used-clear` is 87% on **both** arms — unchanged, and not barred here
  (§6). Note that leg C's printed `VERDICT: FAIL` line folds used-clear into its
  own verdict, so **both** logs say FAIL. That is the harness's own bar, not
  this probe's; bars 1/2/4 pass.

### The survivorship caveat — read before quoting 98%

`n` fell 413 → 384 because **4 of 75 field values were destroyed by the wrap**
and never entered the scored population:

| field | broken as |
|---|---|
| `m_en1.customer` "Brightwork Studios Ltd" | `Brightwo` / `rk Studios Ltd` |
| `m_en4.order_date` "2025-11-11" | `20` / `25-11-11` |
| `m_en6.unit_price` "27.50" | `27.5` / `0` |
| `m_de1.delivery_date` "2025-10-27" | `2025` / `-10-27` |

So the honest statement is **"of the values that survive the round-trip,
citation quality is unchanged"** — not "layout damage is free." End-to-end
field yield is **71/75 (94.7%)**, and those 4 losses are invisible in the top-3
number. The top-1 rise (89% → 91%) is survivorship too: the dropped set includes
a multi-token name, which is the weak citation class.

### The damage was harsher than a real PDF, and that cuts both ways

`cupsfilter` wraps plain text at a fixed ~80-column boundary **by character, not
by word** — hence `Brightwo|rk` with no hyphen. A real digital PDF carries
word-level layout and does not break mid-word unhyphenated. So:

- citation robustness is measured against damage **worse** than reality — the
  pass is conservative, and stronger than the bar asked for;
- the 5.3% field loss is an **artifact of the fixture pipeline**, not a forecast.
  A real digital PDF would lose fewer values, possibly none.

Both lean the same way: the real Arm T result is at least this good. Neither is
measured. A follow-up on genuine PDFs (an ERP invoice, a Word export) is what
would settle the field-loss number; it does not change bar 1.

### Also worth recording

- Leg C has **no tier split**, so bar 1 was scored **all fields together**
  including `short_numeric` (~65% in clean text). That is harder than the bar,
  which was specified on the `distinctive` tier alone. Passing at 98% all-fields
  is therefore a stronger result than §5 asked for.
- Scope: one model, one corpus, one damage class. Not a general claim.
- Running L3H13 (the Qwen 3.6 head) on Qwen 3.8 gives 84% top-3 — the known
  §5.2 failure. It is the arm's default. Every run here needed the override.

## 9. Arm O1 — blocked on the fixture pipeline, 2026-09-08

**Bar 5 FAIL: 53/75 (70.7%) value recall, against a ≥0.95 gate.** Arm O was not
run; per §5 the gate exists precisely to stop before a pointless model run.

**The cause is our fixture generator, not OCR.** Apple Vision's line segmenter
merged visually adjacent source lines into single observations on the tightly
leaded `cupsfilter` render — at least one merged observation (≥1.6× median line
height) in **11 of 15 documents** — and text inside a merged region comes back
scrambled, destroying every field in that block at once. `m_en7` lost all five
of its fields to one merged paragraph.

Loss decomposition: 4 fields were already destroyed by Arm T's character-wrap
(same PDF feeds both pipelines, so this is inherited damage, not OCR's), and
**18 were newly lost to the merge failure**.

Two independent signals say the render, not the recogniser, is at fault:
segmentation failure is geometry-driven and 11/15 is systematic rather than
random; and the residual errors are implausible for a mature OCR on a clean
200 DPI page (`Thanks`→`hanks`, `for`→`tor`, `go`→`ao`, dotless `ı`).

**`cupsfilter -m application/pdf` has now injected two unrepresentative damage
classes** — character-level wrapping (§8) and tight leading (here). It is not a
valid stand-in for a document page, and both arms inherit its defects. Real
documents wrap at word boundaries and are set with normal leading.

**Fix**: render fixtures with AppKit/CoreText (`NSAttributedString` at a real
font and realistic leading) instead of `cupsfilter`. That is word-wrapping by
construction, so it repairs §8's four broken values as well. Re-running **both**
arms on the new fixtures is required — a paired comparison needs both sides from
the same render.

Bar 5's threshold stands. Nothing here is evidence about OCR quality on real
documents; it is evidence that this probe had not yet produced one.

The harness wiring for Arm O (`OCR_SUFFIX` env on `run_ocr_t_probe`, `OCRO=1`
selecting `.ocr.txt`) is in place and was never exercised.

## 10. Arms T and O on a real render — both PASS, 2026-09-08

Fixtures regenerated with an AppKit/CoreText renderer (Helvetica 11pt, 1.4× line
height, native word wrapping, single tall page) replacing `cupsfilter`. Both
arms derive from the **same PDF per document**, so the pairing is exact:
`pdftotext -layout` → Arm T, `pdftoppm -r 200` → Apple Vision → Arm O.

**Bar 5 first, offline: 75/75 on both.** The PDF text layer loses nothing (word
wrapping does not split values), and **Apple Vision loses nothing either** —
70.7% → **100%**. The §9 failure was the fixture in its entirety, not OCR.

| arm | top-1 | top-3 | n | false-alarm | bar 4 | used-clear |
|---|---|---|---|---|---|---|
| baseline, clean text | 369/413 (89%) | **404/413 (97.8%)** | 413 | 0/74 | 1/75 | 87% |
| **Arm T**, PDF text layer | 366/408 (90%) | **399/408 (97.8%)** | 408 | 0/73 | 2/75 | 87% |
| **Arm O**, Vision OCR | 371/413 (90%) | **402/413 (97.3%)** | 413 | 0/74 | 1/75 | 85% |

**The two deltas, decomposed:**

- **baseline → Arm T = −0.03 pts.** Page layout costs nothing.
- **Arm T → Arm O = −0.45 pts.** OCR character error costs nothing.
- baseline → Arm O = −0.48 pts end to end.

**Bar 1 PASS** on both arms (≥0.90 top-3), with margin. **Bar 2 PASS** — zero
confident false receipts on every arm. **Bar 5 PASS.**

**Bar 4**: Arm O matches the baseline exactly (1/75, `m_de2.customer`, the
pre-existing corpus failure). Arm T carries **one extra**: `m_de3.customer`,
where the model emitted **`","`** — a bare comma — instead of
`"steinweg baumarkt"`, at `cov=0.828`. It attended to the span strongly and
produced a comma anyway. That is a genuine extraction failure on Arm T only, not
a formatting nicety; the harness's `NORMALIZED (not verbatim in output)` label
undersells it. Arm O and the baseline both get this field right, so it is not a
layout effect either — it is a single-field, single-arm miss with no established
cause.

**Why `n` is 408 on Arm T — settled by the per-bucket split.** The EN bucket is
**218 on both arms**; the DE bucket is 190 vs 195. The whole 5-token gap sits in
DE, which is exactly where the extra excluded field is. Both arms' EN documents
pass through the same `pdftotext -layout`, so if column-alignment whitespace were
shifting BPE boundaries the EN count would move too. It does not. `n = 408` is
the one excluded field and nothing else.

### What this establishes, and what it does not

Established: **the calibrated text lens is indifferent to how the text reached
it.** Three populations — clean prose, PDF-layout text, and OCR output — agree
on top-3 within half a point, across two unrelated damage classes (character
wrap in §8, word wrap plus recognition error here). The head is reading content,
not formatting.

Not established, and it matters: Arm O1 is the **ceiling**, exactly as §4 said —
a 200 DPI render of a digitally-generated page, clean fonts, no skew, no scan
noise, no photograph. **Arm O2 (real scans) is unrun and needs source
documents that do not exist in this repo.** Nothing here forecasts performance
on a phone photo of a crumpled invoice. Also unrun: **bar 3**, the byte-span →
pixel-box mapping, which is what actually puts a highlight on the page. The
boxes are captured in `/tmp/ocr_probe/<tag>.boxes.json` so a bar-3 run need not
re-OCR.

**One residual worth watching.** Vision still merges visually adjacent lines
into a single observation in **3 of 15 documents** (down from 11 of 15 on the
`cupsfilter` render). None of the three hit a labelled field value, which is
partly luck — the failure mode that destroyed §9's run is reduced, not
eliminated, and it is the thing to check first if a future corpus scores badly.
Other residual OCR errors are cosmetic: a dropped apostrophe (`We'll` → `Well`),
an em-dash flattened to a hyphen. No scrambled text anywhere.

Scope: one model (Qwen3.8-9B, L27H13), one corpus (15 docs, EN+DE, 75 fields),
one OCR engine, single-column business documents. Multi-column reading order is
untested (§4's T2/O2-col was not built).

### Log-reading traps

- Leg C's printed `VERDICT` line folds `used-clear` into its own bar, so **every
  arm prints `FAIL`** while bars 1/2/4/5 pass. Coverage is the known-weak,
  deliberately unbarred arm (§6).
- Arm O reuses `run_ocr_t_probe` via `OCR_SUFFIX`, so its log also prints
  **`ARM T VERDICT`**. The env var is what distinguishes the runs, not the label.
- Every run needs `ATTN_FROZEN_SLOT=6 ATTN_FROZEN_HEAD=13`. Without it the arm
  silently scores L3H13 at 84%.

## 11. Bar 3 — byte span → pixel box, Arm O, 2026-09-08

240 citations (every `fields[].citations[]` entry, not just top-1) across 6
documents, via the **shipped route** — `http_server --attention-lens` on
Qwen3.8-9B, real `POST /v1/extract` responses, not harness internals.
Per-character boxes from `VNRecognizedText.boundingBox(for:)`; index built during
document assembly, never by re-search, as §5 requires.

### PASS on the thing that could have been catastrophically wrong

**Zero positional drift.** No citation mapped to text elsewhere on the page. The
bottom-left→top-left Y flip and the normalised→pixel scaling are both correct,
confirmed visually on annotated full pages (`/tmp/ocr_probe/bar3/`, 20 files):
highlights land tight on the right word of the right line.

### NOT established at the granularity §5 asked for

The automated check compared crop re-OCR to the cited substring **by
containment**, which cannot separate "exact box" from "box covering the whole
enclosing word." The data says it is the latter:

- **195 of 232 citations produce a crop that reads longer than the cited span.**
- Median cited span: **1 byte**. Median crop text: **6 characters**.
- Concretely: `' Berg'` → crop reads `bergblick`; `'ü'` → `stück`;
  `'licht'` → `nordlicht`.

So the spec's "must cover the pixels of the cited bytes" is satisfied — and
overshot. Boxes are **word-level, not character-level**. Cause is not isolated:
Vision's per-character boxes are documented as approximate, and the index may
also be coarse. Not separated by this run.

**This is probably the right granularity anyway**, and it points at a client
rule. A citation's natural unit is a *source token*, which BPE often makes a
single digit or word-piece; a highlight's natural unit is the value. Highlighting
one character of `2025-09-30` would be useless. **The client should union the
rectangles of all of a field's citations**, which yields a value-level highlight
by construction. What must not be claimed is character-exact geometry.

### Gaps

- **Multi-line spans: 0 of 240.** Every real citation fell inside one OCR line,
  so the multi-rectangle union — the failure §5 explicitly names ("a mapping that
  returns the first one is wrong") — was exercised only on a synthetic span. The
  riskiest branch is still untested against real data.
- **6 citations are a single whitespace byte** (space or `\n`) and correctly map
  to **zero** rectangles. A client must render an empty rectangle set as *no
  highlight*, not as an error or a stray box.
- **3 crops too small for Vision to re-read** (`'A'`, `' x'`, `' |'`) — a limit of
  the verification method, not evidence of drift.
- **1 crop mismatch, upstream**: `' Well'` vs `we'll` — the §10 apostrophe drop in
  the original OCR pass. The box is pixel-exact; the document text is what is
  wrong. This is the predicted good behaviour: the citation stays true and the
  highlight walks you to the misread pixels.

**Verdict: bar 3 passes on positional correctness, which is what would have
sunk the idea. Character-exact geometry is unproven and probably not wanted.
The multi-line path needs a document that actually wraps a value.**
