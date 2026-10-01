# VERDICT-IMG — the verdict readout on images

**STATUS: PROBED 2026-09-29..10-01.** Real paper (§9, 12 phone photos):
signature, date and printed stamp (at 0.9) all 12/12 — stamp by a thin margin
over show-through ghosts. Clean arm (§4): PASS on both
models, all three families. Harder arm (§7, 35B-A3B): signature and date PASS
on lures and scans; **stamp FAILS on scans** — a round logo and a printed
"RECEIVED" read as a stamp (1/10 lures answered no), although every real stamp
still scores above every lure (AUROC 1.000). Fresh set (§8): the original
question at a cut of 0.9 PASSES clean and scanned (100% / 98.3%); the stricter
question at 0.5 passes clean but FAILS on scans (86.7%, 12/20 lures). Whether an image input is
built on `/v1/verdict` is the user's decision (an architecture change).

## 1. The question

Can the verdict readout — P(yes) against P(no) off the prefill's last row,
nothing generated — answer yes/no questions about an image? It reads logits,
not attention, so the closed image-attention result
(`docs/note-image-prefill-tap-probe.md`) does not bear on it. The literature
does this routinely for photos (VQAScore, Q-Bench); the open points are
document images, our two Qwen vision models, and the yes-bias of VLMs.

## 2. Setup

* **Images** (`tests/perf/probe_verdict_img_render.swift`): synthetic delivery
  notes, no personal data, 1024 × 1440 px = 1440 image tokens. 5 base notes
  (company, items, stamp place/shape/colour, signature stroke vary) × all 8
  combinations of three marks → 40 images, balanced 20 yes / 20 no per family.
  Plus one blank page (the baseline).

  | family | mark | question |
  |---|---|---|
  | SIG | pen scribble in the "Received by (signature)" box, or the box empty | Is the delivery note signed in the 'Received by' box? |
  | STAMP | a rotated "RECEIVED" stamp, or none | Does the delivery note carry a stamp? |
  | DATE | a handwritten date on the "Delivery date:" line, or the line empty | Is the delivery date filled in? |

* **Prompt.** Image, then `Question: … / Answer with yes or no only.`, Qwen
  chat template, thinking off. One full-depth prefill per question through the
  production path (`prefill_multimodal`), encode once per image.
* **Readout.** p = softmax of the yes tokens against the no tokens (first
  token of every spelling, as `run_lens_verdict`); compliance = the top token
  of the whole vocabulary is a yes/no token.
* **Models.** Qwen3.8-27B Q3_K_M + its mmproj (dense, `qwen35` recipe);
  Qwen3.6-35B-A3B UD-Q3_K_XL + its mmproj (MoE, `qwen36`).

## 3. Bar (set before the run), per model, per family

* **G1 separability:** AUROC of p over the 40 images ≥ 0.90.
* **G2 raw:** accuracy at p ≥ 0.5 ≥ 90%. If G2 passes, no calibration is
  needed.
* **G3 baseline (only if G2 fails):** accuracy ≥ 90% after subtracting the
  blank page's logit for the same question (answer yes when
  logit(p) > logit(p_blank)).
* Compliance reported, not gated.

## 4. Results

| model | family | AUROC | raw @0.5 | min p on yes | max p on no | blank p | blank-subtracted | compliant |
|---|---|---|---|---|---|---|---|---|
| 27B dense | SIG | 1.000 | 100% | 0.999 | 0.004 | 0.023 | 100% | 100% |
| 27B dense | STAMP | 1.000 | 100% | 1.000 | **0.499** | 0.034 | 57.5% | 100% |
| 27B dense | DATE | 1.000 | 100% | 1.000 | 0.005 | 0.020 | 100% | 100% |
| 35B-A3B | SIG | 1.000 | 100% | 0.880 | 0.018 | 0.362 | 100% | 100% |
| 35B-A3B | STAMP | 1.000 | 100% | 0.987 | 0.015 | 0.195 | 100% | 100% |
| 35B-A3B | DATE | 1.000 | 100% | 0.986 | 0.016 | 0.205 | 100% | 100% |

G1 and G2 pass in every cell, so G3 does not apply; the blank-subtracted
column is reported only because it was planned.

* **The 27B confuses a signature with a stamp.** On an unstamped note, P(yes)
  to "carry a stamp?" is 0.03–0.05 when the note is unsigned and **0.34–0.41
  (max 0.499) when it is signed**. Right at 0.5, by a hair. The 35B-A3B shows
  no such pull (every no-item ≤ 0.018, whichever marks are present).
* **The blank page is a poor baseline.** Its P(yes) is not the model's "no"
  level: 0.02–0.03 on the 27B (below the signed-no stamp items, hence 57.5%),
  0.19–0.36 on the 35B-A3B (far above every real no). A blank page is read as
  "no document", not as "a document without the mark". A same-layout unmarked
  note is the natural anchor if one is ever needed.
* **No yes-bias on these items.** Both models say no with confidence on
  unmarked notes; the bias reported in the literature did not show on this
  prompt and these images.

**Cost (M1 Pro 32 GB, 1440 image tokens, full depth, one prefill per
question):**

| model | image encode (first question) | each question |
|---|---|---|
| 27B dense Q3_K_M | ~10.6 s | ~27.3 s |
| 35B-A3B Q3_K_XL | ~10.3 s | ~3.7 s |

The probe re-prefills the image for every question. A route would prefill it
once and resume each question from a snapshot (as `/v1/verdict` does for
text), so the per-question cost would drop to the question's own tokens —
inferred, not measured. The MoE caveat stands: on the 35B-A3B a kept pass is
not bit-identical to a cold one (`project_lens_drift_gate`), so a warm image
verdict there needs its own gate.

## 5. What it does not show

* Clean vector renders: no scan noise, skew, faint or partial stamps, real
  handwriting, photos of paper.
* No lures: a printed name in the signature box, a logo that looks like a
  stamp, initials, a date printed elsewhere on the page.
* One wording per family, five base layouts, English only, one run.
* Yes/no only — the text verdict's third answer (unclear) was not offered.

## 6. Decisions for the user

1. A harder arm (lures + degraded scans) before anything is built — the 27B's
   signature→stamp pull says the lures are where it will break.
2. Which model: the 35B-A3B is 7× faster per question and showed no
   cross-mark pull (its worst no 0.018 vs the 27B's 0.499), though its worst
   yes is lower (0.880 vs 0.999); the 27B is the dense one (bit-identical kept
   passes).
3. An `image` input on `/v1/verdict` — an architecture change.

## 7. Harder arm — lures and scans (35B-A3B only; bar set before the run)

**Why:** §4's clean notes were the easy case; the 27B's signature→stamp pull
says lookalikes are where it breaks. 35B-A3B only, for speed (user, 2026-09-29).

* **Images** (`render DIR hard`): per base (5) and family (3), five variants of
  the asked mark; the other two marks clean and present at random, so all three
  questions are asked on every image (450 questions).

  | variant | SIG | STAMP | DATE |
  |---|---|---|---|
  | yes | pen signature in the box | stamp | handwritten date |
  | yes-hard | faint grey thin signature | faint stamp (28% ink) or half off the page edge | small faint grey date |
  | no | empty box | none | empty line |
  | lure 1 (no) | typed name "J. Miller" in the box | round company logo with initials | order date printed elsewhere |
  | lure 2 (no) | the sender's signature at "Issued by:" | "Status: RECEIVED" printed in the stamp colour | grey "DD.MM.YYYY" placeholder in the field |

* **Two conditions**, same drawings: **clean** (PNG) and **scan** (±1.4° skew,
  blur 1.1 px, contrast 0.8, additive noise, paper tint, JPEG q = 0.45).
* **SIG wording changed** to "signed **by hand** in the 'Received by' box" so a
  typed name is unambiguously a no. STAMP and DATE unchanged.

**Bar, per family, per condition:**

* **H1:** AUROC ≥ 0.90 over the family's 75 questions.
* **H2:** accuracy at p ≥ 0.5 ≥ 90%.
* **H3 lures:** ≥ 9 of the 10 lure images answered no (the asked family's
  lure 1 + lure 2 over 5 bases).
* Yes-hard recall and compliance reported, not gated.

### Results (harder arm, 35B-A3B Q3_K_XL, 450 questions, 100% compliant)

| condition | family | AUROC | acc @0.5 | lures answered no | yes-hard found | H1 | H2 | H3 |
|---|---|---|---|---|---|---|---|---|
| clean | SIG | 1.000 | 100% | 10/10 | 5/5 | pass | pass | pass |
| clean | STAMP | 1.000 | 98.7% | 9/10 | 5/5 | pass | pass | pass |
| clean | DATE | 1.000 | 100% | 10/10 | 5/5 | pass | pass | pass |
| scan | SIG | 1.000 | 100% | 10/10 | 5/5 | pass | pass | pass |
| scan | STAMP | 1.000 | **88.0%** | **1/10** | 5/5 | pass | **FAIL** | **FAIL** |
| scan | DATE | 1.000 | 100% | 10/10 | 5/5 | pass | pass | pass |

Mean P(yes) by the asked mark's variant (min–max over 5 bases):

| | SIG clean | SIG scan | STAMP clean | STAMP scan | DATE clean | DATE scan |
|---|---|---|---|---|---|---|
| yes | 0.83 (0.78–0.88) | 0.89 (0.83–0.93) | 0.995 | 0.998 | 0.993 | 0.986 |
| yes-hard | 0.81 (0.74–0.88) | 0.80 (0.72–0.91) | 0.98 (0.95–1.00) | 0.99 | 0.99 | 0.84 (0.73–0.91) |
| no | 0.003 | 0.016 | 0.006 | 0.011 | 0.007 | 0.029 |
| lure 1 | 0.013 (typed name) | 0.040 | **0.44 (0.34–0.56)** (logo) | **0.80 (0.77–0.83)** | 0.005 (order date) | 0.059 |
| lure 2 | 0.029 (issuer signature) | 0.106 | 0.095 (0.05–0.22) (printed status) | **0.67 (0.30–0.83)** | 0.016 (placeholder) | 0.10 (0.04–0.22) |

Questions about the other two marks on the same images: 0/300 wrong.

* **Signature and date hold.** A typed name, the sender's signature elsewhere,
  an order date elsewhere and a grey placeholder are all answered no, clean and
  scanned; faint signatures and faint dates are all found. Scanning costs
  margin (faint date 0.99 → 0.84, lures up to 0.22) but no answers.
* **Stamp breaks on stamp-shaped things.** A round coloured logo reads as a
  stamp at 0.34–0.56 clean and 0.77–0.83 scanned; "Status: RECEIVED" printed in
  the stamp colour stays low clean (≤ 0.22) but reaches 0.83 scanned. The scan's
  blur and tint erase what separates ink from print.
* **It is a threshold failure, not a separation failure.** Every real stamp,
  faint and half-off-page ones included, scores ≥ 0.946 clean and ≥ 0.988
  scanned; every no, lures included, ≤ 0.563 clean and ≤ 0.827 scanned. A cut
  near 0.9 would have separated all 150 — found after the run, so it is a
  hypothesis for a fresh set, not a pass.
* **Part of this is the question, not the eye.** Whether a round company logo
  is "a stamp" is arguable even for a person looking at a blurry scan. A
  stricter question ("an ink stamp with a date or 'received'") is the other
  lever, again untested.
* The 27B's signature→stamp pull (§4) did not appear on the 35B-A3B here
  either (other-mark questions 0/300 wrong).

### What the harder arm does not show

Still synthetic: rendered, then degraded by filters, not printed and scanned or
photographed; one scan strength; five layouts; English; one run; 35B-A3B only.

### Decisions for the user (updated)

1. Stamp: test a stricter question and/or a per-question cut (≈ 0.9) on a
   **fresh** set of lures and scans — the cut must not be tuned on this set.
2. Real paper: a handful of printed-and-scanned or phone-photographed notes
   (synthetic content, no personal data) — the filters are only a stand-in.
3. Unchanged: which model; an `image` input on `/v1/verdict` (architecture).

## 8. Stricter stamp question, fresh set (35B-A3B; bar set before the run)

**Why:** §7's stamp failure was a threshold failure on a fuzzy word. Two
remedies, tested on images §7 never saw: a stricter question, and §7's
post-hoc cut (0.9) on the original question.

* **Fresh images** (`render DIR stamp2`): 10 new base notes (new companies,
  layouts, stamp positions, shapes and colours; new logo and status-text
  positions). Per base six variants: stamp, faint stamp, stamp half off the
  page, none, round logo (lure 1), "Status: RECEIVED" printed in the stamp
  colour (lure 2); signature and date present at random. Clean and scan → 120
  images, 30 yes / 30 no per condition.
* **Two questions on every image:**
  * STAMP (original): "Does the delivery note carry a stamp?"
  * STAMPX (strict): "Has the delivery note been stamped with an ink stamp (a
    rubber-stamp impression, not a printed logo or printed text)?"

**Bar, per condition (clean, scan):**

* **S1 — strict question at 0.5:** accuracy ≥ 90% (of 60) and ≥ 18/20 lures
  answered no.
* **S2 — original question at the cut 0.9:** the same two bars.
* Reported, not gated: original at 0.5 (does §7's failure replicate), AUROC,
  yes recall per variant.

### Results (fresh set, 35B-A3B Q3_K_XL, 240 questions, 100% compliant)

| condition | question | cut | AUROC | accuracy | lures answered no | stamps found | bar |
|---|---|---|---|---|---|---|---|
| clean | original | 0.5 | 1.000 | 91.7% | 15/20 | 30/30 | (reported) |
| clean | original | 0.9 | 1.000 | 100% | 20/20 | 30/30 | **S2 pass** |
| clean | strict | 0.5 | 1.000 | 100% | 20/20 | 30/30 | **S1 pass** |
| scan | original | 0.5 | 1.000 | 76.7% | 6/20 | 30/30 | (reported) |
| scan | original | 0.9 | 1.000 | 98.3% | 19/20 | 30/30 | **S2 pass** |
| scan | strict | 0.5 | 1.000 | 86.7% | 12/20 | 30/30 | **S1 FAIL** |

Mean P(yes) (min–max over 10 bases):

| variant | original, clean | original, scan | strict, clean | strict, scan |
|---|---|---|---|---|
| stamp | 0.998 | 0.998 | 0.973 | 0.990 |
| faint stamp | 0.995 | 0.993 | 0.971 | 0.974 |
| half off the page | 0.982 | 0.992 | 0.937 (0.88–0.98) | 0.975 |
| none | 0.005 | 0.010 | 0.012 | 0.063 |
| logo (lure 1) | 0.48 (0.21–0.76) | 0.67 (0.37–**0.92**) | 0.08 (0.03–0.17) | 0.33 (0.19–0.59) |
| printed status (lure 2) | 0.11 (0.02–0.31) | 0.61 (0.16–0.89) | 0.07 (0.02–0.21) | 0.54 (0.20–0.75) |

* **§7's failure replicates on fresh images**: the original question at 0.5
  takes 14 of 20 scanned lures for stamps.
* **The cut is the remedy that held.** At 0.9 the original question keeps
  every real stamp (≥ 0.985 scanned, faint and half-off included) and drops
  19 of 20 lures; the one miss is a scanned logo at 0.920.
* **The stricter question fixes clean pages, not scans.** It pushes clean
  lures down to ≤ 0.21 but scanned ones stay at 0.19–0.75, and it lowers real
  stamps a little (half-off 0.88 clean). Naming "printed text" and "logo" in
  the question does not survive the blur that removes the difference.
* Separation is still perfect everywhere (AUROC 1.000). The strict question on
  scans would pass at a cut near 0.9 too (stamps ≥ 0.938, lures ≤ 0.747) —
  found after the run, a hypothesis only.
* What the cut means for a product: a stamp answer is "yes" only at ≥ 0.9;
  0.5–0.9 is "unclear — look". That is a per-question constant, the kind of
  calibration the lens rows already carry, and moving or adding one is the
  user's decision.

## 9. Real paper (35B-A3B; bar set before the captures)

**Why:** every image so far was rendered, and the "scans" were filters. Real
paper adds real pen and pencil, printer toner, a real scanner's compression,
and a phone photo's perspective, shadows and focus. The 0.9 stamp cut was set
on synthetic images and may move.

* **Sheets** (`render temp/paper paper` → `paper_sheets.pdf`, `checklist.md`,
  `truth.tsv`; in the git-ignored `temp/`): 12 A4 pages, new companies —
  6 delivery notes and 6 **invoices** (bills: "Date paid:", "Approved by
  (signature):", a PAID stamp; every invoice prints its invoice date, a
  natural lure for "Date paid"). Lures printed: typed name in the box, a
  printed "Issued by" signature, round logo, "Status: RECEIVED/PAID" text,
  order date elsewhere, DD.MM.YYYY placeholder.
* **By hand (the user):** signatures in pen (4) or light pencil (2) —
  made-up scribbles, never a real signature; dates in pen (4) or light pencil
  (1). Truth: signature 6 yes / 6 no, stamp 5 / 7, date 5 / 7.
* **No rubber stamp is available.** Stamps are **printed** (normal and faint),
  so the stamp arm tests printing and capture, not real ink — the ink cue that
  separated stamps from print in §7 is not in play.
* **Captures:** a phone photo of every sheet, and a scan where possible. Each
  capture converted and resized locally to a long side of 1440 px (the
  synthetic arms' budget) by `py/verdict_img_paper_prep.py`; nothing leaves
  the machine.
* Questions as §7/§8: signature ("signed by hand in the 'Received by' /
  'Approved by' box"), stamp ("carry a stamp?"), date ("delivery date" /
  "'Date paid' field filled in?").

**Bar, per capture type (photo, scan):**

* **P1:** signature and date at 0.5 — each ≥ 11/12.
* **P2:** stamp (printed) at the §8 cut 0.9 — ≥ 11/12.
* Reported, not gated: stamp at 0.5, AUROC, each lure and each pencil mark.

### Results (real paper, 2026-10-01, 35B-A3B Q3_K_XL, flash-attention encoder)

**Captures:** 12 phone photos (no scans), printed in colour, **double-sided** —
so several sheets show the reverse page through the paper: mirrored ghost
stamps on 01, 05, 06, 07 and a ghost logo on 03, none of them planned. Sheet
09's photo cuts off the bottom of the signature box; 04 is cropped on the
right. All pen/pencil marks checked by eye against the checklist before the
run: all present as listed. Phone JPEGs store landscape pixels plus EXIF
orientation 6; the engine's loader ignores EXIF, so `verdict_img_paper_prep.py`
now turns the pixels, resets the tag and refuses a page that is not portrait.

| family | cut | correct | AUROC | bar (≥ 11/12) |
|---|---|---|---|---|
| signature (4 pen, 2 light pencil) | 0.5 | **12/12** | 1.000 | **pass** |
| stamp (printed, incl. one faint) | 0.9 | **12/12** | 1.000 | **pass** |
| date (4 pen, 1 light pencil) | 0.5 | **12/12** | 1.000 | **pass** |

* **Signature and date are clean on real paper.** Every pen and pencil mark
  found (pencil signatures 0.95–0.98, pencil date 0.99); every no ≤ 0.07,
  including the typed name, the printed "Issued by" signature, the order and
  invoice dates printed elsewhere and the placeholders.
* **Stamp passes, but only just, and only because of the 0.9 cut.** Real
  stamps 0.959–0.998 (the faint one lowest). The no-items that came closest
  were the **show-through ghosts** — 0.862 (01, a mirrored ghost stamp), 0.54–0.57
  (03, 06, 07) — and the round logo on 04 at 0.864. At 0.5 the stamp would
  have been **7/12**; at 0.9 the gap between the highest no (0.864) and the
  lowest yes (0.959) is under 0.1.
* **A new real-world lure:** double-sided printing. A stamp on the back page
  shows through as a faint mirrored stamp and is read as a stamp at up to 0.86.
  No synthetic arm had it.

**What this does not show:** 12 sheets, one photo each, one model, one run;
printed stamps, not ink (no rubber stamp available); no scans.

### One real receipt (2026-10-01, user-supplied, kept local in `temp/`)

A real thermal-paper café receipt (a bill), phone photo on a table, German
text; questions and expected answers set before the run. 1530 image tokens.

| question | expected | P(yes) | |
|---|---|---|---|
| signed by hand? | no | 0.016 | ok |
| carries a stamp? (cut 0.9; a round printed logo is on it) | no | 0.013 | ok |
| a date printed on it? | yes | 0.986 | ok |
| a QR code on it? | yes | 0.975 | ok |
| a handwritten note on it? | no | 0.016 | ok |
| paid by card? *(reads text — outside the V1 scope)* | yes | 0.999 | ok |
| total more than 10 euros? *(reads a number — outside the V1 scope)* | no | 0.002 | ok |

7/7, every answer far from its cut. The black-and-white round logo did not
read as a stamp (0.013) — unlike the coloured round logos of §7/§8/§9. The two
reading questions came out right on one clear receipt; one example says
nothing about reading in general (the text verdict fails sums, §1 of
note-lens-verdict-probe.md).

## 10. What an image verdict costs (35B-A3B, measured 2026-09-30)

**Question:** every run above re-read the whole image for every question. A
route would read the image once and answer each question from there, the way
`/v1/verdict` does for text. What does each piece cost, and is the split exact?

**Setup:** `probe-verdict-img` COST mode, 4 images (2 clean notes, 2 scans), 10
questions each, 1440 image tokens. (A) the full re-read per question; (B) the
same prefill split at the end of the image span — image pass, then the
question alone — each part timed; the readouts compared.

| piece | time |
|---|---|
| image encode (vision tower), once per image | **10.0–10.6 s** |
| image pass (prefix + 1440 image tokens through the LLM) | 3.5–3.6 s |
| **the question alone** (~60 tokens) | **0.22 s** |
| full re-read per question (encode already cached) | 3.7–3.8 s |
| snapshot of a slot this size — *stand-in: a text slot of the same row count* | capture 0.13–0.15 s, restore 0.01–0.03 s, **119 MB** |

* **The split is exact.** Question pass started at the rope position after the
  image: margins bit-identical to the full re-read, 40/40 (Δ = 0, same top
  token 40/40). A first attempt started the question at the KV **row** count
  instead (prefix + 1440 rows, where M-RoPE has advanced only prefix +
  max(nx, ny)) and drifted by up to 1.3 logits — the exact trap the engine's
  rows-vs-positions bookkeeping exists for; the fix mirrors
  `drive_prefill_chunks`.
* **But the image pass cannot be kept yet.** On a hybrid model the DeltaNet
  state is overwritten by the first question, so asking a second one needs a
  copy of the post-image state — a snapshot — and `capture_slot` refuses a
  slot holding an M-RoPE image span ("VL sessions are not snapshottable in
  v1"; `architecture.md` §12, `plan-qwen35-vision-impl.md` §4 decision 3:
  the blob records a row count and no rope coordinate). Lifting it = a
  snapshot-header change, a named seam → the user's decision.
* **The encoder dominates.** 10 s of vision tower against 3.5 s of LLM for
  the same image; for one question it is ~70% of the cost. Not investigated
  here (why the ViT is this slow — attention shape, kernel path — is its own
  question). The existing `--image-embed-cache` skips it for a repeated image.

**Per image, N questions (encode included):**

| | N = 1 | N = 3 | N = 10 |
|---|---|---|---|
| today: full re-read per question | 13.9 s | 21.4 s | 47.6 s |
| image once, questions resumed (needs the snapshot change) | ~14.0 s | ~14.5 s | ~16.3 s |
| same, encode cached (image seen before) | ~3.9 s | ~4.4 s | ~6.1 s |

The resumed rows are computed from the measured pieces (image pass + capture
+ N × (restore + question)), with the snapshot priced by the text stand-in —
not a measured end-to-end run, since the snapshot is refused today.

## 11. Why the image encoder takes 10 s (measured 2026-09-30)

**Question:** §10 found the vision tower slower than the whole LLM pass on the
same image (10 s vs 3.5 s). Where does the time go?

**Ruled out:**
* **Not a CPU fallback.** `GGML_SCHED_DEBUG=1` on an encode: one split, all
  Metal. The M1 Pro reports `has bfloat = true`; the BF16 matmul kernels
  (`kernel_mul_mm_bf16_f32`) compile and run on the GPU.
* **Not warm-up.** Encode-only mode (`ENCODE=`): 10.5 / 10.1 / 10.0 s over 3
  reps of the same page.
* **Not the BF16 weights.** A layer's dense part with F16 weights is 5% faster
  (123 vs 129 ms), with F32 slower (148 ms).

**Found — one layer at the page's shape** (`bench-vit-layer`: width 1152,
16 heads × 72, FFN 4304, 5760 patches = 1024 × 1440 px; random weights):

| part | ms / layer | × 27 layers | share |
|---|---|---|---|
| dense (QKV, out, FFN) | 129.7 | 3.5 s | 36% |
| **attention, materialized (as the encoder)** | **234.3** | **6.3 s** | **64%** |
| attention, `ggml_flash_attn_ext`, K/V F32 | 124.7 | 3.4 s | |
| attention, `ggml_flash_attn_ext`, K/V F16 | 92.4 | 2.5 s | |

* The model of the layer reproduces the real encoder: 27 × (dense + attention)
  = **9.8 s** against 10.0 s measured.
* **The cause is the materialized attention:** every layer writes a
  5760 × 5760 score matrix per head (16 heads × 33 M floats ≈ 2.1 GB in F32),
  soft-maxes it and reads it back. Flash attention never materializes it.
* **With flash attention the encoder would take ~6.0 s (F16 K/V) or ~6.9 s
  (F32 K/V)** instead of 10 s — computed from the parts, not a run of a
  changed encoder. Metal's flash attention supports head size 72.
* **Not bit-identical:** on random Q/K/V the flash output differs from the
  materialized one by up to ~8e-5 (F32 and F16 K/V alike). The embeddings
  would shift slightly, so the verdict numbers of §4–§8 would need a re-run
  before trusting them on a changed encoder.
* **Gemma's SigLIP encoder** (`siglip_encoder.cpp`) uses the same
  materialized attention (4096 patches at 896 × 896) — the same change would
  apply there (the cross-family rule).
* The dense half (~1.35 TFLOP/s) is ordinary matmul work; nothing cheap
  found there.
* External reference not obtained: the local llama.cpp build (b9657, June)
  crashes loading this model.

**Per page, one question:** 13.9 s today → ~9.9 s with flash attention
(F16 K/V); the encoder share drops from ~73% to ~61%.

### Landed (2026-09-30, user decision: flash attention, F16 K/V, no flag)

Both encoders (`qwen3vl_encoder`, `siglip_encoder`) now use
`ggml_flash_attn_ext`. Measured on the real towers:

| check | result |
|---|---|
| Qwen page encode (35B-A3B mmproj, 1024 × 1440) | **10.0 → 6.1 s** (first question per page 14.1 → 9.9 s) |
| Qwen embeddings vs the old encoder (2 pages) | rel-L2 1.0–1.2%, worst token cosine 0.993; F32 K/V drifts the same (1.2%, 0.990) — the reduction order, not F16 |
| SigLIP vs the captured llama.cpp reference | **closer**: rel-L2 2.94e-3 → 2.59e-3, min cosine 0.999977 → 0.999984 (coarse gate 4/4; bitwise stays DISABLED) |
| `qwen3vl-encoder-tests` / `multimodal-prefill-tests` (Gemma e2e) | 11/11 / 4/4 |
| MedGemma's greedy description of the e2e X-ray, old vs new encoder | word-for-word identical (the CLI rebuilt from each encoder version — `--target qwenium-cli`; a first attempt built the non-existent target `qwenium` and compared one stale binary with itself) |

**The image verdict re-run on the new encoder (35B-A3B, all 813 questions of
§4, §7, §8):** every gate verdict unchanged — §4 all pass, §7 signature and
date pass / stamp-on-scans fails as before, §8 S2 (original question at 0.9)
passes clean and scanned, S1 (strict question at 0.5) fails scans.

* §4 and §7: **0 of 573 answers flipped**; largest shift 0.15 (a clean logo
  lure, 0.34 → 0.46).
* §8: 5 answers flipped **at 0.5**, every one within 0.06 of 0.5 (lures
  sitting on the line: 0.503 → 0.493, 0.428 → 0.502, …); **none at the gated
  0.9 cut** of the original question. S1's scanned lures went 12/20 → 10/20
  (already a fail).
* A 0.5 cut on lookalike lures sits exactly where encoder noise decides —
  one more reason the stamp needs the 0.9 cut.

## 12. Stamp lures found by the client (2026-10-01)

Printed badges drawn like a stamp and stamps showing through from the back of the
sheet score like real stamps (up to 0.995); no cut separates them; no real stamp
ever scored below 0.5. Full report, fresh set (`render DIR stamp3`) and options:
[note-stamp-lures.md](note-stamp-lures.md). §9's "lures up to 0.86" holds only
for the lures that were on the paper.
**Decision (user, 2026-10-01): the stamp mark never answers yes** (cut 1.0 / 0.5),
so a stamp answer is `no` or `unclear`.

## Reproduce

```
swiftc -O tests/perf/probe_verdict_img_render.swift -o .session-results/verdict_img/render
.session-results/verdict_img/render .session-results/verdict_img
cmake --build build-metal --target probe-verdict-img
OUT=.session-results/verdict_img/r27.tsv ./build-metal/bin/probe-verdict-img          # ~60 min
MODEL_PATH=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf MMPROJ_PATH=models/Qwen3.6-mtp-mmproj-BF16.gguf \
  OUT=.session-results/verdict_img/r35.tsv ./build-metal/bin/probe-verdict-img        # ~15 min
python3 py/verdict_img_score.py .session-results/verdict_img/r27.tsv .session-results/verdict_img/r35.tsv
# harder arm (§7), ~55 min on the 35B-A3B
.session-results/verdict_img/render .session-results/verdict_img_hard hard
MODEL_PATH=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf MMPROJ_PATH=models/Qwen3.6-mtp-mmproj-BF16.gguf \
  DIR=.session-results/verdict_img_hard OUT=.session-results/verdict_img_hard/r35.tsv ./build-metal/bin/probe-verdict-img
python3 py/verdict_img_score.py .session-results/verdict_img_hard/r35.tsv
# fresh stamp set (§8), ~35 min
.session-results/verdict_img/render .session-results/verdict_img_stamp2 stamp2
MODEL_PATH=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf MMPROJ_PATH=models/Qwen3.6-mtp-mmproj-BF16.gguf \
  DIR=.session-results/verdict_img_stamp2 OUT=.session-results/verdict_img_stamp2/r35.tsv ./build-metal/bin/probe-verdict-img
python3 py/verdict_img_score.py .session-results/verdict_img_stamp2/r35.tsv
# real paper (§9): print temp/paper/paper_sheets.pdf, mark per checklist.md, capture into temp/paper/
.session-results/verdict_img/render temp/paper paper
python3 py/verdict_img_paper_prep.py temp/paper
MODEL_PATH=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf MMPROJ_PATH=models/Qwen3.6-mtp-mmproj-BF16.gguf \
  DIR=temp/paper/img OUT=temp/paper/img/r35.tsv ./build-metal/bin/probe-verdict-img
python3 py/verdict_img_score.py temp/paper/img/r35.tsv
# cost (§10), ~5 min
COST=.session-results/verdict_img/b0_s1t1d1.png,.session-results/verdict_img/b3_s0t1d0.png,.session-results/verdict_img_hard/s_b0_SIG_yeshard_s1t1d1.jpg,.session-results/verdict_img_stamp2/s_f3_STAMP_partial_s0t1d1.jpg \
  MODEL_PATH=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf MMPROJ_PATH=models/Qwen3.6-mtp-mmproj-BF16.gguf \
  OUT=.session-results/verdict_img_cost/r35.tsv ./build-metal/bin/probe-verdict-img
# encoder (§11)
ENCODE=.session-results/verdict_img/b0_s1t1d1.png MODEL_PATH=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  MMPROJ_PATH=models/Qwen3.6-mtp-mmproj-BF16.gguf ./build-metal/bin/probe-verdict-img
./build-metal/bin/bench-vit-layer            # WTYPE=bf16|f16|f32, N_POS=5760, REPS=5
```
