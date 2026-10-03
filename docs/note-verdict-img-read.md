# Can the model read what is printed on a page? (after the 2026-10-02 fixes)

2026-10-02. Qwen3.6-35B-A3B UD-Q3_K_XL + Qwen3.6 mmproj, `build-metal`, with the
two image-generation fixes of docs/note-verdict-img-ground.md. Synthetic pages
only. Results in `.session-results/verdict_img_read/`.

## Result in short

**Every read was exact: 120/120 table rows (40/40 tables), 20/20 totals, 40/40 document
numbers, clean and scanned.** On 2026-10-01, reading a real receipt failed
(note-verdict-img-probe.md, "reading text from images", parked by the user).
That ran through the decode path whose fault is now fixed. The receipt itself
was not re-tested (the user's private photo).

## Method

- Pages: `render DIR read` (tests/perf/probe_verdict_img_render.swift). The 10
  layouts of §8 as a delivery note (qty, description) and as an invoice (plus
  an amount column and a total), signed and dated; clean PNG and degraded scan
  = 40 pages, 120 item rows. `read_truth.json` = what is printed.
- Questions (`py/verdict_img_read.py run`), `/v1/chat/completions`, greedy, no
  thinking:
  - TABLE: "Read the items table of this {doc}. Output only JSON: a list with
    one object per row, with the keys "qty" (number), "description" (text)
    [and "amount" (number, without the currency)]."
  - TOTAL (invoices): "What is the total amount on this invoice? Answer with
    the number only, without the currency."
  - NUMBER: "What is the document number of this {doc}? Answer with the
    number only."
- Scoring (`score`): a row is right only if qty, description (case and spacing
  ignored) and amount (to the cent) all match, in order; a table is right only
  if every row is and there are no extra rows.

## Results

| condition | document | rows | tables | totals | numbers |
|---|---|---|---|---|---|
| clean | delivery note | 30/30 | 10/10 | — | 10/10 |
| clean | invoice | 30/30 (amounts 30/30) | 10/10 | 10/10 | 10/10 |
| scan | delivery note | 30/30 | 10/10 | — | 10/10 |
| scan | invoice | 30/30 (amounts 30/30) | 10/10 | 10/10 | 10/10 |

Cost: a table answer ~12.9 s cold (encode + prefill + ~60–100 tokens).

## What this does and does not show

- It shows the model reads clean printed tables on a 1440-token page, also
  through a degraded scan. It does not show handwriting, small print, dense or
  real-world receipts, or photos at an angle.
- A generated read is the model's transcription, not a receipt. Nothing here
  says where on the page a value came from (a box could, see the ground note).
- Not measured: the same pages with the decode fault switched back on (two 35B
  servers do not fit in 32 GB). The link to the 2026-10-01 failure is likely,
  not proven.

## 2026-10-03 — Qwen3.8-27B (+ its own mmproj) vs Qwen3.6-35B-A3B

Question: can one model both read and run the text lens? The 27B has a lens
row (extract/verify, locate, choice, absence) and a projector on disk; two
models together do not fit in 32 GB. Same pages, same prompts, chat route,
greedy, no thinking.

**Reading questions (40 pages, 100 questions):** both exact on every table,
total and document number, clean and scan (27B: 0 misses, like the 35B).
Time per table answer, cold: 35B ~13 s, 27B 52.8 s (median).

**Whole page as lines with boxes** (`lines`, 10 pages, 250 printed runs;
prompt: "Read all the text on this page, line by line. Output only JSON: a
list with one object per line, with the keys "text" and "bbox_2d"."):

| model | runs read | read and box covers ≥ half | lines on nothing | broken JSON pages | s/page (median) |
|---|---|---|---|---|---|
| 35B-A3B | 176/250 | 160 | 0 | 2/10 | 51 |
| 27B | 250/250 | 239 | 0 | 0/10 | 214 |

- The 35B's 74 missing runs come almost all from two pages whose JSON broke
  (all lines packed into one object; boxes without text). Its reading on the
  other 8 pages is near-complete. A grammar-constrained output is the
  obvious fix (not tested).
- The 27B's 11 box misses are short runs (single-digit quantities, the small
  grey "Name / date" caption, one document number on a scan).
- The 27B runs on past its answer (`<|im_start|>user`): its end-of-turn token
  is not a stop token in the server. Cosmetic; the scorer cuts it off.

**Answer:** the 27B can be the single model for reading + the text lens. It
reads as well or better, with sound structure, at about 4x the time.
Missing for that role: compare and the document verdict are not calibrated on
the 27B (head hunts).
