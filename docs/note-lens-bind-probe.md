# Bind probe — this item's value, not the neighbour's

**STATUS: MEASURED 2026-09-24 on Qwen3.8-9B Q4_K_M. The landed LOCATE head
fails binding; the landed SCORE head does it.** In an order listing several
items with the same fields, asking "what is the quantity of the flasks?" lands
on the right item's value 81% EN / 86% DE with the score head (L19 h=11)
against 47% / 43% with the locate head (L11 h=6), whose most common mistake is
the neighbour's quantity. Free (L19 is already loaded); not exposed as a role.

## 1. The question

Locate answers *where is a quantity*. Binding asks *where is the quantity of
THIS item*, in a document that lists several items with the same fields. The
trap is the right kind of value on the wrong line — the "twins" weakness
RETRIEVE2B found (every error right-family-wrong-variant) and the reason
`/v1/locate` returns three spans by default. Invoices, orders, tables, and
CVs ("when did they work at company X?") all need it.

## 2. The probe

`BINDHEAD=1` in `tests/perf/attn_provenance.cpp`.

* **Corpus, generated.** 8 orders per language x layout, EN and DE, three
  layouts of rising difficulty:
  * LINES — `- 120 insulated flasks at 11.25 EUR each` (4 items);
  * TABLE — `insulated flasks | 120 | 11.25` under a header row (4 items);
  * PROSE — `flasks, bolts and cleats, 120, 300 and 24 units respectively,
    at …` — binding by list position only (3 items).
  Quantities and prices are drawn without repeats inside an order, and every
  value's byte span is recorded at generation, so nothing is searched for.
  The probe refuses a document where two values share a token.
* **Questions.** Quantity and unit price of every item, as question keys in
  one prompt (the endpoint's question instruction), order shuffled: 352
  questions.
* **Readout, per head.** The question's attention mass summed over each value
  span (mean or max over the question's rows). The value with the most mass is
  the answer, classified RIGHT, BINDING ERROR (same field, another item) or
  FIELD ERROR (the other field). *Within-field* = argmax among the asked
  field's values only — pure binding.
* **Chance.** Right ~1/8 (lines, table) or 1/6 (prose); within-field 1/4 or 1/3.
* 8 layers x 16 heads x 2 aggregations = 256 candidates; EN and DE each
  held out for the other.

## 3. Results

| head | right EN / DE | binding error EN / DE | field error EN / DE | within-field EN / DE |
|---|---|---|---|---|
| locate L11 h=6, mean | 46.6 / 43.2 | **37.5 / 38.1** | 15.9 / 18.8 | 51.7 / 50.6 |
| locate L11 h=6, max | 35.2 / 36.4 | 43.8 / 37.5 | 21.0 / 26.1 | 44.3 / 48.3 |
| choice L11 h=3, mean | 47.2 / 43.8 | 22.7 / 24.4 | 30.1 / 31.8 | 60.8 / 63.6 |
| absent L19 h=10, mean | 70.5 / 72.2 | 15.3 / 15.9 | 14.2 / 11.9 | 77.3 / 81.2 |
| **score L19 h=11, max** | **80.7 / 85.8** | **5.7 / 8.5** | 13.6 / 5.7 | **90.3 / 89.8** |
| score L19 h=11, mean | 80.1 / 83.5 | 9.1 / 13.1 | 10.8 / 3.4 | 88.6 / 86.9 |
| L15 h=1, mean (sweep #2) | 81.2 / 83.0 | 8.5 / 10.8 | 10.2 / 6.2 | 85.8 / 88.1 |
| inject L11 h=0, mean | 17.6 / 22.7 | 43.2 / 33.5 | 39.2 / 43.8 | 26.7 / 36.4 |

* **Held out:** select on EN → L15 h=1 mean, 83.0% on DE; select on DE → the
  score head L19 h=11 max, 80.7% on EN. The pooled top ten are all at L15 and
  L19.
* **By layout** (right %, EN / DE, binding errors in brackets):

  | | lines | table | prose |
  |---|---|---|---|
  | score L19 h=11, max | 75.0 / 85.9 (10.9 / 10.9) | **92.2 / 85.9** (0.0 / 4.7) | 72.9 / 85.4 (6.2 / 10.4) |
  | locate L11 h=6, mean | 43.8 / 37.5 (45.3 / 48.4) | 62.5 / 65.6 (31.2 / 26.6) | 29.2 / 20.8 (35.4 / 39.6) |

**Reading.** The locate head finds *a* value of the asked kind and is
indifferent to which item it belongs to — its binding errors outnumber its
field errors 2:1. The score head carries the item: it gets roughly 85% right,
picks the wrong item less than 1 time in 10, and still binds by list position
alone in the "respectively" prose. The pattern again: a head landed for one job
(the ordinal) is the right reader for another, and the incumbent pair is a
poor reader of the new job.

## 4. What this is and is not

* **Actionable today, client side.** For documents that list several items,
  ask with `head: "score"` rather than `head: "locate"`, question keys naming
  the item. Free on a locate-only server.
* **The readout knew the candidates.** Mass was compared across the recorded
  value spans; the route returns token spans instead. A client restricting the
  answer to number-like spans reproduces this; the unrestricted peak was not
  measured.
* **Not a role.** No `bind` head exists; whether one should (L19 h=11 or
  L15 h=1) is an architecture decision.
* **Synthetic.** Generated orders, 3–4 items, one template per layout and
  language; 352 questions. Real invoices (multi-page, merged cells, OCR noise)
  are untested.

## Reproduce

```
BINDHEAD=1 QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf ./build-metal/bin/attn-provenance   # ~5 min
```
