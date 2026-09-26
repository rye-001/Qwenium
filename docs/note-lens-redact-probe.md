# Redact probe — can one prefill mark ALL the personal data?

**STATUS: MEASURED 2026-09-24 on Qwen3.8-9B Q4_K_M. Half a mode.** Asked field
by field (name / e-mail / phone / address), the heads already landed touch
**every** personal-data item in the document (100% EN / 100% DE) — but only
about 40% of the tokens they mark are the data itself, so the edges are loose.
Asked open-set ("which parts are personal data?") it fails. One head (L23 h=8)
reads the redaction task from the prompt's closing rows, stably but
moderately, and costs depth. A recipe, not a new capability.

## 1. The question

Locate answers one question with one answer. Redaction is **open-set**: every
name, e-mail address, phone number and personal address, however many there
are. Buyer jobs: GDPR redaction before a document leaves the building, and
anonymised CV screening ("anonyme Bewerbung"). The inject result suggested the
prompt's closing rows follow the *task*; if a "redact personal data"
instruction made them light up the personal data, one ask would highlight
whatever a job needs.

## 2. The probe

`REDACTHEAD=1` in `tests/perf/attn_provenance.cpp`.

* **Documents.** Leg C's 15 order e-mails and the two fictional INJHARD CVs:
  9 EN, 8 DE.
* **Ground truth, marked by hand, policy fixed before the run.** Personal data =
  person names, **all** e-mail addresses (role mailboxes like `orders@` too —
  redaction policies strip them), phone numbers, a person's city or address.
  Not personal = company names, company registration / VAT numbers, order
  numbers, product data. On average ~9% of a document's tokens are personal.
* **Three readouts, per head, per document token:**
  * **R1 per-field recipe** — four question keys (name, e-mail address, phone
    number, postal address or city); a token's score is the max over the four
    keys' mean-row mass. What a client can do today.
  * **R2 one open question** — "Which parts of this document are personal
    data?".
  * **R3 redaction tail** — the instruction asks for the document rewritten
    with personal data replaced by `[REDACTED]`; score = mean of the
    template-tail rows, the readout that found injections.
* **Metrics.** Token AUC (personal vs not, threshold-free); **R-precision** —
  the share of the top-R tokens that are personal, R = the number of personal
  tokens (chance ≈ 9%); **span recall@2R** — the share of marked items with at
  least one token in the top 2R (generous by design: an item counts as found
  if any of its tokens is marked). 8 layers × 16 heads; EN and DE each held out
  for the other.

## 3. Results (EN / DE)

**R1 — per-field recipe:**

| head | AUC | R-precision | span recall@2R |
|---|---|---|---|
| **score L19 h=11** | 0.824 / 0.792 | **40.2 / 38.8** | **100 / 100** |
| locate L11 h=6 | 0.747 / 0.774 | 37.2 / 35.9 | 100 / 100 |
| absent L19 h=10 | 0.695 / 0.681 | 29.6 / 25.0 | 100 / 100 |
| choice L11 h=3 | 0.645 / 0.696 | 23.8 / 27.3 | 88.9 / 96.9 |
| best sweep, L15 h=11 | 0.813 / 0.851 | 40.2 / 44.9 | 100 / 100 |

Held out: select EN → L19 h=2, 34.3% on DE; select DE → L15 h=11, 40.2% on EN.

**R2 — one open question:** landed heads 3–13% R-precision (chance ≈ 9%); the
best sweep head (L15 h=3) 25.8 / 22.9, and the held-out pick drops to 17.2%.
**Fails.**

**R3 — redaction-instruction tail:**

| head | AUC | R-precision | span recall@2R |
|---|---|---|---|
| **L23 h=8** | 0.824 / 0.826 | **31.9 / 36.6** | 100 / 90.6 |
| score L19 h=11 | 0.756 / 0.714 | 20.7 / 8.3 | 92.2 / 78.1 |
| inject L11 h=0 | 0.640 / 0.702 | 9.6 / 15.8 | 83.3 / 81.2 |

Held out: **both directions select L23 h=8** (36.6% on DE, 31.9% on EN) — the
one stable open-set signal found, but moderate, and L23 is 24 blocks against
the locate-only server's 20.

## 4. What this is and is not

* **Usable as a recipe.** Ask the four field questions on the score (or
  locate) head, take the top spans per key, and expand each to its whole
  item — name, address, line — as for locate. Every item was reached; the
  loose edges are the separator-peak habit locate already has.
* **Not open-set.** "Mark all personal data" as one ask does not work on the
  landed heads, and only one deeper head reads the redaction task from the
  closing rows.
* **Regex covers part of it exactly.** E-mail addresses and phone numbers have
  patterns; attention's value is names and addresses — not measured
  separately here.
* **Small and self-marked.** 17 documents, one marking policy; a stricter
  policy (role mailboxes not personal) would move the numbers. The span-recall
  metric is generous; R-precision is the strict one.

## Reproduce

```
REDACTHEAD=1 QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf ./build-metal/bin/attn-provenance   # ~3 min
```
