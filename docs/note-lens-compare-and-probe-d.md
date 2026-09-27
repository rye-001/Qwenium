# Compare landed, probe D closed: attention finds, it does not reason

**STATUS: 2026-09-26. Qwen3.8-9B Q4_K_M.** Part 1 is step 7: AbsenceBench and
the seventh mode, `POST /v1/compare`, which is now served. Part 2 is step 8,
probe D: four readouts from inside the document itself (coreference, split,
surprise, wrong totals). **All four failed the bars set before each run.**
Every probe is a leg of `tests/perf/attn_provenance.cpp`.

## Part 1 — compare (step 7)

### AbsenceBench (ABSBENCH)

AbsenceBench (harveyfin/AbsenceBench, CC-BY-SA 4.0) is a published test of
"what is missing". The model gets an original and a copy with lines removed,
and is scored by micro-F1 over the removed lines. The three validation files
live in `temp/absencebench/`; `py/absencebench_export.py` flattens them to
JSONL (seed 20260926, rows up to 30,000 characters). The head was picked on 20
dev rows per domain and scored on 100 eval rows.

| head | poetry | numbers | code diffs |
|---|---|---|---|
| landed absent head L19 h10 (pre-registered) | 40.3 | 21.6 | 5.3 |
| **compare head L15 h1**, max over the copy's rows | **76.5** | **78.6** | **8.3** |
| GPT-4.1 (published) | 54.3 | 57.5 | — |
| Claude-3.7-Sonnet-thinking (published) | 72.7 | 96.0 | — |

The copies are word-for-word, so a string diff solves AbsenceBench outright.
Its value here is as an external check on the head. The product is the
reworded case, where diff is blind.

> **2026-09-27: the baseline changed.** The median baseline below failed when
> most of an original was missing (51–65% of a summary's drops caught). Compare
> now divides by the mean of the best-covered quarter of units, threshold 0.35
> (COMPARE3, confirmed on fresh trials; AbsenceBench numbers 83.2, poetry 76.5).
> See plan-lens-compare.md. The numbers in this note are the median-baseline ones.

### COMPARE2 — the served shape

The served shape is one generic instruction, the caller's units one per line,
coverage relative to the median unit, and a threshold. On translations (EN/DE
notes of 8 or more sentences, 0–2 sentences dropped), the threshold was chosen
on one direction and scored on the other:

| head | EN→DE flagged / complete copies falsely flagged | DE→EN |
|---|---|---|
| **L15 h1** | **91.7% / 6.2%** | **97.9% / 0.0%** |
| L15 h13 | 97.9% / 31.2% | 89.6% / 0.0% |
| absent L19 h10 | 66.7% / 6.2% | 66.7% / 12.5% |

AbsenceBench through the generic prompt: poetry 78.0, numbers 82.9. Both bars
passed. The threshold chosen independently on translation (0.47 / 0.52) and
on AbsenceBench (0.43 / 0.50) lands in the same place, so 0.50 belongs to the
head, not to one dataset.

### `/v1/compare` — gates (COMPAREGATE, LENSDUMP)

| gate | result |
|---|---|
| G1 other features unchanged | 95 reports (locate, extract, verify, verdict) byte-identical; suite 1034/1034, HTTP tests 18/18 |
| G2 shipped driver at 0.50 | EN→DE 91.7% / 6.2%, DE→EN 97.9% / 0.0%; AbsenceBench numbers 82.9, poetry 77.0 |
| G3 flash on the original's pass | **failed**: 6 of 31,807 flags changed ⇒ the pass stays materialized (compare does not follow the row's split+flash licence) |
| G4 kept original, several revisions | 96/96 identical to cold |
| G5 live | full, verify-only and locate-only servers give the same answer. A ferry notice with 2 of 8 sentences dropped in German gets exactly those 2 flagged (0.10 / 0.17 vs 0.89–1.21). Warm resume 0.64 → 0.30 s. Q8_0 refuses (400). No document text in any log. |

Envelope: at least 8 units, at most 9,762 tokens. Not for repetitive text
(code diffs 8.3). Wire format: `docs/lens-format.md` → Compare. Plan:
`docs/plan-lens-compare.md` (LANDED).

## Part 2 — probe D (step 8)

The shared design: nothing selected is scored on the data it was selected on.
Heads were picked on one language and scored on the other, and every bar was
written before its run.

### COREF — who is "she", "it", "the Supplier"? Bar FAIL; aliases work

112 hand-written items, EN and DE, in three arms:
- **easy:** different gender;
- **hard:** Winograd-style pairs, where one phrase flips the answer and the
  candidates keep their positions;
- **alias:** contract defined terms, with the roles swapped between variants.

Nearest-mention and first-mention both score 50% by construction.

The readout is the reference's own row (ROW), or the mean over the rows to the
end of its sentence (CLAUSE); the score is the max over the candidate's tokens.

| arm | result |
|---|---|
| easy | 100% at L3 h0 / L7 h1: trivial |
| **hard** | **no signal.** ROW is exactly 50.0 on every head: the pair is identical up to the pronoun, so its row is identical. The cue comes *after* the pronoun. CLAUSE's best was 70.8% in-sample (the max of 256 configurations on 24 items, i.e. noise); held out 50.0. |
| **alias** | **works.** The landed compare head L15 h1 (not selected on this data) scores EN 100 / DE 95.8; score L19 h11 91.7 / 100; locate L11 h6 100 / 83.3 |

**Untested caveat:** the template puts each name beside its alias, so "attend
near the earlier copy of the word" would solve it. The harder form
(definitions not adjacent, aliases used much later) was not run.

### SPLIT — where one document ends and the next begins. Bar FAIL

160 streams of short documents, one sentence per line, joined by a single
newline. The readout is the next line's first rows and their attention mass
onto the text before the break. Held out:

| arm | held-out AUC | bar |
|---|---|---|
| same-topic notes | 0.61 / 0.57 | 0.85 |
| tickets back to back | 0.76 / 0.80 | 0.85 |

In-sample the best was 0.95 (EN tickets); English is clearly stronger than
German. On e-mails a greeting/sign-off rule reaches 0.98 / 0.91 without any
model. TextTiling (word overlap) is at chance on short texts (0.42–0.55).

### SURPRISE, SURPRISE2 — a highlighter from the prefill's own logits. Bar FAIL

This is not attention. It runs at full depth with logits at every position
(`set_slice_prefill_head(false)`, no engine change). A word's score is the max
over its tokens. There are 212 damaged documents, each with a clean twin, and
no parameters are selected.

- **SURPRISE** scores raw surprisal.
- **SURPRISE2** scores surprisal minus the entropy of the prediction:
  "surprised where the model was sure".

The bar was fact and real-word errors reaching AUC ≥ 0.90 and top 3 ≥ 70%.

| arm (AUC / damaged word in top 3) | SURPRISE EN | SURPRISE2 EN | SURPRISE DE | SURPRISE2 DE |
|---|---|---|---|---|
| OCR damage | .944 / 95% | .954 / 95% | .873 / 65% | .906 / 75% |
| typos | .961 / 90% | .954 / 85% | .877 / 75% | .903 / 75% |
| real-word errors (form→from, dass→das) | .922 / 75% | .923 / 70% | .823 / 65% | .848 / 60% |
| wrong facts ("Vienna, the capital of Sweden") | .761 / 10% | **.844 / 55%** | .666 / 10% | **.774 / 55%** |
| wrong total (easy: 240 × 12.50) | .728 / 0% | **.925 / 83%** | .823 / 0% | **.971 / 83%** |

- **The model notices:** a damaged word beats its clean twin in 95–100% of
  facts and totals ("Sweden" scores 9.6 nats vs 0.14 for "Austria").
- **Raw surprise confuses novelty with error.** The first word of a note
  ("Reminder:", "NDA") outranks the error. The entropy contrast removes most
  of that noise; sentence-initial words and rare domain words remain.
- **Only confident knowledge shows.** Facts the model is unsure of stay
  invisible ("the Black Sea near Rotterdam", "freezes at 40 degrees").
- **Data flaw:** the German notes corpus is written without umlauts ("fuer",
  "ueber"), and those words themselves rank high. The German numbers are
  understated.

### TOTALCHECK — does this invoice's arithmetic look wrong? Bar FAIL

The SURPRISE2 score restricted to numbers. 120 generated invoices, 2–4 lines
with cents. Exactly one number is changed (a line, the subtotal, the VAT or
the total), by a large change, a digit swap or new cents; each has a clean
twin. The threshold was chosen on one language and scored on the other.

| | EN | DE | bar |
|---|---|---|---|
| detection AUC | 0.829 | 0.886 | — |
| flagged at ≤ 10% false alarms (held out) | 32% | 50% | 80% |
| points at the changed number, top 1 / top 2 | 58% / 80% | 78% / 90% | 70% top 1 |

**Cause:** clean invoices already carry numbers scoring 5–7.7. The 9B is
"sure and wrong" about correct multi-digit products (237 × 184.37). The easy
totals in SURPRISE worked; realistic ones exceed the model's mental
arithmetic. Arithmetic checking belongs in code: find the numbers with
locate/extract and recompute them.

## What it means

* **Attention finds; it does not reason.** Compare, locate, choice and aliases
  are finding jobs, and all work. Who-did-what coreference, document
  structure and arithmetic are reasoning jobs, and none work. This is the old
  law again: attention marks what the model considered, not what it
  concluded.
* **Compare is the result that ships.** It is the one mode where a string diff
  is blind (translations, rewrites) and the numbers hold held out.
* **Surprise-where-sure is real but not yet a mode.** It catches English
  spelling damage and confidently known facts. A pass would need a new run on
  fresh items (real umlauts, new facts, sentence-initial tokens excluded),
  because that change was made after seeing this data.
* **Don't re-propose:** attention-based split; the pronoun's own row for
  Winograd-style coreference; surprisal as an arithmetic checker.
* **Open, small:** a harder alias arm (non-adjacent definitions, late use). If
  it holds, "which party is 'the Supplier'" is free on the compare head.

## Reproduce

```
M=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf
ABSBENCH=1    QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance   # needs temp/absencebench/absencebench.jsonl
COMPARE2=1    QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance
COMPAREGATE=1 QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance
COREF=1       QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance   # COREF_VERBOSE=1 lists misses
SPLIT=1       QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance
SURPRISE=1    QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance   # prints SURPRISE and SURPRISE2
TOTALCHECK=1  QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance
```
