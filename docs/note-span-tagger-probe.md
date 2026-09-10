# Span-tagger probe — can a CPU tagger produce the candidate set? (2026-09-08)

**Verdict: promising for candidates, disqualified for fields, and blocked on a
baseline we never measured.** CPU-only, offline, no engine work. Corpus:
`qdocs_messy_corpus()`, 15 docs (8 EN / 7 DE), 75 labelled fields; every ground
truth value verified byte-exact in its source document.

## The numbers

Exact-match recall (normalised whitespace/case, equality — not containment):

| model | th=0.1 | th=0.5 | median set size | latency/doc | size | licence |
|---|---|---|---|---|---|---|
| GLiNER multi v2.1 | 21/75 (.280) | 17/75 (.227) | 3 → 1 | 100 ms | 1.1 GiB | apache-2.0 |
| **GLiNER bi-base v1.0** | **34/75 (.453)** | 18/75 (.240) | 6 → 1 | 118 ms | 924 MB | apache-2.0 |
| xlm-roberta-base-squad2 (QA) | 19/75 (.253) | 13/75 (.173) | 1 → 0 | 320 ms | 1.0 GiB | cc-by-4.0 |

`urchade/gliner_multi` (v1) is **cc-by-nc-4.0** — disqualified on licence before
evaluation. The licence gate was worth having.

**Latency holds the performance case**: ~100 ms of CPU against a second full GPU
pass over the document.

## The metric was wrong, and it inverts the reading

Exact equality measures *"can this replace the field extractor"*. The candidate
set does not need that. `key_candidates` are **alternative spans a user might
highlight** — a span that contains the value points at the right place, and
bar 3 (§11 of `plan-ocr-lens.md`) established highlights are word-level anyway.

Re-scored as **localisation** (ground truth is a substring of a returned
same-concept span):

| model | exact | boundary (right region, wrong trim) | genuinely absent |
|---|---|---|---|
| GLiNER bi-base v1.0 | 34 | 38 | **3** |
| GLiNER multi v2.1 + label fix | 35 | 39 | **1** |
| xlm-roberta QA | 19 | 31 | 25 |

**Localisation is 96–99% for the taggers**, ~67% for QA. The typical "miss" is
`"45 units"` for `45`, `"0.14 USD"` for `0.14`, `"168.00 USD"` for `168.00` — the
right phrase with units or currency bundled in.

Unchecked: whether the wide spans are *tight*-wide (a word or two) or
uselessly wide (a whole line). The examples are tight; the distribution was not
measured. Check span width before relying on this.

## Label wording dominates everything else

`customer` → **0/15**. `customer company` → **14/15**. Same model, same
documents, one word — **on GLiNER multi v2.1**. Note the attribution: bi-base
scored **12/15** on the bare key, so this sensitivity is model-specific, not a
property of taggers in general. See §"Glosses" below, which weakens this finding
further.

With `customer` alone GLiNER returns the *person* in the email — `Dana Reeve`,
`Frau Kern`, real named individuals — never the ordering company. Changing that
one label moved overall recall .280 → .467 and entity-like recall .541 → .919.
**Bigger than model choice and bigger than threshold choice.**

Label *language* barely mattered: German labels on German documents scored
identically to English labels (9/35 at three thresholds). Semantics, not
language.

**Product consequence.** The key vocabulary is chosen by the *user*. An LLM reads
`customer` in the document's context; a tagger matches it against a label
embedding and fails silently and confidently. A tagger therefore needs a
label-engineering layer between the user's key and the model.

**The format already carries the fix**: `key_vocabulary` accepts `{key, gloss}`,
and the gloss is a natural-language description — exactly what a tagger label
should be. No new request field is needed.

## The numeric wall is real, as predicted

Entity-like {customer, order_number, order_date, delivery_date} vs
numeric/compositional {quantity, unit_price, total}, exact match at th=0.1:

| model | entity-like (n=37) | numeric (n=38) |
|---|---|---|
| GLiNER multi v2.1 | .541 | .026 |
| GLiNER bi-base v1.0 | **.865** | .053 |
| QA | .486 | .026 |

For the best model `unit_price` and `total` are missed on **every single
occurrence**. Dates are never missed. The prediction was written before the run
and it held sharply — but note it is an *exact-match* wall: most numeric misses
are boundary, so the tagger knows where the number is, it just includes the
currency.

## Thresholds do not transfer between checkpoints

GLiNER bi-base never clears 0.7 on any of the 75 fields; GLiNER multi v2.1 does.
Two models, same family, incomparable score scales. This is the
`coverage_used_peak = 0.705` failure mode in miniature — a frozen absolute
threshold is not portable, and any adoption must carry a per-checkpoint
calibration, not a constant.

## The baseline — pass 2, scored by the same code

Measured 2026-09-08 by dumping the shipped finder's spans through the **identical
`analyze.py`** used for the taggers (no reimplemented metric).

| | exact | boundary | **absent** | entity | numeric | latency/doc |
|---|---|---|---|---|---|---|
| **pass 2 (LLM, shipped)** | 34/75 (.453) | 41 | **0** | .865 | .053 | **6301 ms** |
| GLiNER bi-base v1.0 | 34/75 (.453) | 38 | 3 | .865 | .053 | 118 ms |
| GLiNER multi v2.1 + label fix | 35/75 (.467) | 39 | 1 | .919 | — | 100 ms |

Localisation: pass 2 **75/75 (100%)**, bi-base 72/75 (96%), multi+fix 74/75 (98.7%).

**Read the first two rows again.** A 9B LLM doing a full GPU pass and a 924 MB
CPU tagger score **identically** on exact match (34/75), identically on
entity-like fields (.865), and identically on the numeric wall (.053), with raw
un-engineered labels in both cases. The tagger is **53× faster**.

That the two agree to three decimal places on three separate splits says the
~34/75 ceiling is **the ground truth's trim convention**, not model capability —
both find `"45 units"` where the label says `45`. The exact-match metric is
measuring our own annotation style.

**Where pass 2 is genuinely better: zero absent.** It always finds the region;
bi-base misses 3 of 75, multi+fix misses 1. That is the real quality difference,
and it is 1–4% of fields losing an *alternative* span, not a field.

**One asymmetry to keep honest**: GLiNER multi's .467/.919 required changing the
label `customer` → `customer company`. Pass 2 earned .453/.865 on the **raw key
names**. On unengineered labels GLiNER multi scores .280 — far worse than pass 2.
Only **bi-base** matches pass 2 without label engineering, so bi-base is the fair
comparison and the only real candidate.

## Set size — the split that decides it, and it is not a tie

Recall was compared at each model's most permissive threshold. That hid the
cost. Candidate-set size and recall move together, and they must be read
together:

| | recall | median set size |
|---|---|---|
| pass 2 (LLM) | **34/75** | **1** |
| bi-base @ th=0.1 | 34/75 | **6** |
| bi-base @ th=0.5 | **18/75** | 1 |

**At matched recall the tagger's candidate set is 6× looser. At matched set size
its recall halves.** Pass 2 is answering; the tagger is type-matching.

This is not a soft preference — **gate 2 is a shipped gate and requires a median
candidate-set size of 1.** bi-base fails it at the only operating point where its
recall is competitive. A candidate set of six spans per key makes every
unambiguous field look contested, which is the noise gate 2 exists to prevent.

## Glosses as labels — hypothesis refuted, 2026-09-08

The open question was whether a precise label returns *fewer, better* spans, so a
tagger could reach pass 2's recall at pass 2's set size. Tested with the
**authentic production glosses** (`server_extract_smoke.sh` `GLOSS`), verbatim,
no hand-tuning. Recall and median set size read on the same row.

| | best recall **at median set size 1** |
|---|---|
| **pass 2 (LLM, shipped)** | **34/75** |
| bi-base, bare keys | 18/75 (th=0.5) |
| bi-base, glosses | 16/75 (th=0.4) |
| multi v2.1, glosses | 17/75 (th=0.3) |

**No operating point comes close.** Glosses did not help — at the gate-2
compliant point bi-base got slightly *worse* (18 → 16).

`customer` is the sharpest case, and it went the wrong way: bi-base 12/15 bare →
**10/15** glossed, with named individuals (`Pete Hayes`, `Markus Wald`) still
returned in both conditions. The authentic gloss — "the buyer — a company,
person, or their email/domain" — **explicitly licenses person names**. A real
caller's gloss is written for a human reader, not as a discriminative label, and
there is no reason to expect the two to coincide.

`unit_price` and `total` sit at **0 recall at every threshold for both models**,
entirely boundary — a tagger never lifts a bare number without its currency.

## Verdict — CLOSED. Pass 2 stays.

A tagger cannot serve the candidate set. The mechanism is not a tuning problem:
**a tagger scores type-membership, pass 2 answers a question.** Loose enough to
find the value, it finds five other things of the same type; tight enough to
return one span, it drops true positives as fast as false ones. There is no
threshold where it is both selective and complete, and the label is not the lever
— the authentic gloss made it worse.

What was real and is now spent: 53× cheaper, spans-by-construction, and deleting
the second GPU pass. Not worth halving candidate recall or failing a shipped gate.

**Not revived by this**: conflict *detection* (two spans, same key, different
values) was never measured, and the set-size behaviour above makes it look worse
than when it was proposed, not better. Treat it as unsupported speculation unless
someone measures it.

**What this does NOT close**: the exact-match ceiling of ~34/75 that pass 2 and
the taggers hit *identically* is our own annotation trim convention
(`"45 units"` vs `45`). That is a corpus-labelling question worth its own look,
and it is not about taggers.

## Live thread — the typed span index (idea, not a plan)

The rejection above is about **selection**: picking the one span that is the
value. It does not transfer to **coverage**: finding every span of a type. The
numbers that killed the first use support the second — 96–99% localisation,
high recall, loose precision is a bad selector and a good index.

**What it is.** Not `key_candidates`. A separate member answering "what in this
document looks like a date / a company / a price", built by the tagger, never
claiming any of them is the value. At six spans an index is *correct*; a
candidate set at six spans is a false statement about ambiguity. Same output,
different claim, and the claim is what was wrong before.

**Why it is worth anything — the pairing with the omission report:**

    lens   delivery_date  null  absent      ← the model stated no delivery date
    index  date-like:  "KW 44" @ 118–123    ← but something date-shaped is here

`absent` has never meant "not in the document" — that disclaimer is written into
`lens-format.md` and into the OCR client prompt. This closes it. Two readings,
both useful:

- **index hits** → the omission is worth checking, and the reviewer gets the span.
- **index empty** → the absence is corroborated by a **second mechanism that never
  read the model's output**. Every claim the lens makes today derives from one
  model's behaviour; this is the first corroboration, which is a different
  epistemic object from another measurement of the same thing.

**The gate, and it is not optional.** "Index empty ⇒ absence corroborated" is only
as good as the tagger's recall *for that type*, and our data already predicts a
split: `delivery_date` 9/9 and `order_date` 6/6 — never missed at any threshold —
against `unit_price` and `total` at **0 exact at every threshold**. Shipping the
claim flat would be a lie on half the vocabulary.

So it needs a **per-type recall table, and only the types that clear a bar may
carry the corroboration claim** — the same discipline as `kLensCalibrations`,
where a row exists because a probe measured it. Cheap: same corpus, same scripts,
CPU, no engine work.

**Unresolved**: placement (client / sidecar / in-engine encoder — we already run
three bidirectional vision encoders), threshold calibration per checkpoint, and
whether this earns a new report member at all. Recorded as the live thread out of
this probe; not scheduled.
