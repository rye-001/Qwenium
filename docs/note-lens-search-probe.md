# Search probe — which chunk answers the question (and: none of them)

**STATUS: MEASURED 2026-09-24 on Qwen3.8-9B Q4_K_M. A useful prefill-only
mode, built from heads already landed — no new head, no engine change.** With
16 chunks in one prompt the absent head (L19 h=10) picks the chunk that answers
a paraphrased question 100% EN / 96.9% DE (chance 6.2%), and separates "the
answer is here" from "it is not" at AUC 0.994 / 0.979.

## 1. The question

ICR (Chen et al., ICLR 2025, arXiv 2410.02642): attention from a query to
candidate passages ranks them without generating. Locate already finds a LINE
inside one document; this asks whether the same machinery finds the right
CHUNK among many, and whether it can say "none of them".

The bar is usefulness, not a contest: well above chance, held out across
languages, no position artifact.

## 2. The probe

`SEARCHHEAD=1` in `tests/perf/attn_provenance.cpp`.

* **Corpus.** The 55 documents we have — DECIDE's 40 routing notes and Leg C's
  15 order emails — EN (28) and DE (27) kept apart. One **hand-written,
  paraphrased** question per document, avoiding the document's distinctive
  words: only 14.7% (EN) / 11.6% (DE) of a question's words appear in its
  answer. The order emails are the hard part: one topic, differing only by
  customer and product, two of them near-twins of routing notes.
* **Setup A — in one prompt.** 16 chunks (`--- Message k ---` headers) form one
  document, the question is the key, one prefill; a chunk's score is the head's
  attention summed over its tokens. Four shuffled sets per language, every
  pool question asked of every set: 64 positive trials per language, and the
  questions whose document is NOT in the set are the "none" cases.
* **Setup B — one prompt per chunk.** Every chunk prefilled alone with every
  question (1,513 prefills); scores compared ACROSS prefills. What a
  collection bigger than one prompt needs.
* **Signals.** Mass, max-over-query mass, peak, density, and two content-free
  calibrations from an `N/A` key placed AFTER the question (causally invisible
  to it, so the raw readout is untouched): minus, and divided by, the N/A
  key's mass. 8 layers x 16 heads x 6 signals = 768 candidates.
* **Keyword reference.** BM25 on the same chunks and questions, umlauts folded,
  plain and 5-letter-prefix analyzers, question-frame stoplist on the query.
  A reference point for "word matching", not an opponent (§5).

## 3. Results — setup A (16 chunks, one prompt)

| | EN top-1 | DE top-1 | "none" AUC EN / DE |
|---|---|---|---|
| **absent L19 h=10, mass** | **100%** | **96.9%** | **0.994 / 0.979** |
| score L19 h=11, peak | 100% | 98.4% | 0.992 / 0.949 |
| locate L11 h=6, mass | 96.9% | 92.2% | 0.986 / 0.926 |
| choice L11 h=3, maxmass | 100% | 85.9% | 0.958 / 0.902 |
| inject L11 h=0, density | 81.2% | 76.6% | 0.643 / 0.717 |
| BM25, prefix-5 | 51.6% | 31.2% | 0.612 / 0.542 |
| chance | 6.2% | 6.2% | 0.5 |

* **Held out:** select on EN → L11 h=2, 93.8% on DE; select on DE → L15 h=6
  `maxmass`, 98.4% on EN.
* **Position:** 97.9% / 100% / 100% with the answer in the first / middle /
  last third of the prompt.
* **Order emails** (the near-twins): best candidate 100% EN / 94.4% DE; locate
  94.7% / 83.3%.
* **Against word matching, same trials:** head right where BM25 was wrong 31
  (EN) and 43 (DE) times; BM25 right where the head was wrong 0 times.

## 4. Results — setup B (one prompt per chunk, compared across prompts)

| | EN top-1 | DE top-1 | "none" AUC EN / DE |
|---|---|---|---|
| **score L19 h=11, minus N/A** | **100%** | **88.9%** | **0.964 / 0.873** |
| score L19 h=11, mass | 96.4% | 85.2% | 0.948 / 0.853 |
| absent L19 h=10, mass | 92.9% | 85.2% | 0.860 / 0.836 |
| locate L11 h=6, mass | 82.1% | 66.7% | 0.800 / 0.752 |
| BM25, prefix-5 | 42.9% | 25.9% | 0.692 / 0.583 |
| chance | 3.6% | 3.7% | 0.5 |

* **Held out:** select on EN → the score head minus N/A, 88.9% on DE; select
  on DE → L23 h=14, 82.1% on EN.
* **Content-free calibration helps here, a little** (+3.6 points top-1, +0.02
  "none" AUC on the score head). This is ICR's actual use — comparing across
  documents — and the opposite of the within-document result, where it was a
  no-go for every shipped head (note-lens-laya-cross-check.md §9).

## 5. What this is and is not

* **A recipe, not a head.** The best heads sit at L19, the semantic-matching
  cluster the locate-only server already loads. Setup A runs today on
  `/v1/locate`: send the chunks as the document with the question and
  `head: "absent"`, `top_k: 64`, and sum hit mass per chunk. Setup B adds an
  `N/A` question LAST and reads `head: "score"` mass minus N/A mass per chunk.
  The recipe is in lens-format.md.
* **Not a contest.** BM25 is a word-matching reference; paraphrased questions
  are by construction its worst case. A meaning-based retriever was not run
  and is not the bar for this phase.
* **The corpus is easy.** Many of 768 candidates reach 100% on ~28 questions
  per language, so it cannot rank heads against each other — it shows the
  capability, not the best coordinate. Harder: longer chunks, more
  near-duplicates, keyword-style questions mixed in.
* **Self-authored questions**, one per document, short chunks (1–2 sentences
  for routing notes).
* **Setup B costs one prefill per chunk per question.** Prefill-once (compute a
  chunk's KV once, run many questions against it — the warm-document path)
  is what makes it affordable; not built into this probe.

## Reproduce

```
SEARCHHEAD=1 QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf ./build-metal/bin/attn-provenance   # ~25 min
```
