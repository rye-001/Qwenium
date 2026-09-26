# Laya cross-check — an audit of OUR corpora, and what it taught us about `score`

**STATUS: MEASURED 2026-09-21; §9 (content-free calibration) added 2026-09-24.
No constant moved. No calibration changed.**
This note records a cross-check of the shipped lens heads against an
independently trained open-weights model, and the diagnosis of a `score`
failure that turned out to be a defect in the *test*, not in the head.

Read §3 first if you are here about `score`. Read §5 if you are here about the
corpora.

## 1. What Laya is, and why it is the right instrument

`convaiinnovations/laya` (Apache 2.0, HuggingFace) is a non-autoregressive
**encoder** decision model — ModernBERT-large 421M or mmBERT-base 322M, plus a
decision head trained from scratch — built to answer TypeSafe Jev's three typed
primitives: `choice`, `score`, `noul`. Those are three of our four lens jobs.
Trained with RLCD: RL against strictly proper scoring rules, so reporting
honest probabilities maximises reward.

It is the only independent instrument we have for a question we cannot answer
about ourselves: **are the corpora our head numbers were measured on too easy?**
Every document in `decide_choice_corpus()`, `score_corpus_extended()` and
`qdocs_messy_corpus()` was written by the same author as the probe that scores
them. SCOREHEAD already forced a corpus-origin held-out axis for exactly this
reason.

**The test is asymmetric and that is the point.** Laya's own model card reports
its base checkpoints at 0.362 on typed-decisions against a 0.461 majority-class
baseline, and says plainly: *a fast base to specialise, not a zero-shot
decision engine*. So a Laya **loss proves nothing** and must never be reported
as "we beat Laya". Only a Laya **win or near-tie** is informative, and it is
informative *against us*.

| | |
|---|---|
| Checkpoint | `laya/typed-decisions` (the strong arm, 0.766 on their benchmark) |
| German | sent to `laya/multilingual` **explicitly** — their `Router` dispatches on *script*, so Latin-script German would go to the English checkpoint |
| Budgets | `max_len` 1024, `head_max_len` 256–320; nothing truncated |
| Our side | `Qwen3.8-9B-Q4_K_M`, `--attention-lens --lens-locate-only`, 20/33 blocks |
| Driver | `py/laya_cross_check.py`, corpora extracted to `py/corpora/lens_corpora.json` |

## 2. Corpus audit — the result that went against us

Our 40-document routing corpus, four options, same documents both sides:

| | EN | DE | pooled |
|---|---|---|---|
| **Laya** (421M) | **95.0%** | 70.0% | 82.5% |
| 27B `L39 h=7` | 95.0% | 100.0% | 97.5% |
| 9B `L11 h=3` | — | — | 92.5% |

**On the English half a 421M encoder ties our 27B exactly.** A 4-way routing
task over short single-topic business documents does not separate a trained
baseline from an attention head.

We can no longer claim the choice head is *good* on the basis of this corpus.
We can claim it is *not worse than a trained baseline*. Any future choice claim
needs harder documents.

One mitigation: their `typed-decisions` fine-tune covers invoice processing and
customer service — our corpus's exact domain — so "out-of-domain" overstates
their handicap. It does not rescue the corpus.

### Lures — RESOLVED 2026-09-21, and it reverses

The first pass reported Laya at **62.5%** on lures against our **43.8%** and
read it as "lures are our mechanism's weakness". **That was not a valid
comparison** — 43.8% is the *incumbent locate pair* read through
`py/lens_decide1.py`'s own recipe (bare-string keys, `peak`, `max`), so it
varied the head AND the recipe at once. Concluding a capability limit from a
head calibrated for another job is trap #1 in this repo.

`py/lens_lure.py` runs both heads through **one identical harness** — question
vocabulary, `mean`, mass summed over the body — so only `head` changes. The
locate arm is `uncalibrated` by construction and is a control, not a claim.

| corpus | head | all | clean | **lure** | EN | DE |
|---|---|---|---|---|---|---|
| `choice_py` | **choice L11 h=3** | 92.5% | 100.0% | **81.2%** | 95.0 | 90.0 |
| `choice_py` | locate L11 h=6 | 87.5% | 100.0% | 68.8% | 85.0 | 90.0 |
| `choice_cpp` | **choice L11 h=3** | 92.5% | 100.0% | **81.2%** | 95.0 | 90.0 |
| `choice_cpp` | locate L11 h=6 | 82.5% | 95.8% | 62.5% | 85.0 | 80.0 |
| `choice_py` | **choice, 4 rotations** | **97.5%** | 100.0% | **93.8%** | 100.0 | 95.0 |

**Harness validated against the shipped constant**: on `choice_cpp` the choice
arm reproduces the landed provenance exactly — 92.5%, EN 95.0, DE 90.0.

Three results:

* **The head is worth +12.5 to +18.7 points on lures alone** (81.2% vs 68.8% /
  62.5%, same documents, same recipe). The rest of the gap from 43.8% is the
  recipe, not the head.
* **Against Laya on the same documents we lead on lures 81.2% to 62.5%**, and
  on clean documents 100.0% to 91.7%. The first pass had this backwards.
* **Rotating the option order is worth +5 points** (92.5% → 97.5%, lures
  81.2% → 93.8%, DE 90 → 95) at the cost of 4 prefills instead of 1. Position
  bias is real here, as LOCABSENT found it elsewhere. Not landed — it is a
  caller-side choice, and the shipped 92.5% is the single-prefill number.

This also **refines §2**: the corpus is not uniformly easy. Its *clean* half is
saturated for both systems, but its *lure* half separates them by 18.7 points.
Future choice corpora should be mostly lures.

## 3. `score` — the head was right, the test was wrong

The first CV run reported `score` as unshippable. That verdict was **withdrawn**.
Three errors, all mine:

1. **Rounded a fraction the format doc says not to round.** 1.744 rounds to
   `working`, two rungs below the label — which is precisely what the
   "rounding reads low" bullet predicts.
2. **Ran an invalid control.** The score route has no instruction field, only
   keys, so conditioning on a subject means pasting it into all five rungs —
   where it is a common factor that divides out on normalisation. Measured
   ratio across rungs: 1.15–1.43, near-constant. The subject never had a chance
   to move the answer. (The same control *is* valid for Laya, which has a
   separate `instructions` field — see §4.)
3. **Graded against a generous self-authored label.**

### The corpus, and the control that makes it trustworthy

A **monotone ablation of one real CV**: five rungs, each deleting seniority
evidence, with education and skills held constant. Ground truth by
construction; the text is real, not invented. Then the control — a second set
where the *junior* documents are the **longest**, because in the first set
length rose with level and anything scaling with length would score 1.000.

```
                        L0     L1     L2     L3     L4   concordance
lengths rise         0.971  1.394  1.714  1.926  2.210     1.000  MONOTONE
lengths INVERTED     1.058  1.366  1.655  1.900  2.318     1.000  MONOTONE
chars, inverted set   1154    941    793    742    743
```

**Concordance 1.000 both ways.** The head reads the axis, not the word count.
Four ladder phrasings were tried and three scored 1.000, including one built
from single role nouns (`student / intern / engineer / senior engineer /
architect`). The wording was never the problem.

### What the run produced that is new

* **Length bias, 5.7x.** The same content at 2 / 5 / 9 / 14 / 25 words reads
  0.879 / 0.813 / 0.465 / 0.279 / 0.154. Harmless when rungs are balanced;
  a ladder with uneven rungs is biased toward its short ones. This is the
  `mean` form of the filler penalty DECIDE1 found under `max`.
* **Anchors are not optional.** Against reference documents, the real CV's
  1.744 reads *between working (1.714) and senior (1.926)* — legible and
  defensible. Alone it reads as nothing.
* **Removing the negation made it worse.** Rewriting the bottom rung to avoid
  "no …", on the theory that attention has no NOT, produced the *only* ladder
  below 1.000 (0.900). The instinct was wrong here.

All three are now written into `docs/lens-format.md` → *Writing a ladder*.

## 4. Laya, judged fairly

Laya was given the same better test after our head got one.

| test | ours | Laya |
|---|---|---|
| ordinal, 5-rung ablation, both orientations | 1.000 MONOTONE | **1.000 MONOTONE** |
| choice, this CV | `backend` ✓, margin +0.094 | `backend` ✓, margin +0.031 |
| absence, this CV | `publications` **rank 1 of 8**, 2x gap | rank 5 of 8, **inverted** |
| locate | 7 of 11 keys usable | no equivalent |

So **"Laya's score primitive is inert" was too broad** and is corrected here.
It is inert *on the subject axis* — its `instructions` field genuinely did not
move the answer, including for `underwater basket weaving`, which scored above
`backend` (spread 0.045 across seven subjects, three of them nonsense). On a
real ordinal comparison it ranks perfectly, as we do.

**Absence is the one place a gap opened.** `publications` is the only key
actually present in the CV — three named papers — and we rank it first with a
2x gap while Laya ranks it fifth, below four keys that are entirely absent.
Ranking by Laya's output would report a portfolio and a security clearance that
do not exist. That is a document type neither head was calibrated for, and
`noul` is Laya's *best* primitive on its own card.

Per §1 this is the direction that proves nothing on its own, and there is a
framing confound: their `noul` is trained on semantic yes/no about content, not
field-presence over a messy email. Recorded, not claimed.

**Speed, corrected.** On this Mac: ours 1.2–2.2 s per call, Laya 0.5–1.7 s.
Their 33 ms is a T4 GPU figure. Against a 421M model on Metal we are within
~2x, not the 20x the two model cards imply.

## 5. Two corpus defects found in passing

1. **The German half is ASCII-transliterated.** `ueber`, `faellig`,
   `Kuendigung` — **zero real umlauts in 64 DE documents** across
   `decide_choice_corpus()` and `score_corpus_extended()`. The messy corpus
   uses *real* German (6 of 7 DE docs), and `attn_provenance.cpp` contains
   umlauts elsewhere, so this was a choice in those two corpora, not a file
   encoding limit. It is a confound inside numbers already written into shipped
   provenance strings, including the standing claim that German weakness is a
   *model* problem.
   **It does not explain Laya's German drop** — Laya falls just as hard on the
   messy corpus, which has real umlauts (AUC 0.610 DE vs 0.863 EN). Real
   defect, wrong explanation.
2. **`decide_choice_corpus()`'s comment is false.** It claims to mirror
   `py/lens_decide1.py`'s corpus and to be kept in step with it. Same 40 tags,
   same labels, **zero byte-identical documents** — the C++ copy is an abridged
   rewrite. The `lure` flags exist only on the Python side.

## 6. `locate` on a real CV — the `top_k` gap, observed

At `top_k=1` the route returns single-token fragments: `'@gmail'`, `'ich,'`,
`'  '` for the candidate name. At `top_k=8` seven of eleven keys are usable
(phone, years, location, education, employer, email, title), two are weak
(name, skills).

This is the known consequence of the route returning *where the model looked*
rather than an assembled value, and a real document surfaced it immediately. A
client on the default `top_k=3` would see fragments and conclude locate is
broken. The engine-side fix — a per-key total, or span assembly — is a
`qemmi-lens/v4` format change and is **not** proposed here.

Of the two keys not in the CV, `certifications` correctly collapsed to the
lowest mass of all eleven (0.195). `languages` fired at 0.859 because the CV
literally contains `Languages: Python, C++, …` — a key-design collision, not a
fabrication.

## 7. What is NOT claimed

* Nothing here moves a constant. No calibration changed.
* Everything in §2 is one small self-authored corpus; §3, §4 and §6 are **one
  CV**. n=1 is a diagnosis, not a rate.
* No latency claim. Different hardware, 20x the parameters, untuned both sides.
* Laya was run with no temperature refit, which its card asks for. Its
  calibration numbers are therefore not tested here — only its ordering.

## 8. Open, in priority order

1. ~~Lure rate on the landed `choice` head.~~ **DONE 2026-09-21 — see §2.**
   It reversed the finding: we lead Laya on lures 81.2% to 62.5%.
2. **Restore real German** to the choice and score corpora and re-measure. The
   "German weakness is a model problem" claim rests on transliterated text.
3. **Harder choice documents — mostly lures.** §2 now shows the clean half is
   saturated for both systems and only the lure half discriminates.
4. **Is order rotation worth landing?** +5 points for 4x the prefills, measured
   once, on one corpus, on one model. A caller-side recipe today; making it a
   server option would be an architecture decision, not a tweak.
5. Fix the false `decide_choice_corpus()` comment, or re-sync the two corpora.

## 9. Content-free calibration — measured 2026-09-24, a no-go for the shipped heads

**Question.** ICR (Chen et al., ICLR 2025, arXiv 2410.02642) ranks documents
from attention and removes the model's intrinsic bias by subtracting the
attention of a *content-free* query ("N/A"). "Calibrate Before Use" (Zhao et
al., 2021) does the query-side analogue. Would either fix the weaknesses we
have on record — the separator peak (§6), score's rung-length bias (§3), and
"raw mass cannot tell present from absent"?

**Arms.** All on the landed 9B Q4_K_M heads, question vocabulary, `mean`,
mass summed over the body (span coverage checked: 1.000 on choice/score, mean
0.990 on absent):

| arm | what it does | cost |
|---|---|---|
| `raw` | the shipped readout | 1 prefill |
| `docCF` | ICR: an "N/A" key rides in the same prompt; each key's mass minus N/A's | 1 prefill |
| `qNA` | CBU: the same keys asked of the document "N/A"; each key divided by its own content-free mass | 2 |
| `qNEU` | as `qNA`, content-free document = length-matched neutral prose | 2 |
| `loo` | reference-set prior: each key divided by its mean over the OTHER corpus documents | needs a reference set |

Baselines reproduce the record exactly: choice 92.5% / lure 81.2%; score
E-per-level 0.62 / 1.02 / 1.49 / 2.13, concordance 1.0000; absent AUC 0.9973
(0.9948 on record, different key shuffle).

**Results (EN | DE).**

| job | raw | best calibrated arm | verdict |
|---|---|---|---|
| choice_py, all / lure | 95.0 / 87.5 \| 90.0 / 75.0 | `docCF` identical by construction; `qNA`/`qNEU` −5; `loo` DE lure +1 doc | no gain |
| absent AUC | 1.0000 \| 0.9939 | `docCF` 1.0000 \| 0.9912; `qNA` 0.96; `loo` 0.56 (see below) | no gain; EN saturated |
| score, rounded exact | 45.8 \| 41.7 | `docCF` 54.2 \| 54.2, E spreads to 0.42 … 2.21, concordance still 1.0000 | small, scale only |
| locate L11 h6 top3 (key+max) | 92.5 \| 94.3 | `docCF` 92.5 \| 94.3; exact-token top-1 55.0 → 55.0 \| 48.6 → 51.4 | no gain |
| locate L11 h6 top3 (question+mean) | 97.5 \| 91.4 | identical | no gain |

Locate ran in the probe (`LOCHEAD_CALIB=1`, paired: both readouts from one
prefill). Adding the N/A key does not disturb the baseline — pooled top3 is
still 93.3%. Across all 128 heads calibration raises top3 on 81 and lowers
it on 17: **it rescues weak heads, and the per-head search had already
picked heads that carry little of the bias.** The exact-token top-1 barely
moves, so the separator peak is query-specific, not a content-free draw — the
client-side expand-to-line fix stands.

`loo` on absent is invalid, not a finding: in this corpus a key's presence is
constant across documents (`customer` is always present, the six absent
concepts always absent), so a per-key prior divides the label away.

**The one real gain — rung-length skew.** The score corpus's rungs are all
similar length, so it cannot show the bias a query-side arm exists for. A
second run skewed the ladder both ways (2–4-word rungs against ~30-word rungs):

| ladder | raw argmax-exact EN \| DE | `qNA` | `qNEU` | `loo` |
|---|---|---|---|---|
| fwd (L1 short … L4 long) | 87.5 \| 95.8 | 41.7 \| 37.5 | 83.3 \| 70.8 | 91.7 \| 87.5 |
| inv (L1 long … L4 short) | 66.7 \| 62.5 | 33.3 \| 25.0 | 37.5 \| 25.0 | **91.7 \| 83.3** |

Concordance survives the skew under `raw` (0.986–1.000) — the comparator
claim holds — but the argmax does not. Content-free inputs make it worse. A
**reference-set prior repairs it**: divide each rung by its mean mass over a
few real documents of the same kind, and fwd and inv read alike. Caveat: the
prior here came from the test corpus itself (leave-one-out, label-free,
balanced 12 per level); a reference set skewed toward one level would bias
the other way.

**What this means.**
* No engine change, no constant. Content-free calibration is not worth a
  second pass for any shipped head.
* The reference-set prior is a **client-side** recipe for ladders, the same
  family as §3's anchors. It belongs in `../qemmi-lens`, not the server.
* Untested, and the reason ICR calibrates at all: comparing mass **across
  documents** (ranking chunks). Everything above compares queries against one
  document.

## Reproduce

```
python3 py/laya_cross_check.py --checkpoint typed-decisions --de-checkpoint multilingual

./build-metal/bin/qwenium-server -m models/Qwen3.8-9B-Q4_K_M.gguf \
    -c 4096 -s 1 -p 18190 --attention-lens --lens-locate-only
python3 py/lens_lure.py          # §2, both heads through one harness
python3 py/lens_calib.py choice score absent ladder   # §9, ~9 min

# §9 locate arm (probe, stop the server first):
LOCHEAD=1 LOCHEAD_CALIB=1 QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf \
    ./build-metal/bin/attn-provenance
# add LOCHEAD_QUESTIONS=1 LOCHEAD_AGG=mean for the question regime
```

Corpora are extracted to `py/corpora/lens_corpora.json` by a standalone dumper
built from the corpus functions in `tests/perf/attn_provenance.cpp` — the
compiler does the C++ string-literal handling rather than a parser. Laya itself
is not vendored; install it into a throwaway virtualenv, never the global
interpreter. The CV used in §3, §4 and §6 is personal data and is deliberately
**not** in the repo.
