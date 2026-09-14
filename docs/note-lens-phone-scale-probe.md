# Lens at phone scale — does the attention tap survive a model that fits on an iPhone?

**Status: probe complete. Moves no constant.** Date 2026-09-11.
Subject: whether Qemmi-Lens (citation + omission report) survives on a model
small enough to ship inside an iOS app, and what the iOS platform already
provides. Prompted by a raw product idea, not by a defect.

Related: `plan-coverage-layer-search.md` (the matched-FPR instrument used here),
`note-lens-qwen38-probe.md` (the reference Leg C numbers), `plan-ocr-lens.md`
and `../qemmi-lens/docs/plan-page-documents.md` (the capture/display half).

---

## 1. The question, and why it was cheap

The idea: a small Qwen plus the iPhone's local features (Vision OCR, the
Foundation Models framework, Background Assets) as a side product — not a
replacement for the server path.

The whole idea reduces to one measurable question, because of a logical
closure recorded in §8: **the only reason to ship our own runtime on an
iPhone is the attention tap.** Apple already gives away OCR and a 3.18B
model with guided structured output; a small Qwen that only does extraction
is strictly worse than what is already resident on the device. So the bet
lives or dies on whether the tap works at a shippable size.

That is answerable entirely on the Mac, with models already on disk and the
probe harness already written. No iOS work was required to answer it, and
none was done.

## 2. Setup

| model | file | arch/blocks | tapped attn layers | heads |
|---|---|---|---|---|
| Qwen3.5-0.8B | `Qwen3.5-0.8B-BF16.gguf` | qwen35/24 | 6 — 3,7,11,15,19,23 | 8 |
| Qwen3.5-4B | `Qwen3.5-4B-Q8_0.gguf` | qwen35/32 | 8 — 3…31 | 16 |
| Qwen3.5-9B | `Qwen3.5-9B-Q8_0.gguf`, `Qwen3.5-9B-Q4_K_M.gguf` | qwen35/32 | 8 — 3…31 | 16 |
| Qwen3.8-9B | `Qwen3.8-9B-Q8_0.gguf` | qwen35/33 | 8 — 3…31 | 16 |

All runs in `build-metal` (Release + Metal), Apple M1 Pro.

```
ATTN_HEAD_AGG=1 QWEN36_MODEL_PATH=models/<gguf> ./build-metal/bin/attn-provenance
COVERAGE=1     QWEN36_MODEL_PATH=models/<gguf> ./build-metal/bin/attn-provenance
COVSEARCH=1    QWEN36_MODEL_PATH=models/<gguf> ./build-metal/bin/attn-provenance
```

Corpora unchanged: Leg C messy corpus (15 docs, 8 EN / 7 DE, 75 labelled
fields) and COV1. `ATTN_HEAD_AGG` needs no calibration entry — it sweeps every
tapped layer — so the `kLensCalibrations` refusal does not apply to these runs.

## 3. Citation at phone scale (ARM A: max over heads, per layer)

| model | scored tokens | best layer | top1 | top3 |
|---|--:|---|--:|--:|
| Qwen3.5-0.8B | **345** | L19 | 94.5% | **99.1%** |
| Qwen3.5-4B | 413 | L27 | 80.9% | **85.2%** |
| Qwen3.8-9B | 413 | L27 | 86.7% | **92.5%** |

**The clean comparison is 4B vs 9B**: same 8 layers, same 16 heads, same 413
scored tokens, same winning layer, no confounds. **9B → 4B costs 7.3 points.**

The 0.8B's 99.1% is **not a result** — see §4.2.

Frozen-head propagation (G4, single head L3H13, the shipped citation coords):

| model | top1 | top3 | tokens |
|---|--:|--:|--:|
| Qwen3.5-4B | 74.6% | 83.3% | 413 |
| Qwen3.5-9B Q8 | 75.3% | 83.1% | 413 |
| Qwen3.5-9B Q4 | 75.4% | 83.2% | 410 |

## 4. Two measurement defects this probe exposed

Both are properties of instruments we already ship in the probe suite, and
both would mislead anyone who reuses them. They are the most transferable
output of this work.

### 4.1 ARM A (max over heads) is a FLOOR, not a ceiling

Initially read as a generous upper bound on what a single head could do.
It is the opposite — max-over-heads **dilutes a strong head with noise from
the other H−1 heads.** Two same-layer instances:

| model / layer | frozen head | ARM A same layer | gap |
|---|--:|--:|--:|
| Qwen3.8-9B @ L27 | 98% (L27H13, `note-lens-qwen38-probe.md` §5.3) | 92.5% | −5.5 |
| Qwen3.5-4B @ L3 | 83.3% (L3H13) | 75.1% | −8.2 |

So every ARM A figure in §3 **understates** what a properly searched single
head would achieve. The 4B's citation arm is better than 85.2% suggests.

### 4.2 The citation metric silently drops fields the model got WRONG

`attn_provenance.cpp:6046-6048` skips any labelled field whose value the model
did not emit verbatim:

```cpp
if (!qdocs_span_in_prompt(tok, R, f.value, slo, shi)) continue;
size_t gb = R.gen_text.find(f.value);
if (gb == std::string::npos) continue;   // normalized away, unscoreable
```

A wrong extraction is **removed from the denominator, not counted as a miss.**
The 0.8B scored **345** tokens where the 4B and both 9Bs scored 413 — it
botched 68 fields and they vanished. Visible in its own output dump:
`m_en3` customer `"buyer"`, product `"stonegate builders"`; `m_de3` quantity
`"0"`; `m_en8` quantity `"975.00"` (that is the total); `m_de4` delivery `","`,
order_number `","`; `m_de7` customer `"gipfel"`.

**Consequence, and it generalizes well past this probe: the citation metric is
conditional on the extraction being correct, so it systematically flatters weak
models — the worse the model, the more of its failures leave the denominator.**
Any future small-model or cross-model citation comparison must report `cite_n`
alongside the percentage, or it is not a comparison.

This is also the precise product hazard: an accurate map of a wrong extraction,
with the wrong parts invisible to the metric meant to police it.

## 5. Coverage at 4B

### 5.1 COV1 cannot answer — it has one negative per half

```
calib: USED=11 DROPPED=1 VALUE=3 FILLER=5 | held: USED=10 DROPPED=1
```

The "held separation 91%" is 10/11 against **a single negative span**, chosen
from 48 candidates of which four tie at calib 100%. This is the saturation wall
recorded in `plan-coverage-layer-search.md` §11.3, and it is the instrument
that produced the shipped incumbent. It cannot rank layers on a new model.

### 5.2 COVSEARCH — Leg C, 149 spans (75 positive, 74 length-matched fillers)

Matched FPR, every layer pinned to admit 12/74 filler spans:

| layer | threshold | used-clear |
|---|--:|--:|
| L3 | 0.743 | 61/75 = 81% |
| L7 | 0.702 | **67/75 = 89%** |
| L11 *(incumbent)* | 0.716 | 58/75 = 77% |
| L15 | 0.910 | 50/75 = 67% |
| L19 | 0.887 | 51/75 = 68% |
| L23 | 0.757 | 64/75 = 85% |
| L27 | 0.897 | 65/75 = 87% |
| L31 | 0.489 | 54/75 = 72% |

At the shipped coordinates **L11 @ 0.705 → 60/75 = 80%** used-clear, 12/74 filler.

| model | used-clear @ L11 0.705 | EN | DE |
|---|--:|--:|--:|
| Qwen3.6-35B | 84% (63/75) | 88% | 77% |
| Qwen3.8-9B | 87% (65/75) | 90% | 83% |
| **Qwen3.5-4B** | **80%** (60/75) | 75% | 80% |

−7 from the 9B, mirroring the citation drop almost exactly.

**No layer on the 4B clears the ≥90% bar.** The best (L7, 89%) is fitted over 8
candidates with no held-out.

Gate: G1 PASS (candidate L3 @ 0.269 → 100%), **G2 FAIL** (filler-clear 46% vs
16%), G3 FAIL. Verdict KEEP — the same shape as on 9B and 35B: the apparent
win is bought by wrecking the negatives.

### 5.3 The fitted winners are English-only

| layer | EN | DE |
|---|--:|--:|
| L7 *(pooled best)* | 95% | **37%** |
| L3 | 100% | **57%** |
| L19 | 100% | **49%** |
| L27 | 100% | **49%** |
| **L11 *(incumbent)*** | 75% | 80% |

Every layer that looks strong pooled collapses in German. **The only balanced
layer is the incumbent**, and it sits near the bottom of the pooled table.
Operational consequence: if one accepts ~80%, take L11 and get it in both
languages; 89% is not actually available.

## 5b. Coverage at 4B, re-run after the tokenizer fix — §5 was an artifact (2026-09-12)

§5 concluded that the 4B has no usable coverage layer: *"No layer on the 4B
clears the ≥90% bar. The best (L7, 89%)"*, incumbent L11 at 77%, and the 4B 7
points behind the 9B. **That conclusion does not survive correct tokenization.**

`note-lens-mlx-swift-server.md` §7 fixed the engine's pre-tokenizer, which had
been GPT-2's rather than Qwen's. §5 ran before that. Re-running COVSEARCH on the
**same GGUF file**, same corpus, same probe, with only the tokenizer changed:

| layer | §5 (old tokenizer) | now (Q8_0) |
|---|--:|--:|
| L7 | **89%** *(§5's best)* | 65% |
| L11 *(incumbent)* | 77% | **87%** |

The ranking inverts. §5's "best layer" is now second-worst, and the incumbent —
which §5 had in the middle of the pack — is the best layer on the 4B. Span
counts moved too (149 → 146; 74 → 71 negatives), because the filler windows are
built on token boundaries. **A published layer-search result was an artifact of
the tokenizer defect.**

### 5b.1 The 4B is not behind the 9B

At the shipped coordinates, correct tokenization, BF16 weights:

| model | used-clear @ L11 0.705 | filler admitted |
|---|--:|--:|
| Qwen3.8-9B | 65/75 = 87% | 15/71 = 21% |
| **Qwen3.5-4B** | **65/75 = 87%** | **15/71 = 21%** |

Indistinguishable at this resolution. §5's "−7 from the 9B" was the tokenizer,
not the model. **The phone-sized model is not the weak link it appeared to be.**

Independently: the 4B's argmax citation head is **also L27H13** (90.9% top-3 at
BF16, vs the 9B's 98.1%) — the same coordinate, on a different model of the same
family. Two models is not a law, and the calibration table's refusal to
extrapolate stays right, but it is evidence the coordinate is less
model-specific than assumed.

### 5b.2 Where Q4 actually costs: Qwen3.5-4B, Q4_K_M = **2.78 GB** (5.13 BPW)

COVSEARCH pins each build to *its own* incumbent false-positive count, which is
correct for ranking layers **within** a build and wrong for comparing builds —
the pin moves. Pinning every build to one common operating point (15/71 filler
admitted = 21% FPR, BF16's incumbent point):

| layer | BF16 | Q8_0 | **Q4_K_M** |
|---|--:|--:|--:|
| L3 | 83% | 63% | 71% |
| L7 | 64% | 49% | 72% |
| **L11** *(incumbent)* | **87%** | **87%** | **65%** |
| L15 | 56% | 49% | 52% |
| L19 | 39% | 40% | 59% |
| L23 | 73% | 73% | 71% |
| L27 | 80% | 57% | 79% |
| L31 | 47% | 39% | 27% |

- **Q8_0 is free.** 87% → 87% at the incumbent. Same conclusion as the 9B (§6b).
- **Q4 costs the incumbent 22 points** of used-clear at fixed FPR: 87% → 65%.
- **No other layer rescues it.** The best layer at Q4 is L27 at 79%, still 8
  points below what L11 delivers at BF16. Re-calibrating the *layer* does not
  buy the loss back.

### 5b.3 The dangerous part is invisible at the shipped threshold

Read at 0.705 rather than at fixed FPR, Q4 looks almost fine:

| build | used-clear | filler admitted |
|---|--:|--:|
| BF16 | 87% | **21%** |
| Q8_0 | 88% | 24% |
| Q4_K_M | 85% | **31%** |

Recall barely moves (87 → 85). The false-positive rate goes up by half. That
asymmetry is the whole risk: a filler span admitted is a span the report calls
**consulted** when the model did not consult it — so it is a span that should
have appeared in `skipped` and did not. **Q4 does not make the omission report
noisier; it makes it quieter.** It reports the document as more thoroughly read
than it was, which is the one direction an omission report must not fail in.

### 5b.4 What this changes

- **Step 1 is answered: the 4B is viable at BF16/Q8 and is not viable at Q4 on
  the shipped threshold.** The instrument was not the limit, and neither was the
  model — the tokenizer was.
- A 4-bit 4B phone build needs its coverage threshold **re-derived**, and 5b.2
  says re-deriving the *layer* will not recover the loss. What a higher threshold
  buys, at what recall, is the next measurement.
- 2.78 GB at Q4 sits inside the ~2.5 GB envelope figure, so the size question is
  no longer the blocker. The trust question is.

### 5b.5 Limits

- **No held-out split.** All numbers are fitted on the same 146 spans. The
  cross-build comparison in 5b.2 is relative and survives this better than the
  absolute layer selection does, but the absolute percentages are optimistic and
  the §5 caveat about fitting over 8 candidates still applies.
- 15 documents, 8 EN / 7 DE. Not a corpus result.
- **The calibration key does not separate these two models.** `Qwen3.5-4B` and
  `Qwen3.5-9B` are *both* `qwen35`/**32**. `server_lens.h`'s key note lists the
  models it separates and does not list the 4B; its own warning — *"a future
  collision must be resolved by adding a field to the key, never by widening an
  entry"* — is now live. Neither model is calibrated today, so nothing is
  mis-served yet; the moment either gets an entry, the other silently inherits
  it. **This must be fixed before any 4B row lands.**
- Nothing here moves a constant. The 4B has no calibration entry and this probe
  does not propose one.

## 6. Quantization — Q4 vs Q8, matched pair

Same model (Qwen3.5-9B), two quantizations, identical corpus, identical
denominators (149 spans / 75 positive in both). This was the open question
gating on-device deployment, since on-device means 4-bit.

**At the shipped operating point, L11 @ 0.705:**

| | used-clear | filler admitted |
|---|--:|--:|
| Q8_0 (9.5 GB) | 64/75 = **85%** | 13/74 = 18% |
| Q4_K_M (5.7 GB) | 64/75 = **85%** | 14/74 = 19% |

**Identical positives; one extra false positive.** Citation likewise unmoved
(83.1% → 83.2%, §3). Halving the bytes costs one span.

**But the margin thinned.** Holding FPR exactly constant (matched-FPR table):

| layer | Q8 | Q4 | Δ |
|---|--:|--:|--:|
| L3 | 80% | 76% | −4 |
| L7 | 96% | 89% | −7 |
| **L11** | **84%** | **76%** | **−8** |
| L15 | 64% | 65% | +1 |
| L19 | 72% | 68% | −4 |
| L23 | 76% | 69% | −7 |
| L27 | 83% | 73% | −10 |
| L31 | 61% | 48% | −13 |

The two tables are not contradictory. To hold FPR, Q4 needed a **higher**
threshold at L11 (0.744 vs 0.707) — **Q4 pushed more attention mass onto the
filler spans.** The positives held; the negatives got noisier. At a fixed
0.705 that surfaces as one span; if FPR is pinned it costs ~8 points of recall.

**Reading: the shipped configuration survives Q4; the safety margin around it
shrinks.** The download-size lever (Q8 → Q4, half the bytes) is real.

## 6b. Q4 on the CALIBRATED model, with a BF16 reference (2026-09-12)

§6 answered the Q4 question on **Qwen3.5-9B** — which is *not* a calibrated
model (`qwen35`/32, absent from the lens table) — against a Q8_0 reference, and
did not separate weight drift from decode divergence. This section re-runs it on
the model the constants actually belong to (**Qwen3.8-9B**, `qwen35`/33, L27H13
@ 0.705), against a **BF16** reference, with that separation made.

Build: HF bf16 → GGUF (18.4 GB) → `llama-quantize` Q4_K_M (**5.78 GB, 5.02
BPW**). Probe: `MLXCMP` on `attn-provenance`, three builds, same corpus
(15 docs, 8 EN / 7 DE), `ATTN_FROZEN_SLOT=6 ATTN_FROZEN_HEAD=13 ATTN_COV_SLOT=2`.

### 6b.1 The rank survives, cleanly

| build | argmax head | top-3 | top-1 |
|---|---|--:|--:|
| BF16 (18.4 GB) | **L27H13** | 98.1% | 88.9% |
| Q8_0 (9.8 GB) | **L27H13** | 97.8% | 88.6% |
| Q4_K_M (5.8 GB) | **L27H13** | 99.0% | 91.7% |

BF16 reproducing 98.1% also independently re-confirms the shipped calibration
(`note-lens-qwen38-probe.md` §5.3 measured 98%). The citation coordinate is a
property of the weights, not of their precision.

### 6b.2 The confound §6 did not control for

**Q4 changes what the model says.** Generation identical to the BF16 run:

| build | all | EN | DE |
|---|--:|--:|--:|
| Q8_0 | 14/15 | 7/8 | **7/7** |
| Q4_K_M | 4/15 | 4/8 | **0/7** |

This matters because `cov_peak` is a max over **every decode step**. A document
that generates different text produces a different peak for reasons that have
nothing to do with weight drift — so a naive Q8-vs-Q4 span comparison (§6's)
measures the two effects added together. Restricting to documents whose
generation is byte-identical isolates the weight term:

| comparison vs BF16 | spans | median | mean | max | flips @0.705 |
|---|--:|--:|--:|--:|--:|
| **Q8_0** | 138 | 0.0015 | 0.0027 | 0.0220 | **0/138** |
| **Q4_K_M** | 41 | 0.0130 | 0.0298 | 0.1680 | **1/41** |
| *ggml→MLX runtime* (portability note §5) | 149 | 0.0019 | 0.0033 | 0.0204 | 0/149 |

**Q8_0 drift is indistinguishable from the runtime drift** — Q8 is free.
**Q4 is ~7× that on the median, ~9× on the mean, ~8× on the max**, and it flips
verdicts where the runtime flipped none. The risk ordering asserted in the
portability note (quantization ≫ runtime) is confirmed on the calibrated model,
at 7–9× rather than the 12× estimated from a proxy.

### 6b.3 The citation arm is fine; the omission arm is where it costs

`ungrounded_body_mass` (0.538), measured through `/v1/extract` on `m_en1` +
`m_de1`, Q8_0 vs Q4_K_M: **0/16 badge flips**, max |Δ| **0.0394**, skipped-span
counts unchanged (7→7, 4→4). Labelled field VALUES across the whole corpus:
73 identical, 2 changed (one customer gained, one lost — a wash).

One thin margin worth naming: `delivery_date` is the weakest grounded field on
every document, and on `m_en1` it sits at **0.582 against the 0.538 threshold —
a margin of 0.044, only ~1.1× the largest body_mass drift observed (0.0394).**
Nothing flipped, but that field is one bad document away from flipping.

So the split is: **Q4 is affordable for citations and expensive for coverage.**
Coverage is the omission report, which §1 says is the only unique asset.

### 6b.4 Limits — and one that matters

- The clean Q4 subset is **41 spans from 4 English documents**, because Q4's
  generation divergence removed **every German document** from the controlled
  comparison. The cleanest measurement is therefore also the least
  representative: it excludes exactly the language Q4 destabilized most. This is
  the main reason 6b.2's Q4 row should be read as an order of magnitude, not a
  number.
- 0/7 vs 7/7 German generation stability (Q4 vs Q8) is a clean sweep on a small
  corpus. It is consistent with §7's German thread and with the tokenizer finding
  in `note-lens-mlx-swift-server.md` §7 having been a *different* problem — but
  it is 7 documents and it is not yet a finding.
- Nothing here moves a constant. Per CLAUDE.md that is a user decision, and this
  probe does not support one: the shipped 0.705 held on 40 of 41 controlled
  spans.

### 6b.5 Verdict on the gating question

§6 said "the shipped configuration survives Q4; the safety margin around it
shrinks." That survives contact with the calibrated model — with the shrinkage
now quantified as **7–9× the runtime drift, concentrated entirely in the coverage
arm**, and with a second cost §6 could not see: **Q4 changes the model's answers
on 11 of 15 documents, and on 7 of 7 German ones.**

If a 4-bit phone build ships, the coverage threshold needs re-deriving at the
matched-FPR operating point on that build. It does not inherit 0.705.

## 7. The German thread

The one finding that has reproduced at every step, and worsened at each:

| step | EN | DE | gap |
|---|--:|--:|--:|
| Qwen3.6-35B | 88% | 77% | −11 |
| Qwen3.8-9B | 90% | 83% | −7 |
| Qwen3.5-4B, fitted layers | 95–100% | 37–57% | −40 to −50 |
| Qwen3.5-9B, Q8 → Q4 @ L11 | 78% → 78% | 80% → **71%** | −9 from quantization alone |

Quantization damages German while leaving English untouched. Shrinking the
model destroys German on precisely the layers that look best pooled. Every
pooled average in this subsystem's history has concealed this.

It is now the binding risk on the product, not a side observation.

## 8. The iOS platform (survey, 2026-09-11)

**AFMTextV7** — Apple's on-device model, exposed via the Foundation Models
framework: **3.18B params** (+48.77M draft model for speculative decoding),
mixed 2/4-bit QAT at **~3.7 bits/weight average**, **~1 GB on disk**,
**4096-token context**, **already resident** on every Apple-Intelligence
device, free, no keys. Requires iPhone 15 Pro or newer. Guided generation via
`@Generable`, tool calling, native LoRA adapters (Apple ships an adapter
training toolkit; adapters are 160 MB+, Developer-Program and licence gated).
Alongside it `PrivateCloudComputeLanguageModel` at 32K context, also free.

**Attention and logits are not exposed.** `LanguageModelSession` is a text API,
not a tensor API. The weights are OS assets, not a distributable checkpoint;
the runtime is a private ANE-backed framework. **So the omission report cannot
be built on Apple's model at all** — which is the wedge, and also the closure
in §1: without the tap there is no reason to ship our own bytes.

Everything else is available:
- **Runtime** — MLX Swift runs on iPhone and is a full tensor framework
  (attention readable). WWDC26 open-sourced `MLXLanguageModel`.
- **Distribution into Apple's API** — WWDC26's `LanguageModel` /
  `LanguageModelExecutor` protocols let any runtime back a `LanguageModelSession`.
- **Large post-install assets** — Background Assets (`BADownloaderExtension`),
  system-managed, resumable, Wi-Fi/charging aware; Apple pitches it for ML models.
- **OCR / capture** — Vision `VNRecognizeTextRequest`, already used by
  `../qemmi-lens/tools/vision_ocr.swift`.

**No iOS engineering blocks this idea. Only the model does.**

Download arithmetic: Apple's resident model 1 GB; Qwen3.5-4B @ Q4 ≈ 2.5 GB;
Qwen3.8-9B @ Q4 = 5.7 GB. §6 says Q4 is affordable, so 2.5 GB is the realistic
figure for a 4B-class on-device lens.

## 9. Verdict

1. **Citation survives at phone scale.** ~83% frozen-head at 4B, and §4.1 says
   that understates a properly searched head.
2. **Coverage does not.** 80% at 4B with shipped coordinates, no layer clears
   the bar, and the strong-looking layers are English-only (DE 37–57%).
3. **That is backwards from what the product needs.** Citations are the
   commoditized half; the omission report is the only half Apple structurally
   cannot copy.
4. **Q4 is affordable** — one false positive at the shipped operating point,
   with a thinner margin and a −9 German penalty.
5. **The platform is not the obstacle.** Runtime, distribution, asset delivery
   and OCR are all solved and mostly free.

**Recommendation: phone as capture and display, server as adjudication.** The
phone's highest-value contribution is rendering the lens over a photographed
page — cited spans highlighted, skipped clauses greyed — which makes the
omission report legible to a buyer for the first time, and runs the model
server-side where it is calibrated. That work is already specced
(`plan-ocr-lens.md`, `../qemmi-lens/docs/plan-page-documents.md`) and needs no
new science.

The on-device-brain variant is **not killed, not encouraged.** It reopens if
the German deficit is solved, or if a 9B-class model becomes shippable
(Q4 = 5.7 GB, feasible for a managed/vertical deployment, not for consumer).

## 10. Limits

- Q4 was measured on a **9B, not the 4B one would ship.** Small models are
  generally more quantization-fragile. A 4B @ Q4 is a projection (~80% at L11,
  German nearer 70%), not a measurement. Only `Qwen3.5-4B-Q8_0` is on disk;
  a clean test needs the BF16 weights, and requantizing Q8 → Q4 would be
  indicative at best.
- ARM A best-layer figures are **fitted** over 6–8 candidates with no held-out;
  the probe labels them so.
- Every model shows top1 ≈ 0.0% at its **last** tapped attention layer
  (L23/L31) — the argmax is position 0, the attention sink, which `in()`
  excludes via `p >= 1`. Consistent artifact across all four models, not a
  finding.
- Qwen3.5-0.8B, Qwen3.5-4B and Qwen3.5-9B have **no `kLensCalibrations`
  entry** and are refused by the server lens. Nothing here proposes adding one.
- Leg C remains 15 self-authored documents and 75 fields. The corpus
  bottleneck recorded in `plan-coverage-layer-search.md` §13 blocks this thread
  too, and adding phone-class models multiplies it.
- iOS platform facts in §8 are from vendor documentation, not measured here.
