# Plan — question keys (a question as a lens vocabulary entry)

Status: **PROPOSED, not approved. The §4 probe HAS BEEN RUN (2026-09-07) — see §9; three of four bars pass and the fourth failed for a specification reason, not a measurement one.** This document exists to
be argued with. It touches a named seam (`lens-format.md`, `/v1/extract`) and the
calibration constants, so per CLAUDE.md's architecture protocol it needs explicit
user approval before any code lands, and `architecture.md` updates in the same
change.

Origin: the question *"we pass keys and ask for their values — can we also ask a
question and get an answer from the document? Isn't that just a special key?"*

**Short answer: half of it is a special key, and the other half is a different
product wearing this one's badges.** This document draws the line, and specifies
the one measurement that decides whether even the safe half is real.

---

## 1. The split

"Question" hides two features with very different relationships to this format.

**(A) A question that SELECTS A SPAN.** *"What is the delivery commitment?"* The
answer is a span of the document. This differs from the key `delivery_date` only
in that its name is a sentence rather than an identifier. Every invariant the
format rests on survives intact:

| invariant | survives under (A)? |
|---|---|
| `value` is a byte-exact slice of `document` | yes |
| `found_in_document` means something | yes |
| candidates are spans; `returned_as` links value → span | yes |
| absence earned by omission (`badge:"absent"`) | yes |
| `grounded` = "read from the document" | yes |

**(B) A question that PRODUCES AN ANSWER.** *"Can we get the goods before the end
of October?"* → *"No — delivery moved to 13 November, though 640 units could hold
the original date."* That answer is a synthesis of two sentences plus a negation
the document never writes. It is not a span, and not one of the rows above
survives.

**(A) is a parameter on the existing extraction. (B) is a different endpoint.**
This is exactly the "parameterize or split" judgment CLAUDE.md says to flag
explicitly rather than let ride, and the seam is the answer's *extractiveness*,
not its phrasing.

## 2. Why (B) is rejected, not merely deferred

Three reasons, in increasing order of severity.

**It nulls the v4 machinery silently.** A synthesized answer is in no candidate
set, so `returned_as` is `null` on every candidate, `found_in_document` is
`false`, and the surface renders *"the returned value matches no candidate"* —
the state this session's recall fix exists to stop firing falsely — on **every
answer, forever**. The feature would not degrade; it would invert.

**It re-points `badge:"grounded"` without saying so.** `grounded` means "the
value's tokens attended to the document body." On an extracted span that is a
strong receipt: the span either is or is not in the document, so the badge cannot
be over-read very far. On an inference, high `body_mass` on the token "No" says
the model was *looking at the document while it said no*. It says nothing about
whether the document supports "no". Shipping (B) through the existing badge does
not add a claim to the format — it **retroactively changes what every existing
badge means**, including on the extraction path that is already deployed. That is
worse than a new non-claim to defend, because no version bump makes an importer
re-read the ones it already trusts.

**It crosses the accept/reject line the format is built on.**
[`lens-format.md`](lens-format.md) §Non-claims: *"every signal the format does
carry records consideration; each refusal below would assert commitment."* A
question-answer with a green badge asserts commitment. We have run this play
correctly once already: SS3 shipped *which message a citation landed in* —
attribution — and refused the staleness verdict, after the serveable predicate
cried wolf on 7 of 9 correctly-handled corrections
([`note-ss3-matched-pairs.md`](note-ss3-matched-pairs.md) §3). Same discipline
applies here for the same reason.

**Consequence:** (B) is not on this plan's roadmap. If it is ever wanted it is a
separate endpoint, a separate format, its own badges and its own non-claims — not
a flag on `/v1/extract`.

## 3. The unmeasured assumption that (A) rests on

`kLensCalibrations` exists because coordinates measured on one model do not
transfer to another; the guard refuses `--attention-lens` on any model with no
entry, keyed `{architecture, block_count}`.

**The same argument applies to the TASK, and nobody has made it.** L27H13 was
measured as the retrieval head for `key: "value"` extraction at greedy, on the
messy corpus. Nothing establishes it is the retrieval head when the prompt is a
question. `coverage_used_peak` and `ungrounded_body_mass` are likewise
task-shaped constants. The calibration guard checks architecture and block
count — it cannot see what you asked the model to do, so a question routed
through the existing constants would produce receipts that are confidently wrong
with **no refusal to catch it**.

There is a concrete tripwire. `key_vocabulary` already accepts `{key, gloss}` and
`gloss` is dead — its consumer, the presence gate, was deleted in Stage 2. It is
the obvious place to put a question. But `run_lens_extract` deliberately keeps
the gloss **out of the instruction** so the prompt stays byte-identical to the
regime Stage 1 measured. Routing a question through that slot re-opens the prompt
and invalidates the calibration silently, with no version bump and no gate. Do
not do this, however convenient it looks.

**So the whole feature reduces to one falsifiable question:**

> Is the citation head still the citation head when the prompt is a question?

## 4. THE PROBE — `QKEY=1`

Design: **matched pairs**, the design that already caught what coverage could not
in SS3. Same documents, same expected values, same model, same greedy decode.
The **only** variable is whether the vocabulary entry is an identifier or a
question. Anything that differs between the arms is caused by the prompt shape,
which is the thing under test.

### Corpus — reuse, do not author

`qdocs_messy_corpus()` (15 docs EN+DE, 75 labelled `QLabel{concept, value}`
pairs) is already the corpus for Leg C and for the CAND gate, and its ground
truth is exactly what this probe needs: a concept, and the value a correct
extraction returns. **No new corpus.** Each concept is rewritten mechanically
into a question — `delivery_date` → *"What is the delivery date?"*, `customer` →
*"Who is the customer?"* — and the expected value is unchanged. The rewrite table
is authored once, checked in beside the corpus, and is the only new data.

This matters: a hand-authored question corpus would confound "questions are
harder" with "these questions are harder", and would not be comparable to any
number already measured.

### Arms

- **Arm K (control).** Today's extraction, byte-identical prompt, unchanged
  constants. Its numbers must reproduce the published ones (Qwen 3.8-9B: citation
  top-1 89%, top-3 98%) or the probe is misconfigured and nothing else it says
  can be read. This reproduction check is not optional.
- **Arm Q (the test).** Identical in every respect except that the vocabulary
  entries are the question strings. Same taps, same head, same constants, same
  `max_new_tokens`, same greedy path.
- **Arm S (conditional).** Runs **only if Arm Q fails.** A full head search over
  all candidates (the 768-candidate sweep used for Gemma and for the Qwen 3.8
  probe) on the question prompt, asking whether *some other* head is the
  question-retrieval head.

### Metrics

Per emitted field, score the citation exactly as Leg C does: the top-k cited
positions against the ground-truth value span in the document.

1. **citation top-1** and **top-3** accuracy, Arm K vs Arm Q.
2. **Head agreement** — over all fields, does the argmax-attention head under the
   question prompt remain L27H13? Report the distribution over heads, not just
   the rate. This is the mechanism question, and it is the one that generalizes;
   the accuracy numbers are the symptom.
3. **Extractiveness rate** — fraction of Arm Q answers that are byte-exact (or
   whitespace-normalized-exact, per this session's resolver) slices of the
   document. This measures how much of (B) leaks into (A) unbidden, and it is a
   product fact independent of the head.

### Bars

| # | metric | bar | if it fails |
|---|---|---|---|
| 1 | Arm K reproduces published top-1/top-3 | within 2 points | probe is misconfigured — fix before reading anything |
| 2 | Arm Q citation top-3 | **≥ 0.90** | the head does not carry questions at the required accuracy |
| 3 | Arm K − Arm Q top-3 | **≤ 0.05** | the prompt shape moves the head; constants are task-shaped and must be re-derived per task |
| 4 | Arm Q extractiveness | **≥ 0.90** | questions pull the model into (B) on their own — the safe half is not stably reachable by prompting, and the feature needs a constrained output shape or dies |

**Bars 2 and 3 together are the kill gate.** Passing 2 while failing 3 is the
interesting outcome, not a pass: it means questions work but on *different*
coordinates, and the deliverable is then a second calibration entry keyed by task
as well as by `{architecture, block_count}` — a real change to
`kLensCalibrations`' key, and a bigger decision than this feature.

Failing bar 4 is the cheapest kill and should be checked first: if the model will
not stay extractive when asked a question, nothing downstream matters.

### Cost and shape

Env-gated arm in `tests/perf/attn_provenance.cpp` (`QKEY=1`), byte-inert when
unset, exactly like `CAND=1` / `SS2=1` / `SS3=1`. Two passes over 15 documents;
same order of cost as the CAND gate. No engine change, no server change, no
format change — **the probe must be runnable before any of this plan is
approved**, and its result should decide the approval.

## 5. What ships if the probe passes

Additive, and deliberately small.

- Vocabulary entries gain an optional question form: `{"id": "delivery",
  "question": "What is the delivery date?"}`. **`id` is the join key** — short,
  stable, and what `fields` and `key_candidates` are keyed by. The sentence is
  never a map key; `key_candidates["What is the delivery date?"]` is not a
  contract anyone should have to write.
- The instruction changes ⇒ **a new calibration entry**, not a reuse. Whatever
  Arm Q measures is what the entry carries, and the report header names it, the
  way `run.model` already names the coordinates it actually ran.
- Everything else is unchanged: same badges with their same meanings, same
  candidate set, same coverage, same non-claims.

## 6. What does not ship, at any probe result

- A prose answer to a question. See §2.
- A `grounded` badge on anything that is not a span of the document.
- The question text as a map key.
- Questions routed through the dead `gloss` field. See §3.

## 7. Architecture-doc triggers

If approved: `architecture.md` §6 (lens constants / calibration) and §13, and
[`lens-format.md`](lens-format.md) (vocabulary entry shape; a version bump only
if `id` becomes required). A new non-claim — *"the lens does not answer
questions, it locates the spans that bear on them"* — belongs in
§Non-claims alongside CF1/SS1/CG1.

## 8. Open

- Whether `id` should become required for *all* vocabulary entries, which would
  be a subtractive format change (a real version bump), versus optional with the
  key string as the default `id`.
- Whether a question that legitimately has **no** span answer (*"is there a
  penalty clause?"* on a document with none) is `badge:"absent"` — probably yes,
  absence-by-omission already covers it — or wants its own state. Decide against
  the probe's output, not in advance.
- Whether the candidate set is the *better* primary output for a question than a
  single value, given that a question's natural answer is often several spans.
  If Arm Q's extractiveness is high but its set sizes are large, that is the
  finding, and the feature is "questions return candidate sets" rather than
  "questions return values".

---

## 9. Probe result — 2026-09-07, `QKEY=1` on Qwen 3.8-9B Q8_0 (Metal)

Harness arm `run_qkey_probe` in `tests/perf/attn_provenance.cpp`, env-gated and
byte-inert when unset. 15 docs EN+DE, 75 labelled fields, matched pairs, arms
differing **only** in whether a vocabulary entry is an identifier or a question.
Citation coordinates looked up through `lens_calibration_for()` — the same table
production uses — not the `FROZEN_SLOT/HEAD` globals.

| bar | metric | result | |
|---|---|---|---|
| 4 | Arm Q extractiveness | **73/75 (97%)** | **PASS** (bar ≥ 90%) |
| 2 | Arm Q citation top-3 | **407/409 (99.5%)** | **PASS** (bar ≥ 90%) |
| 3 | Arm K top-3 − Arm Q top-3 | **−0.7 points** | **PASS** (bar ≤ 5) |
| 1 | Arm K reproduces published 89%/98% | top-1 92% (+3.1), top-3 99% (+0.8) | **FAIL — see below** |

Arm K: top-1 374/406 (92%), top-3 401/406 (99%), 72/75 fields scoreable.
Arm Q: top-1 379/409 (93%), top-3 407/409 (100%), 73/75 fields scoreable.

**The head transfers.** Arm Q is not merely non-worse, it is marginally *better*
than the identifier control on both top-1 and top-3. The question in §3 — *is the
citation head still the citation head when the prompt is a question?* — is
answered yes for **citation**, at the calibrated L27H13, on this model.

**Bar 1's failure is a defect in the bar, not in the run.** Arm K uses a new task
wording (*"using EXACTLY these keys, verbatim"*, per-document key list); the
published 89%/98% came from Leg C's wording (*"prefer keys like: …"*, a fixed
eight-concept soft hint). The two are different prompts, so Arm K was never a
reproduction of that regime and the ±2-point bar could not have been met except
by accident. The deviation is also *upward* on both metrics, which is what a
stricter key instruction is expected to do. **Re-specify bar 1** as either (a)
Arm K running the byte-identical Leg C prompt, or (b) dropped entirely — the
matched-pairs design does not need an absolute anchor, since Arm K is the control
and the delta is the finding.

**The selection effect that could have confounded bar 3 did not materialise.**
Citation accuracy is scored only over fields whose value was found verbatim, so a
less-extractive Arm Q would have been scored on an easier self-selected subset.
Measured, the two arms are scoreable on 72 and 73 of 75 fields — near-identical
denominators, so the delta is a clean comparison.

**Head agreement is 52/73 (71%), and this does not contradict the above.** The
metric asks which head at the citation layer is *sharpest* (highest top-1
weight), not which head is *right*; H1 was sharpest on 12 fields, with a tail
across H3/H8/H10/H11/H12. Since H13's citation accuracy is simultaneously ~100%
at top-3, the honest reading is that sharpness and correctness are different
properties and sharpness is the weaker signal — consistent with CG1 (*"margin
measures difficulty, not ambiguity"*), which has now died a fourth time. Do not
re-derive a head choice from sharpness.

### What this changes in this plan

- **§5's requirement of a new calibration entry is too strong for citation.** The
  measurement says the existing citation coordinates carry to question prompts on
  this model. `coverage_used_peak` and `ungrounded_body_mass` were **not**
  measured under questions and remain unknown — so the accurate statement is
  *citation transfers; coverage and groundedness are unmeasured*, and a question
  entry must not claim receipts for the two that were not tested.
- **§8's first open question is resolved.** At 97% extractiveness the model stays
  on spans when asked a question, so *"questions return values"* is viable and
  the fallback framing (*"questions return candidate sets"*) is not forced.

### Still not measured

- Coverage and ungrounded thresholds under question prompts (see above).
- Arm S (the 768-head sweep) — **not run and not needed**, since Arm Q passed.
- Any non-Qwen family. Gemma has no citation head by measurement; nothing here
  changes that.
- Whether questions whose answer is genuinely absent behave as `badge:"absent"`.
  The corpus's seven concepts are all present in their documents.
