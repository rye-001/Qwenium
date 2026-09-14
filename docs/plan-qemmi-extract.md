# `qemmi-extract/v1` — extraction without a lens

**STATUS: PROPOSED. Not approved, not built.** This is an architecture change
under CLAUDE.md's protocol — a new server flag, a new format string, and a second
response shape from `/v1/extract`. It needs an explicit decision before any code
lands, and `docs/architecture.md` updates in the same change.

## 1. The problem

`--attention-lens` is **refused fail-loud** on any model without a row in
`kLensCalibrations` (`server_lens.h`, keyed `{architecture, block_count}`). That
refusal is correct and must not be softened: the constants are coordinates
measured on one model, and running them elsewhere returns a confidently-shaped
report computed from someone else's layer and head. A false receipt is worse than
no receipt.

For Gemma this is a **measured** position, not neglect — 0 of 768 candidate heads
clear even a 70% bar against the 90% requirement
(`note-lens-gemma4-probe.md`, `note-lens-gemma-norm-weighted.md`).

But the refusal currently costs more than the lens. It blocks Gemma from
`/v1/extract` **entirely**, including the parts of the endpoint that have nothing
to do with attention and work fine on any model.

## 2. What is actually architecture-bound

Two of eight stages:

| stage | arch-bound? |
|---|---|
| parse request, build instruction | no (chat template already per-family) |
| pass 1 tapped decode | **yes — tap coordinates only** |
| extract JSON from decode text | no |
| attribute values to spans (citation head) | **yes** |
| coverage / grounded badge | **yes** |
| pass 2 candidate finder | no (prompted, taps disarmed) |
| warm document | no (slot/KV mechanics) |

There is no gradient here. Decode-derived signals work anywhere; attention-derived
signals do not work at all. That is a clean seam, and it is where the format
should split.

## 3. The payload

**Top level**

| kept | dropped |
|---|---|
| `format` (= `"qemmi-extract/v1"`), `model` | `citation_source`, `coverage_source` |
| `document`, `raw` | `used_threshold`, `ungrounded_threshold` |
| `fields` | `prompt`, `gen`, `hover`, `heat`, `skipped` |
| `key_candidates` / `candidates_error` | `uncalibrated` |
| `prefix` (warm), `vocabulary_mode` | `doc_lo`/`doc_hi`, `prompt_len` |

**`fields[]`**

| kept | dropped |
|---|---|
| `key`, `value`, `value_span` | `badge`, `body_mass` |
| `found_in_document` | `citations`, `citation_messages` |
| `stated` (bool) | `tier` |

`tier` goes because its only documented purpose is deciding how far to trust a
citation. `doc_lo`/`doc_hi` go because their only consumer is the viewer surface.

## 4. Two rejected alternatives, and why

**`uncalibrated: ["citations","badge","coverage"]` on `qemmi-lens/v4`.** The
`uncalibrated` array is a legitimate disclosure **only while something stays off
it**. That is why it works for question vocabularies: the head is measured, only
the vocabulary mode is not. Here it would name every lens signal in the report —
not a disclosure of degradation, but an empty report with a disclaimer stapled on.
An importer keying off `qemmi-lens/v4` has already been told those members exist.

**A two-state `badge` (`stated` | `absent`).** `badge` is tri-state today and two
of its three states are attention-derived. A client switching on `badge` would
silently lose its `grounded` branch and keep running. A **new member** (`stated`)
cannot be mistaken for the old one.

## 5. Claims and non-claims

**Claims**: values are verbatim-lifted (`found_in_document` checks it); the
vocabulary is answered exhaustively in hint order; `stated:false` means *the model
did not state this concept*.

**Non-claims, on the tin**: where the model looked; whether it read the document
at all; and — the one that costs — **`stated:false` does not mean "not in the
document."** The fidelity gate ("zero confident false receipts") has nothing to
check in this mode: it is **vacuous, not violated**, and the payload must not
imply otherwise.

## 6. Flag interactions — and one open decision

`--attention-lens` is refused alongside `--flash-attn` and `--speculative`
because both break the tap. **extract/v1 has no tap, so neither refusal carries
over** — it can run flash *and* speculative.

**"Free speed" now has a number, and a caveat** (both measured 2026-09-13,
`plan-lens-server-shape.md` §4.2.1/§4.3):

- The number is **8.4%**, not a step change. Flash prefill on Qwen3.8-9B-Q8_0:
  prefill 16.30 → 14.93 s on a ~2.4K-token document, extract end-to-end
  24.16 → 22.13 s. Prefill is 67% of the request at that length, so this is
  most of what flash can buy here. Worth having; not worth reshaping a plan
  around.
- The caveat is that on an **MoE** model, flash does not merely go faster — it
  can change **which values come out**. Measured on Qwen3.6-35B-A3B: a
  prefill-shape perturbation changed one extraction in fifteen outright (82
  generated tokens against 109). **This is now demonstrated rather than
  inferred** (`plan-lens-server-shape.md` §4.4): reading the router directly
  shows 2.3% of expert selections change on Qwen 3.6-35B and 6.4% on Gemma
  4-26B-A4B, while two dense models under the same perturbation change nothing.
  Expert routing is a top-k argmax, so the perturbation is discrete, not small.

That does **not** break any claim in §5 — verbatim lift, exhaustive vocabulary
and `stated:false` are all per-run properties, and both outputs satisfy them.
But it does mean extract/v1 on an MoE is **not reproducible across
configurations**, and anything downstream that diffs two runs (an archive, a
regression suite, a customer comparing yesterday to today) must know that.
Say it on the tin if this ships on an MoE.

**OPEN — and it is the SAME open decision as `plan-lens-server-shape.md` §5.2 /
§4.2.3, carried in two plans under two names.** Here it reads "concurrency vs
warmth"; there it reads "many documents at once vs one document edited fast".
One question, one answer, and it should be recorded in one place:
`--attention-lens` is single-slot because receipts-grade determinism is B=1
(`architecture.md` §1/§11); extract/v1 needs neither; but the warm-document
mechanism assumes exclusive ownership of slot 0, and it buys 92.6–98.9% of
pass-1 prefill on the key-editing loop.

**Proposed resolution (2026-09-13, user's framing — not yet decided):** it is a
**deployment configuration**, not a fork in the design. The two modes are
mutually exclusive *by construction* rather than by policy, so a startup choice
expresses it honestly — slots > 1 for throughput, slots = 1 to keep the warm
document. Neither plan then has to pre-commit on the operator's behalf, and
§4.2.3 over there becomes buildable without answering this first.

## 7. What changed since this was proposed (2026-09-08)

The original motivation was half about **images**. That half is gone:

- A **direct** image path (image → `fields`, no `document` string) leaves
  `value_span` and `found_in_document` with nothing to resolve against — the only
  verifiable claim extract/v1 makes. It would be schema-shaped VQA claiming
  nothing. Rejected.
- A **model-transcription** path is circular: `found_in_document` would verify a
  value against text the same model wrote moments earlier. Rejected.
- **OCR reaches scanned pages through the *calibrated* path** — measured, see
  `plan-ocr-lens.md` §§8–11 — with no engine work at all.

So extract/v1 is now about **Gemma, and handwriting where OCR fails**. That is a
narrower case than when it was specced, and the decision should be made on that
basis, not on "it unlocks images".

**One input that is about to exist.** A planned 2×2 runs the lens drift gate on
`gemma-4-12B-it` (dense) against `gemma-4-26B-A4B-it` (MoE) — same recipe, one
of each — to test whether the MoE instability above is a routing effect that
every MoE inherits. That experiment measures **numerics only** (token identity
and |Δpeak| on a tapped layer) and is emphatically **not** a lens-candidacy
probe for Gemma: the measured position there is unchanged and stands — 0 of 768
candidate heads clear even a 70% bar (§1). But its result bears directly on
this plan's model choice. **It has now been run, and routing is the mechanism**
(§4.4 there): `gemma-4-26B-A4B` gives extract/v1 config-dependent outputs where
`gemma-4-12B` — the same recipe — does not. If extract/v1 ships on Gemma, the
**dense 12B is the reproducible choice** and the MoE 26B is not. That is a
finding this plan can use directly, and it did not require revisiting Gemma's
lens candidacy at all.

## 8. Cost

~1 day: a new flag, a second serializer, a route branch on which flag armed. No
model work, no probes, no new measurement — everything here is already measured or
already architecture-free. It does **not** need a vision model, `--mmproj`, or any
image path.

## 9. The decision

Build it, or record it as dropped. It should not sit as a standing proposal — the
recurring failure mode in this repo is interfaces declared and never adopted.
