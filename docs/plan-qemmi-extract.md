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
over** — it can run flash *and* speculative. Free speed, and it makes a long
document materially cheaper.

**OPEN**: `--attention-lens` is single-slot because receipts-grade determinism is
B=1 (`architecture.md` §1/§11). extract/v1 needs neither. But the warm-document
mechanism assumes exclusive ownership of slot 0. **Concurrency and warmth
conflict; pick one deliberately rather than inheriting single-slot by
copy-paste.**

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

## 8. Cost

~1 day: a new flag, a second serializer, a route branch on which flag armed. No
model work, no probes, no new measurement — everything here is already measured or
already architecture-free. It does **not** need a vision model, `--mmproj`, or any
image path.

## 9. The decision

Build it, or record it as dropped. It should not sit as a standing proposal — the
recurring failure mode in this repo is interfaces declared and never adopted.
