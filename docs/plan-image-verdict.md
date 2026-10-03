# Plan — the image verdict on `/v1/verdict`

**STATUS: DONE 2026-10-01 (P1–P5, §9). The image verdict is served on
`/v1/verdict` (Qwen3.6-35B-A3B UD-Q3_K_XL + its mmproj), with `image_id`
keeping an image across requests. Everything uncommitted.** Decisions taken by the user
(2026-10-01): build it (1), carry the image position in the snapshot (2), a
per-question cut in the model's configuration (3), Qwen3.6-35B-A3B (4). Plus
the instruction: **don't bloat the server; keep concerns separate.** Evidence:
`docs/note-verdict-img-probe.md` (§4–§11, real paper §9).

## 1. What it is

`POST /v1/verdict` with an **image** instead of a document: yes/no questions
about visible marks on one page (signed? stamped? field filled? box ticked?),
each answered `yes` / `no` / `unclear` from P(yes) against P(no) on the
prefill's last row — the probe's readout, nothing generated. **It reads no
attention**: it is not the attention lens and carries no receipt (image
attention readouts are closed, `docs/note-image-prefill-tap-probe.md`).

Out of scope (parked by the user 2026-10-01): reading text or numbers from
the image (items, prices, totals) — OCR → text lens is the path if revived.

## 2. Separation of concerns (the user's constraint)

The image verdict shares the **URL** with the document verdict, nothing else.

| piece | where | size |
|---|---|---|
| the driver: image + questions → report (prompt, image pass, snapshot, one question pass each, readout, bands) | **new** `src/server/image_verdict.{h,cpp}` | the bulk |
| its calibration table (§5) | in `image_verdict.h` — NOT a field on `LensConstants` | small |
| JSON report | `image_verdict_to_json` in the same module | small |
| decode the image, preprocess, expand markers, encoder handle | `ServerVision`: **one** narrow accessor (`prepare(image) → {bitmap, marker tokens}` + encoder ref); no verdict knowledge | ~20 lines |
| route | `http_server.cpp`: parse `image` vs `document` (exactly one), dispatch; the image branch is parse → one integration call → JSON, no logic | ~40–60 lines |
| image position in the snapshot (§4) | `src/session/` — a new snapshot section | contained |
| unit test | **new** `tests/unit/test_image_verdict.cpp` (co-location) | — |

`server_lens.{h,cpp}` (3.5K + 2K lines) is **not touched**. The document
verdict is unchanged byte for byte (gate G0).

**Gating follows the concern, not the URL:** the document branch needs
`--attention-lens` and the row's `verdict_layer` (as today); the image branch
needs `--mmproj`, a full model with its output head (a plain server — not a
truncated lens server), and an image-verdict calibration row. A plain
`--mmproj` server serves the image verdict without `--attention-lens`.

> **Flag (not a blocker):** one URL with two independent gates is the price
> of the approved shape. A separate `/v1/image/verdict` would make the gating
> self-evident. The plan keeps `/v1/verdict` as approved; say if you prefer
> the split.

## 3. Request and response

```json
POST /v1/verdict
{"image": "data:image/jpeg;base64,…",
 "questions": [
   {"id": "s", "mark": "signature", "subject": "delivery note"},
   {"id": "t", "mark": "stamp",     "subject": "invoice"},
   {"id": "q", "question": "Is there a QR code on the receipt?"}],
 "image_id": "optional, 1..256 bytes — keep the image pass (§6)"}
```

* `image` (data URI, decoded by the existing `image_data_uri`) **xor**
  `document`. One image only.
* A question is either a **calibrated mark** (`mark` + `subject`): the server
  renders the wording the cut was measured on — e.g. "Does the {subject} carry
  a stamp?" — or a **free question** (`question`): answered at 0.5 and marked
  `"calibrated": false` (as locate's question vocabulary is today).
* Prompt = the probe's, verbatim: image, then `Question: … / Answer with yes or
  no only.`, thinking off. Answer tokens = `run_lens_verdict`'s yes/no sets.

```json
{"format_version": "qemmi-verdict-image/v1",
 "model": "Qwen3.6-35B-A3B", "config": {"weights": "…", "mmproj": "…"},
 "image": {"tokens": 1440, "grid": [32, 45], "pass": "cold" | "warm"},
 "answers": [
   {"id": "t", "answer": "unclear", "p": {"yes": 0.86, "no": 0.14},
    "mark": "stamp", "cut": {"yes": 1.0, "no": 0.5}, "calibrated": true}]}
```

`answer` = `yes` if p ≥ cut.yes, `no` if p < cut.no, else `unclear`.

## 4. Snapshot: carry the image position (decision 2)

Today `capture_slot` refuses a slot holding an M-RoPE image span: the blob
records a row count, and the rope coordinate (`RopeDivergence{delta,
rows_after}` in `ForwardPassBase`) is lost (`plan-qwen35-vision-impl.md` §4
decision 3). Without it a hybrid model cannot answer a second question from
the post-image state (the first question overwrites the DeltaNet state).

**Proposal:** a new snapshot section `RPOS` holding the slot's
`RopeDivergence`, **written only when the slot has diverged**. Text slots and
Gemma image slots (one position per row, never diverged) write exactly the
blobs they write today. `restore_slot` re-installs the record instead of
dropping it. Needs on `ForwardPassBase`: a getter/setter for one slot's
record (two small methods beside `has_rope_divergence`).

* The manifest matches sections positionally; an optional trailing section
  needs the restore side to accept "present or absent". If that cannot be
  done cleanly, fall back to: always write `RPOS` (delta 0 for text) + bump
  `kSnapshotFormatVersion` — which invalidates persisted prefix-library blobs
  once. **Choose at implementation, with the evidence; byte-identical text
  blobs is the preferred outcome.**
* This settles `plan-qwen35-vision-impl.md` §4 decision 3 and lifts the
  `architecture.md` §12 line "VL sessions are not snapshottable" — both docs
  updated in the same change (§13 trigger: snapshot header/format).
* The CLI's `--image-prefix-cache` and the server's V2 image-prefix cache
  refuse M-RoPE recipes at setup today; lifting those refusals is **not** in
  this plan (separate decision).

## 5. The per-question cut (decision 3) and the model (decision 4)

A small table in `image_verdict.h`, keyed like the lens rows — `{architecture,
block_count, file_type}` plus the projector (`qwen3vl-merger`, projection dim
2048) — so an unmeasured model or mmproj is refused, never inherited:

| model (key) | mark | wording measured | cut yes / no | provenance |
|---|---|---|---|---|
| Qwen3.6-35B-A3B, `qwen35moe`/40/file_type 12 (UD-Q3_K_XL) + Qwen3.6 mmproj | signature | "Is the {subject} signed by hand in the '{box}' box?" | 0.5 / 0.5 | §7, §9: real paper 12/12 |
| same | stamp | "Does the {subject} carry a stamp?" | **1.0 / 0.5 — never yes** (was 0.9 / 0.5 until 2026-10-01) | §8 fresh set, §11 re-run, §9 real paper 12/12, `note-stamp-lures.md` |
| same | date | "Is the {field} filled in?" | 0.5 / 0.5 | §7, §9 |

Only the plain 40-block UD-Q3_K_XL file was measured, so only it gets a row
(the 41-block MTP build and other quants are refused until measured — the
per-model rule). **Stamp changed 2026-10-01 (user decision): never `yes`.**
Printed stamp-shaped badges and crisp show-through, found by the client, score
like real stamps (up to 0.995) and no cut separates them; no real stamp ever
scored below 0.5. A stamp answers `no` or `unclear` (`note-stamp-lures.md`).

## 6. Cost and warm reuse

Per request: encode (6.1 s per page) + image pass (3.5 s) once; each question
≈ 0.22 s + restore. 10 questions ≈ 12 s (today's probe path: ≈ 40 s).

`image_id` keeps the image pass across requests in a store like
`LensDocumentStore` (its own instance and route key, same TTL/size flag
semantics, keyed by image content hash). **Gate G4 before it is served on
MoE:** a warm question must be bit-identical to a cold one. The split at the
image end measured bit-identical (§10), and the warm path uses the same chunk
shapes, but the MoE drift history (`project_lens_drift_gate`: expert-selection
flips under different batch shapes) means it is proven, not assumed. If G4
fails, `image_id` is refused on MoE rows and every request is cold.

## 7. Phases and gates

| phase | work | gate |
|---|---|---|
| P1 | `RPOS` section + `ForwardPassBase` record accessors | **G1a** existing snapshot/prefix round-trip tests unchanged, text blobs byte-identical; **G1b** an image slot (35B-A3B) captured → restored → question logits bit-identical to the uninterrupted pass; **G1c** Gemma (MedGemma) image slot blob byte-identical to today (cross-family falsifier — it never diverges) |
| P2 | `image_verdict` module (driver, table, JSON) + unit test | **G2** P(yes) bit-identical to the probe (`r35_fa.tsv`) on the §4 arm (123 q) and the real-paper set (36 q) — same prompt, same readout, now via one image pass + resumed questions |
| P3 | `ServerVision::prepare`, the route branch, the integration call | **G0** document verdict responses byte-identical before/after; **G3** refusals: both / neither of image+document, image without `--mmproj`, uncalibrated model/mmproj, truncated lens server, >1 image, unknown `mark`, empty questions — each names slot, expected, actual (fail-loud contract); http-server tests pass |
| P4 | `image_id` warm store | **G4** warm = cold bit-identical on the 35B-A3B; otherwise refused on MoE |
| P5 | cost run, docs | end-to-end N = 1 / 3 / 10 timed; `architecture.md` (route, module, `RPOS`, §12 line), `lens-format.md` (response), client brief |

Each phase is its own change; nothing is combined with an optimization.

## 8. Not in this plan

Reading text from images; multi-image requests; Gemma/27B rows (no
measurement — the driver is recipe-agnostic, the table is not); lifting the
image-prefix-cache refusals; any attention readout on images.

## 9. Progress log

**P1 — DONE 2026-10-01.** The optional-section route worked: no manifest
change, no header or format-version bump.

* `ForwardPassBase::rope_record` / `set_rope_record` (fail-loud: delta > 0,
  rows_after within [delta, rows]); `RPOS` section id; `RopeCoordinateSection`
  in `slot_snapshot.cpp` (models/ does not link session/, so the section lives
  with the snapshot code and reaches the record through the two accessors).
  `capture_slot` no longer refuses a diverged slot; `restore_slot` reads the
  blob's section count first and registers RPOS when the blob carries it.
* `test-image-prefix-roundtrip` now runs every family through the production
  API (`make_vision_profile`, `capture_slot`/`restore_slot`, `get_rope_pos` for
  the question and decode positions); its hand-built copy of the section list
  is gone. A non-square Qwen bitmap (512 × 768 → 16 × 24) exercises the
  divergence.
* **G1a** text blob (Qwen3.5-0.8B prefix-library round trip) md5-identical
  before/after; session 10/10, prefix-library 9/9, 149 related ctest cases pass.
* **G1b** Qwen3.6-35B-A3B image slot: 389 rows, rope position 29; warm restore
  = live split = production single call, logits max diff 0, 12 decoded tokens
  identical.
* **G1c** MedGemma image blob md5-identical before/after; rope position = rows
  (262).
* `rope-divergence-tests` 12/12 (3 new: absent record, round trip, refusals).
* The two image-prefix-cache refusals stay; their stated reason was corrected
  (they place the question by rows — not "the format has no coordinate").

**P2 — DONE 2026-10-01.** `src/server/image_verdict.{h,cpp}` in `qinf-server`
(links `qinf-orchestration` + `qinf-image`); nothing in `server_lens`,
`http_server.cpp` or `ServerVision` touched.

* Driver: plan the questions → build each prompt (the probe's, verbatim) →
  check they share the image-inclusive prefix → pin a **materialized** LLM
  prefill (the cuts were measured that way; a `--flash-attn` server must not
  answer from other numbers) → image pass once → `capture_slot` (RPOS) when
  there is more than one question → per question `restore_slot`, prefill the
  question alone from `get_rope_pos` → P(yes) vs P(no) (`run_lens_verdict`'s
  token sets) → band. Slot 0 cleared and the prefill mode restored on any exit.
* Calibration: one row, `{qwen35moe, 40, file_type 12, qwen3vl-merger, 2048}`;
  marks signature (0.5/0.5), stamp (0.9/0.5; 1.0/0.5 since 2026-10-01), date (0.5/0.5), each with the
  probe's wording as a template. Free questions: 0.5, `calibrated: false`.
* `image-verdict-tests` 8/8: table sanity, lookup refuses another quant / the
  41-block MTP build / another projector / a missing file_type, the five
  measured wordings render to the probe's exact strings, every malformed
  question refused naming the field, bands, JSON.
* **G2 PASS: 166/166 P(yes) bit-equal** (exact doubles, same process) to one
  full prefill per question — §4 clean arm 123/123, real paper 36/36, the
  receipt 7/7. Answers at the measured cuts: 123/123, 36/36, 7/7.
* Speed of the same work (35B-A3B, per image): 3 questions 17.1 → 10.3 s
  (synthetic), 18.3 → 11.0 s (paper); 7 questions 35.0 → 12.1 s (receipt).

**P3 — DONE 2026-10-01.** The route and its wiring; `server_lens` untouched.

* `ServerVision`: `prepare_image(bytes)` + plain accessors (encoder, marker ids,
  projector tag, projection dim) — no verdict knowledge. The integration
  assembles `ImageVerdictVision`, exposes `image_verdict_unserved_reason()`
  (startup line + per-request 404) and `image_verdict_json()` (model lock,
  slot 0).
* `handle_image_verdict` (http_server.cpp): returns false — touching nothing —
  unless the body is a JSON object with `"image"`; one dispatch line at the top
  of the `/v1/verdict` handler. A mark's params are the question's other string
  members. `document`, `document_id`, `language` and `image_id` are refused next
  to an image (`image_id` names P4). `--help` mentions it; the stale
  "snapshot carries no rope coordinate" texts are corrected.
* **G0 PASS:** a server built without the dispatch line vs with it, the same 11
  document requests (EN, DE, cold and warm `document_id`, 7 refusals incl. a
  non-JSON body) — byte-identical responses. (A first attempt compared one
  binary with itself: the restored source and the object had the same
  one-second mtime, so make skipped the rebuild — caught by md5.)
* **G3 PASS:** 35B-A3B + mmproj — the valid request returns the G2 doubles
  exactly (sheet 01: signature yes 0.99294, stamp **unclear** 0.86161 — the
  show-through ghost — date yes 0.99443, free QR no, `calibrated: false`), and
  11 malformed requests are refused with 400 naming the field. Not served, 404
  with the reason: no `--mmproj` (9B), an uncalibrated model (MedGemma: its key
  printed against the one row), a truncated lens server (35B `--lens-verify-only`).
  A chat request after an image verdict answers normally (slot left clean).
* Tests: server-lens 175/175, http-server 18/18, image-verdict 8/8, 92 related
  ctest cases.

**P4 — DONE 2026-10-01.** `image_id` (1..256 bytes) keeps the post-image state.

* `ImageVerdictStore` in `image_verdict.{h,cpp}` — its own store, not
  `LensDocumentStore` (that one is built on lens routes, prefill shapes and
  truncation depths). Hit = same id + same image (`Bitmap::content_id` of the
  preprocessed pixels) + same image-inclusive tokens; same id for another
  image → 400; LRU out; idle > TTL dropped on every call. Fixed 4 entries / 15
  min — **no size flag** (a flag is an architecture change; say if wanted).
* Driver: on a hit `restore_slot` the kept blob (no encode, no image pass); on
  a miss the image pass, `capture_slot`, `put`. The snapshot header is built
  after the prefill mode is pinned, so a blob kept under another mode is
  refused by the header check. Report: `image.pass` = `cold` | `warm`.
* `image-verdict-tests` 11/11 (3 new: hit rules, LRU, TTL + zero size).
* **G4 PASS (35B-A3B, MoE):** sheet 01, 3 marks — no id, cold id, warm id give
  the **same doubles** (0.99294…, 0.86161…, 0.99443…). Cold 11.1 s → warm
  **0.7 s**. Same id + other image → 400. After four newer ids the first is
  evicted and runs cold again, same answers. (The MoE drift of the text prefix
  cache does not arise: the kept blob is the very state the cold request
  resumes from.)

**P5 — DONE 2026-10-01.**

* **Cost, end to end over HTTP** (35B-A3B, one A4 phone photo, 1530 image
  tokens, `--slots 1`): cold 10.9 / 11.2 / 12.8 s and warm 0.32 / 0.93 / 2.39 s
  for 1 / 3 / 10 questions; warm == cold in every case. ~0.2 s per extra
  question. (The probe path would have been ~43 s for 10.) Server RSS 17.2 GB
  with three images kept.
* `docs/lens-format.md` — new section "The image verdict" (request, response,
  marks and cuts, bands, image_id, cost, where served).
* Client brief: `../qemmi-lens/docs/engine-update-2026-10-01.md` (API facts
  only; product decisions are the client's).

**Open (the user's calls, not in this plan):** a size flag for the image
store; lifting the image-prefix-cache refusals (they place questions by rows);
more calibration rows (27B, other quants); rubber-stamp ink on real paper.

### 2026-10-01 (later) — stamp never says yes (user decision)

The client found printed stamp-shaped badges and a crisp show-through answering
`yes` above 0.9. A fresh set (`render DIR stamp3`) showed it applies to the whole
lure kind: no cut separates these lures from real stamps
(`docs/note-stamp-lures.md`). Option chosen: **the stamp's yes-cut is 1.0
(`kImageVerdictNeverYes`)**. The band rule never answers yes at that cut, a
saturated p of 1.0 included. cut_no stays 0.5. No p changed, only the band.
* `image-verdict-tests` 12/12, with a never-yes band test. `http-server-test`
  18/18, `server-lens-tests` 175/175.
* Live, cold, on the client's five pages: real stamp, URGENT badge and
  show-through all came back `unclear`, with p values identical to before the
  change. COPY stayed `unclear` and the logo page stayed `no`. The wire shows
  `"cut": {"yes": 1.0, "no": 0.5}`.

### 2026-10-02 — engine fixes, gates re-run, "where" (user: "go with 1", then "go with 2, then 1")
Asking the model for a box (`docs/note-verdict-img-ground.md`) found two engine
faults in image generation (decode masked KV rows by the rope position; block
instead of interleaved M-RoPE). Both fixed. The verdict gates were re-run under
the fixes (1053 synthetic questions): no served decision changed (§6 of that
note). New end-to-end gate `test-image-ground`.
* **`"where": true`** on the image verdict: the model's own box for each yes /
  unclear answer on a mark with a locate wording (a new column of the
  calibration row: the probe's measured wordings). Generated from the kept
  post-image state, mapped onto the uploaded picture through the letterbox the
  `Bitmap` now records. Answers unchanged by it. Unit tests: image-verdict
  16/16, image-loader 11/11.
