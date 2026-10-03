# Where is the mark? The model's own box, checked by covering it

2026-10-02. Idea 1 of the image-verdict follow-ups ("show where the mark is").
Qwen3.6-35B-A3B UD-Q3_K_XL + Qwen3.6 mmproj, `build-metal`. Synthetic pages
only. Results in `.session-results/verdict_img_ground/`.

## Result in short

- **The probe found two faults in our engine.** Before the fix, every box sat
  in the page header. llama.cpp, with the same files and prompts, gave correct
  boxes. Both faults are fixed (§2). Text output does not change.
- **With the fix, the model's box finds the mark.** On 60 marks, 59 boxes have
  IoU ≥ 0.5 against the drawn mark. For "the stamp" on a page with a printed
  stamp-shaped badge and no stamp, all 20 boxes land on the badge (IoU 0.82 to
  0.95).
- **Covering the box removes the answer. Covering a box elsewhere does not.**
  With the model's box painted over in the paper's colour, p(yes) drops below
  0.5 for 59 of 60 real marks. With a box of the same size elsewhere, it drops
  below 0.5 for 0 of 60.
- **A box is not evidence that the mark exists.** On a page with no marks, the
  model still gives a box: the empty signature box, or the empty date line. The
  verdict decides yes / no; the box shows where to look.

## 1. Method

- **Pages:** `render DIR ground` (tests/perf/probe_verdict_img_render.swift).
  The 10 layouts of §8 (note-verdict-img-probe.md), three variants each, clean
  PNG and degraded scan = 60 images:
  - `all`: signature, stamp, date;
  - `lures`: signature, date, no stamp, a stamp-shaped badge ("URGENT" in a
    double ring, docs/note-stamp-lures.md) and "Status: RECEIVED" printed in
    colour;
  - `none`: nothing marked.
  `boxes.tsv` holds the truth: where each mark and lure was drawn. The date's
  truth box is the text's full line height, so it is taller than the ink; date
  IoU stays near 0.67 even for a good box. The scan's skew moves a mark by up
  to ~15 px.
- **The model's box:** `/v1/chat/completions`, greedy, no thinking, prompt
  "Locate {the mark} in the image, output its bbox coordinates using JSON
  format." The model answers `{"bbox_2d": [x1, y1, x2, y2]}` in **0–1000
  relative** coordinates (the absolute-pixel reading scores 0/60).
- **The check:** `tests/perf/probe_verdict_img_occlude.swift` paints the box
  (plus 8 px each side) in the median colour of the ring around it. The
  control is a box of the same size at a seeded random place, off every drawn
  mark and lure. Then `/v1/verdict` asks the three calibrated questions again.
- Script: `py/verdict_img_ground.py ground | score | occlude | report`.

## 2. The two engine faults

The first run gave boxes like `[44, 92, 901, 91]` (y2 < y1), all in the top
10% of the page, and the text ran on after the JSON (`<|im_start|>user …`).
llama.cpp (`build-probe` llama-server) gave the right boxes for the same
request. The yes/no verdict was right on the same pages. Two causes:

1. **Decode after an image saw only the start of the prompt.** An M-RoPE image
   fills nx·ny KV rows but moves the rope position by only max(nx, ny): 1440
   rows, 45 positions on these pages. The decode mask compared KV rows with
   the rope position, so a generated token saw only rows 0..~80. That is the
   prompt head and the image's top rows. It did not see the question or its own
   earlier tokens. Prefill had this fix (`kv_base`, P4); batched decode did
   not. The persistent-graph write path (`KvWriteIndicesInput`, SetRows) also
   wrote the new token's K/V at the position, i.e. over an image row.
   - Fix: `StepContext::kv_rows` (per row: position + the slot's
     rows-minus-positions delta), filled by `ForwardPassBase::set_decode_inputs`
     only for a slot that hosted an image. The mask and the write index use
     `row_kv`. No diverged slot ⇒ no vector ⇒ the old step exactly.
   - Affected: every token generated after an image on a Qwen 3.5-family recipe
     (server chat, CLI). Not affected: the image verdict (prefill only, reads the
     last row), Gemma (no divergence), text.
   - Probably also the cause of the failed receipt reading of 2026-10-01
     (note-verdict-img-probe.md): reading is generation. Not re-tested.
2. **Wrong M-RoPE layout.** We ran MROPE (sections in blocks: all time dims,
   then all row dims, then all column dims). The Qwen 3.5 family is trained
   interleaved (IMROPE), as llama.cpp runs `qwen35` / `qwen35moe`.
   - Fix: `MRopeSections::interleaved`, set by `Qwen35Config::from_metadata`;
     `build_rope_gated` picks `GGML_ROPE_TYPE_IMROPE`.
   - Text: all four components equal ⇒ both layouts equal NEOX bit for bit
     (`MRopeLayout.TextIsBitIdenticalToNeoxUnderBothLayouts`, CPU).
   - Images: every image prefill changes, so every image-verdict number moves
     (§5).

Fix 2 was applied first and alone left the boxes wrong; with fix 1 added they
are right. Fix 1 alone (blocks layout) was not measured. Fix 2 stands on the
reference, not on this probe.

Tests: `attn-mask-input-tests` (`BatchedDecodeAfterImageMasksByKvRow`),
`kv-write-setrows-tests` (`WritesAtTheKvRowAfterAnImage`),
`rope-divergence-tests` (`DecodeInputsCarryEachRowsKvRow`,
`…WithoutAnImageCarryNoKvRows`, `…RefuseFewerSlotsThanRows`), `layer-tests`
(`MRopeLayout.*`, `MRopeSections.LayoutDefaultsToBlocks`).

End-to-end gate: `test-image-ground <model> <mmproj>`
(tests/integration/test_image_ground.cpp). It draws a 768x1024 page (red ring
top-right, blue square bottom-left), runs the production path (preprocess →
prefill_multimodal → decode_step, greedy) and needs IoU >= 0.5 for both boxes.
35B-A3B: ring 0.91, square 0.88, 53 s. With the decode fix switched off: ring
0.07, square 0.00, both FAIL.

## 3. Box accuracy (after the fix)

IoU against the drawn mark; clean (c) and scan (s), 10 each.

| variant | mark | c: IoU ≥ 0.5 | c: median | s: IoU ≥ 0.5 | s: median |
|---|---|---|---|---|---|
| all | signature | 10/10 | 0.81 | 9/10 | 0.72 |
| all | stamp | 10/10 | 0.95 | 10/10 | 0.92 |
| all | date | 10/10 | 0.67 | 10/10 | 0.69 |
| lures | signature | 8/10 | 0.67 | 9/10 | 0.71 |
| lures | date | 10/10 | 0.65 | 10/10 | 0.67 |
| lures | "stamp" → the badge | 20/20 land on the badge, IoU 0.82–0.95 | | | |
| none | any | 60/60 land on no drawn mark: the empty box or line | | | |

For every page with marks, the drawn object a box overlaps most (IoU ≥ 0.3)
is the mark that was asked for (the badge, for "stamp" on lure pages). Cost: one box ≈ 11.3 s cold through the chat route
(encode + prefill + ~30 generated tokens).

## 4. Covering the box

p(yes) of the asked mark, before → after covering (n = 20 per row).

| variant | mark | covered | p < 0.5 after | drop median (min) | other marks, max abs(dp) |
|---|---|---|---|---|---|
| all | signature | model's box | 19/20 | 0.88 (0.19) | 0.010 |
| all | signature | control | 0/20 | 0.00 | 0.007 |
| all | stamp | model's box | 20/20 | 0.99 (0.98) | 0.094 |
| all | stamp | control | 0/20 | 0.00 | 0.073 |
| all | date | model's box | 20/20 | 0.98 (0.94) | 0.109 |
| all | date | control | 0/20 | 0.00 | 0.052 |
| lures | "stamp" (badge) | model's box | 13/20 | 0.57 (0.24) | 0.098 |
| lures | "stamp" (badge) | control | 0/20 | 0.00 | 0.086 |

- The one signature miss (s_h3_all): the box began 10 px below the top
  strokes, which stayed visible (0.91 → 0.72).
- The 7 lure pages still ≥ 0.5 after the badge is covered are all scans
  (0.51–0.73): "Status: RECEIVED" in colour is still on the page, the known
  printed-status lure. One box shows one stamp-like thing, not all of them.

## 5. What this means for the app

- For a `yes` (signature, date) or an `unclear` stamp, the app can show the
  model's box: "here". For the never-yes stamp this is the useful part: the box
  shows the person which stamp-like mark to look at.
- The cover test turns the box into something measured: "covering this box
  changes the answer". It costs one more page read (~10 s).
- Do not show a box for a `no`: the model draws one anyway, on the empty field.
- Not decided (architecture, the user's): a "where" field on `/v1/verdict`, or
  a separate route. Today the box needs the chat route and a JSON parse.

## 6. Re-run of the verdict gates after the fixes

Fix 2 changes every image prefill, so the four synthetic gate sets were run
again with `probe-verdict-img` (2026-10-02, 1053 questions, `r35_im.tsv` next
to the old `r35_fa.tsv` / stamp3 `r35.tsv`). Fix 1 does not touch the verdict.
The 12 real-paper photos were not re-run (private; ask first).

- **Every answer is still a yes or a no token:** 1053 / 1053 compliant.
- **Clean set (§4, 123 questions):** AUROC 1.000 and 100% at 0.5 for all
  three marks, as before. Mean p on yes: signature 0.915 → 0.923, stamp
  0.995 → 0.994, date 0.994 → 0.993.
- **Hard set (§7, 450 questions):** the same pass/fail per family and
  condition. Signature and date pass everywhere; cross-family 0/300 wrong.
  Stamp on scans still fails at 0.5 (lures 2/10 right, was 1/10). Real
  stamps: min 0.959 clean (was 0.934), 0.987 scan (was 0.990).
- **Stamp sets (§8 stamp2, stamp3; 480 questions):** the same picture. Real
  and faint stamps ≥ 0.970 (stamp2) and ≥ 0.981 (stamp3). Badges up to 0.997,
  show-through up to 0.993. The stricter wording is still no fix.

Pooled stamp question, all four sets, the same pool old and new: 206 real
stamps; 120 lures (logos, printed status, badges, show-through).

| cut | real below, old | real below, new | lures at/above, old | lures at/above, new |
|---|---|---|---|---|
| 0.50 | 0 | 0 | 88 | 86 |
| 0.90 | 0 | 0 | 50 | 52 |
| 0.95 | 1 | 0 | 41 | 40 |
| 0.98 | 3 | 5 | 13 | 26 |
| 0.99 | 30 | 26 | 5 | 8 |

(This pool is slightly smaller than note-stamp-lures.md's 207 / 134, which
also counted the client's pages. Old and new here are like for like.)

**Conclusion:** no served decision changes. Signature and date pass at 0.5.
No real stamp is below 0.5 (min 0.959), so a stamp `no` stays reliable, and
no cut separates stamps from lures, so the stamp stays never-yes. No
calibration constant moves. The real-paper figure (36/36) was measured on the
old engine and is not confirmed under the fix.

## 7. Served: `"where": true` on `/v1/verdict` (2026-10-02)

The user chose to serve the box (idea 1). For each yes / unclear answer on a
calibrated mark, the route asks the probe's locate wording from the kept
post-image state and returns `"where": {"box": [x0, y0, x1, y1]}`, fractions
of the uploaded picture (docs/lens-format.md). Measured on the same 60 pages
(`py/verdict_img_ground.py route`, `route.tsv`): per image a cold request
without `where`, then the same `image_id` warm without and with it.

| variant | mark | answers | boxes | IoU ≥ 0.5 | median (min) |
|---|---|---|---|---|---|
| all | signature | yes 20 | 20 | 19/20 | 0.76 (0.33) |
| all | stamp | unclear 20 | 20 | 20/20 | 0.92 (0.88) |
| all | date | yes 20 | 20 | 20/20 | 0.67 (0.59) |
| lures | signature | yes 20 | 20 | 17/20 | 0.67 (0.43) |
| lures | "stamp" | unclear 20 | 20, all on the badge | — | — |
| lures | date | yes 20 | 20 | 20/20 | 0.66 (0.50) |
| none | all three | no 60 | none (by design) | — | — |

- Every box overlaps most with the mark it was asked for.
- The answers are untouched: p cold without `where` = p warm with it, 180/180,
  to the last digit.
- Cost: warm request 0.75 s (median), with three boxes 4.80 s; 1.36 s per box
  (max 1.50). Cold 10.4 s.

## 8. Every stamp-like mark? No (2026-10-02)

Idea: one box shows one candidate, but on 7/20 lure scans a second stamp-like
mark ("Status: RECEIVED" in colour) kept p above 0.5 after the badge was
covered (§4). Asked for all of them (`py/verdict_img_ground.py multi`, two
wordings: "Locate every stamp …" and "Locate every stamp or stamp-like mark
…", "output their bbox coordinates"), on the 60 pages:

| variant | boxes per page | found |
|---|---|---|
| all (one real stamp) | always 1 | the stamp 20/20 |
| lures (badge + status text) | always 1 | the badge 20/20, the status text 0/20 |
| none | always 1 | on the empty signature box 20/20 |

Both wordings gave the same boxes. The model names one stamp and stops; the
printed status text is never boxed. **Not worth serving.** A way that might
work: cover the first box and ask again (§4 already shows p stays above 0.5
when something stamp-like remains), at ~10 s per round.
