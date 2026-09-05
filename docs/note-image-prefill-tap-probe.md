# Image Prefill-Tap Probe — G0: narrow reach above chance; G1: FAIL on the decode-comparable channel (2026-09-03)

**Verdict amended a second time, 2026-09-03, after the chance-baseline and
column-swap follow-ups the first amendment called for.** The INCONCLUSIVE
verdict is now resolved by real evidence, not further argument. The harness,
the tap, the perception gate and every number below are sound — see §1–§4 for
what did not change.

**G0 (reach): PASS, but narrow — not "prefill reaches," rather "a small
minority of heads reach, well above chance."** With a proper chance baseline
(§3), the population of 384 (layer, head) pairs sits at or slightly below
chance at the median (0.52×–1.11× across depths and channels) — the
architect's prediction that "the population sits at or near chance" is
**confirmed**. But the correction to that prediction matters too: it is not
*only* L23H10. LAST-ROW has a second standout, L23H8, running 17.8×–19.3×
chance at every depth — *higher* than L23H10's own 11.1×–15.5× on that same
channel — and a real (if thin) shoulder of the population clears a 5× bar:
7.8–10.2% on LAST-ROW, 0.5–2.1% on ALL-ROWS. This is the round-2 "~6% clear
it" shape again, not a single-head fluke, though the great majority of the
384 pairs are genuinely uninformative.

**G1 (content-dependence): FAIL on LAST-ROW, the channel that actually
matters — and it fails exactly the way the brief predicted a real failure
would look.** The column-swap control (§5) put a second object in the frame
and swapped which side each object sat on. L23H10's LAST-ROW reading —
the query at the last prompt token, i.e. the channel actually comparable to
what a decode-time citation tap would read — **inverts in 2 of 6 valid
conditions**, and the inversion is not noise: in every single condition,
"tracks correctly" **exactly** coincides with "the named object is on the
right," regardless of whether that object is the circle or the square. This
is a clean, textbook positional (left/right) artifact — the same failure
mode that caught 2 of 5 heads in the original decode round 3. **A new PASS on
G1 is not supported.** The broader ALL-ROWS channel (mean over every suffix
query position, not just the last one) tracked the named object correctly in
6/6 valid conditions on the same head — a real and separate result, discussed
in §5, but it is not the decode-comparable channel and does not rescue the
primary claim.

**The "internal control" discrepancy from the first amendment does NOT
dissolve into a like-for-like match — but it also does not confirm a
model-specific ceiling.** §6 compares this probe's LAST-ROW numbers to the
original decode note's "row 0 = 65–67% of image-span mass" figure under the
SAME normalization (sink mass as a fraction of image-span-only mass, not raw
row probability). This probe measures 2.5–5.1% — more than 10× smaller than
the original figure, a real and substantial divergence, not a units artifact.
Combined with the fact that the original figure is a mean over 5
independently-selected heads and this probe's numbers are a population
statistic over 384 unselected heads, **the two measurements remain
genuinely incomparable** on two separate axes (different statistic, and now
a large measured gap even after fixing the normalization). This does **not**
retire the need for a same-model Qwen 3.6 comparison — if anything it
sharpens why one would be informative. Retracting the first amendment's
"model-specific ceiling" framing as unconfirmed, not correcting it to a
different conclusion.

## 1. Provenance

| | |
|---|---|
| Text model | `models/Qwen3.8-27B-Q3_K_M.gguf`, arch `qwen35`, Q3_K_M, 65 blocks |
| Vision projector | `models/Qwen3.8-27B-mmproj-BF16.gguf`, projector `qwen3vl-merger`, patch_size=16, n_merge=2 ⇒ align=32 |
| Attention layers tapped | 16, at `il` = 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 (discovered by scanning the built decode graph for `kq_soft.<il>`, not by reading any GGUF key) |
| Heads | `n_head_q` = 24 ⇒ 384 (layer, head) candidates per reading |
| `--flash-attn` | off (never enabled; confirmed empirically — `kq_soft` tensors existed and every sampled row summed to 1.0 ± 1e-6, which is impossible if flash attention had replaced the chain) |
| Driver | `tests/perf/probe_image_prefill_lens.cpp` (v2 — extends v1 with chance-baseline normalization and the column-swap control) → `build-release/bin/probe-image-prefill-lens` |
| Raw logs | `.session-results/prefill_probe/run_primary.log` (v1, depth sweep, raw numbers only — still valid, cited in §6), `.session-results/prefill_probe/run_v2.log` (this note's numbers) |
| Depth-sweep images | generated at runtime, BMP, 512×512, 16×16 merged-token grid (256 image tokens), object column fixed at grid col 8, object rows {2, 4, 8, 12} of 16, radius ≈0.9×align |
| Swap-control images | generated at runtime, same grid/canvas, red circle + blue square in the SAME row band (rows 8 and 12), columns 4 and 12, two layouts (circle-left/circle-right) |

Row 2 was used in place of the original brief's "row 1" deliberately, to keep
the object's pixel footprint clear of grid row 0 — see §4.

## 2. The tap and the chunk seam (what had to be built beyond the recon)

Unchanged from the first version of this note. `kq_soft.<il>` already exists
in prefill graphs; `get_attention_taps` is decode-shaped and unsafe at
prefill (never called — this harness reads `ggml_graph_get_tensor` +
`ggml_backend_tensor_get` itself, sized off the tensor's own `ne[]`, and
self-tests every `(layer, q, head)` row sums to 1.0 before trusting any
number). `prefill_multimodal` drives an image turn as three separate chunk
graphs (`[prefix][image][suffix]`, `src/engine/multimodal_prefill.cpp`); this
probe reimplements that three-chunk drive manually (prefix via the real
`feed_tokens`; image and suffix chunks via `build_prefill_graph` →
`mark_attention_taps` → alloc → `set_prefill_inputs` → compute → read →
`advance_cache`) so taps can be marked before each chunk's own alloc. v2 adds
one behavioral change: each condition in both Part A and Part B now does a
**fresh `clear_slot`** and re-runs the full prefix/image/suffix sequence,
rather than reusing state across questions — simpler and safer than trying to
rewind a shared KV to ask a second question about the same image.

## 3. Chance baseline (Task 1) — the depth sweep, re-read correctly

**What "chance" means here.** For a target region covering `k` grid cells
out of `n_img` image tokens, chance mass = (that (layer,head) pair's own
total image-span mass, **sink-excluded**) × `k / (n_img − 1)`. This
normalizes against each head's *own* budget for the image span rather than a
single global number, which is the more honest comparison — a head that
mostly ignores the image entirely should not look "concentrated" just because
what little it does look at happens to include the target.

**Target footprint, derived, not assumed:** 3 grid rows × 3 grid columns = 9
of 256 cells = **3.52%** of the image span, identical at every depth (the
generation geometry is constant across depths, as designed).

| depth | channel | image-span share (mean, raw) | sink share (mean, raw) | ratio-to-chance: mean / median / max (head) | clear ≥2× | clear ≥3× | clear ≥5× | L23H10 |
|---|---|---|---|---|---|---|---|---|
| 2  | LAST-ROW | 0.1364 | 0.0039 | 1.93× / 1.02× / **19.33×@L23H8**  | 122/384 (31.8%) | 76/384 (19.8%) | 39/384 (10.2%) | 15.47× |
| 2  | ALL-ROWS | 0.2109 | 0.0108 | 1.18× / 0.62× / 11.05×@L23H10     | 80/384 (20.8%)  | 38/384 (9.9%)   | 8/384 (2.1%)   | 11.05× |
| 4  | LAST-ROW | 0.1417 | 0.0037 | 1.75× / 0.92× / **18.87×@L23H8**  | 108/384 (28.1%) | 64/384 (16.7%) | 31/384 (8.1%)  | 12.97× |
| 4  | ALL-ROWS | 0.2145 | 0.0103 | 1.01× / 0.58× / 9.41×@L23H10      | 58/384 (15.1%)  | 23/384 (6.0%)   | 2/384 (0.5%)   | 9.41×  |
| 8  | LAST-ROW | 0.1401 | 0.0035 | 1.67× / 0.86× / **18.29×@L23H8**  | 103/384 (26.8%) | 61/384 (15.9%) | 30/384 (7.8%)  | 12.65× |
| 8  | ALL-ROWS | 0.2106 | 0.0100 | 0.90× / 0.52× / 8.74×@L23H10      | 48/384 (12.5%)  | 20/384 (5.2%)   | 2/384 (0.5%)   | 8.74×  |
| 12 | LAST-ROW | 0.1355 | 0.0036 | 1.89× / 1.11× / **17.82×@L23H8**  | 116/384 (30.2%) | 75/384 (19.5%) | 36/384 (9.4%)  | 13.36× |
| 12 | ALL-ROWS | 0.2098 | 0.0100 | 0.91× / 0.60× / 8.95×@L23H10      | 38/384 (9.9%)   | 18/384 (4.7%)   | 2/384 (0.5%)   | 8.95×  |

Reading this straight: the **median** pair is at or below chance on every
row of this table (0.52×–1.11×) — most heads carry no signal, confirming the
prediction. The **mean** sits slightly above chance on LAST-ROW (1.67×–1.93×)
mostly because of the long right tail, not because the typical head is
informative. **L23H8, not L23H10, is the single strongest LAST-ROW head**
at every depth, undiminished with depth (17.8×–19.3×) — this was missed in
the first version of this note because it only tracked L23H10 (picked from
the ALL-ROWS channel) forward into LAST-ROW. L23H10 remains the strongest
ALL-ROWS head throughout (8.7×–11.1×). Both heads live in layer 23.

## 4. Assumptions that had to be resolved (a documented standing hazard)

- **Grid ordering.** `image-local token index j → (row, col)` was derived
  from the engine: `src/graph_inputs/mrope_positions_input.cpp` computes
  `row = i / grid_w, col = i % grid_w` for M-RoPE image position
  construction, and this probe uses the identical formula. `mm_grid_for`
  returned exactly the requested 16×16 for every image generated (logged,
  checked).
- **Sink definition is under-specified in the source note and this probe had
  to pick one.** `docs/note-image-lens-probe.md` says "row 0 of the image
  span carries 65–67% of image-span mass, analogous to the BOS sink" —
  readable as the whole first grid row (16 tokens) or the single first
  position (matching the BOS analogy literally). This probe uses the
  single-token definition throughout, including in §6's normalized
  comparison — if a future probe adopts the whole-row reading, that
  comparison should be redone, not assumed to transfer.
- **Where the image span sits in a chunk's KV numbering** vs. the ROPE
  position the chunk was built at — these diverge once M-RoPE makes an image
  span advance position by `max(nx,ny)` but KV rows by `nx·ny`
  (`forward_pass_base.h`'s `get_rope_pos` / `note_span_rows_vs_positions`).
  The harness threads `pos` (rope) and KV-row bookkeeping separately,
  matching `multimodal_prefill.cpp`'s own split; the self-test (rows sum to
  1) would have failed loudly had this been wrong, and did not.
- **`get_attention_taps` decode-shape trap** — flagged in the original
  brief, not re-discovered, restated for completeness: `read_prefill_taps`
  sizes its buffer from the tensor's own `ne[]` rather than assuming
  `n_q==1`, so it is safe at both prefill and decode shapes by construction.

## 5. The column-swap control (Task 3) — this is the one that decides G1

**Design**, mirroring round 3 of `docs/note-image-lens-probe.md`: a red
circle and a blue square in the SAME row band (so depth is held constant),
columns 4 and 12 of 16 (fixed pixel footprints, so target and distractor
cover the identical cell count — fair chance normalization on both sides),
two horizontal layouts (circle-left/square-right, and the mirror), two
questions per image ("what colour is the circle/square? answer with one
word."), at two DEEP rows (8 and 12) — exactly where decode reported exact
0.000 and where the whole reach claim lives. Perception gate first, always:
**2 of 8 conditions failed the gate** (both were the "square on the left,
asked about the square" layout, at both depths — the model's 12-token
free-gen budget ran out mid-sentence, "The user wants the color of the
square. Looking...", never reaching "blue"). Those two are excluded from
scoring, not guessed at. This leaves **6 valid conditions**, covering both
positions for the circle-question and one position each for the
square-question — an honest coverage gap, see §6.

**L23H10, the standout ALL-ROWS head, traced across all 6 valid conditions**
(ratio-to-chance, sink-excluded; "tracks?" = target ratio > distractor
ratio):

| depth | circle is on | asked about | LAST-ROW target / distractor | LAST-ROW tracks? | ALL-ROWS target / distractor | ALL-ROWS tracks? |
|---|---|---|---|---|---|---|
| 8  | left  | circle | 3.89× / 10.16× | **NO — inverts** | 4.19× / 3.27× | yes |
| 8  | left  | square | 12.40× / 2.56×  | yes | 5.84× / 2.29× | yes |
| 8  | right | circle | 8.74× / 5.66×   | yes | 5.93× / 2.84× | yes |
| 12 | left  | circle | 4.05× / 8.48×   | **NO — inverts** | 4.09× / 2.75× | yes |
| 12 | left  | square | 10.34× / 2.66×  | yes | 4.88× / 2.23× | yes |
| 12 | right | circle | 8.11× / 5.19×   | yes | 5.04× / 2.44× | yes |

**LAST-ROW tracks in 4/6 and inverts in exactly the 2 conditions where the
circle sits on the LEFT and the circle is what's asked about.** Read the
table by side instead of by object: LAST-ROW tracks *every single time* the
named object is on the right, and fails *every single time* the named object
is on the left. That is a perfect, deterministic left/right positional
signature, not content tracking with noise — L23H10's LAST-ROW reading is
answering "what's on the right?", and it only *looks* like it is answering
the question when the question and the position happen to agree. This is the
column-swap control doing exactly the job it is designed to do: two of five
heads inverted this way in the original decode round 3, and this is the
same failure mode, once, in a different model and a different tap point.

**ALL-ROWS tracks in 6/6** on this same head, including both conditions
where LAST-ROW inverted. That is a genuine, reproducible difference between
"attention averaged across the whole question" and "attention at the single
query position decode would read" — worth its own investigation some day
(possibly: earlier tokens of the question, e.g. while the model is reading
the word "circle", carry real content-locked attention that a final
positional pull overrides only at the very last position) — but it is not
the decode-comparable channel, and a result that only holds when you average
away the position decode actually occupies does not establish that decode
(or a decode-shaped citation mechanism) would see content tracking. G1 stays
FAIL on that basis.

**A second head is worth flagging, not chasing further:** the population-level
LAST-ROW distractor max across these conditions is repeatedly held by
**L19H5** (20.4×, 23.8×, 22.4×, 23.0×, all far above L23H10's own numbers,
and always on the *distractor* side) — a candidate for an even cleaner
single-side lock. Not characterized further here; flagged for anyone
continuing this line.

## 6. Task 2 — the internal control, compared like with like

The first amendment to this note proposed "the ceiling is model-specific" as
the explanation for why this probe's LAST-ROW reading (max 0.19–0.23 raw,
from the v1 run) did not reproduce the original decode note's exact 0.000.
That framing was too convenient. Comparing properly:

- **Different statistic, confirmed, not dissolved.** The original figure is
  a *mean over 5 heads selected by an independent citation-head search on
  Qwen 3.6*. This probe's headline numbers are a *max, and separately a
  mean, over 384 UNselected heads on Qwen 3.8-27B*. No amount of re-reading
  the existing log turns one into the other — they are different objects by
  construction. The population LAST-ROW mean (raw, from `run_primary.log`)
  is 0.0057–0.0079 across depths — much closer to (though still not exactly)
  the original's 0.000 than the cherry-picked max is, which is the right
  intuition, but "much closer" is not "confirmed equal," and there is no
  principled way to select "the 5 right heads" on this model without
  redoing the citation-head search that produced the original 5.
- **Same normalization, now computed, and it does NOT dissolve the gap.**
  The original 65–67% figure is sink mass **as a fraction of image-span-only
  mass**, not raw row probability — §3's data makes this computable for the
  first time: sink-share ÷ image-span-share, per depth, per channel:

  | depth | LAST-ROW: sink/image-span | ALL-ROWS: sink/image-span |
  |---|---|---|
  | 2  | 0.0039/0.1364 = 2.86% | 0.0108/0.2109 = 5.12% |
  | 4  | 0.0037/0.1417 = 2.61% | 0.0103/0.2145 = 4.80% |
  | 8  | 0.0035/0.1401 = 2.50% | 0.0100/0.2106 = 4.75% |
  | 12 | 0.0036/0.1355 = 2.66% | 0.0100/0.2098 = 4.77% |

  2.5–5.1% vs. the original's 65–67% is more than a 10× gap under the exact
  same normalization convention. This is either a real difference in how
  strongly the "positional sink" phenomenon manifests between decode and
  prefill (or between the two models/quants), or evidence that this probe's
  single-token sink definition (§4) is measuring something narrower than the
  original's — but it is not an artifact of comparing raw-vs-normalized
  units, because both are now normalized the same way.
- **Verdict on Task 2: the discrepancy does not dissolve.** Both axes
  (statistic selection, and now sink magnitude under matched normalization)
  remain open. This does not retroactively confirm "model-specific ceiling"
  either — that was never tested directly. **A same-model, same-tap,
  same-head-selection run against Qwen 3.6 is still the clean way to close
  this**, and remains unbudgeted here per the original brief's fallback-only
  condition (this probe's primary model never failed).

## 7. What was NOT done

- **No third layout/question combo for the square-question, left-position
  case.** 2 of 8 swap-control conditions failed the perception gate (both
  "square on the left, asked about") — not a coverage choice, a real result
  of this specific run (see §5). The swap table in §5 has one fewer
  data point for that specific position/question pairing than for the
  circle-question. A retry with a larger free-gen token budget (12 was used
  here) would likely close this gap without changing anything else.
- **No Qwen 3.6 leg.** Still not run — see §6's closing paragraph. This
  remains the single most valuable follow-up for settling Task 2's open
  question, not (as the first amendment implied) for settling G1, which
  §5's swap control now settles independently of model choice.
- **No investigation of the LAST-ROW/ALL-ROWS divergence itself.** §5 flags
  that L23H10's ALL-ROWS reading tracks correctly where its LAST-ROW reading
  doesn't, and offers a guess (earlier question tokens carry real signal
  that a final positional pull overrides) without testing it. A per-suffix-
  query-position trace (not just the two aggregates used here) would settle
  this cheaply from data this same harness could produce.
- **No characterization of L19H5** (the LAST-ROW distractor-side outlier
  noted in §5) or of L23H8 (§3's LAST-ROW leader, never checked against the
  swap control at all — only L23H10 was traced through Part B). Both are
  candidates for the same kind of positional-lock analysis §5 gave L23H10.
- **No rung-2/3 question**, still rung 1 ("locate a visible thing"), matching
  all prior rounds.
- **No multi-seed / multi-object-size repeat.** One radius, one canvas size,
  fixed columns (4/8/12) throughout.
- **No engine changes**, still. Every tap call is the existing public API;
  `get_attention_taps` correctly avoided. `src/` untouched. Only
  `tests/perf/probe_image_prefill_lens.cpp` and its additive
  `tests/CMakeLists.txt` target changed.

## 8. Honest limits

- One model (Qwen 3.8-27B, `qwen35`, Q3_K_M), one mmproj, one quant, one
  seed, throughout both the depth sweep and the swap control.
- 16 of `qwen35`'s 65 blocks carry attention (the rest are DeltaNet/SSM, per
  the loader's own "16 attention layers (KV), 48 SSM layers" banner); only
  those 16 were tapped. The DeltaNet/SSM layers' role, if any, in image
  grounding is untouched by this probe.
- Grid geometry (16×16, 256 tokens) is well below the ~1024-image-token
  floor `src/vision/image_preprocess.h` documents for grounding-task
  accuracy. The perception gate passed on 12 of 14 total generations across
  both parts regardless (the 2 failures were free-gen budget exhaustion, not
  a wrong color) — this is a coarse forced-choice color task, not the
  fine-grained grounding that floor was written for.
- The swap-control images use a solid square vs. a solid disc of matched
  bounding-box size (so cell-count normalization is exact), but a square and
  a circle are not perceptually identical shapes; this probe cannot rule out
  a shape-specific (rather than pure left/right) component to what looks
  like a positional signature, though the "tracks iff on the right"
  regularity in §5 is total (6/6) and leaves little room for a shape
  confound to also explain it.
- Every "ratio to chance" number normalizes against that (layer,head)
  pair's OWN image-span budget, which is the right comparison for "is this
  head unusually focused," but means a head with a tiny overall image-span
  share can post a large ratio from very little absolute mass — the ratio
  table (§3) should be read alongside the raw image-span-share column, not
  alone.
- The model's free-generation completions continue to carry stray
  chat-template / special-token artifacts (`<|endoftext|><|im_start|>user`
  after clean answers) — harmless where the gate passed, a likely
  contributor to the 2 gate failures where it didn't, not investigated
  further.

## 9. Repro

```
MODEL_PATH=models/Qwen3.8-27B-Q3_K_M.gguf \
MMPROJ_PATH=models/Qwen3.8-27B-mmproj-BF16.gguf \
./build-release/bin/probe-image-prefill-lens
```

Synthetic BMPs are regenerated at runtime under
`.session-results/prefill_probe/` (not committed — session-scoped). Full logs:
`.session-results/prefill_probe/run_primary.log` (v1, raw numbers, still
cited in §6) and `.session-results/prefill_probe/run_v2.log` (this note's
Part A / Part B numbers).
