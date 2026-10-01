# Stamp lures — the image verdict's stamp mark answers "yes" to things drawn like stamps

**2026-10-01. Requested by the qemmi-lens client, which found confident false
"yes" answers above the 0.9 cut.** Qwen3.6-35B-A3B UD-Q3_K_XL + Qwen3.6 mmproj,
the existing `build-metal` binaries (flash-attention encoder). Synthetic images
only. The options in §5 were the user's decision. **Option 4 was chosen and
applied the same day** (stamp: never yes, see §5).

**Bottom line.** The client's finding holds, and it applies to the lure *kind*,
not just to five pages. A printed badge drawn like a stamp (a ring around one
word) and a stamp showing through from the back of the sheet score the same as
real stamps: 0.58–0.995, mostly ≥ 0.9. **No cut separates them** from real
stamps, and the stricter wording does not either. In pixels, both look like a
stamp. Whether a mark was *printed by the form* or *pressed on this side of the
sheet* is not something this readout sees. What does hold: **no real stamp ever
scored below 0.5** (0/207 synthetic, 0/5 real paper), so a stamp `no` stays
reliable.

## 1. Reproduction (cold, no `image_id`, the server on :8082)

The show-through page was regenerated in a scratch copy of the client's
`medical.swift`. The other four pages were regenerated alongside it, and all
four are byte-identical to the client's files.

| page | stamp | p.yes | client saw | correct? |
|---|---|---|---|---|
| round red printed "URGENT" badge | yes | 0.9812 | yes 0.981 | no (false yes) |
| mirrored stamp showing through (alpha 0.22) | yes | 0.9557 | yes 0.956 | no (false yes) |
| big red printed "COPY" | unclear | 0.7596 | unclear 0.760 | yes |
| round printed practice logo, empty stamp box | no | 0.0482 | no 0.048 | yes |
| rectangular blue ink-style stamp | yes | 0.9881 | yes 0.988 | yes |

All five match the client's numbers. The signature answers on these pages were
all right (0.86–0.97).

## 2. Why the calibration missed it

The calibration never contained these two lures. As drawn, it would have
counted both as **positives**.

* **The real stamps in every synthetic set were drawn as a double ring with a
  bold word** (`drawStamp`: outer line 6 px, inner 2.5 px, "RECEIVED"/"PAID").
  The client's URGENT badge is the same drawing: outer 5 px, inner 2 px, a bold
  word. Only the word and the rotation differ.
* **The logo lure (§7/§8) was a filled disc with white initials.** That is a
  different picture, and it is why it stayed lower: 0.29–0.93, half answered no
  at 0.5.
* **The calibration's faint stamp was the stamp at 28% ink, labelled yes.** The
  client's show-through is the same stamp, mirrored, at 22%. Show-through was
  only ever seen *by accident* on the real paper (§9: blurred by the paper,
  0.54–0.86). A crisp synthetic one was never in the set.
* **Pristine renders are not harder in general.** The old logo lure scored
  *lower* clean than scanned (median 0.48 vs 0.67). The new badge scores the
  same clean and scanned (median 0.970 vs 0.976). The faint ghost scores lower
  scanned (0.81 vs 0.97), because blur erases it. Resolution is not the cause
  either: the client's pages are 1364 tokens, the sets were 1440. **The cause
  is the drawing, not the rendering.**
* **The contract was already wrong before this.** "Up to ~0.86" came from the
  real paper. The §8 fresh set already had a scanned logo at 0.925 (0.920
  before the flash encoder). In the provenance string, "lowest stamp 0.959" is
  the real-paper figure; the synthetic minimum is 0.934.

## 3. A fresh set of the two lure kinds (`render DIR stamp3`, 240 questions)

The ten §8 layouts, six variants each, clean and scanned:

* real stamp;
* faint stamp;
* none;
* **badge**: `drawStamp` upright at 80% size in the logo's place, with one of
  URGENT / ORIGINAL / PRIORITY / EXPRESS / COPY in the stamp colour;
* **ghost22** and **ghost12**: the stamp, mirrored, at alpha 0.22 and 0.12,
  where a stamp would sit.

Every image was asked both the original stamp question and the §8 strict one.
240/240 answers were compliant.

P(yes), min / median / max over 10 layouts, with the count at ≥ 0.9:

| variant | original, clean | original, scan | strict, clean | strict, scan |
|---|---|---|---|---|
| real stamp | 0.995 / 0.998 / 0.999 (10) | 0.997 / 0.998 / 1.000 (10) | 0.92 / 0.98 / 0.98 (10) | 0.98 / 0.99 / 1.00 (10) |
| faint stamp | 0.981 / 0.994 / 0.999 (10) | 0.991 / 0.995 / 0.998 (10) | 0.85 / 0.98 / 0.99 (9) | 0.94 / 0.98 / 0.99 (10) |
| none | ≤ 0.027 (0) | ≤ 0.014 (0) | ≤ 0.014 (0) | ≤ 0.077 (0) |
| **badge** | 0.58 / 0.970 / 0.979 (**7**) | 0.93 / 0.976 / 0.994 (**10**) | 0.26 / 0.85 / 0.88 (0) | 0.81 / 0.94 / 0.98 (5) |
| **ghost 0.22** | 0.967 / 0.985 / **0.995** (**10**) | 0.91 / 0.968 / 0.990 (**10**) | 0.87 / 0.94 / 0.96 (8) | 0.80 / 0.88 / 0.96 (4) |
| **ghost 0.12** | 0.954 / 0.970 / 0.980 (**10**) | 0.62 / 0.81 / 0.95 (2) | 0.78 / 0.86 / 0.95 (3) | 0.74 / 0.82 / 0.91 (1) |

Under the shipped wording at the 0.9 cut, **49 of 60 lures answer "yes"**.
None of them answers "no".

## 4. The whole distribution: real stamps vs every lure, all sets

The pool is the four synthetic sets on the current encoder (§4, §7, §8 with the
`r35_fa` re-runs, plus stamp3) and the client's five pages. Real stamps:
n = 207. Lures: n = 134 (filled logo 36, printed status text 34, badge 21,
ghost 41, the client's ring logo 1, COPY 1). The real paper (§9) is quoted from the note, not re-read:
its 5 stamps scored 0.959–0.998 and its lures ≤ 0.864.

| yes-cut | real stamps pushed to unclear | lures still answering yes |
|---|---|---|
| 0.90 (shipped) | 0 / 207 | 52 / 134 |
| 0.95 | 1 / 207 | 43 / 134 |
| 0.97 | 3 / 207 (+ ≥1 of 5 real paper) | 28 / 134 |
| 0.98 | 3 / 207 (+ ≥1 of 5) | 14 / 134 |
| 0.99 | 31 / 207 (+ ≥1 of 5) | 5 / 134 |
| 0.995 | 75 / 207 (+ ≥1 of 5) | 1 / 134 |

* **No cut separates the two.** The highest lure (0.995) sits above 77 real
  stamps. The lowest real stamp (0.934) sits below 47 lures.
* **The low end does separate.** No real stamp scored below 0.5 (min 0.934
  synthetic, 0.959 real paper). Pages with no stamp-like mark scored ≤ 0.252
  (95 pages).
* Answered `no` at 0.5, by lure kind: filled logo 18/36, printed status text
  24/34, **badge 0/20, ghost 0/40**.

## 5. Options and what each costs (measured where possible)

1. **Raise the yes-cut.** Every cut lets lures through and starts eating real
   stamps (table §4). At 0.98, 14 lures still pass and 3 synthetic plus at
   least 1 real-paper stamp drop to unclear. At 0.99, 5 pass and 31 drop. **Not
   a fix.** It would also be a cut fitted to whatever lures were drawn last.
2. **The strict wording as a new calibration row.** On stamp3 at 0.9 it lets
   21/60 lures through, against 49/60 for the shipped wording, and loses 1/40
   real stamps. On the client's pages it is worse: the real rectangular stamp
   drops to 0.766 (unclear) while the badge (0.833) and the ghost (0.924) stay
   above it. §8 already showed it failing on scans. **Not a fix.**
3. **Add badge and show-through to the stamp gate.** This is the right thing
   for the gate whatever is chosen: stamp3 becomes part of the stamp evidence.
   It fixes nothing by itself. The model does not separate these lures, so a
   gate including them fails at every cut.
4. **Stop the stamp mark from saying "yes"**: cut_yes = 1.0, cut_no = 0.5
   unchanged. The stamp then answers `no` (nothing stamp-like) or `unclear`
   (something stamp-like, so a person looks). Measured cost:
   * every real stamp becomes `unclear` (207/207 synthetic, 5/5 real paper);
   * every `no` stays right: 0 real stamps below 0.5;
   * 42 of the 70 old-kind lures (filled logo, printed status text) still
     answer `no`.

   This is a semantic change to the mark ("is something stamp-like here?",
   not "was it stamped?") and a cut change, so it is the user's decision. A
   label for the yes-less answer would also be new, and I have not invented
   one.
5. **Keep 0.9 / 0.5 and correct the contract.** State that `yes` means "a
   stamp-like mark is visible" and that printed badges and show-through land
   there. The client then renders a stamp `yes` as "check it". This costs
   nothing in the engine, but it leaves a confident-looking `yes` on the wire
   that the engine knows it cannot back.

Not measured, and an idea only: follow-up questions aimed at each confounder.
For example, "Is the stamp's text mirrored?" for show-through, or "Is the
round mark part of the printed form?" for badges. These would be new wordings
with new cuts, and the badge question is likely the same fuzzy distinction §8
failed on. One stamp3-sized run costs about 30 minutes.

**Decision (user, 2026-10-01): option 4, applied.** The stamp row is now
`kImageVerdictNeverYes` (1.0) / 0.5, the band rule never answers yes at that
cut, and the contract says so (`lens-format.md`). Option 3, adding stamp3 to
the stamp evidence, is done by this note.

**Recommendation: option 4, plus 3 and the contract correction below.** It is
the only option where the engine claims nothing it has not measured. Its `no`
is reliable on every set, and an `unclear` sends a person to look, which is
what the stamp answer already means in practice. Option 5 is the cheaper
alternative if a stamp `yes` must stay on the wire.

## 6. The contract text no longer holds: proposed wording (superseded by option 4; the applied text is in lens-format.md)

`docs/lens-format.md`, "The image verdict", the bullet after the marks table
currently reads "…score up to ~0.86 on scans and photos; real stamps … scored
≥ 0.959." That is false: lures scored up to 0.995 synthetic, and stamps scored
down to 0.934 synthetic. Proposed text **under the shipped cuts (option 5)**:

> * **`answer`** = `yes` if p ≥ cut.yes, `no` if p < cut.no, else **`unclear`**.
>   Only the stamp has an unclear band. **A stamp `yes` means "something drawn
>   like a stamp is on the page", not "this page was stamped":** a badge the
>   form prints in a stamp's shape (a ring around one word such as "URGENT") and
>   a stamp showing through from the back of the sheet score like real stamps
>   (up to 0.995), and no cut separates them (`docs/note-stamp-lures.md`).
>   Filled round logos and "RECEIVED"/"PAID" printed in colour scored up to
>   0.93 (`unclear` or `yes`). No real stamp scored below 0.5 (synthetic ≥ 0.934,
>   real paper ≥ 0.959), so **a stamp `no` is reliable**. **Treat `unclear` as
>   "look"**, never as no, and show a stamp `yes` as "check it".

Under option 4, the same paragraph would say that the stamp never answers
`yes`. The exact text would follow the user's choice of label.

The same correction applies to the stamp row's provenance string in
`src/server/image_verdict.cpp` ("highest lure 0.864"). That string is
calibration data, so the user decides it too. Neither change is an
architecture change: no route, flag, seam or state kind moves. Option 4
changes a calibration constant.

## Reproduce

```
swiftc -O tests/perf/probe_verdict_img_render.swift -o .session-results/verdict_img/render
.session-results/verdict_img/render .session-results/verdict_img_stamp3 stamp3
MODEL_PATH=models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf MMPROJ_PATH=models/Qwen3.6-mtp-mmproj-BF16.gguf \
  DIR=.session-results/verdict_img_stamp3 OUT=.session-results/verdict_img_stamp3/r35.tsv \
  ./build-metal/bin/probe-verdict-img                                   # ~30 min
```

The client's pages: copy `../qemmi-lens/demo/medical-pages/medical.swift` to a
scratch directory, add `labRequest("med5_lab_showthrough", lure: "showthrough")`,
build it and run it. Send each page cold to `/v1/verdict` with
`{"mark":"stamp","subject":"lab request"}` (or `"prescription"`).
