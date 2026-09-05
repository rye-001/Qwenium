# Does norm-weighted attention beat raw alpha for the Lens? — NEUTRAL (2026-09-04)

**Verdict: NEUTRAL.** Citation holds (Metric A reproduces the shipped numbers
exactly; Metric B is statistically indistinguishable — top-3 is identical,
top-1 is +0.8pp on n=413, well inside noise). Coverage looked like it improved
(87% → 91% used-clear, crossing the 90% bar) until the comparison was made
fair: **once Metric A is also given a threshold recalibrated on Qwen 3.8
instead of the stale Qwen-3.6-derived 0.705, it ties Metric B exactly — 68/75
(91%) for both.** The entire apparent gain was a stale cross-model threshold,
not a property of norm-weighting. `‖V_j‖` carries no additional signal here.

This is a calibration probe on Qwen only, per the gate's scope — Gemma is a
separate, later leg gated on this result (which does not clear it: there is
nothing here worth carrying into a second family).

**Date** 2026-09-04 · **Status** measurement note; no shipped code touched;
`LensConstants` (`src/server/server_lens.h`) and the architecture refusal are
unchanged, as instructed.

## 0. The question

The Qemmi-Lens citation receipt asks: does the argmax/top-3 of a head's
post-softmax attention row `kq_soft.<il>` land inside the source span the
value was copied from? Every prior probe scored raw attention weight `alpha`.
But `output = sum_j alpha_j * V_j`, and value-vector norms `‖V_j‖` vary across
positions — so `alpha` alone may misjudge what a head actually *contributed*.
The standard correction is norm-weighted attention: `score_j = alpha_j *
‖V_j‖`, renormalized over `j`. This note tests whether that correction is a
better estimator, on Qwen, where the right answer is already known
(`docs/note-lens-qwen38-probe.md`).

## 1. Provenance

| | |
|---|---|
| Model | `models/Qwen3.8-9B-Q8_0.gguf`, arch `qwen35`, 33 blocks, Q8_0 |
| Attention layers | 8, at `il` = 3, 7, 11, 15, 19, 23, 27, 31 (tap slots 0..7) |
| Heads | `n_head_q`=16, `n_head_kv`=4 (GQA, group size 4), `head_dim`=256 |
| Citation head | **L27H13** = tap slot 6, head 13 → KV head 13/4 = **3** |
| Coverage layer | layer 11 = tap slot 2 (frozen, no layer search — out of scope) |
| KV cache element type | F32 (`create_forward_pass`'s default; this probe never passes `--kv-type`) |
| `--flash-attn` | off throughout (required — `kq_soft` never materializes under flash) |
| Driver | `tests/perf/attn_provenance.cpp` → `build-release/bin/attn-provenance`, env `NORM_WEIGHTED=1` |
| Env overrides used | `ATTN_FROZEN_SLOT=6 ATTN_FROZEN_HEAD=13 ATTN_COV_SLOT=2` (Qwen 3.8's own head, per `note-lens-qwen38-probe.md`) |
| Corpora | Arm 1: Leg C messy corpus (15 docs, EN+DE, 413 scored value tokens) — same corpus as the ground-truth note. Arm 2: COV1 corpus (`cov_calib()`+`cov_held()`, 12+11 = 23 labeled TARGET spans, USED vs DROPPED) |
| Raw log | `.session-results/qwen38_normweighted_leg_c_cov1.log` (this run; a first run without the fair-threshold cross-check reproduced byte-identical Metric-A numbers and is not separately kept) |
| Process hygiene | Model process exited cleanly both runs; confirmed via `ps aux` no `attn-provenance` process remains |

## 2. The two metrics, scored in the same pass

- **Metric A (shipped):** `score_j = alpha_j`, exactly the tapped `kq_soft` row.
- **Metric B (candidate):** `score_j = alpha_j * ‖V_j‖`, renormalized over all
  `j in [0, n_kv)` so the row sums to 1 (mirroring how Metric A's raw softmax
  row already sums to 1 over all `j` including the BOS position). Ranking
  (top-1/top-3) excludes `j=0` as a candidate, same as Metric A's own
  `topk_head` — the BOS-sink exclusion applies equally to both metrics, for
  the same reason (an attention sink would otherwise win by construction
  regardless of `‖V‖`).

`‖V_j‖` is read directly off `simple_kv_cache::get_v_cache_tensor(cache_layer)`
after generation — no graph change, no new tap (see §5 for the traps this
plumbing exposed).

## 3. Arm 1 — citation (the STOP condition)

Leg C messy corpus, 15 docs (EN+DE), 413 scored value tokens — the same
corpus and the same 369/413, 404/413 numbers `note-lens-qwen38-probe.md`
reports.

| | Metric A (raw alpha) | Metric B (alpha·‖V‖) |
|---|---|---|
| top1, EN | 195/218 (89%) | 196/218 (90%) |
| top1, DE | 174/195 (89%) | 176/195 (90%) |
| **top1, combined** | **369/413 (89.3%)** | **372/413 (90.1%)** |
| top3, EN | 213/218 (98%) | 213/218 (98%) |
| top3, DE | 191/195 (98%) | 191/195 (98%) |
| **top3, combined** | **404/413 (97.8%)** | **404/413 (97.8%)** |

**Metric A reproduces the shipped numbers exactly** — same 369/413 top1,
same per-language 213/218 and 191/195 top3 splits `note-lens-qwen38-probe.md`
reports — confirming the new value-norm plumbing changed nothing about the
existing tap or scoring path. **Metric B's top-3 is bit-for-bit identical to
Metric A's** (404/413 both — expected, since renormalizing a row is a
positive rescale of every candidate and cannot change an argmax/top-3
ranking by itself; the two metrics can only disagree via the BOS-sink
exclusion boundary, which they did not here in a way that moved top-3). Top-1
moved by 3 tokens out of 413 (+0.8pp), which is inside sampling noise for
n=413 (a 95% CI on a ~90% rate at this n is roughly ±3pp). **Gate: PASS —
Metric B does not drop below 95% top-3; the STOP condition does not fire.**
Arm 2 proceeds.

## 4. Arm 2 — coverage separation

### 4.1 COV1: USED-vs-DROPPED separation (the labeled set)

`cov_calib()` + `cov_held()`, 12 + 11 = 23 TARGET spans at layer 11, each
labeled USED (the model's answer actually reflects that span) or DROPPED (a
correction/override the model ignored).

| | Metric A (raw alpha) | Metric B (alpha·‖V‖) |
|---|---|---|
| calib best threshold | 0.453, dir + | 0.469, dir + |
| calib accuracy | 100% | 100% |
| calib AUC | 1.000 | 1.000 |
| **held accuracy** | **91%** | **91%** |
| **held AUC** | **1.000** | **1.000** |
| median FILLER | 0.124 | 0.118 |
| median DROPPED | 0.411 | 0.407 |
| median USED | 0.929 | 0.938 |
| median VALUE (anchor) | 0.948 | 0.951 |

On the labeled separation task itself, the two metrics are indistinguishable:
identical held AUC (1.000, i.e. every USED span already outranks every
DROPPED span at the best threshold — the classes are cleanly separated by
both metrics), identical held accuracy, and near-identical medians (all
differences ≤0.009). This 23-span corpus is small — a tie at AUC=1.000 for
both metrics means it cannot show a *difference* even if one existed, only
that neither metric is confused by it.

### 4.2 The cross-check that mattered: applying each threshold to Leg C's own used spans

The 90%-bar figure the ground-truth note actually reports (87%, `note-lens-qwen38-probe.md`
§5.3) comes from Leg C's 75 used spans against the shipped, Qwen-3.6-derived
constant `0.705` — not from COV1. The gate instructions were explicit that
this constant must not be reused for Metric B, so its own best threshold
(0.469, from §4.1) was applied instead. First pass:

| | threshold | Leg C used-clear |
|---|---|---|
| Metric A @ shipped 0.705 (Qwen 3.6-derived) | 0.705 | 65/75 (87%) |
| Metric B @ COV1-best (0.469, this run) | 0.469 | 68/75 (91%) |

68/75 crosses the 90% bar raw alpha misses — but the comparison is unfair:
**0.705 was calibrated on a different model** (Qwen 3.6). A fair test needs
Metric A's own threshold, recalibrated on Qwen 3.8 the same way Metric B's
was. Re-run with that third column added:

| | threshold | Leg C used-clear |
|---|---|---|
| Metric A @ shipped 0.705 (Qwen 3.6-derived, stale) | 0.705 | 65/75 (87%) |
| **Metric A @ 0.453 (COV1-best on Qwen 3.8, fair)** | 0.453 | **68/75 (91%)** |
| **Metric B @ 0.469 (COV1-best on Qwen 3.8)** | 0.469 | **68/75 (91%)** |

**Metric A, given a fair same-model threshold, ties Metric B exactly — 68/75
(91%) both.** The 87%→91% jump is entirely explained by recalibrating a
threshold that was stale for this model; `‖V_j‖` weighting contributes
nothing measurable once that confound is removed. This is the central
finding of the probe and the reason the verdict is NEUTRAL rather than
BETTER: the promising-looking number in the first pass did not survive a
same-model control.

## 5. Verdict, spelled out

**NEUTRAL.** Citation holds (Metric A reproduces the ground truth exactly;
Metric B does not degrade it — top-3 identical, top-1 noise-level). Coverage
does not move: on the labeled separation task the two metrics tie at every
statistic measured (AUC, held accuracy, class medians), and the one number
that looked like an improvement (used-clear 87%→91%) reproduces identically
under Metric A once Metric A is given the same fair treatment (a threshold
calibrated on the model actually being measured, not inherited from Qwen
3.6). `‖V_j‖` is not the missing ingredient for the coverage arm. Per the
brief: **do not rescue this** — the honest reading is that raw alpha and
norm-weighted alpha carry the same information for both citation and
coverage on this model, at this coverage layer, on this corpus.

What *would* be worth learning from this run: the shipped `0.705` constant is
stale for Qwen 3.8 regardless of metric — a same-model coverage-threshold
recalibration is a real, available 4-point coverage gain (87%→91%) that has
nothing to do with norm-weighting. That is a `LensConstants`/architecture
question (per-model constants), not a probe outcome, and is flagged here for
the user's attention rather than acted on.

## 6. Implementation notes — the traps that were real

- **Cache-layer index ≠ GGUF block index.** `simple_kv_cache` is built
  attention-layers-only: `qwen35.cpp` assigns
  `kv_layer_map_[il] = n_attn_layers++` while scanning blocks 0..32 in
  increasing order and skipping non-attention blocks. `attn_layers` (built in
  `main()` by scanning for `kq_soft.<il>` tensors in the same increasing-`il`
  order) therefore has the property that **tap slot index == cache layer
  index** — block 27 (L27H13) is cache layer 6 (its position in
  `attn_layers`), not 27. Passing the block number into
  `get_v_cache_tensor` would have silently read a different layer's V cache
  under the citation head's own printed name. Verified against
  `src/models/qwen35.cpp:144-181` before writing the reader.
- **GQA mapping confirmed, not assumed.** Qwen 3.8 has real GQA (16 query
  heads, 4 KV heads, group size 4) — not the 1:1 case the brief warned could
  be silently wrong. Confirmed the grouping convention from `ggml_mul_mat`'s
  broadcast rule via the actual call site (`src/layers/attention.cpp`:
  `ggml_mul_mat(ctx, k, q)` with `k` as src0 at `ne[2]=n_head_kv` and `q` as
  src1 at `ne[2]=n_head_q`) — ggml's broadcast index is `i_kv = i_q / (n_head_q
  / n_head_kv)`, i.e. **contiguous** grouping (query heads 0..3 → KV head 0,
  4..7 → KV head 1, ...), not interleaved. L27H13 → KV head 13/4 = 3.
- **KV cache dtype was asserted, not assumed.** `value_norms()` throws unless
  the V cache tensor type is F32 or F16, with an explicit note that this is
  guaranteed rather than lucky: quantized KV requires `--flash-attn`
  (`kv_type_requires_flash_refusal`, `kv_cache_simple.h`), and this probe
  requires flash **off** (the tap needs `kq_soft` to materialize), so any run
  that reaches the reader already has F32/F16. In this run it was F32, the
  `create_forward_pass` default.
- **Layout derived from `ne[]`/`nb[]`, checked not trusted.** `value_norms()`
  asserts `ne[0] == n_head_kv * head_dim` and `nb[1] == n_embd_v *
  element_size` (contiguous row stride) against the tensor's own fields before
  reading, rather than assuming the layout `attention.cpp` happens to build
  today.
- **Sanity check on `‖V_j‖` itself.** `sanity_check_norms()` asserts every
  interior populated position has `‖V‖ > 0`; it never fired across either run
  (15 Leg C docs + 23 COV1 spans, all 4 KV heads, both tap layers) — no
  striding bug produced a block of zeros.
- **Renormalization is a no-op for ranking, on purpose.** Metric B's top-1/
  top-3 comparison against Metric A only differs through arithmetic, not
  through a different candidate set — renormalizing a row by a positive
  constant cannot change its argmax or top-3 order. This is why the top-3
  numbers tied exactly (404/413 both) and only top-1 could move at all (a
  near-tie between the 2nd/3rd-ranked candidates flipping order under the
  `‖V‖` reweighting) — expected, not a bug.

## 7. What was NOT done

- **No Gemma.** Explicitly out of scope for this leg; gated on this result,
  which does not clear the bar for carrying norm-weighting into a second
  family.
- **No images.** Both taps are closed (`docs/note-image-lens-probe.md`,
  `docs/note-image-prefill-tap-probe.md`); out of scope by construction.
- **No `‖W_o(alpha_j V_j)‖`** — the output-projection-weighted refinement the
  brief named as the next rung. Not attempted; this rung did not clear its
  own bar convincingly enough to motivate it, though the negative result here
  (renormalization alone does not help) does not rule out that a
  projection-aware version could behave differently — it simply was not
  tested.
- **No coverage-LAYER search under Metric B.** Layer 11 was held fixed
  throughout (per the gate's scope: "pick the best operating point" meant
  threshold, not layer). Whether some other layer separates USED/DROPPED
  better under Metric B than under Metric A was not asked.
- **`LensConstants` was not changed** and the architecture refusal was not
  touched, as instructed. The stale-0.705-on-Qwen-3.8 observation (§5) is
  reported, not acted on — it is an architecture/constants decision for the
  user, not a probe outcome.
- **COV1's separation corpus is small (23 labeled spans)** — both metrics
  tying at AUC=1.000 there is consistent with "no difference" but could also
  be masking a difference too fine for this corpus to resolve. Leg C's 75
  used spans (used in §4.2) is the larger, more informative population, and
  is where the tie was confirmed.
- **No formal significance test** on the citation top-1 gap (372 vs 369 of
  413). Called "noise-level" from the standard-error estimate in §3, not from
  a computed p-value.
