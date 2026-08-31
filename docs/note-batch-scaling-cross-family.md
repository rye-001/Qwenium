# Batch scaling across three families — the ceiling is a kernel, not a law (2026-08-30)

**P1. One probe (`batch-scaling`), three models, one falsification.** The
inherited `ms/step ≈ 20 + 25.6·B` fit and its "**1.76× amortization ceiling**"
were measured on Qwen 3.6 alone in 2026-07 and have been cited ever since as a
property of the engine. They are not. They are a property of *that recipe on
those ggml-metal kernels*. On a dense Gemma 4 the same probe reaches
**9.60× tokens/wall at B=32 and is still flat**.

Two claims come out of this session and they point in opposite directions:

1. **The shape reproduces on Qwen 3.6; the numbers do not.** Re-fit today:
   `12.3 + 21.0·B` against the inherited `21.7 + 25.4·B`. Slope −18%,
   intercept −43% (fixed overhead fell as persistent-graph and flash landed).
   Net, the **normalized** slope got *worse* — 0.54 → 0.63 — and the ceiling
   fell **1.85× → 1.59×**. On Qwen 3.6 alone the pessimistic reading is now
   *slightly more pessimistic*. This correction is not a softening.
2. **The generalization was never valid**, and not because the reasoning was
   sloppy: the probe was *structurally incapable* of the comparison its number
   was cited for. It opened with
   `if (meta.architecture != "qwen35moe") return 1;`. Every cross-family use of
   "1.76×" rested on a measurement that could not have been taken.

Provenance for everything below: `build-release/bin/batch-scaling`, M1 Pro,
Release + Metal, **2026-08-30**.

---

## 1. Conditions (identical across all three models)

- **CTX = 1024**, prompt = **8 synthetic token ids** `(i*7+3)%1000` —
  tokenizer-free, so prompt length is *exactly* equal across families rather
  than approximately equal. This is the change that made the comparison
  possible at all.
- **6 warmup + 24 timed steps per cell**, **5 passes with B rotated within each
  pass**, medians reported.
- Models run **strictly sequentially, one process at a time** (a 12 GB-class
  model and a 35B MoE do not co-reside on 26.8 GB of working set).

## 2. The three curves, ms/step

| B | Gemma 4-12B-it Q8_0 | Qwen3.5-9B Q4_K_M | Qwen3.6-35B-A3B Q2_K_XL |
|---|---|---|---|
| 1 | 90.3 | 50.2 | 32.0 |
| 2 | 109.3 | 96.6 | 56.2 |
| 4 | 128.6 | 136.7 | 95.3 |
| 8 | 239.3 | 262.3 | 180.0 |
| **run-to-run spread** | ≤2.6% | ≤1.8% | ≤5.7% |

### Fits

| model | `a + b·B` | R² | max resid |
|---|---|---|---|
| Gemma 4-12B-it Q8_0 | **62.4 + 21.2·B** | 0.965 | ±19 ms |
| Qwen3.5-9B Q4_K_M | **26.6 + 29.3·B** | 0.991 | ±11 ms |
| Qwen3.6-35B-A3B Q2_K_XL | **12.3 + 21.0·B** | 0.9995 | ±2 ms |
| *inherited 2026-07 (Qwen3.6)* | *21.7 + 25.4·B* | *0.9997* | *±2 ms* |

Note the R² column is itself a finding. Qwen 3.6 fits a straight line at
0.9995; Gemma 4 fits at 0.965 with ±19 ms of residual. **The Gemma residual is
not noise — it is structure**, and §4 names each piece of it. A linear fit is
the wrong model for the dense curve, which is exactly why the inherited
single-model linear fit generalized so badly.

### Normalized marginal lane cost `b/(a+b)` — the comparable number

Absolute milliseconds are not comparable across a 12B dense Q8_0, a 9B hybrid
Q4_K_M and a 35B MoE Q2_K_XL. The dimensionless ratio is: what fraction of a
B=1 step does one extra lane cost?

| Gemma 4 (dense) | Qwen3.5 (DeltaNet) | Qwen3.6 (DeltaNet+MoE) | inherited |
|---|---|---|---|
| **0.25** | **0.52** | **0.63** | *0.54* |
| ceiling **3.94×** | **1.91×** | **1.59×** | *1.85×* |

The spread is 0.25 → 0.63 — a **2.5× spread in the very quantity that was
being quoted as a single engine constant.** It also orders cleanly by how much
of the step runs through `MUL_MAT_ID` and the per-slot DeltaNet loop, which is
§5.

## 3. The Gemma wide sweep — the falsification

| B | 1 | 4 | 8 | 9 | 10 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| ms/step | 89.4 | 126.3 | 235.7 | 248.7 | 249.8 | 255.4 | 266.2 | 281.3 | 297.4 |
| marginal ms/lane | — | 9.4 | 19.1 | 12.9 | **1.1** | 2.8 | 2.7 | 2.1 | **2.0** |
| tok/wall vs serial | 1.00× | 2.83× | 3.03× | 3.23× | 3.58× | 4.20× | 5.37× | 7.63× | **9.60×** |

Above B=8 a lane costs **~2 ms against an 89 ms B=1 step** — an effective
normalized slope of **~0.02**, and it is *still flat at B=32*. The 3.94×
"ceiling" from the linear fit is not a ceiling either; the linear fit is simply
the wrong shape above the `mul_mm` crossover. There is no measured ceiling for
dense batching inside anything this probe reached.

## 4. The curves are staircases, and every step was predicted before it was measured

This is the methodological result and it is worth more than any single number.
Each discontinuity below was **read out of the ggml-metal dispatch source first,
its location predicted, and then found in the measurement** — not spotted in a
plot and rationalized afterwards.

All in `build-release/_deps/ggml-src/ggml/src/ggml-metal/ggml-metal-ops.cpp`:

**(a) `ggml_metal_op_mul_mat` (~line 2493) — the small-batch `mul_mv_ext`
gate.** It engages for `ne11 ∈ [2,8]` on Q8_0/Q4_0/F16/BF16 but only
`ne11 ∈ [4,8]` for K-quants (Q4_K/Q5_K/Q6_K/Q2_K/Q3_K).
⇒ **Predicted:** a discontinuity at B=4 on Q4_K_M, and specifically a place
where a *larger* batch is *absolutely cheaper*.
⇒ **Measured, Qwen3.5:** B=4 is absolutely cheaper than B=3 — **137.4 vs
143.8 ms**, in all 3 passes, 4.5% apart against a 2.2% run-to-run spread.

**(b) `r1ptg` inside that kernel (~line 2526) — src1 rows per threadgroup**, a
hand-tuned table: `2→2, 3→3, 4→4, 5→5, 6→3, 7→4, 8→4`. B≤5 reads the weights
once; at B=6 the table drops to 3 rows/threadgroup, so the weights are read
twice.
⇒ **Predicted:** the 3rd lane is nearly free, and a cliff at B=5→6.
⇒ **Measured, Gemma 4:** B=2 **108.2** → B=3 **106.7** (the 3rd lane is free —
*negative* marginal cost), then **B=5→6 costs +46.3 ms for one lane.**

**(c) `ne11_mm_min = 8` (line 2471)** — the true matrix×matrix kernel `mul_mm`
engages only *above* B=8.
⇒ **Predicted:** marginal cost collapses above 8.
⇒ **Measured:** **~2 ms/lane from B=10 on** (§3).

**(d) `ggml_metal_op_mul_mat_id` (~line 2695) has only two branches** —
`mul_mm_id` if `ne21 >= ne21_mm_id_min` (**32**, line 2733), else `mul_mv_id`.
There is **no `mul_mv_ext` small-batch path for `MUL_MAT_ID` at all** — the
entire (a)/(b) staircase that dense matmul enjoys does not exist for MoE. And
inside `mul_mv_id`, `_ne1` is hard-coded to `1` with `ne123 = ne20*ne21`
(lines 2865–2866): **one threadgroup grid per *(expert-slot, token)* pair.**

## 5. Two located causes, both software

1. **`src/layers/deltanet.cpp:285`** — `DeltaNetLayer::build_decode` loops over
   slots, builds a **full DeltaNet chain per slot**, and `ggml_concat`s the
   outputs. The existing comment (lines 310–311) says it plainly: *"O(n_slots)
   dispatches rather than one batched op. Deliberately left alone."* It was a
   defensible call when the only consumer was single-slot CLI decode. It is the
   reason both DeltaNet recipes sit at 0.52 and 0.63 where the dense recipe
   sits at 0.25.
2. **The `MUL_MAT_ID` grid** in §4(d). Note the consequence: the **entire ≤10-slot
   workload envelope sits below `ne21_mm_id_min = 32`**, so on the MoE path the
   batched kernel *is never once reached in production*. The engine's declared
   operating window is, by construction, the window in which MoE batching has no
   batched kernel available to it.

Neither is a hardware limit. Both are code, one of ours and one upstream.

## 6. The control, resolved — the July null was a kernel limitation, not a law

The 2026-07 note recorded that `QINF_BATCH_IDENTICAL=1` (same token in every
lane ⇒ maximum expert sharing) produced an **identical curve** on Qwen 3.6
(**224.45 vs 224.69 ms at B=8**) and read it as *"NOT routing/weight-read bound
⇒ no salvage from correlated lanes."*

**That reading is wrong.** `mul_mv_id`'s grid is indexed *by token* (§4d), so
it **structurally cannot share an expert weight load between two tokens routed
to the same expert.** Feeding it perfectly correlated lanes cannot produce a
speedup no matter how much sharing is available, because the kernel has no
mechanism to exploit sharing. The null measured the kernel, not the workload.

The control mechanism itself is sound, and that was checked rather than
assumed. **Dense control, Gemma 4, `QINF_BATCH_IDENTICAL=1` at B=8: 246.1 vs
246.9 ms (0.3%)** — a clean no-op, which is exactly what a model with no expert
routing *requires* the control to produce. A control that produced an effect on
Gemma would have invalidated the Qwen null for the opposite reason. It doesn't;
so the Qwen null is interpretable, and its interpretation is §4(d).

## 7. Two incidental findings

- **The ≤10-slot envelope is the worst operating window on dense models.** It
  is past the `mul_mv_ext` sweet spot (B≤5) and below the `mul_mm` crossover
  (B>8). On Gemma 4, **B=5 and B=16 are both materially better than B=8** —
  the number the server actually runs is sitting in the trough between two
  kernel regimes.

  > **CORRECTED 2026-08-31 — the B=5 half of this does not replicate.**
  > Replicated speedup-vs-serial on the same leg: plain B=5 2.89× / B=8 2.96× /
  > B=16 5.16×; server-realistic (per-lane logits readback + argmax) B=5 2.64×
  > / B=8 2.48× / B=16 4.54×. In plain mode B=5 is *not* materially better than
  > B=8 — a wash, arguably a slight edge to B=8. Server-realistic per-lane work
  > does push B=8 below B=5, so B=8 is a genuine local bad spot under that
  > load, but B=5 is not an actionable improvement, only a smaller loss. B=16
  > still robustly beats B=8 in both modes, but it is outside the declared
  > envelope and past the qwen36 DeltaNet abort at B=11 (below). **Do not act
  > on slot-count guidance from this bullet** — see `architecture.md` §1's
  > matching correction.
- **`qwen35moe` aborts at B=16**:
  `GGML_ASSERT(cgraph->n_nodes < cgraph->size)` in `build_deltanet_layer`,
  reached from `Qwen36ForwardPass::build_decoding_graph`. The decode graph's
  node count is **O(B)** on the DeltaNet path (§5.1 is why) and overflows
  somewhere in `8 < B < 16`. Pre-existing, and just outside the declared
  envelope — recorded in `architecture.md` §12.

  > **CORRECTED 2026-08-31.** "Somewhere in `8 < B < 16`" is now exact, and the
  > crossing is one slot past the declared envelope, not "just outside" it:
  > `n_nodes = 1320·B + 2144` on qwen35moe (30 DeltaNet + 10 attention layers),
  > exact for B≥2 — **B=10 builds at 15344 of 16384 nodes (94% of the limit),
  > B=11 aborts.** `qwen35` (24 DeltaNet + 8 attention layers, no MoE) has the
  > **same defect, previously unrecorded**: `n_nodes = 1056·B + 596`, exact for
  > B≥2 — **B=14 builds (15380 nodes), B=15 aborts**, five slots of margin
  > rather than one. Both fits confirm the mechanism at the op level: each
  > DeltaNet layer contributes exactly **44** graph nodes per slot (14.3
  > `VIEW` + 10 `RESHAPE` + 5 `MUL_MAT` + 2 `CONCAT` + ~2.7 `CPY` + one each of
  > `DELTANET_PRE_STATE`, `DELTANET_POST_STATE`, `GATED_DELTA_NET`, `SSM_CONV`,
  > `TRANSPOSE`, `ADD`, `MUL`, `SIGMOID`, `SILU`, `SOFTPLUS`) — identical
  > per-layer coefficient on both models (1320/30 = 1056/24 = 44). `MUL_MAT_ID`
  > stays exactly constant in B (120 nodes on qwen35moe's 40 MoE layers × 3, 0
  > on qwen35), confirming CLAUDE.md's O(1)-in-expert-count claim structurally,
  > not just by prediction. The crash is now a fail-loud refusal
  > (`validate_deltanet_decode_batch_size`, `src/models/qwen35_family.{h,cpp}`,
  > 2026-08-31) rather than a raw `GGML_ASSERT`; the O(B) growth itself is
  > unfixed and the batching plan that would fix it is
  > [PARKED](plan-deltanet-batched-decode.md). See `architecture.md` §12 for
  > the corrected ledger entry.

## 8. Caveats that must survive into the record

- **The machine was not quiet.** Load average ~6 throughout, a
  `Virtualization.framework` VM at ~22% CPU, browsers open, swap 12.8 GB used
  at start / 18.0 GB at end. **This inflates absolute milliseconds across the
  board.** Do not quote the absolute ms/step figures in this note as the
  engine's speed.
- **What the claims actually rest on is drift-invariant:** the
  **dimensionless ratio** `b/(a+b)`, and the **positions of the
  discontinuities**. Neither moves under a uniform inflation of the time axis.
  Observed within-session thermal drift ~3%.
- **Effects called here are 20–100%, i.e. 10–40× the noise floor.** Nothing in
  §2–§6 is a small-margin call. The one number quoted near the floor is the B=3
  Gemma negative-marginal (1.4%), and it is reported as a *direction* confirmed
  by prediction, not as a magnitude.
- **The CTX reduction did not perturb the shape.** CTX=1024 and CTX=256 Gemma
  runs agree at B=8 (**236 vs 236**), so the allocation reduction used to make
  the wide sweeps fit is not a confound.
- **This probe could not have run cross-family before this session** — the
  `qwen35moe` arch gate. Fixed in the same change as this note.

## 9. How to reproduce

```
build-release/bin/batch-scaling
```

Model via `QWEN36_MODEL_PATH` (the name is historical — the probe now accepts
any registered architecture and prints `arch=…` rather than refusing).

Env knobs, all optional:

| knob | meaning | default |
|---|---|---|
| `QINF_BS_BATCHES` | comma list of B values, in the order run | `1,2,4,8,10` |
| `QINF_BS_MAXB` | max slots the forward pass is built for | max of the batch list |
| `QINF_BS_CTX` | KV ctx per slot | `1024` |
| `QINF_BS_WARMUP` | warmup steps per cell | `6` |
| `QINF_BS_TIMED` | timed steps per cell | `24` |
| `QINF_BS_PASSES` | repeats of the whole batch list | `1` |
| `QINF_BS_PROMPT_TOKENS` | N synthetic ids `(i*7+3)%1000`; `0` = historical text prompt | `0` |
| `QINF_BATCH_IDENTICAL` | same token in every lane (the sharing control) | off |

Rotating B is done by varying the **order** of `QINF_BS_BATCHES` between
passes, so no B is permanently first (cold) or last (hottest).

The probe emits a machine-readable line per cell alongside the table:

```
RESULT,<architecture>,<pass>,<B>,<ms_step>
```

Cross-family runs must be **sequential processes**, one model at a time.

## 10. What this does and does not license

- It **does** justify treating `plan-deltanet-batched-decode`-shaped work
  (batching §5.1's per-slot loop) as measured rather than speculative. Owned by
  another worker; not scoped here.
  **Update 2026-08-31: that plan is now [PARKED](plan-deltanet-batched-decode.md)
  by user decision** — the remaining work is ggml kernel engineering needing a
  dedicated measurement bench, for concurrent-user throughput with no present
  demand. The measurement above still stands as what was proven before parking.
- It **does not** license a general "batching is cheap" claim. On the two
  DeltaNet recipes, at the batch sizes the server actually runs, lanes cost
  0.52–0.63 of a step and the ceiling is 1.59–1.91×. The dense 9.60× is a
  *dense, B=32* number and B=32 is outside the declared envelope.
- It **does not** touch the sleep-time / ghost-slot survivor from the July note
  (baseline there is idle hardware, so no amortization ceiling applies).

## 11. Confirmations and one hard caveat (2026-08-31)

A follow-up session re-ran the dense (Gemma 4) leg to turn §4's *predicted*
kernel-selection discontinuities into *observed* ones, and to replicate the
normalized slopes under more realistic per-lane work. All figures below are
measured; none are estimated.

**The kernel switch is structurally confirmed, not just wall-clock.** The
`ggml_metal_op_mul_mat` `mul_mm` path's Metal function,
`kernel_mul_mm_q8_0_f32`, is **absent from the dispatch log at B=1–8 and
present at B=9–16** — a clean binary transition at exactly B=9, matching
`ne11 > ne11_mm_min(8)` (`ggml-metal-ops.cpp:2580`, §4(c)'s prediction). No
smear across the boundary: the kernel is either dispatched every step in a
cell or never.

**The `mul_mv_ext` small-batch table is likewise confirmed at the dispatch
level.** The `r1ptg` pipeline variant switches from `r1_4` to `r1_5` **exactly
at B=4→5** — the same table (§4(b)) read from source and predicted before
being measured, now directly observed in the pipeline name rather than
inferred from a timing cliff.

**The normalized slope replicates, and moves in the predicted direction under
realistic load.** Plain mode (`batch-scaling`'s own build→alloc→set→compute
loop, §6 Phase 0.3's caveat that this is not what the server does):
normalized slope **0.258** (fit `a + b·B` = 61.6 + 21.5·B) against the
inherited 0.25 — replication, not a new number. Adding **server-realistic
per-lane work** — a full-vocabulary logits readback plus argmax per lane,
mirroring what `run_batched_decode` actually does and the probe does not
(§6 Phase 0.3) — raises it to **0.364** (a=50.6, b=29.0), **+41% relative**.
Direction as predicted: per-lane readback is genuinely O(B) work the bare
graph-build probe cannot see, and it costs more, not less, as B grows.

**The caveat that must survive into the record, because it is the exact
failure this investigation exists to correct:** the B=9 kernel-switch verdict
and the 0.364 server-realistic slope were measured on **Gemma 4-12B-it Q8_0
only**. Two structural reasons this does not transfer:

- K-quant kernel gating differs from Q8_0/F16: `mul_mv_ext` engages for
  `ne11 ∈ [4,8]` on K-quants against `[2,8]` otherwise (§4(a)) — a different
  small-batch staircase shape, not just a shifted one.
- The MoE path has no `mul_mm_id` at all below `ne21_mm_id_min = 32` (§4(d)) —
  the entire ≤10-slot envelope sits below it, so the B=9 crossover this
  session confirmed **structurally cannot happen** on the MoE path in this
  envelope.

So **the above-B=8 marginal-cost collapse is NOT established for Qwen or for
any K-quant model**, and **0.364 is a floor for the flagship, not an
estimate of it**: Qwen 3.6's normalized slope (0.63) is roughly a third of
Gemma 4's plain-mode slope (0.25), so a fixed per-lane vocabulary readback
cost should hurt proportionally more there — a fixed absolute readback cost is
a *larger* fraction of a *smaller* step. This is untested, and deliberately
so: closing it is measurement work, and per the parking decision above, no
further measurement is being scoped from this note.

**Machine state, for provenance.** Load average ~10.9 at both start and end
of the session (a heavier background load than the 2026-08-30 session's ~6);
swap 13.87 GB → 18.01 GB over the session. Per §8, absolute milliseconds are
not quotable from a loaded machine — the slopes and discontinuity positions
above are dimensionless / positional and are what is claimed.
