# DeltaNet Batched Decode — Implementation Plan

> ## PARKED 2026-08-31
>
> **By user decision.** The remaining work below is ggml kernel engineering
> that needs a dedicated measurement bench (§7's risks are real engineering,
> not paperwork), and the payoff is server throughput for concurrent users
> that do not exist today. This is not a rejection of the analysis — it is a
> call that the work is not worth doing *right now*. Nothing below is deleted;
> this header sits on top of it per this repo's own correction idiom
> (`decode-gap-status.md` §4/§15).
>
> **What was proven** (measured, stands independent of parking):
> - The mechanism: `DeltaNetLayer::build_decode`'s per-slot chain
>   (§1, §3.1) drives both the O(B) marginal lane cost (0.52–0.63 normalized,
>   vs. 0.25 dense) and the O(B) decode-graph node count.
> - The two located causes (§1): ours (the per-slot loop) and upstream
>   (`MUL_MAT_ID`'s lack of a small-batch kernel below `ne21=32`).
> - The kernel-selection discontinuities, predicted from source before
>   measurement and then structurally confirmed by dispatch census — see
>   `note-batch-scaling-cross-family.md` §4, §11.
>
> **What was disproven — in this very document.** §3.3's node-count formula
> `1260·B − 30` and its predicted **B≈13** crossing are **wrong**. Direct
> census gives the exact formulas (both confirmed for B≥2, and by direct
> build at the crossing itself):
>
> | model | exact formula | confirmed crossing |
> |---|---|---|
> | qwen35moe (30 DeltaNet + 10 attn) | `n_nodes = 1320·B + 2144` | **B=10 builds (15344 nodes, 94% of budget), B=11 aborts** |
> | qwen35 (24 DeltaNet + 8 attn, dense) | `n_nodes = 1056·B + 596` | **B=14 builds (15380 nodes), B=15 aborts** — same defect, previously unrecorded |
>
> The qwen35moe crossing is **one slot of margin** past the declared ≤10-slot
> envelope, not "just outside" it as §3.3 and `architecture.md` §12 used to
> read before this correction. See `note-batch-scaling-cross-family.md` §7 and
> `architecture.md` §12 for the full corrected numbers and mechanism (the
> uniform 44-nodes-per-DeltaNet-layer-per-slot coefficient, exact on both
> models).
>
> **Phase 1 (the extraction) was deliberately NOT done.** No code in this plan
> was written — not even the byte-for-bit extraction that Phase 1 describes as
> low-risk. It is the first step of a road not being walked; doing it now
> would be waste (a change with no consumer, sitting in the tree until someone
> either finishes the plan or reverts the extraction).
>
> **What did land:** the crash this plan's §3.3/§5.1 describes
> (`GGML_ASSERT(cgraph->n_nodes < cgraph->size)`) is now a fail-loud refusal —
> `validate_deltanet_decode_batch_size` in `src/models/qwen35_family.{h,cpp}`,
> called from both hybrid constructors, refuses an over-limit
> `max_batch_size` before any graph is built. That is a safety net, not
> progress on this plan: the O(B) node growth and the O(B) marginal lane cost
> are both exactly as this document found them.
>
> **If ever resumed:** start from the corrected formulas above, not §3.3's.
> The first gap to close is the one the corrected
> `note-batch-scaling-cross-family.md` §11 flags loudest: every kernel-switch
> and slope number in this plan's Amdahl analysis (§2) and gates (§4, §5) was
> measured on Gemma 4 / the two Qwen hybrids at the specific quant levels
> tested — the K-quant and MoE kernel-gating differences mean the above-B=8
> marginal-cost collapse this plan leans on is **not established for Qwen or
> any K-quant model**, and must be re-measured on the actual DeltaNet recipes
> before this plan's §2 Amdahl arithmetic can be trusted again.

---

**Goal (one line):** make `DeltaNetLayer::build_decode` build **one** DeltaNet
chain over `B` lanes instead of `B` chains concatenated, so the Qwen 3.5-family
hybrids stop paying a full per-slot DeltaNet for every extra server slot.

**Status:** scoped, not started. No code has been written for this. Phase 0 is
measurement only and must complete before Phase 1. **PARKED 2026-08-31 — see
the header above.**

**What this is *not*:** a single-user speed-up. B=1 must stay byte-identical
(§4), so CLI decode is exactly as fast after this change as before it. The
entire payoff is server throughput at B ≥ 2.

---

## 1. Why this plan exists

`src/layers/deltanet.cpp:285` — `DeltaNetLayer::build_decode` slices the batch
input per slot, calls the full single-slot builder once per slot, and
`ggml_concat`s the results. The comment at lines 310–315 states the tradeoff
and closes it: *"O(n_slots) dispatches rather than one batched op. Deliberately
left alone."* That was a defensible call when the only consumer was single-slot
CLI decode. The cross-family batch-scaling probe of 2026-08-30 —
[`note-batch-scaling-cross-family.md`](note-batch-scaling-cross-family.md) —
is the new evidence that retires it.

Measured normalized marginal lane cost `b/(a+b)` (what fraction of a B=1 step
one extra lane costs; M1 Pro, Release + Metal, `build-release/bin/batch-scaling`,
2026-08-30):

| model | fit `a + b·B` | `b/(a+b)` | linear-fit ceiling |
|---|---|---:|---:|
| Gemma 4-12B-it Q8_0 (dense) | 62.4 + 21.2·B | **0.25** | 3.94× |
| Qwen3.5-9B Q4_K_M (DeltaNet, no MoE) | 26.6 + 29.3·B | **0.52** | 1.91× |
| Qwen3.6-35B-A3B Q2_K_XL (DeltaNet + MoE) | 12.3 + 21.0·B | **0.63** | 1.59× |

Above B=8 the dense curve is not linear at all: a lane costs ~2 ms against an
89 ms B=1 step (~0.02 normalized) and Gemma 4 reaches **9.60× tokens/wall at
B=32, still flat**. The curves are staircases sitting on ggml-metal
kernel-selection constants, every step predicted from source before being
measured — see that note's §4.

Two located causes make the hybrids the expensive case, and both are software:

1. **Ours.** The per-slot loop above. It alone takes Qwen3.5 — which has no MoE
   at all — from the dense recipe's 0.25 to 0.52. Everything else in a Qwen 3.5
   decode step is already batched: `build_gated_batched_attention`
   (`src/layers/attention.cpp:603`) issues one `ggml_mul_mat` per projection
   over all `n_batch` rows, and the FFN and output head likewise. **DeltaNet is
   the only per-slot loop left in a hybrid decode graph.**
2. **ggml's.** `ggml_metal_op_mul_mat_id` has exactly two branches —
   `mul_mm_id` if `ne21 >= ne21_mm_id_min` (32,
   `build-*/_deps/ggml-src/ggml/src/ggml-metal/ggml-metal-ops.cpp:2733`), else
   `mul_mv_id`, whose grid is `ne123 = ne20*ne21` with `_ne1` hard-coded to 1
   (same file, lines 2865–2866): one threadgroup grid per *(expert-slot,
   token)* pair, which cannot share an expert weight load between two tokens
   routed to the same expert. `ne21` is the token count, so the **entire
   ≤10-slot envelope sits below 32** and the MoE path never reaches a batched
   kernel in production. That is cause 2, it is upstream, and this plan touches
   it only as a scoped measurement (§6, Phase 0.4).

**A related defect the same loop explains.** `qwen35moe` aborts at B=16 with
`GGML_ASSERT(cgraph->n_nodes < cgraph->size)` inside `build_deltanet_layer`,
reached from `Qwen36ForwardPass::build_decoding_graph`. §3.3 works the
arithmetic and predicts the batched build fixes it as a side effect — as a
*prediction Phase 0.1 must confirm*, not an assumption.

---

## 2. Amdahl before code

Per the performance doctrine (`architecture.md` §10) this section comes first
and is allowed to kill the plan. It does not, but it narrows the claim a lot.

### 2.1 Decompose the marginal lane cost

Two independent subtractions on the table above, which agree:

- **Qwen3.5 − Gemma 4 = 0.52 − 0.25 = 0.27.** Qwen3.5 has no MoE; every part of
  its step except DeltaNet runs through the same dense `ggml_mul_mat` staircase
  Gemma 4 enjoys. So 0.27 is DeltaNet's share of the normalized slope.
- **Qwen3.6 − Qwen3.5 = 0.63 − 0.52 = 0.11.** The only structural difference in
  the FFN seam is routed experts, so 0.11 is MoE's share (cause 2 — untouched
  by this plan).
- Residual for Qwen3.6: `0.63 − 0.11 − 0.27 = 0.25` — exactly the dense
  baseline. The decomposition closes on itself. That is a weak consistency
  check, not a proof: it subtracts across three *different models* at three
  *different quant levels* (**OQ-1**).

### 2.2 Split DeltaNet's 0.27 into replicated vs irreducible

The recurrent state update is genuinely O(B): each lane owns its own
`[head_k_dim, head_v_dim] × num_v_heads` state matrix, and there is nothing to
share. Batching converts B dispatches into 1, but not B state updates into 1.
Two estimators bound that irreducible part.

**(a) Bandwidth floor.** Qwen 3.6: 30 DeltaNet layers; per-slot state =
`head_v_dim(128) × head_k_dim(128) × num_v_heads(32)` = 524,288 F32 = **2.00
MiB** — the "524k-element outer product per step" of
[`phase4-investigation.md`](phase4-investigation.md). Per lane per step the
state alone moves: `gated_delta_net` reads S (2 MiB) and writes the state
snapshot into its packed result (2 MiB); the writeback `ggml_cpy`
(`deltanet.cpp:196-199`) reads that (2 MiB) and writes `rec_all` (2 MiB).
8 MiB × 30 layers = **240 MiB/lane/step** ⇒ **~1.26 ms** at 200 GB/s
(**OQ-4**). Against the 33.3 ms B=1 step the Qwen 3.6 fit implies, that is
**0.038 normalized**. Conv state is negligible: `(4−1) × 8192` = 96 KiB/layer,
2.8 MiB/lane/step.

**(b) Measured sub-stage ceiling.** `phase4-investigation.md` records the
post-PR-4.2.B `state_update` stage delta at **139 µs/layer** at B=1 — the whole
stage: `deltanet_pre_state` + `gated_delta_net` + the state `cpy`. 30 × 139 µs
= **4.2 ms/lane** = **0.125 normalized**. This over-states the irreducible
part (it contains the pre_state fusion and all of the stage's dispatch
overhead, both of which amortize) and it comes from a serializing
microbenchmark on the pre-flash 50.5 ms baseline, so it is not strictly
commensurable with the 33.3 ms step.

⇒ **Irreducible DeltaNet share: 0.04 – 0.13.** ⇒ **Amortizable: 0.14 – 0.23.**

**The irreducible part does not dominate.** That is the answer to the kill
question: 0.04–0.13 against a total DeltaNet share of 0.27 and a total slope of
0.63. Build it.

### 2.3 The falsifiable prediction

| | now | predicted after | ceiling now → after |
|---|---:|---:|---|
| Qwen3.6-35B-A3B Q2_K_XL | 0.63 | **0.40 – 0.49** | 1.59× → **2.0 – 2.5×** |
| Qwen3.5-9B Q4_K_M | 0.52 | **0.29 – 0.38** | 1.91× → **2.6 – 3.4×** |

In absolute terms on the probe's own axis (Qwen 3.6, `12.3 + 21.0·B`, B=1 step
33.3 ms): saving 0.14–0.23 × 33.3 = **4.7 – 7.7 ms per lane** ⇒ `b`: 21.0 →
13.3–16.3 ms.

| B | ms/step now | predicted | tokens/wall gain |
|---:|---:|---:|---:|
| 4 | 96.3 | 65.5 – 77.5 | **1.24 – 1.47×** |
| 8 | 180.3 | 118.7 – 142.7 | **1.26 – 1.52×** |
| 10 | 222.3 | 145.3 – 175.3 | **1.27 – 1.53×** |

**Now discount it.** [`plan-deltanet-fusion.md`](plan-deltanet-fusion.md)
predicted **1.46×** end-to-end and delivered **+7%** (+2–4% from 4.2.A, +5.5%
from 4.2.B). [`note-decode-overhead-probes.md`](note-decode-overhead-probes.md)
§3 records that estimates in that thread ran **1.2–3× optimistic**. Applying
that discount to the excess over 1.0 gives **1.09× – 1.43×** at B=8. The
stop-early line in §6 is set from the bottom of that range.

**Why this is not the fusion precedent repeating.** The fusion attacked a
µs-scale dispatch-overhead quantity on a ±25% instrument, and deferred 4.2.C on
exactly that instrument's inability to see the effect. This attacks an
**O(B) → O(1)** change in dispatch count and shared-weight re-reads, and the
repo has since built a thermally-immune instrument it never pointed at
DeltaNet: the `GGML_METAL_GRAPH_DEBUG=1` dispatch census
([`decode-gap-status.md`](decode-gap-status.md) §8). The census yields **exact
integer predictions** (§5.2) that can be checked before any stopwatch is
started. It is the primary signal here; wall-clock is the second, per "two
signals or it didn't happen."

---

## 3. The design

### 3.1 Parameterize `n_seqs` — the seam ggml already has

`build_deltanet_layer` hardcodes `const int64_t n_seqs = 1;`
(`deltanet.cpp:52`). Everything downstream of it is already written
generically in `(n_seq_tokens, n_seqs)`: the QKV/beta/alpha reshapes, the Q/K/V
strided views out of `ssm_conv`'s output, the `gated_delta_net` result's output
and state views. And the ggml ops themselves are natively batched over
sequences:

- `ggml_gated_delta_net` takes `state: [S_v, S_v, H, n_seqs]`
  (`ggml/include/ggml.h:2584`) and its **Metal grid is
  `(S_v/nsg, H, n_seqs)`** (`ggml-metal-ops.cpp:1896`) — lanes are already a
  parallel axis of the kernel, not a loop inside it.
- `ggml_ssm_conv` takes `sx: [d_conv-1+n_t, d_inner, n_s]` (`ggml.c:5575`).

So the change is: **`n_seqs` becomes a parameter**, and the two state views
span a run of slot columns instead of one. Per CLAUDE.md's parameterize-vs-split
test, this is flagged explicitly as a **parameterize** call, and it is the easy
side of that judgment: `n_seqs` is one axis of one operation, it is the axis
ggml's own ops carry, and no call site gets a knob it does not use.

### 3.2 Contiguous slot runs, not gather/scatter

`DeltaNetState` stores each layer's state as `[rec_slot_floats, n_slots]` and
`[conv_slot_floats, n_slots]` — slot-major, contiguous. A run of slots
`[s, s+n)` is therefore a **contiguous view**, reshapeable to exactly the
`[S_v, S_v, H, n]` and `[k-1, C, n]` shapes the ops want. No copy, no
`ggml_cont` (§3 of `architecture.md`: views are free, `cont` is not).

The server's active set is a `std::set<int>` (`inference_server.h:1190`), so
slots arrive ascending but may be fragmented (`{0, 2, 5}` after slots 1, 3, 4
complete). `build_decode` therefore **partitions `args.slots` into maximal
ascending contiguous runs**, builds one batched chain per run, and concatenates
the run outputs — which reassemble in input-row order precisely because the set
is ascending. Best case (`{0..B-1}`, which is what both the probe and a
freshly-loaded server produce) is one chain. Worst case degrades exactly to
today's behaviour, which is the right failure mode.

The alternative — `ggml_get_rows` on `rec_all` / `ggml_set_rows` back
(`ggml.c:3941`, already the KV write mechanism under `--persistent-graph`) —
is general but costs **+4 MiB per lane per layer** of state traffic (+120
MiB/lane/step on Qwen 3.6, ≈ +0.6 ms/lane), a third to a half of the whole
predicted saving. It is deferred to Phase 3 and gated on measuring how often
fragmentation actually happens (**OQ-6**).

**§9 is not touched.** The state tensors' layout, ownership and *overwrite*
semantics are unchanged: each lane still reads and writes exactly its own
column. Only the *view* spans several columns, and only in the compute. Nothing
here unifies the recurrent state with the KV cache's append semantics, and
nothing may.

### 3.3 The B=16 abort, and why this should fix it

`FP_GRAPH_SIZE = 16384` (`src/models/graph_arena.h:46`) is the `size` in the
failing assert. Counting the tensor-producing `ggml_*` calls in
`build_deltanet_layer` (views, reshapes and transposes all create nodes) gives
**40 per call**, plus the per-slot `ggml_view_2d` of the batch input in
`build_decode` = **41 per layer per slot**. With 30 DeltaNet layers:

- **now:** `30 · (41·B + (B−1)) = 1260·B − 30` ⇒ crosses 16384 at **B ≈ 13.0**,
  consistent with the observed "overflows somewhere in 8 < B < 16".
- **cross-check at B=1:** 1230 DeltaNet nodes against the ~2400-node decode
  graph recorded in `note-decode-overhead-probes.md` — DeltaNet is about half
  the graph, so the per-layer count of 41 is not wildly wrong.
- **batched:** `30 · 41 ≈ 1230`, constant in B. The remaining O(B) node sources
  are small — the `cpy`-mode KV write loop in `build_gated_batched_attention`
  (`attention.cpp:711-720`, 2 nodes × 10 attention layers × B). Predicted total
  ≈ `2400 + ~20·B`, under 16384 well past B=32.

**This is arithmetic, not a measurement.** Phase 0.1 confirms it for free by
printing `gf->n_nodes` at several B. If the observed fit is not ≈ `1260·B + c`,
the count above is wrong and the side-effect claim is withdrawn.

> **RETRACTED 2026-08-31 — the observed fit was not ≈ `1260·B + c`, exactly as
> this paragraph's own escape clause anticipated.** Phase 0.1 (§6) was
> eventually run: the exact census is `n_nodes = 1320·B + 2144` on qwen35moe
> (per-layer coefficient 44, not 41 — the hand count under-counted) and
> `n_nodes = 1056·B + 596` on qwen35 (previously unmeasured). Confirmed
> crossings are **B=11 on qwen35moe** (one slot past the ≤10 envelope, not
> B≈13) and **B=15 on qwen35** (a config this section did not cover at all).
> See the PARKED header at the top of this document and
> `note-batch-scaling-cross-family.md` §7, §11 for the full numbers. The
> qualitative argument here — DeltaNet's per-slot chain drives O(B) node
> growth — was right; only the coefficients were wrong.

### 3.4 Files that change

| file | change |
|---|---|
| `src/layers/deltanet.h` | `build_deltanet_layer` gains `n_seqs`; `slot_idx` becomes the run's first slot. `DeltaNetLayer`'s public surface is unchanged. |
| `src/layers/deltanet.cpp` | `n_seqs` parameterized; state/conv views span the run; `build_decode` partitions into runs. |
| `tests/unit/test_deltanet.cpp` | multi-slot fixture (`N_SLOTS > 1`) + the new gates (§4). |
| `tests/unit/test_qwen35_batched_decode.cpp` | **new** — the two-tier differential for `Qwen35ForwardPass` / `Qwen36ForwardPass`. |
| `tests/unit/batched_decode_differential.h` | **new** — the tier-1/tier-2 harness extracted from `test_gemma_batched_decode.cpp`, which then includes it. The harness is already fully FP-generic (it touches only `run_prefill` / `build_decoding_graph` / `set_decode_inputs` / `get_output_logits_for_slot`), so this is a move, not a rewrite. |
| `tests/CMakeLists.txt` | wire the new test target. |

**Files that must NOT change** — and this is a check, not a hope:
`src/models/qwen35_family.{h,cpp}`, `qwen35.cpp`, `qwen36.cpp`,
`forward_pass_base.*`, `decode_plan`, `decode_step`, anything under
`src/graph_inputs/`, `src/state/`, `src/server/`, and every Gemma recipe. If
any of them needs an edit, that is an interface defect under CLAUDE.md's
pressure test — pause and fix the interface, do not push through. Needing zero
of them **is** the success metric.

> **Note 2026-08-31 — this constraint is about the batching extraction
> (Phases 1–2, never started), not about these files being frozen forever.**
> `qwen35_family.{h,cpp}`, `qwen35.cpp` and `qwen36.cpp` did change on
> 2026-08-31, to add the fail-loud slot-count guard
> (`validate_deltanet_decode_batch_size`) that converts this section's
> `GGML_ASSERT` abort into a named refusal. That is a validation added at the
> constructor boundary, not a change to `build_decoding_graph`'s layer body or
> the DeltaNet call pattern this constraint protects — it does not compete
> with or presuppose Phase 1/2's design. If this plan is ever resumed, the
> guard's limit computation should move to (or be superseded by) whatever
> Phase 2 actually builds, since a batched DeltaNet chain would no longer need
> it in this form.

---

## 4. Gates

Follows `architecture.md` §11 exactly. No new gate species is invented here.

**B=1 is byte-identical. Non-negotiable.** It is the path §11(c)'s receipts
claims rest on (byte-replay holds at B=1; a batched generation was never
replayable and this plan does not change that). `build_decode`'s existing
`if (n_batch == 1)` early return is kept **verbatim** so the B=1 graph is
textually the same graph as today. Gate: `logits-dump` byte-compare across the
commit on Qwen3.5-9B-Q4_K_M **and** Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.

**B>1 cannot be byte-identical and is not promised.** This is a batch-shape
transform and Metal picks kernels by batch size — §11's "…except where the
hardware forbids it". The standard gate is **token-stable + loose logit
ceiling**, with the strict bitwise test **kept but `DISABLED_` and
documented**. That is the `--kv-f16` / `--persistent-graph` / flash-attention /
Gemma-batched-decode precedent, and it is followed, not extended.

| gate | where | tier |
|---|---|---|
| B=1 byte-identical across the commit | `logits-dump`, both hybrids | **blocking** |
| batched-vs-looped per-lane equivalence, CPU backend | `tests/unit/test_deltanet.cpp` (new) | **blocking** |
| decode graph node count independent of B | `tests/unit/test_deltanet.cpp` (new) | blocking |
| Tier-1 single-slot **bitwise** (decode graph ≡ single-token prefill) | `test_qwen35_batched_decode.cpp` (new) | blocking |
| Tier-2 multi-slot **token-stable** + `kDecodeMaxAbsDiff` | same | blocking |
| Tier-2 multi-slot **strict bitwise** | same, `DISABLED_` | recorded |
| Tier-2 with **fragmented slots `{0, 2}`** | same | blocking |
| Gemma 4-12B-it Q8_0 byte-unchanged | `logits-dump` | blocking (§5) |
| Gemma 1–4 batched-decode suite still green | `test_gemma_batched_decode.cpp` | blocking |
| dispatch census hits the predicted integers | `GGML_METAL_GRAPH_DEBUG=1` | blocking |
| `b/(a+b)` improves per §2.3 | `batch-scaling`, both hybrids | stop-early |

**Existing coverage that already applies:** `tests/unit/test_deltanet.cpp`
(CPU, model-free, shapes + finiteness — currently `N_SLOTS = 1`, so it does not
cover multi-slot at all today), `test_deltanet_state.cpp`,
`test_deltanet_pre_state.cpp` / `test_deltanet_post_state.cpp` (the two fused
ops' equivalence, unchanged by this plan), `test_qwen35_family.cpp`,
`test_qwen35_feed_tokens.cpp`.

**On the CPU gate.** The hypothesis is that the CPU backend is **bitwise**
across batch shape — a CPU `mul_mat` output element's reduction order does not
depend on the column count — which would make the batched-vs-looped comparison
a strict equality on CPU while Metal stays token-stable. That hypothesis is not
established anywhere in the repo (**OQ-5**). If it turns out false, record it as
a finding and fall back to a tolerance; do not quietly loosen the gate and move
on.

Co-location holds: `src/layers/deltanet.cpp` ⇒ `tests/unit/test_deltanet.cpp`,
always. The recipe-level differential is an **aspect** test
(`architecture.md` §4), which is why it is `test_qwen35_batched_decode.cpp` and
not a per-recipe file.

---

## 5. The cross-family question, answered

CLAUDE.md requires any forward-pass change to be grounded against a Qwen *and*
a Gemma recipe. **DeltaNet does not exist on Gemma, and never will.** Waving at
that would be dishonest, so here is the concrete resolution.

### 5.1 What is Qwen-only, and what is shared

The *module* is Qwen-only: `layers/deltanet` is reached only from
`build_qwen35_layer_decode` (`qwen35_family.cpp:214-226`). The Gemma leg is
therefore not "port it to Gemma" — it is **proof that Gemma is untouched**,
which is a real obligation because the failure mode this rule exists to catch
is a Qwen-shaped edit leaking into shared machinery.

Concretely, the Gemma leg is:

- **`logits-dump` byte-compare on `models/gemma-4-12B-it-Q8_0.gguf`** across the
  commit, at B=1 and B=4. This model, not another: it is the exact model the
  cross-family probe measured at 0.25, so its curve is directly comparable.
  A **null result is the deliverable.**
- **`batch-scaling` on the same model**, B ∈ {1,2,4,8,16,32}, confirming the
  0.25 slope and the ~2 ms/lane plateau above B=8 are unmoved.
- **The full `test_gemma_batched_decode.cpp` suite green** for Gemma 1, 2, 3
  and 4 — Tier-1 bitwise and Tier-2 token-stable, all four recipes. Cheap
  second leg for local iteration: `gemma-3-1b-it-BF16.gguf`.

### 5.2 The success metric, and how it is checked

CLAUDE.md's pressure test: *"the architecture hosted X without bending"* — zero
logic edits to modules other recipes depend on. §3.4's "must NOT change" list
**is** that metric, made checkable: if the diff touches nothing outside
`src/layers/deltanet.{h,cpp}` and `tests/`, the interface was right. If it
does, pause.

The structural signal that the change actually happened, as exact integers per
decode step on Qwen 3.6 (30 DeltaNet layers), from the
`GGML_METAL_GRAPH_DEBUG=1` census:

| op | now | after |
|---|---|---|
| `MUL_MAT` on the DeltaNet path (qkv, z, beta, alpha, out) | 5·30·B | 5·30 |
| `GATED_DELTA_NET` | 30·B | 30 |
| `SSM_CONV` | 30·B | 30 |
| `DELTANET_PRE_STATE`, `DELTANET_POST_STATE` | 30·B each | 30 each |
| `CONCAT` | 30·B + 30·(B−1) | 30 |

Any of these failing to move — or **disappearing from the Metal log entirely**,
which is what a silent CPU fallback looks like (§3 of `architecture.md`) — is a
blocking finding.

---

## 6. Phasing

Blueprint rule, strictly: **extract first (bit-for-bit identical), stabilize,
then optimize. Never combine extraction with optimization in one step.** Each
phase is independently mergeable with its own gate.

### Phase 0 — measure. No source change.

**0.1 Node-count census.** Print `gf->n_nodes` from `build_decoding_graph` at
B ∈ {1,2,4,8,10} on Qwen3.5-9B and Qwen3.6-35B-A3B; fit. Confirms or refutes
§3.3's `1260·B + c` and the predicted B=13 crossing. Free.

> **Run 2026-08-31 — refuted, as this step exists to check.** Exact fits are
> `1320·B + 2144` (qwen35moe) and `1056·B + 596` (qwen35, not measured before
> now), crossing at B=11 and B=15 respectively, not B≈13. See the PARKED
> header.

**0.2 Dispatch census baseline.** `GGML_METAL_GRAPH_DEBUG=1` op histogram for
one decode step at B=1 and B=8 on Qwen 3.6, by the two-run differencing method
of `decode-gap-status.md` §8. Records the integers §5.2 must move. Free.

**0.3 The real server loop, not the probe.** Every number in §1 and §2 comes
from `tests/perf/batch-scaling`, which **never reads logits and never samples**
(`batch_scaling.cpp` runs `build → alloc → set → compute` and stops).
`run_batched_decode` (`src/server/http_server.cpp:702-745`) does both, **once
per lane**: `get_output_logits_for_slot` copies a full vocab of floats
device→host per lane, then a full-vocab CPU `sample()` per lane. On Qwen 3.5's
242.5K vocab that is ~1 MB of readback plus a 242.5K-element scan per lane —
genuinely O(B) work the probe does not contain at all
(`note-decode-overhead-probes.md` prices B=1 logits readback alone at 1.56 ms).
`decode-gap-status.md` §1 carries the standing warning that the probe loop and
the product loop have differed by 2× before (55.8 vs 31 tok/s, the sampler
defect). **State which loop any number came from.**

Reproduce the B-sweep against a live `qwenium-server`: N concurrent
`/v1/completions` at fixed `max_tokens`, aggregate tok/s, fit `a + b·B`, report
`b/(a+b)`. **Nobody may change the server's slot count on the strength of the
cross-family note's §7 ("B=5 and B=16 beat B=8") until this exists.** Expected
direction: the server's normalized slope is *higher* than the probe's. If it
comes back materially higher, §2.3's predictions are over-stated in proportion
and the gates below must be re-priced before Phase 1.

**0.4 `ne21_mm_id_min: 32 → 8` — a measurement, and an approval item.** One
constant at `ggml-metal-ops.cpp:2733` gates whether the MoE path can reach
`mul_mm_id` inside our envelope. Changing it means **a new file in
`patches/`**, which is a named seam in `architecture.md` §3 — so it **requires
user approval before the file is created**, per the Architecture Doc Protocol.
Scope it strictly as a measurement: patch, rebuild, `batch-scaling` on
Qwen 3.6 at B ∈ {1,2,4,8,10}, revert if it does not help. **It may well be
slower.** The constant is presumably tuned, and `mul_mm_id` pays a
`mul_mm_id_map0` pass plus two extra intermediate buffers
(`ggml-metal-ops.cpp:2745-2779`) that a 10-token batch may not amortize
(**OQ-7**). Independent of Phases 1–3; may run in any order, but must not share
a commit with them.

### Phase 1 — extraction, bit-for-bit

`build_deltanet_layer` gains an explicit `n_seqs`; the recurrent and conv views
span `[slot_base, slot_base + n_seqs)`. **Every call site passes
`n_seqs = 1`.** No batching, no behaviour change.

Gate: B=1 byte-identical on both hybrids; Gemma 4 byte-unchanged (it cannot be
affected — that is the point); `test_deltanet`, `test_qwen35_family`,
`test_deltanet_state` green.

### Phase 2 — batch the compute

`build_decode` partitions into contiguous runs and emits one chain per run.
`n_batch == 1` keeps its early return verbatim.

Gates: the full §4 table. **Stop-early condition: if the Phase-0.2 census
re-run hits its predicted integers but B=8 tokens/wall on Qwen 3.6 improves by
< 1.15×, stop and write the null up.** Do not proceed to Phase 3 hoping the
effects compose — they only compose if each is real in isolation
(`plan-deltanet-fusion.md`'s falsification rule, which held).

### Phase 3 — remove the contiguity condition, only if measured to matter

Phase 2 degrades to today's behaviour on a fragmented slot set. Whether that
matters is empirical, not architectural, so Phase 3 **opens with a
measurement**: log the run count per decode step under realistic concurrent
load. If runs are almost always 1, close as no-action. If fragmentation is
common, the fix is the `get_rows`/`set_rows` gather-scatter of §3.2, whose
+0.6 ms/lane cost must be weighed against the measured fragmentation rate. A
cheaper alternative to price at that point — compacting the server's active
slot set — touches `src/server/` and warm-KV retention and is a separate
decision, not a silent extension of this one.

### Phase 4 — settle the docs

The batched path is not a flag; the loop was never a feature. What does need a
decision is whether `architecture.md` §10 gains a measured number and whether
§12 gains or loses a bullet (the B=16 abort). Per the Architecture Doc
Protocol, surfaced to the user before it lands.

### Stop-early conditions, collected

1. **After 0.3:** if the server loop's `b/(a+b)` on Qwen 3.6 is ≥ 0.80, the
   DeltaNet slice is no longer the dominant lane cost and this plan should be
   re-scoped around the per-lane logits readback instead.
2. **After Phase 2:** the < 1.15× rule above.
3. **At any point:** if the design needs an edit to any file on §3.4's "must
   NOT change" list.

---

## 7. Risks, and what would kill it

1. **The fusion precedent.** `plan-deltanet-fusion.md` predicted **1.46×** and
   delivered **+7%**; its 4.2.C stage was deferred because the ±25% per-step
   instrument could not see the effect. The mitigation is structural, not
   optimistic: the census (§5.2) is exact, thermally immune, and comes *first*.
   If the integers do not move, nothing was built and no stopwatch is needed.
2. **Silent CPU fallback** — the classic ggml trap (`architecture.md` §3). Three
   op shapes are new at `n_seqs = B`: `CONCAT` on a transposed `[1, C, B]` src,
   `SSM_CONV` at `n_s = B`, and our two patched ops. `GATED_DELTA_NET`'s Metal
   gate is `has_simdgroup_reduction && src[2]->ne[0] % 32 == 0`
   (`ggml-metal-device.m:1386`) — independent of `n_seqs` — and its grid is
   already lane-parallel, so it is safe. `DELTANET_POST_STATE`'s gate requires
   `ggml_is_contiguous` on all three sources (same file, 1388–1397); the
   batched `output` view and the `z_4d` reshape *are* contiguous at
   `n_tokens = 1`, but that must be **verified by census, not by reading**.
   ggml offers no "must run on Metal" assertion, so the census is the guard and
   it is a required gate.
3. **`ggml_concat` on a strided source at `n_seqs > 1`.** Already exercised at
   `n_seqs = 1` (`deltanet.cpp:94-95`), and `ggml_transpose` of `[C, 1, B]`
   yields exactly the `[1, C, B]` the batched `ssm_conv` wants. If Metal's
   CONCAT refuses the strided src at B>1, the fallback is one `ggml_cont` of
   the transposed qkv — 32 KiB × B, trivial against the 128 KiB × B the concat
   materializes anyway. Named so it is a known branch, not a surprise.
4. **The §2.1 decomposition subtracts across three different models at three
   different quants** (**OQ-1**). If DeltaNet's real share is below 0.27 the
   payoff shrinks proportionally. Partial mitigation is free: running the probe
   on **Qwen3.5-9B-Q8_0 vs Qwen3.5-9B-Q4_K_M** isolates the K-quant staircase
   term (`mul_mv_ext` engages at `ne11 ∈ [4,8]` for K-quants vs `[2,8]`
   otherwise, `ggml-metal-ops.cpp:2493`) from the DeltaNet term, on one model.
5. **Measurement hygiene.** The source measurement was taken on a loaded machine
   (load ~6, 18 GB swap — cross-family note §8). Absolute ms are not quotable;
   the dimensionless ratio and the discontinuity positions are. Every
   re-measure in this plan reports `b/(a+b)`, not ms.
6. **B>1 will never be byte-identical.** Do not accept a review request for it.
   §11's "…except where the hardware forbids it" governs; the strict test stays
   `DISABLED_`.
7. **Receipts.** §11(c): receipts-grade determinism is per-config **and** B=1.
   Nothing here makes a batched generation replayable, and the plan must not be
   read as claiming so. The B=1 byte gate is what keeps that claim true.

**What would kill it:**

- Phase 0.3 showing the server loop is readback-bound rather than
  DeltaNet-bound.
- Phase 2's census moving as predicted while wall-clock does not — which would
  mean estimator (b) in §2.2 was right and (a) wrong, the irreducible share is
  0.13 rather than 0.04, and there is no Phase 3 worth having.
- Per-lane divergence above `kDecodeMaxAbsDiff` at B>1 — that would mean the
  batched recurrence is not lane-independent, which is a **correctness** kill,
  not a performance one.

---

## 8. Open questions this plan could not close from the repo

- **OQ-1.** The §2.1 decomposition subtracts across three different models at
  three different quant levels. It assumes the non-DeltaNet, non-MoE part of a
  step normalizes to the same 0.25 on all three. Bound-shaped, not an
  attribution. Partial closure route in Risk 4.
- **OQ-2.** Qwen3.5-9B's DeltaNet dimensions (DeltaNet layer count,
  `head_v_dim`, `head_k_dim`, `num_v_heads`) are recorded nowhere in `docs/`,
  so only Qwen 3.6's state-traffic floor could be computed. Cheap to close:
  read the GGUF metadata.
- **OQ-3.** Per-layer decode attention cost on Qwen 3.6 is still "TBD" in
  `phase4-investigation.md` (its Q2, deferred since 2026-05). Without it the
  residual 0.25 in §2.1 cannot be split between attention / head / KV staging.
- **OQ-4.** Achieved memory bandwidth for the state ops on M1 Pro. §2.2(a) uses
  200 GB/s; `decode-gap-status.md` §3 records 179 GB/s achieved on `lm_head`
  and 185 GB/s on `qkv_proj` in the real shape mix. At 179 the floor rises
  1.26 → 1.41 ms — inside the stated range, so the conclusion holds, but the
  figure is an assumption, not a measurement of these ops.
- **OQ-5.** Whether the CPU backend is bitwise across batch shape (the Phase-2
  CPU gate's hypothesis). Not established in the repo;
  `test_deltanet_pre_state` reports "CPU bit-exact" but for a different
  question.
- **OQ-6.** How often the server's active slot set is actually fragmented. No
  telemetry exists. Phase 3 opens by measuring it.
- **OQ-7.** Whether `mul_mm_id` at `ne21 ∈ [8, 32)` beats `mul_mv_id` on M1
  Pro, and whether upstream's 32 was tuned on Apple silicon at all. Exactly
  what Phase 0.4 measures.
- **OQ-8.** The 41-nodes-per-DeltaNet-layer figure in §3.3 is a hand count from
  the source. Phase 0.1 replaces it with a measurement.

---

## 9. Related, and what this does not overlap

- [`note-batch-scaling-cross-family.md`](note-batch-scaling-cross-family.md) —
  the measurement that motivates this plan. Its §10 explicitly leaves this work
  to be scoped here.
- [`plan-deltanet-fusion.md`](plan-deltanet-fusion.md) — the prior DeltaNet
  work: 1.46× predicted, +7% delivered, 4.2.C deferred on instrument limits.
  The cautionary precedent §2.3 discounts against. Its two shipped kernels
  (`patches/0002`–`0005`) are untouched here; batching changes how many times
  they are dispatched, not what they compute.
- [`plan-persistent-decode-graph.md`](plan-persistent-decode-graph.md) — attacks
  the **`a`** term (the ~12 ms galloc replan). This plan attacks **`b`**. They
  compose and neither substitutes for the other. Note the server path still
  rebuilds and reallocs every step (`http_server.cpp:719-720`) — that plan's P4
  is unbuilt. One positive interaction: with a batched DeltaNet the decode
  graph's node count stops exploding with B, which makes a
  keyed-by-active-slot-set persistent graph materially more tractable. The
  graph *shape* still depends on the slot set, so keying is still required.
- [`plan-gemma-batched-decode.md`](plan-gemma-batched-decode.md) — the source of
  the two-tier differential harness this plan reuses, and of the
  cross-family-in-reverse framing §5 follows.
- [`phase4-investigation.md`](phase4-investigation.md) — the sub-stage
  attribution §2.2(b) rests on, and the MoE ≤1.13× ceiling that keeps cause 2
  out of scope beyond Phase 0.4's one-constant measurement.
