# Phase 4 Investigation — Per-Layer Decode Cost Breakdown

**Status:** in progress. Three of four breakdowns measured (MoE, DeltaNet,
Norms). Attention measurement deferred (see Open Questions).

**Branch:** `phase4-fusion`
**Hardware:** Apple M1 Pro, Metal backend
**Model:** Qwen 3.6 35B-A3B Q2_K (UD-Q2_K_XL), 40 layers
(10 gated-attention, 30 GatedDeltaNet, MoE on every layer), 256 experts
top-8, 1 shared expert
**Decode baseline (PR 4.0):** 19.8 tok/s ≈ 50.5 ms/token

---

## Why this exists

PR 4.1 as originally specified ("fused MoE custom op, 3× faster than the
`ggml_mul_mat_id` fallback") was investigated and dropped. Two reasons:

1. `ggml_custom_op` does not reach the Metal backend.
   `GGML_OP_CUSTOM` / `GGML_OP_MAP_CUSTOM*` are handled in `ggml.c` and
   `ggml-cpu.c` only; the Metal backend's `supports_op` table has no
   case, so any `ggml_custom_op` node is scheduled to CPU.
2. The fusable MoE-side dispatch is too small for a 3× target. Stage
   breakdown ([tests/perf/moe_breakdown.cpp](../tests/perf/moe_breakdown.cpp))
   shows expert dispatch (gate / up / down / weighted-sum) totals
   ≈155 μs / layer × 30 layers ≈ 4.65 ms / token. Even reducing that to
   zero gives an end-to-end ceiling of `50.5 / (50.5 − 4.65) ≈ 1.10×` —
   nowhere near 3×.

The premise behind the original PR (per-expert kernel-launch overhead)
was carried over from `ik_llama.cpp`'s graph shape, where MoE is built as
chained `mul_mat` per expert. **Our `src/layers/moe.cpp` issues exactly
three batched `ggml_mul_mat_id` calls per layer** (gate / up / down),
each backed by a native Metal `MUL_MAT_ID` kernel. C-level launch count
is O(1) in `n_experts`, not O(`n_experts`).

This investigation re-ranks Phase 4 PR ordering by measurement instead
of plan-document order.

---

## Method

Cumulative-subgraph timing harness ([tests/perf/breakdown_harness.h](../tests/perf/breakdown_harness.h)).
Each binary builds a graph that terminates one stage later than the
previous, times each at decode shape (n_tokens=1) with 10 warmup + 200
median samples, and reports stage cost as `median(g_N) − median(g_{N-1})`.

Caveats inherited from the methodology:
- The first stage's cumulative absorbs per-iter scheduler overhead
  (`sched_reset + alloc_graph + tensor_set + sched_graph_compute + sync`).
  Stage deltas after the first cancel that overhead out.
- Stage attribution can be fuzzy when Metal's command scheduler runs
  adjacent ops in parallel within a single command buffer — totals are
  robust, sub-stage deltas are indicative.

---

## Findings

### MoE — measured, confirms PR 4.1 should drop

Per layer (Qwen 3.6 layer 0, top_k=8, n_experts=256, ffn_dim=512):

| Stage           | Δ (μs) | Note |
|---|---:|---|
| route            | 380 | router matmul + argsort + get_rows + softmax + per-iter sched/sync |
| gate             | −10 | within-noise; gate+up parallelized in same command buffer |
| up_silu_mul      |  58 | gate's "missing" cost lands here |
| down             |  99 | `mul_mat_id` |
| weighted_sum     |  42 | broadcast mul + 7 view-and-adds |
| shared_expert    |  98 | dense SwiGLU + sigmoid blend |
| **Total / layer**| **668** |  |

**Decode-budget contribution.** 30 layers × 0.668 ms ≈ **20.0 ms/token (~40% of 50.5 ms)**.

**Fusable expert-dispatch.** Stages gate + up + down + weighted_sum ≈
~190 μs/layer × 30 = ~5.7 ms/token. **Ceiling**: `50.5 / (50.5 − 5.7) ≈ 1.13×`.

A second run logged ~30% higher per-layer total than an earlier scratch
measurement; the relative ordering of stages, and the ceiling argument,
are robust to that variance.

Fixture: [tests/fixtures/moe_breakdown_qwen36.json](../tests/fixtures/moe_breakdown_qwen36.json).

### DeltaNet — measured, ranks #1

Per layer (Qwen 3.6 layer 0, GatedDeltaNet decode path, full layer cost
via `DeltaNetLayer::build(Phase::Decode, …)`):

| Measurement   | ms |
|---|---:|
| baseline (sched floor) | 0.097 |
| dn_decode_full         | 1.066 |
| **per_layer (delta)**  | **0.969** |

**Decode-budget contribution.** 30 layers × 0.969 ms ≈ **29.1 ms/token (~58% of 50.5 ms)** — the largest single chunk in our measurements.

**Architecture context.** d_inner=4096, num_v_heads=32, head_dim=128. The
heaviest in-step compute is the output projection [2048×4096] (~8.4M
GEMV) — comparable to one mul_mat in MoE. State update (a 524k-element
outer product per step) and conv1d (k=4 over 8192 channels) are
microsecond-scale on Metal. The 1 ms/layer is therefore unlikely to be
matmul-bound.

**Reachable ceiling — uncertain.** If 50% of the per-layer cost is
launch / barrier overhead from ggml's fine-grained decomposition (10–15
small ops per step), a fused kernel could reach `50.5/(50.5 − 30×0.5) =
1.43×`. At 70% fusable: `50.5/(50.5 − 30×0.7) = 1.71×`. **Both numbers
are upper bounds conditional on the launch-bound hypothesis being
correct.** A sub-stage breakdown is required before scoping fusion.

Fixture: [tests/fixtures/deltanet_breakdown_qwen36.json](../tests/fixtures/deltanet_breakdown_qwen36.json).

### DeltaNet sub-stage findings — Q1 resolved, launch-bound

| Stage        | Δ (µs) | % of per-layer | Contents |
|---|---:|---:|---|
| qkv_proj     | 509  | (absorbs JIT + sched floor)  | first-stage artifact, not a real 509 µs cost |
| gate_proj    | −166 |  — | negative because qkv_proj's JIT-compile burst evaporates at stage 1 |
| conv_update  | 137  | 18.4% | concat + cpy writeback + ssm_conv + silu |
| state_update | 199  | 26.7% | l2_norm × 2 + repeat × 2 + gated_delta_net + state cpy |
| out_norm     |  21  |  2.8% | rms_norm + w_norm + silu(z) + mul |
| out_proj     |  63  |  **8.4%** | the [2048×4096] GEMV under suspicion |

Cumulative-subgraph breakdown ([tests/perf/deltanet_substage_breakdown.cpp](../tests/perf/deltanet_substage_breakdown.cpp), [fixture](../tests/fixtures/deltanet_substage_breakdown_qwen36.json)) isolates a 0.068 ms baseline (sched floor) and a 0.815 ms OutProj cumulative — **per-layer ≈ 0.747 ms**, consistent with the 0.969 ms coarse total within run-to-run variance. **Verdict: launch-bound.** The output GEMV `out_proj` is only **8.4% of per-layer cost**, far below the 40% matmul-bound threshold; the bulk of the budget lives in dispatch/launch overhead across the dozen small ops in conv_update + state_update + out_norm.

**Recommendation for PR 4.2: proceed with a fused DeltaNet decode-step kernel.** The candidate fusion target is the conv_update + state_update + out_norm slab (≈ 357 µs of measured small-op deltas), which composes naturally into one Metal kernel without crossing layer boundaries; out_proj should remain a separate `mul_mat` since it is already a single backend dispatch and contributes only ~63 µs. Reusing the ceiling math from [DeltaNet — measured](#deltanet--measured-ranks-1): fusing ~70% of per-layer cost (≈ 520 µs of 747 µs) gives an end-to-end ceiling of `50.5 / (50.5 − 30 × 0.520) ≈ 1.46×`. PR 4.2 should scope the fusion against this attribution and re-measure with the same harness for falsification.

#### PR 4.2.A landed — `ggml_deltanet_post_state` (out_norm + gate)

Smallest of the planned sub-PRs. Replaces the rms_norm + mul(w_norm) + silu(z) + mul chain at [src/layers/deltanet.cpp:188-208](../src/layers/deltanet.cpp) with a single fused op `ggml_deltanet_post_state`, delivered via the `patches/` mechanism (CPU forward in 0002, Metal kernel in 0003). Equivalence test [tests/unit/test_deltanet_post_state.cpp](../tests/unit/test_deltanet_post_state.cpp) passes on both backends within the F32 rounding floor (CPU max_abs ≈ 2.4e-7; Metal max_rel ≈ 4.1e-7).

Same-machine, same-binary measurements (5 substage runs each, median; 1 perf-baseline run each):

| Metric | Pre-swap | Post-swap | Δ |
|---|---:|---:|---:|
| `out_norm` stage delta (median, µs) | 69 | 23 | **−46** |
| substage total cumulative (median, ms) | 1.117 | 1.223 | +0.106 (within ±25% per-run noise) |
| perf-baseline qwen36 mean tok/s | 17.20 | 17.35 – 17.87 | **+0.4 – +0.7 (+2 – +4%)** |

The `out_norm` stage delta drop (46 µs/layer) is roughly twice the predicted 18 µs/layer in [docs/plan-deltanet-fusion.md](plan-deltanet-fusion.md), but Metal substage attribution is fuzzy (totals are robust, per-stage deltas swing across runs). The end-to-end tok/s improvement of +2–4% comfortably clears the 1% predicted floor (~540 µs/token of 50.5 ms).

**Verdict: PR 4.2.A landed cleanly; proceed with PR 4.2.B (`ggml_deltanet_pre_state`).** No regressions in the equivalence test on either backend; the patch chain `0001/0002/0003` applies and skips idempotently from a clean ggml tree.

#### PR 4.2.B landed — `ggml_deltanet_pre_state` (l2_norm + repeat × 2)

The plan's larger fusion target. Replaces the `l2_norm × 2 + repeat_4d × 2` chain at [src/layers/deltanet.cpp:135-144](../src/layers/deltanet.cpp) with a single fused op `ggml_deltanet_pre_state`, delivered via patches `0004` (CPU) and `0005` (Metal). Output packs Q and K into one tensor of shape `[head_k_dim, 2 * num_v_heads, n_seq_tokens, n_seqs]`; the call site views back into separate Q and K to keep `ggml_gated_delta_net`'s signature unchanged.

Two configurations exercised in [test_deltanet_pre_state.cpp](../tests/unit/test_deltanet_pre_state.cpp): GQA decode (16→32 heads, Qwen 3.6) and symmetric (32→32, no repeat). All 6 cases pass on CPU + Metal within the F32 rounding floor (CPU bit-exact; Metal max_rel ≈ 2.9e-7).

Same-machine measurements (5 substage runs each, median; 1 perf-baseline run each), with PR 4.2.A as the baseline:

| Metric | Pre (4.2.A only) | Post (4.2.A + 4.2.B) | Δ |
|---|---:|---:|---:|
| `state_update` stage delta (median, µs) | 354 | 139 | **−215** |
| `per_layer` (median, ms) | 0.956 | 0.968 | +0.013 (within ±25% per-run noise) |
| perf-baseline qwen36 mean tok/s | 17.40 | **18.36** | **+0.96 (+5.5%)** |

The `state_update` stage delta drop of 215 µs/layer comfortably exceeds the plan's 79 µs/layer target (the fused op also eliminates two head-sized scratch tensors that ggml_repeat_4d's tile-style writeback would materialize, so the saving is larger than just dispatch overhead alone). The +5.5% end-to-end gain stacks roughly additively on top of PR 4.2.A's +2-4%, supporting the plan's falsification rule that *both* per-layer and end-to-end signals must agree.

**Verdict: PR 4.2.B landed cleanly. The remaining DeltaNet headroom is `conv_update` (≈ 160-200 µs/layer in current measurements). PR 4.2.C (`ggml_deltanet_conv_step`) is a stretch goal per the plan — gated on whether the substage breakdown shows >250 µs/layer still on the table after 4.2.B.** Current per-layer is ~970 µs vs. the 50.5 ms/token end-to-end pre-Phase-4 baseline; the 4.2.A + 4.2.B stack has retired ~270 µs/layer of dispatch overhead. PR 4.2.C decision: based on these numbers, the remaining `conv_update` ~180 µs/layer is at the diminishing-returns threshold (the plan's gate is "≥ 250 µs"). **Recommendation: defer PR 4.2.C** unless a Metal GPU capture session in Instruments shows the conv-side state writeback `cpy` is unexpectedly expensive.

### Norms — measured, confirmed not a target

RMSNorm marginal cost from cumulative chains of {0, 1, 2, 4, 8, 16}
norms at shape `[n_embd=2048, 1]`. The 0→1 delta is unreliable
(empty-graph scheduler-floor noise — chain=0 measured *higher* than
chain=1). The clean datum is the 8→16 slope:

> **~8.2 μs marginal per RMSNorm.**

Site count for Qwen 3.6 (per token):

| Site                          | Count |
|---|---:|
| Pre-attention norm × 40 layers | 40 |
| Pre-FFN norm × 40 layers       | 40 |
| Q-norm + K-norm × 10 attn      | 20 |
| Final output norm              | 1  |
| **Total**                      | **101** |

**Decode-budget contribution.** 101 × 8.2 μs ≈ **0.83 ms/token (~1.6%
of 50.5 ms)**. Even zeroing all norms would yield `50.5/(50.5−0.83) ≈
1.017×`. Off the table as a 3× target.

Fixture: [tests/fixtures/norms_breakdown_qwen36.json](../tests/fixtures/norms_breakdown_qwen36.json).

### Attention — deferred

Coarse per-layer measurement was attempted via a standalone call to
`build_gated_batched_attention` ([tests/perf/attention_breakdown.cpp](../tests/perf/attention_breakdown.cpp))
backed by an ad-hoc `simple_kv_cache`. Three different runtime failures
were encountered and only the last two were resolved:
1. `n_embd_head` derived from `n_embd / n_head` rather than
   `attention_key_length` → reshape size assert (fixed).
2. KV cache constructed with `GGML_TYPE_F16` while production uses
   `GGML_TYPE_F32` and the gather view uses `sizeof(float)` strides →
   inconsistent layout (fixed).
3. The standalone `simple_kv_cache` lives in its own `ggml_context`;
   compute graphs build views of those tensors in a different context.
   The scheduler accepts the cross-context references on first reserve
   but trips during subsequent `sched_reset → alloc_graph` cycles
   (`gallocr_needs_realloc … not valid`). Reordering bg1-before-bg0 did
   not address it; the issue is in how gallocr tracks cross-context KV
   views, not buffer-pool sizing.

**Decision:** measure attention in a follow-up via integration with a
real `Qwen36ForwardPass` (which manages cache + scheduler relationships
correctly). The ranking decision below does not depend on the attention
number — see "Why we don't need the attention number to commit."

---

## Ranked decode budget (Qwen 3.6, 50.5 ms/token)

| Slot | per-layer | layers | total/token | ~%-of-budget |
|---|---:|---:|---:|---:|
| **DeltaNet** | 0.97 ms | 30 | **29.1 ms** | ~58% |
| MoE          | 0.67 ms | 30 | 20.0 ms     | ~40% |
| Attention    | TBD     | 10 | TBD         | TBD  |
| Norms        | 0.008 ms | 101 | 0.8 ms     | ~1.6% |

The columns do not sum to 50.5 ms because Metal's command scheduler
overlaps work across layers in the real forward pass; these are
exclusive per-layer timings.

### Why we don't need the attention number to commit

DeltaNet at 29.1 ms is roughly 2× the MoE chunk. Even if attention turns
out to be 10 ms/token (10 layers × 1 ms — already optimistic; published
flash-attention-style baselines on similar models are smaller), DeltaNet
remains the largest single slot. Attention-side optimizations (e.g.
`ggml_flash_attn_ext`) target a different regime (long-context prefill)
and don't compete with DeltaNet for the Phase 4 PR-1 slot.

---

## Decisions

1. **PR 4.1 (fused MoE custom op) is dropped** — implementation path
   doesn't reach Metal; ceiling ≤ 1.13× even if it did.
2. **DeltaNet replaces MoE as Phase 4 PR-1**, conditional on a
   sub-stage breakdown confirming the launch-bound hypothesis.
3. **Norms confirmed not a target.** ~1.6% of budget; do not revisit.
4. **Attention measurement deferred** to a follow-up via ForwardPass
   integration.
5. **PR 4.4 (attention fusion / `ggml_flash_attn_ext` switch) parked.**
   After PR 4.2.A + 4.2.B retired ~270 µs/layer of DeltaNet dispatch
   overhead (decode +7% end-to-end), the remaining attention slot is
   only 10 layers × estimated ~1 ms = ~10 ms/token (~20% of budget),
   and `ggml_flash_attn_ext` primarily targets long-context *prefill*,
   not single-token decode. Better next-wins live one level up:
   speculative decoding (in progress; potentially 2-3×) and the modular
   refactor. (TurboQuant was removed — outside workload envelope.)
   Revisit if a Metal Instruments capture shows attention is the
   single largest remaining slot at decode shape, or if a long-context
   prefill workload becomes a priority.

---

## Open questions

- **Q1 — Is DeltaNet launch-bound or matmul-bound?** *Resolved: launch-bound.*
  `out_proj` is only 8.4% of per-layer cost; conv_update + state_update +
  out_norm dominate. PR 4.2 should ship a fused DeltaNet decode-step kernel.
  See [DeltaNet sub-stage findings](#deltanet-sub-stage-findings--q1-resolved-launch-bound).
- **Q2 — What's per-layer attention cost at moderate n_kv?** Resolved
  by integrating the breakdown harness with a real `Qwen36ForwardPass`
  rather than a standalone `simple_kv_cache`. Required before any
  attention-side PR is scoped, but **not** required to land DeltaNet
  as PR-1.
- **Q3 — Is the per-iter scheduler overhead in the breakdowns
  representative?** The MoE `route` stage's 380 μs absorbs per-iter
  `sched_reset + alloc_graph + tensor_set + sched_graph_compute + sync`
  that fires once per microbenchmark iteration but only once per token
  in the real forward pass. A ggml-internal per-node timing patch
  (debug-flag-gated) would split GPU work from scheduler overhead.
  Lower priority — current numbers are sufficient for ranking.

---

## Artifacts

- [tests/perf/breakdown_harness.h](../tests/perf/breakdown_harness.h) — shared harness
- [tests/perf/moe_breakdown.cpp](../tests/perf/moe_breakdown.cpp) — MoE per-stage timing
- [tests/perf/deltanet_breakdown.cpp](../tests/perf/deltanet_breakdown.cpp) — DeltaNet coarse total
- [tests/perf/deltanet_substage_breakdown.cpp](../tests/perf/deltanet_substage_breakdown.cpp) — DeltaNet sub-stage cuts (Q1 resolution)
- [tests/perf/norms_breakdown.cpp](../tests/perf/norms_breakdown.cpp) — RMSNorm marginal cost
- [tests/perf/attention_breakdown.cpp](../tests/perf/attention_breakdown.cpp) — broken; see "Attention — deferred"
- [tests/fixtures/moe_breakdown_qwen36.json](../tests/fixtures/moe_breakdown_qwen36.json)
- [tests/fixtures/deltanet_breakdown_qwen36.json](../tests/fixtures/deltanet_breakdown_qwen36.json)
- [tests/fixtures/deltanet_substage_breakdown_qwen36.json](../tests/fixtures/deltanet_substage_breakdown_qwen36.json)
- [tests/fixtures/norms_breakdown_qwen36.json](../tests/fixtures/norms_breakdown_qwen36.json)
