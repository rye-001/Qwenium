# Head-only tap — step 3, measured

**STATUS: LANDED (uncommitted) 2026-09-25, approved by the user. Measured on
Qwen3.8-9B Q4_K_M.** `/v1/locate` now copies out only the one attention head
it reads instead of all 16. Values are byte-identical (tested per recipe, Qwen
and Gemma). At a 10K prompt, together with the `cum_bytes` fix: **locate 38.1 s
→ 27.5 s, absent 56.9 s → 45.5 s, peak memory 22.1 GB → 9.2 GB.** A
pre-existing qwen3 defect surfaced on the way and is fixed; its deeper cause
(ggml plan reuse) is open for a decision.

## 1. What changed

* **`DecodePolicy::attention_tap_heads`** — which heads of each tapped layer
  are copied out; empty (default) = every head, today's tap.
* **`ForwardPassBase::set_attention_taps(layers, heads = {})`** — one call sets
  both, so arming taps without heads resets the list: `/v1/verify` can never
  inherit `/v1/locate`'s single head.
* **`mark_attention_taps`** — with a head list, `kq_soft.<il>` stays an
  ordinary intermediate and each selected head's contiguous `[n_kv, n_q]` block
  is copied (`ggml_view_3d` + `ggml_cont`) into its own output
  `kq_tap.<il>.<h>`. Fail-loud on a head out of range or repeated.
* **`get_attention_taps`** — returns the copies as consecutive blocks;
  `AttentionTap::heads` / `block_of(h)` say which model head each block is (the
  identity on a full tap, so existing indexing is unchanged).
* **`run_lens_locate`** — arms `{use_layer}, {use_head}` and reads through
  `block_of`. Extract, verify and the probes keep the full tap.
* **`LOCPERF`** mirrors the shipped path; `LOCPERF_FULLTAP=1` measures the old
  one. architecture.md carries the seam paragraph.

## 2. Measured (LOCPERF, 6 keys, median of 3)

| 10,045-token prompt | before (full tap) | after `cum_bytes` fix | after head-only tap |
|---|---|---|---|
| locate end-to-end | 38.1 s | 34.2 s | **27.5 s** |
| absent end-to-end | 56.9 s | 51.8 s | **45.5 s** |
| tap copied to host | 6,159 MB | 6,159 MB | **385 MB** |
| tap readback (copy + check) | 5.8 / 6.6 s | 6.4 s | **0.12 / 0.23 s** |
| outside the pass | 5.1 s | 0.3–0.6 s | ~0.2 s |
| GPU compute buffer | 8.2 / 8.9 GB | 8.2 / 8.9 GB | 8.2 / 8.9 GB |
| peak footprint | 22.1 GB | — | **9.2 GB** |

At 4,117 tokens: tap 1,034 → 65 MB, readback 209 → 20 ms, compute buffer
1.84 → 1.58 GB (the unpinned `kq_soft` is now reusable), peak 9.4 → 7.2 GB.

**The `cum_bytes` fix verified:** REQHEAD reproduces its note exactly (score
0.954 / 0.928, evidence 100 / 97.2, same held-out picks), so no span offset
moved; "outside the pass" at 4K fell 832 → 77 ms, at 10K 5.1 → 0.3–0.6 s.

**What remains:** the model's own compute (27 s / 45 s at 10K) and ~8–9 GB of
GPU scratch for the materialized attention of the layers below the cut. Only
step 4 (split pass, flash on the document) reduces those.

## 3. Tests

* `HeadSelectedTapEqualsFullTap{Decode,Prefill}` — every selected block equals
  the full tap's rows for that head byte for byte (`memcmp`), logits unchanged;
  heads chosen out of order and non-contiguous.
* `TapHeadOutOfRangeOrRepeatedFailsLoud`, `ArmingWithoutHeadsResetsHeadList`.
* Run per recipe: **44 / 44 pass on qwen3 (Qwen3-0.6B), qwen35
  (Qwen3.8-9B-Q4_K_M), gemma3 (gemma-3-1b) and gemma4 (gemma-4-12B)** —
  including `TapOffByteIdentical` on every leg. Full unit suite 1023 / 1023,
  also with the qwen3 model enabled.

## 4. Found on the way: qwen3's tapped prefill read overwritten memory

> **FIXED 2026-09-27 (user approved option (a)).** Every tapped pass now
> allocates through `ForwardPassBase::alloc_readback_graph`, and
> `get_attention_taps` refuses any other allocation. The hazard was wider than
> stated below: with head-selected taps, two graphs of the same truncation that
> tap DIFFERENT layers have equal node counts, and the recycled slot can hold a
> later layer's `kq_soft` — also in [0, 1], so the range check misses it.
> `SameShapeDifferentTapLayerGetsItsOwnPlan` reproduced it on all four recipes
> (qwen35/qwen3 silently wrong, gemma3/gemma4 refused) and passes now. The
> reserve-the-real-graph design proposed below is WRONG: splitting a graph
> twice leaves stale input copies (logits off by up to 23); the helper
> reserves a one-node graph instead. See architecture.md §12 (the tap seam).

The new prefill tests failed on the qwen3 leg only: a **full** tapped prefill
returned values outside [0, 1] (the `get_attention_taps` guard fired).

* **Trigger (fixed):** `Qwen3ForwardPass::build_prefill_graph` ignored
  `want_logits` and always built the output head. A head-less tapped prefill
  was therefore node-for-node identical to the untapped `run_prefill` of the
  same length that ran before it. qwen3 now prunes the head at one site, like
  qwen35/36 and gemma3/4 (plan-feed-tokens head-presence rule) — which also
  drops an unneeded LM-head matmul.
* **Root cause (open):** `ggml_gallocr_needs_realloc` (ggml-alloc.c) reuses the
  previous memory plan whenever node count and sizes match, **without comparing
  output flags**. So any tapped graph structurally identical to the previous
  untapped graph on the same scheduler can have its tap overwritten — any
  recipe. The guard makes it fail loud (a refusal), never a silent wrong
  receipt, and no lens route is known to trigger it today. The same hazard
  caused the 2026-09-18 verify-only incident.
* **Proposed fix (needs the user's approval — it changes the tap seam's
  protocol):** force a fresh plan for every tapped pass, e.g. a
  `ForwardPassBase` helper that does sched reset + `ggml_backend_sched_reserve`
  + alloc, used by every tap call site. Not done.

## Reproduce

```
M=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf
LOCPERF=1 LOCPERF_TOKENS=10000 QWEN36_MODEL_PATH=$M /usr/bin/time -l ./build-metal/bin/attn-provenance
LOCPERF=1 LOCPERF_TOKENS=10000 LOCPERF_FULLTAP=1 QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance   # old tap in the breakdown
QWEN35_MODEL_PATH=$M QWEN3_MODEL_PATH=$PWD/models/Qwen3-0.6B-Q8_0.gguf \
  GEMMA3_MODEL_PATH=$PWD/models/gemma-3-1b-it-BF16.gguf GEMMA4_MODEL_PATH=$PWD/models/gemma-4-12B-it-Q8_0.gguf \
  ./build-metal/bin/forward-pass-base-tests --gtest_filter='Recipes/ForwardPassTapTest.*'
```
