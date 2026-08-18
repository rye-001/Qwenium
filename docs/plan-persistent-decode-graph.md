# Persistent Decode Graph — Plan

Status: P1 LANDED (e1f95c8). P2 LANDED (80eefc5). P3 DONE (DecodeGraphCache +
opt-in `--persistent-graph` CLI, reuse bitwise-neutral gate green Metal+CPU,
**measured 1.32× decode on Qwen 3.6, stable ×3**). CLI-scoped per the phase
decision; P4 (server) + P5 (MTP head) remain. The byte-identity premise was
FALSIFIED in build (§0.1) → ships OPT-IN, default-off.
Provenance: `note-decode-overhead-probes.md` (2026-07-09 probe hunt) — all
numbers M1 Pro, Metal, Qwen3.6-35B-A3B Q2_K_XL, B=1 unless noted.

## 0.1. Correction (2026-07-10, measured): bucketing is NOT byte-identical

§2.2 claimed provable byte-identity for bucketed n_kv. **Falsified by
`test_decode_kv_bucket`.** Masks are provably correct (pad columns = −inf) and
padding is inert, but widening n_kv past the exact `pos+1` re-blocks the
softmax / scores·V reduction (CPU SIMD lane grouping; Metal kernel tiling), so
the REAL rows round differently. Measured over 40 steps, exact-vs-bucket:

- Metal: token-stable, `max|Δlogit|` ~6e-6 (Qwen3.5) / ~4e-5 (Gemma3), 0 flips.
- CPU: `max|Δlogit|` ~0.07–0.28; Qwen3.5 flipped ONE token in 40 steps, at a
  genuine 0.010 top1-2 tie. SetRows ≡ Cpy to the bit (the fork is bucketing-
  intrinsic, not a set_rows artifact).

This is the standing shape-change fork (architecture.md §11), same status as
speculative decoding: **token-stable modulo ties, not byte-identical.** It is
UNAVOIDABLE for any persistent graph — persistence requires a fixed n_kv, and
any fixed n_kv ≠ exact `pos+1` reblocks the reduction.

**Decision (user, 2026-07-10):** ship persistent-graph OPT-IN
(`--persistent-graph`, default OFF). The default decode path stays exact-n_kv
+ cpy-write = today's byte-reproducible decode, untouched. Bucketing (256) and
the set_rows write turn on together only under the flag. Gate =
**tie-tolerant token-stable** (flips allowed only when the exact top1-2 gap <
0.35) + a loose `max|Δlogit|` ceiling; strict bitwise-across-buckets kept
`DISABLED_`.

## 0. The measured facts this plan is built on

- A decode step spends **~12 ms in `ggml_backend_sched_alloc_graph`** (galloc
  replan) + ~1 ms graph build, per step, every step — 26% of a 46.7 ms step.
- The replan cost is **shape-independent**: fixed-shape alloc = 10.9 ms vs
  growing-shape 12.2 ms. Bucketing n_kv *alone* recovers ~1 ms. (Killed the
  "bucket so galloc reuses its plan" thesis.)
- The measured fix (probe `tests/perf/alloc_reuse.cpp`, PERSIST block): build +
  alloc **once**, then per step only `set_decode_inputs` + `graph_compute`:
  **43.75 → 34.09 ms/step = 22.9 → 29.3 tok/s = 1.28×**.
- `sched_reset` itself is free (0.011 ms).
- Calibration warning (house rule): estimates in this thread ran 1.2–3×
  optimistic three times. The 1.28× is *measured ceiling*; production
  expectation after real-world leakage: **~1.2–1.25× at B=1**.

## 1. Why the decode graph cannot persist today

Verified against the tree (qwen-mtp, 915cae9):

1. **KV write offset is baked structure — THE blocker.**
   `simple_kv_cache::cpy_k/cpy_v` (`src/state/kv_cache_simple.cpp:194,219`)
   freeze `slot*nb[2] + positions[slot]*nb[1]` into a `ggml_view_2d` byte
   offset at graph-build time. A reused graph would overwrite the same KV row
   forever (the probe dodged this by fixing the position).
2. **n_kv dims are exact.** Recipes size mask / gather-indices / gathered-KV
   views at `max_pos+1` (`qwen36.cpp:417-431`, `gemma3.cpp` decode, shared
   `attention.cpp:255-279`), so the shape churns every step.
3. **(Server) slot composition** — n_batch and the slot set are baked per
   step. Handled by keyed rebuild, not by design contortions: a composition
   change is rare (request join/leave) and a 13 ms rebuild amortizes over a
   request's lifetime.

Everything else is **already value-driven** through the typed graph inputs —
`GraphInputSet` was built to separate values from structure and this is its
first real payoff:

- `AttnMaskInput::set_input` sizes from the *tensor's own dim* (`t->ne[0]`)
  and its causal rule (`j <= q_pos`) −inf-gates every padded column with
  **zero changes**. Same for the window rule (Gemma sliding layers) and the
  window-deduplicated per-layer masks.
- `GatherIndicesInput` sizes from the tensor dim; padded entries index
  in-slot rows (bucket ≤ n_ctx_max), which only need to be *finite* (§2.2).
- tokens / positions / sparse ids are input values.
- The probe already ran `set_decode_inputs` + compute on a reused graph — the
  refill machinery works today.

## 2. Design

### 2.1 P1 — value-driven KV write (`ggml_set_rows`)

Our ggml pin (llama.cpp `b8390`) has `ggml_set_rows`, Metal-supported for
F32 src → F16 dst (our cache dtype). This is the same move llama.cpp made for
graph reuse — the write position becomes an input *value*:

- `simple_kv_cache` gains a set_rows-based write beside `cpy_k/cpy_v`: cache
  tensor viewed as 2D rows `[n_embd, n_ctx_max * n_batch_max]`, destination
  row = `slot * n_ctx_max + pos`, indices supplied by a new typed input
  `KvWriteIndicesInput` (one shared index tensor, n_batch entries, referenced
  by every layer's K and V set_rows nodes).
- Side effect: the per-slot `ggml_cpy` loop (2 × n_layers × n_batch nodes)
  collapses to 2 × n_layers set_rows nodes — smaller decode graphs even on
  the rebuild path.
- **Decode path only.** Prefill graphs are built per prompt and don't need it.
  This leaves two write paths to one cache — a flagged fork; unify later iff
  the byte gate stays green (§5 soft-spot entry until then).
- Rollout is capability-gated: `ForwardPassBase::supports_persistent_decode()`
  default **false**; converted in P1: `qwen36` + `qwen35` (both route through
  the shared `build_gated_batched_attention`) and `gemma3` (hand-rolled
  in-recipe, the cross-family falsifier). Other recipes keep today's rebuild
  path untouched until a later sweep.

**Gate P1** (extraction discipline — this lands before any persistence):
set_rows-decode vs cpy-decode, byte-identical logits AND byte-identical
written-KV-region, Qwen3.6 + Gemma3, Metal + CPU.

### 2.2 P2 — bucketed n_kv (the persistence precondition)

- Decode graphs size n_kv at `bucket_up(max_pos+1, 256)` (capped at
  n_ctx_max) instead of exact — one shared helper, used by the converted
  recipes' `build_decoding_graph`.
- **Zero-init KV buffers at allocation.** Cold padded rows flow into the KQ
  matmul before the mask can gate them; uninitialized bits could be NaN, and
  `NaN + (−inf) = NaN` poisons softmax. Finite garbage contributes exactly 0
  after the −inf mask; zeros make that guarantee unconditional. (Stale rows
  after speculative truncation are finite by the same argument.)
- Numerics (as-built, §0.1): the pad is inert (masked to −inf, reads zero
  rows) but widening n_kv reblocks the reduction over the REAL rows, so
  bucketed ≠ exact bit-for-bit. The KQ output-dim argument held; the
  softmax-sum and scores·V reduction axes did not. Token-stable modulo ties.

**Gate P2** (as-built, `test_decode_kv_bucket`): tie-tolerant token-stable
(argmax matches exact except at a <0.35 top1-2 tie) + `max|Δlogit|` < 0.5
ceiling, Qwen3.5 + Gemma3, Metal + CPU, BOTH write modes (proves the fork is
bucketing-intrinsic). Strict bitwise-across-buckets = `DISABLED_`.

### 2.3 P3 — `DecodeGraphCache` + CLI wiring

New module `src/core/decode_graph_cache.{h,cpp}` (+
`tests/unit/test_decode_graph_cache.cpp`), beside its consumers
`decode_step`/`decode_plan`. (Note: adds to the flagged `core/` dumping
ground — accepted for co-location with decode orchestration; the eventual
`core/` split moves them together.)

- Owns a **dedicated `ggml_backend_sched`** for the persistent decode graph
  (two in-tree precedents: the MTP head's private sched, the image-prefill
  dedicated sched). Prefills / `feed_tokens` / MTP verify passes on the main
  sched can never invalidate the persistent allocation.
- Cached `{key, gf}`; **key** = {ordered slot ids, n_kv bucket, head mode,
  output_hidden, diagnostic}. `step()`: key hit ⇒ set inputs + compute;
  miss ⇒ reset + build + alloc (~13 ms) + fail-loud verification that the
  built mask dim equals the keyed bucket (recipe honored the bucket —
  expected/actual named in the error).
- **Rebuild-path fallbacks (v1)**: sparse-head steps (grammar-narrowed rows
  are structural), Bridge-route recipes, `supports_persistent_decode()==false`.
  Grammar-heavy workloads keep today's cost and their existing wins (sparse
  head, forced elision).
- Correct-by-key: speculative KV truncation that drops below the bucket base
  changes the key ⇒ rebuild; bucket-boundary crossings re-alloc every 256
  steps ≈ 0.05 ms/step amortized.
- CLI wiring at `decode_step.cpp:105-110`. **Opt-in `--persistent-graph`,
  default OFF** (§0.1 decision): enabling it flips the ForwardPass to
  {SetRows, bucket 256} and routes decode through the cache; off = today's
  exact-n_kv cpy decode, byte-unchanged. The flag doubles as the A/B seam the
  perf two-signal measurement needs.

**Gate P3**: persistent vs rebuild path byte-identical over a multi-hundred-
token generation crossing ≥2 bucket boundaries, Qwen3.6 + Gemma3, Metal +
CPU; perf ≥1.2× tok/s at B=1 on the probe workload, two signals (per-step
median + end-to-end tok/s).

### 2.4 P4 — server decode loop

Same cache object wired into the inference thread's batched decode
(`http_server.cpp` decode sites); rebuild on slot-composition change.
Expected honestly: **~1.25× at B=1–2 tapering to ~1.05× at B=8** — the 12 ms
is the intercept, and the per-lane 25.6 ms/B dominates at high batch (probe
fit `ms/step ≈ 20 + 25.6·B`). Gate: existing server smokes green + A/B tok/s
at B=1/2/4/8.

### 2.5 P5 — MTP head persistence (unlocks the loop restructure)

`mtp_draft` (`qwen36.cpp:644-647`) pays reset+build+alloc per head step —
~12 ms replan for ~1.3 ms of head math; this is *why* Phase 4 broke even.
The head KV window is < 32 ⇒ one bucket-32 graph persists for an entire
run; mask/pos/hidden are already inputs; the private-KV write gets the same
set_rows treatment. Second instantiation of `DecodeGraphCache` — the
falsifier that it isn't single-site-shaped. The MTP loop restructure
(draft-from-verify-hiddens, D3 already emits all K positions) then rides on
cheap head steps. No end-to-end estimate here — probe after P5 (calibration
rule).

## 3. Non-goals

- No prefill persistence (prompt-shaped, once per request).
- No high-batch win (B≥4 is math-bound; probe-proven).
- No grammar/sparse-head persistence in v1.
- No change to prefill KV writes (v1 fork, flagged).

## 4. Memory & envelope

Dedicated decode sched adds a second scratch allocation. Decode scratch ∝
n_kv_bucket × n_batch (gathered K/V transients); its peak equals today's
end-of-context decode peak, just reached at bucket granularity. Within the
envelope (4K ctx, ≤10 slots) — but **measure and report** in the P3/P4 gates
alongside tok/s.

## 5. architecture.md deltas (this plan is an approval item)

- §4 codemap: `decode_graph_cache` added to the `core/` row.
- §5 decode-loop paragraph: note the opt-in `--persistent-graph` path
  (persistent graph + bucketed n_kv + set_rows write) as token-stable-not-
  byte-identical, same status as speculative; default decode unchanged.
- §6 (server flags) / CLI: `--persistent-graph` listed.
- §10: add the measured persistent-graph number post-P3 (provenance: this
  plan + probe note).
- §12: (a) write-path fork bullet — ALREADY ADDED under P1; (b) note that the
  persistent decode path is opt-in precisely because bucketed n_kv is token-
  stable-not-identical (§0.1), a deliberate limitation not a soft spot to fix.

## 6. Sequencing

P1 → P2 → P3 (CLI, the measured 1.28× realized) → P4 (server) → P5 (MTP) →
then the MTP loop restructure as its own scoped step. Each phase lands only
with its gate green; P1/P2 are pure extraction-style transforms and land
default-inert (nothing persists until P3 turns the key).
