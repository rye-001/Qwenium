# Quantized KV cache + flash attention (2026-09-02) — partial: flag lands, numerics blocked

Candidate A from [`note-ggml-b10582-opportunities.md`](note-ggml-b10582-opportunities.md).
Status: **DONE.** The flag, its guards, the stride fix and its cross-family gate
are all in. Three defects were found and fixed on the way, two of them
pre-existing and months-latent (§3, §4).

## 1. What was built

`--kv-type <f32|f16|q8_0|q4_0>` on both front ends; `--kv-f16` kept as an alias
so existing invocations are unchanged. Default stays F32 (the historical,
byte-identical path).

The type vocabulary lives in `src/state/kv_cache_simple.h`, next to the cache the
type is a property OF: `path_tag()` already folds `type_k`/`type_v` and the
snapshot header already round-trips them through `require(...)`, so a snapshot or
prefix blob captured under one element type is refused fail-loud under another
with no new code. The recipe layer was ALREADY generic (`create_forward_pass`
takes a `ggml_type`); only the front ends hardcoded the F16/F32 binary.

**Quantized types are refused without `--flash-attn`**, fail-loud, before any
weights load. This is structural, not policy: the materialized attention path
transposes V (`ggml_permute(v,1,2,0,3)` + `ggml_cont`), which moves `ne[0]` — the
block dimension of every quantized type — out of position, and Metal's CPY/CONT
has no quantized *source* case, so the node would take ggml's silent CPU fallback
(§3 of architecture.md) rather than fail. It also excludes `--attention-lens`
transitively, via the existing flash/lens refusal.

## 2. Measured: the capacity claim is real

Qwen3-0.6B-Q8_0, `-c 2048`, KV cache as reported at allocation:

| `--kv-type` | KV bytes | vs f32 |
|---|---:|---:|
| f32 (default) | 896.00 MB | 1.00× |
| f16 | 448.00 MB | 2.00× |
| q8_0 | 238.00 MB | 3.76× |
| q4_0 | 126.00 MB | 7.11× |

This is the lever on the `ctx x slots` axis the envelope is written in. It is
**not** a speed lever: KV is ~1% of decode bandwidth against ~12 GB of weights,
so the decode ceiling is ~1% (Amdahl, §10). Note also that the new
`kernel_flash_attn_ext_kv_f16` dequant pre-pass is **prefill-only** —
`ggml_metal_op_flash_attn_ext_use_kv_f16` gates on `src[0]->ne[1] < 32`, and with
`q: [n_embd_k, n_batch, n_head, ne3]` that is n_batch < 32, above our slot count.

## 3. Defect found and FIXED: `--flash-attn` never applied to CLI prefill

`src/cli/complete.cpp` called `run_prefill` at line 172 and
`set_attn_impl(AttnImpl::Flash)` at line 213 — **40 lines too late**. On the `-p`
completion path flash therefore applied only to decode; prefill silently ran
materialized attention, with no diagnostic. architecture.md §5 states flash covers
"**both prefill and decode**" and that "**Prefill is where it pays most**", so the
gap contradicted the documented contract and was invisible.

Measured on Qwen3-0.6B-Q8_0, 2601-token prompt, interleaved A/B, cold run discarded:

| | before fix | after fix |
|---|---:|---:|
| flash off | 3577 ms | 3546 / 3747 ms |
| flash on | 3683 / 3636 ms (**no win**) | **1940 / 2033 ms** |

**~1.83x prefill**, previously not delivered at all on this path. `chat.cpp` has
the same ordering for its SYSTEM-PROMPT prefill only (line 301 before line 337);
per-turn prefills come after and were always fine. **The server is correct** —
`enable_flash_attn()` runs at startup, long before any request prefill.

Anything that measured CLI `-p` prefill with `--flash-attn` before this fix
measured a no-op. Whether §5's "7% at 756 tokens to ~55% at 3000" came from this
path or from a bench binary is **unverified** — treat it as an inherited number
until re-measured.

## 4. BLOCKER: non-F32 KV is mis-read on the batched decode paths

`--kv-type f16`, `q8_0` and `q4_0` all produce degenerate output on the `qwen3`
recipe (`" The!!!!!!!!!!"` for the quantized types; word-salad for f16), while
f32 is coherent. f16 is coherent on `qwen35`, so this is recipe/path-specific,
and it reproduces with flash **off** — so it is not a flash problem.

Cause, and architecture.md §9 predicts it verbatim: *"Attention reads the cache
through views whose strides are derived from the tensor's own type, never
`sizeof(float)`; hardcoding the stride silently mis-reads a non-F32 cache instead
of failing."* Six such hardcodes exist —
`src/layers/attention.cpp:470-472,477-479` (`build_batched_attention`) and
`744-746,749-751` (`build_gated_batched_attention`), i.e. **both batched decode
builders**. Prefill's `build_attention` is correct (`ggml_row_size(v_full->type, ...)`).

So §9's stride claim is **documentation of an intent the code does not honor on
the decode side**. This also means `--kv-f16` has been quietly mis-reading its
own cache on these paths since it shipped — which is a candidate explanation for
the unexplained Gemma 4 MoE `--kv-f16` logit shift recorded in §12, though that
is a hypothesis, not a measurement.

### 4.1 FIXED, with the cross-family gate

Root cause, precisely: the two gather branches return DIFFERENT element types.
`gather_k`/`gather_v` (multi-slot) go through `ggml_get_rows`, whose result type
is always F32 unless the source is I32 — it dequantizes, so a float stride is
CORRECT there. `gather_k_single`/`gather_v_single` (the B==1 fast path) return a
`ggml_reshape_3d` VIEW of the cache and therefore keep the cache's element type,
where a float stride mis-reads it. Both call sites now derive strides from the
gathered tensor's own type via `ggml_row_size(k_gathered->type, n)`, which is
correct for both branches.

`ggml_row_size(GGML_TYPE_F32, n)` is `ggml_type_size(F32)*n/ggml_blck_size(F32)`
= `4n` = `n*sizeof(float)` — an exact integer identity, so the F32 graph is
built with literally the same strides and the default path cannot change.

**Cross-family gate (CLAUDE.md: a Qwen AND a Gemma model):**

* Before/after greedy generation, 40 tokens, `-t 0`, `--kv-type f32`, over
  `qwen35` (Qwen3.5-0.8B), `gemma3` (gemma-3-1b-it) and `qwen3` (Qwen3-0.6B):
  **byte-identical output files, same md5** (`eec05869…`).
* Existing bitwise suites on the exact single-slot path that changed:
  `Gemma3BatchedDecodeTest.Tier1SingleSlotBitwise` and
  `Gemma4BatchedDecodeTest.Tier1SingleSlotBitwise` **PASS** (these memcmp logits),
  plus `Qwen35FeedTokensTest.MidDecodeDifferentialTokenStable`.
* `ctest -j1`: **825 passed, 0 failed, 382 skipped** (was 817; +8 new).

**Effect of the fix**, Qwen3-0.6B-Q8_0, same prompt, greedy: `f32`, `f16` and
`q8_0` now produce **token-identical** output where `f16` previously produced
word-salad. `q4_0` diverges, which is expected at 4 bits and is why it stays
opt-in. So §9's "token-stable" claim for `--kv-f16` is now true on this path
rather than merely asserted.

**Permanent regression gate, model-free:** `tests/unit/test_kv_cache_simple.cpp`
`KVCacheGatherTypeTest`, parameterized over f32/f16/q8_0/q4_0 — pins that
single-slot gather keeps the cache type, and that a non-F32 row stride is NOT the
float stride (so the test cannot silently stop discriminating). Note the fixture
needs a block-aligned width: Q8_0/Q4_0 block at 32 elements, so a 16-wide row is
less than one block and ggml aborts.

## 5. Not done
No server-side test of `--kv-type` under load; no Gemma/quantized coherence leg
beyond the byte gate; no re-measurement of §5's prefill numbers (still an
inherited number); no investigation of whether this stride defect explains §12's
unexplained Gemma 4 MoE `--kv-f16` logit shift.
