# Prefill speed — compute-bound on this machine, not a bug

**STATUS: DIAGNOSED 2026-09-26. Nothing changed in the engine.** The low
prefill throughput behind every prefill-only mode (~590 tok/s through a
locate's 12 of 33 blocks) was the open cost after steps 1–4. Three
measurements on the development machine (**Apple M1 Pro**) say it is the
hardware: our engine matches llama.cpp on short prompts, the whole graph runs
on Metal, and matrix multiplies take 86% of the time. The DeltaNet op, the
suspect, takes 6%.

## 1. Reference — llama.cpp on the same file

`llama-bench -p 512,2048,8192 -n 0 -fa 0,1 -r 2`, a fresh build at
`../llama.cpp/build-bench` (Release, Metal, OpenSSL/curl off — the existing
`build/` there is stale and needs OpenSSL to regenerate).

| model | pp512 | pp2048 | pp8192 |
|---|---|---|---|
| Qwen3.8-9B Q4_K_M (hybrid, 9.2 B params), flash off | 235.6 | 234.3 | 226.3 |
| same, flash on | 236.8 | 234.6 | 214.9 |
| gemma-4-12B-it Q8_0 (dense, 11.9 B params) | 166.7 | 162.4 | — |

Per parameter the two are the same (~2.0–2.2 × 10¹² parameter-tokens/s): the
chip is compute-bound, and the hybrid's DeltaNet layers cost it nothing extra.

## 2. Our engine — PREFPROF

Full-depth prefill (33 blocks, materialized, untapped, no output head), median
of 3 after a warm-up:

| tokens | build + alloc | compute | tok/s | vs llama.cpp |
|---|---|---|---|---|
| 512 | 2.2 ms | 2,119 ms | **241.7** | parity |
| 2,048 | 21.0 ms | 9,507 ms | **215.4** | −8% |
| 8,192 | 327.5 ms | 45,768 ms | **179.0** | −21% |

**Placement (2,048 tokens):** 1,561 nodes, **1 scheduler split, 0 nodes on the
CPU backend** — no fallback.

**Time per op (2,048 tokens,** every node observed through the scheduler's
eval callback; the per-node sync inflates totals, so read the shares):

| op | share | nodes |
|---|---|---|
| MUL_MAT | **85.7%** | 264 |
| GATED_DELTA_NET | **6.3%** | 24 |
| everything else (views, concat, cont, SwiGLU, norms, softmax …) | < 1% each | |

## 3. What it means

* **Short documents: the hardware is the limit.** ~235 tok/s for a 9B at full
  depth is what this GPU does; no engine change moves it much. A faster chip
  (Max / Ultra class) would.
* **The DeltaNet hypothesis is dead.** The fused `gated_delta_net` kernel is 6%
  of prefill; a chunked rewrite would buy little. Not to be re-proposed.
* **The one gap that is ours grows with length** (−8% at 2K, −21% at 8K): we
  build the whole prompt as one graph with materialized attention, while
  llama.cpp works in 512-token ubatches. Closing it (chunking, or flash on
  untapped passes beyond what step 4 already licenses) would recover at most
  ~20% at 8K — and chunked vs one-shot prefill is not bit-identical on Metal,
  so it needs a drift gate like LOCSPLIT. Small gain, real cost: not
  prioritised.
* **The levers that matter are already in:** doing less work — truncating each
  request to the blocks its head needs, the split+flash document pass (step 4),
  and the kept document (73× on a follow-up at 10K).

## Reproduce

```
PREFPROF=1 QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf ./build-metal/bin/attn-provenance
../llama.cpp/build-bench/bin/llama-bench -m models/Qwen3.8-9B-Q4_K_M.gguf -p 512,2048,8192 -n 0 -fa 0,1 -r 2
../llama.cpp/build-bench/bin/llama-bench -m models/gemma-4-12B-it-Q8_0.gguf -p 512,2048 -n 0 -r 2
```
