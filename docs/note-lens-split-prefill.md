# Split locate prefill — step 4, gated per quantization

**STATUS: LANDED (uncommitted) 2026-09-25, approved by the user (option 1: a
Q4_K_M-pinned licence).** `/v1/locate` can prefill in two passes cut at the
document's end — the document untapped (optionally under flash), then the
instruction, keys and template tail tapped. On Qwen3.8-9B **Q4_K_M**,
split+flash passes the drift gate on all five heads and at a 10K prompt runs
locate **27.5 → 23.4 s**, absent **45.5 → 37.4 s**, GPU compute buffer
**8.2 → 2.9 GB**. On **Q8_0** split+flash **fails** the gate (near-ties flip), so
the licence is per quantization: a Q4_K_M-pinned calibration row carries it,
the any-quant row (Q8_0) stays one-shot. Split without flash passes both quants
but is slightly slower and is not licensed.

## 1. The idea

Every row a lens mode reads — key rows, question rows, the template tail —
sits **after** the document, and attention is causal: the document's rows never
see the question, the question's rows see the whole document either way. So
the prompt can be prefilled in two passes at the document's end and the rows
read are computed from the same inputs, up to chunk-boundary rounding. Pass 1
is never tapped, so it may run flash, which never materializes the document's
P × P attention.

## 2. What changed

* **`LensPrefillShape { OneShot, Split, SplitFlash }`** (`server_lens.h`) and a
  parameter on `run_lens_locate` (default `OneShot`, byte-identical to before).
  The cut is the first token that starts at or after the document's end (a
  token straddling the boundary stays in pass 1, so both passes together are
  exactly today's tokens); fail-loud if any read row would fall before it, and
  `SplitFlash` is refused on a recipe without flash.
* **The shape is a per-model, per-quantization licence**:
  `LensConstants::locate_prefill_shape` + `locate_prefill_provenance`, appended
  last (positional aggregates), default `OneShot`.
* **The first `file_type`-pinned row**: Qwen3.8-9B at Q4_K_M
  (`kGgufFileTypeQ4_K_M` = 15, read from the GGUF itself; Q8_0 reads 7). Both 9B
  rows are built by one function, `qwen38_9b_constants()`; the pinned row
  (`qwen38_9b_q4km_constants()`) differs only in the licence, and a test checks
  every coordinate matches.
* **Report**: `/v1/locate` emits `"prefill": "one-shot" | "split" |
  "split+flash"` (additive, no format bump); a split+flash locate is stamped
  `config.attention = "flash-prefill"` by route
  (`RoutePrefill::DocumentPassFlash`), because its document pass ran flash by
  licence, not by server flag.
* **Probe legs**: `LOCSPLIT` (the drift gate) and `LOCPERF_SHAPE` on the timing
  leg. architecture.md and lens-format.md updated.

## 3. The gate (LOCSPLIT)

Against the one-shot pass, per landed head (locate, choice, absent, score,
inject), on: Leg C's 15 order e-mails (key mode, EN/DE), REQHEAD's 6 CVs
(question mode, 72 requirements, EN/DE), and two long documents (~4K and ~8K
tokens). **Bar, set before the run:** top-1 span identical 100%, no key's top
peak moves by its own decision margin (top-1 minus top-2 peak, one-shot), the
winning key (by summed hit mass) never changes, in every group.

| arm | Q4_K_M | Q8_0 |
|---|---|---|
| split | **pass** — worst 0.042 of a margin; long documents: zero drift | **pass** — worst 0.22 of a margin |
| split + flash | **pass** — worst 0.62 of a margin (one inject key), 0.47 (choice, long) | **FAIL** — absent, long documents: top-1 changed on 2 of 12 keys, one near-tie moved 29× its margin; score, German questions: 1 of 36 |

Top-3 order changed on a few near-ties even where top-1 held (worst 91.7%
identical, Q4_K_M split+flash, long documents); the bar is on top-1. The
winning key never changed in any arm.

## 4. Speed and memory (LOCPERF, 10,045-token prompt, 6 keys)

| | one-shot | split | **split + flash** |
|---|---|---|---|
| locate end-to-end | 27.5 s | 28.6 s | **23.4 s** |
| absent end-to-end | 45.5 s | 47.7 s | **37.4 s** |
| GPU compute buffer, locate / absent | 8.2 / 8.9 GB | 8.1 / 8.8 GB | **2.9 / 3.6 GB** |

At 4K the three are within ~0.5 s — flash pays only on long documents. The
process peak footprint read ~9.2 GB for all three; on this machine it does not
appear to count the Metal buffers, so the compute buffer is the memory
measure here. (One-shot already includes the head-only tap and the
`cum_bytes` fix: from the original baseline, 10K locate went 38.1 → 23.4 s and
the peak footprint 22.1 → 9.2 GB.)

## 5. Live check

A `--lens-locate-only` server, the same request on each quant: Q4_K_M reports
`"prefill":"split+flash"`, `"attention":"flash-prefill"`; Q8_0 reports
`"one-shot"`, `"materialized"`; same top span for the first key.

## 6. What this is and is not

* **The licence rests on thin headroom.** The same model at Q8_0 flipped
  near-ties; Q4_K_M's worst key sat at 0.62 of its margin. The gate corpus is
  ~350 keys; a wider corpus could find a Q4_K_M flip too. The provenance says so.
* **Split alone buys nothing yet** — it is the base for keeping a document
  between requests (step 6: pass 1 is exactly what gets kept).
* **Compute still dominates** (23–37 s at 10K). The unexplained low prefill
  throughput (~590 tok/s through 12 blocks even at 1K) is the next lever.
* **Only the 9B is gated.** The 27B and the 35B MoE stay one-shot (MoE is
  expected to fail, as flash prefill did in the extract drift gate).

## Reproduce

```
M=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf     # or Qwen3.8-9B-Q8_0.gguf
LOCSPLIT=1 QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance                               # ~13 min
for S in oneshot split splitflash; do
  LOCPERF=1 LOCPERF_TOKENS=10000 LOCPERF_SHAPE=$S QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance
done
```
