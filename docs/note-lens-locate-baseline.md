# Locate baseline — what today's `/v1/locate` costs, by document length

**STATUS: MEASURED 2026-09-25 on Qwen3.8-9B Q4_K_M, 32 GB Apple Silicon, idle
host.** Step 2 of note-lens-prefill-only-engine.md: the numbers the head-only
tap and the split pass have to beat. The 10K envelope **runs**, at 38 s
(locate) / 57 s (absent) per request and a **22 GB** peak. The tap copy grows
exactly as that note's arithmetic said (6.2 GB at 10K), and one surprise: **5 s
at 10K was a quadratic helper outside the model** (`cum_bytes`), now fixed.

## 1. The measurement

`LOCPERF=1 LOCPERF_TOKENS=<n>` in `tests/perf/attn_provenance.cpp`. One
document length per process, so `/usr/bin/time -l` attributes the peak memory
footprint to that length alone.

* **Document:** the messy corpus repeated to the target prompt length (as
  WARMPERF); **6 keys**, key mode, `top_k` 3.
* **End-to-end:** the shipped `run_lens_locate`, one untimed warmup, median of
  3 — what a request pays.
* **Breakdown:** the same tapped, truncated, materialized prefill replicated
  step by step — graph build + alloc, compute, tap readback (the copy and the
  [0,1] check, both inside `get_attention_taps`) — plus the scheduler's compute
  buffer and the tap's bytes.
* **Heads:** locate (L11, 12 blocks) and absent (L19, 20 blocks — the
  locate-only server's cut).

## 2. Results

| prompt tokens | end-to-end locate / absent | tap (host copy) | GPU compute buffer | peak footprint |
|---|---|---|---|---|
| 1,152 | 2.0 s / 3.3 s | 81 MB | 0.29 / 0.33 GB | 7.0 GB |
| 2,019 | 4.1 s / 6.7 s | 249 MB | 0.61 GB | 7.4 GB |
| 4,117 | 9.8 s / 15.5 s | 1.0 GB | 1.8 GB | 9.4 GB |
| 8,013 | 26.7 s / 38.5 s | 3.9 GB | 5.7 / 5.8 GB | 16.6 GB |
| **10,045** | **38.1 s / 56.9 s** | **6.2 GB** | **8.2 / 8.9 GB** | **22.1 GB** |

The peak footprint includes the whole 9B loaded (the probe loads all blocks;
a `--lens-locate-only` server would hold 20 of 33).

**Breakdown, locate head** (absent in brackets where it differs):

| prompt tokens | build + alloc | compute | tap readback | outside the pass |
|---|---|---|---|---|
| 1,152 | 3 ms | 1.94 s (3.23) | 17 ms | 0.06 s |
| 2,019 | 8 ms | 3.87 s (6.48) | 51 ms | 0.20 s |
| 4,117 | 30 ms | 8.53 s (14.3) | 0.37 s (0.50) | 0.83 s |
| 8,013 | 0.21 s | 19.9 s (33.1) | 2.35 s | 4.18 s |
| 10,045 | 0.21 s | 26.9 s (44.9) | 5.83 s (6.55) | 5.12 s |

"Outside the pass" = end-to-end minus the three timed parts: rendering,
tokenizing, the token→byte map, the per-key aggregation.

## 3. Reading

* **Superlinear, as expected of materialized attention:** 8.7× the tokens
  costs 19× the time (locate), 17× (absent).
* **The tap is exactly 16 × P × P floats** (6,158 MB at 10,045 tokens), and
  reading it back grows from 17 ms to 5.8 s. The head-only tap removes
  nearly all of both (a job reads one head, or bind's short list, and a few
  rows).
* **About 9 GB of the 10K peak is GPU scratch** — the materialized attention
  matrices of the attention layers below the cut. Flash in the document pass
  (split pass) is what removes it; the tap cut alone does not.
* **Compute dominates** at every length (27–45 s at 10K). Only the split pass
  (flash on the document) and the kept document (no re-prefill on the next
  question) reduce it.
* **Unexplained, not diagnosed:** even at 1K the prefill runs ~590 tok/s
  through 12 blocks (~216 tok/s full-depth equivalent) — the same unexplained
  low throughput plan-lens-warm-document §8.2 recorded. Worth its own look.

## 4. The quadratic helper — fixed

The "outside the pass" column grew 0.06 → 0.83 → 4.2 → 5.1 s. The cause is
`cum_bytes` (`server_lens.cpp`; an identical copy in the probe), the
token → byte-offset map every lens route builds: it re-decoded the **whole
prefix for every token** — P decodes of up to P tokens.

`Tokenizer::decode(vector)` is literally the concatenation of the per-token
decodes (`tokenizer.cpp`), so `cum[k+1] = cum[k] + decode(toks[k]).size()` is
the same number in one pass. Both copies now do that. Used by `/v1/extract`,
`/v1/verify` and `/v1/locate`, so all three gain.

Verification: unit tests pass; REQHEAD (spans resolved through the probe's
copy) reproduces its note exactly, so no offset moved; "outside the pass" fell
832 → 77 ms at 4K and 5.1 → 0.3–0.6 s at 10K (locate end-to-end 38.1 → 34.2 s).
The head-only tap that followed is in note-lens-tap-cut.md.

## 5. What this is and is not

* **One machine, one model, one quant.** Absolute times are this host's;
  ratios between arms are the transferable part.
* **Key mode with 6 keys.** Question mode renders a longer suffix; the
  per-key aggregation cost grows with key count (measured earlier: +6.7% from
  1 to 15 keys at short lengths).
* **Probe process, not the server.** The server adds HTTP and JSON, and a
  `--lens-locate-only` server's footprint is smaller by the unloaded blocks.

## Reproduce

```
for T in 1000 2000 4000 8000 10000; do
  LOCPERF=1 LOCPERF_TOKENS=$T QWEN36_MODEL_PATH=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf \
    /usr/bin/time -l ./build-metal/bin/attn-provenance
done   # ~40 min in total; 10K needs ~22 GB
```
