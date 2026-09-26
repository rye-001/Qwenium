# Prefill-only on a full-inference engine — what is worth adapting

**STATUS: ANALYSIS 2026-09-25. Nothing new measured, nothing built.** The
engine was built for prefill + decode; the lens modes use the prefill alone.
No separate engine is needed — the seams are already options on the one
forward pass. One change is worth doing: **split the lens prefill at the
document's end, tap only the part after it.** It removes an O(heads × P²)
memory cost that makes the 10K envelope unreachable for the lens today, lets
the document pass run flash, and makes the document reusable across requests.
Every change here is an architecture decision (the user's), and every one
moves the numbers the head constants were measured on, so each needs the
drift gate.

## 1. Name

**Prefill-only** covers both. The one generated token is not a decode step:
its probabilities are the prefill's own last-position logits, and nothing is
decoded or kept (the PrefillOnly paper, arXiv 2505.07203, uses the name the
same way). Two readouts inside it:

* **attention readout** (0 tokens) — locate, choice, absent, score, inject,
  and the recipes (search, compare, bind, requirements, redact);
* **verdict readout** (1 token) — a yes/no or label probability.

## 2. What a prefill-only job needs

| | full inference | attention readout | verdict readout |
|---|---|---|---|
| depth | all 33 blocks | up to the deepest head read (20 on the 9B) | all 33 |
| output head | every step | none | last position only; 2 tokens matter |
| KV cache | kept, grows | thrown away | thrown away |
| attention | flash where licensed | materialized at the tapped layer | same |
| batching | continuous, pays for decode | little gain: prefill is compute-bound | same |
| speculative decoding, MTP, grammar, persistent graph | used | unused | unused |

Already in the engine, each added as a default-off option:
`want_logits=false` (no output head), `truncate_after_layer` (stop after the
layer read), the attention tap, `--lens-locate-only` (loads 20 of 33 blocks,
no output head, no reservation), per-phase attention implementation
(`/v1/extract` runs a flash prefill under a materialized tapped decode).

## 3. The waste: the tap copies the whole attention matrix

`run_lens_locate` prefills the whole prompt in one tapped pass
(`server_lens.cpp`, `clear_slot(0)` then one `build_prefill_graph`). The tap
is the full `kq_soft` of the tapped layer, `[n_kv, n_q, n_head]` in f32 —
every row, all 16 heads — copied to the host and range-checked element by
element. A mode reads **one head and the rows after the document**.

Computed from the shape, not measured (9B, 16 heads, f32):

| prompt tokens P | tap today (16 × P × P) | what a mode reads |
|---|---|---|
| 1,000 | 64 MB | < 1 MB |
| 3,025 | 590 MB | ~1 MB |
| 4,096 | 1.07 GB | ~1.6 MB |
| 10,240 | **6.7 GB**, plus the same again on the host | ~4 MB |

And the pass runs **every** attention layer below the cut materialized, so a
16 × P × P matrix is built (transiently) at each of them as well. Part of the
measured locate latency curve (367 ms at 326 chars → 4116 ms at 8606, slightly
superlinear) may be the tap's copy and scan — unmeasured.

## 4. The fix: split at the document's end

```
[<|im_start|>user\n] [DOCUMENT]  │ ["Answer each of the following questions …
                                  │   "python": Does the candidate know Python?  …   ← all keys
                                  │   … Output ONLY the JSON object"] [<|im_end|> <|im_start|>assistant …]
 pass 1: untapped, flash          │ pass 2: tapped, materialized — ~60–150 tokens
```

* **Pass 1** prefills the template head and the document, untapped. Its K/V
  (and the DeltaNet state) go into the cache as usual.
* **Pass 2** prefills the instruction, the keys and the template tail from the
  document's end, with the tap on. Its rows look back at the document through
  the cache.
* **Why the readout is the same:** attention is causal. The document's rows
  never see the question; the question's rows see the whole document either
  way. Only float noise from the chunk boundary differs.
* **Tap size:** 16 × S × P with S ≈ 100 suffix tokens — ~20 MB at P = 3,025,
  **~65 MB at 10K** (one head: ~4 MB), against 590 MB and 6.7 GB.
* **Precedent in the tree:** `/v1/verify` already runs this shape (untapped
  prompt pass, then a tapped pass over the answer tokens at offset P).
* **Split in token space**, not text: cut the one-shot token sequence at the
  first token past the document, so both passes together are exactly today's
  tokens.

### 4.1 Flash in pass 1

Pass 1 has no tap, so flash may run there. That is what removes the second
O(P²) cost — the materialized matrix at every attention layer — not only a
speed gain. Licensed per model by the drift gate
(`LensConstants::flash_prefill_ok`, architecture.md):

| model | flash prefill |
|---|---|
| Qwen3.8-9B (dense) | PASS on Q8_0: Δpeak 0.00084 vs margin 0.00126 (~1.5× headroom), prefill 8.4% faster |
| Qwen3.8-27B (dense) | PASS on Q3_K_M, 1.8× headroom (thin) |
| MoE (35B, Gemma 26B) | refused: expert selection flips |

Two cautions: the split and flash **stack** — gate them together, on the
locate readouts, not by reusing the extract-path flash result; and the 9B
result is on **Q8_0** — Q4_K_M, the standard, needs its own run.

### 4.2 Keep pass 1: the warm document

Pass 1 is exactly a reusable prefix. **The design already exists for
`/v1/extract`:** `docs/plan-lens-warm-document.md` (PROPOSED, not approved) —
always split at the document boundary, an explicit `document_id`, fail-loud on
changed bytes, `"prefix": "warm"` disclosed. Measured there on the 9B Q8_0
(2026-09-07): split vs one-shot **token-identical 15/15**, citation top-1/top-3
identical, pass-1 prefill **13.5× at 1K, 91× at 8K** on a hit. What this note
adds:

* the same split for `/v1/locate` and the other prefill-only routes — not in
  that plan, and its gates (extraction tokens, citation accuracy) are not the
  locate readouts' gates;
* the tap-memory argument (§3), which pays even with no cache at all;
* **keep documents in memory, not on disk.** `--prefix-cache` is a disk store;
  a cached CV would be personal data on disk.

Where it pays: questions arriving in separate requests (a UI editing keys),
and search setup B (each chunk prefilled once instead of once per question).
One checklist sent at once gains little — all keys already share one prompt
(1 → 15 keys cost +6.7%).

## 5. Which modes fit the split

| mode | rows read | fits |
|---|---|---|
| locate, choice, absent, score | key / question rows | yes, split at the document's end |
| inject | template-tail rows | yes |
| search A (chunks as the document) | question rows | yes |
| search B (one chunk per prompt) | question + N/A rows | yes — the biggest gain with a kept document |
| bind, requirements, redact per-field | question rows | yes |
| redact, instruction tail | tail rows | yes |
| compare | document B's rows on A | yes, split at A's end; pass 2 is B × (A+B), about half the square |
| `/v1/verify` | answer rows | already split |

**Do not fit** — three banked ideas read rows *inside* the document: #2 split
(document rows on each other), #6 coreference (a pronoun's row on its name),
#1 surprise highlighter (per-position output probabilities). They need the
tap, or the output head, in pass 1 itself; they would need their own cheaper
readout (one head, chunked; a per-position log-prob without building P ×
151K logits).

## 6. The verdict readout needs nothing new

Full depth, `want_logits=true` (the head already computes only the output
rows), the tap armed in the same pass. It cannot run on `--lens-locate-only`
(20 blocks); a full lens server serves both readouts, since depth is set per
request. Open, as a probe question: can the verdict be read at L19–L23 through
the output head (logit lens)? Middle layers often carry more than the last
(Skean et al., *Layer by Layer*, ICML 2025). If so, the verdict fits the
locate-only cut.

## 7. Scheduling

PrefillOnly: prefill-only jobs are compute-bound, batching adds latency for
little throughput; run one at a time, shortest job first (the cost is known
from the prompt length). The lens's one-request-at-a-time rule and the
receipts' unbatched rule are already that shape. **Untested:** "span-only
concurrency = N processes" on one GPU — they likely take turns on the GPU
rather than add throughput. Measure 1 vs 2 processes before relying on it.

## 8. Not worth adopting

* **PrefillOnly's MLP chunking and last-layer-only KV** — they fix activation
  and KV peaks of full-attention models on GPUs. The 9B keeps KV on only 8 of
  33 layers; its peak is the attention matrix (§3).
* **BlockRank's block-sparse attention** (arXiv 2510.05396) — linear-cost
  search, but it changes what the model computes and needs fine-tuning; the
  receipt would no longer be the model's own.
* **Batching prefills** — little gain, and it breaks the unbatched receipts
  rule.

## 9. Decisions (user)

1. **Smallest step, byte-exact:** cut the tap to the one head a job reads
   before it is copied out — 16× less memory, same values. Probes keep the
   full tap.
2. **The split for the prefill-only routes**, flash in pass 1 where licensed —
   drift gate on the locate readouts, both languages, Q4_K_M.
3. **A kept document** for the lens routes, in memory — extends
   plan-lens-warm-document.md.
4. **A verdict route** — only after the verdict probe says it is useful.

Before 2: measure latency and memory at 4K / 8K / 10K on today's path, so the
gain is a measurement, not the §3 arithmetic.

## Sources

PrefillOnly, arXiv 2505.07203 (built on vLLM, ~4.6k lines; P(Yes) as the
score; one request at a time, JCT-ordered). ICR, arXiv 2410.02642. SnapKV,
arXiv 2404.14469 (attention of a query window computed apart from the main
pass). BlockRank, arXiv 2510.05396. Layer by Layer, arXiv 2502.02013. vLLM
pooling runner and disaggregated prefill; llama.cpp `--embedding` /
`--reranking` — prefill-only work served by the same engine.
