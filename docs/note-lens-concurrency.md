# How many users a lens server serves (CONCUR)

**STATUS: 2026-09-27. Qwen3.8-9B Q4_K_M, one Apple M1 Pro (32 GB), engine
fec3db7 + uncommitted document_id/help fixes.** Prefill-only routes only:
locate (all heads), verdict, compare. Documents up to ~3.5K prompt tokens;
10K is out of scope for now. Synthetic English delivery notes, no personal
data. Script: `py/lens_concur.py` (four legs, run against servers you start).

## Answer

**The GPU is the ceiling, and processes do not raise it.** One machine does
about **8 cold ~3.5K locates per minute** or about **200 warm questions per
minute**, whether it runs 1 process or 4. Warm is 25× cheaper than cold, so how
many users a machine serves is decided by how many of them stay warm, and
**today that is 4 per process**: a fifth user taking turns makes every request
cold.

| mix (one machine) | per minute |
|---|---|
| cold ~3.5K locate, new document each time | ~8 |
| warm question on a kept ~3.5K document | ~200 |
| session: one ~3.5K document kept at the absent depth, then 10 questions | ~4 sessions (12.5 s + 10 × 0.35 s) |

## 1. Cost per request (service leg, one full lens server, 3 reps)

Seconds per request, p50 (p95 within 5% everywhere).

| request | ~1.1K prompt tokens | ~3.5K prompt tokens |
|---|---|---|
| locate, cold (L11 cut) | 2.06 | 7.34 |
| absent, cold (L19 cut) | 3.43 | 12.53 |
| locate, warm | 0.25 | 0.29 |
| absent, warm | 0.40 | 0.46 |
| verdict, 1 question, cold (full depth + output head) | 4.52 | 17.90 |
| verdict, each extra question in the same request | +0.28 | +0.44 |
| verdict, warm | 0.30 | 0.35 |
| compare, cold (original + revision ≈ total) | 2.43 | 8.62 |
| compare, warm (original kept, new revision) | 1.13 | 3.82 |

* Cost follows depth: absent (20 blocks) costs 1.7× locate (12), verdict 2.4×.
* **Warm cost barely grows with the document** (0.25 → 0.29 s over 3×).
* Compare warm still pays the revision (it is half the prompt), so it gains
  2.2×, not 25×.
* Ask a verdict all its questions in one request: each extra one is 0.3–0.4 s,
  against 4.5–18 s for a new request.

## 2. Processes on one GPU (scale leg, `--lens-locate-only --slots 1`)

One closed-loop client per process; cold = a new ~3.5K document per request,
warm = each client's own kept document.

| processes | cold req/min | cold latency p50 / p95 | warm req/min | warm latency p50 / p95 |
|---|---|---|---|---|
| 1 | 8.0 | 7.4 / 8.0 s | 183 | 0.29 / 0.46 s |
| 2 | 7.8 | 15.3 / 15.7 s | 203 | 0.52 / 0.84 s |
| 3 | 7.8 | 14.9 / 32.0 s | 203 | 0.84 / 1.47 s |
| 4 | 7.7 | 31.3 / 31.8 s | 201 | 1.03 / 1.77 s |

* **Cold throughput is flat.** Prefill is compute-bound; processes take turns
  on the GPU. This retires the "replicate locate-only for concurrency" advice
  in `architecture.md` §6 for one machine.
* **Processes make cold latency worse than a queue.** Two processes share the
  GPU, so both finish at ~2× (15.3 s). One process with a queue finishes the
  first at 7.4 s and the second at 14.8 s. With three, the sharing is uneven
  (p95 32 s).
* Warm gains 11% from a second process (CPU work overlaps the GPU), then
  nothing.

## 3. One process, 8 clients at once (queue leg, cold ~1.1K locate)

24 requests in 48.9 s = 2.04 s each, the same as alone. Every client waited
~8 service times (p50 16.3 s, p95 16.5 s). **Completion order was strict
round-robin** (0,7,5,6,2,3,4,1 repeated): nobody starved, although
`model_mutex_` promises no order. No timeouts at this depth of queue.

## 4. Kept documents (thrash leg, users taking turns, ~2K documents)

| users | warm turns | mean per turn |
|---|---|---|
| 2 | 6/6 | 0.42 s |
| 4 | 12/12 | 0.43 s |
| 5 | **0/15** | 6.8 s |
| 6 | 0/18 | 7.0 s |
| 8 | 0/24 | 6.8 s |

**A cliff, not a slope.** The store holds 4 entries, least recently used out
first, and users who take turns always evict the next one's document. At 5
users every turn is cold: 16× slower, and the machine drops from ~200 to ~8
requests per minute. A user kept on locate, extract and compare holds 3
entries, so with the warm box on everywhere the cliff should come at 2 users
(inferred from the store's rule, not measured).

## 5. Memory

`footprint` right after start, one locate-only process at `-c 4608`:

| `--slots` | KV cache | footprint |
|---|---|---|
| 10 (the default) | 2880 MB | 6664 MB |
| 1 | 288 MB | 3624 MB |

**A truncated lens server allocates KV for 10 slots and can only use slot 0**
(it refuses chat; every lens route runs single-slot). The default wastes
2.6 GB per process at this context. Run truncated servers with `--slots 1`.

## What it means

1. **One lens process per GPU, with a queue in front.** Replicas belong on
   other machines. The engine already queues fairly in practice; a client that
   wants priority or a shortest-job-first order must build the queue itself.
2. **The binding limit is the kept-document store, not the GPU.** At 4 users
   one machine gives ~200 warm answers a minute; at 5 it gives ~8. Raising the
   store is the lever: now `--lens-kept-documents N` (see below).
3. **`--slots 1` on every truncated lens server** (−2.6 GB each).

**Landed after this note (2026-09-27, user decision): `--lens-kept-documents N`**
(default 4, unchanged behaviour). Thrash rerun on a locate-only `--slots 1`
server with N = 8: **2/4/5/6/8 users all warm** (6/6, 12/12, 15/15, 18/18,
24/24; 0.39–0.42 s a turn), where N = 4 had gone 0/15 at 5 users. Footprint
3625 → 6781 MB after the run (+3.2 GB for 8 entries at ~2K tokens), against
an entry size of ~64 KB per token + ~48 MB DeltaNet state (~175 MB at 2K),
computed from `serialize_slot`; the gap is not explained yet.

**Decisions for the user (nothing changed):**
* whether truncated modes should force or default to one slot — a flag
  semantics change;
* the architecture.md §6 replication sentence, which this note contradicts.

## Reproduce

```
M=models/Qwen3.8-9B-Q4_K_M.gguf
./build-metal/bin/qwenium-server -m $M --attention-lens -c 4608 -p 18140
python3 py/lens_concur.py service --port 18140
python3 py/lens_concur.py queue   --port 18140
python3 py/lens_concur.py thrash  --port 18140
# 4 × ./build-metal/bin/qwenium-server -m $M --attention-lens --lens-locate-only --slots 1 -c 4608 -p 1813{0..3}
python3 py/lens_concur.py scale --ports 18130,18131,18132,18133 --reps 4   # and the 1..3 subsets
```
