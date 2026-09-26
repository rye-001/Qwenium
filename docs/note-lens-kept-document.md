# The kept document — step 6, and the extract warm path it exposed

**STATUS: LANDED (uncommitted) 2026-09-26, approved by the user ("go with step
6, locate only", then "go with option 2" for the extract fix).** A
`document_id` now keeps a document's pass in RAM for later requests, on
`/v1/locate` and `/v1/extract`, as a **snapshot** (KV and DeltaNet state). On
Qwen3.8-9B Q4_K_M a warm report is **bit-identical** to a cold one, and a 10K
follow-up locate drops **22.4 s → 0.30 s**. Building it exposed that
`/v1/extract`'s older warm path, and its candidates pass on every request, gave
**wrong results** on every lens model; both now use the same snapshot store.

## 1. The idea

A lens prompt is `document + instruction`. Every row a lens mode reads sits
after the document, and attention is causal, so the document's pass depends on
the document alone (step 4, docs/note-lens-split-prefill.md). A caller asking
several questions of one document — the UI editing keys, a CV checked against
requirements — can keep that pass and pay only for the instruction.

The model is a hybrid: attention layers keep a KV cache (append: rows can be
rewound) and DeltaNet layers keep a recurrent state (overwrite: the state after
a token cannot be taken back). Keeping a document therefore means keeping
**both**, as a snapshot taken right after the document pass.

## 2. What changed

* **`LensDocumentStore`** (`server_lens.h`/`.cpp`): RAM only, never disk, never
  logged. At most 4 documents, least recently used dropped first, 15-minute idle
  TTL checked on every `/v1/locate` and `/v1/extract` — fixed defaults
  (`kLensDocumentStoreMax`, `kLensDocumentStoreTtl`), not flags. One store per
  server, owned beside slot 0 and guarded by `model_mutex_`.
* **Hit / miss / refusal**, never a stale hit:
  * the only hit test is exact equality of the document pass's tokens;
  * an id sent again with different document bytes is a **400** (a 64-bit hash
    only chooses 400 over a plain miss; it never decides a hit);
  * the same document with a different boundary token is a miss that recomputes;
  * a document kept for a deeper head serves a shallower read (layers under the
    cut are the same computation whatever runs above them); the reverse is a
    miss that recomputes and keeps the deeper one;
  * entries are keyed per **route** (`LensKeptRoute`): locate computes the
    document pass as a truncated tapped graph, extract as a full `run_prefill`,
    so neither serves the other.
* **`/v1/locate`**: `document_id` accepted (1–256 characters); refused on a
  model row without a split licence (Q8_0 is one-shot — there is no document
  pass to keep). Report member `prefix`: `"cold"` (computed and kept) or
  `"warm"` (restored), only when an id was sent.
* **`/v1/extract`**: the old mechanism (`LensWarmDocument`: keep slot 0, rewind
  the position) is **deleted**; the document pass is kept and restored through
  the store (`LensDocPass`). The candidates pass resumes from the same snapshot.
  Wire format unchanged (`"prefix":"warm"` when warm, absent otherwise).
* `qinf-server` links `qinf-snapshot` (`capture_slot` / `restore_slot`).
* Docs: architecture.md, lens-format.md, plan-lens-warm-document.md (header box).

## 3. Locate gate (LOCWARM, 9B Q4_K_M, licensed shape split+flash)

| check | result |
|---|---|
| G1 warm == cold, bit for bit (every span, mass, peak) — 5 heads × 23 documents (15 order e-mails EN/DE key mode, 6 CVs EN/DE question mode, ~4K and ~8K long) | **115 / 115** (first id request also 115 / 115) |
| G2 depth: kept at L19 serves an L11 read, == cold | **23 / 23** |
| G2 depth: kept at L11 misses an L19 read, then warms | **23 / 23** |
| G3 key mode then question mode on one id | warm 15 / 15 |
| G4 one-shot shape + id refused | pass |

| 10,045-token prompt | cold | first id request (stores) | warm | kept |
|---|---|---|---|---|
| locate (L11) | 22.4 s | 23.1 s | **0.30 s** (73.5×) | 673.7 MB |
| absent (L19) | 37.7 s | 39.6 s | **0.45 s** (83.9×) | 673.7 MB |

## 4. The extract warm path was wrong (EXTWARM)

Found while documenting step 6. Extract's warm path rewound slot 0's position:
that restores the KV but not the recurrent state, which by the next request has
absorbed the previous instruction and decode (and is zeroed when the slot is
cleared). Arms per document, candidates on: cold (no id); edit loop (keys K1
with id, then K2 with id); clobber (K1 with id, a `/v1/locate` on another
document, K2 with id).

| | before the fix | after |
|---|---|---|
| edit loop: report == cold | **0 / 6** (citations moved; model output changed on 2 / 6) | **15 / 15** (EN and DE) |
| locate in between: report == cold | **0 / 6** (output changed 6 / 6, still reporting `"prefix":"warm"`) | **15 / 15** |

Why the earlier gate missed it: plan-lens-warm-document.md §8.1 measured warm ==
cold 15/15 by priming the document and prefilling the instruction with **no
decode between** — a sequence the server never runs.

**The candidates pass had the same flaw on every request, id or not**: it
rewound after pass 1 had decoded and cleared the slot, so it read its
instruction with no document in the recurrent layers. CAND's published gates
had measured a fresh one-shot pass 2, not the shipped path. The shipped path,
now resuming from the snapshot, gated directly (15 documents, CAND's concepts):

| gate | bar | result |
|---|---|---|
| median set size, 75 uncontested keys | 1 | **1.0** (0=0 1=73 2=1 3+=1) — pass |
| byte-exactness of candidates | 100% | **78 / 78** — pass |
| producer failures | 0 | 0 |
| same candidate list as a one-shot pass 2 | reported | 105 / 105 keys |

An id reused for different text is now the 400 lens-format.md always promised;
the old code silently ran cold.

## 5. Live checks

* **Locate-only servers.** On Q4_K_M, two requests with one id: the first
  reported `split+flash` / `cold`, the second `warm`, with the same span.
  Reusing the id for another text gave a 400. On Q8_0, the id was refused
  with a 400.
* **A full lens server (Q4_K_M).** The sequence was a cold extract, an id
  extract, a locate on another document, then a warm extract. The warm
  report's fields, candidates and raw output were identical to the cold one.
  Reusing the id gave a 400.
* **Server logs** contained neither document text nor ids.

Full suite: 1022/1022, plus the 18 HTTP integration tests run serially (they
share port 18080 and hang under `ctest -j`).

## 6. What this is and is not

* **Memory is the price.** A kept 10K document is ~674 MB: the snapshot holds
  the f32 KV of all 8 attention layers (including layers above a locate's cut)
  plus the DeltaNet state. Four at 10K ≈ 2.7 GB. Capturing only the layers a
  route reads is an open lever, as is a lower cap.
* **The first request pays a little more** (the capture: +0.7 s locate, +1.9 s
  absent at 10K). Extract with candidates now captures on every request (the
  candidates pass needs it) — not timed here.
* **Idle expiry is checked on the next lens request**, so an idle server holds
  its last documents until something calls it.
* **Gated on the 9B Q4_K_M only.** Other rows: locate refuses an id wherever the
  row is one-shot; extract keeps on any model, gated only here.
* The lens app (`../qemmi-lens`) sends `document_id` to `/v1/extract` behind its
  warm checkbox and says in `locate.ts` that `/v1/locate` refuses it; that
  comment is now stale (app repo, not changed).

## Reproduce

```
M=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf
LOCWARM=1 QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance    # ~13 min + 10K timing
EXTWARM=1 QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance    # ~25 min
```
