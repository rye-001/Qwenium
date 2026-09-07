# Plan — the warm document (`document_id` on `/v1/extract`)

Status: **PROPOSED, not approved. All correctness probes PASS and the speed is measured — see §8. The saving is 13x at 1K and 91x at 8K on pass-1 prefill.** Touches a named seam
(`lens-format.md`, `/v1/extract`) and the prefill path that produces every
receipt the lens reports, so it needs explicit user approval and an
`architecture.md` update in the same change.

Origin: *"if a user opens the UI and works a single document, changing keys or
questions several times, do they benefit from a warm backend?"* Today: **no, not
at all.** This plan says how they would.

---

## 1. The waste, stated exactly

`run_lens_tapped_decode` does `clear_slot(0); set_cache_pos(0,0);` and prefills
the whole prompt, unconditionally. There is no `conversation_id`, no
`PrefixLibrary`, no warm path anywhere in `server_lens.cpp` — the single
occurrence of the word is a comment reading *"This is NOT warm."*

So every `/v1/extract` re-reads the entire document from scratch, and with
`"candidates": true` it does so **twice** (pass 2 re-prefills the same document
under a different instruction).

The prompt is built as `{"user", document + instruction_suffix}` — **document
first, instruction last**. Define:

```
P      = "<|im_start|>user\n" + document      <- thousands of tokens, INVARIANT
suffix = instruction + assistant tag          <- ~40 tokens, changes every edit
```

Changing a key, adding a question, or toggling a gloss changes only `suffix`.
This is a textbook strict-prefix case — the same shape `--prefix-cache` measured
at **2.4x @4K** on the chat path (PR #22) — and the lens takes none of it.

## 2. Design

### 2.1 Always split the prefill at the document boundary

Prefill `P` as its own chunk, then `suffix`, **unconditionally** — on the first
request, on a cache miss, with no `document_id` at all.

This is the load-bearing decision. If the split were conditional on a cache hit,
the *first* extraction of a document (one-shot prefill) would report different
masses from every later one (restore + suffix), because chunked and one-shot
prefill are not bit-identical on Metal. That is a permanent two-path system and
a genuinely nasty surprise: the same document, the same keys, a different answer
the second time.

Unconditional splitting makes **warm == cold by construction**. Every pass 1 runs
the identical numeric path whether or not it hit the cache, `document_id` becomes
purely a cache key rather than a semantics switch, and the gate collapses from
*"maintain a warm/cold delta forever"* to a **one-time** comparison of the new
split path against today's one-shot path. Cost on the first request is two chunks
over the same tokens — negligible.

### 2.2 `document_id`, explicit

```json
{ "document": "...", "document_id": "lease-4471", "key_vocabulary": [...] }
```

Optional. Absent ⇒ today's behaviour exactly (still split, but nothing stored or
restored).

- **Key the snapshot by `(document_id, weights_hash, build_path_tag)`** — the key
  discipline `PrefixLibrary` already enforces and already fails loud on (PR #21).
  A snapshot from other weights or another build path must never be restored.
- **Store the document's hash with the snapshot.** If an id returns with
  *different* document bytes, that is a **fail-loud `400`**, never a silent stale
  hit. Serving a cached prefix for the wrong document would mean reporting
  receipts about text the model did not read — the worst failure this format has.
- **Disclose it.** The response carries `"prefix": "warm"` when a snapshot was
  restored, so a warm citation mass is never unknowingly compared against a cold
  one. Same discipline as `uncalibrated` (docs/plan-question-keys.md).

Why explicit rather than transparent: the caller knows when a document is "the
same" across an edit and the server does not, eviction becomes the caller's
mental model rather than invisible server state, and this repo has already
deleted one buggy transparent prefix cache and re-added it as an opt-in
(`--prefix-cache`, F9).

### 2.3 What each call does

| call | cache | prefills | benefit |
|---|---|---|---|
| **Req#1 Pass#1** | miss — writes the snapshot | `P` + suffix | none (the investment) |
| **Req#1 Pass#2** | miss | `P` + suffix' | none |
| **Req#2 Pass#1** | **hit** — restore `P`, prefill suffix | ~40 tokens | **the whole document prefill** |
| **Req#2 Pass#2** | see §3 — undecided | ? | ? |

Candidates **off**: every iteration after the first is essentially free on
prefill. Candidates **on**: at least one of the two prefills is saved, and §3
decides whether both are.

## 3. Pass 2 — open, and the earlier reasoning was wrong

An earlier version of this design warmed pass 1 only, arguing pass 2 should stay
cold because disarming its taps is what makes flash attention available, and that
warming it would "trade that saving away".

**That argument does not survive inspection, and it was asserted without
measurement.** Two corrections:

- **Warming pass 2 does not give up flash attention.** Flash vs non-flash is how
  attention is *computed*; the KV cache is just K and V tensors, and the flash
  path can consume a restored prefix perfectly well. A warm pass 2 would restore
  `P`, prefill a ~40-token suffix, and **decode with flash exactly as it does
  today**. Flash's benefit on a 40-token prefill is negligible anyway. So the
  supposed trade is largely fictional.
- **The magnitudes were backwards.** Flash makes part of a prefill cheaper; a
  cache hit removes ~99% of the prefill's tokens. A partial reduction on part of
  the work does not beat near-total elimination of it, and the gap widens with
  document length.

What is **genuinely unknown** is narrower, and it is not the performance
question:

> Does pass 2, resuming from a prefix whose K/V were computed on pass 1's
> **armed** (non-flash) path, produce a different candidate set?

Different rounding ⇒ different generated tokens ⇒ possibly different spans. This
is *permissible* in a way pass 1's equivalent is not: pass 2's output is
byte-exact document slices verified against the document, not attention-derived
receipts. A numeric shift changes which spans are offered, not whether a reported
mass is honest. So it is a gate question, not a contract violation.

It also makes the design **tidier** if it works: one snapshot per document
instead of the same document living in two numeric versions inside one request.

**The same first-request discipline applies.** On Req#1 Pass#2 there is nothing
to restore. Either it loses nothing and warms on Req#2 (two paths, first request
differs — the thing §2.1 exists to prevent), or pass 2 picks one path and uses it
always. If warm wins, pass 2 goes warm unconditionally and eats a negligible
first-request cost to keep every request identical.

### 3.1 THE PROBE — `WARM2=1`

Extends the existing `CAND=1` arm in `tests/perf/attn_provenance.cpp`. No server
work, no format change, byte-inert when unset — same shape as `QKEY=1`.

**Arms**, over `qdocs_messy_corpus` (15 docs EN+DE):

- **Arm C (control).** Pass 2 exactly as today: cold, full `P` + suffix prefill,
  taps disarmed.
- **Arm W.** Pass 2 resuming from pass 1's document prefix, prefilling only the
  suffix, decoding with taps disarmed as usual.

**Metrics**

1. **Candidate-set identity** — for every key, is Arm W's candidate list
   *identical* to Arm C's (same spans, same order)? Report exact-match rate, and
   list every document where they differ.
2. **Gate 2 under Arm W** — median set size on uncontested keys. Must still be
   **1** (`0=0 1=71 2=2 3+=2` is the current shape).
3. **Byte-exactness under Arm W** — must stay 100%. A restored prefix must not
   produce spans that fail to resolve.
4. **Wall-clock per document**, both arms, prefill and decode reported
   separately. Expected to favour Arm W; measured so nobody has to trust that.

**Bars**

| # | bar | if it fails |
|---|---|---|
| 1 | Arm W gate 2 median == 1 | pass 2 stays cold; warm pass 1 only |
| 2 | Arm W byte-exactness == 100% | pass 2 stays cold — a restored prefix that breaks span resolution is disqualifying |
| 3 | Arm W is faster than Arm C | pass 2 stays cold; there is no reason to change a path for nothing |
| — | candidate-set identity | **not a bar.** Reported, not gated: a changed set is allowed if it still clears 1 and 2, since pass 2's output is verified against the document rather than trusted. A *large* divergence is a reason to look harder, not an automatic kill. |

Bars 1 and 2 are the real gates. Identity is diagnostic.

## 4. Gates for pass 1

The split path is a change to how every receipt is produced, so it needs the
fork treatment this repo has applied twice already (`feed_tokens`; the Metal
mm-vs-mv fork):

1. **Token stability** — split-path pass 1 emits the same field values as
   one-shot pass 1 on the full messy corpus. Bar: identical.
2. **Mass drift ceiling** — citation masses may move, but citation **top-1 and
   top-3 accuracy must not regress** against the published Qwen 3.8-9B numbers
   (92%/99% as re-measured 2026-09-07 under the QKEY control arm).
3. **Warm == cold** — a restored pass 1 and a cold split pass 1 produce
   byte-identical reports. This one should hold *by construction* under §2.1; if
   it does not, the design is wrong and the split is not where it is claimed.

Gate 3 is the cheap one and should run first: it is a pure consistency check that
needs no baseline.

## 5. What this does not change

Every non-claim survives. Warmth is a statement about how the prefix was
computed, not about what the lens knows. `validated_envelope` and the published
citation numbers were all measured cold and one-shot, so the gates above must
**establish** the split path rather than inherit those numbers.

## 6. Architecture triggers

`architecture.md` §11 (receipts constraints — receipts-grade determinism is
per-config; the split path is a new config and must be named as one), §13, and
`lens-format.md` (the `document_id` request member and the `prefix` response
member). No version bump: both are additive and absent when unused.

## 7. Open

- Snapshot lifetime and eviction. `--attention-lens` is single-slot exclusive, so
  there is no slot contention, but there is still a memory budget and no stated
  policy for how many documents stay warm.
- Whether `document_id` should also warm the **question** vocabulary path. It
  should — the prefix is the document, and the vocabulary lives entirely in the
  suffix — but that composition is untested and both features are unapproved.

---

## 8. Measured — 2026-09-07

### 8.1 Correctness (`WARM1`, `WARM2`) — all six checks pass

Harness-only arms, `build-metal` (Release + Metal), Qwen 3.8-9B Q8_0,
`qdocs_messy_corpus`.

| check | result |
|---|---|
| warm == cold (pass 1) | 15/15 exact |
| token stability, split vs one-shot (pass 1) | 15/15 exact |
| citation top-1 / top-3 | 92% / 99%, **identical across all three arms** |
| candidate-set identity, warm vs cold pass 2 | 75/75 |
| gate 2 under warm pass 2 | median 1.0, `0=0 1=71 2=2 3+=2` |
| byte-exactness under warm pass 2 | 81/81 |

**§2.1's premise was wrong, in our favour.** This plan assumed chunked and
one-shot prefill are not bit-identical on Metal, and built the always-split rule
as a defence against a warm/cold delta. Measured, the split path is
**token-identical** to one-shot (15/15). The defence is still worth keeping —
always-split costs nothing and removes a whole class of first-request-differs
surprise — but it is now belt-and-braces rather than load-bearing. Note the
measurement is token-level identity of emitted fields, which is the bar this repo
uses (the mm-vs-mv fork settled on token-stable), **not** bitwise logit identity;
do not write "byte-identical".

**Warming pass 2 is cleared.** It changes nothing measurable about the candidate
set, so **option 1 (both passes warm) is on the table**, not just option 2.

### 8.2 Speed (`WARMPERF`) — pass 1 prefill, by document length

`WARM2`'s wall-clock was discarded: its timers disagreed with each other (a
"suffix-only" prefill timed slower than the full-document prefill it replaces,
and an armed prefill timed faster than a disarmed one). `WARMPERF` re-measures
with one timer scope for every arm (strictly around `run_prefill`), an untimed
warmup sample, medians over 5 reps, and synthetic documents at controlled
lengths.

| document | cold full prefill | warm suffix prefill | saving |
|---|---|---|---|
| 1 K | 5254.7 ms (1234 tok) | 390.0 ms (82 tok) | **92.6%** (13.5x) |
| 2 K | 9980.9 ms (2190 tok) | 408.7 ms (82 tok) | **95.9%** (24.4x) |
| 4 K | 20676.2 ms (4097 tok) | 456.3 ms (82 tok) | **97.8%** (45.3x) |
| 8 K | 41132.5 ms (8128 tok) | 452.4 ms (82 tok) | **98.9%** (90.9x) |

**Internally consistent, unlike the numbers it replaces.** Cold scales with token
count (4.3 -> 5.1 ms/tok, mildly superlinear as attention grows); warm stays flat
at ~400 ms across an 8x document range because the suffix is always 82 tokens.
That flatness is the signature the measurement is real.

**Read the RATIO, not the absolutes.** The implied prefill throughput is
~200-240 tok/s, which is low for a 9B Q8_0 on this hardware and is **not
explained**. A live `http_server` was holding the same GPU earlier in the
session and had exited by the time the run was checked, so whether this run was
contended is unknown — do not record contention as the cause without measuring
it on an idle host. What the ratio does not depend on: both arms ran
back-to-back under identical conditions, and the cold arm's per-token cost is
stable across four document sizes, so the *relative* saving stands regardless of
what sets the absolute floor. The millisecond figures should not be quoted as
this engine's prefill speed.

**Prefill only.** Decode is unchanged by warming and costs the same in both arms,
so the end-to-end saving per edit is this saving over (prefill + decode) at the
caller's generation length. Pass 1 emits a small JSON object, so prefill is the
dominant term on any document in the 4-10K envelope — which is exactly where the
UI's key-editing loop lives.

### 8.3 What is still not measured

- Real snapshot save/restore through `PrefixLibrary`. Every probe simulates the
  warm hit in-process, which isolates the numerics but validates none of the
  storage, keying, or eviction plumbing.
- Decode-side cost, and therefore the true end-to-end per-edit figure.
- Any document above 8K, and any model other than Qwen 3.8-9B.
