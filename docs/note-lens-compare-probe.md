# Compare probe — two documents in one prompt: what lines up, what is missing

**STATUS: MEASURED 2026-09-24 on Qwen3.8-9B Q4_K_M. A seventh prefill-only
mode, on heads already landed — proven in the probe, NOT exposed by the
server.** Given an original and its translation, one prefill says which part
of the original is missing from the translation, which part is new, and which
parts correspond. At message level it holds at 100% with every shared word
deleted; at single-sentence level about 8 in 10.

Also known as: document comparison / semantic diff; translation QA (the MQM
"omission" and "addition" error categories); redlining (contract versions);
coverage checking (summaries); bitext / cross-document alignment.

## 1. The question

Every other mode reads a KEY's rows (or the template tail) against one
document. This one reads document B's own rows against document A — a new
readout, with no key. Three jobs:

* **align** — each unit of B → the unit of A it restates;
* **omission** — B drops one unit of A: which? Omission is the lens's one
  unique asset; here it is applied *across* documents;
* **addition** — B carries one unit A does not: which?

## 2. The probe

`COMPAREHEAD=1` (easy leg) and `COMPAREHARD=1` (hardened) in
`tests/perf/attn_provenance.cpp`.

* **Ground truth for free.** DECIDE's 20 routing notes exist as parallel
  EN/DE translations (`f_en1` ↔ `f_de1`, …); six of Leg C's order emails are
  parallel too. A = units in one language, B = the same units in the other,
  **shuffled** so position cannot align them, with one dropped (omission
  trials) or one unrelated unit added (addition trials). Both directions are
  run; each is the held-out half for the other.
* **Prompt.** "Document B is meant to be an English (German) translation of
  document A.", then A as `(A1) … (A8)`, B as `(B1) …`, then a question asking
  for the missing (or extra) label.
* **Readout.** For each B unit, the mean over its own rows of the attention
  mass on each A unit — raw, and per A token ("density"). Align = argmax over
  A; omission = the A unit least covered by all of B; addition = the B unit
  whose best match is weakest. Also tried: the template-tail rows on A and B —
  weaker, not used. 8 layers × 16 heads.

## 3. The easy leg (COMPAREHEAD) — and why it was not believed

8 routing notes per trial, 16 trials per job per direction. Align, omission
and addition all read **100%** for dozens of heads, including every landed
one — but with EARLY heads on top (L3, L7; L3 h=13 is the old citation head
that copies tokens), which is the signature of surface matching. The
translations share invoice numbers, "HTTP 500", "clause 11.2",
"OrderValidator.java". Saturated, and not trusted.

## 4. The hardened leg (COMPAREHARD)

**Every word of B that occurs anywhere in A is deleted** — numbers, codes,
names, shared loanwords — so B keeps only translated words (it becomes
telegraphic, which only makes it harder). Measured overlap after stripping:
**0.0%** of B's words. Numbers are DE→EN / EN→DE.

| leg | job | chance | result |
|---|---|---|---|
| stripped routing notes, 8 messages | align | 12.5% | **100 / 100** (absent, score, locate) |
| | omission | 12.5% | **100 / 100** (absent L19 h=10, score L19 h=11, locate L11 h=6) |
| | addition | 11.1% | **100 / 100** (absent, score) |
| near-twin order emails, 6 messages (one topic, same structure) | align | 16.7% | **100 / 100** (L3 h=13, locate) |
| | omission | 16.7% | **100 / 100** (absent L19 h=10) |
| | addition | 14.3% | **100 / 100** (locate L11 h=6); absent 83 / 83 |
| one SENTENCE dropped inside a message | align | — | 100 / 100 (L15 h=13); landed 88–100 |
| | omission | 7.5% | best L15 h=1 density **100 / 81.2**, held out **81.2 / 87.5**; landed score 88 / 75, locate 75 / 75 |

Held out by direction, message level: selecting on either direction and
scoring the other stays at 91.7–100% for align and addition; omission is
100% on the landed absent head both ways, while the per-direction winner
sometimes transfers worse (58.3–75% for L7 h=1) — the landed heads are the
stable choice.

**Reading.** At message level, the heads already landed tell which part of a
translation is missing or extra **by meaning alone, across languages**. The
limit is granularity: a single dropped sentence is found about 8 times in 10,
and EN→DE (A English, B German) is the harder direction.

## 5. What this is and is not

* **Free in depth, not yet reachable.** The heads sit at L11–L19, inside the
  20-block locate-only cut. But the readout — B's own rows on A — is not
  something `/v1/locate` returns today.
* **Two routes, neither taken:** (1) test whether today's API can do it with A
  as the `document` and each B unit as a question key (key rows attend to A;
  a different prompt shape, unmeasured); (2) a `compare` role on the server —
  an architecture change, the user's decision.
* **Translations only.** Contract versions and summary coverage are the same
  mechanism on paper and untested.
* **Small.** 12–16 trials per job per direction; the message-level legs are
  saturated (they show the capability, not the best head). Corpus self-owned;
  the German routing notes are transliterated (ae/oe/ue).
* **No "nothing is missing" answer measured.** Every omission trial had
  exactly one drop, so the mode always names a unit. Whether a clean
  translation can be told from one with a drop is a separate threshold
  question, as it was for search and absence.

## Reproduce

```
M=$PWD/models/Qwen3.8-9B-Q4_K_M.gguf
COMPAREHEAD=1 QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance   # easy leg, ~3 min
COMPAREHARD=1 QWEN36_MODEL_PATH=$M ./build-metal/bin/attn-provenance   # hardened, ~5 min
```
