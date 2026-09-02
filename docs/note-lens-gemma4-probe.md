# Does the Qemmi-Lens citation head exist on Gemma 4? (2026-09-01)

**Verdict: NO-GO for a Gemma 4 lens calibration. Clean NO on constant transfer.**

Two results, both measured on a healthy model *after* a probe defect that had
been silently degrading every Gemma run was found and fixed (§5 — read it before
quoting anything here):

1. **The shipped Qwen 3.6 constants do not transfer to Gemma 4 at all.** L3H13
   scores **0/28 (0.0%)** in-span, rank **312 of 768**. Clean and unambiguous.
2. **No head survives both signals.** The best candidate, L4H7, passes the short
   prompts (75% on selection, **86%** and 66% top1 held-out, probe verdict
   `PASS`) and then **collapses on the independent messy corpus to 41% top1 /
   51% top3** (§6), against a 90% bar that Qwen 3.8's head clears at 98%. That
   is overfitting to the prompt family, and the two-signal rule is what caught
   it. Selection was weak to begin with: ranks 1–4 were a **four-way tie at
   75.0%**.

So: Gemma 4 shows *something* — L4H7 is a genuine within-layer outlier (layer-4
head-mean 0.05 vs its 0.75), and top-3 mass does sit near the source on the short
prompts. But nothing here supports a receipt. **This is a null on the practical
question, and the null is the result.**

**The honest caveat on the first pass.** My first Gemma run reported a
*different* head (L45H12) and a `WEAK` verdict. That run was wrong: the probe
never applied Gemma's mandatory leading BOS, so the model was in its documented
degenerate state. Fixing it changed the selected head, the verdict, and the
teacher-forced agreement (25/28 → 28/28 on Prompt A). The pre-BOS numbers are
retained in §5 only as the record of the defect; **do not quote them**.

**Date** 2026-09-01 · **Status** measurement note; `LensConstants` unchanged,
the server's architecture refusal unchanged (it already refuses `gemma4`).

## 1. Provenance

| | |
|---|---|
| Model | `models/gemma-4-12B-it-Q8_0.gguf`, arch `gemma4`, 48 blocks, Q8_0 dense |
| Attention layers | **all 48** materialize `kq_soft` (no interleave holdout) |
| Heads | 16 ⇒ **768 (layer, head) candidates**, all captured per decode |
| Driver | `tests/perf/attn_provenance.cpp` → `build-release/bin/attn-provenance` |
| Tap | graph-scan on `kq_soft.<il>`, `--flash-attn` OFF |
| BOS | `add_bos_token=true`, id 2, `"<bos>"` — applied (§5) |
| Authoritative logs | `.session-results/gemma4_n3_BOS2.log` (+ `_clean`), `.session-results/gemma4_legC_L4H7.log` |
| Superseded (pre-BOS, do not quote) | `.session-results/gemma4_n3_search.log`, `gemma4_legC_L45H12.log` |

The 26B A4B gemma4 was **not** used: its output is known-incoherent on probe
prompts (architecture.md §12), which would confound the teacher-forced legs
further. The dense 12B was used throughout.

## 2. The search — Prompt A, all 768 candidates

| rank | layer | head | top1/N | in-span | top3/N | bos_mass |
|---|---|---|---|---|---|---|
| **1** | **4** | **7** | 21/28 | **75.0%** | 26/28 | 0.001 |
| 2 | 47 | 0 | 21/28 | 75.0% | 24/28 | 0.042 |
| 3 | 11 | 7 | 21/28 | 75.0% | 23/28 | 0.005 |
| 4 | 7 | 13 | 21/28 | 75.0% | 23/28 | 0.000 |
| 5 | 6 | 5 | 20/28 | 71.4% | 23/28 | 0.007 |
| 6 | 7 | 12 | 20/28 | 71.4% | 22/28 | 0.002 |
| 7 | 11 | 12 | 19/28 | 67.9% | 25/28 | 0.016 |
| … | | | | | | |
| **312** | **3** | **13** ← shipped coords | **0/28** | **0.0%** | 0/28 | — |

**The top of this ranking is a flat crowd, and that matters.** Ranks 1–4 are a
four-way tie at exactly 75.0%; L4H7 wins only on the top-3 tie-break. Compare
Qwen 3.8, where rank 1 was 100% and rank 2 was 92.9% — there the head announced
itself. Here the probe is choosing among four roughly equal candidates, so the
*identity* of Gemma 4's citation head is not settled by this leg even though the
*existence* of one is well supported.

Against that, layer 4's head-*mean* top1 is 0.05 while L4H7 scores 0.75, so L4H7
is a strong within-layer outlier — the signal is real, it is just not unique.

## 3. Held-out prompts — the candidate holds *here* (and only here)

| leg | greedy-agreement | top1 | top3 |
|---|---|---|---|
| Prompt A (selection) | 28/28 | 21/28 (75%) | 26/28 (93%) |
| Prompt B (held-out) | 28/29 | 25/29 (**86%**) | 28/29 (97%) |
| Prompt C (reformat + conflict) | 27/29 | 19/29 (**66%**) | 24/29 (83%) |

Probe verdict: **`PASS`** (frozen top1 86% ≥ 66.7% bar, top3 97% ≥ 80% bar).

Note the shape: top1 *rises* from A (75%) to B (86%). A head selected at 75% that
then scores 86% on unseen data looks like the opposite of overfitting — which is
exactly why §6 matters. **On this evidence alone I would have called this a GO.**
Prompts A/B/C are one corpus family of structurally similar order emails; they
are three samples, not two signals. The messy corpus is the second signal, and it
reverses the conclusion.

## 4. The transfer hypothesis — refuted across families

The Qwen 3.8 leg found that the unmodified Qwen 3.6 constants reproduced Qwen
3.6's own numbers on Qwen 3.8 to within a point. That was a **one-signal claim**
(it rested on a single salvaged run). Gemma 4 is the second, independent test,
and it comes out the other way:

| model | arch | L3H13 in-span top1 | rank |
|---|---|---|---|
| Qwen 3.6 (calibrated) | `qwen35moe` | — (the pinned head) | 1 |
| Qwen 3.8 | `qwen35` | 92.9% | 2 of 128 |
| **Gemma 4 12B** | `gemma4` | **0.0%** | **312 of 768** |

**The constants transfer within the Qwen family and not across families.**
"L3H13 is a retrieval head" is a fact about Qwen-family models, not a general
fact about transformers. This supports keeping the server's architecture refusal
as-is, and it *weakens* the case for widening it into an allowlist — a change
flagged as a decision in the Qwen 3.8 note and not made.

This result survives the §5 defect: it reproduced at 0.0% both before the BOS fix
(rank 277) and after it (rank 312), on a prompt where Gemma's own heads score
75%. The prompt was good enough to find a head; it just was not L3H13's.

## 5. Friction — one defect that invalidated a whole pass

The probe's N3 legs are raw teacher-forced completion (no chat wrapper), which is
why they ran on Gemma 4 unmodified. But three Qwen-shaped assumptions were baked
in, and **the third silently degraded the model rather than failing loudly**:

- **`span_scalars` stack overflow.** It writes `step_src[1 + slot]` into a
  `double[12]` for every tapped layer. At Qwen's 10 layers that fits exactly; at
  Gemma's 48 it runs 37 doubles past the end of a stack buffer. Now guarded
  (indices 1..10 per-layer, 11 = all-layer max, still accumulating over all
  slots). Provably inert at 10 layers ⇒ qwen36 unchanged.
- **`qdocs_chat_prompt` hardcoded `QwenChatTemplate`.** Every QDOCS leg would
  have wrapped a Gemma document in ChatML the model has never seen. Now selects
  `Gemma4ChatTemplate` / `GemmaChatTemplate` / `QwenChatTemplate` by the loaded
  GGUF's architecture.
- **No BOS — the one that mattered.** `gemma4.cpp` sets `add_bos_token = true`
  and both `cli/complete.cpp` and `cli/session_mode.cpp` insert the BOS before
  prefill, but `tok->encode()` does not, and the probe called it directly. A
  Gemma -it model without a leading BOS degenerates. The first pass measured
  that degenerate state end to end.

The BOS fix had to go in as **text**, not as a bare token id: every span here is
found by searching the prompt string and mapping offsets through
`cum_bytes(decode(prefix))`, so prepending a bare id shifts the whole byte map.
The first attempt did exactly that and tripped the probe's own roundtrip guard
(`prompt roundtrip mismatch — token offsets unreliable`) — a fail-loud check
earning its keep. Putting `"<bos>"` in the text keeps tokens and bytes in
lockstep; the body/instruction boundary was widened by the same 5 bytes.

What the fix changed — this is the measurement of how much the defect cost:

| | pre-BOS (degenerate) | post-BOS (authoritative) |
|---|---|---|
| Prompt A greedy-agreement | 25/28 | **28/28** |
| Prompt B greedy-agreement | 21/29 | **28/29** |
| Prompt C greedy-agreement | 13/29 | **27/29** |
| selected head | L45H12 | **L4H7** |
| frozen top1 A / B / C | 93 / 55 / 38% | 75 / **86** / **66**% |
| probe verdict | `WEAK` | **`PASS`** |

A different head, a different verdict, and a Leg C extraction that had been
emitting `"product":"brite"` for three separate fields. **Qwen GGUFs carry
`add_bos_token=false`, so all Qwen legs are byte-identical across this change.**

## 6. Second signal — the messy corpus (Leg C). **L4H7 does not survive.**

`.session-results/gemma4_legC_L4H7.log` — citation **L4H7** (tap-slot 4, head 7,
printed by the run), BOS applied, `Gemma4ChatTemplate`, 15 messy EN+DE documents,
397 scored value tokens.

| metric | Gemma 4 **L4H7** | Qwen 3.8 L27H13 | Qwen 3.6 L3H13 |
|---|---|---|---|
| citation top1-in-span | 161/397 (**41%**) | 369/413 (89%) | — |
| citation top3, EN | 110/209 (**53%**) | 98% | 84% |
| citation top3, DE | 94/188 (**50%**) | 98% | 83% |
| **citation top3, combined** | **204/397 (51%)** ✗ bar ≥90 | 98% ✓ | 84% |

**This is the overfit result the two-signal rule exists to catch.** L4H7 was
selected on Prompt A (75%) and passed the held-out prompts (86%, 66%) — and then
collapses to 41% top1 / 51% top3 on an independent corpus. EN and DE agree
(53% / 50%), so it is not a language artifact. A head that holds on three
structurally similar prompts and halves on real documents is not a citation head;
it is a fit to the prompt family.

**The failure is in the citation signal, not in the generation.** Gemma 4
extracted this corpus about as well as Qwen did — 3/75 values normalized away
(4%) versus Qwen 3.8's 1/75, leaving 72 of 75 labeled values grounded and
verbatim to score. The model found the right values; the attention head just did
not point at where they came from.

### What the other two Leg C numbers do and do not mean

`ungrounded false-alarm 22%` (EN 11%, DE 35%) and `coverage used-clear 19%,
median peak 0.173` both **fail their bars, and both are uninterpretable as
absolutes**: they are computed with `body_mass 0.538` and coverage slot 2 —
Qwen-calibrated constants, and on gemma4 slot 2 is physical layer 2, a layer
nothing ever calibrated. They are reported for completeness and should not be
quoted as Gemma coverage results. The citation row is the load-bearing one, and
it does not depend on those constants.

## 7. The decision this feeds — flash attention vs receipts

`--flash-attn` never materializes `kq_soft`, so a lens and flash attention are
mutually exclusive on the same server — and flash is a **measured 28.8% win on
Gemma 4**. The trade would be: give up 28.8% decode throughput to gain a citation
receipt.

**On these numbers that trade is clearly not worth making.** A receipt whose
top-3 lands on the true source half the time (51%) is not a receipt; the lens's
product claim is that it never lies about where the model looked, and 51% is
much closer to a coin flip than to that claim. Keep flash attention on Gemma 4.

## 8. Decisions raised, not acted on

1. **Do not widen the architecture refusal.** §4 is evidence against the
   allowlist idea floated in the Qwen 3.8 note: the shipped constants are
   worthless off the Qwen family (0.0%, rank 312/768). The refusal correctly
   refuses `gemma4` today and should stay.
2. **Do not add a Gemma entry to `LensConstants`.** There is no head to enter.
3. **Whether to keep pursuing a Gemma 4 lens at all.** This leg says no on the
   current evidence. The cheapest thing that could still change the answer is a
   wider search *scored on the messy corpus directly* rather than selected on
   the short prompts and confirmed afterwards — the flat 4-way tie at the top of
   §2 suggests prompt-A selection has little discriminating power on this model.
   That is a new probe, not a rerun, and it is not obviously worth it.
4. **The probe's Qwen-shaped assumptions are a standing hazard, not a one-off.**
   Three were found in one leg (§5), and the costly one degraded the model
   silently instead of failing loudly. Assume more remain before the next
   cross-family probe is trusted.

## 9. What was NOT done

- **No second BOS-corrected pass on the pre-BOS legs.** Only the N3 search and
  Leg C were re-run after the fix; no other QDOCS leg was re-measured on Gemma.
- **No coverage or ungrounded search on Gemma 4.** `coverage_layer`/`0.705`/
  `body_mass 0.538` were not searched; Leg C evaluates the default slots, which
  on gemma4 point at layer 2 — a *defined* but entirely uncalibrated layer. Do
  not read Gemma coverage numbers as meaningful.
- **No 26B A4B run** (known-incoherent on probe prompts).
- **No images** — `docs/note-image-lens-probe.md` closes that.
- **`LensConstants` unchanged; the server architecture refusal untouched.**
