# SS3 — Gate 2 powered, phrasing confirmed, and the serveable alarm REFUTED (2026-09-05)

**Verdict: the alarm works and cannot be shipped as specified.** Three results,
in descending order of how much they change the plan:

1. **The serveable predicate FAILS.** The alarm SS2 validated uses corpus ground
   truth that `/v1/extract` does not have. Scored side by side on the same run,
   the predicate a server *could* compute false-alarms on **78%** of correctly
   handled corrections (precision 22%). §3 says why, and the reason is
   conceptual, not statistical.
2. **Gate 2 is powered and the alarm holds** on the oracle arm: TP=3 FN=0 FP=0
   TN=9. With SS2, recall is 4/4 and the FRESH false-alarm rate is **0/16**.
3. **SS2's phrasing hypothesis survived a matched-pair test**: STALE 3/6 on
   indirect corrections, **0/6** on explicit ones, with everything else held
   identical within each pair.

**Date** 2026-09-05 · **Status** measurement note; zero `src/` edits from this
leg. Driver `tests/perf/attn_provenance.cpp`, path `SS3=1` (additive, env-gated,
shares `run_ss2()` verbatim — SS2's own eight threads are unchanged and still
reproduce under `SS2=1`). Raw log `.session-results/ss3_matched_pairs.log`.

## 1. Design — matched pairs, because a pile of hard cases proves less

SS2 rested on **one** true positive and named a hypothesis from it: its single
STALE was late + indirect + distractor, while `t_en5` — equally late, equally
distracted, but phrased **explicitly** — survived, so *indirect phrasing is the
discriminator, not position*.

The obvious follow-up (build 8–10 more threads with the hard profile) would have
powered recall and left the hypothesis exactly as open as it was. SS3 instead
builds **six matched pairs**: six unused seeds from the same messy corpus, each
assembled twice into threads identical in seed, corrected concept, corrected
value, distractor text, position (late, 91–92% through the thread) and target
length (5400 tokens), differing **only** in how the correction is phrased. A
split is then attributable; a non-split would have falsified the hypothesis
rather than leaving it standing on one case.

| | |
|---|---|
| Model | `models/Qwen3.8-9B-Q8_0.gguf`, arch `qwen35`, Q8_0, 33 blocks |
| Citation head | **L27H13** (tap slot 6 / head 13), `ATTN_FROZEN_SLOT=6 ATTN_FROZEN_HEAD=13` |
| Decode | FREE — no grammar; tolerant parse, the shipped lens contract |
| Threads | 12 (6 EN, 6 DE), 24–25 messages, 5403–5753 tokens |

## 2. Results

### Gate 0 — citation at thread scale · **PASS**
```
value tokens scored: 317   top1 88% (280/317)   top3 100% (316/317)   [bar top3 >= 85%]
excluded (normalized / not emitted verbatim): 4 of 64 labeled fields
```
SS2 measured 89% / 98% on eight threads; single-document Leg C measured 89% / 98%.
Three independent corpora, one head, the same numbers. **Thread scale costs
nothing** is no longer a single observation.

### Gate 1 + phrasing split — the hypothesis holds
```
explicit correction: STALE 0/6
indirect correction: STALE 3/6
```
All three discordant pairs point the same way, and no pair points the other way.
**Read the strength honestly:** six pairs with perfect direction is exact-test
p ≈ 0.13 (McNemar, the matched-pair test this design calls for) or p ≈ 0.09
unpaired. That is a hypothesis that *survived an experiment built to kill it*,
not a rate. What it licenses is a generator — if more stale cases are needed,
phrase the correction indirectly — and a product warning, not a probability.

The three failures are `en4_ind`, `de2_ind`, `de6_ind`; their explicit twins
`en4_exp`, `de2_exp`, `de6_exp` are byte-identical apart from the correction
sentence and were all handled correctly.

### Gate 2 — two arms, one of which is the product
```
ORACLE     TP=3 FN=0 FP=0 TN=9   precision=100%  recall=100%  FRESH false-alarm= 0%
SERVEABLE  TP=2 FN=1 FP=7 TN=2   precision= 22%  recall= 67%  FRESH false-alarm=78%
```
- **ORACLE** = SS2's predicate: the cited message is superseded for this field,
  where "superseded" comes from corpus ground truth listing which messages state
  a value for the concept. Combined with SS2: **recall 4/4, false alarm 0/16.**
- **SERVEABLE** = the field's citations reach a message *later* than the one the
  emitted value was drawn from. Uses only shipped signals plus caller-declared
  message boundaries — i.e. everything a v3 endpoint would actually have.

**Why both arms were scored.** SS2 scored only the oracle, and an oracle result
is not a product claim. The same class of defect has bitten this probe twice
before (SS1's grammar recommendation that was four days older than its own
refutation; the `L3H13_SLOT` override that was a silent no-op), so the second arm
was added *before* the run rather than discovered after it.

## 3. Why SERVEABLE fails — and it is not noise

Read the FRESH rows. On nine of twelve threads the model handled the correction
correctly, citing message 21 or 22 — the correction itself. The citations for
that same field also reach message 23 or 24:

```
[en4_exp] quantity  emitted=700   FRESH  cited_msg=21  latest_cited=23  -> oracle:silent  serveable:ALARM
[de6_exp] unit_price emitted=29,80 FRESH cited_msg=21  latest_cited=23  -> oracle:silent  serveable:ALARM
```

Message 23 is the **distractor** — the reply that restates the pre-correction
value. Of course the citations reach it: it states a value for that very key. The
model looked at both and chose the correct one. **The serveable predicate turns
"the model considered the old value" into "the model used the old value" — it
converts consideration into commitment, which is the one thing
`lens-format.md` §Non-claims forbids by contract.** The 78% false-alarm rate is
that error being measured, not a threshold needing a tune.

The inverse failure is on the same page. `de2_ind` is genuinely STALE and the
serveable arm stays **silent**:

```
[de2_ind] unit_price emitted=0,12  STALE  cited_msg=23  latest_cited=23  -> oracle:ALARM  serveable:silent
```

The model emitted the stale value and cited **message 23 — the distractor**,
which is *later* in turn order than the correction at 21. The emitted value's
source is the latest cited message, so "cites something earlier than the latest"
is false exactly when the answer is wrong.

**The finding, stated once:** *turn order does not identify supersession.* A
later message can restate an old value, and in a real reply chain it routinely
does — quoted history is what makes threads adversarial in the first place
(SS2 §1). The oracle works because it knows **which message is authoritative for
a key**, and that is a semantic fact about content. A message index does not
carry it, and no threshold over message indices will recover it.

## 4. What this does to the plan

`docs/handoff-qemmi-lens-revisit.md` §4.1 assumed the missing input was **message
boundaries in the request**, and that specifying them was the first thing to do.
That premise is now measured false: boundaries are necessary and **not
sufficient**. The blocking question is not the request shape, it is *what
mechanically identifies the authoritative message for a key* — and that must be
settled before any `lens-format` v3 is designed, or the format will ship a field
the engine cannot honestly fill.

Three candidates, none measured, listed in the order they should be considered:

1. **Per-message extraction (N+1 passes).** Extract each message's own stated
   values for the hinted keys, then report: *"message 3 says 0.14, message 22
   says 0.19; your value came from message 3."* Every part is mechanical and
   contract-safe — it is the format's existing conflict rule (coexisting keys,
   separate citations, no winner named) with turn indices attached. The citation
   is what makes it more than a diff: only the lens can say *which* of the two
   the emitted value came from. **Cost:** N+1 prefills, and this repo deleted an
   N+1 design once already when the presence gate died — reintroduce it with
   eyes open, and price it against the warm prefix cache (handoff §4.2) first.
2. **Mass- or coverage-gated citation spread.** Restrict the spread test to
   citations that clear a consultation bar rather than counting every top-3
   position. Cheap to measure on the existing runs. Caution: COV1 coverage was
   excluded from this alarm *by design* because SS1 found it structurally blind
   to the attended-but-ignored mode, which is precisely the mode here.
3. **Ship the message index and no alarm.** `/v1/extract` gains message
   boundaries and every citation reports which message it landed in — *"this
   value came from message 3 of 24"* — and the human draws the supersession
   conclusion. This claims nothing the lens cannot support, is a real product
   step on its own (an embedding baseline cannot say which message the emitted
   value came from), and does not block on (1) or (2).

## 5. What was NOT done
- **No third leg, one model, one run.** 12 threads on Qwen 3.8-9B.
- **Gate 2's positive class is n=3** (n=4 with SS2). Better than n=1; still not a
  rate. The 0/16 false-alarm figure is the better-supported half, as in SS2.
- **No `src/` edits from this leg**, and no product or server work. (The
  per-model `LensConstants` change landed the same day as a separate, approved
  architecture decision — see `architecture.md` §6.)
- **Candidates 1–3 in §4 are unmeasured.** They are the next probe's job.
- **The 8.3 K OOM (handoff §4.4) is still only avoided**, not resolved: threads
  were capped at 5400 tokens again.
