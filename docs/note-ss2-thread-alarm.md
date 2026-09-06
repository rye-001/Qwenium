# SS2 — coverage-free thread stale-source alarm: Gate 0 PASS, alarm fires (2026-09-04)

**Verdict: ALARM VALIDATED, with one honest asterisk — recall rests on a single
STALE case.** SS1's capstone was INCONCLUSIVE, blocked on three walls. All three
are now down:

| SS1 wall | SS2 result |
|---|---|
| Thread-scale citation unmeasurable (43% top1 / 71% top3, confounded) | **89% top1 / 98% top3 over 207 value tokens — identical to single-document performance.** Gate 0 PASS (bar 85%) |
| No natural STALE (0 cases at 764 tokens) | **1 STALE in 8 correction instances at ~5–6 K tokens.** Stale is real at length |
| Long context never reached (max 764 tok) | 8 threads at **4774–6200 tokens**, EN + DE |

**The load-bearing result is Gate 0, not the headline.** Thread-scale citation
is not degraded *at all* — 98% top-3 matches the single-document figure to the
point (404/413 = 98%). The retrieval head transfers to multi-turn threads
cleanly. SS1's poor 71% was the extractor confound plus a weaker head and quant,
not a property of threads. **That question is now answered.**

**CORRECTION (2026-09-05), after the trivial-rule control leg — read this
before quoting any Gate 2 number.** The originally reported Gate 2 figures
(TP=1 FN=0 FP=0 TN=7 ⇒ precision 100%, false-alarm 0%) are an **ORACLE**
measurement: they use corpus ground truth that `/v1/extract` does not have and
**cannot compute**. Recomputed with only what a real endpoint would see —
citations plus caller-declared message boundaries — the **SERVEABLE** predicate
scores **TP=1 FN=0 FP=2 TN=5 ⇒ precision 33%, FRESH false-alarm 29%**. Two false
alarms in seven correctly-handled corrections. That gap is the cost of shipping
the alarm, not noise, and 33%/29% is the number a product decision must use.

Two further limits found by the same leg:
- **The disambiguation claim is UNTESTED.** Only **15 of 207** scored tokens had
  the value appearing in ≥2 places at all, below the pre-set floor of 20, so the
  trivial-rule control **could not run**. This corpus does not contain enough
  genuine ambiguity to test whether the head resolves near-duplicates or merely
  follows a positional rule. An earlier claim in this note that the threads are
  "adversarial by construction" is therefore **not supported by measurement** —
  quoted history repeats the *messages*, but rarely the *scored values*.
- **On the one inspectable case the head matches two of three trivial rules.**
  `t_de3`: head cited message 0; `R_last` predicts 0 and `R_near` predicts 0;
  only `R_first` (24) differs. n=1, so this refutes nothing — but it is the
  opposite of reassuring, and it corrects an earlier reading of this case as
  evidence *against* a recency rule.

Gate 0 is unaffected: its target spans are confirmed **message-scoped**, so the
98% required landing in the correct message's copy. But because only 15/207
tokens were ambiguous, **Gate 0 measures citation accuracy, not occurrence
disambiguation.** Both statements are true and neither substitutes for the other.

**Date** 2026-09-04 · **Status** measurement note; zero `src/` edits,
`LensConstants` unchanged, server architecture refusal unchanged.

## 0. SECOND CORRECTION (2026-09-05) — the recurring-value corpus and re-score

The original corpus contained almost no ambiguity (15 of 207 scored tokens had
the value in >=2 places), so nothing here had tested occurrence disambiguation.
The corpus was rebuilt with truthful restatements (confirmations, recaps,
cross-references) — **183 of 213 tokens are now genuinely ambiguous** — and
re-run. Three results, in order of how well-powered they are:

**1. The head is NOT a positional artifact. Well-powered, and it stands.**
Over 183 ambiguous tokens: `R_last` 26%, `R_first` 25%, `R_near` 26%. The best
trivial rule disagrees with the head on **74%** of tokens (bar: >=15%). No
"always newest / always oldest / always nearest" rule reproduces what the head
does. This was the threat the control existed to test, and the head survives it.

**2. The head reliably finds a true occurrence. Also well-powered.**
Re-scored with a hit = ANY literal occurrence of the value rather than the
field's designated message: **top1 93% (199/213), top3 99% (211/213)**, versus
strict top1 34% / top3 74%. **Delta +59 / +25 points.**

**3. Therefore Gate 0's "FAIL" is a LABELLING artifact, not a head failure.**
Strict scoring counts a hit only when the citation lands in message 0. Once the
corpus states the same value truthfully in several later messages, that label
asserts a unique correct source where several exist — the head pointing at a
confirmation in message 8 is scored as a miss. The strict number answers "does
the head prefer the ORIGINAL statement" (no, 34%), not "does the head find the
value" (yes, 93%). **The gate needs redefining, not re-thresholding**; the
printed FAIL is against a question that no longer has a well-defined answer.

### What is STILL NOT established
- **Which copy the model actually used.** The head names a message; we cannot
  verify it is the message the generation drew on. That needs ablation (drop a
  copy, re-run, see if the answer moves) — the PROOF1 method. Nothing in this
  corpus supplies that ground truth.
- **The alarm as a shippable predicate.** SERVEABLE (citations + caller-declared
  message boundaries, i.e. what `/v1/extract` could actually compute) scores
  **precision 20%, recall 50%, FRESH false-alarm 67%** — four false alarms in
  six correctly-handled corrections. ORACLE's 100%/100%/0% uses corpus ground
  truth the endpoint does not have. **Quote the SERVEABLE row, never ORACLE.**
- One mitigation worth testing rather than assuming: the alarm's predicate needs
  only the message *class* (superseded vs current), not message *identity*. If
  every copy of a stale value sits in a superseded message, landing on any of
  them is sufficient. That may matter more than strict citation accuracy, and it
  has not been measured.

## 1. Provenance

| | |
|---|---|
| Model | `models/Qwen3.8-9B-Q8_0.gguf`, arch `qwen35`, Q8_0, 33 blocks |
| Attention layers | 8, `il` = 3, 7, 11, 15, 19, 23, 27, 31 (tap slots 0..7) |
| Heads | `n_head_q=16`, `n_head_kv=4` (4:1 GQA) |
| Citation head | **L27H13** = tap slot 6 / head 13 (98% top-3 single-doc), vs SS1's L3H13 at 84% |
| Decode | **FREE — no grammar** (§3.1). Tolerant parse, the shipped lens contract |
| `--flash-attn` | off |
| Driver | `tests/perf/attn_provenance.cpp`, path `SS2=1` (additive, env-gated) |
| Raw log | `.session-results/ss2_thread_alarm_v2.log` (v1, the blocked grammar run: `ss2_thread_alarm.log`) |
| Threads | 8 (4 EN, 4 DE), 4774–6200 tokens, quoted reply history |

## 2. Results

### Gate 0 — citation at thread scale · **PASS**
```
value tokens scored: 207   top1 89% (185/207)   top3 98% (202/207)   [bar top3 >= 85%]
excluded (normalized / not emitted verbatim): 1 of 38 labeled fields
```
Compare SS1: 43% / 71%, with field parsing broken. Compare single-document
Leg C on this head: 89% / 98%. **Thread scale costs nothing.**

### Gate 1 — natural STALE rate · stale occurs at length
8 correction instances: **FRESH 7, STALE 1.**

| thread | lang | position | phrasing | distractor | label |
|---|---|---|---|---|---|
| t_en1 | EN | early | explicit | yes | FRESH |
| t_en3 | EN | mid | indirect | no | FRESH |
| t_en5 | EN | late | explicit | yes | FRESH |
| t_en7 | EN | mid | indirect | no | FRESH |
| t_de1 | DE | early | explicit | no | FRESH |
| **t_de3** | **DE** | **late** | **indirect** | **yes** | **STALE** |
| t_de5 | DE | mid | explicit | no | FRESH |
| t_de7 | DE | early | indirect | yes | FRESH |

**The one failure has the hardest possible profile** — correction at message
22 of 24 (92% through the thread), phrased indirectly, with a distractor
restating the old value. Note t_en5 is also late + distractor but *explicit*,
and survived: on this evidence **indirect phrasing is the discriminator, not
position alone.** One case; a hypothesis, not a finding.

### Gate 2 — alarm precision / recall · promising, under-powered
```
TP=1  FN=0  FP=0  TN=7      precision 100%   recall 100%   FRESH false-alarm 0%
```
The true positive is clean: on `t_de3` the model emitted
`"delivery_date": "2025-09-30"` — the original from message 0 — instead of the
corrected `2025-10-07` at message 22. The frozen head cited **message 0**, which
is superseded for that field, so the alarm fired. Exactly the designed path,
end to end, with no coverage signal involved.

## 3. The two defects this run fixed (both from the v1 run, both real)

### 3.1 The grammar had to go — SS1's own recommendation was stale
SS1 §5 prescribed "a grammar-constrained thread extractor." **SS1 is dated
2026-07-12; the fixed KV grammar was REFUTED by measurement on 2026-07-16 and
REMOVED from the product path in Stage 2 on 2026-07-17**
(`docs/note-nogrammar-refutation.md`; `http_server.cpp`: *"Builds NO grammar:
the lens decodes free"*). The recommendation was four days older than its own
refutation and should not have been followed.

Measured cost of following it: the fixed-KV grammar bounds each pair's *shape*
but not the *number of pairs*, so a looping model emits **syntactically valid
infinite JSON** — v1's `t_en1` invented ~20 nested variants of one key, all
valued `"Priya"`, and never closed. Free decode terminates on `}`. This is the
refutation reproducing on new ground: the grammar's one promised benefit, a
guaranteed parse, is the thing it fails to deliver.

### 3.2 The scorer silently inverted the alarm's only error class
v1 decided FRESH-vs-STALE by substring-searching the **whole generated output**,
testing `has_new` first:
```c
bool has_new = R.gen_text.find(def.corr_new) != std::string::npos;   // WRONG
```
On v1's `t_en3` the model emitted the **stale** `"unit_price":"0.35 eur"` while
the fresh `0.42` appeared under an invented key (`galvanising_supplier_rates`) —
so a genuinely stale emission scored **FRESH**. A false negative on the one case
the alarm exists to catch, produced by the grammar's own invented keys.

Now key-scoped: the emitted value is read from the parsed field whose name is
the labeled concept, via the shipped tolerant parse. Both defects had one root —
the grammar — which is why removing it fixed both.

## 4. What was NOT done
- **No third leg.** 8 threads, one model, one run. Nothing here is a rate.
- **Gate 2 is n=1 on the positive class.** Recall is one coin flip. The
  false-alarm rate (0/7) is the better-supported half.
- **No coverage / COV1 arm** — excluded by design; the alarm is coverage-free.
- **No conflict attribution** (CF1 dead), no product build, no server wiring.
- ~~**v1's GPU OOM at 8292 tokens is unresolved, only avoided**~~ — **RETRACTED
  2026-09-05. This claim was wrong.** It got its own check and did not survive
  it. The only `kIOGPUCommandBufferCallbackErrorOutOfMemory` recorded anywhere in
  this repo is `.session-results/prio2/server_d8b.log`, which is a *different
  configuration entirely*: Qwen **3.6-35B-A3B** (12.5 GB of weights, 1280 MB KV,
  "max batch size: 4 and max ctx: 8192"), failing while answering *"Say OK"*.
  It has nothing to do with 8292 tokens or with a 9 B model. v1's own log
  (`ss2_thread_alarm.log`) ends mid weight-load and never reached a prefill.
  Measured on the exact configuration the claim named — Qwen3.8-9B Q8_0,
  `--ctx-size 9216`, same host:

  | path | prompt tokens | result |
  |---|---|---|
  | CLI, no tap | 8299 | OK (53 s prefill, 156 t/s) |
  | shipped lens `/v1/extract`, taps armed | 8102 | **200**, 5/5 fields grounded |
  | shipped lens `/v1/extract`, taps armed | 8873 | **200**, 3/3 fields |

  Zero OOM in the server log. The 9 B lens runs to ~8.9 K, i.e. the context
  ceiling it was given, so **the ≤10 K envelope holds for this configuration**
  and the thread cap of 5800 was never needed. What *is* real is memory pressure
  on the **35 B at 4 slots × 8192** — a capacity fact, not a bug, and the engine
  already refuses it loudly, names the remedies, and stops the loop cleanly
  (`qinf::engine::require_compute_success`). One caveat kept honest: unified
  memory is shared, so an earlier run competing with a resident 35 B server could
  genuinely have OOM'd; what is refuted is the *attribution* to 8.3 K tokens on a
  9 B.
- **Zero `src/` edits.**

## 5. Honest limits
- Threads are **realistic, not real** — assembled from the existing 15-document
  EN + DE messy corpus with quoted reply history, because no real thread corpus
  exists in this repo. Real mailbox data may behave differently.
- The model wraps output in ```json fences; the tolerant parse absorbs this.
  That is the shipped contract working, but it means the extractor is not
  emitting bare JSON and a stricter consumer would need to handle it.
- One STALE case means the alarm is validated *in principle*. Powering Gate 2
  needs more stale cases, and §2's Gate 1 table says where to find them:
  **late + indirect + distractor.**
