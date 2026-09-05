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

Gate 2 (alarm precision/recall) is **promising but under-powered**: TP=1, FN=0,
FP=0, TN=7 ⇒ precision 100%, recall 100%, FRESH false-alarm 0%. The **0/7
false-alarm rate is the meaningful half** — the alarm stayed silent on every
correctly-handled correction. **Recall rests on n=1 and must not be quoted as a
rate.**

**Date** 2026-09-04 · **Status** measurement note; zero `src/` edits,
`LensConstants` unchanged, server architecture refusal unchanged.

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
- **v1's GPU OOM at 8292 tokens is unresolved, only avoided** — thread targets
  were capped at 5800. A 9 B Q8_0 with a 576 MB KV cache could not complete an
  8.3 K prefill on this host with the tap armed. That is a live data point
  against the ≤10 K envelope and deserves its own check.
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
