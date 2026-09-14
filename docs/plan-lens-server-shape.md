# Plan — the lens-only server shape

**Status (2026-09-13):** Movement 1 **SHIPPED** (`POST /v1/verify`, server +
client). Movement 2: the **gate is shipped and it did its job** — it refused the
first optimisation put in front of it (flash prefill, §4.2.1) on the model the
lens is calibrated on, and it showed that the plan's central number was a
single-model result. Quantized KV is refused on a structural argument (§4.2.2);
document batching is blocked on a product decision (§5.2). §6 remains out of
scope. Open items are listed at the end of §4.3.

**The one-line version:** Movement 2 bought no speed on the lens path. It bought
the ability to find that out before shipping, and one real finding — **an MoE
hybrid does not perturb smoothly, because expert routing is an argmax.**

Written 2026-09-13 from the measurements in §2. **Architecture approval was
required before anything here landed** (CLAUDE.md architecture protocol) and
`architecture.md` was updated in the same change, §5 / §6 / §11.

**Premise.** Qwenium is being used as a *lens server*, not a general inference
engine. This asks what changes if we accept that as the only workload. Two
movements come out of it. They are independent but the second makes the first
cheaper to justify.

---

## 0. How to start a session on this

Read `CLAUDE.md` and `docs/architecture.md` first (session protocol), then §1–§2
here. `docs/note-lens-phone-scale-probe.md` is background, not required.

Three standing constraints:

- **The user commits.** Never run `git commit` / `add` / `stash` / `checkout`.
- **All model runs use `build-metal`** (Release + Metal), never `build/`.
- **Moving a lens calibration constant is a user decision**, not an
  implementation detail. §6 is explicitly out of scope for this plan.

---

## 1. What a lens-only server is, and why the shape differs

A general engine is decode-bound: long generations, many turns, TTFT and
tokens/sec are the SLO. The lens is the inverse — a 2K–10K-token document in,
150–300 tokens of JSON out, no streaming, no multi-turn. The SLO is end-to-end
document latency and **documents per hour**.

Three properties follow, and all three are load-bearing below:

1. **Prefill dominates.** Materialised attention costs `O(n_q × n_kv)`; at
   prefill `n_q` is the whole document, at decode it is 1. So materialising is
   ruinous in the phase we don't tap and nearly free in the phase we do.
2. **The lens only ever reads decode-step attention.** Citations are per
   generated token; coverage is the peak across decode steps. **Prefill rows are
   never tapped.**
3. **The report lives below the deepest tapped layer.** Attention at layer `L`
   cannot depend on layers above `L`, so the provenance is fully determined by
   the first `max(citation_layer, coverage_layer) + 1` layers.

   **How much that saves is per-model, and the two calibrated models differ
   sharply** (corrected 2026-09-13 — an earlier draft of this plan quoted L3/L11
   as if general, which is true only of the 35B):

   | model | citation | coverage | layers needed | of | saving |
   |---|---|---|---|---|---|
   | Qwen3.6-35B-A3B | L3 | L11 | 12 | 40 | **70%** |
   | Qwen3.8-9B | **L27** | L11 | 28 | 33 | 15% |

   So truncation is a large win on the 35B and a minor one on the 9B. **It is
   not the main win on either.** Eliminating the decode loop is: `extract` runs
   one prefill plus ~150–300 sequential decode steps, while `verify` is two
   prefills and no decode at all. That holds regardless of depth, and it is why
   §3 is worth doing even where truncation buys little.

---

## 2. The measurements this rests on

All from `tests/perf/attn_provenance.cpp`, 2026-09-13, Leg C corpus,
**English only** (`ATTN_LENS_EN_ONLY=1` — the ceiling arm; see
§7 for why that matters).

### 2.1 Decision stability — the tolerance budget

The report's decisions are threshold tests on continuous quantities
(`coverage_used_peak` 0.705, `ungrounded_body_mass` 0.538). The property the
product needs is therefore neither byte-identity nor token-stability but
**decision stability**: every span lands on the same side of the threshold.

| quantity | value | how |
|---|---|---|
| nearest span to 0.705, English | **0.0190** | `COVSEARCH` BAND COST block |
| nearest span to 0.705, pooled EN+DE | 0.0034 | same (German spans crowd the line) |
| max drift, single vs chunked prefill | **0.000563** | `BANDDRIFT=1`, 53 body lines, 8 EN docs |
| median drift | 0.000000 | half the lines bit-identical |
| generated tokens changed | **0 / 8 documents** | token identity is a hard gate in the leg |
| coverage decisions crossed 0.705 | **0 / 53** | — |

**Margin is ~34× the measured drift.** The perturbation is single-batch vs
128-token chunked prefill — a real, already-shipped config difference in the
same class as a flash change (`src/engine/multimodal_prefill.cpp:43` names the
"single-vs-chunked Metal precision divergence" outright).

**Limit:** chunked prefill is the same *class* as flash, not flash itself —
flash additionally reorders the softmax reduction in registers. This bounds the
class; it does not settle flash.

### 2.2 Cost of a band, if we ever want one

English, at the shipped operating point: a band of 0.010 withdraws **0 spans**;
0.030 withdraws 1 (a filler, 0% of `skipped[]`); 0.050 withdraws 3 (3%). The
peak distribution is strongly bimodal — positives median 0.912, fillers median
0.236 — which is why 0.705 sits in a valley and any band is cheap.

**Conclusion: no band is needed for the perturbation class we can test.** The
honest move is a bounded reproducibility statement in `lens-format.md`, not a
v5 format bump. Keep the band specified-but-unbuilt until flash exists.

### 2.3 Two corpus defects found and fixed (opt-in)

Both are in the Leg C **filler negatives**, i.e. the denominator of every
false-positive number and of the matched-FPR pin that COVSEARCH uses to select
layers.

- **`product` is asked for by the TASK prompt but never labelled**, so the
  product name sat in filler territory and the model's correct reading of it
  scored as a false positive (3 of 9 on English).
  Flag: `ATTN_PRODUCT_OCCUPIED=1`.
- **Only the *first* occurrence of a labelled value was marked occupied.**
  `m_en2` states `order_number` "5590-B" in the Subject *and* the body; the
  Subject copy was occupied and the body copy was cut into two filler windows,
  both of which cleared. Flag: `ATTN_ALL_OCCURRENCES=1`.

Effect on the incumbent's English false-positive rate: **24% → 22% → 19%**.
Both flags are **opt-in** so `qdocs_messy_corpus()` stays byte-identical for
leg C, the S2/S3 thread work and QKEY, whose stored numbers would otherwise
move silently.

A third class remains and is **not** fixable by labelling: legitimate
supporting context. `m_en4`'s sender domain is `@meridianfoods.example` and its
customer is "Meridian Foods Pte" — the model reads the header to write the
customer name, correctly, and we score it as an error. Five of the seven
remaining false positives are this. Fixing it properly means drawing decoys
from a *different* document.

---

## 3. Movement 1 — split generation from verification

### 3.1 The proposal

```
POST /v1/extract   document               -> JSON + report    (today, unchanged)
POST /v1/verify    document + extraction  -> report only      (new)
```

`verify` teacher-forces a known extraction rather than generating one:
one parallel prefill over `prompt + known JSON`, **no decode loop, no sampling**,
and the forward pass **truncated at the deepest tapped layer**.

### 3.2 Why it is cheap, and why that is exact

Attention at layer `L` cannot depend on layers above `L` — causality through the
residual stream, nothing subtle. So a forward pass truncated after
`max(citation_layer, coverage_layer)` produces **the same tapped rows** as the
full stack. Read the cutoff from the live calibration table, never hardcode it:
it is layer 11 on the 35B but layer 27 on the 9B (§1.3).

Teacher-forcing is likewise exact in principle: position `t` attending to
`[0..t]` sees the same causal mask and the same KV whether it arrived by decode
or by prefill. The difference is batch-vs-single numerics — the documented
mm-vs-mv fork — which §2.1 measures at 5.6e-4 against a 0.019 margin.

### 3.3 Why it earns its keep beyond speed

- **It is the verb the product sells.** "Show me where the model looked" for an
  extraction you already have is a different job from producing one.
- **It makes calibration changes survivable.** If a constant ever moves (see
  §6), every report ever issued goes stale. `verify` re-audits an archive
  without re-generating anything. Without it, moving a constant orphans history.
- **It subsumes the flash phase-split** rather than competing with it:
  `extract` generates with flash everywhere and no tap; `verify` runs
  materialised but shallow. One verb split instead of a per-layer flag.

### 3.4 Qwenium side (engine + server)

1. **Truncated forward pass.** `build_prefill_graph` needs a "stop after layer
   N" mode. This touches the forward pass, so the **cross-family rule applies**:
   it must be expressible on a Gemma recipe too, even though there is no Gemma
   lens claim, or we have built a Qwen-shaped interface.
2. **Teacher-forced tapped prefill.** The tap seam
   (`set_attention_taps` / `mark_attention_taps` / `get_attention_taps`) reads
   `kq_soft.<il>` at decode shape `[n_kv, 1, n_head, 1]`
   (`forward_pass_base.cpp:263`). At prefill the same tensor is
   `[n_kv, n_q, n_head]`. **`get_attention_taps` must learn the prefill shape**,
   and `compute_lens_report` must accept rows that arrive as one block rather
   than per step. This is the largest single piece of work here.
3. **No output head.** `verify` needs no logits; `run_prefill` already honours
   `want_logits=false` via `feed_tokens` on recipes that support it
   (`qwen35.h:125`, `qwen36.h:66`).
4. **New route** in `http_server.cpp` beside `/v1/extract`, same
   `--attention-lens` gate, same calibration refusal, same single-slot lock.
5. **Gate:** `verify(document, extract(document).json)` must reproduce
   `extract(document).report` **decision-for-decision** (not bit-for-bit — see
   §4). That is the correctness test and it is self-checking: no new corpus.

### 3.5 Client side

See §5 — the client work spans both movements and is collected there.

### 3.6 Open question, flagged not resolved

If `verify` accepts *any* extraction — including one produced by a different
model — the lens can attach receipts to a pipeline that is not ours. That is a
large product surface (it dissolves the "ship our gigabytes to the phone"
problem in the iPhone envelope entirely). **But the claim changes**: the report
would say where *our* model looked at *their* answer, not where the producing
model looked. `lens-format.md`'s honesty contract does not currently say that,
and blurring it would be the first dishonest thing in the format. Explore
deliberately or not at all.

---

## 4. Movement 2 — spend the tolerance budget

### 4.1 The reframe

Receipts-grade determinism is currently specified per-config **and B=1**
(`architecture.md` §11). That reading has been treated as byte-identity, and
byte-identity is both too strong (it forbids changes that provably move no
decision) and too weak (it does not survive a driver update, a different GPU, a
ggml bump, or a batch-size change — so it was never the guarantee we could
actually offer a customer).

§2.1 replaces the argument with a number: **margin 0.019, drift 0.00056**.

### 4.2 What that unblocks — three things, one measurement

1. **Phase-scoped flash — BUILT, MEASURED, and then REFUSED by its own gate.**
   `DecodePolicy` now carries one implementation per phase (`attn_impl` decode,
   `prefill_attn_impl` prefill); every recipe's prefill builder reads
   `use_flash_attn_prefill()` — all seven, Qwen **and** Gemma, per the
   cross-family rule. `run_lens_verify` forces its own **tapped** pass
   materialized (it taps prefill, unlike extract) and restores through the RAII
   guard.

   Then the gate was run on both calibrated models, and they disagreed:

   | model | margin | max \|Δpeak\| flash | tokens | verdict |
   |---|---|---|---|---|
   | Qwen3.8-9B-Q8_0 | 0.00126 | 0.00084 | 15/15 identical | **PASS**, 1.5× |
   | Qwen3.6-35B-A3B | 0.0237 | **0.0242** | 15/15 identical | **FAIL** |

   And the control arm is worse than the candidate on the 35B: a **chunked
   prefill** — an already-shipped config difference, not a new one — changes
   **one extraction in fifteen outright**, `m_de2` at 82 generated tokens
   against 109, with max drift 0.0221.

   **The mechanism, which is the part worth keeping.** The 35B is an MoE hybrid
   and **expert routing is a top-k argmax**. An argmax converts a 1e-4
   perturbation into a *different expert*, not a slightly different number. The
   entire "drift is ~1e-4, the margin is 100× that" story — §2.1, §4.1, the
   whole reframe — was measured on the 9B and is a **dense-stack** result. It
   does not transfer to the model this lens was calibrated on, and nobody
   noticed because the measurement had only ever been run on one model.

   **Resolved 2026-09-14 by making the flag per-model**, which is what it always
   was: `LensConstants::flash_prefill_ok` (+ a `flash_prefill_provenance` string,
   same discipline as `LensCalibration::provenance` — a bare boolean is a claim
   with no receipt). The 9B entry carries `true` and names the gate run; the two
   35B entries carry the default `false`. **The default is the safety property**:
   a model nobody has measured is refused rather than inheriting a permission
   measured elsewhere.

   The 8.4% is real (prefill 16.30 → 14.93 s on a ~2.4K-token document, extract
   end-to-end 24.16 → 22.13 s) and is now available on the 9B.

   Three consequences, each of which had to be handled:

   - **The pairing check moved after the model load.** It cannot sit with the
     other argument validation, which deliberately runs before a multi-GB load,
     because the answer depends on which model is loaded. Stated in the code so
     nobody "fixes" it back.
   - **`--attention-lens` with a quantized `--kv-type` needed its own refusal.**
     It used to fall out of the blanket lens/flash refusal; removing that opened
     a hole, because the cache is read by the materialized decode where a
     quantized V takes ggml's silent CPU fallback. That refusal is now
     load-bearing rather than a standing guard.
   - **A licensed configuration is only safe if the report says which one it
     ran under**, which is why the `config` stamp (§4.5) landed with it rather
     than after it.

   Pinned by: `LensCalibration.FlashPrefillIsPerModelAndDefaultsToRefused` (unit,
   no model needed) and a leg in `server_verify_smoke.sh` that starts the 9B with
   both flags and re-runs the full extract→verify check under a flash prefill —
   the fork where extract's tapped rows come from a decode over a flash-written
   KV while verify's tapped pass is forced materialized. It reproduces
   decision-for-decision.

2. **Quantized KV — REFUSED, and the plan was wrong here.** The claim was that
   a flash prefill opens the `--kv-type` gate. It does not. The KV cache is
   written by prefill and **read by decode**, and the lens decode is
   materialized by construction; the materialized path transposes V, which
   moves a quantized type's block dimension out of position, and Metal's
   `CPY`/`CONT` has no quantized source case — so it would take ggml's *silent*
   CPU fallback on precisely the path the receipts come from. The existing
   refusal (quantized ⇒ `--flash-attn`) used to exclude the lens transitively
   through the flash/lens refusal; phase-scoping that refusal opened a hole, so
   a **second, direct refusal** now closes it: `--attention-lens` with a
   quantized `--kv-type` is rejected at startup. Quantized KV cannot reach the
   lens while the tap needs a materialized decode. `--kv-type f16` is
   unaffected by this and remains uncalibrated for receipts (§11).
3. **Document batching — COSTED 2026-09-14, and it does not pay.** Single-slot
   is a *choice*, not a limit: the qwen36 decode KV-gather defect was fixed
   2026-08-29 and multi-slot rows are pinned by `test_gather_indices_input`
   (`architecture.md` §12). But before building it, cost it.

   **Batching cannot touch prefill.** Each document's prefill is its own
   compute-bound work at `n_q` ≈ the document length; running four of them
   together is the same arithmetic in one graph instead of four. Only the
   decode phase batches — and on a lens workload decode is the *small* half.
   Measured on Qwen3.8-9B-Q8_0, a ~2.4K-token document, 160 generated tokens:
   **prefill 16.30 s (67%), decode 7.86 s (33%)**.

   `qwen35` 9B costs **0.52** of a B=1 step per extra lane (`architecture.md`
   §1), so decode throughput is `B / (1 + 0.52(B−1))`:

   | B | decode throughput | s/document | overall |
   |---|---|---|---|
   | 2 | 1.32× | 22.27 | 1.09× |
   | 4 | 1.56× | 21.33 | 1.13× |
   | 5 | 1.62× | 21.14 | **1.14×** |
   | 10 (envelope max) | 1.76× | 20.76 | 1.16× |
   | ∞ | 1.92× | 20.39 | **1.19× ceiling** |

   B=5 rather than 8 because 8 slots sit in the trough between two ggml-metal
   kernel regimes (`architecture.md` §1).

   **Against that, the price is not small.** The lens driver runs slot 0
   directly with its own decode loop, outside the InferenceServer's slot
   lifecycle (`server_lens.cpp`, the `EngineRestore` comment records what
   happened the last time that boundary was crossed carelessly). Multi-slot
   means a new tapped-decode driver with per-slot tap slicing, stop conditions
   and run state; a queueing change to hand it N requests at once; and a
   per-model admissibility gate, because B>1 is a batch-shape change — the same
   class the 35B fails on both the `chunk` and `flash` arms, and for the same
   argmax reason (§4.4). So it would be a 9B-only 1.14×.

   **And it is mutually exclusive with the thing that already works.** The warm
   document buys **92.6–98.9% of pass-1 prefill** on the key-editing loop —
   i.e. it attacks the 67%, where batching attacks the 33% — and it is built on
   single-slot exclusivity (`server_lens.h`). Batching would trade a measured
   two-thirds off one workload for at most a sixth off another.

   **Recommendation: do not build it. Keep single-slot, keep the warm
   document.** This closes the §5.2 question by measurement rather than by
   product preference: "many documents at once" already *works* — requests
   queue and every answer is correct — batching only makes it ~14% faster.
   Revisit if the workload ever inverts (short documents, long generations):
   the win scales with decode share, and even a 50/50 split only reaches 1.31×.

### 4.3 The work — DONE, with one correction and one refusal

**1. The standing gate — DONE.** `BANDDRIFT=1` now takes `DRIFT_ARM` and scores
any registered candidate the same way (`DriftArm` in
`tests/perf/attn_provenance.cpp`). Three criteria, all required: token identity
across the corpus, zero decisions crossed, max |Δpeak| under the margin. **Exit
code is the gate** — 0 PASS, 1 FAIL — so it is wirable, and
`tests/smoke/lens_drift_gate.sh` runs every arm and returns the worst.
Registered arms: `chunk` (the Metal mm-vs-mv fork, today's measurement) and
`flash` (§4.2.1). Adding a third is a struct literal.

**2. The unit mismatch — FIXED, and it mattered in the safe direction.** The
margin is now measured in the same run, on body lines, as the distance from
0.705 to the nearest arm-A line peak: **0.0153**, against the 0.0034 span-level
figure the verdict used to borrow. So the budget was being *understated* 4.5×
— the old verdict was conservative, not wrong-flattering, but it was not a
measurement of the thing it printed.

**3. The claim in `lens-format.md` — WRITTEN.** A new bullet in *Honest limits*:
byte-identical within a config; across configs a bounded **decision** claim with
the two numbers (0.0153 margin vs 0.00061 observed), the corpus it rests on, and
three explicit narrowings — the corpus is small, German is far tighter (item 4
below replaced the estimate with the measurement), and an untested change has no
measured drift rather than a small one. Additive prose,
no member added or removed: **not** a v5 bump, as predicted.

**4. A language axis, because the English arm was flattering us.** The gate
took `DRIFT_LANG=en|all` (default `en`, today's numbers) and was run both ways.
This was listed as "still open" for about an hour; running it changed the
headline.

**Measured. The 9B passes both arms; the 35B fails both:**

| model | arm | corpus | max \|Δpeak\| | margin | tokens | verdict |
|---|---|---|---|---|---|---|
| 9B | `chunk` | EN, 53 lines | 0.000563 | 0.0153 | 8/8 | PASS 27× |
| 9B | `flash` | EN, 53 lines | 0.000609 | 0.0153 | 8/8 | PASS 25× |
| 9B | `chunk` | EN+DE, 98 lines | 0.000602 | 0.00126 | 15/15 | PASS 2.1× |
| 9B | `flash` | EN+DE, 98 lines | 0.000835 | 0.00126 | 15/15 | PASS 1.5× |
| **35B** | `chunk` | EN+DE, 91 lines | 0.0221 | 0.0237 | **14/15** | **FAIL** |
| **35B** | `flash` | EN+DE, 98 lines | **0.0242** | 0.0237 | 15/15 | **FAIL** |

Three things fall out, in increasing order of importance:

- **Which language binds is model-dependent.** German sets the 9B's margin
  (0.00126 against English's 0.0153 — 12× tighter); **English** sets the 35B's
  (0.0237 against German's 0.0539). "German crowds the threshold" is a 9B fact,
  not a law. Run both languages; do not assume which one is the falsifier.
- **The English-only headroom this plan nearly published (25×) was an
  artifact twice over** — of one language and of one model. The gate now prints
  a **THIN** warning under 5× so a 1.5× pass cannot be read as a 25× one.
- **Drift is ~30× larger on the 35B, and it is not a smooth quantity there.**
  Zero decisions crossed on either model, so no shipped report is impugned. But
  the 35B changed an *extraction* under a config difference we already ship, and
  the reason — top-k expert routing — means the perturbation analysis in §2.1
  and §4.1 is a dense-stack result that was never true of the calibrated model.
  **§4.1's reframe survives as a method and not as a number.**

**5. The licence is cashed on a live server, not only in the probe.**
`tests/smoke/server_verify_smoke.sh` gained a third leg: the same
extract-then-verify check re-run against a server started with
`--attention-lens --flash-attn`. That leg is where the fork actually bites —
extract's tapped rows come from a decode over a KV written by a **flash**
prefill, while verify's tapped pass is forced materialized — and it passes:
same 7 fields, same 6 present / 2 skipped, decision for decision. Extract under
a flash prefill also emitted a **byte-identical extraction** to the
materialized one on that document.

**Gates run after the change:** `server_verify_smoke.sh` PASS (all three legs,
9B), `server_extract_smoke.sh` PASS, `decode-policy-tests` 10/10, full `ctest`
939 tests — 6 `HttpServerTest` cases fail under `-j4` and pass serially (port
contention in a mock-engine fixture, untouched by this work).

**Still open:**

- ~~The 35B's instability~~ — **SETTLED the same day, see §4.4.**
- **Whether a per-model gate should unlock flash prefill on the 9B** (§4.2.1) —
  a calibration-table decision, reserved for the user.
- ~~Document batching~~ — **costed and declined 2026-09-14, §4.2.3.**
- The gate is not in `ctest` (it needs a model and a GPU), same as the server
  smokes.

---

### 4.4 Why an MoE is irreproducible — settled, in two steps

§4.3 left the 35B's instability measured but unexplained. Two experiments
closed it.

**Step 1 — a 2×2 across families.** The `chunk` arm, bilingual, 15 documents,
run on a dense and an MoE model from each family. Both Gemma halves are the
**same recipe** (`gemma4`), which is what makes the pair a control rather than
a comparison of two unrelated models:

| family | model | experts | token identity | max \|Δpeak\| |
|---|---|---|---|---|
| Qwen | 3.8-9B-Q8_0 | dense | 15/15 | 0.00060 |
| Gemma | 4-12B-it-Q8_0 | dense | 15/15 | 0.00081 |
| **Qwen** | **3.6-35B-A3B** | **256** | **14/15** | **0.0221** |
| **Gemma** | **4-26B-A4B-it** | **128** | **14/15** | **0.0217** |

Separated perfectly by MoE-ness, across two families, two recipes, different
layer counts (48/30 against 33/40) and different quantizations. Each MoE
changed exactly one extraction in fifteen — Qwen's `m_de2` (82 generated tokens
against 109), Gemma's `m_en1` (148 against 169). This is the cross-family rule
working as designed: Gemma was invited to falsify and confirmed instead.

**Step 2 — read the routing itself** (`MOEROUTE=1`). The 2×2 still admitted two
mechanisms: expert *selection* flipping, or `ggml_mul_mat_id` reducing
differently at a different batch width with the same experts. `moe_idx.<il>` is
already a named node in every MoE graph, so the selection can be read directly.
Prefill only, head-less, both arms:

| model | routing decisions compared | different expert **set** | documents with a flip |
|---|---|---|---|
| Qwen 3.6-35B-A3B (256 experts, top-8) | 116,960 | 2,713 — **2.32%** | 12 / 15 |
| Gemma 4-26B-A4B (128 experts, top-8) | 96,030 | 6,116 — **6.37%** | 14 / 15 |
| Qwen 3.8-9B (dense) | — | no routing to compare (fail-loud control) | — |

**Selection flips. It is not a kernel artifact.** And the flips carry the
argmax's fingerprint: in *both* models the first flip is in the **last slot of
the top-8** — Qwen `[...,48,225]` against `[...,48,38]` at L5, Gemma
`[...,28,53]` against `[...,28,26]` at L8 — the marginal expert, the one nearest
a tie. Flips then concentrate in the **late layers** (Qwen L26–39, Gemma
L19–29), which is the perturbation being amplified down the stack rather than
injected at the end.

**The nuance that keeps this honest: most flips are absorbed.** 2–6% of routing
decisions change, and only **one document in fifteen** changes its answer.
Experts are redundant enough to swallow nearly every flip — but "nearly" is not
a guarantee, and there is no way to know in advance which flip will not be
swallowed. A further 4.7% (Qwen) / 9.8% (Gemma) of decisions picked the same
experts in a different *order*, which is benign: the weighted sum is over the
same set.

**What this settles for the lens.** The perturbation analysis in §2.1 and §4.1
— "drift is ~1e-4, the margin is 100× that" — is a **dense-stack** result. On an
MoE it is not that the numbers are bigger; it is that the quantity is not
continuous, so no margin argument applies to it at all. No kernel fixes this,
and a bigger corpus would not have found it: the corpus was never the problem,
the single-model measurement was.

---

### 4.5 The configuration stamp

Once a configuration is something a *model* may opt into, a report that cannot
say which configuration produced it is a hazard rather than a gap — the whole
point of `verify` plus the client archive is comparing two reports, and a diff
that cannot tell "the document changed" from "the server was configured
differently" will present the second as the first.

`LensReport::config` (additive, optional, **no version bump** — `extraction_origin`'s
reasoning: absence is unambiguous, and additive is the reversible choice):

| member | what it pins |
|---|---|
| `weights` | the metadata weights hash, hex — arch + shape + **quantization** + layout |
| `attention` | `materialized` \| `flash-prefill` \| `flash` |
| `kv_type` | `f32` \| `f16` \| … |

**`weights` matters more than the flag that prompted this.** `model` names the
calibration entry — "Qwen3.8-9B" — and says nothing about quantization, so a
Q8_0 and a Q4_K_M report are indistinguishable today and differ far more than
flash does.

Derived in **one** place (`http_server.cpp`'s `lens_config_stamp()`) from the
server's own state, never assembled per call site. Omitted entirely when
unstamped, which is every in-process caller including the unit tests — so those
payloads stay byte-identical to before the member existed
(`LensReportConfig.UnstampedReportOmitsTheMemberEntirely`).

**Client note:** `../qemmi-lens`'s zod schema is a plain `z.object`, which
*strips* unknown keys rather than rejecting them, so this lands without breaking
the client's fail-loud gate. Surfacing it there — and refusing to diff two
reports whose `config` differs without saying so — is client work, in that repo.

---

## 5. Qemmi-Lens side (`../qemmi-lens`)

The client is a thin TypeScript app — **no trust machinery lives there**;
citations, coverage and badges are computed in the engine. Its jobs are: build
the request, gate the response fail-loud against a zod schema, and render or
export it. Read `../qemmi-lens/docs/handoff.md` (its START HERE) before touching
it; that handoff is the living source of truth for the client, and the body of
the `Qemmi-Lens` memory topic is stale by comparison.

Relevant modules:

| file | what it does |
|---|---|
| `src/lens-format.ts` | zod schema + `LENS_FORMAT_VERSION` / `ACCEPTED_FORMAT_VERSIONS` — the fail-loud version gate |
| `src/client/lens.ts` | the `/v1/extract` client, throws `LensError` |
| `src/cli/index.ts` | commander CLI (`extract`, `presets`, `serve`); `DEFAULT_ENDPOINT` is the full `/v1/extract` URL |
| `src/web/serve.ts`, `src/web/page.ts` | local server with a same-origin `/extract` proxy (plus `/ocr`, `/ocr/highlights`) and the UI |
| `src/page/` | page documents: OCR seam, Apple Vision, byte→pixel index, PDF text layer |
| `src/export.ts`, `src/summary.ts` | JSON/CSV export and terminal summary (pure) |

### 5.1 For Movement 1 (the verify verb)

1. **`src/client/lens.ts` — add `verify()` beside `extract()`.** Same fail-loud
   discipline, same `LensError`. Request carries `document` **and** the
   extraction being audited.
2. **`src/cli/index.ts` — a `verify` command, and split the endpoint.**
   `DEFAULT_ENDPOINT` is currently the full `http://localhost:8080/v1/extract`.
   Two verbs need a **base** URL with the path appended per verb; changing that
   constant is a small breaking change for anyone passing `--endpoint` today, so
   accept both shapes or version the flag.
3. **`src/lens-format.ts` — carry the provenance marker.** §3.5's recommendation:
   a `verify` report should say it was produced by teacher-forcing rather than
   generation. Add it **optional** so every archived v0–v4 payload still
   validates, exactly as `tier` and `presence_grounded` were handled. If the
   marker is optional and additive, **no version bump is needed** — but note the
   v4 precedent in `lens-format.md`: an *absent* member can itself be a fact an
   importer must read, and that argument forced a bump once already. Decide
   deliberately, and if it bumps, `ACCEPTED_FORMAT_VERSIONS` and
   `LENS_FORMAT_VERSION` both move.
4. **`src/web/serve.ts` — a `/verify` route on the same-origin proxy**, and a
   re-audit affordance in `page.ts` distinct from "extract".
5. **The archive.** Re-auditing only means something if the original is kept:
   store `(document, extraction, report, constants)` per run. This is new —
   nothing in the client persists a report today. It is the precondition for §6:
   if a calibration constant ever moves, the archive plus `verify` is what lets
   a user see *which* of their past reports changed.
6. **A diff view** in `src/summary.ts` / `src/export.ts`: given two reports for
   the same document, show which decisions changed — spans that entered or left
   `skipped[]`, badges that flipped. Pure functions, unit-testable with no
   server, which is how the rest of that module is built.

### 5.2 For Movement 2 (the tolerance budget)

Mostly invisible to the client, with one real exception. Flash, quantized KV and
the truncated pass change no wire shape.

1. **Nothing to build for the reproducibility claim — CHECKED, nothing to
   change either.** §4.3 concluded no band is needed, so there is no new member
   to render; the change was a sentence in `lens-format.md`. The client was then
   surveyed for anything that states or implies reproducibility, since that is
   where users actually read the promise. The one candidate is
   `src/web/page.ts:535` (warm vs cold: *"were measured to produce identical
   field values, identical citation accuracy, and identical candidate sets"*),
   and it is already the right shape — a claim attributed to a measurement, not
   an absolute. No client edit needed. Re-check if the UI ever gains wording
   about runs being "the same".
0. **This is the same question `plan-qemmi-extract.md` §6 carries as
   "concurrency vs warmth".** Two plans, two names, one decision — answer it
   once and record it in both. Proposed resolution (2026-09-13): make it a
   **deployment configuration** rather than a design fork, since the two modes
   are mutually exclusive by construction (warm-document owns slot 0). A
   startup choice — slots > 1 for throughput, slots = 1 to keep the warm
   document — lets both plans proceed without pre-committing on the operator's
   behalf.

2. **Document batching is a client product decision, not a server one** —
   **ANSWERED 2026-09-14, and the answer came from the server side after all.**
   §4.2.3 costs it: ceiling 1.19×, realistically 1.14×, 9B-only, and mutually
   exclusive with a warm document worth 92.6–98.9% of pass-1 prefill. The
   product question "which do your users do?" turns out not to need asking:
   one lever is four times the other on its own workload, and the cheap one is
   already built. The text below is kept because the reasoning is still the
   reasoning; only the conclusion moved.

   *(original framing)* If the
   server goes multi-slot, the client could pipeline many documents. But the
   warm-document mechanism (`document_id`) depends on single-slot exclusivity
   and buys 92.6–98.9% of pass-1 prefill on the **key-editing loop** — which is
   a UI flow that lives in `src/web/page.ts`. So the trade is concrete: *many
   documents at once* versus *one document edited fast*. The client knows which
   its users actually do; the server does not. **Answer this before the server
   work starts**, because it decides whether §4.2.3 is worth doing at all.

### 5.3 Housekeeping found while surveying

- `docs/handoff.md` says the client "gates the response fail-loud against the
  `qemmi-lens/v0` schema". It gates against **v0–v4** (`ACCEPTED_FORMAT_VERSIONS`
  in `src/lens-format.ts`). Stale text, worth a one-line fix.
- `architecture.md` records an action item that `ACCEPTED_FORMAT_VERSIONS` must
  add `"qemmi-lens/v4"`, "done in that repo's working tree as of 2026-09-06
  (uncommitted there)". Confirmed present — still uncommitted in that repo.

## 6. Explicitly out of scope: the layer-3 finding

Measured the same day, recorded so it is not lost, **not part of this plan**:
layer 3 carries coverage signal the shipped report discards. LODO,
English, corrected negatives:

| | Qwen3.8-9B | Qwen3.6-35B |
|---|---|---|
| L11 alone (shipped) | 35/40 @ 6/37 | 33/40 @ 5/37 |
| L3 alone | 35/40 @ 7/37 | **40/40 @ 7/37** |
| L3+L11 (OR) | **39/40 @ 6/37** | 38/40 @ 6/37 |

On the 35B — the model the lens was *built* on — L11 is among the worst layers
at a matched filler budget (45/68/88/92 against L3's 78/88/100/100).

**RUN 2026-09-14 — the repo's own gate says KEEP, and the two models disagree
about why.** `COVSEARCH=1` with both corpus fixes and **no** language filter
(bilingual — German is half the corpus and the half that crowds the threshold),
citation tap set per model (`ATTN_FROZEN_SLOT=6` = L27H13 on the 9B, the default
slot 0 = L3H13 on the 35B; getting this wrong scores one model's coordinates on
the other).

**Verdict on both models: `KEEP layer 11 @ 0.705 — the candidate did not clear
the gate`.** That is the repo's criterion and it stands. But look at *which*
criteria failed, because two of them are not measuring what they claim here:

| | 9B | 35B |
|---|---|---|
| G1 used-clear ≥90% | PASS (96%) | PASS (100%) |
| G2 filler-clear ≤ incumbent | FAIL (32% vs 18%) | FAIL (23% vs 20%) |
| G3 plateau | FAIL | FAIL |
| G4 citation unchanged | L27H13 top1 88.6% / top3 97.8% | L3H13 top1 72.3% / top3 83.1% |

- **G3 is degenerate on this data, on both models.** Its discriminator is held
  AUC, and the run prints `best held AUC: layer 3 (1.000) median layer AUC
  1.000` — the metric is **saturated at the ceiling**, so no layer can strictly
  beat the median and the plateau test fails by construction. A FAIL here is
  evidence about the metric, not about the candidate.
- **G2 compares thresholds that were not matched.** The candidate's threshold is
  fitted on the calib split; the incumbent's is 0.705. The gate's own
  **MATCHED-FPR** table is the apples-to-apples version, and it disagrees with
  G2 on one model and confirms it on the other.

**At matched FPR — and this is the result that matters — the two models do not
agree about L3:**

| model | matched budget | L3 | L11 (incumbent) | best layer |
|---|---|---|---|---|
| Qwen3.8-9B | 13/71 fillers | 61/75 = **81%** | 65/75 = **87%** | L7 (87%, ties L11) |
| Qwen3.6-35B | 14/71 fillers | 75/75 = **100%** | 64/75 = **85%** | L3, L35 (100%) |

And the split-half stability is the falsifier the English-only work never had:

| model | EN winner | DE winner | stable? |
|---|---|---|---|
| 9B | L7 (39/40, but DE 18/35 = 51%) | **L11** | **DISAGREE** — no layer is stably better than 11 |
| 35B | L3 (40/40) | L3 (35/35) | **AGREE** — and L35, L39, L23 also beat L11 on both halves |

**What this retires.** §6's earlier reading — "both models agree L3 carries
coverage signal the shipped report discards" — **does not survive**. It was
measured English-only with a bespoke LODO harness. Bilingual, on the repo's own
gate, the 9B says L3 is **worse** than the incumbent at a matched budget, and
its German half is what says so (L3 DE 69% against L11's 77%). One more
English-only conclusion that inverted when German was added; that is now twice
in one week (§4.3 item 4).

**What survives, and is stronger than before.** On the 35B, layer 11 looks like
a genuinely poor operating point: at a matched filler budget **L3, L35 (both
100%), L39 (99%) and L23 (97%) all beat it (85%)**, and L3 is perfectly stable
across the language split. That is not a lone spike — it is a broad plateau of
better layers that the saturated G3 test could not see.

**Still: do not move a constant.** The gate says KEEP, and the honest next step
is not to override it but to **fix it**, because it currently cannot answer:

1. **Replace or rescale G3's discriminator.** Held AUC saturates at 1.000 for
   best *and* median on both models. A test whose outcome is fixed by the
   metric's ceiling is not a test.
2. **Make G2 compare at a matched budget**, which the MATCHED-FPR table already
   computes — the comparison is in the output, it just is not what G2 reads.
3. **Then re-run, per model.** Any candidate must also clear the drift gate
   (§4.4): a threshold fitted on the 35B is fitted on a model whose peaks move
   0.02 under a permitted config change, which is larger than several of the
   layer-to-layer differences above. That requirement did not exist before
   2026-09-13 and it applies to every future recalibration.

**DONE, same day. The gate was fixed and now discriminates — and the two models
split.**

*What changed in the gate* (`tests/perf/attn_provenance.cpp`; probe-only, no
`src/` change, no constant moved):

- **G3** is re-based on the quantity the decision is actually made on: a
  neighbour must also beat the incumbent **at matched FPR**. The held-AUC
  median is still printed, labelled `<< SATURATED` when it is, so nobody
  re-derives it as a criterion.
- **G2** now reads the MATCHED-FPR table that was already in the output, and
  demands the win survive being **selected on one language and scored on the
  other, in both directions, with both halves agreeing on the layer.** This is
  *stricter* than what it replaced: the old G2 compared two different operating
  points, and could be passed by a layer that never generalises across the
  split — which is precisely the 9B's L7 (98% EN, 51% DE).
- The verdict text now also requires a passing candidate to clear the **drift
  gate** before any constant moves, and `DRIFT_THR` was added so the drift gate
  can score a *candidate* operating point rather than only the shipped one.

*Re-run, bilingual, corrected negatives, citation tap per model:*

| | Qwen3.8-9B | Qwen3.6-35B-A3B |
|---|---|---|
| G1 used-clear ≥90% | PASS (99%) | PASS (100%) |
| G2 matched-FPR, cross-language | **FAIL** | **PASS** |
| G3 plateau at matched FPR | **FAIL** (lone spike) | **PASS** (L7 corroborates) |
| G4 citation unchanged | L27H13 88.6% / 97.8% | L3H13 72.3% / 83.1% |
| **verdict** | **KEEP L11 @ 0.705** | **candidate L3 clears G1–G3** |

The 9B fails on substance now, not on a ceiling: its EN winner L7 scores
**18/35** on the held-out German half against the incumbent's **27/35**, and no
layer beats L11 on English either (the DE winner *is* L11, so that half fields
no challenger). On the 35B, L3 beats the incumbent in **both** directions —
DE 35/35 vs 29/35, EN 40/40 vs 35/40 — the halves agree, and pooled at a matched
filler budget it is 75/75 against 64/75.

*And the drift gate at the candidate point says the candidate is the SAFER one:*

| 35B operating point | max \|Δpeak\| | margin | headroom | decisions crossed |
|---|---|---|---|---|
| L11 @ 0.705 (shipped) | 0.0221 | 0.0237 | **1.1×** | 0/91 |
| **L3 @ 0.558 (candidate)** | 0.00299 | 0.0147 | **4.9×** | 0/91 |

The drift gate's overall verdict is still FAIL on the 35B, but only on criterion
1 — token identity, the MoE routing flip (§4.4). That is a property of the
**model**, identical whichever layer is read, so it does not discriminate
between the two operating points. Criteria 2 and 3, which *are* about the
operating point, both pass, and pass better at L3.

## The decision this leaves you

**Moving the 35B's coverage layer to L3 @ 0.558 is now supported by the repo's
own gate, cross-language, at a matched false-positive budget, with a better
drift margin than the incumbent.** That is a calibration-constant move, which
§0 reserves for the user. Nothing has been changed.

Read these before deciding:

- **The corpus is 15 documents**, 75 positive spans and 71 fillers. Everything
  above is limited by it.
- **The threshold 0.558 was fitted on the COV1 calib split** (calib accuracy
  100%), which is small. The *layer* choice is the well-supported half; the
  exact threshold is the weaker half and would benefit from being re-fitted on
  the full corrected corpus.
- **L3 is not uniquely good.** At matched FPR, L35 also reaches 75/75, L39
  74/75 and L23 73/75, all against L11's 64/75. L3 wins on having a neighbour
  (L7) corroborate it. A different tie-break could reasonably pick L35.
- **This does not make the 35B reproducible.** It still changes an extraction on
  one document in fifteen under a permitted config change. A better coverage
  layer is a better *reading*; it is not a fix for the routing flip.
- **The 9B does not move.** Whatever is decided for the 35B, per-model
  calibration is what makes that expressible — the table already supports it.

---

## 7. Limits that no server change fixes

- **The corpus is the binding constraint.** Eight English documents, 40 labelled
  spans, 37 fillers. Every number in §2 and §6 is limited by it, not by the
  engine.
- **English is the ceiling arm, not the shipped rate.** German coverage is
  77% against English's 90% at the shipped operating point, and the German
  spans are the ones crowding the 0.705 threshold. Any number in §2 quoted
  without "English" attached is wrong — and §4.3 shows how wrong: measured on
  the decision unit, the bilingual margin is **0.00126** against English's
  0.0153, a factor of **12**, not the 5.6× the span-level figures suggested.
- **The third negative-set defect (supporting context) is unfixed** and caps how
  good any measured false-positive rate can look on this corpus.

---

## 8. Reproducing §2

All from the repo root, `build-metal/bin/attn-provenance`, model
`models/Qwen3.8-9B-Q8_0.gguf` or `models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`
(both calibrated entries).

```
# coverage gate + BAND COST (decision margin, cost curve)
ATTN_LENS_EN_ONLY=1 ATTN_PRODUCT_OCCUPIED=1 ATTN_ALL_OCCURRENCES=1 \
  COVSEARCH=1 QWEN36_MODEL_PATH=models/<gguf> ./build-metal/bin/attn-provenance

# the drift gate — one arm (chunk | flash). Exit code 0 = PASS.
# ATTN_BAND_CHUNK (default 128) sizes the `chunk` arm.
BANDDRIFT=1 DRIFT_ARM=flash QWEN36_MODEL_PATH=models/<gguf> \
  ./build-metal/bin/attn-provenance

# ...or every arm, with a single verdict:
MODEL=models/<gguf> tests/smoke/lens_drift_gate.sh

# §4.4 — does the perturbation change WHICH expert runs? (MoE models only;
# fails loud on a dense recipe, which is the control.) MOEROUTE_MAX caps the
# document count for a quick run.
MOEROUTE=1 QWEN36_MODEL_PATH=models/<moe gguf> ./build-metal/bin/attn-provenance

# multi-layer vote + per-span dump for offline analysis
ATTN_LENS_EN_ONLY=1 ATTN_PRODUCT_OCCUPIED=1 ATTN_ALL_OCCURRENCES=1 \
  COVCOMBINE=1 COVCOMBINE_DUMP=/tmp/spans.csv \
  QWEN36_MODEL_PATH=models/<gguf> ./build-metal/bin/attn-provenance
```

`COVCOMBINE_DUMP` writes one row per scored span (tag, marker, language, label,
token span, text, per-layer peaks). Use it — validation design needs several
attempts and none of them should cost a GPU run.
