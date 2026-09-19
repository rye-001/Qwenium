# Qwen3.8-27B — citation and coverage calibration

**STATUS: MEASURED, row added to `kLensCalibrations` 2026-09-15.** This note is
the provenance the `qwen35/65` entry names. It records what was measured, on
what, and what is *not* claimed.

## 1. Provenance

| | |
|---|---|
| Model | `models/Qwen3.8-27B-Q3_K_M.gguf`, arch `qwen35`, **65 blocks**, Q3_K_M |
| Attention layers | 16, at `il` = 3, 7, …, 63 (48 SSM layers + 1 NextN head block, excluded) |
| Heads | 24 (`n_head_q`) ⇒ **384 (layer, head) candidates** |
| Driver | `tests/perf/attn_provenance.cpp` → `build-metal/bin/attn-provenance` |
| Tap | `ForwardPassBase::set_attention_taps` → `kq_soft.<il>`, `--flash-attn` off |
| Raw logs | `.session-results/qwen38_27b_{n3_search,legC,covsearch,legC_headsearch,heldout_L19H20,legC_L19H20}.log` |

## 2. The constants

```
citation   L19 H20
coverage   L11 @ 0.705
ungrounded 0.538
```

| arm | measurement | bar | |
|---|---|---|---|
| citation top3 | **378/415 (91%)** — EN 92%, DE 90% | ≥90% | PASS |
| coverage used-clear | **73/75 (97%)** — EN 98%, DE 97% | ≥90% | PASS |
| ungrounded false alarm | **0/75 (0%)** | <10% | PASS |
| verbatim | 0/75 normalized | — | |

`── LEG C VERDICT: PASS ──`, bilingual, 15 messy documents. **This is the first
model to clear every arm of leg C.** Qwen 3.8-9B fails it on coverage (87%
used-clear, 83% on German) despite a better citation head.

## 3. How the citation head was selected — and the method defect it exposed

The default N3 leg selects on three synthetic prompts (A/B/C). On this model
that leg picked **L11H22**, which then scored **84.6% on leg C — rank 14 of
384**. A head selected on synthetic prompts is not the head that survives real
documents, and on the 9B this went unnoticed because its A-winner happened to
also be its leg-C winner.

`LEGCSEARCH` (added here) scores every candidate on the corpus that *judges*
them, reading the taps leg C already captures — no extra forward pass, no
per-family knowledge. It is not `GEMMA4_SEARCH_DUAL`, which recomputes rows from
the KV cache to also score the (dead) norm-weighted metric B and therefore needs
gemma4's split SWA/global layout; that leg refuses `qwen35` fail-loud.

Top of the ranking, bar = top3 ≥90% **pooled and on both halves**:

| rank | layer head | top1 | top3 | EN | DE | |
|---|---|---|---|---|---|---|
| 1 | **L19 H20** | 78.8% | **91.1%** | 91.7% | 90.4% | PASS |
| 2 | L47 H19 | 77.8% | 90.8% | 90.4% | 91.4% | PASS |
| 3 | L63 H14 | 32.8% | 90.8% | 92.2% | 89.3% | fail (DE) |
| 14 | L11 H22 | 74.5% | 84.6% | 86.2% | 82.7% | fail (the N3 pick) |

**Cross-language selection does NOT hold**, and this is the honest caveat:

```
select on EN -> L63 H14 (EN 92.2%) -> scored on DE: 89.3%  [does not hold]
select on DE -> L51 H18 (DE 91.9%) -> scored on EN: 88.1%  [does not hold]
pooled winner is NOT the winner of either half
```

L19H20 wins the pool by being *consistent* rather than best in either language.
That is arguably the right property for a shipped constant, but it is not what
the ranking optimised for, and **DE 90.4% sits exactly on the bar** — four
tokens the other way and this arm fails.

## 4. Held-out confirmation

L19H20 was selected on leg C, so leg C cannot also be its gate. Frozen and
scored on the three N3 prompts, which it did not see:

| prompt | top1 | top3 |
|---|---|---|
| A | 24/28 (86%) | 26/28 (93%) |
| B (held-out) | 25/29 (86%) | 28/29 (97%) |
| C (reformat + date conflict) | 23/29 (79%) | 28/29 (97%) |

`bos_mass` 0.016. All three reformatted fields on C hold (`1.250`→`1.25`,
`8,75`, `10.937,50`), and the conflict readout puts **0.507 mass on the
order-date span against 0.133 on delivery** — it picks the right date.

These prompts are synthetic and small (28–29 scored tokens). They are
independent evidence, not a second corpus-grade gate.

## 5. Coverage — searched, not inherited

`coverage_used_peak = 0.705` is carried from COV1 like every other row, but on
this model the *layer* was searched. COVSEARCH over all 16 candidate layers,
matched-FPR, bilingual:

```
G1 used-clear >=90% (candidate at its own threshold) : FAIL — L59 @ 0.915 = 87%
G2 beats incumbent at matched FPR, cross-language    : FAIL
G3 plateau at matched FPR                            : PASS (L55)
── COVSEARCH VERDICT: KEEP layer 11 @ 0.705 ──
```

At matched FPR L59 reaches 75/75 (100%) against L11's 73/75, but the halves pick
different winners (EN→L59, DE→L7), so the pooled winner is not stable. L7, L27,
L43 and L55 also tie or beat L11 at matched FPR — a **plateau**, not a better
coordinate. L11 is kept on evidence, not by default.

**Zero-cost band 0.0018.** One EN filler span sits at 0.7068, just under the
threshold, and 11 of the 12 nearest spans are fillers — drift at this layer buys
false positives, not misses.

The old G3 discriminator saturated again (held AUC 1.000 best *and* median),
third model running. It stays a labelled diagnostic, not a criterion.

## 6. Probe changes this required

- **`SC_MAX_SLOTS` 10 → 20.** `span_scalars` stored per-layer sources at
  `sc[1+slot]` in a `double[12]`; COVSEARCH refused this model fail-loud rather
  than rank a truncated layer list. Indices 0–10 are written identically, so
  every number measured on the 8- and 10-layer models still reproduces — the
  only index that moved is the all-layer max, which no calibrated path reads.
- **`LEGCSEARCH`** — §3.
- **`ATTN_FROZEN_SLOT/HEAD` now steer the N3 confirmation legs.** The facility
  existed ("so the confirmation legs can be pointed at a candidate found on
  another model", `note-lens-qwen38-probe.md` §2) but the legs still froze A's
  own winner. A's search still runs and prints as a diagnostic.

## 7. Non-claims

- **Q3_K_M only.** The 9B row was measured at Q8_0. Whether these coordinates
  survive another quantization is unmeasured, and nothing here says which way it
  cuts. This is what the report's `config.weights` stamp exists to record.
- **`flash_prefill_ok = true` — measured 2026-09-15, after this note's first
  draft.** The drift gate ran both arms bilingually and both PASS:

  | arm | token identity | decisions crossed | max \|Δpeak\| vs margin | headroom |
  |---|---|---|---|---|
  | `chunk` | 15/15 | 0/98 | 0.000404 vs 0.000991 | 2.5× |
  | `flash` | 15/15 | 0/98 | 0.000544 vs 0.000991 | 1.8× |

  The dense prediction held exactly: **15/15 token-identical on both arms**,
  where Qwen3.6-35B-A3B and Gemma 4-26B-A4B each changed one extraction in
  fifteen (`plan-lens-server-shape.md` §4.4). Three gate runs now say the split
  is dense-vs-MoE, not size and not family.

  Two things to carry with the `true`. **Headroom is thin** — the gate prints
  its own THIN warning at 1.8×; passing says this corpus is clean, not that the
  transform is safe, and a line within 0.00054 of 0.705 could be moved across.
  And **English is the binding language here** (EN margin 0.000991 vs DE
  0.020000) — the reverse of the 9B, where German binds at 0.00126 against
  English's 0.0153. Which half binds is a property of the model, so both must
  be run on every entry.
- **Greedy agreement is 27/28 and 28/29** on the N3 prompts, against 28/28 on
  the 9B at Q8_0. That is the quantization talking, and it is a reason to
  re-measure before trusting these coordinates at a different quant.

## 8. Addendum 2026-09-19 — LOCHEAD: a locate head exists, but it is NOT free

**Result: Qwen3.8-27B has the strongest locate signal of any model measured
(L35 h=16 = 93.3% top1 / 100.0% top3, EN 100.0 / DE 100.0).**

**LANDED 2026-09-19: `locate_layer 27, locate_head 10`** — *not* the best head.
The cut moves 20 → **28 of 65**, the first time locate has set a cut. See
"the cost menu" below for why the shallowest bar-clearing head was taken over
the perfect one, and §8.1 for what landing it changed.

### Provenance

| | |
|---|---|
| Model | `models/Qwen3.8-27B-Q3_K_M.gguf`, arch `qwen35`, block_count 65 |
| Attention layers | 16 tapped (3,7,…,63); 48 SSM; +1 NextN head block correctly excluded |
| Heads | `n_head_q`=24 (`n_head_kv`=4) ⇒ **16 × 24 = 384 candidates** |
| Corpus | Leg C messy corpus, 15 docs EN+DE, **75 keys (EN 40 / DE 35)** — same corpus, same keys as the 9B and Gemma legs |
| Driver | `build-metal/bin/attn-provenance`, `LOCHEAD=1` |
| Raw logs | `.session-results/qwen38_27b_lochead.log`, `…_top60.log`, `…_full.log` |
| Process hygiene | all three runs exit 0; `ps aux` shows no surviving process |

**Quantization caveat, same as §1**: this is Q3_K_M where the 9B's numbers are
Q8_0. A rate measured here is not strictly comparable to the 9B's 96.0%.

### The bar, and why the free zone fails it

Today's cut is `max(citation 19, coverage 11) + 1` = **20 of 65**. A locate layer
≤ 19 is therefore free; above it, residency rises. Bar = top-3 ≥ 90% pooled
**and** on both halves.

Best candidates inside the free zone:

    rank | layer head |  top1    top3  |  EN top3   DE top3
      67 | L15   h=17 |   80.0%   90.7% |    92.5%     88.6%   <- DE FAILS
      76 | L19   h=15 |   65.3%   89.3% |    92.5%     85.7%
      79 | L11   h=9  |   62.7%   89.3% |    92.5%     85.7%

**No free-zone candidate clears the bar.** L15 h=17 clears it pooled (90.7%) and
on English (92.5%) and fails on German (88.6%). This is the twice-burned rule
doing its job: an EN-only reading would have called this a free pass and shipped
a locate route on the 27B at zero cost. German refused it.

### The cost menu, if locate is wanted on this model

    layer head | top1    top3  |  EN      DE     | cut      | vs today
    L27  h=10  | 81.3%   94.7% |  97.5%   91.4%  | 28/65    | +8 blocks
    L35  h=16  | 93.3%  100.0% | 100.0%  100.0%  | 36/65    | +16 blocks

`L27 h=10` is the **shallowest candidate clearing the bar on both halves**.
`L35 h=16` is the best overall; six further candidates also hit 100.0/100.0/100.0
(L39 h=7, L47 h=17, L43 h=22, L47 h=2, L39 h=12, L47 h=1), so the top of this
table is a broad plateau, not a single lucky head.

Both axes move together here and both should be quoted:
- **residency** = `max(citation, coverage, locate) + 1` — what the verify-only
  server keeps resident: 20/65 → 28/65 or 36/65.
- **per-route compute** — `/v1/locate` truncates after `locate_layer` alone, so
  the locate route itself costs 28 or 36 blocks of 65 rather than today's 20.

**LANDED: `L27 h=10`.** The +8 blocks were accepted deliberately. L35's perfect
score was declined: 100.0% vs 94.7% is a 5.3-point gain for a further 8 blocks,
and top-3 is already the mode the client is told to use — the span hull, not a
single span. Where L35 would genuinely pay is top-1 (93.3% vs 81.3%), which is
the rate a caller acting on ONE span is exposed to; if a client ever commits to
top-1 on this model, revisit this trade rather than assuming 94.7% covers it.

### 8.1 What landing it changed

- `src/server/server_lens.h` — the `{qwen35, 65}` row's locate triple.
- `src/server/http_server.cpp` — the "LOCATE BELONGS IN THIS MAX" comment was
  written when the max was inert. **It is now load-bearing**: this is the only
  row where dropping locate from `max(citation, coverage, locate)` leaves every
  other model working while serving `/v1/locate` an unloaded block.
- `tests/unit/test_server_lens.cpp` — `TwentySevenBLocatePairIsSweptAndSetsTheCut`
  asserts the cut is 28 **and** that it would have been 20 without locate, so the
  accepted cost cannot drift unnoticed;
  `LocateDepthDirectionIsNotAFamilyConstant` pins the 9B/27B disagreement.
- `docs/architecture.md` — three passages, two of which claimed locate never
  bites; one was already stale against the code.
- `tests/smoke/server_locate_smoke.sh` — per-model rate list.

### 8.2 Live verification (2026-09-19)

`tests/smoke/server_locate_smoke.sh` run against this model on **both** legs —
`VERIFY_ONLY=1` (the cheap server, where gate 8's shipped defect reproduces) and
the full server. **9/9 gates PASS on both, byte-identical output**, `shape`
confirming the route reads `L27H10`.

The startup banner is the cut proving itself, and the reason the verify-only leg
is the one that matters here:

    --lens-verify-only ON: loaded 28/65 blocks — cut at
    max(citation_layer=19, coverage_layer=11, locate_layer=27) + 1,
    calibration Qwen3.8-27B [qwen35/65]
    Serving POST /v1/verify and POST /v1/locate

`locate_layer=27` is visibly the term winning that max; before this landing the
same banner read 20/65. Gate 8 (no-poison) passing on the verify-only server is
the specific regression check that a locate-then-verify sequence does not corrupt
the citation tap — the 2026-09-18 false-receipt defect — and it now holds with
locate on its own deeper layer.

One gate reading that is expected and not a regression: `floor` reports **3 of 4**
labelled values found (`customer`, `unit_price`, `order_date`; `quantity` missed).
The floor is 2. Four keys on one document cannot re-measure a 94.7% rate, and the
smoke says so itself — the rate lives in `locate_provenance`, not in this gate.

### Two findings beyond the constant

**1. The incumbent citation head is near-worst at locate, on a third model.**
L3 h=13 ranks **372 of 384** (18.7% top-3). Combined with the 9B and Gemma legs,
citation and locate are now independently confirmed as different circuits on
three models. Reading locate off the citation head is not a degraded option; it
is wrong.

**2. The 9B's depth story does NOT generalize — the direction flips.**

| model | citation | locate peak | direction |
|---|---|---|---|
| Qwen3.8-9B (33 blk) | L27 = 82% depth | **L11 = 36%** | locate shallow, citation deep |
| Qwen3.8-27B (65 blk) | L19 = 31% depth | **L35 = 55%** | **citation shallow, locate deep** |
| Gemma 4 12B (48 blk) | ~63% best, no head | L19–24 ≈ 45% | neither; both ~65% |

The 9B result was written up as "locate is shallow, citation is deep", with the
rationale that key→source matching resembles induction / previous-token heads and
should therefore mature early. **That rationale is refuted here.** On the 27B the
shallow layers are visibly weak at locate (L3 = 44.0%, L7 = 84.0%, L11 = 89.3%)
and the circuit does not mature until L35. The claim that survives is only the
weaker one: *locate and citation live at different depths within a model, and the
direction must be measured per model, never assumed.* The 9B's free locate pair
was a happy coincidence of that model, not a law.

### Probe change this required

`LOCHEAD_TOPN` (default 15, unchanged behaviour) makes the ranked-list length
configurable. The decision above turns on rows at rank 67–79, which the fixed
top-15 print could not show; re-running a 6-minute sweep to see one row is the
alternative it removes. Additive and arch-agnostic.
