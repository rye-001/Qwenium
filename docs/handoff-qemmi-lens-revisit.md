# Handoff — Qemmi-Lens revisit: exploit the thread finding (2026-09-05)

**Read this before touching the lens.** It exists because one measurement on
2026-09-04 changed what the lens is *for*, and because the app was built in July
against a July server that has grown substantially since.

**START HERE:** §1 (what changed), then §3 (the one decision that gates
everything), then §7 (what is permanently closed — four dead ends, do not
re-propose them).

---

## 1. What changed — the SS2 result

`docs/note-ss2-thread-alarm.md` (2026-09-04). The retrieval head was validated
on **multi-turn email threads** for the first time:

| | single document | thread (4774–6200 tok) |
|---|---|---|
| citation top-1 | 89% | **89%** |
| citation top-3 | 98% | **98%** |

**No degradation at all.** This closes the last open item from the seven-probe
hunt — memory carried "multi-turn transfer is the honest open frontier" since
July. It is not open any more.

It matters because the threads are **adversarial by construction**: real email
replies quote their history, so the superseded value literally reappears below
the correction. The same string occurs several times in several messages, and
the head still picks the right occurrence.

Two secondary results, both weaker, both honest:
- **Stale is real at length.** 1 of 8 corrections was mishandled at ~5–6 K
  tokens. SS1 saw **0** at 764 tokens. Length is what makes the failure appear.
- **The coverage-free alarm works end to end**: TP=1, FN=0, FP=0, TN=7. The
  **0/7 false-alarm rate is the trustworthy half**; recall is n=1 and must never
  be quoted as a rate.

**The one failure profile:** correction at message 22 of 24, phrased indirectly,
with a distractor restating the old value. A correction just as late but stated
*explicitly* was handled correctly. Hypothesis from one case: **indirect
phrasing is the discriminator, not position.** It is also the recipe for
generating more stale cases.

**Do this first, before any feature work:** power up Gate 2 by building 8–10
threads with the late + indirect + distractor profile. Thread definitions only,
no code change, one run. Building a product on a single true positive is exactly
what the two-signal rule exists to prevent — it is what caught Gemma's L4H7 at
51% after it passed three held-out prompts.

## 2. Why this changes what the lens IS

Single-document citation is commoditized — the market probe (2026-07-20) said so,
and concluded the **omission report is the only unique asset**, with no obvious
buyer. Threads change the job:

> *"This answer came from message 3, which message 22 superseded."*

That is not a citation, it is a supersession claim, and an embedding-retrieval
baseline cannot produce it. It is the omission report finally having a shape
someone needs.

**Scope note:** the lens is DORMANT on the market side per the user (2026-08-21,
*"the concept is correct & that suffices"*). This handoff is an **engineering**
brief. Do not attach a go-to-market; that was closed deliberately.

## 3. THE GATING DECISION — per-model LensConstants

Everything below is blocked on this, and it needs user approval before it lands
(CLAUDE.md architecture protocol: `LensConstants` is a shipped receipts type).

`src/server/server_lens.h` today:
```c
int    citation_head        = 13;      // L3H13
int    citation_layer       = 3;
int    coverage_layer       = 11;
double coverage_used_peak   = 0.705;
double ungrounded_body_mass = 0.538;
constexpr const char* kLensCalibratedArchitecture = "qwen35moe";
```

**The shipped lens runs the weaker head.** Measured on the same messy corpus:

| | Qwen 3.6 L3H13 (shipped) | Qwen 3.8 L27H13 |
|---|---|---|
| citation top-3 | 84% | **98%** |
| ungrounded false-alarm | 7% | **0%** |

Fourteen points of accuracy and a nonzero false-alarm rate, unreachable because
the constants are one frozen struct behind a `qwen35moe`-only refusal.

**And SS2 was validated on Qwen 3.8-9B (`qwen35`) — a model the shipped lens
refuses to run.** We validated on something we do not serve. Either the
constants become per-architecture, or the thread work cannot ship. This was
already flagged as a decision in `note-lens-qwen38-probe.md` §6.2; SS2 makes it
blocking rather than optional.

**Second, smaller constant decision:** `coverage_used_peak = 0.705` was frozen
from COV1 on Qwen 3.6 and is stale. Recalibrated on Qwen 3.8 it scores 68/75
(91%) vs 65/75 (87%), clearing the ≥90% bar it has been failing on *both* Qwen
models (`note-lens-norm-weighted-metric.md`). Caveat: calibrated on 12+11 spans —
treat "recalibration is worth ~4 points" as the finding, **not** the value 0.453.

## 4. Engine / format gaps

**4.1 Wrong unit.** `docs/lens-format.md` is document-shaped; `/v1/extract` takes
`{document, keys}`. A thread alarm needs **message boundaries in the request** —
they cannot be inferred from concatenated text, and the whole alarm depends on
mapping a cited span to a message index. This is a `lens-format` change (a v3, or
a second endpoint) and is the first thing to specify.

**4.2 Warm prefix is unused, and threads are *the* warm case.** `conversation_id`
+ `--prefix-cache` shipped in PR #22 at a measured **2.4×@4K**. A mailbox thread
grows one message at a time — textbook strict-prefix. Every extraction today
re-prefills the whole thread cold. This only became relevant on 2026-09-04:
while the unit was one document, cold was the only option.

**4.3 Single-slot EXCLUSIVE.** `--attention-lens` holds the entire server against
a ≤10-slot envelope. Re-test whether exclusivity is still required or was V1
conservatism; it is a hard throughput ceiling on the lens's own workload.

**4.4 Live envelope defect (independent of the lens).** SS2 v1 hit
`kIOGPUCommandBufferCallbackErrorOutOfMemory` at **8292 tokens** — a 9 B Q8_0
with a 576 MB KV cache could not complete an 8.3 K prefill on this host with the
tap armed. CLAUDE.md claims ≤10 K. Avoided in v2 by capping threads at 5800, not
resolved. `--kv-f16` is the first lever. Worth its own check.

## 5. App-side audit — `../qemmi-lens` (TypeScript, pnpm, 1345 LOC)

| module | LOC | state |
|---|---|---|
| `src/web/page.ts` | 379 | the UI — largest module, see §6 |
| `src/client/lens.ts` | 129 | the driver |
| `src/lens-format.ts` | 97 | zod schema for the v2 format |
| `src/cli/index.ts` | 97 | CLI |
| `src/export.ts` / `summary.ts` / `presets.ts` / `vocabulary.ts` | 63/58/57/39 | support |

**What is good and should not be disturbed:** the fail-loud error contract is
genuinely well done — 422 is surfaced as *"the model's output could not be
parsed... no fields were reported rather than some of them. Route this document
to a human"*, which callers can act on without string-matching. The README's
honesty framing (*"attention marks consideration, not commitment"*, and the
no-grammar explanation) is accurate and matches the engine. Keep both.

**Gaps:**
- **`lens.ts` is stateless single-shot by design** — *"one document + a complete
  key vocabulary"*. No `conversation_id`, no thread, no message boundaries. It
  cannot express the new unit, and it cannot reuse warm KV (§4.2).
- **No supersession concept anywhere.** Badges are `grounded` / `ungrounded` /
  `absent`. A thread needs a fourth state — **stale**: *this value cites message
  3, which message 22 superseded.*
- **Stale footer.** The page footer reads *"Single document, ≤4K tokens
  validated."* Both halves are now wrong: threads at 5–6 K are validated, and
  the envelope is 10 K (subject to §4.4).
- Coverage heat is document-wide; for a thread it must be **per-message**.

## 6. UI brief — for the design pass

`src/web/page.ts` (379 lines) is the whole UI: two panels (Document | Audited
fields), tabs `fields` / `heatmap`, heat modes `coverage` / `citations`, badge
chips, hover tooltips showing up to 6 citations, and JSON/CSV export.

**The single most important design note.** The market probe found the **omission
report is the only unique asset** — the one thing competitors structurally cannot
produce. In the current UI it is `<div id="coverageblock" hidden>`, behind a tab.
**The product's only differentiator is hidden behind a tab.** If one thing
changes in a design pass, it is that.

Second: the UI is built for **one document**. A thread is a different visual
object — ordered messages, the correction highlighted, the superseded message
dimmed but still visible (users need to see *what* it superseded, not just that
it did), and the citation drawn as a link from the emitted value to its source
message.

Third, and this constrains everything: **the honesty contract is load-bearing,
not decoration.** `docs/lens-format.md` §"Non-claims" lists what the lens refuses
to say. The UI must never imply the model *chose* correctly — only where it
looked. Conflicts ship as coexisting keys with separate citations, and the user
resolves them. A design that makes the lens look confident is a design that
breaks the product. Any design pass should read `lens-format.md` §28–71
(the honesty contract and non-claims) before touching the page.

## 7. PERMANENTLY CLOSED — do not re-propose

Four dead ends, each closed by measurement. This list exists to save cycles.

- **Images.** Dead on *both* taps. Decode: target mass exactly 0.000 at depth
  while the model answered correctly (`note-image-lens-probe.md`). Prefill: the
  winning head tracks *iff the named object is on the right* — a left/right
  positional artifact (`note-image-prefill-tap-probe.md`).
- **The fixed KV grammar.** Refuted by measurement 2026-07-16, removed from the
  product path 2026-07-17 (`note-nogrammar-refutation.md`). **Warning:** SS1 §5
  recommends a grammar-constrained extractor — SS1 is dated 2026-07-12, *four
  days before its own refutation*. That stale recommendation was followed once in
  SS2 v1 and produced syntactically valid infinite JSON. Do not follow it.
- **Norm-weighted attention** (`α·‖V‖`). NEUTRAL on Qwen; **inert by
  construction** on Gemma — `gemma4.cpp:447` RMS-norms V with no learned weight,
  forcing `‖V‖ = √head_dim`, so Spearman(rankA, rankB) = 1.0000 over all 768
  candidates (`note-lens-norm-weighted-metric.md`, `note-lens-gemma-norm-weighted.md`).
- **Gemma entirely.** 0 of 768 candidates clear even a 70% bar. A properly-run
  search on the messy corpus tops out at 63% (L7H13) against a 90% requirement.
- **Scalar attention confidence.** Died three separate times (CG1, and twice in
  the margin work: *"margin measures difficulty, not ambiguity"*).

**Cross-family consequence — needs an explicit `architecture.md` entry.**
CLAUDE.md makes Gemma the falsifier for anything touching the forward pass. Gemma
has been *measured* to have no citation head, so the lens is a **Qwen-family
capability by measurement, not by neglect**, and the architecture refusal is the
mechanism that keeps that honest. Write this down as a decision, or a future
reviewer will read it as a violation.

## 8. Suggested order

1. **Power up Gate 2** (§1) — thread definitions only, one run. Cheapest, and
   everything else rests on it.
2. **Decide per-model `LensConstants`** (§3) — user approval; gates all feature
   work; also settle `coverage_used_peak`.
3. **Specify the thread request shape** (§4.1) — `lens-format` v3.
4. **Then** app + UI work (§5, §6), warm prefix (§4.2), exclusivity (§4.3).
5. Independently, whenever: the 8.3 K OOM (§4.4).

## 9. Provenance

Probe notes, all uncommitted in the working tree as of 2026-09-05:
`note-ss2-thread-alarm.md`, `note-lens-norm-weighted-metric.md`,
`note-lens-gemma-norm-weighted.md`, `note-image-prefill-tap-probe.md`.
Prior: `note-stale-source-probe.md` (SS1), `note-lens-qwen38-probe.md`,
`note-lens-gemma4-probe.md`, `note-image-lens-probe.md`,
`note-nogrammar-refutation.md`, `note-draft-pointer-probe.md` (DP1),
`market-probe-qemmi-lens.md`. Harness: `tests/perf/attn_provenance.cpp`
(`SS2=1`, `NORM_WEIGHTED=1`, `GEMMA4_SEARCH_DUAL=1` — all env-gated, byte-inert
when unset). App: `../qemmi-lens`.
