# Plan — the candidate set (server contract)

Status: **approved, 2026-09-06 — on the wire.** Producer decided and gate 2
passed (2026-09-06, probe appended below); gates 1 and 3 remain open (see
*Gates* below) and gate the surface's *contested* state, not this wire shape.
The architect settled the three open implementation decisions this document
left as judgment calls, and they are now shipped exactly as decided:

- **Producer failure** renders as `key_candidates` **omitted entirely** plus a
  top-level `candidates_error` string — never as an empty, unflagged set. This
  yields three distinguishable states: absent + error = the finder failed;
  absent + no error = candidates were not requested; present = it ran.
- **`returned_as`** links by **bidirectional containment** on
  whitespace-normalized text (pass 2 legitimately returns wider or narrower
  spans than the field value). Tiebreak when several candidates qualify:
  **tightest containment first, then earliest `byte_lo`.** At most one
  candidate per occurrence.
- **`anchor`** is the nearest preceding label-like line (a short line ending in
  `:`, or a numbered/underlined heading), else the enclosing line trimmed and
  truncated to ~60 chars, else **`null`** — nullable is deliberate, a byte
  offset is not a place.

This changed what `POST /v1/extract` emits, which is a named seam
([`lens-format.md`](lens-format.md)); `architecture.md` §13 and
`lens-format.md` were updated in the same change (`qemmi-lens/v3` →
`qemmi-lens/v4`). `../qemmi-lens`'s `ACCEPTED_FORMAT_VERSIONS` has since been
given `"qemmi-lens/v4"` in that repo's working tree, together with a zod
`LensCandidate` mirroring this shape (45/45 tests green, 2026-09-06) —
**uncommitted there**, and until it is committed that repo's fail-loud version
gate fails *every* extract call, not only candidate ones.

Scope of *this* document: **only what the server returns.** The surface that
consumes it (card layout, the absent-split, the resolve affordance) is a
separate plan.

## The feature in one line

Alongside the value it already returns, the server returns **every span in the
document that answers the key** — with the emitted value marked as the one that
was *returned*, never as the one that is *right*.

## Why (what today's payload cannot express)

Three of the five things a reviewer needs are structurally unsayable in v3:

| today | the fact that is lost |
|---|---|
| `monthly_rent: "1,375.00 GBP"` `grounded` `body_mass 0.963` | clause 2 of the same contract says `1,450.00 GBP`. Present only as 2 of 8 citations, masses 0.079 / 0.062, mid-pack. |
| `impression` `grounded` `· not verbatim in doc` | the value differs from its source by one character (`L4-5` for `L4-L5`) — the *same* flag fires on whitespace reflow. |
| `line_item` × 3 | there is no denominator. v3's own rationale calls the silently-incomplete list "the one thing this format exists to prevent". |

None of these is a confidence problem. Measured on three documents, the one
wrong value scored `body_mass 0.975` and outranked **7 of 22** correct ones —
CG1 holds, and no threshold separates them. What is missing is not a better
score, it is **negative space**: what the document offered that did not become
the answer.

## Producer

**Cold two-pass** (decided 2026-09-06). Pass 1 is today's extraction, untouched.
Pass 2 is a separate cold inference over the same document, different
instruction, **taps disarmed** — cheaper than pass 1, since disarming the taps
makes flash attention available. Warm reuse is set aside: checkpointing the
document prefix would require splitting prefill at the document boundary, and
chunked-vs-one-shot prefill is not bit-identical on Metal, so it would perturb
pass 1. Cold is the reference; warm must earn its way in by matching it.

## Shape

Candidates are keyed **per key, not per field entry** — v3 may emit several
`fields` entries for one key (occurrences), and duplicating the array into each
would state the denominator N times and make "4 candidates, 3 returned"
unreadable.

Each key maps to a **bare array**. An earlier draft of this section gave each
key an object (`{status, candidates}`) so a per-key `not_addressed` could be
distinguished from `[]`; **that was dropped as unbuildable** — pass 2 emits
lines, and a key the model simply never wrote is indistinguishable on the wire
from a key it wrote as `(none)`. A per-key status would have been a receipt for
a distinction the producer cannot actually make. Emptiness is instead stated at
the **document** level (see *Producer failure* below), which the producer *can*
observe:

```json
"key_candidates": {
  "monthly_rent": [
      { "value": "1,450.00 GBP", "byte_lo": …, "byte_hi": …,
        "anchor": "2. RENT",         "returned_as": null },
      { "value": "1,375.00 GBP", "byte_lo": …, "byte_hi": …,
        "anchor": "AMENDMENT No. 1", "returned_as": 0    } ],
  "pets_policy": [],
  "supplier":    []
}
```

- `value` — **byte-exact slice of `document`**, always. The field's `value` may
  not be verbatim; a candidate's always is. That asymmetry is what makes the
  mis-copy visible without a diff engine.
- `byte_lo` / `byte_hi` — into `document`, same convention as `citations` and
  `value_span`.
- `anchor` — the structural location a human can navigate to (`"2. RENT"`,
  `"AMENDMENT No. 1"`, `"From: line"`). A byte offset is not a place. `anchor`
  carries context so the span itself does not have to (see *Linking*).
- `returned_as` — the `occurrence` of the `fields` entry this candidate was
  returned as, or `null`. At most one candidate per occurrence.

`fields` is **unchanged**. Every v3 member keeps its exact current meaning;
`badge`, `body_mass`, `citations`, `tier`, `occurrence` are untouched, and so
are the attention taps, the calibration, and coverage. Additive on the wire —
no importer breaks, the CLI ignores it.

**Array order is document order (`byte_lo` ascending), and that is load-bearing.**
Not mass, not any ranking. Position is a fact about the document; every other
ordering is a verdict, and CF1 forbids one.

## Producer failure is a state, not an empty set

Absorbed from the probe: a pass-2 parse failure rendered as *"this document
offers no answers"* is the exact confusion this format exists to prevent,
reproduced one level down — the same lesson
[`note-nogrammar-refutation.md`](note-nogrammar-refutation.md) paid for on
pass 1. Emptiness must never be inferred; it must be stated.

**Three top-level states, mutually exclusive on the wire.** Check
`candidates_error` first.

| `key_candidates` | `candidates_error` | meaning |
|---|---|---|
| absent | present (a string) | the finder **failed** on this document — every key is *unknown*, not *nothing found* |
| absent | absent | candidates were **not requested** for this extract |
| present (object, arrays possibly `[]`) | absent | the finder **ran** |

Pass 2 unparseable → `candidates_error` set and `key_candidates` **omitted
entirely**. Report nothing rather than part of it, exactly as pass 1's 422 does.
A v4 response with no `key_candidates` means *the finder failed or was not
asked*; a v3 response means *this server has no finder*. That distinction is the
whole reason the version bumps.

**There is no per-key status.** When `key_candidates` is present the server
guarantees an entry for every vocabulary key, and `[]` there means the finder
ran and reports nothing for that key. It does **not** separate "the model wrote
`key: (none)`" from "the model never mentioned this key" — both collapse to
`[]`, because pass 2's output cannot tell them apart. Do not re-derive a
distinction the wire does not carry.

## Linking `returned_as`

Absorbed from the probe: equality linking scored 62/73 (85%) because pass 2
legitimately returns different span boundaries — wider (`"45 units"` for `45`,
`"PO number 5590-B"` for `5590-B`) and occasionally narrower. Containment
reaches ~99%.

- **Rule.** Bidirectional containment on the **whitespace-normalized** text: a
  candidate links to an occurrence if either string contains the other after
  collapsing runs of whitespace.
- **Tiebreak.** When several candidates contain the value, the **narrowest**
  containing span wins.

The probe asks whether the wider span is the better candidate. **It is not, and
this is a contract decision, not an implementation detail.** A systematically
wider candidate makes *every* field's value differ from its candidate, which
re-creates precisely the failure this feature exists to fix — `· not verbatim in
doc` fired so often it stopped being read. Context belongs in `anchor`, which
exists for it; the span stays tight so the comparison stays sharp.

That combination is worth more than a patch. Whitespace normalization absorbs
the benign case (the radiology `primary_finding`, reflowed line breaks → links
cleanly, no alarm) while a real one-character drift still fails to link (`L4-5`
against `L4-L5` → no link, alarm). The two facts that today share one gray
footnote are separated by the linking rule itself, with no diff engine.

## The states it must express

| candidates | `returned_as` | field `value` | fact |
|---|---|---|---|
| omitted (+ `candidates_error`) | — | any | **finder failed** — says nothing about the document |
| `[]` | — | `null` | true absence — the document says nothing |
| `[]` | — | non-null | derived / fabricated (today's `ungrounded`, re-derived structurally) |
| ≥1 | none | `null` | **recall failure** — an answer was there and was not returned |
| ≥1 | none | non-null | **value matches no candidate** — mis-copy or wrong source |
| 1 | 1 | non-null | ordinary |
| ≥2 | 1 | non-null | **contested** — the reader adjudicates |
| n | k < n | non-null | **incomplete list** — the missing denominator |

Rows 2 and 4 are both `badge:"absent"` today and are different facts with
different actions (accept vs. re-key). Row 5 is where the `L4-5` mis-copy lands,
and it is a strictly stronger signal than `found_in_document:false`, because it
is relative to a candidate rather than to the whole document. Splitting these is
why the set must ship even when it is empty — `[]` and an absent
`key_candidates` are different facts, so the server emits an entry for **every**
key in the vocabulary whenever the map is present at all.

## What a candidate is — the discipline

A candidate is a span the document **offers as an answer to this key**, not a
token of the matching type. Every currency amount in a lease is not a candidate
for `monthly_rent`; the deposit is a different concept that happens to be money.

This is the whole risk. A set that is reliably 6 long is exactly as useless as
the green badge it replaces, and worse, it relocates the trust problem onto the
candidate finder's recall.

## Non-claims (inherited, unchanged)

The candidate set is **attribution, not adjudication** — it reports what the
document offers and refuses to rate the offers. No candidate is marked correct,
chosen, primary, later, or live. `returned_as` is a statement about what the
model emitted, not about which span deserved it. CF1 (no conflict winner), SS1
/SS3 (no staleness), CG1 (no confidence) and "no correctness" all survive
verbatim; nothing here is a route around them.

## Gates

Same bar that killed the fixed grammar, the presence gate and the thread alarm.
On `qdocs_messy_corpus`:

1. **Precision** — fraction of emitted candidates a human accepts as answering
   the key. Bar **≥ 0.90**. *Open — needs human scoring.*
2. **Median set size on uncontested keys** — must be **1**. *PASSED
   2026-09-06: median 1.0, `0=0 1=71 2=2 3+=2`, byte-exactness 81/81.*
3. **Recall on known conflicts** — on a seeded conflict corpus (the lease shape:
   a clause and a later amendment), the losing span appears as a candidate.
   Bar **≥ 0.90**. *Open — the corpus does not exist yet.*

Failing (1) or (2) kills the feature rather than tuning it — a candidate finder
that over-reports trains the reviewer to skip the set, which is the exact
failure mode of `· not verbatim in doc` today.

**Gate 2 is a noise gate, not a proof the feature works.** It says the sets are
not padded; it says nothing about whether the losing span in a conflict is ever
found. The **contested** state — the lease case that motivated this whole plan —
rests entirely on gate 3 and is unvalidated until that corpus exists. It must
not reach the surface before then.

## Version

Bump to **`qemmi-lens/v4`**, additive. It would ride in place by precedent
(v0→v1 were additive), but the client's `ACCEPTED_FORMAT_VERSIONS` check is a
fail-loud gate, and without a bump a reader cannot distinguish *"the finder
failed on this document"* from *"this server does not produce candidates."*
Those are different facts; the format exists to keep them apart.

## Open

- `anchor` derivation on documents with no structural markers (prose, an email
  body). Fall back to what? A line number is honest but not a place.
- Whether `key_candidates` should carry keys the caller never hinted — a span
  answering a concept nobody asked for is the coverage report's job, not this.
- Warm second pass, if it can be shown to match cold bit-for-bit.

---

## Viability measured — the cheap kill gate PASSES (2026-09-06)

Producer decided: **cold two-pass.** Pass 1 is today's extraction, untouched.
Pass 2 is a separate cold inference over the same document with a different
instruction and **taps disarmed** (it needs no citations — which also makes
flash attention available on that pass, so it is cheaper than pass 1).

Warm reuse was investigated and set aside: both `RecurrentState` and
`DeltaNetState` do support `checkpoint`/`restore`, so a warm second pass is
mechanically possible on the hybrid — but taking the checkpoint requires
splitting the prefill at the document boundary, and chunked-vs-one-shot prefill
is not bit-identical on Metal. That would perturb pass 1, which this plan
forbids. Cold is the reference; warm must earn its way in by matching it.

Measured on `qdocs_messy_corpus` (15 EN+DE docs, 75 uncontested keys),
harness path `CAND=1` in `tests/perf/attn_provenance.cpp`:

| gate | result | |
|---|---|---|
| **2 — median set size, uncontested keys** | **median 1.0**, distribution `0=0 1=71 2=2 3+=2` | **PASS** (bar: exactly 1) |
| byte-exactness of candidates | 81/81 (100%) | required by the format |
| `returned_as` coverage | 62/73 exact (85%); **~99% under containment** | recall proxy |
| producer failures | 0 | see below |

Gate 2 passed *despite* a pass-2 instruction that nudges toward splitting
(it tells the model a clause+amendment pair is two spans). That makes the pass
stronger, not weaker.

### Two things this plan should absorb before it lands

**1. Producer failure is a FOURTH state, and §"The states it must express" does
not have it.** The first run reported one document (`m_en1`) as having zero
candidates for every key. It had not: the model emitted every line *unquoted*,
the strict parser dropped all of them, and the probe rendered a parser bug as
*"this document offers no answers."* That is the exact confusion this format
exists to prevent, reproduced inside the feature. `[]` must be distinguishable
from *"the finder produced output we could not read"*, just as `[]` is already
distinguishable from absent-from-the-map. Fixed in the harness by a tolerant
parse plus a loud, separately-counted producer-failure path — the same lesson
[`note-nogrammar-refutation.md`](note-nogrammar-refutation.md) already paid for
on pass 1.

**2. `returned_as` cannot link by equality.** Pass 2 legitimately returns
different span boundaries than the field value — wider (`"45 units"` for `45`,
`"PO number 5590-B"` for `5590-B`) and occasionally narrower (`"300 x"` for a
longer product string). Linking needs **bidirectional containment**, and
"at most one candidate per occurrence" needs a stated tiebreak when several
candidates contain the value. Arguably the wider span is the better candidate —
`"PO number 5590-B"` tells a reviewer where it lives — but that is a contract
decision, not an accident to leave to the implementation.

### Still untested

- **Gate 1 (precision ≥ 0.90)** — needs human judgement; the probe prints every
  candidate set in document order for eyeball scoring on a sample.
- **Gate 3 (conflict recall ≥ 0.90)** — needs the lease-shaped seeded conflict
  corpus, which does not exist yet. Until it does, the *contested* state is
  unvalidated.

---

## UI implementation (separate session)

Decisions and constraints only. Everything below is a choice that cannot be
re-derived from the payload; anything not listed here is the implementer's call.

### The one structural change

**The card unit moves from per-field-entry to per-key.** Candidates are keyed
per key, and the denominator (`n found · k returned`) belongs to the key, not to
an occurrence. Occurrences list *inside* one key card.

This does not violate the existing "cards stay a flat list, never grouped into a
table" rule in `page.ts` — that rule forbids aligning occurrence N of one key
with occurrence N of *another*. Grouping within a single key makes no cross-key
claim.

### Additive on the wire, replacing on the surface

The candidate set must not go behind a disclosure triangle, a tooltip, or a
click. The whole finding this feature answers is that `· not verbatim in doc`
was true, present, and invisible. If a reviewer has to open something to learn a
key was contested, the feature has not shipped.

### State → render

Render each key from the top-level state + its candidate array + the field
entries, per the state table above. Required distinctions:

| state | must read as |
|---|---|
| finder failed (`key_candidates` omitted, `candidates_error` present) | *we don't know what this document offers* — never as an empty set, never as absence. Applies to the **whole document**, not per key |
| `[]`, value `null` | the document offers nothing for this key |
| `[]`, value non-null | value came from nowhere in the document (today's `ungrounded`) |
| ≥1 candidate, none returned, value `null` | **an answer is present that was not returned** — visually distinct from the row above it; these are the two halves of today's single `absent` badge |
| ≥1 candidate, none returned, value non-null | **the returned value matches no candidate** |
| 1 candidate, returned | ordinary — value and candidate stacked (below) |
| ≥2 candidates, 1 returned | contested (see gating) |
| n candidates, k < n returned | `n found · k returned` |

**Stacking.** When a candidate is linked, show the returned value and the
candidate on adjacent lines, left-aligned, same font. That adjacency *is* the
mis-copy detector; do not add a diff renderer, and do not suppress the candidate
line when the two are identical after whitespace normalization — a reader who
only sees the pair when something is wrong cannot calibrate what right looks
like.

### Forbidden

- **No ordering signal.** Candidates render in payload order (document order).
  No sorting, no "primary/alternate", no mass, no percentage, no visual weight
  difference between returned and unreturned beyond a neutral marker. CF1.
- **No count phrased as a total.** `2 found`, never `2 exist`; and nothing that
  reads as *"no conflicts here"*. See gating.
- **No badge derived from candidates.** `badge` keeps its v3 meaning and is
  rendered from `badge` alone.

### Gating on gate 3

Gate 3 (does the *losing* span in a conflict ever get found) is unrun. The
consequence is asymmetric and should be implemented as such:

- **Showing a found rival is safe** — byte-exactness is 100%, so a candidate
  that exists is a real span. Display it.
- **Implying completeness is not.** The absence of a rival must never read as
  reassurance, and a count must never read as a total. This is a copy
  constraint, not a feature flag: get the wording right and no flag is needed.

### v3 compatibility

A payload with no `key_candidates` must render **exactly as today**, unchanged.
The new surface is entered only on `qemmi-lens/v4` with `key_candidates`
present. Accept v4 in the client's version gate.

### Two things that break if left alone

1. **Document marking gets better and should be taken.** Field clicks currently
   mark the top-8 citations, which renders as single-character confetti (a date
   marks as `0 2 5 0 - 2 4 3`). A candidate is a clean span — mark the candidate
   span instead. Keep the citation view where it is, in the Heatmap tab.
2. **The footer provenance line will lie.** It prints `model · citation_source`,
   and pass 2 runs with taps disarmed — the candidate set does not come from the
   citation head. Either scope that line to the view it describes or give the
   candidate set its own provenance. Do not leave it labelling both.

### Prerequisite (not part of the feature)

`page.ts`'s Extract handler clears `#fields`, `#audit`, `#exports`, `#controls`,
`#groundBanner` and `#prov`, but never `#doc`, `#doctok`, `#gentok`, `#instrtok`
or the tooltip, and never re-runs `applyView()`. Observed live: submitting a
second document leaves the *previous* document's tokens, raw output and hover
tooltip on screen, and a 422 renders the refusal directly above a fully
citation-highlighted different document. Fix this first. It is a correctness
defect against "the lens never lies about where the model looked," and this
feature adds per-key state to the same page.

### Out of scope

Accept / correct / escalate per field. Storing a human verdict turns a stateless
viewer into a system of record and pulls in review identity, timestamps, and
re-extraction invalidation. Separate decision, not a UI detail.

### Acceptance

Fixtures covering every row of the state table, including finder-failed
(`candidates_error`) and not-requested (both members absent), each rendering
distinguishably. The server side lands
separately, so build against committed fixtures rather than a live v4 endpoint.
