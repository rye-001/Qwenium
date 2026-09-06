# The Lens Format — Qemmi-Lens interchange contract (v4)

*The versioned data contract that `POST /v1/extract` emits.* This is what a
target system's **importer** reads. Mapping, typing, and validation into your
own schema are **your job, not ours** (plan §0) — this document is the boundary.

Status: **v4** (2026-09-06, docs/plan-candidate-set.md — architect-approved).
Over **v3**, the top level gains **`key_candidates`**: for a request that opted
into the candidate-set producer, every span in the document that could answer
each requested key, not just the one the model returned. `fields` is completely
unchanged — same members, same meaning, same taps and calibration. This is
**additive on the wire** (a v3 importer that ignores unknown top-level members
reads the rest of the payload unaffected), but the version still bumps, because
the **absence** of `key_candidates` is now a fact an importer must be able to
read: a v4 response with no `key_candidates` means *the candidate finder failed
on this document*; a v3 response means *this server has no finder at all*. An
importer stuck on v3 cannot tell those apart — see *`key_candidates` — the
candidate set* below for the full three-state contract.

**Migrating v3 → v4.** Nothing to change for importers that only read `fields`
— it is byte-for-byte the same. Importers that want the candidate set add a
read of the new top-level member; see below for its shape and the three states
it must render distinctly.

Status: **v3** (2026-09-05). Over **v2**, `fields` may carry **more than one
entry per key** — one per occurrence the model emitted, each with its own value
and its own citations, and each tagged `occurrence` (0-based, emission order).
This is a **structural** change, which is why it bumps rather than riding in
place: `fields.length == key_vocabulary.length` was an invariant and no longer
holds, and an importer that maps key → value *last-wins* silently flips from the
first repeated value to the last. The **first entry for each key keeps its v2
value and position**, so a first-match importer is unaffected.

*Why.* Asked for a flat schema on an invoice with three line items, the model
emits the key three times — it is answering correctly and the shape cannot hold
it. v2 read only the first occurrence and then kept only one field, so two thirds
of the answer vanished: no error, no badge, and **coverage did not flag it** (the
un-extracted lines were measured as consulted, since the model does read them —
`note-ss3-matched-pairs.md` era probe, 2026-09-05). A confident, correctly
grounded, silently incomplete answer — the one thing this format exists to
prevent.

*What it is not.* Occurrence order is emission order, which is positional in the
document. The lens does **not** group repeated keys into records: `line_item`
occurrence 1 is not claimed to belong with `quantity` occurrence 1. Record
grouping is a leaf-path design gated on a Leg-B-style measurement of a
repeating-group hint, and is deliberately not this.

Prior status: **v2** (2026-07-17). Over **v1** (2026-07-15) v2 **removes**
`presence_grounded` — a *subtractive* change, hence a real version bump where v1
and v0 rode in place. The fixed KV grammar was **refuted by measurement** and is
off the product path, and the two-pass presence gate it required went with it
([`note-nogrammar-refutation.md`](note-nogrammar-refutation.md)). Nothing else in
the importer surface moved: a concept the document lacks still comes back
`badge:"absent"`, `value:null` — now earned by **omission** (the model simply does
not state it) rather than by a presence verdict. `tier` (A5.3) is unchanged.
Two things are **new** in v2: the endpoint may **refuse** an extraction
(`422 unparseable_extraction`, see the shape contract below), and `gloss` in the
request is now accepted-but-unused. (Both of A5.1's designs — the reverted `null`
value alternative *and* the shipped presence gate — are documented history now;
see [`note-lens-absent-attempt.md`](note-lens-absent-attempt.md) for why each
failed, and the note above for why neither was needed.)
Producer:
[`src/server/server_lens.cpp`](../src/server/server_lens.cpp)
(`lens_report_to_json`). Canonical sample:
[`demo/lens-sample.json`](demo/lens-sample.json) (a real Qwen3.6 extraction).
Viewer: [`demo/attention-lens.html`](demo/attention-lens.html). Sample importer:
[`demo/sample-importer.py`](demo/sample-importer.py).

## The one-sentence honesty contract (load-bearing)

**Attention marks *consideration*, not *commitment*.** The lens reports what the
document says and what the model *consulted* when it wrote each value; it never
claims the model *chose* correctly. Conflicts ship as coexisting keys with
separate citations — resolution belongs to the importer/human. Every consumer of
this format inherits that contract.

**Product invariant: the lens never lies about where the model looked.** "The
model is right" was never the claim; "the record is faithful" is. The lens earns
its keep *most* when the model is wrong — a wrong value carrying a citation is
caught in seconds, where the same wrong value from a black box is caught at the
invoice stage. The value claim therefore degrades gracefully with model quality
instead of cliff-dropping.

## Non-claims — what the lens refuses to say

The lens is a faithful record of *where the model looked*, and nothing more.
Four claims it could superficially seem to support are withheld **by contract**;
an importer that reads them out of the citations/masses is over-reading the
format. These are not gaps to be closed in a later version — they are the
accept/reject line of the seven-probe hunt made contractual: every signal the
format *does* carry records *consideration*; each refusal below would assert
*commitment*, which attention does not carry.

- **No conflict winner (CF1).** When two document spans compete for one concept,
  both light up in that field's top-k `citations`; the format never marks one as
  the chosen source. Attention peaks are bimodal on conflicts and their order
  does not track the correct copy. Competing values ship as separate cited keys —
  the human/importer resolves them. Do not read the top citation as "the winner."
- **No staleness or recency (SS1, re-confirmed by SS2/SS3).** The lens does not
  claim a later value supersedes an earlier one, or that a correction "won."
  That a correction span was *consulted* is displayable; concluding it is the
  *live* value is not. **This survived a direct attempt to lift it.** A thread
  alarm was built and measured: the rule a server can actually compute — *the
  citations reach a message later than the one the value came from* — cried wolf
  on **7 of 9** correctly-handled corrections, and stayed **silent** on the one
  real failure. The reason is structural, not a threshold: a later message
  routinely restates an old value (quoted history, a colleague who is behind), so
  **turn order does not identify supersession**
  ([`note-ss3-matched-pairs.md`](note-ss3-matched-pairs.md) §3). A request may
  now carry `messages` and every citation reports which one it landed in — that
  is **attribution, not a verdict**, and the difference is the whole non-claim.
- **No confidence (CG1).** `body_mass` and citation `mass` are attention masses,
  not calibrated confidence. The most-attended value is not the most-likely-
  correct value; the confidence gap did not separate right answers from wrong.
  Do not threshold a mass as a correctness score.
- **No correctness.** Nothing in the format asserts a value is *right*.
  `grounded` means "read from the document," not "read correctly"; a citation
  locates a source, it does not verify the copy. This is the point, not a
  weakness — see the product invariant above: the receipt is what makes a wrong
  value catchable.

## The shape contract — tolerant parse, loud refusal (SHIPPED, v2)

> **Status: shipped 2026-07-17** (signed off by the architect as drafted). This
> is what **replaces the grammar's guarantee**. Gates: `server-lens-tests`
> (`LensShapeContract`) and `tests/smoke/server_extract_smoke.sh`, which forces a
> real 422 by capping `max_tokens` below what closing the object needs.

The grammar was sold as a moat: *guaranteed well-formed JSON*. Measured, that
guarantee is **false** — a grammar constrains JSON syntax, but nothing stops a
forced value from running off the token budget mid-string, and the output is
unparseable anyway (the constrained arm parsed 14/15; free prompting parsed
15/15). Worse, the constraint *manufactured* the failure it was meant to
prevent. The honest replacement is not a stronger constraint. It is a tolerant
parser and a **loud refusal**:

- **Tolerant on shape.** Free output may carry a markdown fence or a preamble.
  The producer strips a ``` fence and takes the **outermost JSON object** by
  string-aware brace depth (a `{` inside a value — an HTML mail quoting a JSON
  payload — must not close it). Implementation: `lens_find_json_object` in
  [`../src/server/server_lens.cpp`](../src/server/server_lens.cpp). Tolerance is
  bounded and mechanical — it recovers *shape*, and never guesses *content*.
- **Loud on failure.** A document whose output does not parse **returns an
  error**, naming the endpoint, the expectation, and the actual (the fail-loud
  error contract, CLAUDE.md). Never a best-effort parse, never a partial
  extraction, never a repaired object.
- **Values are scalars, or arrays of scalars. Deeper nesting is refused** (422,
  naming the key). An array of scalars is expanded into one occurrence per
  element — the same treatment a repeated key gets, because an element is a
  locatable scalar with its own byte span and therefore its own real citations.
  This matters because the *same document and the same hint* get two encodings of
  one answer: on a three-line invoice Qwen 3.8-9B repeats the key while
  Qwen 3.6-35B writes `"quantity": [7, 19, 43]`. Serving one and refusing the
  other would make the contract depend on which model is loaded. An array holding
  **objects** stays refused: there is no scalar span to cite. The lens reports a flat object of `"key": "value"` pairs; nothing stops a
  free decode emitting nested JSON, and until 2026-09-05 the parse did not refuse
  it — it silently produced wrong fields two ways, both measured
  (`test_server_lens.cpp` `LensNestedOutput`): an array value was read to the
  first comma and shipped as the fragment `[{"desc": "Widget"` with a real badge,
  and because a key resolves by first occurrence, a nested `"quantity":"5"` could
  answer a top-level `"quantity":"45"` — a wrong value carrying citations that
  point at a span the model never read. That is the lens misreporting where the
  model looked, so it is refused rather than summarized. **This is not a ruling
  that arrays are unsupportable:** the trust math is per-value-span and
  generalizes to leaf scalars at any depth (`line_items[2].unit_price` would
  carry its own citations unchanged). What is missing is a hint form for
  *repeating groups* — the complete hint is what holds key names stable (Leg B),
  and you cannot enumerate `line_items[0..N]` in advance. That is a measurement,
  not a parser change. Until it is measured, refuse.
- **An extraction can be refused, and the importer must handle it.** This is the
  contract's whole point: a refusal is a *feature*. An unparseable document that
  fails loudly is strictly better than a constraint that corrupts the output to
  avoid failing — the first is one visibly-skipped document, the second is a
  plausible wrong value that reaches your ERP.

**A refusal is not a bad request.** `/v1/extract` used to answer `400` both when
the *caller* sent something malformed and when the *extraction* failed. Those are
different events with different responses — fix your call vs. retry or route
this document to a human — and an importer must be able to tell them apart
without string-matching a message. v2 splits them:

| status | `error.code` | meaning | importer's move |
|---|---|---|---|
| `400` | `bad_request` | The request was malformed (missing `document`, bad `key_vocabulary`). The caller's fault. | Fix the call. Never retry unchanged. |
| `422` | `unparseable_extraction` | The request was fine; the **model's output for this document** could not be parsed into an object. Carries `raw` — exactly what the model emitted — so the failure is inspectable. | Route the document to a human; retrying may or may not help. Never synthesize fields. |

An importer MUST NOT treat `422` as an empty extraction. Zero fields and a
refusal are different facts: the first says "the document has none of these
concepts", the second says "we do not know what this document says". Collapsing
them re-introduces silent data loss through the back door — the same failure the
grammar produced, one layer up.

## Versioning (fail-loud, snapshot-header precedent)

`format_version` is `"qemmi-lens/v4"`. It is a hard gate, not a hint: an
importer that does not recognize the exact version string MUST refuse the
payload (fail loud), never best-effort-parse it.

**v3** (2026-09-05) bumped for the `fields` occurrence change described above.

**v4** (2026-09-06) adds `key_candidates` (additive — see *`key_candidates`*
below) but still bumps, because on this format an **absent** member is itself
a fact an importer must be able to read: no `key_candidates` on a v4 payload
means the candidate finder failed for this document; a v3 payload has no way
to say that at all. Without the bump, a v3-shaped importer reading a
candidate-enabled response would silently miss the distinction between "this
server never produces candidates" and "it tried and failed" — the exact
confusion this feature exists to prevent, one level up the stack.

**Why v2 is a real bump.** v0→v1 was purely **additive** (`tier`, the
`badge:"absent"`/null-`value` pair, `presence_grounded`), so it rode in place: a
v0 importer that hard-refused unknown fields stayed safe. **v2 is SUBTRACTIVE** —
`presence_grounded` is gone from every field, because the two-pass presence gate
that produced it is gone. A subtractive change cannot ride in place: an importer
reading that column would silently get nothing, which is exactly the failure the
version gate exists to prevent. So the string moves.

**Migrating v1 → v2.** For most importers: nothing to do. `badge:"absent"` and
`value:null` mean the same thing and arrive in the same shape; only their
*derivation* changed (omission by a free decoder, rather than a verdict from a
probe pass). Two real changes: drop any read of `presence_grounded` (it was
already documented as a non-signal — it never separated present from absent), and
**handle `422`** (see the shape contract above) — the endpoint can now refuse.

During the migration window an importer MAY accept `qemmi-lens/v0`..`v3`
(v0/v1/v2 deprecated); new importers should require v4. **`../qemmi-lens`'s
`ACCEPTED_FORMAT_VERSIONS` must add `"qemmi-lens/v4"` or every extract call
fails its own fail-loud version gate** — the gate is on the version, not on
`key_candidates`, so an un-migrated client fails on ordinary text extracts too.
Done in that repo's working tree as of 2026-09-06 (uncommitted there).

## Top-level object

| field | type | meaning |
|---|---|---|
| `format_version` | string | `"qemmi-lens/v4"`. Gate before reading anything else. |
| `model` | string | The pinned model that produced this (Qwen3.6). |
| `validated_envelope` | bool | `true` iff the prompt was ≤ 4K tokens — the measured envelope (plan §1.5). `false` is a **disclosure, not a rejection**: the extraction ran, but beyond where the signals were validated. |
| `citation_source` | string | Human label for the citation head (L3H13, N3). |
| `coverage_source` | string | Human label for the coverage source (layer-11 max-heads, COV1). |
| `used_threshold` | number | Coverage span-peak ≥ this ⇒ a span was "consulted" (0.705). |
| `ungrounded_threshold` | number | `body_mass` ≥ this ⇒ `grounded` (0.538, N3b). |
| `prompt_len` | int | Total prompt tokens (ChatML-wrapped). |
| `doc_lo`, `doc_hi` | int | The document's token range within the prompt: tokens `[doc_lo, doc_hi)` are the document; `[0, doc_lo)` is the chat header and `[doc_hi, prompt_len)` the instruction + assistant tag. All signals are restricted to the document range. |
| `document` | string | The user's raw document, verbatim. **All byte offsets below are relative to this string.** |
| `raw` | string | Exactly what the model emitted, verbatim. Today this is grammar-shaped; note that "grammar-guaranteed well-formed" is **not** a claim the format makes — it was measured false (a constrained value can run off the token budget mid-string). Treat `raw` as evidence, not as a parse guarantee; the shape contract above is the guarantee. |
| `fields` | array | The structured extraction — the importer surface. See below. |
| `key_candidates` | object \| **absent** | v4. Present iff the candidate-set producer ran. See *`key_candidates` — the candidate set* below for the full three-state contract. |
| `candidates_error` | string \| **absent** | v4. Present iff the candidate-set producer **failed**; `key_candidates` is then also absent. |
| `prompt`, `gen`, `hover`, `heat`, `skipped` | arrays | The viewer surface (token-level provenance). See below. |

## `fields[]` — the importer surface

One entry per concept in your **complete key vocabulary** hint (one key per
concept — plan §1.2), in vocabulary order. A concept the document lacks is not
dropped — it comes back as an **absent field** (`badge:"absent"`, `value:null`);
see *How `absent` is earned* below.

| field | type | meaning |
|---|---|---|
| `key` | string | The snake_case key (from your vocabulary). |
| `value` | string \| `null` | The value, **verbatim-lifted** from the document (no normalization). **`null` iff `badge:"absent"`** — the concept is not in the document, so there is no value. |
| `badge` | `"grounded"` \| `"ungrounded"` \| `"absent"` | `grounded` iff `body_mass ≥ ungrounded_threshold`. `ungrounded` flags a value the model wrote **without reading the document** — fabrication-suspect (N3b); does **not** catch attended-but-overruled or wrong-copy (plan §4). **`absent`** = the grounded presence gate judged the concept **not present** (A5.1) — `value` is `null`, `citations` empty, `tier` null. |
| `tier` | `"distinctive"` \| `"short_numeric"` \| `null` | **A5.3** — the machine-readable trust class of the value, so importers apply the two-tier claim (below) without reading this manual. `distinctive` = citations are strong (dates, refs, formatted amounts, addresses/names). `short_numeric` = a short bare integer (a quantity, a bare number) — the **weak** citation class; lean on `badge`+coverage. `null` on an `absent` field (no value to class). Deterministic function of the value shape. |
| ~~`presence_grounded`~~ | — | **REMOVED in v2.** The audit flag of the deleted two-pass presence gate. It was already documented as a non-signal (measured not to separate present from absent), and its producer is gone. A v1 payload may still carry it; ignore it. The presence signal is `badge`. |
| `body_mass` | number | Mean fraction of the citation head's attention on the document while emitting the value's tokens (N3b). On an `absent` field, the Pass-A yes/no answer's body_mass (typically ≪ threshold). |
| `found_in_document` | bool | The value string appears verbatim in `document`. Always `false` on an `absent` field. |
| `value_span` | `{lo, hi}` \| `null` | Byte span `[lo, hi)` of the value's first occurrence in `document` (null if not found, and always null on `absent`). |
| `citations` | array of `{pos, mass, byte_lo, byte_hi}` | Top-k **source spans** the citation head (L3H13) attended to when writing the value, aggregated over its tokens, restricted to the document and sorted by `mass` desc. `pos` = prompt token index; `byte_lo/byte_hi` = its **document-relative** byte span. Empty on `absent`. |

**Two-tier trust (plan §1.3 — read this before trusting a citation).** Citations
are strong for **distinctive** values — dates, delivery/order refs, formatted
amounts, names (88–92% top-3-in-span). They are weaker for **short bare
numerics** (a bare quantity like `45`: ~65%) and multi-token name suffixes. For
those classes, lean on `badge` + the coverage report rather than the exact
citation span. The weak class is written on the tin; do not paper over it. As of
v1 this split is **machine-readable**: switch on `tier` (`distinctive` vs
`short_numeric`) instead of re-deriving value classes — the producer computes it
with the same deterministic rule, and the standing fidelity gate (below) is
scoped to exactly the `distinctive` class.

**Fidelity gate (the standing invariant).** "The lens never lies about where the
model looked" is enforced as a permanent gate, same status as the byte-identity
gates: **zero confident false receipts** — no `grounded`, `distinctive`,
verbatim field whose top citation lands on no occurrence of its value. It is
scoped to the `distinctive` tier (the class the format vouches citations for);
`short_numeric` fields make no such claim and are exempt. Any change to the tap,
grammar, or lens computation must keep the count at zero
(`tests/unit/test_server_lens.cpp`, suite `LensFidelityGate`).

## `key_candidates` — the candidate set (v4, docs/plan-candidate-set.md)

Alongside the value `fields` already returns, an extract that opted into the
candidate-set producer (a per-request internal toggle — no server or CLI flag
exposes it yet) also returns **every span in the document that answers a key**,
not just the one that was returned. `fields` is completely unchanged by this —
same members, same meaning, same taps, calibration, and coverage.

**The three top-level states.** These must never be conflated:

| `key_candidates` | `candidates_error` | meaning |
|---|---|---|
| absent | present | the candidate finder **failed** on this document — says nothing about what the document offers |
| absent | absent | candidates were **not requested** for this extract |
| present (an object, possibly with empty arrays) | absent | the finder **ran** |

A producer failure must never render as an empty, unflagged `key_candidates` —
that is precisely the confusion this feature exists to prevent, one level down
from `fields`' own `422`/absent-field distinctions.

**Shape**, when present: a top-level object keyed by **key**, not by field
entry (one entry per requested key, covering the **complete request
vocabulary** — every key gets an array, `[]` included; `[]` and "this key is
missing from the object" are different facts and this format never conflates
them). Each key's value is an array of candidates:

```json
"key_candidates": {
  "monthly_rent": [
    { "value": "1,450.00 GBP", "byte_lo": 512, "byte_hi": 525,
      "anchor": "2. RENT", "returned_as": null },
    { "value": "1,375.00 GBP", "byte_lo": 980, "byte_hi": 993,
      "anchor": "AMENDMENT No. 1", "returned_as": 0 }
  ],
  "pets_policy": []
}
```

| candidate field | meaning |
|---|---|
| `value` | Byte-exact slice of `document`, always — unlike a field's `value`, which may not be verbatim. This asymmetry is what makes a mis-copy visible without a diff. |
| `byte_lo`, `byte_hi` | Into `document`, same convention as `citations` / `value_span`. |
| `anchor` | The structural location a human can navigate to: the nearest preceding label-like line (a short line ending in `:`, or a numbered/underlined heading), else the enclosing line trimmed and truncated to ~60 chars, else **`null`**. Nullable is deliberate — a byte offset is not a place, and inventing one would be dishonest. |
| `returned_as` | The `occurrence` of the `fields` entry (same key) this candidate was returned as, else `null`. Linking is **bidirectional containment** on whitespace-normalized text (pass 2 legitimately returns wider or narrower spans than the field value — `"45 units"` for `45`). Tiebreak when several candidates qualify: **tightest containment first, then earliest `byte_lo`**. At most one candidate per occurrence. |

**Array order is document order (`byte_lo` ascending), and this is
load-bearing** — not mass, not any ranking. Position is a fact about the
document; any other ordering would be a verdict, and CF1 (no conflict winner)
forbids one.

**Non-claims (unchanged from `fields`).** The candidate set is attribution, not
adjudication. No candidate is marked correct, chosen, primary, or live. CF1, SS1
/SS3 (no staleness), CG1 (no confidence), and "no correctness" all apply here
verbatim.

## How `absent` is earned (by omission)

A hinted concept the document lacks comes back `badge:"absent"`, `value:null`,
`tier:null`, no citations. The rule is **mechanical and boring**, which is the
point:

> A concept in the request's `key_vocabulary` that the model does not state in
> its output — the key is missing, its value is empty, or its value is a JSON
> `null` — is reported `absent`. Fields are emitted in the order you hinted them,
> one per concept.

That is the whole mechanism. There is no probe, no verdict, no second pass.

**Why this used to be hard.** The one fixed KV grammar required a **non-empty**
value for every hinted key, so naming a concept the document lacks *forced* a
fabrication — and on the pinned Qwen3.6-Q2 that fabrication **collapsed the whole
extraction** to `","`. Two designs were built to contain it: a `null` value
alternative in the grammar (reverted — it made every present field decline), then
a two-pass *grounded presence gate* (shipped 2026-07-16, deleted 2026-07-17). Both
were sound engineering aimed at the wrong target. With no grammar the model simply
**declines**, natively and correctly — it emits `"payment_terms": null` unprompted
— and measures **30/30** absent concepts handled against the grammar's **10/30**.
The wound was self-inflicted; closing it deleted the treatment, and the N+1
presence prefills with it. Full record:
[`note-nogrammar-refutation.md`](note-nogrammar-refutation.md);
[`note-lens-absent-attempt.md`](note-lens-absent-attempt.md) keeps the two dead
designs and why each failed.

**What the format does NOT do here.** It does not second-guess a value the model
*did* emit. The presence gate had a "safety net" that re-marked an `ungrounded`,
not-found-verbatim value as `absent`; that died with the gate, deliberately. Alone
it would be the lens judging a value **wrong** — a claim this format refuses (see
*No correctness*). A fabricated value is reported *with its badges*: `ungrounded`,
`found_in_document:false`, a near-zero `body_mass`. Those are the receipt. **You**
decide.

**A quoted `"null"` is a value.** Only the unquoted JSON literal `null` is a
decline. A document that genuinely says `null` extracts as the string.

**Residual failure mode, and which way it fails.** A concept whose document
surface form the model does not connect to the hinted key (a bare `From:` email
address hinted as `customer`) may come back `absent` rather than extracted. That
fails **safe**: the importer sees a missing field, not a fabricated one. The
opposite error — a genuinely-absent concept coming back with a value — is the
one the gate measures at **absent-specificity 1.00**
(`tests/smoke/server_extract_smoke.sh`, including the product-first ordering that
killed the reverted `null` design).

## Viewer surface (token-level provenance)

Consumed by the Attention Lens viewer; an importer can ignore these.

- **`prompt[]`** — `{pos, text, region}` per prompt token. `region` is `"body"`
  (document, `doc_lo ≤ pos < doc_hi`) or `"instr"` (chat header + instruction).
- **`gen[]`** — `{idx, text}` per generated token (the `raw` JSON, tokenized).
- **`hover[]`** — indexed by `gen` idx; each is the citation head's top source
  positions `{pos, mass}` for that generated token (`[]` for idx 0). This is the
  per-token version of a field's aggregated `citations`.
- **`heat[]`** — indexed by prompt `pos`; the coverage source's peak max-heads
  mass on that token across the whole generation (0..1). Drives the heat map.
- **`skipped[]`** — `{lo, hi, peak, text, byte_lo, byte_hi}` document lines whose
  coverage span-peak stayed **below** `used_threshold`: "possibly not
  incorporated." `lo/hi` are token indices; `byte_lo/byte_hi` document-relative
  bytes. This is the **omission audit** (COV1) — the backstop for the weak
  citation classes.

## Honest limits (shipped verbatim, plan §4)

- **Single document per request.** No thread or staleness semantics (SS1).
- **The model never resolves conflicts.** Duplicate/competing values ship as
  separate cited keys; the importer/human decides.
- **Short bare numerics cite weakly**; the coverage audit backstops them.
- **Absent concepts are handled, not banned — but presence recall is imperfect
  and fails safe.** You may now hint a superset for a document class: a concept
  the document lacks comes back `badge:"absent"` (the grounded presence gate,
  above), not a fabrication or a collapse. The cost is presence *recall*: a
  concept whose document surface form the presence word doesn't match — a bare
  `From:` email hinted as `customer`, a value stated only obliquely — may be
  **dropped** (marked `absent`) even though it is arguably present. This fails in
  the safe direction (a missing field, never a fabricated one), and a per-concept
  **gloss** is the lever that recovers most of it (recall 0.75 bare → 0.92
  glossed). A trimmed hint is still the strongest input. (Two earlier absent
  designs — the `null`-in-grammar sentinel and the two-pass presence gate — were
  both **removed**; the grammar that made absence hard is gone. See *How `absent`
  is earned*, [`note-nogrammar-refutation.md`](note-nogrammar-refutation.md) and
  [`note-lens-absent-attempt.md`](note-lens-absent-attempt.md).)
- **Key stability under sampling is unmeasured-and-worse.** The lens decodes
  **greedy**; at `temperature 0.7` free prompting drops to 12/15 key stability
  (a space inside a key: `"delivery_ date"`) where the removed grammar held 14/15
  by construction. Irrelevant today — there is no sampler on this path — but it
  prices any future move to make `/v1/extract` samplable.
- **`ungrounded` catches fabrication-from-nothing**, not attended-but-overruled
  or wrong-copy (a human catches those via the citation).
- **Text only. Per-model constants** (2026-09-05): the coordinates are looked up
  by `{architecture, block_count}` and `--attention-lens` is **refused at startup
  on any model with no calibration entry** — Qwen 3.6-35B-A3B and Qwen 3.8-9B
  today, both Qwen. This is a refusal, not a sanity check: nothing verifies at
  runtime that a calibrated head has not drifted *within* a listed model (the
  offline probe is what catches that). The report's `model` field names the
  entry its numbers came from.
- **Envelope 4K** — longer documents are accepted and disclosed
  (`validated_envelope: false`), never rejected. Unchanged deliberately after the
  2026-09-04 thread work: citation was re-measured at 4774–6200 tokens and did
  not degrade (89% top-1 / 98% top-3, `note-ss2-thread-alarm.md`), but the
  coverage bar and the grounded/ungrounded threshold were **not** re-measured at
  that length. Raising this number means re-measuring all three arms, not editing
  a constant.

## Request

```
POST /v1/extract        (server started with --attention-lens)
{
  "document": "<text>",
  "key_vocabulary": [
    { "key": "customer", "gloss": "the buyer — a company, person, or their email/domain" },
    { "key": "product",  "gloss": "the item ordered" },
    ...
  ],
  "max_tokens": 220
}
```

`key_vocabulary` is an array of **`{key, gloss}` objects**. A bare **string**
element is tolerated as `{key, gloss:""}` for compatibility.

### `messages` — the thread unit (additive, 2026-09-05)

Send **`messages` instead of `document`** when the input is an ordered thread:

```
{
  "messages": [
    { "text": "From: orders@… \nPlease book 45 units at 82.00 GBP each." },
    { "text": "Correction: make that 60 units." },
    …
  ],
  "key_vocabulary": [ … ]
}
```

A bare **string** element is tolerated, as in `key_vocabulary`. The server joins
them with one blank line — the same text a caller who concatenated the thread
themselves would have sent, so the prompt regime the free path was validated on
is unchanged — and echoes the joined text as `document`, so every byte span in
the report still resolves against it.

**Exactly one of `document` / `messages`.** Both, or neither, is **400**
`bad_request`, as is an empty array or an empty `text`. There is no precedence
rule: boundaries cannot be recovered from concatenated text, so guessing which
input wins would be a silent fallback at the boundary of the format.

**What it buys, precisely.** Every citation gains `message` (its index), each
field gains `citation_messages` (the distinct messages its citations landed in,
strongest first), and the report gains `n_messages`. A value therefore reads
*"read from message 23 of 24"*. That is **attribution**. It is *not* a staleness
signal, and the format does not gain one — see the SS1/SS2/SS3 non-claim above
for the measurement that closed that door. A field whose `citation_messages` has
more than one entry is the ordinary conflict case: both are reported, strongest
first, and **no winner is named** (the CF1 non-claim). `message` is `null` and
`citation_messages` is `[]` on a plain `document` request — never `0`, which
would read as "the first message".

**Version.** Additive on both sides, so it rides in place as `qemmi-lens/v2` — a
v2 importer that ignores unknown keys is unaffected, and the same convention v1's
additions rode under. Only a *subtractive* change bumps the version (that is what
v2 itself was).

`key_vocabulary` must name **every** target concept: the complete hint is what
holds key names stable — it, and **not** the (now removed) grammar, is what keeps
the naming zoo shut (Leg B: complete hint ⇒ keys/concept 1.00; omit one and it
fragments; plan §1.2). Dropping the grammar did not reopen this. Dropping the
hint would.

> **`gloss` is accepted and currently UNUSED (v2).** Its only consumer was the
> deleted presence gate's Pass-A question, where it lifted recall 0.75 → 0.92. It
> is kept in the request shape because removing it would be a second breaking
> change and it is a plausible future lever — but it is **not** fed into the
> extraction instruction, which would silently change the exact prompt regime the
> free path was validated on. Send it or don't; today it changes nothing. If a
> glossed instruction is ever wanted, measure it first.

Empty `key_vocabulary` ⇒ **400** `bad_request`. A document that overflows the
context ⇒ **400** (fail-loud, names the parameter). Output that cannot be parsed
⇒ **422** `unparseable_extraction` (see the shape contract — this one is *not*
your fault). The endpoint is single-slot and exclusive; the server 404s it when
`--attention-lens` is off.
