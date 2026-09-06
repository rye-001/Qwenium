#pragma once
// server_lens.h — Qemmi-Lens extraction: document → audited key-value JSON on
// the attention trust layer (docs/plan-qemmi-lens.md, P2/A2).
//
// Two concerns, split so the numerics are testable without a model:
//
//   1. compute_lens_report(LensRun) — PURE. Given one tapped decode run
//      (prompt/gen tokens, byte maps, and per-step kq_soft rows for the two
//      frozen lens layers), it computes the lens report: per-field citations
//      (L3H13, N3), grounded/ungrounded badges (body_mass, N3b), and the
//      document coverage report (layer-11 max-heads span-peak, COV1). No engine,
//      no ggml — the unit test synthesizes rows. This is the relocated probe
//      math (attn_provenance.cpp run_lens_gen / eval_field / parse_fields).
//
//   2. run_lens_extract(...) — the DRIVER. Assembles the ChatML thinking-off
//      prompt from (document, concepts), runs a FREE decode with the P1
//      attention tap armed on the two lens layers, and hands the captured rows
//      to compute_lens_report. Single-slot (inherits the qwen36 slot-0 KV-gather
//      limit, architecture.md §12).
//
// ── No grammar (Stage 2, 2026-07-17) ─────────────────────────────────────────
// The lens path once constrained this decode with ONE fixed KV grammar. It was
// REFUTED by measurement (docs/note-nogrammar-refutation.md): on the Leg C corpus
// the grammar lost on every axis INCLUDING the guaranteed parse it existed for
// (14/15 vs free's 15/15), and its forced non-empty value was the SOLE cause of
// the absent-concept collapse — and therefore of the two-pass presence gate built
// to work around it. Stage 1 re-validated the trust layer over free output
// (top-3 in-span 61/61; like-for-like in-span mass retention mean −1.0%).
//
// The grammar's parse guarantee is replaced by an explicit contract, not by a
// weaker constraint: TOLERANT on shape (strip a fence, take the outermost JSON
// object) and LOUD on failure (LensUnparseableError ⇒ 422, never a partial
// extraction). See docs/lens-format.md §"The shape contract".
//
// `lens_grammar_gbnf()` survives for ONE reason: the QDOCS_S1 probe runs it as a
// CONTROL ARM against the free path through this same driver, so the comparison
// stays reproducible on shipped code. It is not reachable from /v1/extract.
// NOTE: this says nothing about the engine's GBNF machinery or the server's
// per-request `grammar` field on /v1/completions and /v1/chat/completions —
// that is a separate, shipped, unaffected feature.
//
// The constants are PER MODEL, and what guards them is a CALIBRATION REFUSAL at
// server startup, not a numeric self-check: the loaded model is looked up in
// kLensCalibrations by {architecture, block_count}, and --attention-lens is
// refused fail-loud before the server binds if it has no entry
// (lens_calibration_for / lens_calibration_refusal below, called from main() in
// http_server.cpp). There is NO known-answer sanity check on the citation head —
// an earlier version of this comment claimed one and was wrong. Drift of the
// head WITHIN a calibrated model is therefore unguarded at runtime; it is
// caught, if at all, by the offline probe (tests/perf/attn_provenance.cpp).
//
// Two calibrated models today, both Qwen: Qwen 3.6-35B-A3B (L3H13) and
// Qwen 3.8-9B (L27H13). No lens claims for other families, and that is a
// MEASURED position, not neglect — Gemma was searched properly and 0 of 768
// candidate heads clear even a 70% bar against the 90% requirement
// (docs/note-lens-gemma4-probe.md, docs/note-lens-gemma-norm-weighted.md). The
// lens is a Qwen-family capability by measurement; the refusal is the mechanism
// that keeps that honest.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class ForwardPassBase;
class Tokenizer;
struct ModelMetadata;
typedef struct ggml_backend_sched* ggml_backend_sched_t;

namespace qinf {

class GrammarVocab;

// ── Lens constants — one model's measured coordinates ────────────────────────
// The member defaults below ARE the Qwen 3.6 calibration (plan §1.6;
// docs/note-qemmi-docs-p0.md), and they are the only place those numbers are
// written down: the qwen35moe rows of kLensCalibrations{} take them by
// default-construction rather than restating them.
struct LensConstants {
    int    citation_head        = 13;     // L3H13 retrieval head (N3)
    int    citation_layer       = 3;      // physical attention layer for citations
    int    coverage_layer       = 11;     // physical attention layer for coverage (COV1)
    double coverage_used_peak   = 0.705;  // span-peak ≥ this ⇒ "consulted" (COV1)
    double ungrounded_body_mass = 0.538;  // mean body_mass ≥ this ⇒ grounded (N3b)
    int    citation_topk        = 8;      // source positions reported per token/field
    // What the report's `model` field says. It travels WITH the coordinates
    // because it describes them: a report computed from this entry was produced
    // by this model, and a hardcoded name would mislabel every other entry's
    // receipts (it did — `run.model` was a literal "Qwen3.6" string until the
    // table landed, which a Qwen 3.8 extraction would have carried).
    const char* model_label     = "Qwen3.6 (attention lens)";
};

// ── The calibration table — which models the lens may run on ─────────────────
// These are coordinates and thresholds, not a mechanism: "layer 3 head 13" is a
// retrieval head *of one model*, and nothing about it transfers. Run the lens
// on an uncalibrated model and /v1/extract returns a confidently-shaped report
// computed from someone else's coordinates — citations pointing at whatever the
// named layer/head happens to be there, badges off an unvalidated threshold.
// That is a false receipt, so an unlisted model is REFUSED rather than
// best-efforted. The table IS the calibration record: an entry exists because a
// probe measured it, and `provenance` says which one.
//
// ── Why the key is {architecture, block_count} and not architecture alone ────
// Approved 2026-09-05. An architecture string is a FAMILY, not a model:
// `qwen35` hosts Qwen3.5-0.8B (24), Qwen3.5-9B (32), Qwen3.8-9B (33),
// Qwen3.6-27B (64) and Qwen3.8-27B (65) — five different models, five
// different layer counts, one string (src/models/qwen35.h). An arch-keyed
// allowlist would admit all five under coordinates measured on one of them,
// which is the exact false receipt the refusal exists to prevent.
//
// And the key is the RAW GGUF block_count, deliberately, not the decode-stack
// depth (block_count − nextn_predict_layers). Decode depth is the more natural
// quantity, and it is the WRONG key here: it collides Qwen3.5-9B with
// Qwen3.8-9B at 32, and Qwen3.6-27B with Qwen3.8-27B at 64 — in both pairs a
// calibrated model and an uncalibrated one. Raw block_count separates every
// model above, because the trailing MTP head shifts the Qwen 3.8 builds by one.
// That separation is an accident of these files, not a law; it holds for every
// model this repo targets, and a future collision must be resolved by adding a
// field to the key, never by widening an entry to cover a model nobody measured.
struct LensCalibration {
    const char*   architecture;   // GGUF general.architecture
    uint32_t      block_count;    // GGUF <arch>.block_count, raw (see the key note)
    const char*   model;          // the exact model the numbers were measured on
    const char*   provenance;     // the probe note that measured them
    LensConstants constants;
};

inline const std::vector<LensCalibration>& lens_calibrations() {
    static const std::vector<LensCalibration> kLensCalibrations = {
        // Qwen 3.6-35B-A3B, the model the lens was built on. Two GGUF builds of
        // one base model: the MTP build carries a trailing NextN draft block
        // (41 = 40 + 1), the plain build does not (40). The decode stack is the
        // same 40 layers in both and the lens never touches the draft head, so
        // both are the SAME calibration — listed twice rather than keyed on a
        // depth that would collide with uncalibrated models elsewhere.
        {"qwen35moe", 41, "Qwen3.6-35B-A3B (MTP build)",
         "docs/note-qemmi-docs-p0.md (N3, N3b, COV1)", LensConstants{}},
        {"qwen35moe", 40, "Qwen3.6-35B-A3B (plain build, same 40-layer stack)",
         "docs/note-qemmi-docs-p0.md (N3, N3b, COV1)", LensConstants{}},
        // Qwen 3.8-9B. Its own head is L27H13, not L3H13 — 98% top-3 vs 84% on
        // the same messy corpus, and 0% vs 7% ungrounded false alarm
        // (note-lens-qwen38-probe.md §5.3, confirmed on an independent corpus,
        // not overfit to the selection prompt). Thread-scale citation was then
        // validated on this entry at 4774–6200 tokens with no degradation
        // (note-ss2-thread-alarm.md Gate 0: 89% top-1 / 98% top-3).
        //
        // coverage_used_peak stays 0.705 deliberately. It is the weak arm on
        // BOTH models (87% / 84% used-clear against a ≥90% bar) and was never
        // searched — it was frozen from COV1 on one model. A recalibration on
        // 12+11 spans scored ~4 points better (note-lens-norm-weighted-metric.md);
        // "recalibration is worth ~4 points" is the finding, the value it
        // produced is fitted on 23 spans and is not shippable. Decided
        // 2026-09-05: carry 0.705 here until a real coverage-layer search runs.
        {"qwen35", 33, "Qwen3.8-9B",
         "docs/note-lens-qwen38-probe.md §5.3; docs/note-ss2-thread-alarm.md",
         LensConstants{/*citation_head*/ 13, /*citation_layer*/ 27, /*coverage_layer*/ 11,
                       /*coverage_used_peak*/ 0.705, /*ungrounded_body_mass*/ 0.538,
                       /*citation_topk*/ 8, /*model_label*/ "Qwen3.8 (attention lens)"}},
    };
    return kLensCalibrations;
}

// The calibration for one loaded model, or nullptr if it has none.
inline const LensCalibration* lens_calibration_for(const std::string& arch, uint32_t block_count) {
    for (const LensCalibration& c : lens_calibrations())
        if (arch == c.architecture && block_count == c.block_count) return &c;
    return nullptr;
}

// The refusal text. Fail-loud contract order: parameter, expected, actual.
inline std::string lens_calibration_refusal(const std::string& arch, uint32_t block_count) {
    std::string expected;
    for (const LensCalibration& c : lens_calibrations()) {
        if (!expected.empty()) expected += ", ";
        expected += std::string(c.architecture) + "/" + std::to_string(c.block_count) +
                    " (" + c.model + ")";
    }
    return "--attention-lens: expected a model with a calibrated lens entry, one of {" +
           expected + "}, actual architecture '" + arch + "' with block_count " +
           std::to_string(block_count) +
           " — the lens constants are coordinates measured on one model and do not "
           "transfer to another model of the same architecture";
}

// ── Pure-computation input: one tapped decode run ────────────────────────────
// One decode step's tapped rows, flat [n_head * n_kv] row-major [head][kv].
// citation_row is the citation_layer's kq_soft row; coverage_row the
// coverage_layer's. steps[t] is the row computed with gen token t as the query
// (so it is the provenance of gen token t+1 — N3's off-by-one).
struct LensStep {
    int                n_kv = 0;
    std::vector<float> citation_row;
    std::vector<float> coverage_row;
};

struct LensRun {
    std::vector<std::string> prompt_text;   // decoded text per prompt token (len P)
    std::vector<size_t>      prompt_cum;    // cum bytes over prompt (len P+1)
    std::vector<std::string> gen_tok_text;  // decoded text per gen token (len G)
    std::vector<size_t>      gen_cum;       // cum bytes over gen (len G+1)
    std::string document;                   // the user's raw document (values re-found here)
    std::string gen_text;                   // concat of gen_tok_text = the emitted JSON
    // The document's token range within the ChatML-wrapped prompt: tokens
    // [doc_lo, doc_hi) are the document; [0,doc_lo) is the chat header and
    // [doc_hi,P) the instruction + assistant tag. All lens signals (citations,
    // coverage, body_mass) are restricted to the document range — that is where
    // N3b's 0.538 threshold was calibrated. doc_byte_offset is the document's
    // start byte within the prompt (to translate citations to document-relative).
    int    doc_lo = 0, doc_hi = 0;
    size_t doc_byte_offset = 0;
    int  n_head = 0;
    std::vector<LensStep> steps;            // one per gen token; steps.size() == G
    // Byte offset, within `document`, where each request message starts (message
    // i spans [message_offsets[i], message_offsets[i+1])). Empty ⇒ the caller
    // sent a plain `document` and citations carry message -1.
    std::vector<size_t> message_offsets;
    std::string model;                      // passthrough for the report header
    // false ⇒ prompt exceeded the 4 K CALIBRATION floor (a disclosure on the
    // report, not an error). Unrelated to the 10 K workload envelope.
    bool validated_envelope = true;
};

// ── Lens report (the interchange format; P3 versions/documents it) ───────────
// An attended prompt position. `message` is the index of the request message
// this byte range falls in, or -1 when the request sent a plain `document`
// (no boundaries to resolve against). It is a LOCATION, not a verdict — see
// the CF1 non-claim in lens-format.md: the top citation is not "the winner".
struct LensCitation { int pos; double mass; size_t byte_lo, byte_hi; int message = -1; };
struct LensPromptToken { int pos; std::string text; std::string region; };  // region: "body"|"instr"
struct LensCoverageSpan { int lo, hi; double peak; std::string text; size_t byte_lo, byte_hi; };

struct LensField {
    std::string key, value;
    // A5.3 machine-readable trust tier of the VALUE: "distinctive" (citations
    // claimed) | "short_numeric" (weak, coverage-backstopped). "" for absent.
    std::string tier;
    int    gen_lo = -1, gen_hi = -1;      // gen-token span of the value
    bool   found_in_document = false;     // value appears verbatim in the body
    size_t value_byte_lo = 0, value_byte_hi = 0;  // its first byte span (valid iff found)
    bool   grounded = true;               // body_mass ≥ threshold (badge)
    double body_mass = 0.0;               // mean citation-head mass on body positions
    std::vector<LensCitation> citations;  // top-k document source positions (body only)
    // Distinct request-message indices this field's citations landed in, in
    // descending citation mass. Empty when the request sent a plain `document`.
    // Usually one element ("read from message 23"); more than one means the
    // model looked at this key in several messages, which is exactly the
    // coexisting-conflict presentation the format already requires — the lens
    // does NOT say which of them is current (turn order does not identify
    // supersession, docs/note-ss3-matched-pairs.md §3).
    std::vector<int> citation_messages;
    // ── Absent by omission (Stage 2; was the two-pass presence gate, A5.1) ───
    // false ⇒ ABSENT: the model simply did not emit this hinted concept (or
    // emitted it empty). Serializes value:null, badge:"absent". This is now a
    // MECHANICAL read of the parsed output, not a verdict: with no grammar the
    // model declines natively and correctly (30/30 on the Leg C corpus, against
    // the grammar's 10/30 — docs/note-nogrammar-refutation.md), so there is
    // nothing to gate. The N+1 presence prefills are gone with it.
    bool present = true;
    // Which occurrence of this key in the model's output this field is (0-based,
    // emission order). A document whose real structure REPEATS — an invoice with
    // three line items against a flat schema — makes the model emit the key
    // several times. Every occurrence is now reported with its OWN value and its
    // OWN citations; before 2026-09-05 the first was reported three times over
    // and the rest were discarded silently (measured: LensDuplicateKeys).
    // Occurrence order is EMISSION order, which is positional in the document —
    // it is NOT a claim that occurrence 0 pairs with occurrence 0 of another key.
    // Grouping repeated keys into records is a leaf-path design and is not this.
    int occurrence = 0;
};

// One candidate span for one key (docs/plan-candidate-set.md, pass 2). `value`
// is always a byte-exact slice of `document` — non-verbatim spans are dropped
// by the producer before they reach here, never stored and flagged.
struct LensCandidate {
    std::string value;
    size_t      byte_lo = 0, byte_hi = 0;
};

struct LensReport {
    // v2 (Stage 2, 2026-07-17): `presence_grounded` is GONE from every field.
    // v1 folded its additions in place because they were purely additive — a v0
    // importer that hard-refused unknown fields stayed safe. This one REMOVES a
    // column from a shipped shape, so it cannot ride in place: an importer reading
    // presence_grounded would silently get nothing. Subtractive ⇒ version bump.
    // v3 (2026-09-05): `fields` may carry MORE THAN ONE entry per key — one per
    // occurrence the model emitted (see LensField::occurrence). v1's additions
    // rode in place because they were purely additive and a strict importer
    // stayed safe; this changes a STRUCTURAL invariant that importers can have
    // relied on (fields.size() == key_vocabulary.size(), one field per concept),
    // so it cannot ride. An importer that maps key -> value LAST-wins silently
    // flips from the first repeated value to the last — a wrong value, no error.
    // Same severity class as v2's removal of presence_grounded ⇒ version bump.
    // The first entry for each key is unchanged in value and position, so a
    // first-match importer is unaffected.
    // v4 (2026-09-06, docs/plan-candidate-set.md, architect-approved): adds
    // `key_candidates` + top-level `candidates_error`, additive — a v3 importer
    // that ignores unknown top-level members is unaffected. Bumped anyway
    // (not ridden in place) because the ABSENCE of `key_candidates` is now a
    // load-bearing fact: a v4 response with no `key_candidates` member means
    // "the finder failed on this document"; a v3 response means "this server
    // has no finder." An importer stuck on v3 cannot tell those apart, so the
    // version string is what lets it refuse loud instead of reading absence as
    // silence — see ../qemmi-lens ACCEPTED_FORMAT_VERSIONS, which must add
    // "qemmi-lens/v4" or every extract call fails its own fail-loud gate.
    std::string format_version = "qemmi-lens/v4";
    std::string model;
    bool        validated_envelope = true;
    LensConstants k;

    int prompt_len = 0, doc_lo = 0, doc_hi = 0;
    int n_messages = 0;                     // 0 ⇒ the request sent a plain document
    std::string document_text;              // the user's raw document
    std::string raw_json;                   // exactly what the model emitted

    std::vector<LensField>            fields;    // structured, importer-facing
    std::vector<LensPromptToken>      prompt;    // viewer: token stream + region
    std::vector<std::string>          gen;       // viewer: gen token texts
    std::vector<std::vector<LensCitation>> hover; // viewer: per-gen-token citations
    std::vector<double>               heat;      // viewer: per-prompt-token coverage peak
    std::vector<LensCoverageSpan>     skipped;   // "possibly not incorporated" (peak < used)

    // ── Candidate set (docs/plan-candidate-set.md) — producer + wire ─────────
    // Populated only when LensExtractOptions::want_candidates is true (default
    // false ⇒ pass 2 never runs and these stay default-constructed/empty — the
    // off-path is byte-inert). Keyed by concept key. lens_report_to_json emits
    // this (with `anchor` + `returned_as` derived, not stored) as of v4 — see
    // the header comment on `format_version` above.
    std::map<std::string, std::vector<LensCandidate>> key_candidates;
    // true ⇒ pass 2 generated non-empty output that parsed to ZERO candidate
    // lines across the whole key vocabulary — a PRODUCER failure (the model
    // did not honor the requested `key: "span"` / `key: (none)` shape), and
    // must never be read as "the document offers nothing for every key". This
    // is the fourth state docs/plan-candidate-set.md's own states table was
    // missing (absorbed from the m_en1 probe finding). key_candidates stays
    // empty when this is true.
    bool        candidates_producer_failed = false;
    std::string candidates_error;   // set iff candidates_producer_failed
    // Set true iff LensExtractOptions::want_candidates was true for this
    // extract — i.e. pass 2 was attempted at all, success or failure. This is
    // the ONLY way lens_report_to_json can tell "candidates were not
    // requested" (this stays false, key_candidates stays empty, no error) apart
    // from "requested and ran to a legitimate empty result" — both leave
    // key_candidates empty, but only the former must render `key_candidates`
    // absent-with-no-error on the wire; the two are otherwise indistinguishable
    // from key_candidates/candidates_producer_failed alone. Not itself
    // serialized (it is a driver-side fact, not a document fact).
    bool        candidates_requested = false;
};

// Compute the lens report from one tapped run. Pure; fails loud (throws
// std::runtime_error) only on structurally-impossible input (row width vs n_kv).
LensReport compute_lens_report(const LensRun& run, const LensConstants& k = {});

// ── A5.4: "the lens never lies about where the model looked" ─────────────────
// The product invariant, as a pure predicate over a report — promoted out of
// test_server_lens.cpp so the deterministic unit gate and the LIVE gate
// (QDOCS_S1, free-form output) measure it with the SAME ruler rather than two
// drifting copies. Same reason the tier heuristic is shared, not re-implemented.

// Does [c_lo,c_hi) overlap ANY occurrence of `value` in `document` (± tol bytes)?
// "Any occurrence", not just the first: citing a later duplicate (a conflict's
// second copy) is faithful, not a false receipt — the lens names no conflict
// winner (CF1), so either real source is a faithful receipt.
bool lens_cites_a_real_source(const std::string& document, const std::string& value,
                              size_t c_lo, size_t c_hi, long tol = 2);

// THE GATE. Count fields whose confident receipt is not faithful: a grounded,
// distinctive, verbatim value whose top-1 citation lands on no occurrence of
// that value. Anything counted here is a lie the format would be telling.
// Scoped deliberately — ungrounded ⇒ not a confident claim; short_numeric ⇒ the
// weak class the format makes no citation claim for (plan §1.3); not-verbatim ⇒
// its own disclosure (found_in_document=false). Required to be ZERO on any
// corpus, constrained or free.
int lens_count_confident_false_receipts(const LensReport& r, long tol = 2);

// Serialize a report to the lens-format JSON (a superset of the demo's data
// shape so docs/demo/attention-lens.html renders it; P3 owns the spec).
std::string lens_report_to_json(const LensReport& r);

// ── The shape contract (docs/lens-format.md) ─────────────────────────────────
// Thrown when the model's output for a document cannot be parsed into a JSON
// object. A DISTINCT type, because the endpoint must answer 422
// unparseable_extraction rather than 400 bad_request: the request was fine, the
// model's output was not, and an importer has to tell those apart without
// string-matching a message. Never a partial extraction — an unparseable document
// is a loud refusal, which is strictly better than a constraint that corrupts the
// output to avoid a failure it does not actually prevent.
struct LensUnparseableError : std::runtime_error {
    LensUnparseableError(const std::string& what, std::string raw_output)
        : std::runtime_error(what), raw(std::move(raw_output)) {}
    std::string raw;  // exactly what the model emitted, for the 422 body
};

// TOLERANT on shape: skip a ``` fence, then locate the OUTERMOST {...} in `raw`
// by brace-depth (string- and escape-aware, so a brace inside a value does not
// end it). Returns its byte span [lo,hi) within `raw` — byte offsets, not a
// substring, because the gen-token span math must stay anchored to `raw`.
// false ⇒ no object found. Tolerance is bounded and mechanical: it recovers
// SHAPE and never guesses CONTENT.
bool lens_find_json_object(const std::string& raw, size_t& lo, size_t& hi);

// ── Concepts ─────────────────────────────────────────────────────────────────
// A hinted concept. The complete hint is what holds key names stable (Leg B) —
// dropping the grammar does not reopen the naming zoo; dropping the hint would.
//
// `gloss` is ACCEPTED AND CURRENTLY UNUSED. Its only consumer was the deleted
// presence gate's Pass-A question (where it lifted recall 0.75 → 0.92). It is
// kept in the request shape deliberately: removing it would be a second breaking
// change, and it is a plausible future lever. It is NOT fed into the extraction
// instruction — that would silently change the exact prompt regime Stage 1
// validated, on no measurement. If a glossed instruction is ever wanted, measure
// it first.
struct LensConcept { std::string key, gloss; };

// ── Driver ───────────────────────────────────────────────────────────────────
struct LensExtractOptions {
    int  max_new_tokens = 512;   // hard cap on the emitted JSON length
    bool validated_envelope_only = false;  // reserved; false = accept + disclose
    // Per-request toggle for the candidate set (docs/plan-candidate-set.md).
    // Default OFF: pass 1 is untouched either way, and false means run_lens_extract
    // does not run pass 2 at all — no second prefill, no extra decode, no
    // behaviour change of any kind. No server flag / CLI flag exists for this;
    // it is request-scoped only, by design. The wire contract for what
    // want_candidates=true PRODUCES (`key_candidates`, `anchor`, `returned_as`,
    // format_version qemmi-lens/v4) is architect-approved and implemented
    // (docs/plan-candidate-set.md); still out of scope is any route/flag that
    // would let an HTTP caller flip this bit — that remains a separate decision.
    bool want_candidates = false;
    // Message boundaries, as byte offsets into `document`, when the caller sent
    // `messages` instead of a flat `document` (the server joins them and fills
    // this in). Empty ⇒ plain document, and every citation reports message -1.
    // Boundaries buy ATTRIBUTION ("this value was read from message 23"), which
    // is all the lens claims. They deliberately do NOT buy a staleness alarm:
    // measured, a later-message rule cried wolf on 7 of 9 correctly-handled
    // corrections and stayed silent on the real failure, because a later message
    // routinely restates an old value (docs/note-ss3-matched-pairs.md §3).
    std::vector<size_t> message_offsets;
};

// Assemble the ChatML thinking-off prompt from (document, concepts), run the
// FREE tapped decode single-slot, and compute the report — one prefill, one pass.
// Fields come back in `concepts` order; a hinted concept the model did not emit
// comes back absent (value:null, badge:"absent"). Borrows fp/sched/tok by
// reference — owns none of them.
//
// Fails loud: std::runtime_error on empty concepts / empty document / a prompt
// exceeding the model's context; LensUnparseableError (⇒ 422) when the emitted
// output holds no parseable JSON object.
//
// `control_arm_grammar` is a PROBE-ONLY seam and defaults to nullptr — production
// omits it and decodes free. QDOCS_S1 passes the refuted `lens_grammar_gbnf()`
// through it to run the constrained arm against the free one on this same driver,
// which is the only reason the comparison stays honest rather than measuring a
// lookalike. `control_arm_vocab` is the grammar's token table and must be non-null
// exactly when the grammar is. NOT reachable from /v1/extract; unrelated to the
// server's per-request `grammar` field on the OpenAI endpoints.
//
// NOTE: `::Tokenizer` is force-qualified. Tokenizer is a global type, but
// inference_server.h forward-declares a phantom `qinf::Tokenizer`; without
// the `::` this declaration would bind to that phantom inside namespace qinf
// wherever both headers are visible (e.g. http_server.cpp), mismatching the
// definition.
LensReport run_lens_extract(ForwardPassBase* fp, ggml_backend_sched_t sched,
                            ::Tokenizer* tok, const ModelMetadata& meta,
                            uint32_t vocab_size, uint32_t n_ctx_max,
                            const std::string& document,
                            const std::vector<LensConcept>& concepts,
                            const LensExtractOptions& opts,
                            // No default: the constants are per model (see
                            // kLensCalibrations). A defaulted `k` would silently
                            // run Qwen 3.6 coordinates on whatever is loaded —
                            // the false receipt the refusal exists to prevent.
                            const LensConstants& k,
                            GrammarVocab* control_arm_grammar = nullptr,
                            const std::vector<std::string>* control_arm_vocab = nullptr);

// Pure: order `report.fields` by `concepts` and mark absent-by-omission — a
// hinted concept the model did not emit (or emitted empty) becomes a value-null,
// badge:"absent" field. Keys outside the hint are dropped (the complete hint
// defines the surface; Leg B measured 15/15 key stability at greedy, so this is
// not where data hides). No engine — unit-testable with a synthesized report.
//
// Deliberately does NOT second-guess a value the model DID emit. The presence
// gate's "safety net" (re-mark an ungrounded, non-verbatim value as absent) died
// with the gate: it was only defensible as the second of two independent signals.
// Alone it would be the lens judging a value WRONG — a claim the format refuses
// ("No correctness"). The badges already disclose it; the importer decides.
LensReport apply_absent_by_omission(LensReport report,
                                    const std::vector<LensConcept>& concepts);

// The fixed ChatML instruction that names the complete key vocabulary (plan §1.2
// — a complete hint or the naming zoo returns). Exposed for the startup sanity
// check and tests.
std::string lens_build_instruction(const std::vector<std::string>& key_vocabulary);

// The REFUTED fixed KV grammar (GBNF text) — docs/note-nogrammar-refutation.md.
// NOT on the product path: it exists solely so the QDOCS_S1 probe can run it as a
// control arm against the free path (see run_lens_extract's control_arm_grammar).
// Kept as exercised, measured history rather than deleted, the same precedent as
// the other refuted machinery. Do not wire this to an endpoint.
const char* lens_grammar_gbnf();

// ── Candidate set — pass 2 (docs/plan-candidate-set.md) ──────────────────────
// Ported from tests/perf/attn_provenance.cpp's CAND=1 probe
// (CAND_PASS2_TASK_PREFIX / cand_parse_pass2), which measured the cheap-kill
// gate: median candidate-set size 1.0 on 75 uncontested keys, 100%
// byte-exactness. Both are PURE (no model, no engine) and exposed for tests;
// the cold decode that produces the raw text pass 2 parses lives in
// server_lens.cpp and is not model-free, so it is not unit-tested here.

// The pass-2 instruction: a fixed task description (asking for every span
// answering each key, quoted verbatim, one per line, "(none)" if absent) plus
// the complete key vocabulary — built the same way as lens_build_instruction.
std::string lens_cand_pass2_instruction(const std::vector<std::string>& keys);

// Tolerant line parser for pass 2's free-form output. Tolerates bullets,
// numbering, and unquoted `key: span` (a strict `key: "span"`-only parser
// silently dropped an entire document's correct output when the model emitted
// every line unquoted, and rendered it as "this document offers no answers" —
// the exact confusion the candidate-set format exists to prevent, reproduced
// one level down). `(none)` is a real ANSWER (no candidate for that key), not
// a malformed line. An unterminated quote IS malformed and is skipped. Returns
// (key, span) pairs in EMISSION order, restricted to `keys` — the caller
// dedups, byte-checks against the document, and sorts into document order.
std::vector<std::pair<std::string, std::string>>
lens_parse_pass2_candidates(const std::string& text, const std::vector<std::string>& keys);

// PURE (no engine): parses pass 2's raw output with lens_parse_pass2_candidates,
// verifies every candidate is a byte-exact slice of `document` (dropping any
// that are not — the format requires exactness, not a disclosure about it),
// dedups per key, sorts each key's candidates into document order (byte_lo
// ascending), and writes the result into `report.key_candidates`.
//
// FAILS LOUD on a producer failure: non-empty `gen_text` that parses to zero
// candidate lines sets report.candidates_producer_failed + candidates_error
// instead of silently leaving key_candidates empty — see the header comment on
// LensReport::candidates_producer_failed for why (docs/plan-candidate-set.md's
// own states table is missing this as its fourth state).
void lens_apply_pass2_candidates(const std::string& document, const std::string& gen_text,
                                 const std::vector<std::string>& keys, LensReport& report);

// A5.3 trust tier of a (present) value, by VALUE SHAPE — deterministic,
// regex-grade: a short bare integer ⇒ "short_numeric" (the weak citation class,
// plan §1.3, coverage-backstopped); anything with structure ⇒ "distinctive"
// (the class the lens claims citations for). Empty ⇒ "" (used for absent).
std::string lens_value_tier(const std::string& value);

}  // namespace qinf
