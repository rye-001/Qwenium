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
// The frozen constants are Qwen3.6-pinned; the driver runs a startup
// known-answer sanity check and fails loud if the citation head has drifted
// (plan §1.6). No lens claims for other model families yet.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

class ForwardPassBase;
class Tokenizer;
struct ModelMetadata;
typedef struct ggml_backend_sched* ggml_backend_sched_t;

namespace qinf {

class GrammarVocab;

// ── Frozen Qwen3.6 lens constants (plan §1.6; note-qemmi-docs-p0.md) ──────────
struct LensConstants {
    int    citation_head        = 13;     // L3H13 retrieval head (N3)
    int    citation_layer       = 3;      // physical attention layer for citations
    int    coverage_layer       = 11;     // physical attention layer for coverage (COV1)
    double coverage_used_peak   = 0.705;  // span-peak ≥ this ⇒ "consulted" (COV1)
    double ungrounded_body_mass = 0.538;  // mean body_mass ≥ this ⇒ grounded (N3b)
    int    citation_topk        = 8;      // source positions reported per token/field
};

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
    std::string model;                      // passthrough for the report header
    // false ⇒ prompt exceeded the 4 K CALIBRATION floor (a disclosure on the
    // report, not an error). Unrelated to the 10 K workload envelope.
    bool validated_envelope = true;
};

// ── Lens report (the interchange format; P3 versions/documents it) ───────────
struct LensCitation { int pos; double mass; size_t byte_lo, byte_hi; };  // attended prompt position
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
    // ── Absent by omission (Stage 2; was the two-pass presence gate, A5.1) ───
    // false ⇒ ABSENT: the model simply did not emit this hinted concept (or
    // emitted it empty). Serializes value:null, badge:"absent". This is now a
    // MECHANICAL read of the parsed output, not a verdict: with no grammar the
    // model declines natively and correctly (30/30 on the Leg C corpus, against
    // the grammar's 10/30 — docs/note-nogrammar-refutation.md), so there is
    // nothing to gate. The N+1 presence prefills are gone with it.
    bool present = true;
};

struct LensReport {
    // v2 (Stage 2, 2026-07-17): `presence_grounded` is GONE from every field.
    // v1 folded its additions in place because they were purely additive — a v0
    // importer that hard-refused unknown fields stayed safe. This one REMOVES a
    // column from a shipped shape, so it cannot ride in place: an importer reading
    // presence_grounded would silently get nothing. Subtractive ⇒ version bump.
    std::string format_version = "qemmi-lens/v2";
    std::string model;
    bool        validated_envelope = true;
    LensConstants k;

    int prompt_len = 0, doc_lo = 0, doc_hi = 0;
    std::string document_text;              // the user's raw document
    std::string raw_json;                   // exactly what the model emitted

    std::vector<LensField>            fields;    // structured, importer-facing
    std::vector<LensPromptToken>      prompt;    // viewer: token stream + region
    std::vector<std::string>          gen;       // viewer: gen token texts
    std::vector<std::vector<LensCitation>> hover; // viewer: per-gen-token citations
    std::vector<double>               heat;      // viewer: per-prompt-token coverage peak
    std::vector<LensCoverageSpan>     skipped;   // "possibly not incorporated" (peak < used)
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
                            const LensConstants& k = {},
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

// A5.3 trust tier of a (present) value, by VALUE SHAPE — deterministic,
// regex-grade: a short bare integer ⇒ "short_numeric" (the weak citation class,
// plan §1.3, coverage-backstopped); anything with structure ⇒ "distinctive"
// (the class the lens claims citations for). Empty ⇒ "" (used for absent).
std::string lens_value_tier(const std::string& value);

}  // namespace qinf
