// test_server_lens.cpp — Qemmi-Lens pure computation gate (docs/plan-qemmi-lens.md
// P2). compute_lens_report is model-free: it consumes one tapped run (tokens,
// byte maps, per-step kq_soft rows) and emits the lens report. These tests
// synthesize rows so the citation/coverage/badge numerics are checked without a
// model. The driver (run_lens_extract) is covered by the endpoint smoke.

#include <gtest/gtest.h>
#include <cctype>
#include <string>
#include <vector>

#include "../../src/server/server_lens.h"

using namespace qinf;

namespace {

// A controlled fixture: document "buy 45 now\nskip me\n" + a 2-token instruction,
// generated JSON {"qty":"45"}. Body tokens 0..6, instruction tokens 7..8.
// n_head = 2, n_kv = 18 (constant, ≥ every position we index).
struct Fixture {
    static constexpr int N_HEAD = 2;
    static constexpr int N_KV   = 18;
    static constexpr int P      = 9;
    static constexpr int INSTR  = 7;

    LensConstants k;
    LensRun run;

    Fixture() {
        k.citation_head        = 0;      // the fixture puts signal on head 0
        k.coverage_used_peak   = 0.705;
        k.ungrounded_body_mass = 0.538;
        k.citation_topk        = 8;

        run.model    = "test";
        run.n_head   = N_HEAD;
        run.doc_lo   = 0;          // tokens 0..6 are the document
        run.doc_hi   = INSTR;      // tokens 7..8 are the instruction
        run.doc_byte_offset = 0;   // document starts at prompt byte 0 in this fixture
        run.document = "buy 45 now\nskip me\n";

        run.prompt_text = {"buy ", "45", " now", "\n", "skip", " me", "\n", "\nEXTRACT", " json"};
        run.prompt_cum.assign(P + 1, 0);
        for (int i = 0; i < P; ++i) run.prompt_cum[i + 1] = run.prompt_cum[i] + run.prompt_text[i].size();

        run.gen_tok_text = {"{", "\"", "qty", "\"", ":", "\"", "45", "\"", "}"};
        run.gen_text = "{\"qty\":\"45\"}";
        run.gen_cum.assign(run.gen_tok_text.size() + 1, 0);
        for (size_t i = 0; i < run.gen_tok_text.size(); ++i)
            run.gen_cum[i + 1] = run.gen_cum[i] + run.gen_tok_text[i].size();

        const int G = (int)run.gen_tok_text.size();
        run.steps.resize(G);
        for (auto& st : run.steps) {
            st.n_kv = N_KV;
            st.citation_row.assign((size_t)N_HEAD * N_KV, 0.0f);
            st.coverage_row.assign((size_t)N_HEAD * N_KV, 0.0f);
        }
        // Coverage: line [0,3] ("buy 45 now\n") is consulted (peak ≥ 0.705);
        // line [4,6] ("skip me\n") is never consulted ⇒ skipped.
        cov(0, /*head*/0, /*pos*/1, 0.8f);
    }

    // set citation_row[step][head][pos]
    void cite(int step, int head, int pos, float m) {
        run.steps[step].citation_row[(size_t)head * N_KV + pos] = m;
    }
    void cov(int step, int head, int pos, float m) {
        run.steps[step].coverage_row[(size_t)head * N_KV + pos] = m;
    }
};

}  // namespace

// ── Fields parse, value byte-span, and a grounded badge ──────────────────────
TEST(ServerLensCompute, GroundedFieldCitesSource) {
    Fixture fx;
    // Value token "45" is gen index 6 ⇒ provenance is step 5. Attend to the
    // document's "45" at prompt position 1 with high mass ⇒ grounded + cite pos 1.
    fx.cite(5, /*head*/0, /*pos*/1, 0.9f);

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 1u);
    const LensField& f = r.fields[0];
    EXPECT_EQ(f.key, "qty");
    EXPECT_EQ(f.value, "45");
    EXPECT_TRUE(f.found_in_document);
    EXPECT_EQ(f.value_byte_lo, 4u);              // "45" starts at byte 4 of the body
    EXPECT_TRUE(f.grounded);
    EXPECT_NEAR(f.body_mass, 0.9, 1e-5);
    ASSERT_FALSE(f.citations.empty());
    EXPECT_EQ(f.citations[0].pos, 1);            // cited the source token
    EXPECT_EQ(f.citations[0].byte_lo, 4u);
}

// ── The badge flips below the body_mass threshold ────────────────────────────
TEST(ServerLensCompute, UngroundedWhenBodyMassLow) {
    Fixture fx;
    fx.cite(5, /*head*/0, /*pos*/1, 0.1f);        // barely looks at the document
    fx.cite(5, /*head*/0, /*pos*/7, 0.8f);        // mass on the instruction (excluded)

    LensReport r = compute_lens_report(fx.run, fx.k);
    ASSERT_EQ(r.fields.size(), 1u);
    EXPECT_FALSE(r.fields[0].grounded);
    EXPECT_NEAR(r.fields[0].body_mass, 0.1, 1e-5);  // only body positions count
}

// ── Coverage: the un-consulted body line is flagged, the consulted one is not ─
TEST(ServerLensCompute, CoverageFlagsSkippedLine) {
    Fixture fx;
    fx.cite(5, 0, 1, 0.9f);
    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.skipped.size(), 1u);
    EXPECT_EQ(r.skipped[0].lo, 4);                // "skip me\n" line
    EXPECT_LT(r.skipped[0].peak, fx.k.coverage_used_peak);
    EXPECT_NE(r.skipped[0].text.find("skip"), std::string::npos);
    // The consulted line [0,3] must NOT appear.
    for (const auto& s : r.skipped) EXPECT_NE(s.lo, 0);
    EXPECT_NEAR(r.heat[1], 0.8, 1e-5);            // per-token coverage peak
}

// ── Viewer data shape ────────────────────────────────────────────────────────
TEST(ServerLensCompute, ViewerArraysAndJson) {
    Fixture fx;
    fx.cite(5, 0, 1, 0.9f);
    LensReport r = compute_lens_report(fx.run, fx.k);

    EXPECT_EQ(r.prompt.size(), (size_t)Fixture::P);
    EXPECT_EQ(r.prompt[0].region, "body");
    EXPECT_EQ(r.prompt[Fixture::INSTR].region, "instr");
    EXPECT_EQ(r.hover.size(), r.gen.size());

    std::string json = lens_report_to_json(r);
    EXPECT_NE(json.find("\"qty\""), std::string::npos);
    EXPECT_NE(json.find("grounded"), std::string::npos);
    EXPECT_NE(json.find("\"skipped\""), std::string::npos);
    EXPECT_NE(json.find("qemmi-lens/v2"), std::string::npos);  // Stage 2: subtractive bump
}

// ── Fail-loud on structurally-impossible input ───────────────────────────────
TEST(ServerLensCompute, FailsLoudOnStepMismatch) {
    Fixture fx;
    fx.run.steps.pop_back();                      // steps != gen tokens
    EXPECT_THROW(compute_lens_report(fx.run, fx.k), std::runtime_error);
}

// ════════════════════════════════════════════════════════════════════════════
// A5.4 — the standing lens-fidelity gate: ZERO CONFIDENT FALSE RECEIPTS.
//
// Product invariant (plan §5): "the lens never lies about where the model
// looked." A *confident false receipt* is the one way the format could lie — a
// value carrying the lens's STRONG claim (a `grounded` badge on a DISTINCTIVE
// value, the class §1.3 vouches citations for) whose top-1 citation does not
// land on any occurrence of the value in the document. Measured 0 across the
// whole probe program; this gate makes that a permanent, deterministic,
// model-free property with the same status as the byte-identity gates. Any
// change to the tap, grammar, or lens computation must keep the count at zero.
//
// The gate is deliberately SCOPED: short bare numerics (§1.3 — quantities cite
// ~65%, coverage-backstopped) and ungrounded values are not confident claims,
// so a missed citation there is not a false receipt. Scoping the gate to
// exactly the class the format vouches for is what lets it be a hard zero
// instead of a statistical bar (the live smoke keeps the statistical view).
// (A dedicated "absent" value class was tried and reverted — see
// docs/note-lens-absent-attempt.md.)
// ════════════════════════════════════════════════════════════════════════════
namespace {

// The WEAK citation class (§1.3) is exactly the format's short_numeric tier —
// so the gate scopes itself with the SAME promoted heuristic the report emits,
// not a private copy.
bool is_short_bare_numeric(const std::string& v) {
    return lens_value_tier(v) == "short_numeric";
}

// The gate predicates live in server_lens.{h,cpp} — the LIVE free-form gate
// (QDOCS_S1) must measure A5.4 with the same ruler as this deterministic one,
// so these are thin aliases onto the promoted functions, never private copies.
// The tests below own the SCOPING and the TEETH; the module owns the property.
bool cites_a_real_source(const std::string& document, const std::string& value,
                         size_t c_lo, size_t c_hi, long tol) {
    return lens_cites_a_real_source(document, value, c_lo, c_hi, tol);
}

int count_confident_false_receipts(const LensReport& r, long tol = 2) {
    return lens_count_confident_false_receipts(r, tol);
}

// Lens constants for the 2-head fixtures: the signal lives on head 0 (the real
// L3H13 index would run off a 2-head row).
LensConstants fk() { LensConstants k; k.citation_head = 0; return k; }

// Build a LensRun from explicit token streams (body ⧺ instruction, then gen).
// The body tokens ARE the document (doc_byte_offset 0), so a value's citation
// byte span is directly comparable to where the value string lives in the doc.
constexpr int NKV = 24;
LensRun build_run(const std::vector<std::string>& body,
                  const std::vector<std::string>& instr,
                  const std::vector<std::string>& gen, int n_head = 2) {
    LensRun run;
    run.model  = "fidelity-fixture";
    run.n_head = n_head;
    run.prompt_text = body;
    run.prompt_text.insert(run.prompt_text.end(), instr.begin(), instr.end());
    const int P = (int)run.prompt_text.size();
    run.prompt_cum.assign(P + 1, 0);
    for (int i = 0; i < P; ++i)
        run.prompt_cum[i + 1] = run.prompt_cum[i] + run.prompt_text[i].size();
    run.doc_lo = 0; run.doc_hi = (int)body.size(); run.doc_byte_offset = 0;
    for (const auto& b : body) run.document += b;

    run.gen_tok_text = gen;
    run.gen_cum.assign(gen.size() + 1, 0);
    for (size_t i = 0; i < gen.size(); ++i)
        run.gen_cum[i + 1] = run.gen_cum[i] + gen[i].size();
    for (const auto& g : gen) run.gen_text += g;

    run.steps.resize(gen.size());
    for (auto& st : run.steps) {
        st.n_kv = NKV;
        st.citation_row.assign((size_t)n_head * NKV, 0.0f);
        st.coverage_row.assign((size_t)n_head * NKV, 0.0f);
    }
    return run;
}

// Set citation_row[step][head 0][pos] = m (overwriting).
void cite(LensRun& run, int step, int pos, float m) {
    run.steps[step].citation_row[pos] = m;
}

// A four-field order fixture spanning both value classes, every field faithful:
//   due   = 2025-11-20  (distinctive: date)   cite step3  -> pos1
//   ref   = BST-88213   (distinctive: alnum)  cite step7  -> pos3
//   qty   = 875         (weak: bare numeric)  cite step11 -> pos5
//   email = a@b.co      (distinctive: addr)   cite step15 -> pos7
LensRun orders_fixture() {
    std::vector<std::string> body = {
        "due ", "2025-11-20", " ref ", "BST-88213",
        " qty ", "875", " mail ", "a@b.co", "\n"};   // pos 0..8
    std::vector<std::string> instr = {"\nEXTRACT", " json"};
    std::vector<std::string> gen = {
        "{", "\"", "due", "\":\"", "2025-11-20", "\",\"",   // value idx 4  -> step 3
        "ref", "\":\"", "BST-88213", "\",\"",               // value idx 8  -> step 7
        "qty", "\":\"", "875", "\",\"",                     // value idx 12 -> step 11
        "email", "\":\"", "a@b.co", "\"}"};                 // value idx 16 -> step 15
    LensRun run = build_run(body, instr, gen);
    cite(run, 3,  1, 0.9f);
    cite(run, 7,  3, 0.9f);
    cite(run, 11, 5, 0.9f);
    cite(run, 15, 7, 0.9f);
    return run;
}

}  // namespace

// ── The corpus is faithful ⇒ zero confident false receipts (the standing gate) ─
TEST(LensFidelityGate, FaithfulCorpusHasZeroFalseReceipts) {
    LensReport r = compute_lens_report(orders_fixture(), fk());
    ASSERT_EQ(r.fields.size(), 4u);
    // Every distinctive value is grounded and its top citation lands on source;
    // the bare numeric is grounded too but exempt from the citation claim.
    for (const auto& f : r.fields) {
        EXPECT_TRUE(f.grounded) << f.key;
        EXPECT_TRUE(f.found_in_document) << f.key;
    }
    EXPECT_TRUE(is_short_bare_numeric("875"));
    EXPECT_FALSE(is_short_bare_numeric("2025-11-20"));
    EXPECT_FALSE(is_short_bare_numeric("BST-88213"));
    EXPECT_EQ(count_confident_false_receipts(r), 0);
}

// ── The gate has TEETH: a distinctive value cited off-source is caught ─────────
// Without this, a "must be zero" gate that can never be non-zero is worthless.
TEST(LensFidelityGate, FiresOnConfidentMisattribution) {
    LensRun run = orders_fixture();
    // Move `ref`'s citation off BST-88213 (pos3) onto the date (pos1): still
    // grounded, still distinctive, but the confident receipt now points at the
    // wrong span — a lie about where the model looked.
    run.steps[7].citation_row[3] = 0.0f;
    cite(run, 7, 1, 0.9f);

    LensReport r = compute_lens_report(run, fk());
    EXPECT_EQ(count_confident_false_receipts(r), 1);
    for (const auto& f : r.fields)
        if (f.key == "ref") {
            EXPECT_TRUE(f.grounded);
            EXPECT_FALSE(cites_a_real_source(r.document_text, f.value,
                         f.citations[0].byte_lo, f.citations[0].byte_hi, 2));
        }
}

// ── Scoping: a short bare numeric cited off-source is NOT a confident receipt ──
// §1.3 makes no citation claim for this class, so the gate must not over-fire.
TEST(LensFidelityGate, ShortNumericMisattributionIsExempt) {
    LensRun run = orders_fixture();
    run.steps[11].citation_row[5] = 0.0f;   // move qty's citation off 875 (pos5)
    cite(run, 11, 1, 0.9f);                 // onto the date

    LensReport r = compute_lens_report(run, fk());
    for (const auto& f : r.fields)
        if (f.key == "qty") EXPECT_TRUE(f.grounded);  // grounded, but weak-class
    EXPECT_EQ(count_confident_false_receipts(r), 0);
}

// ── Scoping: an ungrounded value is not a confident claim ─────────────────────
TEST(LensFidelityGate, UngroundedMisattributionIsNotConfident) {
    LensRun run = orders_fixture();
    run.steps[15].citation_row[7] = 0.0f;   // email: drop mass off a@b.co
    cite(run, 15, 1, 0.1f);                 // wrong span AND below the threshold

    LensReport r = compute_lens_report(run, fk());
    for (const auto& f : r.fields)
        if (f.key == "email") EXPECT_FALSE(f.grounded);
    EXPECT_EQ(count_confident_false_receipts(r), 0);
}

// ── Conflict/duplicate: citing a later occurrence of the value is faithful ─────
// The reported value_span is the FIRST occurrence; the model here reads the
// second. A single-span check would false-alarm; "any occurrence" does not —
// either real source is a faithful receipt (the lens names no winner, CF1).
TEST(LensFidelityGate, CitingADuplicateOccurrenceIsFaithful) {
    std::vector<std::string> body = {
        "date ", "2025-11-20", " note ", "2025-11-20", "\n"};  // value twice
    std::vector<std::string> instr = {"\nEXTRACT", " json"};
    std::vector<std::string> gen = {
        "{", "\"", "due", "\":\"", "2025-11-20", "\"}"};        // value idx 4 -> step 3
    LensRun run = build_run(body, instr, gen);
    cite(run, 3, 3, 0.9f);   // attend to the SECOND occurrence (pos3), not the first (pos1)

    LensReport r = compute_lens_report(run, fk());
    ASSERT_EQ(r.fields.size(), 1u);
    const LensField& f = r.fields[0];
    EXPECT_TRUE(f.grounded);
    EXPECT_EQ(f.value_byte_lo, 5u);   // report's span is the first occurrence
    // ...but the citation lands on the second occurrence, which is a real source.
    EXPECT_EQ(count_confident_false_receipts(r), 0);
}

// ════════════════════════════════════════════════════════════════════════════
// A5.3 per-field trust tier (qemmi-lens/v1).
// ════════════════════════════════════════════════════════════════════════════

// ── The promoted tier heuristic matches the two-tier claim (§1.3) ─────────────
TEST(LensTier, TierHeuristicByValueShape) {
    EXPECT_EQ(lens_value_tier("875"), "short_numeric");
    EXPECT_EQ(lens_value_tier("2500"), "short_numeric");
    EXPECT_EQ(lens_value_tier("7781"), "short_numeric");     // a bare order number
    EXPECT_EQ(lens_value_tier("2025-11-20"), "distinctive"); // date (hyphens)
    EXPECT_EQ(lens_value_tier("BST-88213"), "distinctive");  // alnum ref
    EXPECT_EQ(lens_value_tier("47.30 EUR"), "distinctive");  // formatted amount
    EXPECT_EQ(lens_value_tier("1234567"), "distinctive");    // >6 digits ⇒ ID-like
    EXPECT_EQ(lens_value_tier(""), "");                      // empty
}

// ── The report and its JSON carry the per-field tier ──────────────────────────
TEST(LensTier, ReportAndJsonCarryTier) {
    LensReport r = compute_lens_report(orders_fixture(), fk());
    ASSERT_EQ(r.fields.size(), 4u);
    for (const auto& f : r.fields) {
        if (f.key == "qty") EXPECT_EQ(f.tier, "short_numeric");
        else EXPECT_EQ(f.tier, "distinctive");
    }
    std::string json = lens_report_to_json(r);
    EXPECT_NE(json.find("\"tier\":\"distinctive\""), std::string::npos);
    EXPECT_NE(json.find("\"tier\":\"short_numeric\""), std::string::npos);
}

// ════════════════════════════════════════════════════════════════════════════
// Stage 2 — absent BY OMISSION (apply_absent_by_omission), the pure half.
//
// The two-pass grounded presence gate that used to live here is DELETED
// (docs/note-nogrammar-refutation.md): it existed only to contain a wound the
// fixed KV grammar inflicted — `value ::= (…)+` forced a non-empty value for
// every hinted key, so a concept the document lacks had to be fabricated. With
// no grammar the model declines natively (30/30 on Leg C vs the grammar's
// 10/30), so absence is a MECHANICAL read of the parsed output, not a verdict.
//
// The invariants that survive the gate's death, and are fixed here: ordering by
// concept, absent injection, collapse-immunity, and the deliberate ABSENCE of a
// safety net (which was only defensible as the second of two signals).
// ════════════════════════════════════════════════════════════════════════════
namespace {

LensField present_field(const std::string& key, const std::string& value,
                        bool grounded, bool found) {
    LensField f;
    f.key = key; f.value = value; f.tier = lens_value_tier(value);
    f.grounded = grounded; f.found_in_document = found; f.present = true;
    f.body_mass = grounded ? 0.9 : 0.1;
    return f;
}

}  // namespace

// ── Fields come back in CONCEPT order, with the un-emitted ones absent ─────────
TEST(LensAbsentByOmission, OrdersByConceptAndInjectsAbsent) {
    LensReport rep;
    rep.document_text = "ACME GmbH ordered 45 units";
    rep.fields = {present_field("quantity", "45", true, true),
                  present_field("customer", "ACME GmbH", true, true)};  // emitted out of order
    std::vector<LensConcept> concepts = {{"customer", ""}, {"payment_terms", "how it is paid"},
                                         {"quantity", ""}};

    LensReport m = apply_absent_by_omission(rep, concepts);

    ASSERT_EQ(m.fields.size(), 3u);
    EXPECT_EQ(m.fields[0].key, "customer");
    EXPECT_EQ(m.fields[0].value, "ACME GmbH");
    EXPECT_TRUE(m.fields[0].present);
    // payment_terms was never emitted ⇒ absent, and it claims nothing.
    EXPECT_EQ(m.fields[1].key, "payment_terms");
    EXPECT_FALSE(m.fields[1].present);
    EXPECT_TRUE(m.fields[1].value.empty());
    EXPECT_TRUE(m.fields[1].tier.empty());
    EXPECT_FALSE(m.fields[1].grounded);
    EXPECT_FALSE(m.fields[1].found_in_document);
    EXPECT_TRUE(m.fields[1].citations.empty());
    EXPECT_EQ(m.fields[2].key, "quantity");
    EXPECT_TRUE(m.fields[2].present);
}

// ── An absent concept does NOT take the present ones down with it ─────────────
// The founding bug: under the grammar, hinting an absent concept collapsed the
// WHOLE extraction to `","`. This is the config that killed the reverted A5.1 —
// absent concept FIRST in the hint order.
TEST(LensAbsentByOmission, PresentFieldsSurviveAnAbsentConceptFirst) {
    LensReport rep;
    rep.document_text = "ACME GmbH ordered 45 units";
    rep.fields = {present_field("customer", "ACME GmbH", true, true),
                  present_field("quantity", "45", true, true)};
    std::vector<LensConcept> concepts = {{"warranty_period", ""}, {"customer", ""},
                                         {"quantity", ""}};

    LensReport m = apply_absent_by_omission(rep, concepts);

    ASSERT_EQ(m.fields.size(), 3u);
    EXPECT_FALSE(m.fields[0].present);                 // absent, first
    EXPECT_EQ(m.fields[1].value, "ACME GmbH");         // intact
    EXPECT_EQ(m.fields[2].value, "45");                // intact
}

// ── An emitted-but-empty value is absent, not a claim ─────────────────────────
// Free decode declines by omitting the key OR by emitting "". Both are "not
// stated"; neither is a value.
TEST(LensAbsentByOmission, EmptyValueIsAbsent) {
    LensReport rep;
    rep.document_text = "ACME GmbH";
    rep.fields = {present_field("customer", "ACME GmbH", true, true),
                  present_field("payment_terms", "", false, false)};
    std::vector<LensConcept> concepts = {{"customer", ""}, {"payment_terms", ""}};

    LensReport m = apply_absent_by_omission(rep, concepts);

    ASSERT_EQ(m.fields.size(), 2u);
    EXPECT_TRUE(m.fields[0].present);
    EXPECT_FALSE(m.fields[1].present);
}

// ── A JSON `null` is the model DECLINING, not a value called "null" ───────────
// Caught live by the smoke gate, 2026-07-17. Freed of the grammar, Qwen3.6
// spontaneously emits `"payment_terms": null` for a concept the document lacks
// (exactly as note-nogrammar-refutation.md predicted). The text scanner read the
// unquoted literal back as the 4-char STRING "null" — non-empty ⇒ reported as a
// present, fabricated field, against a body_mass of 0.0002. Our own machinery
// corrupting a CORRECT decline: the grammar's sin, one layer down.
TEST(LensAbsentByOmission, JsonNullValueIsADeclineNotAValue) {
    std::vector<std::string> body  = {"cust ", "ACME GmbH", "\n"};
    std::vector<std::string> instr = {"\nEXTRACT", " json"};
    std::vector<std::string> gen   = {"{", "\"", "cust", "\":\"", "ACME GmbH", "\",",
                                      "\"", "terms", "\":", " null", "}"};
    LensRun run = build_run(body, instr, gen);
    cite(run, 3, 1, 0.9f);

    LensReport r = compute_lens_report(run, fk());
    LensReport m = apply_absent_by_omission(r, {{"cust", ""}, {"terms", ""}});

    ASSERT_EQ(m.fields.size(), 2u);
    EXPECT_TRUE(m.fields[0].present);
    EXPECT_EQ(m.fields[0].value, "ACME GmbH");
    EXPECT_FALSE(m.fields[1].present) << "JSON null must read as absent, not value \"null\"";
    EXPECT_TRUE(m.fields[1].value.empty());
    EXPECT_NE(lens_report_to_json(m).find("\"badge\":\"absent\""), std::string::npos);
}

// ── ...but a QUOTED "null" is a real value ────────────────────────────────────
// The distinction is the point: a document may genuinely say null. Only the
// unquoted JSON literal is a decline.
TEST(LensAbsentByOmission, QuotedNullIsAValue) {
    std::vector<std::string> body  = {"status ", "null", "\n"};
    std::vector<std::string> instr = {"\nEXTRACT", " json"};
    std::vector<std::string> gen   = {"{", "\"", "status", "\":\"", "null", "\"}"};
    LensRun run = build_run(body, instr, gen);
    cite(run, 3, 1, 0.9f);

    LensReport m = apply_absent_by_omission(compute_lens_report(run, fk()), {{"status", ""}});

    ASSERT_EQ(m.fields.size(), 1u);
    EXPECT_TRUE(m.fields[0].present) << "a quoted \"null\" is a value the document really states";
    EXPECT_EQ(m.fields[0].value, "null");
}

// ── A key outside the complete hint is dropped ────────────────────────────────
// The hint defines the importer-facing surface (Leg B: complete hint ⇒ 15/15 key
// stability at greedy). A stray key is not where data hides.
TEST(LensAbsentByOmission, DropsKeysOutsideTheHint) {
    LensReport rep;
    rep.document_text = "ACME GmbH";
    rep.fields = {present_field("customer", "ACME GmbH", true, true),
                  present_field("vendor_note", "ignore me", true, false)};
    std::vector<LensConcept> concepts = {{"customer", ""}};

    LensReport m = apply_absent_by_omission(rep, concepts);

    ASSERT_EQ(m.fields.size(), 1u);
    EXPECT_EQ(m.fields[0].key, "customer");
}

// ── NO safety net: an emitted value is REPORTED, never overruled ──────────────
// The presence gate re-marked an ungrounded, non-verbatim value as absent
// ("absence decided twice"). That died with the gate, on purpose: alone it would
// be the lens judging a value WRONG — a claim the format refuses ("No
// correctness"). The badges disclose it; the importer decides.
TEST(LensAbsentByOmission, UngroundedNonVerbatimValueIsReportedNotOverruled) {
    LensReport rep;
    rep.document_text = "ACME GmbH ordered 45 units";
    rep.fields = {present_field("customer", "Fabricated Ltd", /*grounded=*/false,
                                /*found=*/false)};
    std::vector<LensConcept> concepts = {{"customer", ""}};

    LensReport m = apply_absent_by_omission(rep, concepts);

    ASSERT_EQ(m.fields.size(), 1u);
    EXPECT_TRUE(m.fields[0].present) << "the lens must not judge a value wrong";
    EXPECT_EQ(m.fields[0].value, "Fabricated Ltd");
    EXPECT_FALSE(m.fields[0].grounded);          // disclosed...
    EXPECT_FALSE(m.fields[0].found_in_document); // ...twice
}

// ── The absent shape serializes value:null / badge:"absent", at v2 ────────────
TEST(LensAbsentByOmission, AbsentFieldSerializesNullValueAndAbsentBadge) {
    LensReport rep;
    rep.document_text = "ACME GmbH";
    rep.fields = {present_field("customer", "ACME GmbH", true, true)};
    std::vector<LensConcept> concepts = {{"customer", ""}, {"payment_terms", ""}};

    std::string json = lens_report_to_json(apply_absent_by_omission(rep, concepts));

    EXPECT_NE(json.find("\"format_version\":\"qemmi-lens/v2\""), std::string::npos);
    // present field: a string value + a real badge
    EXPECT_NE(json.find("\"value\":\"ACME GmbH\""), std::string::npos);
    EXPECT_NE(json.find("\"badge\":\"grounded\""), std::string::npos);
    // absent field: null value, absent badge, tier null
    EXPECT_NE(json.find("\"value\":null"), std::string::npos);
    EXPECT_NE(json.find("\"badge\":\"absent\""), std::string::npos);
    EXPECT_NE(json.find("\"tier\":null"), std::string::npos);
    // the v1 audit flag is GONE (subtractive ⇒ the v2 bump)
    EXPECT_EQ(json.find("presence_grounded"), std::string::npos);
}

// ════════════════════════════════════════════════════════════════════════════
// Stage 2 — the shape contract: tolerant on shape, LOUD on failure.
// (lens-format.md §"The shape contract"; replaces the grammar's false parse
// guarantee — measured 14/15 constrained vs 15/15 free.)
// ════════════════════════════════════════════════════════════════════════════

TEST(LensShapeContract, PlainObject) {
    size_t lo = 0, hi = 0;
    const std::string raw = "{\"a\":\"1\"}";
    ASSERT_TRUE(lens_find_json_object(raw, lo, hi));
    EXPECT_EQ(raw.substr(lo, hi - lo), "{\"a\":\"1\"}");
}

// Tolerant: a markdown fence is stripped. A grammar forecloses fences by
// construction; prompting does not — so the parser absorbs them.
TEST(LensShapeContract, StripsMarkdownFence) {
    size_t lo = 0, hi = 0;
    const std::string raw = "```json\n{\"a\":\"1\"}\n```";
    ASSERT_TRUE(lens_find_json_object(raw, lo, hi));
    EXPECT_EQ(raw.substr(lo, hi - lo), "{\"a\":\"1\"}");
}

// Tolerant: a prose preamble/postamble is skipped.
TEST(LensShapeContract, SkipsProseAroundTheObject) {
    size_t lo = 0, hi = 0;
    const std::string raw = "Sure! Here you go:\n{\"a\":\"1\"}\nHope that helps.";
    ASSERT_TRUE(lens_find_json_object(raw, lo, hi));
    EXPECT_EQ(raw.substr(lo, hi - lo), "{\"a\":\"1\"}");
}

// A brace INSIDE a string value must not close the object. The S1.4 corpus has a
// real case: an HTML mail quoting a JSON payload. A depth counter that ignored
// strings would end the span early and mis-locate every later byte offset — i.e.
// the lens would lie about where the model looked.
TEST(LensShapeContract, BraceInsideAStringDoesNotCloseTheObject) {
    size_t lo = 0, hi = 0;
    const std::string raw = "{\"note\":\"payload {\\\"sku\\\": 1} sent\",\"b\":\"2\"}";
    ASSERT_TRUE(lens_find_json_object(raw, lo, hi));
    EXPECT_EQ(raw.substr(lo, hi - lo), raw);  // the WHOLE object, not a prefix
}

TEST(LensShapeContract, NestedObjectClosesAtTheOutermost) {
    size_t lo = 0, hi = 0;
    const std::string raw = "{\"a\":{\"b\":\"1\"},\"c\":\"2\"}";
    ASSERT_TRUE(lens_find_json_object(raw, lo, hi));
    EXPECT_EQ(raw.substr(lo, hi - lo), raw);
}

// Loud: never closed (ran off the token budget mid-object) ⇒ no object.
TEST(LensShapeContract, UnterminatedObjectIsNotFound) {
    size_t lo = 0, hi = 0;
    EXPECT_FALSE(lens_find_json_object("{\"a\":\"unterminated", lo, hi));
}

TEST(LensShapeContract, NoObjectAtAll) {
    size_t lo = 0, hi = 0;
    EXPECT_FALSE(lens_find_json_object("I could not find anything.", lo, hi));
}

// ── compute_lens_report REFUSES unparseable output (⇒ 422), never partial ─────
TEST(LensShapeContract, ComputeThrowsUnparseableOnJunk) {
    std::vector<std::string> body  = {"date ", "2025-11-20", "\n"};
    std::vector<std::string> instr = {"\nEXTRACT", " json"};
    std::vector<std::string> gen   = {"I'm", " sorry", " I", " cannot"};  // no object
    LensRun run = build_run(body, instr, gen);
    EXPECT_THROW(compute_lens_report(run, fk()), LensUnparseableError);
}

// ── ...and the error names endpoint, expectation, actual, and carries `raw` ────
TEST(LensShapeContract, UnparseableErrorIsFailLoudAndCarriesRaw) {
    std::vector<std::string> body  = {"date ", "2025-11-20", "\n"};
    std::vector<std::string> instr = {"\nEXTRACT", " json"};
    std::vector<std::string> gen   = {"no", " json", " here"};
    LensRun run = build_run(body, instr, gen);
    try {
        compute_lens_report(run, fk());
        FAIL() << "expected LensUnparseableError";
    } catch (const LensUnparseableError& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("/v1/extract"), std::string::npos);  // endpoint
        EXPECT_NE(what.find("expected"), std::string::npos);     // expectation
        EXPECT_NE(what.find("actual"), std::string::npos);       // actual
        EXPECT_EQ(e.raw, "no json here");                        // inspectable
    }
}

// ── A FENCED response still locates value spans correctly ─────────────────────
// The offsets from the parsed object are object-relative; the gen-token math is
// gen_text-relative. If the shift is dropped, a fenced answer silently
// mis-attributes every citation. This is the regression that would make the lens
// lie, so it gets a gate.
TEST(LensShapeContract, FencedOutputStillLocatesTheValueSpan) {
    std::vector<std::string> body  = {"cust ", "ACME GmbH", "\n"};
    std::vector<std::string> instr = {"\nEXTRACT", " json"};
    // The fence tokens shift every byte offset in the object.
    std::vector<std::string> gen = {"```", "json", "\n", "{", "\"", "cust", "\":\"",
                                    "ACME GmbH", "\"}", "\n```"};
    LensRun run = build_run(body, instr, gen);
    cite(run, 6, 1, 0.9f);   // value token idx 7 -> step 6 -> body pos 1 ("ACME GmbH")

    LensReport r = compute_lens_report(run, fk());
    ASSERT_EQ(r.fields.size(), 1u);
    const LensField& f = r.fields[0];
    EXPECT_EQ(f.key, "cust");
    EXPECT_EQ(f.value, "ACME GmbH");
    EXPECT_TRUE(f.found_in_document);
    EXPECT_EQ(f.value_byte_lo, 5u);              // located in the DOCUMENT, not the fence
    EXPECT_EQ(count_confident_false_receipts(r), 0);
}

// ── Architecture guard ───────────────────────────────────────────────────────
// LensConstants is a set of coordinates measured on ONE model (Qwen 3.6,
// `qwen35moe`). Before this guard existed, --attention-lens was accepted on any
// loaded model and /v1/extract answered with a confidently-shaped report
// computed from constants that were never calibrated there. These tests pin the
// predicate the startup refusal is built on, and the fail-loud SHAPE of its
// message (parameter, then expected, then actual — in that order).
//
// The refusal itself lives in QweniumServerIntegration::enable_attention_lens()
// in src/server/http_server.cpp, which is a main()-local type in a binary
// target; it is not linkable from a unit test. The predicate and the message
// are shipped code in server_lens.h and are what that call site consists of.

TEST(LensArchitectureGuard, AcceptsOnlyTheCalibratedArchitecture) {
    EXPECT_STREQ(kLensCalibratedArchitecture, "qwen35moe");
    EXPECT_TRUE(lens_architecture_supported("qwen35moe"));

    // Every other architecture the engine hosts is uncalibrated, including the
    // near neighbour qwen35 (Qwen 3.5/3.8) — same family, different head layout.
    for (const char* arch : {"qwen2", "qwen3", "qwen35", "gemma1", "gemma2",
                             "gemma3", "gemma4", "gemma4uv", ""}) {
        EXPECT_FALSE(lens_architecture_supported(arch)) << "arch: " << arch;
    }
}

TEST(LensArchitectureGuard, RefusalNamesParameterExpectedActualInThatOrder) {
    const std::string msg = lens_architecture_refusal("qwen35");

    const size_t param    = msg.find("--attention-lens");
    const size_t expected = msg.find("qwen35moe");
    const size_t actual   = msg.find("'qwen35'");

    ASSERT_NE(param,    std::string::npos) << msg;
    ASSERT_NE(expected, std::string::npos) << msg;
    ASSERT_NE(actual,   std::string::npos) << msg;
    EXPECT_LT(param,    expected) << msg;   // parameter before expected
    EXPECT_LT(expected, actual)   << msg;   // expected before actual

    // It must say WHY, not just that it refused: an operator who is told only
    // "wrong architecture" will reasonably assume the lens is merely untested
    // here, rather than uncalibrated.
    EXPECT_NE(msg.find("calibrated"), std::string::npos) << msg;
}
