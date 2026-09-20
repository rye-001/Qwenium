// test_server_lens.cpp — Qemmi-Lens pure computation gate (docs/plan-qemmi-lens.md
// P2). compute_lens_report is model-free: it consumes one tapped run (tokens,
// byte maps, per-step kq_soft rows) and emits the lens report. These tests
// synthesize rows so the citation/coverage/badge numerics are checked without a
// model. The driver (run_lens_extract) is covered by the endpoint smoke.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cctype>
#include <set>
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
    EXPECT_NE(json.find("qemmi-lens/v4"), std::string::npos);  // v4: candidate set on the wire (v3 was repeated keys)
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

    EXPECT_NE(json.find("\"format_version\":\"qemmi-lens/v4\""), std::string::npos);
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

// ── Calibration guard ────────────────────────────────────────────────────────
// LensConstants is a set of coordinates measured on ONE model. Before this
// guard existed, --attention-lens was accepted on any loaded model and
// /v1/extract answered with a confidently-shaped report computed from constants
// that were never calibrated there. These tests pin the predicate the startup
// refusal is built on, and the fail-loud SHAPE of its message (parameter, then
// expected, then actual — in that order).
//
// The refusal itself lives in QweniumServerIntegration::enable_attention_lens()
// in src/server/http_server.cpp, which is a main()-local type in a binary
// target; it is not linkable from a unit test. The predicate and the message
// are shipped code in server_lens.h and are what that call site consists of.
//
// The key is {architecture, block_count}, and the second test below is the
// reason: an architecture string is a family, not a model.

TEST(LensCalibrationGuard, AcceptsTheCalibratedModelsWithTheirOwnCoordinates) {
    // Qwen 3.6-35B-A3B — the model the lens was built on. Two GGUF builds of one
    // base model (MTP build 41 = 40 + draft head; plain build 40), same 40-layer
    // decode stack, same calibration.
    for (uint32_t bc : {40u, 41u}) {
        const LensCalibration* c = lens_calibration_for("qwen35moe", bc, kLensAnyFileType);
        ASSERT_NE(c, nullptr) << "block_count: " << bc;
        EXPECT_EQ(c->constants.citation_layer, 3);    // L3H13
        EXPECT_EQ(c->constants.citation_head,  13);
    }

    // Qwen 3.8-9B — its own head is L27H13, not L3H13 (98% vs 84% top-3 on the
    // same messy corpus). This entry is the whole point of the table: before it,
    // the shipped lens ran the weaker head and refused the stronger model.
    const LensCalibration* q38 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q38, nullptr);
    EXPECT_EQ(q38->constants.citation_layer, 27);
    EXPECT_EQ(q38->constants.citation_head,  13);

    // Decided 2026-09-05: coverage stays 0.705 on every entry until a real
    // coverage-layer search runs. The recalibrated alternative was fitted on 23
    // spans — "recalibration is worth ~4 points" is the finding, not the value.
    for (const LensCalibration& c : lens_calibrations())
        EXPECT_DOUBLE_EQ(c.constants.coverage_used_peak, 0.705) << c.model;

    // Qwen 3.8-27B — L19H20, and the first model to clear every arm of leg C
    // (citation 91% top3, coverage 97% used-clear, 0/75 false alarms). Its head
    // is neither model's: L3H13 scores 7.1% here and the 9B's own selection
    // METHOD picks L11H22, which is rank 14 of 384 on real documents
    // (docs/note-lens-qwen38-27b-probe.md §3).
    const LensCalibration* q38_27 = lens_calibration_for("qwen35", 65, kLensAnyFileType);
    ASSERT_NE(q38_27, nullptr);
    EXPECT_EQ(q38_27->constants.citation_layer, 19);
    EXPECT_EQ(q38_27->constants.citation_head,  20);

    // The report's `model` field travels with the coordinates. It was a
    // hardcoded "Qwen3.6" literal until the table landed — which every Qwen 3.8
    // receipt would have carried, mislabelling the one thing the report is for.
    EXPECT_STREQ(lens_calibration_for("qwen35moe", 41, kLensAnyFileType)->constants.model_label,
                 "Qwen3.6 (attention lens)");
    EXPECT_STREQ(q38->constants.model_label,     "Qwen3.8-9B (attention lens)");
    EXPECT_STREQ(q38_27->constants.model_label,  "Qwen3.8-27B (attention lens)");

    // Every label must be DISTINCT, not merely non-empty. "Qwen3.8 (attention
    // lens)" was unambiguous until a second Qwen 3.8 was calibrated; two rows
    // sharing a label would put the same `model` string on reports computed
    // from different layers and heads, which is the exact failure the table was
    // built to end.
    std::set<std::string> labels;
    for (const LensCalibration& c : lens_calibrations())
        if (std::string(c.architecture) != "qwen35moe")   // the two 3.6 builds ARE one calibration
            EXPECT_TRUE(labels.insert(c.constants.model_label).second)
                << "duplicate model_label: " << c.constants.model_label;
    for (const LensCalibration& c : lens_calibrations())
        EXPECT_NE(std::string(c.constants.model_label), "") << c.model;
}

TEST(LensCalibrationGuard, RefusesUncalibratedModelsOfACalibratedArchitecture) {
    // THE hazard the key exists for. `qwen35` is a family, not a model
    // (src/models/qwen35.h): these four are all `qwen35`, none is calibrated,
    // and an architecture-keyed allowlist would have admitted every one of them
    // under Qwen 3.8-9B's coordinates.
    EXPECT_EQ(lens_calibration_for("qwen35", 24, kLensAnyFileType), nullptr);   // Qwen3.5-0.8B
    EXPECT_EQ(lens_calibration_for("qwen35", 32, kLensAnyFileType), nullptr);   // Qwen3.5-9B
    // Qwen3.6-27B is `qwen35`/64/ftype 15 and must stay refused at EVERY
    // file_type, including the sentinel — no row claims {qwen35, 64} today.
    EXPECT_EQ(lens_calibration_for("qwen35", 64, /*Q4_K_M*/ 15), nullptr);  // Qwen3.6-27B
    EXPECT_EQ(lens_calibration_for("qwen35", 64, kLensAnyFileType), nullptr);

    // Note 32 and 64 above: those are exactly the DECODE-STACK depths of the two
    // calibrated Qwen 3.8 builds (33 − 1 MTP, 65 − 1). Keying on decode depth
    // instead of raw block_count would have collided each of them with a
    // calibrated entry. This test is what pins that choice — and the hazard got
    // SHARPER, not milder, when Qwen3.8-27B (65) was calibrated 2026-09-15:
    // under a decode-depth key, Qwen3.6-27B (64 raw) would now be admitted under
    // Qwen3.8-27B's coordinates, which score 7.1% on the model that owns them
    // (docs/note-lens-qwen38-27b-probe.md §1). 64 above is that model.
    EXPECT_EQ(lens_calibration_for("qwen35moe", 39, kLensAnyFileType), nullptr);
    EXPECT_EQ(lens_calibration_for("qwen35moe", 42, kLensAnyFileType), nullptr);
}

// ── The quantization key field ───────────────────────────────────────────────
// No row pins a file_type today (the Ternary-Bonsai-27B row that motivated the
// field was reverted 2026-09-19 with the model). The field stays because the
// collision is not Bonsai-specific — a plain non-MTP Qwen3.8-27B build also
// keys as {qwen35, 64}. This test holds the invariant that matters while no
// row pins: a file_type that no row claims must RESOLVE THE SAME as any other,
// so the extra key component cannot change an answer it was never meant to.
TEST(LensCalibrationGuard, AnUnpinnedTableIgnoresTheFileType) {
    for (const LensCalibration& c : lens_calibrations()) {
        if (c.file_type != kLensAnyFileType) continue;   // pinned rows: other test
        for (uint32_t ft : {0u, 7u, 15u, 41u, 143u, kLensAnyFileType}) {
            const LensCalibration* got =
                lens_calibration_for(c.architecture, c.block_count, ft);
            ASSERT_NE(got, nullptr) << c.architecture << "/" << c.block_count
                                    << " must resolve at file_type " << ft;
            EXPECT_EQ(got, &c) << "file_type " << ft << " changed which row won";
        }
    }
}

// A row that PINS a quantization must beat a row that accepts any, otherwise
// table order would decide which coordinates a caller gets.
TEST(LensCalibrationGuard, APinnedRowWinsOverAnUnrestrictedOne) {
    for (const LensCalibration& c : lens_calibrations()) {
        if (c.file_type == kLensAnyFileType) continue;
        const LensCalibration* hit =
            lens_calibration_for(c.architecture, c.block_count, c.file_type);
        ASSERT_NE(hit, nullptr) << c.model;
        EXPECT_STREQ(hit->model, c.model)
            << "a pinned row was shadowed by an unrestricted one: " << c.model;
    }
}

TEST(LensCalibrationGuard, RefusesEveryOtherFamilyAtEveryDepth) {
    // Gemma is refused by MEASUREMENT, not neglect: a properly-run search over
    // all 768 candidate heads tops out at 63% against a 90% requirement
    // (docs/note-lens-gemma4-probe.md, docs/note-lens-gemma-norm-weighted.md).
    for (const char* arch : {"qwen2", "qwen3", "gemma1", "gemma2", "gemma3",
                             "gemma4", "gemma4uv", ""})
        for (uint32_t bc : {0u, 24u, 32u, 33u, 40u, 41u, 64u, 65u})
            EXPECT_EQ(lens_calibration_for(arch, bc, kLensAnyFileType), nullptr)
                << "arch: " << arch << " block_count: " << bc;
}

TEST(LensCalibrationGuard, RefusalNamesParameterExpectedActualInThatOrder) {
    const std::string msg = lens_calibration_refusal("qwen35", 64, /*Q4_K_M*/ 15);  // Qwen3.6-27B

    const size_t param    = msg.find("--attention-lens");
    const size_t expected = msg.find("qwen35moe");     // first listed entry
    const size_t actual   = msg.find("'qwen35'");

    ASSERT_NE(param,    std::string::npos) << msg;
    ASSERT_NE(expected, std::string::npos) << msg;
    ASSERT_NE(actual,   std::string::npos) << msg;
    EXPECT_LT(param,    expected) << msg;   // parameter before expected
    EXPECT_LT(expected, actual)   << msg;   // expected before actual

    // The actual value must carry the block count too, not just the arch —
    // otherwise the message reads as "wrong architecture" to an operator whose
    // architecture is in fact listed, and the real reason (wrong model of a
    // supported family) is invisible.
    EXPECT_NE(msg.find("64"), std::string::npos) << msg;

    // It must say WHY, not just that it refused: an operator who is told only
    // "not supported" will reasonably assume the lens is merely untested here,
    // rather than uncalibrated.
    EXPECT_NE(msg.find("calibrated"), std::string::npos) << msg;
}

// The startup banner prints the calibrated SET, not just the row that matched,
// and it must print the same set the refusal names — an operator on a
// calibrated model never sees the refusal, so a divergence between the two
// would be invisible until the day someone swaps the model. One builder is the
// mechanism; this pins it.
TEST(LensCalibrationGuard, TheAdvertisedSetIsEveryRowAndIsWhatTheRefusalNames) {
    const std::string list = lens_calibration_list();

    for (const LensCalibration& c : lens_calibrations()) {
        const std::string entry =
            std::string(c.architecture) + "/" + std::to_string(c.block_count) + "/" +
            (c.file_type == kLensAnyFileType ? std::string("any-quant")
                                             : std::string("ftype ") + std::to_string(c.file_type)) +
            " (" + c.model + ")";
        EXPECT_NE(list.find(entry), std::string::npos)
            << "banner list omits a calibrated row: " << entry << "\nlist: " << list;
    }

    // Same string in both places, by construction rather than by coincidence.
    EXPECT_NE(lens_calibration_refusal("gemma4", 30, kLensAnyFileType).find(list), std::string::npos) << list;
}

// ── Message attribution ──────────────────────────────────────────────────────
// The product claim the thread work landed on: every value reports WHICH message
// it was read from ("from message 23 of 24"). It is attribution and nothing more
// — no staleness verdict — because a later-message rule was measured to cry wolf
// on 7 of 9 correctly-handled corrections and stay silent on the real failure
// (docs/note-ss3-matched-pairs.md §3). These tests pin the attribution and the
// -1/empty shape a plain `document` request must still produce.
//
// Fixture body: "buy 45 now\nskip me\n". Splitting at byte 11 makes message 0 =
// "buy 45 now\n" and message 1 = "skip me\n"; the value "45" sits at byte 4.

TEST(LensMessageAttribution, CitationNamesTheMessageItLandedIn) {
    Fixture fx;
    fx.cite(5, /*head*/0, /*pos*/1, 0.9f);       // cite the document's "45"
    fx.run.message_offsets = {0, 11};

    LensReport r = compute_lens_report(fx.run, fx.k);

    EXPECT_EQ(r.n_messages, 2);
    ASSERT_EQ(r.fields.size(), 1u);
    const LensField& f = r.fields[0];
    ASSERT_FALSE(f.citations.empty());
    EXPECT_EQ(f.citations[0].byte_lo, 4u);       // "45"
    EXPECT_EQ(f.citations[0].message, 0);        // ...which is in message 0
    ASSERT_EQ(f.citation_messages.size(), 1u);
    EXPECT_EQ(f.citation_messages[0], 0);
}

TEST(LensMessageAttribution, ResolvesAByteInTheSecondMessage) {
    Fixture fx;
    // Attend to " me" (prompt position 5, byte 15) — past the boundary at 11.
    fx.cite(5, /*head*/0, /*pos*/5, 0.9f);
    fx.run.message_offsets = {0, 11};

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 1u);
    const LensField& f = r.fields[0];
    ASSERT_FALSE(f.citations.empty());
    EXPECT_EQ(f.citations[0].byte_lo, 15u);
    EXPECT_EQ(f.citations[0].message, 1);
}

TEST(LensMessageAttribution, TwoMessagesCoexistWithNoWinnerNamed) {
    Fixture fx;
    // The model looked at this key in BOTH messages — the conflict case. The
    // format reports both, ordered by mass, and names no winner (the CF1
    // non-claim): turn order does not identify which is current.
    fx.cite(5, /*head*/0, /*pos*/5, 0.6f);       // " me"  → message 1
    fx.cite(5, /*head*/0, /*pos*/1, 0.9f);       // "45"   → message 0
    fx.run.message_offsets = {0, 11};

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 1u);
    const LensField& f = r.fields[0];
    ASSERT_EQ(f.citation_messages.size(), 2u);
    EXPECT_EQ(f.citation_messages[0], 0);        // strongest citation first...
    EXPECT_EQ(f.citation_messages[1], 1);        // ...not "the winner"
}

TEST(LensMessageAttribution, PlainDocumentRequestReportsNoMessages) {
    Fixture fx;
    fx.cite(5, /*head*/0, /*pos*/1, 0.9f);
    // No message_offsets — a `document` request, the shape shipped since v1.

    LensReport r = compute_lens_report(fx.run, fx.k);

    EXPECT_EQ(r.n_messages, 0);
    ASSERT_EQ(r.fields.size(), 1u);
    const LensField& f = r.fields[0];
    ASSERT_FALSE(f.citations.empty());
    EXPECT_EQ(f.citations[0].message, -1);       // unknown, not "message 0"
    EXPECT_TRUE(f.citation_messages.empty());

    // ...and it serializes as null, so an importer cannot read a 0 that means
    // "we don't know" as "the first message".
    const std::string j = lens_report_to_json(r);
    EXPECT_NE(j.find("\"message\":null"), std::string::npos) << j;
    EXPECT_NE(j.find("\"n_messages\":0"), std::string::npos) << j;
}

TEST(LensMessageAttribution, SerializesMessageAndCitationMessages) {
    Fixture fx;
    fx.cite(5, /*head*/0, /*pos*/1, 0.9f);
    fx.run.message_offsets = {0, 11};

    const std::string j = lens_report_to_json(compute_lens_report(fx.run, fx.k));

    EXPECT_NE(j.find("\"n_messages\":2"), std::string::npos) << j;
    EXPECT_NE(j.find("\"message\":0"), std::string::npos) << j;
    EXPECT_NE(j.find("\"citation_messages\":[0]"), std::string::npos) << j;
}

// The report's own header must name the coordinates it actually ran, not a
// literal. Both lines were hardcoded "layer 3, head 13" / "layer 11" until
// per-model constants landed — a Qwen 3.8 report (L27H13) would have carried
// someone else's coordinates as its receipt.
TEST(LensMessageAttribution, ReportHeaderNamesTheCoordinatesItActuallyRan) {
    Fixture fx;
    fx.cite(5, /*head*/0, /*pos*/1, 0.9f);
    fx.k.citation_layer = 27;
    fx.k.citation_head  = 0;      // the fixture puts its signal on head 0
    fx.k.coverage_layer = 11;

    const std::string j = lens_report_to_json(compute_lens_report(fx.run, fx.k));

    EXPECT_NE(j.find("layer 27, head 0 (L27H0)"), std::string::npos) << j;
    EXPECT_NE(j.find("layer 11, max over heads"), std::string::npos) << j;
}

// ── Nested output: refused, loudly ───────────────────────────────────────────
// The lens emits and parses a FLAT object of scalar values. Nothing stops a free
// decode emitting nested JSON, and before 2026-09-05 the parse did not refuse it
// — it silently produced wrong fields by two independent mechanisms, both
// measured as characterization tests before the fix and preserved below as the
// refusal cases they became:
//
//   lens_keys   (server_lens.cpp:60) is DEPTH-BLIND — it collects every "..."
//               followed by ':' at any nesting depth, so keys inside an array
//               element surfaced as top-level fields.
//   lens_value_of (:94) is SCALAR-ONLY — a value not starting with '"' was read
//               to the first ',' '}' or newline, so an array yielded a fragment
//               (`[{"desc": "Widget"`) shipped with a real badge and citations;
//               and it resolves a key by the FIRST occurrence of "key", so a
//               nested "quantity":"5" answered a top-level "quantity":"45".
//
// The second was the serious one: a wrong value, normally badged, with citations
// pointing at a span the model did not read — the lens misreporting where the
// model looked, which lens-format.md calls the one thing it must never do.
//
// Refusal (422) is the answer rather than a summary or a dropped key, because
// the lens claims a faithful record and cannot make that claim about a value it
// did not read. This is NOT a ruling that arrays are unsupportable: the trust
// math is per-value-span and generalizes to leaf scalars at any depth. What is
// missing is a hint form for repeating groups, which is a Leg-B-style
// measurement, not a parser change.

namespace {
// Drive the parse with an arbitrary emitted string: one token per byte, so the
// gen-token span math is exact and the citation rows stay inert.
void set_gen(Fixture& fx, const std::string& text) {
    fx.run.gen_text = text;
    fx.run.gen_tok_text.clear();
    for (char c : text) fx.run.gen_tok_text.push_back(std::string(1, c));
    const size_t G = fx.run.gen_tok_text.size();
    fx.run.gen_cum.assign(G + 1, 0);
    for (size_t i = 0; i < G; ++i)
        fx.run.gen_cum[i + 1] = fx.run.gen_cum[i] + fx.run.gen_tok_text[i].size();
    fx.run.steps.assign(G, LensStep{});
    for (auto& st : fx.run.steps) {
        st.n_kv = Fixture::N_KV;
        st.citation_row.assign((size_t)Fixture::N_HEAD * Fixture::N_KV, 0.0f);
        st.coverage_row.assign((size_t)Fixture::N_HEAD * Fixture::N_KV, 0.0f);
    }
}
const LensField* field_named(const LensReport& r, const std::string& key) {
    for (const LensField& f : r.fields) if (f.key == key) return &f;
    return nullptr;
}
}  // namespace

TEST(LensNestedOutput, ArrayValueIsRefusedNotTruncated) {
    Fixture fx;
    set_gen(fx, R"({"line_items": [{"desc": "Widget", "qty": "5"}]})");

    // Was: returned a field whose value was the fragment `[{"desc": "Widget"`.
    EXPECT_THROW(compute_lens_report(fx.run, fx.k), LensUnparseableError);
}

TEST(LensNestedOutput, ObjectValueIsRefused) {
    Fixture fx;
    set_gen(fx, R"({"customer": {"name": "Acme Ltd"}, "quantity": "45"})");

    EXPECT_THROW(compute_lens_report(fx.run, fx.k), LensUnparseableError);
}

TEST(LensNestedOutput, RefusalNamesTheKeyAndCarriesTheRawOutput) {
    Fixture fx;
    const std::string emitted = R"({"line_items": [{"desc": "Widget"}]})";
    set_gen(fx, emitted);

    try {
        compute_lens_report(fx.run, fx.k);
        FAIL() << "expected LensUnparseableError";
    } catch (const LensUnparseableError& e) {
        const std::string msg = e.what();
        // Fail-loud contract order: parameter, expected, actual.
        const size_t param    = msg.find("line_items");
        const size_t expected = msg.find("expected a scalar");
        const size_t actual   = msg.find("actual=");
        ASSERT_NE(param,    std::string::npos) << msg;
        ASSERT_NE(expected, std::string::npos) << msg;
        ASSERT_NE(actual,   std::string::npos) << msg;
        EXPECT_LT(expected, actual) << msg;
        // The 422 body carries exactly what the model emitted, so the failure is
        // inspectable rather than merely reported.
        EXPECT_EQ(e.raw, emitted);
    }
}

TEST(LensNestedOutput, ANestedKeyCanNoLongerAnswerATopLevelLookup) {
    Fixture fx;
    // The document's real answer is 45, and the model emitted 45 at the top
    // level. Before the guard this reported quantity="5" — the nested value —
    // with a normal badge and citations pointing at the wrong span.
    set_gen(fx, R"({"items": [{"quantity": "5"}], "quantity": "45"})");

    EXPECT_THROW(compute_lens_report(fx.run, fx.k), LensUnparseableError);
}

TEST(LensNestedOutput, FlatOutputIsUnaffected) {
    // The control: the shipped flat shape parses exactly as before. A brace or
    // bracket INSIDE a string value is a value, not nesting, and must still pass.
    Fixture fx;
    set_gen(fx, R"({"customer": "Acme [Holdings] Ltd", "quantity": "45"})");

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 2u);
    EXPECT_EQ(field_named(r, "customer")->value, "Acme [Holdings] Ltd");
    EXPECT_EQ(field_named(r, "quantity")->value, "45");
}

TEST(LensNestedOutput, ScalarNonStringValuesStillParse) {
    // Numbers, booleans and null are scalars, not nesting — the guard must not
    // catch them. null stays "the model declined" (absent), not a refusal.
    Fixture fx;
    set_gen(fx, R"({"quantity": 45, "total": null})");

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 2u);
    EXPECT_EQ(field_named(r, "quantity")->value, "45");
    EXPECT_TRUE(field_named(r, "total")->value.empty());   // declined ⇒ absent
}

// ── Repeated keys: every occurrence is reported ──────────────────────────────
// Found by the multi-line invoice probe (2026-09-05). Asked for a flat schema on
// an invoice with three line items, the model emitted all three as REPEATED
// top-level keys. The document was not the problem and the model was not wrong;
// the requested shape simply could not hold what the document said.
//
// Before the fix, two readers disagreed and two thirds of the answer vanished:
// lens_value_of resolved by the FIRST occurrence, so all three line_item fields
// came back as the first item; apply_absent_by_omission then kept only one of
// them. Nothing marked the loss — no error, no badge, and coverage stayed silent
// (measured: the un-extracted lines were NOT flagged as un-consulted). A
// confident, correctly-grounded, silently incomplete answer.
//
// Now each occurrence carries its own value, its own byte span and therefore its
// own citations. What this does NOT do is pair them into records: occurrence
// order is emission order, and line_item[1] is not claimed to belong with
// quantity[1]. Grouping is a leaf-path design and is deliberately not this.

TEST(LensRepeatedKeys, EveryOccurrenceGetsItsOwnValue) {
    Fixture fx;
    set_gen(fx, R"({"line_item": "Bracket", "quantity": "7", )"
                R"("line_item": "Fan", "quantity": "19", )"
                R"("line_item": "Pipe", "quantity": "43"})");

    LensReport r = compute_lens_report(fx.run, fx.k);

    std::vector<std::string> items, qtys;
    for (const LensField& f : r.fields) {
        if (f.key == "line_item") items.push_back(f.value);
        if (f.key == "quantity")  qtys.push_back(f.value);
    }
    EXPECT_EQ(items, (std::vector<std::string>{"Bracket", "Fan", "Pipe"}));
    EXPECT_EQ(qtys,  (std::vector<std::string>{"7", "19", "43"}));
}

TEST(LensRepeatedKeys, OccurrenceIndexIsEmissionOrder) {
    Fixture fx;
    set_gen(fx, R"({"line_item": "Bracket", "line_item": "Fan", "line_item": "Pipe"})");

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 3u);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(r.fields[i].occurrence, i);
    EXPECT_EQ(r.fields[2].value, "Pipe");
}

TEST(LensRepeatedKeys, OrderingPassKeepsAllOccurrencesUnderTheirConcept) {
    Fixture fx;
    set_gen(fx, R"({"line_item": "Bracket", "line_item": "Fan", "customer": "Acme Ltd"})");

    LensReport r = apply_absent_by_omission(
        compute_lens_report(fx.run, fx.k),
        {{"customer", ""}, {"line_item", ""}, {"total", ""}});

    // Concept order still governs: customer, then BOTH line_items, then absent total.
    ASSERT_EQ(r.fields.size(), 4u);
    EXPECT_EQ(r.fields[0].key, "customer");
    EXPECT_EQ(r.fields[1].key, "line_item");
    EXPECT_EQ(r.fields[1].value, "Bracket");
    EXPECT_EQ(r.fields[2].key, "line_item");
    EXPECT_EQ(r.fields[2].value, "Fan");
    EXPECT_EQ(r.fields[3].key, "total");
    EXPECT_FALSE(r.fields[3].present);          // absent-by-omission still works
}

TEST(LensRepeatedKeys, FirstEntryPerKeyIsUnchangedForOldImporters) {
    // The back-compat property that makes this survivable: an importer taking the
    // FIRST match for a key gets exactly what v2 gave it. Later occurrences are
    // appended after, never inserted before. (An importer doing LAST-wins does
    // change behaviour — which is precisely why the format bumps to v3.)
    Fixture fx;
    set_gen(fx, R"({"quantity": "7", "quantity": "43"})");

    LensReport r = apply_absent_by_omission(compute_lens_report(fx.run, fx.k),
                                            {{"quantity", ""}});

    ASSERT_EQ(r.fields.size(), 2u);
    EXPECT_EQ(r.fields[0].value, "7");          // what v2 reported
    EXPECT_EQ(r.fields[1].value, "43");         // what v2 threw away
}

TEST(LensRepeatedKeys, SingleOccurrenceIsUnaffected) {
    // The control: the ordinary flat document is byte-for-byte the same shape.
    Fixture fx;
    set_gen(fx, R"({"customer": "Acme Ltd", "quantity": "45"})");

    LensReport r = apply_absent_by_omission(compute_lens_report(fx.run, fx.k),
                                            {{"customer", ""}, {"quantity", ""}});

    ASSERT_EQ(r.fields.size(), 2u);
    EXPECT_EQ(r.fields[0].value, "Acme Ltd");
    EXPECT_EQ(r.fields[1].value, "45");
    EXPECT_EQ(r.fields[0].occurrence, 0);
    EXPECT_EQ(r.fields[1].occurrence, 0);
}

TEST(LensRepeatedKeys, VersionIsBumpedAndOccurrenceIsSerialized) {
    Fixture fx;
    set_gen(fx, R"({"quantity": "7", "quantity": "43"})");

    const std::string j = lens_report_to_json(compute_lens_report(fx.run, fx.k));

    EXPECT_NE(j.find("qemmi-lens/v4"), std::string::npos) << j;
    EXPECT_NE(j.find("\"occurrence\":1"), std::string::npos) << j;
}

// ── Arrays of scalars: one occurrence per element ────────────────────────────
// The same answer, two encodings. Asked for a flat schema on a three-line
// invoice, Qwen3.8-9B repeats the key ("quantity":"7", ... "quantity":"43")
// while Qwen3.6-35B answers "quantity": [7, 19, 43]. Both are the model doing
// the right thing with a shape that cannot hold three items; refusing one and
// serving the other would be an accident of which model you loaded.
//
// A scalar element is LOCATABLE — it has its own byte span — so the
// per-value-span trust math gives it its own real citations, exactly as a
// repeated key gets. That is the whole test for whether a shape is servable, and
// it is why an array containing OBJECTS stays refused: there is no scalar span
// to cite, and lens_keys would leak the inner keys as top-level fields.

TEST(LensScalarArrays, StringArrayBecomesOneOccurrencePerElement) {
    Fixture fx;
    set_gen(fx, R"({"line_item": ["Bracket", "Fan", "Pipe"]})");

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 3u);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(r.fields[i].occurrence, i);
    EXPECT_EQ(r.fields[0].value, "Bracket");
    EXPECT_EQ(r.fields[1].value, "Fan");
    EXPECT_EQ(r.fields[2].value, "Pipe");
}

TEST(LensScalarArrays, NumberArrayBecomesOneOccurrencePerElement) {
    Fixture fx;
    set_gen(fx, R"({"quantity": [7, 19, 43]})");

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 3u);
    EXPECT_EQ(r.fields[0].value, "7");
    EXPECT_EQ(r.fields[1].value, "19");
    EXPECT_EQ(r.fields[2].value, "43");
}

TEST(LensScalarArrays, EachElementGetsItsOwnGenSpan) {
    // The load-bearing property: elements must not share a span, or all three
    // would carry identical citations and the receipts would be a fiction.
    Fixture fx;
    set_gen(fx, R"({"quantity": [7, 19, 43]})");

    LensReport r = compute_lens_report(fx.run, fx.k);

    ASSERT_EQ(r.fields.size(), 3u);
    EXPECT_LT(r.fields[0].gen_lo, r.fields[1].gen_lo);
    EXPECT_LT(r.fields[1].gen_lo, r.fields[2].gen_lo);
    EXPECT_NE(r.fields[0].gen_hi, r.fields[1].gen_hi);
}

TEST(LensScalarArrays, AnArrayHoldingAnObjectIsStillRefused) {
    Fixture fx;
    set_gen(fx, R"({"line_item": [{"desc": "Bracket"}, {"desc": "Fan"}]})");

    // No scalar span to cite, and lens_keys would leak "desc" as a top-level
    // field. Refused, exactly as before this change.
    EXPECT_THROW(compute_lens_report(fx.run, fx.k), LensUnparseableError);
}

TEST(LensScalarArrays, EmptyArrayAndNullElementsDecline) {
    Fixture fx;
    set_gen(fx, R"({"quantity": [], "total": [null], "tax": "170.50"})");

    LensReport r = apply_absent_by_omission(
        compute_lens_report(fx.run, fx.k),
        {{"quantity", ""}, {"total", ""}, {"tax", ""}});

    ASSERT_EQ(r.fields.size(), 3u);
    EXPECT_FALSE(r.fields[0].present);          // [] ⇒ absent, never a blank value
    EXPECT_FALSE(r.fields[1].present);          // [null] ⇒ absent
    EXPECT_EQ(r.fields[2].value, "170.50");     // the scalar beside them is intact
}

TEST(LensScalarArrays, RefusalMessageNamesBothAcceptedShapes) {
    Fixture fx;
    set_gen(fx, R"({"line_item": [{"desc": "Bracket"}]})");
    try {
        compute_lens_report(fx.run, fx.k);
        FAIL() << "expected LensUnparseableError";
    } catch (const LensUnparseableError& e) {
        const std::string msg = e.what();
        // An operator reading this must learn that arrays are not banned — only
        // ones we cannot cite. Naming only "scalar" would send them to redesign
        // a prompt that already works.
        EXPECT_NE(msg.find("array of scalars"), std::string::npos) << msg;
        EXPECT_NE(msg.find("line_item"), std::string::npos) << msg;
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Candidate set — pass 2 (docs/plan-candidate-set.md). Both functions under
// test are PURE (no engine, no model): lens_parse_pass2_candidates is the
// tolerant line parser ported from the CAND=1 probe, and
// lens_apply_pass2_candidates layers the byte-exactness check, dedup,
// document-order sort, and producer-failure detection on top. The cold
// taps-disarmed decode that PRODUCES the raw text (run_cand_pass2_decode) is
// not model-free and is not exercised here — see the header comment on
// run_lens_extract for what remains untestable without a model.
// ═════════════════════════════════════════════════════════════════════════

TEST(LensCandidatePass2Parse, QuotedInputParsesOneSpanPerKey) {
    const std::vector<std::string> keys = {"monthly_rent", "supplier"};
    auto out = lens_parse_pass2_candidates(
        "monthly_rent: \"1,450.00 GBP\"\nsupplier: \"Acme Ltd\"\n", keys);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].first, "monthly_rent");
    EXPECT_EQ(out[0].second, "1,450.00 GBP");
    EXPECT_EQ(out[1].first, "supplier");
    EXPECT_EQ(out[1].second, "Acme Ltd");
}

TEST(LensCandidatePass2Parse, UnquotedInputStillParses) {
    // m_en1 in the probe corpus emitted every line unquoted; the strict
    // key: "span"-only parser dropped the lot. This is the regression it fixes.
    const std::vector<std::string> keys = {"monthly_rent"};
    auto out = lens_parse_pass2_candidates("monthly_rent: 1,450.00 GBP\n", keys);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].first, "monthly_rent");
    EXPECT_EQ(out[0].second, "1,450.00 GBP");
}

TEST(LensCandidatePass2Parse, MixedQuotedAndUnquotedLinesBothParse) {
    const std::vector<std::string> keys = {"monthly_rent", "supplier"};
    auto out = lens_parse_pass2_candidates(
        "monthly_rent: \"1,450.00 GBP\"\nsupplier: Acme Ltd\n", keys);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].second, "1,450.00 GBP");
    EXPECT_EQ(out[1].second, "Acme Ltd");
}

TEST(LensCandidatePass2Parse, NoneIsARealAnswerNotAMalformedLine) {
    const std::vector<std::string> keys = {"warranty_period"};
    auto out = lens_parse_pass2_candidates("warranty_period: (none)\n", keys);
    EXPECT_TRUE(out.empty());   // a real answer (no candidate) — not dropped as malformed
}

TEST(LensCandidatePass2Parse, UnterminatedQuoteIsMalformedAndSkipped) {
    const std::vector<std::string> keys = {"monthly_rent", "supplier"};
    auto out = lens_parse_pass2_candidates(
        "monthly_rent: \"1,450.00 GBP\nsupplier: \"Acme Ltd\"\n", keys);
    // The first line's quote never closes (it runs into the next line) — malformed,
    // skipped. Only the second, well-formed line survives.
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].first, "supplier");
    EXPECT_EQ(out[0].second, "Acme Ltd");
}

TEST(LensCandidatePass2Parse, UnknownKeyIsIgnored) {
    const std::vector<std::string> keys = {"monthly_rent"};
    auto out = lens_parse_pass2_candidates(
        "monthly_rent: \"1,450.00 GBP\"\nrandom_prose: not a key\n", keys);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].first, "monthly_rent");
}

TEST(LensCandidatePass2Apply, NonVerbatimSpanIsDroppedNotFlagged) {
    // "1450.00 GBP" (no comma, no period) never appears in the document —
    // a paraphrase, not a byte-exact slice. Must be dropped silently, not
    // reported as a producer failure (other keys parsed fine).
    const std::string document = "Clause 2: Rent is 1,450.00 GBP per month.";
    const std::vector<std::string> keys = {"monthly_rent"};
    LensReport r;
    lens_apply_pass2_candidates(document, "monthly_rent: \"1450.00 GBP\"\n", keys, r);
    EXPECT_FALSE(r.candidates_producer_failed);
    EXPECT_TRUE(r.key_candidates["monthly_rent"].empty());
}

TEST(LensCandidatePass2Apply, ByteExactSpanIsKeptWithCorrectOffsets) {
    const std::string document = "Clause 2: Rent is 1,450.00 GBP per month.";
    const std::vector<std::string> keys = {"monthly_rent"};
    LensReport r;
    lens_apply_pass2_candidates(document, "monthly_rent: \"1,450.00 GBP\"\n", keys, r);
    ASSERT_EQ(r.key_candidates["monthly_rent"].size(), 1u);
    const LensCandidate& c = r.key_candidates["monthly_rent"][0];
    EXPECT_EQ(c.value, "1,450.00 GBP");
    EXPECT_EQ(document.substr(c.byte_lo, c.byte_hi - c.byte_lo), "1,450.00 GBP");
}

TEST(LensCandidatePass2Apply, CandidatesSortIntoDocumentOrder) {
    const std::string document = "Amendment: 1,375.00 GBP. Clause 2: 1,450.00 GBP.";
    const std::vector<std::string> keys = {"monthly_rent"};
    LensReport r;
    // Emitted in the OPPOSITE order from how they appear in the document.
    lens_apply_pass2_candidates(
        document,
        "monthly_rent: \"1,450.00 GBP\"\nmonthly_rent: \"1,375.00 GBP\"\n", keys, r);
    ASSERT_EQ(r.key_candidates["monthly_rent"].size(), 2u);
    EXPECT_EQ(r.key_candidates["monthly_rent"][0].value, "1,375.00 GBP");  // earlier byte_lo
    EXPECT_EQ(r.key_candidates["monthly_rent"][1].value, "1,450.00 GBP");
}

TEST(LensCandidatePass2Apply, DuplicateSpanIsNotDoubleCounted) {
    const std::string document = "Rent is 1,450.00 GBP per month.";
    const std::vector<std::string> keys = {"monthly_rent"};
    LensReport r;
    lens_apply_pass2_candidates(
        document,
        "monthly_rent: \"1,450.00 GBP\"\nmonthly_rent: \"1,450.00 GBP\"\n", keys, r);
    EXPECT_EQ(r.key_candidates["monthly_rent"].size(), 1u);
}

TEST(LensCandidatePass2Apply, ProducerFailureIsDistinguishedFromEmptyDocument) {
    // Non-empty generation, zero parseable lines — the fourth state
    // (docs/plan-candidate-set.md): a PRODUCER failure, not "the document
    // offers nothing". Must never render as an empty, unflagged key_candidates.
    const std::string document = "Rent is 1,450.00 GBP per month.";
    const std::vector<std::string> keys = {"monthly_rent"};
    LensReport r;
    lens_apply_pass2_candidates(document, "I cannot help with that request.", keys, r);
    EXPECT_TRUE(r.candidates_producer_failed);
    EXPECT_FALSE(r.candidates_error.empty());
    EXPECT_TRUE(r.key_candidates.empty());
}

TEST(LensCandidatePass2Apply, NoneAnswerCoexistsWithRealCandidatesWithoutProducerFailure) {
    // "(none)" on one key alongside a real span on another: the (none) line
    // must not poison the whole document into a producer failure, and the
    // real candidate on the other key must still land.
    const std::string document = "Rent is 1,450.00 GBP per month.";
    const std::vector<std::string> keys = {"monthly_rent", "warranty_period"};
    LensReport r;
    lens_apply_pass2_candidates(
        document, "monthly_rent: \"1,450.00 GBP\"\nwarranty_period: (none)\n", keys, r);
    EXPECT_FALSE(r.candidates_producer_failed);
    EXPECT_TRUE(r.candidates_error.empty());
    ASSERT_EQ(r.key_candidates["monthly_rent"].size(), 1u);
    EXPECT_EQ(r.key_candidates["monthly_rent"][0].value, "1,450.00 GBP");
    EXPECT_TRUE(r.key_candidates["warranty_period"].empty());
}

TEST(LensCandidatePass2Apply, EmptyGenerationIsNotAProducerFailure) {
    // No output at all (e.g. immediate EOS) is a different fact than non-empty
    // garbage — only the latter is a producer failure.
    const std::string document = "Rent is 1,450.00 GBP per month.";
    const std::vector<std::string> keys = {"monthly_rent"};
    LensReport r;
    lens_apply_pass2_candidates(document, "", keys, r);
    EXPECT_FALSE(r.candidates_producer_failed);
}

// ═════════════════════════════════════════════════════════════════════════
// Candidate set — the WIRE (docs/plan-candidate-set.md, qemmi-lens/v4).
// lens_report_to_json is pure (no engine, no model): these tests build a
// LensReport by hand (fields + key_candidates + the two producer-outcome
// flags) and assert on the parsed JSON, exactly like the fixtures above.
// ═════════════════════════════════════════════════════════════════════════

// ── Question vocabulary (docs/plan-question-keys.md) ─────────────────────────
TEST(LensQuestionVocab, KeyModeInstructionIsUnchanged) {
    // The regime Stage 1 measured must stay byte-identical when no question is
    // asked — this is the guard on "additive".
    const std::vector<std::string> keys = {"customer", "delivery_date"};
    EXPECT_EQ(lens_build_instruction(keys),
              "\n\nExtract the following fields from the document above into a flat "
              "JSON object of \"key\": \"value\" pairs, using exactly these "
              "snake_case keys: customer, delivery_date"
              ". Copy each value verbatim from the document. Output ONLY the JSON "
              "object, nothing else.");
}

TEST(LensQuestionVocab, QuestionInstructionKeysByIdAndDemandsVerbatim) {
    const std::vector<std::string> ids = {"customer", "delivery_date"};
    const std::vector<std::string> qs  = {"Who is the customer?",
                                          "What is the delivery date?"};
    const std::string instr = lens_build_question_instruction(ids, qs);
    // The id is the join key, and the question rides beside it.
    EXPECT_NE(instr.find("\"customer\": Who is the customer?"), std::string::npos);
    EXPECT_NE(instr.find("\"delivery_date\": What is the delivery date?"), std::string::npos);
    // The verbatim demand is load-bearing: 97% extractiveness was measured under
    // an instruction that asks for a span.
    EXPECT_NE(instr.find("verbatim"), std::string::npos);
    EXPECT_NE(instr.find("Do not paraphrase"), std::string::npos);
    // Absence stays earned by omission, same as key mode.
    EXPECT_NE(instr.find("omit that key"), std::string::npos);
}

TEST(LensQuestionWire, KeyModeEmitsNeitherNewMember) {
    LensReport r;
    r.fields = {present_field("customer", "ACME GmbH", true, true)};
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_FALSE(j.contains("vocabulary_mode"));
    EXPECT_FALSE(j.contains("uncalibrated"));
    EXPECT_EQ(j["format_version"], "qemmi-lens/v4");   // additive, no bump
}

TEST(LensQuestionWire, QuestionModeDisclosesWhatWasNotCalibrated) {
    // The probe measured the CITATION head under questions and nothing else.
    // badge and coverage run on identifier-extraction coordinates, and the
    // payload must say so rather than print a receipt it cannot back.
    LensReport r;
    r.question_vocabulary = true;
    r.fields = {present_field("customer", "ACME GmbH", true, true)};
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_EQ(j["vocabulary_mode"], "questions");
    ASSERT_TRUE(j["uncalibrated"].is_array());
    std::vector<std::string> u = j["uncalibrated"].get<std::vector<std::string>>();
    EXPECT_EQ(u, (std::vector<std::string>{"badge", "coverage"}));
}

// ── Warm document (docs/plan-lens-warm-document.md) ──────────────────────────
TEST(LensWarmWire, ColdOmitsThePrefixMember) {
    LensReport r;
    r.fields = {present_field("customer", "ACME GmbH", true, true)};
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_FALSE(j.contains("prefix"))
        << "a caller that never sends document_id must get a byte-identical payload";
}

TEST(LensWarmWire, WarmDisclosesItself) {
    // A warm citation mass must never be compared against a cold one without
    // the reader knowing which is which.
    LensReport r;
    r.prefix_warm = true;
    r.fields = {present_field("customer", "ACME GmbH", true, true)};
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_EQ(j["prefix"], "warm");
}

TEST(LensWarmDocumentState, InvalidateClearsEveryField) {
    LensWarmDocument w;
    w.document_id = "lease-4471"; w.document_hash = 99; w.prefix_tokens = 1234; w.valid = true;
    w.invalidate();
    EXPECT_TRUE(w.document_id.empty());
    EXPECT_EQ(w.document_hash, 0u);
    EXPECT_EQ(w.prefix_tokens, 0u);
    EXPECT_FALSE(w.valid);
}

TEST(LensWarmDocumentState, ReuseRequiresIdAndHashAndValidity) {
    // Mirrors the predicate in run_lens_extract. The hash is the interlock that
    // stops a primed prefix being served for different text — which would report
    // receipts about a document the model never read.
    const size_t h = std::hash<std::string>{}(std::string("the document"));
    LensWarmDocument w;
    w.document_id = "doc-a"; w.document_hash = h; w.prefix_tokens = 10; w.valid = true;

    auto reuse = [&](const std::string& id, size_t hash) {
        return w.valid && !id.empty() && w.document_id == id && w.document_hash == hash;
    };
    EXPECT_TRUE (reuse("doc-a", h));            // same id, same bytes
    EXPECT_FALSE(reuse("doc-a", h + 1));        // SAME id, DIFFERENT document
    EXPECT_FALSE(reuse("doc-b", h));            // different id
    EXPECT_FALSE(reuse("", h));                 // no id ⇒ never warm
    w.valid = false;
    EXPECT_FALSE(reuse("doc-a", h));            // nothing held
}

TEST(LensCandidateWire, FormatVersionIsV4) {
    LensReport r;
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_EQ(j["format_version"], "qemmi-lens/v4");
}

TEST(LensCandidateWire, EveryVocabularyKeyGetsAnEntryIncludingEmpty) {
    LensReport r;
    r.document_text = "Rent is 1,450.00 GBP. Pets: none allowed.";
    r.fields = {present_field("monthly_rent", "1,450.00 GBP", true, true),
                present_field("pets_policy", "none allowed", true, true)};
    r.candidates_requested = true;
    r.key_candidates["monthly_rent"] = {LensCandidate{"1,450.00 GBP", 9, 21}};
    // "pets_policy" is deliberately absent from the map — a key pass 2 found
    // nothing for. The wire must still carry it, as [].

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    ASSERT_TRUE(j.contains("key_candidates"));
    ASSERT_TRUE(j["key_candidates"].contains("monthly_rent"));
    ASSERT_TRUE(j["key_candidates"].contains("pets_policy"));
    EXPECT_EQ(j["key_candidates"]["monthly_rent"].size(), 1u);
    EXPECT_TRUE(j["key_candidates"]["pets_policy"].is_array());
    EXPECT_EQ(j["key_candidates"]["pets_policy"].size(), 0u);
}

TEST(LensCandidateWire, ArrayOrderIsDocumentOrder) {
    LensReport r;
    r.document_text = "Amendment: 1,375.00 GBP. Clause 2: 1,450.00 GBP.";
    r.fields = {present_field("monthly_rent", "1,450.00 GBP", true, true)};
    r.candidates_requested = true;
    // Already producer-sorted by byte_lo ascending — the wire must preserve
    // this, not re-rank by mass or any other criterion (CF1: no verdicts).
    r.key_candidates["monthly_rent"] = {LensCandidate{"1,375.00 GBP", 11, 24},
                                       LensCandidate{"1,450.00 GBP", 36, 49}};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    auto& cands = j["key_candidates"]["monthly_rent"];
    ASSERT_EQ(cands.size(), 2u);
    EXPECT_EQ(cands[0]["value"], "1,375.00 GBP");
    EXPECT_EQ(cands[1]["value"], "1,450.00 GBP");
    EXPECT_LT((int)cands[0]["byte_lo"], (int)cands[1]["byte_lo"]);
}

TEST(LensCandidateWire, ReturnedAsLinksAWiderCandidateToTheNarrowerFieldValue) {
    LensReport r;
    r.document_text = "ACME ordered 45 units on Monday.";
    LensField f = present_field("quantity", "45", true, true);
    f.occurrence = 0;
    r.fields = {f};
    r.candidates_requested = true;
    r.key_candidates["quantity"] = {LensCandidate{"45 units", 13, 21}};  // wider

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_EQ(j["key_candidates"]["quantity"][0]["returned_as"], 0);
}

TEST(LensCandidateWire, ReturnedAsLinksANarrowerCandidateToTheWiderFieldValue) {
    LensReport r;
    r.document_text = "Ship 300 x 45 units total.";
    LensField f = present_field("shipment", "300 x 45 units", true, true);
    f.occurrence = 0;
    r.fields = {f};
    r.candidates_requested = true;
    r.key_candidates["shipment"] = {LensCandidate{"300 x", 5, 10}};  // narrower

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_EQ(j["key_candidates"]["shipment"][0]["returned_as"], 0);
}

TEST(LensCandidateWire, TiebreakPrefersTightestContainmentThenEarliestByteLo) {
    LensReport r;
    r.document_text = "Clause A: 45. Later, clause B says 45 units again.";
    LensField f = present_field("qty", "45", true, true);
    f.occurrence = 0;
    r.fields = {f};
    r.candidates_requested = true;
    // Both candidates contain "45"; the exact-length match (gap 0) must win
    // over the wider one (gap 6), regardless of array position.
    r.key_candidates["qty"] = {LensCandidate{"45 units", 36, 44},
                              LensCandidate{"45", 10, 12}};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    auto& cands = j["key_candidates"]["qty"];
    ASSERT_EQ(cands.size(), 2u);
    // cands[1] is the exact "45" (byte_lo 10) — the tight match.
    EXPECT_EQ(cands[1]["value"], "45");
    EXPECT_EQ(cands[1]["returned_as"], 0);
    EXPECT_TRUE(cands[0]["returned_as"].is_null());
}

TEST(LensCandidateWire, TiebreakOnEqualGapPicksEarliestByteLo) {
    LensReport r;
    r.document_text = "x";  // byte offsets below are synthetic, not sliced from this
    LensField f = present_field("code", "45", true, true);
    f.occurrence = 0;
    r.fields = {f};
    r.candidates_requested = true;
    // Both candidates have the same length gap (1) against "45"; the one with
    // the earlier byte_lo must be chosen. Array order is already the
    // producer's byte_lo-ascending contract — lens_report_to_json does not
    // re-sort, so it is given in that order directly.
    r.key_candidates["code"] = {LensCandidate{"X45", 10, 13}, LensCandidate{"45X", 30, 33}};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    auto& cands = j["key_candidates"]["code"];
    ASSERT_EQ(cands.size(), 2u);
    EXPECT_EQ(cands[0]["value"], "X45");   // byte_lo 10 — earlier
    EXPECT_EQ(cands[0]["returned_as"], 0);
    EXPECT_TRUE(cands[1]["returned_as"].is_null());
}

TEST(LensCandidateWire, ProducerFailureOmitsKeyCandidatesAndSetsError) {
    LensReport r;
    r.fields = {present_field("monthly_rent", "1,450.00 GBP", true, true)};
    r.candidates_requested = true;
    r.candidates_producer_failed = true;
    r.candidates_error = "candidate-set pass 2: 0 parseable lines";

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_FALSE(j.contains("key_candidates"));
    EXPECT_EQ(j["candidates_error"], "candidate-set pass 2: 0 parseable lines");
}

TEST(LensCandidateWire, NotRequestedOmitsBothKeyCandidatesAndError) {
    LensReport r;   // candidates_requested defaults false — pass 2 never ran
    r.fields = {present_field("monthly_rent", "1,450.00 GBP", true, true)};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_FALSE(j.contains("key_candidates"));
    EXPECT_FALSE(j.contains("candidates_error"));
}

TEST(LensCandidateWire, AnchorNullWhenNoStructureExists) {
    LensReport r;
    r.document_text = "45";   // the whole document IS the candidate — no label,
                              // and the "enclosing line" carries no more context
                              // than the span itself.
    r.fields = {present_field("amount", "45", true, true)};
    r.candidates_requested = true;
    r.key_candidates["amount"] = {LensCandidate{"45", 0, 2}};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_TRUE(j["key_candidates"]["amount"][0]["anchor"].is_null());
}

// ── Regression: the hard-wrapped supersession mail (Heliotrope PO-4471) ──────
// Pass 2 enumerates BOTH rivals correctly; every one of these cases is a span
// the apply step used to delete silently, leaving only the SUPERSEDED rival and
// making a correct answer render as "1 found · 0 returned".
namespace {
const char* kWrappedMail =
    "ORIGINAL CONFIRMATION (4 September, superseded)\n"
    "Order reference PO-4471. Quantity 640 units of the BF-22 bulkhead fitting\n"
    "at 18.40 EUR each, giving an order total of 11,776.00 EUR net of VAT.\n"
    "Delivery ex works Gdansk on 30 October 2026. Payment terms 30 days net\n"
    "from invoice date.\n"
    "\n"
    "REVISED CONFIRMATION (this message)\n"
    "Following your increase to 720 units, the unit price falls to 17.65 EUR\n"
    "under the volume band. The revised order total is 12,708.00 EUR net of\n"
    "VAT. Please note that the earlier figure of 11,776.00 EUR is no longer\n"
    "valid.\n"
    "\n"
    "Delivery has moved to 13 November 2026, as the additional 80 units\n"
    "require a second galvanising run.\n";
}  // namespace

TEST(LensCandidatePass2Apply, HardWrappedSpanIsResolvedNotDropped) {
    // "12,708.00 EUR net of VAT" is split by the 70-column wrap: the document
    // has "...net of\nVAT". document.find() misses it; the candidate is real.
    const std::string document = kWrappedMail;
    const std::vector<std::string> keys = {"order_total"};
    LensReport r;
    lens_apply_pass2_candidates(document,
        "order_total: \"11,776.00 EUR net of VAT\"\n"
        "order_total: \"12,708.00 EUR net of VAT\"\n", keys, r);
    ASSERT_EQ(r.key_candidates["order_total"].size(), 2u);
    for (const LensCandidate& c : r.key_candidates["order_total"])
        EXPECT_EQ(document.substr(c.byte_lo, c.byte_hi - c.byte_lo), c.value)
            << "value must stay a byte-exact slice of the document";
    // The wrapped one carries the document's own newline, not the model's space.
    EXPECT_EQ(r.key_candidates["order_total"][1].value, "12,708.00 EUR net of\nVAT");
}

TEST(LensCandidatePass2Apply, TrailingSentencePeriodDoesNotLoseTheSpan) {
    // The model quotes "13 November 2026." — the document has "13 November
    // 2026," (comma). One character, and the whole revised delivery date was
    // dropped, leaving only the superseded 30 October.
    const std::string document = kWrappedMail;
    const std::vector<std::string> keys = {"delivery_date"};
    LensReport r;
    lens_apply_pass2_candidates(document,
        "delivery_date: \"30 October 2026.\"\n"
        "delivery_date: \"13 November 2026.\"\n", keys, r);
    ASSERT_EQ(r.key_candidates["delivery_date"].size(), 2u);
    EXPECT_EQ(r.key_candidates["delivery_date"][0].value, "30 October 2026.");
    EXPECT_EQ(r.key_candidates["delivery_date"][1].value, "13 November 2026");
    for (const LensCandidate& c : r.key_candidates["delivery_date"])
        EXPECT_EQ(document.substr(c.byte_lo, c.byte_hi - c.byte_lo), c.value);
}

TEST(LensCandidatePass2Apply, TolerantMatchStillRejectsARealParaphrase) {
    // The tolerance must not become a fuzzy matcher: a changed CHARACTER is a
    // real difference and still fails every tier.
    const std::string document = kWrappedMail;
    const std::vector<std::string> keys = {"order_total"};
    LensReport r;
    lens_apply_pass2_candidates(document, "order_total: \"12708.00 EUR\"\n", keys, r);
    EXPECT_FALSE(r.candidates_producer_failed);
    EXPECT_TRUE(r.key_candidates["order_total"].empty());
}

TEST(LensCandidatePass2Apply, TwoQuotationsOfOneSpanDeduplicateByRange) {
    const std::string document = kWrappedMail;
    const std::vector<std::string> keys = {"order_total"};
    LensReport r;
    lens_apply_pass2_candidates(document,
        "order_total: \"12,708.00 EUR net of VAT\"\n"
        "order_total: \"12,708.00 EUR net of\nVAT\"\n", keys, r);
    EXPECT_EQ(r.key_candidates["order_total"].size(), 1u);
}

TEST(LensCandidateWire, AnchorNeverCutsMidToken) {
    // The reported defect: "Delivery ex works Gdansk on 30 October 2026.
    // Payment terms 3" — the "3" is the head of "30 days net" and reads as a
    // fact. The enclosing SENTENCE is both shorter and true.
    LensReport r;
    r.document_text = kWrappedMail;
    r.fields = {present_field("delivery_date", "30 October 2026", true, true)};
    r.candidates_requested = true;
    const size_t lo = r.document_text.find("30 October 2026");
    ASSERT_NE(lo, std::string::npos);
    r.key_candidates["delivery_date"] = {LensCandidate{"30 October 2026", lo, lo + 15}};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    const std::string a = j["key_candidates"]["delivery_date"][0]["anchor"];
    EXPECT_EQ(a.substr(a.size() - 1), std::string(")"))
        << "expected the ALL-CAPS section heading, actual '" << a << "'";
    EXPECT_EQ(a, "ORIGINAL CONFIRMATION (4 September, superseded)");
}

TEST(LensCandidateWire, AllCapsSectionHeadingIsALabel) {
    // The heading that tells a reviewer WHICH rival this is. Correspondence
    // heads sections this way instead of numbering them.
    LensReport r;
    r.document_text = kWrappedMail;
    r.fields = {present_field("order_total", "12,708.00 EUR", true, true)};
    r.candidates_requested = true;
    const size_t lo = r.document_text.find("12,708.00 EUR");
    ASSERT_NE(lo, std::string::npos);
    r.key_candidates["order_total"] = {LensCandidate{"12,708.00 EUR", lo, lo + 13}};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_EQ(j["key_candidates"]["order_total"][0]["anchor"],
              "REVISED CONFIRMATION (this message)");
}

TEST(LensCandidateWire, AnchorTruncationLandsOnAWordBoundaryAndIsMarked) {
    LensReport r;
    r.document_text =
        "Freight is DAP Rotterdam and is included and the customs clearance for "
        "this consignment remains entirely with the buyer at 45 units\n";
    r.fields = {present_field("quantity", "45", true, true)};
    r.candidates_requested = true;
    const size_t lo = r.document_text.find("45");
    r.key_candidates["quantity"] = {LensCandidate{"45", lo, lo + 2}};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    const std::string a = j["key_candidates"]["quantity"][0]["anchor"];
    ASSERT_FALSE(a.empty());
    // Marked as a fragment, and the text before the marker ends on a whole word.
    ASSERT_GE(a.size(), 3u);
    EXPECT_EQ(a.substr(a.size() - 3), std::string("\xe2\x80\xa6"));
    const std::string body = a.substr(0, a.size() - 3);
    EXPECT_EQ(r.document_text.compare(0, body.size(), body), 0)
        << "the anchor must be a prefix of the real line, actual '" << body << "'";
    EXPECT_NE(body.back(), ' ');
}

TEST(LensCandidateWire, AnchorIsValidUtf8WhenTheLineIsMultibyte) {
    // A 60-BYTE cut lands inside a multi-byte character on the German half of
    // the corpus and emits invalid UTF-8 into the JSON.
    LensReport r;
    r.document_text =
        "Die Zahlungsbedingungen für die überarbeitete Bestellung müssen "
        "spätestens am Montag bestätigt werden, Betrag 45 EUR\n";
    r.fields = {present_field("amount", "45 EUR", true, true)};
    r.candidates_requested = true;
    const size_t lo = r.document_text.find("45 EUR");
    r.key_candidates["amount"] = {LensCandidate{"45 EUR", lo, lo + 6}};

    const std::string out = lens_report_to_json(r);
    nlohmann::json j = nlohmann::json::parse(out);   // throws on invalid UTF-8
    const std::string a = j["key_candidates"]["amount"][0]["anchor"];
    // Every continuation byte must be preceded by a valid lead byte.
    for (size_t i = 0; i < a.size();) {
        const unsigned char c = (unsigned char)a[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        ASSERT_NE(len, 0u) << "invalid UTF-8 lead byte in anchor '" << a << "'";
        ASSERT_LE(i + len, a.size()) << "truncated UTF-8 sequence in anchor '" << a << "'";
        i += len;
    }
}

TEST(LensCandidateWire, AnchorResolvesToThePrecedingLabelLine) {
    LensReport r;
    r.document_text = "2. RENT:\nMonthly rent is 1,450.00 GBP due on the first.\n";
    r.fields = {present_field("monthly_rent", "1,450.00 GBP", true, true)};
    r.candidates_requested = true;
    const size_t lo = r.document_text.find("1,450.00 GBP");
    r.key_candidates["monthly_rent"] = {LensCandidate{"1,450.00 GBP", lo, lo + 13}};

    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_EQ(j["key_candidates"]["monthly_rent"][0]["anchor"], "2. RENT:");
}

// ── The per-model flash-prefill gate (docs/plan-lens-server-shape.md §4.2.1) ──
// A flash prefill is admissible per MODEL, not per build, because the drift
// gate answers per model: the dense 9B passes it and the MoE 35B fails. The
// table is the record of that, and the two properties worth pinning are that
// the permission travels with the entry and that the DEFAULT is refusal — a
// new model must earn it rather than inherit it.
TEST(LensCalibration, FlashPrefillIsPerModelAndDefaultsToRefused) {
    const LensCalibration* dense = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(dense, nullptr) << "Qwen3.8-9B must stay calibrated";
    EXPECT_TRUE(dense->constants.flash_prefill_ok)
        << "the 9B passed the drift gate: 0/98 decisions moved";
    EXPECT_NE(std::string(dense->constants.flash_prefill_provenance).find("drift gate"),
              std::string::npos)
        << "a bare boolean is a claim with no receipt";

    const LensCalibration* dense27 = lens_calibration_for("qwen35", 65, kLensAnyFileType);
    ASSERT_NE(dense27, nullptr) << "Qwen3.8-27B must stay calibrated";
    EXPECT_TRUE(dense27->constants.flash_prefill_ok)
        << "the 27B passed both drift-gate arms: 15/15 token-identical, 0/98 decisions moved";
    EXPECT_NE(std::string(dense27->constants.flash_prefill_provenance).find("drift gate"),
              std::string::npos);

    for (uint32_t blocks : {40u, 41u}) {
        const LensCalibration* moe = lens_calibration_for("qwen35moe", blocks, kLensAnyFileType);
        ASSERT_NE(moe, nullptr) << "Qwen3.6-35B-A3B must stay calibrated";
        EXPECT_FALSE(moe->constants.flash_prefill_ok)
            << "the 35B FAILED the drift gate, and on an MoE the failure is "
               "categorical: expert routing is an argmax, so no margin applies";
    }

    // The default is what protects a model nobody has measured yet.
    EXPECT_FALSE(LensConstants{}.flash_prefill_ok);

    // Three models, three gate runs, and the split is DENSE vs MoE — not size,
    // not family. Both dense Qwen 3.8 builds are token-identical under a
    // prefill-shape change; both MoE models (this table's 35B, and Gemma
    // 4-26B-A4B outside it) each changed one extraction in fifteen, because
    // expert routing is a top-k argmax and the perturbation is discrete
    // (docs/plan-lens-server-shape.md §4.4). Every entry that says `true` must
    // carry the run that earned it.
    for (const LensCalibration& c : lens_calibrations())
        if (c.constants.flash_prefill_ok)
            EXPECT_NE(std::string(c.constants.flash_prefill_provenance).find("token-identical"),
                      std::string::npos)
                << c.model << " claims flash prefill without naming the gate run";
}

// The configuration stamp is additive: a report nobody stamped serializes
// exactly as it did before the member existed. Every in-process caller — these
// tests included — is in that case, so this is also what keeps the rest of this
// file honest.
TEST(LensReportConfig, UnstampedReportOmitsTheMemberEntirely) {
    LensReport r;
    EXPECT_TRUE(r.config.empty());
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_FALSE(j.contains("config"));

    r.config.weights   = "0123456789abcdef";
    r.config.attention = "flash-prefill";
    r.config.kv_type   = "f16";
    nlohmann::json j2 = nlohmann::json::parse(lens_report_to_json(r));
    ASSERT_TRUE(j2.contains("config"));
    EXPECT_EQ(j2["config"]["weights"],   "0123456789abcdef");
    EXPECT_EQ(j2["config"]["attention"], "flash-prefill");
    EXPECT_EQ(j2["config"]["kv_type"],   "f16");
    EXPECT_EQ(j2["format_version"], "qemmi-lens/v4")
        << "additive, not a version bump — same reasoning as extraction_origin";
}

// The routing digest is additive the same way the config stamp is: a dense
// model, and every in-process caller, serializes exactly as before.
TEST(LensReportRouting, UnstampedReportOmitsTheMemberEntirely) {
    LensReport r;
    EXPECT_TRUE(r.routing.empty());
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_FALSE(j.contains("routing"))
        << "a dense model has no routing to fingerprint and must emit nothing";

    r.routing.digest    = "feedfacecafebeef";
    r.routing.layers    = 3;
    r.routing.top_k     = 8;
    r.routing.positions = 1543;
    r.routing.per_layer = {"aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "cccccccccccccccc"};
    nlohmann::json j2 = nlohmann::json::parse(lens_report_to_json(r));
    ASSERT_TRUE(j2.contains("routing"));
    EXPECT_EQ(j2["routing"]["digest"],    "feedfacecafebeef");
    EXPECT_EQ(j2["routing"]["layers"],    3);
    EXPECT_EQ(j2["routing"]["top_k"],     8);
    EXPECT_EQ(j2["routing"]["positions"], 1543);
    ASSERT_EQ(j2["routing"]["per_layer"].size(), 3u);
    EXPECT_EQ(j2["routing"]["per_layer"][1], "bbbbbbbbbbbbbbbb");
    EXPECT_EQ(j2["format_version"], "qemmi-lens/v4")
        << "additive, not a version bump — same reasoning as the config stamp";
}

// per_layer is the whole reason this is a fingerprint and not a boolean: a
// cross-configuration comparison must be able to say HOW MANY layers diverged,
// because the whole-trace digest differs on essentially every such comparison
// while the extraction changes about 1 time in 15.
TEST(LensReportRouting, PerLayerSurvivesTheRoundTripInOrder) {
    LensReport r;
    r.routing.digest = "0000000000000001";
    r.routing.layers = 4;
    r.routing.top_k  = 2;
    r.routing.per_layer = {"1111111111111111", "2222222222222222",
                           "3333333333333333", "4444444444444444"};
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    ASSERT_EQ(j["routing"]["per_layer"].size(), 4u);
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(j["routing"]["per_layer"][i], r.routing.per_layer[i])
            << "layer order is the localisation; a reordered array is a wrong answer";
}

// An empty digest with non-empty metadata must still be treated as absent —
// otherwise a partially-filled report emits a fingerprint of nothing.
TEST(LensReportRouting, MetadataWithoutADigestIsStillAbsent) {
    LensReport r;
    r.routing.layers    = 40;
    r.routing.top_k     = 8;
    r.routing.positions = 99;
    EXPECT_TRUE(r.routing.empty());
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_FALSE(j.contains("routing"));
}

// The comparison block is absent unless the caller supplied a fingerprint to
// compare against — verify without one serializes as it did before.
TEST(LensReportRouting, ExpectedIsAbsentUnlessSupplied) {
    LensReport r;
    r.routing.digest    = "aaaaaaaaaaaaaaaa";
    r.routing.per_layer = {"1111111111111111", "2222222222222222"};
    r.routing.layers    = 2;
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    ASSERT_TRUE(j.contains("routing"));
    EXPECT_FALSE(j["routing"].contains("expected"));

    r.routing.expected.digest               = "bbbbbbbbbbbbbbbb";
    r.routing.expected.identical            = false;
    r.routing.expected.layers_diverged      = 1;
    r.routing.expected.layers_compared      = 2;
    r.routing.expected.first_diverged_index = 1;
    nlohmann::json j2 = nlohmann::json::parse(lens_report_to_json(r));
    ASSERT_TRUE(j2["routing"].contains("expected"));
    EXPECT_EQ(j2["routing"]["expected"]["digest"],               "bbbbbbbbbbbbbbbb");
    EXPECT_EQ(j2["routing"]["expected"]["identical"],            false);
    EXPECT_EQ(j2["routing"]["expected"]["layers_diverged"],      1);
    EXPECT_EQ(j2["routing"]["expected"]["layers_compared"],      2);
    EXPECT_EQ(j2["routing"]["expected"]["first_diverged_index"], 1);
    EXPECT_EQ(j2["format_version"], "qemmi-lens/v4")
        << "still additive — the comparison does not change the format";
}

// `identical` is the narrow flag; `layers_diverged` is the number a reader
// should weigh. Pin that both survive independently, so a future change cannot
// quietly collapse the magnitude into the boolean.
TEST(LensReportRouting, DivergenceCountIsCarriedSeparatelyFromTheFlag) {
    LensReport r;
    r.routing.digest = "aaaaaaaaaaaaaaaa";
    r.routing.expected.digest          = "aaaaaaaaaaaaaaaa";
    r.routing.expected.identical       = true;
    r.routing.expected.layers_diverged = 0;
    r.routing.expected.layers_compared = 40;
    nlohmann::json j = nlohmann::json::parse(lens_report_to_json(r));
    EXPECT_EQ(j["routing"]["expected"]["identical"], true);
    EXPECT_EQ(j["routing"]["expected"]["layers_diverged"], 0);
    EXPECT_EQ(j["routing"]["expected"]["layers_compared"], 40);
    EXPECT_EQ(j["routing"]["expected"]["first_diverged_index"], -1)
        << "-1 means nothing diverged, and must not be confused with index 0";
}

// ═════════════════════════════════════════════════════════════════════════════
// LOCATE — the pure span finder (../qemmi-lens/docs/plan-locate-and-cut.md §3)
// ═════════════════════════════════════════════════════════════════════════════
//
// lens_locate_spans is the whole of locate's judgement: everything else on that
// route is prompt bookkeeping and byte arithmetic. It runs without a model, so
// it is gated here rather than left to the endpoint smoke.

TEST(LensLocateSpans, GrowsFromThePeakAndStopsAtTheTail) {
    // One hill. tail_frac 0.25 of the 1.0 peak ⇒ 0.3 and 0.4 are in, 0.1 is out.
    std::vector<float> mass = {0.05f, 0.1f, 0.4f, 1.0f, 0.3f, 0.1f, 0.02f};
    auto sp = lens_locate_spans(mass, /*top_k*/ 3, /*tail_frac*/ 0.25, /*max_width*/ 64);
    ASSERT_FALSE(sp.empty());
    EXPECT_EQ(sp[0].first, 2);
    EXPECT_EQ(sp[0].second, 5) << "grew left to 0.4 and right to 0.3, stopping below 0.25*peak";
}

TEST(LensLocateSpans, TwoSeparatedHillsComeBackAsTwoSpansHighestFirst) {
    // 0.3 clears 0.25*0.9 = 0.225 and joins the first hill; 0.1 does not clear
    // 0.25*0.5 = 0.125 and leaves the second one a single position.
    std::vector<float> mass = {0.9f, 0.3f, 0.01f, 0.01f, 0.01f, 0.5f, 0.1f};
    auto sp = lens_locate_spans(mass, 2, 0.25, 64);
    ASSERT_EQ(sp.size(), 2u);
    EXPECT_EQ(sp[0].first, 0);  EXPECT_EQ(sp[0].second, 2);   // the 0.9 hill, highest first
    EXPECT_EQ(sp[1].first, 5);  EXPECT_EQ(sp[1].second, 6);
}

// The twins case, which is the measured weakness of this signal: two adjacent
// near-identical lines must not merge into one span that spells neither.
TEST(LensLocateSpans, AdjacentPeaksStayTwoSpansRatherThanMerging) {
    std::vector<float> mass = {0.02f, 0.9f, 0.8f, 0.02f};
    auto sp = lens_locate_spans(mass, 2, 0.25, 64);
    ASSERT_EQ(sp.size(), 2u);
    // The first span takes [1,3) — 0.8 is above 0.25*0.9 — so the second peak is
    // already consumed and the runner-up is whatever is left, not a duplicate.
    EXPECT_EQ(sp[0].first, 1); EXPECT_EQ(sp[0].second, 3);
    EXPECT_NE(sp[1].first, sp[0].first) << "a second span must not repeat the first";
    for (const auto& a : sp)
        EXPECT_TRUE(a.first >= sp[0].second || a.second <= sp[0].first || &a == &sp[0])
            << "spans must be disjoint";
}

TEST(LensLocateSpans, MaxWidthCapsAPlateau) {
    std::vector<float> mass(40, 1.0f);
    auto sp = lens_locate_spans(mass, 1, 0.25, /*max_width*/ 5);
    ASSERT_EQ(sp.size(), 1u);
    EXPECT_LE(sp[0].second - sp[0].first, 5);
}

// A shorter list than top_k is a real answer — "there is nothing else here" —
// and must never be padded with zero-mass positions the caller would then cut to.
TEST(LensLocateSpans, NeverReturnsAZeroMassSpanToFillTopK) {
    std::vector<float> mass = {0.0f, 0.7f, 0.0f, 0.0f};
    auto sp = lens_locate_spans(mass, 4, 0.25, 64);
    ASSERT_EQ(sp.size(), 1u) << "one hill exists; top_k=4 must not invent three more";
    EXPECT_EQ(sp[0].first, 1);
}

TEST(LensLocateSpans, AllZeroMassYieldsNoSpans) {
    std::vector<float> mass(16, 0.0f);
    EXPECT_TRUE(lens_locate_spans(mass, 3, 0.25, 64).empty());
}

TEST(LensLocateSpans, FailsLoudOnBadParameters) {
    std::vector<float> mass = {1.0f};
    EXPECT_THROW(lens_locate_spans(mass, 0, 0.25, 8), std::runtime_error);
    EXPECT_THROW(lens_locate_spans(mass, 3, -0.1, 8), std::runtime_error);
    EXPECT_THROW(lens_locate_spans(mass, 3, 1.5, 8), std::runtime_error);
    EXPECT_THROW(lens_locate_spans(mass, 3, 0.25, 0), std::runtime_error);
}

// The serializer's two load-bearing facts: no extraction_origin (a locate report
// is not an extraction of either origin), and `uncalibrated` always present so a
// reader never has to decide whether absence means "calibrated" or "old server".
TEST(LensLocateJson, CarriesUncalibratedAndOmitsExtractionOrigin) {
    LensLocateReport r;
    r.model = "Qwen3.6 (attention lens)";
    r.config.weights = "9f3c"; r.config.attention = "materialized"; r.config.kv_type = "f16";
    r.locate_layer = 11; r.locate_head = 5; r.top_k = 3;
    r.locate_provenance = "LOCHEAD 2026-09-18, 75 keys: 82.7% top1 / 89.3% top3";
    r.prompt_len = 136; r.doc_lo = 3; r.doc_hi = 70;
    r.hits.push_back({"unit_price", {LensLocateHit{131, 140, 0.6103, 0.4102, 45, 48}}});
    r.hits.push_back({"nothing_here", {}});

    nlohmann::json j = nlohmann::json::parse(lens_locate_to_json(r));
    EXPECT_EQ(j["format_version"], "qemmi-lens/v4") << "a new endpoint is not a new format";
    // False for a key vocabulary: LOCHEAD measured that regime. The flag now
    // means "question mode", and the provenance carries the rate.
    EXPECT_FALSE(j["uncalibrated"].get<bool>());
    EXPECT_FALSE(j["locate_provenance"].get<std::string>().empty())
        << "a bare layer/head pair is a claim with no receipt";
    EXPECT_FALSE(j.contains("extraction_origin"))
        << "locate makes neither the generated nor the supplied claim";
    // `locate`, not `citation`: after the LOCHEAD sweep these are different
    // heads doing different jobs, and the wire must not invite a reader to
    // compare this against a report's citation.
    EXPECT_EQ(j["locate"]["layer"], 11);
    EXPECT_EQ(j["locate"]["head"], 5);
    EXPECT_FALSE(j.contains("citation"));
    EXPECT_EQ(j["hits"]["unit_price"][0]["byte_lo"], 131);
    EXPECT_EQ(j["hits"]["unit_price"][0]["byte_hi"], 140);
    EXPECT_NEAR(j["hits"]["unit_price"][0]["mass"].get<double>(), 0.6103, 1e-9);
    EXPECT_NEAR(j["hits"]["unit_price"][0]["peak"].get<double>(), 0.4102, 1e-9)
        << "peak is the ordering key and must be published beside the span sum";
    ASSERT_TRUE(j["hits"].contains("nothing_here"));
    EXPECT_TRUE(j["hits"]["nothing_here"].empty())
        << "a key with no hits keeps its slot rather than vanishing";
}

// The 9B's locate pair (LOCHEAD 2026-09-19). Pins the coordinates AND the two
// properties that make them safe to ship: locate does not move the cut, and it
// is not the citation coordinate.
TEST(LensCalibrationGuard, NineBLocatePairIsSweptAndFree) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);

    EXPECT_EQ(q9->constants.locate_layer, 11);
    EXPECT_EQ(q9->constants.locate_head,   6);

    // FREE. The --lens-verify-only cut is max(citation, coverage, locate) + 1.
    // locate == coverage here, so folding it in changes nothing. If a future
    // sweep moves locate past citation, this fails and the server pays for it.
    EXPECT_LE(q9->constants.locate_layer, q9->constants.coverage_layer);
    EXPECT_LE(q9->constants.locate_layer, q9->constants.citation_layer);
    const int cut = std::max({q9->constants.citation_layer,
                              q9->constants.coverage_layer,
                              q9->constants.locate_layer}) + 1;
    EXPECT_EQ(cut, 28) << "the 9B verify-only cut must stay 28 of 33 blocks";

    // Locate is NOT the citation coordinate. Measured on the same corpus:
    // reading locate off the citation layer scores 49.3%, rank 107 of 128.
    EXPECT_NE(q9->constants.locate_layer, q9->constants.citation_layer);

    // /v1/locate is no longer refused on this model.
    EXPECT_GE(q9->constants.locate_layer, 0);
    EXPECT_GE(q9->constants.locate_head,  0);
    EXPECT_STRNE(q9->constants.locate_provenance, "not swept by LOCHEAD");
}

// The 27B's locate pair is the mirror image of the 9B's: DELIBERATELY NOT FREE.
// It is the only row where locate sets the cut, so it is the only row that can
// catch a regression in `max(citation, coverage, locate)` — if someone drops
// locate from that max in http_server.cpp, the 9B keeps working and this model
// silently taps a block the verify-only server never loaded.
TEST(LensCalibrationGuard, TwentySevenBLocatePairIsSweptAndSetsTheCut) {
    const LensCalibration* q27 = lens_calibration_for("qwen35", 65, kLensAnyFileType);
    ASSERT_NE(q27, nullptr);

    EXPECT_EQ(q27->constants.locate_layer, 27);
    EXPECT_EQ(q27->constants.locate_head,  10);

    // NOT free, and that is the point: locate is deeper than BOTH other
    // constants, so it alone decides the cut.
    EXPECT_GT(q27->constants.locate_layer, q27->constants.citation_layer);
    EXPECT_GT(q27->constants.locate_layer, q27->constants.coverage_layer);

    const int cut = std::max({q27->constants.citation_layer,
                              q27->constants.coverage_layer,
                              q27->constants.locate_layer}) + 1;
    EXPECT_EQ(cut, 28) << "the 27B verify-only cut is 28 of 65 — locate sets it";

    // The cost this row accepted, stated as an assertion so it cannot drift
    // unnoticed: without locate the cut would have been 20.
    const int cut_without_locate = std::max(q27->constants.citation_layer,
                                            q27->constants.coverage_layer) + 1;
    EXPECT_EQ(cut_without_locate, 20)
        << "landing locate cost this model 8 blocks; if this number moves, the "
           "trade recorded in note-lens-qwen38-27b-probe.md §8 no longer holds";

    // Direction of the depth profile is the OPPOSITE of the 9B's. Asserted
    // because the 9B write-up originally generalized the wrong way round.
    EXPECT_NE(q27->constants.locate_layer, q27->constants.citation_layer);

    EXPECT_GE(q27->constants.locate_layer, 0);
    EXPECT_GE(q27->constants.locate_head,  0);
    EXPECT_STRNE(q27->constants.locate_provenance, "not swept by LOCHEAD");
}

// Cross-model: the two calibrated Qwen3.8 rows disagree about WHERE locate
// lives, and that disagreement is load-bearing — it is why locate_layer is a
// per-row constant and not a rule of thumb.
TEST(LensCalibrationGuard, LocateDepthDirectionIsNotAFamilyConstant) {
    const LensCalibration* q9  = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    const LensCalibration* q27 = lens_calibration_for("qwen35", 65, kLensAnyFileType);
    ASSERT_NE(q9,  nullptr);
    ASSERT_NE(q27, nullptr);

    // 9B: locate SHALLOWER than citation. 27B: locate DEEPER than citation.
    // Same architecture string, opposite answers.
    EXPECT_LT(q9->constants.locate_layer,  q9->constants.citation_layer);
    EXPECT_GT(q27->constants.locate_layer, q27->constants.citation_layer);
}

// ── config.attention must describe the pass that RAN, not the server's flags ──
//
// lens-format.md's contract for `config` is "the numerical configuration that
// produced THIS report" plus "two reports are only comparable when this
// matches". Both halves broke on /v1/locate: its only pass is the tapped
// prefill, which run_lens_locate forces materialized, so a locate report from
// a --flash-attn server IS byte-comparable with one from a plain server — and
// the old label told readers it was not.
TEST(LensAttentionLabel, LocateIsMaterializedWhateverTheServerFlagsSay) {
    using RP = LensReport::RoutePrefill;
    // Every flag combination, including ones the server cannot currently
    // produce: the route property must not depend on any of them.
    EXPECT_STREQ(lens_attention_label(RP::AlwaysMaterialized, false, false), "materialized");
    EXPECT_STREQ(lens_attention_label(RP::AlwaysMaterialized, false, true),  "materialized");
    EXPECT_STREQ(lens_attention_label(RP::AlwaysMaterialized, true,  false), "materialized");
    EXPECT_STREQ(lens_attention_label(RP::AlwaysMaterialized, true,  true),  "materialized");
}

// The regression, stated as the comparison it protects: a locate report from a
// flash server and one from a plain server must carry the SAME stamp, because
// the pass that produced them is the same pass.
TEST(LensAttentionLabel, TwoLocateReportsAcrossFlashSettingsStayComparable) {
    using RP = LensReport::RoutePrefill;
    EXPECT_STREQ(lens_attention_label(RP::AlwaysMaterialized, /*decode*/ false, /*prefill*/ true),
                 lens_attention_label(RP::AlwaysMaterialized, false, false))
        << "a locate report must not claim a configuration its pass never ran under";
}

// extract and verify are NOT the same case — each has an untapped prompt
// prefill that really does run under the flag, so the fix must not flatten
// them to "materialized" and lose a real numerical difference.
TEST(LensAttentionLabel, ExtractAndVerifyStillReportTheServerConfiguration) {
    using RP = LensReport::RoutePrefill;
    EXPECT_STREQ(lens_attention_label(RP::HonoursServerFlag, false, false), "materialized");
    EXPECT_STREQ(lens_attention_label(RP::HonoursServerFlag, false, true),  "flash-prefill");
    EXPECT_STREQ(lens_attention_label(RP::HonoursServerFlag, true,  false), "flash");
    // Decode-flash wins over prefill-flash: "flash" is the stronger statement
    // and the one a reader must not under-read.
    EXPECT_STREQ(lens_attention_label(RP::HonoursServerFlag, true,  true),   "flash");
}

// The label set is closed — lens-format.md documents exactly three values and
// an importer switches on them.
TEST(LensAttentionLabel, EmitsOnlyTheThreeDocumentedValues) {
    using RP = LensReport::RoutePrefill;
    const std::set<std::string> allowed = {"materialized", "flash-prefill", "flash"};
    for (RP r : {RP::HonoursServerFlag, RP::AlwaysMaterialized})
        for (bool d : {false, true})
            for (bool pf : {false, true})
                EXPECT_TRUE(allowed.count(lens_attention_label(r, d, pf)) == 1)
                    << "undocumented label: " << lens_attention_label(r, d, pf);
}

// ── LensKeyAggregation — the opt-in reduction over a key's query rows ────────
//
// MAX is what LOCHEAD measured and must stay the default, byte-identical to the
// behaviour that predates the enum. MEAN exists for SENTENCE-length keys, where
// max inverts into a defect: one filler token spikes and the wordiest key wins
// regardless of content (DECIDE1 2026-09-20 — a 4-way routing task collapsed
// into a single category until the filler was removed, +25 points).
TEST(LensKeyAggregationContract, MaxIsTheDefaultAndIsNamedOnTheWire) {
    EXPECT_STREQ(lens_key_aggregation_name(LensKeyAggregation::Max),  "max");
    EXPECT_STREQ(lens_key_aggregation_name(LensKeyAggregation::Mean), "mean");
    LensLocateReport r;
    EXPECT_EQ(r.key_aggregation, LensKeyAggregation::Max)
        << "the default must remain MAX: it is the reduction LOCHEAD measured, "
           "and every shipped locate_provenance rate is quoted under it";
}

// A report scored under MEAN sits outside LOCHEAD's measurement, exactly as a
// question vocabulary does. If this ever stops being disclosed, a caller reads
// a 96.0% provenance rate for a request that was never measured at 96.0%.
TEST(LensKeyAggregationContract, MeanIsSerializedSoAReportCannotHideIt) {
    LensLocateReport r;
    r.model = "m"; r.locate_layer = 11; r.locate_head = 6; r.top_k = 3;
    r.locate_provenance = "LOCHEAD 2026-09-19";
    r.key_aggregation = LensKeyAggregation::Mean;
    r.uncalibrated = true;                      // what run_lens_locate sets
    const std::string js = lens_locate_to_json(r);
    EXPECT_NE(js.find("\"key_aggregation\":\"mean\""), std::string::npos)
        << "the reduction must travel with the report";
    EXPECT_NE(js.find("\"uncalibrated\":true"), std::string::npos)
        << "MEAN is off-calibration and must say so, like question mode";

    r.key_aggregation = LensKeyAggregation::Max;
    r.uncalibrated = false;
    EXPECT_NE(lens_locate_to_json(r).find("\"key_aggregation\":\"max\""),
              std::string::npos)
        << "emitted unconditionally: an absent member would make a reader guess "
           "whether it meant max or an old server";
}

// The two reductions must actually differ on a multi-row key, or the parameter
// is decorative. Mirrors the shape of the real loop: rows x document positions,
// one row carrying a lone spike.
TEST(LensKeyAggregationContract, MaxAndMeanDisagreeOnASpikyKey) {
    // 3 query rows over 2 document positions.
    //   row0: [0.90, 0.10]   <- the filler token's spurious spike on position 0
    //   row1: [0.02, 0.50]
    //   row2: [0.02, 0.50]   <- two rows agree on position 1
    // The spike must be big enough to win MAX and still lose MEAN; a first
    // draft of this fixture used 0.05/0.40 and the mean came out 0.333 vs
    // 0.300, i.e. the spike won both and the test proved nothing.
    const std::vector<std::vector<float>> rows = {{0.90f, 0.10f},
                                                  {0.02f, 0.50f},
                                                  {0.02f, 0.50f}};
    std::vector<float> mx(2, 0.0f), mn(2, 0.0f);
    for (const auto& r : rows)
        for (size_t d = 0; d < 2; ++d) {
            if (r[d] > mx[d]) mx[d] = r[d];
            mn[d] += r[d];
        }
    for (float& v : mn) v /= (float)rows.size();

    // MAX hands the spike the win; MEAN hands it to the position two rows agree on.
    EXPECT_GT(mx[0], mx[1]) << "max follows the lone spike";
    EXPECT_LT(mn[0], mn[1]) << "mean follows the agreement — this is the +25 points";
}

// ── The CHOICE pair — a third job on the same layer ─────────────────────────
TEST(LensChoiceHead, NineBCarriesASweptChoicePairOnTheLocateLayer) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    EXPECT_EQ(q9->constants.choice_layer, 11);
    EXPECT_EQ(q9->constants.choice_head,   3);

    // THE POINT OF THIS PAIR: same layer as locate, different head, so serving
    // choice costs a --lens-locate-only server no extra blocks.
    EXPECT_EQ(q9->constants.choice_layer, q9->constants.locate_layer)
        << "choice is free on this model precisely because it shares locate's layer";
    EXPECT_NE(q9->constants.choice_head, q9->constants.locate_head)
        << "...but it is NOT the locate head: reading choice there scores 87.5% "
           "against this pair's 92.5%";

    // The rate is conditional on a recipe, so the provenance must carry it.
    // A bare layer/head pair would be a receipt with its conditions stripped.
    const std::string prov = q9->constants.choice_provenance;
    EXPECT_NE(prov.find("DECIDEHEAD"), std::string::npos);
    EXPECT_NE(prov.find("92.5%"),      std::string::npos);
    EXPECT_NE(prov.find("question"),   std::string::npos) << "instruction shape";
    EXPECT_NE(prov.find("mean"),       std::string::npos) << "key aggregation";
    EXPECT_NE(prov.find("Q4_K_M"),     std::string::npos) << "quant stability";
}

// Every other row must stay unswept rather than inheriting the 9B's coordinate.
// Borrowing a head across models is the false-receipt failure this table exists
// to prevent, and DECIDEHEAD has only run on the 9B.
// Was "every row but the 9B is empty", which stopped being the invariant the
// moment a second model was swept. The invariant was never the model list — it
// is that a row NEVER inherits another model's coordinates. So: layer and head
// are set together or not at all, and a row carrying a pair must carry its own
// provenance rather than the default (the default is what an inherited pair
// would still be wearing).
TEST(LensChoiceHead, ARowCarriesItsOwnPairOrRefuses) {
    for (const LensCalibration& c : lens_calibrations()) {
        const bool has = c.constants.choice_layer >= 0;
        EXPECT_EQ(has, c.constants.choice_head >= 0)
            << c.model << ": layer and head must be set together or not at all";
        EXPECT_FALSE(std::string(c.constants.choice_provenance).empty()) << c.model;
        if (has)
            EXPECT_STRNE(c.constants.choice_provenance, "not swept by DECIDEHEAD")
                << c.model << " carries a pair under the default provenance, which is "
                              "what an inherited coordinate looks like";
    }
}

TEST(LensChoiceHead, HeadRoleIsNamedOnTheWireAndDefaultsToLocate) {
    EXPECT_STREQ(lens_head_role_name(LensHeadRole::Locate), "locate");
    EXPECT_STREQ(lens_head_role_name(LensHeadRole::Choice), "choice");
    LensLocateReport r;
    EXPECT_EQ(r.head_role, LensHeadRole::Locate)
        << "default must stay locate: the route predates the choice pair";
    r.model = "m"; r.locate_layer = 11; r.locate_head = 3; r.top_k = 3;
    r.head_role = LensHeadRole::Choice;
    r.locate_provenance = "DECIDEHEAD 2026-09-20";
    const std::string js = lens_locate_to_json(r);
    EXPECT_NE(js.find("\"head_role\":\"choice\""), std::string::npos)
        << "a reader must be able to tell which job's calibration applies";
    EXPECT_NE(js.find("DECIDEHEAD"), std::string::npos)
        << "provenance must switch with the pair, not stay locate's";
}

// The 9B row is kLensAnyFileType, so it serves Q8_0 AND Q4_K_M — and the same
// pair measures 96.0% top-3 on one and 93.3% on the other. A provenance naming
// only one of them makes every report on the other file overstate itself, which
// is the failure this table exists to prevent. Guarded because the temptation
// to "tidy" a long string back to a single number is real.
TEST(LensChoiceHead, NineBLocateProvenanceNamesBothQuants) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    const std::string p = q9->constants.locate_provenance;
    EXPECT_NE(p.find("Q8_0"),   std::string::npos) << "must name the quant it was swept on";
    EXPECT_NE(p.find("Q4_K_M"), std::string::npos) << "...and the one it also serves";
    EXPECT_NE(p.find("96.0%"),  std::string::npos) << "Q8_0 top-3";
    EXPECT_NE(p.find("93.3%"),  std::string::npos) << "Q4_K_M top-3";
    EXPECT_NE(p.find("88.0%"),  std::string::npos) << "Q8_0 top-1";
    EXPECT_NE(p.find("84.0%"),  std::string::npos) << "Q4_K_M top-1";
    // The row is unpinned, which is WHY both rates have to be here.
    EXPECT_EQ(q9->file_type, kLensAnyFileType);
}

// ── The ABSENT pair — the fourth job, and the first that is NOT free ────────
TEST(LensAbsentHead, NineBCarriesASweptAbsencePairThatCostsDepth) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    EXPECT_EQ(q9->constants.absent_layer, 19);
    EXPECT_EQ(q9->constants.absent_head,  10);

    // Unlike choice, absence does NOT share locate's layer. That is the whole
    // cost of this pair and the reason the locate-only cut moved 12 -> 20:
    // there is no stable shallow absence head (L11 is h=8 on Q4_K_M and h=6 on
    // Q8_0), so depth was paid for a coordinate that survives a file swap.
    EXPECT_GT(q9->constants.absent_layer, q9->constants.locate_layer);
    EXPECT_GT(q9->constants.absent_layer, q9->constants.choice_layer);

    // The locate-only server must load every head it can serve.
    const int locate_only_cut = std::max({q9->constants.locate_layer,
                                          q9->constants.choice_layer,
                                          q9->constants.absent_layer}) + 1;
    EXPECT_EQ(locate_only_cut, 20) << "absence sets the locate-only cut on this model";
    // Verify-only is unaffected: citation is deeper still.
    const int verify_only_cut = std::max({q9->constants.citation_layer,
                                          q9->constants.coverage_layer,
                                          q9->constants.locate_layer,
                                          q9->constants.choice_layer,
                                          q9->constants.absent_layer}) + 1;
    EXPECT_EQ(verify_only_cut, 28) << "citation still dominates the verify-only cut";

    // AUC is a separation, not a rate — the provenance must not let a reader
    // mistake 0.9948 for an accuracy, and must carry both quants.
    const std::string p = q9->constants.absent_provenance;
    EXPECT_NE(p.find("ABSENTHEAD"), std::string::npos);
    EXPECT_NE(p.find("0.9948"),     std::string::npos) << "Q4_K_M AUC";
    EXPECT_NE(p.find("0.9953"),     std::string::npos) << "Q8_0 AUC";
    EXPECT_NE(p.find("89.6%"),      std::string::npos) << "the operating point a buyer hears";
    EXPECT_NE(p.find("SEPARATION"), std::string::npos) << "AUC must not read as a rate";
}

TEST(LensAbsentHead, ARowCarriesItsOwnPairOrRefuses) {
    for (const LensCalibration& c : lens_calibrations()) {
        const bool has = c.constants.absent_layer >= 0;
        EXPECT_EQ(has, c.constants.absent_head >= 0)
            << c.model << ": layer and head must be set together or not at all";
        EXPECT_FALSE(std::string(c.constants.absent_provenance).empty()) << c.model;
        if (has)
            EXPECT_STRNE(c.constants.absent_provenance, "not swept by ABSENTHEAD") << c.model;
    }
}

// Two models now carry an absence pair, and geometry does not transfer between
// them — the 9B's sits at L19 of 33 and the 27B's at L31 of 65. If these ever
// coincide, suspect a copy rather than a coincidence.
TEST(LensAbsentHead, TheTwoSweptModelsLandOnDifferentCoordinates) {
    const LensCalibration* q9  = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    const LensCalibration* q27 = lens_calibration_for("qwen35", 65, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    ASSERT_NE(q27, nullptr);
    ASSERT_GE(q9->constants.absent_layer, 0);
    ASSERT_GE(q27->constants.absent_layer, 0);
    EXPECT_NE(std::make_pair(q9->constants.absent_layer, q9->constants.absent_head),
              std::make_pair(q27->constants.absent_layer, q27->constants.absent_head));
}

// Four jobs, four pairs, all distinct — the per-action design in one assert.
TEST(LensAbsentHead, TheThreeJobsReadThreeDifferentHeads) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    const std::set<std::pair<int,int>> pairs = {
        {q9->constants.locate_layer, q9->constants.locate_head},
        {q9->constants.choice_layer, q9->constants.choice_head},
        {q9->constants.absent_layer, q9->constants.absent_head},
        {q9->constants.score_layer,  q9->constants.score_head}};
    EXPECT_EQ(pairs.size(), 4u) << "locate, choice, absence and score must not collapse "
                                   "onto one coordinate — reuse has measured badly every "
                                   "time";
    EXPECT_STREQ(lens_head_role_name(LensHeadRole::Absent), "absent");
}

// ── SCOREHEAD: the ordinal pair ─────────────────────────────────────────────
TEST(LensScoreHead, NineBCarriesTheSweptOrdinalPair) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    EXPECT_EQ(q9->constants.score_layer, 19);
    EXPECT_EQ(q9->constants.score_head, 11);
    EXPECT_STREQ(lens_head_role_name(LensHeadRole::Score), "score");
}

// The trap this file exists to catch. `kLensCalibrations` uses POSITIONAL
// aggregate initialisation — the /*field*/ markers are comments — so appending
// the score triple anywhere but last silently shifts every pair after it and
// still compiles. Score sits beside absence on the same layer, which makes a
// swap between exactly those two invisible to a layer check; pin the HEADS.
TEST(LensScoreHead, ScoreAndAbsentShareALayerButNotAHead) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    EXPECT_EQ(q9->constants.score_layer, q9->constants.absent_layer)
        << "score is free precisely because it shares absence's layer";
    EXPECT_NE(q9->constants.score_head, q9->constants.absent_head)
        << "adjacent heads, measured differently: h=10 won absence, h=11 the ordinal";
}

// The score pair moves no cut. Absence already pays for layer 19, so a server
// that serves absence serves score for nothing — if this ever fails, the score
// head has drifted off absence's layer and someone is paying depth for it.
TEST(LensScoreHead, ScoreIsFreeOnALocateOnlyServer) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    const int without = std::max({q9->constants.locate_layer,
                                  q9->constants.choice_layer,
                                  q9->constants.absent_layer}) + 1;
    const int with    = std::max({q9->constants.locate_layer,
                                  q9->constants.choice_layer,
                                  q9->constants.absent_layer,
                                  q9->constants.score_layer}) + 1;
    EXPECT_EQ(with, without);
    EXPECT_EQ(with, 20);
}

// An ordinal head is not selected by accuracy and its provenance has to say so
// out loud, because 66.7% sitting alone in a table reads like a bad head rather
// than a number the sweep deliberately did not optimise.
TEST(LensScoreHead, ProvenanceWarnsThatExactMatchIsNotTheNumber) {
    const LensCalibration* q9 = lens_calibration_for("qwen35", 33, kLensAnyFileType);
    ASSERT_NE(q9, nullptr);
    const std::string prov = q9->constants.score_provenance;
    EXPECT_NE(prov.find("NOT THE PRODUCT NUMBER"), std::string::npos);
    EXPECT_NE(prov.find("UNCALIBRATED"), std::string::npos)
        << "a caller that rounds the fraction to an integer level reads low";
    EXPECT_NE(prov.find("Q8_0"), std::string::npos);
    EXPECT_NE(prov.find("Q4_K_M"), std::string::npos)
        << "quant stability was measured for this pair; the receipt must name both";
}

// An unswept model must be refused, not served off a neighbour's coordinates.
// "Unswept" and "swept and declined" are BOTH -1 and both refused — the pair
// is absent either way — but they are different facts and the provenance is
// where the difference lives, so a row may carry -1 with a non-default string.
TEST(LensScoreHead, ModelsWithoutASweptOrdinalPairDeclareItUnmeasured) {
    for (const LensCalibration& c : lens_calibrations()) {
        if (c.constants.score_layer >= 0) continue;
        EXPECT_EQ(c.constants.score_head, -1) << c.model;
        EXPECT_FALSE(std::string(c.constants.score_provenance).empty()) << c.model;
    }
}

// ── The 27B: absence landed, choice and score swept and DECLINED ────────────
TEST(LensTwentySevenB, CarriesTheSweptAbsencePair) {
    const LensCalibration* q27 = lens_calibration_for("qwen35", 65, kLensAnyFileType);
    ASSERT_NE(q27, nullptr);
    EXPECT_EQ(q27->constants.absent_layer, 31);
    EXPECT_EQ(q27->constants.absent_head, 23);
    // Absence sets this model's locate-only cut: 4 blocks deeper than locate.
    EXPECT_GT(q27->constants.absent_layer, q27->constants.locate_layer);
}

// A -1 that means "we measured it and said no" must never read as "we never
// looked". Both rows below were swept on 2026-09-20; a future reader who sees
// the default string would re-run a sweep that has already answered.
TEST(LensTwentySevenB, DeclinedPairsDoNotClaimToBeUnswept) {
    const LensCalibration* q27 = lens_calibration_for("qwen35", 65, kLensAnyFileType);
    ASSERT_NE(q27, nullptr);
    EXPECT_EQ(q27->constants.choice_layer, -1);
    EXPECT_EQ(q27->constants.score_layer, -1);
    const std::string ch = q27->constants.choice_provenance;
    const std::string sc = q27->constants.score_provenance;
    EXPECT_STRNE(q27->constants.choice_provenance, "not swept by DECIDEHEAD");
    EXPECT_STRNE(q27->constants.score_provenance, "not swept by SCOREHEAD");
    EXPECT_NE(ch.find("DECLINED"), std::string::npos);
    EXPECT_NE(sc.find("REFUSED"), std::string::npos);
    // The reason each was declined has to survive, not just the verdict.
    EXPECT_NE(ch.find("NOTHING CHEAPER SURVIVES"), std::string::npos);
    EXPECT_NE(sc.find("ZERO of the"), std::string::npos);
}

// Only one 27B file exists, so the cross-quant agreement that qualified every
// 9B pair was unavailable. That limit belongs in the receipt, not in a doc.
TEST(LensTwentySevenB, EveryDecisionProvenanceNamesTheSingleQuantLimit) {
    const LensCalibration* q27 = lens_calibration_for("qwen35", 65, kLensAnyFileType);
    ASSERT_NE(q27, nullptr);
    for (const char* prov : {q27->constants.choice_provenance,
                             q27->constants.absent_provenance,
                             q27->constants.score_provenance})
        EXPECT_NE(std::string(prov).find("Q3_K_M"), std::string::npos) << prov;
}
