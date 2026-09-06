// server_lens.cpp — Qemmi-Lens extraction (docs/plan-qemmi-lens.md P2/A2).
// The pure computation is the relocated probe math (attn_provenance.cpp
// run_lens_gen / eval_field / parse_fields); the driver reuses the P1 attention
// tap (forward_pass_base set_attention_taps) instead of the probe's inline
// interpose.

#include "server_lens.h"
#include "engine/graph_compute.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>   // the shape contract's real parse (tolerant, then loud)

#include "../models/forward_pass_base.h"
#include "../loader/tokenizer.h"
#include "../loader/chat_template.h"
#include "../sampling/grammar_vocab.h"

namespace qinf {

// ── The REFUTED fixed KV grammar — probe control arm only ────────────────────
// Once the product path's ONE fixed universal KV grammar (plan §1.1, a P0-locked
// decision). REFUTED by measurement 2026-07-16 and taken off the product path in
// Stage 2 (docs/note-nogrammar-refutation.md): on the Leg C corpus it lost on
// every axis INCLUDING the guaranteed parse it existed for (14/15 vs free's
// 15/15), because `value ::= (…)+` forces a non-empty value for every hinted key
// — so a concept the document lacks must be fabricated, and the whole extraction
// can collapse to `","` (m_de7, live). That forced value was the sole cause of the
// absent-concept problem, and therefore of the two-pass presence gate built to
// contain it. Both are now gone.
//
// It survives HERE, and only here, because QDOCS_S1 runs it as a CONTROL ARM
// against the free path through the shipped driver — refuted machinery kept
// exercised and measurable rather than deleted, the standing precedent. Nothing
// on /v1/extract can reach it.
//
// (This engine's parse_term has no `.` wildcard, so the escape-alt is `"\\" [^]`
// — an empty negated class = any char. A pure syntax substitution.)
static const char* LENS_GBNF =
    "root  ::= \"{\" ws pair (\",\" ws pair)* ws \"}\"\n"
    "pair  ::= \"\\\"\" key \"\\\"\" ws \":\" ws \"\\\"\" value \"\\\"\"\n"
    "key   ::= [a-z] [a-z0-9_]*\n"
    "value ::= ([^\"\\\\\\n] | \"\\\\\" [^])+\n"
    "ws    ::= [ \\n\\t]*\n";

const char* lens_grammar_gbnf() { return LENS_GBNF; }

// ═════════════════════════════════════════════════════════════════════════
// Pure computation
// ═════════════════════════════════════════════════════════════════════════
namespace {

// Extract the KEY strings (in order) from a flat-object JSON string: every "key"
// immediately preceding a ':'. Tolerant of the grammar's well-formed output.
// (attn_provenance.cpp qdocs_keys, verbatim behaviour.)
std::vector<std::string> lens_keys(const std::string& json) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < json.size()) {
        if (json[i] != '"') { i++; continue; }
        size_t k0 = i + 1, k1 = json.find('"', k0);
        if (k1 == std::string::npos) break;
        size_t j = k1 + 1;
        while (j < json.size() && (json[j] == ' ' || json[j] == '\t' || json[j] == '\n')) j++;
        if (j < json.size() && json[j] == ':') {
            out.push_back(json.substr(k0, k1 - k0));
            size_t v = j + 1;
            while (v < json.size() && (json[v] == ' ' || json[v] == '\t' || json[v] == '\n')) v++;
            if (v < json.size() && json[v] == '"') {
                size_t ve = json.find('"', v + 1);
                i = (ve == std::string::npos) ? json.size() : ve + 1;
            } else { i = j + 1; }
        } else { i = k1 + 1; }
    }
    return out;
}

// The value string that follows `"key"` in `json`, plus its byte span [vb0,vb1).
// Mirrors parse_fields' value scan (quoted, else until , } or newline).
//
// `is_null` ⇒ the value was the UNQUOTED JSON literal `null`, which is a
// DECLINE and NOT a value. This distinction is load-bearing: freed of the
// grammar, this model spontaneously emits `"payment_terms": null` for a concept
// the document lacks (predicted in docs/note-nogrammar-refutation.md, then
// observed live). A text scanner reads that as the 4-char string "null" and
// hands back a non-empty value — turning the model's CORRECT decline into a
// fabricated field. That is the same class of self-inflicted wound the grammar
// caused, one layer down. A *quoted* "null" is left alone: that is a document
// that really says "null".
// One value the model emitted under a key, with its byte span in `json`.
struct LensValueSpan { std::string value; size_t vb0 = 0, vb1 = 0; bool is_null = false; };

// Read ONE scalar starting at `i`, returning its span. A quoted string spans the
// quotes' interior; anything else runs to the first delimiter. `stop_at_bracket`
// is set when reading inside an array, where ']' also ends a value.
LensValueSpan lens_read_scalar(const std::string& json, size_t i, bool stop_at_bracket) {
    LensValueSpan v;
    if (i < json.size() && json[i] == '"') {
        v.vb0 = i + 1;
        v.vb1 = json.find('"', v.vb0);
        if (v.vb1 == std::string::npos) v.vb1 = json.size();
    } else {
        v.vb0 = i;
        size_t j = i;
        while (j < json.size() && json[j] != ',' && json[j] != '}' && json[j] != '\n' &&
               !(stop_at_bracket && json[j] == ']')) j++;
        v.vb1 = j;
        while (v.vb1 > v.vb0 && (json[v.vb1 - 1] == ' ' || json[v.vb1 - 1] == '\r')) v.vb1--;
        v.is_null = (json.compare(v.vb0, v.vb1 - v.vb0, "null") == 0);
    }
    v.value = json.substr(v.vb0, v.vb1 - v.vb0);
    return v;
}

// Every value the model emitted for the `occurrence`-th appearance of `key`
// (0-based, text order), each with its own byte span.
//
// THREE shapes, and the distinction is the whole point:
//   scalar            -> one value. The ordinary case.
//   array of SCALARS  -> one value PER ELEMENT. Each element is a locatable
//                        scalar with its own span, so the per-value-span trust
//                        math gives each one its own real citations — exactly
//                        what repeated keys get. Refusing these would refuse a
//                        shape we can account for faithfully. Measured need:
//                        asked for a flat schema on a three-line invoice,
//                        Qwen3.8-9B repeats the key while Qwen3.6-35B answers
//                        with `"quantity": [7, 19, 43]`. Same document, same
//                        question, two encodings of one answer.
//   anything deeper   -> REFUSED via `is_nonscalar` (an object value, or an
//                        array holding objects/arrays). Those have no locatable
//                        scalar identity: `lens_keys` is depth-blind so inner
//                        keys leak as top-level fields, and a fragment scanned
//                        to the first comma would ship as a real field with a
//                        badge and citations pointing where the model did not
//                        read (measured: LensNestedOutput).
//
// An empty array, and a null element, yield no value: the model declining, which
// apply_absent_by_omission renders as absent. Never a fabricated blank.
bool lens_values_of(const std::string& json, const std::string& key, size_t occurrence,
                    std::vector<LensValueSpan>& out, bool& is_nonscalar) {
    is_nonscalar = false;
    const std::string needle = "\"" + key + "\"";
    size_t kp = std::string::npos, from = 0;
    for (size_t n = 0; n <= occurrence; ++n) {
        kp = json.find(needle, from);
        if (kp == std::string::npos) return false;
        from = kp + needle.size();
    }
    size_t colon = json.find(':', kp);
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) i++;
    if (i >= json.size()) return false;

    if (json[i] == '{') { is_nonscalar = true; return true; }

    if (json[i] == '[') {
        size_t j = i + 1;
        while (j < json.size()) {
            while (j < json.size() && (json[j] == ' ' || json[j] == '\t' ||
                                       json[j] == '\n' || json[j] == '\r')) j++;
            if (j >= json.size()) break;
            if (json[j] == ']') break;                       // end of array
            if (json[j] == '{' || json[j] == '[') {           // an element we cannot locate
                is_nonscalar = true;
                return true;
            }
            LensValueSpan v = lens_read_scalar(json, j, /*stop_at_bracket=*/true);
            if (!v.is_null && !v.value.empty()) out.push_back(v);
            j = (json[j] == '"') ? v.vb1 + 1 : v.vb1;         // past the closing quote, if any
            while (j < json.size() && json[j] != ',' && json[j] != ']') j++;
            if (j < json.size() && json[j] == ']') break;
            j++;                                              // past the comma
        }
        return true;
    }

    LensValueSpan v = lens_read_scalar(json, i, /*stop_at_bracket=*/false);
    if (!v.is_null) out.push_back(v);
    return true;
}

bool has_alnum(const std::string& s) {
    for (unsigned char c : s) if (std::isalnum(c)) return true;
    return false;
}

}  // namespace

// ── The shape contract: tolerant on shape, loud on failure ───────────────────
bool lens_find_json_object(const std::string& raw, size_t& lo, size_t& hi) {
    // Tolerant step 1: skip a ``` fence (with or without an info string).
    size_t i = raw.find_first_not_of(" \t\r\n");
    if (i == std::string::npos) return false;
    if (raw.compare(i, 3, "```") == 0) {
        const size_t nl = raw.find('\n', i);
        if (nl != std::string::npos) i = nl + 1;  // drop the ``` and any "json" info string
    }
    // Tolerant step 2: the OUTERMOST object, by brace depth. String- and
    // escape-aware — a `{` inside a value (an HTML mail quoting a JSON payload,
    // r_html in the S1.4 corpus) must not open a level, or the span would end in
    // the wrong place and every byte offset after it would be wrong.
    const size_t start = raw.find('{', i);
    if (start == std::string::npos) return false;
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t p = start; p < raw.size(); ++p) {
        const char c = raw[p];
        if (in_str) {
            if (esc)            esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"')  in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') depth++;
        else if (c == '}' && --depth == 0) { lo = start; hi = p + 1; return true; }
    }
    return false;  // never closed — ran off the token budget mid-object
}

namespace {

// Locate the emitted object or FAIL LOUDLY. Named error: endpoint, expectation,
// actual (CLAUDE.md's fail-loud contract). Never a partial extraction — the
// grammar's whole sin was corrupting output to avoid failing.
void lens_locate_or_throw(const std::string& raw, size_t& lo, size_t& hi) {
    if (!lens_find_json_object(raw, lo, hi))
        throw LensUnparseableError(
            "/v1/extract: expected the model to emit a JSON object (a ``` fence and "
            "surrounding prose are tolerated) actual=no complete {...} found in " +
            std::to_string(raw.size()) + " bytes of output — extraction refused, "
            "never partially reported", raw);
    const std::string obj = raw.substr(lo, hi - lo);
    try {
        nlohmann::json::parse(obj);
    } catch (const std::exception& e) {
        throw LensUnparseableError(
            "/v1/extract: expected the emitted JSON object to parse actual=" +
            std::string(e.what()) + " — extraction refused, never partially reported",
            raw);
    }
}

}  // namespace

// A5.3 tier heuristic (see header). The ≤6-digit cut matches the probe corpus's
// bare quantities (45, 875, 2500, 7781); longer all-digit IDs carry enough bits
// to behave distinctively and are not exempted.
std::string lens_value_tier(const std::string& value) {
    if (value.empty()) return "";
    if (value.size() <= 6) {
        bool all_digits = true;
        for (unsigned char c : value) if (!std::isdigit(c)) { all_digits = false; break; }
        if (all_digits) return "short_numeric";
    }
    return "distinctive";
}

// ── A5.4 fidelity predicates (see header) ────────────────────────────────────
bool lens_cites_a_real_source(const std::string& document, const std::string& value,
                              size_t c_lo, size_t c_hi, long tol) {
    if (value.empty()) return false;
    for (size_t occ = document.find(value); occ != std::string::npos;
         occ = document.find(value, occ + 1)) {
        const long o_lo = (long)occ, o_hi = (long)(occ + value.size());
        if ((long)c_lo < o_hi + tol && (long)c_hi > o_lo - tol) return true;
    }
    return false;
}

int lens_count_confident_false_receipts(const LensReport& r, long tol) {
    int n = 0;
    for (const auto& f : r.fields) {
        if (!f.grounded) continue;                          // not a confident claim
        if (lens_value_tier(f.value) == "short_numeric") continue;  // weak class: no claim made
        if (!f.found_in_document) continue;                 // already disclosed
        if (f.citations.empty() ||
            !lens_cites_a_real_source(r.document_text, f.value,
                                      f.citations[0].byte_lo, f.citations[0].byte_hi, tol))
            n++;
    }
    return n;
}

LensReport compute_lens_report(const LensRun& run, const LensConstants& k) {
    const int P  = (int)run.prompt_text.size();
    const int G  = (int)run.gen_tok_text.size();
    const int nh = run.n_head;

    if ((int)run.steps.size() != G)
        throw std::runtime_error(
            "compute_lens_report: steps count expected=" + std::to_string(G) +
            " (one per gen token) actual=" + std::to_string(run.steps.size()));
    if ((int)run.prompt_cum.size() != P + 1)
        throw std::runtime_error(
            "compute_lens_report: prompt_cum length expected=" + std::to_string(P + 1) +
            " actual=" + std::to_string(run.prompt_cum.size()));
    if (run.doc_lo < 0 || run.doc_hi > P || run.doc_lo > run.doc_hi)
        throw std::runtime_error(
            "compute_lens_report: document token range expected 0<=doc_lo<=doc_hi<=" +
            std::to_string(P) + " actual=[" + std::to_string(run.doc_lo) + "," +
            std::to_string(run.doc_hi) + "]");

    const int doc_lo = run.doc_lo, doc_hi = run.doc_hi;

    LensReport r;
    r.model              = run.model;
    r.validated_envelope = run.validated_envelope;
    r.k                  = k;
    r.prompt_len         = P;
    r.doc_lo             = doc_lo;
    r.doc_hi             = doc_hi;
    r.document_text      = run.document;
    r.raw_json           = run.gen_text;

    // Document-relative byte span of a prompt position (for importer citations).
    auto doc_bytes = [&](int p, size_t& blo, size_t& bhi) {
        size_t plo = run.prompt_cum[p], phi = run.prompt_cum[p + 1];
        blo = plo > run.doc_byte_offset ? plo - run.doc_byte_offset : 0;
        bhi = phi > run.doc_byte_offset ? phi - run.doc_byte_offset : 0;
    };

    // Which request message a DOCUMENT-relative byte falls in. -1 when the
    // caller sent a plain `document` (no boundaries), or when the byte lands
    // before the first message. Pure lookup — no claim attached; see the
    // LensCitation comment.
    r.n_messages = (int)run.message_offsets.size();
    auto msg_of = [&](size_t doc_byte) -> int {
        if (run.message_offsets.empty()) return -1;
        auto it = std::upper_bound(run.message_offsets.begin(), run.message_offsets.end(), doc_byte);
        return it == run.message_offsets.begin() ? -1 : (int)(it - run.message_offsets.begin()) - 1;
    };

    for (int p = 0; p < P; ++p)
        r.prompt.push_back({p, run.prompt_text[p],
                            (p >= doc_lo && p < doc_hi) ? "body" : "instr"});
    r.gen = run.gen_tok_text;

    auto row_ptr = [&](const std::vector<float>& row, int head, int n_kv) -> const float* {
        if ((int)row.size() < (head + 1) * n_kv)
            throw std::runtime_error(
                "compute_lens_report: tap row width expected>=" +
                std::to_string((head + 1) * n_kv) + " actual=" + std::to_string(row.size()));
        return row.data() + (size_t)head * n_kv;
    };

    // ── hover: per-gen-token L3H13 top-k source positions (N3) ────────────────
    r.hover.resize(G);
    for (int t = 1; t < G; ++t) {
        const LensStep& st = run.steps[t - 1];
        const int n_kv = st.n_kv;
        const float* rr = row_ptr(st.citation_row, k.citation_head, n_kv);
        std::vector<std::pair<int, double>> v;
        for (int p = 1; p < P && p < n_kv; ++p) v.push_back({p, rr[p]});  // excl idx-0 sink
        const int topk = std::min((int)v.size(), k.citation_topk);
        std::partial_sort(v.begin(), v.begin() + topk, v.end(),
                          [](auto& a, auto& b) { return a.second > b.second; });
        for (int i = 0; i < topk; ++i) {
            int p = v[i].first;
            size_t dblo, dbhi; doc_bytes(p, dblo, dbhi);
            const bool in_doc = p >= doc_lo && p < doc_hi;
            r.hover[t].push_back({p, v[i].second, run.prompt_cum[p], run.prompt_cum[p + 1],
                                  in_doc ? msg_of(dblo) : -1});
        }
    }

    // ── coverage: layer-11 attention mass, max over heads (COV1) ─────────────
    // One shared quantity, two different reductions over it:
    //   mh[t][p] = the largest layer-11 mass any head puts on prompt position
    //              p while decoding gen token t.
    //   heat[p]  = max over steps — per-position intensity, for the viewer.
    //   span-peak of a document line = max over steps of the SUM of mh[t][p]
    //              across that line's positions. This is the COV1 signal: a
    //              line below coverage_used_peak was never consulted.
    r.heat.assign(P, 0.0);
    std::vector<std::vector<double>> mh(G, std::vector<double>(P, 0.0));
    for (int t = 0; t < G; ++t) {
        const LensStep& st = run.steps[t];
        const int n_kv = st.n_kv;
        for (int p = 0; p < P && p < n_kv; ++p) {
            double m = 0.0;
            for (int h = 0; h < nh; ++h) {
                double val = row_ptr(st.coverage_row, h, n_kv)[p];
                if (val > m) m = val;
            }
            mh[t][p] = m;
            if (m > r.heat[p]) r.heat[p] = m;
        }
    }

    // ── skipped: document lines whose span-peak < used_peak (real content) ────
    int lstart = doc_lo;
    for (int p = doc_lo; p < doc_hi; ++p) {
        const bool nl = run.prompt_text[p].find('\n') != std::string::npos;
        if (!nl && p != doc_hi - 1) continue;
        const int lo = lstart, hi = p;
        double span_peak = 0.0;
        for (int t = 0; t < G; ++t) {
            double s = 0.0;
            for (int q = lo; q <= hi; ++q) s += mh[t][q];
            if (s > span_peak) span_peak = s;
        }
        std::string text;
        for (int q = lo; q <= hi; ++q) text += run.prompt_text[q];
        if (span_peak < k.coverage_used_peak && has_alnum(text)) {
            size_t blo, bhi; doc_bytes(lo, blo, bhi); size_t bhi2; doc_bytes(hi, bhi, bhi2);
            r.skipped.push_back({lo, hi, span_peak, text, blo, bhi2});
        }
        lstart = p + 1;
    }

    // ── fields: parse keys, badge (body_mass, N3b), aggregated citations ──────
    // The shape contract: locate the outermost object (tolerating a fence/prose)
    // and validate it, or refuse loudly. Free output is not shape-guaranteed the
    // way the grammar pretended to be, so this is where that pretence is replaced.
    size_t obj_lo = 0, obj_hi = run.gen_text.size();
    lens_locate_or_throw(run.gen_text, obj_lo, obj_hi);
    const std::string obj = run.gen_text.substr(obj_lo, obj_hi - obj_lo);

    // Byte offsets from `obj` are relative to the OBJECT; the gen-token span math
    // (gen_cum) is relative to `gen_text`. Shift by obj_lo or a fenced/prefaced
    // response silently mis-locates every value span — which would be the lens
    // lying about where the model looked, the one thing it must never do.
    // lens_keys returns keys in emission order, duplicates included. Two counters,
    // because they count different things: `key_textual` walks the appearances of
    // the key in the output (so each reads ITS OWN, not all the first), while
    // `key_occurrence` numbers the VALUES that come back — and one appearance can
    // carry several, when the model answers with an array of scalars.
    std::map<std::string, int> key_textual, key_occurrence;
    for (const std::string& key : lens_keys(obj)) {
        std::vector<LensValueSpan> vals;
        bool is_nonscalar = false;
        const size_t textual = (size_t)(key_textual[key]++);
        auto empty_field = [&]() {
            LensField f;
            f.key = key;
            f.occurrence = key_occurrence[key]++;
            r.fields.push_back(f);
        };
        if (!lens_values_of(obj, key, textual, vals, is_nonscalar)) { empty_field(); continue; }
        // A value we cannot locate is REFUSED, not summarized, not partially
        // reported. The lens claims one thing — a faithful record of where the
        // model looked — and it cannot make that claim about a value it did not
        // read. Every softer option is worse: a fragment scanned to the first
        // comma is a wrong value wearing a real badge, and dropping the key
        // silently is the data loss the shape contract exists to prevent. 422
        // says "route this document to a human" (lens-format.md).
        //
        // Note what this does NOT refuse any more: an array of SCALARS, which is
        // handled above as one value per element. Each element is locatable, so
        // each gets its own real citations. Refusing those refused a shape we can
        // account for faithfully — and it is the shape Qwen 3.6-35B chooses for
        // exactly the document Qwen 3.8-9B answers with repeated keys.
        if (is_nonscalar)
            throw LensUnparseableError(
                "/v1/extract: expected a scalar or an array of scalars for key \"" + key +
                "\" actual=a nested object or an array containing one — extraction "
                "refused, never partially reported: a nested value cannot be located "
                "in the document, so its citations would point somewhere the model "
                "did not read",
                run.gen_text);
        // No values = the model declining (JSON null, or an empty array). Absent
        // by omission; claiming the literal "null" would fabricate a field the
        // model explicitly refused.
        if (vals.empty()) { empty_field(); continue; }

        for (const LensValueSpan& vs : vals) {
        LensField f;
        f.key = key;
        f.occurrence = key_occurrence[key]++;
        const std::string& value = vs.value;
        const size_t vb0 = vs.vb0 + obj_lo, vb1 = vs.vb1 + obj_lo;
        f.value = value;
        f.tier  = lens_value_tier(value);
        for (int g = 0; g < G; ++g)
            if (run.gen_cum[g] < vb1 && run.gen_cum[g + 1] > vb0) {
                if (f.gen_lo < 0) f.gen_lo = g;
                f.gen_hi = g;
            }
        size_t sb = run.document.find(value);
        if (!value.empty() && sb != std::string::npos) {
            f.found_in_document = true;
            f.value_byte_lo = sb;                    // document-relative
            f.value_byte_hi = sb + value.size();
        }

        // body_mass + aggregated per-position citations over the value tokens,
        // restricted to the document token range (where N3b was calibrated).
        const int blo = std::max(1, doc_lo);
        std::vector<double> agg(P, 0.0);
        double body_mass_sum = 0.0; int nval = 0;
        for (int v = f.gen_lo; v >= 0 && v <= f.gen_hi; ++v) {
            const int step = v - 1;
            if (step < 0 || step >= G) continue;
            const LensStep& st = run.steps[step];
            const int n_kv = st.n_kv;
            const float* rr = row_ptr(st.citation_row, k.citation_head, n_kv);
            double bm = 0.0;
            for (int j = blo; j < doc_hi && j < n_kv; ++j) {
                bm += rr[j];
                agg[j] += rr[j];
            }
            body_mass_sum += bm;
            nval++;
        }
        f.body_mass = nval ? body_mass_sum / nval : 0.0;
        f.grounded  = f.body_mass >= k.ungrounded_body_mass;

        std::vector<std::pair<int, double>> av;
        for (int p = blo; p < doc_hi; ++p) if (agg[p] > 0.0) av.push_back({p, agg[p]});
        const int topk = std::min((int)av.size(), k.citation_topk);
        std::partial_sort(av.begin(), av.begin() + topk, av.end(),
                          [](auto& a, auto& b) { return a.second > b.second; });
        for (int i = 0; i < topk; ++i) {
            int p = av[i].first;
            size_t cblo, cbhi; doc_bytes(p, cblo, cbhi);
            f.citations.push_back({p, av[i].second / std::max(1, nval), cblo, cbhi, msg_of(cblo)});
        }
        // Distinct messages, first-seen order — and `av` is already sorted by
        // descending mass, so that IS descending citation mass. One element is
        // the ordinary case ("read from message 23"); several mean the model
        // looked at this key in several messages, reported side by side with no
        // winner named (the CF1 non-claim).
        for (const LensCitation& ct : f.citations)
            if (ct.message >= 0 && std::find(f.citation_messages.begin(),
                                             f.citation_messages.end(),
                                             ct.message) == f.citation_messages.end())
                f.citation_messages.push_back(ct.message);
        r.fields.push_back(f);
        }
    }

    return r;
}

// ═════════════════════════════════════════════════════════════════════════
// Serialization
// ═════════════════════════════════════════════════════════════════════════
namespace {
std::string jesc(const std::string& s) {
    std::string o; o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;   case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;   case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}
std::string fnum(double v) { char b[32]; std::snprintf(b, sizeof(b), "%.4f", v); return b; }

// ── Candidate-set wire helpers (docs/plan-candidate-set.md) ──────────────────
// `anchor` and `returned_as` are DERIVED at serialization time, not stored on
// LensCandidate — they depend on `document` and on the sibling `fields`
// entries, and recomputing them here keeps the producer (lens_apply_pass2_candidates)
// untouched, per this change's mandate to build on top of it, not inside it.

std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

// Collapse runs of whitespace to one space and trim. Used ONLY to decide
// whether two spans are "the same answer" for `returned_as` linking — never
// changes what is stored or shown, which stays the raw byte-exact text.
std::string ws_normalize(const std::string& s) {
    std::string o; o.reserve(s.size());
    bool sp = false;
    for (unsigned char c : s) {
        if (std::isspace(c)) { sp = true; continue; }
        if (sp && !o.empty()) o += ' ';
        sp = false;
        o += (char)c;
    }
    return o;
}

bool is_underline_rule(const std::string& t) {
    if (t.size() < 3) return false;
    char c0 = t[0];
    if (c0 != '-' && c0 != '=') return false;
    for (char c : t) if (c != c0) return false;
    return true;
}

// A "label-like" line (docs/plan-candidate-set.md's anchor rule): short, and
// either ends in ':' or reads as a numbered heading ("2. RENT", "IV. Terms").
bool is_label_like_line(const std::string& t) {
    if (t.empty() || t.size() > 60) return false;
    if (t.back() == ':') return true;
    size_t i = 0;
    while (i < t.size() && std::isdigit((unsigned char)t[i])) i++;
    if (i == 0)
        while (i < t.size() && std::string("IVXLCM").find(t[i]) != std::string::npos) i++;
    return i > 0 && i < t.size() && (t[i] == '.' || t[i] == ')') &&
           i + 1 < t.size() && std::isspace((unsigned char)t[i + 1]);
}

// anchor derivation: nearest preceding label-like line (an underlined heading
// counts via the line before a bare '---'/'===' rule), else the enclosing
// line trimmed+truncated, else no anchor at all. `value_norm` is the
// candidate's own whitespace-normalized text: an enclosing line that carries
// nothing beyond the span itself is not a place, it is the span — so that
// case falls through to "no anchor" too (Nullable is deliberate — a byte
// offset is not a place, and inventing one would be dishonest).
bool derive_anchor(const std::string& doc, size_t byte_lo, const std::string& value_norm,
                    std::string& out) {
    if (byte_lo > doc.size()) byte_lo = doc.size();
    std::vector<std::pair<size_t, size_t>> lines;
    size_t start = 0;
    for (size_t i = 0; i <= doc.size(); ++i)
        if (i == doc.size() || doc[i] == '\n') { lines.emplace_back(start, i); start = i + 1; }

    size_t idx = 0;
    for (; idx < lines.size(); ++idx)
        if (byte_lo >= lines[idx].first && byte_lo <= lines[idx].second) break;
    if (idx == lines.size()) idx = lines.empty() ? 0 : lines.size() - 1;
    if (lines.empty()) return false;

    for (long j = (long)idx - 1; j >= 0; --j) {
        std::string t = trim_ws(doc.substr(lines[j].first, lines[j].second - lines[j].first));
        if (is_underline_rule(t)) {
            if (j - 1 >= 0) {
                std::string h = trim_ws(doc.substr(lines[j - 1].first,
                                                   lines[j - 1].second - lines[j - 1].first));
                if (!h.empty() && h.size() <= 60) { out = h; return true; }
            }
            continue;   // the underline itself is a separator, not a label
        }
        if (is_label_like_line(t)) { out = t; return true; }
    }

    std::string enclosing = trim_ws(doc.substr(lines[idx].first, lines[idx].second - lines[idx].first));
    if (enclosing.empty() || ws_normalize(enclosing) == value_norm) return false;
    out = enclosing.size() > 60 ? enclosing.substr(0, 60) : enclosing;
    return true;
}

// returned_as linking: bidirectional containment on whitespace-normalized
// text (pass 2 legitimately returns wider or narrower spans than the field's
// value — docs/plan-candidate-set.md "Linking"). Tiebreak: tightest
// containment first (smallest length gap between the candidate and the
// value it is being linked to), then earliest byte_lo. Greedy, one
// occurrence at a time in occurrence order, each candidate claimed at most
// once — a candidate's `returned_as` is a single occurrence-or-null, so it
// cannot simultaneously answer two occurrences.
std::vector<int> compute_returned_as(const std::vector<LensCandidate>& cands,
                                     const std::vector<LensField>& fields,
                                     const std::string& key) {
    std::vector<int> returned_as(cands.size(), -1);
    struct Occ { int occurrence; std::string norm; };
    std::vector<Occ> occs;
    for (const LensField& f : fields)
        if (f.key == key && f.present && !f.value.empty())
            occs.push_back({f.occurrence, ws_normalize(f.value)});
    std::stable_sort(occs.begin(), occs.end(),
                     [](const Occ& a, const Occ& b) { return a.occurrence < b.occurrence; });

    std::vector<char> claimed(cands.size(), 0);
    for (const Occ& occ : occs) {
        long best = -1, best_gap = -1;
        for (size_t ci = 0; ci < cands.size(); ++ci) {
            if (claimed[ci]) continue;
            std::string cnorm = ws_normalize(cands[ci].value);
            bool contains = cnorm.find(occ.norm) != std::string::npos ||
                           occ.norm.find(cnorm) != std::string::npos;
            if (!contains) continue;
            long gap = std::labs((long)cnorm.size() - (long)occ.norm.size());
            if (best < 0 || gap < best_gap ||
                (gap == best_gap && cands[ci].byte_lo < cands[(size_t)best].byte_lo)) {
                best = (long)ci; best_gap = gap;
            }
        }
        if (best >= 0) { claimed[(size_t)best] = 1; returned_as[(size_t)best] = occ.occurrence; }
    }
    return returned_as;
}
}  // namespace

std::string lens_report_to_json(const LensReport& r) {
    std::string o;
    o += "{\n";
    o += "\"format_version\":\"" + jesc(r.format_version) + "\",\n";
    o += "\"model\":\"" + jesc(r.model) + "\",\n";
    o += "\"validated_envelope\":" + std::string(r.validated_envelope ? "true" : "false") + ",\n";
    // Derived from the calibration entry, never literal: these lines said
    // "layer 3, head 13" on every report until per-model constants landed, which
    // a Qwen 3.8 extraction (L27H13) would have carried as a false receipt —
    // the same defect class as the hardcoded `model` field.
    o += "\"citation_source\":\"layer " + std::to_string(r.k.citation_layer) + ", head " +
         std::to_string(r.k.citation_head) + " (L" + std::to_string(r.k.citation_layer) + "H" +
         std::to_string(r.k.citation_head) + ") \\u2014 N3\",\n";
    o += "\"coverage_source\":\"layer " + std::to_string(r.k.coverage_layer) +
         ", max over heads \\u2014 COV1\",\n";
    o += "\"used_threshold\":" + fnum(r.k.coverage_used_peak) + ",\n";
    o += "\"ungrounded_threshold\":" + fnum(r.k.ungrounded_body_mass) + ",\n";
    o += "\"prompt_len\":" + std::to_string(r.prompt_len) +
         ",\"doc_lo\":" + std::to_string(r.doc_lo) +
         ",\"doc_hi\":" + std::to_string(r.doc_hi) +
         ",\"n_messages\":" + std::to_string(r.n_messages) + ",\n";
    o += "\"document\":\"" + jesc(r.document_text) + "\",\n";
    o += "\"raw\":\"" + jesc(r.raw_json) + "\",\n";

    // fields
    o += "\"fields\":[";
    for (size_t i = 0; i < r.fields.size(); ++i) {
        const LensField& f = r.fields[i];
        o += (i ? "," : "");
        o += "{\"key\":\"" + jesc(f.key) + "\",";
        // ABSENT (Stage 2): the model did not state this hinted concept, so it
        // serializes value:null / badge:"absent". Same importer-facing shape the
        // presence gate produced — now earned mechanically by omission from the
        // parsed output rather than by a yes/no verdict pass.
        o += "\"value\":" + (f.present ? ("\"" + jesc(f.value) + "\"") : std::string("null")) + ",";
        o += "\"badge\":\"" + std::string(f.present ? (f.grounded ? "grounded" : "ungrounded")
                                                    : "absent") + "\",";
        // A5.3: machine-readable trust tier of the value (distinctive|short_numeric);
        // null on absent (no value to class).
        o += "\"tier\":" + (f.tier.empty() ? std::string("null")
                                           : "\"" + jesc(f.tier) + "\"") + ",";
        o += "\"occurrence\":" + std::to_string(f.occurrence) + ",";
        o += "\"body_mass\":" + fnum(f.body_mass) + ",";
        o += "\"found_in_document\":" + std::string(f.found_in_document ? "true" : "false") + ",";
        if (f.found_in_document)
            o += "\"value_span\":{\"lo\":" + std::to_string(f.value_byte_lo) +
                 ",\"hi\":" + std::to_string(f.value_byte_hi) + "},";
        else
            o += "\"value_span\":null,";
        o += "\"citations\":[";
        for (size_t c = 0; c < f.citations.size(); ++c) {
            const LensCitation& ct = f.citations[c];
            o += (c ? "," : "");
            o += "{\"pos\":" + std::to_string(ct.pos) + ",\"mass\":" + fnum(ct.mass) +
                 ",\"byte_lo\":" + std::to_string(ct.byte_lo) +
                 ",\"byte_hi\":" + std::to_string(ct.byte_hi) +
                 ",\"message\":" + (ct.message >= 0 ? std::to_string(ct.message)
                                                   : std::string("null")) + "}";
        }
        o += "],";
        o += "\"citation_messages\":[";
        for (size_t m = 0; m < f.citation_messages.size(); ++m)
            o += (m ? "," : "") + std::to_string(f.citation_messages[m]);
        o += "]}";
    }
    o += "],\n";

    // ── Candidate set (docs/plan-candidate-set.md, qemmi-lens/v4) ────────────
    // Three top-level states, distinguishable by a reader that never has to
    // guess: `key_candidates` present = pass 2 ran; absent + `candidates_error`
    // = the finder failed; absent + no error = candidates were not requested
    // for this extract. A producer failure must NEVER render as an empty,
    // unflagged set — that is the exact confusion this feature exists to fix,
    // one level down (see LensReport::candidates_producer_failed).
    if (r.candidates_producer_failed) {
        o += "\"candidates_error\":\"" + jesc(r.candidates_error) + "\",\n";
    } else if (r.candidates_requested) {
        // Vocabulary = every key `fields` covers, in first-seen (request) order
        // — apply_absent_by_omission guarantees one field entry per requested
        // key, so this is the complete key set, not just the ones with hits.
        std::vector<std::string> vocab;
        {
            std::set<std::string> seen;
            for (const LensField& f : r.fields)
                if (seen.insert(f.key).second) vocab.push_back(f.key);
        }
        static const std::vector<LensCandidate> kNoCands;
        o += "\"key_candidates\":{";
        for (size_t vi = 0; vi < vocab.size(); ++vi) {
            const std::string& key = vocab[vi];
            auto it = r.key_candidates.find(key);
            const std::vector<LensCandidate>& cands =
                (it != r.key_candidates.end()) ? it->second : kNoCands;
            // Every key gets an entry, including one whose array is `[]` — `[]`
            // and absent-from-the-map are different facts (plan §"Shape") and
            // this loop is the reason they cannot be conflated on the wire.
            const std::vector<int> returned_as = compute_returned_as(cands, r.fields, key);
            o += (vi ? "," : "");
            o += "\"" + jesc(key) + "\":[";
            // Array order is document order (byte_lo ascending, already the
            // producer's sort) and that order is LOAD-BEARING — not mass, not
            // any ranking. Position is a fact about the document; any other
            // ordering would be a verdict, and CF1 forbids verdicts.
            for (size_t ci = 0; ci < cands.size(); ++ci) {
                const LensCandidate& c = cands[ci];
                std::string anchor;
                const bool has_anchor = derive_anchor(r.document_text, c.byte_lo,
                                                      ws_normalize(c.value), anchor);
                o += (ci ? "," : "");
                o += "{\"value\":\"" + jesc(c.value) + "\",";
                o += "\"byte_lo\":" + std::to_string(c.byte_lo) +
                     ",\"byte_hi\":" + std::to_string(c.byte_hi) + ",";
                o += "\"anchor\":" + (has_anchor ? ("\"" + jesc(anchor) + "\"")
                                                 : std::string("null")) + ",";
                o += "\"returned_as\":" + (returned_as[ci] >= 0
                                               ? std::to_string(returned_as[ci])
                                               : std::string("null"));
                o += "}";
            }
            o += "]";
        }
        o += "},\n";
    }
    // else: want_candidates was false for this extract — key_candidates and
    // candidates_error both stay absent, meaning "not requested" (distinct
    // from a producer failure, which sets candidates_error with no set).

    // viewer data (demo-compatible)
    o += "\"prompt\":[";
    for (size_t i = 0; i < r.prompt.size(); ++i) {
        const LensPromptToken& p = r.prompt[i];
        o += (i ? "," : "");
        o += "{\"pos\":" + std::to_string(p.pos) + ",\"text\":\"" + jesc(p.text) +
             "\",\"region\":\"" + p.region + "\"}";
    }
    o += "],\n";

    o += "\"gen\":[";
    for (size_t i = 0; i < r.gen.size(); ++i)
        o += (i ? "," : "") + std::string("{\"idx\":") + std::to_string(i) +
             ",\"text\":\"" + jesc(r.gen[i]) + "\"}";
    o += "],\n";

    o += "\"hover\":[";
    for (size_t t = 0; t < r.hover.size(); ++t) {
        o += (t ? "," : "");
        o += "[";
        for (size_t i = 0; i < r.hover[t].size(); ++i) {
            const LensCitation& c = r.hover[t][i];
            o += (i ? "," : "");
            o += "{\"pos\":" + std::to_string(c.pos) + ",\"mass\":" + fnum(c.mass) + "}";
        }
        o += "]";
    }
    o += "],\n";

    o += "\"heat\":[";
    for (size_t p = 0; p < r.heat.size(); ++p) o += (p ? "," : "") + fnum(r.heat[p]);
    o += "],\n";

    o += "\"skipped\":[";
    for (size_t i = 0; i < r.skipped.size(); ++i) {
        const LensCoverageSpan& s = r.skipped[i];
        o += (i ? "," : "");
        o += "{\"lo\":" + std::to_string(s.lo) + ",\"hi\":" + std::to_string(s.hi) +
             ",\"peak\":" + fnum(s.peak) + ",\"text\":\"" + jesc(s.text) +
             "\",\"byte_lo\":" + std::to_string(s.byte_lo) +
             ",\"byte_hi\":" + std::to_string(s.byte_hi) + "}";
    }
    o += "]\n}\n";
    return o;
}

// ═════════════════════════════════════════════════════════════════════════
// Driver
// ═════════════════════════════════════════════════════════════════════════
std::string lens_build_instruction(const std::vector<std::string>& key_vocabulary) {
    // Complete-hint instruction (plan §1.2). Names every target concept — a
    // partial hint reopens the naming zoo for the un-named ones.
    std::string keys;
    for (size_t i = 0; i < key_vocabulary.size(); ++i)
        keys += (i ? ", " : "") + key_vocabulary[i];
    return "\n\nExtract the following fields from the document above into a flat "
           "JSON object of \"key\": \"value\" pairs, using exactly these "
           "snake_case keys: " + keys +
           ". Copy each value verbatim from the document. Output ONLY the JSON "
           "object, nothing else.";
}

// ── Candidate set — pass 2 (docs/plan-candidate-set.md) — ported verbatim ────
// from tests/perf/attn_provenance.cpp's CAND=1 probe (CAND_PASS2_TASK_PREFIX /
// cand_parse_pass2). See server_lens.h for why the tolerance exists.
std::string lens_cand_pass2_instruction(const std::vector<std::string>& keys) {
    static const char* kTaskPrefix =
        "\n\nFor each of the following keys, list every span in the document above "
        "that could answer it — for example a clause and a later amendment that both "
        "give an amount are TWO spans for the same key, not one. Quote each span "
        "EXACTLY as it appears in the document above: do not paraphrase, reformat, "
        "translate, or combine spans. Write one span per line, in the exact form "
        "key: \"span\". A key may have zero, one, or several spans; if it has none, "
        "write key: (none). List every span for one key before moving to the next "
        "key. Output only the list, nothing else. Keys: ";
    std::string task = kTaskPrefix;
    for (size_t i = 0; i < keys.size(); ++i) task += (i ? ", " : "") + keys[i];
    return task;
}

std::vector<std::pair<std::string, std::string>>
lens_parse_pass2_candidates(const std::string& text, const std::vector<std::string>& keys) {
    std::vector<std::pair<std::string, std::string>> out;
    std::set<std::string> keyset(keys.begin(), keys.end());
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        size_t b = 0;
        while (b < line.size() &&
               (std::isspace((unsigned char)line[b]) || line[b] == '-' || line[b] == '*' ||
                std::isdigit((unsigned char)line[b]) || line[b] == '.' || line[b] == ')'))
            b++;
        std::string rest = line.substr(b);
        size_t colon = rest.find(':');
        if (colon == std::string::npos) continue;
        std::string key = rest.substr(0, colon);
        while (!key.empty() && std::isspace((unsigned char)key.back())) key.pop_back();
        if (!keyset.count(key)) continue;
        std::string val = rest.substr(colon + 1);
        std::string span;
        size_t q1 = val.find('"');
        if (q1 != std::string::npos) {
            size_t q2 = val.find('"', q1 + 1);
            if (q2 == std::string::npos) continue;   // unterminated quote — malformed
            span = val.substr(q1 + 1, q2 - q1 - 1);
        } else {
            // TOLERANT PARSE — see server_lens.h for why (m_en1 emitted every
            // line unquoted; a strict parser dropped the lot).
            span = val;
            size_t a = 0;
            while (a < span.size() && std::isspace((unsigned char)span[a])) a++;
            span.erase(0, a);
            while (!span.empty() && std::isspace((unsigned char)span.back())) span.pop_back();
            // "(none)" is a real ANSWER (this key has no candidate), not a parse
            // failure — must not be counted as either a span or a malformed line.
            if (span == "(none)" || span == "none" || span == "-") continue;
        }
        if (!span.empty()) out.emplace_back(key, span);
    }
    return out;
}

void lens_apply_pass2_candidates(const std::string& document, const std::string& gen_text,
                                 const std::vector<std::string>& keys, LensReport& report) {
    auto parsed = lens_parse_pass2_candidates(gen_text, keys);
    if (parsed.empty() && !gen_text.empty()) {
        // FAIL LOUD: a producer failure, not evidence the document offers
        // nothing for every key — the fourth state (see server_lens.h).
        report.candidates_producer_failed = true;
        report.candidates_error =
            "candidate-set pass 2: expected >=1 parseable `key: \"span\"` line from "
            "non-empty output, actual 0 parseable lines from " +
            std::to_string(gen_text.size()) +
            " raw bytes — producer failure, not evidence the document offers nothing";
        return;
    }
    for (auto& kv : parsed) {
        const size_t lo = document.find(kv.second);
        if (lo == std::string::npos) continue;  // not byte-exact — drop, per the format's discipline
        const size_t hi = lo + kv.second.size();
        std::vector<LensCandidate>& cands = report.key_candidates[kv.first];
        bool dup = false;
        for (const LensCandidate& c : cands) if (c.value == kv.second) { dup = true; break; }
        if (dup) continue;
        cands.push_back(LensCandidate{kv.second, lo, hi});
    }
    for (auto& kv : report.key_candidates)
        std::stable_sort(kv.second.begin(), kv.second.end(),
                         [](const LensCandidate& a, const LensCandidate& b) {
                             return a.byte_lo < b.byte_lo;
                         });
}

namespace {
// cum_bytes: token -> cumulative decoded byte offset, via incremental decode
// (byte-level BPE; matches attn_provenance.cpp cum_bytes exactly).
std::vector<size_t> cum_bytes(::Tokenizer* tok, const std::vector<int32_t>& toks) {
    std::vector<size_t> cum(toks.size() + 1, 0);
    for (size_t k = 1; k <= toks.size(); ++k) {
        std::vector<int32_t> pre(toks.begin(), toks.begin() + k);
        cum[k] = tok->decode(pre).size();
    }
    return cum;
}

// Prefill + tapped decode — the extraction core.
// Renders the ChatML thinking-off prompt (document + instruction_suffix), locates
// the document's token range within it, prefills single-slot (slot 0 — the only
// correct qwen36 decode KV gather, architecture.md §12), and runs the tapped
// decode with the citation+coverage taps armed, returning the raw LensRun, which
// compute_lens_report turns into the report.
// The tap is disarmed on return so the default decode path stays byte-inert.
// Fails loud on an empty document or a prompt that exceeds the context.
//
// `grammar == nullptr` ⇒ FREE decode: argmax over the full vocabulary, no
// get_valid_tokens / accept_token / set_sparse_decode_ids, and termination on
// EOS or max_new_tokens alone (there is no accepting state to close on). This is
// the SHIPPED path. Non-null is the QDOCS_S1 control arm ONLY (see the header):
// the tap math and the report are otherwise IDENTICAL, which is what makes the
// two arms comparable — the grammar is the single variable.
//
// Deliberately NO "stop at the first '}'" heuristic (unlike the older probe's
// run_freegen close_char): the caller must see whatever the model freely emits —
// a fence, a preamble, trailing prose — or we would manufacture a parse rate the
// model has not earned. Shape tolerance belongs in the parser, loudly
// (lens_find_json_object / LensUnparseableError), not here.
LensRun run_lens_tapped_decode(ForwardPassBase* fp, ggml_backend_sched_t sched,
                               ::Tokenizer* tok, const ModelMetadata& meta,
                               GrammarVocab* grammar,
                               const std::vector<std::string>& grammar_vocab,
                               uint32_t vocab_size, uint32_t n_ctx_max,
                               const std::string& document,
                               const std::string& instruction_suffix,
                               int max_new_tokens, const LensConstants& k) {
    if (document.empty())
        throw std::runtime_error(
            "run_lens_tapped_decode: document expected non-empty actual=empty");

    // ChatML, thinking off (the production regime; plan §1.4). The document is
    // embedded verbatim so its byte spans locate value sources.
    QwenChatTemplate ct;
    std::vector<ChatMessage> hist = {{"user", document + instruction_suffix}};
    const std::string prompt_text = ct.render(hist, /*add_assistant_prompt=*/true,
                                              /*enable_thinking=*/false);

    std::vector<int32_t> prompt_tokens = tok->encode(prompt_text);
    if ((uint32_t)prompt_tokens.size() >= n_ctx_max)
        throw std::runtime_error(
            "run_lens_tapped_decode: prompt tokens expected < n_ctx_max=" +
            std::to_string(n_ctx_max) + " actual=" + std::to_string(prompt_tokens.size()) +
            " — document too large for the configured context");

    LensRun run;
    run.model              = k.model_label;   // names the calibration entry, not a literal
    run.n_head             = (int)meta.attention_head_count;
    run.document           = document;   // value-source lookups resolve against the doc
    // 4096 is the CALIBRATION floor, not the workload envelope (that is 10 K as
    // of 2026-08-24). The lens constants — L3H13 citations, the body_mass
    // grounded/ungrounded threshold, the coverage bar — were all measured on
    // prompts at or below 4 K, so beyond it the signals still compute but are
    // extrapolated. The flag is a DISCLOSURE on the report, not a refusal; an
    // oversized document is rejected separately, above. Raising this number
    // means re-measuring, not editing it (plan-qemmi-lens.md §1.5).
    run.validated_envelope = prompt_tokens.size() <= 4096;

    // Locate the document's token range within the ChatML-wrapped prompt: the
    // chat header precedes it and the instruction follows, so the document is an
    // interior byte span [doc_pos, doc_end). All lens signals restrict to the
    // tokens overlapping it.
    const size_t doc_pos = prompt_text.find(document);
    if (doc_pos == std::string::npos)
        throw std::runtime_error(
            "run_lens_tapped_decode: document expected to appear verbatim in the "
            "rendered prompt actual=not found — chat template altered it");
    const size_t doc_end = doc_pos + document.size();
    std::vector<size_t> pcum = cum_bytes(tok, prompt_tokens);
    const int P = (int)prompt_tokens.size();
    run.doc_byte_offset = doc_pos;
    run.doc_lo = P; run.doc_hi = 0;
    for (int i = 0; i < P; ++i)
        if (pcum[i] < doc_end && pcum[i + 1] > doc_pos) {  // token i overlaps the document
            if (i < run.doc_lo) run.doc_lo = i;
            run.doc_hi = i + 1;
        }
    if (run.doc_lo > run.doc_hi) { run.doc_lo = 0; run.doc_hi = 0; }
    run.prompt_cum = pcum;
    run.prompt_text.resize(P);
    for (int i = 0; i < P; ++i) run.prompt_text[i] = tok->decode(prompt_tokens[i]);

    // ── Tapped decode, single slot (the probe's run_freegen_grammar sequence,
    //    now using the P1 attention-tap seam). Constrained, or free when
    //    grammar == nullptr. ─────────────────────────────────────────────────
    const bool free_decode = (grammar == nullptr);
    fp->set_attention_taps({k.citation_layer, k.coverage_layer});
    if (!free_decode) grammar->reset();
    fp->clear_slot(0);
    fp->set_cache_pos(0, 0);
    std::vector<float> logits = fp->run_prefill(prompt_tokens, 0, 0, sched);
    const int32_t eos = tok->get_eos_token_id();

    auto argmax_over = [](const std::vector<float>& lg, const std::vector<int32_t>& ids) -> int32_t {
        int32_t best = -1; float bl = -1e30f;
        for (int32_t id : ids)
            if ((size_t)id < lg.size() && lg[id] > bl) { bl = lg[id]; best = id; }
        return best;
    };
    // Free decode: argmax over the real vocabulary. Bounded by vocab_size so a
    // padded logits row can never elect a pad id.
    auto argmax_all = [&](const std::vector<float>& lg) -> int32_t {
        const size_t n = std::min(lg.size(), (size_t)vocab_size);
        int32_t best = -1; float bl = -1e30f;
        for (size_t i = 0; i < n; ++i)
            if (lg[i] > bl) { bl = lg[i]; best = (int32_t)i; }
        return best;
    };

    std::vector<int32_t> valid;
    int32_t next;
    if (free_decode) {
        next = argmax_all(logits);
    } else {
        valid = grammar->get_valid_tokens(grammar_vocab);
        next  = argmax_over(logits, valid);
    }
    std::vector<int32_t> gen_tokens;
    if (next >= 0) {
        if (!free_decode) grammar->accept_token(next, grammar_vocab);
        for (int t = 0; t < max_new_tokens; ++t) {
            const int32_t cur = next;
            gen_tokens.push_back(cur);
            // No accepting state without a grammar — EOS/budget are the only exits.
            const bool closed = free_decode ? false : grammar->is_accepting_state();

            bool use_sparse = false;
            if (!free_decode) {
                valid = grammar->get_valid_tokens(grammar_vocab);
                use_sparse = !valid.empty() && valid.size() < vocab_size / 8;
            }
            // Always set (empty when free) — clears any stale sparse ids.
            fp->set_sparse_decode_ids(use_sparse ? valid : std::vector<int32_t>{});

            std::vector<int32_t>  tks       = {cur};
            std::vector<uint32_t> slots     = {0};
            // Rope position, not the KV row count (they diverge after an
            // image). Identical on this text-only path; kept honest.
            std::vector<int32_t>  positions = {fp->get_rope_pos(0)};
            ggml_cgraph* gf = fp->build_decoding_graph(tks, slots, positions);
            fp->mark_attention_taps(gf);
            ggml_backend_sched_reset(sched);
            ggml_backend_sched_alloc_graph(sched, gf);
            fp->set_decode_inputs(gf, tks, slots, positions);
            qinf::engine::require_compute_success(
                ggml_backend_sched_graph_compute(sched, gf), "lens_tapped_decode");

            std::vector<ForwardPassBase::AttentionTap> taps = fp->get_attention_taps(gf);
            // taps[i] corresponds to attention_taps()[i] — the order this
            // function armed above, NOT layer order. The two coincided while the
            // only calibration was Qwen 3.6's {3, 11} (ascending); Qwen 3.8 arms
            // {27, 11} (descending), so a reordering readback would silently swap
            // the citation and coverage signals and produce a shaped, wrong
            // report. Asserted rather than trusted, in the fail-loud order.
            if (taps.size() != 2 || taps[0].layer != k.citation_layer ||
                taps[1].layer != k.coverage_layer)
                throw std::runtime_error(
                    "lens_tapped_decode: attention taps expected {citation_layer " +
                    std::to_string(k.citation_layer) + ", coverage_layer " +
                    std::to_string(k.coverage_layer) + "}, actual " +
                    (taps.size() != 2 ? std::to_string(taps.size()) + " taps"
                                      : "{" + std::to_string(taps[0].layer) + ", " +
                                        std::to_string(taps[1].layer) + "}"));
            LensStep st;
            st.n_kv         = taps[0].n_kv;
            st.citation_row = std::move(taps[0].rows);   // k.citation_layer
            st.coverage_row = std::move(taps[1].rows);   // k.coverage_layer
            run.steps.push_back(std::move(st));

            std::vector<float> lg = fp->get_output_logits(gf);
            fp->advance_cache(1, 0);

            if (free_decode) { next = argmax_all(lg); }
            else if (valid.empty()) { next = eos; }
            else if (use_sparse) {
                int best_k = -1; float bl = -1e30f;
                for (int i = 0; i < (int)valid.size() && i < (int)lg.size(); ++i)
                    if (lg[i] > bl) { bl = lg[i]; best_k = i; }
                next = best_k >= 0 ? valid[best_k] : eos;
            } else {
                next = argmax_over(lg, valid);
            }
            if (!free_decode && next >= 0 && next != eos) grammar->accept_token(next, grammar_vocab);
            if (closed || next == eos || next < 0) break;
        }
    }
    fp->set_attention_taps({});  // disarm — leave the engine byte-inert for the next request
    // ...and restore the SLOT too, not just the tap. This driver runs slot 0
    // directly (clear_slot/set_cache_pos/run_prefill above), so the
    // InferenceServer's slot lifecycle never learns slot 0 was used and no
    // release ever fires for it. Without this, the slot is left at
    // cache_pos = prompt+generated, and the NEXT request — which prefills at
    // start_pos 0 but calls advance_cache(), which ADDS — decodes with an n_kv
    // inflated by our leftovers and attends over stale lens KV. Measured: an
    // extract left cache_pos=138, the next 24-token request decoded at n_kv=162,
    // and its output differed from the same request run before the extract
    // (exactly one request deep, since that request's own release then cleared
    // the slot). Symmetric with the disarm above: leave the engine as found.
    fp->clear_slot(0);

    run.gen_tok_text.resize(gen_tokens.size());
    for (size_t i = 0; i < gen_tokens.size(); ++i) run.gen_tok_text[i] = tok->decode(gen_tokens[i]);
    run.gen_text = tok->decode(gen_tokens);
    run.gen_cum  = cum_bytes(tok, gen_tokens);
    return run;
}

// Pass 2 of the candidate set (docs/plan-candidate-set.md) — a SEPARATE cold
// inference over the same document with a different instruction, needing no
// citations, so attention taps stay DISARMED for the whole pass (which is also
// what makes flash attention available here, per the plan). This is NOT warm
// reuse of pass 1's KV — the plan explicitly rejects checkpoint/rewind, since
// chunked-vs-one-shot prefill is not bit-identical on Metal and would perturb
// pass 1. Fresh slot: clear_slot + set_cache_pos(0,0) + a full prefill, exactly
// like run_lens_tapped_decode's own reset above, and the same symmetric
// clear_slot(0) on the way out (see the comment on that one for why leaving
// the slot dirty corrupts the next unrelated request).
//
// Free decode only — no grammar, no attention-tap bookkeeping (mark/get_attention_taps
// are never called, because nothing is armed to read). Only called when the
// caller opted into LensExtractOptions::want_candidates; the off path never
// reaches this function at all.
std::string run_cand_pass2_decode(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                  ::Tokenizer* tok, const std::string& document,
                                  const std::vector<std::string>& keys,
                                  uint32_t vocab_size, uint32_t n_ctx_max,
                                  int max_new_tokens) {
    QwenChatTemplate ct;
    std::vector<ChatMessage> hist = {{"user", document + lens_cand_pass2_instruction(keys)}};
    const std::string prompt_text = ct.render(hist, /*add_assistant_prompt=*/true,
                                              /*enable_thinking=*/false);
    std::vector<int32_t> prompt_tokens = tok->encode(prompt_text);
    if ((uint32_t)prompt_tokens.size() >= n_ctx_max)
        throw std::runtime_error(
            "run_cand_pass2_decode: prompt tokens expected < n_ctx_max=" +
            std::to_string(n_ctx_max) + " actual=" + std::to_string(prompt_tokens.size()) +
            " — document too large for the configured context");

    fp->set_attention_taps({});   // disarmed for the whole pass — no citations needed
    fp->clear_slot(0);
    fp->set_cache_pos(0, 0);
    std::vector<float> logits = fp->run_prefill(prompt_tokens, 0, 0, sched);
    const int32_t eos = tok->get_eos_token_id();

    auto argmax_all = [&](const std::vector<float>& lg) -> int32_t {
        const size_t n = std::min(lg.size(), (size_t)vocab_size);
        int32_t best = -1; float bl = -1e30f;
        for (size_t i = 0; i < n; ++i)
            if (lg[i] > bl) { bl = lg[i]; best = (int32_t)i; }
        return best;
    };

    std::vector<int32_t> gen_tokens;
    int32_t next = argmax_all(logits);
    if (next >= 0) {
        for (int t = 0; t < max_new_tokens; ++t) {
            const int32_t cur = next;
            gen_tokens.push_back(cur);

            std::vector<int32_t>  tks       = {cur};
            std::vector<uint32_t> slots     = {0};
            std::vector<int32_t>  positions = {fp->get_rope_pos(0)};
            ggml_cgraph* gf = fp->build_decoding_graph(tks, slots, positions);
            ggml_backend_sched_reset(sched);
            ggml_backend_sched_alloc_graph(sched, gf);
            fp->set_decode_inputs(gf, tks, slots, positions);
            qinf::engine::require_compute_success(
                ggml_backend_sched_graph_compute(sched, gf), "cand_pass2_decode");

            std::vector<float> lg = fp->get_output_logits(gf);
            fp->advance_cache(1, 0);

            next = argmax_all(lg);
            if (next == eos || next < 0) break;
        }
    }
    fp->clear_slot(0);   // leave the engine as found, symmetric with pass 1's own cleanup

    return tok->decode(gen_tokens);
}

// Candidate-set pass 2, orchestrated: runs the cold decode above, parses it
// with the tolerant parser, and folds the result into `report` in place.
// FAILS LOUD on a producer failure (non-empty output, zero parseable lines) by
// setting report.candidates_producer_failed rather than leaving key_candidates
// silently empty — see server_lens.h and docs/plan-candidate-set.md's fourth
// state. Every candidate is verified byte-exact against `document` before it
// is stored; a non-verbatim span is dropped, not flagged (the format requires
// exactness, not a disclosure about its absence).
void run_cand_pass2(ForwardPassBase* fp, ggml_backend_sched_t sched, ::Tokenizer* tok,
                    const std::string& document, const std::vector<std::string>& keys,
                    uint32_t vocab_size, uint32_t n_ctx_max, LensReport& report) {
    // 700, not opts.max_new_tokens: pass 2 must enumerate every span for every
    // key (potentially several per key), a longer output than pass 1's single
    // value per key — matches the budget the CAND=1 probe measured the gate
    // against (docs/plan-candidate-set.md "Viability measured").
    const std::string gen_text =
        run_cand_pass2_decode(fp, sched, tok, document, keys, vocab_size, n_ctx_max, 700);
    lens_apply_pass2_candidates(document, gen_text, keys, report);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════
// Driver — free decode, one prefill, one pass
// ═════════════════════════════════════════════════════════════════════════
LensReport run_lens_extract(ForwardPassBase* fp, ggml_backend_sched_t sched,
                            ::Tokenizer* tok, const ModelMetadata& meta,
                            uint32_t vocab_size, uint32_t n_ctx_max,
                            const std::string& document,
                            const std::vector<LensConcept>& concepts,
                            const LensExtractOptions& opts,
                            const LensConstants& k,
                            GrammarVocab* control_arm_grammar,
                            const std::vector<std::string>* control_arm_vocab) {
    if (concepts.empty())
        throw std::runtime_error(
            "run_lens_extract: concepts expected non-empty (the complete concept "
            "hint) actual=empty — a bare instruction reopens the naming zoo (plan §1.2)");
    if ((control_arm_grammar != nullptr) != (control_arm_vocab != nullptr))
        throw std::runtime_error(
            "run_lens_extract: control_arm_grammar and control_arm_vocab expected "
            "both set or both null actual=" +
            std::string(control_arm_grammar ? "grammar set, vocab null"
                                            : "grammar null, vocab set") +
            " — the probe's control arm needs the grammar's token table");

    std::vector<std::string> keys;
    keys.reserve(concepts.size());
    for (const LensConcept& c : concepts) {
        if (c.key.empty())
            throw std::runtime_error(
                "run_lens_extract: concept key expected non-empty actual=empty");
        keys.push_back(c.key);
    }

    // The gloss is deliberately NOT in the instruction — see the header. The
    // prompt is byte-identical to the regime Stage 1 measured.
    static const std::vector<std::string> kNoVocab;
    // Request metadata, not decode state — set here rather than inside the
    // decode helper, whose job is the forward pass.
    LensRun run = run_lens_tapped_decode(
        fp, sched, tok, meta, control_arm_grammar,
        control_arm_vocab ? *control_arm_vocab : kNoVocab,
        vocab_size, n_ctx_max, document, lens_build_instruction(keys),
        opts.max_new_tokens, k);
    run.message_offsets = opts.message_offsets;
    if (!run.message_offsets.empty() && run.message_offsets.back() >= document.size())
        throw std::runtime_error(
            "run_lens_extract: message_offsets expected offsets within the document "
            "(< " + std::to_string(document.size()) + " bytes), actual last offset " +
            std::to_string(run.message_offsets.back()));

    // Throws LensUnparseableError (⇒ 422) if the output holds no parseable object.
    LensReport report = apply_absent_by_omission(compute_lens_report(run, k), concepts);

    // Pass 2 — candidate set (docs/plan-candidate-set.md), OFF by default and
    // byte-inert when off: this branch is the only place want_candidates is
    // read, and when it is false nothing below it ever runs — no second
    // prefill, no extra decode, no engine call of any kind beyond what pass 1
    // already did above.
    if (opts.want_candidates) {
        // Recorded regardless of what pass 2 produces (success, failure, or a
        // legitimate empty result) — it is the only signal lens_report_to_json
        // has for "not requested" vs. "ran" (see LensReport::candidates_requested).
        report.candidates_requested = true;
        run_cand_pass2(fp, sched, tok, document, keys, vocab_size, n_ctx_max, report);
    }

    return report;
}

LensReport apply_absent_by_omission(LensReport report,
                                    const std::vector<LensConcept>& concepts) {
    std::vector<LensField> out;
    out.reserve(concepts.size());
    for (const LensConcept& c : concepts) {
        // EVERY occurrence of the concept, in emission order — not just the
        // first. Concept order still governs the array, so the first entry for
        // each key keeps its old value and position and a first-match importer
        // is unaffected; what changes is that later occurrences now follow it
        // instead of being dropped. This is why the format bumps to v3
        // (server_lens.h LensReport): fields.size() == concepts.size() was an
        // invariant and no longer holds.
        bool any = false;
        for (const LensField& f : report.fields)
            if (f.key == c.key && !f.value.empty()) { out.push_back(f); any = true; }
        if (any) continue;
        // Absent: the model did not state it. With no grammar it declines
        // natively (30/30 on Leg C), so omission IS the signal — no probe needed.
        LensField f;
        f.key     = c.key;
        f.present = false;
        f.tier    = "";       // no value ⇒ no trust claim to make
        f.grounded = false;
        out.push_back(f);
    }
    report.fields = std::move(out);
    return report;
}

}  // namespace qinf
