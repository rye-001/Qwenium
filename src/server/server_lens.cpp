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
#include <stdexcept>

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
bool lens_value_of(const std::string& json, const std::string& key,
                   std::string& value, size_t& vb0, size_t& vb1, bool& is_null) {
    is_null = false;
    size_t kp = json.find("\"" + key + "\"");
    if (kp == std::string::npos) return false;
    size_t colon = json.find(':', kp);
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) i++;
    if (i < json.size() && json[i] == '"') {
        vb0 = i + 1;
        vb1 = json.find('"', vb0);
        if (vb1 == std::string::npos) vb1 = json.size();
    } else {
        vb0 = i;
        size_t j = i;
        while (j < json.size() && json[j] != ',' && json[j] != '}' && json[j] != '\n') j++;
        vb1 = j;
        while (vb1 > vb0 && (json[vb1 - 1] == ' ' || json[vb1 - 1] == '\r')) vb1--;
        is_null = (json.compare(vb0, vb1 - vb0, "null") == 0);
    }
    value = json.substr(vb0, vb1 - vb0);
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
            r.hover[t].push_back({p, v[i].second, run.prompt_cum[p], run.prompt_cum[p + 1]});
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
    for (const std::string& key : lens_keys(obj)) {
        LensField f;
        f.key = key;
        std::string value; size_t vb0 = 0, vb1 = 0; bool is_null = false;
        if (!lens_value_of(obj, key, value, vb0, vb1, is_null)) { r.fields.push_back(f); continue; }
        // JSON null = the model declining. Leave the value EMPTY so
        // apply_absent_by_omission marks it absent; claiming the literal "null"
        // as a value would fabricate a field the model explicitly refused.
        if (is_null) { r.fields.push_back(f); continue; }
        vb0 += obj_lo; vb1 += obj_lo;
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
            f.citations.push_back({p, av[i].second / std::max(1, nval), cblo, cbhi});
        }
        r.fields.push_back(f);
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
}  // namespace

std::string lens_report_to_json(const LensReport& r) {
    std::string o;
    o += "{\n";
    o += "\"format_version\":\"" + jesc(r.format_version) + "\",\n";
    o += "\"model\":\"" + jesc(r.model) + "\",\n";
    o += "\"validated_envelope\":" + std::string(r.validated_envelope ? "true" : "false") + ",\n";
    o += "\"citation_source\":\"layer 3, head 13 (L3H13) \\u2014 N3\",\n";
    o += "\"coverage_source\":\"layer 11, max over heads \\u2014 COV1\",\n";
    o += "\"used_threshold\":" + fnum(r.k.coverage_used_peak) + ",\n";
    o += "\"ungrounded_threshold\":" + fnum(r.k.ungrounded_body_mass) + ",\n";
    o += "\"prompt_len\":" + std::to_string(r.prompt_len) +
         ",\"doc_lo\":" + std::to_string(r.doc_lo) +
         ",\"doc_hi\":" + std::to_string(r.doc_hi) + ",\n";
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
                 ",\"byte_hi\":" + std::to_string(ct.byte_hi) + "}";
        }
        o += "]}";
    }
    o += "],\n";

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
    run.model              = "Qwen3.6 (attention lens)";
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
            LensStep st;
            st.n_kv         = taps[0].n_kv;
            st.citation_row = std::move(taps[0].rows);   // citation_layer (L3)
            st.coverage_row = std::move(taps[1].rows);   // coverage_layer (layer 11)
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
    LensRun run = run_lens_tapped_decode(
        fp, sched, tok, meta, control_arm_grammar,
        control_arm_vocab ? *control_arm_vocab : kNoVocab,
        vocab_size, n_ctx_max, document, lens_build_instruction(keys),
        opts.max_new_tokens, k);

    // Throws LensUnparseableError (⇒ 422) if the output holds no parseable object.
    return apply_absent_by_omission(compute_lens_report(run, k), concepts);
}

LensReport apply_absent_by_omission(LensReport report,
                                    const std::vector<LensConcept>& concepts) {
    auto emitted = [&](const std::string& key) -> const LensField* {
        for (const LensField& f : report.fields) if (f.key == key) return &f;
        return nullptr;
    };

    std::vector<LensField> out;
    out.reserve(concepts.size());
    for (const LensConcept& c : concepts) {
        const LensField* got = emitted(c.key);
        if (got && !got->value.empty()) { out.push_back(*got); continue; }
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
