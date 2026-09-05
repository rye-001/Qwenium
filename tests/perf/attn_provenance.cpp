// Attention-provenance probe (N3) — does the decode attention row point at the
// value's source span in the prompt?
//
// Hypothesis: at each decode step the engine materializes the post-softmax
// attention row (`kq_soft.<il>` in attention.cpp) — the distribution over KV
// positions the model consulted — and discards it. For copy-shaped extraction,
// does the argmax of that row land on the value's source span? This probe only
// answers "does it land?"; it builds no product.
//
// The tap: qwen36's attention module already names the post-softmax tensor
// `kq_soft.<il>` unconditionally. This probe drives the decode loop itself
// (build -> alloc -> set -> compute -> read), so between build and alloc it
// marks those tensors ggml_set_output (galloc would otherwise reuse the buffer)
// and reads them back after compute. ZERO engine edits: the tap lives entirely
// here, so flag-off is byte-inert by construction (the engine is unchanged).
// Marking an intermediate as output changes buffer liveness, not arithmetic —
// emitted logits are identical (self-test asserts each row sums to ~1/head).
//
// Method: teacher-forced. We construct the JSON extraction ourselves (we know
// the 5 values), so every scored value token's field — hence its source span —
// is known exactly, with no fragile free-gen JSON attribution. We also record
// whether the model's own greedy argmax matched the forced token (how "real"
// each provenance is). Only value tokens (digits/names/dates) are scored; JSON
// keys/punctuation are reported separately, never counted (protocol §4).
//
//   QWEN36_MODEL_PATH=... ./bin/attn-provenance
//   ATTN_TAP_SELFTEST=1   ./bin/attn-provenance   # tap sanity only, then exit

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/sampling/prompt_lookup.h"   // DP1: shipped PLD, called not reimplemented
#include "../../src/sampling/grammar_vocab.h"   // Qemmi-Docs leg A: the fixed KV grammar
#include "../../src/sampling/token_trie.h"      // Qemmi-Docs leg A: literal-candidate narrowing
#include "../../src/loader/chat_template.h"      // Qemmi-Docs: production chat-render (instruct regime)
#include "../../src/server/server_lens.h"        // QDOCS_S1: the SHIPPED lens math, called not reimplemented
#include "ggml.h"
#include "ggml-backend.h"

// ── A field the model must copy out of the prompt ────────────────────────────
struct Field {
    std::string name;    // JSON key
    std::string value;   // value string as it appears in the PROMPT (span lookup)
    int lo = -1, hi = -1; // inclusive prompt-token span of `value`
    std::string cvalue;  // value as emitted in the COMPLETION; empty => == value.
                         // Differs from `value` only under reformatting (Prompt C).
    const std::string& comp_val() const { return cvalue.empty() ? value : cvalue; }
};

// ── Everything collected for one prompt run ──────────────────────────────────
struct PromptRun {
    std::string label;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> comp_tokens;         // teacher-forced completion
    std::vector<Field>   fields;
    std::vector<int>     comp_field;          // per comp token: field idx or -1
    // rows[step][layer_slot] = flat [n_head * n_kv] post-softmax floats.
    // step t feeds comp_tokens[t] (query at prompt_len+t); its row is the
    // provenance for predicting comp_tokens[t+1].
    std::vector<int>                        n_kv_at_step;
    std::vector<std::vector<std::vector<float>>> rows;
    std::vector<int32_t>                    greedy_pred; // model argmax at each step
    int n_head = 0;
};

static const std::vector<int32_t>* g_attn_layers = nullptr;
// GGUF architecture of the loaded model; selects the chat template in
// qdocs_chat_prompt so the QDOCS legs are not silently Qwen-shaped.
static std::string g_arch;
// Model metadata for prompt encoding (BOS contract). Set in main().
static const ModelMetadata* g_meta = nullptr;

// Encode a PROMPT, honouring the model's own add_bos_token contract — the same
// thing cli/complete.cpp and cli/session_mode.cpp do before prefill.
// tok->encode() alone does NOT prepend BOS, and a Gemma -it model without a
// leading BOS degenerates (it repeats a single token; see the server BOS bug).
// Every Gemma number this probe produced before this existed was measured on
// that degenerate state. Qwen GGUFs carry add_bos_token=false, so this is inert
// on the qwen legs.
// The BOS as TEXT, not as a bare token id. Every span in this probe is found by
// searching the prompt STRING and mapping byte offsets back through
// cum_bytes(decode(prefix)); prepending a bare token id shifts that map by the
// BOS's decoded width and silently corrupts every span (the roundtrip guard in
// run_prompt catches it, which is how this was found). Putting the BOS in the
// text keeps tokens and bytes in lockstep, so the guard and the offsets stay
// honest. Empty string on models whose contract does not want a BOS.
static std::string g_bos_text;

static std::string prompt_with_bos(const std::string& text) { return g_bos_text + text; }

static std::vector<int32_t> encode_prompt(Tokenizer* tok, const std::string& text) {
    return tok->encode(text);   // callers pass prompt_with_bos(...) text
}

// Build cumulative byte offsets: cum[k] = bytes of decode(tokens[0:k]).
// Byte-level BPE roundtrips losslessly, so cum aligns with the original text.
static std::vector<size_t> cum_bytes(const Tokenizer* tok,
                                     const std::vector<int32_t>& toks) {
    std::vector<size_t> cum(toks.size() + 1, 0);
    for (size_t k = 1; k <= toks.size(); ++k) {
        std::vector<int32_t> pre(toks.begin(), toks.begin() + k);
        cum[k] = tok->decode(pre).size();
    }
    return cum;
}

// Token span [lo,hi] (inclusive) whose bytes overlap the first occurrence of
// `value` in `text`. Returns false if `value` not found.
static bool find_token_span(const Tokenizer* tok, const std::vector<int32_t>& toks,
                            const std::string& text, const std::string& value,
                            int& lo, int& hi) {
    size_t b0 = text.find(value);
    if (b0 == std::string::npos) return false;
    size_t b1 = b0 + value.size();
    std::vector<size_t> cum = cum_bytes(tok, toks);
    lo = hi = -1;
    for (size_t k = 0; k < toks.size(); ++k) {
        if (cum[k] < b1 && cum[k + 1] > b0) { // token k overlaps [b0,b1)
            if (lo < 0) lo = (int)k;
            hi = (int)k;
        }
    }
    return lo >= 0;
}

// Detokenized ±ctx window around absolute sequence position `pos`, marking the
// token at pos with «».
static std::string ctx_around(const Tokenizer* tok,
                              const std::vector<int32_t>& seq, int pos, int ctx) {
    int a = std::max(0, pos - ctx);
    int b = std::min((int)seq.size() - 1, pos + ctx);
    std::string s;
    for (int i = a; i <= b; ++i) {
        std::string t = tok->decode(seq[i]);
        if (i == pos) s += "«" + t + "»";
        else s += t;
    }
    // collapse newlines for one-line printing
    for (char& c : s) if (c == '\n') c = ' ';
    return s;
}

// Top-k KV positions (excluding index 0 / BOS-sink) for head h of a flat
// [n_head * n_kv] row. Returns {position, mass} sorted desc.
static std::vector<std::pair<int, float>> topk_head(const std::vector<float>& row,
                                                    int h, int n_kv, int k) {
    std::vector<std::pair<int, float>> v;
    const float* r = row.data() + (size_t)h * n_kv;
    for (int j = 1; j < n_kv; ++j) v.push_back({j, r[j]});
    std::partial_sort(v.begin(), v.begin() + std::min((size_t)k, v.size()), v.end(),
                      [](auto& a, auto& b) { return a.second > b.second; });
    if ((int)v.size() > k) v.resize(k);
    return v;
}

// ── Run one prompt: prefill, then teacher-force the completion with the tap ──
static PromptRun run_prompt(ForwardPassBase* fp, ggml_backend_sched_t sched,
                            Tokenizer* tok, const ModelMetadata& meta,
                            const std::string& label,
                            const std::string& prompt_text,
                            const std::string& comp_text,
                            std::vector<Field> fields,
                            bool selftest) {
    PromptRun R;
    R.label = label;
    R.fields = std::move(fields);

    const std::string ptext = prompt_with_bos(prompt_text);
    R.prompt_tokens = encode_prompt(tok, ptext);
    R.comp_tokens   = tok->encode(comp_text);

    // Roundtrip sanity: byte-level BPE must reproduce the text, or our byte
    // offsets are meaningless — fail loud (CLAUDE.md fail-loud contract).
    if (tok->decode(R.prompt_tokens) != ptext)
        throw std::runtime_error("prompt roundtrip mismatch — token offsets unreliable");
    if (tok->decode(R.comp_tokens) != comp_text)
        throw std::runtime_error("completion roundtrip mismatch");

    const int P = (int)R.prompt_tokens.size();

    // Locate each field's source span in the prompt.
    for (auto& f : R.fields) {
        if (!find_token_span(tok, R.prompt_tokens, ptext, f.value, f.lo, f.hi))
            throw std::runtime_error("field value not found in prompt: " + f.value);
    }

    // Label each completion token with the field whose value covers it.
    R.comp_field.assign(R.comp_tokens.size(), -1);
    {
        std::vector<size_t> ccum = cum_bytes(tok, R.comp_tokens);
        for (int fi = 0; fi < (int)R.fields.size(); ++fi) {
            const std::string& cv = R.fields[fi].comp_val();
            size_t b0 = comp_text.find(cv);
            if (b0 == std::string::npos)
                throw std::runtime_error("field comp value not in completion: " + cv);
            size_t b1 = b0 + cv.size();
            for (size_t k = 0; k < R.comp_tokens.size(); ++k)
                if (ccum[k] < b1 && ccum[k + 1] > b0) R.comp_field[k] = fi;
        }
    }

    // Full sequence for context printing.
    std::vector<int32_t> seq = R.prompt_tokens;
    seq.insert(seq.end(), R.comp_tokens.begin(), R.comp_tokens.end());

    // Prefill (untapped — protocol: decode steps only).
    fp->clear_slot(0);
    fp->set_cache_pos(0, 0);
    fp->run_prefill(R.prompt_tokens, 0, 0, sched);

    const int n_head = (int)meta.attention_head_count; // 16
    R.n_head = n_head;

    const int L = (int)R.comp_tokens.size();
    const int n_steps = selftest ? std::min(L, 2) : L;

    for (int t = 0; t < n_steps; ++t) {
        std::vector<int32_t>  tokens    = {R.comp_tokens[t]};
        std::vector<uint32_t> slots     = {0};
        std::vector<int32_t>  positions = {(int)fp->get_cache_pos(0)};

        ggml_cgraph* gf = fp->build_decoding_graph(tokens, slots, positions);

        // THE TAP: mark each attention layer's post-softmax row as an output so
        // galloc retains its buffer. Must happen before alloc.
        std::vector<ggml_tensor*> taps;
        for (int il : *g_attn_layers) {
            std::string nm = "kq_soft." + std::to_string(il);
            ggml_tensor* ts = ggml_graph_get_tensor(gf, nm.c_str());
            if (!ts) throw std::runtime_error("tap tensor missing: " + nm);
            ggml_set_output(ts);
            ggml_build_forward_expand(gf, ts);
            taps.push_back(ts);
        }

        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        fp->set_decode_inputs(gf, tokens, slots, positions);
        ggml_backend_sched_graph_compute(sched, gf);

        // Read back the tapped rows.
        std::vector<std::vector<float>> layer_rows;
        int n_kv = 0;
        for (ggml_tensor* ts : taps) {
            // shape [n_kv, 1, n_head, 1]
            n_kv = (int)ts->ne[0];
            int nh = (int)ts->ne[2];
            std::vector<float> buf((size_t)n_kv * nh);
            ggml_backend_tensor_get(ts, buf.data(), 0, ggml_nbytes(ts));
            layer_rows.push_back(std::move(buf));

            if (selftest && t == 0 && ts == taps.front()) {
                std::printf("[selftest] tap %s dims = [%lld,%lld,%lld,%lld]  n_kv=%d n_head=%d\n",
                            ggml_get_name(ts),
                            (long long)ts->ne[0], (long long)ts->ne[1],
                            (long long)ts->ne[2], (long long)ts->ne[3], n_kv, nh);
            }
        }

        if (selftest) {
            // Each head's softmax row must sum to ~1.0 (proves real data, not
            // a reused/garbage buffer).
            const std::vector<float>& fr = layer_rows.front();
            for (int h = 0; h < n_head; ++h) {
                double s = 0; for (int j = 0; j < n_kv; ++j) s += fr[(size_t)h * n_kv + j];
                if (h < 3)
                    std::printf("[selftest] step %d layer %d head %d  sum=%.6f  argmax@%d\n",
                                t, (*g_attn_layers)[0], h, s,
                                (int)(std::max_element(fr.begin() + (size_t)h * n_kv,
                                       fr.begin() + (size_t)(h + 1) * n_kv) -
                                      (fr.begin() + (size_t)h * n_kv)));
            }
        }

        // Greedy agreement: does the model's own argmax equal the next forced
        // token? (Reads the head that the decode graph already computed.)
        std::vector<float> logits = fp->get_output_logits(gf);
        int32_t best = 0;
        for (int j = 1; j < (int)logits.size(); ++j)
            if (logits[j] > logits[best]) best = j;
        R.greedy_pred.push_back(best);

        R.rows.push_back(std::move(layer_rows));
        R.n_kv_at_step.push_back(n_kv);

        fp->advance_cache(1, 0);
    }

    (void)P; (void)seq;
    return R;
}

// ── Per-(layer,head) scoring over the value tokens of one run ────────────────
struct Score { int n = 0, top1 = 0, top3 = 0; double bos_mass = 0; };

// Score value tokens for attention-layer slot `ls` (index into g_attn_layers)
// and head `h`. Provenance of comp value token v is rows[v-1].
static Score score_lh(const PromptRun& R, int ls, int h, int tol) {
    Score sc;
    for (int v = 1; v < (int)R.comp_tokens.size(); ++v) {
        int fi = R.comp_field[v];
        if (fi < 0) continue;                     // structural — not scored
        int step = v - 1;
        if (step >= (int)R.rows.size()) break;
        const std::vector<float>& row = R.rows[step][ls];
        int n_kv = R.n_kv_at_step[step];
        const Field& f = R.fields[fi];
        auto tk = topk_head(row, h, n_kv, 3);
        auto in_span = [&](int pos) { return pos >= f.lo - tol && pos <= f.hi + tol; };
        sc.n++;
        if (!tk.empty() && in_span(tk[0].first)) sc.top1++;
        for (auto& pr : tk) if (in_span(pr.first)) { sc.top3++; break; }
        sc.bos_mass += row[(size_t)h * n_kv + 0];  // index-0 mass
    }
    return sc;
}

// ═════════════════════════════════════════════════════════════════════════
// N3b — the ungrounded-value alarm. Does the frozen retrieval head's row look
// different when the model INVENTS a value (no prompt source) vs copies one?
// Gated by env ATTN_UNGROUNDED; leaves the A/B/C path (§4-6, committed note)
// completely untouched. Same tap, same head (L3H13, ensemble peer L7H8).
// ═════════════════════════════════════════════════════════════════════════

// Frozen head from N3 (tap-slot index into g_attn_layers, and head).
// Defaults are the Qwen 3.6 coordinates the shipped LensConstants carry.
// ATTN_FROZEN_SLOT / ATTN_FROZEN_HEAD override them so the confirmation legs
// (QDOCS_C, ATTN_UNGROUNDED, COVERAGE) can be pointed at a candidate found on
// ANOTHER model without editing this file — the second corpus is the whole
// point of the search, and a hardcoded head made it unaskable off qwen36.
// Unset ⇒ byte-identical to the committed qwen36 behaviour.
static int FROZEN_SLOT = 0, FROZEN_HEAD = 13;  // layer 3
static int ENS_SLOT    = 1, ENS_HEAD    = 8;   // layer 7 (ensemble stretch)

// Defined below, after L3H13_SLOT / L11_SLOT — it sets those too.
static void apply_frozen_head_overrides();

enum Region { R_IDX0 = 0, R_BODY, R_INSTR, R_OWN };
static const char* region_name(Region r) {
    switch (r) { case R_IDX0: return "IDX0"; case R_BODY: return "BODY";
                 case R_INSTR: return "INSTR"; default: return "OWN"; }
}
static Region region_of(int pos, int instr_tok, int P) {
    if (pos == 0) return R_IDX0;
    if (pos < instr_tok) return R_BODY;
    if (pos < P) return R_INSTR;
    return R_OWN;
}

// CG1 — per-step confidence scalars from the full logit vector the sampler
// already computes (top1/top2 gap etc.), all discarded today.
struct ConfStep {
    float gap = 0;      // logit[top1] - logit[top2]
    float pmargin = 0;  // p(top1) - p(top2)
    float p1 = 0;       // p(top1)
    float entropy = 0;  // full-softmax entropy over the vocab
};
// conf[t] is computed from the step-t logits (which predict gen_tokens[t+1]),
// so the confidence of emitting value token v is conf[v-1] — same off-by-one as
// the attention row. Cheap: two vocab passes/step.
static ConfStep conf_from_logits(const std::vector<float>& lg) {
    ConfStep c;
    int t1 = 0;
    for (int j = 1; j < (int)lg.size(); ++j) if (lg[j] > lg[t1]) t1 = j;
    float top1 = lg[t1], top2 = -1e30f;
    for (int j = 0; j < (int)lg.size(); ++j) if (j != t1 && lg[j] > top2) top2 = lg[j];
    c.gap = top1 - top2;
    // softmax around top1 as the max: e_top1 = 1.
    double Z = 0, S = 0;
    for (float v : lg) { double e = std::exp((double)v - top1); Z += e; S += e * ((double)v - top1); }
    c.p1 = (float)(1.0 / Z);
    c.pmargin = (float)((1.0 - std::exp((double)top2 - top1)) / Z);
    c.entropy = (float)(std::log(Z) - S / Z);
    return c;
}

struct FreeRun {
    std::vector<int32_t> prompt_tokens;
    int P = 0, instr_tok = 0;               // instr_tok = first instruction token
    std::vector<int32_t> gen_tokens;        // free-greedy completion
    std::vector<std::vector<std::vector<float>>> rows; // [step][tap-slot] flat n_head*n_kv
    std::vector<int> n_kv_at_step;
    std::vector<ConfStep> conf;             // [step]; populated only when capture_conf
    int n_head = 0;
    std::string body_text, gen_text;
};

// Free-greedy generate (natural regime), tapping only `tap_layers`. Stops at
// balanced-brace JSON close, EOS, or max_new. rows[t] (query=gen[t]) is the
// provenance for gen[t+1] — same off-by-one as N3.
static FreeRun run_freegen(ForwardPassBase* fp, ggml_backend_sched_t sched,
                           Tokenizer* tok, const ModelMetadata& meta,
                           const std::string& body_text, const std::string& instr_text,
                           const std::vector<int>& tap_layers, int max_new,
                           bool capture_conf = false, char close_char = '}') {
    FreeRun R;
    R.body_text = body_text;
    R.n_head = (int)meta.attention_head_count;
    // BOS lives in the TEXT so tokens and byte offsets stay in lockstep; the
    // body/instruction boundary below must therefore clear the BOS too.
    std::string prompt_text = prompt_with_bos(body_text + instr_text);
    const size_t body_end = g_bos_text.size() + body_text.size();
    R.prompt_tokens = encode_prompt(tok, prompt_text);
    if (tok->decode(R.prompt_tokens) != prompt_text)
        throw std::runtime_error("freegen: prompt roundtrip mismatch");
    R.P = (int)R.prompt_tokens.size();

    std::vector<size_t> cum = cum_bytes(tok, R.prompt_tokens);
    R.instr_tok = R.P;
    for (int k = 0; k < R.P; ++k) if (cum[k] >= body_end) { R.instr_tok = k; break; }

    fp->clear_slot(0); fp->set_cache_pos(0, 0);
    std::vector<float> logits = fp->run_prefill(R.prompt_tokens, 0, 0, sched);
    int32_t next = 0;
    for (int j = 1; j < (int)logits.size(); ++j) if (logits[j] > logits[next]) next = j;

    const int32_t eos = tok->get_eos_token_id();
    for (int t = 0; t < max_new; ++t) {
        if (next == eos) break;
        int32_t cur = next;
        R.gen_tokens.push_back(cur);
        std::string ct = tok->decode(cur);
        // Prompt is primed with an opener; the first close_char in the
        // completion ends it ('}' for a single object, ']' for a Tier-2 array).
        bool closed = ct.find(close_char) != std::string::npos;

        std::vector<int32_t>  tks = {cur};
        std::vector<uint32_t> slots = {0};
        std::vector<int32_t>  positions = {(int)fp->get_cache_pos(0)};
        ggml_cgraph* gf = fp->build_decoding_graph(tks, slots, positions);

        std::vector<ggml_tensor*> taps;
        for (int il : tap_layers) {
            std::string nm = "kq_soft." + std::to_string(il);
            ggml_tensor* ts = ggml_graph_get_tensor(gf, nm.c_str());
            if (!ts) throw std::runtime_error("freegen tap missing: " + nm);
            ggml_set_output(ts); ggml_build_forward_expand(gf, ts); taps.push_back(ts);
        }
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        fp->set_decode_inputs(gf, tks, slots, positions);
        ggml_backend_sched_graph_compute(sched, gf);

        std::vector<std::vector<float>> layer_rows; int n_kv = 0;
        for (ggml_tensor* ts : taps) {
            n_kv = (int)ts->ne[0]; int nh = (int)ts->ne[2];
            std::vector<float> buf((size_t)n_kv * nh);
            ggml_backend_tensor_get(ts, buf.data(), 0, ggml_nbytes(ts));
            layer_rows.push_back(std::move(buf));
        }
        R.rows.push_back(std::move(layer_rows));
        R.n_kv_at_step.push_back(n_kv);

        std::vector<float> lg = fp->get_output_logits(gf);
        if (capture_conf) R.conf.push_back(conf_from_logits(lg));
        next = 0; for (int j = 1; j < (int)lg.size(); ++j) if (lg[j] > lg[next]) next = j;
        fp->advance_cache(1, 0);
        if (closed) break;  // JSON object closed
    }
    R.gen_text = tok->decode(R.gen_tokens);
    return R;
}

// A field parsed out of the generated JSON, classified grounded/invented/refused.
struct GenField {
    std::string name, value;
    int a = -1, b = -1;                 // gen-token span of the value
    bool grounded = false, refused = false;
};

// Boundary-aware grounding: the value must appear in the body as a standalone
// unit, not glued to alnum/./@ (which spuriously "grounds" "0" inside an email
// address or "orders" inside a login). Kills the substring-leakage confound.
static bool grounded_in_body(const std::string& body, const std::string& value) {
    if (value.empty()) return false;
    auto digit = [](char c) { return isdigit((unsigned char)c) != 0; };
    auto alnum = [](char c) { return isalnum((unsigned char)c) != 0; };
    size_t from = 0;
    while (true) {
        size_t p = body.find(value, from);
        if (p == std::string::npos) return false;
        size_t e = p + value.size();
        // Reject only when the match is a sub-token of a larger word/number/
        // email — i.e. glued to alnum or '@', or a '.' that continues a decimal
        // (digit on the far side of the dot). A sentence-terminating '.' is fine.
        bool bad_l = false, bad_r = false;
        if (p > 0) { char c = body[p - 1];
            bad_l = alnum(c) || c == '@' || (c == '.' && p >= 2 && digit(body[p - 2])); }
        if (e < body.size()) { char c = body[e];
            bad_r = alnum(c) || c == '@' || (c == '.' && e + 1 < body.size() && digit(body[e + 1])); }
        if (!bad_l && !bad_r) return true;
        from = p + 1;
    }
}

static std::vector<GenField> parse_fields(Tokenizer* tok, const FreeRun& R,
                                          const std::vector<std::string>& keys) {
    std::vector<GenField> out;
    const std::string& g = R.gen_text;
    std::vector<size_t> gcum = cum_bytes(tok, R.gen_tokens);
    auto is_refusal = [](std::string v) {
        for (char& c : v) c = (char)tolower((unsigned char)c);
        return v.empty() || v == "null" || v == "n/a" || v == "na" ||
               v == "unknown" || v == "tbd" || v == "none" || v == "-" || v == "\"\"";
    };
    for (auto& key : keys) {
        GenField f; f.name = key;
        size_t kp = g.find("\"" + key + "\"");
        if (kp == std::string::npos) { out.push_back(f); continue; }
        size_t colon = g.find(':', kp);
        if (colon == std::string::npos) { out.push_back(f); continue; }
        size_t i = colon + 1; while (i < g.size() && (g[i] == ' ' || g[i] == '\t')) i++;
        size_t vb0, vb1;
        if (i < g.size() && g[i] == '"') {
            vb0 = i + 1; vb1 = g.find('"', vb0); if (vb1 == std::string::npos) vb1 = g.size();
        } else {
            vb0 = i; size_t j = i;
            while (j < g.size() && g[j] != ',' && g[j] != '}' && g[j] != '\n') j++;
            vb1 = j; while (vb1 > vb0 && g[vb1 - 1] == ' ') vb1--;
        }
        f.value = g.substr(vb0, vb1 - vb0);
        for (size_t k = 0; k < R.gen_tokens.size(); ++k)
            if (gcum[k] < vb1 && gcum[k + 1] > vb0) { if (f.a < 0) f.a = (int)k; f.b = (int)k; }
        if (is_refusal(f.value)) f.refused = true;
        else f.grounded = grounded_in_body(R.body_text, f.value);
        out.push_back(f);
    }
    return out;
}

// Field-level scalars from the frozen head (means over the field's value tokens).
struct FE {
    std::string prompt, name, value; bool grounded = false;
    double body_mass = 0, max_body = 0, entropy = 0, body_mass_ens = 0;
    int n = 0, top1_body = 0, first_v = -1;
};

static FE eval_field(const FreeRun& R, const GenField& gf, const std::string& prompt) {
    FE e; e.prompt = prompt; e.name = gf.name; e.value = gf.value; e.grounded = gf.grounded;
    for (int v = gf.a; v <= gf.b; ++v) {
        int step = v - 1;
        if (step < 0 || step >= (int)R.rows.size()) continue;
        int n_kv = R.n_kv_at_step[step];
        auto mass = [&](int slot, int head) {
            const float* r = R.rows[step][slot].data() + (size_t)head * n_kv;
            double bm = 0, mx = 0, ent = 0; int am = 0; double amv = -1;
            for (int j = 0; j < n_kv; ++j) {
                double p = r[j];
                if (p > amv) { amv = p; am = j; }
                if (j >= 1 && j < R.instr_tok) { bm += p; if (p > mx) mx = p; }
                ent += -(p) * std::log(p + 1e-12);
            }
            return std::make_tuple(bm, mx, ent, am);
        };
        auto [bm, mx, ent, am] = mass(FROZEN_SLOT, FROZEN_HEAD);
        double bm2 = std::get<0>(mass(ENS_SLOT, ENS_HEAD));
        e.body_mass += bm; e.max_body += mx; e.entropy += ent;
        e.body_mass_ens += 0.5 * (bm + bm2);
        if (region_of(am, R.instr_tok, R.P) == R_BODY) e.top1_body++;
        if (e.first_v < 0) e.first_v = v;
        e.n++;
    }
    if (e.n) { e.body_mass /= e.n; e.max_body /= e.n; e.entropy /= e.n; e.body_mass_ens /= e.n; }
    return e;
}

// Best single threshold separating grounded(+) from invented(-). dir=+1 =>
// grounded predicted when x>=t; dir=-1 => x<=t.
struct Thr { double t = 0; int dir = 1; double acc = -1; };
static Thr best_threshold(const std::vector<std::pair<double, bool>>& data) {
    std::vector<double> vals; for (auto& d : data) vals.push_back(d.first);
    std::sort(vals.begin(), vals.end());
    std::vector<double> cands; cands.push_back(vals.empty() ? 0 : vals.front() - 1e-6);
    for (size_t i = 0; i + 1 < vals.size(); ++i) cands.push_back(0.5 * (vals[i] + vals[i + 1]));
    if (!vals.empty()) cands.push_back(vals.back() + 1e-6);
    Thr best;
    auto acc_of = [&](double t, int dir) {
        int c = 0; for (auto& d : data) {
            bool pred = dir > 0 ? d.first >= t : d.first <= t;
            if (pred == d.second) c++;
        } return data.empty() ? 0.0 : (double)c / data.size();
    };
    for (double t : cands) for (int dir : {+1, -1}) {
        double a = acc_of(t, dir);
        if (a > best.acc) best = {t, dir, a};
    }
    return best;
}
static double apply_thr(const Thr& thr, const std::vector<std::pair<double, bool>>& data) {
    int c = 0; for (auto& d : data) {
        bool pred = thr.dir > 0 ? d.first >= thr.t : d.first <= thr.t;
        if (pred == d.second) c++;
    } return data.empty() ? 0.0 : (double)c / data.size();
}

struct UPrompt { std::string tag, body, instr; };

// Shared schema keys + prompt sets (N3b and CG1 use byte-identical prompts so
// CG1's greedy generations reproduce N3b's, which is CG1's self-check).
static const std::vector<std::string>& ungrounded_keys() {
    static const std::vector<std::string> keys =
        {"customer", "date", "quantity", "unit_price", "total"};
    return keys;
}
static const std::string& ungrounded_instr() {
    static const std::string INSTR =
        "\nReturn ONLY a single-line JSON object with keys customer, date, "
        "quantity, unit_price, total. Every field is REQUIRED and must be a "
        "concrete value (a name, an ISO date, or a number). Do NOT use null, "
        "empty, or \"unknown\".\nJSON:\n{";  // prime with '{' → forces the object,
                                             // removes think-runaway/empty-reply
    return INSTR;
}
static void ungrounded_prompt_sets(std::vector<UPrompt>& calib,
                                   std::vector<UPrompt>& held) {
    const std::string& INSTR = ungrounded_instr();
    // Each email omits one/two NON-derivable fields; instruction forces a value.
    calib = {
        {"c_date",   // missing: date
         "From: orders@nimbus.example\nSubject: Purchase Order\n\nHello,\n\n"
         "Please process this order for customer Nimbus Supply Co.\n"
         "We need a quantity of 320 units.\nThe unit price is 18.75 USD.\n"
         "The order total comes to 6000.00 USD.\n\nThanks.\n", INSTR},
        {"c_cust",   // missing: customer (generic from-address, no company hint)
         "From: orders@mail-3271.example\nSubject: New Order\n\nHi,\n\n"
         "Please arrange the following order.\nOrder date: 2025-09-08.\n"
         "We require a quantity of 150 units.\nThe unit price is 33.20 EUR.\n"
         "The order total comes to 4980.00 EUR.\n\nRegards.\n", INSTR},
        {"c_money",  // missing: unit_price, total (no money in the email at all)
         "From: buyer@mail-5510.example\nSubject: Order\n\nHello,\n\n"
         "Order for customer Vertex Labs.\nOrder date: 2026-01-19.\n"
         "We need a quantity of 500 units.\n\nThanks.\n", INSTR},
        {"c_qty",    // missing: quantity, total (neither derivable)
         "From: buyer@mail-6120.example\nSubject: Order\n\nHi,\n\n"
         "Order for customer Helios Trading.\nOrder date: 2025-04-11.\n"
         "The unit price is 24.00 USD.\n\nRegards.\n", INSTR},
    };
    held = {
        {"h_date",   // missing: date
         "From: sales@cobalt.example\nSubject: Order Request\n\nHello,\n\n"
         "Order for customer Cobalt Freight.\nWe need a quantity of 90 units.\n"
         "The unit price is 55.00 USD.\nThe order total comes to 4950.00 USD.\n\n"
         "Best.\n", INSTR},
        {"h_cust",   // missing: customer
         "From: orders@mail-8842.example\nSubject: New Order\n\nHi,\n\n"
         "Please process the order below.\nOrder date: 2024-12-03.\n"
         "We require a quantity of 610 units.\nThe unit price is 7.40 EUR.\n"
         "The order total comes to 4514.00 EUR.\n\nThanks.\n", INSTR},
        {"h_money",  // missing: unit_price, total
         "From: buyer@mail-9033.example\nSubject: Order\n\nHello,\n\n"
         "Order for customer Summit Metals.\nOrder date: 2025-07-22.\n"
         "We need a quantity of 275 units.\n\nRegards.\n", INSTR},
        {"h_qty",    // missing: quantity, total
         "From: buyer@mail-2244.example\nSubject: Order\n\nHi,\n\n"
         "Order for customer Delta Foods.\nOrder date: 2026-02-28.\n"
         "The unit price is 3.15 USD.\n\nBest.\n", INSTR},
    };
}

static int run_ungrounded_probe(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                Tokenizer* tok, const ModelMetadata& meta,
                                const std::vector<int32_t>& attn_layers) {
    const std::vector<int> tap_layers = {attn_layers[FROZEN_SLOT], attn_layers[ENS_SLOT]};
    const std::vector<std::string> keys = ungrounded_keys();
    std::vector<UPrompt> calib, held;
    ungrounded_prompt_sets(calib, held);

    struct PromptResult { std::string tag; FreeRun run; std::vector<GenField> fields; };

    auto run_set = [&](std::vector<UPrompt>& set, const char* which,
                       std::vector<FE>& evals) {
        std::vector<PromptResult> results;
        std::printf("\n================ %s SET ================\n", which);
        for (auto& up : set) {
            FreeRun R = run_freegen(fp, sched, tok, meta, up.body, up.instr, tap_layers, 256);
            std::string gt = R.gen_text; for (char& c : gt) if (c == '\n') c = ' ';
            std::printf("\n[%s] P=%d instr_tok=%d gen=%zu tok\n  gen: %s\n",
                        up.tag.c_str(), R.P, R.instr_tok, R.gen_tokens.size(), gt.c_str());
            auto fields = parse_fields(tok, R, keys);
            for (auto& gf : fields) {
                if (gf.a < 0) { std::printf("    %-11s <no value parsed>\n", gf.name.c_str()); continue; }
                if (gf.refused) { std::printf("    %-11s = \"%s\"  [REFUSED]\n",
                                              gf.name.c_str(), gf.value.c_str()); continue; }
                FE e = eval_field(R, gf, up.tag);
                if (e.n == 0) continue;
                std::printf("    %-11s = %-18s %s  body=%.3f max=%.3f ent=%.2f top1body=%d/%d\n",
                            gf.name.c_str(), ("\"" + gf.value + "\"").c_str(),
                            e.grounded ? "GROUND " : "INVENT ",
                            e.body_mass, e.max_body, e.entropy, e.top1_body, e.n);
                evals.push_back(e);
            }
            results.push_back({up.tag, std::move(R), std::move(fields)});
        }
        return results;
    };

    std::vector<FE> cal, hel;
    std::vector<PromptResult> cal_r = run_set(calib, "CALIBRATION", cal);
    std::vector<PromptResult> hel_r = run_set(held, "HELD-OUT", hel);

    auto split = [](const std::vector<FE>& v, int metric) {
        // metric 0=body_mass,1=max_body,2=entropy,3=body_mass_ens
        std::vector<std::pair<double, bool>> d;
        for (auto& e : v) {
            double x = metric == 0 ? e.body_mass : metric == 1 ? e.max_body
                     : metric == 2 ? e.entropy   : e.body_mass_ens;
            d.push_back({x, e.grounded});
        }
        return d;
    };

    int n_g = 0, n_i = 0; for (auto& e : cal) (e.grounded ? n_g : n_i)++;
    std::printf("\n---- discriminator selection (calib: %d grounded, %d invented) ----\n", n_g, n_i);
    const char* mname[4] = {"body_mass", "max_body", "entropy", "body_mass_ENSEMBLE"};
    Thr best; int best_m = -1;
    for (int m = 0; m < 4; ++m) {
        Thr thr = best_threshold(split(cal, m));
        std::printf("  %-19s  thr %.4f dir %+d  calib-acc %.0f%%\n",
                    mname[m], thr.t, thr.dir, 100 * thr.acc);
        // single-scalar winner = m in {0,1,2}; ensemble (m=3) reported separately
        if (m < 3 && thr.acc > best.acc) { best = thr; best_m = m; }
    }
    std::printf("\n  >>> FROZEN single-scalar: %s  thr %.4f dir %+d (calib %.0f%%)\n",
                mname[best_m], best.t, best.dir, 100 * best.acc);

    int hg = 0, hi = 0; for (auto& e : hel) (e.grounded ? hg : hi)++;
    double held_acc = apply_thr(best, split(hel, best_m));
    double held_ens = apply_thr(best_threshold(split(cal, 3)), split(hel, 3));
    std::printf("\n---- HELD-OUT eval (%d grounded, %d invented) ----\n", hg, hi);
    std::printf("  frozen %s: held-out acc %.0f%%\n", mname[best_m], 100 * held_acc);
    std::printf("  (ensemble body_mass, frozen-on-calib): held-out acc %.0f%%\n", 100 * held_ens);

    // Confusion + the scary shape: invented fields that masquerade as grounded
    // (high body mass AND top-1 in body) — a confident FALSE citation.
    std::printf("\n---- held-out per-field verdict (frozen %s) ----\n", mname[best_m]);
    int fp_confident = 0;
    for (auto& e : hel) {
        double x = best_m == 0 ? e.body_mass : best_m == 1 ? e.max_body : e.entropy;
        bool pred_ground = best.dir > 0 ? x >= best.t : x <= best.t;
        bool correct = pred_ground == e.grounded;
        bool scary = !e.grounded && pred_ground && e.top1_body == e.n;
        if (scary) fp_confident++;
        std::printf("  %-7s %-11s %s  pred=%-7s %s%s\n",
                    e.prompt.c_str(), e.name.c_str(), e.grounded ? "GROUND" : "INVENT",
                    pred_ground ? "GROUND" : "INVENT",
                    correct ? "ok" : "**MISS**", scary ? "  <== CONFIDENT-FALSE-CITE" : "");
    }

    // Eyeball: where do invented values point? (frozen head, first value token).
    // Reuses the stored held-out runs — no re-generation.
    std::printf("\n---- eyeball: invented-value provenance (frozen L%d H%d) ----\n",
                attn_layers[FROZEN_SLOT], FROZEN_HEAD);
    for (auto& pr_res : hel_r) {
        const FreeRun& R = pr_res.run;
        std::vector<int32_t> seq = R.prompt_tokens;
        seq.insert(seq.end(), R.gen_tokens.begin(), R.gen_tokens.end());
        for (auto& gf : pr_res.fields) {
            if (gf.a < 0 || gf.refused || gf.grounded) continue;  // invented only
            int v = gf.a, step = v - 1;
            if (step < 0 || step >= (int)R.rows.size()) continue;
            int n_kv = R.n_kv_at_step[step];
            auto tk = topk_head(R.rows[step][FROZEN_SLOT], FROZEN_HEAD, n_kv, 3);
            std::string vt = tok->decode(R.gen_tokens[v]); for (char& c : vt) if (c == '\n') c = ' ';
            std::printf("\n  [%s] INVENTED %s = \"%s\"  (first token «%s»)\n",
                        pr_res.tag.c_str(), gf.name.c_str(), gf.value.c_str(), vt.c_str());
            for (auto& p : tk) {
                Region rg = region_of(p.first, R.instr_tok, R.P);
                std::printf("    pos %3d  mass %.3f  %-5s | %s\n", p.first, p.second,
                            region_name(rg), ctx_around(tok, seq, p.first, 3).c_str());
            }
        }
    }

    std::printf("\n================ VERDICT (N3b) ================\n");
    std::printf("  held-out single-scalar %s: %.0f%%  (PASS >=90%%)\n", mname[best_m], 100 * held_acc);
    if (fp_confident > 0)
        std::printf("  WARNING: %d invented field(s) got a CONFIDENT FALSE citation "
                    "(high body mass, top-1 in body) — citation product must not "
                    "claim groundedness for these.\n", fp_confident);
    if (held_acc >= 0.90 && fp_confident == 0) std::printf("  => PASS\n");
    else if (held_acc >= 0.90)                 std::printf("  => PASS-with-caveat (see WARNING)\n");
    else if (held_ens >= 0.90 || held_acc >= 0.75) std::printf("  => WEAK (needs ensemble/more machinery)\n");
    else                                       std::printf("  => KILL / inconclusive — see tables\n");
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// CG1 — the confidence gap. top1-top2 (and friends) per value token, discarded
// today. Does a logit-gap scalar separate the field classes we already label,
// and is it COMPLEMENTARY to the provenance (body_mass) signal? Gated by
// CONF_GAP=1; N3 (A/B/C) and N3b (ATTN_UNGROUNDED) paths untouched.
// ═════════════════════════════════════════════════════════════════════════
enum { C_GROUND = 0, C_INVENT = 1, C_DERIVED = 2, C_CONFLICT = 3 };
static const char* cls_name(int c) {
    static const char* n[] = {"GROUND", "INVENT", "DERIVED", "CONFLICT"};
    return n[c];
}
struct FConf {
    std::string prompt, name, value; int cls = 0; double body_mass = 0; int n = 0;
    // aggregations: mean over value tokens, weakest-link, first token.
    double g_mean = 0, g_weak = 0, g_first = 0;   // gap  (weak = MIN)
    double pm_mean = 0, pm_weak = 0, pm_first = 0; // prob margin (weak = MIN)
    double p1_mean = 0, p1_weak = 0, p1_first = 0; // p(top1) (weak = MIN)
    double e_mean = 0, e_weak = 0, e_first = 0;    // entropy (weak = MAX)
};
// 12 candidate scalars, extractor by id. For entropy, higher = less confident,
// so best_threshold's direction handles the flip.
static double conf_scalar(const FConf& f, int m) {
    switch (m) {
        case 0: return f.g_mean;  case 1: return f.g_weak;  case 2: return f.g_first;
        case 3: return f.pm_mean; case 4: return f.pm_weak; case 5: return f.pm_first;
        case 6: return f.p1_mean; case 7: return f.p1_weak; case 8: return f.p1_first;
        default:case 9: return f.e_mean; case 10: return f.e_weak; case 11: return f.e_first;
    }
}
static const char* conf_scalar_name(int m) {
    static const char* n[12] = {
        "gap.mean", "gap.weak", "gap.first", "pmargin.mean", "pmargin.weak",
        "pmargin.first", "p1.mean", "p1.weak", "p1.first",
        "entropy.mean", "entropy.weak", "entropy.first"};
    return n[m];
}
static FConf eval_conf(const FreeRun& R, const GenField& gf) {
    FConf e; e.name = gf.name; e.value = gf.value;
    double gmin = 1e30, pmmin = 1e30, p1min = 1e30, emax = -1e30;
    for (int v = gf.a; v <= gf.b; ++v) {
        int step = v - 1;
        if (step < 0 || step >= (int)R.conf.size()) continue;
        const ConfStep& c = R.conf[step];
        e.g_mean += c.gap; e.pm_mean += c.pmargin; e.p1_mean += c.p1; e.e_mean += c.entropy;
        gmin = std::min(gmin, (double)c.gap); pmmin = std::min(pmmin, (double)c.pmargin);
        p1min = std::min(p1min, (double)c.p1); emax = std::max(emax, (double)c.entropy);
        if (v == gf.a) { e.g_first = c.gap; e.pm_first = c.pmargin;
                         e.p1_first = c.p1; e.e_first = c.entropy; }
        e.n++;
    }
    if (e.n) { e.g_mean /= e.n; e.pm_mean /= e.n; e.p1_mean /= e.n; e.e_mean /= e.n;
               e.g_weak = gmin; e.pm_weak = pmmin; e.p1_weak = p1min; e.e_weak = emax; }
    return e;
}
static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static int run_confgap_probe(ForwardPassBase* fp, ggml_backend_sched_t sched,
                             Tokenizer* tok, const ModelMetadata& meta,
                             const std::vector<int32_t>& attn_layers) {
    const std::vector<int> tap_layers = {attn_layers[FROZEN_SLOT], attn_layers[ENS_SLOT]};
    const std::vector<std::string> keys = ungrounded_keys();
    std::vector<UPrompt> calib, held;
    ungrounded_prompt_sets(calib, held);
    // Conflict prompts (two candidate dates → the schema key "date" is ambiguous).
    std::vector<UPrompt> conflict = {
        {"x_conf1",
         "From: sales@initech.example\nSubject: Order Confirmation\n\nHi,\n\n"
         "Order for customer Initech LLC.\nOrder date: 2024-06-09.\n"
         "Requested delivery date: 2024-07-20.\nWe need a quantity of 1250 units.\n"
         "The unit price is 8.75 USD.\nThe order total comes to 10937.50 USD.\n\n"
         "Best.\n", ungrounded_instr()},
        {"x_conf2",
         "From: ops@zenith.example\nSubject: PO\n\nHello,\n\n"
         "Order for customer Zenith Works.\nInvoice date: 2025-03-05.\n"
         "Shipment date: 2025-03-19.\nWe need a quantity of 44 units.\n"
         "The unit price is 12.00 USD.\nThe order total comes to 528.00 USD.\n\n"
         "Regards.\n", ungrounded_instr()},
    };

    auto num = [](const std::vector<GenField>& fs, const std::string& nm) -> double {
        for (auto& f : fs) if (f.name == nm && f.a >= 0) return strtod(f.value.c_str(), nullptr);
        return std::nan("");
    };
    auto is_derived_total = [&](const GenField& gf, const std::vector<GenField>& fs,
                                const FreeRun& R) {
        if (gf.name != "total") return false;
        if (grounded_in_body(R.body_text, gf.value)) return false; // copied, not derived
        double t = strtod(gf.value.c_str(), nullptr);
        double q = num(fs, "quantity"), p = num(fs, "unit_price");
        if (std::isnan(t) || std::isnan(q) || std::isnan(p)) return false;
        return std::fabs(t - q * p) <= std::max(1.0, 0.02 * std::fabs(t));
    };

    struct Row { std::string prompt; FConf c; };
    std::vector<Row> cal, hel, cnf;
    std::vector<double> struct_gaps;  // structural-token sanity

    auto process = [&](std::vector<UPrompt>& set, std::vector<Row>& out, bool is_conflict) {
        for (auto& up : set) {
            FreeRun R = run_freegen(fp, sched, tok, meta, up.body, up.instr,
                                    tap_layers, 256, /*capture_conf=*/true);
            std::string gt = R.gen_text; for (char& c : gt) if (c == '\n') c = ' ';
            std::printf("\n[%s] gen=%zu tok  %s\n", up.tag.c_str(), R.gen_tokens.size(), gt.c_str());
            auto fields = parse_fields(tok, R, keys);
            // structural tokens = gen tokens not in any parsed value span (excl idx 0)
            std::vector<bool> is_val(R.gen_tokens.size(), false);
            for (auto& gf : fields) if (gf.a >= 0)
                for (int v = gf.a; v <= gf.b; ++v) if (v >= 0 && v < (int)is_val.size()) is_val[v] = true;
            for (int v = 1; v < (int)R.gen_tokens.size(); ++v)
                if (!is_val[v] && v - 1 < (int)R.conf.size()) struct_gaps.push_back(R.conf[v - 1].gap);

            for (auto& gf : fields) {
                if (gf.a < 0 || gf.refused) continue;
                int cls;
                if (is_conflict && gf.name == "date") cls = C_CONFLICT;
                else if (grounded_in_body(R.body_text, gf.value)) cls = C_GROUND;
                else if (is_derived_total(gf, fields, R)) cls = C_DERIVED;
                else cls = C_INVENT;
                FE bm = eval_field(R, gf, up.tag);
                FConf c = eval_conf(R, gf);
                c.prompt = up.tag; c.cls = cls; c.body_mass = bm.body_mass;
                std::printf("    %-11s %-9s %-16s body=%.3f | gap.first=%.2f gap.mean=%.2f "
                            "p1.first=%.3f ent.first=%.2f\n",
                            gf.name.c_str(), cls_name(cls), ("\"" + gf.value + "\"").c_str(),
                            c.body_mass, c.g_first, c.g_mean, c.p1_first, c.e_first);
                out.push_back({up.tag, c});
            }
        }
    };

    std::printf("\n================ CALIBRATION ================\n");
    process(calib, cal, false);
    std::printf("\n================ HELD-OUT ================\n");
    process(held, hel, false);
    std::printf("\n================ CONFLICT (descriptive) ================\n");
    process(conflict, cnf, true);

    // Self-check: greedy generations must reproduce N3b's body_mass byte-for-byte.
    std::printf("\n---- self-check vs N3b note (body_mass must match) ----\n");
    struct Chk { const char* p; const char* f; double want; };
    std::vector<Chk> checks = {{"c_date", "customer", 0.850}, {"c_date", "quantity", 0.968},
                               {"h_money", "quantity", 0.980}, {"h_date", "unit_price", 0.977}};
    bool ok = true;
    for (auto& ck : checks) {
        double got = -1;
        for (auto& r : cal) if (r.prompt == ck.p && r.c.name == ck.f) got = r.c.body_mass;
        for (auto& r : hel) if (r.prompt == ck.p && r.c.name == ck.f) got = r.c.body_mass;
        bool m = std::fabs(got - ck.want) < 0.02;
        ok = ok && m;
        std::printf("  %-7s %-11s want %.3f got %.3f  %s\n", ck.p, ck.f, ck.want, got,
                    m ? "ok" : "**MISMATCH**");
    }
    if (!ok) { std::printf("\n  STOP: generations did not reproduce N3b — CG1 invalid.\n"); return 1; }

    // Structural sanity row.
    std::printf("\n  structural tokens (keys/braces/quotes): median gap = %.2f  (expect huge)\n",
                median(struct_gaps));

    // ── Scalar selection: grounded vs invented FIELDS on calibration ──────────
    auto gi = [](const std::vector<Row>& rs, int m) {
        std::vector<std::pair<double, bool>> d;
        for (auto& r : rs) if (r.c.cls == C_GROUND || r.c.cls == C_INVENT)
            d.push_back({conf_scalar(r.c, m), r.c.cls == C_GROUND});
        return d;
    };
    std::printf("\n---- scalar selection (calib grounded-vs-invented) ----\n");
    Thr best; int best_m = -1;
    for (int m = 0; m < 12; ++m) {
        Thr thr = best_threshold(gi(cal, m));
        std::printf("  %-14s thr %8.3f dir %+d  calib %.0f%%\n",
                    conf_scalar_name(m), thr.t, thr.dir, 100 * thr.acc);
        if (thr.acc > best.acc) { best = thr; best_m = m; }
    }
    double held_acc = apply_thr(best, gi(hel, best_m));
    std::printf("\n  >>> FROZEN %s  thr %.3f dir %+d (calib %.0f%%)  ->  HELD-OUT %.0f%%\n",
                conf_scalar_name(best_m), best.t, best.dir, 100 * best.acc, 100 * held_acc);

    // ── Per-class medians (descriptive, pooled) ──────────────────────────────
    std::vector<Row> all; all.insert(all.end(), cal.begin(), cal.end());
    all.insert(all.end(), hel.begin(), hel.end()); all.insert(all.end(), cnf.begin(), cnf.end());
    std::printf("\n---- per-class medians (pooled, descriptive) ----\n");
    std::printf("  class     n   gap.first  gap.mean  p1.first  ent.first  body_mass\n");
    for (int c = 0; c <= 3; ++c) {
        std::vector<double> gf, gm, p1, ef, bm; int n = 0;
        for (auto& r : all) if (r.c.cls == c) {
            gf.push_back(r.c.g_first); gm.push_back(r.c.g_mean); p1.push_back(r.c.p1_first);
            ef.push_back(r.c.e_first); bm.push_back(r.c.body_mass); n++;
        }
        if (!n) continue;
        std::printf("  %-8s %3d   %8.2f  %8.2f  %8.3f  %8.2f  %8.3f\n", cls_name(c), n,
                    median(gf), median(gm), median(p1), median(ef), median(bm));
    }

    // ── 2×2 complementarity on held-out (provenance × gap) ────────────────────
    std::printf("\n---- 2x2 complementarity (HELD-OUT): {body_mass>=0.538} x {gap-scalar grounded} ----\n");
    const double BM_THR = 0.538;
    auto gap_ground = [&](const FConf& c) {
        double x = conf_scalar(c, best_m);
        return best.dir > 0 ? x >= best.t : x <= best.t;
    };
    const char* cell_name[2][2] = {{"prov-FLAG & gap-FLAG", "prov-FLAG & gap-ok"},
                                   {"prov-ok  & gap-FLAG", "prov-ok  & gap-ok"}};
    std::vector<std::string> cells[2][2];
    for (auto& r : hel) {
        int pi = r.c.body_mass >= BM_THR ? 1 : 0;   // 1 = prov says grounded
        int gj = gap_ground(r.c) ? 1 : 0;           // 1 = gap says grounded
        cells[pi][gj].push_back(r.prompt + ":" + r.c.name + "(" + cls_name(r.c.cls) + ")");
    }
    for (int pi = 1; pi >= 0; --pi) for (int gj = 1; gj >= 0; --gj) {
        std::printf("  [%-20s] %zu:", cell_name[pi][gj], cells[pi][gj].size());
        for (auto& s : cells[pi][gj]) std::printf(" %s", s.c_str());
        std::printf("\n");
    }
    std::printf("  (complementarity win = non-grounded fields in 'prov-ok & gap-FLAG': "
                "gap catches what provenance passes)\n");

    // ── Eyeball: 3 tokens with gap + class ────────────────────────────────────
    std::printf("\n---- eyeball (value token | gap | class) ----\n");
    int shown = 0;
    for (auto& r : all) {
        if (shown >= 3) break;
        if (r.c.cls == C_GROUND && shown != 0) continue;  // 1 ground, then invent/derived
        std::printf("  %-7s %-10s %-8s value \"%s\"  gap.first=%.2f p1.first=%.3f body_mass=%.3f\n",
                    r.prompt.c_str(), r.c.name.c_str(), cls_name(r.c.cls), r.c.value.c_str(),
                    r.c.g_first, r.c.p1_first, r.c.body_mass);
        shown++;
    }

    // ── Verdict ───────────────────────────────────────────────────────────────
    // class ordering check: cleanly-invented gap.first median below grounded's.
    auto med_first = [&](int c) { std::vector<double> v;
        for (auto& r : all) if (r.c.cls == c) v.push_back(r.c.g_first); return median(v); };
    double g_ground = med_first(C_GROUND), g_invent = med_first(C_INVENT),
           g_conf = med_first(C_CONFLICT), g_deriv = med_first(C_DERIVED);
    bool ordered = g_invent < g_ground && (g_conf < g_ground || g_deriv < g_ground);
    // redundancy: does gap flag differ from provenance flag on held-out?
    int diff = 0; for (auto& r : hel) {
        bool pv = r.c.body_mass >= BM_THR, gp = gap_ground(r.c);
        if (pv != gp) diff++;
    }
    std::printf("\n================ VERDICT (CG1) ================\n");
    std::printf("  held-out grounded-vs-invented: %.0f%%  (PASS >=90%%)\n", 100 * held_acc);
    std::printf("  class order gap.first: GROUND %.2f | INVENT %.2f | CONFLICT %.2f | DERIVED %.2f  (%s)\n",
                g_ground, g_invent, g_conf, g_deriv, ordered ? "ordered as hypothesized" : "NOT ordered");
    std::printf("  provenance/gap disagree on %d/%zu held fields (%s)\n", diff, hel.size(),
                diff > 0 ? "complementary" : "REDUNDANT");
    if (held_acc >= 0.90 && ordered) std::printf("  => PASS\n");
    else if (ordered && held_acc >= 0.70) std::printf("  => WEAK (ordering holds; 90%% bar missed)\n");
    else if (diff == 0) std::printf("  => KILL (redundant with provenance)\n");
    else std::printf("  => KILL / inconclusive — see tables\n");
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// DP1 — the retrieval head as a draft POINTER. When the model emits e_t, the
// frozen L3H13 row points at source(e_t)=prompt[p]; if a copy is in progress,
// prompt[p+1..p+K] is a speculative draft with a TRAINED pointer (no draft
// model, no extra compute). Offline acceptance simulation only — no engine
// change, no speed claim. Compared head-to-head with the shipped PLD.
// Gated by DRAFT_POINTER=1; N3/N3b/CG1 paths untouched.
// ═════════════════════════════════════════════════════════════════════════
static int lcp_tok(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    int n = (int)std::min(a.size(), b.size()), i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}
static int common_bytes(const std::string& a, const std::string& b) {
    int n = (int)std::min(a.size(), b.size()), i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

struct Rec {
    int tier; bool held;
    // pointer (BODY argmax)
    bool ptr_raw = false; double maxmass = 0, bodymass = 0; int ptr_L = 0; bool tokmis = false;
    // PLD (shipped)
    bool pld_raw = false; int pld_L = 0; int ngram_occ = 0;
    // OWN self-copy (descriptive)
    bool own_raw = false; int own_L = 0;
};

static std::vector<UPrompt> make_tier2(bool held) {
    // Multi-line-item orders → long verbatim product-name copies (where drafting
    // pays). One reformatted price per email (a known pointer-failure class).
    const std::string I =
        "\nExtract every line item as a JSON array; each element has keys "
        "product, quantity, unit_price, copied exactly from the email. "
        "Output only the JSON array.\nJSON:\n[";
    if (!held) return {
        {"t2_a",
         "From: procurement@harborline.example\nSubject: Purchase Order 4471\n\n"
         "Hello,\n\nPlease fulfill the following order:\n"
         "- Stainless Steel Compression Fitting, quantity 120, unit price 4.75\n"
         "- Heavy Duty Rubber Gasket Seal, quantity 340, unit price 1.20\n"
         "- Brass Ball Valve Half Inch, quantity 85, unit price 9.60\n"
         "- Galvanized Steel Mounting Bracket, quantity 200, unit price 3.15\n\n"
         "Thank you.\n", I},
        {"t2_b",
         "From: orders@meadowfield.example\nSubject: Supply Request\n\n"
         "Hi,\n\nWe would like to order:\n"
         "- Organic Whole Wheat Flour Sack, quantity 50, unit price 12.40\n"
         "- Cold Pressed Sunflower Oil Drum, quantity 18, unit price 74.00\n"
         "- Himalayan Pink Rock Salt Bag, quantity 220, unit price 2.85\n\n"
         "Regards.\n", I},
        {"t2_c",
         "From: purchasing@nordictech.example\nSubject: Hardware Order\n\n"
         "Hello,\n\nRequested items:\n"
         "- Wireless Mechanical Keyboard Backlit, quantity 40, unit price 89.90\n"
         "- Ultrawide Curved Monitor 34 Inch, quantity 12, unit price 415.00\n"
         "- USB C Docking Station Dual, quantity 60, unit price 128.50\n"
         "- Noise Cancelling Headset Pro, quantity 75, unit price 199.00\n\n"
         "Best.\n", I},
    };
    return {
        {"t2_d",
         "From: buyer@crestwood.example\nSubject: PO 8890\n\n"
         "Hello,\n\nOrder details:\n"
         "- Powder Coated Steel Shelving Unit, quantity 30, unit price 64.20\n"
         "- Interlocking Rubber Floor Tile, quantity 500, unit price 3.40\n"
         "- Adjustable Workbench Vise Clamp, quantity 45, unit price 27.75\n\n"
         "Thanks.\n", I},
        {"t2_e",
         "From: orders@bluepeak.example\nSubject: Catering Supplies\n\n"
         "Hi,\n\nPlease send:\n"
         "- Compostable Kraft Paper Plates, quantity 1200, unit price 0.18\n"
         "- Double Walled Insulated Coffee Cups, quantity 800, unit price 0.42\n"
         "- Biodegradable Wooden Cutlery Set, quantity 950, unit price 0.31\n"
         "- Recycled Napkin Dispenser Refill, quantity 300, unit price 1.95\n\n"
         "Regards.\n", I},
        {"t2_f",
         "From: procurement@ironclad.example\nSubject: Parts Order\n\n"
         "Hello,\n\nWe require:\n"
         "- Tungsten Carbide Drill Bit Set, quantity 65, unit price 43.30\n"
         "- Hydraulic Quick Release Coupler, quantity 140, unit price 8.90\n"
         "- Anti Vibration Rubber Mounting Pad, quantity 260, unit price 2.10\n\n"
         "Best.\n", I},
    };
}

// Simulate pointer / PLD / OWN acceptance for one generation.
static std::vector<Rec> simulate(const FreeRun& R, Tokenizer* tok, int tier, bool held) {
    const int K = 8;
    qinf::PromptLookup pld({/*ngram*/3, /*max_draft*/5, /*min_draft*/1});  // shipped params
    const auto& prompt = R.prompt_tokens;
    const auto& gen = R.gen_tokens;
    const int P = R.P, G = (int)gen.size();
    std::vector<int32_t> seq = prompt; seq.insert(seq.end(), gen.begin(), gen.end());
    std::vector<Rec> out;

    for (int j = 0; j + 1 < G; ++j) {
        Rec r; r.tier = tier; r.held = held;
        std::vector<int32_t> actual(gen.begin() + j + 1,
                                    gen.begin() + std::min(G, j + 1 + K));
        // PLD: needle = last 3 of generated-so-far; haystack = prompt.
        std::vector<int32_t> generated(gen.begin(), gen.begin() + j + 1);
        std::vector<int32_t> pd = pld.find_draft(prompt, generated);
        r.pld_raw = !pd.empty();
        r.pld_L = lcp_tok(pd, actual);
        if ((int)generated.size() >= 3) {
            const int32_t* nd = generated.data() + generated.size() - 3;
            for (int i = 0; i + 3 <= P; ++i)
                if (prompt[i] == nd[0] && prompt[i+1] == nd[1] && prompt[i+2] == nd[2]) r.ngram_occ++;
        }
        // Pointer: row from the forward pass that emitted gen[j] = rows[j-1].
        if (j >= 1) {
            const std::vector<float>& row = R.rows[j - 1][FROZEN_SLOT];
            int n_kv = R.n_kv_at_step[j - 1];
            const float* rr = row.data() + (size_t)FROZEN_HEAD * n_kv;
            int body_hi = std::min(R.instr_tok, n_kv);
            int p = -1; double best = -1; double bmass = 0;
            for (int q = 1; q < body_hi; ++q) { bmass += rr[q]; if (rr[q] > best) { best = rr[q]; p = q; } }
            if (p >= 0) {
                r.ptr_raw = true; r.maxmass = best; r.bodymass = bmass;
                std::vector<int32_t> D(prompt.begin() + std::min(P, p + 1),
                                       prompt.begin() + std::min(P, p + 1 + K));
                r.ptr_L = lcp_tok(D, actual);
                if (r.ptr_L == 0 && !D.empty() && !actual.empty()) {
                    std::string dt, at;
                    for (int x : D) dt += tok->decode(x);
                    for (int x : actual) at += tok->decode(x);
                    r.tokmis = common_bytes(dt, at) > 0;
                }
            }
            // OWN self-copy: argmax over OWN region [P, n_kv).
            int q = -1; double bo = -1;
            for (int u = P; u < n_kv; ++u) if (rr[u] > bo) { bo = rr[u]; q = u; }
            if (q >= 0 && q + 1 < (int)seq.size()) {
                r.own_raw = true;
                std::vector<int32_t> D(seq.begin() + q + 1,
                                       seq.begin() + std::min((int)seq.size(), q + 1 + K));
                r.own_L = lcp_tok(D, actual);
            }
        }
        out.push_back(r);
    }
    return out;
}

// Aggregate metrics over a filtered set of steps.
struct Agg { double E_ptr = 0, E_pld = 0, E_hyb = 0, E_own = 0;
             double cov_ptr = 0, cov_pld = 0; double acc1_ptr = 0, acc1_pld = 0; int n = 0; };
static Agg aggregate(const std::vector<Rec>& rs, int gate, double thr) {
    Agg a; a.n = (int)rs.size(); if (!a.n) return a;
    int off_ptr = 0, off_pld = 0, hit_ptr = 0, hit_pld = 0;
    double sp = 0, spl = 0, sh = 0, so = 0;
    for (auto& r : rs) {
        double gm = gate == 0 ? r.maxmass : r.bodymass;
        bool poff = r.ptr_raw && gm >= thr;
        int pL = poff ? r.ptr_L : 0;
        sp += pL; spl += r.pld_L; so += r.own_raw ? r.own_L : 0;
        sh += std::max(pL, r.pld_L);
        if (poff) { off_ptr++; if (r.ptr_L >= 1) hit_ptr++; }
        if (r.pld_raw) { off_pld++; if (r.pld_L >= 1) hit_pld++; }
    }
    a.E_ptr = sp / a.n; a.E_pld = spl / a.n; a.E_hyb = sh / a.n; a.E_own = so / a.n;
    a.cov_ptr = (double)off_ptr / a.n; a.cov_pld = (double)off_pld / a.n;
    a.acc1_ptr = off_ptr ? (double)hit_ptr / off_ptr : 0;
    a.acc1_pld = off_pld ? (double)hit_pld / off_pld : 0;
    return a;
}

static int run_draft_pointer_probe(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                   Tokenizer* tok, const ModelMetadata& meta,
                                   const std::vector<int32_t>& attn_layers) {
    const std::vector<int> tap_layers = {attn_layers[FROZEN_SLOT], attn_layers[ENS_SLOT]};
    const std::vector<std::string> keys = ungrounded_keys();
    std::vector<UPrompt> t1c, t1h; ungrounded_prompt_sets(t1c, t1h);
    std::vector<UPrompt> t2c = make_tier2(false), t2h = make_tier2(true);

    struct GenRec { std::string tag; int tier; bool held; FreeRun run; };
    std::vector<GenRec> gens;
    std::vector<std::pair<std::string, double>> selfcheck;  // (tag:field, body_mass)

    auto run_group = [&](std::vector<UPrompt>& set, int tier, bool held, char close) {
        for (auto& up : set) {
            FreeRun R = run_freegen(fp, sched, tok, meta, up.body, up.instr, tap_layers,
                                    tier == 1 ? 256 : 420, false, close);
            std::string gt = R.gen_text; for (char& c : gt) if (c == '\n') c = ' ';
            if (gt.size() > 140) gt = gt.substr(0, 140) + "...";
            std::printf("[%s t%d %s] gen=%zu  %s\n", up.tag.c_str(), tier,
                        held ? "held" : "cal", R.gen_tokens.size(), gt.c_str());
            if (tier == 1) {  // self-check hooks
                auto fields = parse_fields(tok, R, keys);
                for (auto& gf : fields) if (gf.a >= 0 && !gf.refused) {
                    FE e = eval_field(R, gf, up.tag);
                    selfcheck.push_back({up.tag + ":" + gf.name, e.body_mass});
                }
            }
            gens.push_back({up.tag, tier, held, std::move(R)});
        }
    };
    std::printf("\n---- generations ----\n");
    run_group(t1c, 1, false, '}'); run_group(t1h, 1, true, '}');
    run_group(t2c, 2, false, ']'); run_group(t2h, 2, true, ']');

    // Self-check vs N3b note.
    std::printf("\n---- self-check vs N3b (body_mass) ----\n");
    struct Chk { const char* k; double w; };
    std::vector<Chk> checks = {{"c_date:customer", 0.850}, {"c_date:quantity", 0.968},
                               {"h_money:quantity", 0.980}, {"h_date:unit_price", 0.977}};
    bool ok = true;
    for (auto& ck : checks) { double got = -1;
        for (auto& s : selfcheck) if (s.first == ck.k) got = s.second;
        bool m = std::fabs(got - ck.w) < 0.02; ok = ok && m;
        std::printf("  %-18s want %.3f got %.3f %s\n", ck.k, ck.w, got, m ? "ok" : "**MISMATCH**");
    }
    if (!ok) { std::printf("  STOP: generations diverged from N3b — DP1 invalid.\n"); return 1; }

    // Simulate all.
    std::vector<Rec> all;
    for (auto& g : gens) { auto v = simulate(g.run, tok, g.tier, g.held);
        all.insert(all.end(), v.begin(), v.end()); }
    auto filt = [&](bool held, int tier /*0=both*/) {
        std::vector<Rec> r; for (auto& x : all)
            if (x.held == held && (tier == 0 || x.tier == tier)) r.push_back(x); return r;
    };
    std::vector<Rec> cal = filt(false, 0), hel = filt(true, 0);

    // ── Gate sweep on calibration: max E[accepted]/step s.t. acc1 >= 0.60 ─────
    std::printf("\n---- gate sweep (calibration) ----\n");
    std::printf("  gate         thr    E[acc]/step  cov   acc@1\n");
    int best_gate = 0; double best_thr = 0, best_E = -1;
    for (int g = 0; g < 2; ++g) {
        for (double t = 0.0; t <= (g == 0 ? 0.7 : 0.9) + 1e-9; t += 0.1) {
            Agg a = aggregate(cal, g, t);
            std::printf("  %-10s %5.2f  %8.3f    %.2f  %.2f\n",
                        g == 0 ? "maxmass" : "bodymass", t, a.E_ptr, a.cov_ptr, a.acc1_ptr);
            if (a.acc1_ptr >= 0.60 && a.E_ptr > best_E) { best_E = a.E_ptr; best_gate = g; best_thr = t; }
        }
    }
    if (best_E < 0) {  // nothing hit 0.60 — fall back to max acc1
        double bestacc = -1;
        for (int g = 0; g < 2; ++g) for (double t = 0.0; t <= 0.9; t += 0.1) {
            Agg a = aggregate(cal, g, t);
            if (a.acc1_ptr > bestacc) { bestacc = a.acc1_ptr; best_gate = g; best_thr = t; best_E = a.E_ptr; }
        }
        std::printf("  (no gate reached acc@1>=0.60; picked max acc@1)\n");
    }
    std::printf("  >>> FROZEN gate=%s thr=%.2f\n", best_gate == 0 ? "maxmass" : "bodymass", best_thr);

    Agg ung = aggregate(cal, 0, 0.0);  // ungated bound (maxmass thr 0)
    std::printf("  ungated (calib) pointer E[acc]/step=%.3f cov=%.2f acc@1=%.2f\n",
                ung.E_ptr, ung.cov_ptr, ung.acc1_ptr);

    // ── Held-out tables per tier ──────────────────────────────────────────────
    auto report = [&](const char* label, const std::vector<Rec>& rs) {
        Agg a = aggregate(rs, best_gate, best_thr);
        std::printf("  %-16s  ptr E=%.3f cov=%.2f acc@1=%.2f | PLD E=%.3f cov=%.2f acc@1=%.2f | "
                    "HYB E=%.3f | OWN E=%.3f  (n=%d)\n",
                    label, a.E_ptr, a.cov_ptr, a.acc1_ptr, a.E_pld, a.cov_pld, a.acc1_pld,
                    a.E_hyb, a.E_own, a.n);
        return a;
    };
    std::printf("\n---- HELD-OUT (gated pointer=%s@%.2f, K=8; PLD shipped n=3/k=5) ----\n",
                best_gate == 0 ? "maxmass" : "bodymass", best_thr);
    report("Tier1 (short)", filt(true, 1));
    report("Tier2 (long)",  filt(true, 2));
    Agg pooled = report("POOLED held",  hel);

    // ── Onset / ambiguity / tokenization (held-out, gated pointer) ────────────
    int onset_ptr_only = 0, pld_only = 0, ambig_ptr_win = 0, tokmis = 0, both = 0;
    for (auto& r : hel) {
        double gm = best_gate == 0 ? r.maxmass : r.bodymass;
        bool poff = r.ptr_raw && gm >= best_thr;
        bool pw = poff && r.ptr_L >= 1;
        bool lw = r.pld_raw && r.pld_L >= 1;
        if (pw && !lw) onset_ptr_only++;
        if (lw && !pw) pld_only++;
        if (pw && lw) both++;
        if (pw && r.ngram_occ > 1) ambig_ptr_win++;
        if (r.tokmis) tokmis++;
    }
    std::printf("\n  complementarity (held, tokens accepted@>=1): pointer-only=%d  PLD-only=%d  both=%d\n",
                onset_ptr_only, pld_only, both);
    std::printf("  ambiguous-ngram steps where pointer accepts (n-gram occurs >1x): %d\n", ambig_ptr_win);
    std::printf("  tokenization-mismatch failures (pointer text matched, token-ids didn't): %d\n", tokmis);

    // ── Eyeball: 3 steps (step | pointer pos | draft | accepted-len) ──────────
    std::printf("\n---- eyeball (Tier2 held, pointer draft vs actual) ----\n");
    int shown = 0;
    for (auto& g : gens) {
        if (g.tier != 2 || !g.held || shown >= 3) continue;
        auto recs = simulate(g.run, tok, 2, true);
        const auto& prompt = g.run.prompt_tokens; const auto& gen = g.run.gen_tokens;
        for (int j = 1; j + 1 < (int)gen.size() && shown < 3; ++j) {
            const Rec& r = recs[j];
            double gm = best_gate == 0 ? r.maxmass : r.bodymass;
            if (!(r.ptr_raw && gm >= best_thr && r.ptr_L >= 3)) continue;  // show strong hits
            int n_kv = g.run.n_kv_at_step[j - 1];
            const float* rr = g.run.rows[j - 1][FROZEN_SLOT].data() + (size_t)FROZEN_HEAD * n_kv;
            int p = -1; double best = -1;
            for (int q = 1; q < std::min(g.run.instr_tok, n_kv); ++q) if (rr[q] > best) { best = rr[q]; p = q; }
            std::string dr, ac;
            for (int x = p + 1; x < std::min((int)prompt.size(), p + 1 + r.ptr_L + 1); ++x) dr += tok->decode(prompt[x]);
            for (int x = j + 1; x < std::min((int)gen.size(), j + 1 + r.ptr_L + 1); ++x) ac += tok->decode(gen[x]);
            for (char& c : dr) if (c=='\n') c=' '; for (char& c : ac) if (c=='\n') c=' ';
            std::printf("  [%s] after «%s» ptr->pos%d maxmass=%.2f  draft \"%s\" | actual \"%s\"  acceptL=%d\n",
                        g.tag.c_str(), tok->decode(gen[j]).c_str(), p, best, dr.c_str(), ac.c_str(), r.ptr_L);
            shown++;
        }
    }

    // ── Amdahl line ───────────────────────────────────────────────────────────
    // Persistent-graph decode C_dec=34ms; verify pass ~one full step C_ver=46ms
    // (plan-mtp-decode.md §9: 3-tok verify ~55ms on the 52ms-baseline path).
    // speedup = C_dec*(1+E_all) / (cov*C_ver + (1-cov)*C_dec).  E_all = coverage
    // * mean-accepted-given-draft = the reported E[acc]/step (hybrid).
    const double C_dec = 34.0, C_ver = 46.0;
    auto amdahl = [&](double E_all, double cov) {
        return C_dec * (1.0 + E_all) / (cov * C_ver + (1.0 - cov) * C_dec);
    };
    Agg hp = aggregate(hel, best_gate, best_thr);
    double sp_ptr = amdahl(hp.E_ptr, hp.cov_ptr);
    double sp_pld = amdahl(hp.E_pld, hp.cov_pld);
    double sp_hyb = amdahl(hp.E_hyb, std::max(hp.cov_ptr, hp.cov_pld));
    std::printf("\n---- Amdahl (C_dec=34ms, C_ver=46ms; optimistic — ignores hybrid refeed tax) ----\n");
    std::printf("  ceiling tok/s multiplier: pointer %.2fx | PLD %.2fx | HYBRID %.2fx\n",
                sp_ptr, sp_pld, sp_hyb);
    std::printf("  (anchor: §9 shipped PLD 1.08 tok/step measured ~flat on this model)\n");

    // ── Verdict ───────────────────────────────────────────────────────────────
    double hyb_vs_pld = hp.E_pld > 1e-6 ? hp.E_hyb / hp.E_pld : (hp.E_hyb > 0 ? 99 : 1);
    std::printf("\n================ VERDICT (DP1) ================\n");
    std::printf("  HYBRID/PLD E[acc] ratio = %.2f (PASS needs >=1.20)\n", hyb_vs_pld);
    std::printf("  gated pointer acc@1 (pooled held) = %.2f (PASS needs >=0.60)\n", pooled.acc1_ptr);
    std::printf("  Amdahl HYBRID ceiling = %.2fx (build-worthy needs >=1.15)\n", sp_hyb);
    bool pass = hyb_vs_pld >= 1.20 && pooled.acc1_ptr >= 0.60;
    if (pass && sp_hyb >= 1.15) std::printf("  => PASS (and build-worthy)\n");
    else if (pass)              std::printf("  => PASS but NOT worth building (Amdahl < 1.15x)\n");
    else if (pooled.acc1_ptr < 0.50) std::printf("  => KILL (acc@1 < 0.50 — head marks READ, not next-emit)\n");
    else                        std::printf("  => WEAK (see tables: redundant-with-PLD / tier-specific / ungated-only)\n");
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// COV1 — coverage. The inverse of N3: if NO generated token ever attends to a
// prompt span, was that span ignored? Product: the missed-lines audit (dropped
// line item, buried override) — errors of OMISSION citations can't see. Labels
// come from the OUTPUT (USED/DROPPED); VALUE/FILLER anchor the scale. Gated by
// COVERAGE=1; N3/N3b/CG1/DP1 untouched. Decode-step rows only (prefill tap is
// out of scope). Taps ALL 10 attention layers.
// ═════════════════════════════════════════════════════════════════════════
enum { CT_TARGET = 0, CT_VALUE = 1, CT_FILLER = 2 };
struct CovTarget { std::string marker, used; int cls; };
struct CovPrompt { std::string tag, body, instr; char close; std::vector<CovTarget> tg; };
struct SpanRec {
    std::string prompt, marker; int cls; int label; int len; // label: 1 USED,0 DROPPED,-1 anchor
    double sc[12][4];   // [source][scalar]; sources: 0=L3H13, 1..10=layer slot, 11=all10
};
static const char* cov_scalar_name(int s) {
    static const char* n[4] = {"peak", "mean", "maxsingle", "hitrate"}; return n[s];
}

// Fill sc[12][4] for span [lo,hi] over all decode steps.
static void span_scalars(const FreeRun& R, int lo, int hi, double sc[12][4],
                         const std::vector<int32_t>& attn_layers) {
    double peak[12] = {0}, sum[12] = {0}, mx[12] = {0}; int hit[12] = {0};
    int nst = (int)R.rows.size();
    const int nslot = (int)attn_layers.size();  // 10
    for (int t = 0; t < nst; ++t) {
        int n_kv = R.n_kv_at_step[t];
        double step_src[12]; for (int s = 0; s < 12; ++s) step_src[s] = 0;
        double ms_src[12];   for (int s = 0; s < 12; ++s) ms_src[s] = 0;
        double glob = 0, glob_ms = 0;
        for (int slot = 0; slot < nslot; ++slot) {
            double layer_best = 0, layer_ms = 0;
            for (int h = 0; h < R.n_head; ++h) {
                const float* rr = R.rows[t][slot].data() + (size_t)h * n_kv;
                double s = 0, mp = 0;
                for (int q = lo; q <= hi && q < n_kv; ++q) { s += rr[q]; if (rr[q] > mp) mp = rr[q]; }
                if (s > layer_best) layer_best = s;
                if (mp > layer_ms) layer_ms = mp;
                if (slot == 0 && h == FROZEN_HEAD) { step_src[0] = s; ms_src[0] = mp; }  // L3H13
            }
            // Indices 1..10 are the per-layer slots and 11 is the all-layer max,
            // so only the first 10 tapped layers get their own entry. On qwen36
            // (nslot=10) that is every layer and this guard never fires; on a
            // 48-layer gemma4 it is what stops `step_src[1+slot]` running off the
            // end of a double[12] and corrupting the stack. The all-layer max
            // below still accumulates over ALL slots.
            if (1 + slot < 11) { step_src[1 + slot] = layer_best; ms_src[1 + slot] = layer_ms; }
            if (layer_best > glob) glob = layer_best;
            if (layer_ms > glob_ms) glob_ms = layer_ms;
        }
        step_src[11] = glob; ms_src[11] = glob_ms;                          // all10
        for (int s = 0; s < 12; ++s) {
            if (step_src[s] > peak[s]) peak[s] = step_src[s];
            sum[s] += step_src[s];
            if (step_src[s] >= 0.05) hit[s]++;
            if (ms_src[s] > mx[s]) mx[s] = ms_src[s];
        }
    }
    for (int s = 0; s < 12; ++s) {
        sc[s][0] = peak[s]; sc[s][1] = nst ? sum[s] / nst : 0;
        sc[s][2] = mx[s];   sc[s][3] = nst ? (double)hit[s] / nst : 0;
    }
}

static std::vector<CovPrompt> cov_calib() {
    const std::string OBJ = "\nExtract the order as a JSON object with keys "
        "customer, date, quantity, unit_price.\nJSON:\n{";
    const std::string ARR = "\nExtract every line item as a JSON array; each "
        "element has keys product, quantity, unit_price.\nJSON:\n[";
    const std::string ARR3 = "\nExtract at most 3 line items as a JSON array; each "
        "element has keys product, quantity, unit_price.\nJSON:\n[";
    return {
      {"ov_buried",
       "From: orders@apex.example\nSubject: Order\n\nHello,\n\n"
       "Order for customer Apex Tooling Co.\nOrder date: 2025-08-14.\n"
       "Quantity: 200 units.\nUnit price: 15.00 USD.\n\nThanks,\nRita Nolan\n\n"
       "P.S. Correction: the quantity should be 500, not 200.\n", OBJ, '}',
       {{"the quantity should be 500", "500", CT_TARGET},
        {"Apex Tooling Co", "Apex", CT_VALUE}, {"Hello,", "", CT_FILLER}}},
      {"ov_top",
       "From: sales@blueridge.example\nSubject: Order (PRICE UPDATE)\n\n"
       "Note up front: the unit price must be 42.00, overriding any figure below.\n\n"
       "Hello,\n\nOrder for customer Blue Ridge Supply.\nOrder date: 2025-09-01.\n"
       "Quantity: 90 units.\nUnit price: 12.00 USD.\n\nRegards.\n", OBJ, '}',
       {{"the unit price must be 42.00", "42.00", CT_TARGET},
        {"Blue Ridge Supply", "Blue Ridge", CT_VALUE}, {"Hello,", "", CT_FILLER}}},
      {"li_aftersig",
       "From: buyer@harbor.example\nSubject: Supplies\n\nPlease send:\n"
       "- Carbon Fiber Tripod Stand, quantity 20, unit price 55.00\n"
       "- Aluminum Clamp Mount, quantity 60, unit price 8.50\n"
       "- Weatherproof Camera Housing, quantity 15, unit price 120.00\n\n"
       "Thanks,\nSam Ortiz\n\nP.S. Also add Portable LED Light Panel, quantity 30, unit price 45.00.\n",
       ARR, ']',
       {{"Portable LED Light Panel", "Portable LED Light Panel", CT_TARGET},
        {"Carbon Fiber Tripod Stand", "Carbon Fiber Tripod Stand", CT_TARGET},
        {"Aluminum Clamp Mount", "Aluminum Clamp Mount", CT_TARGET},
        {"Weatherproof Camera Housing", "Weatherproof Camera Housing", CT_TARGET},
        {"Please send:", "", CT_FILLER}}},
      {"li_cap",
       "From: orders@meadow.example\nSubject: Bulk Order\n\nWe would like:\n"
       "- Organic Wheat Flour Sack, quantity 50, unit price 12.40\n"
       "- Cold Pressed Sunflower Oil, quantity 18, unit price 74.00\n"
       "- Pink Rock Salt Bag, quantity 220, unit price 2.85\n"
       "- Dried Basil Leaf Pouch, quantity 95, unit price 6.10\n"
       "- Whole Black Peppercorn Jar, quantity 140, unit price 9.30\n\nRegards.\n",
       ARR3, ']',
       {{"Organic Wheat Flour Sack", "Organic Wheat Flour Sack", CT_TARGET},
        {"Cold Pressed Sunflower Oil", "Cold Pressed Sunflower Oil", CT_TARGET},
        {"Pink Rock Salt Bag", "Pink Rock Salt Bag", CT_TARGET},
        {"Dried Basil Leaf Pouch", "Dried Basil Leaf Pouch", CT_TARGET},
        {"Whole Black Peppercorn Jar", "Whole Black Peppercorn Jar", CT_TARGET},
        {"We would like:", "", CT_FILLER}}},
      {"ov_format",
       "From: orders@zen.example\nSubject: Order\n\nHello,\n\n"
       "Formatting rule: always write the date in ISO 8601 like 2025-03-05.\n\n"
       "Order for customer Zenith Works.\nOrder date: March 5, 2025.\n"
       "Quantity: 44 units.\nUnit price: 12.00 USD.\n\nRegards.\n", OBJ, '}',
       {{"write the date in ISO 8601", "2025-03-05", CT_TARGET},
        {"Zenith Works", "Zenith", CT_VALUE}, {"Hello,", "", CT_FILLER}}},
    };
}
static std::vector<CovPrompt> cov_held() {
    const std::string OBJ = "\nExtract the order as a JSON object with keys "
        "customer, date, quantity, unit_price.\nJSON:\n{";
    const std::string ARR = "\nExtract every line item as a JSON array; each "
        "element has keys product, quantity, unit_price.\nJSON:\n[";
    const std::string ARR3 = "\nExtract at most 3 line items as a JSON array; each "
        "element has keys product, quantity, unit_price.\nJSON:\n[";
    return {
      {"ov_buried2",
       "From: orders@crest.example\nSubject: PO\n\nHello,\n\n"
       "Order for customer Crestwood Metals.\nOrder date: 2024-11-20.\n"
       "Quantity: 75 units.\nUnit price: 30.00 USD.\n\nBest,\nDan Pryce\n\n"
       "P.S. Please note the quantity is actually 610, disregard the 75 above.\n", OBJ, '}',
       {{"the quantity is actually 610", "610", CT_TARGET},
        {"Crestwood Metals", "Crestwood", CT_VALUE}, {"Hello,", "", CT_FILLER}}},
      {"ov_top2",
       "From: sales@sum.example\nSubject: Order\n\n"
       "Up front: use unit price 7.40, this overrides the price listed below.\n\n"
       "Hi,\n\nOrder for customer Summit Foods.\nOrder date: 2025-07-22.\n"
       "Quantity: 275 units.\nUnit price: 99.00 USD.\n\nThanks.\n", OBJ, '}',
       {{"use unit price 7.40", "7.40", CT_TARGET},
        {"Summit Foods", "Summit", CT_VALUE}, {"Hi,", "", CT_FILLER}}},
      {"li_aftersig2",
       "From: buyer@iron.example\nSubject: Parts\n\nWe require:\n"
       "- Tungsten Carbide Drill Set, quantity 65, unit price 43.30\n"
       "- Hydraulic Quick Coupler, quantity 140, unit price 8.90\n\n"
       "Best,\nLee\n\nP.S. Add also Anti Vibration Mounting Pad, quantity 260, unit price 2.10.\n",
       ARR, ']',
       {{"Anti Vibration Mounting Pad", "Anti Vibration Mounting Pad", CT_TARGET},
        {"Tungsten Carbide Drill Set", "Tungsten Carbide Drill Set", CT_TARGET},
        {"Hydraulic Quick Coupler", "Hydraulic Quick Coupler", CT_TARGET},
        {"We require:", "", CT_FILLER}}},
      {"li_cap2",
       "From: orders@blue.example\nSubject: Catering\n\nPlease send:\n"
       "- Compostable Kraft Paper Plates, quantity 1200, unit price 0.18\n"
       "- Insulated Coffee Cups, quantity 800, unit price 0.42\n"
       "- Wooden Cutlery Set, quantity 950, unit price 0.31\n"
       "- Napkin Dispenser Refill, quantity 300, unit price 1.95\n"
       "- Recycled Straws Box, quantity 500, unit price 0.75\n\nRegards.\n",
       ARR3, ']',
       {{"Compostable Kraft Paper Plates", "Compostable Kraft Paper Plates", CT_TARGET},
        {"Insulated Coffee Cups", "Insulated Coffee Cups", CT_TARGET},
        {"Wooden Cutlery Set", "Wooden Cutlery Set", CT_TARGET},
        {"Napkin Dispenser Refill", "Napkin Dispenser Refill", CT_TARGET},
        {"Recycled Straws Box", "Recycled Straws Box", CT_TARGET},
        {"Please send:", "", CT_FILLER}}},
      {"ov_nord",
       "From: orders@nord.example\nSubject: Order\n\nHi,\n\n"
       "Order for customer Nordic Tech.\nOrder date: 2025-04-11.\n"
       "Quantity: 12 units.\nUnit price: 415.00 USD.\n\nCheers,\nMara\n\n"
       "P.S. Scratch that quantity — it should be 360 units.\n", OBJ, '}',
       {{"it should be 360", "360", CT_TARGET},
        {"Nordic Tech", "Nordic", CT_VALUE}, {"Hi,", "", CT_FILLER}}},
    };
}

static int run_coverage_probe(ForwardPassBase* fp, ggml_backend_sched_t sched,
                              Tokenizer* tok, const ModelMetadata& meta,
                              const std::vector<int32_t>& attn_layers) {
    std::vector<int> tap_layers(attn_layers.begin(), attn_layers.end());  // all 10
    std::vector<SpanRec> cal, hel;
    bool sanity_sum_done = false;

    auto run_set = [&](std::vector<CovPrompt> set, std::vector<SpanRec>& out) {
        for (auto& cp : set) {
            FreeRun R = run_freegen(fp, sched, tok, meta, cp.body, cp.instr,
                                    tap_layers, 320, false, cp.close);
            std::string gt = R.gen_text; for (char& c : gt) if (c == '\n') c = ' ';
            if (gt.size() > 130) gt = gt.substr(0, 130) + "...";
            std::printf("[%s] gen=%zu  %s\n", cp.tag.c_str(), R.gen_tokens.size(), gt.c_str());
            if (!sanity_sum_done && !R.rows.empty()) {  // rows sum to 1 sanity
                int n_kv = R.n_kv_at_step[0]; double s = 0;
                for (int q = 0; q < n_kv; ++q) s += R.rows[0][0][q];
                std::printf("  [sanity] L3 head0 step0 row sum = %.5f\n", s);
                sanity_sum_done = true;
            }
            for (auto& tg : cp.tg) {
                int lo, hi;
                if (!find_token_span(tok, R.prompt_tokens, cp.body, tg.marker, lo, hi)) {
                    std::printf("  WARN marker not found: %s\n", tg.marker.c_str());
                    continue;
                }
                SpanRec r; r.prompt = cp.tag; r.marker = tg.marker; r.cls = tg.cls; r.len = hi - lo + 1;
                r.label = tg.cls != CT_TARGET ? -1
                        : (R.gen_text.find(tg.used) != std::string::npos ? 1 : 0);
                span_scalars(R, lo, hi, r.sc, attn_layers);
                out.push_back(r);
            }
        }
    };
    std::printf("\n---- generations + labels ----\n");
    run_set(cov_calib(), cal);
    std::printf("---- held ----\n");
    run_set(cov_held(), hel);

    // Self-check: N3 citation reproduces — a VALUE span's L3H13 peak is high.
    double vpeak = 0; int vn = 0;
    for (auto& r : cal) if (r.cls == CT_VALUE) { vpeak += r.sc[0][0]; vn++; }
    std::printf("\n[self-check] VALUE-span L3H13 peak mean = %.3f (N3: consulted spans peak high)\n",
                vn ? vpeak / vn : 0);

    auto count = [](const std::vector<SpanRec>& v, int cls, int lab) {
        int n = 0; for (auto& r : v) if (r.cls == cls && (lab < -1 || r.label == lab)) n++; return n; };
    std::printf("  calib: USED=%d DROPPED=%d VALUE=%d FILLER=%d | held: USED=%d DROPPED=%d\n",
                count(cal, CT_TARGET, 1), count(cal, CT_TARGET, 0),
                count(cal, CT_VALUE, -2), count(cal, CT_FILLER, -2),
                count(hel, CT_TARGET, 1), count(hel, CT_TARGET, 0));

    // ── Scalar+source selection (calib TARGET USED vs DROPPED) ────────────────
    auto pairs = [&](const std::vector<SpanRec>& v, int s, int sc) {
        std::vector<std::pair<double, bool>> d;
        for (auto& r : v) if (r.cls == CT_TARGET) d.push_back({r.sc[s][sc], r.label == 1});
        return d;
    };
    auto src_name = [&](int s) -> std::string {
        if (s == 0) return "L3H13";
        if (s == 11) return "all10";
        return "layer" + std::to_string(attn_layers[s - 1]);
    };
    std::printf("\n---- selection (calib USED vs DROPPED; best per source) ----\n");
    Thr best; int best_s = 0, best_sc = 0;
    for (int s = 0; s < 12; ++s) for (int sc = 0; sc < 4; ++sc) {
        Thr t = best_threshold(pairs(cal, s, sc));
        if (t.acc > best.acc) { best = t; best_s = s; best_sc = sc; }
    }
    // print a few notable sources at the winning scalar
    for (int s : {0, 11}) {
        for (int sc = 0; sc < 4; ++sc) {
            Thr t = best_threshold(pairs(cal, s, sc));
            std::printf("  %-8s %-9s thr %.3f dir %+d  calib %.0f%%\n",
                        src_name(s).c_str(), cov_scalar_name(sc), t.t, t.dir, 100 * t.acc);
        }
    }
    std::printf("  >>> FROZEN source=%s scalar=%s thr=%.3f dir %+d (calib %.0f%%)\n",
                src_name(best_s).c_str(), cov_scalar_name(best_sc), best.t, best.dir, 100 * best.acc);

    double held_acc = apply_thr(best, pairs(hel, best_s, best_sc));
    std::printf("  HELD-OUT: %.0f%%\n", 100 * held_acc);

    // ── Per-class medians (frozen source/scalar, pooled) ──────────────────────
    std::vector<SpanRec> all = cal; all.insert(all.end(), hel.begin(), hel.end());
    auto med_cls = [&](int cls, int lab) {
        std::vector<double> v;
        for (auto& r : all) if (r.cls == cls && (lab < -1 || r.label == lab)) v.push_back(r.sc[best_s][best_sc]);
        return median(v);
    };
    double mF = med_cls(CT_FILLER, -2), mD = med_cls(CT_TARGET, 0),
           mU = med_cls(CT_TARGET, 1), mV = med_cls(CT_VALUE, -2);
    std::printf("\n---- per-class medians (frozen %s.%s) ----\n", src_name(best_s).c_str(),
                cov_scalar_name(best_sc));
    // Spec ordering is FILLER <= DROPPED <= USED (VALUE is a top anchor, both
    // it and USED are "consulted" so they tie ~high; not part of the gate).
    bool ordered = mF <= mD && mD <= mU;
    std::printf("  FILLER %.3f  <=  DROPPED %.3f  <=  USED %.3f   [VALUE anchor %.3f]  (%s)\n",
                mF, mD, mU, mV, ordered ? "ordering holds" : "ordering BROKEN");

    // DROPPED-span dump — the kill-shape evidence (esp. naturally-ignored
    // overrides: are they LOW mass [coverage works] or HIGH [attended-but-ignored]?).
    std::printf("  DROPPED spans (frozen scalar): ");
    for (auto& r : all) if (r.cls == CT_TARGET && r.label == 0)
        std::printf("%s/%.2f ", r.prompt.c_str(), r.sc[best_s][best_sc]);
    std::printf("\n");

    // ── Confusion (held TARGET spans, with names) ─────────────────────────────
    std::printf("\n---- held-out confusion (frozen) ----\n");
    for (auto& r : hel) if (r.cls == CT_TARGET) {
        double x = r.sc[best_s][best_sc];
        bool pred_used = best.dir > 0 ? x >= best.t : x <= best.t;
        bool ok = pred_used == (r.label == 1);
        std::printf("  %-11s %-30s len%2d  %s  true=%-7s pred=%-7s %s\n",
                    r.prompt.c_str(), ("\"" + r.marker.substr(0, 26) + "\"").c_str(), r.len,
                    "", r.label == 1 ? "USED" : "DROPPED", pred_used ? "USED" : "DROPPED",
                    ok ? "ok" : "**MISS**");
    }

    // ── Eyeball ───────────────────────────────────────────────────────────────
    std::printf("\n---- eyeball (span | frozen scalar | label) ----\n");
    int shown = 0;
    for (auto& r : hel) { if (shown >= 3) break; if (r.cls != CT_TARGET) continue;
        std::printf("  %-11s \"%s\"  %s=%.3f  %s\n", r.prompt.c_str(), r.marker.c_str(),
                    cov_scalar_name(best_sc), r.sc[best_s][best_sc],
                    r.label == 1 ? "USED" : "DROPPED");
        shown++;
    }

    // ── Verdict + kill-shape attribution ──────────────────────────────────────
    bool shape1 = mD >= 0.7 * mU && mU > 0;                 // DROPPED ~ USED
    bool shape2 = mU <= 1.5 * std::max(mF, 1e-6);           // USED ~ FILLER
    std::printf("\n================ VERDICT (COV1) ================\n");
    std::printf("  held USED-vs-DROPPED separation: %.0f%%  (PASS >=90%%, WEAK 75-90%%)\n", 100 * held_acc);
    if (held_acc >= 0.90 && ordered) std::printf("  => PASS\n");
    else if (held_acc >= 0.75)       std::printf("  => WEAK (see per-type breakdown)\n");
    else if (shape1)                 std::printf("  => KILL shape 1 (attended-but-ignored: DROPPED~USED mass)\n");
    else if (shape2)                 std::printf("  => KILL shape 2 (indirect flow: USED~FILLER mass; DeltaNet/prefill carries use)\n");
    else                             std::printf("  => KILL / inconclusive — see medians\n");
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// CF1 — the conflict flag. N3 §6 saw once: two competing dates → the emitted
// token's row split its mass bimodally across BOTH spans. CF1 formalizes it:
// from the row alone, tell "≥2 candidate sources" (CONFLICT) from "single clean
// source" (CLEAN), and localize both. CG1 proved confidence can't see this
// (conflicted fields are maximally confident); the mass split is the only
// signal. Gated by CONFLICT_FLAG=1; prior paths untouched. Zero engine edits.
// ═════════════════════════════════════════════════════════════════════════
struct Peaks { int p1 = -1, p2 = -1; double m1 = 0, m2 = 0; };

// Segment BODY [lo,hi) into disjoint peaks (mass≥0.02; a ≥4-token gap starts a
// new peak). Return the two largest by summed mass.
static Peaks segment(const std::vector<double>& row, int lo, int hi) {
    struct G { double mass = 0, best = -1; int pos = -1; };
    std::vector<G> gs; int prev = -100;
    for (int q = lo; q < hi && q < (int)row.size(); ++q) {
        if (row[q] < 0.02) continue;
        if (q - prev >= 4) gs.push_back(G());
        G& g = gs.back(); g.mass += row[q]; if (row[q] > g.best) { g.best = row[q]; g.pos = q; }
        prev = q;
    }
    std::sort(gs.begin(), gs.end(), [](const G& a, const G& b) { return a.mass > b.mass; });
    Peaks r;
    if (gs.size() >= 1) { r.p1 = gs[0].pos; r.m1 = gs[0].mass; }
    if (gs.size() >= 2) { r.p2 = gs[1].pos; r.m2 = gs[1].mass; }
    return r;
}
// Row for source s at step: s==0 = L3H13 (proper distribution); s>=1 = layer
// slot (s-1) max-over-heads.
static std::vector<double> cf_row(const FreeRun& R, int step, int s, int n_kv) {
    std::vector<double> row(n_kv, 0.0);
    if (s == 0) { const float* rr = R.rows[step][0].data() + (size_t)FROZEN_HEAD * n_kv;
        for (int q = 0; q < n_kv; ++q) row[q] = rr[q]; }
    else { int slot = s - 1;
        for (int h = 0; h < R.n_head; ++h) { const float* rr = R.rows[step][slot].data() + (size_t)h * n_kv;
            for (int q = 0; q < n_kv; ++q) if (rr[q] > row[q]) row[q] = rr[q]; } }
    return row;
}

enum { K_CLEAN = 0, K_CONFLICT = 1, K_DERIVED = 2, K_DUP = 3 };
struct CfField { std::string name; int kind; std::string v1, v2; };
struct CfPrompt { std::string tag, body, instr; std::vector<CfField> f; };
struct CfRec { std::string prompt, field; int kind; int label; // label 1 CONFLICT,0 CLEAN
    int emit_lo = -1, emit_hi = -1, alt_lo = -1, alt_hi = -1;
    Peaks pk[11][2]; };  // [source][scope 0=first,1=mean]

static std::vector<CfPrompt> cf_set(bool held);

static int run_conflict_probe(ForwardPassBase* fp, ggml_backend_sched_t sched,
                              Tokenizer* tok, const ModelMetadata& meta,
                              const std::vector<int32_t>& attn_layers) {
    std::vector<int> tap_layers(attn_layers.begin(), attn_layers.end());
    const std::vector<std::string> keys = ungrounded_keys();

    // ── Self-check: reproduce N3 §6 Prompt C split (teacher-forced) ───────────
    {
        std::string pC =
            "From: sales@initech.example\nSubject: Order Confirmation\n\nHi,\n\n"
            "Order for customer Initech LLC.\nOrder date: 2024-06-09.\n"
            "Requested delivery date: 2024-07-20.\nQuantity: 1.250 units.\n"
            "Unit price: EUR 8,75.\nOrder total: EUR 10.937,50.\n\nBest.\n\n"
            "Extract as JSON with keys customer, order_date, quantity, unit_price, total:\n";
        std::string cC = "{\"customer\": \"Initech LLC\", \"order_date\": \"2024-06-09\", "
            "\"quantity\": 1250, \"unit_price\": 8.75, \"total\": 10937.50}";
        std::vector<Field> fC = {{"customer","Initech LLC",-1,-1,""},
            {"order_date","2024-06-09",-1,-1,""},{"quantity","1.250",-1,-1,"1250"},
            {"unit_price","8,75",-1,-1,"8.75"},{"total","10.937,50",-1,-1,"10937.50"}};
        PromptRun R = run_prompt(fp, sched, tok, meta, "Cself", pC, cC, fC, false);
        int dlo, dhi; find_token_span(tok, R.prompt_tokens, pC, "2024-07-20", dlo, dhi);
        const Field& od = R.fields[1];
        int v = -1; for (int i = 1; i < (int)R.comp_tokens.size(); ++i)
            if (R.comp_field[i] == 1) { v = i; break; }
        int step = v - 1, n_kv = R.n_kv_at_step[step];
        const float* rr = R.rows[step][0].data() + (size_t)13 * n_kv;
        double mo = 0, md = 0;
        for (int q = od.lo; q <= od.hi; ++q) mo += rr[q];
        for (int q = dlo; q <= dhi; ++q) md += rr[q];
        std::printf("[self-check N3 §6] ORDER span=%.3f DELIVERY span=%.3f (N3: 0.582/0.193)\n", mo, md);
    }

    std::vector<CfRec> cal, hel;
    auto run_set = [&](std::vector<CfPrompt> set, std::vector<CfRec>& out) {
        for (auto& cp : set) {
            FreeRun R = run_freegen(fp, sched, tok, meta, cp.body, cp.instr, tap_layers, 300, false, '}');
            std::string gt = R.gen_text; for (char& c : gt) if (c == '\n') c = ' ';
            if (gt.size() > 120) gt = gt.substr(0, 120) + "...";
            std::printf("[%s] %s\n", cp.tag.c_str(), gt.c_str());
            auto gfs = parse_fields(tok, R, keys);
            for (auto& cf : cp.f) {
                const GenField* gf = nullptr;
                for (auto& g : gfs) if (g.name == cf.name && g.a >= 0) { gf = &g; break; }
                if (!gf) continue;
                CfRec rec; rec.prompt = cp.tag; rec.field = cf.name; rec.kind = cf.kind;
                rec.label = (cf.kind == K_CONFLICT) ? 1 : 0;
                if (cf.kind == K_CONFLICT) {
                    std::string emit, alt;
                    if (R.gen_text.find(cf.v1) != std::string::npos &&
                        gf->value.find(cf.v1) != std::string::npos) { emit = cf.v1; alt = cf.v2; }
                    else if (gf->value.find(cf.v2) != std::string::npos) { emit = cf.v2; alt = cf.v1; }
                    else { std::printf("  (excl %s: emitted neither cand: \"%s\")\n",
                                       cf.name.c_str(), gf->value.c_str()); continue; }
                    find_token_span(tok, R.prompt_tokens, cp.body, emit, rec.emit_lo, rec.emit_hi);
                    find_token_span(tok, R.prompt_tokens, cp.body, alt, rec.alt_lo, rec.alt_hi);
                }
                // compute peaks for all sources/scopes
                int a = gf->a, b = gf->b;
                for (int s = 0; s < 11; ++s) {
                    // scope 0: first value token
                    int st0 = a - 1;
                    if (st0 >= 0 && st0 < (int)R.rows.size()) {
                        int n_kv = R.n_kv_at_step[st0];
                        int hi = std::min(R.instr_tok, n_kv);
                        rec.pk[s][0] = segment(cf_row(R, st0, s, n_kv), 1, hi);
                    }
                    // scope 1: mean over value tokens' provenance steps
                    int lo_s = a - 1, hi_s = b - 1, cnt = 0;
                    std::vector<double> mean; int n_kv0 = 0;
                    for (int st = lo_s; st <= hi_s; ++st) {
                        if (st < 0 || st >= (int)R.rows.size()) continue;
                        int n_kv = R.n_kv_at_step[st];
                        if (mean.empty()) { mean.assign(n_kv, 0.0); n_kv0 = n_kv; }
                        std::vector<double> r = cf_row(R, st, s, n_kv);
                        for (int q = 0; q < n_kv0 && q < n_kv; ++q) mean[q] += r[q];
                        cnt++;
                    }
                    if (cnt) { for (double& x : mean) x /= cnt;
                        rec.pk[s][1] = segment(mean, 1, std::min(R.instr_tok, n_kv0)); }
                }
                out.push_back(rec);
            }
        }
    };
    std::printf("\n---- calibration ----\n"); run_set(cf_set(false), cal);
    std::printf("---- held ----\n"); run_set(cf_set(true), hel);

    auto cnt = [](const std::vector<CfRec>& v, int k) {
        int n = 0; for (auto& r : v) if (r.kind == k) n++; return n; };
    std::printf("\n  calib: CONFLICT=%d CLEAN=%d DERIVED=%d DUP=%d | held: CONFLICT=%d CLEAN=%d DERIVED=%d DUP=%d\n",
                cnt(cal, K_CONFLICT), cnt(cal, K_CLEAN), cnt(cal, K_DERIVED), cnt(cal, K_DUP),
                cnt(hel, K_CONFLICT), cnt(hel, K_CLEAN), cnt(hel, K_DERIVED), cnt(hel, K_DUP));

    // ── Selection: CONFLICT vs CLEAN on calib; sweep source×scope×θ×ρ ─────────
    auto predict = [](const CfRec& r, int s, int sc, double th, double rho) {
        const Peaks& p = r.pk[s][sc];
        return p.m2 >= th && p.m1 > 0 && (p.m2 / p.m1) >= rho;
    };
    auto acc_on = [&](const std::vector<CfRec>& v, int s, int sc, double th, double rho) {
        int ok = 0, n = 0;
        for (auto& r : v) if (r.kind == K_CLEAN || r.kind == K_CONFLICT) {
            n++; bool pc = predict(r, s, sc, th, rho); if (pc == (r.kind == K_CONFLICT)) ok++;
        }
        return n ? (double)ok / n : 0;
    };
    int bs = 0, bsc = 0; double bth = 0, brho = 0, bacc = -1;
    for (int s = 0; s < 11; ++s) for (int sc = 0; sc < 2; ++sc)
        for (double th = 0.05; th <= 0.30001; th += 0.05)
            for (double rho = 0.1; rho <= 0.50001; rho += 0.1) {
                double a = acc_on(cal, s, sc, th, rho);
                if (a > bacc) { bacc = a; bs = s; bsc = sc; bth = th; brho = rho; }
            }
    auto sname = [&](int s) { return s == 0 ? std::string("L3H13")
        : "layer" + std::to_string(attn_layers[s - 1]) + "-maxH"; };
    std::printf("\n---- selection ----\n  >>> FROZEN source=%s scope=%s theta=%.2f rho=%.1f (calib %.0f%%)\n",
                sname(bs).c_str(), bsc == 0 ? "first" : "mean", bth, brho, 100 * bacc);
    double hacc = acc_on(hel, bs, bsc, bth, brho);
    std::printf("  HELD-OUT CONFLICT-vs-CLEAN: %.0f%%\n", 100 * hacc);

    // ── Confusion incl DERIVED / DUP (held) + localization ────────────────────
    std::printf("\n---- held-out confusion (frozen) ----\n");
    int loc_ok = 0, loc_n = 0;
    for (auto& r : hel) {
        bool pc = predict(r, bs, bsc, bth, brho);
        const Peaks& p = r.pk[bs][bsc];
        const char* kn = r.kind == K_CONFLICT ? "CONFLICT" : r.kind == K_CLEAN ? "CLEAN"
                       : r.kind == K_DERIVED ? "DERIVED" : "DUP";
        bool correct = (r.kind == K_CONFLICT) == pc;
        std::printf("  %-7s %-11s %-9s pred=%-8s m1=%.2f@%d m2=%.2f@%d %s\n",
                    r.prompt.c_str(), r.field.c_str(), kn, pc ? "CONFLICT" : "clean",
                    p.m1, p.p1, p.m2, p.p2,
                    r.kind <= K_CONFLICT ? (correct ? "ok" : "**MISS**") : "");
        if (r.kind == K_CONFLICT && pc) {  // localization
            loc_n++;
            bool l1 = p.p1 >= r.emit_lo - 2 && p.p1 <= r.emit_hi + 2;
            bool l2 = p.p2 >= r.alt_lo - 2 && p.p2 <= r.alt_hi + 2;
            if (l1 && l2) loc_ok++;
        }
    }
    std::printf("  localization: %d/%d detected conflicts put peak1@emitted±2 AND peak2@alt±2\n",
                loc_ok, loc_n);
    // DERIVED / DUP false-alarm rates
    auto far = [&](int k) { int n = 0, f = 0; for (auto& r : hel) if (r.kind == k) {
        n++; if (predict(r, bs, bsc, bth, brho)) f++; }
        std::printf("  %s flagged CONFLICT: %d/%d\n", k == K_DERIVED ? "DERIVED" : "DUP", f, n); };
    far(K_DERIVED); far(K_DUP);

    // ── Eyeball ───────────────────────────────────────────────────────────────
    std::printf("\n---- eyeball (field | peaks | label) ----\n");
    int shown = 0;
    for (auto& r : hel) { if (shown >= 3) break; if (r.kind != K_CONFLICT) continue;
        const Peaks& p = r.pk[bs][bsc];
        std::printf("  %-7s %-10s CONFLICT emit[%d..%d] alt[%d..%d]  p1=%.2f@%d p2=%.2f@%d\n",
                    r.prompt.c_str(), r.field.c_str(), r.emit_lo, r.emit_hi, r.alt_lo, r.alt_hi,
                    p.m1, p.p1, p.m2, p.p2);
        shown++;
    }

    // ── Per-class median m2 ───────────────────────────────────────────────────
    std::vector<CfRec> all = cal; all.insert(all.end(), hel.begin(), hel.end());
    auto medm2 = [&](int k) { std::vector<double> v;
        for (auto& r : all) if (r.kind == k) v.push_back(r.pk[bs][bsc].m2); return median(v); };
    std::printf("\n  median 2nd-peak mass: CLEAN %.3f  CONFLICT %.3f  DERIVED %.3f  DUP %.3f\n",
                medm2(K_CLEAN), medm2(K_CONFLICT), medm2(K_DERIVED), medm2(K_DUP));

    // ── Verdict ───────────────────────────────────────────────────────────────
    double locf = loc_n ? (double)loc_ok / loc_n : 0;
    double der_far = 0; { int n = 0, f = 0; for (auto& r : hel) if (r.kind == K_DERIVED) { n++;
        if (predict(r, bs, bsc, bth, brho)) f++; } der_far = n ? (double)f / n : 0; }
    std::printf("\n================ VERDICT (CF1) ================\n");
    std::printf("  detection %.0f%% (PASS>=90), localization %.0f%% (PASS>=80), DERIVED false-alarm %.0f%%\n",
                100 * hacc, 100 * locf, 100 * der_far);
    if (hacc >= 0.90 && locf >= 0.80 && der_far < 0.5) std::printf("  => PASS\n");
    else if (hacc >= 0.90) std::printf("  => WEAK (detection ok; localization<80%% or DERIVED inseparable)\n");
    else std::printf("  => KILL / inconclusive — see tables\n");
    return 0;
}

static std::vector<CfPrompt> cf_set(bool held) {
    const std::string OBJ = "\nExtract the order as a JSON object with keys "
        "customer, date, quantity, unit_price, total.\nJSON:\n{";
    if (!held) return {
      {"cf1","From: buyer@apex.example\nSubject: Order\n\nHello,\n\n"
       "Order for customer Apex Tooling.\nOrder date: 2024-06-09.\n"
       "Requested delivery date: 2024-08-15.\nWe need a quantity of 40 boxes.\n"
       "Unit price: 18.50 USD.\n\nThanks,\nRita\n\n"
       "P.S. Correction: the quantity should be 45, not 40.\n", OBJ,
       {{"quantity",K_CONFLICT,"40","45"},{"date",K_CONFLICT,"2024-06-09","2024-08-15"},
        {"customer",K_CLEAN,"Apex Tooling",""},{"unit_price",K_CLEAN,"18.50",""},
        {"total",K_DERIVED,"",""}}},
      {"cf2","From: orders@blue.example\nSubject: PO\n\nHi,\n\n"
       "Order for customer Blue Ridge Supply.\nOrder date: 2025-03-05.\n"
       "We need a quantity of 120 units.\nUnit price: 7.25 USD.\n\nRegards,\nDan\n\n"
       "P.S. Actually, change the quantity to 150.\n", OBJ,
       {{"quantity",K_CONFLICT,"120","150"},{"date",K_CLEAN,"2025-03-05",""},
        {"customer",K_CLEAN,"Blue Ridge Supply",""},{"unit_price",K_CLEAN,"7.25",""},
        {"total",K_DERIVED,"",""}}},
      {"cf3","From: sales@sum.example\nSubject: Order\n\nHello,\n\n"
       "Order for customer Summit Metals.\nInvoice date: 2025-07-22.\n"
       "Shipment date: 2025-09-01.\nWe need a quantity of 300 units.\n"
       "Unit price: 12.00 USD.\n\nBest.\n", OBJ,
       {{"date",K_CONFLICT,"2025-07-22","2025-09-01"},{"quantity",K_CLEAN,"300",""},
        {"customer",K_CLEAN,"Summit Metals",""},{"unit_price",K_CLEAN,"12.00",""},
        {"total",K_DERIVED,"",""}}},
      {"cf4","From: buyer@nord.example\nSubject: Order\n\nHi,\n\n"
       "Order for customer Nordic Tech.\nOrder date: 2024-11-20.\n"
       "Expected delivery date: 2024-12-10.\nQuantity: 60 units.\n"
       "Unit price: 9.99 USD, that is 9.99 per unit.\n\nCheers,\nMara\n\n"
       "P.S. Scratch that — make the quantity 75.\n", OBJ,
       {{"quantity",K_CONFLICT,"60","75"},{"date",K_CONFLICT,"2024-11-20","2024-12-10"},
        {"unit_price",K_DUP,"9.99",""},{"customer",K_CLEAN,"Nordic Tech",""},
        {"total",K_DERIVED,"",""}}},
      {"cf5","From: procurement@iron.example\nSubject: Parts\n\nHello,\n\n"
       "Order for customer Ironclad Works.\nOrder date: 2025-04-11.\n"
       "Requested delivery date: 2025-05-20.\nWe require a quantity of 200 pieces.\n"
       "Unit price: 43.30 USD.\n\nThanks.\n\nP.S. Update the quantity to 260 pieces.\n", OBJ,
       {{"quantity",K_CONFLICT,"200","260"},{"date",K_CONFLICT,"2025-04-11","2025-05-20"},
        {"customer",K_CLEAN,"Ironclad Works",""},{"unit_price",K_CLEAN,"43.30",""},
        {"total",K_DERIVED,"",""}}},
    };
    return {
      {"h1","From: buyer@crest.example\nSubject: Order\n\nHello,\n\n"
       "Order for customer Crestwood Metals.\nOrder date: 2024-02-14.\n"
       "Delivery date: 2024-03-20.\nQuantity: 80 units.\nUnit price: 6.40 USD.\n\n"
       "Best,\nDan\n\nP.S. Correction: the quantity should be 90.\n", OBJ,
       {{"quantity",K_CONFLICT,"80","90"},{"date",K_CONFLICT,"2024-02-14","2024-03-20"},
        {"customer",K_CLEAN,"Crestwood Metals",""},{"unit_price",K_CLEAN,"6.40",""},
        {"total",K_DERIVED,"",""}}},
      {"h2","From: orders@meadow.example\nSubject: PO\n\nHi,\n\n"
       "Order for customer Meadowfield Foods.\nOrder date: 2025-06-18.\n"
       "Quantity: 500 units.\nUnit price: 2.85 USD.\n\nRegards.\n\n"
       "P.S. Please change the quantity to 550.\n", OBJ,
       {{"quantity",K_CONFLICT,"500","550"},{"date",K_CLEAN,"2025-06-18",""},
        {"customer",K_CLEAN,"Meadowfield Foods",""},{"unit_price",K_CLEAN,"2.85",""},
        {"total",K_DERIVED,"",""}}},
      {"h3","From: sales@harbor.example\nSubject: Order\n\nHello,\n\n"
       "Order for customer Harbor Line Co.\nInvoice date: 2025-10-05.\n"
       "Shipment date: 2025-11-12.\nQuantity: 44 units.\nUnit price: 128.50 USD.\n\nBest.\n", OBJ,
       {{"date",K_CONFLICT,"2025-10-05","2025-11-12"},{"quantity",K_CLEAN,"44",""},
        {"customer",K_CLEAN,"Harbor Line Co",""},{"unit_price",K_CLEAN,"128.50",""},
        {"total",K_DERIVED,"",""}}},
      {"h4","From: buyer@zen.example\nSubject: Order\n\nHi,\n\n"
       "Order for customer Zenith Works, that is Zenith Works Ltd.\n"
       "Order date: 2024-09-08.\nExpected delivery date: 2024-10-01.\n"
       "Quantity: 15 units.\nUnit price: 415.00 USD.\n\nCheers.\n\n"
       "P.S. Make the quantity 20.\n", OBJ,
       {{"quantity",K_CONFLICT,"15","20"},{"date",K_CONFLICT,"2024-09-08","2024-10-01"},
        {"customer",K_DUP,"Zenith Works",""},{"unit_price",K_CLEAN,"415.00",""},
        {"total",K_DERIVED,"",""}}},
      {"h5","From: orders@delta.example\nSubject: Order\n\nHello,\n\n"
       "Order for customer Delta Foods.\nOrder date: 2026-02-28.\n"
       "Requested delivery date: 2026-03-15.\nQuantity: 30 units.\n"
       "Unit price: 3.15 USD.\n\nThanks.\n\nP.S. Update the quantity to 36.\n", OBJ,
       {{"quantity",K_CONFLICT,"30","36"},{"date",K_CONFLICT,"2026-02-28","2026-03-15"},
        {"customer",K_CLEAN,"Delta Foods",""},{"unit_price",K_CLEAN,"3.15",""},
        {"total",K_DERIVED,"",""}}},
    };
}

// ═════════════════════════════════════════════════════════════════════════
// SS1 — the stale-source alarm. The capstone: do the two PASS signals (N3
// citations + COV1 coverage) TRANSFER to multi-turn email threads and catch the
// ignored correction? Product: "emitted value cites Email 1; the correction in
// Email 2 was never consulted." NO conflict-attribution (CF1 dead). Frozen rules
// travel AS-IS (primary arm = transfer, no re-selection). Gated STALE_SOURCE=1.
// COV1 frozen coverage = source index 3 (layer-11 max-heads) peak, sc[3][0].
// ═════════════════════════════════════════════════════════════════════════
static const int COV_SRC = 3;          // span_scalars index for layer-11 max-heads
static const double COV_THR = 0.705;   // COV1 frozen threshold

enum { S_CLEAN = 0, S_CORR = 1, S_FILLER = 2 };
struct SsTarget { std::string name; int kind; std::string old_val, new_val; };
struct SsThread { std::string tag; std::string body, instr; std::vector<SsTarget> t; };
struct SsRec {
    std::string thread, field; int kind; int bucket;   // bucket 0 short,1 long
    int label = -1;                                     // 1 FRESH,0 STALE,-1 anchor
    double cov_new = 0, cov_old = 0;                    // coverage peak of new / old span
    bool cite_t1 = false, cite_t3 = false;             // emitted value cited in its source span
    bool t1_new = false, t1_old = false;               // reversal readout (FRESH)
};

static std::vector<SsThread> ss_threads(bool held);

static int run_stale_probe(ForwardPassBase* fp, ggml_backend_sched_t sched,
                           Tokenizer* tok, const ModelMetadata& meta,
                           const std::vector<int32_t>& attn_layers) {
    std::vector<int> tap(attn_layers.begin(), attn_layers.end());
    const std::vector<std::string> keys = {"customer", "delivery_city", "quantity", "unit_price"};

    auto cov_peak = [&](const FreeRun& R, int lo, int hi) {
        double sc[12][4]; span_scalars(R, lo, hi, sc, attn_layers); return sc[COV_SRC][0];
    };
    auto cite = [&](const FreeRun& R, int a, int lo, int hi, SsRec& r, bool set_span) {
        int step = a - 1; if (step < 0 || step >= (int)R.rows.size()) return;
        int n_kv = R.n_kv_at_step[step];
        auto tk = topk_head(R.rows[step][0], FROZEN_HEAD, n_kv, 3);
        auto in = [](int p, int l, int h) { return p >= l - 2 && p <= h + 2; };
        if (set_span) { if (!tk.empty()) r.cite_t1 = in(tk[0].first, lo, hi);
            for (auto& p : tk) if (in(p.first, lo, hi)) { r.cite_t3 = true; break; } }
        return;
    };

    // ── Self-checks: N3 citation + COV1 USED/DROPPED reproduce on ONE email ───
    {
        std::string b = "From: orders@apex.example\nSubject: Order\n\nHello,\n\n"
            "Order for customer Apex Tooling.\nDeliver to Hamburg.\n"
            "Quantity: 200 units.\nUnit price: 18.50 USD.\n\nThanks.\n";
        std::string ins = "\nExtract as JSON with keys customer, delivery_city, quantity, "
            "unit_price.\nJSON:\n{";
        FreeRun R = run_freegen(fp, sched, tok, meta, b, ins, tap, 120, false, '}');
        int n_kv = R.n_kv_at_step.empty() ? 0 : R.n_kv_at_step[0];
        double s = 0; for (int q = 0; q < n_kv; ++q) s += R.rows[0][0][q];
        auto gfs = parse_fields(tok, R, keys);
        int clo, chi; find_token_span(tok, R.prompt_tokens, b, "Apex Tooling", clo, chi);
        double cust_peak = cov_peak(R, clo, chi);
        std::printf("[self-check] rows sum=%.4f | COV1 'Apex Tooling' peak=%.3f (USED, >0.705 expect) | ", s, cust_peak);
        for (auto& g : gfs) if (g.name == "customer" && g.a >= 0) {
            SsRec tmp; cite(R, g.a, clo, chi, tmp, true);
            std::printf("N3 cust top3-in-span=%d\n", tmp.cite_t3 ? 1 : 0);
        }
    }

    std::vector<SsRec> cal, hel;
    auto run_set = [&](std::vector<SsThread> set, std::vector<SsRec>& out) {
        for (auto& th : set) {
            FreeRun R = run_freegen(fp, sched, tok, meta, th.body, th.instr, tap, 300, false, '}');
            int bucket = R.P < 900 ? 0 : 1;
            std::string gt = R.gen_text; for (char& c : gt) if (c == '\n') c = ' ';
            if (gt.size() > 110) gt = gt.substr(0, 110) + "...";
            std::printf("[%s P=%d %s] %s\n", th.tag.c_str(), R.P, bucket ? "LONG" : "short", gt.c_str());
            auto gfs = parse_fields(tok, R, keys);
            for (auto& tg : th.t) {
                SsRec r; r.thread = th.tag; r.field = tg.name; r.kind = tg.kind; r.bucket = bucket;
                const GenField* gf = nullptr;
                for (auto& g : gfs) if (g.name == tg.name && g.a >= 0) { gf = &g; break; }
                if (tg.kind == S_FILLER) {
                    int lo, hi; if (find_token_span(tok, R.prompt_tokens, th.body, tg.new_val, lo, hi))
                        r.cov_new = cov_peak(R, lo, hi);
                    out.push_back(r); continue;
                }
                if (!gf) { std::printf("  (no field %s)\n", tg.name.c_str()); continue; }
                if (tg.kind == S_CORR) {
                    int nlo, nhi, olo, ohi;
                    bool nf = find_token_span(tok, R.prompt_tokens, th.body, tg.new_val, nlo, nhi);
                    bool of = find_token_span(tok, R.prompt_tokens, th.body, tg.old_val, olo, ohi);
                    if (!nf || !of) { std::printf("  (span miss %s)\n", tg.name.c_str()); continue; }
                    bool emit_new = gf->value.find(tg.new_val) != std::string::npos;
                    bool emit_old = gf->value.find(tg.old_val) != std::string::npos;
                    if (!emit_new && !emit_old) { std::printf("  (excl %s: emitted \"%s\")\n",
                        tg.name.c_str(), gf->value.c_str()); continue; }
                    r.label = emit_new ? 1 : 0;
                    r.cov_new = cov_peak(R, nlo, nhi); r.cov_old = cov_peak(R, olo, ohi);
                    // citation in emitted source span; reversal readout
                    int slo = emit_new ? nlo : olo, shi = emit_new ? nhi : ohi;
                    cite(R, gf->a, slo, shi, r, true);
                    int step = gf->a - 1, n_kv = R.n_kv_at_step[step];
                    auto tk = topk_head(R.rows[step][0], FROZEN_HEAD, n_kv, 3);
                    auto inr = [](int p, int l, int h) { return p >= l - 2 && p <= h + 2; };
                    if (!tk.empty()) { r.t1_new = inr(tk[0].first, nlo, nhi); r.t1_old = inr(tk[0].first, olo, ohi); }
                } else {  // CLEAN
                    int lo, hi; if (!find_token_span(tok, R.prompt_tokens, th.body, tg.new_val, lo, hi)) continue;
                    r.cov_new = cov_peak(R, lo, hi);
                    cite(R, gf->a, lo, hi, r, true);
                }
                out.push_back(r);
            }
        }
    };
    std::printf("\n---- calibration threads ----\n"); run_set(ss_threads(false), cal);
    std::printf("---- held threads ----\n"); run_set(ss_threads(true), hel);

    auto cnt = [](const std::vector<SsRec>& v, int k, int lab) { int n = 0;
        for (auto& r : v) if (r.kind == k && (lab < -1 || r.label == lab)) n++; return n; };
    std::printf("\n  calib: FRESH=%d STALE=%d CLEAN=%d | held: FRESH=%d STALE=%d CLEAN=%d\n",
                cnt(cal, S_CORR, 1), cnt(cal, S_CORR, 0), cnt(cal, S_CLEAN, -2),
                cnt(hel, S_CORR, 1), cnt(hel, S_CORR, 0), cnt(hel, S_CLEAN, -2));

    // ── TRANSFER 1: frozen coverage separates FRESH vs STALE correction spans ─
    auto cov_report = [&](const std::vector<SsRec>& v, int bucket) {
        int n = 0, ok = 0, fresh = 0, stale = 0;
        for (auto& r : v) if (r.kind == S_CORR && (bucket < 0 || r.bucket == bucket)) {
            bool pf = r.cov_new >= COV_THR; n++;
            if (pf == (r.label == 1)) ok++;
            if (r.label == 1) fresh++; else stale++;
        }
        std::printf("    %-14s n=%d (FRESH %d STALE %d)  frozen-coverage acc=%.0f%%\n",
                    bucket < 0 ? "ALL" : bucket ? "long" : "short", n, fresh, stale,
                    n ? 100.0 * ok / n : 0);
    };
    std::printf("\n---- TRANSFER: coverage FRESH-vs-STALE (frozen layer11 peak>=0.705) ----\n");
    std::printf("  [held]\n"); cov_report(hel, -1); cov_report(hel, 0); cov_report(hel, 1);
    std::printf("  [calib]\n"); cov_report(cal, -1);

    // ── TRANSFER 2: frozen-head citation top-1/top-3 by length bucket ─────────
    auto cite_report = [&](const std::vector<SsRec>& v, int bucket) {
        int n = 0, t1 = 0, t3 = 0;
        for (auto& r : v) if (r.kind != S_FILLER && (bucket < 0 || r.bucket == bucket)) {
            n++; if (r.cite_t1) t1++; if (r.cite_t3) t3++;
        }
        std::printf("    %-6s n=%d  top1=%.0f%% top3=%.0f%%\n", bucket < 0 ? "ALL"
                    : bucket ? "long" : "short", n, n ? 100.0*t1/n : 0, n ? 100.0*t3/n : 0);
    };
    std::printf("\n---- TRANSFER: citation (frozen L3H13) by bucket, pooled cal+held ----\n");
    std::vector<SsRec> allr = cal; allr.insert(allr.end(), hel.begin(), hel.end());
    cite_report(allr, -1); cite_report(allr, 0); cite_report(allr, 1);

    // ── CF1-reversal rate on FRESH corrections ────────────────────────────────
    int rn = 0, r_new = 0, r_old = 0;
    for (auto& r : allr) if (r.kind == S_CORR && r.label == 1) { rn++;
        if (r.t1_new) r_new++; if (r.t1_old && !r.t1_new) r_old++; }
    std::printf("\n  FRESH-emission top-1 lands: newer-span %d/%d, old-span-only %d/%d (CF1 reversal readout)\n",
                r_new, rn, r_old, rn);

    // ── Per-class coverage medians + confusion ────────────────────────────────
    auto medc = [&](int k, int lab) { std::vector<double> v;
        for (auto& r : allr) if (r.kind == k && (lab < -1 || r.label == lab)) v.push_back(r.cov_new);
        return median(v); };
    std::printf("\n  median coverage peak: FILLER %.3f  STALE-corr %.3f  FRESH-corr %.3f  CLEAN %.3f\n",
                medc(S_FILLER, -2), medc(S_CORR, 0), medc(S_CORR, 1), medc(S_CLEAN, -2));
    std::printf("\n---- held correction confusion ----\n");
    for (auto& r : hel) if (r.kind == S_CORR) {
        bool pf = r.cov_new >= COV_THR;
        std::printf("  %-4s %-13s %-6s cov_new=%.2f cov_old=%.2f  pred=%-5s %s\n",
                    r.thread.c_str(), r.field.c_str(), r.label ? "FRESH" : "STALE",
                    r.cov_new, r.cov_old, pf ? "FRESH" : "STALE",
                    pf == (r.label == 1) ? "ok" : "**MISS**");
    }

    // ── Verdict ───────────────────────────────────────────────────────────────
    int hn = 0, hok = 0; for (auto& r : hel) if (r.kind == S_CORR) { hn++;
        if ((r.cov_new >= COV_THR) == (r.label == 1)) hok++; }
    double cov_acc = hn ? (double)hok / hn : 0;
    int cn = 0, c3 = 0; for (auto& r : allr) if (r.kind != S_FILLER) { cn++; if (r.cite_t3) c3++; }
    double cite3 = cn ? (double)c3 / cn : 0;
    double fresh_med = medc(S_CORR, 1), stale_med = medc(S_CORR, 0);
    int held_stale = 0, held_long = 0; for (auto& r : hel) {
        if (r.kind == S_CORR && r.label == 0) held_stale++;
        if (r.kind == S_CORR && r.bucket == 1) held_long++; }
    std::printf("\n================ VERDICT (SS1) ================\n");
    std::printf("  coverage FRESH-vs-STALE held=%.0f%% (PASS>=90); citation top3=%.0f%% (PASS>=90)\n",
                100 * cov_acc, 100 * cite3);
    std::printf("  held natural STALE=%d, held LONG-bucket corrections=%d\n", held_stale, held_long);
    bool inverted = stale_med >= fresh_med;  // the available STALE is attended-but-ignored
    if (held_stale < 3)
        std::printf("  => INCONCLUSIVE: too few natural STALE to test the bar (model too thorough); "
                    "the STALE we can induce is attended-but-ignored (cov high) — coverage can't catch it. "
                    "Long-context untested (padding short). See tables.\n");
    else if (cov_acc >= 0.90 && cite3 >= 0.90) std::printf("  => PASS (signals transfer to threads)\n");
    else if (inverted) std::printf("  => KILL: STALE spans are attended-but-ignored (coverage inverted)\n");
    else std::printf("  => WEAK (see tables)\n");
    return 0;
}

// Clean inline thread format (the "----- Email N -----" banners confused the
// model into echoing structure). Firm "output only JSON" kills the garbage.
static const char* SS_INS =
    "\n\nUsing the information above, output ONLY a JSON object with keys "
    "customer, delivery_city, quantity, unit_price. Do not repeat the emails "
    "or add other keys.\nJSON:\n{";
// ~200-token distractor email for padding long threads (no order fields).
static std::string ss_pad(int n) {
    std::string p;
    for (int i = 0; i < n; ++i) p +=
        "\nEmail (logistics, later): Hi team, circling back on the carrier "
        "schedule. The depot is closed Thursday for maintenance so pickups move "
        "to Friday. The fuel surcharge table was refreshed but does not touch "
        "this order. Pallet wrapping stock is low; a reorder is pending with the "
        "supplier. Payroll reminded everyone to file timesheets by end of week. "
        "The break room coffee machine is fixed. Parking level 2 is resurfaced "
        "Monday. None of this changes the order details. Regards, Sven from ops.\n";
    return p;
}
static std::vector<SsThread> ss_threads(bool held) {
    const std::string INS = SS_INS;
    if (!held) return {
      {"t1", "Email 1 (Mon): From Rita at Apex. Please set up order 7781 for "
       "customer Apex Tooling. Deliver to Hamburg. Quantity 200 units. Unit price 18.50.\n"
       "Email 2 (Tue): From Rita. Quick correction: deliver to Bremen, our Hamburg "
       "warehouse moved. Everything else is unchanged.\n", INS,
       {{"delivery_city",S_CORR,"Hamburg","Bremen"},{"customer",S_CLEAN,"","Apex Tooling"},
        {"quantity",S_CLEAN,"","200"},{"unit_price",S_CLEAN,"","18.50"}}},
      {"t2", "Email 1 (Mon): From Dan at Blue Ridge. Order for customer Blue Ridge "
       "Supply. Deliver to Lyon. Quantity 120 units. Unit price 7.25.\n"
       "Email 2 (Tue): From Dan. By the way, please bump the quantity to 180.\n"
       "Email 3 (Tue): From the warehouse desk. Confirmed, 120 units are staged and "
       "ready to ship to Lyon on schedule.\n", INS,   // distractor re-quotes OLD 120
       {{"quantity",S_CORR,"120","180"},{"delivery_city",S_CLEAN,"","Lyon"},
        {"customer",S_CLEAN,"","Blue Ridge Supply"},{"unit_price",S_CLEAN,"","7.25"}}},
      {"t3", "Email 1 (Mon): From Sam at Summit. Customer Summit Metals, deliver to "
       "Turin, quantity 300, unit price 12.00.\n"
       "Email 2 (Wed): From Sam. Heads up, our supplier raised rates; the unit price "
       "is now 13.50 per unit.\n", INS,
       {{"unit_price",S_CORR,"12.00","13.50"},{"delivery_city",S_CLEAN,"","Turin"},
        {"customer",S_CLEAN,"","Summit Metals"},{"quantity",S_CLEAN,"","300"}}},
      {"t4", "Email 1 (Mon): From Mara at Nordic. Customer Nordic Tech, deliver to "
       "Oslo, quantity 60, unit price 9.99.\n"
       "Email 2 (Tue): From Mara. Please deliver to Gothenburg instead of Oslo.\n"
       "Email 3 (Wed): From the shipping desk. Order confirmed for delivery to Oslo "
       "as originally requested; truck booked.\n", INS,   // distractor re-quotes OLD Oslo
       {{"delivery_city",S_CORR,"Oslo","Gothenburg"},{"customer",S_CLEAN,"","Nordic Tech"},
        {"quantity",S_CLEAN,"","60"},{"unit_price",S_CLEAN,"","9.99"}}},
      {"t5", "Email 1 (Mon): From Lee at Ironclad. Customer Ironclad Works, deliver to "
       "Bilbao, quantity 200, unit price 43.30.\n"
       "Email 2 (Tue): From Lee. Two changes: deliver to Valencia, and change the "
       "quantity to 260.\n", INS,
       {{"delivery_city",S_CORR,"Bilbao","Valencia"},{"quantity",S_CORR,"200","260"},
        {"customer",S_CLEAN,"","Ironclad Works"},{"unit_price",S_CLEAN,"","43.30"}}},
    };
    return {
      {"h1", "Email 1 (Mon): From procurement at Crestwood. Customer Crestwood Metals, "
       "deliver to Genoa, quantity 80, unit price 6.40.\n"
       "Email 2 (Tue): Correction: deliver to Naples, not Genoa.\n", INS,
       {{"delivery_city",S_CORR,"Genoa","Naples"},{"customer",S_CLEAN,"","Crestwood Metals"},
        {"quantity",S_CLEAN,"","80"},{"unit_price",S_CLEAN,"","6.40"}}},
      {"h2", "Email 1 (Mon): From orders at Meadowfield. Customer Meadowfield Foods, "
       "deliver to Porto, quantity 500, unit price 2.85.\n"
       "Email 2 (Tue): Could you make the quantity 650 on this one.\n"
       "Email 3 (Tue): From dispatch. Noted; the 500 units currently on the pick list "
       "will move to the Porto lane first thing.\n", INS,   // distractor re-quotes OLD 500
       {{"quantity",S_CORR,"500","650"},{"delivery_city",S_CLEAN,"","Porto"},
        {"customer",S_CLEAN,"","Meadowfield Foods"},{"unit_price",S_CLEAN,"","2.85"}}},
      {"h3", "Email 1 (Mon): From Harbor Line. Customer Harbor Line Co, deliver to Cork, "
       "quantity 44, unit price 128.50.\n"
       "Email 2 (Wed): The unit price is now 149.00 following the new quote.\n", INS,
       {{"unit_price",S_CORR,"128.50","149.00"},{"delivery_city",S_CLEAN,"","Cork"},
        {"customer",S_CLEAN,"","Harbor Line Co"},{"quantity",S_CLEAN,"","44"}}},
      {"h4", "Email 1 (Mon): From Zenith. Customer Zenith Works, deliver to Malmo, "
       "quantity 15, unit price 415.00.\n" + ss_pad(6) +
       "\nEmail (final, from Zenith): One change on order 300: deliver to Aarhus "
       "instead of Malmo. Thanks.\n", INS,
       {{"delivery_city",S_CORR,"Malmo","Aarhus"},{"customer",S_CLEAN,"","Zenith Works"},
        {"quantity",S_CLEAN,"","15"},{"unit_price",S_CLEAN,"","415.00"}}},
      {"h5", "Email 1 (Mon): From Delta. Customer Delta Foods, deliver to Nantes, "
       "quantity 30, unit price 3.15.\n"
       "Email 2 (Tue): Change the quantity to 36 and the unit price to 3.60.\n" + ss_pad(6) +
       "\nEmail (dispatch): The original 30 units are boxed and waiting on the Nantes "
       "dock for final sign-off.\n", INS,   // long + distractor re-quotes OLD 30
       {{"quantity",S_CORR,"30","36"},{"unit_price",S_CORR,"3.15","3.60"},
        {"delivery_city",S_CLEAN,"","Nantes"},{"customer",S_CLEAN,"","Delta Foods"}}},
    };
}

// ═════════════════════════════════════════════════════════════════════════
// LENS — data generator for the "Attention Lens" HTML demo. NOT a probe: dumps
// ONE real run's citation (L3H13) + coverage (layer-11 max-heads) data as JSON
// for docs/demo/attention-lens.html. Frozen sources/thresholds travel as-is.
// Gated by LENS=1; all prior paths untouched; zero engine edits.
// ═════════════════════════════════════════════════════════════════════════
static std::string jesc(const std::string& s) {
    std::string o; o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;   case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;    case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                     else o += (char)c;
        }
    }
    return o;
}

static int run_lens_gen(ForwardPassBase* fp, ggml_backend_sched_t sched,
                        Tokenizer* tok, const ModelMetadata& meta,
                        const std::vector<int32_t>& attn_layers) {
    const std::vector<int> tap = {attn_layers[FROZEN_SLOT], attn_layers[2]};  // {L3, layer11}
    const std::vector<std::string> keys = {"customer", "product", "quantity", "unit_price", "delivery"};
    std::string body =
        "From: mueller@nordwind-gmbh.de\n"
        "Subject: Order\n"
        "Hello,\n"
        "we'd like to order 40 boxes of A4 copy paper\n"
        "at 3.20 EUR per box.\n"
        "Delivery to our Hamburg warehouse.\n"
        "PS: Please make it 45 boxes, not 40.\n"
        "PPS: The invoice must mention project code NW-771.\n";
    std::string instr =
        "\nExtract the order as a JSON object with keys customer, product, quantity, "
        "unit_price, delivery.\nJSON:\n{";

    FreeRun R = run_freegen(fp, sched, tok, meta, body, instr, tap, 200, false, '}');
    const int P = R.P, G = (int)R.gen_tokens.size();

    // Self-check: rows sum to 1.0 (L3, head 0, step 0).
    int n_kv0 = R.n_kv_at_step.empty() ? 0 : R.n_kv_at_step[0];
    double s0 = 0; for (int q = 0; q < n_kv0; ++q) s0 += R.rows[0][0][q];
    std::printf("[LENS self-check] rows sum=%.5f  P=%d gen=%d\n", s0, P, G);
    std::printf("[LENS] gen: %s\n", R.gen_text.c_str());

    // Self-check: quantity value's L3H13 citation lands in a PS/line-2 span.
    auto gfs = parse_fields(tok, R, keys);
    for (auto& g : gfs) if (g.name == "quantity" && g.a >= 1) {
        int step = g.a - 1, n_kv = R.n_kv_at_step[step];
        auto tkq = topk_head(R.rows[step][0], FROZEN_HEAD, n_kv, 3);
        std::printf("[LENS self-check] quantity=\"%s\" L3H13 top-3 prompt pos:", g.value.c_str());
        for (auto& pr : tkq) if (pr.first < P)
            std::printf(" %d(%.2f)=\"%s\"", pr.first, pr.second, tok->decode(R.prompt_tokens[pr.first]).c_str());
        std::printf("\n");
    }

    // layer-11 max-over-heads mass per (step, prompt position). heat[p] (per-
    // token color) = max over steps. The not-incorporated LIST instead uses
    // COV1's per-SPAN semantics (max over steps of the SUM over the span), at
    // line granularity — so a line whose value was copied at some step reads
    // USED even if no single token peaks ≥ 0.705.
    std::vector<std::vector<double>> mh(G, std::vector<double>(P, 0.0));
    std::vector<double> heat(P, 0.0);
    for (int t = 0; t < G; ++t) {
        int n_kv = R.n_kv_at_step[t];
        for (int p = 0; p < P; ++p) {
            double m = 0;
            for (int h = 0; h < R.n_head; ++h) { double v = R.rows[t][1][(size_t)h * n_kv + p]; if (v > m) m = v; }
            mh[t][p] = m; if (m > heat[p]) heat[p] = m;
        }
    }
    // Body lines (split on tokens containing '\n'); each line's COV1 span-peak.
    struct Line { int lo, hi; double span_peak; std::string text; };
    std::vector<Line> lines; int lstart = 0;
    for (int p = 0; p < R.instr_tok; ++p) {
        std::string tt = tok->decode(R.prompt_tokens[p]);
        bool nl = tt.find('\n') != std::string::npos;
        if (nl || p == R.instr_tok - 1) {
            int lo = lstart, hi = p; double sp = 0; std::string tx;
            for (int t = 0; t < G; ++t) { double s = 0; for (int q = lo; q <= hi; ++q) s += mh[t][q]; if (s > sp) sp = s; }
            for (int q = lo; q <= hi; ++q) tx += tok->decode(R.prompt_tokens[q]);
            lines.push_back({lo, hi, sp, tx}); lstart = p + 1;
        }
    }

    FILE* f = std::fopen("docs/demo/lens-data.json", "w");
    if (!f) { std::printf("[LENS] ERROR: cannot open docs/demo/lens-data.json (mkdir first)\n"); return 1; }
    std::fprintf(f, "{\n");
    std::fprintf(f, "\"model\":\"Qwen3.6-35B-A3B-MTP UD-Q2_K_XL (Metal, greedy)\",\n");
    std::fprintf(f, "\"citation_source\":\"layer 3, head 13 (L3H13) \\u2014 N3, 97%% top-1\",\n");
    std::fprintf(f, "\"coverage_source\":\"layer 11, max over heads \\u2014 COV1\",\n");
    std::fprintf(f, "\"used_threshold\":0.705,\n");
    std::fprintf(f, "\"bands\":{\"filler\":0.16,\"dropped_lo\":0.2,\"dropped_hi\":0.58,\"used\":0.77},\n");
    std::fprintf(f, "\"prompt_len\":%d,\"instr_tok\":%d,\n", P, R.instr_tok);

    std::fprintf(f, "\"prompt\":[");
    for (int p = 0; p < P; ++p)
        std::fprintf(f, "%s{\"pos\":%d,\"text\":\"%s\",\"region\":\"%s\"}", p ? "," : "", p,
                     jesc(tok->decode(R.prompt_tokens[p])).c_str(), p < R.instr_tok ? "body" : "instr");
    std::fprintf(f, "],\n");

    std::fprintf(f, "\"gen\":[");
    for (int t = 0; t < G; ++t)
        std::fprintf(f, "%s{\"idx\":%d,\"text\":\"%s\"}", t ? "," : "", t,
                     jesc(tok->decode(R.gen_tokens[t])).c_str());
    std::fprintf(f, "],\n");

    // hover[t] = frozen L3H13 top-8 PROMPT positions for gen[t] (row at t-1).
    std::fprintf(f, "\"hover\":[");
    for (int t = 0; t < G; ++t) {
        std::fprintf(f, "%s[", t ? "," : "");
        if (t >= 1) {
            int step = t - 1, n_kv = R.n_kv_at_step[step];
            const float* rr = R.rows[step][0].data() + (size_t)FROZEN_HEAD * n_kv;
            std::vector<std::pair<int, double>> v;
            for (int p = 1; p < P; ++p) v.push_back({p, rr[p]});   // prompt positions, excl idx 0
            std::partial_sort(v.begin(), v.begin() + std::min((size_t)8, v.size()), v.end(),
                              [](auto& a, auto& b) { return a.second > b.second; });
            int n = std::min((size_t)8, v.size());
            for (int i = 0; i < n; ++i)
                std::fprintf(f, "%s{\"pos\":%d,\"mass\":%.4f}", i ? "," : "", v[i].first, v[i].second);
        }
        std::fprintf(f, "]");
    }
    std::fprintf(f, "],\n");

    std::fprintf(f, "\"heat\":[");
    for (int p = 0; p < P; ++p) std::fprintf(f, "%s%.4f", p ? "," : "", heat[p]);
    std::fprintf(f, "],\n");

    // skipped: whole BODY lines whose COV1 span-peak < 0.705 (with real content).
    std::fprintf(f, "\"skipped\":[");
    bool first = true;
    for (auto& L : lines) {
        if (L.span_peak >= 0.705) continue;
        bool has = false; for (char c : L.text) if (std::isalnum((unsigned char)c)) has = true;
        if (!has) continue;
        std::fprintf(f, "%s{\"lo\":%d,\"hi\":%d,\"peak\":%.4f,\"text\":\"%s\"}",
                     first ? "" : ",", L.lo, L.hi, L.span_peak, jesc(L.text).c_str());
        first = false;
    }
    std::fprintf(f, "]\n}\n");
    std::fclose(f);
    std::printf("[LENS] wrote docs/demo/lens-data.json\n");
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// QEMMI-DOCS P0 — document → relaxed key-value JSON on the attention trust
// layer. Five legs A–E. Gated QDOCS_A / QDOCS_B / QDOCS_C / QDOCS_D. Legs
// share the ONE fixed KV grammar and a grammar-driven free-gen; all seven
// prior probe paths (N3/N3b/CG1/DP1/COV1/CF1/SS1/LENS) are untouched. Zero
// src/ edits — the grammar/trie/head are driven from here exactly as the
// production decode_step sequence (peek → sparse ids → build → accept).
// ═════════════════════════════════════════════════════════════════════════

// The ONE fixed universal KV grammar (brief §"fixed KV grammar"). Flat object,
// arbitrary snake_case keys, values = verbatim strings. NOTE: GBNF's `.`
// wildcard is unsupported by this engine's parse_term (it has no dot case), so
// the value rule's `"\\" .` escape-alt is written `"\\" [^]` — an empty NEGATED
// char class, which matches ANY char (grammar_vocab.cpp:316, allowed=!in_set
// with in_set=false). Behaviourally identical; purely a syntax substitution.
static const char* QDOCS_GBNF = R"GBNF(root  ::= "{" ws pair ("," ws pair)* ws "}"
pair  ::= "\"" key "\"" ws ":" ws "\"" value "\""
key   ::= [a-z] [a-z0-9_]*
value ::= ([^"\\\n] | "\\" [^])+
ws    ::= [ \n\t]*
)GBNF";

// Tap-slot (index into attn_layers) for the CITATION layer and the COVERAGE
// layer. Defaults are the Qwen 3.6 coordinates: slot 0 = physical layer 3,
// slot 2 = physical layer 11 (span_scalars sc index 1+2=3).
// These are a SECOND pair of slot constants alongside FROZEN_SLOT/ENS_SLOT, and
// they are the ones the QDOCS legs actually read — qdocs_eval_field takes its
// layer from L3H13_SLOT and only its head from FROZEN_HEAD. So overriding
// FROZEN_SLOT alone silently leaves leg C scoring layer 3, which is exactly the
// kind of "the flag did nothing and the label lied" trap that makes a probe
// report a confirmation it never ran. apply_frozen_head_overrides keeps them in
// sync; ATTN_COV_SLOT moves the coverage layer independently.
static int L3H13_SLOT = 0;
static int L11_SLOT   = 2;

static void apply_frozen_head_overrides() {
    if (const char* s = std::getenv("ATTN_FROZEN_SLOT")) FROZEN_SLOT = std::atoi(s);
    if (const char* h = std::getenv("ATTN_FROZEN_HEAD")) FROZEN_HEAD = std::atoi(h);
    if (const char* s = std::getenv("ATTN_ENS_SLOT"))    ENS_SLOT    = std::atoi(s);
    if (const char* h = std::getenv("ATTN_ENS_HEAD"))    ENS_HEAD    = std::atoi(h);
    // The QDOCS legs read the citation LAYER from L3H13_SLOT, so it must follow
    // ATTN_FROZEN_SLOT or the override is a no-op there (see the note above).
    L3H13_SLOT = FROZEN_SLOT;
    if (const char* c = std::getenv("ATTN_COV_SLOT")) L11_SLOT = std::atoi(c);
    if (std::getenv("ATTN_FROZEN_SLOT") || std::getenv("ATTN_FROZEN_HEAD") ||
        std::getenv("ATTN_COV_SLOT")) {
        std::fprintf(stderr,
                     "[override] citation tap-slot=%d head=%d, coverage tap-slot=%d "
                     "(defaults 0/13/2 = Qwen 3.6 L3H13 + layer 11)\n",
                     FROZEN_SLOT, FROZEN_HEAD, L11_SLOT);
    }
}

// Render a document + extraction task through Qwen's ChatML template with
// thinking OFF (the model goes straight to the JSON answer). This is the
// production regime for Qemmi-Docs — the raw-text `{`-prime of N3/N3b/COV1 was
// a probe convenience that this thinking model needs replaced with the real
// instruct format. The returned string is the full prompt; the document is
// embedded verbatim so body.find(value) still locates source spans.
static const char* QDOCS_TASK =
    "\n\nExtract every fact from the email above into a flat JSON object of "
    "\"key\": \"value\" pairs. Use short snake_case keys. Copy each value "
    "verbatim from the email. Output ONLY the JSON object, nothing else.";
static std::string qdocs_chat_prompt(const std::string& document,
                                     const std::string& task = QDOCS_TASK) {
    // Pick the template by the loaded architecture. Hardcoding QwenChatTemplate
    // here made every QDOCS leg silently Qwen-shaped: on a gemma4 model it would
    // wrap the document in ChatML the model has never seen, and any resulting
    // "no citation head" reading would be measuring the wrapper, not the model.
    // g_arch is set in main() from the GGUF.
    std::vector<ChatMessage> hist = {{"user", document + task}};
    if (g_arch.rfind("gemma4", 0) == 0) {
        Gemma4ChatTemplate ct;
        return ct.render(hist, /*add_assistant_prompt=*/true, /*enable_thinking=*/false);
    }
    if (g_arch.rfind("gemma", 0) == 0) {
        GemmaChatTemplate ct;
        return ct.render(hist, /*add_assistant_prompt=*/true, /*enable_thinking=*/false);
    }
    QwenChatTemplate ct;
    return ct.render(hist, /*add_assistant_prompt=*/true, /*enable_thinking=*/false);
}

// Per-decode-step grammar bookkeeping for leg A (elision-hole analysis + sparse
// counting) and leg E fallback accounting.
struct GStep {
    int    n_valid = 0;      // |valid set| used to predict THIS step's next token
    bool   sparse  = false;  // sparse LM head fired (|valid| < vocab/8)
    bool   forced  = false;  // |valid|==1 ⇒ production would ELIDE (no attention row)
    std::string forced_tok;  // decoded string of the single legal token (if forced)
};

// Grammar-constrained free-greedy generation, tapping `tap_layers` — the leg A/
// B/C/D production shape. Mirrors run_freegen but every emitted token is drawn
// from the grammar's valid set (greedy within the mask), the sparse LM head is
// armed exactly as decode_step does, and grammar state advances via accept_token.
// The prompt is NOT pre-primed with '{' — the grammar emits the opener itself.
static FreeRun run_freegen_grammar(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                   Tokenizer* tok, const ModelMetadata& meta,
                                   const std::string& body_text, const std::string& instr_text,
                                   const std::vector<int>& tap_layers, int max_new,
                                   qinf::GrammarVocab* gr,
                                   const std::vector<std::string>& vocab,
                                   uint32_t vocab_size,
                                   std::vector<GStep>* trace) {
    FreeRun R;
    R.body_text = body_text;
    R.n_head = (int)meta.attention_head_count;
    // BOS lives in the TEXT so tokens and byte offsets stay in lockstep; the
    // body/instruction boundary below must therefore clear the BOS too.
    std::string prompt_text = prompt_with_bos(body_text + instr_text);
    const size_t body_end = g_bos_text.size() + body_text.size();
    R.prompt_tokens = encode_prompt(tok, prompt_text);
    if (tok->decode(R.prompt_tokens) != prompt_text)
        throw std::runtime_error("freegen_grammar: prompt roundtrip mismatch");
    R.P = (int)R.prompt_tokens.size();
    std::vector<size_t> cum = cum_bytes(tok, R.prompt_tokens);
    R.instr_tok = R.P;
    for (int k = 0; k < R.P; ++k) if (cum[k] >= body_end) { R.instr_tok = k; break; }

    gr->reset();
    fp->clear_slot(0); fp->set_cache_pos(0, 0);
    std::vector<float> logits = fp->run_prefill(R.prompt_tokens, 0, 0, sched);
    const int32_t eos = tok->get_eos_token_id();

    // Pick the FIRST gen token under the grammar (prefill head is dense).
    auto argmax_over = [&](const std::vector<float>& lg,
                           const std::vector<int32_t>& ids) -> int32_t {
        int32_t best = -1; float bl = -1e30f;
        for (int32_t id : ids) if ((size_t)id < lg.size() && lg[id] > bl) { bl = lg[id]; best = id; }
        return best;
    };
    std::vector<int32_t> valid = gr->get_valid_tokens(vocab);
    int32_t next = argmax_over(logits, valid);
    if (next < 0) { R.gen_text.clear(); return R; }
    gr->accept_token(next, vocab);

    for (int t = 0; t < max_new; ++t) {
        int32_t cur = next;
        R.gen_tokens.push_back(cur);
        std::string ct = tok->decode(cur);
        bool closed = gr->is_accepting_state();  // object completed after `cur`

        // Peek the valid set for the UPCOMING token (grammar state after `cur`).
        valid = gr->get_valid_tokens(vocab);
        GStep gs;
        gs.n_valid = (int)valid.size();
        gs.forced  = (valid.size() == 1);
        if (gs.forced) gs.forced_tok = tok->decode(valid[0]);
        const bool use_sparse = !valid.empty() && valid.size() < vocab_size / 8;
        gs.sparse = use_sparse;
        if (trace) trace->push_back(gs);

        fp->set_sparse_decode_ids(use_sparse ? valid : std::vector<int32_t>{});

        std::vector<int32_t>  tks = {cur};
        std::vector<uint32_t> slots = {0};
        std::vector<int32_t>  positions = {(int)fp->get_cache_pos(0)};
        ggml_cgraph* gf = fp->build_decoding_graph(tks, slots, positions);

        std::vector<ggml_tensor*> taps;
        for (int il : tap_layers) {
            std::string nm = "kq_soft." + std::to_string(il);
            ggml_tensor* ts = ggml_graph_get_tensor(gf, nm.c_str());
            if (!ts) throw std::runtime_error("freegen_grammar tap missing: " + nm);
            ggml_set_output(ts); ggml_build_forward_expand(gf, ts); taps.push_back(ts);
        }
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        fp->set_decode_inputs(gf, tks, slots, positions);
        ggml_backend_sched_graph_compute(sched, gf);

        std::vector<std::vector<float>> layer_rows; int n_kv = 0;
        for (ggml_tensor* ts : taps) {
            n_kv = (int)ts->ne[0]; int nh = (int)ts->ne[2];
            std::vector<float> buf((size_t)n_kv * nh);
            ggml_backend_tensor_get(ts, buf.data(), 0, ggml_nbytes(ts));
            layer_rows.push_back(std::move(buf));
        }
        R.rows.push_back(std::move(layer_rows));
        R.n_kv_at_step.push_back(n_kv);

        std::vector<float> lg = fp->get_output_logits(gf);
        fp->advance_cache(1, 0);

        if (valid.empty()) { next = eos; }
        else if (use_sparse) {
            // Sparse head: lg is aligned to the sparse ids (== valid order).
            int best_k = -1; float bl = -1e30f;
            for (int k = 0; k < (int)valid.size() && k < (int)lg.size(); ++k)
                if (lg[k] > bl) { bl = lg[k]; best_k = k; }
            next = best_k >= 0 ? valid[best_k] : eos;
        } else {
            next = argmax_over(lg, valid);
        }
        if (next >= 0 && next != eos) gr->accept_token(next, vocab);
        if (closed) break;
        if (next == eos || next < 0) break;
    }
    R.gen_text = tok->decode(R.gen_tokens);
    return R;
}

// Citation score (frozen L3H13) for a set of known value strings that appear
// verbatim in the body. For each value: locate its gen-token span and its body
// source span; each gen value token at gen-index g is scored top-1/top-3-in-span
// against rows[g-1] (query=gen[g-1] — N3's off-by-one). DE/EN split by `is_de`.
struct QCite { int n = 0, top1 = 0, top3 = 0; };
static QCite qdocs_cite(Tokenizer* tok, const FreeRun& R,
                        const std::vector<std::string>& values, int TOL) {
    QCite cs;
    std::vector<size_t> gcum = cum_bytes(tok, R.gen_tokens);
    std::vector<size_t> pcum = cum_bytes(tok, R.prompt_tokens);
    for (const std::string& v : values) {
        // source span in the PROMPT (body is a prefix of prompt)
        size_t sb = R.body_text.find(v);
        if (sb == std::string::npos) continue;
        size_t se = sb + v.size();
        int slo = -1, shi = -1;
        for (int k = 0; k < R.P; ++k)
            if (pcum[k] < se && pcum[k + 1] > sb) { if (slo < 0) slo = k; shi = k; }
        if (slo < 0) continue;
        // gen-token span of the emitted value (first occurrence in gen_text)
        size_t gb = R.gen_text.find(v);
        if (gb == std::string::npos) continue;   // model normalized it away
        size_t ge = gb + v.size();
        for (int g = 0; g < (int)R.gen_tokens.size(); ++g) {
            if (!(gcum[g] < ge && gcum[g + 1] > gb)) continue;  // token g inside value
            if (g < 1 || g - 1 >= (int)R.rows.size()) continue;
            auto tk = topk_head(R.rows[g - 1][L3H13_SLOT], FROZEN_HEAD,
                                R.n_kv_at_step[g - 1], 3);
            auto in = [&](int p) { return p >= slo - TOL && p <= shi + TOL; };
            cs.n++;
            if (!tk.empty() && in(tk[0].first)) cs.top1++;
            for (auto& pr : tk) if (in(pr.first)) { cs.top3++; break; }
        }
    }
    return cs;
}

// Layer-11 max-heads peak coverage (COV1 frozen) for one body span.
static double qdocs_cov_peak(const FreeRun& R, const std::string& span_text,
                             Tokenizer* tok, const std::vector<int32_t>& attn_layers) {
    std::vector<size_t> pcum = cum_bytes(tok, R.prompt_tokens);
    size_t sb = R.body_text.find(span_text);
    if (sb == std::string::npos) return -1;
    size_t se = sb + span_text.size();
    int lo = -1, hi = -1;
    for (int k = 0; k < R.P; ++k)
        if (pcum[k] < se && pcum[k + 1] > sb) { if (lo < 0) lo = k; hi = k; }
    if (lo < 0) return -1;
    double sc[12][4];
    span_scalars(R, lo, hi, sc, attn_layers);
    return sc[1 + L11_SLOT][0];   // layer-11 max-heads peak
}

// Extract the KEY strings (in order) from a flat-object JSON string: every
// "key" that immediately precedes a ':'. Tolerant of the well-formed output the
// grammar guarantees. Used by legs B/C/D.
static std::vector<std::string> qdocs_keys(const std::string& json) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < json.size()) {
        if (json[i] != '"') { i++; continue; }
        size_t k0 = i + 1, k1 = json.find('"', k0);
        if (k1 == std::string::npos) break;
        size_t j = k1 + 1; while (j < json.size() && (json[j] == ' ' || json[j] == '\t' || json[j] == '\n')) j++;
        if (j < json.size() && json[j] == ':') {
            out.push_back(json.substr(k0, k1 - k0));
            // skip the value string that follows so we don't read it as a key
            size_t v = j + 1; while (v < json.size() && (json[v] == ' ' || json[v] == '\t' || json[v] == '\n')) v++;
            if (v < json.size() && json[v] == '"') { size_t ve = json.find('"', v + 1); i = (ve == std::string::npos) ? json.size() : ve + 1; }
            else i = j + 1;
        } else { i = k1 + 1; }
    }
    return out;
}

// Map a model-chosen key to a canonical CORE concept, an AUX concept (a sensible
// non-order fact — header/currency/tax/contact — NOT junk), or "" (junk). This
// mapper IS the hand-labeling of the concept space (brief: "hand-label each
// email's ground-truth concepts"); it is disclosed in the note. EN + DE.
static std::string qdocs_concept_of(std::string k) {
    for (char& c : k) c = (char)std::tolower((unsigned char)c);
    auto has = [&](const char* s) { return k.find(s) != std::string::npos; };
    // Order: most specific first.
    if (has("deliver") || has("liefer") || has("versand") || has("ship")) return "delivery_date";
    if (has("order_date") || has("orderdate") || has("bestelldatum") ||
        has("auftragsdatum") || k == "datum" || (has("date") && !has("update"))) return "order_date";
    if (has("customer") || has("kunde") || has("client") || has("buyer") ||
        has("besteller") || has("auftraggeber") || has("firma") || has("company")) {
        if (has("number") || has("nummer") || has("_no") || has("_id") || has("nr")) return "aux"; // customer_number
        return "customer";
    }
    if ((has("order") || has("po") || has("bestell") || has("auftrag") || has("reference") || has("referenz")) &&
        (has("number") || has("no") || has("_id") || has("nummer") || has("nr") || has("po_") || k == "po")) return "order_number";
    if (has("quant") || k == "qty" || has("menge") || has("anzahl") || has("stückzahl") ||
        has("stueckzahl") || has("units")) return "quantity";
    if (has("unit_price") || has("unitprice") || has("stückpreis") || has("stueckpreis") ||
        has("einzelpreis") || has("price_per") || has("preis_pro") || (has("price") && has("unit"))) return "unit_price";
    if (has("total") || has("gesamt") || has("summe") || has("grand") || has("amount_due") ||
        has("betrag") || (has("sum") && !has("summary"))) return "total";
    if (has("price") || has("preis") || has("cost") || has("kosten")) return "unit_price"; // bare price → unit_price
    if (has("product") || has("produkt") || has("item") || has("artikel") || has("article") ||
        has("ware") || has("bezeichnung")) return "product";
    // AUX — sensible facts that are not the eight order concepts (not junk).
    if (has("from") || has("sender") || has("email") || has("mail") || has("absender")) return "aux";
    if (has("subject") || has("betreff")) return "aux";
    if (has("currency") || has("waehrung") || has("währung")) return "aux";
    if (has("vat") || has("tax") || has("mwst") || has("ust")) return "aux";
    if (has("contact") || has("phone") || has("tel") || has("address") || has("adresse") ||
        has("note") || has("payment") || has("zahlung") || has("term")) return "aux";
    return ""; // junk
}

// A leg-B/C/D document: the raw email, its DE flag, and the CORE concepts a
// human labels as present (ground truth for coverage).
struct QDoc { std::string tag; bool de; std::string document; std::vector<std::string> present; };

static const char* QDOCS_CORE[8] = {"customer", "product", "quantity", "unit_price",
                                    "total", "order_date", "delivery_date", "order_number"};

// 12 varied order emails, 6 EN + 6 DE, differing wording/layout. Flat single-item
// orders (the flat-object grammar; record-groups are deferred per P0 scope).
static std::vector<QDoc> qdocs_legb_corpus() {
    return {
      {"en1", false,
       "From: orders@acme-corp.example\nSubject: Purchase Order\n\nHello,\n\n"
       "Please process an order for customer Acme Corporation.\n"
       "Product: Stainless Steel Bolt M6.\nQuantity: 240 units.\n"
       "Unit price: 12.50 USD.\nOrder total: 3000.00 USD.\nOrder date: 2026-03-14.\n\nThanks.\n",
       {"customer","product","quantity","unit_price","total","order_date"}},
      {"en2", false,
       "From: purchasing@globex.example\nSubject: New Order\n\nHi team,\n\n"
       "Kindly arrange an order for Globex Industries.\nItem: Aluminium Bracket.\n"
       "We require 875 pieces at 47.30 EUR each.\nDelivery date: 2025-11-20.\n"
       "PO number: PO-88213.\n\nRegards.\n",
       {"customer","product","quantity","unit_price","delivery_date","order_number"}},
      {"en3", false,
       "From: buyer@harbor.example\nSubject: Reorder\n\nHello,\n\n"
       "Client Harbor Freight Ltd would like to reorder.\n"
       "Article: Weatherproof Camera Housing, qty 15, price per unit 120.00 GBP.\n"
       "Total due: 1800.00 GBP.\n\nBest,\nSam\n",
       {"customer","product","quantity","unit_price","total"}},
      {"en4", false,
       "From: procurement@nimbus.example\nSubject: Order 4471\n\nDear Sir,\n\n"
       "Nimbus Supply Co. orders 320 units of Copper Wire Spool.\n"
       "The unit cost is 18.75 USD.\nRequested delivery: 2025-12-01.\n"
       "Order reference: 4471.\n\nSincerely.\n",
       {"customer","product","quantity","unit_price","delivery_date","order_number"}},
      {"en5", false,
       "From: sales@initech.example\nSubject: Confirmation\n\nHi,\n\n"
       "Order for Initech LLC confirmed.\nProduct name: Hydraulic Coupler.\n"
       "Amount: 60 units.\nPrice: 8.90 EUR per unit.\nGrand total: 534.00 EUR.\n"
       "Ordered on 2024-06-09.\n\nThanks.\n",
       {"customer","product","quantity","unit_price","total","order_date"}},
      {"en6", false,
       "From: orders@meadow.example\nSubject: Bulk\n\nHello,\n\n"
       "Buyer: Meadow Foods.\nWe would like 1200 boxes of Compostable Plates.\n"
       "Unit price 0.18 USD.\nShip date 2025-08-30.\n\nRegards.\n",
       {"customer","product","quantity","unit_price","delivery_date"}},
      {"de1", true,
       "Von: bestellung@muster-gmbh.example\nBetreff: Bestellung\n\nSehr geehrte Damen und Herren,\n\n"
       "bitte bearbeiten Sie eine Bestellung für den Kunden Muster GmbH.\n"
       "Artikel: Edelstahlschraube M6.\nMenge: 240 Stück.\n"
       "Einzelpreis: 12,50 EUR.\nGesamtpreis: 3000,00 EUR.\nBestelldatum: 2026-03-14.\n\nMit freundlichen Grüßen.\n",
       {"customer","product","quantity","unit_price","total","order_date"}},
      {"de2", true,
       "Von: einkauf@nordwind.example\nBetreff: Neue Bestellung\n\nHallo,\n\n"
       "wir möchten für die Firma Nordwind AG bestellen.\nProdukt: Aluminiumhalter.\n"
       "Wir benötigen 875 Stück zu je 47,30 EUR.\nLiefertermin: 2025-11-20.\n"
       "Bestellnummer: BST-88213.\n\nViele Grüße.\n",
       {"customer","product","quantity","unit_price","delivery_date","order_number"}},
      {"de3", true,
       "Von: kunde@hafen.example\nBetreff: Nachbestellung\n\nGuten Tag,\n\n"
       "der Auftraggeber Hafen Handel GmbH möchte nachbestellen.\n"
       "Bezeichnung: Wetterfestes Kameragehäuse, Anzahl 15, Stückpreis 120,00 EUR.\n"
       "Gesamtbetrag: 1800,00 EUR.\n\nBeste Grüße,\nSabine\n",
       {"customer","product","quantity","unit_price","total"}},
      {"de4", true,
       "Von: beschaffung@wolke.example\nBetreff: Auftrag 4471\n\nSehr geehrter Herr,\n\n"
       "die Wolke Handels KG bestellt 320 Einheiten Kupferdrahtspule.\n"
       "Der Stückpreis beträgt 18,75 EUR.\nGewünschter Liefertermin: 2025-12-01.\n"
       "Auftragsnummer: 4471.\n\nMit freundlichen Grüßen.\n",
       {"customer","product","quantity","unit_price","delivery_date","order_number"}},
      {"de5", true,
       "Von: vertrieb@technik.example\nBetreff: Bestätigung\n\nHallo,\n\n"
       "Bestellung für Technik Süd GmbH bestätigt.\nProduktname: Hydraulikkupplung.\n"
       "Menge: 60 Stück.\nPreis: 8,90 EUR pro Stück.\nGesamtsumme: 534,00 EUR.\n"
       "Bestellt am 2024-06-09.\n\nDanke.\n",
       {"customer","product","quantity","unit_price","total","order_date"}},
      {"de6", true,
       "Von: bestellung@wiese.example\nBetreff: Großbestellung\n\nGuten Tag,\n\n"
       "Besteller: Wiese Lebensmittel.\nWir möchten 1200 Kartons Kompostierbare Teller.\n"
       "Einzelpreis 0,18 EUR.\nVersanddatum 2025-08-30.\n\nMit freundlichen Grüßen.\n",
       {"customer","product","quantity","unit_price","delivery_date"}},
    };
}

// Token span [lo,hi] in the prompt covering the first occurrence of `text`.
static bool qdocs_span_in_prompt(Tokenizer* tok, const FreeRun& R,
                                 const std::string& text, int& lo, int& hi) {
    std::vector<size_t> pcum = cum_bytes(tok, R.prompt_tokens);
    size_t b0 = R.body_text.find(text); if (b0 == std::string::npos) return false;
    size_t b1 = b0 + text.size(); lo = hi = -1;
    for (int k = 0; k < R.P; ++k) if (pcum[k] < b1 && pcum[k + 1] > b0) { if (lo < 0) lo = k; hi = k; }
    return lo >= 0;
}

// Per grounded field: citation (frozen L3H13, top-1/3-in-span), coverage
// (frozen layer-11 max-heads peak over the SOURCE span — valid even if the
// value was normalized), and body_mass (N3b ungrounded discriminator, L3H13
// mass on the DOCUMENT region). found_verbatim=false ⇒ the model normalized
// the value (no gen-verbatim match) — citation/body_mass unscoreable, counted.
struct QFieldEval {
    bool found_verbatim = false;
    int cite_n = 0, cite_t1 = 0, cite_t3 = 0;
    double cov_peak = -1, body_mass = -1;
};
static QFieldEval qdocs_eval_field(Tokenizer* tok, const FreeRun& R, const std::string& value,
                                   int doc_lo, int doc_hi,
                                   const std::vector<int32_t>& attn_layers, int TOL) {
    QFieldEval e;
    int slo, shi;
    if (!qdocs_span_in_prompt(tok, R, value, slo, shi)) return e;   // not in body → skip
    double sc[12][4]; span_scalars(R, slo, shi, sc, attn_layers);
    e.cov_peak = sc[1 + L11_SLOT][0];

    std::vector<size_t> gcum = cum_bytes(tok, R.gen_tokens);
    size_t gb = R.gen_text.find(value);
    if (gb == std::string::npos) return e;   // normalized away
    e.found_verbatim = true;
    size_t ge = gb + value.size();
    double bmsum = 0; int bmn = 0;
    for (int g = 0; g < (int)R.gen_tokens.size(); ++g) {
        if (!(gcum[g] < ge && gcum[g + 1] > gb)) continue;
        if (g < 1 || g - 1 >= (int)R.rows.size()) continue;
        int n_kv = R.n_kv_at_step[g - 1];
        auto tk = topk_head(R.rows[g - 1][L3H13_SLOT], FROZEN_HEAD, n_kv, 3);
        auto in = [&](int p) { return p >= slo - TOL && p <= shi + TOL; };
        e.cite_n++;
        if (!tk.empty() && in(tk[0].first)) e.cite_t1++;
        for (auto& pr : tk) if (in(pr.first)) { e.cite_t3++; break; }
        const float* rr = R.rows[g - 1][L3H13_SLOT].data() + (size_t)FROZEN_HEAD * n_kv;
        double bm = 0; for (int j = std::max(1, doc_lo); j <= doc_hi && j < n_kv; ++j) bm += rr[j];
        bmsum += bm; bmn++;
    }
    e.body_mass = bmn ? bmsum / bmn : -1;
    return e;
}

// A labeled grounded field: canonical concept + the value's verbatim body form.
struct QLabel { std::string concept, value; };
struct QMessy { std::string tag; bool de; std::string document; std::vector<QLabel> fields; };

// ── Leg B — key-name stability (the product-shape killer) ────────────────────
static int run_qdocs_leg_b(ForwardPassBase* fp, ggml_backend_sched_t sched,
                           Tokenizer* tok, const ModelMetadata& meta,
                           const std::vector<int32_t>& attn_layers) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ QEMMI-DOCS LEG B — key-name stability (2 arms, 12 docs EN+DE) ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const uint32_t vocab_size = (uint32_t)vocab.size();
    qinf::TokenTrie trie; trie.build(vocab);
    auto gr = qinf::GrammarVocab::parse_impl(QDOCS_GBNF);
    gr->set_token_trie(&trie);
    std::vector<int> tap1 = {attn_layers[0]};   // minimal tap; leg B needs keys only
    auto corpus = qdocs_legb_corpus();

    const char* BARE = "\n\nExtract every fact from the email above into a flat JSON "
        "object of \"key\": \"value\" pairs. Use short snake_case keys. Copy each value "
        "verbatim from the email. Output ONLY the JSON object, nothing else.";
    const char* HINT = "\n\nExtract every fact from the email above into a flat JSON "
        "object of \"key\": \"value\" pairs. Use short snake_case keys — prefer keys like: "
        "customer, quantity, unit_price, delivery, order_date, product. Copy each value "
        "verbatim from the email. Output ONLY the JSON object, nothing else.";
    // The brief's example hint omits `total` and `order_number`; this arm names
    // the full concept vocabulary to test whether the ≤1.2 bar is ACHIEVABLE.
    const char* HINT2 = "\n\nExtract every fact from the email above into a flat JSON "
        "object of \"key\": \"value\" pairs. Use short snake_case keys — prefer keys like: "
        "customer, product, quantity, unit_price, total, order_date, delivery, order_number. "
        "Copy each value verbatim from the email. Output ONLY the JSON object, nothing else.";

    struct ArmAgg {
        std::map<std::string, std::set<std::string>> keys_for;   // concept → distinct keys
        int present_pairs = 0, covered_pairs = 0;                // coverage numerator/denom
        int total_keys = 0, junk_keys = 0;
    };

    auto run_arm = [&](const char* task, const char* label) -> ArmAgg {
        ArmAgg A;
        std::printf("\n──────── ARM: %s ────────\n", label);
        for (const QDoc& d : corpus) {
            std::string prompt = qdocs_chat_prompt(d.document, task);
            std::vector<GStep> tr;
            FreeRun R = run_freegen_grammar(fp, sched, tok, meta, prompt, "", tap1,
                                            260, gr.get(), vocab, vocab_size, &tr);
            std::vector<std::string> keys = qdocs_keys(R.gen_text);
            std::printf("  [%s%s] %s\n", d.tag.c_str(), d.de ? " DE" : "", R.gen_text.c_str());

            std::set<std::string> concepts_hit;   // core concepts this doc produced
            for (const std::string& k : keys) {
                A.total_keys++;
                std::string c = qdocs_concept_of(k);
                if (c.empty()) { A.junk_keys++; std::printf("      junk key: \"%s\"\n", k.c_str()); }
                else if (c != "aux") { A.keys_for[c].insert(k); concepts_hit.insert(c); }
            }
            for (const std::string& c : d.present) {
                A.present_pairs++;
                if (concepts_hit.count(c)) A.covered_pairs++;
                else std::printf("      MISSED concept: %s\n", c.c_str());
            }
        }
        return A;
    };

    auto report = [&](const ArmAgg& A, const char* label, bool bar) {
        std::printf("\n──────── METRICS: %s ────────\n", label);
        double kpc_sum = 0; int kpc_n = 0;
        std::printf("  keys-per-concept (distinct key strings mapped to each core concept):\n");
        for (const char* c : QDOCS_CORE) {
            auto it = A.keys_for.find(c);
            if (it == A.keys_for.end()) continue;
            kpc_sum += it->second.size(); kpc_n++;
            std::printf("    %-14s %zu  {", c, it->second.size());
            bool f = true; for (const std::string& k : it->second) { std::printf("%s%s", f ? "" : ", ", k.c_str()); f = false; }
            std::printf("}\n");
        }
        double kpc = kpc_n ? kpc_sum / kpc_n : 0;
        double cov = A.present_pairs ? 100.0 * A.covered_pairs / A.present_pairs : 0;
        double junk = A.total_keys ? 100.0 * A.junk_keys / A.total_keys : 0;
        std::printf("  ── keys-per-concept = %.2f  (ideal 1.0%s)\n", kpc, bar ? ", bar ≤1.2" : "");
        std::printf("  ── concept coverage = %.0f%% (%d/%d)%s\n", cov, A.covered_pairs, A.present_pairs, bar ? "  bar ≥90%" : "");
        std::printf("  ── junk-key rate    = %.0f%% (%d/%d)%s\n", junk, A.junk_keys, A.total_keys, bar ? "  bar <10%" : "");
        if (bar) {
            bool pass = kpc <= 1.2 && cov >= 90 && junk < 10;
            std::printf("  ── HINT-ARM VERDICT: %s (kpc %s1.2, cov %s90%%, junk %s10%%)\n",
                        pass ? "PASS" : "FAIL",
                        kpc <= 1.2 ? "≤" : ">", cov >= 90 ? "≥" : "<", junk < 10 ? "<" : "≥");
        }
    };

    ArmAgg bare = run_arm(BARE, "bare instruction (no bar)");
    ArmAgg hint = run_arm(HINT, "soft key-vocab hint — brief's example (BARRED)");
    ArmAgg hin2 = run_arm(HINT2, "soft key-vocab hint — complete vocabulary (BARRED)");
    report(bare, "bare instruction", false);
    report(hint, "soft key-vocab hint (brief's example — omits total/order_number)", true);
    report(hin2, "soft key-vocab hint (complete vocabulary)", true);
    return 0;
}

// 15 realistic-messy synthetic order emails (constructed, not real): greetings,
// signatures, legal disclaimers, quoted reply fragments, typos, mixed DE/EN,
// European number/currency formats. Labeled grounded fields = (concept, verbatim
// body value). 8 EN + 7 DE.
static std::vector<QMessy> qdocs_messy_corpus() {
    return {
      {"m_en1", false,
       "From: p.hayes@brightwork.example\nSubject: RE: Restock request\n\n"
       "Hi Dana,\n\nThanks for the quick turnaround. Please go ahead and book the order for "
       "Brightwork Studios Ltd — 45 units of the Matte Black Easel Stand at 82.00 GBP each. "
       "Order date 2025-10-06. We'll need it by 2025-10-24.\n\nCheers,\nPete Hayes\n"
       "Procurement Lead | Brightwork Studios | +44 20 7946 0102\n\n"
       "-----------------------------------------\n"
       "This message and any attachments are confidential and intended solely for the addressee. "
       "If you have received it in error, please notify the sender.\n\n"
       "On Fri, 3 Oct 2025, Dana Reeve wrote:\n> Can you confirm the quantity and delivery window?\n",
       {{"customer","Brightwork Studios Ltd"},{"quantity","45"},{"unit_price","82.00"},
        {"order_date","2025-10-06"},{"delivery_date","2025-10-24"}}},
      {"m_en2", false,
       "From: orders@vertex-labs.example\nSubject: PO 5590-B\n\nHello,\n\n"
       "Please find our order below. Custmer: Vertex Labs Inc. We require 1,200 units of "
       "Nitrile Exam Gloves (M) at 0.14 USD/unit. Total 168.00 USD. PO number 5590-B.\n\n"
       "Regards,\nA. Okonkwo\n\n*** DISCLAIMER *** The information transmitted is intended only "
       "for the person to whom it is addressed and may contain confidential material.\n",
       {{"customer","Vertex Labs Inc."},{"quantity","1,200"},{"unit_price","0.14"},
        {"total","168.00"},{"order_number","5590-B"}}},
      {"m_en3", false,
       "From: buyer@stonegate.example\nSubject: order\n\nhi there\n\njust need a quick reorder — "
       "stonegate builders, 300 x galvanised coach bolt 10mm, 0.35 eur each pls. deliver to site by "
       "2025-09-30 if poss. order ref SG-7742. thanks!! sent from my phone\n",
       {{"customer","stonegate builders"},{"quantity","300"},{"unit_price","0.35"},
        {"delivery_date","2025-09-30"},{"order_number","SG-7742"}}},
      {"m_en4", false,
       "From: k.tan@meridianfoods.example\nSubject: Weekly order — please confirm\n\n"
       "Dear supplier,\n\nMeridian Foods Pte would like to place this week's order: 640 cartons of "
       "Cold-Pressed Coconut Oil (1L), unit price 3.90 SGD, grand total 2496.00 SGD. Ordered 2025-11-11.\n\n"
       "Best regards,\nKelvin Tan\nHead of Supply\nMeridian Foods Pte Ltd\n"
       "Reg. No. 200812345K | www.meridianfoods.example\n\n"
       "Please consider the environment before printing this email.\n",
       {{"customer","Meridian Foods Pte"},{"quantity","640"},{"unit_price","3.90"},
        {"total","2496.00"},{"order_date","2025-11-11"}}},
      {"m_en5", false,
       "From: procurement@arcticgear.example\nSubject: Fwd: order sheet\n\n"
       "Team, forwarding the numbers. Arctic Gear Co needs 88 of the Insulated Flask 750ml, "
       "price per unit 11.25 CAD. Requested delivery 2026-01-15. Thanks.\n\n"
       "> ---------- Forwarded message ----------\n"
       "> From: warehouse@arcticgear.example\n> the 750ml not the 500ml this time\n",
       {{"customer","Arctic Gear Co"},{"quantity","88"},{"unit_price","11.25"},
        {"delivery_date","2026-01-15"}}},
      {"m_en6", false,
       "From: sam@harbourpoint.example\nSubject: quote accepted\n\nHi,\n\n"
       "we accept your quote. Harbour Point Marine, qty 24, Stainless Cleat 6in, at 27.50 usd, "
       "total 660.00 usd. our po is HP-2231, date 2025-12-02.\n\nregards, sam\n"
       "This email has been scanned for viruses by our security appliance.\n",
       {{"customer","Harbour Point Marine"},{"quantity","24"},{"unit_price","27.50"},
        {"total","660.00"},{"order_number","HP-2231"},{"order_date","2025-12-02"}}},
      {"m_en7", false,
       "From: orders@lumenoptics.example\nSubject: New PO attached (details in body too)\n\n"
       "Good morning,\n\nKindly process for Lumen Optics GmbH: 150 pieces, Anti-Reflective Lens Blank, "
       "at 6.80 EUR/pc. Delivery required 2025-10-19. Order no. LO-0098.\n\n"
       "Mit freundlichen Grüßen / Kind regards,\nElena Vogt\n",
       {{"customer","Lumen Optics GmbH"},{"quantity","150"},{"unit_price","6.80"},
        {"delivery_date","2025-10-19"},{"order_number","LO-0098"}}},
      {"m_en8", false,
       "From: reorders@pinnacle.example\nSubject: same as last month\n\n"
       "hey — same as last month for Pinnacle Sports: 500 Rubber Grip Tape rolls @ 1.95 GBP. "
       "total's 975.00 GBP. need by 2025-11-28. cheers\n\n"
       "Sent from Outlook for iOS\n",
       {{"customer","Pinnacle Sports"},{"quantity","500"},{"unit_price","1.95"},
        {"total","975.00"},{"delivery_date","2025-11-28"}}},
      {"m_de1", true,
       "Von: einkauf@bergblick.example\nBetreff: AW: Bestellung KW42\n\n"
       "Hallo Frau Kern,\n\nvielen Dank. Bitte bestellen Sie für die Bergblick Handels GmbH "
       "80 Stück des Wanderstock Alu Pro zu je 24,90 EUR. Bestelldatum 2025-10-13. "
       "Liefertermin bis 2025-10-27.\n\nMit freundlichen Grüßen\nMarkus Wald\n"
       "Einkauf | Bergblick Handels GmbH | Tel. 089 123456\n\n"
       "________________________________\n"
       "Diese E-Mail enthält vertrauliche Informationen. Sollten Sie nicht der richtige "
       "Adressat sein, informieren Sie bitte den Absender.\n",
       {{"customer","Bergblick Handels GmbH"},{"quantity","80"},{"unit_price","24,90"},
        {"order_date","2025-10-13"},{"delivery_date","2025-10-27"}}},
      {"m_de2", true,
       "Von: bestellung@nordlicht.example\nBetreff: Bestellung Nr. 7781\n\nSehr geehrte Damen und Herren,\n\n"
       "die Nordlicht GmbH bestellt 2.500 Stück Einweghandschuhe Nitril (M) zum Einzelpreis von "
       "0,12 EUR. Gesamtbetrag 300,00 EUR. Bestellnummer 7781.\n\n"
       "Freundliche Grüße,\nA. Schmitt\n\n*** Vertraulichkeitshinweis *** Diese Nachricht ist "
       "ausschließlich für den vorgesehenen Empfänger bestimmt.\n",
       {{"customer","Nordlicht GmbH"},{"quantity","2.500"},{"unit_price","0,12"},
        {"total","300,00"},{"order_number","7781"}}},
      {"m_de3", true,
       "Von: kaufhaus@steinweg.example\nBetreff: nachbestellung\n\nhallo,\n\n"
       "kurze nachbestellung — steinweg baumarkt, 300 x verzinkte schlossschraube 10mm, je 0,35 eur. "
       "lieferung bitte bis 2025-09-30 an die filiale. bestellref SW-7742. danke!\n"
       "von meinem iphone gesendet\n",
       {{"customer","steinweg baumarkt"},{"quantity","300"},{"unit_price","0,35"},
        {"delivery_date","2025-09-30"},{"order_number","SW-7742"}}},
      {"m_de4", true,
       "Von: k.brand@suedfrucht.example\nBetreff: Wochenbestellung — bitte bestätigen\n\n"
       "Guten Tag,\n\ndie Südfrucht Handel KG möchte die Wochenbestellung aufgeben: 640 Kartons "
       "Kaltgepresstes Kokosöl (1L), Stückpreis 3,90 EUR, Gesamtsumme 2.496,00 EUR. Bestellt am 2025-11-11.\n\n"
       "Beste Grüße,\nKarin Brand\nLeitung Beschaffung\nSüdfrucht Handel KG\n"
       "USt-IdNr. DE123456789\n",
       {{"customer","Südfrucht Handel KG"},{"quantity","640"},{"unit_price","3,90"},
        {"total","2.496,00"},{"order_date","2025-11-11"}}},
      {"m_de5", true,
       "Von: beschaffung@polarausruestung.example\nBetreff: WG: bestellliste\n\n"
       "Team, ich leite die zahlen weiter. Polar Ausrüstung GmbH benötigt 88 Stück der "
       "Isolierflasche 750ml, stückpreis 11,25 EUR. Gewünschte lieferung 2026-01-15. Danke.\n\n"
       "> ---------- Weitergeleitete Nachricht ----------\n"
       "> Von: lager@polarausruestung.example\n> diesmal die 750ml, nicht die 500ml\n",
       {{"customer","Polar Ausrüstung GmbH"},{"quantity","88"},{"unit_price","11,25"},
        {"delivery_date","2026-01-15"}}},
      {"m_de6", true,
       "Von: sabine@hafenpunkt.example\nBetreff: angebot angenommen\n\nHallo,\n\n"
       "wir nehmen ihr angebot an. Hafenpunkt Marine, menge 24, Edelstahlklampe 15cm, zu 27,50 eur, "
       "gesamt 660,00 eur. unsere bestellnr ist HP-2231, datum 2025-12-02.\n\ngrüße, sabine\n"
       "Diese E-Mail wurde von unserer Sicherheitssoftware auf Viren geprüft.\n",
       {{"customer","Hafenpunkt Marine"},{"quantity","24"},{"unit_price","27,50"},
        {"total","660,00"},{"order_number","HP-2231"},{"order_date","2025-12-02"}}},
      {"m_de7", true,
       "Von: bestellungen@gipfel.example\nBetreff: wie letzten monat\n\n"
       "hey — wie letzten monat für Gipfel Sport: 500 Griffband-Rollen à 1,95 EUR. "
       "gesamt 975,00 EUR. benötigt bis 2025-11-28. danke\n\n"
       "Gesendet mit Outlook für iOS\n",
       {{"customer","Gipfel Sport"},{"quantity","500"},{"unit_price","1,95"},
        {"total","975,00"},{"delivery_date","2025-11-28"}}},
    };
}

// One generic filler block (unrelated older thread + catalog boilerplate). No
// numbers/names that collide with leg-D core values. Repeated to pad context.
static const char* QDOCS_FILLER_BLOCK =
    "\n\n----- earlier in this thread -----\n"
    "On an earlier day, a colleague wrote: Thanks for the update, noted for the file. "
    "Please keep the usual terms and the standard packaging; nothing else changes on our side. "
    "The logistics team confirmed the loading dock hours remain the same and the carrier is unchanged.\n"
    "> Reference catalogue excerpt (for context only, not part of any order):\n"
    "> - Utility Widget, general purpose, sold by the case, various finishes available on request\n"
    "> - Assorted Fasteners, mixed sizes, supplied in bulk sacks, grade as per house standard\n"
    "> - Packing Foam Sheets, protective, cut to common dimensions, neutral colour\n"
    "> - Shipping Labels, blank, thermal compatible, supplied on rolls for the printer\n"
    "> Please disregard the reference lines above; they are boilerplate appended by the mail system.\n"
    "Kind regards, and thank you for your continued cooperation over the past several seasons.\n";

// Grow filler (older-thread blocks) after `core` until the CHAT-RENDERED prompt
// reaches `target` tokens. Returns the document (core + filler); prints nothing.
static std::string qdocs_pad_to(Tokenizer* tok, const std::string& core,
                                const std::string& task, int target, int& actual_out) {
    std::string doc = core;
    for (int guard = 0; guard < 400; ++guard) {
        std::string prompt = qdocs_chat_prompt(doc, task);
        int n = (int)tok->encode(prompt).size();
        if (n >= target) { actual_out = n; return doc; }
        doc += QDOCS_FILLER_BLOCK;
    }
    actual_out = (int)tok->encode(qdocs_chat_prompt(doc, task)).size();
    return doc;
}

// Fast per-field eval for leg D: taps only 2 layers (slot 0 = layer 3 for
// L3H13 citation, slot 1 = layer 11 for COV1 coverage), so long-context runs
// don't pay the 10-layer readback + galloc re-reserve every step. Coverage peak
// = max over steps of (max over heads of span-mass on the layer-11 slot) —
// identical definition to span_scalars sc[layer][0].
static QFieldEval qdocs_eval_field_fast(Tokenizer* tok, const FreeRun& R,
                                        const std::string& value,
                                        int l3_slot, int l11_slot, int TOL) {
    QFieldEval e;
    int slo, shi;
    if (!qdocs_span_in_prompt(tok, R, value, slo, shi)) return e;
    // coverage: layer-11 max-heads peak across steps
    double peak = 0;
    for (int t = 0; t < (int)R.rows.size(); ++t) {
        int n_kv = R.n_kv_at_step[t];
        for (int h = 0; h < R.n_head; ++h) {
            const float* rr = R.rows[t][l11_slot].data() + (size_t)h * n_kv;
            double s = 0; for (int j = slo; j <= shi && j < n_kv; ++j) s += rr[j];
            if (s > peak) peak = s;
        }
    }
    e.cov_peak = peak;
    // citation: frozen L3H13
    std::vector<size_t> gcum = cum_bytes(tok, R.gen_tokens);
    size_t gb = R.gen_text.find(value);
    if (gb == std::string::npos) return e;
    e.found_verbatim = true;
    size_t ge = gb + value.size();
    for (int g = 0; g < (int)R.gen_tokens.size(); ++g) {
        if (!(gcum[g] < ge && gcum[g + 1] > gb)) continue;
        if (g < 1 || g - 1 >= (int)R.rows.size()) continue;
        auto tk = topk_head(R.rows[g - 1][l3_slot], FROZEN_HEAD, R.n_kv_at_step[g - 1], 3);
        auto in = [&](int p) { return p >= slo - TOL && p <= shi + TOL; };
        e.cite_n++;
        if (!tk.empty() && in(tk[0].first)) e.cite_t1++;
        for (auto& pr : tk) if (in(pr.first)) { e.cite_t3++; break; }
    }
    return e;
}

// ── Leg D — context length (the envelope leg) ────────────────────────────────
static int run_qdocs_leg_d(ForwardPassBase* fp, ggml_backend_sched_t sched,
                           Tokenizer* tok, const ModelMetadata& meta,
                           const std::vector<int32_t>& attn_layers) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ QEMMI-DOCS LEG D — context length (1K / 2K / 4K buckets)      ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    const int TOL = 2;
    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const uint32_t vocab_size = (uint32_t)vocab.size();
    qinf::TokenTrie trie; trie.build(vocab);
    auto gr = qinf::GrammarVocab::parse_impl(QDOCS_GBNF);
    gr->set_token_trie(&trie);
    // Tap only the 2 layers the signals need: layer 3 (L3H13 citation, slot 0)
    // and layer 11 (COV1 coverage, slot 1). Keeps long-context runs tractable.
    std::vector<int> taps2 = {attn_layers[0], attn_layers[2]};   // {3, 11}
    const int L3_S = 0, L11_S = 1;
    const char* TASK = "\n\nExtract the order in this email thread into a flat JSON "
        "object of \"key\": \"value\" pairs. Use short snake_case keys — prefer keys like: "
        "customer, product, quantity, unit_price, total, order_date, delivery, order_number. "
        "Ignore boilerplate and quoted older messages. Copy each value verbatim. "
        "Output ONLY the JSON object, nothing else.";

    // 3 core emails with globally-distinctive labeled values (won't collide with filler).
    std::vector<QMessy> cores = {
      {"core_a", false,
       "From: orders@zephyrdyn.example\nSubject: Firm order\n\nHello,\n\n"
       "Please book a firm order for Zephyr Dynamics Ltd: 417 units of the Titanium Hinge Assembly "
       "at 63.75 CHF each. Order date 2025-07-08. Our order number is ZD-9931.\n\nThank you.\n",
       {{"customer","Zephyr Dynamics Ltd"},{"quantity","417"},{"unit_price","63.75"},
        {"order_date","2025-07-08"},{"order_number","ZD-9931"}}},
      {"core_b", false,
       "From: buyer@halcyontex.example\nSubject: Order confirmation\n\nHi,\n\n"
       "Halcyon Textiles confirms an order for 930 metres of the Herringbone Wool Bolt at 4.65 AUD, "
       "for a total of 4324.50 AUD. Requested delivery 2026-02-19.\n\nRegards.\n",
       {{"customer","Halcyon Textiles"},{"quantity","930"},{"unit_price","4.65"},
        {"total","4324.50"},{"delivery_date","2026-02-19"}}},
      {"core_c", false,
       "From: procurement@ironwoodco.example\nSubject: PO IW-3300\n\nDear supplier,\n\n"
       "Ironwood Components orders 58 pieces of the Forged Bracket Type-K at 112.40 NZD each. "
       "Order date 2025-05-22. Purchase order IW-3300.\n\nSincerely.\n",
       {{"customer","Ironwood Components"},{"quantity","58"},{"unit_price","112.40"},
        {"order_date","2025-05-22"},{"order_number","IW-3300"}}},
    };
    const int targets[3] = {1024, 2048, 4096};

    struct Bkt { int cite_n=0, t1=0, t3=0, used=0, clear=0; std::vector<double> peaks; std::vector<int> toks; };
    Bkt bkt[3];

    for (int bi = 0; bi < 3; ++bi) {
        std::printf("\n════════ BUCKET ~%d tokens ════════\n", targets[bi]);
        for (const QMessy& core : cores) {
            int actual = 0;
            std::string doc = qdocs_pad_to(tok, core.document, TASK, targets[bi], actual);
            std::string prompt = qdocs_chat_prompt(doc, TASK);
            std::vector<GStep> tr;
            FreeRun R = run_freegen_grammar(fp, sched, tok, meta, prompt, "", taps2,
                                            160, gr.get(), vocab, vocab_size, &tr);
            bkt[bi].toks.push_back(actual);
            std::printf("\n  [%s] actual_prompt_tokens=%d\n    gen: %s\n",
                        core.tag.c_str(), actual, R.gen_text.c_str());
            for (const QLabel& f : core.fields) {
                QFieldEval e = qdocs_eval_field_fast(tok, R, f.value, L3_S, L11_S, TOL);
                if (e.cov_peak >= 0) { bkt[bi].used++; bkt[bi].peaks.push_back(e.cov_peak);
                                       if (e.cov_peak >= 0.705) bkt[bi].clear++; }
                if (e.found_verbatim) { bkt[bi].cite_n += e.cite_n; bkt[bi].t1 += e.cite_t1; bkt[bi].t3 += e.cite_t3; }
                std::printf("    %-13s \"%s\"  cite t3 %d/%d  cov=%.3f%s\n",
                            f.concept.c_str(), f.value.c_str(), e.cite_t3, e.cite_n, e.cov_peak,
                            e.found_verbatim ? "" : "  [normalized]");
            }
        }
    }

    auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size()/2]; };
    std::printf("\n──────── LEG D SUMMARY ────────\n");
    std::printf("  bucket | tok range        | citation top3 | coverage used-clear | med peak\n");
    for (int bi = 0; bi < 3; ++bi) {
        int lo = 1e9, hi = 0; for (int t : bkt[bi].toks) { lo = std::min(lo, t); hi = std::max(hi, t); }
        std::printf("  ~%-4d | %4d..%-4d (n=%zu) | %3d/%-3d (%3.0f%%) | %2d/%-2d (%3.0f%%)      | %.3f\n",
                    targets[bi], lo, hi, bkt[bi].toks.size(),
                    bkt[bi].t3, bkt[bi].cite_n, bkt[bi].cite_n ? 100.0 * bkt[bi].t3 / bkt[bi].cite_n : 0,
                    bkt[bi].clear, bkt[bi].used, bkt[bi].used ? 100.0 * bkt[bi].clear / bkt[bi].used : 0,
                    med(bkt[bi].peaks));
    }
    double t3_4k = bkt[2].cite_n ? 100.0 * bkt[2].t3 / bkt[2].cite_n : 0;
    std::printf("  => 4K top3 = %.0f%% (bar ≥85%% ⇒ full-envelope claim)\n", t3_4k);
    return 0;
}

// ── Leg C — messy-corpus robustness (the real-world leg) ─────────────────────
static int run_qdocs_leg_c(ForwardPassBase* fp, ggml_backend_sched_t sched,
                           Tokenizer* tok, const ModelMetadata& meta,
                           const std::vector<int32_t>& attn_layers) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ QEMMI-DOCS LEG C — messy-corpus robustness (15 docs EN+DE)    ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    const int TOL = 2;
    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const uint32_t vocab_size = (uint32_t)vocab.size();
    qinf::TokenTrie trie; trie.build(vocab);
    auto gr = qinf::GrammarVocab::parse_impl(QDOCS_GBNF);
    gr->set_token_trie(&trie);
    std::vector<int> taps10(attn_layers.begin(), attn_layers.end());
    // Complete key-vocabulary hint (leg B's winning arm).
    const char* TASK = "\n\nExtract every fact from the email above into a flat JSON "
        "object of \"key\": \"value\" pairs. Use short snake_case keys — prefer keys like: "
        "customer, product, quantity, unit_price, total, order_date, delivery, order_number. "
        "Copy each value verbatim from the email. Output ONLY the JSON object, nothing else.";

    auto corpus = qdocs_messy_corpus();

    // Accumulators, split EN / DE.
    struct Acc { int cite_n=0, t1=0, t3=0; int used_spans=0, used_clear=0;
                 std::vector<double> used_peaks; int grounded=0, false_alarm=0;
                 int labeled=0, normalized=0; };
    Acc en, de;
    auto pick = [&](bool d) -> Acc& { return d ? de : en; };

    for (const QMessy& d : corpus) {
        std::string prompt = qdocs_chat_prompt(d.document, TASK);
        std::vector<GStep> tr;
        FreeRun R = run_freegen_grammar(fp, sched, tok, meta, prompt, "", taps10,
                                        320, gr.get(), vocab, vocab_size, &tr);
        std::printf("\n[%s%s] %s\n", d.tag.c_str(), d.de ? " DE" : "", R.gen_text.c_str());

        int doc_lo, doc_hi;
        if (!qdocs_span_in_prompt(tok, R, d.document, doc_lo, doc_hi)) {
            // document is huge; find its first line at least
            doc_lo = 0; doc_hi = R.P - 1;
        }
        Acc& A = pick(d.de);
        for (const QLabel& f : d.fields) {
            A.labeled++;
            QFieldEval e = qdocs_eval_field(tok, R, f.value, doc_lo, doc_hi, attn_layers, TOL);
            // coverage on the source span (independent of verbatim emission)
            if (e.cov_peak >= 0) {
                A.used_spans++; A.used_peaks.push_back(e.cov_peak);
                if (e.cov_peak >= 0.705) A.used_clear++;
            }
            if (!e.found_verbatim) { A.normalized++;
                std::printf("    %-13s \"%s\"  NORMALIZED (not verbatim in output)  cov=%.3f\n",
                            f.concept.c_str(), f.value.c_str(), e.cov_peak);
                continue;
            }
            A.cite_n += e.cite_n; A.t1 += e.cite_t1; A.t3 += e.cite_t3;
            A.grounded++;
            bool alarm = e.body_mass < 0.538;   // predicted invented despite being grounded
            if (alarm) A.false_alarm++;
            std::printf("    %-13s \"%s\"  cite t1 %d/%d t3 %d/%d  cov=%.3f  body_mass=%.3f%s\n",
                        f.concept.c_str(), f.value.c_str(), e.cite_t1, e.cite_n, e.cite_t3, e.cite_n,
                        e.cov_peak, e.body_mass, alarm ? "  << FALSE ALARM" : "");
        }
    }

    auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end());
        return v[v.size() / 2]; };
    auto report = [&](const char* label, const Acc& A) {
        std::printf("\n──────── %s ────────\n", label);
        // Name the head that was ACTUALLY scored, not the qwen36 default: with
        // ATTN_FROZEN_SLOT/HEAD set, a hardcoded "L3H13" label makes the log lie.
        std::printf("  citation (frozen L%dH%d) on value tokens: top1 %d/%d (%.0f%%)  top3 %d/%d (%.0f%%)  [bar top3 ≥90%%]\n",
                    (*g_attn_layers)[FROZEN_SLOT], FROZEN_HEAD,
                    A.t1, A.cite_n, A.cite_n ? 100.0 * A.t1 / A.cite_n : 0,
                    A.t3, A.cite_n, A.cite_n ? 100.0 * A.t3 / A.cite_n : 0);
        std::printf("  coverage: used spans clearing 0.705 = %d/%d (%.0f%%)  median used-peak %.3f  [COV1 USED median 0.937]\n",
                    A.used_clear, A.used_spans, A.used_spans ? 100.0 * A.used_clear / A.used_spans : 0,
                    med(A.used_peaks));
        std::printf("  ungrounded false-alarm on grounded fields: %d/%d (%.0f%%)  [bar <10%%]\n",
                    A.false_alarm, A.grounded, A.grounded ? 100.0 * A.false_alarm / A.grounded : 0);
        std::printf("  verbatim-ness: normalized %d/%d (%.0f%% of labeled values re-rendered)\n",
                    A.normalized, A.labeled, A.labeled ? 100.0 * A.normalized / A.labeled : 0);
    };
    report("EN", en);
    report("DE", de);
    Acc all; // combined
    all.cite_n=en.cite_n+de.cite_n; all.t1=en.t1+de.t1; all.t3=en.t3+de.t3;
    all.used_spans=en.used_spans+de.used_spans; all.used_clear=en.used_clear+de.used_clear;
    all.grounded=en.grounded+de.grounded; all.false_alarm=en.false_alarm+de.false_alarm;
    all.labeled=en.labeled+de.labeled; all.normalized=en.normalized+de.normalized;
    for (double p : en.used_peaks) all.used_peaks.push_back(p);
    for (double p : de.used_peaks) all.used_peaks.push_back(p);
    report("COMBINED", all);

    double t3 = all.cite_n ? 100.0 * all.t3 / all.cite_n : 0;
    double fa = all.grounded ? 100.0 * all.false_alarm / all.grounded : 0;
    bool pass = t3 >= 90 && fa < 10 && (all.used_spans ? (double)all.used_clear / all.used_spans >= 0.9 : false);
    std::printf("\n── LEG C VERDICT: %s (top3 %.0f%%%s90, false-alarm %.0f%%%s10, used-clear %.0f%%) ──\n",
                pass ? "PASS" : "FAIL", t3, t3 >= 90 ? "≥" : "<", fa, fa < 10 ? "<" : "≥",
                all.used_spans ? 100.0 * all.used_clear / all.used_spans : 0);
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// SS2 — the coverage-free stale-source alarm, on a grammar-constrained thread
// extractor, at genuinely long (4K-8K token) context. Supersedes SS1
// (docs/note-stale-source-probe.md), which was blocked by three walls: no
// natural STALE at short context, an attended-but-ignored STALE mode that
// COV1 coverage structurally cannot see, and long context never reached
// (max 764 tokens). SS1's own §5 named this follow-up: "the coverage-free
// alarm (citation-of-emitted-value + structural turn-order), on a
// grammar-constrained thread extractor, at genuinely long context."
//
// Design (three deliberate changes from SS1):
//  (a) COVERAGE-FREE. Alarm = citation + structural turn order only. For each
//      emitted field value, the frozen citation head's top-1 gives the prompt
//      span it came from; that span is mapped to WHICH EMAIL in the thread it
//      sits in (byte-range containment, computed at thread-construction time
//      — see the trap note below); ALARM fires iff that email is superseded
//      by a later email that corrects the field. Never asks "was the
//      correction consulted" (coverage's question) — asks "which email did
//      the emitted value cite."
//  (b) GRAMMAR-CONSTRAINED extraction (QDOCS_GBNF, the same fixed KV grammar
//      as Leg C) — SS1's thread-scale citation was confounded by garbage
//      keys and echoed structure; the grammar pins the key set so field
//      parsing cannot break.
//  (c) STRONGER HEAD + MODEL: Qwen3.8-9B, L27H13 (98% top-3 / 89% top-1 on
//      the messy corpus per docs/note-lens-qwen38-probe.md), vs SS1's
//      Qwen3.6 Q2_K_XL L3H13 (84% top-3). Set via ATTN_FROZEN_SLOT=6
//      ATTN_FROZEN_HEAD=13 (defaults stay Qwen 3.6-shaped; unset ⇒ untouched).
//
// TRAP (the one this leg is warned is most dangerous): real threads quote
// their own history, so the SAME field value can appear more than once in
// the final rendered document (original statement + every later quote of
// it). A naive `text.find(value)` search — which qdocs_span_in_prompt/
// qdocs_cite/qdocs_eval_field all use for SHORT single documents — would
// silently grab the WRONG occurrence here and make the alarm look correct
// when it isn't. SS2 sidesteps this by construction instead of by
// disambiguation: s2_build() tracks the EXACT byte span of every field
// mention and every message's own envelope as the nested-quote document is
// assembled bottom-up (oldest message innermost, exactly how a real mail
// client renders a reply chain), remapping spans through each quoting/
// prefixing step. No text search over the assembled document ever happens
// for a short value string; the only `.find()` calls are (1) each message's
// OWN new content against itself (unique by construction, one call site,
// authored to be single-occurrence — verified by hand for this corpus), and
// (2) locating the whole multi-KB document blob within the ChatML-wrapped
// prompt once (safe: long unique blob, not a short repeatable value).
//
// Threads are built on the SAME 15-doc EN+DE messy corpus Leg C uses
// (qdocs_messy_corpus()) — one seed document per thread becomes the oldest
// (innermost) message; a correction reply is added; realistic filler replies
// (logistics/admin chatter, no digits, cannot collide with tracked field
// values) pad the thread to 4K-8K tokens, verified against the actual
// tokenizer, not assumed. Explicit ("Correction: ...") and indirect ("prices
// went up...") phrasing, and early/mid/late correction position, are varied
// across the 8 threads. Four of the eight threads also carry a natural
// distractor reply — a later, unrelated-sender message that restates the
// PRE-correction value (mirrors SS1's t4/h1/h2/h4/h5 threads, which SS1
// itself treated as natural, not induced) — to test whether the model is
// pulled back to stale values the way SS1's 35B model sometimes was.
//
// Gated SS2=1. Zero engine edits; all prior probe paths (N3/N3b/CG1/DP1/
// COV1/CF1/SS1/LENS/QDOCS A-D/S1/NORM_WEIGHTED) untouched.
// ═════════════════════════════════════════════════════════════════════════

// One message in a reconstructed reply chain. `pre` = header/greeting (no
// field text, never searched); `body` = this message's own new content
// (may state 0+ field values, must contain each verbatim exactly once);
// `post` = signoff; `quote_lead` = the "On <date>, X wrote:" line introducing
// the quoted prior message (empty for the root/oldest message).
struct S2Msg {
    std::string pre, body, post, quote_lead;
    std::vector<std::pair<std::string, std::string>> fields;   // concept -> value (in `body`)
};

// Prefix every line of `s` with `prefix` (email-style quoting), remapping
// byte spans given in `s`'s own coordinates into the new (quoted) string's
// coordinates. This is the span-tracking primitive that lets SS2 avoid ever
// re-searching the assembled document for a short, possibly-repeated value.
static std::string s2_quote(const std::string& s, const std::string& prefix,
                            std::vector<std::pair<size_t, size_t>>& spans) {
    std::vector<size_t> map(s.size() + 1);
    std::string out; out.reserve(s.size() * 2 + prefix.size());
    out += prefix;
    for (size_t i = 0; i < s.size(); ++i) {
        map[i] = out.size();
        out += s[i];
        if (s[i] == '\n' && i + 1 < s.size()) out += prefix;
    }
    map[s.size()] = out.size();
    for (auto& sp : spans) { sp.first = map[sp.first]; sp.second = map[sp.second]; }
    return out;
}

// One tracked byte span in the fully-assembled document: a field mention
// (tag = concept name) or a whole message's own envelope (tag = "__msg__",
// used to classify an arbitrary cited byte position to "which message").
struct S2Span { int msg_idx; std::string tag; size_t lo, hi; };
struct S2Built { std::string text; std::vector<S2Span> spans; };

// Assemble `msgs` (oldest first) into the final nested-quote document a
// recipient would see (newest message on top, each reply quoting everything
// before it in full — real client behaviour, and why the OLD value
// naturally reappears below a correction with no extra engineering: msg0's
// content is nested, unmodified, inside every later message).
static S2Built s2_build(const std::vector<S2Msg>& msgs) {
    S2Built out;
    std::vector<std::pair<size_t, size_t>> live;
    std::vector<std::pair<int, std::string>> tags;
    std::string cur;
    for (size_t i = 0; i < msgs.size(); ++i) {
        const S2Msg& m = msgs[i];
        std::string top = m.pre + m.body + m.post + m.quote_lead;
        if (i == 0) {
            cur = top;
        } else {
            std::string quoted = s2_quote(cur, "> ", live);
            for (auto& sp : live) { sp.first += top.size(); sp.second += top.size(); }
            cur = top + quoted;
        }
        size_t body_off = m.pre.size();
        for (auto& fv : m.fields) {
            size_t p = m.body.find(fv.second);
            if (p == std::string::npos)
                throw std::runtime_error("SS2 s2_build: value \"" + fv.second +
                    "\" not found in its own message body (msg " + std::to_string(i) +
                    ", concept " + fv.first + ") — thread corpus authoring bug");
            live.push_back({body_off + p, body_off + p + fv.second.size()});
            tags.push_back({(int)i, fv.first});
        }
        live.push_back({0, top.size()});
        tags.push_back({(int)i, "__msg__"});
    }
    out.text = cur;
    for (size_t k = 0; k < live.size(); ++k)
        out.spans.push_back({tags[k].first, tags[k].second, live[k].first, live[k].second});
    return out;
}

static std::string s2_qname(bool de, int i) {
    static const char* EN[4] = {"Priya", "Alex", "Sam", "Jordan"};
    static const char* DE[4] = {"Jonas", "Mira", "Tobias", "Katrin"};
    return (de ? DE : EN)[i % 4];
}
static std::string s2_qdate(int i) {
    // 2026-02-xx: disjoint from every tracked field value (no field uses this
    // range, so a filler/reply date can never collide with a field value) AND
    // chronologically AFTER every date that appears in any seed document or
    // correction (latest is 2026-01-29, t_en5's corrected delivery_date) — a
    // first version used 2024 dates, which are EARLIER than the seed orders'
    // own 2025/2026 dates despite being replies later in the thread; the
    // model correctly flagged that as a chronological contradiction and
    // refused to extract on 2 of 8 threads (disclosed in the deliverable).
    static const char* D[8] = {"2026-02-03", "2026-02-05", "2026-02-07", "2026-02-10",
                               "2026-02-12", "2026-02-14", "2026-02-17", "2026-02-19"};
    return D[i % 8];
}
// Realistic, unrelated reply chatter — zero digits by construction, so it
// can never collide with a tracked (numeric or date) field value. Cycled;
// senders/dates vary per instance so the thread doesn't look mechanical.
static const char* S2_FILLER_EN[8] = {
    "Just a quick note that the loading dock will be repainted next week, so please use the "
    "side entrance for any deliveries until further notice. Nothing about this order changes on our end.",
    "Heads up — our usual courier is switching regional depots, so tracking links might look a "
    "little different for a while. Everything else stays exactly the same, so no action needed from you.",
    "FYI the office is quieter than usual this week because of a company retreat, so replies might "
    "be a touch slower than normal. We still have everything on file and nothing else needs to change.",
    "Reminder from the compliance team: please keep using the standard packaging slips going forward, "
    "no exceptions. This is unrelated to your order and is just a routine note to all our supplier contacts.",
    "Just letting you know our warehouse system had a scheduled maintenance window over the weekend. "
    "Everything is back online now and the details already on file are unaffected.",
    "Also wanted to say thanks again for being such an easy partner to work with — the whole team "
    "really appreciates the smooth back and forth on this account.",
    "Small heads up, our accounting team switched over to a new invoicing portal, so future paperwork "
    "will look a little different, but the underlying order information stays unchanged.",
    "One more housekeeping note: our support line hours are shifting slightly for the season, opening "
    "a bit later in the morning. That does not affect anything already agreed here.",
};
static const char* S2_FILLER_DE[8] = {
    "Nur kurz zur Info: die Ladezone wird nächste Woche neu gestrichen, bitte nutzen Sie solange den "
    "Seiteneingang für Lieferungen. An dieser Bestellung ändert sich dadurch nichts.",
    "Kurze Notiz — unser gewohnter Kurierdienst wechselt gerade den regionalen Standort, daher können "
    "Sendungsverfolgungen vorübergehend anders aussehen. Alles andere bleibt wie gehabt.",
    "Zur Information: bei uns ist es diese Woche wegen einer Teamklausur etwas ruhiger, Antworten "
    "können daher etwas länger dauern. Die Bestellung liegt uns weiterhin vollständig vor.",
    "Erinnerung von der Compliance-Abteilung: bitte weiterhin die üblichen Verpackungshinweise "
    "verwenden, ohne Ausnahme. Das betrifft nicht Ihre Bestellung, sondern ist eine routinemäßige "
    "Erinnerung an alle Lieferantenkontakte.",
    "Kurzer Hinweis: unser Lagerverwaltungssystem hatte am Wochenende ein geplantes Wartungsfenster. "
    "Alles läuft wieder normal und die hinterlegten Daten sind davon nicht betroffen.",
    "Noch etwas — vielen Dank für die weiterhin angenehme Zusammenarbeit, das ganze Team schätzt die "
    "reibungslose Kommunikation sehr.",
    "Kleiner Hinweis: unsere Buchhaltung ist auf ein neues Rechnungsportal umgestiegen, künftige "
    "Unterlagen sehen daher etwas anders aus, die zugrunde liegenden Angaben bleiben aber unverändert.",
    "Eine letzte organisatorische Notiz: unsere Erreichbarkeit verschiebt sich saisonal leicht nach "
    "hinten am Morgen. Das hat keinen Einfluss auf das hier bereits Vereinbarte.",
};

// A thread specification: one seed document from qdocs_messy_corpus() (the
// oldest/innermost message), one correction reply, and (optionally) one
// distractor reply that restates the pre-correction value.
struct S2ThreadDef {
    std::string tag; bool de; std::string seed_tag;
    std::string corr_concept, corr_body, corr_new;
    std::string position;   // "early" | "mid" | "late"
    bool explicit_ph;
    bool has_distractor; std::string distr_body;
    int target_tokens;
};

static std::vector<S2ThreadDef> ss2_thread_defs() {
    return {
      {"t_en1", false, "m_en1", "quantity",
       "Correction on the Brightwork order: we actually need 60 units of the Matte Black Easel "
       "Stand — please update the paperwork accordingly.", "60", "early", true, true,
       "Warehouse note: confirming 45 units of the Matte Black Easel Stand are already palletised "
       "and ready for the Brightwork pickup, right on schedule.", 4600},
      {"t_en3", false, "m_en3", "unit_price",
       "heads up, the galvanising supplier put their rates up again this month, so it works out "
       "to 0.42 eur each on our side now for the coach bolts.", "0.42", "mid", false, false, "", 5800},
      {"t_en5", false, "m_en5", "delivery_date",
       "One correction on the Arctic Gear order: the requested delivery date is now 2026-01-29 — "
       "our dock schedule shifted, please plan around the new date.", "2026-01-29", "late", true, true,
       "Logistics note: the carrier still has 2026-01-15 pencilled in for the Arctic Gear pickup "
       "and everything looks on track for that slot.", 5600},
      {"t_en7", false, "m_en7", "quantity",
       "quick update from the lab — turns out we only need enough Anti-Reflective Lens Blanks to "
       "cover 130 units this run, the rest of the batch got reassigned elsewhere.", "130", "mid", false, false, "", 5000},
      {"t_de1", true, "m_de1", "unit_price",
       "Korrektur zur Bergblick-Bestellung: der Einzelpreis für den Wanderstock Alu Pro beträgt "
       "jetzt 27,50 EUR — der Hersteller hat die Preise angepasst.", "27,50", "early", true, false, "", 4400},
      {"t_de3", true, "m_de3", "delivery_date",
       "kurz zur info, die spedition kann den termin diese woche nicht mehr schaffen, das rutscht "
       "jetzt eher richtung 2025-10-07.", "2025-10-07", "late", false, true,
       "Kurzer Zwischenstand von der Filiale: für 2025-09-30 ist die Anlieferung der Schrauben "
       "weiterhin so im System eingetragen, sieht soweit gut aus.", 5400},
      {"t_de5", true, "m_de5", "quantity",
       "Korrektur: wir benötigen jetzt 96 Stück der Isolierflasche 750ml statt vorher, ein "
       "zweites Team ist dazugekommen.", "96", "mid", true, false, "", 5200},
      {"t_de7", true, "m_de7", "unit_price",
       "achso, der Großhändler hat das Griffband teurer gemacht, das kommt jetzt eher auf 2,10 "
       "EUR raus.", "2,10", "early", false, true,
       "Kurze Rückmeldung vom Lager: für das Griffband steht bei uns weiterhin 1,95 EUR pro Rolle "
       "in der Preisliste, falls das noch relevant ist.", 4800},
    };
}

// Build the message chain for one thread def, padding with filler replies
// (position-aware: growth happens on the side that preserves the requested
// early/mid/late placement) until the ChatML-rendered prompt reaches
// `def.target_tokens`, verified against the real tokenizer each step.
static std::vector<S2Msg> s2_build_thread(Tokenizer* tok, const std::string& task,
                                          const S2ThreadDef& def, const QMessy& seed,
                                          int& corr_idx_out, std::string& corr_old_out) {
    bool de = def.de;
    std::vector<S2Msg> msgs;
    S2Msg m0; m0.body = seed.document;
    for (auto& f : seed.fields) m0.fields.push_back({f.concept, f.value});
    msgs.push_back(m0);

    corr_old_out.clear();
    for (auto& f : seed.fields) if (f.concept == def.corr_concept) corr_old_out = f.value;
    if (corr_old_out.empty())
        throw std::runtime_error("SS2: corr_concept '" + def.corr_concept + "' not found in seed " + def.tag);

    int fi = 0;
    std::string prev_name = "the original sender";
    auto mk_filler = [&]() -> S2Msg {
        const char* txt = (de ? S2_FILLER_DE : S2_FILLER_EN)[fi % 8];
        std::string name = s2_qname(de, fi), date = s2_qdate(fi);
        S2Msg m;
        m.pre = "From: " + name + "\nDate: " + date + "\n\n" + (de ? "Hallo,\n\n" : "Hi,\n\n");
        m.body = txt;
        m.post = de ? "\n\nViele Grüße\n" + name + "\n" : "\n\nBest,\n" + name + "\n";
        m.quote_lead = de ? ("\nAm " + date + " schrieb " + prev_name + ":\n")
                          : ("\nOn " + date + ", " + prev_name + " wrote:\n");
        prev_name = name; fi++;
        return m;
    };
    auto mk_named = [&](const std::string& body, const std::string& concept,
                        const std::string& value) -> S2Msg {
        std::string name = s2_qname(de, fi), date = s2_qdate(fi);
        S2Msg m;
        m.pre = "From: " + name + "\nDate: " + date + "\n\n" + (de ? "Hallo,\n\n" : "Hi,\n\n");
        m.body = body;
        m.post = de ? "\n\nViele Grüße\n" + name + "\n" : "\n\nBest,\n" + name + "\n";
        m.quote_lead = de ? ("\nAm " + date + " schrieb " + prev_name + ":\n")
                          : ("\nOn " + date + ", " + prev_name + " wrote:\n");
        if (!concept.empty()) m.fields = {{concept, value}};
        prev_name = name; fi++;
        return m;
    };

    int pre_seed = def.position == "late" ? 6 : def.position == "mid" ? 3 : 1;
    for (int i = 0; i < pre_seed; ++i) msgs.push_back(mk_filler());

    msgs.push_back(mk_named(def.corr_body, def.corr_concept, def.corr_new));
    int corr_pos = (int)msgs.size() - 1;

    if (def.has_distractor) {
        msgs.push_back(mk_filler());
        msgs.push_back(mk_named(def.distr_body, def.corr_concept, corr_old_out));
    }

    auto token_len = [&]() {
        S2Built b = s2_build(msgs);
        return (int)tok->encode(qdocs_chat_prompt(b.text, task)).size();
    };
    bool toggle = true;
    for (int guard = 0; guard < 300 && token_len() < def.target_tokens; ++guard) {
        S2Msg f = mk_filler();
        if (def.position == "late") { msgs.insert(msgs.begin() + corr_pos, f); corr_pos++; }
        else if (def.position == "early") { msgs.push_back(f); }
        else {
            if (toggle) { msgs.insert(msgs.begin() + corr_pos, f); corr_pos++; }
            else { msgs.push_back(f); }
            toggle = !toggle;
        }
    }
    corr_idx_out = corr_pos;
    return msgs;
}

static bool s2_bytes_to_toks(const std::vector<size_t>& cum, int n, size_t b0, size_t b1,
                             int& lo, int& hi) {
    lo = hi = -1;
    for (int k = 0; k < n; ++k) if (cum[k] < b1 && cum[k + 1] > b0) { if (lo < 0) lo = k; hi = k; }
    return lo >= 0;
}
static int s2_classify(const std::vector<std::pair<size_t, size_t>>& env, size_t pos) {
    for (size_t i = 0; i < env.size(); ++i)
        if (pos >= env[i].first && pos < env[i].second) return (int)i;
    return -1;
}

// One scored field instance (the corrected field, one per thread).
struct S2AlarmRec {
    std::string thread, position; bool de, explicit_ph, has_distractor;
    int label = -1;           // 1 FRESH (emitted current), 0 STALE (emitted superseded), -1 excluded
    bool alarm = false;
    int cited_msg = -1, current_msg = -1;
};

static int run_ss2(ForwardPassBase* fp, ggml_backend_sched_t sched,
                   Tokenizer* tok, const ModelMetadata& meta,
                   const std::vector<int32_t>& attn_layers) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ SS2 — coverage-free thread stale-source alarm (grammar, long) ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");
    std::printf("citation tap: L%dH%d (slot %d)  [defaults 0/13 = Qwen3.6 L3H13; override via "
                "ATTN_FROZEN_SLOT/ATTN_FROZEN_HEAD]\n", (*g_attn_layers)[FROZEN_SLOT], FROZEN_HEAD, FROZEN_SLOT);

    const int TOL = 2;
    // NO GRAMMAR. The fixed KV grammar was REFUTED by measurement 2026-07-16 and
    // REMOVED from the product path in Stage 2 (docs/note-nogrammar-refutation.md);
    // the shipped lens decodes free in one prefill with a tolerant parse
    // (server_lens.h; http_server.cpp: "Builds NO grammar").
    // SS1 §5 recommended a grammar-constrained thread extractor — but SS1 is dated
    // 2026-07-12, FOUR DAYS BEFORE that refutation, so the recommendation is stale.
    // Measured cost of having followed it: the fixed-KV grammar bounds each pair's
    // SHAPE but not the NUMBER of pairs, so a looping model emits syntactically
    // valid infinite JSON (t_en1 invented ~20 nested key variants, never closed).
    // Free decode terminates on '}' — the stopping rule the grammar never had.
    static const std::vector<std::string> S2_KEYS = {
        "customer","product","quantity","unit_price","total",
        "order_date","delivery_date","order_number"};
    std::vector<int> tap(attn_layers.begin(), attn_layers.end());
    // Deliberately tighter than Leg C's "extract every fact" hint: Leg C's
    // single-email documents bound "every fact" naturally, but SS2's threads
    // carry many pages of realistic-but-irrelevant filler (logistics/admin
    // chatter) whose whole purpose is padding length, not being extracted.
    // A first run with Leg C's looser hint spiraled into a repetitive,
    // never-closing key explosion over that filler on long threads — a real,
    // disclosed finding (see docs/note-ss2-thread-alarm.md "What was NOT
    // done"), not a scoring bug, but avoided here by naming the exact ORDER
    // key set and explicitly excluding admin/courtesy content.
    const char* TASK = "\n\nThis is an email thread about ONE order, with possibly one or more "
        "corrections later in the thread, plus unrelated administrative/logistics/courtesy "
        "remarks mixed in. Extract ONLY the order's CURRENT field values (after applying any "
        "corrections) as a flat JSON object with exactly these keys, omitting any that never "
        "appear: customer, product, quantity, unit_price, total, order_date, delivery_date, "
        "order_number. Copy each value verbatim from the email it currently holds. Ignore all "
        "administrative, logistics, and courtesy remarks — do not create keys for them. Output "
        "ONLY the JSON object, nothing else.";

    auto corpus = qdocs_messy_corpus();
    auto find_seed = [&](const std::string& tag) -> const QMessy& {
        for (auto& d : corpus) if (d.tag == tag) return d;
        throw std::runtime_error("SS2: seed tag not found: " + tag);
    };

    struct CiteAcc { int n = 0, t1 = 0, t3 = 0; };
    CiteAcc cite_all;
    std::vector<S2AlarmRec> alarms;
    int excluded = 0, labeled_total = 0;

    for (auto& def : ss2_thread_defs()) {
        const QMessy& seed = find_seed(def.seed_tag);
        int corr_idx = -1; std::string corr_old;
        std::vector<S2Msg> msgs = s2_build_thread(tok, TASK, def, seed, corr_idx, corr_old);
        S2Built built = s2_build(msgs);
        std::string prompt = qdocs_chat_prompt(built.text, TASK);
        size_t base = prompt.find(built.text);
        if (base == std::string::npos)
            throw std::runtime_error("SS2: assembled thread text not found verbatim in its own "
                                     "ChatML prompt (thread " + def.tag + ") — cannot locate spans");

        FreeRun R = run_freegen(fp, sched, tok, meta, prompt, "", tap, 380);
        int actual_tokens = R.P;
        std::printf("\n[%s%s] msgs=%zu corr@%d/%zu (%.0f%%) pos=%s phrasing=%s distractor=%s "
                    "target=%d actual=%d tok\n  gen: %s\n",
                    def.tag.c_str(), def.de ? " DE" : "", msgs.size(), corr_idx, msgs.size() - 1,
                    100.0 * corr_idx / std::max<size_t>(1, msgs.size() - 1), def.position.c_str(),
                    def.explicit_ph ? "explicit" : "indirect", def.has_distractor ? "yes" : "no",
                    def.target_tokens, actual_tokens, R.gen_text.c_str());

        std::vector<size_t> pcum = cum_bytes(tok, R.prompt_tokens);
        // Tolerant parse of the free output — the shipped lens contract.
        auto gfs = parse_fields(tok, R, S2_KEYS);
        std::vector<size_t> gcum = cum_bytes(tok, R.gen_tokens);

        // message envelopes, absolute byte ranges in the prompt
        std::vector<std::pair<size_t, size_t>> env(msgs.size());
        std::map<std::string, std::vector<std::pair<int, std::pair<size_t, size_t>>>> field_spans;
        for (auto& sp : built.spans) {
            if (sp.tag == "__msg__") env[sp.msg_idx] = {base + sp.lo, base + sp.hi};
            else field_spans[sp.tag].push_back({sp.msg_idx, {base + sp.lo, base + sp.hi}});
        }

        auto score_field = [&](const std::string& concept, const std::string& emitted_value,
                               const std::vector<std::pair<size_t, size_t>>& target_spans) -> int {
            // returns the classified msg_idx of the top-1 citation on the LAST
            // scored gen-token (majority not needed for single/short values;
            // multi-token values use the final token, which is where the
            // value's identity is fully resolved).
            size_t gb = R.gen_text.find(emitted_value);
            if (gb == std::string::npos) return -2;   // normalized away
            size_t ge = gb + emitted_value.size();
            int last_cited = -1;
            for (int g = 0; g < (int)R.gen_tokens.size(); ++g) {
                if (!(gcum[g] < ge && gcum[g + 1] > gb)) continue;
                if (g < 1 || g - 1 >= (int)R.rows.size()) continue;
                int n_kv = R.n_kv_at_step[g - 1];
                auto tk = topk_head(R.rows[g - 1][L3H13_SLOT], FROZEN_HEAD, n_kv, 3);
                cite_all.n++;
                bool t1 = false, t3 = false;
                for (auto& [tlo, thi] : target_spans) {
                    int slo, shi;
                    if (!s2_bytes_to_toks(pcum, R.P, tlo, thi, slo, shi)) continue;
                    auto in = [&](int p) { return p >= slo - TOL && p <= shi + TOL; };
                    if (!tk.empty() && in(tk[0].first)) t1 = true;
                    for (auto& pr : tk) if (in(pr.first)) { t3 = true; break; }
                }
                if (t1) cite_all.t1++;
                if (t3) cite_all.t3++;
                if (!tk.empty()) last_cited = s2_classify(env, pcum[tk[0].first]);
            }
            return last_cited;
        };

        for (auto& f : seed.fields) {
            labeled_total++;
            if (f.concept == def.corr_concept) {
                // "superseded" = every OTHER message that stated a value for
                // this concept (msg0's original, plus the distractor if it
                // restates the old value too) — the alarm's ground-truth
                // predicate, independent of which value was actually emitted.
                std::vector<int> superseded_msgs;
                for (auto& [mi, span] : field_spans[f.concept]) if (mi != corr_idx) superseded_msgs.push_back(mi);

                // Which value did the model emit UNDER THIS KEY? Deliberately NOT a
                // whole-output substring search. That was a real false negative: on
                // t_en3 the model emitted the STALE "unit_price":"0.35 eur" while the
                // fresh 0.42 sat under an invented key, and because has_new is tested
                // first the run scored a genuinely stale emission as FRESH — silently
                // inverting the one error class this alarm exists to catch.
                std::string emitted_here;
                for (auto& gf : gfs) if (gf.name == f.concept) { emitted_here = gf.value; break; }
                bool has_new = !emitted_here.empty() &&
                               emitted_here.find(def.corr_new) != std::string::npos;
                bool has_old = !emitted_here.empty() &&
                               emitted_here.find(corr_old)    != std::string::npos;
                std::string emitted; int label; std::vector<std::pair<size_t, size_t>> target;
                if (has_new) {
                    emitted = def.corr_new; label = 1;
                    for (auto& [mi, span] : field_spans[f.concept]) if (mi == corr_idx) target.push_back(span);
                } else if (has_old) {
                    emitted = corr_old; label = 0;
                    for (auto& [mi, span] : field_spans[f.concept]) if (mi != corr_idx) target.push_back(span);
                } else { excluded++; continue; }
                int cited = score_field(f.concept, emitted, target);
                if (cited == -2) { excluded++; continue; }
                S2AlarmRec r; r.thread = def.tag; r.position = def.position; r.de = def.de;
                r.explicit_ph = def.explicit_ph; r.has_distractor = def.has_distractor;
                r.label = label; r.cited_msg = cited; r.current_msg = corr_idx;
                // ALARM fires iff the citation lands in a message that is
                // superseded for THIS field — not merely "not the current
                // message" (a filler-message citation is noise, not a
                // stale-source finding, and must not count as either).
                r.alarm = std::find(superseded_msgs.begin(), superseded_msgs.end(), cited) != superseded_msgs.end();
                alarms.push_back(r);
                std::printf("  [ALARM FIELD] %-13s emitted=%-12s label=%-5s cited_msg=%-3d "
                            "current_msg=%-3d -> %s\n", f.concept.c_str(), emitted.c_str(),
                            label ? "FRESH" : "STALE", cited, corr_idx, r.alarm ? "ALARM" : "silent");
            } else {
                std::string ev;
                for (auto& gf : gfs) if (gf.name == f.concept) { ev = gf.value; break; }
                if (ev.find(f.value) == std::string::npos) { excluded++; continue; }
                std::vector<std::pair<size_t, size_t>> target;
                for (auto& [mi, span] : field_spans[f.concept]) if (mi == 0) target.push_back(span);
                score_field(f.concept, f.value, target);
            }
        }
    }

    // ── Gate 0: is thread-scale citation measurable? ─────────────────────────
    double t1p = cite_all.n ? 100.0 * cite_all.t1 / cite_all.n : 0;
    double t3p = cite_all.n ? 100.0 * cite_all.t3 / cite_all.n : 0;
    std::printf("\n================ GATE 0 — citation at thread scale ================\n");
    std::printf("  value tokens scored: %d  top1=%.0f%% (%d/%d)  top3=%.0f%% (%d/%d)  [bar top3>=85%%]\n",
                cite_all.n, t1p, cite_all.t1, cite_all.n, t3p, cite_all.t3, cite_all.n);
    std::printf("  excluded (normalized / not emitted verbatim): %d / %d labeled fields\n",
                excluded, labeled_total);
    bool gate0 = cite_all.n > 0 && t3p >= 85.0;
    std::printf("  => %s\n", gate0 ? "GATE 0 PASS" : "GATE 0 FAIL — citation not measurable at thread scale");

    // ── Gate 1: natural STALE rate ────────────────────────────────────────────
    int fresh = 0, stale = 0;
    for (auto& r : alarms) { if (r.label == 1) fresh++; else if (r.label == 0) stale++; }
    std::printf("\n================ GATE 1 — natural STALE rate ================\n");
    std::printf("  correction instances scored: %d (FRESH %d, STALE %d)\n", (int)alarms.size(), fresh, stale);
    std::printf("  %-8s %-6s %-9s %-9s %-6s %-6s\n", "thread", "lang", "position", "phrasing", "distr", "label");
    for (auto& r : alarms)
        std::printf("  %-8s %-6s %-9s %-9s %-6s %-6s\n", r.thread.c_str(), r.de ? "DE" : "EN",
                    r.position.c_str(), r.explicit_ph ? "explicit" : "indirect",
                    r.has_distractor ? "yes" : "no", r.label ? "FRESH" : "STALE");
    bool gate1 = stale > 0;
    std::printf("  => %s\n", gate1 ? "STALE occurs naturally at length" :
                "NO natural STALE at 4K-8K tokens (model honored every correction)");

    // ── Gate 2: does the alarm work? ──────────────────────────────────────────
    std::printf("\n================ GATE 2 — alarm precision/recall ================\n");
    if (!gate1) {
        std::printf("  (skipped: Gate 1 yielded 0 natural STALE — nothing to detect)\n");
        int fp_n = 0, fp_fired = 0;
        for (auto& r : alarms) if (r.label == 1) { fp_n++; if (r.alarm) fp_fired++; }
        std::printf("  FRESH false-alarm rate (fired despite correct answer): %d/%d (%.0f%%)\n",
                    fp_fired, fp_n, fp_n ? 100.0 * fp_fired / fp_n : 0);
    } else {
        int tp = 0, fn = 0, fp_n = 0, tn = 0;
        for (auto& r : alarms) {
            if (r.label == 0) { if (r.alarm) tp++; else fn++; }
            else { if (r.alarm) fp_n++; else tn++; }
        }
        double precision = (tp + fp_n) ? 100.0 * tp / (tp + fp_n) : 0;
        double recall = (tp + fn) ? 100.0 * tp / (tp + fn) : 0;
        double falarm = (fp_n + tn) ? 100.0 * fp_n / (fp_n + tn) : 0;
        std::printf("  TP=%d FN=%d FP=%d TN=%d\n", tp, fn, fp_n, tn);
        std::printf("  precision=%.0f%%  recall=%.0f%%  FRESH false-alarm rate=%.0f%%\n",
                    precision, recall, falarm);
    }

    std::printf("\n================ VERDICT (SS2) ================\n");
    if (!gate0) std::printf("  => STILL BLOCKED: Gate 0 failed (top3 %.0f%% < 85%% bar) — citation "
                            "not measurable at thread scale.\n", t3p);
    else if (!gate1) std::printf("  => NO JOB: Gate 0 passed but natural STALE is 0/%d at 4K-8K "
                                 "tokens — the model honored every correction; the alarm has no "
                                 "job on this model at this envelope.\n", (int)alarms.size());
    else std::printf("  => ALARM VALIDATED (see Gate 2 precision/recall above) — but n is small "
                     "(%d STALE case%s); read the numbers as a signal, not a settled rate.\n",
                     stale, stale == 1 ? "" : "s");
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// Norm-weighted attention (Metric B) — a calibration probe, gated by
// NORM_WEIGHTED=1. docs/note-lens-norm-weighted-metric.md.
//
// Metric A (shipped): score_j = alpha_j (the raw post-softmax kq_soft row).
// Metric B (candidate): score_j = alpha_j * ||V_j||, renormalized over j so
// the row sums to 1 -- the standard correction for what a head actually
// contributes to `output = sum_j alpha_j * V_j`.
//
// ||V_j|| is read straight off the persisted V cache
// (simple_kv_cache::get_v_cache_tensor) -- no graph change, no new tap. Every
// symbol above this point (Leg C, COV1, N3, ...) is untouched; this section
// only ADDS reader functions and two new driver legs.
// ═════════════════════════════════════════════════════════════════════════

// ‖V_j‖ for KV positions [0, n_kv) of one (cache-layer, kv-head).
//
// `cache_layer` is NOT the GGUF block index `il` (e.g. 27 for L27H13).
// simple_kv_cache is built ATTENTION-LAYERS-ONLY: qwen35.cpp assigns
// `kv_layer_map_[il] = n_attn_layers++` while walking blocks 0..block_count-1
// in increasing order and skipping non-attention blocks, so the compact cache
// index for tap slot `s` is `s` itself -- `attn_layers` (main()) is built by
// the identical increasing-il scan over `kq_soft.<il>` tensors, so slot index
// and cache index coincide by construction. Passing the block number here
// instead would silently read a DIFFERENT layer's V cache under the citation
// head's own printed name -- the exact trap this probe was warned about.
//
// Quantized KV (Q8_0/Q4_0) is out of reach here by construction, not by
// omission: this probe requires --flash-attn OFF (kq_soft never materializes
// under flash -- DecodePolicy::is_attn_impl_coherent() enforces the pairing
// elsewhere), and a quantized KV cache is refused without flash attention
// (kv_type_requires_flash_refusal, kv_cache_simple.h). So any run that reaches
// this function already has an F32 or F16 V cache; anything else fails loud
// instead of silently misreading quantized blocks as floats.
static std::vector<float> value_norms(simple_kv_cache* kv, int cache_layer, int kv_head,
                                      int head_dim, int n_head_kv, int n_kv) {
    ggml_tensor* vt = kv->get_v_cache_tensor(cache_layer);
    if (!vt)
        throw std::runtime_error("value_norms: expected a V cache tensor for cache layer " +
                                 std::to_string(cache_layer) + ", actual nullptr");
    if (vt->type != GGML_TYPE_F32 && vt->type != GGML_TYPE_F16)
        throw std::runtime_error("value_norms: expected V cache type f32 or f16 (quantized KV "
            "requires --flash-attn, which this probe forbids), actual " +
            std::string(ggml_type_name(vt->type)));
    const int n_embd_v = n_head_kv * head_dim;
    if ((int)vt->ne[0] != n_embd_v)
        throw std::runtime_error("value_norms: expected V cache ne[0]=" +
            std::to_string(n_embd_v) + " (n_head_kv*head_dim), actual " + std::to_string(vt->ne[0]));
    if (kv_head < 0 || kv_head >= n_head_kv)
        throw std::runtime_error("value_norms: expected kv_head in [0," + std::to_string(n_head_kv) +
            "), actual " + std::to_string(kv_head));
    // Slot 0 (the only slot this single-slot probe ever uses): byte offset
    // within the tensor is just position * nb[1]. A freshly-allocated ggml
    // tensor is contiguous, so nb[1] must equal n_embd_v * element_size --
    // asserted explicitly (from the tensor's own nb[], not assumed) rather
    // than trusted, per the layout trap this probe was warned about.
    const size_t elem = ggml_type_size(vt->type);
    if (vt->nb[1] != (size_t)n_embd_v * elem)
        throw std::runtime_error("value_norms: expected V cache row stride " +
            std::to_string((size_t)n_embd_v * elem) + " bytes (contiguous), actual " +
            std::to_string(vt->nb[1]));
    std::vector<uint8_t> raw((size_t)n_kv * n_embd_v * elem);
    ggml_backend_tensor_get(vt, raw.data(), 0, raw.size());

    std::vector<float> out(n_kv, 0.0f);
    for (int j = 0; j < n_kv; ++j) {
        double ss = 0;
        const uint8_t* rowb = raw.data() + (size_t)j * n_embd_v * elem;
        if (vt->type == GGML_TYPE_F32) {
            const float* row = reinterpret_cast<const float*>(rowb);
            for (int d = 0; d < head_dim; ++d) { double x = row[(size_t)kv_head * head_dim + d]; ss += x * x; }
        } else {
            const ggml_fp16_t* row = reinterpret_cast<const ggml_fp16_t*>(rowb);
            for (int d = 0; d < head_dim; ++d) {
                double x = ggml_fp16_to_fp32(row[(size_t)kv_head * head_dim + d]);
                ss += x * x;
            }
        }
        out[j] = (float)std::sqrt(ss);
    }
    return out;
}

// Sanity check (trap #3): ||V_j|| should be smooth and positive with no zeros
// in the populated interior -- a block of exact zeros means the read is past
// the populated rows, or striding wrong. Position 0 (BOS) and the very last
// written row are excluded: BOS can legitimately be a near-zero-norm sink,
// and there is no guarantee here about what lies immediately after the
// current write cursor.
static void sanity_check_norms(const std::vector<float>& vnorm, const char* where) {
    if (vnorm.size() < 4) return;
    for (size_t j = 1; j + 2 < vnorm.size(); ++j) {
        if (vnorm[j] <= 0.0f)
            throw std::runtime_error(std::string("value_norms sanity: expected every populated "
                "interior position to have ||V||>0, actual 0 at position ") + std::to_string(j) +
                " of " + std::to_string(vnorm.size()) + " (" + where + ")");
    }
}

// Metric B's analogue of topk_head: rank KV positions by alpha_j * ||V_j||
// instead of alpha_j, renormalized over j so the row sums to 1 (a positive
// rescale of every candidate, so it does not change the ranking by itself --
// done anyway so the reported score is on the same 0..1 "mass" scale as
// Metric A, for honest side-by-side printing). Same BOS-sink exclusion
// (j starts at 1) as Metric A, for the same reason: an attention sink would
// otherwise win every top-1 by construction regardless of ||V||.
static std::vector<std::pair<int, float>> topk_head_weighted(
        const std::vector<float>& row, int h, int n_kv, int k,
        const std::vector<float>& vnorm) {
    const float* r = row.data() + (size_t)h * n_kv;
    double tot = 0;
    for (int j = 0; j < n_kv; ++j) tot += (double)r[j] * vnorm[j];
    const float inv = tot > 0 ? (float)(1.0 / tot) : 0.0f;
    std::vector<std::pair<int, float>> v;
    for (int j = 1; j < n_kv; ++j) v.push_back({j, (float)((double)r[j] * vnorm[j] * inv)});
    std::partial_sort(v.begin(), v.begin() + std::min((size_t)k, v.size()), v.end(),
                      [](auto& a, auto& b) { return a.second > b.second; });
    if ((int)v.size() > k) v.resize(k);
    return v;
}

// Metric B's analogue of the raw-alpha span-mass sum used throughout (topk
// excepted): renormalized weighted mass on [lo,hi], as a fraction of the
// row's total renormalized mass over ALL j in [0,n_kv) (mirroring how Metric
// A's raw row already sums to 1 over all j including the BOS position).
static double span_mass_weighted(const std::vector<float>& row, int h, int n_kv,
                                 int lo, int hi, const std::vector<float>& vnorm) {
    const float* r = row.data() + (size_t)h * n_kv;
    double tot = 0;
    for (int j = 0; j < n_kv; ++j) tot += (double)r[j] * vnorm[j];
    if (tot <= 0) return 0;
    double s = 0;
    for (int q = lo; q <= hi && q < n_kv; ++q) s += (double)r[q] * vnorm[q];
    return s / tot;
}

// Rank-based AUC (Mann-Whitney U / (n_pos*n_neg)): P(random positive scores
// higher than random negative), with average-rank tie handling. A model-free
// separation figure, independent of any chosen threshold.
static double auc_of(const std::vector<std::pair<double, bool>>& data) {
    std::vector<std::pair<double, bool>> v = data;
    std::sort(v.begin(), v.end());
    const int n = (int)v.size();
    std::vector<double> rank(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && v[j].first == v[i].first) j++;
        double avg_rank = (i + j - 1) / 2.0 + 1.0;  // 1-indexed
        for (int k2 = i; k2 < j; ++k2) rank[k2] = avg_rank;
        i = j;
    }
    double sum_rank_pos = 0; int n_pos = 0, n_neg = 0;
    for (int k2 = 0; k2 < n; ++k2) { if (v[k2].second) { sum_rank_pos += rank[k2]; n_pos++; } else n_neg++; }
    if (n_pos == 0 || n_neg == 0) return -1;
    double u = sum_rank_pos - n_pos * (n_pos + 1) / 2.0;
    return u / ((double)n_pos * n_neg);
}

// Paired A/B citation+coverage eval for one grounded field value. Mirrors
// qdocs_eval_field exactly for Metric A (byte-identical numbers) and adds the
// Metric B twin computed from the SAME rows in the SAME pass.
struct QFieldEvalDual {
    bool found_verbatim = false;
    int cite_n = 0, t1A = 0, t3A = 0, t1B = 0, t3B = 0;
    double cov_peakA = -1, cov_peakB = -1;
};
static QFieldEvalDual qdocs_eval_field_dual(Tokenizer* tok, const FreeRun& R, const std::string& value,
        const std::vector<int32_t>& attn_layers, int TOL,
        const std::vector<std::vector<float>>& vnorm_cite,  // [kv_head][pos] at L3H13_SLOT's cache layer
        const std::vector<std::vector<float>>& vnorm_cov,   // [kv_head][pos] at L11_SLOT's cache layer
        int group) {
    QFieldEvalDual e;
    int slo, shi;
    if (!qdocs_span_in_prompt(tok, R, value, slo, shi)) return e;

    double sc[12][4]; span_scalars(R, slo, shi, sc, attn_layers);
    e.cov_peakA = sc[1 + L11_SLOT][0];

    double peakB = 0;
    for (int t = 0; t < (int)R.rows.size(); ++t) {
        int n_kv = R.n_kv_at_step[t];
        double layer_best = 0;
        for (int h = 0; h < R.n_head; ++h) {
            int kh = h / group;
            double s = span_mass_weighted(R.rows[t][L11_SLOT], h, n_kv, slo, shi, vnorm_cov[kh]);
            if (s > layer_best) layer_best = s;
        }
        if (layer_best > peakB) peakB = layer_best;
    }
    e.cov_peakB = peakB;

    std::vector<size_t> gcum = cum_bytes(tok, R.gen_tokens);
    size_t gb = R.gen_text.find(value);
    if (gb == std::string::npos) return e;   // normalized away
    e.found_verbatim = true;
    size_t ge = gb + value.size();
    const int kh_cite = FROZEN_HEAD / group;
    for (int g = 0; g < (int)R.gen_tokens.size(); ++g) {
        if (!(gcum[g] < ge && gcum[g + 1] > gb)) continue;
        if (g < 1 || g - 1 >= (int)R.rows.size()) continue;
        int n_kv = R.n_kv_at_step[g - 1];
        auto in = [&](int p) { return p >= slo - TOL && p <= shi + TOL; };
        e.cite_n++;
        auto tkA = topk_head(R.rows[g - 1][L3H13_SLOT], FROZEN_HEAD, n_kv, 3);
        if (!tkA.empty() && in(tkA[0].first)) e.t1A++;
        for (auto& pr : tkA) if (in(pr.first)) { e.t3A++; break; }
        auto tkB = topk_head_weighted(R.rows[g - 1][L3H13_SLOT], FROZEN_HEAD, n_kv, 3, vnorm_cite[kh_cite]);
        if (!tkB.empty() && in(tkB[0].first)) e.t1B++;
        for (auto& pr : tkB) if (in(pr.first)) { e.t3B++; break; }
    }
    return e;
}

// ── ARM 1 — citation, paired A/B, on the SAME Leg C messy corpus (15 docs
// EN+DE, 413 scored value tokens) that produced the shipped 98%/89% numbers.
// Metric A must reproduce those numbers or the plumbing here is wrong.
// used_peaks_AB collects every used-span's (peakA, peakB) pair for Arm 2's
// final cross-check against Leg C's own population.
static int run_qdocs_leg_c_dual(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                Tokenizer* tok, const ModelMetadata& meta,
                                const std::vector<int32_t>& attn_layers,
                                std::vector<std::pair<double, double>>* used_peaks_AB) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ NORM-WEIGHTED ARM 1 — citation, Leg C messy corpus (paired A/B) ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    const int TOL = 2;
    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const uint32_t vocab_size = (uint32_t)vocab.size();
    qinf::TokenTrie trie; trie.build(vocab);
    auto gr = qinf::GrammarVocab::parse_impl(QDOCS_GBNF);
    gr->set_token_trie(&trie);
    std::vector<int> taps10(attn_layers.begin(), attn_layers.end());
    const char* TASK = "\n\nExtract every fact from the email above into a flat JSON "
        "object of \"key\": \"value\" pairs. Use short snake_case keys — prefer keys like: "
        "customer, product, quantity, unit_price, total, order_date, delivery, order_number. "
        "Copy each value verbatim from the email. Output ONLY the JSON object, nothing else.";

    auto corpus = qdocs_messy_corpus();
    simple_kv_cache* kv = fp->snapshot_kv_cache();
    if (!kv) throw std::runtime_error("run_qdocs_leg_c_dual: expected a KV cache, actual nullptr");
    const int n_head_kv = (int)meta.attention_head_count_kv;
    const int head_dim  = (int)meta.attention_value_length;
    const int group     = (int)meta.attention_head_count / n_head_kv;

    struct Acc { int cite_n = 0, t1A = 0, t3A = 0, t1B = 0, t3B = 0;
                 int used_spans = 0, used_clearA = 0;
                 std::vector<double> peaksA, peaksB; int grounded = 0, labeled = 0; };
    Acc en, de;
    auto pick = [&](bool d) -> Acc& { return d ? de : en; };

    for (const QMessy& d : corpus) {
        std::string prompt = qdocs_chat_prompt(d.document, TASK);
        std::vector<GStep> tr;
        FreeRun R = run_freegen_grammar(fp, sched, tok, meta, prompt, "", taps10,
                                        320, gr.get(), vocab, vocab_size, &tr);
        std::printf("\n[%s%s] %s\n", d.tag.c_str(), d.de ? " DE" : "", R.gen_text.c_str());

        int n_kv_final = R.n_kv_at_step.empty() ? R.P : R.n_kv_at_step.back();
        std::vector<std::vector<float>> vnorm_cite(n_head_kv), vnorm_cov(n_head_kv);
        for (int kh = 0; kh < n_head_kv; ++kh) {
            vnorm_cite[kh] = value_norms(kv, L3H13_SLOT, kh, head_dim, n_head_kv, n_kv_final);
            sanity_check_norms(vnorm_cite[kh], "leg-c-dual citation layer");
            vnorm_cov[kh] = value_norms(kv, L11_SLOT, kh, head_dim, n_head_kv, n_kv_final);
            sanity_check_norms(vnorm_cov[kh], "leg-c-dual coverage layer");
        }

        Acc& A = pick(d.de);
        for (const QLabel& f : d.fields) {
            A.labeled++;
            QFieldEvalDual e = qdocs_eval_field_dual(tok, R, f.value, attn_layers, TOL,
                                                     vnorm_cite, vnorm_cov, group);
            if (e.cov_peakA >= 0) {
                A.used_spans++; A.peaksA.push_back(e.cov_peakA); A.peaksB.push_back(e.cov_peakB);
                if (e.cov_peakA >= 0.705) A.used_clearA++;
                if (used_peaks_AB) used_peaks_AB->push_back({e.cov_peakA, e.cov_peakB});
            }
            if (!e.found_verbatim) continue;
            A.cite_n += e.cite_n; A.t1A += e.t1A; A.t3A += e.t3A; A.t1B += e.t1B; A.t3B += e.t3B;
            A.grounded++;
        }
    }

    auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end());
        return v[v.size() / 2]; };
    auto report = [&](const char* label, const Acc& A) {
        std::printf("\n──────── %s ────────\n", label);
        std::printf("  citation A (raw alpha)     top1 %d/%d (%.0f%%)  top3 %d/%d (%.0f%%)\n",
                    A.t1A, A.cite_n, A.cite_n ? 100.0 * A.t1A / A.cite_n : 0,
                    A.t3A, A.cite_n, A.cite_n ? 100.0 * A.t3A / A.cite_n : 0);
        std::printf("  citation B (alpha*||V||)   top1 %d/%d (%.0f%%)  top3 %d/%d (%.0f%%)\n",
                    A.t1B, A.cite_n, A.cite_n ? 100.0 * A.t1B / A.cite_n : 0,
                    A.t3B, A.cite_n, A.cite_n ? 100.0 * A.t3B / A.cite_n : 0);
        std::printf("  coverage used-clear A @0.705: %d/%d (%.0f%%)  median peakA %.3f  median peakB %.3f\n",
                    A.used_clearA, A.used_spans, A.used_spans ? 100.0 * A.used_clearA / A.used_spans : 0,
                    med(A.peaksA), med(A.peaksB));
    };
    report("EN", en);
    report("DE", de);
    Acc all;
    all.cite_n = en.cite_n + de.cite_n; all.t1A = en.t1A + de.t1A; all.t3A = en.t3A + de.t3A;
    all.t1B = en.t1B + de.t1B; all.t3B = en.t3B + de.t3B;
    all.used_spans = en.used_spans + de.used_spans; all.used_clearA = en.used_clearA + de.used_clearA;
    for (double p : en.peaksA) all.peaksA.push_back(p);
    for (double p : de.peaksA) all.peaksA.push_back(p);
    for (double p : en.peaksB) all.peaksB.push_back(p);
    for (double p : de.peaksB) all.peaksB.push_back(p);
    report("COMBINED", all);

    double t3A = all.cite_n ? 100.0 * all.t3A / all.cite_n : 0;
    double t3B = all.cite_n ? 100.0 * all.t3B / all.cite_n : 0;
    double t1A = all.cite_n ? 100.0 * all.t1A / all.cite_n : 0;
    double t1B = all.cite_n ? 100.0 * all.t1B / all.cite_n : 0;
    std::printf("\n── ARM 1 GATE: metric A top1 %.1f%% top3 %.1f%%  (repro check: want ~89%%/~98%%)\n"
                "               metric B top1 %.1f%% top3 %.1f%%  (STOP condition: top3 <95%%) ──\n",
                t1A, t3A, t1B, t3B);
    return 0;
}

// ── ARM 2 — coverage separation, paired A/B, on the COV1 corpus (cov_calib +
// cov_held, the only corpus here with an explicit USED-vs-DROPPED label) at
// the frozen coverage layer (layer 11 / L11_SLOT — no layer search, per the
// gate's scope). Reports medians, AUC, and each metric's OWN best threshold
// (found independently — the 0.705 constant is Metric-A-only and is never
// applied to Metric B's differently-scaled quantity).
static int run_coverage_probe_dual(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                   Tokenizer* tok, const ModelMetadata& meta,
                                   const std::vector<int32_t>& attn_layers,
                                   Thr* out_thrA, Thr* out_thrB) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ NORM-WEIGHTED ARM 2 — coverage separation, COV1 corpus (paired) ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");
    std::vector<int> tap_layers(attn_layers.begin(), attn_layers.end());
    simple_kv_cache* kv = fp->snapshot_kv_cache();
    if (!kv) throw std::runtime_error("run_coverage_probe_dual: expected a KV cache, actual nullptr");
    const int n_head_kv = (int)meta.attention_head_count_kv;
    const int head_dim  = (int)meta.attention_value_length;
    const int group     = (int)meta.attention_head_count / n_head_kv;

    struct SpanRecW { std::string prompt, marker; int cls = 0, label = 0, len = 0;
                       double peakA = 0, peakB = 0; };
    std::vector<SpanRecW> cal, hel;

    auto run_set = [&](std::vector<CovPrompt> set, std::vector<SpanRecW>& out) {
        for (auto& cp : set) {
            FreeRun R = run_freegen(fp, sched, tok, meta, cp.body, cp.instr,
                                    tap_layers, 320, false, cp.close);
            int n_kv_final = R.n_kv_at_step.empty() ? R.P : R.n_kv_at_step.back();
            std::vector<std::vector<float>> vnorm(n_head_kv);
            for (int kh = 0; kh < n_head_kv; ++kh) {
                vnorm[kh] = value_norms(kv, L11_SLOT, kh, head_dim, n_head_kv, n_kv_final);
                sanity_check_norms(vnorm[kh], "cov1-dual coverage layer");
            }
            for (auto& tg : cp.tg) {
                int lo, hi;
                if (!find_token_span(tok, R.prompt_tokens, cp.body, tg.marker, lo, hi)) {
                    std::printf("  WARN marker not found: %s\n", tg.marker.c_str());
                    continue;
                }
                SpanRecW r; r.prompt = cp.tag; r.marker = tg.marker; r.cls = tg.cls; r.len = hi - lo + 1;
                r.label = tg.cls != CT_TARGET ? -1 : (R.gen_text.find(tg.used) != std::string::npos ? 1 : 0);
                double sc[12][4]; span_scalars(R, lo, hi, sc, attn_layers);
                r.peakA = sc[1 + L11_SLOT][0];
                double peakB = 0;
                for (int t = 0; t < (int)R.rows.size(); ++t) {
                    int n_kv = R.n_kv_at_step[t];
                    double layer_best = 0;
                    for (int h = 0; h < R.n_head; ++h) {
                        int kh = h / group;
                        double s = span_mass_weighted(R.rows[t][L11_SLOT], h, n_kv, lo, hi, vnorm[kh]);
                        if (s > layer_best) layer_best = s;
                    }
                    if (layer_best > peakB) peakB = layer_best;
                }
                r.peakB = peakB;
                out.push_back(r);
            }
        }
    };
    std::printf("\n---- generations + labels ----\n");
    run_set(cov_calib(), cal);
    std::printf("---- held ----\n");
    run_set(cov_held(), hel);

    auto pairsA = [&](const std::vector<SpanRecW>& v) {
        std::vector<std::pair<double, bool>> d;
        for (auto& r : v) if (r.cls == CT_TARGET) d.push_back({r.peakA, r.label == 1});
        return d;
    };
    auto pairsB = [&](const std::vector<SpanRecW>& v) {
        std::vector<std::pair<double, bool>> d;
        for (auto& r : v) if (r.cls == CT_TARGET) d.push_back({r.peakB, r.label == 1});
        return d;
    };

    Thr thrA = best_threshold(pairsA(cal));
    Thr thrB = best_threshold(pairsB(cal));
    double heldA = apply_thr(thrA, pairsA(hel));
    double heldB = apply_thr(thrB, pairsB(hel));
    double aucA_cal = auc_of(pairsA(cal)), aucA_hel = auc_of(pairsA(hel));
    double aucB_cal = auc_of(pairsB(cal)), aucB_hel = auc_of(pairsB(hel));

    std::printf("\n---- separation, TARGET USED vs DROPPED (calib=%zu, held=%zu spans, layer %d) ----\n",
                pairsA(cal).size(), pairsA(hel).size(), attn_layers[L11_SLOT]);
    std::printf("  metric A (raw alpha)     calib-best thr=%.3f dir%+d acc=%.0f%%  AUC(calib)=%.3f  "
                "HELD acc=%.0f%% AUC(held)=%.3f\n",
                thrA.t, thrA.dir, 100 * thrA.acc, aucA_cal, 100 * heldA, aucA_hel);
    std::printf("  metric B (alpha*||V||)   calib-best thr=%.3f dir%+d acc=%.0f%%  AUC(calib)=%.3f  "
                "HELD acc=%.0f%% AUC(held)=%.3f\n",
                thrB.t, thrB.dir, 100 * thrB.acc, aucB_cal, 100 * heldB, aucB_hel);

    std::vector<SpanRecW> all = cal; all.insert(all.end(), hel.begin(), hel.end());
    auto med_cls = [&](int cls, int lab, bool useB) {
        std::vector<double> v;
        for (auto& r : all) if (r.cls == cls && (lab < -1 || r.label == lab)) v.push_back(useB ? r.peakB : r.peakA);
        return median(v);
    };
    std::printf("\n---- per-class medians (COV1 pooled, calib+held) ----\n");
    std::printf("  A: FILLER %.3f  DROPPED %.3f  USED %.3f  [VALUE anchor %.3f]\n",
                med_cls(CT_FILLER, -2, false), med_cls(CT_TARGET, 0, false),
                med_cls(CT_TARGET, 1, false), med_cls(CT_VALUE, -2, false));
    std::printf("  B: FILLER %.3f  DROPPED %.3f  USED %.3f  [VALUE anchor %.3f]\n",
                med_cls(CT_FILLER, -2, true), med_cls(CT_TARGET, 0, true),
                med_cls(CT_TARGET, 1, true), med_cls(CT_VALUE, -2, true));

    if (out_thrA) *out_thrA = thrA;
    if (out_thrB) *out_thrB = thrB;
    return 0;
}

// Combined driver: Arm 1 (citation, Leg C) then Arm 2 (coverage separation,
// COV1), then a final cross-check applying Metric B's OWN best threshold
// (picked on COV1, independently of Metric A's frozen 0.705) to Leg C's own
// used-span population — the number that pairs directly against the shipped
// 87%-at-0.705 result docs/note-lens-qwen38-probe.md reports.
static int run_norm_weighted_probe(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                   Tokenizer* tok, const ModelMetadata& meta,
                                   const std::vector<int32_t>& attn_layers) {
    std::vector<std::pair<double, double>> used_peaks_AB;
    int rc1 = run_qdocs_leg_c_dual(fp, sched, tok, meta, attn_layers, &used_peaks_AB);
    if (rc1) return rc1;

    Thr thrA, thrB;
    int rc2 = run_coverage_probe_dual(fp, sched, tok, meta, attn_layers, &thrA, &thrB);
    if (rc2) return rc2;

    int clearA_frozen = 0, clearA_own = 0, clearB = 0;
    for (auto& pr : used_peaks_AB) {
        if (pr.first >= 0.705) clearA_frozen++;
        bool a_clear_own = thrA.dir > 0 ? pr.first >= thrA.t : pr.first <= thrA.t;
        if (a_clear_own) clearA_own++;
        bool b_clear = thrB.dir > 0 ? pr.second >= thrB.t : pr.second <= thrB.t;
        if (b_clear) clearB++;
    }
    int n = (int)used_peaks_AB.size();
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ ARM 2 CROSS-CHECK — Leg C used-span clear-rate at each metric's║\n");
    std::printf("║ OWN best operating point (thresholds picked on COV1, THIS run) ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");
    std::printf("  metric A @ 0.705 (frozen, COV1-derived on Qwen 3.6):       %d/%d (%.0f%%)  [bar >=90%%]\n",
                clearA_frozen, n, n ? 100.0 * clearA_frozen / n : 0);
    std::printf("  metric A @ %.3f dir%+d (COV1-derived on THIS run, fair):   %d/%d (%.0f%%)  [bar >=90%%]\n",
                thrA.t, thrA.dir, clearA_own, n, n ? 100.0 * clearA_own / n : 0);
    std::printf("  metric B @ %.3f dir%+d (COV1-derived on THIS run):         %d/%d (%.0f%%)  [bar >=90%%]\n",
                thrB.t, thrB.dir, clearB, n, n ? 100.0 * clearB / n : 0);
    std::printf("  (the A-frozen vs A-own-threshold pair isolates how much of any A-vs-B gap is\n"
                "   metric shape vs. simply recalibrating the threshold on THIS model/corpus.)\n");
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════
// GEMMA4 candidate-space SEARCH — every (layer,head) candidate scored on the
// Leg C messy corpus DIRECTLY, Metric A and Metric B in the same pass. Gated
// GEMMA4_SEARCH_DUAL=1. docs/note-lens-gemma-norm-weighted.md.
//
// This does NOT select on Prompt A/B/C and confirm on the messy corpus
// afterward -- note-lens-gemma4-probe.md §2 found that procedure has almost
// no discriminating power on Gemma (ranks 1-4 tied at exactly 75.0%). Every
// candidate here is scored directly against the same 15-doc EN+DE messy
// corpus and grammar/task Leg C uses, so a reproduction of L4H7's documented
// 41%/51% under Metric A is the plumbing sanity check (§6 of that note).
//
// Metric B needs ||V_j|| per candidate, and Gemma 4 splits attention across
// TWO disjoint KV caches with DIFFERENT (n_kv_heads, head_dim) shapes: global
// layers (V==K, no attn_v.weight, wide head_dim, few KV heads) and sliding
// layers (separate V, narrow head_dim, more KV heads). Neither the per-layer
// kind nor the two caches are reachable through the single-cache accessor
// every other leg in this file uses (snapshot_kv_cache() returns nullptr on
// gemma4 -- gemma4.h overrides the MULTI-cache snapshot_kv_caches() instead).
// The per-layer kind is derived here from tensor-inventory presence of
// blk.<il>.attn_v.weight -- the exact same test Gemma4Config::from_metadata
// (gemma4.cpp) uses internally. This file does not include gemma4.h, so the
// derivation is reproduced against meta.raw_kv / meta.tensor_inventory
// rather than reached through the recipe's own config object; the two caches
// snapshot_kv_caches() returns are then matched to (global, swa) by SHAPE
// (n_layers, V ne[0]) rather than trusted by the documented return order --
// "derive it, do not assume it" applied to a second, harder cache-layer trap
// than the single-cache one the brief named.
// ═════════════════════════════════════════════════════════════════════════

// ||V_j|| for every kv_head of one (cache, cache_layer) in ONE tensor read.
// The single-head value_norms() above would re-read the same layer's full V
// tensor once per kv_head; at 48 candidate slots (vs. the 1-2 frozen slots
// every other leg reads) that redundancy is worth avoiding. Runs the exact
// same checks (type, ne[0], row stride) value_norms() runs, so this is not
// more trusting of the tensor than the reader it stands in for.
static std::vector<std::vector<float>> value_norms_all_heads(
        simple_kv_cache* kv, int cache_layer, int head_dim, int n_head_kv, int n_kv) {
    ggml_tensor* vt = kv->get_v_cache_tensor(cache_layer);
    if (!vt)
        throw std::runtime_error("value_norms_all_heads: expected a V cache tensor for cache layer " +
                                 std::to_string(cache_layer) + ", actual nullptr");
    if (vt->type != GGML_TYPE_F32 && vt->type != GGML_TYPE_F16)
        throw std::runtime_error("value_norms_all_heads: expected V cache type f32 or f16, actual " +
            std::string(ggml_type_name(vt->type)));
    const int n_embd_v = n_head_kv * head_dim;
    if ((int)vt->ne[0] != n_embd_v)
        throw std::runtime_error("value_norms_all_heads: expected V cache ne[0]=" +
            std::to_string(n_embd_v) + " (n_head_kv*head_dim), actual " + std::to_string(vt->ne[0]));
    const size_t elem = ggml_type_size(vt->type);
    if (vt->nb[1] != (size_t)n_embd_v * elem)
        throw std::runtime_error("value_norms_all_heads: expected V cache row stride " +
            std::to_string((size_t)n_embd_v * elem) + " bytes (contiguous), actual " +
            std::to_string(vt->nb[1]));
    std::vector<uint8_t> raw((size_t)n_kv * n_embd_v * elem);
    ggml_backend_tensor_get(vt, raw.data(), 0, raw.size());

    std::vector<std::vector<float>> out(n_head_kv, std::vector<float>(n_kv, 0.0f));
    for (int j = 0; j < n_kv; ++j) {
        const uint8_t* rowb = raw.data() + (size_t)j * n_embd_v * elem;
        for (int kh = 0; kh < n_head_kv; ++kh) {
            double ss = 0;
            if (vt->type == GGML_TYPE_F32) {
                const float* row = reinterpret_cast<const float*>(rowb);
                for (int d = 0; d < head_dim; ++d) { double x = row[(size_t)kh * head_dim + d]; ss += x * x; }
            } else {
                const ggml_fp16_t* row = reinterpret_cast<const ggml_fp16_t*>(rowb);
                for (int d = 0; d < head_dim; ++d) {
                    double x = ggml_fp16_to_fp32(row[(size_t)kh * head_dim + d]);
                    ss += x * x;
                }
            }
            out[kh][j] = (float)std::sqrt(ss);
        }
    }
    return out;
}

// Allocation-free top-3-in-span scan. topk_head/topk_head_weighted above
// build+partial_sort a vector per call; this leg calls a per-position scan
// ~48 slots x 16 heads x ~400 scored tokens x 2 metrics times (~600K calls),
// where a heap allocation per call would dominate wall time. Position 0 (the
// BOS-sink) is excluded, matching topk_head's own convention, for the same
// reason: an attention sink would otherwise win top-1 by construction.
static void top3_positions(const float* r, int n_kv, int& p1, int& p2, int& p3) {
    p1 = p2 = p3 = -1;
    float v1 = -1e30f, v2 = -1e30f, v3 = -1e30f;
    for (int j = 1; j < n_kv; ++j) {
        float x = r[j];
        if (x > v1)      { v3 = v2; p3 = p2; v2 = v1; p2 = p1; v1 = x; p1 = j; }
        else if (x > v2) { v3 = v2; p3 = p2; v2 = x;  p2 = j; }
        else if (x > v3) { v3 = x;  p3 = j; }
    }
}
// Metric B's analogue: ranks by alpha_j * ||V_j|| instead of alpha_j.
// Unnormalized -- renormalizing by the row's total is a positive rescale
// that cannot change which positions are top-1/top-3 (see topk_head_weighted's
// own comment), so the normalization that matters for honest score PRINTING
// is skipped here on purpose; only membership is accumulated at this scale.
static void top3_positions_weighted(const float* r, const float* vnorm, int n_kv,
                                    int& p1, int& p2, int& p3) {
    p1 = p2 = p3 = -1;
    float v1 = -1e30f, v2 = -1e30f, v3 = -1e30f;
    for (int j = 1; j < n_kv; ++j) {
        float x = r[j] * vnorm[j];
        if (x > v1)      { v3 = v2; p3 = p2; v2 = v1; p2 = p1; v1 = x; p1 = j; }
        else if (x > v2) { v3 = v2; p3 = p2; v2 = x;  p2 = j; }
        else if (x > v3) { v3 = x;  p3 = j; }
    }
}

struct Gemma4KvLayout {
    std::vector<bool> is_global;      // size block_count
    std::vector<int>  cache_idx;      // per-layer index WITHIN its own cache
    uint32_t head_dim_swa = 0, head_dim_global = 0;
    uint32_t n_kv_heads_swa = 0, n_kv_heads_global = 0;
    uint32_t n_swa_layers = 0, n_global_layers = 0;
};

// Reproduces Gemma4Config::from_metadata's per-layer-kind derivation
// (gemma4.cpp) read-only, from meta.raw_kv / meta.tensor_inventory, so this
// probe stays out of src/ and does not include gemma4.h.
static Gemma4KvLayout resolve_gemma4_kv_layout(const ModelMetadata& meta) {
    if (meta.architecture.rfind("gemma4", 0) != 0)
        throw std::runtime_error("resolve_gemma4_kv_layout: expected arch 'gemma4*', actual '" +
                                 meta.architecture + "'");
    Gemma4KvLayout L;
    const auto& inv = meta.tensor_inventory;
    L.is_global.assign(meta.block_count, false);
    L.cache_idx.assign(meta.block_count, -1);
    int swa_idx = 0, glb_idx = 0;
    for (uint32_t il = 0; il < meta.block_count; ++il) {
        const std::string vk = "blk." + std::to_string(il) + ".attn_v.weight";
        bool has_v = inv.find(vk) != inv.end();
        L.is_global[il] = !has_v;
        L.cache_idx[il] = has_v ? swa_idx++ : glb_idx++;
    }
    L.n_swa_layers    = (uint32_t)swa_idx;
    L.n_global_layers = (uint32_t)glb_idx;
    L.head_dim_global = meta.raw_kv.get_uint32("gemma4.attention.key_length");
    L.head_dim_swa    = meta.raw_kv.get_uint32("gemma4.attention.key_length_swa");
    auto k_out_dim = [&](uint32_t il) -> uint64_t {
        const std::string kk = "blk." + std::to_string(il) + ".attn_k.weight";
        auto it = inv.find(kk);
        if (it == inv.end() || it->second.shape.size() < 2)
            throw std::runtime_error("resolve_gemma4_kv_layout: tensor '" + kk +
                                     "' missing or rank-deficient");
        return it->second.shape[1];
    };
    bool seen_swa = false, seen_glb = false;
    for (uint32_t il = 0; il < meta.block_count && !(seen_swa && seen_glb); ++il) {
        if (!seen_swa && !L.is_global[il]) {
            L.n_kv_heads_swa = (uint32_t)(k_out_dim(il) / L.head_dim_swa); seen_swa = true;
        }
        if (!seen_glb && L.is_global[il]) {
            L.n_kv_heads_global = (uint32_t)(k_out_dim(il) / L.head_dim_global); seen_glb = true;
        }
    }
    if (L.n_swa_layers && !L.n_kv_heads_swa)
        throw std::runtime_error("resolve_gemma4_kv_layout: n_kv_heads_swa expected >0, actual 0");
    if (L.n_global_layers && !L.n_kv_heads_global)
        throw std::runtime_error("resolve_gemma4_kv_layout: n_kv_heads_global expected >0, actual 0");
    return L;
}

// Matches fp->snapshot_kv_caches() to (global, swa) by SHAPE (n_layers, V
// ne[0]) rather than by the documented return order -- see the section
// banner above.
struct Gemma4Caches { simple_kv_cache* global = nullptr; simple_kv_cache* swa = nullptr; };
static Gemma4Caches resolve_gemma4_caches(ForwardPassBase* fp, const Gemma4KvLayout& L) {
    auto caches = fp->snapshot_kv_caches();
    Gemma4Caches out;
    for (simple_kv_cache* kv : caches) {
        if (!kv) continue;
        uint32_t nl = kv->get_n_layers();
        ggml_tensor* v0 = nl ? kv->get_v_cache_tensor(0) : nullptr;
        uint32_t ne0 = v0 ? (uint32_t)v0->ne[0] : 0;
        bool matches_global = L.n_global_layers && nl == L.n_global_layers &&
                              ne0 == L.n_kv_heads_global * L.head_dim_global;
        bool matches_swa    = L.n_swa_layers && nl == L.n_swa_layers &&
                              ne0 == L.n_kv_heads_swa * L.head_dim_swa;
        if (matches_global && matches_swa)
            throw std::runtime_error("resolve_gemma4_caches: a cache matches BOTH global and swa "
                "shapes (n_layers=" + std::to_string(nl) + " ne0=" + std::to_string(ne0) +
                ") -- ambiguous, refusing to guess");
        if (matches_global) {
            if (out.global) throw std::runtime_error("resolve_gemma4_caches: two caches match global shape");
            out.global = kv;
        } else if (matches_swa) {
            if (out.swa) throw std::runtime_error("resolve_gemma4_caches: two caches match swa shape");
            out.swa = kv;
        } else {
            throw std::runtime_error("resolve_gemma4_caches: cache (n_layers=" + std::to_string(nl) +
                " ne0=" + std::to_string(ne0) + ") matches neither global nor swa shape");
        }
    }
    if (L.n_global_layers && !out.global)
        throw std::runtime_error("resolve_gemma4_caches: expected a global-shaped cache, found none");
    if (L.n_swa_layers && !out.swa)
        throw std::runtime_error("resolve_gemma4_caches: expected a swa-shaped cache, found none");
    return out;
}

// Average-rank (ties share the mean rank), rank 1 = highest score.
static std::vector<double> rank_desc_avg(const std::vector<double>& v) {
    int n = (int)v.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    std::vector<double> rank(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && v[idx[j]] == v[idx[i]]) j++;
        double avg = (i + j - 1) / 2.0 + 1.0;
        for (int k2 = i; k2 < j; ++k2) rank[idx[k2]] = avg;
        i = j;
    }
    return rank;
}
static double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    int n = (int)a.size();
    double ma = 0, mb = 0;
    for (int i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double num = 0, da = 0, db = 0;
    for (int i = 0; i < n; ++i) {
        double xa = a[i] - ma, xb = b[i] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    return (da > 0 && db > 0) ? num / std::sqrt(da * db) : 0.0;
}

static int run_gemma4_search_dual(ForwardPassBase* fp, ggml_backend_sched_t sched,
                                  Tokenizer* tok, const ModelMetadata& meta,
                                  const std::vector<int32_t>& attn_layers) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ GEMMA4 SEARCH — all (layer,head) candidates, Leg C corpus, A/B ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    const int TOL = 2;
    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const uint32_t vocab_size = (uint32_t)vocab.size();
    qinf::TokenTrie trie; trie.build(vocab);
    auto gr = qinf::GrammarVocab::parse_impl(QDOCS_GBNF);
    gr->set_token_trie(&trie);
    std::vector<int> taps(attn_layers.begin(), attn_layers.end());
    // Byte-identical to run_qdocs_leg_c's TASK — the L4H7 41%/51% reproduction
    // check below is only meaningful if every setting matches that leg.
    const char* TASK = "\n\nExtract every fact from the email above into a flat JSON "
        "object of \"key\": \"value\" pairs. Use short snake_case keys — prefer keys like: "
        "customer, product, quantity, unit_price, total, order_date, delivery, order_number. "
        "Copy each value verbatim from the email. Output ONLY the JSON object, nothing else.";

    auto corpus = qdocs_messy_corpus();

    const int S = (int)attn_layers.size();
    const int H = (int)meta.attention_head_count;
    const int C = S * H;
    std::printf("candidates: %d slots x %d heads = %d\n", S, H, C);

    Gemma4KvLayout L = resolve_gemma4_kv_layout(meta);
    Gemma4Caches   K = resolve_gemma4_caches(fp, L);
    std::printf("kv layout: swa layers=%u kv_heads=%u head_dim=%u | global layers=%u kv_heads=%u head_dim=%u\n",
                L.n_swa_layers, L.n_kv_heads_swa, L.head_dim_swa,
                L.n_global_layers, L.n_kv_heads_global, L.head_dim_global);

    std::vector<simple_kv_cache*> slot_cache(S);
    std::vector<int> slot_cache_layer(S), slot_head_dim(S), slot_n_kv_heads(S), slot_group(S);
    for (int s = 0; s < S; ++s) {
        int il = attn_layers[s];
        bool g = L.is_global[il];
        slot_cache[s]       = g ? K.global : K.swa;
        slot_cache_layer[s] = L.cache_idx[il];
        slot_head_dim[s]    = g ? (int)L.head_dim_global   : (int)L.head_dim_swa;
        slot_n_kv_heads[s]  = g ? (int)L.n_kv_heads_global : (int)L.n_kv_heads_swa;
        if (H % slot_n_kv_heads[s] != 0)
            throw std::runtime_error("run_gemma4_search_dual: expected n_head (" + std::to_string(H) +
                ") a multiple of n_kv_heads (" + std::to_string(slot_n_kv_heads[s]) + ") at il=" +
                std::to_string(il));
        slot_group[s] = H / slot_n_kv_heads[s];
    }

    std::vector<long> t1A_en(C, 0), t3A_en(C, 0), t1A_de(C, 0), t3A_de(C, 0);
    std::vector<long> t1B_en(C, 0), t3B_en(C, 0), t1B_de(C, 0), t3B_de(C, 0);
    int cite_n_en = 0, cite_n_de = 0;

    for (const QMessy& d : corpus) {
        std::string prompt = qdocs_chat_prompt(d.document, TASK);
        std::vector<GStep> tr;
        FreeRun R = run_freegen_grammar(fp, sched, tok, meta, prompt, "", taps,
                                        320, gr.get(), vocab, vocab_size, &tr);
        std::printf("\n[%s%s] %s\n", d.tag.c_str(), d.de ? " DE" : "", R.gen_text.c_str());

        int n_kv_final = R.n_kv_at_step.empty() ? R.P : R.n_kv_at_step.back();
        std::vector<std::vector<std::vector<float>>> vnorm_by_slot(S);
        for (int s = 0; s < S; ++s) {
            vnorm_by_slot[s] = value_norms_all_heads(slot_cache[s], slot_cache_layer[s],
                                                     slot_head_dim[s], slot_n_kv_heads[s], n_kv_final);
            for (auto& v : vnorm_by_slot[s]) sanity_check_norms(v, "gemma4-search value-norm layer");
        }
        // Diagnostic (first doc only): ||V_j|| spread for the eventual winner
        // (L7H13, a sliding layer) and for the first global layer found --
        // gemma4.cpp RMS-norms V with NO learned weight before it is cached
        // ("Vcur = ggml_rms_norm(ctx0, Vcur, eps)"), which forces every V_j to
        // a near-constant L2 norm BY CONSTRUCTION. If that is why Metric A/B
        // tie, this spread should be tiny relative to the mean.
        if (&d == &corpus.front()) {
            auto stats = [&](int s, int kh) {
                const auto& v = vnorm_by_slot[s][kh];
                double mn = 1e30, mx = -1e30, sum = 0, sumsq = 0; int n = 0;
                for (size_t j = 1; j + 1 < v.size(); ++j) {  // skip BOS + tail, as sanity_check_norms does
                    double x = v[j]; mn = std::min(mn, x); mx = std::max(mx, x);
                    sum += x; sumsq += x * x; n++;
                }
                double mean = n ? sum / n : 0;
                double var = n ? sumsq / n - mean * mean : 0;
                double sd = var > 0 ? std::sqrt(var) : 0;
                std::printf("  ||V|| slot=%d kv_head=%d: n=%d min=%.4f mean=%.4f max=%.4f sd=%.4f cv=%.4f%%\n",
                            s, kh, n, mn, mean, mx, sd, mean ? 100.0 * sd / mean : 0);
            };
            std::printf("\n── ||V_j|| SPREAD (doc 1 only, sanity for the A/B tie) ──\n");
            for (int s = 0; s < S; ++s) if (attn_layers[s] == 7) { stats(s, 13 / slot_group[s]); break; }
            for (int s = 0; s < S; ++s) if (L.is_global[attn_layers[s]]) { stats(s, 0); break; }
        }

        std::vector<size_t> gcum = cum_bytes(tok, R.gen_tokens);
        for (const QLabel& f : d.fields) {
            int slo, shi;
            if (!qdocs_span_in_prompt(tok, R, f.value, slo, shi)) continue;
            size_t gb = R.gen_text.find(f.value);
            if (gb == std::string::npos) continue;   // normalized away, unscoreable
            size_t ge = gb + f.value.size();
            auto in = [&](int p) { return p >= 1 && p >= slo - TOL && p <= shi + TOL; };

            for (int g = 0; g < (int)R.gen_tokens.size(); ++g) {
                if (!(gcum[g] < ge && gcum[g + 1] > gb)) continue;
                if (g < 1 || g - 1 >= (int)R.rows.size()) continue;
                int n_kv = R.n_kv_at_step[g - 1];
                if (d.de) cite_n_de++; else cite_n_en++;
                for (int s = 0; s < S; ++s) {
                    const float* rowBase = R.rows[g - 1][s].data();
                    int group = slot_group[s];
                    for (int h = 0; h < H; ++h) {
                        int cand = s * H + h;
                        const float* r = rowBase + (size_t)h * n_kv;
                        int a1, a2, a3; top3_positions(r, n_kv, a1, a2, a3);
                        bool a1in = in(a1), a3in = in(a1) || in(a2) || in(a3);
                        int b1, b2, b3;
                        top3_positions_weighted(r, vnorm_by_slot[s][h / group].data(), n_kv, b1, b2, b3);
                        bool b1in = in(b1), b3in = in(b1) || in(b2) || in(b3);
                        if (d.de) {
                            if (a1in) t1A_de[cand]++; if (a3in) t3A_de[cand]++;
                            if (b1in) t1B_de[cand]++; if (b3in) t3B_de[cand]++;
                        } else {
                            if (a1in) t1A_en[cand]++; if (a3in) t3A_en[cand]++;
                            if (b1in) t1B_en[cand]++; if (b3in) t3B_en[cand]++;
                        }
                    }
                }
            }
        }
    }

    const int cite_n = cite_n_en + cite_n_de;
    std::printf("\nscored value tokens: EN %d  DE %d  combined %d\n", cite_n_en, cite_n_de, cite_n);

    // Rank all C candidates under each metric: primary=top1 combined count,
    // tie-break=top3 combined count (same convention main()'s Prompt-A
    // "global best" search below uses).
    std::vector<double> compA(C), compB(C);
    std::vector<long> t1A(C), t3A(C), t1B(C), t3B(C);
    for (int c = 0; c < C; ++c) {
        t1A[c] = t1A_en[c] + t1A_de[c]; t3A[c] = t3A_en[c] + t3A_de[c];
        t1B[c] = t1B_en[c] + t1B_de[c]; t3B[c] = t3B_en[c] + t3B_de[c];
        compA[c] = (double)t1A[c] * (cite_n + 1) + (double)t3A[c];
        compB[c] = (double)t1B[c] * (cite_n + 1) + (double)t3B[c];
    }
    // Diagnostic: how many of the 768 candidates does Metric B actually move
    // at the raw hit-count level (before any ranking/tie-break)? Spearman
    // alone cannot distinguish "every candidate identical" from "a few tiny,
    // rank-order-preserving nudges" at 4-decimal printing precision.
    int diff_t1 = 0, diff_t3 = 0; long max_abs_t1_diff = 0, max_abs_t3_diff = 0;
    for (int c = 0; c < C; ++c) {
        if (t1A[c] != t1B[c]) { diff_t1++; max_abs_t1_diff = std::max(max_abs_t1_diff, std::abs(t1A[c] - t1B[c])); }
        if (t3A[c] != t3B[c]) { diff_t3++; max_abs_t3_diff = std::max(max_abs_t3_diff, std::abs(t3A[c] - t3B[c])); }
    }
    std::printf("\n── METRIC A vs B, RAW HIT-COUNT DIFFERENCES (of %d candidates) ──\n", C);
    std::printf("  top1 differs on %d candidates (max |delta|=%ld)\n", diff_t1, max_abs_t1_diff);
    std::printf("  top3 differs on %d candidates (max |delta|=%ld)\n", diff_t3, max_abs_t3_diff);

    std::vector<double> rankA = rank_desc_avg(compA);
    std::vector<double> rankB = rank_desc_avg(compB);
    double spearman = pearson(rankA, rankB);

    int bestA = 0, bestB = 0;
    for (int c = 1; c < C; ++c) {
        if (compA[c] > compA[bestA]) bestA = c;
        if (compB[c] > compB[bestB]) bestB = c;
    }

    auto layer_of = [&](int c) { return attn_layers[c / H]; };
    auto head_of  = [&](int c) { return c % H; };
    auto pct = [&](long n) { return cite_n ? 100.0 * n / cite_n : 0.0; };
    auto printCand = [&](const std::string& label, int c) {
        std::printf("  %-14s L%dH%-3d  A: top1 %ld/%d (%.1f%%) top3 %ld/%d (%.1f%%)  |  "
                    "B: top1 %ld/%d (%.1f%%) top3 %ld/%d (%.1f%%)  |  rankA=%.1f rankB=%.1f\n",
                    label.c_str(), layer_of(c), head_of(c),
                    t1A[c], cite_n, pct(t1A[c]), t3A[c], cite_n, pct(t3A[c]),
                    t1B[c], cite_n, pct(t1B[c]), t3B[c], cite_n, pct(t3B[c]), rankA[c], rankB[c]);
    };
    std::printf("\n── BEST CANDIDATES ──\n");
    printCand("best under A", bestA);
    printCand("best under B", bestB);

    int clearA70 = 0, clearA80 = 0, clearA90 = 0, clearB70 = 0, clearB80 = 0, clearB90 = 0;
    for (int c = 0; c < C; ++c) {
        double t3a = pct(t3A[c]), t3b = pct(t3B[c]);
        if (t3a >= 70) clearA70++; if (t3a >= 80) clearA80++; if (t3a >= 90) clearA90++;
        if (t3b >= 70) clearB70++; if (t3b >= 80) clearB80++; if (t3b >= 90) clearB90++;
    }
    std::printf("\n── CLEAR-RATE COUNTS (top3, of %d candidates) ──\n", C);
    std::printf("  metric A: >=70%% %d   >=80%% %d   >=90%% %d\n", clearA70, clearA80, clearA90);
    std::printf("  metric B: >=70%% %d   >=80%% %d   >=90%% %d\n", clearB70, clearB80, clearB90);

    std::printf("\n── SPEARMAN(rankA, rankB) over all %d candidates: %.4f ──\n", C, spearman);

    std::vector<int> orderA(C), orderB(C);
    for (int c = 0; c < C; ++c) { orderA[c] = c; orderB[c] = c; }
    std::sort(orderA.begin(), orderA.end(), [&](int a, int b) { return compA[a] > compA[b]; });
    std::sort(orderB.begin(), orderB.end(), [&](int a, int b) { return compB[a] > compB[b]; });
    std::printf("\n── TOP 10 UNDER METRIC A ──\n");
    for (int i = 0; i < 10 && i < C; ++i) printCand("#" + std::to_string(i + 1), orderA[i]);
    std::printf("\n── TOP 10 UNDER METRIC B ──\n");
    for (int i = 0; i < 10 && i < C; ++i) printCand("#" + std::to_string(i + 1), orderB[i]);

    // BROKEN check: metric A at L4H7 must reproduce the documented 41%/51%
    // baseline (note-lens-gemma4-probe.md §6), or this leg's plumbing -- not
    // the metric -- is what is wrong.
    for (int c = 0; c < C; ++c) {
        if (layer_of(c) == 4 && head_of(c) == 7) {
            std::printf("\n── L4H7 REPRODUCTION CHECK (metric A only, vs note-lens-gemma4-probe.md §6) ──\n");
            std::printf("  documented: top1 161/397 (41%%)  top3 204/397 (51%%)\n");
            std::printf("  this run:   top1 %ld/%d (%.1f%%)  top3 %ld/%d (%.1f%%)\n",
                        t1A[c], cite_n, pct(t1A[c]), t3A[c], cite_n, pct(t3A[c]));
            break;
        }
    }
    return 0;
}

// ── Leg A — fixed grammar × tap sanity (the everything-stops leg) ────────────
static int run_qdocs_leg_a(ForwardPassBase* fp, ggml_backend_sched_t sched,
                           Tokenizer* tok, const ModelMetadata& meta,
                           const std::vector<int32_t>& attn_layers) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ QEMMI-DOCS LEG A — fixed grammar × tap sanity                 ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    const int TOL = 2;
    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const uint32_t vocab_size = (uint32_t)vocab.size();
    std::printf("vocab_size=%u\n", vocab_size);

    // Build the trie once (production narrows LITERAL candidates through it).
    qinf::TokenTrie trie;
    trie.build(vocab);
    std::printf("trie tokens indexed=%zu\n", trie.token_count());

    auto gr = qinf::GrammarVocab::parse_impl(QDOCS_GBNF);
    gr->set_token_trie(&trie);
    std::printf("grammar parsed OK (root/pair/key/value/ws)\n");

    // One grounded order email, rendered through the production chat template
    // (thinking off). The grammar emits the '{' opener itself — no prime.
    std::string document =
        "From: purchasing@globex.example\nSubject: New Order\n\n"
        "Hi team,\n\n"
        "Kindly arrange an order for customer Globex Industries.\n"
        "Order date: 2025-11-02.\n"
        "We require a quantity of 875 units.\n"
        "The unit price is 47.30 EUR.\n"
        "The order total comes to 41387.50 EUR.\n\n"
        "Regards.\n";
    std::string body = qdocs_chat_prompt(document);
    std::string instr = "";
    std::vector<std::string> values = {
        "Globex Industries", "2025-11-02", "875", "47.30", "41387.50"};
    const std::string filler = "Hi team,";

    std::vector<int> taps10(attn_layers.begin(), attn_layers.end());  // all 10 attn layers

    // ── Constrained arm (KV grammar, sparse head) ──
    std::vector<GStep> trace;
    FreeRun G = run_freegen_grammar(fp, sched, tok, meta, body, instr, taps10,
                                    220, gr.get(), vocab, vocab_size, &trace);
    std::printf("\n[constrained] gen (%zu tok): %s\n", G.gen_tokens.size(), G.gen_text.c_str());

    // Q1 — rows exist under the sparse-head graph and sum to 1.0.
    std::printf("\n── Q1: kq_soft rows under sparse head ──\n");
    int rows_bad = 0; double worst = 1.0; int checked = 0;
    for (int t = 0; t < (int)G.rows.size(); ++t) {
        int n_kv = G.n_kv_at_step[t];
        for (int slot = 0; slot < (int)taps10.size(); ++slot) {
            for (int h = 0; h < G.n_head; ++h) {
                double s = 0; const float* rr = G.rows[t][slot].data() + (size_t)h * n_kv;
                for (int j = 0; j < n_kv; ++j) s += rr[j];
                checked++;
                if (std::fabs(s - 1.0) > 1e-3) { rows_bad++; if (std::fabs(s - 1.0) > std::fabs(worst - 1.0)) worst = s; }
            }
        }
    }
    int n_sparse_steps = 0; for (auto& g : trace) if (g.sparse) n_sparse_steps++;
    std::printf("  rows checked=%d  bad(|sum-1|>1e-3)=%d  worst_sum=%.6f\n", checked, rows_bad, worst);
    std::printf("  sparse-head steps=%d / %zu decode steps  (rest dense)\n",
                n_sparse_steps, trace.size());
    std::printf("  => rows %s under sparse head\n", rows_bad == 0 ? "MATERIALIZE & sum to 1.0 (PASS)"
                                                                  : "CORRUPT (KILL)");

    // Q2 — signals match constrained vs unconstrained.
    std::printf("\n── Q2: signals constrained vs unconstrained ──\n");
    FreeRun U = run_freegen(fp, sched, tok, meta, body, instr, taps10, 220, false, '}');
    std::printf("[unconstrained] gen (%zu tok): %s\n", U.gen_tokens.size(), U.gen_text.c_str());

    QCite cg = qdocs_cite(tok, G, values, TOL);
    QCite cu = qdocs_cite(tok, U, values, TOL);
    auto pct = [](int a, int b) { return b ? 100.0 * a / b : 0.0; };
    std::printf("\n  citation (frozen L3H13) on value tokens:\n");
    std::printf("    constrained : top1 %d/%d (%.0f%%)  top3 %d/%d (%.0f%%)\n",
                cg.top1, cg.n, pct(cg.top1, cg.n), cg.top3, cg.n, pct(cg.top3, cg.n));
    std::printf("    unconstr.   : top1 %d/%d (%.0f%%)  top3 %d/%d (%.0f%%)\n",
                cu.top1, cu.n, pct(cu.top1, cu.n), cu.top3, cu.n, pct(cu.top3, cu.n));
    double d_top3 = pct(cg.top3, cg.n) - pct(cu.top3, cu.n);
    std::printf("    Δtop3 = %+.0f points  (bar: within 5)  => %s\n", d_top3,
                std::fabs(d_top3) <= 5.0 ? "MATCH" : "DIVERGE");

    std::printf("\n  coverage (frozen layer-11 max-heads peak, thr 0.705):\n");
    auto band = [](double p) { return p < 0 ? "n/a" : (p >= 0.705 ? "USED" : "DROPPED/FILLER"); };
    bool bands_ok = true;
    for (const std::string& v : values) {
        double pg = qdocs_cov_peak(G, v, tok, attn_layers);
        double pu = qdocs_cov_peak(U, v, tok, attn_layers);
        bool same = (pg >= 0.705) == (pu >= 0.705);
        if (!same) bands_ok = false;
        std::printf("    %-20s constrained %.3f [%s]  unconstr %.3f [%s]  %s\n",
                    v.c_str(), pg, band(pg), pu, band(pu), same ? "" : "<< BAND FLIP");
    }
    {
        double pg = qdocs_cov_peak(G, filler, tok, attn_layers);
        double pu = qdocs_cov_peak(U, filler, tok, attn_layers);
        std::printf("    %-20s constrained %.3f [%s]  unconstr %.3f [%s]  (FILLER anchor)\n",
                    filler.c_str(), pg, band(pg), pu, band(pu));
    }
    std::printf("    => coverage band assignments %s\n", bands_ok ? "UNCHANGED (PASS)" : "CHANGED (FAIL)");

    // Eyeball: 3 value tokens under the grammar → frozen L3H13 top-3 attended.
    std::printf("\n  === eyeball (constrained, frozen L3H13): value token → top-3 attended ===\n");
    {
        std::vector<size_t> gcum = cum_bytes(tok, G.gen_tokens);
        std::vector<int32_t> seq = G.prompt_tokens;
        seq.insert(seq.end(), G.gen_tokens.begin(), G.gen_tokens.end());
        int shown = 0;
        for (const std::string& v : {std::string("Globex Industries"),
                                     std::string("2025-11-02"), std::string("47.30")}) {
            size_t gb = G.gen_text.find(v); if (gb == std::string::npos) continue;
            for (int g = 0; g < (int)G.gen_tokens.size(); ++g) {
                if (!(gcum[g] < gb + v.size() && gcum[g + 1] > gb)) continue;
                if (g < 1 || g - 1 >= (int)G.rows.size()) break;
                auto tk = topk_head(G.rows[g - 1][L3H13_SLOT], FROZEN_HEAD, G.n_kv_at_step[g - 1], 3);
                std::string vt = tok->decode(G.gen_tokens[g]);
                for (char& c : vt) if (c == '\n') c = ' ';
                std::printf("  «%s» (value «%s»)\n", vt.c_str(), v.c_str());
                for (int r = 0; r < (int)tk.size(); ++r)
                    std::printf("    #%d pos %3d m%.3f | %s\n", r + 1, tk[r].first, tk[r].second,
                                ctx_around(tok, seq, tk[r].first, 3).c_str());
                break;
            }
            if (++shown >= 3) break;
        }
    }

    // Q3 — elision-hole analysis: which token classes collapse to one legal token?
    std::printf("\n── Q3: forced-token-elision holes ──\n");
    auto is_structural = [](const std::string& s) {
        for (char c : s) if (c != '{' && c != '}' && c != '"' && c != ':' &&
                             c != ',' && c != ' ' && c != '\n' && c != '\t') return false;
        return !s.empty();
    };
    int n_forced = 0, n_forced_value = 0;
    std::printf("  forced (|valid|==1) steps — the elision candidates (no attention row):\n");
    for (int t = 0; t < (int)trace.size(); ++t) {
        if (!trace[t].forced) continue;
        n_forced++;
        bool structural = is_structural(trace[t].forced_tok);
        if (!structural) n_forced_value++;
        std::string vis = trace[t].forced_tok;
        for (char& c : vis) if (c == '\n') c = ' ';
        std::printf("    step %-3d forced=\"%s\" %s\n", t, vis.c_str(),
                    structural ? "[structural]" : "[NON-STRUCTURAL — audit hole!]");
    }
    std::printf("  forced steps=%d  non-structural(value/key)=%d\n", n_forced, n_forced_value);
    std::printf("  => %s\n", n_forced_value == 0
                ? "only structural tokens elide — NO value/key elided (PASS)"
                : "a VALUE/KEY token is elidable — audit hole reported");

    // ── Verdict ──
    std::printf("\n── LEG A VERDICT ──\n");
    bool kill = (rows_bad != 0);
    std::printf("  Q1 rows: %s | Q2 Δtop3=%+.0f, bands %s | Q3 non-struct elisions=%d\n",
                rows_bad == 0 ? "OK" : "CORRUPT", d_top3,
                bands_ok ? "unchanged" : "changed", n_forced_value);
    if (kill) std::printf("  => KILL (rows missing/corrupt under sparse head) — STOP\n");
    else if (std::fabs(d_top3) <= 5.0 && bands_ok)
        std::printf("  => PASS — grammar path preserves the tap and the frozen signals\n");
    else
        std::printf("  => PASS-with-caveat — rows intact; signal drift noted above\n");
    return 0;
}

// ── S1 — STAGE 1 GATE: does the trust layer still tell the truth free-form? ──
// The fixed KV grammar is refuted by measurement on extraction quality
// (docs/note-nogrammar-refutation.md). Before removing it, ONE load-bearing
// question remains: the lens tap math parses the emitted JSON to locate value
// spans, and it has only ever been measured over grammar-shaped output. If
// free-form output breaks the receipts, the grammar has found a real
// justification and Stage 2 is off (docs/handoff-nogrammar-stages.md §S1.2).
//
// Both arms drive the SHIPPED run_lens_extract + compute_lens_report over the
// canonical Leg C corpus, differing ONLY by the grammar pointer (nullptr =
// free). Hint = each document's own labelled (truly present) concepts PLUS two
// concepts verified absent from all 15, exactly as the refutation measured.
//
// THE GATE (re-specified 2026-07-17 — see below for why the original bar could
// not discriminate). It splits TRUTHFULNESS from USEFULNESS:
//
//  - TRUTHFULNESS ("the lens never lies about where the model looked") is gated
//    at ZERO by the DETERMINISTIC LensFidelityGate in tests/unit/test_server_lens.cpp
//    — model-free, synthesized rows, and it has teeth (FiresOnConfidentMisattribution).
//    That is the real instrument; a live corpus cannot gate a by-construction property.
//  - USEFULNESS is what this live probe gates:
//      (a) free-arm top-3 in-span == 100% — the value's own source appears in the
//          reported citation SET (the format ships citations[], not one citation), and
//      (b) LIKE-FOR-LIKE: on fields where BOTH arms emit the SAME value, free's
//          top-1 is never worse than constrained's.
//
// Why not the original "zero top-1 false receipts, free-form"? Measured, it
// rejects the SHIPPED constrained path too (1), so it discriminates nothing:
//  1. Every free-arm miss is ONE characterised class — a `customer` value cited to
//     the sender's email DOMAIN (customer="Harbour Point Marine" → "point" inside
//     sam@harbourpoint.example). The model really did read the domain to get the
//     company name, so the receipt is FAITHFUL; it just is not the verbatim body
//     occurrence. Top-3 contains that occurrence in every case.
//  2. The constrained arm shows the IDENTICAL case on m_en6 (mass 0.216 vs 0.213)
//     ⇒ the phenomenon is grammar-independent. It is the model, not the constraint.
//  3. The bar REWARDS COLLAPSE: under the grammar m_de7 collapses to 67 junk `", "`
//     fields, which are not verbatim in the document and are therefore excluded —
//     a DESTROYED document scores zero false receipts. Absolute counts across
//     different claim sets are confounded; (b) fixes that by construction.
// (plan-qemmi-lens.md §5 scoped A5.4's zero to the FIXTURE corpus all along; the
// handoff extended it to a live corpus without measuring that the claim held.)
//
//   QDOCS_S1=1 ./bin/attn-provenance
static int run_qdocs_s1(ForwardPassBase* fp, ggml_backend_sched_t sched,
                        Tokenizer* tok, const ModelMetadata& meta, uint32_t n_ctx) {
    std::printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║ STAGE 1 GATE — the trust layer over FREE-FORM output (15 docs)║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    const int TOL = 2;
    // Verified absent from all 15 Leg C documents — any value for either is a
    // fabrication (mirrors py/lens_legc_nogrammar.py ABSENT).
    const std::vector<std::string> ABSENT = {"payment_terms", "warranty_period"};

    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const uint32_t vocab_size = (uint32_t)vocab.size();
    qinf::TokenTrie trie; trie.build(vocab);
    auto gr = qinf::GrammarVocab::parse_impl(qinf::lens_grammar_gbnf());
    gr->set_token_trie(&trie);

    auto corpus = qdocs_messy_corpus();
    std::printf("corpus=%zu docs  eos_id=%d  ctx=%u\n",
                corpus.size(), (int)tok->get_eos_token_id(), n_ctx);

    // One emitted field, kept per doc/key so the two arms can be compared
    // LIKE-FOR-LIKE (same key, same value) instead of by confounded totals.
    // top1_mass vs in_span_mass is the MARGIN: how decisively the out-of-span
    // citation beat the value's own source. A hair-thin margin means the two
    // arms flipped a near-tie, not that the grammar bought anything.
    struct FieldRec {
        std::string value;
        bool   checked = false, top1 = false, top3 = false;
        double top1_mass = 0.0;     // mass of citations[0]
        double in_span_mass = 0.0;  // mass of the best citation that IS in-span (0 = none)
        int    in_span_rank = -1;   // its rank in citations[] (0-based)
    };
    struct Arm {
        int docs = 0, fields_located = 0;
        int receipts = 0, top1 = 0, top3 = 0, false_receipts = 0;
        int fid_ok = 0, fid_tot = 0, absent_ok = 0, absent_tot = 0;
        std::vector<std::string> fabricated, empty_docs, false_docs, unparseable;
        std::vector<std::map<std::string, FieldRec>> per_doc;  // [doc index][key]
    };

    // Does any of the top-n citations land on a real source of the value?
    auto cited_in_span = [&](const qinf::LensField& f, const std::string& doc, int topn) {
        for (int i = 0; i < (int)f.citations.size() && i < topn; ++i)
            if (qinf::lens_cites_a_real_source(doc, f.value, f.citations[i].byte_lo,
                                                  f.citations[i].byte_hi, TOL))
                return true;
        return false;
    };
    auto norm = [](std::string s) {
        std::string o; bool sp = false;
        for (char c : s) {
            if (std::isspace((unsigned char)c)) { sp = !o.empty(); continue; }
            if (sp) { o += ' '; sp = false; }
            o += (char)std::tolower((unsigned char)c);
        }
        return o;
    };
    // A DECLINE is the model saying "not stated". `","` is NOT a decline — it is
    // the collapse (m_de7 emits it for every key, present ones included), i.e.
    // garbage. Counting it as declined FLATTERS the grammar's absent score. The
    // list matches py/lens_legc_nogrammar.py exactly: the two probes must never
    // score the same event differently.
    auto declined = [&](const std::string& v) {
        const std::string n = norm(v);
        return n.empty() || n == "null" || n == "none" || n == "n/a" || n == "-";
    };

    auto run_arm = [&](const char* tag, qinf::GrammarVocab* grammar) -> Arm {
        Arm A;
        std::printf("\n%s\n ARM: %s\n%s\n", std::string(72, '=').c_str(),
                    grammar ? "FIXED KV GRAMMAR" : "NO GRAMMAR (free decode)",
                    std::string(72, '=').c_str());
        for (const QMessy& d : corpus) {
            std::vector<qinf::LensConcept> concepts;
            for (const QLabel& f : d.fields) concepts.push_back({f.concept, ""});
            for (const std::string& a : ABSENT) concepts.push_back({a, ""});

            qinf::LensExtractOptions opts;
            opts.max_new_tokens = 400;
            // `grammar` is the CONTROL ARM seam (null ⇒ the shipped free path).
            // Both arms run the same driver, which is the whole point: the grammar
            // is the only variable, and the free arm under test is product code.
            qinf::LensReport r;
            try {
                r = qinf::run_lens_extract(
                    fp, sched, tok, meta, vocab_size, n_ctx, d.document, concepts,
                    opts, qinf::LensConstants{}, grammar, grammar ? &vocab : nullptr);
            } catch (const qinf::LensUnparseableError& e) {
                // The shape contract firing. Report it as the loud refusal it is —
                // do NOT skip the doc silently, that would flatter the arm.
                A.docs++;
                A.unparseable.push_back(d.tag);
                A.fid_tot += (int)d.fields.size();
                A.absent_tot += (int)ABSENT.size();
                A.per_doc.emplace_back();
                std::printf("\n[%s%s] 422 UNPARSEABLE — extraction refused\n  %.140s\n",
                            d.tag.c_str(), d.de ? " DE" : "", e.raw.c_str());
                continue;
            }

            A.docs++;
            A.fields_located += (int)r.fields.size();
            if (r.fields.empty()) A.empty_docs.push_back(d.tag);

            const int fr = qinf::lens_count_confident_false_receipts(r, TOL);
            A.false_receipts += fr;
            if (fr) A.false_docs.push_back(d.tag);

            // Header first — the diagnostics below belong to THIS document.
            std::printf("\n[%s%s] fields=%zu false_receipts=%d\n  %s\n",
                        d.tag.c_str(), d.de ? " DE" : "", r.fields.size(), fr,
                        r.raw_json.substr(0, 160).c_str());

            // A gate that fails must say WHY, or it cannot be acted on. For each
            // false receipt print the value, where the receipt points, and where
            // the value actually occurs — enough to tell a LIE (the lens claims a
            // source the model never read) from a DEFINITION artifact (the model
            // truly read that span, but it sits outside this value's occurrence).
            for (const auto& f : r.fields) {
                if (!f.grounded || !f.found_in_document) continue;
                if (qinf::lens_value_tier(f.value) != "distinctive") continue;
                if (f.citations.empty()) { std::printf("    !! %s: NO CITATIONS\n", f.key.c_str()); continue; }
                if (cited_in_span(f, r.document_text, 1)) continue;
                const auto& c = f.citations[0];
                std::string at = c.byte_lo < r.document_text.size()
                    ? r.document_text.substr(c.byte_lo, std::min<size_t>(c.byte_hi - c.byte_lo, 24))
                    : std::string("<out of doc>");
                std::printf("    !! FALSE RECEIPT %s=\"%s\"\n", f.key.c_str(), f.value.c_str());
                std::printf("       top-1 cites [%zu,%zu) = \"%s\" (mass %.3f)\n",
                            c.byte_lo, c.byte_hi, at.c_str(), c.mass);
                std::printf("       value occurs at:");
                for (size_t o = r.document_text.find(f.value); o != std::string::npos;
                     o = r.document_text.find(f.value, o + 1))
                    std::printf(" [%zu,%zu)", o, o + f.value.size());
                std::printf("\n       top-3:");
                for (int i = 0; i < (int)f.citations.size() && i < 3; ++i) {
                    const auto& ci = f.citations[i];
                    std::string t = ci.byte_lo < r.document_text.size()
                        ? r.document_text.substr(ci.byte_lo, std::min<size_t>(ci.byte_hi - ci.byte_lo, 16))
                        : std::string("?");
                    std::printf("  [%zu)=\"%s\"", ci.byte_lo, t.c_str());
                }
                std::printf("\n");
            }

            // In-span citation rate over the fields the format makes a claim for.
            A.per_doc.emplace_back();
            for (const auto& f : r.fields) {
                FieldRec rec; rec.value = f.value;
                if (f.grounded && f.found_in_document &&
                    qinf::lens_value_tier(f.value) == "distinctive") {
                    rec.checked = true;
                    rec.top1 = cited_in_span(f, r.document_text, 1);
                    rec.top3 = cited_in_span(f, r.document_text, 3);
                    if (!f.citations.empty()) rec.top1_mass = f.citations[0].mass;
                    for (int i = 0; i < (int)f.citations.size(); ++i)
                        if (qinf::lens_cites_a_real_source(r.document_text, f.value,
                                f.citations[i].byte_lo, f.citations[i].byte_hi, TOL)) {
                            rec.in_span_mass = f.citations[i].mass;
                            rec.in_span_rank = i;
                            break;  // citations are mass-sorted: first hit is the best
                        }
                    A.receipts++;
                    A.top1 += rec.top1;
                    A.top3 += rec.top3;
                }
                A.per_doc.back().emplace(f.key, rec);  // first wins (a collapse repeats keys)
            }
            // Fidelity vs the corpus labels; absent handling on the two planted keys.
            auto value_of = [&](const std::string& key) -> const qinf::LensField* {
                for (const auto& f : r.fields) if (f.key == key) return &f;
                return nullptr;
            };
            for (const QLabel& lf : d.fields) {
                A.fid_tot++;
                const qinf::LensField* got = value_of(lf.concept);
                if (!got) continue;
                const std::string t = norm(lf.value), g = norm(got->value);
                if (!g.empty() && (t.find(g) != std::string::npos || g.find(t) != std::string::npos))
                    A.fid_ok++;
            }
            for (const std::string& key : ABSENT) {
                A.absent_tot++;
                const qinf::LensField* got = value_of(key);
                if (!got || declined(got->value)) A.absent_ok++;
                else A.fabricated.push_back(d.tag + ":" + key + "=\"" + got->value + "\"");
            }
        }
        return A;
    };

    Arm ng = run_arm("NG", nullptr);
    Arm gp = run_arm("GR", gr.get());

    auto pct = [](int a, int b) { return b ? 100.0 * a / b : 0.0; };
    auto report = [&](const char* name, const Arm& A) {
        std::printf("\n  -- %s --\n", name);
        // NOT the gate (see the header): counts the customer→sender-domain class,
        // where the lens is truthful and top-3 finds the body mention. Reported
        // because it is the number the original, mis-specified bar looked at.
        std::printf("     confident false receipts  %d        <- reported, NOT gated\n",
                    A.false_receipts);
        std::printf("     in-span citation  top-1   %d/%d (%.0f%%)   top-3 %d/%d (%.0f%%)\n",
                    A.top1, A.receipts, pct(A.top1, A.receipts),
                    A.top3, A.receipts, pct(A.top3, A.receipts));
        std::printf("     fields located            %d\n", A.fields_located);
        std::printf("     value fidelity            %d/%d\n", A.fid_ok, A.fid_tot);
        std::printf("     absent handled            %d/%d\n", A.absent_ok, A.absent_tot);
        if (!A.fabricated.empty()) {
            std::printf("     FABRICATED (%zu):\n", A.fabricated.size());
            for (size_t i = 0; i < A.fabricated.size() && i < 6; ++i)
                std::printf("       %s\n", A.fabricated[i].c_str());
        }
        if (!A.empty_docs.empty()) {
            std::printf("     NO FIELDS LOCATED:");
            for (const auto& t : A.empty_docs) std::printf(" %s", t.c_str());
            std::printf("\n");
        }
        if (!A.false_docs.empty()) {
            std::printf("     FALSE-RECEIPT DOCS:");
            for (const auto& t : A.false_docs) std::printf(" %s", t.c_str());
            std::printf("\n");
        }
        if (!A.unparseable.empty()) {
            std::printf("     422 UNPARSEABLE (refused, not partially reported):");
            for (const auto& t : A.unparseable) std::printf(" %s", t.c_str());
            std::printf("\n");
        }
    };
    report("NO GRAMMAR (free)", ng);
    report("FIXED KV GRAMMAR", gp);

    std::printf("\n%s\n", std::string(72, '=').c_str());
    std::printf("%-26s%14s%14s\n", "axis", "no grammar", "grammar");
    std::printf("%-26s%14d%14d\n", "confident false receipts", ng.false_receipts, gp.false_receipts);
    std::printf("%-26s%13.0f%%%13.0f%%\n", "in-span top-1", pct(ng.top1, ng.receipts), pct(gp.top1, gp.receipts));
    std::printf("%-26s%13.0f%%%13.0f%%\n", "in-span top-3", pct(ng.top3, ng.receipts), pct(gp.top3, gp.receipts));
    std::printf("%-26s%14d%14d\n", "fields located", ng.fields_located, gp.fields_located);
    std::printf("%-26s%9d/%-4d%9d/%-4d\n", "value fidelity", ng.fid_ok, ng.fid_tot, gp.fid_ok, gp.fid_tot);
    std::printf("%-26s%9d/%-4d%9d/%-4d\n", "absent handled", ng.absent_ok, ng.absent_tot, gp.absent_ok, gp.absent_tot);

    // ── LIKE-FOR-LIKE: only fields where both arms emit the SAME value ────────
    // Absolute totals are confounded — the arms make different claims on
    // different values, and a collapsed doc silently drops out of the denominator.
    // Comparing identical (key, value) pairs removes both effects.
    // (b) is RANK-FREE: it asks the question the rank flip only proxied — did
    // removing the grammar make the value's OWN source less attended? Rank order
    // between two genuine sources turns on hundredths of mass, and handoff §4 is
    // explicit: Metal is token-stable-not-byte-identical, gate on structure and
    // rates, NOT exact masses. A rank-flip gate resolves exactly the tie-breaks
    // that invariant forbids. Mass retention does not.
    //
    // Teeth: if free-form broke the value-span location (the S1.2 fear), in-span
    // mass COLLAPSES toward zero — a −100% move, not a wobble.
    //
    // MEASURED (2026-07-17): the per-pair cross-arm spread on IDENTICAL values is
    // min −31.3% / mean −1.0% / max +10.8% over 42 pairs — so no useful per-pair
    // tolerance fits under the noise, and a threshold tuned to pass would be
    // circular. The spread is also partly an artifact: citations[].mass is
    // normalized by the value's token count, and the arms tokenize the same value
    // differently (`": "` vs `":"`), so per-pair masses are not strictly
    // commensurable across arms. Hence the criterion is applied to the MEAN — a
    // rate, which is what handoff §4 permits — with the distribution always
    // printed so a real collapse (−100%) is unmissable, plus a constant-free
    // structural check (every like-for-like field must cite a real source at all).
    const double MEAN_TOL = 0.10;   // mean retention floor; measured mean is −1.0%
    const double REPORT_AT = 0.20;  // per-pair outliers listed for inspection, NOT gated
    int lfl = 0, outliers = 0, no_source = 0;
    double rel_min = 1e9, rel_max = -1e9, rel_sum = 0.0; int rel_n = 0;
    std::vector<std::string> detail;
    for (size_t i = 0; i < corpus.size() && i < ng.per_doc.size() && i < gp.per_doc.size(); ++i) {
        for (const auto& kv : ng.per_doc[i]) {
            auto it = gp.per_doc[i].find(kv.first);
            if (it == gp.per_doc[i].end()) continue;
            const FieldRec& nf = kv.second;
            const FieldRec& gf = it->second;
            if (!nf.checked || !gf.checked || nf.value != gf.value) continue;
            lfl++;
            // Structural, constant-free: the value's own source must be cited AT ALL.
            if (nf.in_span_rank < 0) {
                no_source++;
                char b[256];
                std::snprintf(b, sizeof(b), "\n     %s:%s = \"%s\" — free cites NO real source",
                              corpus[i].tag.c_str(), kv.first.c_str(), nf.value.c_str());
                detail.push_back(b);
                continue;
            }
            if (gf.in_span_mass <= 0.0) continue;  // no grammar baseline to compare against
            const double rel = (nf.in_span_mass - gf.in_span_mass) / gf.in_span_mass;
            rel_min = std::min(rel_min, rel); rel_max = std::max(rel_max, rel);
            rel_sum += rel; rel_n++;
            if (rel < -REPORT_AT) {
                outliers++;  // reported, not gated — see MEAN_TOL above
                char b[384];
                std::snprintf(b, sizeof(b),
                    "\n     %s:%s = \"%s\" — in-span mass free %.4f vs grammar %.4f (%+.1f%%)",
                    corpus[i].tag.c_str(), kv.first.c_str(), nf.value.c_str(),
                    nf.in_span_mass, gf.in_span_mass, 100.0 * rel);
                detail.push_back(b);
            }
        }
    }
    std::printf("\nlike-for-like (same key AND same value, both checked): %d pairs\n", lfl);
    std::printf("   in-span MASS retention, free vs grammar (rank-free):\n");
    std::printf("     min %+.1f%%   mean %+.1f%%   max %+.1f%%   over %d pairs\n",
                100.0 * rel_min, 100.0 * (rel_n ? rel_sum / rel_n : 0.0), 100.0 * rel_max, rel_n);
    std::printf("     GATED on the mean (floor -%.0f%%); a broken value-span location reads -100%%\n",
                100.0 * MEAN_TOL);
    std::printf("   free cites no real source: %d (gated)  |  per-pair outliers below -%.0f%%: %d (reported only)",
                no_source, 100.0 * REPORT_AT, outliers);
    for (const auto& s : detail) std::printf("%s", s.c_str());
    std::printf("\n");

    // Stage 1 exit criteria, re-specified 2026-07-17 (see the header block).
    const double rel_mean = rel_n ? rel_sum / rel_n : 0.0;
    const bool gate_top3 = (ng.receipts > 0) && (ng.top3 == ng.receipts);
    const bool gate_lfl  = (lfl > 0) && (no_source == 0) && (rel_mean >= -MEAN_TOL);
    std::printf("\nSTAGE 1 GATE (a): free top-3 in-span == 100%%       : %s (%d/%d)\n",
                gate_top3 ? "PASS" : "FAIL", ng.top3, ng.receipts);
    std::printf("STAGE 1 GATE (b): free retains in-span mass       : %s (%d pairs, mean %+.1f%%, %d unsourced)\n",
                gate_lfl ? "PASS" : "FAIL", lfl, 100.0 * rel_mean, no_source);
    std::printf("\nFYI (not a gate): top-1 absolute %.0f%% free vs %.0f%% grammar — confounded,\n"
                "  the arms claim different values and a collapsed doc scores no false receipts.\n",
                pct(ng.top1, ng.receipts), pct(gp.top1, gp.receipts));
    std::printf("\n%s\n", (gate_top3 && gate_lfl)
                ? "STAGE 1 (S1.2) GREEN — free-form does not degrade the receipts."
                : "STAGE 1 (S1.2) RED — the grammar may have a real justification; record it.");
    return (gate_top3 && gate_lfl) ? 0 : 1;
}

int main() {
    const char* env = std::getenv("QWEN36_MODEL_PATH");
    std::string path = env ? env : "models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf";
    const bool selftest = std::getenv("ATTN_TAP_SELFTEST") != nullptr;
    // 2048 for the short-prompt probes; leg D needs ≥4K + generation margin.
    // QDOCS_S1 also takes the margin: free decode has no accepting state to
    // close on, so a run can spend its whole 400-token budget before EOS.
    // SS2 threads target up to 8K prompt tokens (the workload-envelope ceiling,
    // CLAUDE.md) plus a 380-token grammar-decode margin.
    // KV *capacity* only — decode uses exact n_kv, so prior paths are byte-inert.
    const uint32_t CTX = std::getenv("SS2") ? 9216
                        : (std::getenv("QDOCS_D") || std::getenv("QDOCS_S1")) ? 5120 : 2048;
    const int TOL = 2;

    apply_frozen_head_overrides();
    register_builtin_models();
    std::cerr << "Loading " << path << " ...\n";
    Model model;
    model.load_metadata(path);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    auto fp = create_forward_pass(model, &meta, CTX, 1);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();

    // ── Discover attention layers from the BUILT DECODE GRAPH ────────────────
    // Was: `meta.raw_kv.get_uint32("qwen35moe.full_attention_interval")`, a key
    // only a qwen35moe GGUF carries — so this probe could not even load any
    // other architecture, and was structurally incapable of asking whether the
    // lens constants transfer. Same defect, same fix as the P1 gate
    // (tests/unit/test_forward_pass_base.cpp): the seam is the TENSOR NAME
    // `kq_soft.<il>` that layers/attention.cpp assigns inside build_attn_mha,
    // the single funnel every attention builder passes through. Scanning for it
    // needs no per-family knowledge and automatically excludes blocks held out
    // of the decode stack (a qwen35 GGUF's trailing NextN/MTP head builds no
    // nodes). On qwen36 this yields exactly the old fai-derived list.
    std::vector<int32_t> attn_layers;
    {
        std::vector<int32_t> warm = tok->encode("Hello");
        if (warm.empty()) {
            std::fprintf(stderr, "tap discovery: expected a non-empty warm-up "
                                 "tokenization, actual 0 tokens\n");
            return 1;
        }
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
        fp->run_prefill(warm, 0, 0, sched);
        std::vector<int32_t>  t = {warm.back()};
        std::vector<uint32_t> s = {0};
        std::vector<int32_t>  p = {(int32_t)fp->get_cache_pos(0)};
        ggml_cgraph* gscan = fp->build_decoding_graph(t, s, p);
        for (uint32_t il = 0; il < meta.block_count; ++il) {
            const std::string nm = "kq_soft." + std::to_string(il);
            if (ggml_graph_get_tensor(gscan, nm.c_str())) attn_layers.push_back((int32_t)il);
        }
        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
    }
    if (attn_layers.empty()) {
        std::fprintf(stderr, "tap discovery: expected ≥1 `kq_soft.<il>` tensor in "
                             "the decode graph, actual 0 (arch '%s' does not "
                             "materialize attention and cannot host the lens tap)\n",
                     meta.architecture.c_str());
        return 1;
    }
    g_attn_layers = &attn_layers;
    g_arch = meta.architecture;
    g_meta = &meta;
    // Honour the model's own add_bos_token contract, as cli/complete.cpp and
    // cli/session_mode.cpp do before prefill. A Gemma -it model without a
    // leading BOS degenerates (repeats a single token), so every Gemma number
    // measured without this was taken on a broken state. Qwen GGUFs carry
    // add_bos_token=false, so g_bos_text stays empty and the qwen legs are
    // byte-identical.
    if (meta.add_bos_token && meta.bos_token_id >= 0) {
        g_bos_text = tok->decode({(int32_t)meta.bos_token_id});
        std::printf("prompt BOS: add_bos_token=true id=%d text=\"%s\" (%zu bytes)\n",
                    meta.bos_token_id, g_bos_text.c_str(), g_bos_text.size());
        if (g_bos_text.empty()) {
            std::fprintf(stderr, "prompt BOS: expected a non-empty decoding for "
                                 "bos_token_id %d, actual empty string — the "
                                 "text-level BOS scheme cannot represent it\n",
                         meta.bos_token_id);
            return 1;
        }
    }

    // Validate the slot overrides against the model actually loaded: a stale
    // ATTN_FROZEN_SLOT from a deeper model would otherwise index past the tap
    // vector and read garbage that still looks like a probability row.
    for (auto& sv : {std::make_pair("ATTN_FROZEN_SLOT/ATTN_COV_SLOT citation", FROZEN_SLOT),
                     std::make_pair("ATTN_COV_SLOT coverage", L11_SLOT)}) {
        if (sv.second < 0 || sv.second >= (int)attn_layers.size()) {
            std::fprintf(stderr, "%s: expected a tap slot in [0,%zu) for arch '%s' "
                                 "(%zu attention layers), actual %d\n",
                         sv.first, attn_layers.size(), meta.architecture.c_str(),
                         attn_layers.size(), sv.second);
            return 1;
        }
    }

    std::printf("\nmodel map: arch=%s block_count=%u\n",
                meta.architecture.c_str(), meta.block_count);
    std::printf("attention layers (%zu): ", attn_layers.size());
    for (int il : attn_layers) std::printf("%d ", il);
    std::printf("\nn_head_q=%u n_head_kv=%u\n\n",
                meta.attention_head_count, meta.attention_head_count_kv);

    // N3b — the ungrounded-value alarm (sibling probe, own prompts). Leaves the
    // A/B/C path below untouched.
    if (std::getenv("ATTN_UNGROUNDED"))
        return run_ungrounded_probe(fp.get(), sched, tok, meta, attn_layers);

    // CG1 — the confidence gap (top1-top2), reusing N3b's labeled prompts.
    if (std::getenv("CONF_GAP"))
        return run_confgap_probe(fp.get(), sched, tok, meta, attn_layers);

    // DP1 — the retrieval head as a speculative draft pointer (vs shipped PLD).
    if (std::getenv("DRAFT_POINTER"))
        return run_draft_pointer_probe(fp.get(), sched, tok, meta, attn_layers);

    // COV1 — coverage: does zero attention mass reveal a SKIPPED prompt span?
    if (std::getenv("COVERAGE"))
        return run_coverage_probe(fp.get(), sched, tok, meta, attn_layers);

    // CF1 — conflict flag: does bimodal attention mass detect ≥2 candidate sources?
    if (std::getenv("CONFLICT_FLAG"))
        return run_conflict_probe(fp.get(), sched, tok, meta, attn_layers);

    // SS1 — stale-source alarm: do citations+coverage transfer to multi-turn threads?
    if (std::getenv("STALE_SOURCE"))
        return run_stale_probe(fp.get(), sched, tok, meta, attn_layers);

    // LENS — data generator for the Attention Lens HTML demo (not a probe).
    if (std::getenv("LENS"))
        return run_lens_gen(fp.get(), sched, tok, meta, attn_layers);

    // Qemmi-Docs P0 — leg A: fixed KV grammar × attention tap sanity (the gate).
    if (std::getenv("QDOCS_A"))
        return run_qdocs_leg_a(fp.get(), sched, tok, meta, attn_layers);
    // Qemmi-Docs P0 — leg B: key-name stability (2 arms × 12 EN+DE docs).
    if (std::getenv("QDOCS_B"))
        return run_qdocs_leg_b(fp.get(), sched, tok, meta, attn_layers);
    // Qemmi-Docs P0 — leg C: messy-corpus robustness (frozen signals, 15 docs).
    if (std::getenv("QDOCS_C"))
        return run_qdocs_leg_c(fp.get(), sched, tok, meta, attn_layers);
    // Norm-weighted attention calibration (docs/note-lens-norm-weighted-metric.md):
    // Metric A (raw alpha) vs Metric B (alpha*||V||) on citation (Leg C corpus)
    // and coverage separation (COV1 corpus), paired, same pass.
    if (std::getenv("NORM_WEIGHTED"))
        return run_norm_weighted_probe(fp.get(), sched, tok, meta, attn_layers);
    // GEMMA4 SEARCH — all (layer,head) candidates scored on the Leg C messy
    // corpus directly, Metric A vs Metric B, gemma4-only (the caches this
    // reads have no Qwen equivalent). docs/note-lens-gemma-norm-weighted.md.
    if (std::getenv("GEMMA4_SEARCH_DUAL"))
        return run_gemma4_search_dual(fp.get(), sched, tok, meta, attn_layers);
    // Qemmi-Docs P0 — leg D: context length (1K/2K/4K buckets, real token counts).
    if (std::getenv("QDOCS_D"))
        return run_qdocs_leg_d(fp.get(), sched, tok, meta, attn_layers);
    // Stage 1 gate — the trust layer over FREE-FORM output, on the shipped lens
    // math (docs/handoff-nogrammar-stages.md §S1.2). Gates the grammar removal.
    if (std::getenv("QDOCS_S1"))
        return run_qdocs_s1(fp.get(), sched, tok, meta, CTX);

    // SS2 — coverage-free stale-source alarm on grammar-constrained email
    // threads at 4K-8K tokens (supersedes SS1's inconclusive short-context
    // composition). docs/note-ss2-thread-alarm.md.
    if (std::getenv("SS2"))
        return run_ss2(fp.get(), sched, tok, meta, attn_layers);

    // ── Prompt A (calibration) ───────────────────────────────────────────────
    std::string promptA =
        "From: orders@acme-corp.example\n"
        "Subject: Purchase Order\n\n"
        "Hello,\n\n"
        "Please process the following order for customer Acme Corporation.\n"
        "Order date: 2026-03-14.\n"
        "We need a quantity of 240 units.\n"
        "The unit price is 12.50 USD.\n"
        "The order total comes to 3000.00 USD.\n\n"
        "Thanks.\n\n"
        "Extract the order as JSON with keys customer, date, quantity, unit_price, total:\n";
    std::string compA =
        "{\"customer\": \"Acme Corporation\", \"date\": \"2026-03-14\", "
        "\"quantity\": 240, \"unit_price\": 12.50, \"total\": 3000.00}";
    std::vector<Field> fieldsA = {
        {"customer",   "Acme Corporation", -1, -1},
        {"date",       "2026-03-14",       -1, -1},
        {"quantity",   "240",              -1, -1},
        {"unit_price", "12.50",            -1, -1},
        {"total",      "3000.00",          -1, -1},
    };

    // ── Prompt B (held-out, same shape) ──────────────────────────────────────
    std::string promptB =
        "From: purchasing@globex.example\n"
        "Subject: New Order\n\n"
        "Hi team,\n\n"
        "Kindly arrange an order for customer Globex Industries.\n"
        "Order date: 2025-11-02.\n"
        "We require a quantity of 875 units.\n"
        "The unit price is 47.30 EUR.\n"
        "The order total comes to 41387.50 EUR.\n\n"
        "Regards.\n\n"
        "Extract the order as JSON with keys customer, date, quantity, unit_price, total:\n";
    std::string compB =
        "{\"customer\": \"Globex Industries\", \"date\": \"2025-11-02\", "
        "\"quantity\": 875, \"unit_price\": 47.30, \"total\": 41387.50}";
    std::vector<Field> fieldsB = {
        {"customer",   "Globex Industries", -1, -1},
        {"date",       "2025-11-02",        -1, -1},
        {"quantity",   "875",               -1, -1},
        {"unit_price", "47.30",             -1, -1},
        {"total",      "41387.50",          -1, -1},
    };

    // ── Self-test path: tap sanity, then exit ────────────────────────────────
    if (selftest) {
        std::printf("=== TAP SELF-TEST (2 steps) ===\n");
        run_prompt(fp.get(), sched, tok, meta, "A", promptA, compA, fieldsA, true);
        std::printf("[selftest] OK if dims=[n_kv,1,%u,1] and every head sum≈1.0\n",
                    meta.attention_head_count);
        return 0;
    }

    // ── Run A, select best (layer,head) ──────────────────────────────────────
    PromptRun A = run_prompt(fp.get(), sched, tok, meta, "A", promptA, compA, fieldsA, false);

    auto report_fields = [&](const PromptRun& R) {
        std::printf("[%s] prompt_tokens=%zu comp_tokens=%zu\n",
                    R.label.c_str(), R.prompt_tokens.size(), R.comp_tokens.size());
        for (auto& f : R.fields)
            std::printf("    field %-11s value=\"%s\"  prompt span [%d..%d]  ctx: %s\n",
                        f.name.c_str(), f.value.c_str(), f.lo, f.hi,
                        ctx_around(tok, R.prompt_tokens, f.lo, 3).c_str());
        // count value tokens
        int nv = 0; for (int v = 1; v < (int)R.comp_tokens.size(); ++v)
            if (R.comp_field[v] >= 0) nv++;
        int agree = 0; for (int v = 1; v < (int)R.comp_tokens.size(); ++v)
            if (R.comp_field[v] >= 0 && v - 1 < (int)R.greedy_pred.size()
                && R.greedy_pred[v - 1] == R.comp_tokens[v]) agree++;
        std::printf("    value tokens scored: %d   greedy-agreement on them: %d/%d\n",
                    nv, agree, nv);
    };

    std::printf("\n================ PROMPT A (calibration) ================\n");
    report_fields(A);

    // Per-layer: best head + head-mean top1 fraction.
    std::printf("\n  layer | best_head  top1/N  top3/N | head-MEAN top1  bos_mass\n");
    std::printf("  ------+-------------------------------+--------------------------\n");
    int    best_ls = -1, best_h = -1;
    double best_frac = -1; int best_top3 = -1;
    for (int ls = 0; ls < (int)attn_layers.size(); ++ls) {
        int    lbest_h = -1; double lbest_frac = -1; int lbest_t1 = 0, lbest_t3 = 0, lbest_n = 0;
        double mean_t1 = 0, mean_bos = 0;
        for (int h = 0; h < A.n_head; ++h) {
            Score sc = score_lh(A, ls, h, TOL);
            double frac = sc.n ? (double)sc.top1 / sc.n : 0;
            mean_t1  += frac;
            mean_bos += sc.n ? sc.bos_mass / sc.n : 0;
            if (frac > lbest_frac) { lbest_frac = frac; lbest_h = h;
                                     lbest_t1 = sc.top1; lbest_t3 = sc.top3; lbest_n = sc.n; }
            // global best: prefer top1 fraction, tie-break top3
            if (frac > best_frac ||
                (frac == best_frac && sc.top3 > best_top3)) {
                best_frac = frac; best_top3 = sc.top3; best_ls = ls; best_h = h;
            }
        }
        mean_t1 /= A.n_head; mean_bos /= A.n_head;
        std::printf("  %5d | h=%-2d      %2d/%-2d   %2d/%-2d  |   %.2f          %.3f\n",
                    attn_layers[ls], lbest_h, lbest_t1, lbest_n, lbest_t3, lbest_n,
                    mean_t1, mean_bos);
    }

    // ── Full ranked (layer,head) table ───────────────────────────────────────
    // The per-layer summary above shows only each layer's winner, which hides
    // the shape of the field: whether ONE head stands out (a retrieval head) or
    // the top of the ranking is a flat crowd of near-ties (no head, just noise
    // with a lucky argmax). That distinction is the whole go/no-go question when
    // asking whether the lens constants transfer to another model, so print the
    // ranking itself rather than its argmax.
    {
        struct Cand { int layer, head, top1, top3, n; double frac, bos; };
        std::vector<Cand> cands;
        for (int ls = 0; ls < (int)attn_layers.size(); ++ls)
            for (int h = 0; h < A.n_head; ++h) {
                Score sc = score_lh(A, ls, h, TOL);
                cands.push_back({attn_layers[ls], h, sc.top1, sc.top3, sc.n,
                                 sc.n ? (double)sc.top1 / sc.n : 0.0,
                                 sc.n ? sc.bos_mass / sc.n : 0.0});
            }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.frac != b.frac) return a.frac > b.frac;
            return a.top3 > b.top3;
        });
        const int SHOW = 20;
        std::printf("\n  === RANKED (layer,head) on A — top %d of %zu candidates ===\n",
                    SHOW, cands.size());
        std::printf("  rank | layer head | top1/N  (in-span)  top3/N | bos_mass\n");
        std::printf("  -----+------------+--------------------------+---------\n");
        for (int i = 0; i < SHOW && i < (int)cands.size(); ++i) {
            const Cand& c = cands[i];
            std::printf("  %4d | L%-4d H%-3d | %2d/%-2d   (%5.1f%%)   %2d/%-2d | %.3f\n",
                        i + 1, c.layer, c.head, c.top1, c.n, 100.0 * c.frac,
                        c.top3, c.n, c.bos);
        }
        // The Qwen 3.6 frozen coordinates, scored HERE as a control — on the
        // pinned model this must be at/near the top; on any other model it is
        // the direct measurement of whether the shipped constants transfer.
        for (int ls = 0; ls < (int)attn_layers.size(); ++ls) {
            if (attn_layers[ls] != 3) continue;
            if (13 >= A.n_head) break;
            Score sc = score_lh(A, ls, 13, TOL);
            int rank = 1;
            for (const Cand& c : cands)
                if (c.frac > (sc.n ? (double)sc.top1 / sc.n : 0.0)) rank++;
            std::printf("  CONTROL L3H13 (the shipped LensConstants coordinates): "
                        "top1 %d/%d (%.1f%%), top3 %d/%d — rank %d of %zu\n",
                        sc.top1, sc.n, sc.n ? 100.0 * sc.top1 / sc.n : 0.0,
                        sc.top3, sc.n, rank, cands.size());
        }
    }

    Score bA = score_lh(A, best_ls, best_h, TOL);
    std::printf("\n  >>> SELECTED on A: layer %d, head %d  ->  top1 %d/%d (%.0f%%), top3 %d/%d (%.0f%%)\n",
                attn_layers[best_ls], best_h,
                bA.top1, bA.n, 100.0 * bA.top1 / bA.n,
                bA.top3, bA.n, 100.0 * bA.top3 / bA.n);

    // ── Run B, frozen eval ───────────────────────────────────────────────────
    PromptRun B = run_prompt(fp.get(), sched, tok, meta, "B", promptB, compB, fieldsB, false);
    std::printf("\n================ PROMPT B (held-out, FROZEN layer %d head %d) ================\n",
                attn_layers[best_ls], best_h);
    report_fields(B);
    Score bB = score_lh(B, best_ls, best_h, TOL);
    std::printf("\n  FROZEN eval on B: top1 %d/%d (%.0f%%), top3 %d/%d (%.0f%%), bos_mass %.3f\n",
                bB.top1, bB.n, 100.0 * bB.top1 / bB.n,
                bB.top3, bB.n, 100.0 * bB.top3 / bB.n,
                bB.n ? bB.bos_mass / bB.n : 0);

    // Per-field breakdown on B for the frozen head.
    std::printf("\n  per-field (B, frozen head):\n");
    for (int fi = 0; fi < (int)B.fields.size(); ++fi) {
        int n = 0, t1 = 0, t3 = 0;
        for (int v = 1; v < (int)B.comp_tokens.size(); ++v) {
            if (B.comp_field[v] != fi) continue;
            int step = v - 1; if (step >= (int)B.rows.size()) break;
            auto tk = topk_head(B.rows[step][best_ls], best_h, B.n_kv_at_step[step], 3);
            const Field& f = B.fields[fi];
            auto in = [&](int p){ return p >= f.lo - TOL && p <= f.hi + TOL; };
            n++; if (!tk.empty() && in(tk[0].first)) t1++;
            for (auto& pr : tk) if (in(pr.first)) { t3++; break; }
        }
        std::printf("    %-11s span[%d..%d]  top1 %d/%d  top3 %d/%d\n",
                    B.fields[fi].name.c_str(), B.fields[fi].lo, B.fields[fi].hi, t1, n, t3, n);
    }

    // ── 3 verbatim eyeball examples (from B, frozen head) ────────────────────
    std::printf("\n  === eyeball (B, frozen layer %d head %d): value token -> top-3 attended ===\n",
                attn_layers[best_ls], best_h);
    std::vector<int32_t> seqB = B.prompt_tokens;
    seqB.insert(seqB.end(), B.comp_tokens.begin(), B.comp_tokens.end());
    int shown = 0;
    for (int v = 1; v < (int)B.comp_tokens.size() && shown < 3; ++v) {
        int fi = B.comp_field[v]; if (fi < 0) continue;
        int step = v - 1; if (step >= (int)B.rows.size()) break;
        // pick the FIRST value token of each of the first 3 fields
        if (fi != shown) continue;
        auto tk = topk_head(B.rows[step][best_ls], best_h, B.n_kv_at_step[step], 3);
        std::string vt = tok->decode(B.comp_tokens[v]);
        for (char& c : vt) if (c == '\n') c = ' ';
        std::printf("\n  value token «%s» (field %s, source span [%d..%d])\n",
                    vt.c_str(), B.fields[fi].name.c_str(), B.fields[fi].lo, B.fields[fi].hi);
        for (int r = 0; r < (int)tk.size(); ++r) {
            bool in = tk[r].first >= B.fields[fi].lo - TOL && tk[r].first <= B.fields[fi].hi + TOL;
            std::printf("    #%d pos %3d  mass %.3f  %s | %s\n",
                        r + 1, tk[r].first, tk[r].second, in ? "IN " : "out",
                        ctx_around(tok, seqB, tk[r].first, 3).c_str());
        }
        shown++;
    }

    // ── Prompt C (boundary): reformatted values + a DATE CONFLICT ────────────
    // Same frozen head. Tests two things the verbatim PASS can't: (1) does the
    // head still point at the source when the emitted value is REFORMATTED
    // (European→plain, thousands stripped) rather than a byte copy? (WEAK line)
    // (2) with two candidate dates in the prompt, is the date row bimodal over
    // both spans, or does it lock onto the one it emits? (stretch)
    std::string promptC =
        "From: sales@initech.example\n"
        "Subject: Order Confirmation\n\n"
        "Hi,\n\n"
        "Order for customer Initech LLC.\n"
        "Order date: 2024-06-09.\n"
        "Requested delivery date: 2024-07-20.\n"
        "Quantity: 1.250 units.\n"
        "Unit price: EUR 8,75.\n"
        "Order total: EUR 10.937,50.\n\n"
        "Best.\n\n"
        "Extract as JSON with keys customer, order_date, quantity, unit_price, total:\n";
    std::string compC =
        "{\"customer\": \"Initech LLC\", \"order_date\": \"2024-06-09\", "
        "\"quantity\": 1250, \"unit_price\": 8.75, \"total\": 10937.50}";
    std::vector<Field> fieldsC = {
        {"customer",   "Initech LLC", -1, -1, ""},          // verbatim control
        {"order_date", "2024-06-09",  -1, -1, ""},          // verbatim, but conflicted
        {"quantity",   "1.250",       -1, -1, "1250"},      // reformatted: 1.250 -> 1250
        {"unit_price", "8,75",        -1, -1, "8.75"},      // reformatted: 8,75 -> 8.75
        {"total",      "10.937,50",   -1, -1, "10937.50"},  // reformatted
    };

    PromptRun C = run_prompt(fp.get(), sched, tok, meta, "C", promptC, compC, fieldsC, false);
    std::printf("\n================ PROMPT C (boundary: reformatting + conflict, FROZEN L%d H%d) ================\n",
                attn_layers[best_ls], best_h);
    report_fields(C);
    Score bC = score_lh(C, best_ls, best_h, TOL);
    std::printf("\n  FROZEN eval on C: top1 %d/%d (%.0f%%), top3 %d/%d (%.0f%%)\n",
                bC.top1, bC.n, bC.n ? 100.0 * bC.top1 / bC.n : 0,
                bC.top3, bC.n, bC.n ? 100.0 * bC.top3 / bC.n : 0);
    std::printf("  per-field (C, frozen head) — verbatim vs reformatted:\n");
    for (int fi = 0; fi < (int)C.fields.size(); ++fi) {
        int n = 0, t1 = 0, t3 = 0, agree = 0;
        for (int v = 1; v < (int)C.comp_tokens.size(); ++v) {
            if (C.comp_field[v] != fi) continue;
            int step = v - 1; if (step >= (int)C.rows.size()) break;
            auto tk = topk_head(C.rows[step][best_ls], best_h, C.n_kv_at_step[step], 3);
            const Field& f = C.fields[fi];
            auto in = [&](int p){ return p >= f.lo - TOL && p <= f.hi + TOL; };
            n++; if (!tk.empty() && in(tk[0].first)) t1++;
            for (auto& pr : tk) if (in(pr.first)) { t3++; break; }
            if (C.greedy_pred[step] == C.comp_tokens[v]) agree++;
        }
        bool reform = !C.fields[fi].cvalue.empty();
        std::printf("    %-11s %-11s span[%d..%d]  top1 %d/%d  top3 %d/%d  greedy-agree %d/%d\n",
                    C.fields[fi].name.c_str(), reform ? "[REFORMAT]" : "[verbatim]",
                    C.fields[fi].lo, C.fields[fi].hi, t1, n, t3, n, agree, n);
    }

    // Conflict bimodality: the order_date token's mass over the TWO date spans.
    {
        int order_lo, order_hi, deliv_lo, deliv_hi;
        find_token_span(tok, C.prompt_tokens, promptC, "2024-06-09", order_lo, order_hi);
        find_token_span(tok, C.prompt_tokens, promptC, "2024-07-20", deliv_lo, deliv_hi);
        // first completion token of order_date
        int dv = -1, ofi = 1; // order_date is field index 1
        for (int v = 1; v < (int)C.comp_tokens.size(); ++v)
            if (C.comp_field[v] == ofi) { dv = v; break; }
        std::printf("\n  conflict readout: order_date span[%d..%d] vs delivery span[%d..%d]\n",
                    order_lo, order_hi, deliv_lo, deliv_hi);
        if (dv > 0) {
            int step = dv - 1;
            const std::vector<float>& row = C.rows[step][best_ls];
            int n_kv = C.n_kv_at_step[step];
            auto mass_in = [&](int lo, int hi){ double m = 0;
                for (int j = std::max(1, lo - TOL); j <= hi + TOL && j < n_kv; ++j)
                    m += row[(size_t)best_h * n_kv + j]; return m; };
            std::printf("    date token «%s»: mass on ORDER span = %.3f, on DELIVERY span = %.3f\n",
                        tok->decode(C.comp_tokens[dv]).c_str(),
                        mass_in(order_lo, order_hi), mass_in(deliv_lo, deliv_hi));
            auto tk = topk_head(row, best_h, n_kv, 5);
            std::vector<int32_t> seqC = C.prompt_tokens;
            seqC.insert(seqC.end(), C.comp_tokens.begin(), C.comp_tokens.end());
            for (int r = 0; r < (int)tk.size(); ++r)
                std::printf("      top%d pos %3d mass %.3f | %s\n", r + 1, tk[r].first,
                            tk[r].second, ctx_around(tok, seqC, tk[r].first, 3).c_str());
        }
    }

    // ── Verdict ──────────────────────────────────────────────────────────────
    double t1f = bB.n ? (double)bB.top1 / bB.n : 0;
    double t3f = bB.n ? (double)bB.top3 / bB.n : 0;
    std::printf("\n================ VERDICT ================\n");
    std::printf("  frozen top1=%.0f%% (PASS needs >=66.7%%), top3=%.0f%% (PASS needs >=80%%)\n",
                100 * t1f, 100 * t3f);
    if (t1f >= 2.0 / 3.0 && t3f >= 0.80) std::printf("  => PASS\n");
    else if (t3f >= 0.80)                std::printf("  => WEAK (top3 aligns, top1 does not)\n");
    else                                 std::printf("  => KILL / WEAK — see tables\n");
    return 0;
}
