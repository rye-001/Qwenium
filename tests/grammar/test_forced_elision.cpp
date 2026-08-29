// test_forced_elision.cpp — Phase B differential (docs/plan-grammar-guided-decode.md).
//
// Forced-token elision: when the grammar leaves exactly one valid token, the
// next token(s) are determined WITHOUT a forward pass. decode_step rolls the
// deterministic run and batch-advances model state via feed_tokens (KV append
// + recurrent overwrite, no LM head). This must not change WHAT is generated —
// only how fast.
//
// Gate (the documented Metal fork, per the owner decision reused from
// test_qwen35_feed_tokens.cpp / the out_ids slice): feed_tokens(span) vs
// N×single-step is NOT bitwise-reproducible on Metal (mm-vs-mv). So the
// contract Phase B's done-criterion #1 must be — and is here — **token-stable**:
// the generated token-ID sequence with forced elision ON must be IDENTICAL to
// the sequence with it OFF (same grammar + greedy → deterministic ids; the
// only divergence Metal can inject is in branching-position logits, asserted
// token-stable). Strict bytewise is not claimed (unattainable by construction).
//
// Combinatorial: iterates over resolved DecodePlan values (not raw flag
// products) — {diagnostic} × {forced elision} for a Unified recipe.
// Every legal plan must be token-stable vs the canonical baseline; illegal
// combinations never reach here (resolve_decode_plan throws). Same harness
// shape as test_sparse_differential.cpp: real model + grammar, per-case slot
// cloned from one prefill, greedy. Self-skips when the model is absent.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "engine/decode_step.h"
#include "engine/decode_plan.h"
#include "engine/model.h"
#include "../../src/loader/chat_template.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/models/model_registry.h"
#include "../../src/sampling/grammar_vocab.h"
#include "../../src/sampling/sampling.h"

#include "ggml-backend.h"
#include "ggml.h"

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "ERROR: cannot open " << path << "\n"; std::exit(1); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

// Decode n_tokens with forced elision on/off. Returns the token-ID stream
// (NOT including first_token; grammar already advanced past first_token).
std::vector<int32_t> run_decode(
    ForwardPassBase* fp, ggml_backend_sched_t sched,
    qinf::Sampler* sampler, const std::vector<std::string>& vocab,
    uint32_t vocab_size, int32_t first_token, uint32_t slot, int n_tokens,
    const std::vector<int32_t>& stop_ids, bool forced, bool force_dense,
    size_t* elided = nullptr) {

    std::vector<int32_t> out;
    auto is_stop = [&](int32_t t) {
        return std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end();
    };
    int32_t next = first_token;
    std::vector<int32_t> history;
    std::vector<int32_t> forced_run;

    for (int i = 0; i < n_tokens && !is_stop(next); ) {
        next = decode_step(fp, sched, sampler, next, slot, history, vocab,
                           vocab_size, force_dense,
                           forced ? &forced_run : nullptr);
        bool stop_hit = false;
        if (forced) {
            if (elided) *elided += forced_run.size();
            for (int32_t ft : forced_run) {
                if (is_stop(ft)) { stop_hit = true; break; }
                out.push_back(ft);
                history.push_back(ft);
                if (++i >= n_tokens) { stop_hit = true; break; }
            }
        }
        if (stop_hit) break;
        if (is_stop(next)) break;
        out.push_back(next);
        history.push_back(next);
        ++i;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const char* env_model = std::getenv("QWENIUM_MODEL_PATH");
    std::string model_path = env_model ? env_model
                                       : "./Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf";
    std::string grammar_path = "grammar/order-management.gbnf";
    std::string system_prompt_path = "tests/system_prompt_order_mngmt.txt";
    std::string user_query = "get all platinum customers";
    int n_tokens = 40;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--grammar") && i + 1 < argc) grammar_path = argv[++i];
        else if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n_tokens = std::stoi(argv[++i]);
    }

    if (FILE* f = std::fopen(model_path.c_str(), "rb")) std::fclose(f);
    else { std::cerr << "[skip] model not found: " << model_path << "\n"; return 0; }

    ggml_log_set([](ggml_log_level lvl, const char* t, void*) {
        if (lvl == GGML_LOG_LEVEL_ERROR) std::fprintf(stderr, "%s", t);
    }, nullptr);
    ggml_backend_load_all();
    register_builtin_models();

    Model model;
    try { model.load_metadata(model_path); model.load_tensors(); }
    catch (const std::exception& e) {
        std::cerr << "ERROR loading model: " << e.what() << "\n"; return 1;
    }
    Tokenizer* tok = model.get_tokenizer();
    const auto raw = tok->get_vocabulary();
    std::vector<std::string> vocab;
    vocab.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        vocab.push_back(tok->decode(static_cast<int32_t>(i)));
    const auto& meta = model.get_metadata();
    const uint32_t vocab_size = meta.vocab_size;

    auto gstr = slurp(grammar_path);
    auto grammar = qinf::GrammarVocab::parse_impl(gstr);
    if (!grammar) { std::cerr << "ERROR: grammar parse failed\n"; return 1; }

    const ChatTemplate* tmpl = lookup_chat_template(meta.architecture);
    if (!tmpl) { std::cerr << "ERROR: no chat template\n"; return 1; }
    std::string sys_turn  = tmpl->render({{"system", slurp(system_prompt_path)}}, false);
    std::string user_turn = tmpl->render({{"user", user_query}}, true);
    auto sys_tokens  = tok->encode(sys_turn);
    auto user_tokens = tok->encode(user_turn);

    // Combinatorial differential over resolved DecodePlan values (NOT raw
    // flag products): for a Unified recipe the legal plans are the
    // {diagnostic} × {forced elision} grid. Every legal plan must produce a
    // token-stable-identical stream to the canonical baseline (Optimized, no
    // forced elision). Plans that route through feed_tokens are token-stable,
    // not bytewise, by the documented Metal fork — that is the asserted
    // contract here (Phase B done-criterion #1, fork-rewritten).
    struct Case {
        const char* name;
        bool        forced;       // pass a forced_run sink to decode_step
        bool        force_dense;  // decode_step force_dense + slice_prefill off
    };
    const Case cases[] = {
        {"Optimized/forced=off", false, false},  // [0] = baseline
        {"Optimized/forced=on ", true,  false},
        {"ForceDense/forced=off", false, true },
        {"ForceDense/forced=on ", true,  true },
    };
    constexpr int kNCases = sizeof(cases) / sizeof(cases[0]);

    auto fp = create_forward_pass(model, &meta, 4096, kNCases + 1);
    ggml_backend_sched_t sched = model.get_scheduler();
    const uint32_t prefill_slot = 0;
    fp->run_prefill(sys_tokens, 0, prefill_slot, sched);

    const std::vector<int32_t> stop_ids = meta.stop_token_ids;
    auto make_sampler = [&]() {
        auto s = std::make_unique<qinf::GreedySampler>();
        s->set_grammar(grammar.get());
        s->build_token_trie(vocab);
        for (int32_t id : stop_ids) s->add_eos_token_id(id);
        return s;
    };
    auto fresh_start = [&](qinf::Sampler* s, uint32_t slot) -> int32_t {
        grammar->reset();
        std::vector<float> logits = fp->run_prefill(
            user_tokens, static_cast<int32_t>(sys_tokens.size()), slot, sched);
        std::vector<float> last(logits.end() - vocab_size, logits.end());
        std::vector<int32_t> hist;
        int32_t t = static_cast<int32_t>(s->sample(last, hist, vocab));
        grammar->accept_token(t, vocab);
        return t;
    };
    auto diag_str = [](DecodeDiagnostic d) {
        return d == DecodeDiagnostic::Optimized ? "Optimized" : "ForceDense";
    };

    std::vector<int32_t> base_stream;
    int32_t base_first = 0;
    bool all_ok = true;

    for (int c = 0; c < kNCases; ++c) {
        const Case& cs = cases[c];
        const uint32_t slot = static_cast<uint32_t>(c + 1);
        fp->clone_slot(prefill_slot, slot, sys_tokens.size());
        // The collapsed diagnostic seam: ForceDense drives BOTH inputs
        // (decode force_dense + the prefill out_ids slice toggle) together.
        fp->set_slice_prefill_head(!cs.force_dense);

        // Resolve once and show the plan this case actually spans (illegal
        // combinations would have thrown in resolve — none here, all legal).
        DecodePlan plan = resolve_decode_plan(fp.get(), cs.forced,
                                              cs.force_dense);
        std::cerr << "[plan " << cs.name << "]"
                  << " diagnostic=" << diag_str(plan.diagnostic)
                  << " allow_forced_elision=" << plan.allow_forced_elision
                  << " sparse_head_allowed=" << plan.sparse_head_allowed << "\n";

        auto s = make_sampler();
        int32_t first = fresh_start(s.get(), slot);
        size_t elided = 0;
        auto stream = run_decode(fp.get(), sched, s.get(), vocab, vocab_size,
                                 first, slot, n_tokens, stop_ids, cs.forced,
                                 cs.force_dense, &elided);

        std::cerr << "[" << cs.name << "] elided=" << elided << " first="
                  << first << " n=" << stream.size() << "\n";

        if (c == 0) { base_stream = stream; base_first = first; continue; }

        bool ok = (first == base_first) && (stream.size() == base_stream.size());
        for (size_t i = 0; ok && i < stream.size(); ++i)
            ok = (stream[i] == base_stream[i]);
        if (!ok) {
            all_ok = false;
            std::cerr << "[FAIL] plan '" << cs.name << "' diverged from the "
                         "Optimized/forced=off baseline (token-stable "
                         "required).\n";
            size_t n = std::min(stream.size(), base_stream.size());
            for (size_t i = 0; i < n; ++i)
                if (stream[i] != base_stream[i]) {
                    std::cerr << "  first divergence @" << i << " base="
                              << base_stream[i] << " ('" << vocab[base_stream[i]]
                              << "') case=" << stream[i] << " ('"
                              << vocab[stream[i]] << "')\n";
                    break;
                }
        } else {
            std::cerr << "[ok] plan '" << cs.name << "' token-stable vs "
                         "baseline (" << (1 + stream.size()) << " tokens)\n";
        }
    }

    if (!all_ok) {
        std::cerr << "[FAIL] DecodePlan combinatorial differential: at least "
                     "one legal plan changed the token stream.\n";
        return 1;
    }
    std::cerr << "[PASS] DecodePlan combinatorial differential: all "
              << kNCases << " legal plans token-stable vs baseline.\n";
    return 0;
}
