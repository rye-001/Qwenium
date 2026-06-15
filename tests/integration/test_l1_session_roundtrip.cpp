// test_l1_session_roundtrip.cpp — Phase 1b L1 falsifier (model-based, standalone).
//
// Drives a real model through prefill + decode, snapshots the L1 control state
// mid-decode (token span + sampler RNG + grammar cursor), then restores into a
// FRESH forward pass via feed_tokens(span) and resumes. Two gates, mirroring the
// Phase-3 fidelity/substitutability split (docs/plan-session-snapshot.md):
//
//   GATE A (continuous-vs-restored) — ACTIVE: token-stable.
//     The restored continuation must produce the IDENTICAL token-id sequence as
//     the uninterrupted live run. This is the strongest gate feed_tokens permits
//     vs sequential decode (its documented contract is token-stable, low FP bits
//     may differ — forward_pass_base.h). Proves control-state completeness: if
//     the RNG, grammar cursor, or token span were not fully captured, the branch
//     would diverge.
//
//   GATE B (restore-vs-restore) — ACTIVE: bitwise on CPU.
//     Two independent restores of the same blob must yield BIT-IDENTICAL
//     first-resume logits (memcmp) and identical token sequences. This is the
//     user-chosen "bitwise branch reproduction" tier: a given snapshot always
//     reproduces the exact same branch. On Metal this is DISABLED (mm-vs-mv /
//     galloc reuse); here we run the CPU build.
//
// Also prints the continuous-vs-restored logits max_abs_diff as the noise-floor
// measurement (the DISABLED strict-bitwise observation for GATE A).
//
// Usage: test-l1-session-roundtrip <model.gguf>
// Run against a Qwen recipe AND a Gemma recipe (CLAUDE.md cross-family rule):
//   Qwen3.5-0.8B-BF16 (qwen35: attention + SSM — also falsifies recurrent rebuild)
//   gemma-3-1b-it-BF16 (gemma3).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

#include "core/model.h"
#include "loader/tokenizer.h"
#include "models/forward_pass_base.h"
#include "models/model_registry.h"
#include "sampling/grammar_vocab.h"
#include "sampling/sampling.h"
#include "sampling/sampling_snapshot.h"
#include "session/compat_header.h"
#include "session/session_manifest.h"
#include "session/snapshot_io.h"
#include "state/token_sequence_section.h"

namespace {

using qwenium::GrammarStateSection;
using qwenium::GrammarVocab;
using qwenium::SamplerStateSection;
using qwenium::Sampler;
using qwenium::TemperatureSampler;
using qinf::session::CompatHeader;
using qinf::session::SessionManifest;
using qinf::session::SnapshotReader;
using qinf::session::SnapshotWriter;
using qinf::state::TokenSequenceSection;

// Permissive char-class grammar: stays valid across many tokens (never forces
// termination within the run) yet keeps a live cursor that must be restored.
const char* kGbnf = "root ::= [A-Za-z0-9 .,!?']+\n";

constexpr uint32_t kCtx = 512;
constexpr int kWarmupK = 6;    // tokens decoded before the snapshot
constexpr int kContinueM = 10; // tokens decoded after, on each branch
constexpr float kTemp = 0.8f;

std::unique_ptr<TemperatureSampler> make_sampler(const ModelMetadata& meta,
                                                 GrammarVocab* grammar) {
    auto s = std::make_unique<TemperatureSampler>(
        kTemp, /*repetition_penalty=*/1.0f, /*lookback=*/64, /*top_k=*/40,
        /*top_p=*/0.95f);
    s->set_grammar(grammar);
    for (int32_t id : meta.stop_token_ids) s->add_eos_token_id(id);
    return s;
}

struct StepOut {
    int32_t token;
    std::vector<float> logits;  // last-position vocab logits
};

// One decode step via the single-token prefill bridge (works for every recipe
// and exposes logits). Feeds `token` at the slot's current position, samples
// the next token, advances the grammar.
StepOut decode_one(ForwardPassBase* fp, ggml_backend_sched_t sched,
                   Sampler* sampler, int32_t token, uint32_t slot,
                   std::vector<int32_t>& history,
                   const std::vector<std::string>& vocab, size_t vocab_size) {
    int pos = static_cast<int>(fp->get_cache_pos(slot));
    std::vector<float> logits = fp->run_prefill({token}, pos, slot, sched);
    std::vector<float> tail(logits.end() - vocab_size, logits.end());
    std::vector<float> tail_copy = tail;
    int next = sampler->sample(tail_copy, history, vocab);
    sampler->accept_token(next);
    return {static_cast<int32_t>(next), std::move(tail)};
}

struct Branch {
    std::vector<int32_t> seq;
    std::vector<float> first_logits;
};

// Restore the blob into a fresh forward pass + fresh sampler/grammar, rebuild KV
// (and recurrent) via feed_tokens(span), then resume kContinueM steps.
Branch restore_and_continue(const std::vector<uint8_t>& blob,
                            const CompatHeader& header, Model& model,
                            const ModelMetadata& meta,
                            ggml_backend_sched_t sched,
                            const std::vector<std::string>& vocab,
                            size_t vocab_size) {
    auto fp = create_forward_pass(model, &meta, kCtx, /*max_batch=*/1);
    auto grammar = GrammarVocab::parse_impl(kGbnf);
    auto sampler = make_sampler(meta, grammar.get());

    std::vector<int32_t> tokens;
    uint32_t gen_start = 0;
    SessionManifest m;
    TokenSequenceSection ts(tokens, gen_start);
    SamplerStateSection ss(*sampler);
    GrammarStateSection gs(*grammar);
    m.add(&ts);
    m.add(&ss);
    m.add(&gs);
    SnapshotReader r(blob);
    m.restore(r, header);

    // The token vector's last element is the not-yet-fed "current" token; the
    // rest is the KV span (prompt + already-decoded tokens).
    std::vector<int32_t> feed_span(tokens.begin(), tokens.end() - 1);
    int32_t cur = tokens.back();
    fp->feed_tokens(feed_span, /*slot=*/0, sched);

    Branch out;
    for (int i = 0; i < kContinueM; ++i) {
        StepOut s = decode_one(fp.get(), sched, sampler.get(), cur, /*slot=*/0,
                               tokens, vocab, vocab_size);
        if (i == 0) out.first_logits = s.logits;
        out.seq.push_back(s.token);
        tokens.push_back(s.token);
        cur = s.token;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.gguf>\n";
        return 64;
    }
    const std::string model_path = argv[1];

    ggml_backend_load_all();
    register_builtin_models();

    Model model;
    try {
        model.load_metadata(model_path);
        model.load_tensors();
    } catch (const std::exception& e) {
        std::cerr << "load failed: " << e.what() << "\n";
        return 1;
    }

    const ModelMetadata& meta = model.get_metadata();
    Tokenizer* tok = model.get_tokenizer();
    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const size_t vocab_size = meta.vocab_size;
    ggml_backend_sched_t sched = model.get_scheduler();

    std::cout << "=== L1 session round-trip: " << meta.architecture << " ("
              << meta.model_name << ") ===\n";

    // --- Prompt (+ BOS per the model's contract; Gemma -it degenerates without) ---
    std::vector<int32_t> prompt = tok->encode("Once upon a time in a small town");
    if (meta.add_bos_token && meta.bos_token_id >= 0)
        prompt.insert(prompt.begin(), meta.bos_token_id);
    const uint32_t n_prompt = static_cast<uint32_t>(prompt.size());

    auto fp = create_forward_pass(model, &meta, kCtx, /*max_batch=*/1);
    if (!fp->feed_tokens_supported()) {
        std::cerr << "FAIL: recipe '" << meta.architecture
                  << "' does not support feed_tokens — L1 restore is "
                     "feed_tokens(span)+resume.\n";
        return 2;
    }

    auto grammar = GrammarVocab::parse_impl(kGbnf);
    auto sampler = make_sampler(meta, grammar.get());

    // --- Prefill the prompt, sample the first generated token ---
    std::vector<float> logits = fp->run_prefill(prompt, 0, /*slot=*/0, sched);
    std::vector<float> tail(logits.end() - vocab_size, logits.end());
    std::vector<float> tail_copy = tail;
    std::vector<int32_t> tokens = prompt;
    int32_t cur = static_cast<int32_t>(sampler->sample(tail_copy, tokens, vocab));
    sampler->accept_token(cur);
    tokens.push_back(cur);

    // --- Warmup decode to the snapshot point ---
    for (int k = 0; k < kWarmupK; ++k) {
        StepOut s = decode_one(fp.get(), sched, sampler.get(), cur, /*slot=*/0,
                               tokens, vocab, vocab_size);
        tokens.push_back(s.token);
        cur = s.token;
    }
    // KV now holds tokens[:-1]; tokens.back() == cur is the unfed current token.

    // --- Capture L1 snapshot ---
    CompatHeader header;
    header.weights_hash = 0xC0FFEEC0FFEEull;
    header.arch_id = static_cast<uint32_t>(std::hash<std::string>{}(meta.architecture));
    header.block_count = meta.block_count;
    header.embedding_length = meta.embedding_length;
    header.vocab_size = meta.vocab_size;

    std::vector<uint8_t> blob;
    {
        uint32_t gen_start = n_prompt;
        SessionManifest m;
        TokenSequenceSection ts(tokens, gen_start);
        SamplerStateSection ss(*sampler);
        GrammarStateSection gs(*grammar);
        m.add(&ts);
        m.add(&ss);
        m.add(&gs);
        SnapshotWriter w;
        m.capture(w, header);
        blob = w.buffer();
    }
    std::cout << "captured: " << tokens.size() << " tokens, blob "
              << blob.size() << " bytes (gen_start=" << n_prompt << ")\n";

    // --- Continuous continuation on the LIVE forward pass ---
    std::vector<int32_t> cont_seq;
    std::vector<float> cont_first_logits;
    {
        int32_t ccur = cur;
        for (int i = 0; i < kContinueM; ++i) {
            StepOut s = decode_one(fp.get(), sched, sampler.get(), ccur,
                                   /*slot=*/0, tokens, vocab, vocab_size);
            if (i == 0) cont_first_logits = s.logits;
            cont_seq.push_back(s.token);
            tokens.push_back(s.token);
            ccur = s.token;
        }
    }

    // --- Two independent restores ---
    Branch r1 = restore_and_continue(blob, header, model, meta, sched, vocab, vocab_size);
    Branch r2 = restore_and_continue(blob, header, model, meta, sched, vocab, vocab_size);

    // --- GATE A: continuous-vs-restored token sequence (token-stable) ---
    bool gate_a = (cont_seq == r1.seq);

    // --- GATE B: restore-vs-restore first-resume logits bitwise + seq equal ---
    bool gate_b_seq = (r1.seq == r2.seq);
    bool gate_b_bits =
        r1.first_logits.size() == r2.first_logits.size() &&
        std::memcmp(r1.first_logits.data(), r2.first_logits.data(),
                    r1.first_logits.size() * sizeof(float)) == 0;

    // --- Noise floor: continuous-vs-restored logits max_abs_diff (DISABLED) ---
    double max_abs_diff = 0.0;
    if (cont_first_logits.size() == r1.first_logits.size()) {
        for (size_t i = 0; i < cont_first_logits.size(); ++i) {
            max_abs_diff = std::max(
                max_abs_diff,
                static_cast<double>(std::fabs(cont_first_logits[i] - r1.first_logits[i])));
        }
    }

    auto print_seq = [](const char* tag, const std::vector<int32_t>& s) {
        std::cout << "  " << tag << ":";
        for (int32_t t : s) std::cout << ' ' << t;
        std::cout << "\n";
    };
    print_seq("continuous", cont_seq);
    print_seq("restored#1", r1.seq);
    print_seq("restored#2", r2.seq);
    std::cout << "GATE A (continuous==restored, token-stable): "
              << (gate_a ? "PASS" : "FAIL") << "\n";
    std::cout << "GATE B (restore-vs-restore bitwise on CPU): "
              << ((gate_b_seq && gate_b_bits) ? "PASS" : "FAIL")
              << " [seq=" << (gate_b_seq ? "ok" : "DIFF")
              << ", logits=" << (gate_b_bits ? "bit-identical" : "DIFF") << "]\n";
    std::cout << "[noise floor] continuous-vs-restored logits max_abs_diff = "
              << max_abs_diff << " (strict-bitwise GATE A is DISABLED)\n";

    bool ok = gate_a && gate_b_seq && gate_b_bits;
    std::cout << (ok ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return ok ? 0 : 3;
}
