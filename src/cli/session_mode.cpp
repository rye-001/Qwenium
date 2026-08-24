#include "session_mode.h"

#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../core/decode_step.h"
#include "../core/model.h"
#include "../loader/tokenizer.h"
#include "../models/forward_pass_base.h"
#include "../models/model_registry.h"
#include "../sampling/grammar_vocab.h"
#include "../sampling/sampling.h"
#include "../sampling/sampling_snapshot.h"
#include "../session/compat_header.h"
#include "../session/session_manifest.h"
#include "../session/snapshot_io.h"
#include "../state/token_sequence_section.h"

namespace {

using qinf::session::CompatHeader;
using qinf::session::SessionManifest;
using qinf::session::SnapshotReader;
using qinf::session::SnapshotWriter;
using qinf::state::TokenSequenceSection;
using qwenium::GrammarStateSection;
using qwenium::SamplerStateSection;

// Compat-header discriminants available from the loaded model. weights_hash is
// the model's content identity (computed at load); build_path_tag stays default
// here (L1 restore is feed_tokens, path-independent). arch_id + shape +
// weights_hash give a fail-loud cross-version / cross-weights refusal.
CompatHeader make_header(const ModelMetadata& m) {
    CompatHeader h;
    h.arch_id = static_cast<uint32_t>(std::hash<std::string>{}(m.architecture));
    h.weights_hash = m.weights_hash;
    h.block_count = m.block_count;
    h.embedding_length = m.embedding_length;
    h.attention_head_count = m.attention_head_count;
    h.attention_head_count_kv = m.attention_head_count_kv;
    h.attention_key_length = m.attention_key_length;
    h.vocab_size = m.vocab_size;
    return h;
}

}  // namespace

int run_session(Model& model, const CliArgs& args,
                std::unique_ptr<qwenium::GrammarVocab>& grammar,
                const std::function<void(int32_t)>& log_token) {
    const ModelMetadata& meta = model.get_metadata();
    Tokenizer* tok = model.get_tokenizer();
    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const size_t vocab_size = meta.vocab_size;
    ggml_backend_sched_t sched = model.get_scheduler();

    register_builtin_models();
    std::unique_ptr<ForwardPassBase> fp =
        create_forward_pass(model, &meta, args.context_length, /*max_batch=*/1,
                            args.kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    if (!fp->feed_tokens_supported()) {
        std::cerr << "run_session: recipe \"" << meta.architecture
                  << "\" expected feed_tokens support (L1 restore is "
                     "feed_tokens(span)+resume), got unsupported.\n";
        return 2;
    }

    // Sampler + grammar mirror complete.cpp's setup so the session decodes on
    // the production path.
    std::unique_ptr<qwenium::Sampler> sampler;
    if (args.temperature > 0.0f) {
        sampler = std::make_unique<qwenium::TemperatureSampler>(
            args.temperature, args.repetition_penalty, 64, args.top_k, args.top_p);
    } else {
        // Greedy is a true argmax unless --repeat-penalty was explicitly
        // given (see complete.cpp).
        sampler = std::make_unique<qwenium::GreedySampler>(
            args.repetition_penalty_set ? args.repetition_penalty : 1.0f);
    }
    if (grammar) sampler->set_grammar(grammar.get());
    sampler->build_token_trie(vocab);
    for (int32_t id : meta.stop_token_ids) sampler->add_eos_token_id(id);

    auto is_stop = [&](int32_t t) {
        for (int32_t s : meta.stop_token_ids)
            if (t == s) return true;
        return false;
    };

    // The three L1 Control sections. GrammarStateSection is present iff a grammar
    // is active; save and load MUST agree (the manifest validates section
    // count/id and refuses a mismatch fail-loud).
    std::vector<int32_t> tokens;
    uint32_t gen_start = 0;
    auto build_manifest = [&](SessionManifest& m,
                              std::unique_ptr<TokenSequenceSection>& ts,
                              std::unique_ptr<SamplerStateSection>& ss,
                              std::unique_ptr<GrammarStateSection>& gs) {
        ts = std::make_unique<TokenSequenceSection>(tokens, gen_start);
        ss = std::make_unique<SamplerStateSection>(*sampler);
        m.add(ts.get());
        m.add(ss.get());
        if (grammar) {
            gs = std::make_unique<GrammarStateSection>(*grammar);
            m.add(gs.get());
        }
    };

    int32_t next;  // the next token to feed (current, unfed)

    if (!args.load_session_path.empty()) {
        // ── RESUME ───────────────────────────────────────────────────────────
        std::vector<uint8_t> blob =
            qinf::session::load_from_file(args.load_session_path);
        SessionManifest m;
        std::unique_ptr<TokenSequenceSection> ts;
        std::unique_ptr<SamplerStateSection> ss;
        std::unique_ptr<GrammarStateSection> gs;
        build_manifest(m, ts, ss, gs);

        SnapshotReader r(blob);
        try {
            m.restore(r, make_header(meta));  // fail-loud on cross-version mismatch
        } catch (const std::exception& e) {
            std::cerr << "run_session: refusing to load \""
                      << args.load_session_path << "\": " << e.what() << "\n";
            return 4;
        }
        if (tokens.empty()) {
            std::cerr << "run_session: loaded session has an empty token span\n";
            return 4;
        }

        // tokens.back() is the not-yet-fed current token; the rest is the KV
        // span. Pop the current token into `next` so `tokens` becomes exactly the
        // already-fed prefix — matching the fresh path's loop invariant (tokens =
        // KV span, next = token to feed). Then rebuild KV (and recurrent).
        next = tokens.back();
        tokens.pop_back();
        fp->feed_tokens(tokens, /*slot=*/0, sched);
        std::cout << "[session] resumed " << (tokens.size() + 1)
                  << " tokens (gen_start=" << gen_start << ") from "
                  << args.load_session_path << "\n";
    } else {
        // ── FRESH: prefill prompt, sample first token ────────────────────────
        tokens = tok->encode(args.prompt);
        if (meta.add_bos_token && meta.bos_token_id >= 0)
            tokens.insert(tokens.begin(), meta.bos_token_id);
        gen_start = static_cast<uint32_t>(tokens.size());

        std::vector<float> logits = fp->run_prefill(tokens, 0, /*slot=*/0, sched);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());
        next = static_cast<int32_t>(sampler->sample(tail, tokens, vocab));
        sampler->accept_token(next);
    }

    // ── Decode loop ──────────────────────────────────────────────────────────
    // `generated` counts tokens past gen_start; on resume it starts at the
    // restored count so --max-tokens stays a total bound across save+load.
    int generated = static_cast<int>(tokens.size()) - static_cast<int>(gen_start);
    const bool saving = !args.save_session_path.empty();

    while (generated < args.max_tokens) {
        if (is_stop(next)) break;
        log_token(next);
        std::cout << make_readable(tok->decode(next)) << std::flush;
        tokens.push_back(next);
        ++generated;

        // Snapshot point: tokens.back()==next is the current unfed token, KV
        // holds tokens[:-1]. This is exactly the state a resume reconstructs.
        if (saving && generated == args.save_session_at) {
            CompatHeader header = make_header(meta);
            SessionManifest m;
            std::unique_ptr<TokenSequenceSection> ts;
            std::unique_ptr<SamplerStateSection> ss;
            std::unique_ptr<GrammarStateSection> gs;
            build_manifest(m, ts, ss, gs);
            SnapshotWriter w;
            m.capture(w, header);
            qinf::session::save_to_file(args.save_session_path, w.buffer());
            std::cout << "\n[session] saved " << tokens.size() << " tokens, "
                      << w.size() << " bytes to " << args.save_session_path
                      << " (resume continues this run)\n";
        }

        next = decode_step(fp.get(), sched, sampler.get(), next, /*slot=*/0,
                           tokens, vocab, vocab_size);
    }

    std::cout << "\n--- End of Session ---\n";
    return 0;
}
