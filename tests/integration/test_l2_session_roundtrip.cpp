// test_l2_session_roundtrip.cpp — Phase 3 L2 falsifier (model-based, standalone).
//
// L2 = L1 superset: it additionally serializes the AppendKV cache and (where the
// recipe has one) the authoritative OverwriteRecurrent state, and restores by
// MEMCPYing those bytes into a fresh forward pass — NO feed_tokens, NO prefill.
// Two distinct gates (docs/plan-session-snapshot.md, Phase 3):
//
//   GATE 1 — Serialization fidelity — ACTIVE: byte-identical.
//     L2-restore-and-continue vs the SAME live session that produced the blob.
//     The KV/recurrent are a memcpy of the live frozen floats and the decode
//     path is batch=1 in both, so this is byte-exact. Validates the writer/
//     reader, not the forward pass.
//
//   GATE 2 — Substitutability — ACTIVE: token-stable.
//     L2-restore (frozen bytes) vs L1-restore (feed_tokens recompute of the same
//     span). Fresh recompute != frozen floats (mm-vs-mv / batched-vs-sequential),
//     so the achievable gate is token-stable; strict bitwise is the DISABLED
//     max_abs_diff measurement. If this goes RED: STOP, do not ship L2 — it is
//     forward-pass nondeterminism beyond the accepted floor.
//
// Plus a build_path_tag mismatch must be refused fail-loud.
//
// Usage: test-l2-session-roundtrip <model.gguf>
// Cross-family (CLAUDE.md): Qwen3.5-0.8B (qwen35: KV + DeltaNet recurrent — the
// authoritative-recurrent path) AND gemma-3-1b-it (gemma3: KV only).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

#include "engine/model.h"
#include "loader/tokenizer.h"
#include "models/forward_pass_base.h"
#include "models/model_registry.h"
#include "sampling/sampling.h"
#include "sampling/sampling_snapshot.h"  // SamplerStateSection
#include "session/compat_header.h"
#include "session/session_manifest.h"
#include "session/snapshot_io.h"
#include "state/deltanet_state.h"
#include "state/kv_cache_simple.h"
#include "state/token_sequence_section.h"

namespace {

using qinf::session::CompatHeader;
using qinf::session::SessionManifest;
using qinf::session::SnapshotReader;
using qinf::session::SnapshotWriter;
using qinf::state::TokenSequenceSection;
using qwenium::Sampler;
using qwenium::SamplerStateSection;
using qwenium::TemperatureSampler;

constexpr uint32_t kCtx = 512;
constexpr int kWarmupK = 6;
constexpr int kContinueM = 10;
constexpr float kTemp = 0.8f;

// No grammar here on purpose: L2's subject is the tensor path (KV / recurrent),
// and the grammar cursor was already validated in the L1 falsifier. A grammar
// mask over a raw-prompted small model collapses generation to a repeated token
// (a fixed point), which would let a BROKEN KV restore pass the gates vacuously.
// Repetition penalty (1.3) ON keeps the continuation varied so a wrong KV
// actually diverges; it reads the restored token vector, so it stays
// deterministic and identical across all branches.
std::unique_ptr<TemperatureSampler> make_sampler(const ModelMetadata& meta) {
    auto s = std::make_unique<TemperatureSampler>(kTemp, 1.3f, 64, 40, 0.95f);
    for (int32_t id : meta.stop_token_ids) s->add_eos_token_id(id);
    return s;
}

// build_path_tag is derived from the live KV cache's kernel-path identity
// (backend + dtype + ctx), so two processes on the same build/model/backend
// produce the same tag (blob loads) while a different path produces a different
// tag (blob refused fail-loud) — the empirical "is build_path_tag sufficient?"
// hinge for cross-process / cross-backend L2.
CompatHeader make_header(const ModelMetadata& m, const simple_kv_cache* kv) {
    CompatHeader h;
    h.arch_id = static_cast<uint32_t>(std::hash<std::string>{}(m.architecture));
    h.weights_hash = m.weights_hash;
    h.block_count = m.block_count;
    h.embedding_length = m.embedding_length;
    h.vocab_size = m.vocab_size;
    h.build_path_tag = kv->path_tag();
    return h;
}

struct StepOut {
    int32_t token;
    std::vector<float> logits;
};

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

// Adds the control sections + (for L2) the KV / recurrent sections, in a fixed
// order shared by capture and restore. The section objects are owned by the
// out-params so they outlive the manifest's use.
struct Sections {
    std::unique_ptr<TokenSequenceSection> ts;
    std::unique_ptr<SamplerStateSection> ss;
    std::unique_ptr<KvCacheSection> kv;
    std::unique_ptr<DeltaNetStateSection> dn;
};

void build_manifest(SessionManifest& m, Sections& sec, bool with_tensors,
                    ForwardPassBase* fp, std::vector<int32_t>& tokens,
                    uint32_t& gen_start, Sampler& sampler) {
    sec.ts = std::make_unique<TokenSequenceSection>(tokens, gen_start);
    sec.ss = std::make_unique<SamplerStateSection>(sampler);
    m.add(sec.ts.get());
    m.add(sec.ss.get());
    if (with_tensors) {
        sec.kv = std::make_unique<KvCacheSection>(*fp->snapshot_kv_cache(), 0);
        m.add(sec.kv.get());
        if (fp->snapshot_recurrent()) {
            sec.dn = std::make_unique<DeltaNetStateSection>(*fp->snapshot_recurrent(), 0);
            m.add(sec.dn.get());
        }
    }
}

struct Branch {
    std::vector<int32_t> seq;
    std::vector<float> first_logits;
};

// L2 restore: memcpy KV (+recurrent) from the blob, NO feed_tokens, resume.
// The expected header is re-derived from THIS process's fresh forward pass — so
// a cross-process loader independently computes build_path_tag and the manifest
// refuses a blob whose path differs. `override_header` forces a tag for the
// fail-loud test.
Branch restore_l2(const std::vector<uint8_t>& blob, Model& model,
                  const ModelMetadata& meta, ggml_backend_sched_t sched,
                  const std::vector<std::string>& vocab, size_t vocab_size,
                  const CompatHeader* override_header = nullptr) {
    auto fp = create_forward_pass(model, &meta, kCtx, 1);
    CompatHeader expected = override_header
        ? *override_header
        : make_header(meta, fp->snapshot_kv_cache());
    auto sampler = make_sampler(meta);
    std::vector<int32_t> tokens;
    uint32_t gen_start = 0;
    SessionManifest m;
    Sections sec;
    build_manifest(m, sec, /*with_tensors=*/true, fp.get(), tokens, gen_start,
                   *sampler);
    SnapshotReader r(blob);
    m.restore(r, expected);  // sets KV cursor + memcpy KV/recurrent

    int32_t cur = tokens.back();  // KV already holds [0,N); cur is the unfed token
    Branch out;
    for (int i = 0; i < kContinueM; ++i) {
        StepOut s = decode_one(fp.get(), sched, sampler.get(), cur, 0, tokens,
                               vocab, vocab_size);
        if (i == 0) out.first_logits = s.logits;
        out.seq.push_back(s.token);
        tokens.push_back(s.token);
        cur = s.token;
    }
    return out;
}

// L1 restore: control only, rebuild KV via feed_tokens(span), resume.
Branch restore_l1(const std::vector<uint8_t>& blob, Model& model,
                  const ModelMetadata& meta, ggml_backend_sched_t sched,
                  const std::vector<std::string>& vocab, size_t vocab_size) {
    auto fp = create_forward_pass(model, &meta, kCtx, 1);
    CompatHeader expected = make_header(meta, fp->snapshot_kv_cache());
    auto sampler = make_sampler(meta);
    std::vector<int32_t> tokens;
    uint32_t gen_start = 0;
    SessionManifest m;
    Sections sec;
    build_manifest(m, sec, /*with_tensors=*/false, fp.get(), tokens, gen_start,
                   *sampler);
    SnapshotReader r(blob);
    m.restore(r, expected);

    std::vector<int32_t> feed_span(tokens.begin(), tokens.end() - 1);
    int32_t cur = tokens.back();
    fp->feed_tokens(feed_span, 0, sched);
    Branch out;
    for (int i = 0; i < kContinueM; ++i) {
        StepOut s = decode_one(fp.get(), sched, sampler.get(), cur, 0, tokens,
                               vocab, vocab_size);
        if (i == 0) out.first_logits = s.logits;
        out.seq.push_back(s.token);
        tokens.push_back(s.token);
        cur = s.token;
    }
    return out;
}

bool logits_bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    if (a.size() != b.size()) return 1e9;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

void dump_tokens(const std::string& path, const std::vector<int32_t>& seq) {
    std::ofstream f(path);
    for (int32_t t : seq) f << t << "\n";
}

void dump_logits(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    // Modes:
    //   <model>                                  in-process all-gates run
    //   <model> save <l2blob> <l1blob> <refpfx>  capture blobs to disk + dump the
    //                                            live continuation (GATE 1 ref)
    //   <model> load <l2blob> <l1blob> <outpfx>  resume from disk (fresh process):
    //                                            dump L2 + L1(feed_tokens) results
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <model.gguf> [save|load <l2blob> <l1blob> <prefix>]\n";
        return 64;
    }
    const std::string mode = (argc >= 3) ? argv[2] : "inproc";
    ggml_backend_load_all();
    register_builtin_models();

    Model model;
    try {
        model.load_metadata(argv[1]);
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

    std::cout << "=== L2 session round-trip: " << meta.architecture << " ("
              << meta.model_name << ") mode=" << mode << " ===\n";

    // ── load: a fresh process resumes the on-disk blobs ──────────────────────
    if (mode == "load") {
        if (argc < 6) {
            std::cerr << "load: expected <l2blob> <l1blob> <out_prefix>\n";
            return 64;
        }
        const std::string l2blob = argv[3], l1blob = argv[4], outp = argv[5];
        Branch l2, l1;
        try {
            l2 = restore_l2(qinf::session::load_from_file(l2blob), model, meta,
                            sched, vocab, vocab_size);
            l1 = restore_l1(qinf::session::load_from_file(l1blob), model, meta,
                            sched, vocab, vocab_size);
        } catch (const std::exception& e) {
            std::cerr << "load: refused/failed: " << e.what() << "\n";
            return 4;
        }
        dump_tokens(outp + ".l2.tok", l2.seq);
        dump_logits(outp + ".l2.logits", l2.first_logits);
        dump_tokens(outp + ".l1.tok", l1.seq);
        std::cout << "[load] wrote " << outp << ".l2.{tok,logits} + " << outp
                  << ".l1.tok\n";
        return 0;
    }

    auto fp = create_forward_pass(model, &meta, kCtx, 1);
    if (!fp->snapshot_kv_cache()) {
        std::cerr << "FAIL: recipe '" << meta.architecture
                  << "' has no snapshot_kv_cache accessor (needed for L2)\n";
        return 2;
    }
    const bool has_recurrent = fp->snapshot_recurrent() != nullptr;
    std::cout << "lanes: AppendKV" << (has_recurrent ? " + OverwriteRecurrent" : "")
              << "\n";

    std::vector<int32_t> prompt = tok->encode("Once upon a time in a small town");
    if (meta.add_bos_token && meta.bos_token_id >= 0)
        prompt.insert(prompt.begin(), meta.bos_token_id);
    const uint32_t n_prompt = static_cast<uint32_t>(prompt.size());

    auto sampler = make_sampler(meta);

    std::vector<float> logits = fp->run_prefill(prompt, 0, 0, sched);
    std::vector<float> tail(logits.end() - vocab_size, logits.end());
    std::vector<float> tail_copy = tail;
    std::vector<int32_t> tokens = prompt;
    int32_t cur = static_cast<int32_t>(sampler->sample(tail_copy, tokens, vocab));
    sampler->accept_token(cur);
    tokens.push_back(cur);

    for (int k = 0; k < kWarmupK; ++k) {
        StepOut s = decode_one(fp.get(), sched, sampler.get(), cur, 0, tokens,
                               vocab, vocab_size);
        tokens.push_back(s.token);
        cur = s.token;
    }

    CompatHeader header = make_header(meta, fp->snapshot_kv_cache());

    // Capture L2 (control + KV + recurrent) and L1 (control only) blobs.
    std::vector<uint8_t> blob_l2, blob_l1;
    {
        uint32_t gen_start = n_prompt;
        SessionManifest m;
        Sections sec;
        build_manifest(m, sec, /*with_tensors=*/true, fp.get(), tokens, gen_start,
                       *sampler);
        SnapshotWriter w;
        m.capture(w, header);
        blob_l2 = w.buffer();
    }
    {
        uint32_t gen_start = n_prompt;
        SessionManifest m;
        Sections sec;
        build_manifest(m, sec, /*with_tensors=*/false, fp.get(), tokens, gen_start,
                       *sampler);
        SnapshotWriter w;
        m.capture(w, header);
        blob_l1 = w.buffer();
    }
    std::cout << "captured: " << tokens.size() << " tokens; L2 blob "
              << blob_l2.size() << " B, L1 blob " << blob_l1.size() << " B\n";

    // Live continuation (the session that produced the blob) — the GATE 1 ref.
    std::vector<int32_t> cont_seq;
    std::vector<float> cont_first_logits;
    {
        int32_t ccur = cur;
        for (int i = 0; i < kContinueM; ++i) {
            StepOut s = decode_one(fp.get(), sched, sampler.get(), ccur, 0, tokens,
                                   vocab, vocab_size);
            if (i == 0) cont_first_logits = s.logits;
            cont_seq.push_back(s.token);
            tokens.push_back(s.token);
            ccur = s.token;
        }
    }

    // ── save: persist blobs + the live continuation, for a fresh-process load ──
    if (mode == "save") {
        if (argc < 6) {
            std::cerr << "save: expected <l2blob> <l1blob> <ref_prefix>\n";
            return 64;
        }
        qinf::session::save_to_file(argv[3], blob_l2);
        qinf::session::save_to_file(argv[4], blob_l1);
        const std::string refp = argv[5];
        dump_tokens(refp + ".tok", cont_seq);
        dump_logits(refp + ".logits", cont_first_logits);
        std::cout << "[save] L2 blob " << blob_l2.size() << " B, L1 blob "
                  << blob_l1.size() << " B, ref " << refp
                  << ".{tok,logits}; build_path_tag=" << header.build_path_tag
                  << " weights_hash=" << header.weights_hash << "\n";
        return 0;
    }

    Branch l2 = restore_l2(blob_l2, model, meta, sched, vocab, vocab_size);
    Branch l1 = restore_l1(blob_l1, model, meta, sched, vocab, vocab_size);

    // GATE 1 — fidelity (byte): L2 restore reproduces the live session exactly.
    bool gate1 = (l2.seq == cont_seq) &&
                 logits_bit_equal(l2.first_logits, cont_first_logits);
    // GATE 2 — substitutability (token-stable): L2 frozen bytes vs L1 recompute.
    bool gate2 = (l2.seq == l1.seq);
    double floor = max_abs_diff(l2.first_logits, l1.first_logits);

    auto print_seq = [](const char* t, const std::vector<int32_t>& s) {
        std::cout << "  " << t << ":";
        for (int32_t x : s) std::cout << ' ' << x;
        std::cout << "\n";
    };
    print_seq("live      ", cont_seq);
    print_seq("L2 restore", l2.seq);
    print_seq("L1 restore", l1.seq);
    std::cout << "GATE 1 (L2 fidelity vs live, BYTE): " << (gate1 ? "PASS" : "FAIL")
              << " [seq=" << (l2.seq == cont_seq ? "ok" : "DIFF") << ", logits="
              << (logits_bit_equal(l2.first_logits, cont_first_logits)
                      ? "bit-identical" : "DIFF") << "]\n";
    std::cout << "GATE 2 (L2 substitutability vs L1 feed_tokens, token-stable): "
              << (gate2 ? "PASS" : "FAIL (STOP — do not ship L2)") << "\n";
    std::cout << "[noise floor] L2-vs-L1 logits max_abs_diff = " << floor
              << " (strict-bitwise substitutability is DISABLED)\n";

    // build_path_tag must be load-bearing: a mismatched tag is refused fail-loud.
    bool gate_tag = false;
    {
        CompatHeader bad = header;
        bad.build_path_tag ^= 0x1ull;
        try {
            restore_l2(blob_l2, model, meta, sched, vocab, vocab_size, &bad);
        } catch (const std::exception& e) {
            gate_tag = std::string(e.what()).find("build_path_tag") != std::string::npos;
        }
    }
    std::cout << "GATE 3 (build_path_tag mismatch refused fail-loud): "
              << (gate_tag ? "PASS" : "FAIL") << "\n";

    bool ok = gate1 && gate2 && gate_tag;
    std::cout << (ok ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return ok ? 0 : 3;
}
