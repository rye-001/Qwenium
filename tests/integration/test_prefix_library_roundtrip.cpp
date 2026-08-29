// test_prefix_library_roundtrip.cpp — Phase 4 prefix-library falsifier
// (model-based, standalone). docs/plan-session-snapshot.md, Phase 4.
//
// A warm prefix blob is a slot's KV (+ authoritative recurrent state) after
// prefilling a recurring PREFIX (e.g. a system prompt). The library stores it
// keyed by hash(prefix); a later run loads it, SKIPS the prefix prefill, and
// starts the user turn at pos = len(prefix). This harness is the gate that
// decides whether that substitution is safe to ship.
//
// Three branches over one fixed (prefix, turn):
//   REF  — fresh fp: prefill(prefix ++ turn) from scratch, decode.        (scratch)
//   LIVE — fresh fp: prefill(prefix); CAPTURE blob; prefill(turn); decode. (producer)
//   WARM — fresh fp: LOAD blob (memcpy KV, no prefill); prefill(turn); decode. (consumer)
//
// Decode is a DETERMINISTIC argmax with a repetition penalty applied in-harness
// (NO sampler RNG: TemperatureSampler seeds mt19937 from random_device, so two
// branches would draw differently at the first near-tie and diverge spuriously).
// The penalty keeps the continuation varied so a WRONG prefix KV actually
// diverges, while staying a pure function of the logits + history.
//
// Gates:
//   GATE 1 — SHIP GATE — fidelity+substitutability (BYTE): WARM == LIVE.
//     LIVE is the chunked from-scratch path the CLI ACTUALLY runs without a
//     cache: prefill(prefix), then prefill(turn), then decode. WARM replaces the
//     prefix prefill with a blob memcpy. Identical prefix KV bytes + identical
//     subsequent path ⇒ byte-identical logits and tokens. This is the real
//     claim: the warm prefix blob is a perfect substitute for re-prefilling the
//     prefix. If RED: STOP — the serialize/deserialize or restore is wrong.
//   GATE 2 — INFORMATIONAL — WARM vs a single combined prefill(prefix++turn).
//     Token-stable EXPECTED to differ on a small model: combined-batch vs
//     chunked prefill is the mm-vs-mv batch-shape fork ([[feedback_flag_repro
//     _conflicts]]), NOT a cache failure — the CLI never combines these. Report
//     the token-stability + noise floor; do not gate ship on it.
//   GATE 3 — a build_path_tag / weights_hash mismatch is refused fail-loud by
//     the library + manifest (never silently re-prefilled — the F9 rule).
//
// Usage: test-prefix-library-roundtrip <model.gguf>
// Cross-family (CLAUDE.md): Qwen3.5-0.8B (KV + DeltaNet recurrent) AND
// gemma-3-1b-it (KV only).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

#include "engine/model.h"
#include "session/prefix_library.h"
#include "loader/tokenizer.h"
#include "models/forward_pass_base.h"
#include "models/model_registry.h"
#include "session/compat_header.h"
#include "session/session_manifest.h"
#include "session/snapshot_io.h"
#include "state/deltanet_state.h"
#include "state/kv_cache_simple.h"

namespace {

using qinf::session::CompatHeader;
using qinf::session::SessionManifest;
using qinf::session::SnapshotReader;
using qinf::session::SnapshotWriter;

constexpr uint32_t kCtx = 512;
constexpr int kDecodeN = 12;
constexpr float kRepPenalty = 1.3f;

// Deterministic next-token rule: repetition-penalize the logits of tokens
// already in `history` (the standard divide-positive / multiply-negative form),
// then take the argmax. Pure function of (logits, history) — no RNG — so two
// branches with byte-identical logits ALWAYS agree, and the penalty keeps the
// continuation KV-sensitive (a wrong prefix KV diverges instead of collapsing).
int32_t pick(std::vector<float> logits, const std::vector<int32_t>& history) {
    for (int32_t t : history) {
        if (t >= 0 && static_cast<size_t>(t) < logits.size()) {
            float& l = logits[t];
            l = (l > 0.0f) ? l / kRepPenalty : l * kRepPenalty;
        }
    }
    int32_t best = 0;
    float best_v = logits.empty() ? 0.0f : logits[0];
    for (size_t i = 1; i < logits.size(); ++i)
        if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<int32_t>(i); }
    return best;
}

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

// The prefix blob's sections, in a fixed order shared by capture and restore:
// the AppendKV slot span plus (where the recipe has one) the authoritative
// OverwriteRecurrent state — both for slot 0.
struct PrefixSections {
    std::unique_ptr<KvCacheSection> kv;
    std::unique_ptr<DeltaNetStateSection> dn;
};
void build_sections(SessionManifest& m, PrefixSections& sec, ForwardPassBase* fp) {
    sec.kv = std::make_unique<KvCacheSection>(*fp->snapshot_kv_cache(), 0);
    m.add(sec.kv.get());
    if (fp->snapshot_recurrent()) {
        sec.dn = std::make_unique<DeltaNetStateSection>(*fp->snapshot_recurrent(), 0);
        m.add(sec.dn.get());
    }
}

struct Branch {
    std::vector<int32_t> seq;
    std::vector<float> first_logits;  // the prefill tail (logits that pick token 0)
};

// Given `fp` whose slot-0 KV cursor is at the end of `context` (the full fed
// prefix++turn) and `prefill_tail` = the prefill's last-position logits, decode
// kDecodeN tokens deterministically (pick = rep-penalty + argmax over the
// running history). All branches funnel through here so they differ ONLY in how
// they reached this state (combined / chunked / blob-loaded prefix).
Branch run_decode(ForwardPassBase* fp, ggml_backend_sched_t sched,
                  std::vector<int32_t> context,
                  const std::vector<float>& prefill_tail, size_t vocab_size) {
    Branch out;
    out.first_logits = prefill_tail;
    int32_t cur = pick(prefill_tail, context);
    out.seq.push_back(cur);
    context.push_back(cur);
    for (int i = 1; i < kDecodeN; ++i) {
        int pos = static_cast<int>(fp->get_cache_pos(0));
        std::vector<float> logits = fp->run_prefill({cur}, pos, 0, sched);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());
        cur = pick(tail, context);
        out.seq.push_back(cur);
        context.push_back(cur);
    }
    return out;
}

bool logits_bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.gguf>\n";
        return 64;
    }
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
    const size_t vocab_size = meta.vocab_size;
    ggml_backend_sched_t sched = model.get_scheduler();

    // The recurring PREFIX (system-prompt-shaped) and the per-request TURN.
    std::vector<int32_t> prefix = tok->encode(
        "You are a terse assistant. Answer in one short sentence.");
    if (meta.add_bos_token && meta.bos_token_id >= 0)
        prefix.insert(prefix.begin(), meta.bos_token_id);
    std::vector<int32_t> turn = tok->encode(" What is the capital of France?");
    const uint32_t n_prefix = static_cast<uint32_t>(prefix.size());

    std::cout << "=== prefix-library round-trip: " << meta.architecture << " ("
              << meta.model_name << ") n_prefix=" << n_prefix
              << " n_turn=" << turn.size() << " ===\n";

    std::vector<int32_t> context = prefix;
    context.insert(context.end(), turn.begin(), turn.end());

    // ── REF (informational): prefill prefix++turn as ONE combined batch. ─────
    Branch ref;
    {
        auto fp = create_forward_pass(model, &meta, kCtx, 1);
        std::vector<float> logits = fp->run_prefill(context, 0, 0, sched);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());
        ref = run_decode(fp.get(), sched, context, tail, vocab_size);
    }

    // ── LIVE (ship reference): chunked from-scratch — prefill(prefix),
    //    CAPTURE the blob, prefill(turn), decode. This is what the CLI runs
    //    without a cache. ──────────────────────────────────────────────────────
    const std::string dir = std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp")
                          + "/qinf_prefixlib_test";
    CompatHeader header;
    std::vector<uint8_t> blob;
    Branch live;
    const uint64_t key = PrefixLibrary::key_for(prefix);
    {
        auto fp = create_forward_pass(model, &meta, kCtx, 1);
        if (!fp->snapshot_kv_cache()) {
            std::cerr << "FAIL: recipe '" << meta.architecture
                      << "' has no snapshot_kv_cache accessor (needed for L2)\n";
            return 2;
        }
        header = make_header(meta, fp->snapshot_kv_cache());

        fp->run_prefill(prefix, 0, 0, sched);  // prefix only → KV[0,n_prefix)

        // Capture the warm prefix blob (header + KV [+ recurrent]) and publish it.
        {
            SessionManifest m;
            PrefixSections sec;
            build_sections(m, sec, fp.get());
            SnapshotWriter w;
            m.capture(w, header);
            blob = w.buffer();
        }
        PrefixLibrary(dir, header).store(key, blob);

        std::vector<float> logits =
            fp->run_prefill(turn, static_cast<int>(n_prefix), 0, sched);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());
        live = run_decode(fp.get(), sched, context, tail, vocab_size);
    }
    std::cout << "captured prefix blob: " << blob.size() << " B, key=" << key
              << ", build_path_tag=" << header.build_path_tag
              << " weights_hash=" << header.weights_hash << "\n";

    // ── WARM (consumer): LOAD blob (no prefix prefill); prefill(turn); decode. ─
    Branch warm;
    {
        auto fp = create_forward_pass(model, &meta, kCtx, 1);
        CompatHeader expected = make_header(meta, fp->snapshot_kv_cache());
        PrefixLibrary lib(dir, expected);
        std::vector<uint8_t> loaded;
        if (!lib.try_load(key, loaded)) {
            std::cerr << "FAIL: warm load missed key " << key << "\n";
            return 3;
        }
        SessionManifest m;
        PrefixSections sec;
        build_sections(m, sec, fp.get());
        SnapshotReader r(loaded);
        m.restore(r, expected);  // memcpy KV(+recurrent) + set cursor to n_prefix

        if (fp->get_cache_pos(0) != n_prefix) {
            std::cerr << "FAIL: restored cursor expected " << n_prefix << ", got "
                      << fp->get_cache_pos(0) << "\n";
            return 3;
        }
        std::vector<float> logits =
            fp->run_prefill(turn, static_cast<int>(n_prefix), 0, sched);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());
        warm = run_decode(fp.get(), sched, context, tail, vocab_size);
    }

    auto print_seq = [](const char* t, const std::vector<int32_t>& s) {
        std::cout << "  " << t << ":";
        for (int32_t x : s) std::cout << ' ' << x;
        std::cout << "\n";
    };
    print_seq("REF (combined)", ref.seq);
    print_seq("LIVE(chunked) ", live.seq);
    print_seq("WARM(consumer)", warm.seq);

    bool gate1 = (warm.seq == live.seq) &&
                 logits_bit_equal(warm.first_logits, live.first_logits);
    bool gate2 = (warm.seq == ref.seq);
    double floor = max_abs_diff(warm.first_logits, ref.first_logits);

    std::cout << "GATE 1 [SHIP] (WARM==LIVE chunked-from-scratch, BYTE): "
              << (gate1 ? "PASS" : "FAIL (STOP — do not ship)")
              << " [seq=" << (warm.seq == live.seq ? "ok" : "DIFF") << ", logits="
              << (logits_bit_equal(warm.first_logits, live.first_logits)
                      ? "bit-identical" : "DIFF") << "]\n";
    std::cout << "GATE 2 [info] (WARM==REF combined-batch prefill, token-stable): "
              << (gate2 ? "PASS" : "DIFF (mm-vs-mv batch-shape fork, not a cache bug)")
              << "\n";
    std::cout << "[noise floor] WARM-vs-REF(combined) logits max_abs_diff = " << floor
              << " (combined != chunked on Metal; strict-bitwise DISABLED)\n";

    // GATE 3 — a mismatched header is refused fail-loud by the library.
    bool gate3 = false;
    {
        CompatHeader bad = header;
        bad.build_path_tag ^= 0x1ull;
        PrefixLibrary badlib(dir, bad);
        std::vector<uint8_t> out;
        try {
            badlib.try_load(key, out);
        } catch (const std::exception& e) {
            gate3 = std::string(e.what()).find("build_path_tag") != std::string::npos;
        }
    }
    std::cout << "GATE 3 (mismatched header refused fail-loud): "
              << (gate3 ? "PASS" : "FAIL") << "\n";

    // Ship gate = fidelity (WARM==LIVE) + refusal. GATE 2 is informational (the
    // combined-vs-chunked prefill fork is orthogonal to the cache).
    bool ok = gate1 && gate3;
    std::cout << (ok ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return ok ? 0 : 3;
}
