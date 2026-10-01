// test_image_prefix_roundtrip.cpp — Vision V2 image-prefix KV blob falsifier
// (model-based, standalone). docs/plan-session-snapshot.md, Vision V2.
//
// V2 = Phase 4's prefix library with the cache key extended to include the
// IMAGE. The cacheable region is "everything up to and including the image
// soft-token span" — fixed across questions about the SAME image — and the
// per-question TEXT after the image is the variable suffix. A warm image-prefix
// blob is that slot's KV (+ recurrent) AFTER the chunked [text-prefix | image]
// prefill. On a hit a later question LOADS the blob (skipping BOTH the ViT
// encode and the image-position prefill) and prefills only its own text.
//
// This harness is the gate that decides whether that substitution is safe. It
// reuses Phase 4's machinery unchanged — capture_slot / restore_slot /
// PrefixLibrary — adding only the image-extended key and the capture/restore
// taken at the post-image boundary (the seam-reuse invariant: V2 hosts without
// bending the existing modules).
//
// Three branches over one fixed (preceding-context, image, question):
//   REF  — fresh fp: prefill_multimodal[preceding | image | question] in ONE
//          call (the production CLI image-turn path), decode.        (production)
//   LIVE — fresh fp: prefill_multimodal[preceding | image] (no question);
//          CAPTURE the post-image blob; prefill(question); decode.   (producer)
//   WARM — fresh fp: LOAD blob (memcpy KV, NO encode, NO image prefill);
//          prefill(question); decode.                                (consumer)
//
// Decode is a DETERMINISTIC argmax + repetition penalty (NO sampler RNG), so two
// branches with byte-identical logits ALWAYS agree and a wrong image-prefix KV
// actually diverges. Same rule as test_prefix_library_roundtrip.cpp.
//
// Gates:
//   GATE 1 — SHIP GATE — substitutability (BYTE): WARM == LIVE. The warm
//     image-prefix blob is a perfect substitute for [encode + image prefill].
//     Identical post-image KV bytes + identical question prefill ⇒ byte-identical
//     logits and tokens. If RED: STOP — the serialize/restore around the image
//     span is wrong (or the forward pass is nondeterministic past the floor).
//   GATE 2 — also strong — LIVE == REF (BYTE expected): splitting the production
//     single call into [text-prefix | image] + [question] does not change the
//     result (the image chunk's KV is identical whether fed head-less or as the
//     last chunk; the question is its own batch in both). If byte, WARM == REF
//     too ⇒ the cache reproduces the production path EXACTLY. Reported, not
//     ship-gating (a token-stable divergence here is the run_prefill-vs-
//     feed_tokens image-chunk seam, not a cache bug).
//   GATE 3 — a mismatched header is refused fail-loud by the library + manifest
//     (never silently re-prefilled — the F9 rule).
//
// Usage: test-image-prefix-roundtrip <model.gguf> <mmproj.gguf>
// Cross-family (CLAUDE.md): a Gemma3 (SigLIP, bidirectional) AND a Gemma4
// (gemma4uv, causal) image path.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

#include "engine/model.h"
#include "engine/multimodal_prefill.h"
#include "session/prefix_library.h"
#include "loader/tokenizer.h"
#include "models/forward_pass_base.h"
#include "models/model_registry.h"
#include "session/compat_header.h"
#include "session/session_manifest.h"
#include "session/slot_snapshot.h"
#include "session/snapshot_io.h"
#include "state/deltanet_state.h"
#include "state/kv_cache_simple.h"
#include "vision/bitmap.h"
#include "vision/i_vision_encoder.h"
#include "vision/vision_loader.h"
#include "vision/vision_model.h"
#include "vision/vision_profile.h"

namespace {

using qinf::session::CompatHeader;

constexpr uint32_t kCtx = 1024;
constexpr int kDecodeN = 12;
constexpr float kRepPenalty = 1.3f;

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

// build_path_tag over ALL of the recipe's KV caches (Gemma4 has two). A fold
// that returns the single path_tag unchanged for one cache, so single-cache
// recipes are byte-identical to the Phase 4 header.
uint64_t combined_path_tag(const std::vector<simple_kv_cache*>& caches) {
    uint64_t tag = 0;
    bool first = true;
    for (const simple_kv_cache* c : caches) {
        const uint64_t pt = c->path_tag();
        tag = first ? pt : (tag * 1099511628211ull) ^ pt;
        first = false;
    }
    return tag;
}

CompatHeader make_header(const ModelMetadata& m,
                         const std::vector<simple_kv_cache*>& caches) {
    CompatHeader h;
    h.arch_id = static_cast<uint32_t>(std::hash<std::string>{}(m.architecture));
    h.weights_hash = m.weights_hash;
    h.block_count = m.block_count;
    h.embedding_length = m.embedding_length;
    h.vocab_size = m.vocab_size;
    h.build_path_tag = combined_path_tag(caches);
    return h;
}

struct Branch {
    std::vector<int32_t> seq;
    std::vector<float> first_logits;  // the question-prefill tail (picks token 0)
};

// Decode kDecodeN tokens deterministically from a slot whose cursor is at the
// end of `context` and whose prefill tail is `prefill_tail`. All branches funnel
// through here so they differ ONLY in how they reached this post-question state.
Branch run_decode(ForwardPassBase* fp, ggml_backend_sched_t sched,
                  std::vector<int32_t> context,
                  const std::vector<float>& prefill_tail, size_t vocab_size) {
    Branch out;
    out.first_logits = prefill_tail;
    int32_t cur = pick(prefill_tail, context);
    out.seq.push_back(cur);
    context.push_back(cur);
    for (int i = 1; i < kDecodeN; ++i) {
        // The ROPE position, not the KV row count: identical for Gemma (one
        // position per row), the only correct one after an M-RoPE image span.
        int pos = fp->get_rope_pos(0);
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

// A synthetic, deterministic gray bitmap sized for the encoder family, with a
// producer-set content_id (the encoder is content-blind). SigLIP wants a square
// at image_size; gemma4uv wants multiples of the effective patch (48).
qinf::vision::Bitmap make_gray_bitmap(const qinf::vision::VisionModel& vmodel) {
    using PT = qinf::vision::VisionProjectorType;
    const auto& cfg = vmodel.config();
    qinf::vision::Bitmap bmp;
    bmp.channels = 3;
    if (cfg.projector_type == PT::Gemma3Siglip) {
        bmp.width = bmp.height = static_cast<int>(cfg.image_size);
    } else if (cfg.projector_type == PT::Qwen3VlMerger) {
        // 512×768 → 16×24 = 384 tokens; NON-square, so the M-RoPE span's
        // position advance (max(nx, ny) = 24) differs from both sides and from
        // its 384 rows — the case the snapshot's RPOS section exists for.
        bmp.width = 512;
        bmp.height = 768;
    } else {  // Gemma4Uv: 480×480 → 10×10 = 100 tokens, within budget
        bmp.width = bmp.height = 480;
    }
    bmp.pixels.assign(
        static_cast<size_t>(3) * bmp.width * bmp.height, 0.5f);
    bmp.content_id = 0x5EEDC0FFEEull;  // producer-set (chunk-list builder's job)
    return bmp;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <mmproj.gguf>\n";
        return 64;
    }
    ggml_backend_load_all();
    register_builtin_models();

    Model model;
    try {
        model.load_metadata(argv[1], /*allow_multimodal=*/true);
        model.load_tensors();
    } catch (const std::exception& e) {
        std::cerr << "load failed: " << e.what() << "\n";
        return 1;
    }
    const ModelMetadata& meta = model.get_metadata();
    Tokenizer* tok = model.get_tokenizer();
    const size_t vocab_size = meta.vocab_size;
    ggml_backend_sched_t sched = model.get_scheduler();

    // Vision encoder + a synthetic fixed image (own graph / scheduler; shared
    // backend, mirroring the CLI image path).
    ggml_backend_t backend = model.has_metal_backend() ? model.get_backend_metal()
                                                       : model.get_backend_cpu();
    qinf::vision::VisionModel vmodel;
    qinf::vision::VisionLoader vloader;
    try {
        vloader.parse_metadata(argv[2], vmodel);
        vloader.load_tensors(vmodel, backend);
    } catch (const std::exception& e) {
        std::cerr << "mmproj load failed: " << e.what() << "\n";
        return 1;
    }
    // The production projector dispatch: encoder, markers and soft-token id for
    // whichever family the mmproj is (gemma3-siglip, gemma4uv, qwen3vl-merger).
    qinf::vision::VisionProfile vprofile = qinf::vision::make_vision_profile(
        vmodel, backend, tok->get_vocabulary(), "test-image-prefix-roundtrip: argv[2]");
    qinf::vision::IVisionEncoder* encoder = vprofile.encoder.get();

    qinf::vision::Bitmap bmp = make_gray_bitmap(vmodel);
    const uint32_t n_img = encoder->mm_tokens_for(bmp);
    const int32_t soft_id = vprofile.soft_id;

    // The FIXED preceding context (everything before the image span — the V2
    // reuse condition) and the per-question variable TURN. The image-inclusive
    // prefix = preceding ++ N soft-token placeholders; the question follows.
    std::vector<int32_t> preceding = tok->encode("Look at this image.");
    if (meta.add_bos_token && meta.bos_token_id >= 0)
        preceding.insert(preceding.begin(), meta.bos_token_id);
    const int32_t span_start = static_cast<int32_t>(preceding.size());

    std::vector<int32_t> image_prefix = preceding;
    image_prefix.insert(image_prefix.end(), n_img, soft_id);  // N placeholders
    const uint32_t n_prefix = static_cast<uint32_t>(image_prefix.size());

    std::vector<int32_t> question = tok->encode(" Describe what is shown here.");

    std::vector<int32_t> full = image_prefix;  // production single-call tokens
    full.insert(full.end(), question.begin(), question.end());

    std::vector<int32_t> context = full;  // decode history starts at end of `full`

    std::cout << "=== image-prefix round-trip: " << meta.architecture << " ("
              << meta.model_name << ") projector=" << vprofile.projector_tag
              << " n_prefix=" << n_prefix << " (text=" << span_start
              << "+img=" << n_img << ") n_question=" << question.size() << " ===\n";

    const std::vector<ImagePromptChunk> chunks = {{&bmp, span_start}};

    // ── REF (production): the CLI image-turn path — one prefill_multimodal call
    //    over [preceding | image | question]. ────────────────────────────────────
    Branch ref;
    {
        auto fp = create_forward_pass(model, &meta, kCtx, 1);
        std::vector<float> logits = prefill_multimodal(
            *fp, *encoder, sched, full, chunks, 0, 0, nullptr, nullptr);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());
        ref = run_decode(fp.get(), sched, context, tail, vocab_size);
    }

    // ── LIVE (producer): chunked [preceding | image] (NO question), CAPTURE the
    //    post-image blob, then prefill(question), decode. ─────────────────────────
    const std::string dir = std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp")
                          + "/qinf_imgprefix_test";
    CompatHeader header;
    std::vector<uint8_t> blob;
    Branch live;
    int live_rope_pos = -1;   // the question's position after the image span
    const uint64_t key = PrefixLibrary::key_for(preceding, bmp.content_id);
    {
        auto fp = create_forward_pass(model, &meta, kCtx, 1);
        if (fp->snapshot_kv_caches().empty()) {
            std::cerr << "FAIL: recipe '" << meta.architecture
                      << "' has no snapshot_kv_caches accessor (needed for V2)\n";
            return 2;
        }
        header = make_header(meta, fp->snapshot_kv_caches());

        // Encode + chunked-prefill ONLY [preceding | image]; cursor → n_prefix.
        prefill_multimodal(*fp, *encoder, sched, image_prefix, chunks, 0, 0,
                           nullptr, nullptr);
        if (fp->get_cache_pos(0) != n_prefix) {
            std::cerr << "FAIL: post-image cursor expected " << n_prefix << ", got "
                      << fp->get_cache_pos(0) << "\n";
            return 2;
        }

        // Capture the warm image-prefix blob through the production API (header
        // + KV [+ recurrent] [+ RPOS for an M-RoPE span]) and publish it.
        blob = qinf::snapshot::capture_slot(*fp, 0, header);
        PrefixLibrary(dir, header).store(key, blob);
        live_rope_pos = fp->get_rope_pos(0);

        std::vector<float> logits =
            fp->run_prefill(question, live_rope_pos, 0, sched);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());
        live = run_decode(fp.get(), sched, context, tail, vocab_size);
    }
    std::cout << "post-image rows=" << n_prefix << " rope position=" << live_rope_pos << "\n";
    std::cout << "captured image-prefix blob: " << blob.size() << " B, key=" << key
              << ", build_path_tag=" << header.build_path_tag
              << " content_id=" << bmp.content_id << "\n";

    // ── WARM (consumer): LOAD blob (no encode, no image prefill); prefill(question). ─
    Branch warm;
    {
        auto fp = create_forward_pass(model, &meta, kCtx, 1);
        CompatHeader expected = make_header(meta, fp->snapshot_kv_caches());
        PrefixLibrary lib(dir, expected);
        std::vector<uint8_t> loaded;
        if (!lib.try_load(key, loaded)) {
            std::cerr << "FAIL: warm load missed key " << key << "\n";
            return 3;
        }
        qinf::snapshot::restore_slot(*fp, 0, loaded, expected);  // KV(+recurrent)(+RPOS)

        if (fp->get_cache_pos(0) != n_prefix) {
            std::cerr << "FAIL: restored cursor expected " << n_prefix << ", got "
                      << fp->get_cache_pos(0) << "\n";
            return 3;
        }
        if (fp->get_rope_pos(0) != live_rope_pos) {
            std::cerr << "FAIL: restored rope position expected " << live_rope_pos
                      << ", got " << fp->get_rope_pos(0) << "\n";
            return 3;
        }
        std::vector<float> logits =
            fp->run_prefill(question, fp->get_rope_pos(0), 0, sched);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());
        warm = run_decode(fp.get(), sched, context, tail, vocab_size);
    }

    auto print_seq = [](const char* t, const std::vector<int32_t>& s) {
        std::cout << "  " << t << ":";
        for (int32_t x : s) std::cout << ' ' << x;
        std::cout << "\n";
    };
    print_seq("REF (production)", ref.seq);
    print_seq("LIVE(split+cap) ", live.seq);
    print_seq("WARM(consumer)  ", warm.seq);

    bool gate1 = (warm.seq == live.seq) &&
                 logits_bit_equal(warm.first_logits, live.first_logits);
    bool gate2 = (live.seq == ref.seq) &&
                 logits_bit_equal(live.first_logits, ref.first_logits);
    double floor = max_abs_diff(warm.first_logits, ref.first_logits);

    std::cout << "GATE 1 [SHIP] (WARM==LIVE, image-prefix blob is a byte-exact "
                 "substitute): "
              << (gate1 ? "PASS" : "FAIL (STOP — do not ship)")
              << " [seq=" << (warm.seq == live.seq ? "ok" : "DIFF") << ", logits="
              << (logits_bit_equal(warm.first_logits, live.first_logits)
                      ? "bit-identical" : "DIFF") << "]\n";
    std::cout << "GATE 2 [strong] (LIVE==REF, split prefill == production single "
                 "call): "
              << (gate2 ? "PASS (byte)" : "DIFF (run_prefill-vs-feed_tokens image "
                                          "seam; not a cache bug)")
              << "\n";
    std::cout << "[noise floor] WARM-vs-REF logits max_abs_diff = " << floor << "\n";

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

    bool ok = gate1 && gate3;  // ship = substitutability + refusal; GATE 2 reported
    std::cout << (ok ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return ok ? 0 : 3;
}
