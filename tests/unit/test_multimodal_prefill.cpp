// test_multimodal_prefill.cpp — Phase 5 of docs/plan-gemma-vision-impl.md.
//
// Gate: SYNTHETIC ORCHESTRATOR CONTROL FLOW (owner decision). The orchestrator
// prefill_multimodal() is the one top-level site bridging the vision subsystem
// and the gemma3 recipe. Its *components* are already anchored on real models:
//   - encoder output vs llama.cpp reference  → test_vision_encoder.cpp (P2.4)
//   - embedding substitution into the stream → test_gemma3_image_substitution (P3)
//   - image-span bidirectional mask          → test_attn_mask_input + P4 wiring
// What is NOT yet anchored is the orchestrator's own control flow, so that is
// what this gate covers, WITHOUT the ~8 GB medgemma-4B load + ~11 min real
// encode the full e2e composition would require (deferred; needs a captured
// llama.cpp next-token reference that does not exist yet).
//
// The three branches:
//   (a) no images          → plain text prefill, identical to run_prefill
//   (b) more than one image → fail-loud (Phase 5 is single-image; Phase 7)
//   (c) null bitmap         → fail-loud
//
// Construction note: the SiglipEncoder is built from the mmproj (its ctor only
// validates against the projection weight — cheap, no forward pass) and is
// never encode()d here, so the gemma-3-1b text model (embed 1152) and the
// 2560-dim mmproj coexist fine; the single-image path that would bridge them is
// the deferred e2e case, not exercised.
//
// Self-skips unless BOTH a Gemma 3 text model and the mmproj are present.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "engine/model.h"
#include "engine/multimodal_prefill.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/gemma3.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/image/image_loader.h"
#include "../../src/image/image_prompt.h"
#include "../../src/vision/bitmap.h"
#include "../../src/vision/siglip_encoder.h"
#include "../../src/vision/vision_loader.h"
#include "../../src/vision/vision_model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef QINF_TEST_HAS_METAL
#include "ggml-metal.h"
#endif

using qinf::vision::Bitmap;
using qinf::vision::SiglipEncoder;
using qinf::vision::VisionLoader;
using qinf::vision::VisionModel;

namespace {

std::string gemma3_path() {
    if (const char* e = std::getenv("GEMMA3_MODEL_PATH"))
        if (e[0]) return std::string(e);
    return "gemma-3-1b-it-BF16.gguf";
}
std::string mmproj_path() {
    if (const char* e = std::getenv("QINF_MMPROJ_PATH"))
        if (e[0]) return std::string(e);
    return "./mmproj-BF16.gguf";
}
bool exists(const std::string& p) { std::ifstream f(p); return f.good(); }

// End-to-end gate model: must be the mmproj-COMPATIBLE Gemma 3 text model
// (embed 2560), not the gemma-3-1b orchestrator-control default (1152, which
// cannot host this mmproj). Defaults to the on-disk medgemma checkpoint.
std::string e2e_model_path() {
    if (const char* e = std::getenv("QINF_E2E_MODEL_PATH"))
        if (e[0]) return std::string(e);
    return "medgemma-1.5-4b-it-BF16.gguf";
}
std::string e2e_image_path() {
    if (const char* e = std::getenv("QINF_E2E_IMAGE_PATH"))
        if (e[0]) return std::string(e);
    return "./Fracturedribsmarked.jpg";
}

// Encoder backend for the e2e gate. CPU by default (slow but portable, ~minutes
// for one 896² SigLIP encode); QINF_TEST_METAL runs it on Metal. Mirrors
// test_vision_encoder.cpp::make_test_backend.
ggml_backend_t make_encoder_backend() {
#ifdef QINF_TEST_HAS_METAL
    if (const char* e = std::getenv("QINF_TEST_METAL"); e && e[0])
        return ggml_backend_metal_init();
#endif
    return ggml_backend_cpu_init();
}

#define SKIP_IF_MISSING_MODELS()                                            \
    do {                                                                    \
        if (!exists(gemma3_path()))                                         \
            GTEST_SKIP() << "Gemma 3 text model not found at "             \
                         << gemma3_path();                                  \
        if (!exists(mmproj_path()))                                         \
            GTEST_SKIP() << "mmproj not found at " << mmproj_path();        \
    } while (0)

#define SKIP_IF_MISSING_E2E()                                               \
    do {                                                                    \
        if (!exists(e2e_model_path()))                                      \
            GTEST_SKIP() << "e2e Gemma 3 (mmproj-compatible) model not "    \
                            "found at " << e2e_model_path();                \
        if (!exists(mmproj_path()))                                         \
            GTEST_SKIP() << "mmproj not found at " << mmproj_path();        \
        if (!exists(e2e_image_path()))                                      \
            GTEST_SKIP() << "e2e image not found at " << e2e_image_path();  \
    } while (0)

const std::vector<int32_t> kTokens = {2, 1841, 603, 573, 9876, 235292, 108, 17};

// Holds a loaded text recipe + a (constructed-but-never-encoded) vision encoder
// + the CPU backend the encoder borrows. One per test, RAII teardown.
struct Rig {
    Model model;
    std::unique_ptr<ForwardPassBase> fp;
    Gemma3ForwardPass* g3 = nullptr;

    VisionModel vmodel;
    VisionLoader vloader;
    ggml_backend_t vbackend = nullptr;
    std::unique_ptr<SiglipEncoder> encoder;

    // model_path defaults to the orchestrator-control gemma-3-1b; the e2e gate
    // passes the mmproj-compatible model. context_len must cover the image
    // block (boi + 256 soft + eoi + prompt) for the e2e path.
    void load(const std::string& model_path = gemma3_path(),
              bool metal_encoder = false,
              uint32_t context_len = 128,
              bool allow_multimodal = false) {
        register_builtin_models();
        model.load_metadata(model_path, allow_multimodal);
        model.load_tensors();
        const ModelMetadata& meta = model.get_metadata();
        fp = create_forward_pass(model, &meta, context_len, /*max_batch=*/1);
        g3 = dynamic_cast<Gemma3ForwardPass*>(fp.get());

        vloader.parse_metadata(mmproj_path(), vmodel);
        vbackend = metal_encoder ? make_encoder_backend()
                                 : ggml_backend_cpu_init();
        vloader.load_tensors(vmodel, vbackend);
        encoder = std::make_unique<SiglipEncoder>(
            vmodel, vbackend, vmodel.config().projection_dim);
    }
    ~Rig() {
        encoder.reset();
        if (vbackend) ggml_backend_free(vbackend);
    }
};

}  // namespace

// ── (a) No images → plain text prefill (passthrough identity) ────────────────

TEST(MultimodalPrefill, EmptyImagesEqualsTextPrefill) {
    SKIP_IF_MISSING_MODELS();
    Rig r; r.load();
    ASSERT_NE(r.g3, nullptr);

    std::vector<float> baseline =
        r.fp->run_prefill(kTokens, 0, 0, r.model.get_scheduler());
    std::vector<float> orchestrated = prefill_multimodal(
        *r.g3, *r.encoder, r.model.get_scheduler(), kTokens, /*images=*/{}, 0, 0);

    ASSERT_EQ(orchestrated.size(), baseline.size());
    for (size_t i = 0; i < baseline.size(); ++i)
        ASSERT_EQ(orchestrated[i], baseline[i])
            << "empty-images orchestrator diverged from run_prefill at " << i;
}

// ── (b) More than one image → fail-loud (single-image scope) ─────────────────

TEST(MultimodalPrefill, RejectsMultipleImages) {
    SKIP_IF_MISSING_MODELS();
    Rig r; r.load();
    ASSERT_NE(r.g3, nullptr);

    Bitmap a, b;
    std::vector<ImagePromptChunk> two = {{&a, 1}, {&b, 4}};
    EXPECT_THROW(prefill_multimodal(*r.g3, *r.encoder, r.model.get_scheduler(),
                                    kTokens, two, 0, 0),
                 std::runtime_error);
}

// ── (c) Null bitmap → fail-loud ──────────────────────────────────────────────

TEST(MultimodalPrefill, RejectsNullBitmap) {
    SKIP_IF_MISSING_MODELS();
    Rig r; r.load();
    ASSERT_NE(r.g3, nullptr);

    std::vector<ImagePromptChunk> one = {{nullptr, 2}};
    EXPECT_THROW(prefill_multimodal(*r.g3, *r.encoder, r.model.get_scheduler(),
                                    kTokens, one, 0, 0),
                 std::runtime_error);
}

// ── (d) END-TO-END gate: bidirectional image attention is coherent ───────────
//
// The real composition the file header deferred: load the mmproj-compatible
// Gemma 3 model, encode a real image, run the FULL bidi prefill, and assert the
// first sampled token (argmax @ temp 0) is a real token — NOT an image control
// token. This gates the *effect* of the image-span bidirectional mask, which
// the value-level tests (test_attn_mask_input) cannot: with bidi armed, the
// model previously degenerated into emitting <start_of_image> forever. Root
// cause was global-layer linear RoPE scaling (gemma3.rope.scaling.factor) being
// dropped (freq_scale=1.0 instead of 0.125) — causal hid it, bidi amplified it
// into incoherence. A regression in either freq_scale wiring OR the bidi
// default flips the argmax back to <start_of_image> and trips this gate.
//
// Heavy + opt-in: ~8 GB model load + one SigLIP encode (CPU minutes; set
// QINF_TEST_METAL=1 for Metal). Self-skips unless the e2e model + mmproj + a
// real image are all present.
TEST(MultimodalPrefill, BidiImageAttentionProducesNonImageNextToken) {
    SKIP_IF_MISSING_E2E();
    Rig r;
    r.load(e2e_model_path(), /*metal_encoder=*/true, /*context_len=*/512,
           /*allow_multimodal=*/true);
    ASSERT_NE(r.g3, nullptr);

    Tokenizer* tok = r.model.get_tokenizer();
    ASSERT_NE(tok, nullptr);
    const auto& vocab = tok->get_vocabulary();
    auto find_id = [&](const std::string& s) -> int32_t {
        for (size_t i = 0; i < vocab.size(); ++i)
            if (vocab[i] == s) return static_cast<int32_t>(i);
        return -1;
    };
    const int32_t boi  = find_id("<start_of_image>");
    const int32_t eoi  = find_id("<end_of_image>");
    const int32_t soft = find_id("<image_soft_token>");
    ASSERT_GE(boi, 0);
    ASSERT_GE(eoi, 0);
    ASSERT_GE(soft, 0)
        << "model vocab lacks the Gemma 3 image control tokens";

    const int32_t  bos   = r.model.get_metadata().bos_token_id;
    const uint32_t n_img = r.encoder->mm_tokens_per_image();

    // Faithful Gemma 3 image turn — MUST end at the assistant-turn boundary
    // (<start_of_turn>model\n), because that is exactly where the pre-fix bug
    // emitted <start_of_image>. A bare "...<eoi>Describe this image." sequence
    // does NOT reproduce it (the correctly-scaled local layers dominate a
    // mid-sentence prediction), so the generation point has to match cli/chat
    // .cpp's rendered turn. encode() recognizes the <…> control tokens and
    // emits one <start_of_image>, which expand_image_markers splices into
    // <start_of_image> soft×n_img <end_of_image>.
    const std::string rendered =
        "<start_of_turn>user\n\n\n<start_of_image>\n\n"
        "Describe this image.<end_of_turn>\n<start_of_turn>model\n";
    std::vector<int32_t> pre = tok->encode(rendered);
    pre.insert(pre.begin(), bos);
    auto expanded =
        qinf::image::expand_image_markers(pre, boi, soft, eoi, n_img);

    qinf::vision::Bitmap bmp =
        qinf::image::load_image_to_bitmap(e2e_image_path(), 896);
    std::vector<ImagePromptChunk> chunks = {{&bmp, expanded.span_start}};

    std::vector<float> logits = prefill_multimodal(
        *r.g3, *r.encoder, r.model.get_scheduler(),
        expanded.tokens, chunks, /*pos=*/0, /*slot=*/0);

    // Last vocab_size floats = final-position logits (mirrors cli/chat.cpp).
    const size_t vocab_size = r.model.get_metadata().vocab_size;
    ASSERT_GE(logits.size(), vocab_size);
    const float* last = logits.data() + (logits.size() - vocab_size);
    int32_t argmax = 0;
    for (size_t i = 1; i < vocab_size; ++i)
        if (last[i] > last[argmax]) argmax = static_cast<int32_t>(i);

    EXPECT_NE(argmax, boi)
        << "bidi degenerated to <start_of_image> — freq_scale / bidi regression";
    EXPECT_NE(argmax, soft)  << "first token is an image soft token";
    EXPECT_NE(argmax, eoi)   << "first token is <end_of_image>";
}
