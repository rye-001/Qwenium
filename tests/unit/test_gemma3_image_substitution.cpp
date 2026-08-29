// test_gemma3_image_substitution.cpp — Phase 3 of docs/plan-gemma-vision-impl.md.
//
// Gate: SELF-CONSISTENCY (reference-free). The plan defers the authoritative
// llama.cpp logits reference to the Phase 5 end-to-end test — capturing it here
// would require driving llama.cpp's full mtmd multi-batch decode (llama.cpp has
// no single mixed text+image graph; it injects image embeddings as a separate
// llama_decode batch), which IS the P5 path. P3 isolates the recipe-side
// wiring: that armed image embeddings
//   (a) flow into the gemma3 prefill residual stream (output changes),
//   (b) are consumed-on-use (one prefill, then text-only again),
//   (c) are validated fail-loud (payload size + span bounds).
//
// NOT asserted numerically here (grounded in code vs llama.cpp gemma3.cpp:93
//   `ggml_scale(inpL, ubatch.token ? sqrtf(n_embd) : 1.0f)`, verified e2e at P5):
//   that image rows enter UNSCALED and text positions stay bit-identical.
//
// Self-skips when no Gemma 3 TEXT model is available (GEMMA3_MODEL_PATH or the
// default below). The multimodal MedGemma checkpoint is refused by the loader
// (Phase 1), so this needs a text-only Gemma 3 GGUF.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/gemma3.h"

#include "ggml.h"
#include "ggml-backend.h"

namespace {

std::string get_gemma3_model_path() {
    if (const char* e = std::getenv("GEMMA3_MODEL_PATH"))
        if (e[0]) return std::string(e);
    return "gemma-3-1b-it-BF16.gguf";
}

bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

#define SKIP_IF_NO_MODEL()                                                  \
    do {                                                                    \
        if (!file_exists(get_gemma3_model_path()))                          \
            GTEST_SKIP() << "Gemma 3 text model not found at "              \
                         << get_gemma3_model_path()                         \
                         << " — set GEMMA3_MODEL_PATH to override";         \
    } while (0)

// Loads a Gemma 3 model + builds a Gemma3ForwardPass. Returns nullptr-ish via
// ASSERTs in the caller. Kept as a helper so each test is self-contained.
struct Loaded {
    Model model;
    std::unique_ptr<ForwardPassBase> fp;
    Gemma3ForwardPass* g3 = nullptr;   // non-owning view of fp (for set_image_*)
    uint32_t vocab = 0;
    uint32_t n_embd = 0;
};

void load(Loaded& L) {
    register_builtin_models();
    L.model.load_metadata(get_gemma3_model_path());
    L.model.load_tensors();
    const ModelMetadata& meta = L.model.get_metadata();
    L.vocab  = meta.vocab_size;
    L.n_embd = meta.embedding_length;
    L.fp = create_forward_pass(L.model, &meta, /*context_len=*/128, /*max_batch=*/1);
    L.g3 = dynamic_cast<Gemma3ForwardPass*>(L.fp.get());
}

// Deterministic synthetic image-embedding block, [n_embd, n_img] row-major
// (n_embd fastest) — the layout VisionEncoder::encode produces. Small bounded
// values so the substituted residual stays in a sane range.
std::vector<float> make_synthetic_embd(uint32_t n_embd, uint32_t n_img) {
    std::vector<float> v(static_cast<size_t>(n_embd) * n_img);
    for (uint32_t t = 0; t < n_img; ++t)
        for (uint32_t e = 0; e < n_embd; ++e)
            v[static_cast<size_t>(t) * n_embd + e] =
                0.01f * static_cast<float>((e + 7 * t) % 17) - 0.08f;
    return v;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// Last-token vocab slice from a (possibly multi-position) prefill logits buffer.
std::vector<float> last_logits(const std::vector<float>& logits, uint32_t vocab) {
    return std::vector<float>(logits.end() - vocab, logits.end());
}

const std::vector<int32_t> kTokens = {2, 1841, 603, 573, 9876, 235292, 108, 17};

}  // namespace

// ── Substitution flows into the residual stream and changes the output ───────

TEST(Gemma3ImageSubstitution, SubstitutedEmbeddingsChangeLogits) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g3, nullptr) << "create_forward_pass did not yield a Gemma3ForwardPass";

    std::vector<float> base = last_logits(
        L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()), L.vocab);
    EXPECT_TRUE(all_finite(base)) << "text-only baseline produced NaN/Inf";

    L.g3->set_image_embeddings(make_synthetic_embd(L.n_embd, 3), /*span_start=*/2, 3);
    std::vector<float> out = last_logits(
        L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()), L.vocab);
    EXPECT_TRUE(all_finite(out)) << "image-substituted prefill produced NaN/Inf";

    size_t differing = 0;
    for (uint32_t i = 0; i < L.vocab; ++i)
        if (out[i] != base[i]) ++differing;
    EXPECT_GT(differing, 0u)
        << "image-substituted logits identical to text-only baseline — the "
        << "armed embeddings did not reach the residual stream";
}

// ── Consume-on-use: a second prefill without re-arming is text-only again ─────

TEST(Gemma3ImageSubstitution, ArmingIsConsumedAfterOnePrefill) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g3, nullptr);

    std::vector<float> base = last_logits(
        L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()), L.vocab);

    L.g3->set_image_embeddings(make_synthetic_embd(L.n_embd, 3), 2, 3);
    (void) L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler());  // consumes arm

    std::vector<float> again = last_logits(
        L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()), L.vocab);
    for (uint32_t i = 0; i < L.vocab; ++i)
        ASSERT_EQ(again[i], base[i])
            << "prefill after consuming the image arm diverged from the "
            << "text-only baseline at logit " << i << " — arm not consumed";
}

// ── Fail-loud: payload size mismatch ─────────────────────────────────────────

TEST(Gemma3ImageSubstitution, RejectsWrongSizedPayload) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g3, nullptr);

    // Claims n_img=3 but supplies only 2 tokens' worth of data.
    L.g3->set_image_embeddings(make_synthetic_embd(L.n_embd, 2), 1, 3);
    EXPECT_THROW(L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()),
                 std::runtime_error);
}

// ── Fail-loud: span runs past the sequence ───────────────────────────────────

TEST(Gemma3ImageSubstitution, RejectsOutOfRangeSpan) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g3, nullptr);

    const std::vector<int32_t> short_tokens = {2, 1841, 603, 573};  // 4 tokens
    // span_start=3 + n_img=3 = 6 > 4 tokens.
    L.g3->set_image_embeddings(make_synthetic_embd(L.n_embd, 3), 3, 3);
    EXPECT_THROW(L.fp->run_prefill(short_tokens, 0, 0, L.model.get_scheduler()),
                 std::runtime_error);
}

// ── Phase 4: gemma3 builds the per-layer bidi mask with the right span ────────
//
// Recipe-level wiring check (the unit-level mask formula is gated in
// test_attn_mask_input.cpp). Arms an image span, builds the prefill graph, and
// reads back EVERY layer's "kq_mask.{il}" tensor — proving gemma3 (a) converts
// the local span index to absolute (pos + span_start) and (b) passes the bidi
// span to every layer. Because Gemma 3 is 5:1 local:global, asserting on all
// layers covers both window kinds. The window itself does not bite this 8-token
// sequence, so the discriminating signal is the bidi relaxation: an image query
// attends to a LATER image key (impossible under a causal mask) on every layer.
//
// NOTE: the image-span bidi mask is DEFAULT-OFF in gemma3 (causal) because it
// corrupts real multimodal generation end-to-end — a known, unresolved bug (see
// docs/gemma-vision-handoff.md). This test still validates the bidi WIRING (the
// recipe builds the per-layer mask with the right span) for the eventual fix, so
// it opts the span back in via QINF_GEMMA3_IMAGE_BIDI. It is a value/wiring gate,
// NOT an end-to-end correctness gate — the latter is what is currently broken.
TEST(Gemma3ImageSubstitution, PerLayerBidiMaskWiring) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g3, nullptr);

    setenv("QINF_GEMMA3_IMAGE_BIDI", "1", 1);  // exercise the (default-off) bidi path

    const uint32_t span = 2, n_img = 3;  // image positions 2,3,4 (pos=0)
    L.g3->set_image_embeddings(make_synthetic_embd(L.n_embd, n_img), span, n_img);

    auto* sched = L.model.get_scheduler();
    ggml_backend_sched_reset(sched);
    ggml_cgraph* gf = L.g3->build_prefill_graph(kTokens, /*pos=*/0, 0);
    ASSERT_EQ(ggml_backend_sched_alloc_graph(sched, gf), true);
    L.g3->set_prefill_inputs(gf, kTokens, 0);  // populates the mask inputs

    const uint32_t n_tok = static_cast<uint32_t>(kTokens.size());  // 8
    const uint32_t n_kv  = n_tok;                                  // pos=0
    const uint32_t n_layers = L.model.get_metadata().block_count;
    ASSERT_GT(n_layers, 0u);

    uint32_t layers_seen = 0;
    for (uint32_t il = 0; il < n_layers; ++il) {
        std::string name = "kq_mask." + std::to_string(il);
        ggml_tensor* m = ggml_graph_get_tensor(gf, name.c_str());
        ASSERT_NE(m, nullptr) << "missing mask tensor " << name;
        ASSERT_EQ(static_cast<uint32_t>(m->ne[0]), n_kv);
        std::vector<float> mask(static_cast<size_t>(n_kv) * n_tok);
        ggml_backend_tensor_get(m, mask.data(), 0, mask.size() * sizeof(float));
        auto cell = [&](uint32_t q, uint32_t j) { return mask[q * n_kv + j]; };

        // Bidi within span [2,5): image query attends a LATER image key.
        EXPECT_FLOAT_EQ(cell(2, 4), 0.0f) << name << " (q=2 future img key j=4)";
        EXPECT_FLOAT_EQ(cell(2, 3), 0.0f) << name;
        EXPECT_FLOAT_EQ(cell(4, 2), 0.0f) << name << " (reverse)";
        // Causality preserved outside the span: text query never sees the future.
        EXPECT_EQ(cell(0, 5), -INFINITY) << name << " (text q=0, future key)";
        // Image query does NOT see text after the span closes.
        EXPECT_EQ(cell(4, 7), -INFINITY) << name << " (img q=4, future text j=7)";
        // Text after the span attends causally back to the whole image span.
        EXPECT_FLOAT_EQ(cell(7, 2), 0.0f) << name;
        ++layers_seen;
    }
    EXPECT_EQ(layers_seen, n_layers);

    unsetenv("QINF_GEMMA3_IMAGE_BIDI");  // don't leak into other tests
}
