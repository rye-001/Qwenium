// test_gemma4_image_substitution.cpp — Gemma 4 recipe hook
// (docs/plan-gemma4-vision-impl.md §6, §8 gate 3).
//
// Gate: SELF-CONSISTENCY (reference-free), mirroring the Gemma 3 substitution
// test. Isolates the recipe-side wiring: armed image embeddings
//   (a) flow into the gemma4 prefill residual stream (output changes),
//   (b) are consumed-on-use (one prefill, then text-only again),
//   (c) are validated fail-loud (payload size + span bounds).
//
// The ONE structural difference from Gemma 3: image attention is plain causal —
// there is NO bidirectional image mask — so there is no per-layer mask-wiring
// test. The authoritative image→next-token check is the §8 e2e gate.
//
// Uses the dense gemma-4-12B-it text export (a text-only checkpoint: no mm.
// tensors, no vision_config keys, vocab placeholder is <|image> not the Gemma 3
// <image_soft_token> the loader guard trips on). Self-skips if absent.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/gemma4.h"

#include "ggml.h"
#include "ggml-backend.h"

namespace {

std::string get_gemma4_model_path() {
    if (const char* e = std::getenv("GEMMA4_MODEL_PATH"))
        if (e[0]) return std::string(e);
    return "gemma-4-12B-it-Q8_0.gguf";
}

bool file_exists(const std::string& p) { std::ifstream f(p); return f.good(); }

#define SKIP_IF_NO_MODEL()                                                  \
    do {                                                                    \
        if (!file_exists(get_gemma4_model_path()))                          \
            GTEST_SKIP() << "Gemma 4 text model not found at "              \
                         << get_gemma4_model_path()                         \
                         << " — set GEMMA4_MODEL_PATH to override";         \
    } while (0)

struct Loaded {
    Model model;
    std::unique_ptr<ForwardPassBase> fp;
    Gemma4ForwardPass* g4 = nullptr;
    uint32_t vocab = 0;
    uint32_t n_embd = 0;
};

void load(Loaded& L) {
    register_builtin_models();
    L.model.load_metadata(get_gemma4_model_path());
    L.model.load_tensors();
    const ModelMetadata& meta = L.model.get_metadata();
    L.vocab  = meta.vocab_size;
    L.n_embd = meta.embedding_length;
    L.fp = create_forward_pass(L.model, &meta, /*context_len=*/128, /*max_batch=*/1);
    L.g4 = dynamic_cast<Gemma4ForwardPass*>(L.fp.get());
}

// Synthetic [n_embd, n_img] block, n_embd fastest — the Gemma4UvEncoder layout.
std::vector<float> make_synthetic_embd(uint32_t n_embd, uint32_t n_img) {
    std::vector<float> v(static_cast<size_t>(n_embd) * n_img);
    for (uint32_t t = 0; t < n_img; ++t)
        for (uint32_t e = 0; e < n_embd; ++e)
            v[static_cast<size_t>(t) * n_embd + e] =
                0.01f * static_cast<float>((e + 7 * t) % 17) - 0.08f;
    return v;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

std::vector<float> last_logits(const std::vector<float>& logits, uint32_t vocab) {
    return std::vector<float>(logits.end() - vocab, logits.end());
}

const std::vector<int32_t> kTokens = {2, 1841, 603, 573, 9876, 235292, 108, 17};

}  // namespace

// ── Substitution flows into the residual stream and changes the output ───────
TEST(Gemma4ImageSubstitution, SubstitutedEmbeddingsChangeLogits) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g4, nullptr) << "create_forward_pass did not yield a Gemma4ForwardPass";

    std::vector<float> base = last_logits(
        L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()), L.vocab);
    EXPECT_TRUE(all_finite(base)) << "text-only baseline produced NaN/Inf";

    L.g4->set_image_embeddings(make_synthetic_embd(L.n_embd, 3), /*span_start=*/2, 3);
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
TEST(Gemma4ImageSubstitution, ArmingIsConsumedAfterOnePrefill) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g4, nullptr);

    std::vector<float> base = last_logits(
        L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()), L.vocab);

    L.g4->set_image_embeddings(make_synthetic_embd(L.n_embd, 3), 2, 3);
    (void) L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler());  // consumes arm

    std::vector<float> again = last_logits(
        L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()), L.vocab);
    for (uint32_t i = 0; i < L.vocab; ++i)
        ASSERT_EQ(again[i], base[i])
            << "prefill after consuming the image arm diverged from the "
            << "text-only baseline at logit " << i << " — arm not consumed";
}

// ── Fail-loud: payload size mismatch ─────────────────────────────────────────
TEST(Gemma4ImageSubstitution, RejectsWrongSizedPayload) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g4, nullptr);

    L.g4->set_image_embeddings(make_synthetic_embd(L.n_embd, 2), 1, 3);  // claims 3, gives 2
    EXPECT_THROW(L.fp->run_prefill(kTokens, 0, 0, L.model.get_scheduler()),
                 std::runtime_error);
}

// ── Regression: the substituted residual is pinned as a graph output ─────────
// Without ggml_set_output on inpL_image_subst, galloc reuses its buffer across
// the server's alternating graph shapes and the 2nd+ image request degenerates
// into token-soup (docs/server-image-multirequest-bug.md). The pin lives in the
// shared ForwardPassBase::build_image_substitution; assert it directly so a
// future "prune unused outputs" cleanup can't silently reintroduce the bug.
TEST(Gemma4ImageSubstitution, SubstitutedResidualIsPinnedAsOutput) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g4, nullptr);

    L.g4->set_image_embeddings(make_synthetic_embd(L.n_embd, 3), /*span_start=*/2, 3);
    ggml_cgraph* gf = L.fp->build_prefill_graph(kTokens, 0, 0);
    ggml_tensor* subst = ggml_graph_get_tensor(gf, "inpL_image_subst");
    ASSERT_NE(subst, nullptr)
        << "inpL_image_subst node absent — image substitution did not build";
    EXPECT_TRUE(subst->flags & GGML_TENSOR_FLAG_OUTPUT)
        << "inpL_image_subst is not pinned as a graph output — galloc may reuse "
        << "its buffer and corrupt multi-request image prefill";
}

// ── Fail-loud: span runs past the sequence ───────────────────────────────────
TEST(Gemma4ImageSubstitution, RejectsOutOfRangeSpan) {
    SKIP_IF_NO_MODEL();
    Loaded L; load(L);
    ASSERT_NE(L.g4, nullptr);

    const std::vector<int32_t> short_tokens = {2, 1841, 603, 573};  // 4 tokens
    L.g4->set_image_embeddings(make_synthetic_embd(L.n_embd, 3), 3, 3);  // 3+3 > 4
    EXPECT_THROW(L.fp->run_prefill(short_tokens, 0, 0, L.model.get_scheduler()),
                 std::runtime_error);
}
