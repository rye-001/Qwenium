// test_vision_loader.cpp — Phase 2.1 of docs/plan-gemma-vision-impl.md.
//
// Opens the Gemma 3 mmproj GGUF, asserts SigLIPConfig fields populate from
// the GGUF metadata, and asserts the critical tensors are present with the
// expected shapes. Self-skips when the mmproj file is absent (mirrors the
// SKIP_IF_NO_MODEL pattern used by test_qwen36_feed_tokens).
//
// Phase 2.1 deliverable gate: this file. Phase 2.3 will add a separate
// encoder-forward differential (`test_vision_encoder.cpp`) gated against
// `tests/fixtures/vision/siglip_gray_896.bin`.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../../src/vision/vision_loader.h"
#include "../../src/vision/vision_model.h"
#include "../../src/vision/vision_profile.h"

using qinf::vision::VisionLoader;
using qinf::vision::VisionModel;

namespace {

std::string get_mmproj_path() {
    if (const char* e = std::getenv("QINF_MMPROJ_PATH"))
        if (e[0]) return std::string(e);
    return "./mmproj-BF16.gguf";
}

#define SKIP_IF_NO_MMPROJ()                                                \
    do {                                                                   \
        FILE* _f = std::fopen(get_mmproj_path().c_str(), "rb");            \
        if (!_f) GTEST_SKIP() << "mmproj GGUF not found at "                \
                              << get_mmproj_path()                         \
                              << " — skipping (set QINF_MMPROJ_PATH to "   \
                                 "override)";                              \
        std::fclose(_f);                                                   \
    } while (0)


// The Qwen 3.5-family mmproj (P1 of docs/plan-qwen35-vision-impl.md). Separate
// env var because it is a different file from the Gemma 3 one above.
std::string get_qwen_mmproj_path() {
    if (const char* e = std::getenv("QINF_QWEN_MMPROJ_PATH"))
        if (e[0]) return std::string(e);
    return "./models/Qwen3.6-mtp-mmproj-BF16.gguf";
}

#define SKIP_IF_NO_QWEN_MMPROJ()                                              \
    do {                                                                      \
        FILE* _f = std::fopen(get_qwen_mmproj_path().c_str(), "rb");          \
        if (!_f) GTEST_SKIP() << "Qwen mmproj GGUF not found at "             \
                              << get_qwen_mmproj_path()                       \
                              << " — skipping (set QINF_QWEN_MMPROJ_PATH to " \
                                 "override)";                                 \
        std::fclose(_f);                                                      \
    } while (0)

}  // namespace

// ── parse_metadata populates SigLIPConfig from clip.vision.* keys ────────────

TEST(VisionLoader, ParsesGemma3MmprojConfig) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_mmproj_path(), model);

    const auto& cfg = model.config();

    // SigLIP-So400M at Gemma 3 dimensions, verified by GGUF inspection of
    // ./mmproj-BF16.gguf:
    EXPECT_EQ(cfg.image_size,        896u);
    EXPECT_EQ(cfg.patch_size,        14u);
    EXPECT_EQ(cfg.num_channels,      3u);
    EXPECT_EQ(cfg.hidden_size,       1152u);
    EXPECT_EQ(cfg.num_layers,        27u);
    EXPECT_EQ(cfg.num_attn_heads,    16u);
    EXPECT_EQ(cfg.projection_dim,    2560u);   // = Gemma 3 4B embedding_length
    EXPECT_EQ(cfg.mm_tokens_per_image, 256u);  // derived: (896/14/4)²
    EXPECT_EQ(cfg.pool_factor,       4u);
    EXPECT_GT(cfg.intermediate_size, 0u);      // exact value not part of contract
    EXPECT_GT(cfg.layer_norm_eps,    0.0f);
    EXPECT_LT(cfg.layer_norm_eps,    1.0f);    // sanity

    EXPECT_EQ(model.mmproj_path(), get_mmproj_path());
}

// ── Required tensor presence + shape ─────────────────────────────────────────

TEST(VisionLoader, ParseMetadataValidatesProjectionShape) {
    SKIP_IF_NO_MMPROJ();
    // The shape cross-checks live inside parse_metadata and throw on
    // mismatch. We simply confirm parse_metadata succeeds on the real
    // mmproj — i.e. the declared projection_dim (2560) and hidden_size
    // (1152) agree with mm.input_projection.weight's shape.
    VisionModel model;
    VisionLoader loader;
    EXPECT_NO_THROW(loader.parse_metadata(get_mmproj_path(), model));
}

// ── load_tensors populates the weight map on a CPU backend ───────────────────

TEST(VisionLoader, LoadTensorsPopulatesWeightMapOnCPU) {
    SKIP_IF_NO_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_mmproj_path(), model);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ASSERT_NE(backend, nullptr) << "ggml_backend_cpu_init failed";

    loader.load_tensors(model, backend);

    // 439 tensors per GGUF inspection: 27 layers × 16 + 5 (mm.input_projection,
    // mm.soft_emb_norm, v.patch_embd.{weight,bias}, v.position_embd.weight) + 2
    // (v.post_ln.{weight,bias}) = 439.
    EXPECT_EQ(model.tensors().size(), 439u);

    // Critical weight handles non-null and reachable by name.
    const auto& tensors = model.tensors();
    auto require = [&](const std::string& name) -> ggml_tensor* {
        auto it = tensors.find(name);
        EXPECT_NE(it, tensors.end()) << "tensor missing: " << name;
        return (it == tensors.end()) ? nullptr : it->second;
    };
    EXPECT_NE(require("mm.input_projection.weight"), nullptr);
    EXPECT_NE(require("mm.soft_emb_norm.weight"),    nullptr);
    EXPECT_NE(require("v.patch_embd.weight"),        nullptr);
    EXPECT_NE(require("v.patch_embd.bias"),          nullptr);
    EXPECT_NE(require("v.position_embd.weight"),     nullptr);
    EXPECT_NE(require("v.post_ln.weight"),           nullptr);
    EXPECT_NE(require("v.post_ln.bias"),             nullptr);

    // Spot-check every layer index exists for one tensor kind.
    for (uint32_t il = 0; il < model.config().num_layers; ++il) {
        const std::string key = "v.blk." + std::to_string(il) + ".attn_q.weight";
        EXPECT_NE(require(key), nullptr) << key;
    }

    // mm.input_projection.weight: ggml ne = [projection_dim, hidden_size, 1, 1]
    auto* proj = require("mm.input_projection.weight");
    ASSERT_NE(proj, nullptr);
    EXPECT_EQ(proj->ne[0], 2560);
    EXPECT_EQ(proj->ne[1], 1152);

    // VisionModel owns the context + buffer (destructor frees both).
    EXPECT_NE(model.weight_context(),  nullptr);
    EXPECT_NE(model.backend_buffer(),  nullptr);

    // Cleanup: model destructor frees its ctx/buffer; we still own the
    // backend handle.
    ggml_backend_free(backend);
}

// ── P1 gate: the Qwen 3.5-family mmproj parses ───────────────────────────────
//
// This is the first step of the vision port that tests the plan against the
// file rather than against static analysis. Every number below was read off
// models/Qwen3.6-mtp-mmproj-BF16.gguf on 2026-08-25 and matches §3.6 of
// docs/plan-qwen35-vision-impl.md.

TEST(VisionLoader, ParsesQwen3VlMergerMmprojConfig) {
    SKIP_IF_NO_QWEN_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_qwen_mmproj_path(), model);

    const auto& cfg = model.config();
    EXPECT_EQ(cfg.projector_type,
              qinf::vision::VisionProjectorType::Qwen3VlMerger);

    // image_size is the POSITION-EMBEDDING GRID (768/16 = 48 ⇒ 48² = 2304
    // entries), not a fixed input size — resolution is dynamic.
    EXPECT_EQ(cfg.image_size,        768u);
    EXPECT_EQ(cfg.patch_size,        16u);
    EXPECT_EQ(cfg.hidden_size,       1152u);
    EXPECT_EQ(cfg.num_layers,        27u);
    EXPECT_EQ(cfg.num_attn_heads,    16u);
    EXPECT_EQ(cfg.intermediate_size, 4304u);
    EXPECT_FLOAT_EQ(cfg.layer_norm_eps, 1e-6f);

    // The 2×2 merge comes from clip.vision.spatial_merge_size — NOT from
    // projector.scale_factor, which is the Gemma 4 key and is absent here.
    EXPECT_EQ(cfg.n_merge,     2u);
    EXPECT_EQ(cfg.pool_factor, 1u);

    // Seam A's invariant: projection_dim equals the host text model's
    // embedding_length. 2048 == Qwen3.6-35B-A3B. Note llama.cpp computes
    // projection_dim × spatial_merge_size² = 8192 here and refuses to load
    // (ggml-org/llama.cpp#20899, closed "not planned"); mm.2.weight is
    // [4608, 2048], so 2048 is the correct answer and 8192 is their bug.
    EXPECT_EQ(cfg.projection_dim, 2048u);

    // Dynamic resolution ⇒ per-image token count, decided at encode time.
    EXPECT_EQ(cfg.mm_tokens_per_image, 0u);

    // Read from clip.vision.image_{mean,std} — this projector normalizes like
    // SigLIP (0.5/0.5), unlike gemma4uv's [0,0,0]/[1,1,1].
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(cfg.image_mean[c], 0.5f) << "channel " << c;
        EXPECT_FLOAT_EQ(cfg.image_std[c],  0.5f) << "channel " << c;
    }
}

// P3 landed the encoder, so this projector is now DISPATCHABLE — the profile
// no longer refuses it for being unknown. What it still refuses is a text-only
// vocabulary, which is the same contract the other two projectors have.
//
// (Before P3 this test asserted the opposite: a refusal naming P3. Keeping a
// test here either way is the point — the dispatch's answer for this projector
// is pinned, it just changed from "not yet" to "vocab, please".)
TEST(VisionLoader, Qwen3VlMergerIsDispatchableAndRefusesOnVocabOnly) {
    SKIP_IF_NO_QWEN_MMPROJ();

    VisionModel model;
    VisionLoader loader;
    loader.parse_metadata(get_qwen_mmproj_path(), model);

    const std::vector<std::string> empty_vocab;
    try {
        qinf::vision::make_vision_profile(model, nullptr, empty_vocab,
                                          "test: parameter '--mmproj'");
        FAIL() << "expected make_vision_profile to refuse a text-only vocab";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        // The refusal must now be about the vocabulary...
        EXPECT_NE(msg.find("text-only"), std::string::npos) << msg;
        EXPECT_NE(msg.find("<|vision_start|>"), std::string::npos) << msg;
        // ...and NOT about the projector being unhostable.
        EXPECT_EQ(msg.find("P3"), std::string::npos)
            << "projector should be hostable after P3: " << msg;
    }
}
