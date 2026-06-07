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
