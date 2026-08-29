// test_multimodal_check.cpp — load-time refusal of multimodal-only
// checkpoints.  Detection logic is exercised via constructed metadata
// fixtures; no model-file dependency.
//
// See src/loader/multimodal_check.h and docs/plan-multimodal-eval.md.

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "engine/model.h"
#include "../../src/loader/gguf_loader.h"
#include "../../src/loader/multimodal_check.h"

namespace {

// Minimal text-only ModelMetadata that passes every detection branch by
// default.  Tests add the specific multimodal signal under test on top.
ModelMetadata make_text_only_meta() {
    ModelMetadata m;
    m.architecture     = "gemma3";
    m.model_name       = "test-text-only";
    m.block_count      = 26;
    m.embedding_length = 1152;
    return m;
}

constexpr const char* kNoSidecar = "/tmp/qinf_multimodal_check_no_sidecar/model.gguf";

}  // namespace

// ── Negative case: clean text-only metadata is not flagged ────────────────────

TEST(MultimodalCheck, TextOnlyMetadataPassesDetection) {
    auto m = make_text_only_meta();
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    EXPECT_FALSE(r.is_multimodal) << "clean metadata tripped: "
                                    << r.trigger_field;
}

TEST(MultimodalCheck, TextOnlyMetadataDoesNotThrow) {
    auto m = make_text_only_meta();
    EXPECT_NO_THROW(
        multimodal_check::enforce_text_only_or_throw(m, kNoSidecar));
}

// ── GGUF metadata key signals ────────────────────────────────────────────────

TEST(MultimodalCheck, MmTokensPerImageKeyFires) {
    auto m = make_text_only_meta();
    m.raw_kv.set("mm_tokens_per_image", uint32_t{256});
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    ASSERT_TRUE(r.is_multimodal);
    EXPECT_EQ(r.trigger_field,  "mm_tokens_per_image");
    EXPECT_EQ(r.expected_value, "absent (text-only checkpoint)");
    EXPECT_EQ(r.actual_value,   "present");
}

TEST(MultimodalCheck, ImageTokenIdKeyFires) {
    auto m = make_text_only_meta();
    m.raw_kv.set("image_token_id", uint32_t{262144});
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    ASSERT_TRUE(r.is_multimodal);
    EXPECT_EQ(r.trigger_field, "image_token_id");
}

TEST(MultimodalCheck, VisionConfigHiddenSizeKeyFires) {
    auto m = make_text_only_meta();
    m.raw_kv.set("vision_config.hidden_size", uint32_t{1152});
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    ASSERT_TRUE(r.is_multimodal);
    EXPECT_EQ(r.trigger_field, "vision_config.hidden_size");
}

TEST(MultimodalCheck, BoiEoiTokenIndicesFire) {
    auto m1 = make_text_only_meta();
    m1.raw_kv.set("boi_token_index", uint32_t{255999});
    EXPECT_TRUE(multimodal_check::detect_multimodal_only(m1, kNoSidecar).is_multimodal);

    auto m2 = make_text_only_meta();
    m2.raw_kv.set("eoi_token_index", uint32_t{256000});
    EXPECT_TRUE(multimodal_check::detect_multimodal_only(m2, kNoSidecar).is_multimodal);
}

// ── Architecture-string signal ────────────────────────────────────────────────

TEST(MultimodalCheck, ConditionalGenerationArchSuffixFires) {
    auto m = make_text_only_meta();
    m.architecture = "Gemma3ForConditionalGeneration";
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    ASSERT_TRUE(r.is_multimodal);
    EXPECT_EQ(r.trigger_field, "general.architecture");
    EXPECT_EQ(r.actual_value,  "Gemma3ForConditionalGeneration");
}

// ── Tensor inventory signal ──────────────────────────────────────────────────

TEST(MultimodalCheck, VisionTowerTensorFires) {
    auto m = make_text_only_meta();
    TensorMetadata t;
    t.name = "vision_tower.blk.0.attn_q.weight";
    m.tensor_inventory[t.name] = t;
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    ASSERT_TRUE(r.is_multimodal);
    EXPECT_NE(r.trigger_field.find("vision_tower.blk.0.attn_q.weight"), std::string::npos);
}

TEST(MultimodalCheck, MmProjectorTensorFires) {
    auto m = make_text_only_meta();
    TensorMetadata t;
    t.name = "mm.0.weight";
    m.tensor_inventory[t.name] = t;
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    ASSERT_TRUE(r.is_multimodal);
    EXPECT_NE(r.trigger_field.find("mm.0.weight"), std::string::npos);
}

// ── Vocab image-patch placeholder signal ─────────────────────────────────────

TEST(MultimodalCheck, ImageSoftTokenInVocabFires) {
    auto m = make_text_only_meta();
    // Pad up to a realistic vocab and place the image patch token where
    // MedGemma carries it (262144).  Index value doesn't matter for the
    // signal — presence anywhere in id_to_token suffices.
    m.id_to_token.resize(10);
    m.id_to_token[7] = "<image_soft_token>";
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    ASSERT_TRUE(r.is_multimodal);
    EXPECT_EQ(r.trigger_field,  "vocab[7]");
    EXPECT_EQ(r.actual_value,   "found '<image_soft_token>'");
}

TEST(MultimodalCheck, StockGemmaVocabShapeDoesNotFire) {
    // Stock Gemma 3 1B carries <start_of_image> / <end_of_image> as
    // unused placeholders inherited from the shared SentencePiece
    // tokenizer, but NOT <image_soft_token>.  Verify those alone don't
    // false-positive — otherwise we'd refuse the text-only Gemma we
    // explicitly support.
    auto m = make_text_only_meta();
    m.id_to_token.resize(5);
    m.id_to_token[1] = "<start_of_image>";
    m.id_to_token[2] = "<end_of_image>";
    auto r = multimodal_check::detect_multimodal_only(m, kNoSidecar);
    EXPECT_FALSE(r.is_multimodal) << "false positive on stock Gemma vocab shape: "
                                    << r.trigger_field;
}

// ── Sibling config.json signal ───────────────────────────────────────────────

TEST(MultimodalCheck, SiblingConfigJsonWithVisionConfigFires) {
    // Write a temporary directory + config.json next to a fake gguf path.
    const std::string dir       = "/tmp/qinf_multimodal_check_with_sidecar";
    const std::string gguf_path = dir + "/model.gguf";
    const std::string cfg_path  = dir + "/config.json";

    std::system(("mkdir -p " + dir).c_str());
    {
        std::ofstream f(cfg_path);
        ASSERT_TRUE(f.good()) << "could not write " << cfg_path;
        f << R"({
            "architectures": ["Gemma3ForConditionalGeneration"],
            "vision_config": { "hidden_size": 1152 }
        })";
    }

    auto m = make_text_only_meta();
    auto r = multimodal_check::detect_multimodal_only(m, gguf_path);
    EXPECT_TRUE(r.is_multimodal);
    EXPECT_EQ(r.trigger_field, "sibling config.json: vision_config");

    std::remove(cfg_path.c_str());
    std::system(("rmdir " + dir).c_str());
}

TEST(MultimodalCheck, SiblingConfigJsonWithoutVisionConfigDoesNotFire) {
    const std::string dir       = "/tmp/qinf_multimodal_check_text_sidecar";
    const std::string gguf_path = dir + "/model.gguf";
    const std::string cfg_path  = dir + "/config.json";

    std::system(("mkdir -p " + dir).c_str());
    {
        std::ofstream f(cfg_path);
        ASSERT_TRUE(f.good());
        f << R"({ "architectures": ["Gemma3ForCausalLM"] })";
    }

    auto m = make_text_only_meta();
    auto r = multimodal_check::detect_multimodal_only(m, gguf_path);
    EXPECT_FALSE(r.is_multimodal);

    std::remove(cfg_path.c_str());
    std::system(("rmdir " + dir).c_str());
}

// ── Throw shape (CLAUDE.md fail-loud contract) ────────────────────────────────

TEST(MultimodalCheck, EnforceThrowsWithFieldExpectedActualAndDocPointer) {
    auto m = make_text_only_meta();
    m.raw_kv.set("mm_tokens_per_image", uint32_t{256});

    try {
        multimodal_check::enforce_text_only_or_throw(m, kNoSidecar);
        FAIL() << "expected GGUFLoadError, none thrown";
    } catch (const GGUFLoadError& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("multimodal-only checkpoint refused at load"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("field 'mm_tokens_per_image'"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("expected absent (text-only checkpoint)"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("got present"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("docs/plan-multimodal-eval.md"),
                  std::string::npos) << msg;
    } catch (const std::exception& e) {
        FAIL() << "wrong exception type: " << e.what();
    }
}
