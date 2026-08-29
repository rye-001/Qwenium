// test_vision_profile.cpp — co-located unit test for src/vision/vision_profile.cpp
// (CLAUDE.md test co-location).
//
// Scope: the DISPATCH, which is what P0 of docs/plan-qwen35-vision-impl.md adds
// — exhaustive, fail-loud, no silent default. Everything reachable here runs
// BEFORE the encoder is constructed, so no mmproj, no weights, no backend and
// no model file are required; the encode paths themselves stay covered by
// multimodal-prefill-tests, which needs real files.
//
// The refusal cases are the point: an unregistered projector type must throw
// rather than fall through to Gemma4Uv (the bug the old duplicated if/else
// would have shipped the moment a third type existed), and a text-only vocab
// paired with a multimodal mmproj must be named as such.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "vision/vision_profile.h"
#include "vision/vision_model.h"

namespace {

using qinf::vision::VisionProjectorType;

// A VisionModel carrying config only — never loaded, never encoded. Enough to
// drive every branch that precedes encoder construction.
std::unique_ptr<qinf::vision::VisionModel> config_only_model(
        VisionProjectorType type, const std::string& path = "test.mmproj") {
    auto m = std::make_unique<qinf::vision::VisionModel>();
    m->config().projector_type = type;
    m->config().projection_dim = 2048;
    m->set_mmproj_path(path);
    return m;
}

const std::vector<std::string>& empty_vocab() {
    static const std::vector<std::string> v;
    return v;
}

TEST(VisionProfileTest, ToStringCoversEveryRegisteredProjector) {
    EXPECT_EQ(qinf::vision::to_string(VisionProjectorType::Gemma3Siglip),
              "Gemma3Siglip");
    EXPECT_EQ(qinf::vision::to_string(VisionProjectorType::Gemma4Uv), "Gemma4Uv");
}

// The P0 deliverable: an unhandled projector type refuses instead of defaulting.
TEST(VisionProfileTest, UnhandledProjectorTypeThrowsAndNamesItself) {
    // A type the enum does not register — stands in for "P1 added a value to
    // VisionProjectorType and to the loader but not to the factory".
    const auto bogus = static_cast<VisionProjectorType>(999);
    auto model = config_only_model(bogus, "unknown-projector.mmproj");

    try {
        qinf::vision::make_vision_profile(*model, nullptr, empty_vocab(),
                                       "test: parameter '--mmproj'");
        FAIL() << "expected make_vision_profile to refuse an unhandled type";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        // Fail-loud contract: slot, expected, actual — in that order.
        EXPECT_NE(msg.find("test: parameter '--mmproj'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("expected"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Gemma3Siglip"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Gemma4Uv"), std::string::npos) << msg;
        EXPECT_NE(msg.find("actual"), std::string::npos) << msg;
        // and the file that caused it, so the message is actionable
        EXPECT_NE(msg.find("unknown-projector.mmproj"), std::string::npos) << msg;
    }
}

TEST(VisionProfileTest, Gemma3TextOnlyVocabIsRefusedByName) {
    auto model = config_only_model(VisionProjectorType::Gemma3Siglip);
    try {
        qinf::vision::make_vision_profile(*model, nullptr, empty_vocab(),
                                       "run_chat: parameter '--image'");
        FAIL() << "expected make_vision_profile to refuse a text-only vocab";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("run_chat: parameter '--image'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("<start_of_image>"), std::string::npos) << msg;
        EXPECT_NE(msg.find("<image_soft_token>"), std::string::npos) << msg;
        EXPECT_NE(msg.find("<end_of_image>"), std::string::npos) << msg;
        EXPECT_NE(msg.find("text-only"), std::string::npos) << msg;
    }
}

TEST(VisionProfileTest, Gemma4TextOnlyVocabIsRefusedByName) {
    auto model = config_only_model(VisionProjectorType::Gemma4Uv);
    try {
        qinf::vision::make_vision_profile(*model, nullptr, empty_vocab(),
                                       "ServerVision: parameter '--mmproj'");
        FAIL() << "expected make_vision_profile to refuse a text-only vocab";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("ServerVision: parameter '--mmproj'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("<|image>"), std::string::npos) << msg;
        EXPECT_NE(msg.find("<image|>"), std::string::npos) << msg;
    }
}

// A vocab missing only ONE of the three markers is still a refusal — the
// original inline checks were `boi < 0 || eoi < 0 || soft < 0` and that
// three-way requirement is worth pinning.
TEST(VisionProfileTest, PartialMarkerVocabIsStillRefused) {
    auto model = config_only_model(VisionProjectorType::Gemma3Siglip);
    const std::vector<std::string> partial = {
        "<start_of_image>", "<image_soft_token>",  // <end_of_image> absent
    };
    EXPECT_THROW(
        qinf::vision::make_vision_profile(*model, nullptr, partial, "test: slot"),
        std::runtime_error);
}

}  // namespace
