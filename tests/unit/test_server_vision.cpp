// test_server_vision.cpp — co-located test for src/server/server_vision.cpp.
//
// ServerVision serves image input on /v1/chat/completions, and its plain
// accessors (prepare_image, encoder, the marker ids, projector_tag,
// projection_dim) are what a client reads to run its own image passes — the
// qemmi-lens-server image verdict does (architecture.md §6). Those accessors
// had no test of their own until 2026-10-03; the image verdict's gates covered
// them end to end and left with the lens.
//
// One leg per projector family, each a (text model, mmproj) pair, self-skipping
// when either file is absent. One model resident at a time (architecture.md §5).
//
//   QWEN36_MODEL_PATH    + QINF_QWEN_MMPROJ_PATH    — qwen3vl merger (Qwen 3.6-35B-A3B)
//   QINF_E2E_MODEL_PATH  + QINF_MMPROJ_PATH         — gemma3 SigLIP (medgemma-1.5-4b)
//   GEMMA4_MODEL_PATH    + QINF_GEMMA4_MMPROJ_PATH  — gemma4uv (Gemma 4 12B)
//
// The image is the synthetic 157x97 preprocessing fixture
// (tests/fixtures/vision/preproc_input.png) — non-square, so a letterbox or a
// dynamic resize has to preserve the aspect ratio to pass.

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/model.h"
#include "loader/tokenizer.h"
#include "models/model_registry.h"
#include "models/forward_pass_base.h"
#include "vision/bitmap.h"
#include "vision/i_vision_encoder.h"
#include "server_vision.h"

namespace {

struct Leg {
    const char* name;
    const char* model_env;
    const char* mmproj_env;
    const char* projector_tag;
    const char* boi;
    const char* soft;
    const char* eoi;
};
const Leg kLegs[] = {
    {"qwen3vl",  "QWEN36_MODEL_PATH",   "QINF_QWEN_MMPROJ_PATH",   "qwen3vl-merger",
     "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"},
    {"gemma3",   "QINF_E2E_MODEL_PATH", "QINF_MMPROJ_PATH",        "gemma3-siglip",
     "<start_of_image>", "<image_soft_token>", "<end_of_image>"},
    {"gemma4uv", "GEMMA4_MODEL_PATH",   "QINF_GEMMA4_MMPROJ_PATH", "gemma4uv",
     "<|image>", "<|image|>", "<image|>"},
};

std::string env_or_empty(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? std::string(v) : std::string();
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("test_server_vision: fixture '" + path + "' expected readable, actual not");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

const char* kFixture = QINF_TEST_FIXTURES_DIR "/vision/preproc_input.png";
constexpr int kFixtureW = 157, kFixtureH = 97;

}  // namespace

class ServerVisionTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        leg_ = &kLegs[GetParam()];
        const std::string model_path  = env_or_empty(leg_->model_env);
        const std::string mmproj_path = env_or_empty(leg_->mmproj_env);
        if (model_path.empty() || mmproj_path.empty())
            GTEST_SKIP() << leg_->model_env << " / " << leg_->mmproj_env << " not both set — skipping this leg";
        register_builtin_models();
        model_ = std::make_unique<Model>();
        model_->load_metadata(model_path, /*allow_multimodal=*/true);
        model_->load_tensors();
        const auto& meta = model_->get_metadata();
        tokenizer_ = std::make_unique<Tokenizer>(&meta, lookup_tokenizer_config(meta.architecture));
        fp_ = create_forward_pass(*model_, &meta, 1024, 1);
        vision_ = std::make_unique<ServerVision>(*model_, *fp_, *tokenizer_, mutex_, 1024,
                                                 mmproj_path, "", "");
    }
    void TearDown() override {
        vision_.reset(); fp_.reset(); tokenizer_.reset(); model_.reset();
    }

    const Leg* leg_ = nullptr;
    std::unique_ptr<Model> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<ForwardPassBase> fp_;
    std::mutex mutex_;
    std::unique_ptr<ServerVision> vision_;
};

// The accessors name THIS projector: its tag, its three marker tokens (checked
// against the vocabulary's own text, not against the profile that produced
// them), a render prefix that opens with the begin marker, and soft tokens as
// wide as the text model's embedding.
TEST_P(ServerVisionTest, AccessorsNameTheLoadedProjector) {
    const std::vector<std::string> vocab = tokenizer_->get_vocabulary();
    EXPECT_EQ(vision_->projector_tag(), leg_->projector_tag);
    ASSERT_GE(vision_->boi_id(), 0);
    ASSERT_GE(vision_->soft_id(), 0);
    ASSERT_GE(vision_->eoi_id(), 0);
    ASSERT_LT((size_t)vision_->boi_id(), vocab.size());
    ASSERT_LT((size_t)vision_->soft_id(), vocab.size());
    ASSERT_LT((size_t)vision_->eoi_id(), vocab.size());
    EXPECT_EQ(vocab[vision_->boi_id()], leg_->boi);
    EXPECT_EQ(vocab[vision_->soft_id()], leg_->soft);
    EXPECT_EQ(vocab[vision_->eoi_id()], leg_->eoi);
    EXPECT_NE(vision_->image_marker_prefix().find(leg_->boi), std::string::npos)
        << "marker prefix '" << vision_->image_marker_prefix() << "' expected to contain '" << leg_->boi << "'";
    EXPECT_EQ(vision_->projection_dim(), model_->get_metadata().embedding_length)
        << "projector '" << leg_->projector_tag << "' soft tokens must be as wide as the text embedding";
    EXPECT_EQ(vision_->encoder().projection_dim(), vision_->projection_dim());
}

// prepare_image is deterministic and its content rectangle — the part of the
// canvas the uploaded picture occupies, which a caller maps model coordinates
// back through — lies inside the canvas and keeps the picture's aspect ratio.
TEST_P(ServerVisionTest, PrepareImageIsDeterministicAndKeepsTheContentRect) {
    const std::vector<uint8_t> bytes = read_file(kFixture);
    const qinf::vision::Bitmap a = vision_->prepare_image(bytes);
    const qinf::vision::Bitmap b = vision_->prepare_image(bytes);
    EXPECT_EQ(a.content_id, b.content_id);
    EXPECT_EQ(a.pixels, b.pixels);
    ASSERT_GT(a.width, 0);
    ASSERT_GT(a.height, 0);
    ASSERT_GT(a.content_w, 0);
    ASSERT_GT(a.content_h, 0);
    EXPECT_GE(a.content_x, 0);
    EXPECT_GE(a.content_y, 0);
    EXPECT_LE(a.content_x + a.content_w, a.width);
    EXPECT_LE(a.content_y + a.content_h, a.height);
    const double want = (double)kFixtureW / kFixtureH;
    const double got  = (double)a.content_w / a.content_h;
    EXPECT_NEAR(got / want, 1.0, 0.10)
        << "projector '" << leg_->projector_tag << "': content rect " << a.content_w << "x" << a.content_h
        << " on a " << a.width << "x" << a.height << " canvas expected the fixture's aspect "
        << kFixtureW << ":" << kFixtureH;
}

TEST_P(ServerVisionTest, EmptyImageFailsLoud) {
    EXPECT_THROW(vision_->prepare_image({}), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(
    Projectors, ServerVisionTest, ::testing::Values(0, 1, 2),
    [](const ::testing::TestParamInfo<int>& info) { return std::string(kLegs[info.param].name); });
