// test_image_verdict.cpp — co-located unit test for src/server/image_verdict.cpp.
// Everything that needs no model: the calibration table and its lookup, the
// question wording (a calibrated mark must render to EXACTLY the string its cut
// was measured on), the fail-loud refusals, the cut bands, and the JSON shape.
// The driver itself is gated against the probe on the real model (G2,
// docs/plan-image-verdict.md §7).

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

#include "engine/model.h"
#include "image_verdict.h"

using namespace qinf;

namespace {

const ImageVerdictCalibration& row35() { return image_verdict_calibrations().at(0); }

ModelMetadata meta35(uint32_t file_type = 12) {
    ModelMetadata m;
    m.architecture = "qwen35moe";
    m.block_count = 40;
    m.raw_kv.set("general.file_type", file_type);
    return m;
}

ImageVerdictVision vision35() {
    ImageVerdictVision v;
    v.projector = "qwen3vl-merger";
    v.projection_dim = 2048;
    return v;
}

ImageVerdictQuestion mark(const std::string& id, const std::string& name,
                          std::map<std::string, std::string> params) {
    ImageVerdictQuestion q;
    q.id = id;
    q.mark = name;
    q.params = std::move(params);
    return q;
}

ImageVerdictQuestion free_q(const std::string& id, const std::string& text) {
    ImageVerdictQuestion q;
    q.id = id;
    q.question = text;
    return q;
}

}  // namespace

// ── The table ────────────────────────────────────────────────────────────────

TEST(ImageVerdictCalibration, RowsAreWellFormed) {
    for (const ImageVerdictCalibration& c : image_verdict_calibrations()) {
        std::set<std::string> names;
        for (const ImageVerdictMark& m : c.marks) {
            EXPECT_TRUE(names.insert(m.name).second) << c.model << ": duplicate mark " << m.name;
            EXPECT_LE(m.cut_no, m.cut_yes) << m.name;
            EXPECT_GT(m.cut_no, 0.0) << m.name;
            EXPECT_LE(m.cut_yes, kImageVerdictNeverYes) << m.name;
        }
    }
}

TEST(ImageVerdictCalibration, FindsTheMeasuredFileAndRefusesEverythingElse) {
    EXPECT_EQ(image_verdict_calibration_for(meta35(), vision35()), &row35());

    EXPECT_EQ(image_verdict_calibration_for(meta35(/*file_type=*/15), vision35()), nullptr);  // other quant
    ModelMetadata mtp = meta35();
    mtp.block_count = 41;                                                                     // MTP build
    EXPECT_EQ(image_verdict_calibration_for(mtp, vision35()), nullptr);
    ImageVerdictVision other = vision35();
    other.projector = "gemma3-siglip";
    EXPECT_EQ(image_verdict_calibration_for(meta35(), other), nullptr);
    ModelMetadata no_ft;
    no_ft.architecture = "qwen35moe";
    no_ft.block_count = 40;                                                                   // no general.file_type
    EXPECT_EQ(image_verdict_calibration_for(no_ft, vision35()), nullptr);

    const std::string why = image_verdict_refusal(meta35(15), vision35());
    EXPECT_NE(why.find("expected a measured calibration row"), std::string::npos) << why;
    EXPECT_NE(why.find("file_type 15"), std::string::npos) << why;
}

// ── Wording: a calibrated mark renders to the string its cut was measured on ─

TEST(ImageVerdictQuestions, MarksRenderTheMeasuredWordingExactly) {
    const auto p = plan_image_verdict_questions(row35(), {
        mark("s", "signature", {{"subject", "delivery note"}, {"box", "Received by"}}),
        mark("i", "signature", {{"subject", "invoice"}, {"box", "Approved by"}}),
        mark("t", "stamp", {{"subject", "delivery note"}}),
        mark("d", "date", {{"field", "delivery date"}}),
        mark("p", "date", {{"field", "'Date paid' field"}}),
    });
    // The probe's exact strings (tests/perf/probe_verdict_img_render.swift).
    EXPECT_EQ(p[0].text, "Is the delivery note signed by hand in the 'Received by' box?");
    EXPECT_EQ(p[1].text, "Is the invoice signed by hand in the 'Approved by' box?");
    EXPECT_EQ(p[2].text, "Does the delivery note carry a stamp?");
    EXPECT_EQ(p[3].text, "Is the delivery date filled in?");
    EXPECT_EQ(p[4].text, "Is the 'Date paid' field filled in?");
    EXPECT_DOUBLE_EQ(p[2].cut_yes, kImageVerdictNeverYes);   // the stamp never answers yes
    EXPECT_DOUBLE_EQ(p[2].cut_no, 0.5);
    for (const auto& q : p) EXPECT_TRUE(q.calibrated);
}

TEST(ImageVerdictQuestions, AFreeQuestionIsAnsweredAtHalfAndMarkedUncalibrated) {
    const auto p = plan_image_verdict_questions(row35(), {free_q("q", "Is there a QR code on the receipt?")});
    EXPECT_EQ(p[0].text, "Is there a QR code on the receipt?");
    EXPECT_FALSE(p[0].calibrated);
    EXPECT_DOUBLE_EQ(p[0].cut_yes, 0.5);
    EXPECT_DOUBLE_EQ(p[0].cut_no, 0.5);
    EXPECT_TRUE(p[0].mark.empty());
}

TEST(ImageVerdictQuestions, RefusesMalformedQuestionsFailLoud) {
    const auto& c = row35();
    auto refused = [&](std::vector<ImageVerdictQuestion> qs, const std::string& expect) {
        try {
            plan_image_verdict_questions(c, qs);
            ADD_FAILURE() << "expected a refusal containing: " << expect;
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(expect), std::string::npos) << e.what();
        }
    };
    refused({}, "'questions' expected >= 1");
    refused({free_q("", "x?")}, "expected a non-empty id");
    refused({free_q("a", "x?"), free_q("a", "y?")}, "duplicate 'a'");
    ImageVerdictQuestion both = mark("b", "stamp", {{"subject", "invoice"}});
    both.question = "x?";
    refused({both}, "exactly one of 'mark' and 'question', actual both");
    ImageVerdictQuestion neither;
    neither.id = "n";
    refused({neither}, "actual neither");
    refused({mark("u", "checkbox", {})}, "expected one of {signature, stamp, date}");
    refused({mark("m", "stamp", {})}, "params.subject: expected a non-empty value");
    refused({mark("e", "stamp", {{"subject", ""}})}, "actual empty");
    refused({mark("x", "stamp", {{"subject", "invoice"}, {"box", "x"}})}, "params.box");
    refused({mark("l", "stamp", {{"subject", "in\nvoice"}})}, "expected one line");
    ImageVerdictQuestion fp = free_q("f", "x?");
    fp.params["subject"] = "invoice";
    refused({fp}, "expected none with a free 'question'");
}

// ── Bands ────────────────────────────────────────────────────────────────────

TEST(ImageVerdictBand, SplitCutsGiveThreeBands) {
    EXPECT_EQ(image_verdict_band(0.959, 0.9, 0.5), ImageVerdictAnswer::Yes);
    EXPECT_EQ(image_verdict_band(0.9, 0.9, 0.5), ImageVerdictAnswer::Yes);        // the cut itself
    EXPECT_EQ(image_verdict_band(0.864, 0.9, 0.5), ImageVerdictAnswer::Unclear);
    EXPECT_EQ(image_verdict_band(0.5, 0.9, 0.5), ImageVerdictAnswer::Unclear);
    EXPECT_EQ(image_verdict_band(0.499, 0.9, 0.5), ImageVerdictAnswer::No);
}

TEST(ImageVerdictBand, ANeverYesMarkAnswersOnlyNoOrUnclear) {
    // The stamp row (docs/note-stamp-lures.md): real stamps and lookalikes both
    // land in unclear; a saturated p of exactly 1.0 included.
    EXPECT_EQ(image_verdict_band(0.9881, kImageVerdictNeverYes, 0.5), ImageVerdictAnswer::Unclear);  // real stamp
    EXPECT_EQ(image_verdict_band(0.9812, kImageVerdictNeverYes, 0.5), ImageVerdictAnswer::Unclear);  // printed badge
    EXPECT_EQ(image_verdict_band(1.0, kImageVerdictNeverYes, 0.5), ImageVerdictAnswer::Unclear);
    EXPECT_EQ(image_verdict_band(0.5, kImageVerdictNeverYes, 0.5), ImageVerdictAnswer::Unclear);
    EXPECT_EQ(image_verdict_band(0.0482, kImageVerdictNeverYes, 0.5), ImageVerdictAnswer::No);      // nothing stamp-like
}

TEST(ImageVerdictBand, EqualCutsHaveNoUnclearBand) {
    EXPECT_EQ(image_verdict_band(0.5, 0.5, 0.5), ImageVerdictAnswer::Yes);
    EXPECT_EQ(image_verdict_band(0.4999, 0.5, 0.5), ImageVerdictAnswer::No);
}

// ── JSON ─────────────────────────────────────────────────────────────────────

TEST(ImageVerdictJson, CarriesTheAnswerCutsAndCalibrationFlag) {
    ImageVerdictReport rep;
    rep.model = "m";
    rep.image_tokens = 1440;
    rep.grid_w = 32;
    rep.grid_h = 45;
    ImageVerdictResult r;
    r.id = "t";
    r.mark = "stamp";
    r.question = "Does the invoice carry a stamp?";
    r.p_yes = 0.86;
    r.cut_yes = kImageVerdictNeverYes;
    r.cut_no = 0.5;
    r.calibrated = true;
    r.answer = image_verdict_band(r.p_yes, r.cut_yes, r.cut_no);
    r.prompt_tokens = 1490;
    rep.answers.push_back(r);
    const nlohmann::json j = nlohmann::json::parse(image_verdict_to_json(rep));
    EXPECT_EQ(j["format_version"], "qemmi-verdict-image/v1");
    EXPECT_EQ(j["image"]["tokens"], 1440);
    EXPECT_EQ(j["image"]["grid"][1], 45);
    const auto& a = j["answers"][0];
    EXPECT_EQ(a["answer"], "unclear");
    EXPECT_EQ(a["mark"], "stamp");
    EXPECT_DOUBLE_EQ(a["cut"]["yes"].get<double>(), 1.0);
    EXPECT_NEAR(a["p"]["no"].get<double>(), 0.14, 1e-12);
    EXPECT_TRUE(a["calibrated"].get<bool>());
}

// ── The store (image_id) ─────────────────────────────────────────────────────

namespace {
ImageVerdictStore::Entry entry(uint64_t hash, std::vector<int32_t> prefix) {
    ImageVerdictStore::Entry e;
    e.image_hash = hash;
    e.prefix_tokens = std::move(prefix);
    e.blob = std::make_shared<const std::vector<uint8_t>>(std::vector<uint8_t>{1, 2, 3});
    return e;
}
}  // namespace

TEST(ImageVerdictStore, HitNeedsTheSameImageAndPromptTokens) {
    ImageVerdictStore st(4, std::chrono::seconds(60));
    const auto t0 = ImageVerdictStore::Clock::now();
    EXPECT_EQ(st.find("a", 7, {1, 2}, t0), nullptr);                       // empty: a miss
    st.put("a", entry(7, {1, 2}), t0);
    ASSERT_NE(st.find("a", 7, {1, 2}, t0), nullptr);                       // hit
    EXPECT_EQ(st.find("a", 7, {1, 2, 3}, t0), nullptr);                    // other prompt: a miss
    EXPECT_THROW(st.find("a", 8, {1, 2}, t0), std::runtime_error);         // other image: refused
}

TEST(ImageVerdictStore, EvictsTheLeastRecentlyUsed) {
    ImageVerdictStore st(2, std::chrono::seconds(600));
    const auto t0 = ImageVerdictStore::Clock::now();
    st.put("a", entry(1, {1}), t0);
    st.put("b", entry(2, {2}), t0 + std::chrono::seconds(1));
    ASSERT_NE(st.find("a", 1, {1}, t0 + std::chrono::seconds(2)), nullptr);   // a is now the newer
    st.put("c", entry(3, {3}), t0 + std::chrono::seconds(3));                // evicts b
    EXPECT_EQ(st.size(), 2u);
    EXPECT_NE(st.find("a", 1, {1}, t0 + std::chrono::seconds(4)), nullptr);
    EXPECT_EQ(st.find("b", 2, {2}, t0 + std::chrono::seconds(4)), nullptr);
    EXPECT_NE(st.find("c", 3, {3}, t0 + std::chrono::seconds(4)), nullptr);
}

TEST(ImageVerdictStore, ExpiresIdleEntriesAndRefusesZeroSize) {
    ImageVerdictStore st(4, std::chrono::seconds(10));
    const auto t0 = ImageVerdictStore::Clock::now();
    st.put("a", entry(1, {1}), t0);
    st.expire(t0 + std::chrono::seconds(5));
    EXPECT_EQ(st.size(), 1u);
    st.expire(t0 + std::chrono::seconds(11));
    EXPECT_EQ(st.size(), 0u);
    EXPECT_THROW(ImageVerdictStore(0, std::chrono::seconds(10)), std::runtime_error);
}
