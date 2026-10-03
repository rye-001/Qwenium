#pragma once
// image_verdict.h — POST /v1/verdict with an IMAGE (docs/plan-image-verdict.md).
//
// One page, yes/no questions about visible marks (signed? stamped? field
// filled?), each answered yes / no / unclear from P(yes) against P(no) on the
// prefill's last row — nothing generated, and NO attention read: this is not
// the attention lens and carries no receipt (image attention readouts are
// closed, docs/note-image-prefill-tap-probe.md). The evidence is
// docs/note-verdict-img-probe.md.
//
// Separation of concerns: this module knows nothing of HTTP or of the server's
// classes. The route parses and dispatches; ServerVision supplies the vision
// handles (ImageVerdictVision); this module owns the prompt, the image pass,
// the per-question passes, the readout, the cut bands, its calibration table
// and its JSON. It does not touch server_lens.
//
// Per request: the image is encoded and prefilled ONCE (up to the end of the
// image span), the slot captured (the snapshot carries the M-RoPE position —
// the RPOS section), and every question restored from it and prefilled alone
// from the rope position. That split is bit-identical to one full prefill per
// question (note §10, test-image-prefix-roundtrip).

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

class ForwardPassBase;
class Tokenizer;
struct ModelMetadata;

namespace qinf::vision {
class IVisionEncoder;
struct Bitmap;
}  // namespace qinf::vision

namespace qinf {

// ── The vision handles the driver needs (filled by ServerVision, or a probe) ──
struct ImageVerdictVision {
    qinf::vision::IVisionEncoder* encoder = nullptr;   // borrowed
    std::string marker_prefix;                         // rendered before the question
    int32_t     boi_id = -1, soft_id = -1, eoi_id = -1;
    std::string projector;                             // e.g. "qwen3vl-merger"
    uint32_t    projection_dim = 0;
};

// ── Calibration: which models, which marks, which cuts ───────────────────────
// A calibrated mark is a question the server words itself — exactly as it was
// measured — so its cut applies. `wording` holds {placeholders} filled from the
// question's params. answer = yes if p >= cut_yes, no if p < cut_no, else
// unclear (cut_no <= cut_yes; equal cuts mean no unclear band). cut_yes =
// kImageVerdictNeverYes means the mark never answers yes — only no or unclear
// (the stamp: lookalikes score like real stamps, docs/note-stamp-lures.md).
inline constexpr double kImageVerdictNeverYes = 1.0;

struct ImageVerdictMark {
    const char* name;
    const char* wording;
    double      cut_yes;
    double      cut_no;
    const char* provenance;
    // What to ask the model to locate for "where" ("Locate {locate} in the
    // image, …"), with the same {placeholders} as `wording` (a subset is fine).
    // nullptr ⇒ this mark gets no box.
    const char* locate = nullptr;
};

// Keyed like the lens rows — {architecture, block_count, general.file_type} —
// plus the projector, so an unmeasured model, quantization or mmproj is
// refused, never inherited.
struct ImageVerdictCalibration {
    const char* architecture;
    uint32_t    block_count;
    uint32_t    file_type;
    const char* projector;
    uint32_t    projection_dim;
    const char* model;
    std::vector<ImageVerdictMark> marks;
};

const std::vector<ImageVerdictCalibration>& image_verdict_calibrations();

// nullptr when no row matches; image_verdict_refusal says why, fail-loud style.
const ImageVerdictCalibration* image_verdict_calibration_for(const ::ModelMetadata& meta,
                                                             const ImageVerdictVision& vision);
std::string image_verdict_refusal(const ::ModelMetadata& meta, const ImageVerdictVision& vision);

// ── Questions ────────────────────────────────────────────────────────────────
// Either a calibrated mark (`mark` + the params its wording names) or a free
// `question` (answered at 0.5, calibrated = false). Exactly one of the two.
struct ImageVerdictQuestion {
    std::string id;
    std::string mark;
    std::map<std::string, std::string> params;
    std::string question;
};

// A question made ready: the text the model sees and the cuts that apply.
struct ImageVerdictPlanned {
    std::string id, mark, text;
    std::string locate;           // the filled locate wording; empty ⇒ no box
    double      cut_yes = 0.5, cut_no = 0.5;
    bool        calibrated = false;
};

// Validate and word the questions (no model needed). Fail-loud on: no
// questions, an empty or duplicate id, both or neither of mark/question, a mark
// this row does not calibrate, a missing / unused / empty param.
std::vector<ImageVerdictPlanned> plan_image_verdict_questions(
    const ImageVerdictCalibration& cal, const std::vector<ImageVerdictQuestion>& questions);

enum class ImageVerdictAnswer { Yes, No, Unclear };
ImageVerdictAnswer image_verdict_band(double p_yes, double cut_yes, double cut_no);
const char* image_verdict_answer_name(ImageVerdictAnswer a);

// ── The report ───────────────────────────────────────────────────────────────
struct ImageVerdictResult {
    std::string id, mark, question;
    ImageVerdictAnswer answer = ImageVerdictAnswer::Unclear;
    double p_yes = 0.0;
    double cut_yes = 0.5, cut_no = 0.5;
    bool   calibrated = false;
    int    prompt_tokens = 0;
    // "where" (asked only for a yes / unclear answer on a mark with a locate
    // wording): the model's own box for the mark, as fractions [x0, y0, x1, y1]
    // of the uploaded picture. A pointer, not evidence — on an empty field the
    // model still draws a box (docs/note-verdict-img-ground.md).
    bool   where_asked = false;
    bool   has_box = false;          // false: the model's answer held no box
    double box[4] = {0, 0, 0, 0};
};

struct ImageVerdictReport {
    std::string model;
    uint32_t    image_tokens = 0, grid_w = 0, grid_h = 0;
    bool        warm = false;   // the image pass was resumed from the store
    std::vector<ImageVerdictResult> answers;
};

// ── Keeping an image across requests (`image_id`) ─────────────────────────────
// The post-image state of one request, kept so a later request with the same
// image_id and the same image skips the encode and the image pass. Its own
// store, not LensDocumentStore (that one is the lens's: routes, prefill
// shapes, truncation depths). A hit needs the same image (the preprocessed
// pixels' content id) AND the same image-inclusive prompt tokens; the same id
// for a different image is refused, not silently replaced. Least recently used
// out first; entries idle longer than the TTL are dropped on every call.
// Bounded by count (each entry is one snapshot: ~0.1 GB for a page on the
// 35B-A3B — KV rows of the image span plus the DeltaNet state).
inline constexpr size_t kImageVerdictStoreMax = 4;
inline constexpr std::chrono::seconds kImageVerdictStoreTtl{15 * 60};

class ImageVerdictStore {
public:
    using Clock = std::chrono::steady_clock;
    struct Entry {
        uint64_t             image_hash = 0;
        std::vector<int32_t> prefix_tokens;   // up to the end of the image span — the hit test
        uint32_t             image_tokens = 0, grid_w = 0, grid_h = 0;
        std::shared_ptr<const std::vector<uint8_t>> blob;   // capture_slot after the image pass
        Clock::time_point    last_used{};
    };

    ImageVerdictStore(size_t max_images, std::chrono::seconds ttl);

    void expire(Clock::time_point now);
    // The kept image this request may resume from, or nullptr (a miss — also
    // when the prompt tokens differ). Throws when `id` is held for another
    // image. A hit refreshes its age.
    const Entry* find(const std::string& id, uint64_t image_hash,
                      const std::vector<int32_t>& prefix_tokens, Clock::time_point now);
    // Store (or replace) `id`, evicting the least recently used entry when full.
    void put(const std::string& id, Entry entry, Clock::time_point now);

    size_t size() const { return entries_.size(); }
    size_t max_images() const { return max_; }

private:
    size_t max_;
    std::chrono::seconds ttl_;
    std::map<std::string, Entry> entries_;
};

// The driver. Slot 0, EXCLUSIVE: the caller holds the model lock. Leaves slot 0
// cleared and the engine's prefill attention mode as it found it. With a store
// and an image_id (both or neither), the image pass is kept / resumed. With
// `where`, every yes / unclear answer on a mark with a locate wording also gets
// the model's box: the post-image state restored, the locate prompt prefilled,
// a short greedy generation (~1 s), the box parsed. Nothing else changes: the
// answers and p values are the same with or without it.
// The model's box: the first [x0, y0, x1, y1] in `text`, Qwen-VL's 0..1000
// coordinates of the canvas the encoder saw. False when there is none or it is
// not a box (x1 <= x0 or y1 <= y0).
bool parse_image_verdict_box(const std::string& text, double rel[4]);

// Map a 0..1000 canvas box onto the uploaded picture (the Bitmap's content
// rect, i.e. without the letterbox), as fractions clamped to [0, 1].
void image_verdict_box_on_picture(const double rel[4], const qinf::vision::Bitmap& canvas, double out[4]);

// The prompt that asks for a box ("Locate … in the image, output its bbox
// coordinates using JSON format." — docs/note-verdict-img-ground.md §1).
std::string image_verdict_locate_prompt(const std::string& locate);

ImageVerdictReport run_image_verdict(::ForwardPassBase* fp, ggml_backend_sched_t sched,
                                     ::Tokenizer* tok, const ::ModelMetadata& meta,
                                     uint32_t n_ctx_max, const ImageVerdictVision& vision,
                                     const qinf::vision::Bitmap& image,
                                     const ImageVerdictCalibration& cal,
                                     const std::vector<ImageVerdictQuestion>& questions,
                                     ImageVerdictStore* store = nullptr,
                                     const std::string& image_id = std::string(),
                                     bool where = false);

std::string image_verdict_to_json(const ImageVerdictReport& report);

}  // namespace qinf
