#include "image_verdict.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>

#include "nlohmann/json.hpp"

#include "engine/model.h"
#include "engine/decode_step.h"
#include "engine/multimodal_prefill.h"
#include "image/image_prompt.h"
#include "loader/chat_template.h"
#include "loader/tokenizer.h"
#include "models/forward_pass_base.h"
#include "models/model_registry.h"   // lookup_chat_template
#include "sampling/sampling.h"
#include "session/slot_snapshot.h"
#include "vision/bitmap.h"
#include "vision/i_vision_encoder.h"

namespace qinf {

// ── The table ────────────────────────────────────────────────────────────────

const std::vector<ImageVerdictCalibration>& image_verdict_calibrations() {
    // One row per measured {model file, mmproj}. Wordings are the probe's
    // exact strings with the varying parts as {params}; cuts are the measured
    // ones (docs/note-verdict-img-probe.md). Measured on the flash-attention
    // encoder (§11) with a materialized LLM prefill (run_image_verdict pins it);
    // re-run after the 2026-10-02 engine fixes, no decision changed
    // (docs/note-verdict-img-ground.md §6). The locate wordings are the
    // grounding probe's (same note, §1): boxes IoU >= 0.5 on 59/60 marks.
    static const std::vector<ImageVerdictCalibration> kRows = {
        {"qwen35moe", 40, 12, "qwen3vl-merger", 2048,
         "Qwen3.6-35B-A3B (UD-Q3_K_XL) + Qwen3.6 mmproj",
         {
             {"signature", "Is the {subject} signed by hand in the '{box}' box?", 0.5, 0.5,
              "note §7 (lures + scans 100%), §9 real paper 12/12",
              "the handwritten signature in the '{box}' box"},
             // Never yes (user decision 2026-10-01): printed stamp-shaped badges and
             // show-through score like real stamps (up to 0.995) and no cut separates
             // them; no real stamp ever scored below 0.5, so no / unclear both hold.
             {"stamp", "Does the {subject} carry a stamp?", kImageVerdictNeverYes, 0.5,
              "note-stamp-lures.md (all sets: real stamps >= 0.934, 0/207 below 0.5; "
              "lures up to 0.995, badge + show-through 0/60 below 0.5), §9 real paper stamps >= 0.959",
              "the stamp"},
             {"date", "Is the {field} filled in?", 0.5, 0.5,
              "note §7 (lures + scans 100%), §9 real paper 12/12",
              "the handwritten {field}"},
         }},
    };
    return kRows;
}

namespace {

uint32_t file_type_of(const ::ModelMetadata& meta) {
    const std::optional<uint32_t> ft = meta.raw_kv.get_uint32_opt("general.file_type");
    return ft ? *ft : 0xFFFFFFFFu;
}

std::string key_text(const ::ModelMetadata& meta, const ImageVerdictVision& v) {
    return "{architecture '" + meta.architecture + "', block_count " + std::to_string(meta.block_count) +
           ", file_type " + std::to_string(file_type_of(meta)) + ", projector '" + v.projector +
           "', projection_dim " + std::to_string(v.projection_dim) + "}";
}

}  // namespace

const ImageVerdictCalibration* image_verdict_calibration_for(const ::ModelMetadata& meta,
                                                             const ImageVerdictVision& vision) {
    const uint32_t ft = file_type_of(meta);
    for (const ImageVerdictCalibration& c : image_verdict_calibrations())
        if (meta.architecture == c.architecture && meta.block_count == c.block_count &&
            ft == c.file_type && vision.projector == c.projector &&
            vision.projection_dim == c.projection_dim)
            return &c;
    return nullptr;
}

std::string image_verdict_refusal(const ::ModelMetadata& meta, const ImageVerdictVision& vision) {
    std::string rows;
    for (const ImageVerdictCalibration& c : image_verdict_calibrations())
        rows += std::string(rows.empty() ? "" : "; ") + c.model;
    return "image verdict: model + mmproj expected a measured calibration row (" + rows +
           "), actual " + key_text(meta, vision) +
           " — unmeasured models, quantizations and projectors are refused, never inherited";
}

// ── Questions and bands ──────────────────────────────────────────────────────

std::vector<ImageVerdictPlanned> plan_image_verdict_questions(
    const ImageVerdictCalibration& cal, const std::vector<ImageVerdictQuestion>& questions) {
    if (questions.empty())
        throw std::runtime_error("image verdict: 'questions' expected >= 1, actual 0");
    std::set<std::string> ids;
    std::vector<ImageVerdictPlanned> out;
    for (size_t i = 0; i < questions.size(); ++i) {
        const ImageVerdictQuestion& q = questions[i];
        const std::string at = "image verdict: questions[" + std::to_string(i) + "]";
        if (q.id.empty()) throw std::runtime_error(at + ".id: expected a non-empty id, actual empty");
        if (!ids.insert(q.id).second)
            throw std::runtime_error(at + ".id: expected unique ids, actual duplicate '" + q.id + "'");
        if (q.mark.empty() == q.question.empty())
            throw std::runtime_error(at + ": expected exactly one of 'mark' and 'question', actual " +
                                     (q.mark.empty() ? "neither" : "both"));
        ImageVerdictPlanned p;
        p.id = q.id;
        if (!q.question.empty()) {
            if (!q.params.empty())
                throw std::runtime_error(at + ".params: expected none with a free 'question', actual " +
                                         std::to_string(q.params.size()));
            p.text = q.question;                     // free question: 0.5, uncalibrated
            out.push_back(p);
            continue;
        }
        const ImageVerdictMark* m = nullptr;
        std::string known;
        for (const ImageVerdictMark& mk : cal.marks) {
            known += std::string(known.empty() ? "" : ", ") + mk.name;
            if (q.mark == mk.name) m = &mk;
        }
        if (!m)
            throw std::runtime_error(at + ".mark: expected one of {" + known + "} for " + cal.model +
                                     ", actual '" + q.mark + "'");
        // Fill {placeholders}; every placeholder needs a param, every param a placeholder.
        std::string text = m->wording;
        std::set<std::string> used;
        for (size_t a = text.find('{'); a != std::string::npos; a = text.find('{', a)) {
            const size_t b = text.find('}', a);
            const std::string key = text.substr(a + 1, b - a - 1);
            const auto it = q.params.find(key);
            if (it == q.params.end() || it->second.empty())
                throw std::runtime_error(at + ".params." + key + ": expected a non-empty value for mark '" +
                                         q.mark + "' (\"" + m->wording + "\"), actual " +
                                         (it == q.params.end() ? "missing" : "empty"));
            if (it->second.find('\n') != std::string::npos)
                throw std::runtime_error(at + ".params." + key + ": expected one line, actual a newline");
            text.replace(a, b - a + 1, it->second);
            a += it->second.size();
            used.insert(key);
        }
        for (const auto& kv : q.params)
            if (!used.count(kv.first))
                throw std::runtime_error(at + ".params." + kv.first + ": expected only the params of mark '" +
                                         q.mark + "' (\"" + m->wording + "\"), actual an unused one");
        if (m->locate) {
            // The locate wording's placeholders are a subset of the wording's,
            // so every one has a validated, non-empty, one-line param.
            std::string loc = m->locate;
            for (size_t a = loc.find('{'); a != std::string::npos; a = loc.find('{', a)) {
                const size_t b = loc.find('}', a);
                const std::string key = loc.substr(a + 1, b - a - 1);
                if (!used.count(key))
                    throw std::runtime_error(at + ": mark '" + q.mark + "' locate wording expected only the "
                                             "wording's {params}, actual {" + key + "}");
                const std::string& v = q.params.at(key);
                loc.replace(a, b - a + 1, v);
                a += v.size();
            }
            p.locate = loc;
        }
        p.mark = q.mark;
        p.text = text;
        p.cut_yes = m->cut_yes;
        p.cut_no = m->cut_no;
        p.calibrated = true;
        out.push_back(p);
    }
    return out;
}

ImageVerdictAnswer image_verdict_band(double p_yes, double cut_yes, double cut_no) {
    // A saturated p of exactly 1.0 must not turn a never-yes mark into a yes.
    if (cut_yes < kImageVerdictNeverYes && p_yes >= cut_yes) return ImageVerdictAnswer::Yes;
    if (p_yes < cut_no) return ImageVerdictAnswer::No;
    return ImageVerdictAnswer::Unclear;
}

// ── "where": the model's own box ─────────────────────────────────────────────

bool parse_image_verdict_box(const std::string& text, double rel[4]) {
    static const std::regex kBox(
        R"(\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\])");
    std::smatch m;
    if (!std::regex_search(text, m, kBox)) return false;
    for (int k = 0; k < 4; ++k) rel[k] = std::stod(m[k + 1].str());
    return rel[2] > rel[0] && rel[3] > rel[1];
}

void image_verdict_box_on_picture(const double rel[4], const qinf::vision::Bitmap& canvas, double out[4]) {
    if (canvas.width <= 0 || canvas.height <= 0)
        throw std::runtime_error("image_verdict_box_on_picture: canvas expected positive size, actual " +
                                 std::to_string(canvas.width) + "x" + std::to_string(canvas.height));
    const bool whole = canvas.content_w <= 0 || canvas.content_h <= 0;
    const double cx = whole ? 0 : canvas.content_x, cy = whole ? 0 : canvas.content_y;
    const double cw = whole ? canvas.width : canvas.content_w, ch = whole ? canvas.height : canvas.content_h;
    for (int k = 0; k < 4; ++k) {
        const bool x = (k % 2 == 0);
        const double px = rel[k] / 1000.0 * (x ? canvas.width : canvas.height);   // canvas pixels
        const double f = (px - (x ? cx : cy)) / (x ? cw : ch);
        out[k] = std::min(1.0, std::max(0.0, f));
    }
}

std::string image_verdict_locate_prompt(const std::string& locate) {
    return "Locate " + locate + " in the image, output its bbox coordinates using JSON format.";
}

const char* image_verdict_answer_name(ImageVerdictAnswer a) {
    switch (a) {
        case ImageVerdictAnswer::Yes: return "yes";
        case ImageVerdictAnswer::No: return "no";
        case ImageVerdictAnswer::Unclear: return "unclear";
    }
    return "unclear";
}

// ── The store ────────────────────────────────────────────────────────────────

ImageVerdictStore::ImageVerdictStore(size_t max_images, std::chrono::seconds ttl)
    : max_(max_images), ttl_(ttl) {
    if (max_images == 0)
        throw std::runtime_error("ImageVerdictStore: max_images expected >= 1, actual 0");
}

void ImageVerdictStore::expire(Clock::time_point now) {
    for (auto it = entries_.begin(); it != entries_.end();)
        it = (now - it->second.last_used > ttl_) ? entries_.erase(it) : std::next(it);
}

const ImageVerdictStore::Entry* ImageVerdictStore::find(const std::string& id, uint64_t image_hash,
                                                        const std::vector<int32_t>& prefix_tokens,
                                                        Clock::time_point now) {
    const auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    if (it->second.image_hash != image_hash)
        throw std::runtime_error("\"image_id\": '" + id + "' expected for the image it was kept with, actual "
                                 "a different image — use a new image_id for a new image");
    if (it->second.prefix_tokens != prefix_tokens) return nullptr;   // same image, other prompt: a miss
    it->second.last_used = now;
    return &it->second;
}

void ImageVerdictStore::put(const std::string& id, Entry entry, Clock::time_point now) {
    entry.last_used = now;
    if (!entries_.count(id) && entries_.size() >= max_) {
        auto lru = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it)
            if (it->second.last_used < lru->second.last_used) lru = it;
        entries_.erase(lru);
    }
    entries_[id] = std::move(entry);
}

// ── The driver ───────────────────────────────────────────────────────────────

ImageVerdictReport run_image_verdict(::ForwardPassBase* fp, ggml_backend_sched_t sched,
                                     ::Tokenizer* tok, const ::ModelMetadata& meta,
                                     uint32_t n_ctx_max, const ImageVerdictVision& vision,
                                     const qinf::vision::Bitmap& image,
                                     const ImageVerdictCalibration& cal,
                                     const std::vector<ImageVerdictQuestion>& questions,
                                     ImageVerdictStore* store, const std::string& image_id,
                                     bool where) {
    if ((store == nullptr) != image_id.empty())
        throw std::runtime_error(std::string("run_image_verdict: a store and an image_id expected together, actual ") +
                                 (store ? "a store without an id" : "an id without a store"));
    if (!fp || !tok || !vision.encoder)
        throw std::runtime_error("run_image_verdict: forward pass, tokenizer and encoder expected, actual a null");
    const std::vector<ImageVerdictPlanned> planned = plan_image_verdict_questions(cal, questions);

    // The answer tokens, exactly as run_lens_verdict builds them (yes / no only):
    // the first token of every spelling, the two sets disjoint.
    auto first_tokens = [&](std::initializer_list<const char*> words) {
        std::vector<int32_t> ids;
        for (const char* w : words)
            for (const std::string s : {std::string(w), std::string(" ") + w}) {
                const std::vector<int32_t> t = tok->encode(s);
                if (!t.empty() && std::find(ids.begin(), ids.end(), t[0]) == ids.end()) ids.push_back(t[0]);
            }
        return ids;
    };
    const std::vector<int32_t> yes_ids = first_tokens({"Yes", "yes", "YES", "Ja", "ja"});
    const std::vector<int32_t> no_ids = first_tokens({"No", "no", "NO", "Nein", "nein"});
    for (int32_t t : yes_ids)
        if (std::find(no_ids.begin(), no_ids.end(), t) != no_ids.end())
            throw std::runtime_error("run_image_verdict: answer first tokens expected disjoint, actual shared '" +
                                     tok->decode(t) + "'");

    // The prompt, as measured: image, then "Question: … / Answer with yes or no
    // only.", the family's chat template, thinking off, BOS when the model has one.
    const ChatTemplate* tmpl_ptr = lookup_chat_template(meta.architecture);
    if (!tmpl_ptr)
        throw std::runtime_error("run_image_verdict: a chat template expected for architecture '" +
                                 meta.architecture + "', actual none registered");
    const ChatTemplate& tmpl = *tmpl_ptr;
    const uint32_t n_img = vision.encoder->mm_tokens_for(image);
    uint32_t grid_w = 0, grid_h = 0;
    vision.encoder->mm_grid_for(image, grid_w, grid_h);
    // One user turn: the image, then `user_text`. Questions and "where" share it,
    // so both resume from the same post-image state.
    auto build_turn = [&](const std::string& user_text, int& span_start) {
        std::vector<ChatMessage> turn = {{"user", vision.marker_prefix + user_text}};
        const std::string prompt = tmpl.render(turn, /*add_assistant_prompt=*/true, /*enable_thinking=*/false);
        qinf::image::ExpandedImagePrompt built = qinf::image::expand_image_markers(
            tok->encode(prompt), vision.boi_id, vision.soft_id, vision.eoi_id, n_img);
        std::vector<int32_t> tokens = std::move(built.tokens);
        span_start = built.span_start;
        if (meta.bos_token_id >= 0) { tokens.insert(tokens.begin(), meta.bos_token_id); span_start += 1; }
        if (tokens.size() >= n_ctx_max)
            throw std::runtime_error("run_image_verdict: prompt tokens expected < n_ctx_max=" +
                                     std::to_string(n_ctx_max) + ", actual " + std::to_string(tokens.size()));
        return tokens;
    };
    auto build = [&](const std::string& question, int& span_start) {
        return build_turn("Question: " + question + "\nAnswer with yes or no only.", span_start);
    };
    std::vector<std::vector<int32_t>> prompts(planned.size());
    int span_start = 0;
    for (size_t i = 0; i < planned.size(); ++i) {
        int s = 0;
        prompts[i] = build(planned[i].text, s);
        if (i == 0) span_start = s;
        else if (s != span_start)
            throw std::runtime_error("run_image_verdict: image span expected at token " + std::to_string(span_start) +
                                     " for every question, actual " + std::to_string(s) + " for '" + planned[i].id + "'");
    }
    const size_t img_end = static_cast<size_t>(span_start) + n_img;
    const std::vector<int32_t> image_inclusive(prompts[0].begin(), prompts[0].begin() + img_end);
    for (size_t i = 1; i < prompts.size(); ++i)
        if (prompts[i].size() <= img_end || !std::equal(image_inclusive.begin(), image_inclusive.end(), prompts[i].begin()))
            throw std::runtime_error("run_image_verdict: question '" + planned[i].id +
                                     "' expected to share the image-inclusive prefix, actual differs");

    // The calibration was measured with a MATERIALIZED LLM prefill; a server
    // started with --flash-attn must not answer from different numbers. Pinned
    // here, put back (and slot 0 cleared) however this returns.
    struct EngineRestore {
        ForwardPassBase* fp;
        ForwardPassBase::AttnImpl prefill_impl;
        ~EngineRestore() {
            fp->set_prefill_attn_impl(prefill_impl);
            fp->clear_slot(0);
            fp->set_cache_pos(0, 0);
            fp->reset_rope_pos(0);
        }
    } restore{fp, fp->prefill_attn_impl()};
    fp->set_prefill_attn_impl(ForwardPassBase::AttnImpl::Materialized);

    // The post-image state: kept from an earlier request (image_id hit), or the
    // image pass now — encode + prefill up to the end of the image span, once.
    // The header is built AFTER the prefill mode is pinned: the mode salts it,
    // so a blob kept under another mode can never be restored here.
    const qinf::session::CompatHeader header =
        qinf::snapshot::make_snapshot_header(meta, fp->snapshot_kv_caches());
    const auto now = ImageVerdictStore::Clock::now();
    const ImageVerdictStore::Entry* kept =
        store ? store->find(image_id, image.content_id, image_inclusive, now) : nullptr;
    std::shared_ptr<const std::vector<uint8_t>> blob;
    fp->clear_slot(0);
    fp->set_cache_pos(0, 0);
    fp->reset_rope_pos(0);
    if (kept) {
        blob = kept->blob;
        qinf::snapshot::restore_slot(*fp, 0, *blob, header);
    } else {
        const std::vector<ImagePromptChunk> chunks = {{&image, span_start}};
        (void)prefill_multimodal(*fp, *vision.encoder, sched, image_inclusive, chunks, 0, 0);
        // Keep the post-image state for every question after the first (the
        // first question overwrites the recurrent state) and for the store. The
        // blob carries the rope position (RPOS), so get_rope_pos is exact after
        // each restore.
        if (planned.size() > 1 || store || where)
            blob = std::make_shared<const std::vector<uint8_t>>(qinf::snapshot::capture_slot(*fp, 0, header));
        if (store) {
            ImageVerdictStore::Entry e;
            e.image_hash = image.content_id;
            e.prefix_tokens = image_inclusive;
            e.image_tokens = n_img;
            e.grid_w = grid_w;
            e.grid_h = grid_h;
            e.blob = blob;
            store->put(image_id, std::move(e), now);
        }
    }

    const size_t n_vocab = tok->get_vocabulary().size();
    ImageVerdictReport rep;
    rep.model = cal.model;
    rep.image_tokens = n_img;
    rep.grid_w = grid_w;
    rep.grid_h = grid_h;
    rep.warm = kept != nullptr;
    for (size_t i = 0; i < planned.size(); ++i) {
        if (i > 0) qinf::snapshot::restore_slot(*fp, 0, *blob, header);
        const std::vector<int32_t> suffix(prompts[i].begin() + img_end, prompts[i].end());
        // The question starts at the ROPE position after the image (prefix +
        // max(nx, ny) under M-RoPE), never the KV row count.
        const int q_pos = fp->get_rope_pos(0);
        fp->note_span_rows_vs_positions(0, (uint32_t)suffix.size(), (uint32_t)suffix.size());
        const std::vector<float> logits = fp->run_prefill(suffix, q_pos, 0, sched);
        if (logits.size() < n_vocab)
            throw std::runtime_error("run_image_verdict: logits expected >= one row of " + std::to_string(n_vocab) +
                                     ", actual " + std::to_string(logits.size()));
        const float* row = logits.data() + (logits.size() - n_vocab);
        auto lse = [&](const std::vector<int32_t>& ids) {
            double m = -1e30;
            for (int32_t t : ids) m = std::max(m, (double)row[t]);
            double s = 0;
            for (int32_t t : ids) s += std::exp((double)row[t] - m);
            return m + std::log(s);
        };
        const double margin = lse(yes_ids) - lse(no_ids);
        if (!std::isfinite(margin))
            throw std::runtime_error("run_image_verdict: yes/no logits expected finite for question '" +
                                     planned[i].id + "', actual non-finite");
        ImageVerdictResult r;
        r.id = planned[i].id;
        r.mark = planned[i].mark;
        r.question = planned[i].text;
        r.p_yes = 1.0 / (1.0 + std::exp(-margin));
        r.cut_yes = planned[i].cut_yes;
        r.cut_no = planned[i].cut_no;
        r.calibrated = planned[i].calibrated;
        r.answer = image_verdict_band(r.p_yes, r.cut_yes, r.cut_no);
        r.prompt_tokens = (int)prompts[i].size();
        rep.answers.push_back(r);
    }

    // "where": the model's own box for each yes / unclear answer on a mark with a
    // locate wording (docs/note-verdict-img-ground.md). The answers above are
    // final; this only adds the box. Each locate prompt resumes from the
    // post-image state like a question, then a short greedy generation.
    if (where) {
        constexpr int kWhereMaxTokens = 64;   // a box answer is ~30 tokens
        const std::vector<std::string>& vocab = tok->get_vocabulary();
        const int32_t eos = tok->get_eos_token_id();
        const std::vector<int32_t> im_end = tok->encode("<|im_end|>");
        for (size_t i = 0; i < planned.size(); ++i) {
            ImageVerdictResult& r = rep.answers[i];
            if (planned[i].locate.empty() || r.answer == ImageVerdictAnswer::No) continue;
            int s = 0;
            const std::vector<int32_t> lp = build_turn(image_verdict_locate_prompt(planned[i].locate), s);
            if (s != span_start || lp.size() <= img_end ||
                !std::equal(image_inclusive.begin(), image_inclusive.end(), lp.begin()))
                throw std::runtime_error("run_image_verdict: the locate prompt of '" + planned[i].id +
                                         "' expected to share the image-inclusive prefix, actual differs");
            if (lp.size() + kWhereMaxTokens >= n_ctx_max)
                throw std::runtime_error("run_image_verdict: locate prompt + " + std::to_string(kWhereMaxTokens) +
                                         " tokens expected < n_ctx_max=" + std::to_string(n_ctx_max) +
                                         ", actual " + std::to_string(lp.size() + kWhereMaxTokens));
            qinf::snapshot::restore_slot(*fp, 0, *blob, header);
            const std::vector<int32_t> suffix(lp.begin() + img_end, lp.end());
            const int l_pos = fp->get_rope_pos(0);
            fp->note_span_rows_vs_positions(0, (uint32_t)suffix.size(), (uint32_t)suffix.size());
            std::vector<float> logits = fp->run_prefill(suffix, l_pos, 0, sched);
            if (logits.size() < n_vocab)
                throw std::runtime_error("run_image_verdict: locate logits expected >= one row of " +
                                         std::to_string(n_vocab) + ", actual " + std::to_string(logits.size()));
            std::vector<float> tail(logits.end() - n_vocab, logits.end());
            qinf::GreedySampler sampler(/*repetition_penalty=*/1.0f);
            std::vector<int32_t> history = lp;
            int32_t next = static_cast<int32_t>(sampler.sample(tail, history, vocab));
            std::string text;
            for (int k = 0; k < kWhereMaxTokens; ++k) {
                if (next == eos || (im_end.size() == 1 && next == im_end[0])) break;
                text += tok->decode(next);
                history.push_back(next);
                if (text.find("bbox") != std::string::npos &&
                    text.find(']', text.find("bbox")) != std::string::npos) break;   // the box is closed
                next = decode_step(fp, sched, &sampler, next, /*slot=*/0, history, vocab, (uint32_t)n_vocab);
            }
            double rel[4];
            r.where_asked = true;
            r.has_box = parse_image_verdict_box(text, rel);
            if (r.has_box) image_verdict_box_on_picture(rel, image, r.box);
        }
    }
    return rep;
}

// ── JSON ─────────────────────────────────────────────────────────────────────

std::string image_verdict_to_json(const ImageVerdictReport& rep) {
    nlohmann::json answers = nlohmann::json::array();
    for (const ImageVerdictResult& r : rep.answers) {
        nlohmann::json a = {
            {"id", r.id},
            {"answer", image_verdict_answer_name(r.answer)},
            {"p", {{"yes", r.p_yes}, {"no", 1.0 - r.p_yes}}},
            {"question", r.question},
            {"cut", {{"yes", r.cut_yes}, {"no", r.cut_no}}},
            {"calibrated", r.calibrated},
            {"prompt_len", r.prompt_tokens},
        };
        if (!r.mark.empty()) a["mark"] = r.mark;
        if (r.where_asked) {
            if (r.has_box) {
                nlohmann::json b = nlohmann::json::array();
                for (double v : r.box) b.push_back(std::round(v * 10000.0) / 10000.0);
                a["where"] = {{"box", b}};
            } else {
                a["where"] = {{"box", nullptr}};
            }
        }
        answers.push_back(a);
    }
    const nlohmann::json out = {
        {"format_version", "qemmi-verdict-image/v1"},
        {"model", rep.model},
        {"image", {{"tokens", rep.image_tokens}, {"grid", {rep.grid_w, rep.grid_h}},
                   {"pass", rep.warm ? "warm" : "cold"}}},
        {"answers", answers},
    };
    return out.dump(1);
}

}  // namespace qinf
