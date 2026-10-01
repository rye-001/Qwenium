// VERDICT-IMG probe — does the verdict readout (P(yes) against P(no) off the
// prefill's last row, nothing generated) answer yes/no questions about an
// IMAGE? Logits only: no attention is read, so the closed image-attention
// result (docs/note-image-prefill-tap-probe.md) does not bear on it.
//
// Images and manifest come from probe_verdict_img_render.swift (synthetic
// delivery notes: signature / stamp / delivery date, 2^3 variants per base,
// plus a blank page as the yes-bias baseline). One full-depth prefill per
// question through the production path (prefill_multimodal: encode once per
// image, chunked [pre-image | image | question] over one KV), thinking off.
// Answer tokens: the first token of every spelling, as run_lens_verdict.
//
//   MODEL_PATH=models/Qwen3.8-27B-Q3_K_M.gguf \
//   MMPROJ_PATH=models/Qwen3.8-27B-mmproj-BF16.gguf \
//   DIR=.session-results/verdict_img OUT=.session-results/verdict_img/r27.tsv \
//   ./build-metal/bin/probe-verdict-img
//
// Output TSV: image base family expected p_yes p_no top_token compliant ms n_img
//
// DRIVER mode (DRIVER=1, with DIR/OUT as above): the G2 gate of
// docs/plan-image-verdict.md — per image, every manifest question through BOTH
// the probe path above (one full prefill per question) and run_image_verdict
// (image once, questions resumed from a snapshot), comparing P(yes) as exact
// doubles. Prints the mismatch count; 0 is the gate.
//
// COST mode (COST=img1,img2,... paths from the cwd; no manifest needed): per
// image, (A) the path above — a full re-read per question — and (B) the same
// prefill split at the end of the image span, each part timed: the image pass
// and the question alone. Resuming (B) from a snapshot is NOT possible yet:
// capture_slot refuses a slot holding an image span (Qwen's M-RoPE makes KV
// rows and rope positions diverge; v1 snapshots cannot say so). As a stand-in
// for the snapshot's price, a TEXT slot of the same row count is captured and
// restored (time and size).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "engine/model.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/loader/chat_template.h"
#include "../../src/vision/vision_model.h"
#include "../../src/vision/vision_loader.h"
#include "../../src/vision/vision_profile.h"
#include "../../src/vision/i_vision_encoder.h"
#include "../../src/vision/bitmap.h"
#include "../../src/image/image_loader.h"
#include "../../src/image/image_prompt.h"
#include "../../src/engine/multimodal_prefill.h"
#include "../../src/session/image_embedding_cache.h"
#include "session/slot_snapshot.h"
#include "../../src/server/image_verdict.h"
#include "ggml-backend.h"

namespace {

struct Item { std::string image, base, family, question, expected; };

std::vector<Item> read_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("probe-verdict-img: manifest expected at '" + path + "', actual missing");
    std::vector<Item> v;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> c;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) c.push_back(cell);
        if (c.size() != 5)
            throw std::runtime_error("probe-verdict-img: manifest row expected 5 cells, actual " +
                                     std::to_string(c.size()) + ": " + line);
        v.push_back({c[0], c[1], c[2], c[3], c[4]});
    }
    return v;
}

}  // namespace

int main() {
    auto env = [](const char* k, const char* d) { const char* v = std::getenv(k); return std::string(v ? v : d); };
    const std::string model_path  = env("MODEL_PATH", "models/Qwen3.8-27B-Q3_K_M.gguf");
    const std::string mmproj_path = env("MMPROJ_PATH", "models/Qwen3.8-27B-mmproj-BF16.gguf");
    const std::string dir         = env("DIR", ".session-results/verdict_img");
    const std::string out_path    = env("OUT", ".session-results/verdict_img/results.tsv");
    const uint32_t CTX = 2048;

    register_builtin_models();
    std::cerr << "Loading text model " << model_path << " ...\n";
    Model model;
    model.load_metadata(model_path, /*allow_multimodal=*/true);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    auto fp = create_forward_pass(model, &meta, CTX, 1);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();

    std::cerr << "Loading vision projector " << mmproj_path << " ...\n";
    ggml_backend_t backend = model.has_metal_backend() ? model.get_backend_metal() : model.get_backend_cpu();
    qinf::vision::VisionModel vmodel;
    qinf::vision::VisionLoader vloader;
    vloader.parse_metadata(mmproj_path, vmodel);
    vloader.load_tensors(vmodel, backend);
    qinf::vision::VisionProfile vprofile = qinf::vision::make_vision_profile(
        vmodel, backend, tok->get_vocabulary(), "probe-verdict-img: parameter 'MMPROJ_PATH'");
    if (vprofile.projector_tag != "qwen3vl-merger")
        throw std::runtime_error("probe-verdict-img: projector expected 'qwen3vl-merger', actual '" +
                                 vprofile.projector_tag + "'");

    // Answer tokens exactly as run_lens_verdict builds them (yes / no only here).
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
    const std::vector<int32_t> no_ids  = first_tokens({"No", "no", "NO", "Nein", "nein"});
    for (int32_t t : yes_ids)
        if (std::find(no_ids.begin(), no_ids.end(), t) != no_ids.end())
            throw std::runtime_error("probe-verdict-img: answer first tokens expected disjoint, actual shared '" +
                                     tok->decode(t) + "'");

    const ChatTemplate& tmpl = *lookup_chat_template(meta.architecture);
    const int32_t bos_id = meta.bos_token_id;
    const size_t n_vocab = tok->get_vocabulary().size();

    // The prompt for one question: [BOS] + the rendered turn, image span expanded.
    auto build = [&](const std::string& question, uint32_t n_img, int& span_start) {
        std::vector<ChatMessage> turn = {{"user", vprofile.marker_prefix + "Question: " + question +
                                                  "\nAnswer with yes or no only."}};
        const std::string prompt = tmpl.render(turn, /*add_assistant_prompt=*/true, /*enable_thinking=*/false);
        qinf::image::ExpandedImagePrompt built = qinf::image::expand_image_markers(
            tok->encode(prompt), vprofile.boi_id, vprofile.soft_id, vprofile.eoi_id, n_img);
        std::vector<int32_t> tokens = std::move(built.tokens);
        span_start = built.span_start;
        if (bos_id >= 0) { tokens.insert(tokens.begin(), bos_id); span_start += 1; }
        if (tokens.size() >= CTX)
            throw std::runtime_error("probe-verdict-img: prompt tokens expected < " + std::to_string(CTX) +
                                     ", actual " + std::to_string(tokens.size()));
        return tokens;
    };
    struct Readout { double p_yes, margin; int32_t top; bool compliant; };
    auto read_row = [&](const std::vector<float>& logits) {
        if (logits.size() < n_vocab)
            throw std::runtime_error("probe-verdict-img: logits expected >= n_vocab " + std::to_string(n_vocab) +
                                     ", actual " + std::to_string(logits.size()));
        const float* row = logits.data() + (logits.size() - n_vocab);   // last row
        auto lse = [&](const std::vector<int32_t>& ids) {
            double m = -1e30;
            for (int32_t t : ids) m = std::max(m, (double)row[t]);
            double s = 0;
            for (int32_t t : ids) s += std::exp((double)row[t] - m);
            return m + std::log(s);
        };
        Readout r;
        r.margin = lse(yes_ids) - lse(no_ids);
        r.p_yes = 1.0 / (1.0 + std::exp(-r.margin));
        r.top = 0;
        for (size_t j = 1; j < n_vocab; ++j) if (row[j] > row[r.top]) r.top = (int32_t)j;
        r.compliant = std::find(yes_ids.begin(), yes_ids.end(), r.top) != yes_ids.end() ||
                      std::find(no_ids.begin(), no_ids.end(), r.top) != no_ids.end();
        return r;
    };
    auto ms_since = [](std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    if (env("DRIVER", "") == "1") {
        const std::vector<Item> items = read_manifest(dir + "/manifest.tsv");
        qinf::ImageVerdictVision v;
        v.encoder = vprofile.encoder.get();
        v.marker_prefix = vprofile.marker_prefix;
        v.boi_id = vprofile.boi_id;
        v.soft_id = vprofile.soft_id;
        v.eoi_id = vprofile.eoi_id;
        v.projector = vprofile.projector_tag;
        v.projection_dim = vmodel.config().projection_dim;
        const qinf::ImageVerdictCalibration* cal = qinf::image_verdict_calibration_for(meta, v);
        if (!cal) throw std::runtime_error(qinf::image_verdict_refusal(meta, v));
        std::ofstream o(out_path);
        o << "image\tquestion\tp_probe\tp_driver\tbit_equal\tanswer\tprobe_ms\tdriver_ms\n";
        size_t total = 0, mismatches = 0;
        for (size_t i = 0; i < items.size();) {
            size_t j = i;
            while (j < items.size() && items[j].image == items[i].image) ++j;
            qinf::vision::Bitmap bm = qinf::image::load_image_to_bitmap(dir + "/" + items[i].image, vprofile.preprocess);
            const uint32_t n_img = vprofile.encoder->mm_tokens_for(bm);
            // Probe path: one full prefill per question (encode cached across them).
            ImageEmbeddingCache cache;
            std::vector<double> ref;
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t k = i; k < j; ++k) {
                int ss = 0;
                const std::vector<int32_t> tokens = build(items[k].question, n_img, ss);
                fp->clear_slot(0); fp->set_cache_pos(0, 0); fp->reset_rope_pos(0);
                const std::vector<ImagePromptChunk> chunks = {{&bm, ss}};
                ref.push_back(read_row(prefill_multimodal(*fp, *vprofile.encoder, sched, tokens, chunks, 0, 0, &cache)).p_yes);
            }
            const double probe_ms = ms_since(t0);
            // Driver path: free questions with the manifest's exact text.
            std::vector<qinf::ImageVerdictQuestion> qs;
            for (size_t k = i; k < j; ++k) {
                qinf::ImageVerdictQuestion q;
                q.id = items[k].family + "_" + std::to_string(k - i);
                q.question = items[k].question;
                qs.push_back(q);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const qinf::ImageVerdictReport rep =
                qinf::run_image_verdict(fp.get(), sched, tok, meta, CTX, v, bm, *cal, qs);
            const double driver_ms = ms_since(t1);
            for (size_t k = i; k < j; ++k) {
                const qinf::ImageVerdictResult& r = rep.answers[k - i];
                const bool eq = r.p_yes == ref[k - i];
                ++total;
                mismatches += !eq;
                char buf[512];
                std::snprintf(buf, sizeof buf, "%s\t%s\t%.17g\t%.17g\t%d\t%s\t%.0f\t%.0f\n", items[k].image.c_str(),
                              items[k].family.c_str(), ref[k - i], r.p_yes, eq ? 1 : 0,
                              qinf::image_verdict_answer_name(r.answer), probe_ms, driver_ms);
                o << buf;
            }
            o.flush();
            std::fprintf(stderr, "[%zu/%zu] %s probe %.0f ms, driver %.0f ms, mismatches so far %zu\n", j,
                         items.size(), items[i].image.c_str(), probe_ms, driver_ms, mismatches);
            i = j;
        }
        std::printf("G2: %zu questions, %zu P(yes) not bit-equal to the probe path\n", total, mismatches);
        return mismatches == 0 ? 0 : 3;
    }

    // ENCODE mode (ENCODE=image path): the vision tower alone, REPS times.
    const std::string encode_path = env("ENCODE", "");
    if (!encode_path.empty()) {
        const int reps = std::stoi(env("REPS", "3"));
        qinf::vision::Bitmap bm = qinf::image::load_image_to_bitmap(encode_path, vprofile.preprocess);
        std::printf("encode %s: %dx%d px, %u merged tokens\n", encode_path.c_str(), bm.width, bm.height,
                    vprofile.encoder->mm_tokens_for(bm));
        std::vector<float> e;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            e = vprofile.encoder->encode(bm);
            std::printf("  rep %d: %.0f ms (%zu floats)\n", r, ms_since(t0), e.size());
            std::fflush(stdout);
        }
        // DUMP=path: the embeddings as raw float32, to diff encoder versions.
        const std::string dump = env("DUMP", "");
        if (!dump.empty()) {
            std::ofstream d(dump, std::ios::binary);
            d.write(reinterpret_cast<const char*>(e.data()), (std::streamsize)(e.size() * sizeof(float)));
            std::printf("  dumped %zu floats to %s\n", e.size(), dump.c_str());
        }
        return 0;
    }

    const std::string cost = env("COST", "");
    if (!cost.empty()) {
        std::vector<std::string> paths;
        { std::stringstream ss(cost); std::string p; while (std::getline(ss, p, ',')) if (!p.empty()) paths.push_back(p); }
        const std::vector<std::string> qs = {
            "Is the delivery note signed in the 'Received by' box?",
            "Does the delivery note carry a stamp?",
            "Is the delivery date filled in?",
            "Is there a table of delivered items?",
            "Is the company name printed at the top of the page?",
            "Is there a handwritten note in the margin?",
            "Is the 'Order ref' field filled in?",
            "Is there a company logo?",
            "Is the delivery note written in German?",
            "Is any part of the page torn or missing?",
        };
        const qinf::session::CompatHeader header =
            qinf::snapshot::make_snapshot_header(meta, fp->snapshot_kv_caches());
        std::ofstream cout_tsv(out_path);
        cout_tsv << "image\tquestion\tfull_ms\tquestion_only_ms\tp_full\tp_split\tmargin_full\tmargin_split\n";
        std::printf("image  n_img  encode_ms  full_q_ms  image_pass_ms  question_only_ms  "
                    "text_capture_ms  text_restore_ms  blob_MB  max|dmargin|  answers_agree  top_agree\n");
        for (const std::string& path : paths) {
            qinf::vision::Bitmap bm = qinf::image::load_image_to_bitmap(path, vprofile.preprocess);
            const uint32_t n_img = vprofile.encoder->mm_tokens_for(bm);

            // (A) a full re-read per question; the first call also encodes.
            ImageEmbeddingCache c_full;
            std::vector<Readout> full(qs.size());
            std::vector<double> full_ms(qs.size());
            for (size_t q = 0; q < qs.size(); ++q) {
                int ss = 0;
                const std::vector<int32_t> tokens = build(qs[q], n_img, ss);
                const auto t0 = std::chrono::steady_clock::now();
                fp->clear_slot(0); fp->set_cache_pos(0, 0); fp->reset_rope_pos(0);
                const std::vector<ImagePromptChunk> chunks = {{&bm, ss}};
                full[q] = read_row(prefill_multimodal(*fp, *vprofile.encoder, sched, tokens, chunks, 0, 0, &c_full));
                full_ms[q] = ms_since(t0);
            }
            double full_rest = 0;
            for (size_t q = 1; q < qs.size(); ++q) full_rest += full_ms[q];
            full_rest /= (double)(qs.size() - 1);
            const double encode_ms = full_ms[0] - full_rest;

            // (B) the same prefill split at the image end: image pass, then the question alone.
            int ss0 = 0;
            const std::vector<int32_t> t0tok = build(qs[0], n_img, ss0);
            const int img_end = ss0 + (int)n_img;
            const std::vector<int32_t> image_inclusive(t0tok.begin(), t0tok.begin() + img_end);
            ImageEmbeddingCache c_res;
            double image_sum = 0, res_sum = 0, max_dm = 0;
            int agree = 0, top_agree = 0;
            for (size_t q = 0; q < qs.size(); ++q) {
                int ss = 0;
                const std::vector<int32_t> tokens = build(qs[q], n_img, ss);
                if (ss != ss0 || !std::equal(image_inclusive.begin(), image_inclusive.end(), tokens.begin()))
                    throw std::runtime_error("probe-verdict-img: question " + std::to_string(q) +
                                             " expected to share the image-inclusive prefix, actual differs");
                const std::vector<int32_t> suffix(tokens.begin() + img_end, tokens.end());
                auto t0 = std::chrono::steady_clock::now();
                fp->clear_slot(0); fp->set_cache_pos(0, 0); fp->reset_rope_pos(0);
                const std::vector<ImagePromptChunk> chunks0 = {{&bm, ss0}};
                (void)prefill_multimodal(*fp, *vprofile.encoder, sched, image_inclusive, chunks0, 0, 0, &c_res);
                if (q > 0) image_sum += ms_since(t0);            // q = 0 also encodes
                // Start at the ROPE position after the image (prefix + max(nx, ny) under
                // M-RoPE), not the KV row count — exactly as drive_prefill_chunks does.
                const int q_pos = (int)fp->get_rope_pos(0);
                const auto tq = std::chrono::steady_clock::now();
                fp->note_span_rows_vs_positions(0, (uint32_t)suffix.size(), (uint32_t)suffix.size());
                const Readout r = read_row(fp->run_prefill(suffix, q_pos, 0, sched));
                const double qms = ms_since(tq);
                res_sum += qms;
                max_dm = std::max(max_dm, std::fabs(r.margin - full[q].margin));
                agree += (r.p_yes >= 0.5) == (full[q].p_yes >= 0.5);
                top_agree += r.top == full[q].top;
                char buf[512];
                std::snprintf(buf, sizeof buf, "%s\t%zu\t%.0f\t%.0f\t%.6f\t%.6f\t%.6f\t%.6f\n", path.c_str(), q,
                              full_ms[q], qms, full[q].p_yes, r.p_yes, full[q].margin, r.margin);
                cout_tsv << buf;
            }
            const double image_ms = image_sum / (double)(qs.size() - 1);

            // Stand-in for the snapshot's price: a text slot with the same row count.
            std::vector<int32_t> filler = tok->encode("The quick brown fox jumps over the lazy dog. ");
            std::vector<int32_t> text;
            while ((int)text.size() < img_end) text.insert(text.end(), filler.begin(), filler.end());
            text.resize((size_t)img_end);
            fp->clear_slot(0); fp->set_cache_pos(0, 0); fp->reset_rope_pos(0);
            (void)fp->run_prefill(text, 0, 0, sched);
            auto t0 = std::chrono::steady_clock::now();
            const std::vector<uint8_t> blob = qinf::snapshot::capture_slot(*fp, 0, header);
            const double capture_ms = ms_since(t0);
            t0 = std::chrono::steady_clock::now();
            qinf::snapshot::restore_slot(*fp, 0, blob, header);
            const double restore_ms = ms_since(t0);
            cout_tsv.flush();
            std::printf("%s  %u  %.0f  %.0f  %.0f  %.0f  %.0f  %.0f  %.1f  %.2e  %d/%zu  %d/%zu\n", path.c_str(),
                        n_img, encode_ms, full_rest, image_ms, res_sum / (double)qs.size(), capture_ms, restore_ms,
                        blob.size() / 1048576.0, max_dm, agree, qs.size(), top_agree, qs.size());
            std::fflush(stdout);
        }
        return 0;
    }

    const std::vector<Item> items = read_manifest(dir + "/manifest.tsv");
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("probe-verdict-img: OUT expected writable, actual '" + out_path + "'");
    out << "image\tbase\tfamily\texpected\tp_yes\tp_no\ttop_token\tcompliant\tms\tn_img\n";

    std::string cur_image;
    qinf::vision::Bitmap bitmap;
    std::unique_ptr<ImageEmbeddingCache> cache;
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& it = items[i];
        const auto t0 = std::chrono::steady_clock::now();
        if (it.image != cur_image) {
            bitmap = qinf::image::load_image_to_bitmap(dir + "/" + it.image, vprofile.preprocess);
            cache = std::make_unique<ImageEmbeddingCache>();   // encode once per image
            cur_image = it.image;
        }
        const uint32_t n_img = vprofile.encoder->mm_tokens_for(bitmap);
        int span_start = 0;
        const std::vector<int32_t> tokens = build(it.question, n_img, span_start);

        fp->clear_slot(0); fp->set_cache_pos(0, 0); fp->reset_rope_pos(0);
        const std::vector<ImagePromptChunk> chunks = {{&bitmap, span_start}};
        const Readout r = read_row(prefill_multimodal(*fp, *vprofile.encoder, sched, tokens, chunks, 0, 0, cache.get()));
        const double ms = ms_since(t0);

        std::string top_s = tok->decode(r.top);
        for (char& c : top_s) if (c == '\t' || c == '\n') c = ' ';
        char buf[512];
        std::snprintf(buf, sizeof buf, "%s\t%s\t%s\t%s\t%.6f\t%.6f\t%s\t%d\t%.0f\t%u\n",
                      it.image.c_str(), it.base.c_str(), it.family.c_str(), it.expected.c_str(),
                      r.p_yes, 1.0 - r.p_yes, top_s.c_str(), r.compliant ? 1 : 0, ms, n_img);
        out << buf; out.flush();
        std::fprintf(stderr, "[%zu/%zu] %s", i + 1, items.size(), buf);
    }
    return 0;
}
