// Prefill image-lens probe (v2) — follow-up to docs/note-image-prefill-tap-probe.md.
//
// v1 (see git history / the note's §1-§6) found G0 "promising" and G1
// "qualified pass" from a single-object row sweep. The architect's review
// found the row-sweep-only G1 argument circular (the control band overlapped
// the target at the shallowest depth) and the G0 number uninterpretable
// without a chance baseline. This version:
//
//   PART A — re-runs the same 4-depth single-object sweep (rows 2/4/8/12,
//   16x16 grid, col 8 fixed), this time computing the ADDITIONAL diagnostic
//   v1 never persisted: total image-span mass per (layer,head), which is the
//   denominator needed for "ratio to uniform-over-image-span" (chance = k/n_img
//   of the image span's own mass, sink-excluded, k = target-cell count). Reports
//   mean/median/max ratio and how many of the 384 (layer,head) pairs clear a
//   stated bar — the analogue of the original decode probe's round-2 "~6%
//   clear it" figure.
//
//   PART B — the column-swap (here: row-band-constant, LEFT/RIGHT-swap)
//   control the original brief wrongly forbade: two objects (red circle, blue
//   square) in the SAME row band, at DEEP rows (8, 12), two horizontal
//   layouts (object positions swapped), two questions per image (which
//   object is named). Reports per-head target vs distractor mass, ratio to
//   chance, and whether it follows the NAMED object or a fixed side.
//
// Same tap machinery as v1: kq_soft.<il> read directly (never
// get_attention_taps, which is decode-shaped and unsafe at prefill), rows
// self-tested to sum to 1.0, chunked prefill ([prefix|image|suffix], one
// shared KV) driven manually so taps can be marked before each chunk's alloc.
//
//   MODEL_PATH=... MMPROJ_PATH=... ./bin/probe-image-prefill-lens

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "engine/model.h"
#include "engine/graph_compute.h"
#include "../../src/models/model_registry.h"
#include "../../src/models/forward_pass_base.h"
#include "../../src/models/i_image_embeddable.h"
#include "../../src/loader/tokenizer.h"
#include "../../src/loader/chat_template.h"
#include "../../src/vision/vision_model.h"
#include "../../src/vision/vision_loader.h"
#include "../../src/vision/vision_profile.h"
#include "../../src/vision/i_vision_encoder.h"
#include "../../src/vision/bitmap.h"
#include "../../src/image/image_loader.h"
#include "../../src/image/image_prompt.h"
#include "ggml.h"
#include "ggml-backend.h"

namespace {

// ── BMP writers (no external deps; stb_image decodes BMP) ──────────────────
void bmp_header(std::ofstream& f, int w, int h) {
    const int row_pad = (4 - (w * 3) % 4) % 4;
    const uint32_t pixel_data_size = static_cast<uint32_t>(w * 3 + row_pad) * h;
    const uint32_t file_size = 54 + pixel_data_size;
    auto put16 = [&](uint16_t v) { f.write(reinterpret_cast<char*>(&v), 2); };
    auto put32 = [&](uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
    auto put32s = [&](int32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
    f.write("BM", 2);
    put32(file_size); put16(0); put16(0); put32(54);
    put32(40); put32s(w); put32s(h); put16(1); put16(24); put32(0);
    put32(pixel_data_size); put32s(2835); put32s(2835); put32(0); put32(0);
}

void write_bmp_circle(const std::string& path, int w, int h, int cx, int cy, int radius) {
    const int row_pad = (4 - (w * 3) % 4) % 4;
    const int row_size = w * 3 + row_pad;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_bmp_circle: cannot open " + path);
    bmp_header(f, w, h);
    const long long r2 = static_cast<long long>(radius) * radius;
    std::vector<unsigned char> buf(row_size, 255);
    for (int row = h - 1; row >= 0; --row) {
        std::fill(buf.begin(), buf.end(), (unsigned char)255);
        for (int col = 0; col < w; ++col) {
            const long long dx = col - cx, dy = row - cy;
            if (dx * dx + dy * dy <= r2) { buf[col*3+0]=20; buf[col*3+1]=20; buf[col*3+2]=220; }  // red (BGR)
        }
        for (int p = 0; p < row_pad; ++p) buf[w * 3 + p] = 0;
        f.write(reinterpret_cast<char*>(buf.data()), row_size);
    }
}

// Red circle + blue square, same row band, at (circle_cx,cy) and (square_cx,cy).
// Square half-width == radius, so both objects cover an identical-sized grid
// footprint (fair chance normalization between target and distractor).
void write_bmp_two_objects(const std::string& path, int w, int h, int cy,
                           int circle_cx, int square_cx, int radius) {
    const int row_pad = (4 - (w * 3) % 4) % 4;
    const int row_size = w * 3 + row_pad;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_bmp_two_objects: cannot open " + path);
    bmp_header(f, w, h);
    const long long r2 = static_cast<long long>(radius) * radius;
    std::vector<unsigned char> buf(row_size, 255);
    for (int row = h - 1; row >= 0; --row) {
        std::fill(buf.begin(), buf.end(), (unsigned char)255);
        for (int col = 0; col < w; ++col) {
            const long long dxc = col - circle_cx, dyc = row - cy;
            const bool in_circle = dxc*dxc + dyc*dyc <= r2;
            const bool in_square = std::abs(col - square_cx) <= radius && std::abs(row - cy) <= radius;
            if (in_circle)      { buf[col*3+0]=20;  buf[col*3+1]=20;  buf[col*3+2]=220; }  // red
            else if (in_square) { buf[col*3+0]=220; buf[col*3+1]=20;  buf[col*3+2]=20;  }  // blue
        }
        for (int p = 0; p < row_pad; ++p) buf[w * 3 + p] = 0;
        f.write(reinterpret_cast<char*>(buf.data()), row_size);
    }
}

std::set<int> overlapping_cells(int center, int radius, int align, int n_cells) {
    std::set<int> out;
    int lo = std::max(0, (center - radius) / align);
    int hi = std::min(n_cells - 1, (center + radius) / align);
    for (int i = lo; i <= hi; ++i) out.insert(i);
    return out;
}

// ── Raw prefill attention tensor readback ───────────────────────────────────
// kq_soft.<il> ne = [n_kv, n_q, n_head, 1]; element (kv,q,h) at
// data[kv + n_kv*(q + n_q*h)].
struct RawTap {
    int layer = -1;
    int64_t n_kv = 0, n_q = 0, n_head = 0;
    std::vector<float> data;
    float at(int64_t kv, int64_t q, int64_t h) const { return data[kv + n_kv*(q + n_q*h)]; }
};

std::vector<RawTap> read_prefill_taps(ggml_cgraph* gf, const std::vector<int32_t>& layers) {
    std::vector<RawTap> out;
    out.reserve(layers.size());
    for (int il : layers) {
        std::string nm = "kq_soft." + std::to_string(il);
        ggml_tensor* ts = ggml_graph_get_tensor(gf, nm.c_str());
        if (!ts) throw std::runtime_error("read_prefill_taps: tensor '" + nm + "' absent");
        if (ts->type != GGML_TYPE_F32)
            throw std::runtime_error("read_prefill_taps: tensor '" + nm + "' not f32");
        RawTap t; t.layer = il; t.n_kv = ts->ne[0]; t.n_q = ts->ne[1]; t.n_head = ts->ne[2];
        t.data.resize((size_t)ggml_nelements(ts));
        ggml_backend_tensor_get(ts, t.data.data(), 0, ggml_nbytes(ts));
        out.push_back(std::move(t));
    }
    return out;
}

void selftest_rows_sum_to_one(const std::vector<RawTap>& taps, const char* label) {
    double max_dev = 0.0; int checked = 0;
    for (const auto& t : taps) {
        int64_t q_stride = std::max<int64_t>(1, t.n_q / 8);
        for (int64_t q = 0; q < t.n_q; q += q_stride)
            for (int64_t h = 0; h < t.n_head; ++h) {
                double s = 0.0;
                for (int64_t kv = 0; kv < t.n_kv; ++kv) s += t.at(kv, q, h);
                max_dev = std::max(max_dev, std::abs(s - 1.0)); ++checked;
            }
    }
    std::printf("[selftest %s] checked %d rows, max|sum-1|=%.6f\n", label, checked, max_dev);
    if (max_dev > 0.01) throw std::runtime_error(std::string("selftest[") + label + "] failed");
}

// Per (layer,head): mass on target cells, distractor cells (optional), the
// single sink token (image-local index 0), and the WHOLE image span
// (sink-included) — averaged over query rows [q_lo,q_hi] inclusive.
struct PairMass2 { int layer, head; double target, distractor, sink, total; };

std::vector<PairMass2> masses_over_qrange(
    const std::vector<RawTap>& taps, int prefix_len, int n_img, int64_t q_lo, int64_t q_hi,
    const std::set<int>& obj_rows, const std::set<int>& obj_cols,
    const std::set<int>* dis_rows, const std::set<int>* dis_cols, int grid_w) {
    std::vector<PairMass2> out;
    for (const auto& t : taps) {
        int64_t lo = std::max<int64_t>(0, q_lo), hi = std::min<int64_t>(t.n_q - 1, q_hi);
        for (int64_t h = 0; h < t.n_head; ++h) {
            double target_sum = 0, dis_sum = 0, sink_sum = 0, total_sum = 0;
            int64_t nrows = 0;
            for (int64_t q = lo; q <= hi; ++q) {
                ++nrows;
                for (int j = 0; j < n_img; ++j) {
                    int64_t kv = prefix_len + j;
                    float v = t.at(kv, q, h);
                    total_sum += v;
                    if (j == 0) sink_sum += v;
                    int row = j / grid_w, col = j % grid_w;
                    if (obj_rows.count(row) && obj_cols.count(col)) target_sum += v;
                    if (dis_rows && dis_cols && dis_rows->count(row) && dis_cols->count(col)) dis_sum += v;
                }
            }
            double n = (double)std::max<int64_t>(1, nrows);
            out.push_back({t.layer, (int)h, target_sum/n, dis_sum/n, sink_sum/n, total_sum/n});
        }
    }
    return out;
}

struct RatioPoint { int layer, head; double ratio; };

// ratio = observed_mass / chance_mass, chance_mass = (total-sink) * k/(n_img-1)
// (uniform spread over the image span excluding the single sink token).
std::vector<RatioPoint> to_ratios(const std::vector<PairMass2>& v, int k, int n_img, bool use_target) {
    std::vector<RatioPoint> out; out.reserve(v.size());
    for (auto& p : v) {
        double denom_total = p.total - p.sink;
        double chance = denom_total * (k / (double)(n_img - 1));
        double m = use_target ? p.target : p.distractor;
        out.push_back({p.layer, p.head, (chance > 1e-12) ? m / chance : 0.0});
    }
    return out;
}

struct RStats { double mean=0, median=0, max=-1; int max_layer=-1, max_head=-1; };
RStats stats_of(std::vector<RatioPoint> pts) {
    RStats s;
    if (pts.empty()) return s;
    double sum = 0;
    for (auto& p : pts) { sum += p.ratio; if (p.ratio > s.max) { s.max = p.ratio; s.max_layer = p.layer; s.max_head = p.head; } }
    s.mean = sum / pts.size();
    std::vector<double> vals; vals.reserve(pts.size());
    for (auto& p : pts) vals.push_back(p.ratio);
    std::sort(vals.begin(), vals.end());
    size_t m = vals.size() / 2;
    s.median = (vals.size() % 2 == 0) ? 0.5 * (vals[m-1] + vals[m]) : vals[m];
    return s;
}
int count_clear(const std::vector<RatioPoint>& pts, double bar) {
    int c = 0; for (auto& p : pts) if (p.ratio >= bar) ++c; return c;
}
double find_ratio(const std::vector<RatioPoint>& pts, int layer, int head) {
    for (auto& p : pts) if (p.layer == layer && p.head == head) return p.ratio;
    return -1.0;
}

}  // namespace

int main() {
    const char* model_env  = std::getenv("MODEL_PATH");
    const char* mmproj_env = std::getenv("MMPROJ_PATH");
    const std::string model_path  = model_env  ? model_env  : "models/Qwen3.8-27B-Q3_K_M.gguf";
    const std::string mmproj_path = mmproj_env ? mmproj_env : "models/Qwen3.8-27B-mmproj-BF16.gguf";
    const uint32_t CTX = 2048;
    const int WATCH_LAYER = 23, WATCH_HEAD = 10;  // v1's standout head, L23H10

    register_builtin_models();
    std::cerr << "Loading text model " << model_path << " ...\n";
    Model model;
    model.load_metadata(model_path, /*allow_multimodal=*/true);
    model.load_tensors();
    const auto& meta = model.get_metadata();
    auto fp = create_forward_pass(model, &meta, CTX, 1);
    ggml_backend_sched_t sched = model.get_scheduler();
    Tokenizer* tok = model.get_tokenizer();

    IImageEmbeddable* embeddable = dynamic_cast<IImageEmbeddable*>(fp.get());
    if (!embeddable) { std::fprintf(stderr, "probe: expected IImageEmbeddable recipe\n"); return 1; }

    std::cerr << "Loading vision projector " << mmproj_path << " ...\n";
    ggml_backend_t backend = model.has_metal_backend() ? model.get_backend_metal() : model.get_backend_cpu();
    qinf::vision::VisionModel vmodel;
    qinf::vision::VisionLoader vloader;
    vloader.parse_metadata(mmproj_path, vmodel);
    vloader.load_tensors(vmodel, backend);
    qinf::vision::VisionProfile vprofile = qinf::vision::make_vision_profile(
        vmodel, backend, tok->get_vocabulary(), "probe_image_prefill_lens: parameter '--mmproj'");
    const uint32_t align = vmodel.config().patch_size * vmodel.config().n_merge;
    std::printf("vision: patch_size=%u n_merge=%u align=%u projector=%s\n",
               vmodel.config().patch_size, vmodel.config().n_merge, align, vprofile.projector_tag.c_str());

    std::vector<int32_t> attn_layers;
    {
        std::vector<int32_t> warm = tok->encode("Hello");
        fp->clear_slot(0); fp->set_cache_pos(0, 0);
        fp->run_prefill(warm, 0, 0, sched);
        std::vector<int32_t> t = {warm.back()}; std::vector<uint32_t> s = {0};
        std::vector<int32_t> p = {(int32_t)fp->get_cache_pos(0)};
        ggml_cgraph* gscan = fp->build_decoding_graph(t, s, p);
        for (uint32_t il = 0; il < meta.block_count; ++il) {
            std::string nm = "kq_soft." + std::to_string(il);
            if (ggml_graph_get_tensor(gscan, nm.c_str())) attn_layers.push_back((int32_t)il);
        }
        fp->clear_slot(0); fp->set_cache_pos(0, 0);
    }
    if (attn_layers.empty()) { std::fprintf(stderr, "tap discovery: 0 layers\n"); return 1; }
    std::printf("arch=%s block_count=%u n_head_q=%u\nattention layers (%zu): ",
               meta.architecture.c_str(), meta.block_count, meta.attention_head_count, attn_layers.size());
    for (int il : attn_layers) std::printf("%d ", il);
    std::printf("\n\n");

    const int GRID = 16;
    const int canvas = (int)align * GRID;
    const int radius = (int)(0.9 * align);
    const int32_t bos_id = meta.bos_token_id;
    const int32_t eos = tok->get_eos_token_id();

    // Runs one full [prefix|image|suffix] chunked prefill for a given bitmap
    // file + question. Returns taps for last-row and all-rows q-ranges, plus
    // the perception-gated free-gen answer. Fresh clear_slot every call.
    struct RunOut {
        std::string answer;
        std::vector<RawTap> taps_q;  // suffix chunk
        int prefix_len; uint32_t n_img, grid_w, grid_h;
    };
    auto run_one = [&](const std::string& bmp_path, const std::string& question) -> RunOut {
        fp->clear_slot(0); fp->set_cache_pos(0, 0); fp->reset_rope_pos(0);
        qinf::vision::Bitmap bitmap = qinf::image::load_image_to_bitmap(bmp_path, vprofile.preprocess);
        uint32_t grid_w = 0, grid_h = 0;
        vprofile.encoder->mm_grid_for(bitmap, grid_w, grid_h);
        const uint32_t n_img = vprofile.encoder->mm_tokens_for(bitmap);
        std::vector<float> embd = vprofile.encoder->encode(bitmap);

        std::vector<ChatMessage> turn;
        if (vprofile.wants_thinking) turn.push_back({"system", ""});
        turn.push_back({"user", vprofile.marker_prefix + question});
        const ChatTemplate& tmpl = *lookup_chat_template(meta.architecture);
        std::string turn_prompt = tmpl.render(turn, true,
            vprofile.wants_thinking ? std::optional<bool>(true) : std::nullopt);
        std::vector<int32_t> raw = tok->encode(turn_prompt);
        qinf::image::ExpandedImagePrompt built = qinf::image::expand_image_markers(
            raw, vprofile.boi_id, vprofile.soft_id, vprofile.eoi_id, n_img);
        std::vector<int32_t> tokens = built.tokens;
        int span_start = built.span_start;
        if (bos_id >= 0) { tokens.insert(tokens.begin(), bos_id); span_start += 1; }

        const uint32_t img_pos_advance = embeddable->image_span_is_2d() ? std::max(grid_w, grid_h) : n_img;
        std::vector<int32_t> prefix_tokens(tokens.begin(), tokens.begin() + span_start);
        std::vector<int32_t> image_tokens(tokens.begin() + span_start, tokens.begin() + span_start + n_img);
        std::vector<int32_t> suffix_tokens(tokens.begin() + span_start + n_img, tokens.end());

        int pos = 0;
        if (!prefix_tokens.empty()) {
            fp->note_span_rows_vs_positions(0, (uint32_t)prefix_tokens.size(), (uint32_t)prefix_tokens.size());
            fp->feed_tokens(prefix_tokens, 0, sched, pos);
            pos += (int)prefix_tokens.size();
        }
        const int prefix_len = span_start;

        embeddable->set_image_embeddings(std::move(embd), 0, n_img, grid_w, grid_h);
        fp->note_span_rows_vs_positions(0, n_img, img_pos_advance);
        fp->set_attention_taps(attn_layers);
        ggml_backend_sched_reset(sched);
        ggml_cgraph* gf_img = fp->build_prefill_graph(image_tokens, pos, 0, false);
        fp->mark_attention_taps(gf_img);
        ggml_backend_sched_alloc_graph(sched, gf_img);
        fp->set_prefill_inputs(gf_img, image_tokens, pos);
        qinf::engine::require_compute_success(ggml_backend_sched_graph_compute(sched, gf_img), "image chunk");
        fp->advance_cache(n_img, 0);
        pos += (int)img_pos_advance;

        fp->note_span_rows_vs_positions(0, (uint32_t)suffix_tokens.size(), (uint32_t)suffix_tokens.size());
        fp->set_attention_taps(attn_layers);
        ggml_backend_sched_reset(sched);
        ggml_cgraph* gf_q = fp->build_prefill_graph(suffix_tokens, pos, 0, true);
        fp->mark_attention_taps(gf_q);
        ggml_backend_sched_alloc_graph(sched, gf_q);
        fp->set_prefill_inputs(gf_q, suffix_tokens, pos);
        qinf::engine::require_compute_success(ggml_backend_sched_graph_compute(sched, gf_q), "suffix chunk");
        std::vector<RawTap> taps_q = read_prefill_taps(gf_q, attn_layers);
        selftest_rows_sum_to_one(taps_q, "suffix-chunk");
        std::vector<float> logits = fp->get_output_logits(gf_q);
        fp->advance_cache((uint32_t)suffix_tokens.size(), 0);
        pos += (int)suffix_tokens.size();

        int32_t next = 0;
        for (size_t j = 1; j < logits.size(); ++j) if (logits[j] > logits[next]) next = (int32_t)j;
        std::vector<int32_t> gen;
        for (int t = 0; t < 12; ++t) {
            if (next == eos) break;
            gen.push_back(next);
            std::vector<int32_t> tks = {next}; std::vector<uint32_t> slots = {0};
            std::vector<int32_t> positions = {(int32_t)fp->get_cache_pos(0)};
            ggml_cgraph* gfd = fp->build_decoding_graph(tks, slots, positions);
            ggml_backend_sched_reset(sched);
            ggml_backend_sched_alloc_graph(sched, gfd);
            fp->set_decode_inputs(gfd, tks, slots, positions);
            qinf::engine::require_compute_success(ggml_backend_sched_graph_compute(sched, gfd), "decode");
            std::vector<float> lg = fp->get_output_logits(gfd);
            next = 0; for (size_t j = 1; j < lg.size(); ++j) if (lg[j] > lg[next]) next = (int32_t)j;
            fp->advance_cache(1, 0);
        }
        RunOut R; R.answer = tok->decode(gen); R.taps_q = std::move(taps_q);
        R.prefix_len = prefix_len; R.n_img = n_img; R.grid_w = grid_w; R.grid_h = grid_h;
        return R;
    };

    // ═══════════════════════ PART A — depth sweep w/ chance baseline ═══════════
    std::printf("################ PART A: depth sweep + chance baseline ################\n");
    const int col_target = 8;
    const int cx = col_target * (int)align + (int)align / 2;
    struct DepthCase { int row; std::string path; };
    std::vector<DepthCase> cases = {
        {2,  ".session-results/prefill_probe/depth_r2.bmp"},
        {4,  ".session-results/prefill_probe/depth_r4.bmp"},
        {8,  ".session-results/prefill_probe/depth_r8.bmp"},
        {12, ".session-results/prefill_probe/depth_r12.bmp"},
    };
    for (auto& c : cases) {
        int cy = c.row * (int)align + (int)align / 2;
        write_bmp_circle(c.path, canvas, canvas, cx, cy, radius);
    }
    const std::string q_circle = "What colour is the circle? Answer with one word.";

    for (auto& c : cases) {
        std::printf("---- depth row %d/16 ----\n", c.row);
        RunOut R = run_one(c.path, q_circle);
        std::string lower = R.answer; for (auto& ch : lower) ch = (char)tolower((unsigned char)ch);
        bool pass = lower.find("red") != std::string::npos;
        std::printf("answer=\"%s\" perception=%s grid=%ux%u n_img=%u\n",
                   R.answer.c_str(), pass ? "PASS" : "FAIL", R.grid_w, R.grid_h, R.n_img);
        if (!pass) { std::printf("SKIPPED (perception gate fail)\n\n"); continue; }

        int cy = c.row * (int)align + (int)align / 2;
        std::set<int> obj_rows = overlapping_cells(cy, radius, (int)align, (int)R.grid_h);
        std::set<int> obj_cols = overlapping_cells(cx, radius, (int)align, (int)R.grid_w);
        const int k = (int)(obj_rows.size() * obj_cols.size());
        std::printf("target cells: %d rows x %d cols = %d of %u (%.2f%% of image span)\n",
                   (int)obj_rows.size(), (int)obj_cols.size(), k, R.n_img, 100.0 * k / R.n_img);

        auto lr = masses_over_qrange(R.taps_q, R.prefix_len, (int)R.n_img,
                                     (int64_t)R.taps_q[0].n_q - 1, (int64_t)R.taps_q[0].n_q - 1,
                                     obj_rows, obj_cols, nullptr, nullptr, (int)R.grid_w);
        auto ar = masses_over_qrange(R.taps_q, R.prefix_len, (int)R.n_img, 0, R.taps_q[0].n_q - 1,
                                     obj_rows, obj_cols, nullptr, nullptr, (int)R.grid_w);

        for (auto& pair : {std::make_pair("LAST-ROW", &lr), std::make_pair("ALL-ROWS", &ar)}) {
            const char* label = pair.first; auto* v = pair.second;
            // image span's share of raw row mass (population mean, sink-included)
            double total_mean = 0; for (auto& p : *v) total_mean += p.total; total_mean /= v->size();
            double sink_mean = 0;  for (auto& p : *v) sink_mean  += p.sink;  sink_mean  /= v->size();
            auto ratios = to_ratios(*v, k, (int)R.n_img, true);
            RStats s = stats_of(ratios);
            double watch = find_ratio(ratios, WATCH_LAYER, WATCH_HEAD);
            std::printf("%-9s image-span-share(mean)=%.4f sink-share(mean)=%.4f | ratio-to-chance: "
                       "mean=%.2fx median=%.2fx max=%.2fx@L%dH%d | clear>=2x:%d/%zu >=3x:%d/%zu >=5x:%d/%zu "
                       "| L%dH%d(watch)=%.2fx\n",
                       label, total_mean, sink_mean, s.mean, s.median, s.max, s.max_layer, s.max_head,
                       count_clear(ratios, 2.0), v->size(), count_clear(ratios, 3.0), v->size(),
                       count_clear(ratios, 5.0), v->size(), WATCH_LAYER, WATCH_HEAD, watch);
        }
        std::printf("\n");
    }

    // ═══════════════════════ PART B — column-swap control ═══════════════════════
    std::printf("################ PART B: same-row-band left/right swap control ################\n");
    const int col_left = 4, col_right = 12;
    const int cx_left = col_left * (int)align + (int)align / 2;
    const int cx_right = col_right * (int)align + (int)align / 2;
    const std::string q_square = "What colour is the square? Answer with one word.";

    struct SwapCase { int row; bool circle_on_left; std::string path; };
    std::vector<SwapCase> swaps = {
        {8,  true,  ".session-results/prefill_probe/swap_r8_A1.bmp"},   // circle left, square right
        {8,  false, ".session-results/prefill_probe/swap_r8_A2.bmp"},   // circle right, square left
        {12, true,  ".session-results/prefill_probe/swap_r12_A1.bmp"},
        {12, false, ".session-results/prefill_probe/swap_r12_A2.bmp"},
    };
    for (auto& sw : swaps) {
        int cy = sw.row * (int)align + (int)align / 2;
        int circle_cx = sw.circle_on_left ? cx_left : cx_right;
        int square_cx = sw.circle_on_left ? cx_right : cx_left;
        write_bmp_two_objects(sw.path, canvas, canvas, cy, circle_cx, square_cx, radius);
    }

    // L23H10 trace across all 8 conditions, for the final judgement.
    struct WatchRow { int row; std::string layout; std::string question; double target_ratio, distractor_ratio; bool track_correct; };
    std::vector<WatchRow> watch_trace;

    for (auto& sw : swaps) {
        int cy = sw.row * (int)align + (int)align / 2;
        int circle_cx = sw.circle_on_left ? cx_left : cx_right;
        int square_cx = sw.circle_on_left ? cx_right : cx_left;
        std::string layout = sw.circle_on_left ? "circle-LEFT" : "circle-RIGHT";

        for (auto& qa : std::vector<std::pair<std::string,std::string>>{
                {"circle", q_circle}, {"square", q_square}}) {
            const std::string& named = qa.first;
            const std::string& question = qa.second;
            std::printf("---- row=%d layout=%s question=%s ----\n", sw.row, layout.c_str(), named.c_str());
            RunOut R = run_one(sw.path, question);
            std::string lower = R.answer; for (auto& ch : lower) ch = (char)tolower((unsigned char)ch);
            std::string expect = (named == "circle") ? "red" : "blue";
            bool pass = lower.find(expect) != std::string::npos;
            std::printf("answer=\"%s\" expect=%s perception=%s\n", R.answer.c_str(), expect.c_str(), pass ? "PASS" : "FAIL");
            if (!pass) { std::printf("SKIPPED (perception gate fail)\n\n"); continue; }

            std::set<int> obj_rows = overlapping_cells(cy, radius, (int)align, (int)R.grid_h);
            std::set<int> named_cols = overlapping_cells(named == "circle" ? circle_cx : square_cx,
                                                          radius, (int)align, (int)R.grid_w);
            std::set<int> other_cols = overlapping_cells(named == "circle" ? square_cx : circle_cx,
                                                          radius, (int)align, (int)R.grid_w);
            const int k = (int)(obj_rows.size() * named_cols.size());

            auto lr = masses_over_qrange(R.taps_q, R.prefix_len, (int)R.n_img,
                                         (int64_t)R.taps_q[0].n_q - 1, (int64_t)R.taps_q[0].n_q - 1,
                                         obj_rows, named_cols, &obj_rows, &other_cols, (int)R.grid_w);
            auto ar = masses_over_qrange(R.taps_q, R.prefix_len, (int)R.n_img, 0, R.taps_q[0].n_q - 1,
                                         obj_rows, named_cols, &obj_rows, &other_cols, (int)R.grid_w);

            for (auto& pair : {std::make_pair("LAST-ROW", &lr), std::make_pair("ALL-ROWS", &ar)}) {
                const char* label = pair.first; auto* v = pair.second;
                auto target_ratios = to_ratios(*v, k, (int)R.n_img, true);
                auto dis_ratios    = to_ratios(*v, k, (int)R.n_img, false);
                RStats st = stats_of(target_ratios), sd = stats_of(dis_ratios);
                double wt = find_ratio(target_ratios, WATCH_LAYER, WATCH_HEAD);
                double wd = find_ratio(dis_ratios, WATCH_LAYER, WATCH_HEAD);
                std::printf("%-9s target: mean=%.2fx max=%.2fx@L%dH%d | distractor: mean=%.2fx max=%.2fx@L%dH%d | "
                           "L%dH%d(watch): target=%.2fx distractor=%.2fx %s\n",
                           label, st.mean, st.max, st.max_layer, st.max_head,
                           sd.mean, sd.max, sd.max_layer, sd.max_head,
                           WATCH_LAYER, WATCH_HEAD, wt, wd, (wt > wd) ? "[TRACKS NAMED]" : "[FAILS/INVERTS]");
                if (std::string(label) == "ALL-ROWS")
                    watch_trace.push_back({sw.row, layout, named, wt, wd, wt > wd});
            }
            std::printf("\n");
        }
    }

    std::printf("################ WATCH HEAD L%dH%d — ALL-ROWS SWAP-CONTROL SUMMARY ################\n",
               WATCH_LAYER, WATCH_HEAD);
    std::printf("%-5s %-13s %-8s %-14s %-14s %s\n", "row", "layout", "asked", "target(x)", "distractor(x)", "tracks?");
    int n_track = 0, n_total = 0;
    for (auto& w : watch_trace) {
        std::printf("%-5d %-13s %-8s %-14.2f %-14.2f %s\n", w.row, w.layout.c_str(), w.question.c_str(),
                   w.target_ratio, w.distractor_ratio, w.track_correct ? "yes" : "NO");
        ++n_total; if (w.track_correct) ++n_track;
    }
    std::printf("TRACKS NAMED OBJECT: %d/%d conditions\n", n_track, n_total);

    return 0;
}
