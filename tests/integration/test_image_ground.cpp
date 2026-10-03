// test_image_ground.cpp — end-to-end gate for GENERATION after an image on a
// Qwen-VL recipe: ask the model for a box, compare it with where the shape was
// drawn (docs/note-verdict-img-ground.md).
//
// Why a box: describing an image survives a lot of brokenness, because the
// DeltaNet layers carry a summary of the whole page. Coordinates do not. Two
// faults passed the coherence smokes and were found by asking for a box
// (2026-10-02): batched decode masked KV rows by the rope position (a generated
// token saw only rows 0..position — the prompt head and the image's top rows),
// and the recipe ran block M-RoPE where the family is trained interleaved.
// Before the fixes every box sat in the top tenth of the page.
//
// The page is drawn here — white, a red ring top-right, a blue square
// bottom-left, non-square (768x1024) so rows and columns differ — and goes
// through the production path: PPM bytes → load_image_to_bitmap_from_memory
// (the projector's preprocessing) → chat template (thinking off) →
// expand_image_markers → prefill_multimodal → decode_step (greedy, no
// penalty). The box is read as Qwen-VL's 0..1000 relative bbox_2d.
//
// Gate: IoU >= 0.5 against the drawn shape, for both shapes.
//
// Usage: test-image-ground <model.gguf> <mmproj.gguf>
// Qwen-VL only (the qwen3vl-merger projector): grounding is a trained Qwen-VL
// skill, not a property of the engine. Other projectors exit 77 (skipped). The
// shared code these faults lived in is gated model-free for every recipe
// (rope-divergence-tests, attn-mask-input-tests, kv-write-setrows-tests,
// layer-tests MRopeLayout).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include "ggml-backend.h"

#include "engine/model.h"
#include "engine/decode_step.h"
#include "engine/multimodal_prefill.h"
#include "image/image_loader.h"
#include "image/image_prompt.h"
#include "loader/tokenizer.h"
#include "models/forward_pass_base.h"
#include "models/model_registry.h"
#include "sampling/sampling.h"
#include "vision/bitmap.h"
#include "vision/i_vision_encoder.h"
#include "vision/vision_loader.h"
#include "vision/vision_model.h"
#include "vision/vision_profile.h"

namespace {

constexpr int kW = 768, kH = 1024;
constexpr uint32_t kCtx = 2048;
constexpr int kMaxNew = 64;

struct Box { double x0, y0, x1, y1; };

double iou(const Box& a, const Box& b) {
    const double ix = std::max(0.0, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const double iy = std::max(0.0, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    const double inter = ix * iy;
    const double u = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return u > 0 ? inter / u : 0.0;
}

// The page as binary PPM (P6) — a format stb decodes, so the test runs the same
// decode + resize + normalize as an uploaded image.
std::vector<uint8_t> draw_page(Box& ring, Box& square) {
    std::vector<uint8_t> rgb(static_cast<size_t>(kW) * kH * 3, 255);
    auto put = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        uint8_t* p = &rgb[(static_cast<size_t>(y) * kW + x) * 3];
        p[0] = r; p[1] = g; p[2] = b;
    };
    const double cx = 560, cy = 250, r_out = 100, r_in = 84;   // red ring, top-right
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            const double d = std::hypot(x + 0.5 - cx, y + 0.5 - cy);
            if (d <= r_out && d >= r_in) put(x, y, 210, 20, 30);
        }
    ring = {cx - r_out, cy - r_out, cx + r_out, cy + r_out};
    square = {110, 690, 300, 880};                                // blue square, bottom-left
    for (int y = static_cast<int>(square.y0); y < static_cast<int>(square.y1); ++y)
        for (int x = static_cast<int>(square.x0); x < static_cast<int>(square.x1); ++x)
            put(x, y, 30, 60, 200);
    const std::string head = "P6\n" + std::to_string(kW) + " " + std::to_string(kH) + "\n255\n";
    std::vector<uint8_t> ppm(head.begin(), head.end());
    ppm.insert(ppm.end(), rgb.begin(), rgb.end());
    return ppm;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <mmproj.gguf>\n";
        return 64;
    }
    ggml_backend_load_all();
    register_builtin_models();

    Model model;
    model.load_metadata(argv[1], /*allow_multimodal=*/true);
    model.load_tensors();
    const ModelMetadata& meta = model.get_metadata();
    Tokenizer* tok = model.get_tokenizer();
    ggml_backend_sched_t sched = model.get_scheduler();
    auto fp = create_forward_pass(model, &meta, kCtx, 1);

    ggml_backend_t backend = model.has_metal_backend() ? model.get_backend_metal()
                                                       : model.get_backend_cpu();
    qinf::vision::VisionModel vmodel;
    qinf::vision::VisionLoader vloader;
    vloader.parse_metadata(argv[2], vmodel);
    vloader.load_tensors(vmodel, backend);
    qinf::vision::VisionProfile vp = qinf::vision::make_vision_profile(
        vmodel, backend, tok->get_vocabulary(), "test-image-ground: argv[2]");
    if (vp.projector_tag != "qwen3vl-merger") {
        std::cout << "SKIPPED: projector '" << vp.projector_tag
                  << "' — grounding is a Qwen-VL skill (qwen3vl-merger only)\n";
        return 77;
    }
    const ChatTemplate* tmpl = lookup_chat_template(meta.architecture);
    if (!tmpl) {
        std::cerr << "test-image-ground: a chat template expected for '" << meta.architecture
                  << "', actual none registered\n";
        return 1;
    }

    Box ring_truth{}, square_truth{};
    const std::vector<uint8_t> ppm = draw_page(ring_truth, square_truth);
    const qinf::vision::Bitmap bmp =
        qinf::image::load_image_to_bitmap_from_memory(ppm.data(), ppm.size(), vp.preprocess);
    const uint32_t n_img = vp.encoder->mm_tokens_for(bmp);
    uint32_t gw = 0, gh = 0;
    vp.encoder->mm_grid_for(bmp, gw, gh);
    std::cout << "=== image-ground: " << meta.architecture << " page " << kW << "x" << kH
              << " -> " << gw << "x" << gh << " = " << n_img << " image tokens ===\n";

    const std::vector<std::string>& vocab = tok->get_vocabulary();
    const uint32_t vocab_size = static_cast<uint32_t>(meta.vocab_size);
    const int32_t eos = tok->get_eos_token_id();
    const std::vector<int32_t> im_end = tok->encode("<|im_end|>");

    struct Case { const char* name; const char* what; Box truth; };
    const Case cases[] = {
        {"ring", "the red circle", ring_truth},
        {"square", "the blue square", square_truth},
    };
    const std::regex bbox(R"(\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\])");

    int failed = 0;
    for (const Case& c : cases) {
        std::vector<ChatMessage> turn = {{"user", vp.marker_prefix + "Locate " + c.what +
                                          " in the image, output its bbox coordinates using JSON format."}};
        const std::string prompt = tmpl->render(turn, /*add_assistant_prompt=*/true, /*enable_thinking=*/false);
        qinf::image::ExpandedImagePrompt built = qinf::image::expand_image_markers(
            tok->encode(prompt), vp.boi_id, vp.soft_id, vp.eoi_id, n_img);
        std::vector<int32_t> tokens = std::move(built.tokens);
        int32_t span_start = built.span_start;
        if (meta.bos_token_id >= 0) { tokens.insert(tokens.begin(), meta.bos_token_id); span_start += 1; }

        fp->clear_slot(0);
        fp->set_cache_pos(0, 0);
        fp->reset_rope_pos(0);
        const std::vector<ImagePromptChunk> chunks = {{&bmp, span_start}};
        std::vector<float> logits = prefill_multimodal(*fp, *vp.encoder, sched, tokens, chunks, 0, 0);
        std::vector<float> tail(logits.end() - vocab_size, logits.end());

        qinf::GreedySampler sampler(/*repetition_penalty=*/1.0f);
        std::vector<int32_t> history = tokens;
        int32_t next = static_cast<int32_t>(sampler.sample(tail, history, vocab));
        std::string text;
        for (int i = 0; i < kMaxNew; ++i) {
            if (next == eos || (im_end.size() == 1 && next == im_end[0])) break;
            text += tok->decode(next);
            history.push_back(next);
            if (text.find(']') != std::string::npos && text.find("bbox") != std::string::npos) break;
            next = decode_step(fp.get(), sched, &sampler, next, /*slot=*/0, history, vocab, vocab_size);
        }

        std::smatch m;
        if (!std::regex_search(text, m, bbox)) {
            std::cout << "  FAIL " << c.name << ": no bbox in the answer: " << text << "\n";
            ++failed;
            continue;
        }
        // Qwen-VL boxes are 0..1000 relative to the image.
        const Box got{std::stod(m[1]) * kW / 1000.0, std::stod(m[2]) * kH / 1000.0,
                      std::stod(m[3]) * kW / 1000.0, std::stod(m[4]) * kH / 1000.0};
        const double v = iou(got, c.truth);
        char line[256];
        std::snprintf(line, sizeof(line),
                      "  %s %s: box [%.0f %.0f %.0f %.0f] truth [%.0f %.0f %.0f %.0f] IoU %.2f (gate >= 0.5)\n",
                      v >= 0.5 ? "PASS" : "FAIL", c.name, got.x0, got.y0, got.x1, got.y1,
                      c.truth.x0, c.truth.y0, c.truth.x1, c.truth.y1, v);
        std::cout << line;
        if (v < 0.5) ++failed;
    }
    std::cout << (failed == 0 ? "PASSED" : "FAILED") << " (" << failed << " of 2 boxes failed)\n";
    return failed == 0 ? 0 : 1;
}
