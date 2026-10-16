// What an image embedding costs (docs/plan-embed-image.md phase 6): resident
// memory with both GGUFs loaded, and seconds per image, each step timed on its
// own — preprocess (decode + resize + normalise), encode (the gemma4v tower),
// the text pass (splice + 24 layers + pool) — median of REPS after one warm-up.
// Metal. Images are the synthetic fixtures plus any extra PNGs given.
//
//   ./build-metal/bin/embed-image-cost MODEL MMPROJ [extra.png ...]
//   (REPS=5, BUDGET=280 — the max soft tokens; 1120 is the card's ceiling)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <mach/mach.h>

#include "engine/embed.h"
#include "engine/model.h"
#include "image/image_loader.h"
#include "loader/tokenizer.h"
#include "models/model_registry.h"
#include "vision/vision_loader.h"
#include "vision/vision_model.h"
#include "vision/vision_profile.h"

namespace {

// Physical footprint (what Activity Monitor calls Memory): includes Metal
// allocations on unified memory, which plain RSS does not.
double footprint_mb() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
        return -1.0;
    return static_cast<double>(info.phys_footprint) / (1024.0 * 1024.0);
}

double median_ms(int reps, const std::function<void()>& f) {
    f();   // warm-up
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const auto a = std::chrono::steady_clock::now();
        f();
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ [extra.png ...]\n", argv[0]);
        return 2;
    }
    const int reps   = std::getenv("REPS") ? std::atoi(std::getenv("REPS")) : 5;
    const int budget = std::getenv("BUDGET") ? std::atoi(std::getenv("BUDGET")) : 280;

    const double m0 = footprint_mb();
    register_builtin_models();
    Model model;
    model.load_metadata(argv[1]);
    model.load_tensors();
    auto fp = create_forward_pass(model, &model.get_metadata(), 4096, 1);
    const double m1 = footprint_mb();

    ggml_backend_t backend = model.get_backend_metal() ? model.get_backend_metal() : model.get_backend_cpu();
    qinf::vision::VisionModel vmodel;
    qinf::vision::VisionLoader vloader;
    vloader.parse_metadata(argv[2], vmodel);
    vloader.load_tensors(vmodel, backend);
    auto prof = qinf::vision::make_vision_profile(vmodel, backend, model.get_metadata().id_to_token, "embed-image-cost");
    prof.preprocess = qinf::vision::gemma4v_preprocess(48, 70, budget);
    const double vision_mb = ggml_backend_buffer_get_size(vmodel.backend_buffer()) / (1024.0 * 1024.0);
    const double m2 = footprint_mb();

    const ModelMetadata& meta = model.get_metadata();
    const TokenizerConfig cfg = lookup_tokenizer_config(meta.architecture);
    const std::string arch = meta.architecture;

    std::printf("footprint: start %.0f MB, + text model %.0f MB, + image tower %.0f MB (its weights buffer %.0f MB)\n",
                m0, m1 - m0, m2 - m1, vision_mb);
    std::printf("budget %d soft tokens, %d reps (median), Metal\n\n", budget, reps);

    // A text for scale: the first fixture prompt (18 tokens).
    {
        const auto toks = qinf::engine::embed::frame(
            cfg, meta, model.get_tokenizer()->encode("task: search result | query: Which planet is known as the Red Planet?"));
        const double t = median_ms(reps, [&] { qinf::engine::embed::run(*fp, model.get_scheduler(), toks, arch); });
        std::printf("%-28s %5zu tokens                                  text pass %7.1f ms\n",
                    "text (18 tokens)", toks.size(), t);
    }

    std::vector<std::string> paths;
    for (const char* n : {"red_circle_224", "blue_squares_640x480", "green_triangle_1200x300",
                          "stripes_300x1200", "page_850x1100", "yellow_tiny_40", "checker_512"})
        paths.push_back(std::string("tests/fixtures/embed_image/") + n + ".png");
    for (int i = 3; i < argc; ++i) paths.push_back(argv[i]);

    std::printf("%-28s %6s %9s %10s %9s %11s %9s\n", "image", "tokens", "canvas", "preprocess", "encode", "text pass", "total");
    double peak = m2;
    for (const std::string& p : paths) {
        qinf::vision::Bitmap bmp;
        const double t_pre = median_ms(reps, [&] { bmp = qinf::image::load_image_to_bitmap(p, prof.preprocess); });
        const uint32_t n = prof.encoder->mm_tokens_for(bmp);
        std::vector<float> soft;
        const double t_enc = median_ms(reps, [&] { soft = prof.encoder->encode(bmp); });
        size_t framed = 0;
        const double t_txt = median_ms(reps, [&] {
            auto in = qinf::engine::embed::image_input(cfg, meta, prof.boi_id, prof.soft_id, prof.eoi_id, soft, n);
            framed = in.tokens.size();
            qinf::engine::embed::run(*fp, model.get_scheduler(), std::move(in), arch);
        });
        peak = std::max(peak, footprint_mb());
        const std::string name = p.substr(p.find_last_of('/') + 1);
        char canvas[32];
        std::snprintf(canvas, sizeof(canvas), "%dx%d", bmp.width, bmp.height);
        std::printf("%-28s %6zu %9s %7.1f ms %6.1f ms %8.1f ms %6.1f ms\n", name.c_str(), framed, canvas,
                    t_pre, t_enc, t_txt, t_pre + t_enc + t_txt);
    }
    std::printf("\npeak footprint during the runs: %.0f MB\n", peak);
    return 0;
}
