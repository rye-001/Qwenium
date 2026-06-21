#include "image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO_DEPRECATED
#include "../../third_party/stb/stb_image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace qinf::cli {

namespace {

inline float lerp(float s, float e, float t) { return s + (e - s) * t; }

// Aspect-preserving bilinear resize of an interleaved 8-bit RGB source to
// new_w × new_h, written interleaved into `dst`. Byte-for-byte faithful to
// llama.cpp mtmd `img_tool::resize_bilinear`: ALIGN-CORNERS coordinate mapping
// (px = x * (src-1)/(dst-1), NOT half-pixel) and a uint8 intermediate (the
// interpolated value is truncated to a byte before normalization). Matching
// both is load-bearing — the encoder was trained on exactly this resampling.
void resize_bilinear_u8(const unsigned char* src, int w, int h,
                        std::vector<unsigned char>& dst, int new_w, int new_h) {
    dst.resize(static_cast<size_t>(3) * new_w * new_h);
    const float x_ratio = new_w > 1 ? static_cast<float>(w - 1) / (new_w - 1) : 0.0f;
    const float y_ratio = new_h > 1 ? static_cast<float>(h - 1) / (new_h - 1) : 0.0f;
    auto at = [&](int x, int y, int c) -> float {
        return static_cast<float>(src[(static_cast<size_t>(y) * w + x) * 3 + c]);
    };
    for (int y = 0; y < new_h; ++y) {
        const float py = y * y_ratio;
        const int y0 = std::min(static_cast<int>(py), h - 1);
        const int y1 = std::min(y0 + 1, h - 1);
        const float yf = py - y0;
        for (int x = 0; x < new_w; ++x) {
            const float px = x * x_ratio;
            const int x0 = std::min(static_cast<int>(px), w - 1);
            const int x1 = std::min(x0 + 1, w - 1);
            const float xf = px - x0;
            for (int c = 0; c < 3; ++c) {
                const float top = lerp(at(x0, y0, c), at(x1, y0, c), xf);
                const float bot = lerp(at(x0, y1, c), at(x1, y1, c), xf);
                dst[(static_cast<size_t>(y) * new_w + x) * 3 + c] =
                    static_cast<unsigned char>(lerp(top, bot, yf));
            }
        }
    }
}

// Qwen-VL "smart_resize" — llama.cpp img_tool::calc_size_preserved_ratio (the
// min/max-pixels variant). Returns (w_bar, h_bar), both multiples of `align`,
// with min_px ≤ w_bar·h_bar ≤ max_px, preserving aspect ratio. Token count is
// then (w_bar/align)·(h_bar/align). Faithful to mtmd-image.cpp.
void smart_resize_dims(int w, int h, int align, int min_px, int max_px,
                       int& out_w, int& out_h) {
    auto round_by = [align](float x) {
        return static_cast<int>(std::round(x / align)) * align; };
    auto ceil_by  = [align](float x) {
        return static_cast<int>(std::ceil (x / align)) * align; };
    auto floor_by = [align](float x) {
        return static_cast<int>(std::floor(x / align)) * align; };

    int h_bar = std::max(align, round_by(static_cast<float>(h)));
    int w_bar = std::max(align, round_by(static_cast<float>(w)));

    if (static_cast<long long>(h_bar) * w_bar > max_px) {
        const float beta = std::sqrt(static_cast<float>(h) * w / max_px);
        h_bar = std::max(align, floor_by(h / beta));
        w_bar = std::max(align, floor_by(w / beta));
    } else if (static_cast<long long>(h_bar) * w_bar < min_px) {
        const float beta = std::sqrt(static_cast<float>(min_px) / (static_cast<float>(h) * w));
        h_bar = ceil_by(h * beta);
        w_bar = ceil_by(w * beta);
    }
    out_w = w_bar;
    out_h = h_bar;
}

// FNV-1a over the normalized pixel bytes — a content-derived id for the
// per-session encode cache (dedups the same image across turns).
uint64_t content_hash(const std::vector<float>& pixels) {
    uint64_t hsh = 1469598103934665603ull;
    const auto* bytes = reinterpret_cast<const unsigned char*>(pixels.data());
    const size_t n = pixels.size() * sizeof(float);
    for (size_t i = 0; i < n; ++i) {
        hsh ^= bytes[i];
        hsh *= 1099511628211ull;
    }
    return hsh;
}

}  // namespace

ImagePreprocess gemma3_preprocess(int target) {
    ImagePreprocess pp;
    pp.sizing       = ImagePreprocess::Sizing::FixedSquarePadCeil;
    pp.fixed_target = target;
    pp.mean[0]   = pp.mean[1]   = pp.mean[2]   = 0.5f;
    pp.stddev[0] = pp.stddev[1] = pp.stddev[2] = 0.5f;
    return pp;
}

ImagePreprocess gemma4uv_preprocess(int align, int min_tokens, int max_tokens) {
    ImagePreprocess pp;
    pp.sizing     = ImagePreprocess::Sizing::DynSmartResize;
    pp.align      = align;
    pp.min_tokens = min_tokens;
    pp.max_tokens = max_tokens;
    pp.mean[0]   = pp.mean[1]   = pp.mean[2]   = 0.0f;   // verified mmproj kv [0,0,0]
    pp.stddev[0] = pp.stddev[1] = pp.stddev[2] = 1.0f;   // verified mmproj kv [1,1,1]
    return pp;
}

namespace {

// Shared core: turn an already-decoded interleaved 8-bit RGB buffer into the
// preprocessed channel-planar Bitmap.The file- and memory-based entry points
// differ only in how `data` is produced, so this is byte-identical between them.
qinf::vision::Bitmap bitmap_from_rgb(const unsigned char* data, int w, int h,
                                     const ImagePreprocess& pp) {
    // ── Target canvas dims (aspect ratio handled by the PAD_CEIL letterbox) ───
    int target_w, target_h;
    if (pp.sizing == ImagePreprocess::Sizing::FixedSquarePadCeil) {
        target_w = target_h = pp.fixed_target;
    } else {
        const int patch_area = pp.align * pp.align;  // cur_merge==1 ⇒ align²
        smart_resize_dims(w, h, pp.align,
                          pp.min_tokens * patch_area, pp.max_tokens * patch_area,
                          target_w, target_h);
    }

    qinf::vision::Bitmap bmp;
    bmp.channels = 3;
    bmp.width    = target_w;
    bmp.height   = target_h;
    const size_t plane = static_cast<size_t>(target_w) * target_h;

    // Pre-fill with the normalized black-pad value per channel: uint8 0 →
    // (0/255 − mean)/stddev. The active (resized) region overwrites its
    // sub-rectangle below; whatever is left is the letterbox pad in encoder space.
    bmp.pixels.resize(static_cast<size_t>(3) * plane);
    for (int c = 0; c < 3; ++c) {
        const float pad = (0.0f - pp.mean[c]) / pp.stddev[c];
        std::fill(bmp.pixels.begin() + static_cast<long>(c * plane),
                  bmp.pixels.begin() + static_cast<long>((c + 1) * plane), pad);
    }

    // ── Aspect-preserving resize (ceil, clamp) + center-composite (PAD_CEIL) ──
    const float scale = std::min(static_cast<float>(target_w) / w,
                                 static_cast<float>(target_h) / h);
    const int new_w = std::min(static_cast<int>(std::ceil(w * scale)), target_w);
    const int new_h = std::min(static_cast<int>(std::ceil(h * scale)), target_h);

    std::vector<unsigned char> resized;
    resize_bilinear_u8(data, w, h, resized, new_w, new_h);  // caller owns/frees `data`

    const int off_x = (target_w - new_w) / 2;
    const int off_y = (target_h - new_h) / 2;
    for (int ry = 0; ry < new_h; ++ry) {
        const int oy = off_y + ry;
        for (int rx = 0; rx < new_w; ++rx) {
            const int ox = off_x + rx;
            for (int c = 0; c < 3; ++c) {
                const float v =
                    static_cast<float>(resized[(static_cast<size_t>(ry) * new_w + rx) * 3 + c]);
                bmp.pixels[c * plane + static_cast<size_t>(oy) * target_w + ox] =
                    (v / 255.0f - pp.mean[c]) / pp.stddev[c];
            }
        }
    }

    bmp.content_id = content_hash(bmp.pixels);
    return bmp;
}

}  // namespace

qinf::vision::Bitmap load_image_to_bitmap(const std::string& path,
                                          const ImagePreprocess& pp) {
    int w = 0, h = 0, src_channels = 0;
    // Force 3 channels (RGB); stb drops alpha and expands grayscale.
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &src_channels, 3);
    if (data == nullptr)
        throw std::runtime_error(
            std::string("image_loader: file '") + path +
            "': expected a decodable JPEG/PNG/image, got: " +
            (stbi_failure_reason() ? stbi_failure_reason() : "unknown stb error"));
    if (w <= 0 || h <= 0) {
        stbi_image_free(data);
        throw std::runtime_error(
            std::string("image_loader: file '") + path +
            "': expected positive dimensions, got: " + std::to_string(w) +
            "x" + std::to_string(h));
    }
    qinf::vision::Bitmap bmp = bitmap_from_rgb(data, w, h, pp);
    stbi_image_free(data);
    return bmp;
}

qinf::vision::Bitmap load_image_to_bitmap_from_memory(const uint8_t* data,
                                                      size_t len,
                                                      const ImagePreprocess& pp) {
    int w = 0, h = 0, src_channels = 0;
    unsigned char* rgb = stbi_load_from_memory(
        data, static_cast<int>(len), &w, &h, &src_channels, 3);
    if (rgb == nullptr)
        throw std::runtime_error(
            std::string("image_loader: parameter 'image bytes': expected a "
            "decodable JPEG/PNG/image, got: ") +
            (stbi_failure_reason() ? stbi_failure_reason() : "unknown stb error"));
    if (w <= 0 || h <= 0) {
        stbi_image_free(rgb);
        throw std::runtime_error(
            std::string("image_loader: parameter 'image bytes': expected positive "
            "dimensions, got: ") + std::to_string(w) + "x" + std::to_string(h));
    }
    qinf::vision::Bitmap bmp = bitmap_from_rgb(rgb, w, h, pp);
    stbi_image_free(rgb);
    return bmp;
}

qinf::vision::Bitmap load_image_to_bitmap(const std::string& path, int target) {
    return load_image_to_bitmap(path, gemma3_preprocess(target));
}

}  // namespace qinf::cli
