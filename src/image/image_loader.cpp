#include "image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO_DEPRECATED
#include "../../third_party/stb/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace qinf::image {

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

// Pillow-compatible separable bicubic resize of an interleaved 8-bit RGB source
// to new_w × new_h (docs/plan-embed-image.md, gemma4v). Byte-for-byte port of
// llama.cpp mtmd `img_tool::resize_pillow` with RESIZE_ALGO_BICUBIC at
// ceefd16da (itself from Pillow's Resample.c): Keys cubic with a = −0.5,
// pixel-centre mapping, the filter widened by the scale when downsampling,
// weights normalised then rounded to 22-bit fixed point (+/−0.5 then truncate),
// a horizontal pass then a vertical pass, each through a uint8 intermediate with
// a 0.5 rounding bias and a clamp. An axis whose size does not change is not
// resampled at all.
void resize_pillow_bicubic_u8(const unsigned char* src, int w, int h,
                              std::vector<unsigned char>& dst, int new_w, int new_h) {
    constexpr int kPrecisionBits = 32 - 8 - 2;
    constexpr double kSupport = 2.0;   // bicubic
    auto filter = [](double x) -> double {
        constexpr double a = -0.5;
        if (x < 0.0) x = -x;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    auto clip8 = [](int v) -> unsigned char {
        return static_cast<unsigned char>(v < 0 ? 0 : (v > 255 ? 255 : v));
    };
    // Coefficients for one axis: bounds[2·o] = first input index, bounds[2·o+1]
    // = count; weights[o·ksize + k] in fixed point. Returns ksize.
    auto weights_for = [&](int in_size, int out_size, std::vector<int>& bounds,
                           std::vector<int32_t>& weights) -> int {
        const double scale = static_cast<double>(in_size) / out_size;
        const double filterscale = scale < 1.0 ? 1.0 : scale;
        const double support = kSupport * filterscale;
        const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
        std::vector<double> pre(static_cast<size_t>(out_size) * ksize);
        bounds.assign(static_cast<size_t>(out_size) * 2, 0);
        for (int xx = 0; xx < out_size; ++xx) {
            const double center = (xx + 0.5) * scale;
            const double ss = 1.0 / filterscale;
            int xmin = static_cast<int>(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = static_cast<int>(center + support + 0.5);
            if (xmax > in_size) xmax = in_size;
            xmax -= xmin;
            double ww = 0.0;
            int x = 0;
            for (; x < xmax; ++x) {
                const double wv = filter((x + xmin - center + 0.5) * ss);
                pre[static_cast<size_t>(xx) * ksize + x] = wv;
                ww += wv;
            }
            for (x = 0; x < xmax; ++x)
                if (ww != 0.0) pre[static_cast<size_t>(xx) * ksize + x] /= ww;
            for (; x < ksize; ++x) pre[static_cast<size_t>(xx) * ksize + x] = 0;
            bounds[xx * 2 + 0] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        weights.resize(static_cast<size_t>(out_size) * ksize);
        const double fxp = std::ldexp(1.0, kPrecisionBits);
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = static_cast<int32_t>(pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5));
        return ksize;
    };
    auto horizontal = [&](const unsigned char* in, int in_nx, int in_ny, int out_nx) {
        std::vector<int> b;
        std::vector<int32_t> k;
        const int ksize = weights_for(in_nx, out_nx, b, k);
        std::vector<unsigned char> out(static_cast<size_t>(out_nx) * in_ny * 3);
        for (int yy = 0; yy < in_ny; ++yy) {
            const unsigned char* row = in + static_cast<size_t>(yy) * in_nx * 3;
            unsigned char* orow = out.data() + static_cast<size_t>(yy) * out_nx * 3;
            for (int xx = 0; xx < out_nx; ++xx) {
                const int xmin = b[xx * 2], xcnt = b[xx * 2 + 1];
                const int32_t* kk = &k[static_cast<size_t>(xx) * ksize];
                const unsigned char* p = row + static_cast<size_t>(xmin) * 3;
                int32_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xcnt; ++x, p += 3) {
                    s0 += p[0] * kk[x];
                    s1 += p[1] * kk[x];
                    s2 += p[2] * kk[x];
                }
                orow[xx * 3 + 0] = clip8(s0 >> kPrecisionBits);
                orow[xx * 3 + 1] = clip8(s1 >> kPrecisionBits);
                orow[xx * 3 + 2] = clip8(s2 >> kPrecisionBits);
            }
        }
        return out;
    };
    auto vertical = [&](const unsigned char* in, int in_nx, int in_ny, int out_ny) {
        std::vector<int> b;
        std::vector<int32_t> k;
        const int ksize = weights_for(in_ny, out_ny, b, k);
        const size_t row_elems = static_cast<size_t>(in_nx) * 3;
        std::vector<unsigned char> out(row_elems * out_ny);
        std::vector<int32_t> acc(row_elems);
        for (int yy = 0; yy < out_ny; ++yy) {
            const int ymin = b[yy * 2], ycnt = b[yy * 2 + 1];
            const int32_t* kk = &k[static_cast<size_t>(yy) * ksize];
            std::fill(acc.begin(), acc.end(), 1 << (kPrecisionBits - 1));
            for (int y = 0; y < ycnt; ++y) {
                const unsigned char* row = in + static_cast<size_t>(ymin + y) * row_elems;
                const int32_t wv = kk[y];
                for (size_t i = 0; i < row_elems; ++i) acc[i] += row[i] * wv;
            }
            unsigned char* orow = out.data() + static_cast<size_t>(yy) * row_elems;
            for (size_t i = 0; i < row_elems; ++i) orow[i] = clip8(acc[i] >> kPrecisionBits);
        }
        return out;
    };
    const bool need_h = new_w != w, need_v = new_h != h;
    if (need_h && need_v) {
        const std::vector<unsigned char> tmp = horizontal(src, w, h, new_w);
        dst = vertical(tmp.data(), new_w, h, new_h);
    } else if (need_h) {
        dst = horizontal(src, w, h, new_w);
    } else if (need_v) {
        dst = vertical(src, w, h, new_h);
    } else {
        dst.assign(src, src + static_cast<size_t>(3) * w * h);
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

namespace {

// Shared core: turn an already-decoded interleaved 8-bit RGB buffer into the
// preprocessed channel-planar Bitmap.The file- and memory-based entry points
// differ only in how `data` is produced, so this is byte-identical between them.
qinf::vision::Bitmap bitmap_from_rgb(const unsigned char* data, int w, int h,
                                     const qinf::vision::ImagePreprocess& pp) {
    // ── Target canvas dims (aspect ratio handled by the PAD_CEIL letterbox) ───
    int target_w, target_h;
    if (pp.sizing == qinf::vision::ImagePreprocess::Sizing::FixedSquarePadCeil) {
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
    if (pp.resample == qinf::vision::ImagePreprocess::Resample::PillowBicubic)
        resize_pillow_bicubic_u8(data, w, h, resized, new_w, new_h);
    else
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
    bmp.content_x = off_x;
    bmp.content_y = off_y;
    bmp.content_w = new_w;
    bmp.content_h = new_h;
    return bmp;
}

}  // namespace

qinf::vision::Bitmap load_image_to_bitmap(const std::string& path,
                                          const qinf::vision::ImagePreprocess& pp) {
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
                                                      const qinf::vision::ImagePreprocess& pp) {
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
    return load_image_to_bitmap(path, qinf::vision::gemma3_preprocess(target));
}

}  // namespace qinf::image
