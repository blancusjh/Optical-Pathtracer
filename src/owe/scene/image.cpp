#include "owe/scene/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "stb_image.h"

namespace owe {

double srgbToLinear(double c) { return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); }

const float* ImageRGB::srgbLut() {
    static const std::array<float, 256> lut = [] {
        std::array<float, 256> t{};
        for (int i = 0; i < 256; ++i) t[size_t(i)] = float(srgbToLinear(i / 255.0));
        return t;
    }();
    return lut.data();
}

Vec3 ImageRGB::bilinear(double u, double v) const {
    const double x = (u - std::floor(u)) * width - 0.5, y = (v - std::floor(v)) * height - 0.5;
    const double fx = std::floor(x), fy = std::floor(y);
    const double tx = x - fx, ty = y - fy;
    auto wrap = [](int i, int n) { return ((i % n) + n) % n; };
    const int x0 = wrap(int(fx), width), x1 = wrap(int(fx) + 1, width);
    const int y0 = wrap(int(fy), height), y1 = wrap(int(fy) + 1, height);
    return texel(x0, y0) * ((1 - tx) * (1 - ty)) + texel(x1, y0) * (tx * (1 - ty)) + texel(x0, y1) * ((1 - tx) * ty) +
           texel(x1, y1) * (tx * ty);
}

size_t imageBytes(const ImageRGB& img) { return img.hdr ? img.rgb.size() * sizeof(float) : img.srgb8.size() * 4; }

void halveImage(ImageRGB& img) {
    if (img.width < 2 && img.height < 2) return;
    const int W = std::max(1, img.width / 2), H = std::max(1, img.height / 2);
    auto src = [&](int x, int y) { return size_t(std::min(y, img.height - 1)) * img.width + size_t(std::min(x, img.width - 1)); };
    if (img.hdr) {
        std::vector<float> out(size_t(W) * H * 3);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < 3; ++c)
                    out[3 * (size_t(y) * W + x) + c] =
                        0.25f * (img.rgb[3 * src(2 * x, 2 * y) + c] + img.rgb[3 * src(2 * x + 1, 2 * y) + c] +
                                 img.rgb[3 * src(2 * x, 2 * y + 1) + c] + img.rgb[3 * src(2 * x + 1, 2 * y + 1) + c]);
        img.rgb.swap(out);
    } else {
        // The encoded bytes are averaged as stored: colour maps lose a little contrast, data maps
        // (normals, roughness) stay as they are meant to be read.
        std::vector<uint32_t> out(size_t(W) * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const uint32_t q[4] = {img.srgb8[src(2 * x, 2 * y)], img.srgb8[src(2 * x + 1, 2 * y)],
                                       img.srgb8[src(2 * x, 2 * y + 1)], img.srgb8[src(2 * x + 1, 2 * y + 1)]};
                uint32_t v = 0;
                for (int c = 0; c < 3; ++c) {
                    uint32_t s = 0;
                    for (uint32_t t : q) s += (t >> (8 * c)) & 0xFFu;
                    v |= ((s + 2) / 4) << (8 * c);
                }
                out[size_t(y) * W + x] = v;
            }
        img.srgb8.swap(out);
    }
    img.width = W;
    img.height = H;
}

bool imageIsHdr(const std::string& path) { return stbi_is_hdr(path.c_str()) != 0; }

bool imageSize(const std::string& path, int& width, int& height) {
    int n = 0;
    return stbi_info(path.c_str(), &width, &height, &n) != 0;
}

std::shared_ptr<const ImageRGB> loadImage(const std::string& path) {
    return std::make_shared<const ImageRGB>(decodeImage(path));
}

ImageRGB decodeImage(const std::string& path) {
    ImageRGB out;
    ImageRGB* img = &out;
    img->path = path;
    int n = 0;
    if (stbi_is_hdr(path.c_str())) {
        float* data = stbi_loadf(path.c_str(), &img->width, &img->height, &n, 3);
        if (!data) throw std::runtime_error("cannot read image " + path + ": " + stbi_failure_reason());
        img->hdr = true;
        img->rgb.assign(data, data + size_t(img->width) * img->height * 3);
        stbi_image_free(data);
    } else {
        unsigned char* data = stbi_load(path.c_str(), &img->width, &img->height, &n, 3);
        if (!data) throw std::runtime_error("cannot read image " + path + ": " + stbi_failure_reason());
        const size_t count = size_t(img->width) * img->height;
        img->srgb8.resize(count);
        for (size_t i = 0; i < count; ++i)
            img->srgb8[i] = uint32_t(data[3 * i]) | uint32_t(data[3 * i + 1]) << 8 | uint32_t(data[3 * i + 2]) << 16;
        stbi_image_free(data);
    }
    if (img->width <= 0 || img->height <= 0) throw std::runtime_error("empty image " + path);
    return out;
}

}  // namespace owe
