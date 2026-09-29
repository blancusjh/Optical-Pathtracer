#include "owe/scene/image.hpp"

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

std::shared_ptr<const ImageRGB> loadImage(const std::string& path) {
    auto img = std::make_shared<ImageRGB>();
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
        float lut[256];
        for (int i = 0; i < 256; ++i) lut[i] = float(srgbToLinear(i / 255.0));
        const size_t count = size_t(img->width) * img->height;
        img->rgb.resize(count * 3);
        img->srgb8.resize(count);
        for (size_t i = 0; i < count; ++i) {
            for (int c = 0; c < 3; ++c) img->rgb[3 * i + c] = lut[data[3 * i + c]];
            img->srgb8[i] = uint32_t(data[3 * i]) | uint32_t(data[3 * i + 1]) << 8 | uint32_t(data[3 * i + 2]) << 16;
        }
        stbi_image_free(data);
    }
    if (img->width <= 0 || img->height <= 0) throw std::runtime_error("empty image " + path);
    return img;
}

}  // namespace owe
