// Images for textures and environment maps, held in linear RGB. PNG and JPEG files are sRGB-encoded
// and decoded to linear on load; Radiance HDR files are linear already. Reflectance textures use
// them through the RGB reflectance basis (spectra in [0, 1] for colours in [0, 1]); environment maps
// through the RGB illuminant basis.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "owe/core/math.hpp"

namespace owe {

struct ImageRGB {
    std::string path;
    int width = 0, height = 0;
    bool hdr = false;         // linear radiance (Radiance HDR); else an sRGB-encoded 8-bit image
    std::vector<float> rgb;   // linear, row-major from the top row
    std::vector<uint32_t> srgb8;  // 8-bit images: the original encoded texels (R | G << 8 | B << 16)

    Vec3 texel(int x, int y) const {
        const size_t i = 3 * (size_t(y) * width + x);
        return {rgb[i], rgb[i + 1], rgb[i + 2]};
    }
    // Bilinear lookup, repeating: u across, v down (v = 0 at the top row).
    Vec3 bilinear(double u, double v) const;
};

// Loads PNG, JPEG, BMP, TGA or HDR; throws with the reason on failure.
std::shared_ptr<const ImageRGB> loadImage(const std::string& path);
// An image file's size from its header, without decoding it; false if it cannot be read.
bool imageSize(const std::string& path, int& width, int& height);
double srgbToLinear(double c);

}  // namespace owe
