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
    std::vector<float> rgb;   // HDR images: linear, row-major from the top row
    // 8-bit images: the encoded texels (R | G << 8 | B << 16), row-major from the top row; decoded
    // on lookup (4 bytes a texel rather than 16: scans carry several 8192² photographs).
    std::vector<uint32_t> srgb8;

    Vec3 texel(int x, int y) const {
        const size_t i = size_t(y) * width + x;
        if (hdr) return {rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]};
        const float* lut = srgbLut();
        const uint32_t v = srgb8[i];
        return {lut[v & 0xFFu], lut[(v >> 8) & 0xFFu], lut[(v >> 16) & 0xFFu]};
    }
    static const float* srgbLut();  // linear values of the 256 sRGB codes
    // Bilinear lookup, repeating: u across, v down (v = 0 at the top row).
    Vec3 bilinear(double u, double v) const;
};

// Loads PNG, JPEG, BMP, TGA or HDR; throws with the reason on failure.
std::shared_ptr<const ImageRGB> loadImage(const std::string& path);
ImageRGB decodeImage(const std::string& path);  // the same, as a value (no shared copy)
// Halves an image in each direction (a 2 × 2 box filter, which keeps its mean; an odd last row or
// column is averaged with its neighbour).
void halveImage(ImageRGB& img);
// The bytes an image takes (4 per 8-bit texel, 12 per HDR texel).
size_t imageBytes(const ImageRGB& img);
// An image file's size from its header, without decoding it; false if it cannot be read.
bool imageSize(const std::string& path, int& width, int& height);
// Whether an image file holds HDR (Radiance) data, from its header.
bool imageIsHdr(const std::string& path);
double srgbToLinear(double c);

}  // namespace owe
