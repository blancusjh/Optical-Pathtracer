#include "owe/render/output.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#ifdef OWE_HAVE_ZLIB
#include <zlib.h>
#endif

namespace owe {

void writePFM(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "PF\n" << img.width << " " << img.height << "\n-1.0\n";
    std::vector<float> row(size_t(img.width) * 3);
    for (int y = img.height - 1; y >= 0; --y) {  // PFM stores bottom-to-top
        for (int x = 0; x < img.width; ++x) {
            double rgb[3];
            xyzToLinearSRGB(img.at(x, y), rgb);
            for (int c = 0; c < 3; ++c) row[3 * x + c] = float(rgb[c]);
        }
        out.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size() * sizeof(float)));
    }
}

namespace {
uint32_t crc32(const unsigned char* data, size_t n, uint32_t crc = 0) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}
void be32(std::vector<unsigned char>& v, uint32_t x) {
    v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}
void chunk(std::ofstream& out, const char* type, const std::vector<unsigned char>& data) {
    std::vector<unsigned char> buf;
    be32(buf, uint32_t(data.size()));
    buf.insert(buf.end(), type, type + 4);
    buf.insert(buf.end(), data.begin(), data.end());
    uint32_t c = crc32(buf.data() + 4, buf.size() - 4);
    be32(buf, c);
    out.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(buf.size()));
}
double srgbEncode(double v) {
    v = clampd(v, 0, 1);
    return v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1 / 2.4) - 0.055;
}
}  // namespace

namespace {
// Bradford chromatic adaptation matrix from white (Xw, Yw, Zw) to D65.
void bradfordToD65(const XYZ& w, double M[3][3]) {
    const double B[3][3] = {{0.8951, 0.2664, -0.1614}, {-0.7502, 1.7135, 0.0367}, {0.0389, -0.0685, 1.0296}};
    const double Bi[3][3] = {{0.9869929, -0.1470543, 0.1599627}, {0.4323053, 0.5183603, 0.0492912},
                             {-0.0085287, 0.0400428, 0.9684867}};
    const double d65[3] = {0.95047, 1.0, 1.08883};
    double src[3], dst[3];
    for (int i = 0; i < 3; ++i) {
        src[i] = B[i][0] * w.x + B[i][1] * w.y + B[i][2] * w.z;
        dst[i] = B[i][0] * d65[0] + B[i][1] * d65[1] + B[i][2] * d65[2];
    }
    double T[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) T[i][j] = dst[i] / src[i] * B[i][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M[i][j] = Bi[i][0] * T[0][j] + Bi[i][1] * T[1][j] + Bi[i][2] * T[2][j];
}
}  // namespace

double autoExposureEV(const Image& img) {
    // Meter the current image, protecting bright detail in a mostly dark room.
    // The old log mean alone could blow out a projection surrounded by black walls.
    double s = 0;
    size_t n = 0;
    std::vector<double> luminance;
    luminance.reserve(size_t(img.width) * img.height);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            double Y = img.at(x, y).y;
            if (Y > 0 && std::isfinite(Y)) {
                s += std::log(Y + 1e-12); n++; luminance.push_back(Y);
            }
        }
    double avg = n ? std::exp(s / n) : 1;
    double metered = 0.18 / std::max(avg, 1e-12);
    if (!luminance.empty()) {
        size_t high = size_t(0.99 * (luminance.size() - 1));
        std::nth_element(luminance.begin(), luminance.begin() + high, luminance.end());
        metered = std::min(metered, 0.6 / luminance[high]);
    }
    // A planet may occupy only a few pixels in a wide free-eye view. A percentile
    // meter discards it and raises a weak surrounding reflection to mid-grey.
    // Give the central 10% (the observer's fixation) additional highlight protection.
    // A small bright lamp elsewhere in a room should not black out the whole frame.
    // Require a neighbouring pixel so a lone noisy sample cannot set this limit.
    double compactHighlight = 0;
    int x0 = std::max(0, img.width / 2 - std::max(1, img.width / 20));
    int x1 = std::min(img.width - 1, img.width / 2 + std::max(1, img.width / 20));
    int y0 = std::max(0, img.height / 2 - std::max(1, img.height / 20));
    int y1 = std::min(img.height - 1, img.height / 2 + std::max(1, img.height / 20));
    for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
        double a = 0, b = 0;
        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            double value = img.at(x + dx, y + dy).y;
            if (!std::isfinite(value)) continue;
            if (value > a) { b = a; a = value; }
            else if (value > b) b = value;
        }
        compactHighlight = std::max(compactHighlight, b);
    }
    if (compactHighlight > 0) metered = std::min(metered, 0.6 / compactHighlight);
    return std::log2(metered);
}

std::vector<unsigned char> displayRGB8(const Image& img, double exposureEV, bool autoExposure, double whiteKelvin) {
    double scale = std::exp2(exposureEV + (autoExposure ? autoExposureEV(img) : 0));
    double wb[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    if (whiteKelvin > 0) bradfordToD65(Spectrum::blackbody(whiteKelvin, 1.0).toXYZ(), wb);
    std::vector<unsigned char> out;
    out.reserve(size_t(img.height) * img.width * 3);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            XYZ c0 = img.at(x, y);
            XYZ c{(wb[0][0] * c0.x + wb[0][1] * c0.y + wb[0][2] * c0.z) * scale,
                  (wb[1][0] * c0.x + wb[1][1] * c0.y + wb[1][2] * c0.z) * scale,
                  (wb[2][0] * c0.x + wb[2][1] * c0.y + wb[2][2] * c0.z) * scale};
            double rgb[3];
            xyzToLinearSRGB(c, rgb);
            // Gentle highlight roll-off (display only).
            for (int k = 0; k < 3; ++k) {
                double v = std::max(0.0, rgb[k]);
                v = v / (1 + v / 4.0) * 1.25;
                out.push_back((unsigned char)std::lround(255 * srgbEncode(v)));
            }
        }
    }
    return out;
}

void writePNG(const std::string& path, const Image& img, double exposureEV, bool autoExposure, double whiteKelvin) {
    writeRGB8PNG(path, img.width, img.height, displayRGB8(img, exposureEV, autoExposure, whiteKelvin));
}

void writeRGB8PNG(const std::string& path, int width, int height, const std::vector<unsigned char>& rgb) {
    std::vector<unsigned char> raw;
    raw.reserve(size_t(height) * (width * 3 + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);  // PNG filter type: none
        raw.insert(raw.end(), rgb.begin() + size_t(y) * width * 3, rgb.begin() + size_t(y + 1) * width * 3);
    }
    std::vector<unsigned char> z;
#ifdef OWE_HAVE_ZLIB
    uLongf zlen = compressBound(uLong(raw.size()));
    z.resize(zlen);
    if (compress2(z.data(), &zlen, raw.data(), uLong(raw.size()), 9) != Z_OK) throw std::runtime_error("zlib failure");
    z.resize(zlen);
#else
    // zlib stream with stored (uncompressed) deflate blocks.
    z = {0x78, 0x01};
    size_t pos = 0;
    uint32_t a = 1, b = 0;
    for (unsigned char ch : raw) { a = (a + ch) % 65521; b = (b + a) % 65521; }
    do {
        size_t len = std::min<size_t>(65535, raw.size() - pos);
        bool last = pos + len == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(len & 0xFF); z.push_back(len >> 8);
        z.push_back(~len & 0xFF); z.push_back((~len >> 8) & 0xFF);
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
    } while (pos < raw.size());
    be32(z, (b << 16) | a);
#endif

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    out.write(reinterpret_cast<const char*>(sig), 8);
    std::vector<unsigned char> ihdr;
    be32(ihdr, uint32_t(width));
    be32(ihdr, uint32_t(height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    chunk(out, "IHDR", ihdr);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
}

}  // namespace owe
