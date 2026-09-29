#include "owe/render/output.hpp"

#include "owe/render/exposure.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
Mat3 inverse(const double a[3][3]);
}

Image readPFM(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string magic;
    int w = 0, h = 0;
    double scale = 0;
    in >> magic >> w >> h >> scale;
    in.get();  // the single whitespace before the data
    if (magic != "PF" || w <= 0 || h <= 0 || scale == 0) throw std::runtime_error(path + ": not a colour PFM");
    std::vector<float> data(size_t(w) * h * 3);
    in.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
    if (!in) throw std::runtime_error(path + ": truncated");
    if (scale > 0)  // big-endian
        for (float& f : data) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            u = (u >> 24) | ((u >> 8) & 0xFF00u) | ((u << 8) & 0xFF0000u) | (u << 24);
            std::memcpy(&f, &u, 4);
        }
    // The exact inverse of the XYZ → linear sRGB matrix writePFM used.
    const double m[3][3] = {{3.2404542, -1.5371385, -0.4985314}, {-0.9692660, 1.8760108, 0.0415560}, {0.0556434, -0.2040259, 1.0572252}};
    static const Mat3 toXYZ = inverse(m);
    Image img;
    img.width = w;
    img.height = h;
    img.xyz.resize(data.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float* c = &data[(size_t(h - 1 - y) * w + x) * 3];  // PFM stores bottom-to-top
            for (int k = 0; k < 3; ++k)
                img.xyz[(size_t(y) * w + x) * 3 + k] = toXYZ.m[k][0] * c[0] + toXYZ.m[k][1] * c[1] + toXYZ.m[k][2] * c[2];
        }
    return img;
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

double autoExposureEV(const Image& img) { return meterEV(img); }

namespace {
// Oklab (B. Ottosson, 2020) from and to linear sRGB.
void linearSRGBToOklab(const double c[3], double lab[3]) {
    double l = std::cbrt(0.4122214708 * c[0] + 0.5363325363 * c[1] + 0.0514459929 * c[2]);
    double m = std::cbrt(0.2119034982 * c[0] + 0.6806995451 * c[1] + 0.1073969566 * c[2]);
    double s = std::cbrt(0.0883024619 * c[0] + 0.2817188376 * c[1] + 0.6299787005 * c[2]);
    lab[0] = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
    lab[1] = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    lab[2] = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
}
void oklabToLinearSRGB(const double lab[3], double c[3]) {
    double l = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2];
    double m = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2];
    double s = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2];
    l = l * l * l;
    m = m * m * m;
    s = s * s * s;
    c[0] = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    c[1] = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    c[2] = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
}
}  // namespace

void gamutMapLinearSRGB(double rgb[3]) {
    if (rgb[0] >= 0 && rgb[1] >= 0 && rgb[2] >= 0) return;
    double lab[3];
    linearSRGBToOklab(rgb, lab);
    if (!(lab[0] > 0) || !std::isfinite(lab[0] + lab[1] + lab[2])) {
        rgb[0] = rgb[1] = rgb[2] = 0;
        return;
    }
    // The largest chroma fraction (same lightness and hue) whose sRGB components are all ≥ 0.
    double lo = 0, hi = 1;
    for (int i = 0; i < 24; ++i) {
        double k = 0.5 * (lo + hi), t[3] = {lab[0], lab[1] * k, lab[2] * k}, c[3];
        oklabToLinearSRGB(t, c);
        (c[0] >= 0 && c[1] >= 0 && c[2] >= 0 ? lo : hi) = k;
    }
    double t[3] = {lab[0], lab[1] * lo, lab[2] * lo};
    oklabToLinearSRGB(t, rgb);
    for (int k = 0; k < 3; ++k) rgb[k] = std::max(0.0, rgb[k]);
}


Tone toneFromName(const std::string& name) {
    if (name == "agx") return Tone::AgX;
    if (name == "standard") return Tone::Standard;
    throw std::runtime_error("unknown tone '" + name + "' (agx or standard)");
}

namespace {

// AgX's inset of the sRGB primaries toward white, and the polynomial fit of its default contrast
// sigmoid (Troy Sobotka's AgX, in the minimal form of Benjamin Wrensch, 2023). The published rows
// sum to 1 within 1.4e-4; they are normalised to exactly 1, so a grey stays exactly grey.
constexpr double kAgxInset[3][3] = {{0.842479062253094, 0.0784335999999992, 0.0792237451477643},
                                    {0.0423282422610123, 0.878468636469772, 0.0791661274605434},
                                    {0.0423756549057051, 0.0784336, 0.879142973793104}};
constexpr double kAgxMinEv = -12.47393, kAgxMaxEv = 4.026069;

Mat3 inverse(const double a[3][3]) {
    Mat3 r;
    const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                       a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const int i1 = (j + 1) % 3, i2 = (j + 2) % 3, j1 = (i + 1) % 3, j2 = (i + 2) % 3;
            r.m[i][j] = (a[i1][j1] * a[i2][j2] - a[i1][j2] * a[i2][j1]) / det;
        }
    return r;
}

double agxSigmoid(double x) {
    const double x2 = x * x, x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

Mat3 agxInset() {
    Mat3 r;
    for (int i = 0; i < 3; ++i) {
        const double sum = kAgxInset[i][0] + kAgxInset[i][1] + kAgxInset[i][2];
        for (int j = 0; j < 3; ++j) r.m[i][j] = kAgxInset[i][j] / sum;
    }
    return r;
}

}  // namespace

void applyTone(Tone tone, double rgb[3]) {
    if (tone == Tone::AgX) {
        static const Mat3 inset = agxInset(), outset = inverse(inset.m);
        double v[3];
        for (int i = 0; i < 3; ++i) {
            double x = inset.m[i][0] * rgb[0] + inset.m[i][1] * rgb[1] + inset.m[i][2] * rgb[2];
            x = std::log2(std::max(x, 1e-12));
            v[i] = agxSigmoid(std::clamp((x - kAgxMinEv) / (kAgxMaxEv - kAgxMinEv), 0.0, 1.0));
        }
        for (int i = 0; i < 3; ++i) {
            double y = outset.m[i][0] * v[0] + outset.m[i][1] * v[1] + outset.m[i][2] * v[2];
            rgb[i] = std::clamp(std::pow(std::max(y, 0.0), 2.2), 0.0, 1.0);
        }
        return;
    }
    // Standard: a gentle roll-off below white; a colour beyond the display's range fades toward
    // white at its own luminance, as a bright star or a caustic's core looks white, instead of
    // clipping per channel (which turns a blue-white star cyan). Colours within range are untouched.
    double top = 0;
    for (int k = 0; k < 3; ++k) {
        rgb[k] = std::max(0.0, rgb[k]);
        rgb[k] = rgb[k] / (1 + rgb[k] / 4.0) * 1.25;
        top = std::max(top, rgb[k]);
    }
    if (top > 1) {
        double Y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        double t = Y >= 1 ? 1.0 : (top - 1) / (top - Y);
        for (int k = 0; k < 3; ++k) rgb[k] = Y >= 1 ? 1.0 : rgb[k] + t * (Y - rgb[k]);
    }
}

std::vector<unsigned char> displayRGB8(const Image& img, double exposureEV, bool autoExposure, double whiteKelvin, Tone tone) {
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
            // Every spectral colour lies outside sRGB. Clipping its negative component to zero
            // collapsed whole ranges of wavelengths into one hue (a rainbow became bands); the
            // chroma is reduced instead, at constant perceptual lightness and hue.
            gamutMapLinearSRGB(rgb);
            applyTone(tone, rgb);
            for (int k = 0; k < 3; ++k) out.push_back((unsigned char)std::lround(255 * srgbEncode(rgb[k])));
        }
    }
    return out;
}

void writePNG(const std::string& path, const Image& img, double exposureEV, bool autoExposure, double whiteKelvin, Tone tone) {
    writeRGB8PNG(path, img.width, img.height, displayRGB8(img, exposureEV, autoExposure, whiteKelvin, tone));
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
