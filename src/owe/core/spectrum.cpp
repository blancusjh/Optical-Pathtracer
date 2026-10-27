#include "owe/core/spectrum.hpp"

#include <cstdio>
#include <functional>
#include <stdexcept>

namespace owe {

namespace {
double integrateVisible(const std::function<double(double)>& f) {
    // Composite Simpson with 0.5 nm steps.
    const int n = int((LambdaMax - LambdaMin) * 2);
    const double h = (LambdaMax - LambdaMin) / n;
    double s = f(LambdaMin) + f(LambdaMax);
    for (int i = 1; i < n; ++i) s += f(LambdaMin + i * h) * ((i & 1) ? 4 : 2);
    return s * h / 3;
}
}  // namespace

double cieYIntegral() {
    static const double v = integrateVisible([](double l) { return cieY(l); });
    return v;
}

void xyzToLinearSRGB(const XYZ& c, double rgb[3]) {
    for (int i = 0; i < 3; ++i)
        rgb[i] = kXyzToLinearSRGB[i][0] * c.x + kXyzToLinearSRGB[i][1] * c.y + kXyzToLinearSRGB[i][2] * c.z;
}

Spectrum Spectrum::constant(double v) {
    Spectrum s;
    s.kind_ = Kind::Constant;
    s.a_ = v;
    return s;
}

Spectrum Spectrum::blackbody(double kelvin, double y) {
    if (!(kelvin > 0)) throw std::runtime_error("blackbody temperature must be positive");
    Spectrum s;
    s.kind_ = Kind::Blackbody;
    s.a_ = kelvin;
    s.scale_ = 1.0;
    double lum = s.luminance();
    s.scale_ = y / lum;
    return s;
}

Spectrum Spectrum::tabulated(std::vector<double> lambdas, std::vector<double> values) {
    if (lambdas.size() != values.size() || lambdas.empty())
        throw std::runtime_error("tabulated spectrum needs equal-length, non-empty tables");
    for (size_t i = 1; i < lambdas.size(); ++i)
        if (!(lambdas[i] > lambdas[i - 1])) throw std::runtime_error("tabulated spectrum wavelengths must increase");
    Spectrum s;
    s.kind_ = Kind::Tabulated;
    s.lambdas_ = std::make_shared<const std::vector<double>>(std::move(lambdas));
    s.values_ = std::make_shared<const std::vector<double>>(std::move(values));
    return s;
}

namespace {
// Maps an sRGB triple to partition-basis weights so the spectrum reproduces that colour
// (white-balanced to the 6504 K reference white). White maps to equal weights, i.e. a flat
// spectrum; reflectance weights are clamped to [0,1] so they remain physical.
struct RgbCalibration {
    double inv[3][3];
    double white[3];
    RgbCalibration() {
        double M[3][3];
        for (int k = 0; k < 3; ++k) {
            XYZ c;
            double inv0 = 1.0 / cieYIntegral();
            for (double l = LambdaMin; l <= LambdaMax; l += 0.5) {
                double w = rgbBasis(l)[k] * planck(l, 6504.0) * 0.5;
                c.x += w * cieX(l) * inv0;
                c.y += w * cieY(l) * inv0;
                c.z += w * cieZ(l) * inv0;
            }
            double rgb[3];
            xyzToLinearSRGB(c, rgb);
            for (int i = 0; i < 3; ++i) M[i][k] = rgb[i];
        }
        for (int i = 0; i < 3; ++i) white[i] = M[i][0] + M[i][1] + M[i][2];
        const Mat3 Minv = Mat3::fromRows(M).inverse();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) inv[i][j] = Minv.m[i][j];
    }
    void weights(double r, double g, double b, double w[3]) const {
        double c[3] = {r * white[0], g * white[1], b * white[2]};
        for (int i = 0; i < 3; ++i) w[i] = inv[i][0] * c[0] + inv[i][1] * c[1] + inv[i][2] * c[2];
    }
};
const RgbCalibration& rgbCalibration() {
    static const RgbCalibration c;
    return c;
}
}  // namespace

void rgbToBasisWeights(double r, double g, double b, double w[3]) { rgbCalibration().weights(r, g, b, w); }
void rgbBasisMatrix(double A[3][3]) {
    const auto& c = rgbCalibration();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) A[i][j] = c.inv[i][j] * c.white[j];
}
double rgbIlluminantNorm() { return Spectrum::rgbIlluminant(1, 1, 1).paramScale(); }

Spectrum Spectrum::rgbReflectance(double r, double g, double b) {
    Spectrum s;
    s.kind_ = Kind::RgbReflectance;
    double w[3];
    rgbCalibration().weights(r, g, b, w);
    s.a_ = clamp(w[0], 0, 1); s.b_ = clamp(w[1], 0, 1); s.c_ = clamp(w[2], 0, 1);
    return s;
}

Spectrum Spectrum::rgbIlluminant(double r, double g, double b) {
    Spectrum s;
    s.kind_ = Kind::RgbIlluminant;
    double w[3];
    rgbCalibration().weights(r, g, b, w);
    s.a_ = std::max(0.0, w[0]); s.b_ = std::max(0.0, w[1]); s.c_ = std::max(0.0, w[2]);
    static const double norm = [] {
        Spectrum w;
        w.kind_ = Kind::RgbIlluminant;
        w.a_ = w.b_ = w.c_ = 1;
        return 1.0 / w.luminance();
    }();
    s.scale_ = norm;
    return s;
}

double Spectrum::eval(double l) const {
    switch (kind_) {
        case Kind::Constant: return scale_ * a_;
        case Kind::Blackbody: return scale_ * planck(l, a_);
        case Kind::Tabulated: {
            const auto& L = *lambdas_;
            const auto& V = *values_;
            if (l <= L.front()) return scale_ * V.front();
            if (l >= L.back()) return scale_ * V.back();
            size_t i = size_t(std::upper_bound(L.begin(), L.end(), l) - L.begin());
            double t = (l - L[i - 1]) / (L[i] - L[i - 1]);
            return scale_ * (V[i - 1] * (1 - t) + V[i] * t);
        }
        case Kind::RgbReflectance: return scale_ * dot(Vec3(a_, b_, c_), rgbBasis(l));
        case Kind::RgbIlluminant: return scale_ * dot(Vec3(a_, b_, c_), rgbBasis(l)) * rgbIlluminantBase(l);
    }
    return 0;
}

double Spectrum::maxValue() const {
    double m = 0;
    for (double l = LambdaMin; l <= LambdaMax; l += 1.0) m = std::max(m, std::abs(eval(l)));
    return m;
}

XYZ Spectrum::toXYZ() const {
    double inv = 1.0 / cieYIntegral();
    XYZ r;
    r.x = integrateVisible([&](double l) { return eval(l) * cieX(l); }) * inv;
    r.y = integrateVisible([&](double l) { return eval(l) * cieY(l); }) * inv;
    r.z = integrateVisible([&](double l) { return eval(l) * cieZ(l); }) * inv;
    return r;
}

double Spectrum::luminance() const {
    return integrateVisible([&](double l) { return eval(l) * cieY(l); }) / cieYIntegral();
}

std::string Spectrum::describe() const {
    char buf[160];
    switch (kind_) {
        case Kind::Constant: std::snprintf(buf, sizeof buf, "constant(%g)", a_ * scale_); break;
        case Kind::Blackbody: std::snprintf(buf, sizeof buf, "blackbody(T=%gK, scale=%g)", a_, scale_); break;
        case Kind::Tabulated: std::snprintf(buf, sizeof buf, "tabulated(%zu samples, scale=%g)", lambdas_->size(), scale_); break;
        case Kind::RgbReflectance: std::snprintf(buf, sizeof buf, "rgb_reflectance(%g,%g,%g)", a_, b_, c_); break;
        case Kind::RgbIlluminant: std::snprintf(buf, sizeof buf, "rgb_illuminant(%g,%g,%g)x%g", a_, b_, c_, scale_); break;
    }
    return buf;
}

}  // namespace owe
