// Hero-wavelength spectral sampling (Wilkie et al. 2014): each path carries four
// stratified wavelengths. Secondary wavelengths share the hero's directions and are
// terminated only where that would be wrong (refraction between dispersive media).
#pragma once

#include <array>

#include "owe/core/spectrum.hpp"

namespace owe {

constexpr int NW = 4;

struct Spec4 {
    std::array<double, NW> v{};
    Spec4() = default;
    explicit Spec4(double x) { v.fill(x); }
    double& operator[](int i) { return v[size_t(i)]; }
    double operator[](int i) const { return v[size_t(i)]; }
    Spec4& operator+=(const Spec4& o) { for (int i = 0; i < NW; ++i) v[i] += o.v[i]; return *this; }
    Spec4& operator*=(const Spec4& o) { for (int i = 0; i < NW; ++i) v[i] *= o.v[i]; return *this; }
    Spec4& operator*=(double s) { for (auto& x : v) x *= s; return *this; }
    double maxValue() const { double m = v[0]; for (double x : v) m = std::max(m, x); return m; }
    double sum() const { double s = 0; for (double x : v) s += x; return s; }
    bool isZero() const { for (double x : v) if (x != 0) return false; return true; }
    bool finite() const { for (double x : v) if (!std::isfinite(x)) return false; return true; }
};
inline Spec4 operator*(Spec4 a, const Spec4& b) { return a *= b; }
inline Spec4 operator*(Spec4 a, double s) { return a *= s; }
inline Spec4 operator+(Spec4 a, const Spec4& b) { return a += b; }

struct Wavelengths {
    std::array<double, NW> lambda{};
    std::array<double, NW> pdf{};  // 0 marks a terminated secondary wavelength

    static Wavelengths sample(double u) {
        Wavelengths w;
        for (int i = 0; i < NW; ++i) {
            double up = u + double(i) / NW;
            if (up >= 1) up -= 1;
            w.lambda[size_t(i)] = sampleVisibleWavelength(up);
            w.pdf[size_t(i)] = visibleWavelengthPdf(w.lambda[size_t(i)]);
        }
        return w;
    }
    static Wavelengths single(double lambdaNm) {
        Wavelengths w;
        w.lambda.fill(lambdaNm);
        w.pdf.fill(0);
        w.pdf[0] = 1;
        return w;
    }
    double hero() const { return lambda[0]; }
    bool secondaryTerminated() const {
        for (int i = 1; i < NW; ++i)
            if (pdf[size_t(i)] != 0) return false;
        return true;
    }
    // Keep only the hero; its estimate then stands for all four strata.
    void terminateSecondary() {
        if (secondaryTerminated()) return;
        for (int i = 1; i < NW; ++i) pdf[size_t(i)] = 0;
        pdf[0] /= NW;
    }
    Spec4 eval(const Spectrum& s) const {
        Spec4 r;
        for (int i = 0; i < NW; ++i) r[i] = s.eval(lambda[size_t(i)]);
        return r;
    }
    // XYZ contribution of a spectral sample: (1/NW) Σ L(λᵢ) cmf(λᵢ)/pdf(λᵢ) / ∫ȳ.
    XYZ toXYZ(const Spec4& L) const {
        XYZ c;
        double inv = 1.0 / (NW * cieYIntegral());
        for (int i = 0; i < NW; ++i) {
            if (pdf[size_t(i)] == 0 || L[i] == 0) continue;
            double w = L[i] / pdf[size_t(i)] * inv;
            double l = lambda[size_t(i)];
            c.x += w * cieX(l);
            c.y += w * cieY(l);
            c.z += w * cieZ(l);
        }
        return c;
    }
};

}  // namespace owe
