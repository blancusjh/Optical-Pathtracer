// Interface optics kernels: Snell, Fresnel, microfacets. Header-only and free of
// scene state so the same code can be ported verbatim to GPU shading languages.
#pragma once

#include <complex>

#include "math.hpp"
#include "sampling.hpp"

namespace owe {

// Snell refraction of propagation direction d at a surface with normal n
// (either orientation), going from index ni into index nt.
// Returns false on total internal reflection.
inline bool refractDirection(const Vec3& d, const Vec3& n, double ni, double nt, Vec3& out) {
    Vec3 nn = dot(d, n) < 0 ? n : -n;
    double cosi = -dot(d, nn);
    double eta = ni / nt;
    double k = 1 - eta * eta * (1 - cosi * cosi);
    if (k < 0) return false;
    out = normalize(d * eta + nn * (eta * cosi - std::sqrt(k)));
    return true;
}

// Fresnel amplitude-squared reflectances for s and p polarisation.
// cosI is the cosine of the incidence angle (≥ 0). Returns false on TIR (Rs = Rp = 1).
inline bool fresnelDielectricSP(double cosI, double ni, double nt, double& Rs, double& Rp) {
    cosI = clampd(cosI, 0, 1);
    double sinT2 = sqr(ni / nt) * (1 - cosI * cosI);
    if (sinT2 >= 1) { Rs = Rp = 1; return false; }
    double cosT = std::sqrt(1 - sinT2);
    double rs = (ni * cosI - nt * cosT) / (ni * cosI + nt * cosT);
    double rp = (nt * cosI - ni * cosT) / (nt * cosI + ni * cosT);
    Rs = rs * rs;
    Rp = rp * rp;
    return true;
}

// Unpolarised dielectric reflectance, pbrt convention: eta = n_inside/n_outside
// relative to the +z hemisphere; cosThetaI may be negative (incidence from inside).
inline double fresnelDielectric(double cosThetaI, double eta) {
    cosThetaI = clampd(cosThetaI, -1, 1);
    if (cosThetaI < 0) { eta = 1 / eta; cosThetaI = -cosThetaI; }
    double sin2T = (1 - cosThetaI * cosThetaI) / (eta * eta);
    if (sin2T >= 1) return 1;
    double cosT = safeSqrt(1 - sin2T);
    double rParl = (eta * cosThetaI - cosT) / (eta * cosThetaI + cosT);
    double rPerp = (cosThetaI - eta * cosT) / (cosThetaI + eta * cosT);
    return (rParl * rParl + rPerp * rPerp) / 2;
}

// Unpolarised reflectance of a conductor with complex relative index eta = (n + ik)/n_medium.
inline double fresnelConductor(double cosThetaI, std::complex<double> eta) {
    using C = std::complex<double>;
    cosThetaI = clampd(cosThetaI, 0, 1);
    double sin2I = 1 - cosThetaI * cosThetaI;
    C sin2T = C(sin2I) / (eta * eta);
    C cosT = std::sqrt(C(1) - sin2T);
    C rParl = (eta * cosThetaI - cosT) / (eta * cosThetaI + cosT);
    C rPerp = (C(cosThetaI) - eta * cosT) / (C(cosThetaI) + eta * cosT);
    return (std::norm(rParl) + std::norm(rPerp)) / 2;
}

// Local shading-frame helpers (normal = +z).
inline double cosTheta(const Vec3& w) { return w.z; }
inline double absCosTheta(const Vec3& w) { return std::abs(w.z); }
inline bool sameHemisphere(const Vec3& a, const Vec3& b) { return a.z * b.z > 0; }

// pbrt-v4 style refraction of wi (pointing away from the surface) about n.
// eta = n_t/n_i relative to n's hemisphere; etap receives the effective ratio.
inline bool refractLocal(Vec3 wi, Vec3 n, double eta, double& etap, Vec3& wt) {
    double cosI = dot(n, wi);
    if (cosI < 0) { eta = 1 / eta; cosI = -cosI; n = -n; }
    double sin2I = std::max(0.0, 1 - cosI * cosI);
    double sin2T = sin2I / (eta * eta);
    if (sin2T >= 1) return false;
    double cosT = safeSqrt(1 - sin2T);
    wt = -wi / eta + n * (cosI / eta - cosT);
    etap = eta;
    return true;
}

// Isotropic Trowbridge–Reitz (GGX) microfacet distribution with visible-normal sampling.
class GGX {
public:
    explicit GGX(double alpha) : a_(std::max(alpha, 1e-4)) {}
    double alpha() const { return a_; }
    bool effectivelySmooth() const { return a_ < 1e-3; }

    double D(const Vec3& wm) const {
        double c2 = wm.z * wm.z;
        if (c2 <= 0) return 0;
        double tan2 = (1 - c2) / c2;
        double c4 = c2 * c2;
        if (c4 < 1e-16) return 0;
        double e = tan2 / (a_ * a_);
        return 1 / (Pi * a_ * a_ * c4 * sqr(1 + e));
    }
    double lambda(const Vec3& w) const {
        double c2 = w.z * w.z;
        if (c2 <= 0) return 0;
        double tan2 = (1 - c2) / c2;
        return (std::sqrt(1 + a_ * a_ * tan2) - 1) / 2;
    }
    double G1(const Vec3& w) const { return 1 / (1 + lambda(w)); }
    double G(const Vec3& wo, const Vec3& wi) const { return 1 / (1 + lambda(wo) + lambda(wi)); }
    // Distribution of visible normals.
    double D(const Vec3& w, const Vec3& wm) const {
        if (absCosTheta(w) == 0) return 0;
        return G1(w) / absCosTheta(w) * D(wm) * std::abs(dot(w, wm));
    }
    double pdf(const Vec3& w, const Vec3& wm) const { return D(w, wm); }
    Vec3 sampleWm(const Vec3& w, double u1, double u2) const {
        Vec3 wh = normalize(Vec3(a_ * w.x, a_ * w.y, w.z));
        if (wh.z < 0) wh = -wh;
        Vec3 T1 = (wh.z < 0.99999) ? normalize(cross(Vec3(0, 0, 1), wh)) : Vec3(1, 0, 0);
        Vec3 T2 = cross(wh, T1);
        // Uniform disk (polar mapping).
        double r = std::sqrt(u1), phi = 2 * Pi * u2;
        double px = r * std::cos(phi), py = r * std::sin(phi);
        double h = std::sqrt(std::max(0.0, 1 - px * px));
        double s = (1 + wh.z) / 2;
        py = (1 - s) * h + s * py;
        double pz = safeSqrt(1 - px * px - py * py);
        Vec3 nh = T1 * px + T2 * py + wh * pz;
        return normalize(Vec3(a_ * nh.x, a_ * nh.y, std::max(1e-6, nh.z)));
    }

private:
    double a_;
};

}  // namespace owe
