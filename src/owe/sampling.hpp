// Deterministic random numbers and warping functions.
#pragma once

#include "math.hpp"

namespace owe {

// PCG32 (O'Neill). Streams are fully determined by (seed, stream id), which
// makes every render reproducible independent of thread scheduling.
class Rng {
public:
    Rng() { seed(0, 0); }
    Rng(uint64_t seedValue, uint64_t stream) { seed(seedValue, stream); }
    void seed(uint64_t seedValue, uint64_t stream) {
        state_ = 0;
        inc_ = (stream << 1u) | 1u;
        nextU32();
        state_ += seedValue;
        nextU32();
    }
    uint32_t nextU32() {
        uint64_t old = state_;
        state_ = old * 6364136223846793005ULL + inc_;
        uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = uint32_t(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    // Uniform in [0,1) with 53 bits.
    double uniform() {
        uint64_t a = nextU32() >> 5, b = nextU32() >> 6;
        return (double(a) * 67108864.0 + double(b)) * (1.0 / 9007199254740992.0);
    }

private:
    uint64_t state_, inc_;
};

inline uint64_t mixBits(uint64_t v) {
    v ^= v >> 31;
    v *= 0x7fb5d329728ea185ULL;
    v ^= v >> 27;
    v *= 0x81dadef4bc2dd44dULL;
    v ^= v >> 33;
    return v;
}
inline uint64_t hashCombine(uint64_t a, uint64_t b) { return mixBits(a ^ (mixBits(b) + 0x9e3779b97f4a7c15ULL)); }

struct Vec2 {
    double x = 0, y = 0;
};

inline Vec2 sampleUniformDiskConcentric(double u1, double u2) {
    double a = 2 * u1 - 1, b = 2 * u2 - 1;
    if (a == 0 && b == 0) return {0, 0};
    double r, phi;
    if (std::abs(a) > std::abs(b)) { r = a; phi = (Pi / 4) * (b / a); }
    else { r = b; phi = Pi / 2 - (Pi / 4) * (a / b); }
    return {r * std::cos(phi), r * std::sin(phi)};
}

inline Vec3 sampleCosineHemisphere(double u1, double u2) {
    Vec2 d = sampleUniformDiskConcentric(u1, u2);
    double z = safeSqrt(1 - d.x * d.x - d.y * d.y);
    return {d.x, d.y, z};
}

inline Vec3 sampleUniformSphere(double u1, double u2) {
    double z = 1 - 2 * u1;
    double r = safeSqrt(1 - z * z);
    double phi = 2 * Pi * u2;
    return {r * std::cos(phi), r * std::sin(phi), z};
}

// Uniform direction inside a cone of half-angle acos(cosMax) around +z.
inline Vec3 sampleUniformCone(double u1, double u2, double cosMax) {
    double cosT = (1 - u1) + u1 * cosMax;
    double sinT = safeSqrt(1 - cosT * cosT);
    double phi = 2 * Pi * u2;
    return {std::cos(phi) * sinT, std::sin(phi) * sinT, cosT};
}

inline double powerHeuristic(double fPdf, double gPdf) {
    double f = fPdf * fPdf, g = gPdf * gPdf;
    if (std::isinf(f)) return 1;
    if (f + g == 0) return 0;
    return f / (f + g);
}

// Henyey–Greenstein phase function; wo and wi both point away from the scattering point.
inline double hgPhase(double cosTheta, double g) {
    double denom = 1 + g * g + 2 * g * cosTheta;
    return (1 - g * g) / (4 * Pi * denom * std::sqrt(std::max(denom, 1e-300)));
}
// Samples wi given wo (both pointing away). Returns the phase value (== pdf).
inline double sampleHg(const Vec3& wo, double g, double u1, double u2, Vec3& wi) {
    double cosTheta;
    if (std::abs(g) < 1e-3) cosTheta = 1 - 2 * u1;
    else cosTheta = -1 / (2 * g) * (1 + g * g - sqr((1 - g * g) / (1 + g - 2 * g * u1)));
    double sinTheta = safeSqrt(1 - cosTheta * cosTheta);
    double phi = 2 * Pi * u2;
    Frame f(wo);  // cosTheta is measured from wo; g > 0 favours wi near -wo
    wi = f.toWorld({sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta});
    return hgPhase(dot(wo, wi), g);
}

}  // namespace owe
