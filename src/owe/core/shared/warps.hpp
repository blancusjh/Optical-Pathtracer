// Random numbers, sample warps, phase functions and multiple-importance weights, shared by the
// reference and the GPU kernels (see prelude.hpp). Both draw their uniforms in the same order, so a
// GPU path follows the CPU path with the same seed until float rounding makes a decision differ.
#pragma once

#include "owe/core/shared/prelude.hpp"

OWE_SHARED_BEGIN

// ---------------------------------------------------------------- hashing and PCG32 (O'Neill)

OWE_FN uint64_t mixBits(uint64_t v) {
    v ^= v >> 31;
    v *= 0x7fb5d329728ea185ULL;
    v ^= v >> 27;
    v *= 0x81dadef4bc2dd44dULL;
    v ^= v >> 33;
    return v;
}
OWE_FN uint64_t hashCombine(uint64_t a, uint64_t b) { return mixBits(a ^ (mixBits(b) + 0x9e3779b97f4a7c15ULL)); }

// One step of a PCG32 stream: the next 32-bit output.
OWE_FN uint pcg32Next(OWE_INOUT(uint64_t) state, uint64_t inc) {
    uint64_t old = state;
    state = old * 6364136223846793005ULL + inc;
    uint xorshifted = uint(((old >> 18u) ^ old) >> 27u);
    uint rot = uint(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
}

// ---------------------------------------------------------------- frames and warps

// Orthonormal basis around a unit vector (Duff et al. 2017).
OWE_FN void orthonormalBasis(Real3 n, OWE_OUT(Real3) b1, OWE_OUT(Real3) b2) {
    Real sgn = copysign(Real(1), n.z);
    Real a = -1 / (sgn + n.z);
    Real b = n.x * n.y * a;
    b1 = Real3(1 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    b2 = Real3(b, sgn + n.y * n.y * a, -n.y);
}

OWE_FN Real2 sampleUniformDiskConcentric(Real u1, Real u2) {
    Real a = 2 * u1 - 1, b = 2 * u2 - 1;
    if (a == 0 && b == 0) return Real2(0, 0);
    Real r, phi;
    if (abs(a) > abs(b)) {
        r = a;
        phi = (Pi / 4) * (b / a);
    } else {
        r = b;
        phi = Pi / 2 - (Pi / 4) * (a / b);
    }
    return Real2(r * cos(phi), r * sin(phi));
}

// Two independent standard normal deviates (Box–Muller).
OWE_FN Real2 sampleStandardNormal2(Real u1, Real u2) {
    Real r = sqrt(-2 * log(1 - u1)), phi = 2 * Pi * u2;
    return Real2(r * cos(phi), r * sin(phi));
}

OWE_FN Real3 sampleCosineHemisphere(Real u1, Real u2) {
    Real2 d = sampleUniformDiskConcentric(u1, u2);
    return Real3(d.x, d.y, safeSqrt(1 - d.x * d.x - d.y * d.y));
}

OWE_FN Real3 sampleUniformSphere(Real u1, Real u2) {
    Real z = 1 - 2 * u1;
    Real r = safeSqrt(1 - z * z);
    Real phi = 2 * Pi * u2;
    return Real3(r * cos(phi), r * sin(phi), z);
}

// Uniform direction inside a cone of half-angle acos(cosMax) around +z.
OWE_FN Real3 sampleUniformCone(Real u1, Real u2, Real cosMax) {
    Real cosT = (1 - u1) + u1 * cosMax;
    Real sinT = safeSqrt(1 - cosT * cosT);
    Real phi = 2 * Pi * u2;
    return Real3(cos(phi) * sinT, sin(phi) * sinT, cosT);
}

// Uniform points of plane apertures (z = 0): an annulus, a rectangle, an ellipse.
OWE_FN Real3 sampleAnnulus(Real rMin, Real rMax, Real u1, Real u2) {
    Real r = sqrt(rMin * rMin + u1 * (rMax * rMax - rMin * rMin));
    Real phi = 2 * Pi * u2;
    return Real3(r * cos(phi), r * sin(phi), 0);
}
OWE_FN Real3 sampleRect(Real halfX, Real halfY, Real u1, Real u2) {
    return Real3((2 * u1 - 1) * halfX, (2 * u2 - 1) * halfY, 0);
}
OWE_FN Real3 sampleEllipse(Real semiX, Real semiY, Real u1, Real u2) {
    Real r = sqrt(u1), phi = 2 * Pi * u2;
    return Real3(semiX * r * cos(phi), semiY * r * sin(phi), 0);
}
// A uniform point of triangle (v0, v1, v2).
OWE_FN Real3 sampleTriangle(Real3 v0, Real3 v1, Real3 v2, Real u1, Real u2) {
    Real su = sqrt(u1);
    Real b0 = 1 - su, b1 = u2 * su;
    return v0 * b0 + v1 * b1 + v2 * (1 - b0 - b1);
}

// The solid-angle density at ref of a point p (normal n) drawn uniformly from an area; false when
// p is seen edge-on or coincides with ref.
OWE_FN bool areaToSolidAngle(Real3 ref, Real3 p, Real3 n, Real area, OWE_OUT(Real) pdfW) {
    pdfW = 0;
    Real3 w = p - ref;
    Real d2 = dot(w, w);
    if (!(d2 > 0)) return false;
    Real cosL = abs(dot(n, w)) / sqrt(d2);
    if (cosL <= kGrazing) return false;
    pdfW = d2 / (cosL * area);
    return true;
}

// The cone a sphere of radius R about the origin subtends at ref outside it: 1 − cos of its
// half-angle, computed without cancellation for small cones (0 when ref is inside the sphere).
OWE_FN Real sphereConeSize(Real3 ref, Real R) {
    Real dc2 = dot(ref, ref);
    if (dc2 <= R * R * (1 + kRelTol)) return 0;
    Real sin2Max = R * R / dc2;
    return sin2Max / (1 + safeSqrt(1 - sin2Max));
}
// A direction from ref uniformly inside that cone (oneMinusCos from sphereConeSize).
OWE_FN Real3 sampleSphereCone(Real3 ref, Real oneMinusCos, Real u1, Real u2) {
    Real3 axis = -ref / sqrt(dot(ref, ref));
    Real3 s, t;
    orthonormalBasis(axis, s, t);
    Real oneMinusCosT = u1 * oneMinusCos;
    Real cosT = 1 - oneMinusCosT;
    Real sinT = safeSqrt(oneMinusCosT * (2 - oneMinusCosT));
    Real phi = 2 * Pi * u2;
    return s * (sinT * cos(phi)) + t * (sinT * sin(phi)) + axis * cosT;
}

// ---------------------------------------------------------------- MIS and phase functions

OWE_FN Real powerHeuristic(Real fPdf, Real gPdf) {
    Real f = fPdf * fPdf, g = gPdf * gPdf;
    if (isinf(f)) return 1;
    if (f + g == 0) return 0;
    return f / (f + g);
}

// Henyey–Greenstein phase function; wo and wi both point away from the scattering point.
OWE_FN Real hgPhase(Real cosTheta, Real g) {
    Real denom = 1 + g * g + 2 * g * cosTheta;
    return (1 - g * g) / (4 * Pi * denom * sqrt(max(denom, kTiny)));
}
// Samples wi given wo (both pointing away); returns the phase value, which is its pdf.
OWE_FN Real sampleHg(Real3 wo, Real g, Real u1, Real u2, OWE_OUT(Real3) wi) {
    Real cosTheta;
    if (abs(g) < 1e-3) cosTheta = 1 - 2 * u1;
    else cosTheta = -1 / (2 * g) * (1 + g * g - sqr((1 - g * g) / (1 + g - 2 * g * u1)));
    Real sinTheta = safeSqrt(1 - cosTheta * cosTheta);
    Real phi = 2 * Pi * u2;
    Real3 s, t;
    orthonormalBasis(wo, s, t);  // cosTheta is measured from wo; g > 0 favours wi near -wo
    wi = s * (sinTheta * cos(phi)) + t * (sinTheta * sin(phi)) + wo * cosTheta;
    return hgPhase(dot(wo, wi), g);
}

OWE_SHARED_END
