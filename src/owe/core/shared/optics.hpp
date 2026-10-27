// Interface optics shared by the reference and the GPU kernels (see prelude.hpp): Snell's law,
// Fresnel's equations for dielectrics and conductors, and the GGX microfacet lobes. Lobes work in
// a local frame whose +z is the surface normal; wo points back along the incident ray, wi along
// the scattered one. `radiance` selects radiance transport (camera paths), which carries the
// 1/η² of refraction; importance transport (light paths) does not.
#pragma once

#include "owe/core/shared/prelude.hpp"

OWE_SHARED_BEGIN

// Snell refraction of propagation direction d at a surface with normal n (either orientation),
// from index ni into index nt. False on total internal reflection.
OWE_FN bool refractDirection(Real3 d, Real3 n, Real ni, Real nt, OWE_OUT(Real3) outDir) {
    Real3 nn = dot(d, n) < 0 ? n : -n;
    Real cosi = -dot(d, nn);
    Real eta = ni / nt;
    Real k = 1 - eta * eta * (1 - cosi * cosi);
    outDir = d;
    if (k < 0) return false;
    outDir = normalize(d * eta + nn * (eta * cosi - sqrt(k)));
    return true;
}

// Fresnel reflectances for s and p polarisation; cosI ≥ 0. False on TIR (Rs = Rp = 1).
OWE_FN bool fresnelDielectricSP(Real cosI, Real ni, Real nt, OWE_OUT(Real) Rs, OWE_OUT(Real) Rp) {
    cosI = clamp(cosI, Real(0), Real(1));
    Real sinT2 = sqr(ni / nt) * (1 - cosI * cosI);
    Rs = 1;
    Rp = 1;
    if (sinT2 >= 1) return false;
    Real cosT = sqrt(1 - sinT2);
    Real rs = (ni * cosI - nt * cosT) / (ni * cosI + nt * cosT);
    Real rp = (nt * cosI - ni * cosT) / (nt * cosI + ni * cosT);
    Rs = rs * rs;
    Rp = rp * rp;
    return true;
}

// Unpolarised dielectric reflectance (pbrt convention): eta = n_inside / n_outside relative to the
// +z hemisphere; cosThetaI may be negative (incidence from inside).
OWE_FN Real fresnelDielectric(Real cosThetaI, Real eta) {
    cosThetaI = clamp(cosThetaI, Real(-1), Real(1));
    if (cosThetaI < 0) {
        eta = 1 / eta;
        cosThetaI = -cosThetaI;
    }
    Real sin2T = (1 - cosThetaI * cosThetaI) / (eta * eta);
    if (sin2T >= 1) return 1;
    Real cosT = safeSqrt(1 - sin2T);
    Real rParl = (eta * cosThetaI - cosT) / (eta * cosThetaI + cosT);
    Real rPerp = (cosThetaI - eta * cosT) / (cosThetaI + eta * cosT);
    return (rParl * rParl + rPerp * rPerp) / 2;
}

// Complex numbers as (re, im).
OWE_FN Real2 cmul(Real2 a, Real2 b) { return Real2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
OWE_FN Real2 cdiv(Real2 a, Real2 b) {
    Real d = dot(b, b);
    return Real2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}
OWE_FN Real2 csqrt(Real2 z) {  // principal branch
    Real r = sqrt(dot(z, z));
    return Real2(sqrt(max(Real(0), (r + z.x) / 2)), copysign(sqrt(max(Real(0), (r - z.x) / 2)), z.y));
}

// Unpolarised reflectance of a conductor with complex relative index eta = (n + ik) / n_medium.
OWE_FN Real fresnelConductor(Real cosThetaI, Real2 eta) {
    cosThetaI = clamp(cosThetaI, Real(0), Real(1));
    Real2 c = Real2(cosThetaI, 0);
    Real2 sin2T = cdiv(Real2(1 - cosThetaI * cosThetaI, 0), cmul(eta, eta));
    Real2 cosT = csqrt(Real2(1, 0) - sin2T);
    Real2 ec = eta * cosThetaI, ect = cmul(eta, cosT);
    Real2 rParl = cdiv(ec - cosT, ec + cosT);
    Real2 rPerp = cdiv(c - ect, c + ect);
    return (dot(rParl, rParl) + dot(rPerp, rPerp)) / 2;
}

OWE_FN bool sameHemisphere(Real3 a, Real3 b) { return a.z * b.z > 0; }

// Refraction of wi (pointing away from the surface) about n, pbrt-v4 style: eta = n_t / n_i
// relative to n's hemisphere; etap receives the effective ratio.
OWE_FN bool refractLocal(Real3 wi, Real3 n, Real eta, OWE_OUT(Real) etap, OWE_OUT(Real3) wt) {
    Real cosI = dot(n, wi);
    etap = 1;
    wt = wi;
    if (cosI < 0) {
        eta = 1 / eta;
        cosI = -cosI;
        n = -n;
    }
    Real sin2T = max(Real(0), 1 - cosI * cosI) / (eta * eta);
    if (sin2T >= 1) return false;
    Real cosT = safeSqrt(1 - sin2T);
    wt = -wi / eta + n * (cosI / eta - cosT);
    etap = eta;
    return true;
}

// The probability of taking the reflection at a smooth dielectric of reflectance R: R itself, or
// kept within [floor, 1 − floor] (a sampling choice that explores rare ghost reflections; the
// branch weights R/pR and T/(1 − pR) keep the estimate unbiased).
OWE_FN Real fresnelBranchProbability(Real R, Real floor) {
    return floor > 0 && R < 1 ? clamp(R, floor, 1 - floor) : R;
}

// ---------------------------------------------------------------- GGX (Trowbridge–Reitz)

// Isotropic GGX with roughness α (clamped away from a delta), visible-normal sampling.
OWE_CONST Real kSmoothRoughness = 1e-3;  // below this α a surface is treated as smooth (a delta)
OWE_FN Real ggxAlpha(Real roughness) { return max(roughness, Real(1e-4)); }

OWE_FN Real ggxD(Real a, Real3 wm) {
    Real c2 = wm.z * wm.z;
    if (c2 <= 0) return 0;
    Real c4 = c2 * c2;
    if (c4 < 1e-16) return 0;
    Real e = (1 - c2) / c2 / (a * a);
    return 1 / (Pi * a * a * c4 * sqr(1 + e));
}
OWE_FN Real ggxLambda(Real a, Real3 w) {
    Real c2 = w.z * w.z;
    if (c2 <= 0) return 0;
    return (sqrt(1 + a * a * ((1 - c2) / c2)) - 1) / 2;
}
OWE_FN Real ggxG1(Real a, Real3 w) { return 1 / (1 + ggxLambda(a, w)); }
OWE_FN Real ggxG(Real a, Real3 wo, Real3 wi) { return 1 / (1 + ggxLambda(a, wo) + ggxLambda(a, wi)); }
// The density of visible normals wm seen from w: the pdf of ggxSampleWm.
OWE_FN Real ggxPdf(Real a, Real3 w, Real3 wm) {
    if (w.z == 0) return 0;
    return ggxG1(a, w) / abs(w.z) * ggxD(a, wm) * abs(dot(w, wm));
}
OWE_FN Real3 ggxSampleWm(Real a, Real3 w, Real u1, Real u2) {
    Real3 wh = normalize(Real3(a * w.x, a * w.y, w.z));
    if (wh.z < 0) wh = -wh;
    Real3 T1 = wh.z < 0.99999 ? normalize(cross(Real3(0, 0, 1), wh)) : Real3(1, 0, 0);
    Real3 T2 = cross(wh, T1);
    Real r = sqrt(u1), phi = 2 * Pi * u2;
    Real px = r * cos(phi), py = r * sin(phi);
    Real h = sqrt(max(Real(0), 1 - px * px));
    Real s = (1 + wh.z) / 2;
    py = (1 - s) * h + s * py;
    Real pz = safeSqrt(1 - px * px - py * py);
    Real3 nh = T1 * px + T2 * py + wh * pz;
    return normalize(Real3(a * nh.x, a * nh.y, max(Real(1e-6), nh.z)));
}

// A sampled lobe direction (local frame) with its weight f·|cos|/pdf, its pdf, and the Fresnel
// split (diagnostics).
struct LobeSample {
    Real3 wi;
    Real weight;
    Real pdf;
    bool transmitted;
    Real R, T;
};

// Rough dielectric (Walter et al. 2007, pbrt-v4's formulation); eta = n_back / n_front.
OWE_FN bool sampleRoughDielectric(Real a, Real3 wo, Real eta, Real uc, Real u1, Real u2, bool radiance,
                                  OWE_OUT(LobeSample) s) {
    Real3 wm = ggxSampleWm(a, wo, u1, u2);
    Real R = fresnelDielectric(dot(wo, wm), eta), T = 1 - R;
    s.wi = wo;
    s.weight = 0;
    s.pdf = 0;
    s.transmitted = false;
    s.R = R;
    s.T = T;
    if (uc < R / (R + T)) {
        s.wi = reflect(-wo, wm);
        if (!sameHemisphere(wo, s.wi)) return false;
        s.pdf = ggxPdf(a, wo, wm) / (4 * abs(dot(wo, wm))) * R / (R + T);
        Real f = ggxD(a, wm) * ggxG(a, wo, s.wi) * R / (4 * abs(s.wi.z) * abs(wo.z));
        s.weight = f * abs(s.wi.z) / s.pdf;
    } else {
        Real etap;
        Real3 wi;
        if (!refractLocal(wo, wm, eta, etap, wi)) return false;
        if (sameHemisphere(wo, wi) || wi.z == 0) return false;
        s.wi = wi;
        s.transmitted = true;
        Real denom = sqr(dot(wi, wm) + dot(wo, wm) / etap);
        s.pdf = ggxPdf(a, wo, wm) * (abs(dot(wi, wm)) / denom) * T / (R + T);
        Real ft = T * ggxD(a, wm) * ggxG(a, wo, wi) * abs(dot(wi, wm) * dot(wo, wm) / (wi.z * wo.z * denom));
        if (radiance) ft /= sqr(etap);
        s.weight = ft * abs(wi.z) / s.pdf;
    }
    return isfinite(s.weight) && s.pdf > 0;
}

OWE_FN Real evalRoughDielectric(Real a, Real3 wo, Real3 wi, Real eta, bool radiance, OWE_OUT(Real) pdf) {
    pdf = 0;
    Real co = wo.z, ci = wi.z;
    if (co == 0 || ci == 0) return 0;
    bool refl = co * ci > 0;
    Real etap = 1;
    if (!refl) etap = co > 0 ? eta : 1 / eta;
    Real3 wm = wi * etap + wo;
    if (dot(wm, wm) == 0) return 0;
    wm = normalize(wm);
    if (wm.z < 0) wm = -wm;
    if (dot(wm, wi) * ci < 0 || dot(wm, wo) * co < 0) return 0;
    Real R = fresnelDielectric(dot(wo, wm), eta), T = 1 - R;
    if (refl) {
        pdf = ggxPdf(a, wo, wm) / (4 * abs(dot(wo, wm))) * R / (R + T);
        return ggxD(a, wm) * ggxG(a, wo, wi) * R / abs(4 * ci * co);
    }
    Real denom = sqr(dot(wi, wm) + dot(wo, wm) / etap);
    pdf = ggxPdf(a, wo, wm) * abs(dot(wi, wm)) / denom * T / (R + T);
    Real ft = ggxD(a, wm) * T * ggxG(a, wo, wi) * abs(dot(wi, wm) * dot(wo, wm) / (denom * ci * co));
    if (radiance) ft /= sqr(etap);
    return ft;
}

// Rough conductor seen from the upper hemisphere (w.z > 0): reflectance F·tint with complex
// relative index eta.
OWE_FN bool sampleRoughConductor(Real a, Real3 w, Real2 eta, Real tint, Real u1, Real u2, OWE_OUT(LobeSample) s) {
    Real3 wm = ggxSampleWm(a, w, u1, u2);
    s.wi = reflect(-w, wm);
    s.weight = 0;
    s.pdf = 0;
    s.transmitted = false;
    s.T = 0;
    s.R = fresnelConductor(abs(dot(w, wm)), eta) * tint;
    if (s.wi.z <= 0) return false;
    s.pdf = ggxPdf(a, w, wm) / (4 * abs(dot(w, wm)));
    if (!(s.pdf > 0)) return false;
    Real f = ggxD(a, wm) * s.R * ggxG(a, w, s.wi) / (4 * s.wi.z * w.z);
    s.weight = f * s.wi.z / s.pdf;
    return isfinite(s.weight);
}
OWE_FN Real evalRoughConductor(Real a, Real3 wo, Real3 wi, Real2 eta, Real tint, OWE_OUT(Real) pdf) {
    pdf = 0;
    Real3 wm = wo + wi;
    if (dot(wm, wm) == 0) return 0;
    wm = normalize(wm);
    Real F = fresnelConductor(abs(dot(wo, wm)), eta) * tint;
    pdf = ggxPdf(a, wo, wm) / (4 * abs(dot(wo, wm)));
    return ggxD(a, wm) * F * ggxG(a, wo, wi) / (4 * wi.z * wo.z);
}

// ---------------------------------------------------------------- sampling guides

// The solid-angle density at p of directions toward a uniform point of the disc (centre, unit
// normal, radius). generatingDisk: the direction is known to come from that disc (rebuilding its
// point loses precision for a millimetre pupil metres away, and a false rejection would leave the
// mixture's other components alone with enormous weights).
OWE_FN Real diskAimPdf(Real3 p, Real3 wi, Real3 center, Real3 normal, Real radius, bool generatingDisk) {
    if (!(radius > 0)) return 0;
    Real dn = dot(wi, normal);
    if (abs(dn) < 1e-12) return 0;
    Real t = dot(center - p, normal) / dn;
    Real3 delta = p + wi * t - center;
    if (t <= 0 || (!generatingDisk && dot(delta, delta) > radius * radius)) return 0;
    return t * t / (abs(dn) * Pi * radius * radius);
}

OWE_SHARED_END
