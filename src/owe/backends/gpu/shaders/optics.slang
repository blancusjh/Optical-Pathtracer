// Interface optics: Snell, Fresnel (dielectric, conductor), GGX, and boundary scattering.
// Float port of optics.hpp and scattering.cpp; both transport directions are supported.
#pragma once
#include "common.slang"
#include "spectrum.slang"
#include "texture.slang"

static const uint kModeRadiance = 0, kModeImportance = 1;

// Snell refraction of propagation direction d (normal of either orientation), ni → nt.
bool refractDirection(float3 d, float3 n, float ni, float nt, out float3 outDir) {
    float3 nn = dot(d, n) < 0 ? n : -n;
    float cosi = -dot(d, nn);
    float eta = ni / nt;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    outDir = 0;
    if (k < 0) return false;
    outDir = normalize(d * eta + nn * (eta * cosi - sqrt(k)));
    return true;
}

// Unpolarised dielectric reflectance (pbrt convention, eta = n_inside / n_outside).
float fresnelDielectric(float cosThetaI, float eta) {
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);
    if (cosThetaI < 0) { eta = 1 / eta; cosThetaI = -cosThetaI; }
    float sin2T = (1 - cosThetaI * cosThetaI) / (eta * eta);
    if (sin2T >= 1) return 1;
    float cosT = safeSqrt(1 - sin2T);
    float rParl = (eta * cosThetaI - cosT) / (eta * cosThetaI + cosT);
    float rPerp = (cosThetaI - eta * cosT) / (cosThetaI + eta * cosT);
    return (rParl * rParl + rPerp * rPerp) / 2;
}

float2 cmul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
float2 cdiv(float2 a, float2 b) {
    float d = dot(b, b);
    return float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}
float2 csqrt(float2 z) {  // principal branch, as std::sqrt
    float r = length(z);
    float re = sqrt(max(0.0, 0.5 * (r + z.x)));
    float im = sqrt(max(0.0, 0.5 * (r - z.x)));
    return float2(re, copySign(im, z.y));
}
// Unpolarised reflectance of a conductor with complex relative index eta = (n + ik) / n_medium.
float fresnelConductor(float cosThetaI, float2 eta) {
    cosThetaI = clamp(cosThetaI, 0.0, 1.0);
    float sin2I = 1 - cosThetaI * cosThetaI;
    float2 sin2T = cdiv(float2(sin2I, 0), cmul(eta, eta));
    float2 cosT = csqrt(float2(1, 0) - sin2T);
    float2 ec = eta * cosThetaI;
    float2 rParl = cdiv(ec - cosT, ec + cosT);
    float2 ect = cmul(eta, cosT);
    float2 rPerp = cdiv(float2(cosThetaI, 0) - ect, float2(cosThetaI, 0) + ect);
    return (dot(rParl, rParl) + dot(rPerp, rPerp)) / 2;
}

bool sameHemisphere(float3 a, float3 b) { return a.z * b.z > 0; }

bool refractLocal(float3 wi, float3 n, float eta, out float etap, out float3 wt) {
    float cosI = dot(n, wi);
    etap = 1;
    wt = 0;
    if (cosI < 0) { eta = 1 / eta; cosI = -cosI; n = -n; }
    float sin2I = max(0.0, 1 - cosI * cosI);
    float sin2T = sin2I / (eta * eta);
    if (sin2T >= 1) return false;
    float cosT = safeSqrt(1 - sin2T);
    wt = -wi / eta + n * (cosI / eta - cosT);
    etap = eta;
    return true;
}

// Isotropic Trowbridge–Reitz (GGX) with visible-normal sampling.
struct GGX {
    float a;
    __init(float alpha) { a = max(alpha, 1e-4); }
    float D(float3 wm) {
        float c2 = wm.z * wm.z;
        if (c2 <= 0) return 0;
        float tan2 = (1 - c2) / c2;
        float c4 = c2 * c2;
        if (c4 < 1e-16) return 0;
        float e = tan2 / (a * a);
        return 1 / (kPi * a * a * c4 * sqr(1 + e));
    }
    float lambda(float3 w) {
        float c2 = w.z * w.z;
        if (c2 <= 0) return 0;
        float tan2 = (1 - c2) / c2;
        return (sqrt(1 + a * a * tan2) - 1) / 2;
    }
    float G1(float3 w) { return 1 / (1 + lambda(w)); }
    float G(float3 wo, float3 wi) { return 1 / (1 + lambda(wo) + lambda(wi)); }
    float Dv(float3 w, float3 wm) {
        if (abs(w.z) == 0) return 0;
        return G1(w) / abs(w.z) * D(wm) * abs(dot(w, wm));
    }
    float pdf(float3 w, float3 wm) { return Dv(w, wm); }
    float3 sampleWm(float3 w, float u1, float u2) {
        float3 wh = normalize(float3(a * w.x, a * w.y, w.z));
        if (wh.z < 0) wh = -wh;
        float3 T1 = (wh.z < 0.99999) ? normalize(cross(float3(0, 0, 1), wh)) : float3(1, 0, 0);
        float3 T2 = cross(wh, T1);
        float r = sqrt(u1), phi = 2 * kPi * u2;
        float px = r * cos(phi), py = r * sin(phi);
        float h = sqrt(max(0.0, 1 - px * px));
        float s = (1 + wh.z) / 2;
        py = (1 - s) * h + s * py;
        float pz = safeSqrt(1 - px * px - py * py);
        float3 nh = T1 * px + T2 * py + wh * pz;
        return normalize(float3(a * nh.x, a * nh.y, max(1e-6, nh.z)));
    }
};

// ---------------------------------------------------------------- interfaces

struct Optics {
    uint type, texKind, reflectance, backAbsorbs, texA, texB, texC, conductorN, conductorK;
    float roughness, texScale, texParam;
    float4 sampleAimCenter, sampleAimNormal;
    float4 roughMap;
    float4 relief;
    bool isDelta() {
        // Stained glass: gain, density, haze, reflection scale in sampleAimCenter.
        if (type == kSurfStainedGlass) return sampleAimCenter.z <= 0 && sampleAimCenter.w <= 0;
        return (type == kSurfDielectric || type == kSurfConductor) ? roughness < 1e-3 : type == kSurfMirror;
    }
    // Whether scattering reads the refractive indices on either side (diffuse and mirror surfaces
    // do not, so an interface at another wavelength needs no index evaluation).
    bool usesIndices() { return type == kSurfDielectric || type == kSurfConductor; }
};
Optics loadOptics(uint id) {
    GOptics g = g_optics[id];
    Optics o;
    o.type = g.a.x;
    o.texKind = g.a.y;
    o.reflectance = g.a.z;
    o.backAbsorbs = g.a.w;
    o.texA = g.b.x;
    o.texB = g.b.y;
    o.texC = g.b.z;
    o.conductorN = g.b.w;
    o.conductorK = g.c.x;
    o.roughness = asfloat(g.c.y);
    o.texScale = asfloat(g.c.z);
    o.texParam = asfloat(g.c.w);
    o.sampleAimCenter = g.sampleAimCenter;
    o.sampleAimNormal = g.sampleAimNormal;
    o.roughMap = g.roughMap;
    o.relief = g.relief;
    return o;
}

// The reflectance (or a stained glass's colour) of optics o at wavelength l, from texture weights.
float opticsAlbedo(Optics o, float3 tex, float l) {
    if (o.texKind == kTexNone) return spectrumEval(o.reflectance, l);
    float r = tex.x * spectrumEval(o.texA, l);
    if (tex.y != 0) r += tex.y * spectrumEval(o.texB, l);
    if (tex.z != 0) r += tex.z * spectrumEval(o.texC, l);
    return r;
}

// A thin stained-glass slab met at geometric incidence cosI from a medium of index nOut: specular
// reflectance R, total transmittance T (straight and diffuse, before the haze split) and diffuse
// reflectance Rd. Fresnel F at each face, internal transmittance a = τ₀^(1/cos θₜ) with
// τ₀ = clamp(gain·colour)^density, incoherent inter-reflections:
//   T = (1−F)² a / (1 − F² a²),   R = F + (1−F)² F a² / (1 − F² a²),   Rd = (1 − R − T)·ρ.
float3 stainedSlab(Optics o, float3 tex, float cosI, float lambda, float nOut) {
    float eta = spectrumEval(o.conductorN, lambda) / nOut;
    float c = clamp(cosI, 0.0, 1.0);
    float F = fresnelDielectric(c, eta);
    float cosT = safeSqrt(1 - (1 - c * c) / (eta * eta));
    float colour = opticsAlbedo(o, tex, lambda);
    float tau0 = pow(saturate(o.sampleAimCenter.x * colour), o.sampleAimCenter.y);
    float a = cosT > 1e-6 ? pow(tau0, 1 / cosT) : 0.0;
    float den = 1 - F * F * a * a;
    float T = den > 1e-12 ? (1 - F) * (1 - F) * a / den : 0.0;
    float R = den > 1e-12 ? F + (1 - F) * (1 - F) * F * a * a / den : 1.0;
    float Rd = max(0.0, 1 - R - T) * saturate(o.sampleAimCenter.w * colour);
    return float3(R, T, Rd);
}
// The probabilities of a stained glass's branches: specular reflection, straight passage, diffuse
// transmission, diffuse reflection. The same at every wavelength (taken at 550 nm in air), half in
// proportion to each branch's value there and half uniform over the branches the material has, so
// a branch that carries light at any wavelength is sampled.
static const float kAirIndex = 1.00028;
float4 stainedProbs(Optics o, float3 tex, float cosI) {
    float3 s = stainedSlab(o, tex, cosI, 550.0, kAirIndex);
    float h = o.sampleAimCenter.z;
    float4 v = float4(s.x, (1 - h) * s.y, h * s.y, s.z);
    float4 on = float4(1, h < 1 ? 1 : 0, h > 0 ? 1 : 0, o.sampleAimCenter.w > 0 ? 1 : 0);
    float sum = v.x + v.y + v.z + v.w;
    float4 p = on / (on.x + on.y + on.z + on.w);
    return sum > 0 ? 0.5 * p + 0.5 * v / sum : p;
}

struct Interface {
    Optics optics;
    float3 n;              // world normal pointing to the front region
    float3 ns;             // shading normal (smooth meshes; else n): the frame scattering happens in
    float3 p;
    float nFront, nBack;   // refractive indices at `lambda`
    float lambda;
    float fresnelFloor;
    float3 tex;            // texture mixing weights at the hit

    float albedo() {
        if (optics.texKind == kTexNone) return spectrumEval(optics.reflectance, lambda);
        float r = tex.x * spectrumEval(optics.texA, lambda);
        if (tex.y != 0) r += tex.y * spectrumEval(optics.texB, lambda);
        if (tex.z != 0) r += tex.z * spectrumEval(optics.texC, lambda);
        return r;
    }
    float2 conductorEta(float nMedium) {
        return float2(spectrumEval(optics.conductorN, lambda), spectrumEval(optics.conductorK, lambda)) / nMedium;
    }
    bool smooth() { return any(ns != n); }
    // A material map: the roughness at this point, from the texture weights (set once they are).
    [mutating] void applyMaterialMap() {
        if (optics.roughMap.w != 0) optics.roughness = max(0.0, dot(tex, optics.roughMap.xyz));
    }
    // A direction lies on the same side of the shading and the geometric surface: scattering
    // about ns then reflects or transmits exactly where the geometry does (regions stay right).
    bool agrees(float3 w) { return dot(w, ns) * dot(w, n) > 0; }
};

// Shading normals (Veach 1997, §5.3). Scattering is evaluated in the frame of ns, and the cosine
// of the rendering equation is |wi·ns|; transported the other way (particles, photons), the
// adjoint of that BSDF carries |wo·ns| |wi·n| / (|wo·n| |wi·ns|), so that both directions
// estimate the same image. wo: toward where the path came from; wi: where it goes.
float adjointCorrection(Interface it, float3 wo, float3 wi, uint mode) {
    if (mode == kModeRadiance || !it.smooth()) return 1;
    float den = abs(dot(wo, it.n)) * abs(dot(wi, it.ns));
    return den > 0 ? abs(dot(wo, it.ns)) * abs(dot(wi, it.n)) / den : 0.0;
}
// The factor multiplying f(wo, wi) for a connection toward wi (next-event estimation, a particle's
// link to the observer): |wi·ns|, times the adjoint correction for importance.
float scatterCosine(Interface it, float3 wo, float3 wi, uint mode) {
    if (!it.smooth()) return abs(dot(wi, it.n));
    return abs(dot(wi, it.ns)) * adjointCorrection(it, wo, wi, mode);
}

struct ScatterSample {
    float3 wi;
    float weight;  // f·|cosθ|/pdf, including the radiance n² factor when applicable
    float pdf;
    bool delta;
    bool transmitted;
};

bool sampleRoughDielectric(GGX mf, float3 wo, float eta, float uc, float u1, float u2, uint mode, inout ScatterSample s,
                           out float3 wiLocal) {
    float3 wm = mf.sampleWm(wo, u1, u2);
    float R = fresnelDielectric(dot(wo, wm), eta), T = 1 - R;
    wiLocal = 0;
    if (uc < R / (R + T)) {
        wiLocal = reflectDir(-wo, wm);
        if (!sameHemisphere(wo, wiLocal)) return false;
        s.pdf = mf.pdf(wo, wm) / (4 * abs(dot(wo, wm))) * R / (R + T);
        float f = mf.D(wm) * mf.G(wo, wiLocal) * R / (4 * abs(wiLocal.z) * abs(wo.z));
        s.weight = f * abs(wiLocal.z) / s.pdf;
        s.transmitted = false;
    } else {
        float etap;
        if (!refractLocal(wo, wm, eta, etap, wiLocal)) return false;
        if (sameHemisphere(wo, wiLocal) || wiLocal.z == 0) return false;
        float denom = sqr(dot(wiLocal, wm) + dot(wo, wm) / etap);
        float dwmdwi = abs(dot(wiLocal, wm)) / denom;
        s.pdf = mf.pdf(wo, wm) * dwmdwi * T / (R + T);
        float ft = T * mf.D(wm) * mf.G(wo, wiLocal) * abs(dot(wiLocal, wm) * dot(wo, wm) / (wiLocal.z * wo.z * denom));
        if (mode == kModeRadiance) ft /= sqr(etap);
        s.weight = ft * abs(wiLocal.z) / s.pdf;
        s.transmitted = true;
    }
    s.delta = false;
    return isfinite(s.weight) && s.pdf > 0;
}

float evalRoughDielectric(GGX mf, float3 wo, float3 wi, float eta, uint mode, out float pdf) {
    pdf = 0;
    float co = wo.z, ci = wi.z;
    if (co == 0 || ci == 0) return 0;
    bool refl = co * ci > 0;
    float etap = 1;
    if (!refl) etap = co > 0 ? eta : 1 / eta;
    float3 wm = wi * etap + wo;
    if (dot(wm, wm) == 0) return 0;
    wm = normalize(wm);
    if (wm.z < 0) wm = -wm;
    if (dot(wm, wi) * ci < 0 || dot(wm, wo) * co < 0) return 0;
    float R = fresnelDielectric(dot(wo, wm), eta), T = 1 - R;
    if (refl) {
        pdf = mf.pdf(wo, wm) / (4 * abs(dot(wo, wm))) * R / (R + T);
        return mf.D(wm) * mf.G(wo, wi) * R / abs(4 * ci * co);
    }
    float denom = sqr(dot(wi, wm) + dot(wo, wm) / etap);
    pdf = mf.pdf(wo, wm) * abs(dot(wi, wm)) / denom * T / (R + T);
    float ft = mf.D(wm) * T * mf.G(wo, wi) * abs(dot(wi, wm) * dot(wo, wm) / (denom * ci * co));
    if (mode == kModeRadiance) ft /= sqr(etap);
    return ft;
}

// Samples the continuation of a path arriving along propagation direction d.
float screenAimPdf(Interface it, float3 wi, bool generatingDisk = false) {
    float4 aim = it.optics.sampleAimCenter;
    float3 normal = it.optics.sampleAimNormal.xyz;
    if (aim.w <= 0) return 0;
    float dn = dot(wi, normal);
    if (abs(dn) < 1e-12) return 0;
    float t = dot(aim.xyz - it.p, normal) / dn;
    float3 delta = it.p + wi * t - aim.xyz;
    if (t <= 0 || (!generatingDisk && dot(delta, delta) > aim.w * aim.w)) return 0;
    return t * t / (abs(dn) * kPi * aim.w * aim.w);
}

bool sampleScatterIn(Interface it, float3 d, float uc, float u1, float u2, uint mode, out ScatterSample s);

bool sampleScatter(Interface it, float3 d, float uc, float u1, float u2, uint mode, out ScatterSample s) {
    if (!it.smooth()) return sampleScatterIn(it, d, uc, u1, u2, mode, s);
    // Where the shading and the geometric surface disagree about a direction's side, the sample is
    // lost (grazing directions on strongly curved smooth meshes): never a crossing the geometry
    // does not make.
    if (!it.agrees(-d) || !sampleScatterIn(it, d, uc, u1, u2, mode, s) || !it.agrees(s.wi)) return false;
    s.weight *= adjointCorrection(it, -d, s.wi, mode);
    return s.weight > 0;
}

// Scattering in the shading frame (ns).
bool sampleScatterIn(Interface it, float3 d, float uc, float u1, float u2, uint mode, out ScatterSample s) {
    Optics o = it.optics;
    Frame frame = Frame(it.ns);
    float3 wo = frame.toLocal(-d);
    bool fromFront = wo.z > 0;
    s.wi = 0;
    s.weight = 0;
    s.pdf = 0;
    s.delta = false;
    s.transmitted = false;

    switch (o.type) {
        case kSurfDielectric: {
            float eta = it.nBack / it.nFront;
            if (o.roughness < 1e-3 || eta == 1) {
                float ni = fromFront ? it.nFront : it.nBack;
                float nt = fromFront ? it.nBack : it.nFront;
                float R = fresnelDielectric(wo.z, eta);
                s.delta = true;
                float pR = R;
                if (it.fresnelFloor > 0 && R < 1) pR = clamp(R, it.fresnelFloor, 1 - it.fresnelFloor);
                if (uc < pR) {
                    s.wi = frame.toWorld(float3(-wo.x, -wo.y, wo.z));
                    s.weight = R / pR;
                    s.transmitted = false;
                } else {
                    float3 wt;
                    if (!refractDirection(d, it.ns, ni, nt, wt)) return false;
                    s.wi = wt;
                    s.weight = (1 - R) / (1 - pR) * (mode == kModeRadiance ? sqr(ni / nt) : 1.0);
                    s.transmitted = true;
                }
                return true;
            }
            GGX mf = GGX(o.roughness);
            float3 wiL;
            if (!sampleRoughDielectric(mf, wo, eta, uc, u1, u2, mode, s, wiL)) return false;
            s.wi = frame.toWorld(wiL);
            return true;
        }
        case kSurfDiffuse: {
            if (!fromFront && o.backAbsorbs != 0) return false;
            float albedo = it.albedo();
            if (albedo <= 0) return false;
            float3 w = sampleCosineHemisphere(u1, u2);
            if (!fromFront) w.z = -w.z;
            s.wi = frame.toWorld(w);
            s.pdf = abs(w.z) * kInvPi;
            s.weight = albedo;
            float share = fromFront && mode == kModeRadiance ? o.sampleAimNormal.w : 0;
            if (share > 0) {
                if (uc < share) {
                    float3 a, b;
                    orthonormalBasis(o.sampleAimNormal.xyz, a, b);
                    float2 disk = sampleUniformDiskConcentric(u1, u2);
                    s.wi = normalize(o.sampleAimCenter.xyz + (a * disk.x + b * disk.y) * o.sampleAimCenter.w - it.p);
                }
                float c = dot(s.wi, it.ns);
                if (c <= 0) return false;
                s.pdf = (1 - share) * c * kInvPi + share * screenAimPdf(it, s.wi, uc < share);
                s.weight = albedo * c * kInvPi / s.pdf;
            }
            return s.pdf > 0;
        }
        case kSurfStainedGlass: {
            // The straight passage was decided before (the kernels' passStainedGlass, a crossing
            // like a null boundary's): the other branches share the rest, 1 − pS, of the samples.
            float cosI = abs(dot(d, it.n));
            float4 p = stainedProbs(o, it.tex, cosI);
            float3 sl = stainedSlab(o, it.tex, cosI, it.lambda, kAirIndex);
            float x = uc * (1 - p.y);
            if (x < p.x) {
                s.wi = frame.toWorld(float3(-wo.x, -wo.y, wo.z));
                s.weight = sl.x / p.x;
                s.delta = true;
                return s.weight > 0;
            }
            bool transmit = x < p.x + p.z;
            float pb = transmit ? p.z : p.w;
            float value = transmit ? o.sampleAimCenter.z * sl.y : sl.z;
            if (!(pb > 0) || !(value > 0)) return false;
            float3 w = sampleCosineHemisphere(u1, u2);
            if (fromFront == transmit) w.z = -w.z;
            s.wi = frame.toWorld(w);
            s.pdf = pb * abs(w.z) * kInvPi;
            s.weight = value / pb;
            s.transmitted = transmit;
            return true;
        }
        case kSurfMirror: {
            if (!fromFront && o.backAbsorbs != 0) return false;
            float r = spectrumEval(o.reflectance, it.lambda);
            if (r <= 0) return false;
            s.wi = frame.toWorld(float3(-wo.x, -wo.y, wo.z));
            s.weight = r;
            s.delta = true;
            return true;
        }
        case kSurfConductor: {
            if (!fromFront && o.backAbsorbs != 0) return false;
            float nMed = fromFront ? it.nFront : it.nBack;
            float2 eta = it.conductorEta(nMed);
            float3 w = fromFront ? wo : float3(wo.x, wo.y, -wo.z);
            if (o.roughness < 1e-3) {
                float F = fresnelConductor(w.z, eta) * it.albedo();
                s.wi = frame.toWorld(float3(-wo.x, -wo.y, wo.z));
                s.weight = F;
                s.delta = true;
                return F > 0;
            }
            GGX mf = GGX(o.roughness);
            float3 wm = mf.sampleWm(w, u1, u2);
            float3 wiL = reflectDir(-w, wm);
            if (wiL.z <= 0) return false;
            float pdf = mf.pdf(w, wm) / (4 * abs(dot(w, wm)));
            float F = fresnelConductor(abs(dot(w, wm)), eta) * it.albedo();
            float f = mf.D(wm) * F * mf.G(w, wiL) / (4 * wiL.z * w.z);
            if (!(pdf > 0)) return false;
            if (!fromFront) wiL.z = -wiL.z;
            s.wi = frame.toWorld(wiL);
            s.pdf = pdf;
            s.weight = f * abs(wiL.z) / pdf;
            return isfinite(s.weight);
        }
        default: return false;
    }
}

// f(wo, wi) of the non-delta components (wo = −d) and the pdf of sampling wi. With shading normals,
// f of the shading frame; the cosine (and adjoint correction) is scatterCosine's.
float evalScatter(Interface it, float3 d, float3 wiWorld, uint mode, out float pdf) {
    pdf = 0;
    Optics o = it.optics;
    if (o.isDelta()) return 0;
    if (it.smooth() && !(it.agrees(-d) && it.agrees(wiWorld))) return 0;
    Frame frame = Frame(it.ns);
    float3 wo = frame.toLocal(-d);
    float3 wi = frame.toLocal(wiWorld);
    bool fromFront = wo.z > 0;
    switch (o.type) {
        case kSurfDiffuse: {
            if (!fromFront && o.backAbsorbs != 0) return 0;
            if (!sameHemisphere(wo, wi)) return 0;
            pdf = abs(wi.z) * kInvPi;
            float share = fromFront && mode == kModeRadiance ? o.sampleAimNormal.w : 0;
            if (share > 0) pdf = (1 - share) * pdf + share * screenAimPdf(it, wiWorld);
            return it.albedo() * kInvPi;
        }
        case kSurfConductor: {
            if (!fromFront && o.backAbsorbs != 0) return 0;
            if (!sameHemisphere(wo, wi)) return 0;
            if (!fromFront) { wo.z = -wo.z; wi.z = -wi.z; }
            GGX mf = GGX(o.roughness);
            float3 wm = wo + wi;
            if (dot(wm, wm) == 0) return 0;
            wm = normalize(wm);
            float nMed = fromFront ? it.nFront : it.nBack;
            float F = fresnelConductor(abs(dot(wo, wm)), it.conductorEta(nMed)) * it.albedo();
            pdf = mf.pdf(wo, wm) / (4 * abs(dot(wo, wm)));
            return mf.D(wm) * F * mf.G(wo, wi) / (4 * wi.z * wo.z);
        }
        case kSurfDielectric: {
            GGX mf = GGX(o.roughness);
            return evalRoughDielectric(mf, wo, wi, it.nBack / it.nFront, mode, pdf);
        }
        case kSurfStainedGlass: {
            float cosI = abs(dot(d, it.n));
            float4 p = stainedProbs(o, it.tex, cosI);
            float3 sl = stainedSlab(o, it.tex, cosI, it.lambda, kAirIndex);
            bool same = sameHemisphere(wo, wi);
            pdf = (same ? p.w : p.z) * abs(wi.z) * kInvPi;
            return (same ? sl.z : o.sampleAimCenter.z * sl.y) * kInvPi;
        }
        default: return 0;
    }
}

bool isDispersive(Interface hero, Interface other) {
    float a = hero.nBack / hero.nFront, b = other.nBack / other.nFront;
    return abs(a - b) > 1e-12 * abs(a);
}

// Throughput multiplier at another wavelength for the direction the hero sampled.
float secondaryScatterWeight(Interface it, float3 d, ScatterSample s, uint mode) {
    Optics o = it.optics;
    if (!s.delta) {
        float pdf;
        float f = evalScatter(it, d, s.wi, mode, pdf);
        return s.pdf > 0 ? f * scatterCosine(it, -d, s.wi, mode) / s.pdf : 0.0;
    }
    float cosI = abs(dot(d, it.ns));
    switch (o.type) {
        case kSurfDielectric: return s.weight;  // (a dispersive interface ends the secondaries)
        case kSurfStainedGlass: {  // the specular reflection (straight passage is the kernels')
            float c = abs(dot(d, it.n));
            float4 p = stainedProbs(o, it.tex, c);
            return p.x > 0 ? stainedSlab(o, it.tex, c, it.lambda, kAirIndex).x / p.x : 0.0;
        }
        case kSurfMirror: return spectrumEval(o.reflectance, it.lambda) * adjointCorrection(it, -d, s.wi, mode);
        case kSurfConductor: {
            bool fromFront = dot(d, it.n) < 0;
            float nMed = fromFront ? it.nFront : it.nBack;
            return fresnelConductor(cosI, it.conductorEta(nMed)) * it.albedo() *
                   adjointCorrection(it, -d, s.wi, mode);
        }
        default: return 0;
    }
}
