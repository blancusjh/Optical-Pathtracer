// Boundary scattering in the kernels: the reference's Snell, Fresnel and GGX lobes
// (owe/core/shared/optics.hpp) applied to the scene's optics records, plus what only the GPU
// renders: shading normals and stained glass. Both transport directions are supported.
#pragma once
#include "owe/core/shared/optics.hpp"
#include "common.slang"
#include "spectrum.slang"
#include "texture.slang"

static const uint kModeRadiance = 0, kModeImportance = 1;

// ---------------------------------------------------------------- interfaces

struct Optics {
    uint type, texKind, reflectance, backAbsorbs, texA, texB, texC, conductorN, conductorK;
    float roughness, texScale, texParam;
    float4 sampleAimCenter, sampleAimNormal;
    float4 roughMap;
    float4 relief;
    uint4 maps;
    float4 mapParams;
    bool isDelta() {
        // Stained glass: gain, density, haze, reflection scale in sampleAimCenter.
        if (type == kSurfStainedGlass) return sampleAimCenter.z <= 0 && sampleAimCenter.w <= 0;
        return (type == kSurfDielectric || type == kSurfConductor) ? roughness < kSmoothRoughness : type == kSurfMirror;
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
    o.maps = g.maps;
    o.mapParams = g.mapParams;
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

    float albedo() { return opticsAlbedo(optics, tex, lambda); }
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

// The density of a diffuse screen's sampling guide toward wi.
float screenAimPdf(Interface it, float3 wi, bool generatingDisk = false) {
    float4 aim = it.optics.sampleAimCenter;
    return diskAimPdf(it.p, wi, aim.xyz, it.optics.sampleAimNormal.xyz, aim.w, generatingDisk);
}

// A lobe sample (local frame) as a world sample.
ScatterSample fromLobe(Frame frame, LobeSample l) {
    ScatterSample s;
    s.wi = frame.toWorld(l.wi);
    s.weight = l.weight;
    s.pdf = l.pdf;
    s.delta = false;
    s.transmitted = l.transmitted;
    return s;
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

// The samplers of each kind of surface, in the shading frame: wo = −d there, fromFront its side.

bool sampleDielectric(Interface it, Frame frame, float3 d, float3 wo, bool fromFront, float uc, float u1, float u2,
                      uint mode, inout ScatterSample s) {
    Optics o = it.optics;
    float eta = it.nBack / it.nFront;
    if (o.roughness < kSmoothRoughness || eta == 1) {
        float ni = fromFront ? it.nFront : it.nBack;
        float nt = fromFront ? it.nBack : it.nFront;
        float R = fresnelDielectric(wo.z, eta);
        s.delta = true;
        float pR = fresnelBranchProbability(R, it.fresnelFloor);
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
    LobeSample l;
    if (!sampleRoughDielectric(ggxAlpha(o.roughness), wo, eta, uc, u1, u2, mode == kModeRadiance, l)) return false;
    s = fromLobe(frame, l);
    return true;
}

// Cosine-weighted; a share of radiance samples aims at the screen the surface faces (sampleAim).
bool sampleDiffuse(Interface it, Frame frame, bool fromFront, float uc, float u1, float u2, uint mode,
                   inout ScatterSample s) {
    Optics o = it.optics;
    if (!fromFront && o.backAbsorbs != 0) return false;
    float albedo = it.albedo();
    if (albedo <= 0) return false;
    float3 w = sampleCosineHemisphere(u1, u2);
    if (!fromFront) w.z = -w.z;
    s.wi = frame.toWorld(w);
    s.pdf = abs(w.z) * InvPi;
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
        s.pdf = (1 - share) * c * InvPi + share * screenAimPdf(it, s.wi, uc < share);
        s.weight = albedo * c * InvPi / s.pdf;
    }
    return s.pdf > 0;
}

// The straight passage was decided before (the kernels' passStainedGlass, a crossing like a null
// boundary's): the other branches share the rest, 1 − pS, of the samples.
bool sampleStainedGlass(Interface it, Frame frame, float3 d, float3 wo, bool fromFront, float uc, float u1, float u2,
                        inout ScatterSample s) {
    Optics o = it.optics;
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
    s.pdf = pb * abs(w.z) * InvPi;
    s.weight = value / pb;
    s.transmitted = transmit;
    return true;
}

bool sampleMirror(Interface it, Frame frame, float3 wo, bool fromFront, inout ScatterSample s) {
    if (!fromFront && it.optics.backAbsorbs != 0) return false;
    float r = spectrumEval(it.optics.reflectance, it.lambda);
    if (r <= 0) return false;
    s.wi = frame.toWorld(float3(-wo.x, -wo.y, wo.z));
    s.weight = r;
    s.delta = true;
    return true;
}

bool sampleConductor(Interface it, Frame frame, float3 wo, bool fromFront, float u1, float u2, inout ScatterSample s) {
    Optics o = it.optics;
    if (!fromFront && o.backAbsorbs != 0) return false;
    float nMed = fromFront ? it.nFront : it.nBack;
    float2 eta = it.conductorEta(nMed);
    float3 w = fromFront ? wo : float3(wo.x, wo.y, -wo.z);
    if (o.roughness < kSmoothRoughness) {
        float F = fresnelConductor(w.z, eta) * it.albedo();
        s.wi = frame.toWorld(float3(-wo.x, -wo.y, wo.z));
        s.weight = F;
        s.delta = true;
        return F > 0;
    }
    LobeSample l;
    if (!sampleRoughConductor(ggxAlpha(o.roughness), w, eta, it.albedo(), u1, u2, l)) return false;
    if (!fromFront) l.wi.z = -l.wi.z;
    s = fromLobe(frame, l);
    return true;
}

// Scattering in the shading frame (ns).
bool sampleScatterIn(Interface it, float3 d, float uc, float u1, float u2, uint mode, out ScatterSample s) {
    Frame frame = Frame(it.ns);
    float3 wo = frame.toLocal(-d);
    bool fromFront = wo.z > 0;
    s.wi = 0;
    s.weight = 0;
    s.pdf = 0;
    s.delta = false;
    s.transmitted = false;
    switch (it.optics.type) {
        case kSurfDielectric: return sampleDielectric(it, frame, d, wo, fromFront, uc, u1, u2, mode, s);
        case kSurfDiffuse: return sampleDiffuse(it, frame, fromFront, uc, u1, u2, mode, s);
        case kSurfStainedGlass: return sampleStainedGlass(it, frame, d, wo, fromFront, uc, u1, u2, s);
        case kSurfMirror: return sampleMirror(it, frame, wo, fromFront, s);
        case kSurfConductor: return sampleConductor(it, frame, wo, fromFront, u1, u2, s);
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
            pdf = abs(wi.z) * InvPi;
            float share = fromFront && mode == kModeRadiance ? o.sampleAimNormal.w : 0;
            if (share > 0) pdf = (1 - share) * pdf + share * screenAimPdf(it, wiWorld);
            return it.albedo() * InvPi;
        }
        case kSurfConductor: {
            if (!fromFront && o.backAbsorbs != 0) return 0;
            if (!sameHemisphere(wo, wi)) return 0;
            if (!fromFront) { wo.z = -wo.z; wi.z = -wi.z; }
            float nMed = fromFront ? it.nFront : it.nBack;
            return evalRoughConductor(ggxAlpha(o.roughness), wo, wi, it.conductorEta(nMed), it.albedo(), pdf);
        }
        case kSurfDielectric:
            return evalRoughDielectric(ggxAlpha(o.roughness), wo, wi, it.nBack / it.nFront, mode == kModeRadiance, pdf);
        case kSurfStainedGlass: {
            float cosI = abs(dot(d, it.n));
            float4 p = stainedProbs(o, it.tex, cosI);
            float3 sl = stainedSlab(o, it.tex, cosI, it.lambda, kAirIndex);
            bool same = sameHemisphere(wo, wi);
            pdf = (same ? p.w : p.z) * abs(wi.z) * InvPi;
            return (same ? sl.z : o.sampleAimCenter.z * sl.y) * InvPi;
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
