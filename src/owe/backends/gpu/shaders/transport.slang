// The transport engine for camera paths: float port of Tracer::radiance (transport.cpp) with the
// same structure, region bookkeeping, next-event estimation, MIS and random-number order.
#pragma once
#include "optics.slang"
#include "world.slang"

// The first region inconsistency of a pass (as TransportStats::first on the CPU): words 8–15 of
// g_counters hold a claim flag, the boundary, the ray's region, the boundary's region on the side
// met, the previous boundary and the camera-relative position (float bits).
void noteInconsistency(uint boundary, uint rayRegion, uint boundaryRegion, uint previous, float3 p) {
    uint old;
    InterlockedCompareExchange(g_counters[8], 0u, 1u, old);
    if (old != 0) return;
    g_counters[9] = boundary;
    g_counters[10] = rayRegion;
    g_counters[11] = boundaryRegion;
    g_counters[12] = previous;
    g_counters[13] = asuint(p.x);
    g_counters[14] = asuint(p.y);
    g_counters[15] = asuint(p.z);
}

// ---------------------------------------------------------------- environment

// Environment map (world.cpp: Environment::mapRGB, sampleMap, mapPdf).
float2 envMapUV(float3 dir) {
    GGlobals G = globals();
    float c = clamp(dot(dir, G.up.xyz), -1.0, 1.0);
    float u = (atan2(dot(dir, G.envE2.xyz), dot(dir, G.envE1.xyz)) - G.envParams.x) / (2 * kPi);
    return float2(u - floor(u), acos(c) / kPi);
}
float envMapRadiance(float3 dir, float l) {
    GGlobals G = globals();
    float2 uv = envMapUV(dir);
    float3 c = imageBilinear(G.envMap.x, uv.x, uv.y);
    float3 w = max(float3(dot(G.rgbA[0].xyz, c), dot(G.rgbA[1].xyz, c), dot(G.rgbA[2].xyz, c)), 0.0);
    return G.envParams.y * G.envParams.w * dot(w, rgbBasis(l)) * planck(l, 6504.0) * 1e-13;
}
// Binary search: the last index i in [0, n) with cdf[i] <= a (cdf at scalars[at..at+n]).
uint cdfIndex(uint at, uint n, float a) {
    uint lo = 0, len = n;
    while (len > 0) {
        uint half = len / 2;
        if (g_scalars[at + lo + half] <= a) { lo += half + 1; len -= half + 1; }
        else len = half;
    }
    return lo == 0 ? 0 : lo - 1;
}
bool sampleEnvMap(float u1, float u2, out float3 dir, out float pdf) {
    GGlobals G = globals();
    dir = float3(0, 0, 1);
    pdf = 0;
    const float total = G.envParams.z;
    const uint W = uint(G.envE1.w), H = uint(G.envE2.w);
    if (!(total > 0)) return false;
    const float a = u1 * total;
    const uint j = min(cdfIndex(G.envMap.y, H + 1, a), H - 1);
    const float m0 = g_scalars[G.envMap.y + j], m1 = g_scalars[G.envMap.y + j + 1];
    const float tRow = clamp((a - m0) / max(m1 - m0, 1e-30), 0.0, 1.0);
    const uint row = G.envMap.z + j * (W + 1);
    const float b = u2 * g_scalars[G.envMore.x + j];
    const uint i = min(cdfIndex(row, W + 1, b), W - 1);
    const float c0 = g_scalars[row + i], c1 = g_scalars[row + i + 1];
    const float tCol = clamp((b - c0) / max(c1 - c0, 1e-30), 0.0, 1.0);
    const float u = (float(i) + tCol) / float(W), v = (float(j) + tRow) / float(H);
    const float theta = kPi * v, phi = 2 * kPi * u + G.envParams.x;
    const float st = sin(theta);
    if (!(st > 0)) return false;
    dir = (G.envE1.xyz * cos(phi) + G.envE2.xyz * sin(phi)) * st + G.up.xyz * cos(theta);
    pdf = (c1 - c0) / total * float(W) * float(H) / (2 * kPi * kPi * st);
    return pdf > 0;
}
float envMapPdf(float3 dir) {
    GGlobals G = globals();
    const float total = G.envParams.z;
    if (!(total > 0)) return 0;
    const uint W = uint(G.envE1.w), H = uint(G.envE2.w);
    float c = clamp(dot(dir, G.up.xyz), -1.0, 1.0);
    float st = sqrt(max(0.0, 1 - c * c));
    if (!(st > 0)) return 0;
    float2 uv = envMapUV(dir);
    uint i = min(uint(uv.x * float(W)), W - 1), j = min(uint(uv.y * float(H)), H - 1);
    uint row = G.envMap.z + j * (W + 1);
    return (g_scalars[row + i + 1] - g_scalars[row + i]) / total * float(W) * float(H) / (2 * kPi * kPi * st);
}

float envSky(float3 dir, float l) {
    GGlobals G = globals();
    switch (G.sky.x) {
        case 1: return spectrumEval(G.sky.y, l);
        case 2: {
            float c = dot(dir, G.up.xyz);
            if (c < 0) return spectrumEval(G.sky.w, l);
            float t = pow(c, 0.45);
            return spectrumEval(G.sky.z, l) * (1 - t) + spectrumEval(G.sky.y, l) * t;
        }
        case 3: return envMapRadiance(dir, l);
        default: return 0;
    }
}
float envSun(float3 dir, float l) {
    GGlobals G = globals();
    return (G.sun.x != 0 && dot(dir, G.sunDir.xyz) >= G.sunDir.w) ? spectrumEval(G.sun.y, l) : 0;
}

// ---------------------------------------------------------------- interfaces at hits

struct HitInterface {
    Interface it;
    uint front, back;
};

HitInterface makeInterface(SurfaceHit hit, float lambda) {
    GBoundary b = g_boundaries[hit.boundary];
    HitInterface h;
    h.it.p = hit.p;
    h.front = b.links.x;
    h.back = b.links.y;
    h.it.optics = loadOptics(b.links.z);
    h.it.n = hit.n;
    h.it.ns = hit.ns;
    h.it.nFront = regionIndex(h.front, lambda);
    h.it.nBack = regionIndex(h.back, lambda);
    h.it.lambda = lambda;
    h.it.fresnelFloor = globals().misc.z;
    h.it.tex = float3(1, 0, 0);
    if (h.it.optics.texKind != kTexNone)
        h.it.tex = textureWeights(h.it.optics.texKind, h.it.optics.texScale, h.it.optics.texParam, hit.pLocal, hit.nLocal,
                                  hit.uv, hit.hasUv);
    h.it.applyMaterialMap();
    if (h.it.optics.relief.y > 0) {
        // The relief's slope tilts the shading normal: n − (∇h − (∇h·n) n), in the tangent plane.
        float3 g = xfVector(b.toWorld, reliefGradient(hit.pLocal, h.it.optics.relief));
        float3 ns = h.it.ns;
        float3 tilted = normalize(ns - (g - dot(g, ns) * ns));
        if (dot(tilted, hit.n) > 1e-3) h.it.ns = tilted;
    }
    return h;
}
// The same interface at another wavelength: only the indices change.
Interface atWavelength(HitInterface base, float lambda) {
    Interface it = base.it;
    if (it.optics.usesIndices()) {
        it.nFront = regionIndex(base.front, lambda);
        it.nBack = regionIndex(base.back, lambda);
    }
    it.lambda = lambda;
    return it;
}

// ---------------------------------------------------------------- lights

struct LightSample {
    float3 wi;
    float dist;
    float4 Le;
    float pdf;  // solid angle, including light selection
};

uint pickLight(float u) {
    uint n = globals().counts.y;
    uint lo = 0, len = n;  // first cdf > u (std::upper_bound)
    while (len > 0) {
        uint half = len / 2;
        if (g_lights[lo + half].cdf <= u) { lo += half + 1; len -= half + 1; }
        else len = half;
    }
    return min(lo, n - 1);
}

bool sampleLightFrom(float3 p, Wavelengths wl, inout Rng rng, out LightSample ls) {
    GGlobals G = globals();
    ls = (LightSample)0;
    if (G.counts.y == 0) return false;
    uint li = pickLight(rng.uniform());
    GLight L = g_lights[li];
    float sel = L.pdf;
    float u1 = rng.uniform();
    float u2 = rng.uniform();
    float u3 = rng.uniform();
    if (L.kind == 1) {
        Frame f = Frame(G.sunDir.xyz);
        ls.wi = f.toWorld(sampleUniformCone(u1, u2, G.sunDir.w));
        ls.dist = infinity();
        ls.Le = wl.eval(G.sun.y);
        ls.pdf = sel / G.misc.x;
        return true;
    }
    if (L.kind == 2) {
        float pdfDir;
        if (!sampleEnvMap(u1, u2, ls.wi, pdfDir)) return false;
        ls.dist = infinity();
        [ForceUnroll] for (int i = 0; i < NW; ++i) ls.Le[i] = envSky(ls.wi, wl.lambda[i]);
        ls.pdf = sel * pdfDir;
        return true;
    }
    GBoundary b = g_boundaries[L.boundary];
    float3 pL, nL;
    float pdfW;
    if (!sampleFrom(b, xfPoint(b.toLocal, p), u1, u2, u3, pL, nL, pdfW)) return false;
    float3 pW = xfPoint(b.toWorld, pL);
    float3 nW = normalize(xfVector(b.toWorld, nL));
    float3 d = pW - p;
    ls.dist = length(d);
    if (!(ls.dist > 0)) return false;
    ls.wi = d / ls.dist;
    GEmission e = g_emissions[b.links.w];
    float cosL = dot(nW, -ls.wi);
    ls.Le = ((cosL > 0 && e.front != 0) || (cosL < 0 && e.back != 0)) ? wl.eval(e.radiance) : float4(0);
    ls.pdf = sel * pdfW;
    return true;
}

float lightPdf(float3 ref, SurfaceHit hit) {
    GBoundary b = g_boundaries[hit.boundary];
    uint li = b.mesh.x;
    if (li == kGpuNone) return 0;
    return g_lights[li].pdf * pdfFrom(b, xfPoint(b.toLocal, ref), hit.pLocal, hit.nLocal);
}

float sunPdf() {
    GGlobals G = globals();
    if (G.sun.x == 0 || G.counts.y == 0) return 0;
    return g_lights[G.counts.y - 1].pdf / G.misc.x;
}

float4 sigmaT(uint medium, Wavelengths wl) {
    GMedium m = g_media[medium];
    float4 r;
    [ForceUnroll] for (int i = 0; i < NW; ++i) r[i] = spectrumEval(m.props.x, wl.lambda[i]) + spectrumEval(m.props.y, wl.lambda[i]);
    return r;
}

// Transmittance along a straight segment through null boundaries; 0 if blocked. finalRegion receives
// the region the segment ends in.
float4 transmittance(float3 o, uint region, float3 dir, float dist, Wavelengths wl, float scale, out uint finalRegion) {
    finalRegion = region;
    float4 Tr = 1;
    float remaining = dist;
    // The segment stops short of its end point (on the light), by more than the float error of the
    // light point and of the ray's hit on the light. `dist` is measured from `o` itself.
    float margin = isinf(dist) ? 0 : 1e-5 * dist + surfaceEpsilon(o, scale);
    // Any matter on the segment blocks it: find that without ordering. Only when the segment crosses
    // a null boundary do regions (and media) change along it, which needs the ordered walk below.
    {
        float tmax = isinf(dist) ? infinity() : dist - margin;
        if (tmax <= 0) return Tr;
        bool sawNull;
        if (occluded(o, dir, tmax, sawNull)) return 0;
        if (!sawNull) {
            uint m = mediumOf(region);
            if (mediumAbsorbs(m) || mediumScatters(m)) {
                if (isinf(dist)) return 0;
                Tr = exp(-sigmaT(m, wl) * dist);
            }
            if (isinf(dist) && region != globals().counts.z) return 0;
            return Tr;
        }
    }
    for (int guard = 0; guard < 256; ++guard) {
        SurfaceHit h;
        float tmax = isinf(remaining) ? infinity() : remaining - margin;
        if (tmax <= 0) break;
        bool found = intersectWorld(o, dir, tmax, h);
        float seg = found ? h.t : remaining;
        uint m = mediumOf(region);
        if (mediumAbsorbs(m) || mediumScatters(m)) {
            if (isinf(seg)) return 0;
            float4 sT = sigmaT(m, wl);
            Tr *= exp(-sT * seg);
        }
        if (!found) {
            if (isinf(dist) && region != globals().counts.z) return 0;
            finalRegion = region;
            return Tr;
        }
        GBoundary b = g_boundaries[h.boundary];
        if (g_optics[b.links.z].a.x != kSurfNull) return 0;
        region = dot(dir, h.n) < 0 ? b.links.y : b.links.x;
        o = offsetOrigin(h.p, h.n, dir, h.scale);
        scale = h.scale;
        if (!isinf(remaining)) remaining -= h.t;
    }
    finalRegion = region;
    return Tr;
}
float4 transmittance(float3 o, uint region, float3 dir, float dist, Wavelengths wl, float scale) {
    uint finalRegion;
    return transmittance(o, region, dir, dist, wl, scale, finalRegion);
}

// A next-event estimate waiting for its shadow ray: value (the path throughput already applied) is
// added when the segment o + t·d, t < dist, is clear. Tracing it after the scatter sample has been
// drawn keeps the surface interface out of the live state during the shadow traversal (register
// pressure); the shadow ray draws no random numbers, so the streams are unchanged.
struct ShadowRay {
    float3 o, d;
    float dist, scale;
    uint region;
    float4 value;
};

ShadowRay prepareDirect(SurfaceHit hit, HitInterface base, float3 d, Wavelengths wl, inout Rng rng, uint region,
                        float4 beta) {
    ShadowRay sr = (ShadowRay)0;
    LightSample ls;
    if (!sampleLightFrom(hit.p, wl, rng, ls) || isZero(ls.Le) || ls.pdf <= 0) return sr;
    float4 f = 0;
    float pdfScatter = 0;
    [ForceUnroll] for (int i = 0; i < NW; ++i) {
        if (wl.pdf[i] == 0) continue;
        Interface it = i == 0 ? base.it : atWavelength(base, wl.lambda[i]);
        float pdf;
        f[i] = evalScatter(it, d, ls.wi, kModeRadiance, pdf);
        if (i == 0) pdfScatter = pdf;
    }
    if (isZero(f)) return sr;
    sr.region = dot(ls.wi, hit.n) > 0 ? base.front : base.back;
    sr.o = offsetOrigin(hit.p, hit.n, ls.wi, hit.scale);
    sr.d = ls.wi;
    sr.scale = hit.scale;
    // The origin moved along the ray (up to 20 surface epsilons at grazing angles): measure the
    // distance to the light from there, or the margin in transmittance() may not cover it.
    sr.dist = isinf(ls.dist) ? ls.dist : ls.dist - dot(sr.o - hit.p, ls.wi);
    float w = powerHeuristic(ls.pdf, pdfScatter);
    sr.value = beta * f * ls.Le * (scatterCosine(base.it, -d, ls.wi, kModeRadiance) * w / ls.pdf);
    return sr;
}

float4 traceShadow(ShadowRay sr, Wavelengths wl) {
    if (isZero(sr.value)) return 0;
    return sr.value * transmittance(sr.o, sr.region, sr.d, sr.dist, wl, sr.scale);
}

// Zero the throughput of wavelengths whose pdf was cleared.
float4 maskTerminated(Wavelengths wl, float4 beta) {
    [ForceUnroll] for (int i = 1; i < NW; ++i)
        if (wl.pdf[i] == 0) beta[i] = 0;
    return beta;
}

// A camera path in flight: the state Tracer::radiance keeps across its loop, advanced one segment
// at a time by stepPath (one traversal site, one shadow-ray site: a smaller kernel).
struct PathState {
    float3 o, d;
    float scale;       // error scale of the frame o was computed in (ray offsets)
    uint region;
    int depth;
    float4 L, beta;    // radiance gathered so far; throughput
    Wavelengths wl;    // secondary wavelengths may be terminated on the way
    float weight;      // the detector's importance-to-pdf ratio for this sample
    bool specular;
    float prevPdf;
    float3 prevP;
    // Hybrid: paths eye → D → S⁺ → light belong to light tracing (the partition is disjoint).
    int scatterEvents, nonDeltaEvents, deltaSinceNonDelta;
    bool firstIsNonDelta;
#ifdef OWE_SPPM
    // SPPM camera paths (sppm.slang): the first non-specular surface reached through specular events
    // only is the visible point. The path then takes next-event estimation and one scatter sample
    // there, and ends after that sample's segment (its emission, MIS-weighted): everything beyond is
    // the photons'. A path that scatters in a medium first has no visible point and is path traced.
    bool vpDone;
    bool chromatic;    // an absorbing medium on the way (the throughput depends on λ)
    float pathLength;  // unfolded length from the eye to the visible point
    float4 vpP;        // position; w: boundary (bits)
    float4 vpN;        // geometric normal; w: flags (bits)
    float4 vpD;        // arriving direction; w: unused
    float4 vpBeta;     // throughput arriving (with the detector's weight)
    float4 vpTex;      // texture weights; w: the shading normal (octahedral bits) where flag 8
#endif
};

PathState startPath(float3 o, float3 d, uint region, Wavelengths wl, float weight) {
    PathState p = (PathState)0;
    p.o = o;
    p.d = d;
    p.region = region;
    p.wl = wl;
    p.weight = weight;
    p.beta = maskTerminated(wl, float4(1));
    p.specular = true;
    p.prevP = o;
    return p;
}

void noteScatter(inout PathState p, bool delta) {
    if (delta) ++p.deltaSinceNonDelta;
    else {
        if (p.scatterEvents == 0) p.firstIsNonDelta = true;
        ++p.nonDeltaEvents;
        p.deltaSinceNonDelta = 0;
    }
    ++p.scatterEvents;
}

// Next-event estimation at a medium scattering point (phase anisotropy g), as a pending shadow ray.
ShadowRay prepareDirectMedium(float3 x, float3 d, float g, Wavelengths wl, inout Rng rng, uint region, float4 beta,
                              float scale) {
    ShadowRay sr = (ShadowRay)0;
    LightSample ls;
    if (!sampleLightFrom(x, wl, rng, ls) || isZero(ls.Le) || ls.pdf <= 0) return sr;
    float ph = hgPhase(dot(-d, ls.wi), g);
    sr.o = x;
    sr.d = ls.wi;
    sr.dist = ls.dist;
    sr.scale = scale;
    sr.region = region;
    sr.value = beta * ph * ls.Le * (powerHeuristic(ls.pdf, ph) / ls.pdf);
    return sr;
}

// Advances a camera path by one segment: one iteration of Tracer::radiance's loop, with the same
// random-number order. False when the path has ended (p.L then holds its radiance).
// stats: paths, segments, inconsistencies, leaks.
bool stepPath(inout PathState p, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    const uint ambient = G.counts.z;
    const bool partition = (pc.flags & kFlagPartition) != 0;
    const float hero = p.wl.hero();
    const int depth = p.depth++;

    SurfaceHit hit;
    bool found = intersectWorld(p.o, p.d, infinity(), hit);
    stats.y++;
    uint med = mediumOf(p.region);
    float seg = found ? hit.t : infinity();

    if (mediumScatters(med)) {
        // Distance sampling with the hero's extinction; secondaries are reweighted.
        float4 sT = sigmaT(med, p.wl);
        float t = -log(1 - rng.uniform()) / sT.x;
        if (t < seg) {
#ifdef OWE_SPPM
            if (p.vpDone) return false;  // light scattered in a medium on the way belongs to the photons
#endif
            float3 x = p.o + p.d * t;
            float pdf0 = sT.x * exp(-sT.x * t);
            [ForceUnroll] for (int i = 0; i < NW; ++i)
                p.beta[i] *= spectrumEval(g_media[med].props.y, p.wl.lambda[i]) * exp(-sT[i] * t) / pdf0;
            if (depth >= maxDepth) return false;
            float g = g_media[med].g.x;
            ShadowRay shadow = prepareDirectMedium(x, p.d, g, p.wl, rng, p.region, p.beta, p.scale);
            float u2 = rng.uniform();  // Draw2 order: last argument first
            float u1 = rng.uniform();
            float3 wi;
            float ph = sampleHg(-p.d, g, u1, u2, wi);
            p.prevPdf = ph;
            p.specular = false;
            p.prevP = x;
            noteScatter(p, false);
            p.o = x;
            p.d = wi;
            bool alive = true;
            if (depth >= rrDepth) {
                float q = min(1.0, maxValue(p.beta));
                if (rng.uniform() >= q) alive = false;
                else p.beta *= 1 / q;
            }
            p.L += traceShadow(shadow, p.wl);
            return alive;
        }
        if (isinf(seg)) return false;
        float pass0 = exp(-sT.x * seg);
        p.beta *= exp(-sT * seg) / pass0;
    } else if (mediumAbsorbs(med)) {
        if (!found) return false;
        [ForceUnroll] for (int i = 0; i < NW; ++i) p.beta[i] *= exp(-spectrumEval(g_media[med].props.x, p.wl.lambda[i]) * seg);
#ifdef OWE_SPPM
        p.chromatic = true;
#endif
    }

    const bool ownedByLightTracing =
        partition && p.nonDeltaEvents == 1 && p.firstIsNonDelta && p.deltaSinceNonDelta >= 1;
    if (!found) {
        if (p.region == ambient && !ownedByLightTracing) {
            float4 sky, sun;
            [ForceUnroll] for (int i = 0; i < NW; ++i) {
                sky[i] = envSky(p.d, p.wl.lambda[i]);
                sun[i] = envSun(p.d, p.wl.lambda[i]);
            }
            float w = (p.specular || isZero(sun)) ? 1.0 : powerHeuristic(p.prevPdf, sunPdf());
            // An environment map is also a light for next-event estimation: MIS.
            float wSky = 1;
            if (G.envMap.w != kGpuNone && !p.specular) wSky = powerHeuristic(p.prevPdf, g_lights[G.envMap.w].pdf * envMapPdf(p.d));
            p.L += p.beta * (sky * wSky + sun * w);
        } else if (p.region != ambient) {
            stats.w++;
        }
        return false;
    }

    GBoundary b = g_boundaries[hit.boundary];
    bool fromFront = dot(p.d, hit.n) < 0;
    uint inc = fromFront ? b.links.x : b.links.y;
    uint other = fromFront ? b.links.y : b.links.x;
    if (inc != p.region) {
        stats.z++;
        noteInconsistency(hit.boundary, p.region, inc, kGpuNone, hit.p);
        p.region = inc;
    }
    uint surfType = g_optics[b.links.z].a.x;
#ifdef OWE_SPPM
    if (!p.vpDone) p.pathLength += hit.t;
#endif

    // Emission.
    if (b.links.w != kGpuNone) {
        GEmission e = g_emissions[b.links.w];
        if (((fromFront && e.front != 0) || (!fromFront && e.back != 0)) && !ownedByLightTracing) {
            float w = 1;
            if (!p.specular) {
                float pl = lightPdf(p.prevP, hit);
                if (pl > 0) w = powerHeuristic(p.prevPdf, pl);
            }
            p.L += p.beta * p.wl.eval(e.radiance) * w;
        }
    }

    if (surfType == kSurfNull) {
        p.region = other;
        p.o = offsetOrigin(hit.p, hit.n, p.d, hit.scale);
        p.scale = hit.scale;
        return depth < 4 * maxDepth;
    }
#ifdef OWE_SPPM
    if (p.vpDone) return false;  // the visible point's scatter sample has met matter: done
#endif
    if (surfType == kSurfAbsorber || surfType == kSurfDetector) return false;
    if (depth >= maxDepth) return false;

    HitInterface hi = makeInterface(hit, hero);
    if (surfType == kSurfDielectric && !p.wl.secondaryTerminated()) {
        [ForceUnroll] for (int i = 1; i < NW; ++i)
            if (p.wl.pdf[i] != 0 && isDispersive(hi.it, atWavelength(hi, p.wl.lambda[i]))) {
                p.wl.terminateSecondary();
                p.beta = maskTerminated(p.wl, p.beta);
                break;
            }
    }
#ifdef OWE_SPPM
    if (!hi.it.optics.isDelta() && p.nonDeltaEvents == 0) {
        p.vpDone = true;
        p.vpP = float4(hit.p, asfloat(hit.boundary));
        // Flags: valid; secondaries terminated; spectrally free (the eye sees the point directly, so
        // its throughput is the same at every wavelength and it can take a photon of any λ); smooth
        // (a shading normal, packed in vpTex.w).
        const bool free = p.scatterEvents == 0 && !p.chromatic;
        const bool smooth = hi.it.smooth();
        p.vpN = float4(hit.n, asfloat(1u | (p.wl.secondaryTerminated() ? 2u : 0u) | (free ? 4u : 0u) | (smooth ? 8u : 0u)));
        p.vpD = float4(p.d, 0);
        p.vpBeta = p.beta * p.weight;
        p.vpTex = float4(hi.it.tex, asfloat(smooth ? octEncode(hi.it.ns) : 0u));
    }
#endif
    ShadowRay shadow = (ShadowRay)0;
    if (!hi.it.optics.isDelta()) shadow = prepareDirect(hit, hi, p.d, p.wl, rng, p.region, p.beta);
    float u3 = rng.uniform();  // Draw3 order: last argument first
    float u2 = rng.uniform();
    float u1 = rng.uniform();
    ScatterSample s;
    bool alive = sampleScatter(hi.it, p.d, u1, u2, u3, kModeRadiance, s) && s.weight > 0;
    if (alive) {
        p.beta.x *= s.weight;
        [ForceUnroll] for (int i = 1; i < NW; ++i)
            if (p.wl.pdf[i] != 0) p.beta[i] *= secondaryScatterWeight(atWavelength(hi, p.wl.lambda[i]), p.d, s, kModeRadiance);
        if (s.transmitted) p.region = other;
        p.specular = s.delta;
        p.prevPdf = s.pdf;
        p.prevP = hit.p;
        noteScatter(p, s.delta);
        p.o = offsetOrigin(hit.p, hit.n, s.wi, hit.scale);
        p.scale = hit.scale;
        p.d = s.wi;
        if (depth >= rrDepth) {
            float q = min(1.0, maxValue(p.beta));
            if (rng.uniform() >= q) alive = false;
            else p.beta *= 1 / q;
        }
    }
    p.L += traceShadow(shadow, p.wl);  // the interface is no longer needed
    return alive;
}
