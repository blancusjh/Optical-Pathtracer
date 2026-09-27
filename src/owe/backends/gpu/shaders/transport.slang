// The transport engine for camera paths: float port of Tracer::radiance (transport.cpp) with the
// same structure, region bookkeeping, next-event estimation, MIS and random-number order.
#pragma once
#include "optics.slang"
#include "world.slang"

// ---------------------------------------------------------------- environment

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
    h.it.nFront = regionIndex(h.front, lambda);
    h.it.nBack = regionIndex(h.back, lambda);
    h.it.lambda = lambda;
    h.it.fresnelFloor = globals().misc.z;
    h.it.tex = float3(1, 0, 0);
    if (h.it.optics.texKind != kTexNone)
        h.it.tex = textureWeights(h.it.optics.texKind, h.it.optics.texScale, h.it.optics.texParam, hit.pLocal, hit.nLocal);
    return h;
}
// The same interface at another wavelength: only the indices change.
Interface atWavelength(HitInterface base, float lambda) {
    Interface it = base.it;
    it.nFront = regionIndex(base.front, lambda);
    it.nBack = regionIndex(base.back, lambda);
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
    sr.value = beta * f * ls.Le * (abs(dot(ls.wi, hit.n)) * w / ls.pdf);
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
            p.L += p.beta * (sky + sun * w);
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
        p.region = inc;
    }
    uint surfType = g_optics[b.links.z].a.x;

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
