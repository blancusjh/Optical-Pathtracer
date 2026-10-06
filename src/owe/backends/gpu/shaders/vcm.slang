// Bidirectional path tracing and vertex connection and merging (Georgiev, Křivánek, Davidovič &
// Slusallek 2012), the `bdpt` and `vcm` integrators, with a light vertex cache (Davidovič et al.
// 2014) in place of per-pixel light subpaths.
//
// One iteration:
//  1. light pass (vcm_light_kernel): N_L light subpaths (one per pixel) leave the lights as photons
//     do. Every non-specular vertex is stored in the cache with its throughput and the partial MIS
//     quantities dVCM, dVC, dVM of Georgiev's recursive formulation, and is connected to the eye
//     (light tracing).
//  2. (vcm) the cached vertices enter the multi-level hash of sppm.slang, as photons for merging.
//  3. camera pass (vcm_camera_kernel): one camera path per pixel; at every non-specular vertex it
//     takes next-event estimation, a connection to one light vertex drawn uniformly from the cache
//     (weighted by the cache's size over N_L: the expected sum over one light subpath's vertices), and
//     (vcm) merges the cached vertices within the merging radius. Emission it meets is weighted too.
// Every strategy is weighted by the balance heuristic over all the others (bdpt: the merging
// strategies are absent), so each light path is counted once in total. The merging radius at a point
// is a few pixel footprints at its distance from the eye, shrinking over iterations as SPPM's; the MIS
// factor of merging there, η = π r² N_L, is a function of position both path ends evaluate alike.
//
// Spectra: an iteration's light and camera paths are split into G wavelength groups (as SPPM's):
// light path j is in group j mod G, pixel p in a group drawn afresh each iteration, and a group's
// paths share four wavelengths interleaved with the other groups' (4G per iteration). A camera path
// connects to and merges with its own group's light vertices only, and light tracing splats only to
// its group's pixels (scaled by G): each group is a whole VCM estimator with N_L/G light paths, which
// is the number every MIS weight counts. A dispersive event on either side leaves the hero alone. Pdfs are converted between solid angle and area with the
// geometric normal throughout; a shading normal enters only the contributions (optics.slang).
// Participating media that scatter are not supported (the renderer falls back to path tracing).
#pragma once
#include "sppm.slang"

struct LightVertex {
    float4 p;     // camera-relative position; w: boundary (bits)
    float4 n;     // geometric normal (toward the front region); w: shading normal (octahedral bits) if smooth
    float4 d;     // direction the light arrived along; w: flags (bits)
    float4 beta;  // light subpath throughput arriving (flux per path)
    float4 tex;   // texture weights; w: dVCM
    float4 mis;   // x: dVC, y: dVM, z: the light path's guide cell (bits; kGpuNone: none), w: its group (bits)
};
static const uint kLvTerminated = 1u, kLvSmooth = 2u;

[[vk::binding(25, 0)]] RWStructuredBuffer<LightVertex> g_lightVertices;
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> g_vcmCounters;  // [0] vertices offered this iteration, [1] dropped this pass (cache full), [2] counting pass: paths with vertices, [4 + g] group g's vertices offered

// Push constants (pc.extra): x wavelength rotation (float), y hash mask, z merging radius per metre of
// distance from the eye (float; 0: bdpt, no merging), w smallest merging radius (float).
// pc.pad0: the cache's capacity per group; pc.pad1: N_L/G, the light subpaths of a group;
// pc.flags bits 24–31: G, the wavelength groups.
bool merging() { return asfloat(pc.extra.z) > 0; }
uint cacheCapacity() { return pc.pad0; }
float lightPaths() { return float(pc.pad1); }
uint vcmGroups() { return max(pc.flags >> 24, 1u); }
Wavelengths groupWavelengths(uint g) {
    float u = iterationRotation() + float(g) / float(4 * vcmGroups());
    return Wavelengths.sample(u - floor(u));
}
uint pixelGroup(uint pixel) { return uint(mixBits(pack64(pixel, pc.pass) ^ 0x2545f4914f6cdd1dULL) % uint64_t(vcmGroups())); }
// Group g's cache: vertices [g·capacity, g·capacity + count), its count at g_vcmCounters[4 + g].
uint groupCount(uint g) { return min(g_vcmCounters[4 + g], cacheCapacity()); }
float mergeRadius(float3 p) { return max(asfloat(pc.extra.z) * length(p), asfloat(pc.extra.w)); }
// η = π r² N_L: merging's weight relative to connections at p; its inverse. Both 0 without merging.
float etaAt(float3 p) {
    if (!merging()) return 0;
    float r = mergeRadius(p);
    return kPi * r * r * lightPaths();
}
float invEtaAt(float3 p) {
    float e = etaAt(p);
    return e > 0 ? 1 / e : 0.0;
}

float mis(float x) { return x; }

// Debugging: strategies left out (pc.flags bits 8–15, OWE_VCM_MASK): 1 emission met after a vertex,
// 2 next-event estimation, 4 connections, 8 merging, 16 light tracing, 32 emission met after a
// non-specular vertex.
bool skip(uint bit) { return ((pc.flags >> 8) & bit) != 0; }
// A counting pass (pc.flags bit 16): light subpaths only count their vertices, and how many of them
// had any (g_vcmCounters[2]), which sizes the cache.
bool counting() { return (pc.flags & (1u << 16)) != 0; }  // balance heuristic (the power heuristic overflows float at astronomical distances)

// The interface at a cached light vertex, at wavelength lambda.
Interface lightVertexInterface(LightVertex v, float lambda) {
    GBoundary b = g_boundaries[asuint(v.p.w)];
    Interface it;
    it.optics = loadOptics(b.links.z);
    it.n = v.n.xyz;
    it.ns = (asuint(v.d.w) & kLvSmooth) != 0 ? octDecode(asuint(v.n.w)) : it.n;
    it.p = v.p.xyz;
    it.nFront = it.nBack = 1;
    if (it.optics.usesIndices()) {
        it.nFront = regionIndex(b.links.x, lambda);
        it.nBack = regionIndex(b.links.y, lambda);
    }
    it.lambda = lambda;
    it.fresnelFloor = globals().misc.z;
    it.tex = v.tex.xyz;
    it.applyMaterialMap();
    return it;
}

// ---------------------------------------------------------------- lights

// The emission side of a light for MIS: the pdf of emitting a particle from its point in its
// direction (with the choice of the light among the emitters), and the next-event pdf.
struct EmissionPdfs {
    float emissionPdfW;  // pdf of position × direction, including the choice of the light
    float cosAtLight;    // at the emitter (1 at infinity)
};

float emitterRadius() { return globals().misc.w * 1.001; }  // the disc distant light leaves from

// The emission pdf of an area light at a point with normal-cosine cosL toward the path.
float areaEmissionPdf(GBoundary b, uint li, float cosL) {
    GGlobals G = globals();
    GEmission em = g_emissions[b.links.w];
    float sidePdf = (em.front != 0 && em.back != 0) ? 0.5 : 1.0;
    return (1 - G.emit.x) * g_lights[li].pdf / b.misc.y * sidePdf * abs(cosL) * kInvPi;
}
// Distant light arriving from direction `toward` (sky and sun parts): emission pdfs and NEE pdfs.
float skyEmissionPdf(float3 toward) {
    GGlobals G = globals();
    const uint envLight = G.envMap.w;
    const float pSkyPhoton = G.emit.x + (envLight != kGpuNone ? (1 - G.emit.x) * g_lights[envLight].pdf : 0.0);
    float pdfDir = G.sky.x == 3 ? envMapPdf(toward) : 1 / (4 * kPi);
    float R = emitterRadius();
    return pSkyPhoton * pdfDir / (kPi * R * R);
}
float skyDirectPdf(float3 toward) {
    GGlobals G = globals();
    return G.envMap.w != kGpuNone ? g_lights[G.envMap.w].pdf * envMapPdf(toward) : 0.0;
}
// Sun light paths are guided as SPPM's photons (particle.slang): a share drawn from a map of where
// earlier ones proved useful over the emission disc. The disc's density at q (its plane's coordinates
// about the centre), mixture and uniform share together; its cell for learning (kGpuNone outside).
float sunDiscPdf(float2 q, float R, out uint cell) {
    const float size = 2 * R / float(kGuideRes);
    int2 ci = int2(floor((q + R) / size));
    const bool inGrid = all(ci >= 0) && all(ci < int(kGuideRes));
    cell = inGrid ? uint(ci.y) * kGuideRes + uint(ci.x) : kGpuNone;
    const float uniform = dot(q, q) <= R * R ? 1 / (kPi * R * R) : 0.0;
    if (!(guideTotal() > 0)) return uniform;
    return (inGrid ? kGuideShare * guideCellProbability(cell) / (size * size) : 0.0) + (1 - kGuideShare) * uniform;
}
// The emission pdf of sunlight arriving at x from direction `toward`: the sun's choice, its cone, and
// the disc's density where the ray back from x crosses it.
float sunEmissionPdf(float3 x, float3 toward) {
    GGlobals G = globals();
    if (G.sun.x == 0 || G.counts.y == 0) return 0;
    const float R = emitterRadius();
    const float3 dir = -toward;
    const float3 c0 = G.emit.yzw - dir * (1.01 * R);
    float3 a1, a2;
    orthonormalBasis(dir, a1, a2);
    const float3 p = x - dir * dot(x - c0, dir) - c0;
    uint cell;
    return (1 - G.emit.x) * g_lights[G.counts.y - 1].pdf / G.misc.x * sunDiscPdf(float2(dot(p, a1), dot(p, a2)), R, cell);
}
// A light path found useful: its uniform-share sun emission cell counts for the guide.
void noteUseful(uint cell) {
    if (cell != kGpuNone) InterlockedAdd(g_guide[cell], 1u);
}

// A light subpath's start: the particle (emitParticle's sampling, without guiding) and what MIS needs.
struct VcmEmission {
    float3 o, d;
    uint region;
    float4 beta;
    float scale;
    float emissionPdfW;
    float usedCos;       // cos at the emitter for an area light, 1 at infinity
    bool infinite;
    float directPdfW;    // infinite: next-event pdf of the direction (area lights: at the first hit)
    uint light;          // area light: its entry in g_lights (kGpuNone otherwise)
    float3 pL, nL;       // area light: the point and normal in the emitter's frame
    uint guideCell;      // a sun path of the uniform share: its cell of the guide map (else kGpuNone)
};

bool emitVcm(Wavelengths wl, inout Rng rng, out VcmEmission e) {
    GGlobals G = globals();
    const float pSky = G.emit.x;
    e = (VcmEmission)0;
    e.light = kGpuNone;
    e.guideCell = kGpuNone;
    bool fromSky = rng.uniform() < pSky;
    uint li = 0;
    if (!fromSky) li = pickLight(rng.uniform());
    GLight L = g_lights[li];
    const uint envLight = G.envMap.w;
    if (!fromSky && li == envLight) fromSky = true;
    if (fromSky || L.kind == 1) {
        float R = emitterRadius();
        float3 toward;
        float pdfDir;
        float4 Le;
        float u2 = rng.uniform();
        float u1 = rng.uniform();
        if (fromSky) {
            if (G.sky.x == 3) {
                if (!sampleEnvMap(u1, u2, toward, pdfDir)) return false;
            } else {
                toward = sampleUniformSphere(u1, u2);
                pdfDir = 1 / (4 * kPi);
            }
            [ForceUnroll] for (int i = 0; i < NW; ++i) Le[i] = envSky(toward, wl.lambda[i]);
            e.emissionPdfW = skyEmissionPdf(toward);
            e.directPdfW = skyDirectPdf(toward);
        } else {
            Frame f = Frame(G.sunDir.xyz);
            toward = f.toWorld(sampleUniformCone(u1, u2, G.sunDir.w));
            Le = wl.eval(G.sun.y);
            e.directPdfW = sunPdf();
        }
        float3 dir = -toward;
        float3 a1, a2;
        orthonormalBasis(dir, a1, a2);
        float v2 = rng.uniform();
        float v1 = rng.uniform();
        float2 q = sampleUniformDiskConcentric(v1, v2) * R;  // on the disc, about its centre
        if (!fromSky) {
            const bool guided = guideTotal() > 0 && rng.uniform() < kGuideShare;
            if (guided) {
                const float size = 2 * R / float(kGuideRes);
                uint c = guideSample(v1);
                q = float2(float(c % kGuideRes) + v2, float(c / kGuideRes) + rng.uniform()) * size - R;
            }
            uint cell;
            e.emissionPdfW = (1 - G.emit.x) * g_lights[li].pdf / G.misc.x * sunDiscPdf(q, R, cell);
            if (!guided) e.guideCell = cell;  // the uniform share teaches the map
        }
        e.o = G.emit.yzw - dir * (1.01 * R) + a1 * q.x + a2 * q.y;
        e.d = dir;
        e.region = G.counts.z;
        e.scale = maxAbsComponent(e.o);
        e.beta = e.emissionPdfW > 0 ? Le / e.emissionPdfW : float4(0);
        e.usedCos = 1;
        e.infinite = true;
    } else {
        GBoundary b = g_boundaries[L.boundary];
        GEmission em = g_emissions[b.links.w];
        float4 Le = wl.eval(em.radiance);
        float u3 = rng.uniform();
        float u2 = rng.uniform();
        float u1 = rng.uniform();
        float3 pL, nL;
        if (!sampleArea(b, u1, u2, u3, pL, nL)) return false;
        float3 p = xfPoint(b.toWorld, pL), n = normalize(xfVector(b.toWorld, nL));
        bool front = em.front != 0;
        if (em.front != 0 && em.back != 0) front = rng.uniform() < 0.5;
        float3 ns = front ? n : -n;
        float c2 = rng.uniform();
        float c1 = rng.uniform();
        float3 dir = Frame(ns).toWorld(sampleCosineHemisphere(c1, c2));
        float cosL = abs(dot(n, dir));
        e.emissionPdfW = areaEmissionPdf(b, li, cosL);
        if (!(e.emissionPdfW > 0)) return false;
        e.beta = Le * (cosL / e.emissionPdfW);
        e.o = offsetOrigin(p, n, dir, b.misc.x);
        e.scale = b.misc.x;
        e.d = dir;
        e.region = front ? b.links.x : b.links.y;
        e.usedCos = cosL;
        e.infinite = false;
        e.light = li;
        e.pL = pL;
        e.nL = nL;
    }
    e.beta = maskTerminated(wl, e.beta);
    return !isZero(e.beta);
}

// Next-event estimation with the emission pdfs MIS needs (sampleLightFrom plus the light's side).
struct VcmLightSample {
    float3 wi;
    float dist;
    float4 Le;
    float directPdfW;    // including the light's choice
    float emissionPdfW;
    float cosAtLight;
};

bool sampleLightVcm(float3 p, Wavelengths wl, inout Rng rng, out VcmLightSample ls) {
    GGlobals G = globals();
    ls = (VcmLightSample)0;
    if (G.counts.y == 0) return false;
    uint li = pickLight(rng.uniform());
    GLight L = g_lights[li];
    float sel = L.pdf;
    float u1 = rng.uniform();
    float u2 = rng.uniform();
    float u3 = rng.uniform();
    ls.cosAtLight = 1;
    if (L.kind == 1) {
        Frame f = Frame(G.sunDir.xyz);
        ls.wi = f.toWorld(sampleUniformCone(u1, u2, G.sunDir.w));
        ls.dist = infinity();
        ls.Le = wl.eval(G.sun.y);
        ls.directPdfW = sel / G.misc.x;
        ls.emissionPdfW = sunEmissionPdf(p, ls.wi);
        return true;
    }
    if (L.kind == 2) {
        float pdfDir;
        if (!sampleEnvMap(u1, u2, ls.wi, pdfDir)) return false;
        ls.dist = infinity();
        [ForceUnroll] for (int i = 0; i < NW; ++i) ls.Le[i] = envSky(ls.wi, wl.lambda[i]);
        ls.directPdfW = sel * pdfDir;
        ls.emissionPdfW = skyEmissionPdf(ls.wi);
        return true;
    }
    GBoundary b = g_boundaries[L.boundary];
    float3 pL, nL;
    float pdfW;
    if (!sampleFrom(b, xfPoint(b.toLocal, p), u1, u2, u3, pL, nL, pdfW)) return false;
    float3 pW = xfPoint(b.toWorld, pL);
    float3 nW = normalize(xfVector(b.toWorld, nL));
    float3 dv = pW - p;
    ls.dist = length(dv);
    if (!(ls.dist > 0)) return false;
    ls.wi = dv / ls.dist;
    GEmission e = g_emissions[b.links.w];
    float cosL = dot(nW, -ls.wi);
    ls.Le = ((cosL > 0 && e.front != 0) || (cosL < 0 && e.back != 0)) ? wl.eval(e.radiance) : float4(0);
    ls.directPdfW = sel * pdfW;
    ls.cosAtLight = abs(cosL);
    ls.emissionPdfW = areaEmissionPdf(b, li, cosL);
    return true;
}

// ---------------------------------------------------------------- light subpaths

void addVcmCounter(uint index) { InterlockedAdd(g_vcmCounters[index], 1u); }

// Traces one light subpath: caches its non-specular vertices and connects each to the eye.
void traceLightVcm(inout Wavelengths wl, uint group, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    const uint ambient = G.counts.z;
    const float NL = lightPaths();
    stats.x++;
    VcmEmission em;
    if (!emitVcm(wl, rng, em)) return;
    float3 o = em.o, d = em.d;
    uint region = em.region;
    float4 beta = em.beta;
    float scale = em.scale;
    const float hero = wl.hero();
    float dVCM = 0, dVC = mis(em.usedCos / em.emissionPdfW), dVM = 0;
    float segment = 0;  // distance since the last scattering vertex (null boundaries do not scatter)
    bool first = true, counted = false;
    uint lastBoundary = kGpuNone;

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = intersectWorld(o, d, infinity(), hit);
        if (found && depth == 0 && em.infinite) refineDistantHit(d, hit);
        stats.y++;
        uint med = mediumOf(region);
        // Media are extinction only here: paths that scatter in them are the path tracer's
        // (kFlagMediumOnly), so these walks carry exp(−σt·t), absorption and out-scattering.
        if (mediumAbsorbs(med) || mediumScatters(med)) {
            if (!found) return;
            beta *= exp(-sigmaT(med, wl) * hit.t);
        }
        if (!found) {
            if (region != ambient) stats.w++;
            return;
        }
        GBoundary b = g_boundaries[hit.boundary];
        bool fromFront = dot(d, hit.n) < 0;
        uint inc = fromFront ? b.links.x : b.links.y;
        uint other = fromFront ? b.links.y : b.links.x;
        if (inc != region) {
            stats.z++;
            noteInconsistency(hit.boundary, region, inc, lastBoundary, hit.p);
            region = inc;
        }
        lastBoundary = hit.boundary;
        segment += hit.t;
        uint surfType = g_optics[b.links.z].a.x;
        if (surfType == kSurfNull) {
            region = other;
            o = offsetOrigin(hit.p, hit.n, d, hit.scale);
            scale = hit.scale;
            if (depth >= 4 * maxDepth) return;
            continue;
        }
        if (surfType == kSurfStainedGlass && passStainedGlass(hit, d, wl, beta, rng)) {
            o = offsetOrigin(hit.p, hit.n, d, hit.scale);
            scale = hit.scale;
            if (depth >= 4 * maxDepth) return;
            continue;
        }
        if (surfType == kSurfDetector || surfType == kSurfAbsorber) return;
        // The MIS quantities arrive at this vertex (SmallVCM's update on hitting a surface).
        const float cosIn = abs(dot(hit.n, d));
        if (first) {
            if (em.infinite) {
                dVCM = mis(em.directPdfW / em.emissionPdfW);
            } else {
                // Next-event estimation would have chosen the emitter point from here: its pdf
                // (solid angle) times the emitter's cosine is the area pdf times the distance².
                GBoundary lb = g_boundaries[g_lights[em.light].boundary];
                float directPdfW = g_lights[em.light].pdf * pdfFrom(lb, xfPoint(lb.toLocal, hit.p), em.pL, em.nL);
                dVCM = mis(directPdfW * em.usedCos / em.emissionPdfW);
            }
            dVM = dVC * invEtaAt(hit.p);
            first = false;
        } else {
            dVCM *= mis(segment * segment);
        }
        segment = 0;
        dVCM /= mis(cosIn);
        dVC /= mis(cosIn);
        dVM /= mis(cosIn);

        HitInterface hi = makeInterface(hit, hero);
        if (!hi.it.optics.isDelta()) {
            // Into the cache.
            uint slot;
            InterlockedAdd(g_vcmCounters[0], 1u);
            InterlockedAdd(g_vcmCounters[4 + group], 1u, slot);
            if (counting()) {
                if (!counted) InterlockedAdd(g_vcmCounters[2], 1u);  // nothing stored, nothing splatted
                counted = true;
            } else if (slot < cacheCapacity()) {
                LightVertex v;
                const bool smooth = hi.it.smooth();
                v.p = float4(hit.p, asfloat(hit.boundary));
                v.n = float4(hit.n, asfloat(smooth ? octEncode(hi.it.ns) : 0u));
                v.d = float4(d, asfloat((wl.secondaryTerminated() ? kLvTerminated : 0u) | (smooth ? kLvSmooth : 0u)));
                v.beta = beta;
                v.tex = float4(hi.it.tex, dVCM);
                v.mis = float4(dVC, dVM, asfloat(em.guideCell), asfloat(group));
                g_lightVertices[group * cacheCapacity() + slot] = v;
            } else {
                addVcmCounter(1);
            }
            // To the eye (light tracing, t = 1).
            float3 q;
            int px, py;
            float factor;
            // Only to the pixels of this path's group this iteration (their camera paths share its
            // wavelengths), scaled by G: each pixel takes 1/G of the light paths.
            if (!counting() && connectObserver(hit.p, rng, q, px, py, factor) && !skip(16) &&
                pixelGroup(uint(py) * pc.width + uint(px)) == group) {
                float3 wv = q - hit.p;
                float dist = length(wv);
                float3 wc = wv / dist;
                float4 f = 0;
                float pdfFwd = 0;
                [ForceUnroll] for (int i = 0; i < NW; ++i) {
                    if (wl.pdf[i] == 0) continue;
                    float pdf;
                    f[i] = evalScatter(i == 0 ? hi.it : atWavelength(hi, wl.lambda[i]), d, wc, kModeImportance, pdf);
                }
                if (!isZero(f)) {
                    float pdfRev;
                    evalScatter(hi.it, -wc, -d, kModeRadiance, pdfRev);
                    const float cameraPdfA = factor * abs(dot(hit.n, wc));
                    const float wLight = mis(cameraPdfA / NL) * (etaAt(hit.p) + dVCM + dVC * mis(pdfRev));
                    const float w = 1 / (wLight + 1);
                    uint startRegion = dot(wc, hit.n) > 0 ? hi.front : hi.back, endRegion;
                    float3 origin = offsetOrigin(hit.p, hit.n, wc, hit.scale);
                    float4 Tr = transmittance(origin, startRegion, wc, dist - dot(origin - hit.p, wc), wl, hit.scale, endRegion);
                    if (!isZero(Tr) && endRegion == G.det.x) {
                        deposit(px, py, beta * f * Tr * (scatterCosine(hi.it, -d, wc, kModeImportance) * factor * w * float(vcmGroups())), wl);
                        noteUseful(em.guideCell);
                    }
                }
            }
        }
        if (depth >= maxDepth) return;
        if (surfType == kSurfDielectric && !wl.secondaryTerminated()) {
            [ForceUnroll] for (int i = 1; i < NW; ++i)
                if (wl.pdf[i] != 0 && isDispersive(hi.it, atWavelength(hi, wl.lambda[i]))) {
                    wl.terminateSecondary();
                    beta = maskTerminated(wl, beta);
                    break;
                }
        }
        float u3 = rng.uniform();
        float u2 = rng.uniform();
        float u1 = rng.uniform();
        ScatterSample s;
        if (!sampleScatter(hi.it, d, u1, u2, u3, kModeImportance, s) || s.weight <= 0) return;
        const float cosOut = abs(dot(hit.n, s.wi));
        if (s.delta) {
            dVCM = 0;
            dVC *= mis(cosOut);
            dVM *= mis(cosOut);
        } else {
            float pdfRev;
            evalScatter(hi.it, -s.wi, -d, kModeRadiance, pdfRev);
            const float eta = etaAt(hit.p);
            dVC = mis(cosOut / s.pdf) * (dVC * mis(pdfRev) + dVCM + eta);
            dVM = mis(cosOut / s.pdf) * (dVM * mis(pdfRev) + dVCM * (eta > 0 ? 1 / eta : 0.0) + 1);
            dVCM = mis(1 / s.pdf);
        }
        const float w0 = s.weight;
        beta.x *= s.weight;
        [ForceUnroll] for (int i = 1; i < NW; ++i)
            if (wl.pdf[i] != 0) beta[i] *= secondaryScatterWeight(atWavelength(hi, wl.lambda[i]), d, s, kModeImportance);
        if (s.transmitted) region = other;
        o = offsetOrigin(hit.p, hit.n, s.wi, hit.scale);
        scale = hit.scale;
        d = s.wi;
        if (depth >= rrDepth) {
            float q = min(1.0, w0);
            if (q < 1) {
                if (rng.uniform() >= q) return;
                beta *= 1 / q;
            }
        }
    }
}

// ---------------------------------------------------------------- camera subpaths

// XYZ of a contribution whose light side may have left only the hero wavelength.
float3 joinedXYZ(Wavelengths camera, bool lightTerminated, float4 value) {
    Wavelengths w = camera;
    if (lightTerminated) w.terminateSecondary();
    return w.toXYZ(maskTerminated(w, value), globals().misc.y);
}

// Traces one camera path from pixel (px, py); returns its contribution (XYZ, with the camera's weight).
float3 traceCameraVcm(float px, float py, Wavelengths wlIn, uint group, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    const uint ambient = G.counts.z;
    const float NL = lightPaths();
    Wavelengths wl = wlIn;
    float3 o, d;
    float weight;
    if (!generateRay(px, py, rng, o, d, weight) || weight <= 0) return 0;
    stats.x++;
    const float W = float(G.limits.z), H = float(G.limits.w);
    const float pixelArea = (2 * G.lens.x / W) * (2 * G.lens.y / H);
    const float cosCam = dot(d, G.fwd.xyz);
    if (!(cosCam > 0)) return 0;
    const float cameraPdfW = 1 / (pixelArea * cosCam * cosCam * cosCam);
    float dVCM = mis(NL / cameraPdfW), dVC = 0, dVM = 0;
    float4 beta = maskTerminated(wl, float4(weight));
    uint region = G.det.x;
    float3 L = 0;
    float segment = 0;
    float3 prevP = o;
    int pathLength = 0;
    bool nonDelta = false;  // debugging (strategy mask 32)
    const uint M = groupCount(group);

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = intersectWorld(o, d, infinity(), hit);
        stats.y++;
        uint med = mediumOf(region);
        if (mediumAbsorbs(med) || mediumScatters(med)) {  // extinction only (see traceLightVcm)
            if (!found) break;
            beta *= exp(-sigmaT(med, wl) * hit.t);
        }
        if (!found) {
            if (region == ambient) {
                // The sky and the sun at infinity: seen directly, or weighted against next-event
                // estimation and the light subpaths that leave them.
                float4 sky, sun;
                [ForceUnroll] for (int i = 0; i < NW; ++i) {
                    sky[i] = envSky(d, wl.lambda[i]);
                    sun[i] = envSun(d, wl.lambda[i]);
                }
                float wSky = 1, wSun = 1;
                if (pathLength > 0) {
                    wSky = 1 / (1 + mis(skyDirectPdf(d)) * dVCM + mis(skyEmissionPdf(d)) * dVC);
                    wSun = 1 / (1 + mis(sunPdf()) * dVCM + mis(sunEmissionPdf(prevP, d)) * dVC);
                    if (skip(1) || (skip(32) && nonDelta)) wSky = wSun = 0;
                }
                L += wl.toXYZ(beta * (sky * wSky + sun * wSun), G.misc.y);
            } else {
                stats.w++;
            }
            break;
        }
        GBoundary b = g_boundaries[hit.boundary];
        bool fromFront = dot(d, hit.n) < 0;
        uint inc = fromFront ? b.links.x : b.links.y;
        uint other = fromFront ? b.links.y : b.links.x;
        if (inc != region) {
            stats.z++;
            noteInconsistency(hit.boundary, region, inc, kGpuNone, hit.p);
            region = inc;
        }
        segment += hit.t;
        uint surfType = g_optics[b.links.z].a.x;
        if (surfType == kSurfNull) {
            region = other;
            o = offsetOrigin(hit.p, hit.n, d, hit.scale);
            if (depth >= 4 * maxDepth) break;
            continue;
        }
        if (surfType == kSurfStainedGlass && passStainedGlass(hit, d, wl, beta, rng)) {
            o = offsetOrigin(hit.p, hit.n, d, hit.scale);
            if (depth >= 4 * maxDepth) break;
            continue;
        }
        const float cosIn = abs(dot(hit.n, d));
        const float dist = segment;
        dVCM *= mis(dist * dist);
        dVCM /= mis(cosIn);
        dVC /= mis(cosIn);
        dVM /= mis(cosIn);
        segment = 0;
        ++pathLength;

        // Emission met (s = 0).
        if (b.links.w != kGpuNone) {
            GEmission e = g_emissions[b.links.w];
            if ((fromFront && e.front != 0) || (!fromFront && e.back != 0)) {
                float w = 1;
                if (pathLength > 1 && b.mesh.x != kGpuNone) {
                    const float directPdfA = lightPdf(prevP, hit) * cosIn / (dist * dist);
                    const float emissionPdfW = areaEmissionPdf(b, b.mesh.x, cosIn);
                    w = 1 / (1 + mis(directPdfA) * dVCM + mis(emissionPdfW) * dVC);
                }
                L += wl.toXYZ(beta * wl.eval(e.radiance) * w, G.misc.y);
            }
        }
        if (surfType == kSurfAbsorber || surfType == kSurfDetector) break;
        if (depth >= maxDepth) break;

        HitInterface hi = makeInterface(hit, wl.hero());
        if (surfType == kSurfDielectric && !wl.secondaryTerminated()) {
            [ForceUnroll] for (int i = 1; i < NW; ++i)
                if (wl.pdf[i] != 0 && isDispersive(hi.it, atWavelength(hi, wl.lambda[i]))) {
                    wl.terminateSecondary();
                    beta = maskTerminated(wl, beta);
                    break;
                }
        }
        if (!hi.it.optics.isDelta()) {
            nonDelta = true;
            const float etaX = etaAt(hit.p);
            // Next-event estimation (s = 1).
            {
                VcmLightSample ls;
                if (sampleLightVcm(hit.p, wl, rng, ls) && !isZero(ls.Le) && ls.directPdfW > 0 && !skip(2)) {
                    float4 f = 0;
                    float pdfFwd = 0;
                    [ForceUnroll] for (int i = 0; i < NW; ++i) {
                        if (wl.pdf[i] == 0) continue;
                        float pdf;
                        f[i] = evalScatter(i == 0 ? hi.it : atWavelength(hi, wl.lambda[i]), d, ls.wi, kModeRadiance, pdf);
                        if (i == 0) pdfFwd = pdf;
                    }
                    if (!isZero(f)) {
                        float pdfRev;
                        evalScatter(hi.it, -ls.wi, -d, kModeRadiance, pdfRev);
                        const float cosToLight = abs(dot(hit.n, ls.wi));
                        const float wLight = mis(pdfFwd / ls.directPdfW);
                        const float wCamera = mis(ls.emissionPdfW * cosToLight / (ls.directPdfW * ls.cosAtLight)) *
                                              (etaX + dVCM + dVC * mis(pdfRev));
                        const float w = 1 / (wLight + 1 + wCamera);
                        uint startRegion = dot(ls.wi, hit.n) > 0 ? hi.front : hi.back;
                        float3 so = offsetOrigin(hit.p, hit.n, ls.wi, hit.scale);
                        float sdist = isinf(ls.dist) ? ls.dist : ls.dist - dot(so - hit.p, ls.wi);
                        float4 Tr = transmittance(so, startRegion, ls.wi, sdist, wl, hit.scale);
                        if (!isZero(Tr))
                            L += wl.toXYZ(beta * f * ls.Le * Tr * (scatterCosine(hi.it, -d, ls.wi, kModeRadiance) * w / ls.directPdfW),
                                          G.misc.y);
                    }
                }
            }
            // Connection to a light vertex (s, t ≥ 2), drawn uniformly from the cache.
            if (M > 0 && !skip(4)) {
                const uint k = group * cacheCapacity() + min(uint(rng.uniform() * float(M)), M - 1);
                LightVertex v = g_lightVertices[k];
                float3 dv = v.p.xyz - hit.p;
                float dist2 = dot(dv, dv);
                if (dist2 > 0) {
                    const float dl = sqrt(dist2);
                    const float3 dir = dv / dl;
                    const float3 dLight = v.d.xyz;
                    const bool lightTerminated = (asuint(v.d.w) & kLvTerminated) != 0;
                    Wavelengths wj = wl;
                    if (lightTerminated) wj.terminateSecondary();
                    float4 fc = 0, fl = 0;
                    float pdfCamFwd = 0, pdfLightFwd = 0;
                    Interface itl = lightVertexInterface(v, wj.lambda.x);
                    [ForceUnroll] for (int i = 0; i < NW; ++i) {
                        if (wj.pdf[i] == 0) continue;
                        float pdf;
                        fc[i] = evalScatter(i == 0 ? hi.it : atWavelength(hi, wj.lambda[i]), d, dir, kModeRadiance, pdf);
                        if (i == 0) pdfCamFwd = pdf;
                        fl[i] = evalScatter(i == 0 ? itl : lightVertexInterface(v, wj.lambda[i]), dLight, -dir, kModeImportance, pdf);
                        if (i == 0) pdfLightFwd = pdf;
                    }
                    if (!isZero(fc * fl)) {
                        float pdfCamRev, pdfLightRev;
                        evalScatter(hi.it, -dir, -d, kModeRadiance, pdfCamRev);
                        evalScatter(itl, dir, -dLight, kModeRadiance, pdfLightRev);
                        const float cosX = abs(dot(hit.n, dir)), cosY = abs(dot(v.n.xyz, dir));
                        const float camPdfA = pdfCamFwd * cosY / dist2, lightPdfA = pdfLightFwd * cosX / dist2;
                        const float wLight = mis(camPdfA) * (etaAt(v.p.xyz) + v.tex.w + v.mis.x * mis(pdfLightRev));
                        const float wCamera = mis(lightPdfA) * (etaX + dVCM + dVC * mis(pdfCamRev));
                        const float w = 1 / (wLight + 1 + wCamera);
                        uint startRegion = dot(dir, hit.n) > 0 ? hi.front : hi.back, endRegion;
                        float3 so = offsetOrigin(hit.p, hit.n, dir, hit.scale);
                        float4 Tr = transmittance(so, startRegion, dir, dl - dot(so - hit.p, dir), wj, hit.scale, endRegion);
                        GBoundary lb = g_boundaries[asuint(v.p.w)];
                        const uint arrival = dot(-dir, v.n.xyz) > 0 ? lb.links.x : lb.links.y;
                        if (!isZero(Tr) && endRegion == arrival) {
                            const float g = scatterCosine(hi.it, -d, dir, kModeRadiance) * scatterCosine(itl, -dLight, -dir, kModeImportance) / dist2;
                            L += joinedXYZ(wl, lightTerminated, beta * fc * v.beta * fl * Tr * (g * w * float(M) / NL));
                            noteUseful(asuint(v.mis.z));
                        }
                    }
                }
            }
            // Merging with the light vertices about this point (vcm).
            if (merging() && !skip(8)) {
                const float r = mergeRadius(hit.p), r2 = r * r;
                const float invEta = 1 / etaX;
                // The levels light vertices within reach live on: their radii span those at the
                // nearest and farthest distance from the eye in reach (almost always one level).
                const float dx = length(hit.p);
                const int levelLo = gridLevel(mergeRadius(hit.p * (max(dx - r, 0.0) / max(dx, 1e-30))));
                const int levelHi = gridLevel(mergeRadius(hit.p * ((dx + r) / max(dx, 1e-30))));
                for (int level = levelLo; level <= levelHi; ++level) {
                    const float size = cellSize(level);
                    const int3 base = cellOf(hit.p - 0.5 * size, size);
                    for (int c8 = 0; c8 < 8; ++c8) {
                        const int3 c = base + int3(c8 & 1, (c8 >> 1) & 1, (c8 >> 2) & 1);
                        for (uint i = g_gridHead[gridHash(c, level)]; i != kGpuNone; i = g_gridNext[i]) {
                            const float4 vp = g_lightVertices[i].p;
                            const float3 dp = vp.xyz - hit.p;
                            if (dot(dp, dp) > r2) continue;
                            if (gridLevel(mergeRadius(vp.xyz)) != level || any(cellOf(vp.xyz, size) != c)) continue;
                            LightVertex v = g_lightVertices[i];
                            if (abs(dot(dp, hit.n)) > 0.25 * r || dot(v.n.xyz, hit.n) < 0.5) continue;
                            if (asuint(v.mis.w) != group) continue;  // other wavelengths
                            const float3 wi = -v.d.xyz;
                            const bool lightTerminated = (asuint(v.d.w) & kLvTerminated) != 0;
                            Wavelengths wj = wl;
                            if (lightTerminated) wj.terminateSecondary();
                            float4 fc = 0;
                            float pdfFwd = 0;
                            [ForceUnroll] for (int j = 0; j < NW; ++j) {
                                if (wj.pdf[j] == 0) continue;
                                float pdf;
                                fc[j] = evalScatter(j == 0 ? hi.it : atWavelength(hi, wj.lambda[j]), d, wi, kModeRadiance, pdf);
                                if (j == 0) pdfFwd = pdf;
                            }
                            if (isZero(fc)) continue;
                            float pdfRev;
                            evalScatter(hi.it, -wi, -d, kModeRadiance, pdfRev);
                            const float wLight = v.tex.w * invEta + v.mis.y * mis(pdfFwd);
                            const float wCamera = dVCM * invEta + dVM * mis(pdfRev);
                            const float w = 1 / (wLight + 1 + wCamera);
                            // A shading normal: the camera's cosine over the density per geometric area.
                            float gather = 1;
                            if (hi.it.smooth()) {
                                const float cn = abs(dot(wi, hit.n));
                                gather = cn > 0 ? abs(dot(wi, hi.it.ns)) / cn : 0.0;
                            }
                            L += joinedXYZ(wl, lightTerminated, beta * fc * v.beta * (gather * w * invEta));
                            noteUseful(asuint(v.mis.z));
                        }
                    }
                }
            }
        }

        float u3 = rng.uniform();
        float u2 = rng.uniform();
        float u1 = rng.uniform();
        ScatterSample s;
        if (!sampleScatter(hi.it, d, u1, u2, u3, kModeRadiance, s) || s.weight <= 0) break;
        const float cosOut = abs(dot(hit.n, s.wi));
        if (s.delta) {
            dVCM = 0;
            dVC *= mis(cosOut);
            dVM *= mis(cosOut);
        } else {
            float pdfRev;
            evalScatter(hi.it, -s.wi, -d, kModeRadiance, pdfRev);
            const float eta = etaAt(hit.p);
            dVC = mis(cosOut / s.pdf) * (dVC * mis(pdfRev) + dVCM + eta);
            dVM = mis(cosOut / s.pdf) * (dVM * mis(pdfRev) + dVCM * (eta > 0 ? 1 / eta : 0.0) + 1);
            dVCM = mis(1 / s.pdf);
        }
        beta.x *= s.weight;
        [ForceUnroll] for (int i = 1; i < NW; ++i)
            if (wl.pdf[i] != 0) beta[i] *= secondaryScatterWeight(atWavelength(hi, wl.lambda[i]), d, s, kModeRadiance);
        if (s.transmitted) region = other;
        prevP = hit.p;
        o = offsetOrigin(hit.p, hit.n, s.wi, hit.scale);
        d = s.wi;
        if (depth >= rrDepth) {
            float q = min(1.0, maxValue(beta));
            if (rng.uniform() >= q) break;
            beta *= 1 / q;
        }
    }
    return L;
}
