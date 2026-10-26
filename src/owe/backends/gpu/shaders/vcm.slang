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
// is the number every MIS weight counts. A dispersive event on either side leaves the hero alone.
// Pdfs are converted between solid angle and area with the geometric normal throughout; a shading
// normal enters only the contributions (optics.slang).
// Participating media are extinction only here: paths that scatter in them are the path tracer's
// (kFlagMediumOnly).
#pragma once
#include "sppm.slang"

struct LightVertex {
    float4 p;     // camera-relative position; w: boundary (bits)
    float4 n;     // geometric normal (toward the front region); w: shading normal (octahedral bits) if smooth
    float4 d;     // direction the light arrived along; w: low 2 flag bits, upper 30: primitive slot + 1
    float4 beta;  // light subpath throughput arriving (flux per path)
    float4 tex;   // texture weights; w: dVCM
    float4 mis;   // x: dVC, y: dVM, z: the light path's guide cell (bits; kGpuNone: none), w: its group (bits)
};
static const uint kLvTerminated = 1u, kLvSmooth = 2u;
// Keep the 96-byte cache layout. Zero means unknown; mesh IDs are global triangle slots,
// including their owner and original-face identity. Slots beyond the encoding range stay unknown.
uint cachedPrimitive(uint flags) {
    uint encoded = flags >> 2;
    return encoded == 0 ? kGpuNone : encoded - 1;
}
uint cacheFlags(uint flags, uint primitive) {
    return flags | ((primitive < 0x3fffffffu ? primitive + 1 : 0u) << 2);
}

float cachedSurfaceScale(GBoundary b, float3 p) {
    if (!hasSphereFrame(b)) return b.scale;
    uint data = b.data;
    float4 toPole[3];
    [ForceUnroll] for (int k = 0; k < 3; ++k) toPole[k] = g_shapeData[data + 1 + k];
    return max(maxAbsComponent(xfPoint(toPole, p)),
               max(abs(g_shapeData[data + 4].w), max(abs(g_shapeData[data + 5].w), abs(g_shapeData[data + 6].w))));
}

// The interface at a cached light vertex, at wavelength lambda.
HitInterface lightVertexInterface(LightVertex v, float lambda) {
    const float3 ns = (asuint(v.d.w) & kLvSmooth) != 0 ? octDecode(asuint(v.n.w)) : v.n.xyz;
    return cachedInterface(asuint(v.p.w), v.n.xyz, ns, v.p.xyz, v.tex.xyz, lambda);
}

[[vk::binding(25, 0)]] RWStructuredBuffer<LightVertex> g_lightVertices;
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> g_vcmCounters;  // [0] vertices offered this iteration, [1] dropped this pass (cache full), [2] counting pass: paths with vertices, [4 + g] group g's vertices offered

// Named push fields are shared with the host through push_fields.inc.
bool merging() { return pc.mergeRadiusPerMetre > 0; }
uint cacheCapacity() { return pc.cacheCapacity; }
float lightPaths() { return float(pc.lightPathsPerGroup); }
uint vcmGroups() { return max(pc.flags >> 24, 1u); }
Wavelengths groupWavelengths(uint g) {
    float u = iterationRotation() + float(g) / float(4 * vcmGroups());
    return Wavelengths.sample(u - floor(u));
}
uint pixelGroup(uint pixel) { return uint(mixBits(pack64(pixel, pc.pass) ^ 0x2545f4914f6cdd1dULL) % uint64_t(vcmGroups())); }
// Group g's cache: vertices [g·capacity, g·capacity + count), its count at g_vcmCounters[4 + g].
uint groupCount(uint g) { return min(g_vcmCounters[4 + g], cacheCapacity()); }
float mergeRadius(float3 p) { return max(pc.mergeRadiusPerMetre * length(p), pc.mergeRadiusFloor); }
// η = π r² N_L: merging's weight relative to connections at p; its inverse. Both 0 without merging.
float etaAt(float3 p) {
    if (!merging()) return 0;
    float r = mergeRadius(p);
    return Pi * r * r * lightPaths();
}
float invEtaAt(float3 p) {
    float e = etaAt(p);
    return e > 0 ? 1 / e : 0.0;
}

// Debugging: strategies left out (pc.flags bits 8–15, explicit conformance settings): 1 emission met after a vertex,
// 2 next-event estimation, 4 connections, 8 merging, 16 light tracing, 32 emission met after a
// non-specular vertex.
bool skip(uint bit) { return ((pc.flags >> 8) & bit) != 0; }
// A counting pass (pc.flags bit 16): light subpaths only count their vertices, and how many of them
// had any (g_vcmCounters[2]), which sizes the cache.
bool counting() { return (pc.flags & (1u << 16)) != 0; }

// The MIS quantities use the balance heuristic (the power heuristic overflows float at
// astronomical distances).

// ---------------------------------------------------------------- the emission side of lights

// The emission pdf of an area light at a point with normal-cosine cosL toward the path (with the
// choice of the light among the emitters).
float areaEmissionPdf(GBoundary b, uint li, float cosL) {
    GGlobals G = globals();
    GEmission em = g_emissions[b.emission];
    float sidePdf = (em.front != 0 && em.back != 0) ? 0.5 : 1.0;
    return (1 - G.emit.x) * g_lights[li].pdf / b.area * sidePdf * abs(cosL) * InvPi;
}
// Sky light arriving from direction `toward`: the pdf of emitting it as a particle does.
float skyEmissionPdf(float3 toward) {
    GGlobals G = globals();
    const uint envLight = G.envMap.w;
    const float pSkyPhoton = G.emit.x + (envLight != kGpuNone ? (1 - G.emit.x) * g_lights[envLight].pdf : 0.0);
    float pdfDir = G.sky.x == 3 ? envMapPdf(toward) : 1 / (4 * Pi);
    float R = emitterRadius();
    return pSkyPhoton * pdfDir / (Pi * R * R);
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
// The pdf of emitting, as a light subpath, the light that next-event sample ls brings to p.
float lightEmissionPdf(LightSample ls, float3 p) {
    GLight L = g_lights[ls.light];
    if (L.kind == 1) return sunEmissionPdf(p, ls.wi);
    if (L.kind == 2) return skyEmissionPdf(ls.wi);
    return areaEmissionPdf(g_boundaries[L.boundary], ls.light, ls.cosAtLight);
}
// A light path found useful: its uniform-share sun emission cell counts for the guide.
void noteUseful(uint cell) {
    if (cell != kGpuNone) InterlockedAdd(g_guide[cell], 1u);
}

// The MIS quantities after a scattering sample s at the vertex (SmallVCM's update on sampling).
void updateVcmMis(HitInterface hi, SurfaceHit hit, float3 d, ScatterSample s, inout float dVCM, inout float dVC,
                  inout float dVM) {
    const float cosOut = abs(dot(hit.n, s.wi));
    if (s.delta) {
        dVCM = 0;
        dVC *= cosOut;
        dVM *= cosOut;
        return;
    }
    float pdfRev;
    evalScatter(hi.it, -s.wi, -d, kModeRadiance, pdfRev);
    const float eta = etaAt(hit.p);
    dVC = cosOut / s.pdf * (dVC * pdfRev + dVCM + eta);
    dVM = cosOut / s.pdf * (dVM * pdfRev + dVCM * (eta > 0 ? 1 / eta : 0.0) + 1);
    dVCM = 1 / s.pdf;
}

// ---------------------------------------------------------------- light subpaths

// Traces one light subpath: caches its non-specular vertices and connects each to the eye.
void traceLightVcm(inout Wavelengths wl, uint group, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    const float NL = lightPaths();
    stats.x++;
    Emitted em;
    if (!emitParticle(wl, rng, false, em) || !(em.emissionPdfW > 0)) return;
    PathRay ray = em.ray;
    float4 beta = em.beta;
    float dVCM = 0, dVC = em.usedCos / em.emissionPdfW, dVM = 0;
    float segment = 0;  // distance since the last scattering vertex (null boundaries do not scatter)
    bool first = true, counted = false;

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = traceFrom(em, ray, depth, hit);
        stats.y++;
        if (!attenuate(mediumOf(ray.region), found ? hit.t : infinity(), wl, beta)) return;
        if (!found) {
            if (ray.region != G.counts.z) stats.w++;
            return;
        }
        const Crossing c = meetBoundary(hit, ray.d, ray.region, ray.departed, stats);
        segment += hit.t;
        if (hit.ambiguousContact) return;
        if (passThrough(hit, c, ray, wl, beta, rng)) {
            if (depth >= 4 * maxDepth) return;
            continue;
        }
        if (c.surface == kSurfDetector || c.surface == kSurfAbsorber) return;
        const float3 d = ray.d;
        // The MIS quantities arrive at this vertex (SmallVCM's update on hitting a surface).
        const float cosIn = abs(dot(hit.n, d));
        if (first) {
            if (em.distant) {
                dVCM = em.directPdfW / em.emissionPdfW;
            } else {
                // Next-event estimation would have chosen the emitter point from here: its pdf
                // (solid angle) times the emitter's cosine is the area pdf times the distance².
                GBoundary lb = g_boundaries[g_lights[em.light].boundary];
                float directPdfW = g_lights[em.light].pdf * pdfFrom(lb, xfPoint(lb.toLocal, hit.p), em.pL, em.nL);
                dVCM = directPdfW * em.usedCos / em.emissionPdfW;
            }
            dVM = dVC * invEtaAt(hit.p);
            first = false;
        } else {
            dVCM *= segment * segment;
        }
        segment = 0;
        dVCM /= cosIn;
        dVC /= cosIn;
        dVM /= cosIn;

        HitInterface hi = makeInterface(hit, wl.hero());
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
                v.d = float4(d, asfloat(cacheFlags((wl.secondaryTerminated() ? kLvTerminated : 0u) |
                                                  (smooth ? kLvSmooth : 0u), hit.prim)));
                v.beta = beta;
                v.tex = float4(hi.it.tex, dVCM);
                v.mis = float4(dVC, dVM, asfloat(em.guideCell), asfloat(group));
                g_lightVertices[group * cacheCapacity() + slot] = v;
            } else {
                InterlockedAdd(g_vcmCounters[1], 1u);
            }
            // To the eye (light tracing, t = 1): only to the pixels of this path's group this
            // iteration (their camera paths share its wavelengths), scaled by G: each pixel takes
            // 1/G of the light paths.
            ObserverLink k;
            if (!counting() && linkObserver(hit.p, rng, k) && !skip(16) &&
                pixelGroup(uint(k.py) * pc.width + uint(k.px)) == group) {
                float pdfFwd;
                float4 f = evalScatterSpectral(hi, wl, d, k.dir, kModeImportance, pdfFwd);
                if (!isZero(f)) {
                    float pdfRev;
                    evalScatter(hi.it, -k.dir, -d, kModeRadiance, pdfRev);
                    const float cameraPdfA = k.factor * abs(dot(hit.n, k.dir));
                    const float wLight = cameraPdfA / NL * (etaAt(hit.p) + dVCM + dVC * pdfRev);
                    const float w = 1 / (wLight + 1);
                    Endpoint start = endpoint(hit, dot(k.dir, hit.n) > 0 ? hi.front : hi.back);
                    float4 Tr = connectionTransmittance(start, observerEnd(k.q), wl);
                    if (!isZero(Tr)) {
                        deposit(k.px, k.py, beta * f * Tr * (scatterCosine(hi.it, -d, k.dir, kModeImportance) * k.factor * w *
                                                             float(vcmGroups())), wl);
                        noteUseful(em.guideCell);
                    }
                }
            }
        }
        if (depth >= maxDepth) return;
        terminateIfDispersive(hi, wl, beta);
        ScatterSample s;
        if (!scatterAtSurface(hit, c, hi, ray, wl, beta, rng, kModeImportance, s)) return;
        updateVcmMis(hi, hit, d, s, dVCM, dVC, dVM);
        if (depth >= rrDepth && s.weight < 1 && !survives(s.weight, beta, rng)) return;
    }
}

// ---------------------------------------------------------------- camera subpaths

// Wavelengths stay fixed along a camera path. Cache only their colour matching
// values: the pdf can still change when either subpath terminates secondaries.
struct CameraCie {
    float3 cmf[NW];

    static CameraCie at(Wavelengths wl) {
        CameraCie cache;
        [ForceUnroll] for (int i = 0; i < NW; ++i) {
            const float l = wl.lambda[i];
            cache.cmf[i] = float3(cieX(l), cieY(l), cieZ(l));
        }
        return cache;
    }

    float3 toXYZ(Wavelengths wl, float4 value, float cieYIntegral) {
        float3 c = 0;
        const float inv = 1.0 / (NW * cieYIntegral);
        [ForceUnroll] for (int i = 0; i < NW; ++i) {
            if (wl.pdf[i] == 0 || value[i] == 0) continue;
            const float w = value[i] / wl.pdf[i] * inv;
            c += w * cmf[i];
        }
        return c;
    }
};

// XYZ of a contribution whose light side may have left only the hero wavelength.
float3 joinedXYZ(CameraCie cie, Wavelengths camera, bool lightTerminated, float4 value) {
    Wavelengths w = camera;
    if (lightTerminated) w.terminateSecondary();
    return cie.toXYZ(w, maskTerminated(w, value), globals().misc.y);
}

// A diffuse camera vertex's spectral response is independent of the connected/gathered
// direction. Evaluate its albedo once per hit, rather than once per gathered photon.
float4 cameraDiffuseResponse(HitInterface hi, Wavelengths wl) {
    float4 f = 0;
    if (hi.it.optics.type == kSurfDiffuse) {
        [ForceUnroll] for (int i = 0; i < NW; ++i)
            if (wl.pdf[i] != 0) f[i] = (i == 0 ? hi.it : atWavelength(hi, wl.lambda[i])).albedo() * InvPi;
    }
    return f;
}

// Camera-side packet BSDF evaluation. Diffuse geometry and its sampling pdf are common to
// all wavelengths; other material models retain the general spectral evaluation.
float4 evalCameraVcm(HitInterface hi, Wavelengths wl, float4 diffuseF, float3 d, float3 wi, out float pdfFwd) {
    pdfFwd = 0;
    if (hi.it.optics.type == kSurfDiffuse) {
        Interface it = hi.it;
        if (it.smooth() && !(it.agrees(-d) && it.agrees(wi))) return 0;
        const float cosIn = dot(-d, it.ns), cosOut = dot(wi, it.ns);
        const bool front = cosIn > 0;
        if ((!front && it.optics.backAbsorbs != 0) || !(cosIn * cosOut > 0)) return 0;
        pdfFwd = abs(cosOut) * InvPi;
        const float share = front ? it.optics.sampleAimNormal.w : 0;
        if (share > 0) pdfFwd = (1 - share) * pdfFwd + share * screenAimPdf(it, wi);
        return maskTerminated(wl, diffuseF);
    }
    return evalScatterSpectral(hi, wl, d, wi, kModeRadiance, pdfFwd);
}

// Traces one camera path from pixel (px, py); returns its contribution (XYZ, with the camera's weight).
float3 traceCameraVcm(float px, float py, Wavelengths wlIn, uint group, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    const float NL = lightPaths();
    Wavelengths wl = wlIn;
    float3 o, d0;
    float weight;
    if (!generateRay(px, py, rng, o, d0, weight) || weight <= 0) return 0;
    stats.x++;
    const float W = float(G.limits.z), H = float(G.limits.w);
    const float pixelArea = (2 * G.lens.x / W) * (2 * G.lens.y / H);
    const float cosCam = dot(d0, G.fwd.xyz);
    if (!(cosCam > 0)) return 0;
    const CameraCie cie = CameraCie.at(wl);
    const float cameraPdfW = 1 / (pixelArea * cosCam * cosCam * cosCam);
    float dVCM = NL / cameraPdfW, dVC = 0, dVM = 0;
    float4 beta = maskTerminated(wl, float4(weight));
    PathRay ray = pathRay(o, d0, G.det.x, 0);
    float3 L = 0;
    float segment = 0;
    float3 prevP = o;
    int pathLength = 0;
    bool nonDelta = false;  // debugging (strategy mask 32)
    const uint M = groupCount(group);

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = ray.trace(hit);
        stats.y++;
        if (!attenuate(mediumOf(ray.region), found ? hit.t : infinity(), wl, beta)) break;
        if (!found) {
            if (ray.region == G.counts.z) {
                // The sky and the sun at infinity: seen directly, or weighted against next-event
                // estimation and the light subpaths that leave them.
                float4 sky, sun;
                [ForceUnroll] for (int i = 0; i < NW; ++i) {
                    sky[i] = envSky(ray.d, wl.lambda[i]);
                    sun[i] = envSun(ray.d, wl.lambda[i]);
                }
                float wSky = 1, wSun = 1;
                if (pathLength > 0) {
                    wSky = 1 / (1 + skyDirectPdf(ray.d) * dVCM + skyEmissionPdf(ray.d) * dVC);
                    wSun = 1 / (1 + sunPdf() * dVCM + sunEmissionPdf(prevP, ray.d) * dVC);
                    if (skip(1) || (skip(32) && nonDelta)) wSky = wSun = 0;
                }
                L += cie.toXYZ(wl, beta * (sky * wSky + sun * wSun), G.misc.y);
            } else {
                stats.w++;
            }
            break;
        }
        const Crossing c = meetBoundary(hit, ray.d, ray.region, ray.departed, stats);
        segment += hit.t;
        if (hit.ambiguousContact) break;
        if (passThrough(hit, c, ray, wl, beta, rng)) {
            if (depth >= 4 * maxDepth) break;
            continue;
        }
        const float3 d = ray.d;
        const float cosIn = abs(dot(hit.n, d));
        const float dist = segment;
        dVCM *= dist * dist;
        dVCM /= cosIn;
        dVC /= cosIn;
        dVM /= cosIn;
        segment = 0;
        ++pathLength;

        // Emission met (s = 0).
        GBoundary b = g_boundaries[hit.boundary];
        if (b.emission != kGpuNone) {
            GEmission e = g_emissions[b.emission];
            if ((c.fromFront && e.front != 0) || (!c.fromFront && e.back != 0)) {
                float w = 1;
                if (pathLength > 1 && b.light != kGpuNone) {
                    const float directPdfA = lightPdf(prevP, hit) * cosIn / (dist * dist);
                    const float emissionPdfW = areaEmissionPdf(b, b.light, cosIn);
                    w = 1 / (1 + directPdfA * dVCM + emissionPdfW * dVC);
                }
                L += cie.toXYZ(wl, beta * wl.eval(e.radiance) * w, G.misc.y);
            }
        }
        if (c.surface == kSurfAbsorber || c.surface == kSurfDetector) break;
        if (depth >= maxDepth) break;

        HitInterface hi = makeInterface(hit, wl.hero());
        terminateIfDispersive(hi, wl, beta);
        if (!hi.it.optics.isDelta()) {
            nonDelta = true;
            const float etaX = etaAt(hit.p);
            const float4 diffuseF = cameraDiffuseResponse(hi, wl);
            // Next-event estimation (s = 1).
            LightSample ls;
            if (sampleLightFrom(hit.p, wl, rng, ls) && !isZero(ls.Le) && ls.pdf > 0 && !skip(2)) {
                float pdfFwd;
                const float4 f = evalCameraVcm(hi, wl, diffuseF, d, ls.wi, pdfFwd);
                if (!isZero(f)) {
                    float pdfRev;
                    evalScatter(hi.it, -ls.wi, -d, kModeRadiance, pdfRev);
                    const float cosToLight = abs(dot(hit.n, ls.wi));
                    const float wLight = pdfFwd / ls.pdf;
                    const float wCamera = lightEmissionPdf(ls, hit.p) * cosToLight / (ls.pdf * ls.cosAtLight) *
                                          (etaX + dVCM + dVC * pdfRev);
                    const float w = 1 / (wLight + 1 + wCamera);
                    const uint startRegion = dot(ls.wi, hit.n) > 0 ? hi.front : hi.back;
                    const float4 Tr = isinf(ls.dist) ? shadowTransmittance(endpoint(hit, startRegion), ls.wi, ls.dist, wl)
                                                     : connectionTransmittance(ls.endpoint, surfaceEnd(hit, startRegion), wl);
                    if (!isZero(Tr))
                        L += cie.toXYZ(wl, beta * f * ls.Le * Tr * (scatterCosine(hi.it, -d, ls.wi, kModeRadiance) * w / ls.pdf),
                                       G.misc.y);
                }
            }
            // Connection to a light vertex (s, t ≥ 2), drawn uniformly from the cache.
            if (M > 0 && !skip(4)) {
                const uint k = group * cacheCapacity() + min(uint(rng.uniform() * float(M)), M - 1);
                LightVertex v = g_lightVertices[k];
                float3 dv = v.p.xyz - hit.p;
                float dist2 = dot(dv, dv);
                if (dist2 > 0) {
                    const float3 dir = dv / sqrt(dist2);
                    const float3 dLight = v.d.xyz;
                    const bool lightTerminated = (asuint(v.d.w) & kLvTerminated) != 0;
                    Wavelengths wj = wl;
                    if (lightTerminated) wj.terminateSecondary();
                    float pdfCamFwd, pdfLightFwd;
                    const float4 fc = evalCameraVcm(hi, wj, diffuseF, d, dir, pdfCamFwd);
                    HitInterface hl = lightVertexInterface(v, wj.lambda.x);
                    const float4 fl = evalScatterSpectral(hl, wj, dLight, -dir, kModeImportance, pdfLightFwd);
                    if (!isZero(fc * fl)) {
                        float pdfCamRev, pdfLightRev;
                        evalScatter(hi.it, -dir, -d, kModeRadiance, pdfCamRev);
                        evalScatter(hl.it, dir, -dLight, kModeRadiance, pdfLightRev);
                        const float cosX = abs(dot(hit.n, dir)), cosY = abs(dot(v.n.xyz, dir));
                        const float camPdfA = pdfCamFwd * cosY / dist2, lightPdfA = pdfLightFwd * cosX / dist2;
                        const float wLight = camPdfA * (etaAt(v.p.xyz) + v.tex.w + v.mis.x * pdfLightRev);
                        const float wCamera = lightPdfA * (etaX + dVCM + dVC * pdfCamRev);
                        const float w = 1 / (wLight + 1 + wCamera);
                        GBoundary lb = g_boundaries[asuint(v.p.w)];
                        const Endpoint lightEnd = endpoint(v.p.xyz, dot(-dir, v.n.xyz) > 0 ? lb.front : lb.back,
                                                           cachedSurfaceScale(lb, v.p.xyz), asuint(v.p.w), v.n.xyz,
                                                           cachedPrimitive(asuint(v.d.w)));
                        const float4 Tr = connectionTransmittance(lightEnd, surfaceEnd(hit, dot(dir, hit.n) > 0 ? hi.front : hi.back), wj);
                        if (!isZero(Tr)) {
                            const float g = scatterCosine(hi.it, -d, dir, kModeRadiance) * scatterCosine(hl.it, -dLight, -dir, kModeImportance) / dist2;
                            L += joinedXYZ(cie, wl, lightTerminated, beta * fc * v.beta * fl * Tr * (g * w * float(M) / NL));
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
                        const int3 cell = base + int3(c8 & 1, (c8 >> 1) & 1, (c8 >> 2) & 1);
                        for (uint i = g_gridHead[gridHash(cell, level)]; i != kGpuNone; i = g_gridNext[i]) {
                            const float4 vp = g_lightVertices[i].p;
                            const float3 dp = vp.xyz - hit.p;
                            if (dot(dp, dp) > r2) continue;
                            if (gridLevel(mergeRadius(vp.xyz)) != level || any(cellOf(vp.xyz, size) != cell)) continue;
                            LightVertex v = g_lightVertices[i];
                            if (abs(dot(dp, hit.n)) > 0.25 * r || dot(v.n.xyz, hit.n) < 0.5) continue;
                            if (asuint(v.mis.w) != group) continue;  // other wavelengths
                            const float3 wi = -v.d.xyz;
                            const bool lightTerminated = (asuint(v.d.w) & kLvTerminated) != 0;
                            Wavelengths wj = wl;
                            if (lightTerminated) wj.terminateSecondary();
                            float pdfFwd;
                            const float4 fc = evalCameraVcm(hi, wj, diffuseF, d, wi, pdfFwd);
                            if (isZero(fc)) continue;
                            float pdfRev;
                            evalScatter(hi.it, -wi, -d, kModeRadiance, pdfRev);
                            const float wLight = v.tex.w * invEta + v.mis.y * pdfFwd;
                            const float wCamera = dVCM * invEta + dVM * pdfRev;
                            const float w = 1 / (wLight + 1 + wCamera);
                            // A shading normal: the camera's cosine over the density per geometric area.
                            float gather = 1;
                            if (hi.it.smooth()) {
                                const float cn = abs(dot(wi, hit.n));
                                gather = cn > 0 ? abs(dot(wi, hi.it.ns)) / cn : 0.0;
                            }
                            L += joinedXYZ(cie, wl, lightTerminated, beta * fc * v.beta * (gather * w * invEta));
                            noteUseful(asuint(v.mis.z));
                        }
                    }
                }
            }
        }

        ScatterSample s;
        if (!scatterAtSurface(hit, c, hi, ray, wl, beta, rng, kModeRadiance, s)) break;
        updateVcmMis(hi, hit, d, s, dVCM, dVC, dVM);
        prevP = hit.p;
        if (depth >= rrDepth && !survives(min(1.0, maxValue(beta)), beta, rng)) break;
    }
    return L;
}
