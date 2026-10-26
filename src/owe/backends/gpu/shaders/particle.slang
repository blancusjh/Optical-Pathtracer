// Particles (light tracing): float port of Tracer::traceParticle (transport.cpp). Emitters (area
// lights, sun, sky) → world → the detector: virtual observers receive connections from every
// non-specular vertex; surface sensors receive the particles that reach them. SPPM photons and VCM
// light subpaths leave the emitters the same way (emitParticle).
//
// Splats are accumulated exactly. A pixel receives contributions from many threads in no fixed
// order; float atomics would make the image depend on that order. Instead every splat is added
// as integers: its 24-bit significand, shifted to its binary exponent, lands in two of
// kSplatBins 64-bit bins (each 32 bits of exponent range, carries propagated with 32-bit atomics,
// which every Vulkan device has). Integer addition is associative, so particle images are bitwise
// reproducible like path-traced ones. The bins span splat exponents 2^-80 .. 2^48, far beyond any
// physical contribution; significand bits below 2^-80 are dropped (values under ~1e-17).
#pragma once
#include "detector.slang"
#include "kernel.slang"
#include "transport.slang"

static const int kSplatBins = 4;
static const int kSplatExp0 = -80;  // weight of bin 0's least significant bit: 2^kSplatExp0
static const uint kSplatWords = 3 * kSplatBins * 2;  // per pixel: XYZ × bins × (lo, hi)

// Adds v exactly to the fixed-point bins at words [base, base + 2·kSplatBins) of buf.
void addFixedPoint(RWStructuredBuffer<uint> buf, uint base, float v) {
    if (!(v > 0) || isinf(v)) return;  // zero, negative or not finite: nothing to add
    uint bits = asuint(v);
    int e = int((bits >> 23) & 0xFFu);
    if (e == 0) return;  // subnormal
    uint mant = (bits & 0x7FFFFFu) | 0x800000u;
    int p = e - 150 - kSplatExp0;  // position of the significand's lowest bit above bin 0's
    if (p < 0) {
        if (p <= -24) return;
        mant >>= uint(-p);
        p = 0;
    }
    int b = p >> 5, r = p & 31;
    if (b >= kSplatBins) return;
    add64(buf, base + uint(b) * 2, mant << uint(r));
    uint high = r == 0 ? 0u : mant >> uint(32 - r);
    if (high != 0 && b + 1 < kSplatBins) add64(buf, base + uint(b + 1) * 2, high);
}
void addFixedPointXYZ(RWStructuredBuffer<uint> buf, uint base, float3 c) {
    addFixedPoint(buf, base, c.x);
    addFixedPoint(buf, base + 2 * kSplatBins, c.y);
    addFixedPoint(buf, base + 4 * kSplatBins, c.z);
}

void deposit(int px, int py, float4 value, Wavelengths wl) {
    addFixedPointXYZ(g_splat, (uint(py) * uint(pc.width) + uint(px)) * kSplatWords, wl.toXYZ(value, globals().misc.y));
}

// IdealObserver::connect: from point x, a pupil point q, the pixel, and the factor W such that the
// pixel receives β·f·|cos_x|·W·Tr / N (the observer sits at the camera-relative origin); dir: the
// unit direction from x to q.
struct ObserverLink {
    float3 q, dir;
    int px, py;
    float factor;
};
bool linkObserver(float3 x, inout Rng rng, out ObserverLink k) {
    GGlobals G = globals();
    k = (ObserverLink)0;
    const float pupil = G.lens.z, focus = G.lens.w;
    const float3 fwd = G.fwd.xyz, right = G.right.xyz, up = G.upv.xyz;
    if (pupil > 0) {
        float2 u = draw2(rng);
        float2 dk = sampleUniformDiskConcentric(u.x, u.y);
        k.q = (right * dk.x + up * dk.y) * pupil;
    }
    float3 w = x - k.q;
    float dist2 = dot(w, w);
    if (dist2 <= 0) return false;
    float3 om = w / sqrt(dist2);
    float c = dot(om, fwd);
    if (c <= 1e-9) return false;
    float sx, sy;
    if (pupil > 0 && focus > 0) {
        float3 F = k.q + om * (focus / c);
        sx = dot(F, right) / focus;
        sy = dot(F, up) / focus;
    } else {
        sx = dot(om, right) / c;
        sy = dot(om, up) / c;
    }
    const float W = float(G.limits.z), H = float(G.limits.w);
    float2 f = filterOffset(rng);  // a direction just outside the field still reaches the border pixels
    float fx = (sx / G.lens.x + 1) * 0.5 * W + f.x;
    float fy = (1 - sy / G.lens.y) * 0.5 * H + f.y;
    if (!(fx >= 0 && fy >= 0 && fx < W && fy < H)) return false;
    k.px = min(int(fx), int(W) - 1);
    k.py = min(int(fy), int(H) - 1);
    float pixelArea = (2 * G.lens.x / W) * (2 * G.lens.y / H);
    k.factor = 1 / (dist2 * pixelArea * c * c * c);
    k.dir = normalize(k.q - x);
    return true;
}

// SurfaceSensor::pixelOfHit on the sensor's own boundary (local coordinates).
bool sensorPixel(float3 pLocal, inout Rng rng, out int px, out int py, out float pixelArea) {
    GGlobals G = globals();
    const float W = float(G.limits.z), H = float(G.limits.w);
    float fx = (pLocal.x / G.sensorSize.x + 1) * 0.5 * W;
    float fy = (1 - pLocal.y / G.sensorSize.y) * 0.5 * H;
    if (G.det.y != 0) fx = W - fx;
    if (G.det.z != 0) fy = H - fy;
    float2 f = filterOffset(rng);
    fx += f.x;
    fy += f.y;
    px = py = 0;
    pixelArea = (2 * G.sensorSize.x / W) * (2 * G.sensorSize.y / H);
    if (!(fx >= 0 && fy >= 0 && fx < W && fy < H)) return false;
    px = min(int(fx), int(W) - 1);
    py = min(int(fy), int(H) - 1);
    return true;
}

// Connects a particle vertex to the observer (surface: the hit's interface at the hero wavelength;
// otherwise a medium scattering point with phase anisotropy g).
void connectVertex(SurfaceHit hit, bool surface, HitInterface base, float3 d, float g, uint region, float4 beta,
                   Wavelengths wl, inout Rng rng) {
    ObserverLink k;
    if (!linkObserver(hit.p, rng, k)) return;
    float4 f = 0;
    float cosX = 1;
    uint startRegion = region;
    if (!surface) {
        f = hgPhase(dot(-d, k.dir), g);
    } else {
        float pdf;
        f = evalScatterSpectral(base, wl, d, k.dir, kModeImportance, pdf);
        if (isZero(f)) return;
        cosX = scatterCosine(base.it, -d, k.dir, kModeImportance);
        startRegion = dot(k.dir, hit.n) > 0 ? base.front : base.back;
    }
    Endpoint start = surface ? endpoint(hit, startRegion) : endpoint(hit.p, startRegion, hit.scale);
    float4 Tr = connectionTransmittance(start, observerEnd(k.q), wl);
    if (isZero(Tr)) return;
    deposit(k.px, k.py, beta * f * Tr * (cosX * k.factor), wl);
}

// A ray from the disc beyond the world (sun and sky particles) meets it with the float error of its
// far origin, microns along the ray at tens of metres: enough to start the next segment under the
// surface it scatters from. The hit is found again from just before it, where float is precise.
void refineDistantHit(float3 d, inout SurfaceHit hit) {
    const float back = 1e-5 * (1 + maxAbsComponent(hit.p));
    SurfaceHit h;
    if (intersectWorld(hit.p - d * back, d, 4 * back, h)) hit = h;
}

#ifdef OWE_PHOTON_GUIDE
// Photon guiding (SPPM, VCM): a map over the sun's emission disc (kGuideRes² cells over its bounding
// square) counts where photons from the uniform share turned out useful, i.e. reached a visible
// point. A share kGuideShare of sun photons is then drawn from the map, the rest uniformly over the
// disc; every photon is weighted by the mixture density, so the estimate is unchanged (the uniform
// share keeps full support) and only its noise falls. It learns indirect geometry too: a lens far
// from the caustic it forms, a window a room is lit through.
// Layout of g_guide: counts [0, N), normalised CDF [N, 2N) (float bits), total [2N] (float bits).
[[vk::binding(22, 0)]] RWStructuredBuffer<uint> g_guide;
static const uint kGuideRes = 128, kGuideCells = kGuideRes * kGuideRes;
static const float kGuideShare = 0.75;
float guideTotal() { return asfloat(g_guide[2 * kGuideCells]); }
uint guideSample(float u) {  // first cell whose CDF exceeds u
    uint lo = 0, len = kGuideCells;
    while (len > 0) {
        uint half = len / 2;
        if (asfloat(g_guide[kGuideCells + lo + half]) <= u) { lo += half + 1; len -= half + 1; }
        else len = half;
    }
    return min(lo, kGuideCells - 1);
}
float guideCellProbability(uint c) {
    float hi = asfloat(g_guide[kGuideCells + c]), lo = c > 0 ? asfloat(g_guide[kGuideCells + c - 1]) : 0.0;
    return hi - lo;
}
// The sun's emission density at q (coordinates in the disc's plane about its centre): the map's
// mixture with the uniform share; cell: q's cell of the map (kGpuNone outside it).
float sunDiscPdf(float2 q, float R, out uint cell) {
    const float size = 2 * R / float(kGuideRes);
    int2 ci = int2(floor((q + R) / size));
    const bool inGrid = all(ci >= 0) && all(ci < int(kGuideRes));
    cell = inGrid ? uint(ci.y) * kGuideRes + uint(ci.x) : kGpuNone;
    const float uniform = dot(q, q) <= R * R ? 1 / (Pi * R * R) : 0.0;
    if (!(guideTotal() > 0)) return uniform;
    return (inGrid ? kGuideShare * guideCellProbability(cell) / (size * size) : 0.0) + (1 - kGuideShare) * uniform;
}
#endif

// The disc distant light (sun, sky) leaves from, beyond the world: its radius.
float emitterRadius() { return globals().misc.w * 1.001; }

// A particle leaving an emitter: its first ray, its throughput (flux per particle), and what VCM's
// MIS needs of its emission.
struct Emitted {
    PathRay ray;
    float4 beta;
    uint guideCell;      // photon guiding: the emission cell of a sun photon from the uniform share, else kGpuNone
    bool distant;        // from the sun or sky: its origin is on a disc beyond the world
    float emissionPdfW;  // pdf of the emitted position and direction, with the choice of the light
    float directPdfW;    // distant: next-event estimation's pdf of the direction
    float usedCos;       // |cos| at an area emitter (1 for distant light)
    uint light;          // area emitter: its entry in g_lights (kGpuNone otherwise)
    float3 pL, nL;       // area emitter: the point and normal in its own frame
};

// Samples an emission: a light (area emitter, sun) or the sky, by power. With connectToObserver, an
// area emitter's own point is also connected to a virtual observer (the emitter seen directly), in
// the random-number order of Tracer::traceParticle. False when nothing is emitted.
bool emitParticle(Wavelengths wl, inout Rng rng, bool connectToObserver, out Emitted e) {
    GGlobals G = globals();
    const float pSky = G.emit.x;
    e = (Emitted)0;
    e.guideCell = kGpuNone;
    e.light = kGpuNone;
    e.usedCos = 1;
    bool fromSky = rng.uniform() < pSky;
    uint li = 0;
    if (!fromSky) li = pickLight(rng.uniform());
    GLight L = g_lights[li];
    // An environment map is both the sky and a light: either way a sky photon, emitted with the
    // combined probability of the two choices (Tracer::traceParticle).
    const uint envLight = G.envMap.w;
    if (!fromSky && li == envLight) fromSky = true;
    const float pSkyPhoton = pSky + (envLight != kGpuNone ? (1 - pSky) * g_lights[envLight].pdf : 0.0);
    if (fromSky || L.kind == 1) {
        // Distant emission (sky dome or sun): pick a direction, then a point of a disc covering the world.
        const float R = emitterRadius();
        float3 toward;  // direction from which the light arrives
        float pdfDir, sel;
        float4 Le;
        float2 u = draw2(rng);
        if (fromSky) {
            if (G.sky.x == 3) {
                if (!sampleEnvMap(u.x, u.y, toward, pdfDir)) return false;
            } else {
                toward = sampleUniformSphere(u.x, u.y);
                pdfDir = 1 / (4 * Pi);
            }
            [ForceUnroll] for (int i = 0; i < NW; ++i) Le[i] = envSky(toward, wl.lambda[i]);
            sel = pSkyPhoton;
            e.directPdfW = skyDirectPdf(toward);
        } else {
            toward = sampleSunDirection(u.x, u.y);
            pdfDir = 1 / G.misc.x;
            Le = wl.eval(G.sun.y);
            sel = (1 - pSky) * L.pdf;
            e.directPdfW = sunPdf();
        }
        const float3 dir = -toward;
        float3 a1, a2;
        orthonormalBasis(dir, a1, a2);
        float2 q;  // position in the disc's plane, relative to its centre
        float pdfA = 1 / (Pi * R * R);
#ifdef OWE_PHOTON_GUIDE
        if (!fromSky && guideTotal() > 0) {
            const bool guided = rng.uniform() < kGuideShare;
            float2 v = draw2(rng);
            if (guided) {
                uint c = guideSample(v.x);
                q = float2(float(c % kGuideRes) + v.y, float(c / kGuideRes) + rng.uniform()) * (2 * R / float(kGuideRes)) - R;
            } else {
                q = sampleUniformDiskConcentric(v.x, v.y) * R;
            }
            uint cell;
            pdfA = sunDiscPdf(q, R, cell);
            if (!(pdfA > 0)) return false;
            if (!guided) e.guideCell = cell;  // the uniform share teaches the map
        } else
#endif
        {
            float2 v = draw2(rng);
            q = sampleUniformDiskConcentric(v.x, v.y) * R;
#ifdef OWE_PHOTON_GUIDE
            uint cell;
            sunDiscPdf(q, R, cell);
            if (!fromSky) e.guideCell = cell;  // before the map has learnt anything every sun photon teaches it
#endif
        }
        e.ray = pathRay(G.emit.yzw - dir * (1.01 * R) + a1 * q.x + a2 * q.y, dir, G.counts.z, 0);
        e.ray.scale = maxAbsComponent(e.ray.o);
        e.emissionPdfW = sel * pdfDir * pdfA;
        e.beta = Le / e.emissionPdfW;
        e.distant = true;
    } else {
        const float sel = (1 - pSky) * L.pdf;
        GBoundary b = g_boundaries[L.boundary];
        GEmission em = g_emissions[b.emission];
        float4 Le = wl.eval(em.radiance);
        float3 u = draw3(rng);
        uint sampledPrimitive;
        if (!sampleArea(b, u.x, u.y, u.z, e.pL, e.nL, sampledPrimitive)) return false;
        float3 p = xfPoint(b.toWorld, e.pL), n = normalize(xfVector(b.toWorld, e.nL));
        const float pdfA = 1 / b.area;
        // Emitter seen directly by a virtual observer (path tracing owns this under the partition).
        ObserverLink k;
        if (connectToObserver && linkObserver(p, rng, k)) {
            float cosL = dot(n, k.dir);
            if ((cosL > 0 && em.front != 0) || (cosL < 0 && em.back != 0)) {
                Endpoint start = endpoint(p, cosL > 0 ? b.front : b.back, b.scale, L.boundary, n, sampledPrimitive);
                float4 Tr = connectionTransmittance(start, observerEnd(k.q), wl);
                if (!isZero(Tr)) deposit(k.px, k.py, Le * Tr * (abs(cosL) * k.factor / (sel * pdfA)), wl);
            }
        }
        float sidePdf = 1;
        bool front = em.front != 0;
        if (em.front != 0 && em.back != 0) {
            front = rng.uniform() < 0.5;
            sidePdf = 0.5;
        }
        float2 c = draw2(rng);
        float3 dir = Frame(front ? n : -n).toWorld(sampleCosineHemisphere(c.x, c.y));
        e.ray = pathRay(p, dir, front ? b.front : b.back, b.scale);
        e.ray.departed = L.boundary;
        e.ray.departedPrimitive = sampledPrimitive;
        e.ray.normal = n;
        // β = Le·cos / (sel · pdfA · sidePdf · cos/π)
        e.beta = Le * (Pi / (sel * pdfA * sidePdf));
        e.usedCos = abs(dot(n, dir));
        e.emissionPdfW = sel * pdfA * sidePdf * e.usedCos * InvPi;
        e.light = li;
    }
    e.beta = maskTerminated(wl, e.beta);
    return !isZero(e.beta);
}

// A segment of an emitted particle's walk: the first one, from a distant emitter, has its hit found
// again near it.
bool traceFrom(Emitted em, PathRay ray, int depth, out SurfaceHit hit) {
    bool found = ray.trace(hit);
    if (found && depth == 0 && em.distant) refineDistantHit(ray.d, hit);
    return found;
}

// Emits one particle (four wavelengths; secondaries may be terminated at dispersive interfaces) and
// splats its contributions. stats: paths, segments, region inconsistencies, leaks.
void traceParticle(inout Wavelengths wl, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const bool partition = (pc.flags & kFlagPartition) != 0;
    const bool observer = G.counts.w == 0;
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    stats.x++;

    Emitted em;
    if (!emitParticle(wl, rng, observer && !partition, em)) return;
    PathRay ray = em.ray;
    float4 beta = em.beta;
    int nonDeltaEvents = 0, deltaEvents = 0;

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = traceFrom(em, ray, depth, hit);
        stats.y++;
        uint med = mediumOf(ray.region);
        // Under the caustic partition, connect only at the first non-specular vertex after ≥1 specular one.
        const bool mayConnect = !partition || (nonDeltaEvents == 0 && deltaEvents >= 1);
        float t;
        const uint event = sampleMedium(med, found ? hit.t : infinity(), wl, beta, rng, t);
        if (event == kMediumEnd) break;
        if (event == kMediumScatter) {
            SurfaceHit mh = (SurfaceHit)0;
            mh.p = ray.o + ray.d * t;
            mh.scale = ray.scale;
            float g = g_media[med].g.x;
            if (observer && mayConnect) connectVertex(mh, false, (HitInterface)0, ray.d, g, ray.region, beta, wl, rng);
            ++nonDeltaEvents;
            if (partition) break;  // nothing further can be connected
            if (depth >= maxDepth) break;
            float2 u = draw2(rng);
            float3 wi;
            sampleHg(-ray.d, g, u.x, u.y, wi);
            ray.leaveVolume(mh.p, wi);
            if (depth >= rrDepth && !survives(0.95, beta, rng)) break;  // any survival probability in (0,1] is unbiased
            continue;
        }
        if (!found) {
            if (ray.region != G.counts.z) stats.w++;
            break;
        }
        const Crossing c = meetBoundary(hit, ray, stats);
        if (hit.ambiguousContact) break;
        if (passThrough(hit, c, ray, wl, beta, rng)) {
            if (depth >= 4 * maxDepth) break;
            continue;
        }
        // A diffuse measurement screen records incident flux and then scatters the reflected
        // fraction. An absorbing detector records flux and terminates the path.
        if (c.fromFront && hit.boundary == G.target.x) {
            int px, py;
            float area;
            if (sensorPixel(hit.pLocal, rng, px, py, area)) deposit(px, py, beta * (1 / area), wl);
        }
        if (c.surface == kSurfDetector || c.surface == kSurfAbsorber) break;
        HitInterface hi = makeInterface(hit, wl.hero());
        if (!hi.it.optics.isDelta()) {
            if (observer && mayConnect) connectVertex(hit, true, hi, ray.d, 0, ray.region, beta, wl, rng);
            ++nonDeltaEvents;
            if (partition) break;  // later vertices are owned by path tracing
        }
        if (depth >= maxDepth) break;
        terminateIfDispersive(hi, wl, beta);
        ScatterSample s;
        if (!scatterAtSurface(hit, c, hi, ray, wl, beta, rng, kModeImportance, s)) break;
        if (s.delta) ++deltaEvents;
        // Particles carry flux, so roulette against the per-event throughput.
        if (depth >= rrDepth && s.weight < 1 && !survives(s.weight, beta, rng)) break;
    }
}
