// Particles (light tracing): float port of Tracer::traceParticle (transport.cpp). Emitters (area
// lights, sun, sky) → world → the detector: virtual observers receive connections from every
// non-specular vertex; surface sensors receive the particles that reach them.
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
#include "transport.slang"

static const int kSplatBins = 4;
static const int kSplatExp0 = -80;  // weight of bin 0's least significant bit: 2^kSplatExp0
static const uint kSplatWords = 3 * kSplatBins * 2;  // per pixel: XYZ × bins × (lo, hi)

// Adds x to the 64-bit counter at words (index, index + 1).
void atomicAdd64(uint index, uint x) {
    if (x == 0) return;
    uint old;
    InterlockedAdd(g_splat[index], x, old);
    if (old > 0xFFFFFFFFu - x) InterlockedAdd(g_splat[index + 1], 1u);
}

void splatChannel(uint base, float v) {
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
    atomicAdd64(base + uint(b) * 2, mant << uint(r));
    uint high = r == 0 ? 0u : mant >> uint(32 - r);
    if (high != 0 && b + 1 < kSplatBins) atomicAdd64(base + uint(b + 1) * 2, high);
}

void deposit(int px, int py, float4 value, Wavelengths wl) {
    float3 c = wl.toXYZ(value, globals().misc.y);
    uint base = (uint(py) * uint(pc.width) + uint(px)) * kSplatWords;
    splatChannel(base, c.x);
    splatChannel(base + 2 * kSplatBins, c.y);
    splatChannel(base + 4 * kSplatBins, c.z);
}

// IdealObserver::connect: from point x, a pupil point q, the pixel, and the factor W such that the
// pixel receives β·f·|cos_x|·W·Tr / N (the observer sits at the camera-relative origin).
bool connectObserver(float3 x, inout Rng rng, out float3 q, out int px, out int py, out float factor) {
    GGlobals G = globals();
    q = 0;
    px = py = 0;
    factor = 0;
    const float pupil = G.lens.z, focus = G.lens.w;
    const float3 fwd = G.fwd.xyz, right = G.right.xyz, up = G.upv.xyz;
    if (pupil > 0) {
        float u2 = rng.uniform();  // Draw2 order: last argument first
        float u1 = rng.uniform();
        float2 dk = sampleUniformDiskConcentric(u1, u2);
        q = (right * dk.x + up * dk.y) * pupil;
    }
    float3 w = x - q;
    float dist2 = dot(w, w);
    if (dist2 <= 0) return false;
    float3 om = w / sqrt(dist2);
    float c = dot(om, fwd);
    if (c <= 1e-9) return false;
    float sx, sy;
    if (pupil > 0 && focus > 0) {
        float3 F = q + om * (focus / c);
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
    px = min(int(fx), int(W) - 1);
    py = min(int(fy), int(H) - 1);
    float pixelArea = (2 * G.lens.x / W) * (2 * G.lens.y / H);
    factor = 1 / (dist2 * pixelArea * c * c * c);
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
    float3 q;
    int px, py;
    float factor;
    if (!connectObserver(hit.p, rng, q, px, py, factor)) return;
    float3 wv = q - hit.p;
    float dist = length(wv);
    float3 wc = wv / dist;
    float4 f = 0;
    float cosX = 1;
    float3 origin = hit.p;
    uint startRegion = region;
    if (!surface) {
        f = hgPhase(dot(-d, wc), g);
    } else {
        [ForceUnroll] for (int i = 0; i < NW; ++i) {
            if (wl.pdf[i] == 0) continue;
            float pdfDummy;
            f[i] = evalScatter(i == 0 ? base.it : atWavelength(base, wl.lambda[i]), d, wc, kModeImportance, pdfDummy);
        }
        if (isZero(f)) return;
        cosX = scatterCosine(base.it, -d, wc, kModeImportance);
        startRegion = dot(wc, hit.n) > 0 ? base.front : base.back;
        origin = offsetOrigin(hit.p, hit.n, wc, hit.scale);
    }
    uint endRegion;
    float4 Tr = transmittance(origin, startRegion, wc, dist - dot(origin - hit.p, wc), wl, hit.scale, endRegion);
    if (isZero(Tr) || endRegion != globals().det.x) return;
    deposit(px, py, beta * f * Tr * (cosX * factor), wl);
}

// A particle leaving an emitter: its first ray, region and throughput (flux per particle).
struct Emitted {
    float3 o, d;
    uint region;
    float4 beta;
    float scale;
    uint guideCell;  // photon guiding: the emission cell of a sun photon from the uniform share, else kGpuNone
    bool distant;    // from the sun or sky: its origin is on a disc beyond the world
};

// A ray from the disc beyond the world (sun and sky particles) meets it with the float error of its
// far origin, microns along the ray at tens of metres: enough to start the next segment under the
// surface it scatters from. The hit is found again from just before it, where float is precise.
void refineDistantHit(float3 d, inout SurfaceHit hit) {
    const float back = 1e-5 * (1 + maxAbsComponent(hit.p));
    SurfaceHit h;
    if (intersectWorld(hit.p - d * back, d, 4 * back, h)) hit = h;
}

#ifdef OWE_PHOTON_GUIDE
// Photon guiding (SPPM): a map over the sun's emission disc (kGuideRes² cells over its bounding
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
#endif

// Samples an emission: a light (area emitter, sun) or the sky, by power. With connectToObserver, an
// area emitter's own point is also connected to a virtual observer (the emitter seen directly), in
// the random-number order of Tracer::traceParticle. False when nothing is emitted.
bool emitParticle(Wavelengths wl, inout Rng rng, bool connectToObserver, out Emitted e) {
    GGlobals G = globals();
    const uint ambient = G.counts.z;
    const float pSky = G.emit.x;
    e = (Emitted)0;
    e.guideCell = kGpuNone;
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
        // Distant emission (sky dome or sun): pick a direction, then a disk covering the world.
        float R = G.misc.w * 1.001;
        float3 toward;  // direction from which the light arrives
        float pdfDir, sel;
        float4 beta;
        if (fromSky) {
            float u2 = rng.uniform();
            float u1 = rng.uniform();
            if (G.sky.x == 3) {
                if (!sampleEnvMap(u1, u2, toward, pdfDir)) return false;
            } else {
                toward = sampleUniformSphere(u1, u2);
                pdfDir = 1 / (4 * kPi);
            }
            [ForceUnroll] for (int i = 0; i < NW; ++i) beta[i] = envSky(toward, wl.lambda[i]);
            sel = pSkyPhoton;
        } else {
            Frame f = Frame(G.sunDir.xyz);
            float u2 = rng.uniform();
            float u1 = rng.uniform();
            toward = f.toWorld(sampleUniformCone(u1, u2, G.sunDir.w));
            pdfDir = 1 / G.misc.x;
            beta = wl.eval(G.sun.y);
            sel = (1 - pSky) * L.pdf;
        }
        float3 dir = -toward;
        float3 a1, a2;
        orthonormalBasis(dir, a1, a2);
        const float3 c0 = G.emit.yzw - dir * (1.01 * R);  // the world's disc, upstream of it
#ifdef OWE_PHOTON_GUIDE
        if (!fromSky && guideTotal() > 0) {
            const bool guided = rng.uniform() < kGuideShare;
            float v2 = rng.uniform();
            float v1 = rng.uniform();
            const float cell = 2 * R / float(kGuideRes);
            float2 q;  // position in the disc's plane, relative to its centre
            if (guided) {
                uint c = guideSample(v1);
                q = float2(float(c % kGuideRes) + v2, float(c / kGuideRes) + rng.uniform()) * cell - R;
            } else {
                q = sampleUniformDiskConcentric(v1, v2) * R;
            }
            int2 ci = int2(floor((q + R) / cell));
            const bool inGrid = all(ci >= 0) && all(ci < int(kGuideRes));
            const uint c = inGrid ? uint(ci.y) * kGuideRes + uint(ci.x) : 0u;
            const float pdfA = (inGrid ? kGuideShare * guideCellProbability(c) / (cell * cell) : 0) +
                               (dot(q, q) <= R * R ? (1 - kGuideShare) / (kPi * R * R) : 0);
            if (!(pdfA > 0)) return false;
            e.o = c0 + a1 * q.x + a2 * q.y;
            e.beta = beta * (1 / (pdfDir * sel * pdfA));
            if (!guided && inGrid) e.guideCell = c;  // the uniform share teaches the map
        } else
#endif
        {
            float v2 = rng.uniform();
            float v1 = rng.uniform();
            float2 dk = sampleUniformDiskConcentric(v1, v2);
            e.o = c0 + (a1 * dk.x + a2 * dk.y) * R;
            e.beta = beta * (kPi * R * R / (pdfDir * sel));
#ifdef OWE_PHOTON_GUIDE
            if (!fromSky) {  // before the map has learnt anything every sun photon teaches it
                int2 ci = int2(floor((dk * R + R) / (2 * R / float(kGuideRes))));
                if (all(ci >= 0) && all(ci < int(kGuideRes))) e.guideCell = uint(ci.y) * kGuideRes + uint(ci.x);
            }
#endif
        }
        e.d = dir;
        e.region = ambient;
        e.distant = true;
        e.scale = maxAbsComponent(e.o);
    } else {
        float sel = (1 - pSky) * L.pdf;
        GBoundary b = g_boundaries[L.boundary];
        GEmission em = g_emissions[b.links.w];
        float4 Le = wl.eval(em.radiance);
        float u3 = rng.uniform();  // Draw3 order: last argument first
        float u2 = rng.uniform();
        float u1 = rng.uniform();
        float3 pL, nL;
        if (!sampleArea(b, u1, u2, u3, pL, nL)) return false;
        float3 p = xfPoint(b.toWorld, pL), n = normalize(xfVector(b.toWorld, nL));
        float pdfA = 1 / b.misc.y;
        // Emitter seen directly by a virtual observer (path tracing owns this under the partition).
        if (connectToObserver) {
            float3 q;
            int px, py;
            float factor;
            if (connectObserver(p, rng, q, px, py, factor)) {
                float3 wv = q - p;
                float dist = length(wv);
                float3 wc = wv / dist;
                float cosL = dot(n, wc);
                if ((cosL > 0 && em.front != 0) || (cosL < 0 && em.back != 0)) {
                    uint startRegion = cosL > 0 ? b.links.x : b.links.y, endRegion;
                    float3 from = offsetOrigin(p, n, wc, b.misc.x);
                    float4 Tr = transmittance(from, startRegion, wc, dist - dot(from - p, wc), wl, b.misc.x, endRegion);
                    if (!isZero(Tr) && endRegion == G.det.x)
                        deposit(px, py, Le * Tr * (abs(cosL) * factor / (sel * pdfA)), wl);
                }
            }
        }
        float sidePdf = 1;
        bool front = em.front != 0;
        if (em.front != 0 && em.back != 0) {
            front = rng.uniform() < 0.5;
            sidePdf = 0.5;
        }
        float3 ns = front ? n : -n;
        Frame f = Frame(ns);
        float c2 = rng.uniform();
        float c1 = rng.uniform();
        float3 dir = f.toWorld(sampleCosineHemisphere(c1, c2));
        // β = Le·cos / (sel · pdfA · sidePdf · cos/π)
        e.beta = Le * (kPi / (sel * pdfA * sidePdf));
        e.o = offsetOrigin(p, n, dir, b.misc.x);
        e.scale = b.misc.x;
        e.d = dir;
        e.region = front ? b.links.x : b.links.y;
    }
    e.beta = maskTerminated(wl, e.beta);
    return !isZero(e.beta);
}

// Emits one particle (four wavelengths; secondaries may be terminated at dispersive interfaces) and
// splats its contributions. stats: paths, segments, region inconsistencies, leaks.
void traceParticle(inout Wavelengths wl, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const bool partition = (pc.flags & kFlagPartition) != 0;
    const bool observer = G.counts.w == 0;
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    const uint ambient = G.counts.z;
    stats.x++;

    Emitted em;
    if (!emitParticle(wl, rng, observer && !partition, em)) return;
    float3 o = em.o, d = em.d;
    uint region = em.region;
    float4 beta = em.beta;
    float scale = em.scale;
    const float hero = wl.hero();
    const uint target = G.target.x;
    int nonDeltaEvents = 0, deltaEvents = 0;
    uint lastBoundary = kGpuNone;

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = intersectWorld(o, d, infinity(), hit);
        if (found && depth == 0 && em.distant) refineDistantHit(d, hit);
        stats.y++;
        uint med = mediumOf(region);
        float seg = found ? hit.t : infinity();
        // Under the caustic partition, connect only at the first non-specular vertex after ≥1 specular one.
        const bool mayConnect = !partition || (nonDeltaEvents == 0 && deltaEvents >= 1);
        if (mediumScatters(med)) {
            float4 sT = sigmaT(med, wl);
            float t = -log(1 - rng.uniform()) / sT.x;
            if (t < seg) {
                float3 p = o + d * t;
                float pdf0 = sT.x * exp(-sT.x * t);
                [ForceUnroll] for (int i = 0; i < NW; ++i)
                    beta[i] *= spectrumEval(g_media[med].props.y, wl.lambda[i]) * exp(-sT[i] * t) / pdf0;
                SurfaceHit mh = (SurfaceHit)0;
                mh.p = p;
                mh.scale = scale;
                float g = g_media[med].g.x;
                if (observer && mayConnect) connectVertex(mh, false, (HitInterface)0, d, g, region, beta, wl, rng);
                ++nonDeltaEvents;
                if (partition) break;  // nothing further can be connected
                if (depth >= maxDepth) break;
                float u2 = rng.uniform();
                float u1 = rng.uniform();
                float3 wi;
                sampleHg(-d, g, u1, u2, wi);
                o = p;
                d = wi;
                if (depth >= rrDepth) {
                    const float q = 0.95;  // any survival probability in (0,1] is unbiased
                    if (rng.uniform() >= q) break;
                    beta *= 1 / q;
                }
                continue;
            }
            if (isinf(seg)) break;
            float pass0 = exp(-sT.x * seg);
            beta *= exp(-sT * seg) / pass0;
        } else if (mediumAbsorbs(med)) {
            if (!found) break;
            [ForceUnroll] for (int i = 0; i < NW; ++i) beta[i] *= exp(-spectrumEval(g_media[med].props.x, wl.lambda[i]) * seg);
        }
        if (!found) {
            if (region != ambient) stats.w++;
            break;
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
        uint surfType = g_optics[b.links.z].a.x;
        if (surfType == kSurfNull) {
            region = other;
            o = offsetOrigin(hit.p, hit.n, d, hit.scale);
            scale = hit.scale;
            if (depth >= 4 * maxDepth) break;
            continue;
        }
        if (surfType == kSurfStainedGlass && passStainedGlass(hit, d, wl, beta, rng)) {
            o = offsetOrigin(hit.p, hit.n, d, hit.scale);
            scale = hit.scale;
            if (depth >= 4 * maxDepth) break;
            continue;
        }
        // A diffuse measurement screen records incident flux and then scatters the reflected
        // fraction. An absorbing detector records flux and terminates the path.
        if (fromFront && hit.boundary == target) {
            int px, py;
            float area;
            if (sensorPixel(hit.pLocal, rng, px, py, area)) deposit(px, py, beta * (1 / area), wl);
        }
        if (surfType == kSurfDetector || surfType == kSurfAbsorber) break;
        HitInterface hi = makeInterface(hit, hero);
        if (!hi.it.optics.isDelta()) {
            if (observer && mayConnect) connectVertex(hit, true, hi, d, 0, region, beta, wl, rng);
            ++nonDeltaEvents;
            if (partition) break;  // later vertices are owned by path tracing
        }
        if (depth >= maxDepth) break;
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
        bool ok = sampleScatter(hi.it, d, u1, u2, u3, kModeImportance, s);
        if (!ok || s.weight <= 0) break;
        float w0 = s.weight;
        beta.x *= s.weight;
        [ForceUnroll] for (int i = 1; i < NW; ++i)
            if (wl.pdf[i] != 0) beta[i] *= secondaryScatterWeight(atWavelength(hi, wl.lambda[i]), d, s, kModeImportance);
        if (s.delta) ++deltaEvents;
        if (s.transmitted) region = other;
        o = offsetOrigin(hit.p, hit.n, s.wi, hit.scale);
        scale = hit.scale;
        d = s.wi;
        if (depth >= rrDepth) {
            // Particles carry flux, so roulette against the per-event throughput.
            float q = min(1.0, w0);
            if (q < 1) {
                if (rng.uniform() >= q) break;
                beta *= 1 / q;
            }
        }
    }
}
