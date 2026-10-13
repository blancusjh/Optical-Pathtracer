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

// A stained glass's mask (raw red, nearest texel; sampleAimNormal.x: threshold) says stone here.
bool stainedIsStone(Optics o, SurfaceHit hit) {
    if (o.conductorK == kGpuNone || !hit.hasUv) return false;
    return imageNearestRaw(o.conductorK, hit.uv.x, hit.uv.y) < o.sampleAimNormal.x;
}
// The straight transmittance of the stained glass at a hit, per wavelength, and the probability pS
// that a path takes it (both 0 where the mask says stone).
float4 stainedStraight(SurfaceHit hit, float3 d, Wavelengths wl, out float pS) {
    pS = 0;
    Optics o = loadOptics(g_boundaries[hit.boundary].optics);
    if (stainedIsStone(o, hit)) return 0;
    float3 tex = o.texKind != kTexNone ? textureWeights(o.texKind, o.texScale, o.texParam, hit.pLocal, hit.nLocal, hit.uv, hit.hasUv)
                                       : float3(1, 0, 0);
    float cosI = abs(dot(d, hit.n));
    pS = stainedProbs(o, tex, cosI).y;
    float keep = 1 - o.sampleAimCenter.z;
    float4 T = 0;
    [ForceUnroll] for (int i = 0; i < NW; ++i)
        if (wl.pdf[i] != 0) T[i] = keep * stainedSlab(o, tex, cosI, wl.lambda[i], kAirIndex).y;
    return T;
}
// At a stained glass, a path passes straight through with probability pS, carrying T/pS: a crossing
// like a null boundary's (no scattering vertex; region unchanged), which keeps it consistent with
// next-event estimation, whose shadow rays take the same T. Otherwise the glass scatters it.
bool passStainedGlass(SurfaceHit hit, float3 d, Wavelengths wl, inout float4 beta, inout Rng rng) {
    float pS;
    float4 T = stainedStraight(hit, d, wl, pS);
    if (!(pS > 0) || rng.uniform() >= pS) return false;
    beta *= T / pS;
    return true;
}

// The shading normal tilted by a tangent-space normal map (OpenGL convention: green toward the top
// of the picture, i.e. decreasing v; DirectX flips it). The tangent frame: the triangle's dP/du and
// −dP/dv for uv mapping; the projection's axes for planar and triplanar mapping, each axis's tilt
// added to the shading normal with the triplanar weights.
float3 tangentToWorld(float3 t, float3 T, float3 U, float3 N) {
    T = T - N * dot(N, T);
    float lt = length(T);
    if (!(lt > 1e-12)) return N;
    T /= lt;
    U = U - N * dot(N, U) - T * dot(T, U);
    float lu = length(U);
    if (!(lu > 1e-12)) return N;
    U /= lu;
    return normalize(T * t.x + U * t.y + N * t.z);
}
float3 normalMapTexel(Optics o, uint rec, float u, float v) {
    float3 t = imageBilinearRaw(rec, u, v) * 2 - 1;
    if (o.mapParams.y != 0) t.y = -t.y;
    t.xy *= o.mapParams.x;
    return t;
}
float3 normalMapped(GBoundary b, SurfaceHit hit, Optics o) {
    const uint word = o.maps.x, rec = word & 0x3FFFFFFFu;
    const float scale = asfloat(o.maps.z);
    const float3 N = hit.ns;
    float3 dpdu, dpdv;
    if ((word & 0x40000000u) != 0 && hit.hasUv && hit.tri != kGpuNone && meshUvFrame(hit.tri, dpdu, dpdv))
        return tangentToWorld(normalMapTexel(o, rec, hit.uv.x, hit.uv.y), xfVector(b.toWorld, dpdu),
                              -xfVector(b.toWorld, dpdv), N);
    const float3 X = xfVector(b.toWorld, float3(1, 0, 0)), Y = xfVector(b.toWorld, float3(0, 1, 0)),
                 Z = xfVector(b.toWorld, float3(0, 0, 1));
    const float3 p = hit.pLocal;
    if ((word & 0x80000000u) == 0) return tangentToWorld(normalMapTexel(o, rec, 0.5 + p.x / scale, 0.5 - p.y / scale), X, Y, N);
    float3 a = pow(abs(hit.nLocal), float3(4));
    a /= max(a.x + a.y + a.z, 1e-30);
    float3 sum = N;
    sum += a.x * (tangentToWorld(normalMapTexel(o, rec, 0.5 + p.y / scale, 0.5 - p.z / scale), Y, Z, N) - N);
    sum += a.y * (tangentToWorld(normalMapTexel(o, rec, 0.5 + p.x / scale, 0.5 - p.z / scale), X, Z, N) - N);
    sum += a.z * (tangentToWorld(normalMapTexel(o, rec, 0.5 + p.x / scale, 0.5 - p.y / scale), X, Y, N) - N);
    return normalize(sum);
}

struct HitInterface {
    Interface it;
    uint front, back;
};

HitInterface makeInterface(SurfaceHit hit, float lambda) {
    GBoundary b = g_boundaries[hit.boundary];
    HitInterface h;
    h.it.p = hit.p;
    h.front = b.front;
    h.back = b.back;
    h.it.optics = loadOptics(b.optics);
    if (h.it.optics.type == kSurfStainedGlass && stainedIsStone(h.it.optics, hit))
        h.it.optics = loadOptics(asuint(h.it.optics.sampleAimNormal.y));  // the mask says stone here
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
    if (h.it.optics.maps.y != kGpuNone) {
        // A roughness map: α is the square of the channel's value (perceptual roughness).
        float3 c = mapValue(h.it.optics.maps.y, asfloat(h.it.optics.maps.w), hit.pLocal, hit.nLocal, hit.uv, hit.hasUv);
        float r = c[int(h.it.optics.mapParams.z)];
        h.it.optics.roughness = r * r;
    }
    if (h.it.optics.maps.x != kGpuNone) {
        float3 mapped = normalMapped(b, hit, h.it.optics);
        if (dot(mapped, hit.n) > 1e-3) h.it.ns = mapped;
    }
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

// Finite light endpoints retain the sampled face, including mesh sampling-copy triangle slots.
// Infinite lights use the directional shadow path and do not read this metadata.

struct LightSample {
    float3 wi;
    float dist;
    float4 Le;
    float pdf;  // solid angle, including light selection
    Endpoint endpoint;
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
    uint sampledPrimitive;
    if (!sampleFrom(b, xfPoint(b.toLocal, p), u1, u2, u3, pL, nL, pdfW, sampledPrimitive)) return false;
    float3 pW = xfPoint(b.toWorld, pL);
    float3 nW = normalize(xfVector(b.toWorld, nL));
    float3 d = pW - p;
    ls.dist = length(d);
    if (!(ls.dist > 0)) return false;
    ls.wi = d / ls.dist;
    GEmission e = g_emissions[b.emission];
    float cosL = dot(nW, -ls.wi);
    ls.endpoint.p = pW;
    ls.endpoint.normal = nW;
    ls.endpoint.scale = b.scale;
    ls.endpoint.region = cosL > 0 ? b.front : b.back;
    ls.endpoint.boundary = L.boundary;
    ls.endpoint.primitive = sampledPrimitive;
    ls.Le = ((cosL > 0 && e.front != 0) || (cosL < 0 && e.back != 0)) ? wl.eval(e.radiance) : float4(0);
    ls.pdf = sel * pdfW;
    return true;
}

float lightPdf(float3 ref, SurfaceHit hit) {
    GBoundary b = g_boundaries[hit.boundary];
    uint li = b.light;
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
// Scalar parameters keep this non-inlined kernel boundary out of Function-storage
// aggregates, which caused severe driver spilling in the particle kernel. Public
// visibility callers use the single Endpoint adapter below.
[noinline]
float4 walkTransmittance(float3 o, uint region, float3 dir, float dist, Wavelengths wl, float scale,
                         out uint finalRegion, uint departed, float3 normal, uint endpointRegion,
                         uint endpointBoundary, float3 endpointNormal, float endpointError,
                         uint departedPrimitive, uint endpointPrimitive, float3 endpointPoint) {
    finalRegion = region;
    float4 Tr = 1;
    float remaining = dist;
    bool explicitEndpoint = endpointRegion != kGpuNone;
    // The segment stops short of its end point (on the light), by more than the float error of the
    // light point and of the ray's hit on the light. `dist` is measured from `o` itself.
    float margin = isinf(dist) ? 0 : (explicitEndpoint ? 0 : 1e-5 * dist + surfaceEpsilon(o, scale));
    // Prove the spawn neighborhood clear before using the normal fast shadow query.
    // A local passable interface requires the ordered walk; opaque contact blocks immediately.
    float3 fastOrigin = o;
    float certifiedAdvance = -1;
    SurfaceHit contact;
    bool pendingContact = false;
    float initialMax = isinf(dist) ? infinity() : dist - margin;
    if (initialMax <= 0) return Tr;
    if (departed != kGpuNone) {
        float window = spawnAdvance(o, normal, dir, scale, initialMax);
        fastOrigin = o + dir * window;
        pendingContact = intersectContactNeighborhood(o, dir, window, departed, region, scale, normal, departedPrimitive, false, contact);
        if (!pendingContact) certifiedAdvance = window;
        if (pendingContact) {
            GBoundary boundary = g_boundaries[contact.boundary];
            uint incident = dot(dir, contact.n) < 0 ? boundary.front : boundary.back;
            if (incident != region || contact.ambiguousContact ||
                !isPassableType(g_optics[boundary.optics].a.x)) return 0;
        }
    }
    if (!pendingContact && !explicitEndpoint) {
        float advance = max(0, dot(fastOrigin - o, dir));
        float tmax = isinf(dist) ? infinity() : initialMax - advance;
        if (tmax <= 0) return Tr;
        bool sawNull;
        if (occluded(fastOrigin, dir, tmax, sawNull)) return 0;
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
        float tmax = isinf(remaining) ? infinity() : (explicitEndpoint ? remaining + endpointError : remaining - margin);
        if (tmax <= 0) { finalRegion = region; return Tr; }
        bool found;
        if (guard == 0 && pendingContact) { h = contact; found = true; }
        else found = intersectSpawnWorld(o, dir, tmax, departed, region, normal, scale, departedPrimitive, h,
                                         false, guard == 0 ? certifiedAdvance : -1);
        float seg = found ? min(h.t, remaining) : remaining;
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
        if (explicitEndpoint && h.boundary == endpointBoundary && !h.ambiguousContact &&
            abs(h.t - remaining) <= endpointError && dot(h.n, endpointNormal) > 0 && region == endpointRegion &&
            (g_boundaries[h.boundary].shape != kShapeMesh || sameMeshFace(h.prim, endpointPrimitive))) {
            finalRegion = region;
            return Tr;
        }
        if (explicitEndpoint && dot(h.p - endpointPoint, dir) > 0) {
            finalRegion = region;
            return Tr;
        }
        GBoundary b = g_boundaries[h.boundary];
        uint incident = dot(dir, h.n) < 0 ? b.front : b.back;
        if (incident != region || h.ambiguousContact) return 0;
        if (g_optics[b.optics].a.x == kSurfStainedGlass) {
            float pS;
            Tr *= stainedStraight(h, dir, wl, pS);
            if (isZero(Tr)) return 0;
            o = h.p;
            departed = h.boundary;
            departedPrimitive = h.prim;
            normal = h.n;
            scale = h.scale;
            if (!isinf(remaining)) remaining -= h.t;
            continue;
        }
        if (g_optics[b.optics].a.x != kSurfNull) return 0;
        region = dot(dir, h.n) < 0 ? b.back : b.front;
        o = h.p;
        departed = h.boundary;
        departedPrimitive = h.prim;
        normal = h.n;
        scale = h.scale;
        if (!isinf(remaining)) remaining -= h.t;
    }
    finalRegion = region;
    return 0;  // guard exhaustion cannot certify the untraced remainder
}
float4 transmittanceSpawn(Endpoint start, float3 dir, float dist, Wavelengths wl,
                          out uint finalRegion, Endpoint destination) {
    return walkTransmittance(start.p, start.region, dir, dist, wl, start.scale, finalRegion,
                             start.boundary, start.normal, destination.region, destination.boundary,
                             destination.normal, destination.error, start.primitive,
                             destination.primitive, destination.p);
}

float4 transmittance(float3 o, uint region, float3 dir, float dist, Wavelengths wl, float scale, out uint finalRegion) {
    return transmittanceSpawn(endpoint(o, region, scale, kGpuNone, float3(0), kGpuNone), dir, dist, wl, finalRegion, endpoint(float3(0), kGpuNone));
}
float4 transmittanceSurface(SurfaceHit hit, uint region, float3 dir, float dist, Wavelengths wl, out uint finalRegion) {
    return transmittanceSpawn(endpoint(hit.p, region, hit.scale, hit.boundary, hit.n, hit.prim), dir, dist, wl, finalRegion, endpoint(float3(0), kGpuNone));
}
float4 transmittanceSurface(SurfaceHit hit, uint region, float3 dir, float dist, Wavelengths wl) {
    uint finalRegion;
    return transmittanceSurface(hit, region, dir, dist, wl, finalRegion);
}
float4 transmittance(float3 o, uint region, float3 dir, float dist, Wavelengths wl, float scale) {
    uint finalRegion;
    return transmittance(o, region, dir, dist, wl, scale, finalRegion);
}

// Finite connections preserve both exact endpoint anchors. There is no scalar exclusion
// interval: only the actual destination face is excluded, after checking its neighbors.

float4 connectionTransmittance(Endpoint source, Endpoint target, Wavelengths wl) {
    float3 delta = source.p - target.p;
    float distance = length(delta);
    if (!(distance > 0)) return 0;
    float3 direction = delta / distance;
    source.error = contactError(source.p, max(distance, source.scale));
    SurfaceHit neighbor;
    if (intersectContactNeighborhood(source.p, -direction, source.error, source.boundary, source.region,
                                     source.scale, source.normal, source.primitive, true, neighbor)) {
        if (neighbor.ambiguousContact || !isPassableType(g_optics[g_boundaries[neighbor.boundary].optics].a.x)) return 0;
    }
    uint arrival;
    float4 tr = transmittanceSpawn(target, direction, distance, wl, arrival, source);
    return arrival == source.region ? tr : float4(0);
}

// A next-event estimate waiting for its shadow ray: value (the path throughput already applied) is
// added when the segment o + t·d, t < dist, is clear. Tracing it after the scatter sample has been
// drawn keeps the surface interface out of the live state during the shadow traversal (register
// pressure); the shadow ray draws no random numbers, so the streams are unchanged.
struct ShadowRay {
    float3 o, d;
    float dist, scale;
    uint region, departed, departedPrimitive;
    float3 normal;
    Endpoint endpoint;
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
    sr.o = hit.p;
    sr.departed = hit.boundary;
    sr.departedPrimitive = hit.prim;
    sr.normal = hit.n;
    sr.d = ls.wi;
    sr.scale = hit.scale;
    sr.dist = ls.dist;
    sr.endpoint = ls.endpoint;
    float w = powerHeuristic(ls.pdf, pdfScatter);
    sr.value = beta * f * ls.Le * (scatterCosine(base.it, -d, ls.wi, kModeRadiance) * w / ls.pdf);
    return sr;
}

float4 traceShadow(ShadowRay sr, Wavelengths wl) {
    if (isZero(sr.value)) return 0;
    if (!isinf(sr.dist)) {
        Endpoint e = sr.endpoint;
        return sr.value * connectionTransmittance(endpoint(e.p, e.region, e.scale, e.boundary, e.normal, e.primitive), endpoint(sr.o, sr.region, maxAbsComponent(sr.o), sr.departed, sr.normal, sr.departedPrimitive), wl);
    }
    uint finalRegion;
    return sr.value * transmittanceSpawn(endpoint(sr.o, sr.region, sr.scale, sr.departed, sr.normal, sr.departedPrimitive), sr.d, sr.dist, wl, finalRegion, endpoint(float3(0), kGpuNone));
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
    uint region, departed, departedPrimitive;
    float3 normal;
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
    bool mediumScattered;  // a medium scattering vertex so far (kFlagMediumOnly counts only such paths)
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
    p.departed = globals().counts.w == 0 ? kGpuNone : globals().target.x;
    p.departedPrimitive = globals().counts.w == 0 ? kGpuNone : 0;
    p.normal = globals().sensorNormal.xyz;
    p.scale = globals().counts.w == 0 ? 0 : max(globals().sensorSize.x, globals().sensorSize.y);
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
    sr.departed = kGpuNone;
    sr.departedPrimitive = kGpuNone;
    sr.d = ls.wi;
    sr.dist = ls.dist;
    sr.endpoint = ls.endpoint;
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
    const bool counts = (pc.flags & kFlagMediumOnly) == 0 || p.mediumScattered;  // this path's light is ours
    const float hero = p.wl.hero();
    const int depth = p.depth++;

    SurfaceHit hit;
    bool found = intersectSpawnWorld(p.o, p.d, infinity(), p.departed, p.region, p.normal, p.scale, p.departedPrimitive, hit, depth == 0 && G.counts.w != 0);
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
            p.mediumScattered = true;
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
            p.departed = kGpuNone;
            p.departedPrimitive = kGpuNone;
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
        if (p.region == ambient && !ownedByLightTracing && counts) {
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
    uint inc = fromFront ? b.front : b.back;
    uint other = fromFront ? b.back : b.front;
    if (inc != p.region || hit.ambiguousContact) {
        stats.z++;
        noteInconsistency(hit.boundary, p.region, inc, p.departed, hit.p);
        p.region = inc;
    }
    if (hit.ambiguousContact) return false;
    uint surfType = g_optics[b.optics].a.x;
#ifdef OWE_SPPM
    if (!p.vpDone) p.pathLength += hit.t;
#endif

    // Emission.
    if (b.emission != kGpuNone) {
        GEmission e = g_emissions[b.emission];
        if (((fromFront && e.front != 0) || (!fromFront && e.back != 0)) && !ownedByLightTracing && counts) {
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
        p.o = hit.p;
        p.departed = hit.boundary;
        p.departedPrimitive = hit.prim;
        p.normal = hit.n;
        p.scale = hit.scale;
        return depth < 4 * maxDepth;
    }
    if (surfType == kSurfStainedGlass && passStainedGlass(hit, p.d, p.wl, p.beta, rng)) {
        p.o = hit.p;
        p.departed = hit.boundary;
        p.departedPrimitive = hit.prim;
        p.normal = hit.n;
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
    if (!hi.it.optics.isDelta() && counts) shadow = prepareDirect(hit, hi, p.d, p.wl, rng, p.region, p.beta);
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
        p.o = hit.p;
        p.departed = hit.boundary;
        p.departedPrimitive = hit.prim;
        p.normal = hit.n;
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
