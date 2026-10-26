// The transport engine for camera paths: float port of Tracer::radiance (transport.cpp) with the
// same structure, region bookkeeping, next-event estimation, MIS and random-number order.
#pragma once
#include "owe/core/shared/environment.hpp"
#include "optics.slang"
#include "world.slang"

// ---------------------------------------------------------------- environment

// Environment map (world.cpp: Environment::mapRGB, sampleMap, mapPdf; owe/core/shared/environment.hpp).
float2 envMapUV(float3 dir) {
    GGlobals G = globals();
    return envMapUV(dir, G.up.xyz, G.envE1.xyz, G.envE2.xyz, G.envParams.x);
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
    float st;
    dir = envMapDirection((float(i) + tCol) / float(W), (float(j) + tRow) / float(H), G.up.xyz, G.envE1.xyz, G.envE2.xyz,
                          G.envParams.x, st);
    if (!(st > 0)) return false;
    pdf = envMapPdf((c1 - c0) / total, float(W), float(H), st);
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
    return envMapPdf((g_scalars[row + i + 1] - g_scalars[row + i]) / total, float(W), float(H), st);
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
    float3 a = triplanarWeights(hit.nLocal);
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
        float3 g = xfVector(b.toWorld, reliefGradient(hit.pLocal, h.it.optics.relief.x, h.it.optics.relief.y, int(h.it.optics.relief.z)));
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

// The interface at a cached path vertex (an SPPM visible point, a VCM light vertex), from what the
// cache keeps: boundary, normals, position and texture weights.
HitInterface cachedInterface(uint boundary, float3 n, float3 ns, float3 p, float3 tex, float lambda) {
    GBoundary b = g_boundaries[boundary];
    HitInterface h;
    h.front = b.front;
    h.back = b.back;
    h.it.optics = loadOptics(b.optics);
    h.it.n = n;
    h.it.ns = ns;
    h.it.p = p;
    h.it.nFront = h.it.nBack = 1;
    h.it.lambda = 0;
    h.it.fresnelFloor = globals().misc.z;
    h.it.tex = tex;
    h.it.applyMaterialMap();
    h.it = atWavelength(h, lambda);
    return h;
}

// ---------------------------------------------------------------- lights

// Finite light endpoints retain the sampled face, including mesh sampling-copy triangle slots.
// Infinite lights use the directional shadow path and do not read this metadata.

struct LightSample {
    float3 wi;
    float dist;
    float4 Le;
    float pdf;        // solid angle, including light selection
    uint light;       // its entry in g_lights
    float cosAtLight; // |cos| at an area light's point (1 at infinity)
    Endpoint endpoint;
};

// A direction toward a uniform point of the sun's disc.
float3 sampleSunDirection(float u1, float u2) {
    GGlobals G = globals();
    return Frame(G.sunDir.xyz).toWorld(sampleUniformCone(u1, u2, G.sunDir.w));
}

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
    ls.light = li;
    ls.cosAtLight = 1;
    if (L.kind == 1) {
        ls.wi = sampleSunDirection(u1, u2);
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
    ls.cosAtLight = abs(cosL);
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
// Next-event estimation's pdf of a sky direction (the environment map's, as a light; 0 without one).
float skyDirectPdf(float3 toward) {
    GGlobals G = globals();
    return G.envMap.w != kGpuNone ? g_lights[G.envMap.w].pdf * envMapPdf(toward) : 0.0;
}

float4 sigmaT(uint medium, Wavelengths wl) {
    GMedium m = g_media[medium];
    float4 r;
    r = wl.eval(m.props.x) + wl.eval(m.props.y);
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

// Transmittance along a ray of direction dir from start: to distance dist, or (infinity) out of
// the world, which it must leave from the ambient region.
float4 shadowTransmittance(Endpoint start, float3 dir, float dist, Wavelengths wl) {
    uint finalRegion;
    return transmittanceSpawn(start, dir, dist, wl, finalRegion, endpoint(float3(0), kGpuNone));
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

// The surface end of a finite connection (the camera side: its error scale follows its position).
Endpoint surfaceEnd(SurfaceHit hit, uint region) {
    return endpoint(hit.p, region, maxAbsComponent(hit.p), hit.boundary, hit.n, hit.prim);
}
// The observer's end of a connection: pupil point q in the detector's region.
Endpoint observerEnd(float3 q) { return endpoint(q, globals().det.x, maxAbsComponent(q)); }

// ---------------------------------------------------------------- the steps every path takes

// A ray in flight with its departure anchor: the boundary, face and normal it left (contact
// queries exclude only that face's own root) and the error scale of the frame its origin was
// computed in. A ray from a volume point, the eye or a distant emitter has no departed boundary.
struct PathRay {
    float3 o, d, normal;
    float scale;
    uint region, departed, departedPrimitive;

    // Leaves the surface at hit along dir (the region is the caller's).
    [mutating] void leave(SurfaceHit hit, float3 dir) {
        o = hit.p;
        d = dir;
        departed = hit.boundary;
        departedPrimitive = hit.prim;
        normal = hit.n;
        scale = hit.scale;
    }
    // Leaves volume point x along dir.
    [mutating] void leaveVolume(float3 x, float3 dir) {
        o = x;
        d = dir;
        departed = kGpuNone;
        departedPrimitive = kGpuNone;
    }
    bool trace(out SurfaceHit hit, bool forwardOnly = false) {
        return intersectSpawnWorld(o, d, infinity(), departed, region, normal, scale, departedPrimitive, hit, forwardOnly);
    }
};
PathRay pathRay(float3 o, float3 d, uint region, float scale) {
    PathRay r;
    r.o = o;
    r.d = d;
    r.normal = 0;
    r.scale = scale;
    r.region = region;
    r.departed = kGpuNone;
    r.departedPrimitive = kGpuNone;
    return r;
}

// What the medium of the ray's region does to a segment of length seg (infinity: unbounded): a
// scattering point at t < seg, distance-sampled with the hero's extinction (the secondaries
// reweighted), or passage with Beer–Lambert attenuation. End: the path cannot go on (an unbounded
// segment through absorbing or scattering matter). Tracer::radiance's sampleMedium.
static const uint kMediumPass = 0, kMediumScatter = 1, kMediumEnd = 2;
uint sampleMedium(uint med, float seg, Wavelengths wl, inout float4 beta, inout Rng rng, out float t) {
    t = seg;
    if (mediumScatters(med)) {
        float4 sT = sigmaT(med, wl);
        t = -log(1 - rng.uniform()) / sT.x;
        if (t < seg) {
            float pdf0 = sT.x * exp(-sT.x * t);
            beta *= wl.eval(g_media[med].props.y) * exp(-sT * t) / pdf0;
            return kMediumScatter;
        }
        if (isinf(seg)) return kMediumEnd;
        beta *= exp(-sT * seg) / exp(-sT.x * seg);
    } else if (mediumAbsorbs(med)) {
        if (isinf(seg)) return kMediumEnd;
        beta *= exp(-wl.eval(g_media[med].props.x) * seg);
    }
    return kMediumPass;
}
// Media as extinction only (VCM's paths: those scattering in a medium are the path tracer's).
// False for an unbounded segment through absorbing or scattering matter.
bool attenuate(uint med, float seg, Wavelengths wl, inout float4 beta) {
    if (!mediumAbsorbs(med) && !mediumScatters(med)) return true;
    if (isinf(seg)) return false;
    beta *= exp(-sigmaT(med, wl) * seg);
    return true;
}

// The sides of a boundary met along d. A ray arriving from a region other than the boundary's on
// that side counts as an inconsistency (stats.z; the first one is recorded) and continues from the
// boundary's region.
struct Crossing {
    uint inc, other;  // the region the ray must arrive from, and the one beyond
    bool fromFront;
    uint surface;     // the boundary's surface type
};
// The first region inconsistency of a pass (as TransportStats::first on the CPU), in g_counters:
// word 8 claims it; 9 the boundary, 10 the ray's region, 11 the boundary's region on the side
// met, 12 the boundary the ray left, 13–15 the camera-relative position (float bits); the
// segment: 16–18 its origin, 19–21 its direction, 22 the primitive it left, 23 the one it met.
void noteInconsistency(SurfaceHit hit, PathRay ray, uint boundaryRegion) {
    uint old;
    InterlockedCompareExchange(g_counters[8], 0u, 1u, old);
    if (old != 0) return;
    g_counters[9] = hit.boundary;
    g_counters[10] = ray.region;
    g_counters[11] = boundaryRegion;
    g_counters[12] = ray.departed;
    [ForceUnroll] for (int k = 0; k < 3; ++k) {
        g_counters[13 + k] = asuint(hit.p[k]);
        g_counters[16 + k] = asuint(ray.o[k]);
        g_counters[19 + k] = asuint(ray.d[k]);
    }
    g_counters[22] = ray.departedPrimitive;
    g_counters[23] = hit.prim;
}

Crossing meetBoundary(SurfaceHit hit, inout PathRay ray, inout uint4 stats) {
    GBoundary b = g_boundaries[hit.boundary];
    Crossing c;
    c.fromFront = dot(ray.d, hit.n) < 0;
    c.inc = c.fromFront ? b.front : b.back;
    c.other = c.fromFront ? b.back : b.front;
    c.surface = g_optics[b.optics].a.x;
    if (c.inc != ray.region || hit.ambiguousContact) {
        stats.z++;
        noteInconsistency(hit, ray, c.inc);
        ray.region = c.inc;
    }
    return c;
}

// A null boundary only changes the region; a stained glass may pass the path straight on (its
// straight share, passStainedGlass). True when the ray went on past the boundary.
bool passThrough(SurfaceHit hit, Crossing c, inout PathRay ray, Wavelengths wl, inout float4 beta, inout Rng rng) {
    if (c.surface == kSurfNull) {
        ray.region = c.other;
        ray.leave(hit, ray.d);
        return true;
    }
    if (c.surface == kSurfStainedGlass && passStainedGlass(hit, ray.d, wl, beta, rng)) {
        ray.leave(hit, ray.d);
        return true;
    }
    return false;
}

// Zero the throughput of wavelengths whose pdf was cleared.
float4 maskTerminated(Wavelengths wl, float4 beta) {
    [ForceUnroll] for (int i = 1; i < NW; ++i)
        if (wl.pdf[i] == 0) beta[i] = 0;
    return beta;
}

// At a dispersive dielectric the secondary wavelengths would refract elsewhere: they end, and the
// hero carries the path for all four.
void terminateIfDispersive(HitInterface hi, inout Wavelengths wl, inout float4 beta) {
    if (hi.it.optics.type != kSurfDielectric || wl.secondaryTerminated()) return;
    [ForceUnroll] for (int i = 1; i < NW; ++i)
        if (wl.pdf[i] != 0 && isDispersive(hi.it, atWavelength(hi, wl.lambda[i]))) {
            wl.terminateSecondary();
            beta = maskTerminated(wl, beta);
            return;
        }
}

// f(wo = −d, wi) at every live wavelength; heroPdf: the hero's pdf of sampling wi.
float4 evalScatterSpectral(HitInterface hi, Wavelengths wl, float3 d, float3 wi, uint mode, out float heroPdf) {
    float4 f = 0;
    heroPdf = 0;
    [ForceUnroll] for (int i = 0; i < NW; ++i) {
        if (wl.pdf[i] == 0) continue;
        float pdf;
        f[i] = evalScatter(i == 0 ? hi.it : atWavelength(hi, wl.lambda[i]), d, wi, mode, pdf);
        if (i == 0) heroPdf = pdf;
    }
    return f;
}

// The throughput after scattering sample s, drawn at the hero wavelength, at every live wavelength.
void weighScatter(inout float4 beta, HitInterface hi, Wavelengths wl, float3 d, ScatterSample s, uint mode) {
    beta.x *= s.weight;
    [ForceUnroll] for (int i = 1; i < NW; ++i)
        if (wl.pdf[i] != 0) beta[i] *= secondaryScatterWeight(atWavelength(hi, wl.lambda[i]), d, s, mode);
}

// Scatters at a surface: samples the continuation at the hero wavelength, weighs every wavelength
// and turns the ray (its region too, when transmitted). False when the path ends there.
bool scatterAtSurface(SurfaceHit hit, Crossing c, HitInterface hi, inout PathRay ray, Wavelengths wl, inout float4 beta,
                      inout Rng rng, uint mode, out ScatterSample s) {
    float3 u = draw3(rng);
    if (!sampleScatter(hi.it, ray.d, u.x, u.y, u.z, mode, s) || s.weight <= 0) return false;
    weighScatter(beta, hi, wl, ray.d, s, mode);
    if (s.transmitted) ray.region = c.other;
    ray.leave(hit, s.wi);
    return true;
}

// ---------------------------------------------------------------- camera paths

// A next-event estimate waiting for its shadow ray: value (the path throughput already applied) is
// added when the segment from start along d, to dist, is clear. Tracing it after the scatter sample
// has been drawn keeps the surface interface out of the live state during the shadow traversal
// (register pressure); the shadow ray draws no random numbers, so the streams are unchanged.
struct ShadowRay {
    Endpoint start, light;
    float3 d;
    float dist;
    float4 value;
};

ShadowRay prepareDirect(SurfaceHit hit, HitInterface base, float3 d, Wavelengths wl, inout Rng rng, float4 beta) {
    ShadowRay sr = (ShadowRay)0;
    LightSample ls;
    if (!sampleLightFrom(hit.p, wl, rng, ls) || isZero(ls.Le) || ls.pdf <= 0) return sr;
    float pdfScatter;
    float4 f = evalScatterSpectral(base, wl, d, ls.wi, kModeRadiance, pdfScatter);
    if (isZero(f)) return sr;
    sr.start = endpoint(hit, dot(ls.wi, hit.n) > 0 ? base.front : base.back);
    sr.d = ls.wi;
    sr.dist = ls.dist;
    sr.light = ls.endpoint;
    float w = powerHeuristic(ls.pdf, pdfScatter);
    sr.value = beta * f * ls.Le * (scatterCosine(base.it, -d, ls.wi, kModeRadiance) * w / ls.pdf);
    return sr;
}

// Next-event estimation at a medium scattering point x (phase anisotropy g), as a pending shadow ray.
ShadowRay prepareDirectMedium(float3 x, float3 d, float g, Wavelengths wl, inout Rng rng, uint region, float4 beta,
                              float scale) {
    ShadowRay sr = (ShadowRay)0;
    LightSample ls;
    if (!sampleLightFrom(x, wl, rng, ls) || isZero(ls.Le) || ls.pdf <= 0) return sr;
    float ph = hgPhase(dot(-d, ls.wi), g);
    sr.start = endpoint(x, region, scale);
    sr.d = ls.wi;
    sr.dist = ls.dist;
    sr.light = ls.endpoint;
    sr.value = beta * ph * ls.Le * (powerHeuristic(ls.pdf, ph) / ls.pdf);
    return sr;
}

float4 traceShadow(ShadowRay sr, Wavelengths wl) {
    if (isZero(sr.value)) return 0;
    if (isinf(sr.dist)) return sr.value * shadowTransmittance(sr.start, sr.d, sr.dist, wl);
    // A finite connection's surface end takes the error scale of its position.
    Endpoint start = sr.start;
    start.scale = maxAbsComponent(start.p);
    return sr.value * connectionTransmittance(sr.light, start, wl);
}

// A camera path in flight: the state Tracer::radiance keeps across its loop, advanced one segment
// at a time by stepPath (one traversal site, one shadow-ray site: a smaller kernel).
struct PathState {
    PathRay ray;
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
    GGlobals G = globals();
    PathState p = (PathState)0;
    p.ray = pathRay(o, d, region, 0);
    if (G.counts.w != 0) {  // a surface sensor: the ray leaves its boundary
        p.ray.departed = G.target.x;
        p.ray.departedPrimitive = 0;
        p.ray.scale = max(G.sensorSize.x, G.sensorSize.y);
    }
    p.ray.normal = G.sensorNormal.xyz;
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

// Advances a camera path by one segment: one iteration of Tracer::radiance's loop, with the same
// random-number order. False when the path has ended (p.L then holds its radiance).
// stats: paths, segments, inconsistencies, leaks.
bool stepPath(inout PathState p, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    const uint ambient = G.counts.z;
    const bool partition = (pc.flags & kFlagPartition) != 0;
    const bool counts = (pc.flags & kFlagMediumOnly) == 0 || p.mediumScattered;  // this path's light is ours
    const int depth = p.depth++;

    SurfaceHit hit;
    bool found = p.ray.trace(hit, depth == 0 && G.counts.w != 0);
    stats.y++;
    uint med = mediumOf(p.ray.region);
    float t;
    const uint event = sampleMedium(med, found ? hit.t : infinity(), p.wl, p.beta, rng, t);
    if (event == kMediumEnd) return false;
    if (event == kMediumScatter) {
#ifdef OWE_SPPM
        if (p.vpDone) return false;  // light scattered in a medium on the way belongs to the photons
#endif
        if (depth >= maxDepth) return false;
        float3 x = p.ray.o + p.ray.d * t;
        float g = g_media[med].g.x;
        p.mediumScattered = true;
        ShadowRay shadow = prepareDirectMedium(x, p.ray.d, g, p.wl, rng, p.ray.region, p.beta, p.ray.scale);
        float2 u = draw2(rng);
        float3 wi;
        p.prevPdf = sampleHg(-p.ray.d, g, u.x, u.y, wi);
        p.specular = false;
        p.prevP = x;
        noteScatter(p, false);
        p.ray.leaveVolume(x, wi);
        bool alive = depth < rrDepth || survives(min(1.0, maxValue(p.beta)), p.beta, rng);
        p.L += traceShadow(shadow, p.wl);
        return alive;
    }
#ifdef OWE_SPPM
    if (mediumAbsorbs(med) && !mediumScatters(med)) p.chromatic = true;
#endif

    const bool ownedByLightTracing =
        partition && p.nonDeltaEvents == 1 && p.firstIsNonDelta && p.deltaSinceNonDelta >= 1;
    if (!found) {
        if (p.ray.region == ambient && !ownedByLightTracing && counts) {
            float4 sky, sun;
            [ForceUnroll] for (int i = 0; i < NW; ++i) {
                sky[i] = envSky(p.ray.d, p.wl.lambda[i]);
                sun[i] = envSun(p.ray.d, p.wl.lambda[i]);
            }
            float w = (p.specular || isZero(sun)) ? 1.0 : powerHeuristic(p.prevPdf, sunPdf());
            // An environment map is also a light for next-event estimation: MIS.
            float wSky = 1;
            if (G.envMap.w != kGpuNone && !p.specular) wSky = powerHeuristic(p.prevPdf, g_lights[G.envMap.w].pdf * envMapPdf(p.ray.d));
            p.L += p.beta * (sky * wSky + sun * w);
        } else if (p.ray.region != ambient) {
            stats.w++;
        }
        return false;
    }

    const Crossing c = meetBoundary(hit, p.ray, stats);
    if (hit.ambiguousContact) return false;
#ifdef OWE_SPPM
    if (!p.vpDone) p.pathLength += hit.t;
#endif

    // Emission.
    GBoundary b = g_boundaries[hit.boundary];
    if (b.emission != kGpuNone) {
        GEmission e = g_emissions[b.emission];
        if (((c.fromFront && e.front != 0) || (!c.fromFront && e.back != 0)) && !ownedByLightTracing && counts) {
            float w = 1;
            if (!p.specular) {
                float pl = lightPdf(p.prevP, hit);
                if (pl > 0) w = powerHeuristic(p.prevPdf, pl);
            }
            p.L += p.beta * p.wl.eval(e.radiance) * w;
        }
    }

    if (passThrough(hit, c, p.ray, p.wl, p.beta, rng)) return depth < 4 * maxDepth;
#ifdef OWE_SPPM
    if (p.vpDone) return false;  // the visible point's scatter sample has met matter: done
#endif
    if (c.surface == kSurfAbsorber || c.surface == kSurfDetector) return false;
    if (depth >= maxDepth) return false;

    HitInterface hi = makeInterface(hit, p.wl.hero());
    terminateIfDispersive(hi, p.wl, p.beta);
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
        p.vpD = float4(p.ray.d, 0);
        p.vpBeta = p.beta * p.weight;
        p.vpTex = float4(hi.it.tex, asfloat(smooth ? octEncode(hi.it.ns) : 0u));
    }
#endif
    ShadowRay shadow = (ShadowRay)0;
    if (!hi.it.optics.isDelta() && counts) shadow = prepareDirect(hit, hi, p.ray.d, p.wl, rng, p.beta);
    ScatterSample s;
    bool alive = scatterAtSurface(hit, c, hi, p.ray, p.wl, p.beta, rng, kModeRadiance, s);
    if (alive) {
        p.specular = s.delta;
        p.prevPdf = s.pdf;
        p.prevP = hit.p;
        noteScatter(p, s.delta);
        alive = depth < rrDepth || survives(min(1.0, maxValue(p.beta)), p.beta, rng);
    }
    p.L += traceShadow(shadow, p.wl);  // the interface is no longer needed
    return alive;
}
