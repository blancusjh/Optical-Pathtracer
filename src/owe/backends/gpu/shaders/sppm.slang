// Stochastic progressive photon mapping (Hachisuka & Jensen 2009), the `sppm` integrator.
//
// One iteration:
//  1. camera pass (sppm_camera_kernel): one camera path per pixel follows specular events (smooth
//     glass, mirrors, smooth metal, with Fresnel branching and Beer–Lambert absorption) to its first
//     non-specular surface, the visible point. Emission met along the specular chain, and the direct
//     light at the visible point (next-event estimation and one scatter-sampled segment, combined by
//     MIS), are the camera pass's: it is the path tracer (stepPath) ending after that segment. A path
//     that scatters in a medium first is path traced to the end instead (no visible point).
//  2. the visible points enter a multi-level spatial hash (sppm_grid_kernel): a point of radius r
//     lives in one cell of size 2^level ≥ 2r, so a photon finds every point within reach in the 2×2×2
//     cells about it on each level in use, whatever the spread of radii (a desk and a mountain).
//  3. photons (sppm_photon_kernel) leave the lights by power (sun photons guided by where earlier
//     photons proved useful, particle.slang) and deposit at every non-specular
//     surface after at least one scattering event (direct light belongs to the camera pass, so nothing
//     is counted twice). A deposit adds β_vp · f_vp(wo, wi) · β_photon to each visible point within
//     its radius, on a disc in the point's tangent plane with a matching surface orientation (no
//     leakage through thin walls or onto nearby parallel surfaces).
//  4. each pixel updates its radius and flux (sppm_update_kernel): N' = N + αM, r' = r·√(N'/(N+M)),
//     τ' = (τ + Φ)·(r'/r)². The estimate is L = L_direct/iterations + τ / (N_emitted·π·r²).
//
// Spectra: an iteration's camera paths and photons are split into wavelength groups; within
// a group they share four wavelengths (a hero rotation over iterations), so a photon and a visible
// point seen through glass merge only within their group and agree on λ exactly; a visible point the
// eye sees directly merges every photon at the photon's own wavelengths. A dispersive event on either
// side leaves the hero alone, and dispersion is exact on both sides of the diffuse surface.
// Accumulation into the visible points uses the exact fixed-point bins of particle splats, so images
// are bitwise reproducible like the other integrators'.
#pragma once
#define OWE_PHOTON_GUIDE 1
#include "particle.slang"

struct VisiblePoint {
    float4 p;     // camera-relative position; w: gather radius
    float4 n;     // geometric normal (toward the front region); w: boundary (bits)
    float4 d;     // direction of the camera ray arriving at the point; w: flags (bits)
    float4 beta;  // camera path throughput with the detector's weight, before the point's scattering
    float4 tex;   // texture weights at the point
};
// Written by the camera pass (transport.slang's stepPath under OWE_SPPM).
static const uint kVpValid = 1u, kVpTerminated = 2u, kVpFree = 4u, kVpSmooth = 8u;  // flags; bits 8–15: wavelength group

struct SppmPixel {
    float4 tau;   // xyz: accumulated flux τ (XYZ); w: N, the accumulated (fractional) photon count
    float4 misc;  // x: gather radius (0 until the pixel's first visible point)
};

[[vk::binding(16, 0)]] RWStructuredBuffer<VisiblePoint> g_vp;
[[vk::binding(17, 0)]] RWStructuredBuffer<SppmPixel> g_sppm;
// Per pixel: this iteration's flux as XYZ exact fixed-point bins (as particle splats), then M.
[[vk::binding(18, 0)]] RWStructuredBuffer<uint> g_flux;
[[vk::binding(19, 0)]] RWStructuredBuffer<uint> g_gridHead;  // hash bucket → first visible point
[[vk::binding(20, 0)]] RWStructuredBuffer<uint> g_gridNext;  // visible point → next in its bucket
[[vk::binding(21, 0)]] RWStructuredBuffer<int> g_levels;  // [0], [1]: lowest and highest hash level in use

static const uint kFluxWords = 3 * kSplatBins * 2 + 2;  // XYZ bins, M, pad
static const uint kFluxCount = 3 * kSplatBins * 2;

float iterationRotation() { return pc.wavelengthRotation; }
uint gridMask() { return pc.gridMask; }

// Wavelength groups: an iteration's paths and photons are split into groups (pc.wavelengthGroups) whose hero
// wavelengths interleave (group g rotates by g/(4·groups), between the four stratified wavelengths
// of a set), so the union covers the spectrum evenly. A visible point seen through glass (its own
// wavelengths fixed by dispersion on the way) takes the photons of its group; one the eye sees
// directly takes every photon, at the photon's wavelengths: 16 spectral samples per iteration where
// a single shared set had one, at the price of 1/groups of the photons for points seen through glass.
uint wavelengthGroups() { return max(pc.wavelengthGroups, 1u); }
float groupRotation(uint g) {
    float u = iterationRotation() + float(g) / float(4 * wavelengthGroups());
    return u - floor(u);
}
float sppmAlpha() { return pc.photonAlpha; }
float initialRadiusPerMetre() { return pc.initialRadiusPerMetre; }

// ---------------------------------------------------------------- the hash grid

int gridLevel(float r) { return int(ceil(log2(max(2 * r, 1e-30)))); }
float cellSize(int level) { return exp2(float(level)); }
int3 cellOf(float3 p, float size) { return int3(clamp(floor(p / size), -2.0e9, 2.0e9)); }
uint gridHash(int3 c, int level) {
    uint h = (uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u) ^ (uint(level) * 2654435761u);
    return h & gridMask();
}

// ---------------------------------------------------------------- photons

// This iteration's flux in a pixel's fixed-point bins (addFixedPoint's).
float fluxRead(uint base) {
    float v = 0;
    [ForceUnroll] for (int b = 0; b < kSplatBins; ++b) {
        uint lo = g_flux[base + uint(b) * 2], hi = g_flux[base + uint(b) * 2 + 1];
        if ((lo | hi) != 0) v += ldexp(float(hi) * 4294967296.0 + float(lo), float(kSplatExp0 + 32 * b));
    }
    return v;
}

// A photon arriving at a non-specular surface point along d with throughput beta: adds its
// contribution to every visible point within reach.
// Returns whether the photon reached any visible point (for photon guiding).
bool gatherPhoton(SurfaceHit hit, float3 d, float4 beta, Wavelengths wl, uint group) {
    bool useful = false;
    const int lo = g_levels[0], hi = g_levels[1];
    if (lo > hi) return false;  // no visible points in this iteration
    const float3 wi = -d;
    for (int level = lo; level <= hi; ++level) {
        const float size = cellSize(level);
        const int3 base = cellOf(hit.p - 0.5 * size, size);
        for (int k = 0; k < 8; ++k) {
            const int3 c = base + int3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
            for (uint i = g_gridHead[gridHash(c, level)]; i != kGpuNone; i = g_gridNext[i]) {
                // Position and radius first (16 bytes); the rest of the point only when it is in reach.
                const float4 vp = g_vp[i].p;
                const float r = vp.w;
                const float3 dp = hit.p - vp.xyz;
                if (dot(dp, dp) > r * r) continue;
                // The point's own cell (buckets are shared by hash collisions; this counts it once).
                if (gridLevel(r) != level || any(cellOf(vp.xyz, size) != c)) continue;
                VisiblePoint v = g_vp[i];
                // A disc in the point's tangent plane, on a surface facing the same way.
                if (abs(dot(dp, v.n.xyz)) > 0.25 * r || dot(v.n.xyz, hit.n) < 0.5) continue;
                const uint flags = asuint(v.d.w);
                const bool free = (flags & kVpFree) != 0;
                if (!free && ((flags >> 8) & 0xFFu) != group) continue;  // wavelengths must agree
                Wavelengths w = wl;  // the photon's (a free point is evaluated at them)
                if (!free && (flags & kVpTerminated) != 0) w.terminateSecondary();
                const float3 ns = (flags & kVpSmooth) != 0 ? octDecode(asuint(v.tex.w)) : v.n.xyz;
                float pdf;
                float4 f = evalScatterSpectral(cachedInterface(asuint(v.n.w), v.n.xyz, ns, v.p.xyz, v.tex.xyz, w.lambda.x), w,
                                               v.d.xyz, wi, kModeRadiance, pdf);
                // A shading normal: the camera's cosine |wi·ns| over the photons' density per
                // geometric area, which carries |wi·n|.
                if ((flags & kVpSmooth) != 0) {
                    const float cn = abs(dot(wi, v.n.xyz));
                    f *= cn > 0 ? abs(dot(wi, ns)) / cn : 0.0;
                }
                // A point bound to its group sees only that group's photons, 1/groups of them: its share
                // is scaled back up, so both kinds of point estimate the same flux.
                float4 value = maskTerminated(w, (free ? float4(v.beta.x) : v.beta * float(wavelengthGroups())) * f * beta);
                if (isZero(value)) continue;
                const uint fb = i * kFluxWords;
                addFixedPointXYZ(g_flux, fb, w.toXYZ(value, globals().misc.y));
                InterlockedAdd(g_flux[fb + kFluxCount], 1u);
                useful = true;
            }
        }
    }
    return useful;
}

// Emits one photon and gathers it at every non-specular surface after its first scattering event.
void traceSppmPhoton(inout Wavelengths wl, uint group, inout Rng rng, inout uint4 stats) {
    GGlobals G = globals();
    const int maxDepth = G.limits.x, rrDepth = G.limits.y;
    stats.x++;
    Emitted em;
    if (!emitParticle(wl, rng, false, em)) return;
    PathRay ray = em.ray;
    float4 beta = em.beta;
    bool useful = false;
    int events = 0;  // scattering events so far (direct light is the camera pass's)
    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = traceFrom(em, ray, depth, hit);
        stats.y++;
        uint med = mediumOf(ray.region);
        float t;
        const uint event = sampleMedium(med, found ? hit.t : infinity(), wl, beta, rng, t);
        if (event == kMediumEnd) break;
        if (event == kMediumScatter) {
            if (depth >= maxDepth) break;
            float2 u = draw2(rng);
            float3 wi;
            sampleHg(-ray.d, g_media[med].g.x, u.x, u.y, wi);
            ray.leaveVolume(ray.o + ray.d * t, wi);
            ++events;
            if (depth >= rrDepth && !survives(0.95, beta, rng)) break;
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
        if (c.surface == kSurfDetector || c.surface == kSurfAbsorber) break;
        HitInterface hi = makeInterface(hit, wl.hero());
        if (!hi.it.optics.isDelta() && events >= 1) useful = gatherPhoton(hit, ray.d, beta, wl, group) || useful;
        if (depth >= maxDepth) break;
        terminateIfDispersive(hi, wl, beta);
        ScatterSample s;
        if (!scatterAtSurface(hit, c, hi, ray, wl, beta, rng, kModeImportance, s)) break;
        ++events;
        if (depth >= rrDepth && s.weight < 1 && !survives(s.weight, beta, rng)) break;
    }
    if (useful && em.guideCell != kGpuNone) InterlockedAdd(g_guide[em.guideCell], 1u);
}
