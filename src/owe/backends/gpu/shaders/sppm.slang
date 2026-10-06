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

// Push constants of the SPPM kernels (pc.extra): x wavelength rotation (float), y hash mask,
// z α (float), w initial radius per metre of path (float: radius scale × pixel angle); pc.pad0: the
// smallest initial radius (float).
float iterationRotation() { return asfloat(pc.extra.x); }
uint gridMask() { return pc.extra.y; }

// Wavelength groups: an iteration's paths and photons are split into groups (pc.pad1) whose hero
// wavelengths interleave (group g rotates by g/(4·groups), between the four stratified wavelengths
// of a set), so the union covers the spectrum evenly. A visible point seen through glass (its own
// wavelengths fixed by dispersion on the way) takes the photons of its group; one the eye sees
// directly takes every photon, at the photon's wavelengths: 16 spectral samples per iteration where
// a single shared set had one, at the price of 1/groups of the photons for points seen through glass.
uint wavelengthGroups() { return max(pc.pad1, 1u); }
float groupRotation(uint g) {
    float u = iterationRotation() + float(g) / float(4 * wavelengthGroups());
    return u - floor(u);
}
float sppmAlpha() { return asfloat(pc.extra.z); }
float initialRadiusPerMetre() { return asfloat(pc.extra.w); }

// ---------------------------------------------------------------- the hash grid

int gridLevel(float r) { return int(ceil(log2(max(2 * r, 1e-30)))); }
float cellSize(int level) { return exp2(float(level)); }
int3 cellOf(float3 p, float size) { return int3(clamp(floor(p / size), -2.0e9, 2.0e9)); }
uint gridHash(int3 c, int level) {
    uint h = (uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u) ^ (uint(level) * 2654435761u);
    return h & gridMask();
}

// ---------------------------------------------------------------- flux accumulation

void fluxAdd64(uint index, uint x) {
    if (x == 0) return;
    uint old;
    InterlockedAdd(g_flux[index], x, old);
    if (old > 0xFFFFFFFFu - x) InterlockedAdd(g_flux[index + 1], 1u);
}
// splatChannel (particle.slang) into this iteration's flux bins.
void fluxChannel(uint base, float v) {
    if (!(v > 0) || isinf(v)) return;
    uint bits = asuint(v);
    int e = int((bits >> 23) & 0xFFu);
    if (e == 0) return;
    uint mant = (bits & 0x7FFFFFu) | 0x800000u;
    int p = e - 150 - kSplatExp0;
    if (p < 0) {
        if (p <= -24) return;
        mant >>= uint(-p);
        p = 0;
    }
    int b = p >> 5, r = p & 31;
    if (b >= kSplatBins) return;
    fluxAdd64(base + uint(b) * 2, mant << uint(r));
    uint high = r == 0 ? 0u : mant >> uint(32 - r);
    if (high != 0 && b + 1 < kSplatBins) fluxAdd64(base + uint(b + 1) * 2, high);
}
float fluxRead(uint base) {
    float v = 0;
    [ForceUnroll] for (int b = 0; b < kSplatBins; ++b) {
        uint lo = g_flux[base + uint(b) * 2], hi = g_flux[base + uint(b) * 2 + 1];
        if ((lo | hi) != 0) v += ldexp(float(hi) * 4294967296.0 + float(lo), float(kSplatExp0 + 32 * b));
    }
    return v;
}

// ---------------------------------------------------------------- photons

// The interface at a visible point at wavelength lambda.
Interface visibleInterface(VisiblePoint v, GBoundary b, float lambda) {
    Interface it;
    it.optics = loadOptics(b.links.z);
    it.n = v.n.xyz;
    it.ns = (asuint(v.d.w) & kVpSmooth) != 0 ? octDecode(asuint(v.tex.w)) : it.n;
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
                GBoundary vb = g_boundaries[asuint(v.n.w)];
                float4 f = 0;
                [ForceUnroll] for (int j = 0; j < NW; ++j) {
                    if (w.pdf[j] == 0) continue;
                    float pdf;
                    f[j] = evalScatter(visibleInterface(v, vb, w.lambda[j]), v.d.xyz, wi, kModeRadiance, pdf);
                }
                // A shading normal: the camera's cosine |wi·ns| over the photons' density per
                // geometric area, which carries |wi·n|.
                if ((flags & kVpSmooth) != 0) {
                    const float cn = abs(dot(wi, v.n.xyz));
                    f *= cn > 0 ? abs(dot(wi, octDecode(asuint(v.tex.w)))) / cn : 0.0;
                }
                // A point bound to its group sees only that group's photons, 1/groups of them: its share
                // is scaled back up, so both kinds of point estimate the same flux.
                float4 value = maskTerminated(w, (free ? float4(v.beta.x) : v.beta * float(wavelengthGroups())) * f * beta);
                if (isZero(value)) continue;
                float3 xyz = w.toXYZ(value, globals().misc.y);
                const uint fb = i * kFluxWords;
                fluxChannel(fb, xyz.x);
                fluxChannel(fb + 2 * kSplatBins, xyz.y);
                fluxChannel(fb + 4 * kSplatBins, xyz.z);
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
    const uint ambient = G.counts.z;
    stats.x++;
    Emitted em;
    if (!emitParticle(wl, rng, false, em)) return;
    bool useful = false;
    float3 o = em.o, d = em.d;
    uint region = em.region;
    float4 beta = em.beta;
    const float hero = wl.hero();
    int events = 0;  // scattering events so far (direct light is the camera pass's)
    uint lastBoundary = kGpuNone;
    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = intersectWorld(o, d, infinity(), hit);
        if (found && depth == 0 && em.distant) refineDistantHit(d, hit);
        stats.y++;
        uint med = mediumOf(region);
        float seg = found ? hit.t : infinity();
        if (mediumScatters(med)) {
            float4 sT = sigmaT(med, wl);
            float t = -log(1 - rng.uniform()) / sT.x;
            if (t < seg) {
                float pdf0 = sT.x * exp(-sT.x * t);
                [ForceUnroll] for (int i = 0; i < NW; ++i)
                    beta[i] *= spectrumEval(g_media[med].props.y, wl.lambda[i]) * exp(-sT[i] * t) / pdf0;
                if (depth >= maxDepth) break;
                float u2 = rng.uniform();
                float u1 = rng.uniform();
                float3 wi;
                sampleHg(-d, g_media[med].g.x, u1, u2, wi);
                o = o + d * t;
                d = wi;
                ++events;
                if (depth >= rrDepth) {
                    const float q = 0.95;
                    if (rng.uniform() >= q) break;
                    beta *= 1 / q;
                }
                continue;
            }
            if (isinf(seg)) break;
            beta *= exp(-sT * seg) / exp(-sT.x * seg);
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
            if (depth >= 4 * maxDepth) break;
            continue;
        }
        if (surfType == kSurfStainedGlass && passStainedGlass(hit, d, wl, beta, rng)) {
            o = offsetOrigin(hit.p, hit.n, d, hit.scale);
            if (depth >= 4 * maxDepth) break;
            continue;
        }
        if (surfType == kSurfDetector || surfType == kSurfAbsorber) break;
        HitInterface hi = makeInterface(hit, hero);
        if (!hi.it.optics.isDelta() && events >= 1) useful = gatherPhoton(hit, d, beta, wl, group) || useful;
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
        if (!sampleScatter(hi.it, d, u1, u2, u3, kModeImportance, s) || s.weight <= 0) break;
        float w0 = s.weight;
        beta.x *= s.weight;
        [ForceUnroll] for (int i = 1; i < NW; ++i)
            if (wl.pdf[i] != 0) beta[i] *= secondaryScatterWeight(atWavelength(hi, wl.lambda[i]), d, s, kModeImportance);
        if (s.transmitted) region = other;
        o = offsetOrigin(hit.p, hit.n, s.wi, hit.scale);
        d = s.wi;
        ++events;
        if (depth >= rrDepth) {
            float q = min(1.0, w0);
            if (q < 1) {
                if (rng.uniform() >= q) break;
                beta *= 1 / q;
            }
        }
    }
    if (useful && em.guideCell != kGpuNone) InterlockedAdd(g_guide[em.guideCell], 1u);
}
