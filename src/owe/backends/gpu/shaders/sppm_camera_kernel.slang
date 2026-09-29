// SPPM camera pass (sppm.slang): one thread per pixel and one camera path per iteration: the path
// tracer's step (transport.slang) with the visible-point rule of OWE_SPPM. The path's own radiance
// (emission through specular events, direct light at its visible point) accumulates in g_pixels
// over the iterations of a pass; the visible point goes to g_vp for the photons.
#define OWE_SPPM 1
#include "detector.slang"
#include "transport.slang"
#include "sppm.slang"

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    const uint pixel = tid.y * pc.width + tid.x;
    PixelState ps = g_pixels[pixel];
    if (pc.sampleBase == 0) {  // the pass's first iteration
        ps.acc = 0;
        ps.comp = 0;
        ps.stats = 0;
    }
    // One stream per (pixel, iteration); every path of the iteration shares its wavelengths.
    Rng rng;
    rng.seed(hashCombine(pack64(pc.seedLo, pc.seedHi) ^ 0x3c6ef372fe94f82bULL, uint64_t(pixel)), uint64_t(pc.pass));
    // The pixel's wavelength group this iteration.
    const uint group = uint(mixBits(pack64(pixel, pc.pass) ^ 0x2545f4914f6cdd1dULL) % uint64_t(wavelengthGroups()));
    Wavelengths wl = Wavelengths.sample(groupRotation(group));
    float px = float(tid.x) + rng.uniform();
    float py = float(tid.y) + rng.uniform();
    float2 f = filterOffset(rng);
    VisiblePoint vp = (VisiblePoint)0;
    uint4 stats = ps.stats;
    float3 o, d;
    float weight;
    float3 c = 0;
    if (generateRay(px + f.x, py + f.y, rng, o, d, weight) && weight > 0) {
        PathState path = startPath(o, d, globals().det.x, wl, weight);
        stats.x++;
        while (stepPath(path, rng, stats)) {}
        if (!isZero(path.L)) c = path.wl.toXYZ(path.L * path.weight, globals().misc.y);
        if (path.vpDone) {
            SppmPixel s = g_sppm[pixel];
            if (s.misc.x == 0) {
                // The pixel's first visible point sets its initial radius: a few pixel footprints at
                // the distance the path travelled (unfolded through its specular events).
                s.misc.x = max(initialRadiusPerMetre() * path.pathLength, asfloat(pc.pad0));
                g_sppm[pixel] = s;
            }
            vp.p = float4(path.vpP.xyz, s.misc.x);
            vp.n = float4(path.vpN.xyz, path.vpP.w);
            vp.d = float4(path.vpD.xyz, asfloat(asuint(path.vpN.w) | (group << 8)));
            vp.beta = path.vpBeta;
            vp.tex = path.vpTex;
        }
    }
    g_vp[pixel] = vp;
    if (isfinite(c.x + c.y + c.z)) {
        precise float3 y = c - ps.comp.xyz;
        precise float3 t = ps.acc.xyz + y;
        ps.comp.xyz = (t - ps.acc.xyz) - y;
        ps.acc.xyz = t;
    }
    ps.stats = stats;
    g_pixels[pixel] = ps;
}
