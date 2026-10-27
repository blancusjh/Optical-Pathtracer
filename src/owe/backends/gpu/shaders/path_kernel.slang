// Path-tracing kernel: one thread per pixel, samples [sampleBase, sampleBase + sampleCount) of the
// pass per dispatch. A pass may be split over several dispatches (short GPU submissions); the
// pixel's random stream and running sums persist in g_pixels, so the result does not depend on how
// a pass is split.
//
// Each sample's path is advanced one segment at a time (stepPath). Starting a lane's next sample
// as soon as its path ends (path regeneration) was ~5% faster with hardware traversal, but the
// interleaved loop is miscompiled by at least one driver (Mesa's Intel compiler loses the device),
// so samples run one after another.
//
// Groups are 16×16 pixels: the driver gives a kernel a register budget from its group size, and
// 256 threads get 128 registers per thread (64 threads get 168), so more threads stay resident to
// hide memory and traversal latency. The small spill this costs is the better trade (~5% on
// hardware traversal, ~7% on software, measured on an RTX 4060).
#include "detector.slang"
#include "kernel.slang"
#include "transport.slang"

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;
    const uint pixel = tid.y * pc.width + tid.x;
    PixelState ps = g_pixels[pixel];
    Rng rng;
    if (pc.sampleBase == 0) {
        // As the CPU reference: one PCG32 stream per (pixel, pass).
        rng = pathStream(0, uint64_t(pixel));
        ps.acc = 0;
        ps.comp = 0;
        ps.stats = 0;
        ps.acc.w = rng.uniform();  // hero-wavelength rotation of this pixel's strata
    } else {
        rng.state = pack64(ps.rng.x, ps.rng.y);
        rng.inc = pack64(ps.rng.z, ps.rng.w);
    }
    for (uint j = 0; j < pc.sampleCount; ++j) {
        const uint s = pc.sampleBase + j;
        // Hero wavelengths are stratified across the pixel's samples in this pass.
        float u = (float(s) + rng.uniform()) / float(pc.spp) + ps.acc.w;
        Wavelengths wl = Wavelengths.sample(u - floor(u));
        float px = float(tid.x) + rng.uniform();
        float py = float(tid.y) + rng.uniform();
        float2 f = filterOffset(rng);
        float3 o, d;
        float weight;
        if (!generateRay(px + f.x, py + f.y, rng, o, d, weight) || weight <= 0) continue;
        PathState path = startPath(o, d, globals().det.x, wl, weight);
        ps.stats.x++;
        while (stepPath(path, rng, ps.stats)) {}
        if (!isZero(path.L)) accumulate(ps, path.wl.toXYZ(path.L * path.weight, globals().misc.y));
    }
    uint2 st = unpack64(rng.state), inc = unpack64(rng.inc);
    ps.rng = uint4(st.x, st.y, inc.x, inc.y);
    g_pixels[pixel] = ps;
}
