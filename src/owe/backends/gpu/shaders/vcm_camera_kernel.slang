// VCM camera pass (vcm.slang): one thread per pixel and one camera path per iteration; its
// contributions (every strategy but light tracing, which splats) accumulate in g_pixels over the
// iterations of a pass.
#include "vcm.slang"

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
    Rng rng;
    rng.seed(hashCombine(pack64(pc.seedLo, pc.seedHi) ^ 0x2b992ddfa23249d6ULL, uint64_t(pixel)), uint64_t(pc.pass));
    const uint group = pixelGroup(pixel);
    Wavelengths wl = groupWavelengths(group);
    float px = float(tid.x) + rng.uniform();
    float py = float(tid.y) + rng.uniform();
    float2 f = filterOffset(rng);
    uint4 stats = ps.stats;
    float3 c = traceCameraVcm(px + f.x, py + f.y, wl, group, rng, stats);
    if (isfinite(c.x + c.y + c.z)) {
        precise float3 y = c - ps.comp.xyz;
        precise float3 t = ps.acc.xyz + y;
        ps.comp.xyz = (t - ps.acc.xyz) - y;
        ps.acc.xyz = t;
    }
    ps.stats = stats;
    g_pixels[pixel] = ps;
}
