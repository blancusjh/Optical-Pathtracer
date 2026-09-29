// SPPM photon guiding: the normalised CDF of the guide map's accumulated counts (particle.slang),
// for the next iteration's sun photons. One group of 256 threads, 64 cells each.
#include "sppm.slang"

groupshared float gs_part[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 lid : SV_GroupThreadID) {
    const uint t = lid.x, per = kGuideCells / 256, base = t * per;
    float sum = 0;
    for (uint i = 0; i < per; ++i) sum += float(g_guide[base + i]);
    gs_part[t] = sum;
    GroupMemoryBarrierWithGroupSync();
    // Inclusive scan of the 256 partial sums (Hillis–Steele).
    for (uint off = 1; off < 256; off *= 2) {
        float v = t >= off ? gs_part[t - off] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        gs_part[t] += v;
        GroupMemoryBarrierWithGroupSync();
    }
    const float total = gs_part[255];
    float run = gs_part[t] - sum;
    for (uint i = 0; i < per; ++i) {
        run += float(g_guide[base + i]);
        g_guide[kGuideCells + base + i] = asuint(total > 0 ? run / total : 0.0);
    }
    if (t == 255) g_guide[2 * kGuideCells] = asuint(total);
}
