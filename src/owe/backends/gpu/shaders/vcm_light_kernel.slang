// VCM light pass (vcm.slang): one thread per light subpath, subpaths [base, base + sampleCount) of the
// iteration. Each has its own PCG32 stream (seed, index, iteration) and the iteration's wavelengths;
// its splats to the eye are exact integer additions, and the cache order does not change the result
// beyond the connections' random choice.
#include "vcm.slang"

static const uint kGroup = 256;
groupshared uint gs_stats[4];

void addCounter(uint index, uint x) {
    if (x == 0) return;
    uint old;
    InterlockedAdd(g_counters[index], x, old);
    if (old > 0xFFFFFFFFu - x) InterlockedAdd(g_counters[index + 1], 1u);
}

[shader("compute")]
[numthreads(kGroup, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    if (lid.x < 4) gs_stats[lid.x] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint4 stats = 0;
    if (tid.x < pc.sampleCount) {
        uint64_t index = pack64(pc.sampleBase, pc.baseHi) + uint64_t(tid.x);
        Rng rng;
        rng.seed(hashCombine(pack64(pc.seedLo, pc.seedHi) ^ 0x4f1bbcdcbfa53e0bULL, index), uint64_t(pc.pass));
        const uint group = uint(index % uint64_t(vcmGroups()));
        Wavelengths wl = groupWavelengths(group);
        traceLightVcm(wl, group, rng, stats);
    }
    [ForceUnroll] for (int k = 0; k < 4; ++k)
        if (stats[k] != 0) InterlockedAdd(gs_stats[k], stats[k]);
    GroupMemoryBarrierWithGroupSync();
    if (lid.x < 4) addCounter(lid.x * 2, gs_stats[lid.x]);
}
