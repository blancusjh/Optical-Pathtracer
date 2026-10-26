// VCM light pass (vcm.slang): one thread per light subpath, subpaths [base, base + sampleCount) of the
// iteration. Each has its own PCG32 stream (seed, index, iteration) and the iteration's wavelengths;
// its splats to the eye are exact integer additions, and the cache order does not change the result
// beyond the connections' random choice.
#include "vcm.slang"

static const uint kGroup = 256;

[shader("compute")]
[numthreads(kGroup, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    beginGroupStats(lid.x);
    uint4 stats = 0;
    if (tid.x < pc.sampleCount) {
        uint64_t index = pack64(pc.sampleBase, pc.baseHi) + uint64_t(tid.x);
        Rng rng = pathStream(0x4f1bbcdcbfa53e0bULL, index);
        const uint group = uint(index % uint64_t(vcmGroups()));
        Wavelengths wl = groupWavelengths(group);
        traceLightVcm(wl, group, rng, stats);
    }
    endGroupStats(stats, lid.x);
}
