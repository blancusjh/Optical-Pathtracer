// SPPM photon pass (sppm.slang): one thread per photon, photons [base, base + sampleCount) of the
// iteration. Each photon has its own PCG32 stream (seed, index, iteration) and the iteration's
// wavelengths; its deposits are exact integer additions, so the result depends neither on how the
// pass is split nor on the order of deposits.
#include "sppm.slang"

static const uint kGroup = 256;

[shader("compute")]
[numthreads(kGroup, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    beginGroupStats(lid.x);
    uint4 stats = 0;
    if (tid.x < pc.sampleCount) {
        uint64_t index = pack64(pc.sampleBase, pc.baseHi) + uint64_t(tid.x);
        Rng rng = pathStream(0x7a3c9e51d2b84f06ULL, index);
        const uint group = uint(index % uint64_t(wavelengthGroups()));
        Wavelengths wl = Wavelengths.sample(groupRotation(group));
        traceSppmPhoton(wl, group, rng, stats);
    }
    endGroupStats(stats, lid.x);
}
