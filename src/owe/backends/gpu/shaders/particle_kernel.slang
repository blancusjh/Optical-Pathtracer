// Particle kernel (light and hybrid integrators): one thread per particle, particles
// [base, base + sampleCount) of the pass. Each particle has its own PCG32 stream (seed, index,
// pass), so the result does not depend on how a pass is split into dispatches; splats are exact
// (particle.slang), so it does not depend on their order either.
#include "particle.slang"

static const uint kGroup = 64;
groupshared uint gs_stats[4];

// Adds x to the 64-bit counter at words (index, index + 1) of g_counters.
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
        rng.seed(hashCombine(pack64(pc.seedLo, pc.seedHi) ^ 0x5bd1e995ULL, index), uint64_t(pc.pass));
        Wavelengths wl = Wavelengths.sample(rng.uniform());
        traceParticle(wl, rng, stats);
    }
    // One global addition per group and counter: the counters are shared by every thread.
    [ForceUnroll] for (int k = 0; k < 4; ++k)
        if (stats[k] != 0) InterlockedAdd(gs_stats[k], stats[k]);
    GroupMemoryBarrierWithGroupSync();
    if (lid.x < 4) addCounter(lid.x * 2, gs_stats[lid.x]);
}
