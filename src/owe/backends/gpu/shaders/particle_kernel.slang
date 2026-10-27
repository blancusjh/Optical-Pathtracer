// Particle kernel (light and hybrid integrators): one thread per particle, particles
// [base, base + sampleCount) of the pass. Each particle has its own PCG32 stream (seed, index,
// pass), so the result does not depend on how a pass is split into dispatches; splats are exact
// (particle.slang), so it does not depend on their order either.
#include "particle.slang"

static const uint kGroup = 256;  // as the path kernel: a register budget for more resident threads

[shader("compute")]
[numthreads(kGroup, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    beginGroupStats(lid.x);
    uint4 stats = 0;
    if (tid.x < pc.sampleCount) {
        Rng rng = pathStream(0x5bd1e995ULL, pack64(pc.sampleBase, pc.baseHi) + uint64_t(tid.x));
        Wavelengths wl = Wavelengths.sample(rng.uniform());
        traceParticle(wl, rng, stats);
    }
    endGroupStats(stats, lid.x);
}
