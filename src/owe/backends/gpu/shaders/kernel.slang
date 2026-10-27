// What every kernel does around its paths: a path's random stream, the pixels' running sums and
// the pass's statistics.
#pragma once
#include "common.slang"
#include "scene.slang"

// The PCG32 stream of path `index` in this pass: one per (seed ^ salt, index, pass), so a result
// depends neither on how a pass is split into dispatches nor on the order threads run.
Rng pathStream(uint64_t salt, uint64_t index) {
    Rng rng;
    rng.seed(hashCombine(pack64(pc.seedLo, pc.seedHi) ^ salt, index), uint64_t(pc.pass));
    return rng;
}

// Adds c to a pixel's running sums. Compensated summation keeps a long pass exact to float
// precision; a value that is not finite is dropped, never poisoning the accumulation.
void accumulate(inout PixelState ps, float3 c) {
    if (!isfinite(c.x + c.y + c.z)) return;
    precise float3 y = c - ps.comp.xyz;
    precise float3 t = ps.acc.xyz + y;
    ps.comp.xyz = (t - ps.acc.xyz) - y;
    ps.acc.xyz = t;
}

// Adds x to the 64-bit counter at words (index, index + 1) of buf.
void add64(RWStructuredBuffer<uint> buf, uint index, uint x) {
    if (x == 0) return;
    uint old;
    InterlockedAdd(buf[index], x, old);
    if (old > 0xFFFFFFFFu - x) InterlockedAdd(buf[index + 1], 1u);
}

// Statistics of one-thread-per-path kernels (paths, segments, inconsistencies, leaks): summed in
// the group, then one global addition per group and counter.
groupshared uint gs_stats[4];
void beginGroupStats(uint lid) {
    if (lid < 4) gs_stats[lid] = 0;
    GroupMemoryBarrierWithGroupSync();
}
void endGroupStats(uint4 stats, uint lid) {
    [ForceUnroll] for (int k = 0; k < 4; ++k)
        if (stats[k] != 0) InterlockedAdd(gs_stats[k], stats[k]);
    GroupMemoryBarrierWithGroupSync();
    if (lid < 4) add64(g_counters, lid * 2, gs_stats[lid]);
}
