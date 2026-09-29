// SPPM: the visible points of this iteration enter the multi-level hash (sppm.slang), and the range
// of levels in use is recorded for the photons.
#include "sppm.slang"

groupshared int gs_lo, gs_hi;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    if (lid.x == 0) {
        gs_lo = 0x7fffffff;
        gs_hi = -0x7fffffff;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint pixel = tid.x;
    if (pixel < pc.width * pc.height) {
        VisiblePoint v = g_vp[pixel];
        if ((asuint(v.d.w) & kVpValid) != 0) {
            const int level = gridLevel(v.p.w);
            uint old;
            InterlockedExchange(g_gridHead[gridHash(cellOf(v.p.xyz, cellSize(level)), level)], pixel, old);
            g_gridNext[pixel] = old;
            InterlockedMin(gs_lo, level);
            InterlockedMax(gs_hi, level);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid.x == 0 && gs_lo <= gs_hi) {
        InterlockedMin(g_levels[0], gs_lo);
        InterlockedMax(g_levels[1], gs_hi);
    }
}
