// VCM: the iteration's cached light vertices enter the multi-level hash (sppm.slang) at the level of
// the merging radius where they lie, for the camera pass to merge.
#include "vcm.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x, g = i / max(cacheCapacity(), 1u);
    if (g >= vcmGroups() || i - g * cacheCapacity() >= groupCount(g)) return;
    const float3 p = g_lightVertices[i].p.xyz;
    const int level = gridLevel(mergeRadius(p));
    uint old;
    InterlockedExchange(g_gridHead[gridHash(cellOf(p, cellSize(level)), level)], i, old);
    g_gridNext[i] = old;
}
