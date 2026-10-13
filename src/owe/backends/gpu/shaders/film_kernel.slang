// Accepted pass accumulation. Pixel sums stay on the device until resolve(); particle
// bins remain exact during transport, then enter a compensated float sum per channel.
#include "scene.slang"

struct FilmPixel {
    float4 camera, cameraCompensation;
    float4 light, lightCompensation;
};
[[vk::binding(27, 0)]] RWStructuredBuffer<FilmPixel> g_film;
groupshared uint4 groupStats[256];

void accumulate(inout float4 sum, inout float4 compensation, float4 value) {
    float4 y = value - compensation;
    float4 t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
}

// Global diagnostic counters use two 32-bit atomic words, including carry.
void addCounter(uint index, uint64_t value) {
    uint old;
    uint low = uint(value);
    InterlockedAdd(g_counters[index], low, old);
    InterlockedAdd(g_counters[index + 1], uint(value >> 32) + uint(old > 0xFFFFFFFFu - low));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    uint pixel = tid.x;
    uint4 stats = 0;
    if (pixel < pc.width * pc.height) {
        FilmPixel film = g_film[pixel];
        if ((pc.flags & 1u) != 0) {
            PixelState state = g_pixels[pixel];
            accumulate(film.camera, film.cameraCompensation, float4(state.acc.xyz, 0));
            stats = state.stats;
        }
        if ((pc.flags & 2u) != 0) {
            // Add bins from small to large with compensation, without a lossy intermediate sum.
            for (uint bin = 0; bin < 4; ++bin) {
                float4 value = 0;
                for (uint channel = 0; channel < 3; ++channel) {
                    uint base = pixel * 24 + channel * 8 + bin * 2;
                    value[channel] = float(g_splat[base]) * exp2(float(-80 + 32 * int(bin))) +
                                     float(g_splat[base + 1]) * exp2(float(-48 + 32 * int(bin)));
                }
                accumulate(film.light, film.lightCompensation, value);
            }
        }
        g_film[pixel] = film;
    }
    groupStats[lane] = stats;
    GroupMemoryBarrierWithGroupSync();
    // One atomic per workgroup and counter rather than one contended atomic per pixel.
    if (lane < 4) {
        uint64_t total = 0;
        for (uint i = 0; i < 256; ++i) total += uint64_t(groupStats[i][lane]);
        addCounter(lane * 2, total);
    }
}
