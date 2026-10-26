// Frames, transforms and the random-number stream of the kernels. Warps, phase functions and MIS
// weights are the reference's own (owe/core/shared/warps.hpp, compiled here in float); where the
// CPU reference consumes random numbers, this code consumes them in the same order, so a GPU path
// follows the CPU path with the same seed until float rounding makes a decision differ.
#pragma once
#include "owe/core/shared/warps.hpp"

struct Frame {
    float3 s, t, n;
    __init(float3 normal) {
        n = normal;
        orthonormalBasis(n, s, t);
    }
    float3 toLocal(float3 v) { return float3(dot(v, s), dot(v, t), dot(v, n)); }
    float3 toWorld(float3 v) { return s * v.x + t * v.y + n * v.z; }
};

// Rows [R | t] of a rigid transform.
float3 xfPoint(float4 r[3], float3 p) {
    return float3(dot(r[0].xyz, p) + r[0].w, dot(r[1].xyz, p) + r[1].w, dot(r[2].xyz, p) + r[2].w);
}
float3 xfVector(float4 r[3], float3 v) { return float3(dot(r[0].xyz, v), dot(r[1].xyz, v), dot(r[2].xyz, v)); }

uint64_t pack64(uint lo, uint hi) { return (uint64_t(hi) << 32) | uint64_t(lo); }
uint2 unpack64(uint64_t v) { return uint2(uint(v), uint(v >> 32)); }

// PCG32 exactly as the reference's Rng. uniform() draws the same two 32-bit words as the CPU's
// 53-bit uniform and keeps its leading 24 bits, so both backends see the same random numbers up to
// float precision.
struct Rng {
    uint64_t state, inc;

    [mutating] void seed(uint64_t seedValue, uint64_t stream) {
        state = 0;
        inc = (stream << 1u) | 1u;
        nextU32();
        state += seedValue;
        nextU32();
    }
    [mutating] uint nextU32() { return pcg32Next(state, inc); }
    [mutating] float uniform() {
        uint a = nextU32();
        nextU32();
        return float(a >> 8) * (1.0 / 16777216.0);
    }
};

// Uniforms for one sampling call in the reference's order (Draw2, Draw3: the last one first).
float2 draw2(inout Rng rng) {
    float u2 = rng.uniform();
    float u1 = rng.uniform();
    return float2(u1, u2);
}
float3 draw3(inout Rng rng) {
    float u3 = rng.uniform();
    float u2 = rng.uniform();
    float u1 = rng.uniform();
    return float3(u1, u2, u3);
}

// Russian roulette with survival probability q (drawing one uniform): false when the path ends;
// a survivor's throughput is divided by q.
bool survives(float q, inout float4 beta, inout Rng rng) {
    if (rng.uniform() >= q) return false;
    beta *= 1 / q;
    return true;
}
