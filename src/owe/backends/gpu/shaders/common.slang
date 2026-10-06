// Math, random numbers and warps. A float port of math.hpp and sampling.hpp; where the CPU
// reference consumes random numbers, this code consumes them in the same order, so a GPU path
// follows the CPU path with the same seed until float rounding makes a decision differ.
#pragma once

static const float kPi = 3.14159265358979323846;
static const float kInvPi = 0.31830988618379067154;

float infinity() { return asfloat(0x7f800000u); }
float sqr(float x) { return x * x; }
float safeSqrt(float x) { return sqrt(max(0.0, x)); }
float maxAbsComponent(float3 a) { return max(max(abs(a.x), abs(a.y)), abs(a.z)); }
float copySign(float mag, float sgn) { return asfloat((asuint(mag) & 0x7fffffffu) | (asuint(sgn) & 0x80000000u)); }
float atanhf(float x) { return 0.5 * log((1.0 + x) / (1.0 - x)); }
// IEEE remainder of x by y (result in [-y/2, y/2]).
float remainderf(float x, float y) { return x - y * round(x / y); }

float3 reflectDir(float3 d, float3 n) { return d - 2.0 * dot(d, n) * n; }

// Orthonormal basis around a unit vector (Duff et al. 2017).
void orthonormalBasis(float3 n, out float3 b1, out float3 b2) {
    float sgn = copySign(1.0, n.z);
    float a = -1.0 / (sgn + n.z);
    float b = n.x * n.y * a;
    b1 = float3(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    b2 = float3(b, sgn + n.y * n.y * a, -n.y);
}

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

// ---------------------------------------------------------------- hashing and PCG32

uint64_t mixBits(uint64_t v) {
    v ^= v >> 31;
    v *= 0x7fb5d329728ea185ULL;
    v ^= v >> 27;
    v *= 0x81dadef4bc2dd44dULL;
    v ^= v >> 33;
    return v;
}
uint64_t hashCombine(uint64_t a, uint64_t b) { return mixBits(a ^ (mixBits(b) + 0x9e3779b97f4a7c15ULL)); }

uint64_t pack64(uint lo, uint hi) { return (uint64_t(hi) << 32) | uint64_t(lo); }
uint2 unpack64(uint64_t v) { return uint2(uint(v), uint(v >> 32)); }

// PCG32 exactly as sampling.hpp. uniform() draws the same two 32-bit words as the CPU's 53-bit
// uniform and keeps its leading 24 bits, so both backends see the same random numbers up to float
// precision.
struct Rng {
    uint64_t state, inc;

    [mutating] void seed(uint64_t seedValue, uint64_t stream) {
        state = 0;
        inc = (stream << 1u) | 1u;
        nextU32();
        state += seedValue;
        nextU32();
    }
    [mutating] uint nextU32() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint xorshifted = uint(((old >> 18u) ^ old) >> 27u);
        uint rot = uint(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }
    [mutating] float uniform() {
        uint a = nextU32();
        nextU32();
        return float(a >> 8) * (1.0 / 16777216.0);
    }
};

// ---------------------------------------------------------------- warps

float2 sampleUniformDiskConcentric(float u1, float u2) {
    float a = 2 * u1 - 1, b = 2 * u2 - 1;
    if (a == 0 && b == 0) return float2(0, 0);
    float r, phi;
    if (abs(a) > abs(b)) { r = a; phi = (kPi / 4) * (b / a); }
    else { r = b; phi = kPi / 2 - (kPi / 4) * (a / b); }
    return float2(r * cos(phi), r * sin(phi));
}

// Two independent standard normal deviates (Box–Muller).
float2 sampleStandardNormal2(float u1, float u2) {
    float r = sqrt(-2 * log(1 - u1)), phi = 2 * kPi * u2;
    return float2(r * cos(phi), r * sin(phi));
}

float3 sampleCosineHemisphere(float u1, float u2) {
    float2 d = sampleUniformDiskConcentric(u1, u2);
    float z = safeSqrt(1 - d.x * d.x - d.y * d.y);
    return float3(d.x, d.y, z);
}

float3 sampleUniformSphere(float u1, float u2) {
    float z = 1 - 2 * u1;
    float r = safeSqrt(1 - z * z);
    float phi = 2 * kPi * u2;
    return float3(r * cos(phi), r * sin(phi), z);
}

float3 sampleUniformCone(float u1, float u2, float cosMax) {
    float cosT = (1 - u1) + u1 * cosMax;
    float sinT = safeSqrt(1 - cosT * cosT);
    float phi = 2 * kPi * u2;
    return float3(cos(phi) * sinT, sin(phi) * sinT, cosT);
}

float powerHeuristic(float fPdf, float gPdf) {
    float f = fPdf * fPdf, g = gPdf * gPdf;
    if (isinf(f)) return 1;
    if (f + g == 0) return 0;
    return f / (f + g);
}

float hgPhase(float cosTheta, float g) {
    float denom = 1 + g * g + 2 * g * cosTheta;
    return (1 - g * g) / (4 * kPi * denom * sqrt(max(denom, 1e-30)));
}
float sampleHg(float3 wo, float g, float u1, float u2, out float3 wi) {
    float cosTheta;
    if (abs(g) < 1e-3) cosTheta = 1 - 2 * u1;
    else cosTheta = -1 / (2 * g) * (1 + g * g - sqr((1 - g * g) / (1 + g - 2 * g * u1)));
    float sinTheta = safeSqrt(1 - cosTheta * cosTheta);
    float phi = 2 * kPi * u2;
    Frame f = Frame(wo);
    wi = f.toWorld(float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta));
    return hgPhase(dot(wo, wi), g);
}
