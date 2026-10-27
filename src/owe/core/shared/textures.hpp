// Solid textures in boundary-local coordinates, shared by the reference and the GPU kernels (see
// prelude.hpp). The lattice hash is 64-bit on both, so patterns match texel for texel; only the
// lattice values' precision differs (53 bits on the CPU, 24 on the GPU).
#pragma once

#include "owe/core/shared/warps.hpp"

OWE_SHARED_BEGIN

// Texture kinds, in the order of Texture::Kind.
OWE_CONST uint kTexNone = 0, kTexChecker = 1, kTexNoise = 2, kTexRings = 3, kTexTerrain = 4, kTexBands = 5,
               kTexRadial = 6, kTexMarble = 7, kTexImage = 8;

#ifdef __SLANG__
OWE_FN Real latticeValue(uint64_t k) { return float(uint(k >> 40)) * (1.0 / 16777216.0); }
#else
OWE_FN Real latticeValue(uint64_t k) { return double(k >> 11) * (1.0 / 9007199254740992.0); }
#endif

// The value at lattice corner c (bit 0: x + 1, bit 1: y + 1, bit 2: z + 1) of the cell at (ix, iy, iz):
// hashCombine(hashCombine(x, y), z). hashCombine mixes its second argument first, so the y and z
// mixes (my0, my1, mz0, mz1) are shared between corners.
OWE_FN Real latticeCorner(int c, int ix, uint64_t my0, uint64_t my1, uint64_t mz0, uint64_t mz1) {
    uint64_t kxy = mixBits(uint64_t(uint(ix + (c & 1))) ^ (((c >> 1) & 1) != 0 ? my1 : my0));
    return latticeValue(mixBits(kxy ^ (((c >> 2) & 1) != 0 ? mz1 : mz0)));
}

// Value noise in [0, 1) with a smoothstep fade; grad receives its exact gradient (the fade is C¹,
// its derivative 6t(1 − t)).
OWE_FN Real valueNoise3Grad(Real3 p, OWE_OUT(Real3) grad) {
    Real3 f = Real3(floor(p.x), floor(p.y), floor(p.z));
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    Real3 u = p - f;
    Real3 t = Real3(u.x * u.x * (3 - 2 * u.x), u.y * u.y * (3 - 2 * u.y), u.z * u.z * (3 - 2 * u.z));
    Real3 dt = Real3(6 * u.x * (1 - u.x), 6 * u.y * (1 - u.y), 6 * u.z * (1 - u.z));
    const uint64_t C = 0x9e3779b97f4a7c15ULL;
    uint64_t my0 = mixBits(uint64_t(uint(iy))) + C, my1 = mixBits(uint64_t(uint(iy + 1))) + C;
    uint64_t mz0 = mixBits(uint64_t(uint(iz))) + C, mz1 = mixBits(uint64_t(uint(iz + 1))) + C;
    Real r = 0;
    grad = Real3(0, 0, 0);
    for (int c = 0; c < 8; ++c) {
        bool dx = (c & 1) != 0, dy = ((c >> 1) & 1) != 0, dz = ((c >> 2) & 1) != 0;
        Real wx = dx ? t.x : 1 - t.x, wy = dy ? t.y : 1 - t.y, wz = dz ? t.z : 1 - t.z;
        Real gx = dx ? dt.x : -dt.x, gy = dy ? dt.y : -dt.y, gz = dz ? dt.z : -dt.z;
        Real v = latticeCorner(c, ix, my0, my1, mz0, mz1);
        r += wx * wy * wz * v;
        grad += Real3(gx * wy * wz, wx * gy * wz, wx * wy * gz) * v;
    }
    return r;
}
OWE_FN Real valueNoise3(Real3 p) {
    Real3 grad;
    return valueNoise3Grad(p, grad);
}

// Fractal sum of value noise, normalised to [0, 1).
OWE_FN Real fbm3(Real3 p, int octaves) {
    Real sum = 0, amp = 0.5, norm = 0;
    Real3 q = p;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * valueNoise3(q);
        norm += amp;
        amp *= 0.5;
        q = q * 2.03 + Real3(17.1, 5.3, 9.7);
    }
    return sum / norm;
}

// The gradient of a relief height depth·fbm(p / scale) of the given octaves.
OWE_FN Real3 reliefGradient(Real3 p, Real scale, Real depth, int octaves) {
    Real3 q = p / scale, sum = Real3(0, 0, 0);
    Real amp = 0.5, norm = 0, freq = 1;
    for (int i = 0; i < octaves; ++i) {
        Real3 g;
        valueNoise3Grad(q, g);
        sum += g * (amp * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
        q = q * 2.03 + Real3(17.1, 5.3, 9.7);
    }
    return sum * (depth / (scale * norm));
}

// Blend weights of the three axis projections of a triplanar mapping: the normal's fourth powers.
OWE_FN Real3 triplanarWeights(Real3 n) {
    Real3 a = Real3(pow(abs(n.x), Real(4)), pow(abs(n.y), Real(4)), pow(abs(n.z), Real(4)));
    Real sum = a.x + a.y + a.z;
    return a / (sum > 0 ? sum : Real(1));
}

// The mixing weights of a procedural texture's spectra a, b, c at local point p with normal n
// (wavelength independent); scale: cell or feature size; param: snow line, or turbulence.
OWE_FN Real3 proceduralWeights(uint kind, Real scale, Real param, Real3 p, Real3 n) {
    if (kind == kTexChecker) {
        Real3 q = p / scale;
        int s = int(floor(q.x + kRelTol)) + int(floor(q.y + kRelTol)) + int(floor(q.z + kRelTol));
        return (s & 1) != 0 ? Real3(0, 1, 0) : Real3(1, 0, 0);
    }
    if (kind == kTexNoise) {
        Real t = saturate((fbm3(p / scale, 6) - 0.25) * 2);
        return Real3(1 - t, t, 0);
    }
    if (kind == kTexRings) {
        Real r = sqrt(p.x * p.x + p.z * p.z) / scale + param * fbm3(p / scale, 4);
        Real t = 0.5 + 0.5 * sin(2 * Pi * r);
        t = t * t;
        return Real3(1 - t, t, 0);
    }
    if (kind == kTexBands) {
        Real z = p.z / scale + param * (fbm3(Real3(p.x, p.y, p.z * 4) / scale, 5) - 0.5);
        Real t = 0.5 + 0.5 * sin(2 * Pi * z) * (0.6 + 0.4 * sin(2 * Pi * z * 0.37 + 1.3));
        Real mix = saturate(fbm3(Real3(0, 0, z * 3.1), 3) * 1.4 - 0.2);
        return Real3((1 - t) * (1 - 0.35 * mix), t * (1 - 0.35 * mix), 0.35 * mix);
    }
    if (kind == kTexRadial) {
        Real r = sqrt(p.x * p.x + p.y * p.y) / scale;
        Real t = saturate(fbm3(Real3(r * 3, 0.5, 0.5), 6) * 1.6 - 0.3);
        return Real3(1 - t, t, 0);
    }
    if (kind == kTexMarble) {
        Real3 q = p / scale;
        Real v = abs(sin(q.x * 0.9 + q.y * 0.5 + q.z * 0.3 + param * 6 * fbm3(q * 0.7, 6)));
        Real t = pow(1 - v, Real(6));  // thin dark veins
        Real base = 0.94 + 0.06 * fbm3(q * 3, 4);
        return Real3(base * (1 - t), t, 0);
    }
    if (kind == kTexTerrain) {
        // Grass on gentle slopes, rock on steep ones, snow above the snow line.
        Real jitter = fbm3(p / scale, 5) - 0.5;
        Real slope = 1 - abs(n.z);
        Real rock = saturate((slope + 0.3 * jitter - 0.25) * 6);
        Real snow = 0;
        if (param > 0) snow = saturate((p.z - param + 40 * jitter) / 30) * saturate(1 - slope * 1.6);
        return Real3((1 - rock) * (1 - snow), rock * (1 - snow), snow);
    }
    return Real3(1, 0, 0);
}

OWE_SHARED_END
