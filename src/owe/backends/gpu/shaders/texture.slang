// Solid textures in boundary-local coordinates: float port of Texture (world.cpp). The lattice
// hash is the CPU's 64-bit hash, so patterns match the reference texel for texel.
#pragma once
#include "common.slang"
#include "spectrum.slang"

float valueNoise3(float3 p) {
    float3 f = floor(p);
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    float3 t = p - f;
    t = t * t * (3 - 2 * t);
    const uint64_t C = 0x9e3779b97f4a7c15ULL;
    uint64_t my0 = mixBits(uint64_t(uint(iy))) + C, my1 = mixBits(uint64_t(uint(iy + 1))) + C;
    uint64_t mz0 = mixBits(uint64_t(uint(iz))) + C, mz1 = mixBits(uint64_t(uint(iz + 1))) + C;
    float r = 0;
    for (int c = 0; c < 8; ++c) {
        int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
        float w = (dx != 0 ? t.x : 1 - t.x) * (dy != 0 ? t.y : 1 - t.y) * (dz != 0 ? t.z : 1 - t.z);
        uint64_t kxy = mixBits(uint64_t(uint(ix + dx)) ^ (dy != 0 ? my1 : my0));
        uint64_t k = mixBits(kxy ^ (dz != 0 ? mz1 : mz0));
        r += w * (float(uint(k >> 40)) * (1.0 / 16777216.0));
    }
    return r;
}

float fbm3(float3 p, int octaves) {
    float sum = 0, amp = 0.5, norm = 0;
    float3 q = p;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * valueNoise3(q);
        norm += amp;
        amp *= 0.5;
        q = q * 2.03 + float3(17.1, 5.3, 9.7);
    }
    return sum / norm;
}

// Value noise and its exact gradient (the smoothstep fade is C¹, its derivative 6t(1 − t)).
float valueNoise3Grad(float3 p, out float3 grad) {
    float3 f = floor(p);
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    float3 u = p - f;
    float3 t = u * u * (3 - 2 * u), dt = 6 * u * (1 - u);
    const uint64_t C = 0x9e3779b97f4a7c15ULL;
    uint64_t my0 = mixBits(uint64_t(uint(iy))) + C, my1 = mixBits(uint64_t(uint(iy + 1))) + C;
    uint64_t mz0 = mixBits(uint64_t(uint(iz))) + C, mz1 = mixBits(uint64_t(uint(iz + 1))) + C;
    float r = 0;
    grad = 0;
    for (int c = 0; c < 8; ++c) {
        int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
        float3 w = float3(dx != 0 ? t.x : 1 - t.x, dy != 0 ? t.y : 1 - t.y, dz != 0 ? t.z : 1 - t.z);
        float3 dw = float3(dx != 0 ? dt.x : -dt.x, dy != 0 ? dt.y : -dt.y, dz != 0 ? dt.z : -dt.z);
        uint64_t kxy = mixBits(uint64_t(uint(ix + dx)) ^ (dy != 0 ? my1 : my0));
        uint64_t k = mixBits(kxy ^ (dz != 0 ? mz1 : mz0));
        float v = float(uint(k >> 40)) * (1.0 / 16777216.0);
        r += w.x * w.y * w.z * v;
        grad += float3(dw.x * w.y * w.z, w.x * dw.y * w.z, w.x * w.y * dw.z) * v;
    }
    return r;
}

// The gradient of a relief's height depth·fbm(p / scale), fbm3's octaves.
float3 reliefGradient(float3 p, float4 relief) {
    float3 q = p / relief.x, sum = 0;
    float amp = 0.5, norm = 0, freq = 1;
    const int octaves = int(relief.z);
    for (int i = 0; i < octaves; ++i) {
        float3 g;
        valueNoise3Grad(q, g);
        sum += g * (amp * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
        q = q * 2.03 + float3(17.1, 5.3, 9.7);
    }
    return sum * (relief.y / (relief.x * norm));
}

static const uint kTexNone = 0, kTexChecker = 1, kTexNoise = 2, kTexRings = 3, kTexTerrain = 4, kTexBands = 5,
                  kTexRadial = 6, kTexMarble = 7, kTexImage = 8;

// ---------------------------------------------------------------- images (image.hpp)

float srgbDecode(uint v) {
    float c = float(v & 0xFFu) * (1.0 / 255.0);
    return c <= 0.04045 ? c * (1.0 / 12.92) : pow((c + 0.055) * (1.0 / 1.055), 2.4);
}
float3 imageTexel(uint4 hdr, int x, int y) {
    uint i = uint(y) * hdr.y + uint(x);
    if ((hdr.w & 1u) != 0) {
        uint o = hdr.x + 3 * i;
        return float3(asfloat(g_texels[o]), asfloat(g_texels[o + 1]), asfloat(g_texels[o + 2]));
    }
    uint v = g_texels[hdr.x + i];
    return float3(srgbDecode(v), srgbDecode(v >> 8), srgbDecode(v >> 16));
}
// Bilinear, repeating: u across, v down (ImageRGB::bilinear).
float3 imageBilinear(uint record, float u, float v) {
    uint4 hdr = uint4(g_texels[4 * record], g_texels[4 * record + 1], g_texels[4 * record + 2], g_texels[4 * record + 3]);
    const int W = int(hdr.y), H = int(hdr.z);
    float x = (u - floor(u)) * float(W) - 0.5, y = (v - floor(v)) * float(H) - 0.5;
    float fx = floor(x), fy = floor(y), tx = x - fx, ty = y - fy;
    int x0 = ((int(fx) % W) + W) % W, x1 = (x0 + 1) % W;
    int y0 = ((int(fy) % H) + H) % H, y1 = (y0 + 1) % H;
    return imageTexel(hdr, x0, y0) * ((1 - tx) * (1 - ty)) + imageTexel(hdr, x1, y0) * (tx * (1 - ty)) +
           imageTexel(hdr, x0, y1) * ((1 - tx) * ty) + imageTexel(hdr, x1, y1) * (tx * ty);
}

// Mixing weights of the spectra a, b, c at a hit (wavelength independent).
float3 textureWeights(uint kind, float scale, float param, float3 p, float3 n, float2 uv, bool hasUv) {
    switch (kind) {
        case kTexChecker: {
            float3 q = p / scale;
            int s = int(floor(q.x + 1e-6)) + int(floor(q.y + 1e-6)) + int(floor(q.z + 1e-6));
            return (s & 1) != 0 ? float3(0, 1, 0) : float3(1, 0, 0);
        }
        case kTexNoise: {
            float t = clamp((fbm3(p / scale, 6) - 0.25) * 2.0, 0.0, 1.0);
            return float3(1 - t, t, 0);
        }
        case kTexRings: {
            float r = sqrt(p.x * p.x + p.z * p.z) / scale + param * fbm3(p / scale, 4);
            float t = 0.5 + 0.5 * sin(2 * kPi * r);
            t = t * t;
            return float3(1 - t, t, 0);
        }
        case kTexBands: {
            float z = p.z / scale + param * (fbm3(float3(p.x, p.y, p.z * 4) / scale, 5) - 0.5);
            float t = 0.5 + 0.5 * sin(2 * kPi * z) * (0.6 + 0.4 * sin(2 * kPi * z * 0.37 + 1.3));
            float mix = clamp(fbm3(float3(0, 0, z * 3.1), 3) * 1.4 - 0.2, 0.0, 1.0);
            return float3((1 - t) * (1 - 0.35 * mix), t * (1 - 0.35 * mix), 0.35 * mix);
        }
        case kTexRadial: {
            float r = sqrt(p.x * p.x + p.y * p.y) / scale;
            float t = clamp(fbm3(float3(r * 3.0, 0.5, 0.5), 6) * 1.6 - 0.3, 0.0, 1.0);
            return float3(1 - t, t, 0);
        }
        case kTexMarble: {
            float3 q = p / scale;
            float v = abs(sin(q.x * 0.9 + q.y * 0.5 + q.z * 0.3 + param * 6 * fbm3(q * 0.7, 6)));
            float t = pow(1 - v, 6);
            float base = 0.94 + 0.06 * fbm3(q * 3.0, 4);
            return float3(base * (1 - t), t, 0);
        }
        case kTexTerrain: {
            float jitter = fbm3(p / scale, 5) - 0.5;
            float slope = 1 - abs(n.z);
            float rock = clamp((slope + 0.3 * jitter - 0.25) * 6, 0.0, 1.0);
            float snow = 0;
            if (param > 0) snow = clamp((p.z - param + 40 * jitter) / 30.0, 0.0, 1.0) * clamp(1 - slope * 1.6, 0.0, 1.0);
            return float3((1 - rock) * (1 - snow), rock * (1 - snow), snow);
        }
        case kTexImage: {
            // param: the image record, the mapping in the top bits (31: triplanar, 30: uv; a
            // surface without texture coordinates takes the planar projection).
            const uint bits = asuint(param), rec = bits & 0x3FFFFFFFu;
            float3 c;
            if ((bits & 0x40000000u) != 0 && hasUv) {
                c = imageBilinear(rec, uv.x, uv.y);
            } else if ((bits & 0x80000000u) == 0) {
                c = imageBilinear(rec, 0.5 + p.x / scale, 0.5 - p.y / scale);
            } else {
                float3 a = pow(abs(n), float3(4));
                a /= max(a.x + a.y + a.z, 1e-30);
                c = imageBilinear(rec, 0.5 + p.y / scale, 0.5 - p.z / scale) * a.x +
                    imageBilinear(rec, 0.5 + p.x / scale, 0.5 - p.z / scale) * a.y +
                    imageBilinear(rec, 0.5 + p.x / scale, 0.5 - p.y / scale) * a.z;
            }
            return clamp(c, 0.0, 1.0);
        }
        default: return float3(1, 0, 0);
    }
}
