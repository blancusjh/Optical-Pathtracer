// Image textures and data maps read from the texel buffer, and the solid textures of the
// reference (owe/core/shared/textures.hpp), whose patterns match it texel for texel.
#pragma once
#include "owe/core/shared/textures.hpp"
#include "common.slang"
#include "spectrum.slang"

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

// The red channel as stored, without decoding (masks: classification data, not colour), at the
// nearest texel; u across, v down, repeating.
float imageNearestRaw(uint record, float u, float v) {
    uint4 hdr = uint4(g_texels[4 * record], g_texels[4 * record + 1], g_texels[4 * record + 2], g_texels[4 * record + 3]);
    const int W = int(hdr.y), H = int(hdr.z);
    int x = clamp(int(floor((u - floor(u)) * float(W))), 0, W - 1);
    int y = clamp(int(floor((v - floor(v)) * float(H))), 0, H - 1);
    uint i = uint(y) * hdr.y + uint(x);
    if ((hdr.w & 1u) != 0) return asfloat(g_texels[hdr.x + 3 * i]);
    return float(g_texels[hdr.x + i] & 0xFFu) * (1.0 / 255.0);
}

// Data maps (normals, roughness): texels as stored, without sRGB decoding, bilinear and repeating.
float3 imageTexelRaw(uint4 hdr, int x, int y) {
    uint i = uint(y) * hdr.y + uint(x);
    if ((hdr.w & 1u) != 0) {
        uint o = hdr.x + 3 * i;
        return float3(asfloat(g_texels[o]), asfloat(g_texels[o + 1]), asfloat(g_texels[o + 2]));
    }
    uint v = g_texels[hdr.x + i];
    return float3(float(v & 0xFFu), float((v >> 8) & 0xFFu), float((v >> 16) & 0xFFu)) * (1.0 / 255.0);
}
float3 imageBilinearRaw(uint record, float u, float v) {
    uint4 hdr = uint4(g_texels[4 * record], g_texels[4 * record + 1], g_texels[4 * record + 2], g_texels[4 * record + 3]);
    const int W = int(hdr.y), H = int(hdr.z);
    float x = (u - floor(u)) * float(W) - 0.5, y = (v - floor(v)) * float(H) - 0.5;
    float fx = floor(x), fy = floor(y), tx = x - fx, ty = y - fy;
    int x0 = ((int(fx) % W) + W) % W, x1 = (x0 + 1) % W;
    int y0 = ((int(fy) % H) + H) % H, y1 = (y0 + 1) % H;
    return imageTexelRaw(hdr, x0, y0) * ((1 - tx) * (1 - ty)) + imageTexelRaw(hdr, x1, y0) * (tx * (1 - ty)) +
           imageTexelRaw(hdr, x0, y1) * ((1 - tx) * ty) + imageTexelRaw(hdr, x1, y1) * (tx * ty);
}
// A data map's value at a hit, placed like an image texture (word: record | 31 triplanar | 30 uv).
float3 mapValue(uint word, float scale, float3 p, float3 n, float2 uv, bool hasUv) {
    const uint rec = word & 0x3FFFFFFFu;
    if ((word & 0x40000000u) != 0 && hasUv) return imageBilinearRaw(rec, uv.x, uv.y);
    if ((word & 0x80000000u) == 0) return imageBilinearRaw(rec, 0.5 + p.x / scale, 0.5 - p.y / scale);
    float3 a = triplanarWeights(n);
    return imageBilinearRaw(rec, 0.5 + p.y / scale, 0.5 - p.z / scale) * a.x +
           imageBilinearRaw(rec, 0.5 + p.x / scale, 0.5 - p.z / scale) * a.y +
           imageBilinearRaw(rec, 0.5 + p.x / scale, 0.5 - p.y / scale) * a.z;
}

// Mixing weights of the spectra a, b, c at a hit (wavelength independent).
float3 textureWeights(uint kind, float scale, float param, float3 p, float3 n, float2 uv, bool hasUv) {
    if (kind != kTexImage) return proceduralWeights(kind, scale, param, p, n);
    // param: the image record, the mapping in the top bits (31: triplanar, 30: uv; a surface
    // without texture coordinates takes the planar projection).
    const uint bits = asuint(param), rec = bits & 0x3FFFFFFFu;
    float3 c;
    if ((bits & 0x40000000u) != 0 && hasUv) {
        c = imageBilinear(rec, uv.x, uv.y);
    } else if ((bits & 0x80000000u) == 0) {
        c = imageBilinear(rec, 0.5 + p.x / scale, 0.5 - p.y / scale);
    } else {
        float3 a = triplanarWeights(n);
        c = imageBilinear(rec, 0.5 + p.y / scale, 0.5 - p.z / scale) * a.x +
            imageBilinear(rec, 0.5 + p.x / scale, 0.5 - p.z / scale) * a.y +
            imageBilinear(rec, 0.5 + p.x / scale, 0.5 - p.y / scale) * a.z;
    }
    return saturate(c);
}
