// Spectral quantities: float port of spectrum.cpp, medium.cpp (index models) and wavelengths.hpp.
// A path carries four hero-stratified wavelengths, so spectral values are float4.
#pragma once
#include "common.slang"
#include "scene.slang"

static const float kLambdaMin = 360.0;
static const float kLambdaMax = 830.0;
static const int NW = 4;

float lobe(float x, float mu, float s1, float s2) {
    float t = (x - mu) / (x < mu ? s1 : s2);
    return exp(-0.5 * t * t);
}
float cieX(float l) {
    return 1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2);
}
float cieY(float l) { return 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1); }
float cieZ(float l) { return 1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8); }

float sampleVisibleWavelength(float u) {
    return clamp(538.0 - 138.888889 * atanhf(0.85691062 - 1.82750197 * u), kLambdaMin, kLambdaMax);
}
float visibleWavelengthPdf(float l) {
    if (l < kLambdaMin || l > kLambdaMax) return 0;
    return 0.0039398042 / sqr(cosh(0.0072 * (l - 538.0)));
}

// Planck spectral radiance B(λ,T) in W·m⁻²·sr⁻¹·m⁻¹ for λ in nm. Evaluated in micrometres
// (c1 = 2hc² = 1.191e8 W·µm⁴·m⁻²·sr⁻¹, c2 = hc/k = 14388 µm·K): in metres λ⁵ ≈ 1e-32, and an
// optimiser that reassociates the constant factors (legal in Vulkan) underflows float to zero.
float planck(float lambdaNm, float kelvin) {
    const float c1 = 1.191042972e8, c2 = 1.438776877e4;
    float l = lambdaNm * 1e-3;
    float l2 = l * l;
    return 1e6 * c1 / (l2 * l2 * l * (exp(c2 / (l * kelvin)) - 1));
}

float logistic(float x) { return 1.0 / (1.0 + exp(-x)); }
float3 rgbBasis(float l) {
    float hiEdge = logistic((l - 590.0) / 11.0);
    float loEdge = logistic((l - 492.0) / 11.0);
    return float3(hiEdge, loEdge - hiEdge, 1.0 - loEdge);
}

static const uint kSpecConstant = 0;
static const uint kSpecBlackbody = 1;
static const uint kSpecTabulated = 2;
static const uint kSpecRgbReflectance = 3;
static const uint kSpecRgbIlluminant = 4;

float tableEval(uint table, uint count, float l) {
    uint vals = table + count;
    if (l <= g_scalars[table]) return g_scalars[vals];
    if (l >= g_scalars[table + count - 1]) return g_scalars[vals + count - 1];
    // First entry greater than l (std::upper_bound).
    uint lo = 0, n = count;
    while (n > 0) {
        uint half = n / 2;
        if (g_scalars[table + lo + half] <= l) { lo += half + 1; n -= half + 1; }
        else n = half;
    }
    uint i = lo;
    float l0 = g_scalars[table + i - 1], l1 = g_scalars[table + i];
    float t = (l - l0) / (l1 - l0);
    return g_scalars[vals + i - 1] * (1 - t) + g_scalars[vals + i] * t;
}

float spectrumEval(uint id, float l) {
    GSpectrum s = g_spectra[id];
    float scale = s.abcs.w;
    switch (s.info.x) {
        case kSpecConstant: return scale * s.abcs.x;
        case kSpecBlackbody: return scale * planck(l, s.abcs.x);
        case kSpecTabulated: return scale * tableEval(s.info.y, s.info.z, l);
        case kSpecRgbReflectance: return scale * dot(s.abcs.xyz, rgbBasis(l));
        case kSpecRgbIlluminant: return scale * dot(s.abcs.xyz, rgbBasis(l)) * planck(l, 6504.0) * 1e-13;
        default: return 0;
    }
}
bool spectrumIsZero(uint id) {
    GSpectrum s = g_spectra[id];
    return s.info.x == kSpecConstant && (s.abcs.x == 0 || s.abcs.w == 0);
}

// ---------------------------------------------------------------- refractive index

static const uint kIndexConstant = 0;
static const uint kIndexCauchy = 1;
static const uint kIndexSellmeier = 2;
static const uint kIndexTabulated = 3;
static const uint kIndexCiddor = 4;

float ciddorAir(float s2) { return 1.0 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2); }

float mediumIndex(uint medium, float lambdaNm) {
    GMedium m = g_media[medium];
    float l = lambdaNm * 1e-3;  // μm
    float l2 = l * l;
    uint c = m.index.z;
    float v = 1;
    switch (m.index.x) {
        case kIndexConstant: v = g_scalars[c]; break;
        case kIndexCauchy: v = g_scalars[c] + g_scalars[c + 1] / l2 + g_scalars[c + 2] / (l2 * l2); break;
        case kIndexSellmeier: {
            float s = 1;
            uint n = m.index.w;
            for (uint i = 0; i < n; ++i) s += g_scalars[c + i] * l2 / (l2 - g_scalars[c + n + i]);
            v = sqrt(s);
            break;
        }
        case kIndexTabulated: v = spectrumEval(c, lambdaNm); break;
        case kIndexCiddor: v = ciddorAir(1.0 / l2); break;
        default: break;
    }
    if (m.index.y != 0) v *= ciddorAir(1.0 / l2);
    return v;
}
uint mediumOf(uint region) { return g_regions[region]; }
float regionIndex(uint region, float lambdaNm) { return mediumIndex(mediumOf(region), lambdaNm); }
bool mediumAbsorbs(uint medium) { return (g_media[medium].props.z & 1u) != 0; }
bool mediumScatters(uint medium) { return (g_media[medium].props.z & 2u) != 0; }

// ---------------------------------------------------------------- hero wavelengths

struct Wavelengths {
    float4 lambda;
    float4 pdf;  // 0 marks a terminated secondary wavelength

    static Wavelengths sample(float u) {
        Wavelengths w;
        [ForceUnroll] for (int i = 0; i < NW; ++i) {
            float up = u + float(i) / NW;
            if (up >= 1) up -= 1;
            w.lambda[i] = sampleVisibleWavelength(up);
            w.pdf[i] = visibleWavelengthPdf(w.lambda[i]);
        }
        return w;
    }
    float hero() { return lambda.x; }
    bool secondaryTerminated() { return pdf.y == 0 && pdf.z == 0 && pdf.w == 0; }
    [mutating] void terminateSecondary() {
        if (secondaryTerminated()) return;
        pdf = float4(pdf.x / NW, 0, 0, 0);
    }
    float4 eval(uint spectrum) {
        return float4(spectrumEval(spectrum, lambda.x), spectrumEval(spectrum, lambda.y), spectrumEval(spectrum, lambda.z),
                      spectrumEval(spectrum, lambda.w));
    }
    // XYZ contribution: (1/NW) Σ L(λᵢ) cmf(λᵢ)/pdf(λᵢ) / ∫ȳ.
    float3 toXYZ(float4 L, float cieYIntegral) {
        float3 c = 0;
        float inv = 1.0 / (NW * cieYIntegral);
        [ForceUnroll] for (int i = 0; i < NW; ++i) {
            if (pdf[i] == 0 || L[i] == 0) continue;
            float w = L[i] / pdf[i] * inv;
            float l = lambda[i];
            c += w * float3(cieX(l), cieY(l), cieZ(l));
        }
        return c;
    }
};

bool isZero(float4 v) { return all(v == 0); }
float maxValue(float4 v) { return max(max(v.x, v.y), max(v.z, v.w)); }
