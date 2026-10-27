// Spectral formulas shared by the reference and the GPU kernels (see prelude.hpp): colour matching,
// wavelength sampling, Planck's law, the RGB basis spectra and refractive-index models.
#pragma once

#include "owe/core/shared/prelude.hpp"

OWE_SHARED_BEGIN

OWE_CONST Real LambdaMin = 360.0;  // nm
OWE_CONST Real LambdaMax = 830.0;  // nm

// CIE 1931 2° colour matching functions (Wyman, Sloan & Shirley 2013 multi-lobe fit).
OWE_FN Real cieLobe(Real x, Real mu, Real s1, Real s2) {
    Real t = (x - mu) / (x < mu ? s1 : s2);
    return exp(-0.5 * t * t);
}
OWE_FN Real cieX(Real l) {
    return 1.056 * cieLobe(l, 599.8, 37.9, 31.0) + 0.362 * cieLobe(l, 442.0, 16.0, 26.7) -
           0.065 * cieLobe(l, 501.1, 20.4, 26.2);
}
OWE_FN Real cieY(Real l) { return 0.821 * cieLobe(l, 568.8, 46.9, 40.5) + 0.286 * cieLobe(l, 530.9, 16.3, 31.1); }
OWE_FN Real cieZ(Real l) { return 1.217 * cieLobe(l, 437.0, 11.8, 36.0) + 0.681 * cieLobe(l, 459.0, 26.0, 13.8); }

// Importance sampling of visible wavelengths, pdf ∝ 1/cosh²(0.0072 (λ − 538)). The inverse CDF can
// round a hair outside the range at u → 0 or 1.
OWE_FN Real sampleVisibleWavelength(Real u) {
    return clamp(538 - 138.888889 * atanh(0.85691062 - 1.82750197 * u), LambdaMin, LambdaMax);
}
OWE_FN Real visibleWavelengthPdf(Real l) {
    if (l < LambdaMin || l > LambdaMax) return 0;
    return 0.0039398042 / sqr(cosh(0.0072 * (l - 538)));
}

// Planck spectral radiance B(λ, T) in W·m⁻²·sr⁻¹·m⁻¹ for λ in nm, evaluated in micrometres
// (c1 = 2hc² = 1.191e8 W·µm⁴·m⁻²·sr⁻¹, c2 = hc/k = 14388 µm·K): in metres λ⁵ ≈ 1e-32, and a float
// optimiser that reassociates the constant factors (legal in Vulkan) underflows to zero.
OWE_FN Real planck(Real lambdaNm, Real kelvin) {
    Real c1 = 1.191042972e8, c2 = 1.438776877e4;
    Real l = lambdaNm * 1e-3;
    Real l2 = l * l;
    return 1e6 * c1 / (l2 * l2 * l * (exp(c2 / (l * kelvin)) - 1));
}

// Partition-of-unity basis of the RGB spectra (red, green, blue): logistic edges at 492 and 590 nm.
OWE_FN Real logistic(Real x) { return 1 / (1 + exp(-x)); }
OWE_FN Real3 rgbBasis(Real l) {
    Real hiEdge = logistic((l - 590) / 11);
    Real loEdge = logistic((l - 492) / 11);
    return Real3(hiEdge, loEdge - hiEdge, 1 - loEdge);  // green ≥ 0: the logistic is monotone
}
// The RGB illuminant's base: a 6504 K Planck spectrum, scaled near unit magnitude.
OWE_FN Real rgbIlluminantBase(Real l) { return planck(l, 6504.0) * 1e-13; }

// Refractive-index models; l2 is the squared wavelength in µm².
OWE_FN Real ciddorAir(Real l2) {
    Real s2 = 1 / l2;
    return 1 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2);
}
OWE_FN Real cauchyIndex(Real A, Real B, Real C, Real l2) { return A + B / l2 + C / (l2 * l2); }
// One term B l²/(l² − C) of a Sellmeier sum; n = sqrt(1 + Σ terms).
OWE_FN Real sellmeierTerm(Real B, Real C, Real l2) { return B * l2 / (l2 - C); }

OWE_SHARED_END
