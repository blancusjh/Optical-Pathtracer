// Spectral quantities. Light transport is performed per wavelength; colour
// only appears when a detector integrates its spectral response.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "owe/core/math.hpp"

namespace owe {

constexpr double LambdaMin = 360.0;  // nm
constexpr double LambdaMax = 830.0;  // nm

// CIE 1931 2° colour matching functions (Wyman, Sloan & Shirley 2013 multi-lobe fit).
double cieX(double lambdaNm);
double cieY(double lambdaNm);
double cieZ(double lambdaNm);
// ∫ ȳ(λ) dλ over [LambdaMin, LambdaMax] for the fit above.
double cieYIntegral();

// Importance sampling of visible wavelengths (pdf ∝ 1/cosh²(0.0072(λ-538))).
double sampleVisibleWavelength(double u);
double visibleWavelengthPdf(double lambdaNm);

// Planck spectral radiance B(λ,T) in W·m⁻²·sr⁻¹·m⁻¹ for λ in nm.
double planck(double lambdaNm, double kelvin);

struct XYZ {
    double x = 0, y = 0, z = 0;
};
void xyzToLinearSRGB(const XYZ& c, double rgb[3]);

// A spectral distribution. Immutable after construction and cheap to copy.
class Spectrum {
public:
    enum class Kind { Constant, Blackbody, Tabulated, RgbReflectance, RgbIlluminant };

    Spectrum() = default;  // constant zero
    static Spectrum constant(double v);
    // Planck spectrum at temperature T, scaled so its luminance-like Y equals `y`.
    static Spectrum blackbody(double kelvin, double y = 1.0);
    // Piecewise-linear tabulation; clamped outside the table.
    static Spectrum tabulated(std::vector<double> lambdas, std::vector<double> values);
    // Smooth reflectance built from a partition of unity, so any rgb in [0,1]³
    // stays a physically valid reflectance in [0,1] at every wavelength.
    static Spectrum rgbReflectance(double r, double g, double b);
    // Emission spectrum: rgb partition times a 6504 K Planck base, normalised so (1,1,1) has Y = 1.
    static Spectrum rgbIlluminant(double r, double g, double b);

    double eval(double lambdaNm) const;
    double operator()(double lambdaNm) const { return eval(lambdaNm); }
    Spectrum scaled(double s) const { Spectrum r = *this; r.scale_ *= s; return r; }
    Kind kind() const { return kind_; }
    bool isZero() const { return kind_ == Kind::Constant && (a_ == 0 || scale_ == 0); }
    double maxValue() const;  // conservative upper bound over the visible range
    // Y tristimulus normalised so that a constant spectrum of 1 has Y = 1.
    double luminance() const;
    XYZ toXYZ() const;
    std::string describe() const;
    // Raw parameters (for backends that re-implement eval): a, b, c, scale and, for tables, the
    // wavelengths and values (null otherwise).
    double paramA() const { return a_; }
    double paramB() const { return b_; }
    double paramC() const { return c_; }
    double paramScale() const { return scale_; }
    const std::vector<double>* tableLambdas() const { return lambdas_.get(); }
    const std::vector<double>* tableValues() const { return values_.get(); }

private:
    Kind kind_ = Kind::Constant;
    double a_ = 0, b_ = 0, c_ = 0;
    double scale_ = 1;
    std::shared_ptr<const std::vector<double>> lambdas_, values_;
};

// Partition-of-unity basis used by rgbReflectance/rgbIlluminant.
void rgbBasis(double lambdaNm, double& br, double& bg, double& bb);
// The calibrated weights of the three basis spectra for linear sRGB (r, g, b): a linear map A·rgb
// (Spectrum::rgbReflectance clamps them to [0, 1], rgbIlluminant to ≥ 0).
void rgbToBasisWeights(double r, double g, double b, double w[3]);
void rgbBasisMatrix(double A[3][3]);
// rgbIlluminant(r, g, b) = norm · Σ max(0, wₖ) Bₖ(λ) · B(λ, 6504 K) · 1e-13, with this norm (Y = 1 for
// white).
double rgbIlluminantNorm();

}  // namespace owe
