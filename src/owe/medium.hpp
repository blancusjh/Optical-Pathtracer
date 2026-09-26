// Constitutive optical properties of matter.
#pragma once

#include <string>
#include <vector>

#include "spectrum.hpp"

namespace owe {

// Refractive index n(λ). Wavelengths are in nm at the API; formulas use μm.
class IndexModel {
public:
    enum class Kind { Constant, Cauchy, Sellmeier, Tabulated, Ciddor };

    static IndexModel constant(double n);
    // n = A + B/λ² + C/λ⁴ (λ in μm).
    static IndexModel cauchy(double A, double B, double C = 0);
    // n² = 1 + Σ Bᵢ λ²/(λ² − Cᵢ) (λ in μm, Cᵢ in μm²).
    static IndexModel sellmeier(std::vector<double> B, std::vector<double> C);
    static IndexModel tabulated(std::vector<double> lambdasNm, std::vector<double> n);
    // Standard dry air, 15 °C, 101.325 kPa, 450 ppm CO₂ (Ciddor 1996 two-term form).
    static IndexModel ciddorAir();

    // Absolute refractive index. Catalog glass data are relative to air; such
    // models multiply by n_air(λ) so interfaces with the ambient air are exact.
    double n(double lambdaNm) const;
    double nRelative(double lambdaNm) const;  // as tabulated (relative to air if flagged)
    IndexModel relativeToAir() const { IndexModel m = *this; m.relAir_ = true; return m; }
    bool isRelativeToAir() const { return relAir_; }
    Kind kind() const { return kind_; }
    std::string describe() const;

private:
    Kind kind_ = Kind::Constant;
    std::vector<double> a_, b_;
    Spectrum table_;
    bool relAir_ = false;
};

// A homogeneous optical medium. Opaque media are never entered by light;
// they only mark the interior of solid bodies whose boundaries are opaque.
struct Medium {
    std::string name;
    IndexModel index = IndexModel::constant(1.0);
    Spectrum absorption = Spectrum::constant(0);  // σₐ(λ) in 1/m (Beer–Lambert)
    Spectrum scattering = Spectrum::constant(0);  // σₛ(λ) in 1/m
    double g = 0;                                 // Henyey–Greenstein anisotropy
    bool opaque = false;

    double n(double lambdaNm) const { return index.n(lambdaNm); }
    bool scatters() const { return !scattering.isZero(); }
    bool absorbs() const { return !absorption.isZero(); }
};

// Abbe number V_d = (n_d − 1)/(n_F − n_C).
double abbeNumber(const IndexModel& m);
constexpr double LambdaD = 587.5618;  // helium d line, nm
constexpr double LambdaF = 486.1327;  // hydrogen F line, nm
constexpr double LambdaC = 656.2725;  // hydrogen C line, nm

// Built-in catalog. Returns false when the name is unknown.
bool catalogMedium(const std::string& name, Medium& out);
std::vector<std::string> catalogMediumNames();

// Complex index (n + ik) of a metal.
struct ConductorSpectrum {
    std::string name;
    Spectrum n, k;
};
bool catalogConductor(const std::string& name, ConductorSpectrum& out);
std::vector<std::string> catalogConductorNames();

}  // namespace owe
