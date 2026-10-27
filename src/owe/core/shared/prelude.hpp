// The dialect of owe/core/shared: formulas written once and compiled twice, by C++ for the
// double-precision reference and by Slang for the float GPU kernels. Real is double or float;
// Real2 and Real3 are owe::Vec2/Vec3 or float2/float3. The common subset both languages accept:
// free functions (OWE_FN), out and inout parameters (OWE_OUT, OWE_INOUT), plain structs set
// field by field, and the math functions named alike in both (sqrt, abs, min, max, clamp, dot,
// cross, normalize, ...). Tolerances that depend on the precision are named below.
#pragma once

#ifdef __SLANG__

typedef float Real;
typedef float2 Real2;
typedef float3 Real3;
#define OWE_FN
#define OWE_OUT(T) out T
#define OWE_INOUT(T) inout T
#define OWE_CONST static const
#define OWE_SHARED_BEGIN
#define OWE_SHARED_END

static const float Pi = 3.14159265358979323846;
static const float InvPi = 0.31830988618379067154;
float sqr(float x) { return x * x; }
float safeSqrt(float x) { return sqrt(max(0.0, x)); }
float maxAbsComponent(float3 a) { return max(max(abs(a.x), abs(a.y)), abs(a.z)); }
float infinity() { return asfloat(0x7f800000u); }
// IEEE remainder of x by y (result in [-y/2, y/2]).
float remainder(float x, float y) { return x - y * round(x / y); }

#else

#include <cstdint>

#include "owe/core/vector.hpp"

namespace owe {
using Real = double;
using Real2 = Vec2;
using Real3 = Vec3;
using uint = std::uint32_t;
using std::abs, std::acos, std::asin, std::atan2, std::atanh, std::copysign, std::cos, std::cosh, std::exp,
    std::floor, std::isfinite, std::isinf, std::log, std::max, std::min, std::pow, std::remainder, std::sin, std::sqrt;
inline double infinity() { return Inf; }
}  // namespace owe
#define OWE_FN inline
#define OWE_OUT(T) T&
#define OWE_INOUT(T) T&
#define OWE_CONST constexpr
#define OWE_SHARED_BEGIN namespace owe {
#define OWE_SHARED_END }

#endif

OWE_SHARED_BEGIN

// Tolerances by precision (double reference | float kernels). A float evaluation keeps about
// seven significant digits, so each float tolerance sits a few ulps above its rounding.
#ifdef __SLANG__
OWE_CONST Real kTiny = 1e-30;        // magnitudes below this are treated as zero in divisions
OWE_CONST Real kRefine = 1e-3;       // a re-solved root moves by less than kRefine (1 + |t|)
OWE_CONST Real kBisect = 2e-7;       // bisection stops at this relative bracket width
OWE_CONST int kBisectSteps = 40;
OWE_CONST Real kGrazing = 1e-7;      // a light point seen at a smaller cosine is edge-on
OWE_CONST Real kAngleTol = 1e-5;     // angular slack of trimmed analytic patches (radians)
OWE_CONST Real kRelTol = 1e-6;       // relative slack of inside/outside tests
#else
OWE_CONST Real kTiny = 1e-300;
OWE_CONST Real kRefine = 1e-6;
OWE_CONST Real kBisect = 1e-15;
OWE_CONST int kBisectSteps = 80;
OWE_CONST Real kGrazing = 1e-12;
OWE_CONST Real kAngleTol = 1e-9;
OWE_CONST Real kRelTol = 1e-9;
#endif

OWE_SHARED_END
