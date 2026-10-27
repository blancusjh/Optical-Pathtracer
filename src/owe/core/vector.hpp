// Scalars and small vectors of the CPU reference (IEEE-754 double). The GPU kernels use Slang's
// float vectors; formulas both sides evaluate are written once in owe/core/shared.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace owe {

constexpr double Pi = 3.14159265358979323846;
constexpr double InvPi = 1.0 / Pi;
constexpr double Inf = std::numeric_limits<double>::infinity();
constexpr double SpeedOfLight = 299792458.0;  // m/s

inline double sqr(double x) { return x * x; }
inline double clamp(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }
inline double saturate(double x) { return clamp(x, 0.0, 1.0); }
inline double safeSqrt(double x) { return std::sqrt(std::max(0.0, x)); }
inline double radians(double deg) { return deg * (Pi / 180.0); }
inline double degrees(double rad) { return rad * (180.0 / Pi); }

struct Vec2 {
    double x = 0, y = 0;
    constexpr Vec2() = default;
    constexpr Vec2(double x_, double y_) : x(x_), y(y_) {}
    double operator[](int i) const { return i == 0 ? x : y; }
    Vec2 operator-() const { return {-x, -y}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    Vec2& operator*=(double s) { x *= s; y *= s; return *this; }
};
inline Vec2 operator+(Vec2 a, const Vec2& b) { return a += b; }
inline Vec2 operator-(Vec2 a, const Vec2& b) { return a -= b; }
inline Vec2 operator*(Vec2 a, double s) { return a *= s; }
inline Vec2 operator*(double s, Vec2 a) { return a *= s; }
inline Vec2 operator/(const Vec2& a, double s) { return a * (1.0 / s); }
inline double dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
inline double length(const Vec2& a) { return std::hypot(a.x, a.y); }
inline Vec2 normalize(const Vec2& a) { return a * (1 / length(a)); }

struct Vec3 {
    double x = 0, y = 0, z = 0;
    constexpr Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
    Vec3& operator/=(double s) { return *this *= 1.0 / s; }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
inline Vec3 operator*(Vec3 a, double s) { return a *= s; }
inline Vec3 operator*(double s, Vec3 a) { return a *= s; }
inline Vec3 operator/(const Vec3& a, double s) { return a * (1.0 / s); }
inline Vec3 mul(const Vec3& a, const Vec3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double lengthSq(const Vec3& a) { return dot(a, a); }
inline double length(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a) { return a / length(a); }
inline Vec3 vmin(const Vec3& a, const Vec3& b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
inline Vec3 vmax(const Vec3& a, const Vec3& b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
inline double maxAbsComponent(const Vec3& a) { return std::max({std::abs(a.x), std::abs(a.y), std::abs(a.z)}); }
// Mirror reflection of an incident propagation direction d about normal n (Slang's reflect).
inline Vec3 reflect(const Vec3& d, const Vec3& n) { return d - 2.0 * dot(d, n) * n; }

}  // namespace owe
