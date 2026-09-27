// Core double-precision geometry for the CPU reference tracer.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace owe {

constexpr double Pi = 3.14159265358979323846;
constexpr double InvPi = 1.0 / Pi;
constexpr double Inf = std::numeric_limits<double>::infinity();
constexpr double SpeedOfLight = 299792458.0;  // m/s

inline double sqr(double x) { return x * x; }
inline double clampd(double x, double lo, double hi) { return std::min(hi, std::max(lo, x)); }
inline double safeSqrt(double x) { return std::sqrt(std::max(0.0, x)); }
inline double radians(double deg) { return deg * (Pi / 180.0); }
inline double degrees(double rad) { return rad * (180.0 / Pi); }

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

// Mirror reflection of an incident propagation direction d about normal n.
inline Vec3 reflect(const Vec3& d, const Vec3& n) { return d - 2.0 * dot(d, n) * n; }

struct Mat3 {
    // Row-major.
    double m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static Mat3 fromColumns(const Vec3& a, const Vec3& b, const Vec3& c) {
        Mat3 r;
        for (int i = 0; i < 3; ++i) { r.m[i][0] = a[i]; r.m[i][1] = b[i]; r.m[i][2] = c[i]; }
        return r;
    }
    Vec3 col(int j) const { return {m[0][j], m[1][j], m[2][j]}; }
    Vec3 operator*(const Vec3& v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }
    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
        return r;
    }
    Mat3 transposed() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
        return r;
    }
    // Rotation by `angle` radians about unit `axis` (Rodrigues).
    static Mat3 rotation(const Vec3& axisIn, double angle) {
        Vec3 a = normalize(axisIn);
        double c = std::cos(angle), s = std::sin(angle), t = 1 - c;
        Mat3 r;
        r.m[0][0] = t * a.x * a.x + c;       r.m[0][1] = t * a.x * a.y - s * a.z; r.m[0][2] = t * a.x * a.z + s * a.y;
        r.m[1][0] = t * a.x * a.y + s * a.z; r.m[1][1] = t * a.y * a.y + c;       r.m[1][2] = t * a.y * a.z - s * a.x;
        r.m[2][0] = t * a.x * a.z - s * a.y; r.m[2][1] = t * a.y * a.z + s * a.x; r.m[2][2] = t * a.z * a.z + c;
        return r;
    }
};

// Orthonormal basis around a unit vector (Duff et al. 2017).
inline void orthonormalBasis(const Vec3& n, Vec3& b1, Vec3& b2) {
    double sign = std::copysign(1.0, n.z);
    double a = -1.0 / (sign + n.z);
    double b = n.x * n.y * a;
    b1 = {1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x};
    b2 = {b, sign + n.y * n.y * a, -n.y};
}

struct Frame {
    Vec3 s, t, n;
    Frame() : s(1, 0, 0), t(0, 1, 0), n(0, 0, 1) {}
    explicit Frame(const Vec3& normal) : n(normal) { orthonormalBasis(n, s, t); }
    Vec3 toLocal(const Vec3& v) const { return {dot(v, s), dot(v, t), dot(v, n)}; }
    Vec3 toWorld(const Vec3& v) const { return s * v.x + t * v.y + n * v.z; }
};

// Rigid transform x_parent = R x_local + t. Rigid transforms preserve ray
// parameters, so hit distances need no rescaling between frames.
struct Transform {
    Mat3 R;
    Vec3 t;
    Vec3 point(const Vec3& p) const { return R * p + t; }
    Vec3 vector(const Vec3& v) const { return R * v; }
    Transform inverse() const {
        Transform r;
        r.R = R.transposed();
        r.t = -(r.R * t);
        return r;
    }
    Transform operator*(const Transform& o) const {  // this ∘ o
        Transform r;
        r.R = R * o.R;
        r.t = R * o.t + t;
        return r;
    }
    static Transform translate(const Vec3& v) { Transform r; r.t = v; return r; }
    static Transform rotate(const Vec3& axis, double angle) { Transform r; r.R = Mat3::rotation(axis, angle); return r; }
    // Rotation mapping local +z onto `axis`; `up` hints where local +y should point.
    static Transform alignZ(const Vec3& axisIn, const Vec3& upHint = {0, 1, 0}) {
        Vec3 z = normalize(axisIn);
        Vec3 up = upHint;
        if (std::abs(dot(normalize(up), z)) > 0.999) up = std::abs(z.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 0, 1);
        Vec3 x = normalize(cross(up, z));
        Vec3 y = cross(z, x);
        Transform r;
        r.R = Mat3::fromColumns(x, y, z);
        return r;
    }
};

struct Ray {
    Vec3 o, d;
    Vec3 at(double t) const { return o + d * t; }
};

struct AABB {
    Vec3 lo{Inf, Inf, Inf}, hi{-Inf, -Inf, -Inf};
    bool valid() const { return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z; }
    void expand(const Vec3& p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void expand(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    Vec3 centroid() const { return (lo + hi) * 0.5; }
    Vec3 extent() const { return hi - lo; }
    double surfaceArea() const {
        if (!valid()) return 0;
        Vec3 e = extent();
        return 2.0 * (e.x * e.y + e.y * e.z + e.z * e.x);
    }
    AABB transformed(const Transform& xf) const {
        AABB r;
        for (int i = 0; i < 8; ++i) {
            Vec3 c{(i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z};
            r.expand(xf.point(c));
        }
        return r;
    }
    // Slab test; invD may contain infinities.
    bool intersect(const Vec3& o, const Vec3& invD, double tmin, double tmax, double& tEnter) const {
        for (int a = 0; a < 3; ++a) {
            double t0 = (lo[a] - o[a]) * invD[a];
            double t1 = (hi[a] - o[a]) * invD[a];
            if (t0 > t1) std::swap(t0, t1);
            if (std::isnan(t0)) t0 = -Inf;  // 0 * inf when the origin lies on a slab
            if (std::isnan(t1)) t1 = Inf;
            tmin = std::max(tmin, t0);
            tmax = std::min(tmax, t1 * (1.0 + 4e-16));
            if (tmin > tmax) return false;
        }
        tEnter = tmin;
        return true;
    }
};

// A ray prepared for many slab tests: the near and far planes of each axis are chosen once by
// the direction's sign instead of being sorted per box. Same result as AABB::intersect (a NaN
// from an origin on a slab plane with a zero direction component never narrows the interval).
struct RaySlabs {
    double o[3], inv[3];
    bool neg[3];
    explicit RaySlabs(const Ray& r)
        : o{r.o.x, r.o.y, r.o.z}, inv{1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z},
          neg{1.0 / r.d.x < 0, 1.0 / r.d.y < 0, 1.0 / r.d.z < 0} {}
    bool hit(const AABB& b, double tmin, double tmax, double& tEnter) const {
        const double lo[3] = {b.lo.x, b.lo.y, b.lo.z}, hi[3] = {b.hi.x, b.hi.y, b.hi.z};
        for (int a = 0; a < 3; ++a) {
            double t0 = ((neg[a] ? hi[a] : lo[a]) - o[a]) * inv[a];
            double t1 = ((neg[a] ? lo[a] : hi[a]) - o[a]) * inv[a] * (1.0 + 4e-16);
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmin > tmax) return false;
        }
        tEnter = tmin;
        return true;
    }
};

// Displacement used to move a new ray origin off a surface. It scales with coordinate
// magnitude so it stays well above double rounding (ulp ≈ 2.2e-16 relative, a few ulp after
// rigid transforms) at any world scale, yet far below optical feature sizes: 10 pm at the
// origin, 1 nm for an instrument standing 100 m away, 0.1 µm at 10 km.
inline double surfaceEpsilon(const Vec3& p) { return 1e-11 * (1.0 + maxAbsComponent(p)); }
// Moves along the new ray itself so the origin stays exactly on the refracted or
// reflected line (no lateral error); grazing directions fall back to the normal.
inline Vec3 offsetOrigin(const Vec3& p, const Vec3& n, const Vec3& dir) {
    double e = surfaceEpsilon(p);
    double c = dot(dir, n);
    if (std::abs(c) > 0.05) return p + dir * (e / std::abs(c));
    return c >= 0 ? p + n * e : p - n * e;
}

}  // namespace owe
