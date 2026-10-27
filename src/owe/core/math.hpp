// Frames, rigid transforms, rays and boxes of the CPU reference tracer (IEEE-754 double).
#pragma once

#include "owe/core/shared/warps.hpp"
#include "owe/core/vector.hpp"

namespace owe {

// The unit direction at azimuth az (from +y toward +x) and elevation el above the xy plane.
inline Vec3 directionFromAzimuthElevation(double az, double el) {
    return {std::cos(el) * std::sin(az), std::cos(el) * std::cos(az), std::sin(el)};
}

// Conservative absolute bound for contact-anchor arithmetic, not a physical gap.
inline double contactError(double scale) {
    return 64 * std::numeric_limits<double>::epsilon() * std::max(1e-12, scale);
}

struct Mat3 {
    // Row-major.
    double m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static Mat3 fromColumns(const Vec3& a, const Vec3& b, const Vec3& c) {
        Mat3 r;
        for (int i = 0; i < 3; ++i) { r.m[i][0] = a[i]; r.m[i][1] = b[i]; r.m[i][2] = c[i]; }
        return r;
    }
    static Mat3 fromRows(const double rows[3][3]) {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = rows[i][j];
        return r;
    }
    // The cofactor matrix: det · (M⁻¹)ᵀ, which carries surface normals (the inverse transpose up to
    // the determinant's scale).
    Mat3 cofactor() const {
        Mat3 c;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) c.m[i][j] = cofactorOf(i, j);
        return c;
    }
    double determinant() const { return m[0][0] * cofactorOf(0, 0) + m[0][1] * cofactorOf(0, 1) + m[0][2] * cofactorOf(0, 2); }
    Mat3 inverse() const {
        Mat3 r = cofactor().transposed();
        const double inv = 1 / determinant();
        for (auto& row : r.m)
            for (double& x : row) x *= inv;
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
    double cofactorOf(int i, int j) const {
        const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
        return m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1];
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
    // The minimal rotation taking unit vector `from` to unit vector `to` (a half turn when opposite).
    static Transform rotateTo(const Vec3& from, const Vec3& to) {
        const Vec3 axis = cross(from, to);
        if (length(axis) > 1e-12) return rotate(axis, std::acos(clamp(dot(from, to), -1, 1)));
        if (dot(from, to) > 0) return Transform{};
        Vec3 a, b;
        orthonormalBasis(to, a, b);
        return rotate(a, Pi);
    }
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
    // Spawned rays retain their surface anchor. Only the departing boundary's numerical
    // self root is excluded; a distinct contacting boundary must remain visible.
    uint32_t departedBoundary = 0xFFFFFFFEu;
    uint32_t regionHint = 0xFFFFFFFEu;
    Vec3 departedNormal{};
    uint32_t departedPrimitive = 0xFFFFFFFEu;
    double departureError = 0;
    bool forwardOnly = false;  // exact nonnegative interval for endpoint-local proofs
    Vec3 at(double t) const { return o + d * t; }
};

struct AABB {
    Vec3 lo{Inf, Inf, Inf}, hi{-Inf, -Inf, -Inf};
    bool valid() const { return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z; }
    void expand(const Vec3& p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void expand(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    Vec3 centroid() const { return (lo + hi) * 0.5; }
    // Corner i: bit 0 chooses hi.x, bit 1 hi.y, bit 2 hi.z.
    Vec3 corner(int i) const { return {(i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z}; }
    Vec3 extent() const { return hi - lo; }
    double surfaceArea() const {
        if (!valid()) return 0;
        Vec3 e = extent();
        return 2.0 * (e.x * e.y + e.y * e.z + e.z * e.x);
    }
    AABB transformed(const Transform& xf) const {
        AABB r;
        for (int i = 0; i < 8; ++i) r.expand(xf.point(corner(i)));
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
