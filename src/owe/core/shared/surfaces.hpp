// Ray–surface algebra shared by the reference and the GPU kernels (see prelude.hpp): quadric
// roots refined for distant origins, optical sags, liquid-wave heights, lathe patches, plane
// apertures and the trims of walls and domes. Everything is in the shape's own frame.
#pragma once

#include "owe/core/shared/prelude.hpp"

OWE_SHARED_BEGIN

// ---------------------------------------------------------------- quadric roots

// Numerically stable real roots of a t² + b t + c = 0, ascending; returns their count.
OWE_FN int solveQuadratic(Real a, Real b, Real c, OWE_OUT(Real) t0, OWE_OUT(Real) t1) {
    t0 = 0;
    t1 = 0;
    if (abs(a) < kTiny || abs(a) < 1e-14 * abs(b)) {
        if (b == 0) return 0;
        t0 = -c / b;
        t1 = t0;
        return 1;
    }
    Real disc = b * b - 4 * a * c;
    if (disc < 0) return 0;
    Real q = -(b + copysign(sqrt(disc), b)) / 2;
    t0 = q / a;
    t1 = q != 0 ? c / q : t0;
    if (t0 > t1) {
        Real tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    return 2;
}

// The quadrics whose roots are refined: a conic sag c(x²+y²) + c(1+k) z² − 2z = 0, a sphere and an
// infinite cylinder of radius R about the origin and the z axis, and a cone x² + y² = (r0 + k(z − z0))².
OWE_CONST uint kQuadSag = 0, kQuadSphere = 1, kQuadCylinder = 2, kQuadCone = 3;
struct Quadric {
    uint kind;
    Real a, b, c;  // sag: c, c(1 + k); sphere, cylinder: R²; cone: r0, z0, k
};
OWE_FN Quadric makeQuadric(uint kind, Real a, Real b, Real c) {
    Quadric q;
    q.kind = kind;
    q.a = a;
    q.b = b;
    q.c = c;
    return q;
}

OWE_FN void quadricCoeffs(Quadric q, Real3 o, Real3 d, OWE_OUT(Real) A, OWE_OUT(Real) B, OWE_OUT(Real) C) {
    if (q.kind == kQuadSag) {
        A = q.a * (d.x * d.x + d.y * d.y) + q.b * d.z * d.z;
        B = 2 * (q.a * (o.x * d.x + o.y * d.y) + q.b * o.z * d.z - d.z);
        C = q.a * (o.x * o.x + o.y * o.y) + q.b * o.z * o.z - 2 * o.z;
    } else if (q.kind == kQuadSphere) {
        A = dot(d, d);
        B = 2 * dot(o, d);
        C = dot(o, o) - q.a;
    } else if (q.kind == kQuadCylinder) {
        A = d.x * d.x + d.y * d.y;
        B = 2 * (o.x * d.x + o.y * d.y);
        C = o.x * o.x + o.y * o.y - q.a;
    } else {
        Real rho = q.a + q.c * (o.z - q.b);
        A = d.x * d.x + d.y * d.y - q.c * q.c * d.z * d.z;
        B = 2 * (o.x * d.x + o.y * d.y - q.c * rho * d.z);
        C = o.x * o.x + o.y * o.y - rho * rho;
    }
}

// Distant origins lose precision in the coefficients (cancellation in C and the discriminant).
// Re-solving from the approximate hit point, where every quantity is small, and adding the
// correction restores it. A correction above kRefine (1 + |t|) is not a refinement and is ignored.
OWE_FN Real refineRoot(Quadric q, Real3 o, Real3 d, Real t) {
    Real A, B, C, u0, u1;
    quadricCoeffs(q, o + d * t, d, A, B, C);
    if (solveQuadratic(A, B, C, u0, u1) == 0) return t;
    Real du = abs(u0) < abs(u1) ? u0 : u1;
    return abs(du) < kRefine * (1 + abs(t)) ? t + du : t;
}
// Refinement moves a root by less than kRefine (1 + |t|): one farther outside (tmin, tmax) stays
// outside, and its refinement is skipped.
OWE_FN bool mayLieIn(Real t, Real tmin, Real tmax) {
    Real m = kRefine * (1 + abs(t));
    return !(t + m <= tmin || t - m >= tmax);
}
// The root t refined, if it may lie in (tmin, tmax) and is inside once refined.
OWE_FN bool refinedRootIn(Quadric q, Real3 o, Real3 d, Real t, Real tmin, Real tmax, OWE_OUT(Real) tr) {
    tr = t;
    if (!mayLieIn(t, tmin, tmax)) return false;
    tr = refineRoot(q, o, d, t);
    return tr > tmin && tr < tmax;
}

// Sphere of radius R about the origin: roots in the geometric form (the discriminant from the
// closest approach, which does not cancel when the ray starts far away), ascending.
OWE_FN bool sphereRoots(Real3 o, Real3 d, Real R, OWE_OUT(Real) t0, OWE_OUT(Real) t1) {
    Real b = dot(o, d);
    Real3 perp = o - d * b;
    Real disc = R * R - dot(perp, perp);
    t0 = 0;
    t1 = 0;
    if (disc < 0) return false;
    Real c = dot(o, o) - R * R;
    Real q = -b - copysign(sqrt(disc), b);
    t0 = q;
    t1 = q != 0 ? c / q : q;
    if (t0 > t1) {
        Real tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    return true;
}
// Infinite cylinder of radius R about the z axis: roots of A t² + 2 b t + c = 0 in the same
// geometric form (b² − A c = A (R² − |perp|²), perp the closest approach to the axis), ascending.
OWE_FN bool cylinderRoots(Real3 o, Real3 d, Real R, OWE_OUT(Real) t0, OWE_OUT(Real) t1) {
    t0 = 0;
    t1 = 0;
    Real A = d.x * d.x + d.y * d.y;
    if (A < kTiny) return false;
    Real b = o.x * d.x + o.y * d.y;
    Real px = o.x - d.x * (b / A), py = o.y - d.y * (b / A);
    Real disc = A * (R * R - (px * px + py * py));
    if (disc < 0) return false;
    Real q = -(b + copysign(sqrt(disc), b));
    Real c = o.x * o.x + o.y * o.y - R * R;
    t0 = q / A;
    t1 = q != 0 ? c / q : t0;
    if (t0 > t1) {
        Real tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    return true;
}

// Clips (ta, tb) to the slab lo ≤ o + t d ≤ hi along one axis; false when nothing remains.
OWE_FN bool clipSlab(Real o, Real d, Real lo, Real hi, OWE_INOUT(Real) ta, OWE_INOUT(Real) tb) {
    if (abs(d) < kTiny) return o >= lo && o <= hi && ta < tb;
    Real s0 = (lo - o) / d, s1 = (hi - o) / d;
    ta = max(ta, min(s0, s1));
    tb = min(tb, max(s0, s1));
    return ta < tb;
}
// Clips (ta, tb) to the infinite cylinder of radius R about the z axis.
OWE_FN bool clipCylinder(Real3 o, Real3 d, Real R, OWE_INOUT(Real) ta, OWE_INOUT(Real) tb) {
    Real a = d.x * d.x + d.y * d.y;
    if (a < kTiny) return o.x * o.x + o.y * o.y <= R * R && ta < tb;
    Real q0, q1;
    if (solveQuadratic(a, 2 * (o.x * d.x + o.y * d.y), o.x * o.x + o.y * o.y - R * R, q0, q1) < 2) return false;
    ta = max(ta, q0);
    tb = min(tb, q1);
    return ta < tb;
}

// ---------------------------------------------------------------- optical sags

// The conic part of a rotationally symmetric sag z(r) (curvature c, conic constant k) and its
// slope dz/dr; an even asphere adds Σ A_{2m} r^{2m} (m ≥ 2) to both.
OWE_FN Real conicSag(Real c, Real k, Real r2) { return c * r2 / (1 + safeSqrt(1 - (1 + k) * c * c * r2)); }
OWE_FN Real conicSlope(Real c, Real k, Real r) {
    Real root = safeSqrt(1 - (1 + k) * c * c * (r * r));
    return root > 0 ? c * r / root : (c != 0 ? infinity() : Real(0));
}
// The unit normal (toward +z) of a sag with slope s at radius r of point p.
OWE_FN Real3 sagNormal(Real3 p, Real r, Real s) {
    if (r < kTiny) return Real3(0, 0, 1);
    return normalize(Real3(-s * p.x / r, -s * p.y / r, 1));
}
// A root on the conic's other sheet (beyond its apex sphere) is not the surface.
OWE_FN bool onConicSheet(Real c, Real k, Real z) { return c == 0 || (1 + k) * c * z <= 1 + kRelTol; }

// ---------------------------------------------------------------- liquid waves

// Smoothstep taper of coordinate u toward the edge at ±half over `margin`, and its derivative.
OWE_FN void waveTaper(Real u, Real half, Real margin, OWE_OUT(Real) w, OWE_OUT(Real) dw) {
    w = 1;
    dw = 0;
    if (!(margin > 0)) return;
    Real e = half - abs(u);
    if (e >= margin) return;
    if (e <= 0) {
        w = 0;
        return;
    }
    Real t = e / margin;
    w = t * t * (3 - 2 * t);
    dw = -(u < 0 ? Real(-1) : Real(1)) * 6 * t * (1 - t) / margin;
}

// Cubic Hermite bases at u ∈ [0, 1] for the two ends of an interval: value bases P and slope
// bases Q, and their derivatives.
OWE_FN void hermiteBases(Real u, OWE_OUT(Real2) P, OWE_OUT(Real2) Q, OWE_OUT(Real2) dP, OWE_OUT(Real2) dQ) {
    P = Real2(2 * u * u * u - 3 * u * u + 1, -2 * u * u * u + 3 * u * u);
    Q = Real2(u * u * u - 2 * u * u + u, u * u * u - u * u);
    dP = Real2(6 * u * u - 6 * u, -6 * u * u + 6 * u);
    dQ = Real2(3 * u * u - 4 * u + 1, 3 * u * u - 2 * u);
}

// ---------------------------------------------------------------- lathe patches

// The cone frustum's outward normal at p: (x, y, −ρk) with ρ = r(z), k = dr/dz.
OWE_FN Real3 coneNormal(Real3 p, Real rho, Real k) { return normalize(Real3(p.x, p.y, -rho * k)); }

// Torus patch: the arc of radius rho about (R, zc) in the (r, z) half-plane, for arc angles
// a0 ≤ a ≤ a1, revolved about z. F < 0 inside the tube.
OWE_FN Real torusF(Real R, Real zc, Real rho, Real3 p) {
    return sqr(sqrt(p.x * p.x + p.y * p.y) - R) + sqr(p.z - zc) - rho * rho;
}
OWE_FN bool torusOnArc(Real R, Real zc, Real a0, Real a1, Real3 p) {
    Real a = atan2(p.z - zc, sqrt(p.x * p.x + p.y * p.y) - R);
    for (int k = -1; k <= 1; ++k) {
        Real w = a + k * 2 * Pi;
        if (w >= a0 - kAngleTol && w <= a1 + kAngleTol) return true;
    }
    return false;
}
// Normal away from the arc's centre.
OWE_FN Real3 torusNormal(Real R, Real zc, Real3 p) {
    Real rho = sqrt(p.x * p.x + p.y * p.y);
    Real3 c = rho > 0 ? Real3(p.x * R / rho, p.y * R / rho, zc) : Real3(0, 0, zc);
    return normalize(p - c);
}

// ---------------------------------------------------------------- trims

// Plane apertures (z = 0): a disk or annulus (a = rMax, b = rMin), a rectangle (half sizes a, b),
// an ellipse (semi-axes a, b), a disk of radius a with a hole of radius b centred at (hx, hy).
OWE_CONST uint kApDisk = 0, kApRect = 1, kApEllipse = 2, kApDiskHole = 3;
OWE_FN bool inPlaneAperture(uint kind, Real a, Real b, Real hx, Real hy, Real3 p) {
    if (kind == kApDisk) {
        Real r2 = p.x * p.x + p.y * p.y;
        return r2 <= a * a && r2 >= b * b;
    }
    if (kind == kApRect) return abs(p.x) <= a && abs(p.y) <= b;
    if (kind == kApEllipse) return sqr(p.x / a) + sqr(p.y / b) <= 1;
    return p.x * p.x + p.y * p.y <= a * a && sqr(p.x - hx) + sqr(p.y - hy) >= b * b;
}

// A rectangular opening in a round wall of radius R: centred on an azimuth (from +y toward +x),
// `width` along the wall, between heights sill and top.
OWE_FN bool inWallOpening(Real R, Real azimuth, Real width, Real sill, Real top, Real3 p) {
    Real dAz = remainder(atan2(p.x, p.y) - azimuth, 2 * Pi);
    return abs(dAz) * R < width / 2 && p.z > sill && p.z < top;
}

// An observatory dome of radius R (z ≥ 0) with a slit of constant linear width centred on an
// azimuth, open from the horizon up to elevation top.
OWE_FN bool domeSolid(Real R, Real azimuth, Real width, Real top, Real3 p) {
    if (p.z < 0) return false;
    if (width <= 0) return true;
    Real el = asin(clamp(p.z / R, Real(-1), Real(1)));
    if (el >= top) return true;
    Real dAz = remainder(atan2(p.x, p.y) - azimuth, 2 * Pi);
    // Constant linear width: the half-angle grows toward the zenith.
    return abs(sin(dAz)) * R * cos(el) >= width / 2 || cos(dAz) < 0;
}

// ---------------------------------------------------------------- triangles

// A triangle's (unnormalised) normal as the cross product of its two shorter edges. For a sliver,
// the two long edges are nearly parallel and their cross product cancels; the shorter pair does not.
OWE_FN Real3 triangleNormal(Real3 v0, Real3 v1, Real3 v2) {
    Real3 a = v1 - v0, b = v2 - v0, c = v2 - v1;
    Real la = dot(a, a), lb = dot(b, b), lc = dot(c, c);
    if (lc >= la && lc >= lb) return cross(a, b);
    return la >= lb ? cross(b, c) : cross(a, c);
}

OWE_SHARED_END
