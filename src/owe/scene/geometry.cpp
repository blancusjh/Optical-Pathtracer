#include "owe/scene/geometry.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace owe {

// ---------------------------------------------------------------- Shape

void Shape::sampleArea(double, double, double, Vec3& p, Vec3& n) const {
    p = {};
    n = {0, 0, 1};
}

bool Shape::sampleFrom(const Vec3& ref, double u1, double u2, double u3, Vec3& p, Vec3& n, double& pdfW) const {
    if (!canSample() || area() <= 0) return false;
    sampleArea(u1, u2, u3, p, n);
    Vec3 w = p - ref;
    double d2 = lengthSq(w);
    if (d2 <= 0) return false;
    double cosL = std::abs(dot(n, w)) / std::sqrt(d2);
    if (cosL <= 1e-12) return false;
    pdfW = d2 / (cosL * area());
    return true;
}

double Shape::pdfFrom(const Vec3& ref, const Vec3& p, const Vec3& n) const {
    if (!canSample() || area() <= 0) return 0;
    Vec3 w = p - ref;
    double d2 = lengthSq(w);
    double cosL = std::abs(dot(n, w)) / std::sqrt(d2);
    if (cosL <= 1e-12) return 0;
    return d2 / (cosL * area());
}

namespace {
// Numerically stable real roots of a t² + b t + c = 0, ascending. Returns count.
int solveQuadratic(double a, double b, double c, double& t0, double& t1) {
    if (std::abs(a) < 1e-300 || std::abs(a) < 1e-14 * std::abs(b)) {
        if (b == 0) return 0;
        t0 = t1 = -c / b;
        return 1;
    }
    double disc = b * b - 4 * a * c;
    if (disc < 0) return 0;
    double s = std::sqrt(disc);
    double q = -0.5 * (b + std::copysign(s, b));
    t0 = q / a;
    t1 = (q != 0) ? c / q : t0;
    if (t0 > t1) std::swap(t0, t1);
    return 2;
}
// Distant origins lose precision in the quadratic's coefficients (cancellation in c and in
// the discriminant). Re-solving from the approximate hit point, where all quantities are
// small, and adding the correction restores near-full precision at any distance.
template <class Coeffs>
double refineRoot(const Ray& r, double t, Coeffs coeffs) {
    Vec3 o1 = r.at(t);
    double A, B, C;
    coeffs(o1, A, B, C);
    double u0, u1;
    int n = solveQuadratic(A, B, C, u0, u1);
    if (n == 0) return t;
    double du = std::abs(u0) < std::abs(u1) ? u0 : u1;
    return std::abs(du) < 1e-6 * (1 + std::abs(t)) ? t + du : t;
}
// refineRoot moves a root by less than 1e-6 (1 + |t|): a root farther than that outside
// (tmin, tmax) stays outside, so its refinement is skipped without changing any result.
inline bool mayLieIn(double t, double tmin, double tmax) {
    double m = 1e-6 * (1 + std::abs(t));
    return !(t + m <= tmin || t - m >= tmax);
}
}  // namespace

// ---------------------------------------------------------------- SagSurface

SagSurface::SagSurface(double curvature, double conic, std::vector<double> asphere, double rMax, double rMin)
    : c_(curvature), k_(conic), A_(std::move(asphere)), rMax_(rMax), rMin_(rMin) {
    while (!A_.empty() && A_.back() == 0) A_.pop_back();
    if (!(rMax_ > 0) || rMin_ < 0 || rMin_ >= rMax_) throw std::runtime_error("sag surface: invalid aperture radii");
    if (1 - (1 + k_) * c_ * c_ * rMax_ * rMax_ < 0)
        throw std::runtime_error("sag surface: aperture exceeds the domain of the conic (|R| too small for semi-diameter)");
    zMin_ = Inf;
    zMax_ = -Inf;
    const int N = 512;
    for (int i = 0; i <= N; ++i) {
        double r = rMin_ + (rMax_ - rMin_) * i / N;
        double z = sag(r);
        zMin_ = std::min(zMin_, z);
        zMax_ = std::max(zMax_, z);
    }
    // Account for extrema between samples of high-order aspheres.
    double pad = 1e-9 + 1e-3 * (zMax_ - zMin_);
    zMin_ -= pad;
    zMax_ += pad;
}

double SagSurface::sag(double r) const {
    double r2 = r * r;
    double z = c_ * r2 / (1 + safeSqrt(1 - (1 + k_) * c_ * c_ * r2));
    double rp = r2 * r2;  // r^4
    for (double a : A_) { z += a * rp; rp *= r2; }
    return z;
}

double SagSurface::dsag(double r) const {
    double r2 = r * r;
    double root = safeSqrt(1 - (1 + k_) * c_ * c_ * r2);
    double d = root > 0 ? c_ * r / root : (c_ != 0 ? Inf : 0);
    double m = 4;
    double rp = r2 * r;  // r^3
    for (double a : A_) { d += a * m * rp; rp *= r2; m += 2; }
    return d;
}

Vec3 SagSurface::normalAt(const Vec3& p) const {
    double r = std::sqrt(p.x * p.x + p.y * p.y);
    if (r < 1e-300) return {0, 0, 1};
    double s = dsag(r);
    return normalize(Vec3(-s * p.x / r, -s * p.y / r, 1));
}

bool SagSurface::accept(const Vec3& p) const {
    double r2 = p.x * p.x + p.y * p.y;
    return r2 <= rMax_ * rMax_ && r2 >= rMin_ * rMin_;
}

bool SagSurface::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    return A_.empty() ? intersectQuadric(r, tmin, tmax, h) : intersectNumeric(r, tmin, tmax, h);
}

bool SagSurface::intersectQuadric(const Ray& ray, double tmin, double tmax, LocalHit& h) const {
    const Vec3& d = ray.d;
    double t0, t1;
    int n;
    // c(x²+y²) + c(1+k) z² − 2z = 0
    double ck = c_ * (1 + k_);
    auto coeffs = [&](const Vec3& o, double& A, double& B, double& C) {
        A = c_ * (d.x * d.x + d.y * d.y) + ck * d.z * d.z;
        B = 2 * (c_ * (o.x * d.x + o.y * d.y) + ck * o.z * d.z - d.z);
        C = c_ * (o.x * o.x + o.y * o.y) + ck * o.z * o.z - 2 * o.z;
    };
    if (c_ == 0) {
        if (d.z == 0) return false;
        t0 = t1 = -ray.o.z / d.z;
        n = 1;
    } else {
        double A, B, C;
        coeffs(ray.o, A, B, C);
        n = solveQuadratic(A, B, C, t0, t1);
    }
    for (int i = 0; i < n; ++i) {
        double t = i == 0 ? t0 : t1;
        if (c_ != 0) {
            if (!mayLieIn(t, tmin, tmax)) continue;
            t = refineRoot(ray, t, coeffs);
        }
        if (t <= tmin || t >= tmax) continue;
        Vec3 p = ray.at(t);
        if (!accept(p)) continue;
        if (c_ != 0 && (1 + k_) * c_ * p.z > 1 + 1e-12) continue;  // wrong sheet of the quadric
        h.t = t;
        h.p = p;
        h.n = normalAt(p);
        h.prim = 0;
        return true;
    }
    return false;
}

bool SagSurface::intersectNumeric(const Ray& ray, double tmin, double tmax, LocalHit& h) const {
    const Vec3& o = ray.o;
    const Vec3& d = ray.d;
    double ta = tmin, tb = tmax;
    // Clip to the cylinder r ≤ rMax.
    double a = d.x * d.x + d.y * d.y;
    double b = 2 * (o.x * d.x + o.y * d.y);
    double c = o.x * o.x + o.y * o.y - rMax_ * rMax_;
    if (a < 1e-300) {
        if (c > 0) return false;
    } else {
        double q0, q1;
        if (solveQuadratic(a, b, c, q0, q1) < 2) return false;
        ta = std::max(ta, q0);
        tb = std::min(tb, q1);
    }
    // Clip to the slab zMin ≤ z ≤ zMax.
    if (std::abs(d.z) < 1e-300) {
        if (o.z < zMin_ || o.z > zMax_) return false;
    } else {
        double s0 = (zMin_ - o.z) / d.z, s1 = (zMax_ - o.z) / d.z;
        if (s0 > s1) std::swap(s0, s1);
        ta = std::max(ta, s0);
        tb = std::min(tb, s1);
    }
    if (!(ta < tb)) return false;
    auto f = [&](double t) {
        Vec3 p = ray.at(t);
        return sag(std::sqrt(p.x * p.x + p.y * p.y)) - p.z;
    };
    const int N = 96;
    double tPrev = ta, fPrev = f(ta);
    for (int i = 1; i <= N; ++i) {
        double tCur = ta + (tb - ta) * i / N;
        double fCur = f(tCur);
        if ((fPrev <= 0 && fCur >= 0) || (fPrev >= 0 && fCur <= 0)) {
            if (fPrev == 0 && tPrev <= tmin) { tPrev = tCur; fPrev = fCur; continue; }
            double lo = tPrev, hi = tCur, flo = fPrev;
            for (int it = 0; it < 80 && hi - lo > 1e-15 * (1 + std::abs(hi)); ++it) {
                double mid = 0.5 * (lo + hi);
                double fm = f(mid);
                if ((fm <= 0) == (flo <= 0)) { lo = mid; flo = fm; } else { hi = mid; }
            }
            double t = 0.5 * (lo + hi);
            Vec3 p = ray.at(t);
            if (t > tmin && t < tmax && accept(p)) {
                h.t = t;
                h.p = p;
                h.n = normalAt(p);
                h.prim = 0;
                return true;
            }
        }
        tPrev = tCur;
        fPrev = fCur;
    }
    return false;
}

AABB SagSurface::bounds() const {
    AABB b;
    b.expand(Vec3(-rMax_, -rMax_, zMin_));
    b.expand(Vec3(rMax_, rMax_, zMax_));
    return b;
}

std::string SagSurface::describe() const {
    std::ostringstream os;
    os << "sag(R=" << (c_ == 0 ? std::string("inf") : std::to_string(1.0 / c_)) << ", k=" << k_;
    for (size_t i = 0; i < A_.size(); ++i) os << ", A" << 4 + 2 * i << "=" << A_[i];
    os << ", r=[" << rMin_ << "," << rMax_ << "])";
    return os.str();
}

void SagSurface::sampleArea(double u1, double u2, double, Vec3& p, Vec3& n) const {
    double r = std::sqrt(rMin_ * rMin_ + u1 * (rMax_ * rMax_ - rMin_ * rMin_));
    double phi = 2 * Pi * u2;
    p = {r * std::cos(phi), r * std::sin(phi), 0};
    n = {0, 0, 1};
}

// ---------------------------------------------------------------- PlaneShape

std::shared_ptr<PlaneShape> PlaneShape::disk(double rMax, double rMin) {
    if (!(rMax > 0) || rMin < 0 || rMin >= rMax) throw std::runtime_error("disk: invalid radii");
    auto s = std::make_shared<PlaneShape>();
    s->ap_ = Aperture::Disk;
    s->a_ = rMax;
    s->b_ = rMin;
    return s;
}
std::shared_ptr<PlaneShape> PlaneShape::diskWithHole(double rMax, double holeR, double hx, double hy) {
    if (!(rMax > 0) || !(holeR > 0) || std::sqrt(hx * hx + hy * hy) + holeR >= rMax)
        throw std::runtime_error("disk with hole: the hole must lie inside the disk");
    auto s = std::make_shared<PlaneShape>();
    s->ap_ = Aperture::DiskHole;
    s->a_ = rMax;
    s->b_ = holeR;
    s->hx_ = hx;
    s->hy_ = hy;
    return s;
}

std::shared_ptr<PlaneShape> PlaneShape::rect(double hx, double hy) {
    if (!(hx > 0 && hy > 0)) throw std::runtime_error("rect: invalid size");
    auto s = std::make_shared<PlaneShape>();
    s->ap_ = Aperture::Rect;
    s->a_ = hx;
    s->b_ = hy;
    return s;
}
std::shared_ptr<PlaneShape> PlaneShape::ellipse(double ax, double ay) {
    if (!(ax > 0 && ay > 0)) throw std::runtime_error("ellipse: invalid size");
    auto s = std::make_shared<PlaneShape>();
    s->ap_ = Aperture::Ellipse;
    s->a_ = ax;
    s->b_ = ay;
    return s;
}

bool PlaneShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    if (r.d.z == 0) return false;
    double t = -r.o.z / r.d.z;
    if (t <= tmin || t >= tmax) return false;
    Vec3 p = r.at(t);
    p.z = 0;
    switch (ap_) {
        case Aperture::Disk: {
            double r2 = p.x * p.x + p.y * p.y;
            if (r2 > a_ * a_ || r2 < b_ * b_) return false;
            break;
        }
        case Aperture::Rect:
            if (std::abs(p.x) > a_ || std::abs(p.y) > b_) return false;
            break;
        case Aperture::Ellipse:
            if (sqr(p.x / a_) + sqr(p.y / b_) > 1) return false;
            break;
        case Aperture::DiskHole:
            if (p.x * p.x + p.y * p.y > a_ * a_ || sqr(p.x - hx_) + sqr(p.y - hy_) < b_ * b_) return false;
            break;
    }
    h.t = t;
    h.p = p;
    h.n = {0, 0, 1};
    h.prim = 0;
    return true;
}

AABB PlaneShape::bounds() const {
    AABB b;
    double ex = a_, ey = (ap_ == Aperture::Disk || ap_ == Aperture::DiskHole) ? a_ : b_;
    double pad = 1e-9 * (1 + std::max(ex, ey));
    b.expand(Vec3(-ex, -ey, -pad));
    b.expand(Vec3(ex, ey, pad));
    return b;
}

std::string PlaneShape::describe() const {
    char buf[128];
    switch (ap_) {
        case Aperture::Disk: std::snprintf(buf, sizeof buf, "plane-disk(r=[%g,%g])", b_, a_); break;
        case Aperture::Rect: std::snprintf(buf, sizeof buf, "plane-rect(%g x %g)", 2 * a_, 2 * b_); break;
        case Aperture::Ellipse: std::snprintf(buf, sizeof buf, "plane-ellipse(%g x %g)", 2 * a_, 2 * b_); break;
        case Aperture::DiskHole:
            std::snprintf(buf, sizeof buf, "plane-disk(r=%g, hole r=%g at %g,%g)", a_, b_, hx_, hy_);
            break;
    }
    return buf;
}

double PlaneShape::area() const {
    switch (ap_) {
        case Aperture::Disk: return Pi * (a_ * a_ - b_ * b_);
        case Aperture::Rect: return 4 * a_ * b_;
        case Aperture::Ellipse: return Pi * a_ * b_;
        case Aperture::DiskHole: return Pi * (a_ * a_ - b_ * b_);
    }
    return 0;
}

void PlaneShape::sampleArea(double u1, double u2, double, Vec3& p, Vec3& n) const {
    n = {0, 0, 1};
    switch (ap_) {
        case Aperture::Disk: {
            double r = std::sqrt(b_ * b_ + u1 * (a_ * a_ - b_ * b_));
            double phi = 2 * Pi * u2;
            p = {r * std::cos(phi), r * std::sin(phi), 0};
            return;
        }
        case Aperture::Rect: p = {(2 * u1 - 1) * a_, (2 * u2 - 1) * b_, 0}; return;
        case Aperture::Ellipse: {
            double r = std::sqrt(u1), phi = 2 * Pi * u2;
            p = {a_ * r * std::cos(phi), b_ * r * std::sin(phi), 0};
            return;
        }
        case Aperture::DiskHole: p = {}; return;  // not sampleable (see canSample)
    }
}

// ---------------------------------------------------------------- SphereShape

bool SphereShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    // Geometric formulation for accuracy with distant origins.
    double b = dot(r.o, r.d);
    Vec3 perp = r.o - r.d * b;
    double disc = R_ * R_ - lengthSq(perp);
    if (disc < 0) return false;
    double s = std::sqrt(disc);
    double c = lengthSq(r.o) - R_ * R_;
    double q = -b - std::copysign(s, b);
    double t0 = q, t1 = (q != 0) ? c / q : q;
    if (t0 > t1) std::swap(t0, t1);
    auto coeffs = [&](const Vec3& o, double& A, double& B, double& C) {
        A = dot(r.d, r.d);
        B = 2 * dot(o, r.d);
        C = lengthSq(o) - R_ * R_;
    };
    auto accept = [&](double t0, double& t) {
        if (!mayLieIn(t0, tmin, tmax)) return false;
        t = refineRoot(r, t0, coeffs);
        return !(t <= tmin || t >= tmax);
    };
    double t;
    if (!accept(t0, t) && !accept(t1, t)) return false;
    h.t = t;
    h.p = r.at(t);
    h.n = h.p / R_;
    h.n = normalize(h.n);
    h.prim = 0;
    return true;
}

AABB SphereShape::bounds() const {
    AABB b;
    b.expand(Vec3(-R_, -R_, -R_));
    b.expand(Vec3(R_, R_, R_));
    return b;
}

std::string SphereShape::describe() const { return "sphere(r=" + std::to_string(R_) + ")"; }

void SphereShape::sampleArea(double u1, double u2, double, Vec3& p, Vec3& n) const {
    n = sampleUniformSphere(u1, u2);
    p = n * R_;
}

bool SphereShape::sampleFrom(const Vec3& ref, double u1, double u2, double u3, Vec3& p, Vec3& n, double& pdfW) const {
    double dc2 = lengthSq(ref);
    if (dc2 <= R_ * R_ * (1 + 1e-9)) return Shape::sampleFrom(ref, u1, u2, u3, p, n, pdfW);
    double dc = std::sqrt(dc2);
    double sin2Max = R_ * R_ / dc2;
    double cosMax = safeSqrt(1 - sin2Max);
    double oneMinusCos = sin2Max / (1 + cosMax);
    Vec3 axis = -ref / dc;
    Frame f(axis);
    // Sample cosθ in [cosMax, 1] with good precision for small cones.
    double oneMinusCosT = u1 * oneMinusCos;
    double cosT = 1 - oneMinusCosT;
    double sinT = safeSqrt(oneMinusCosT * (2 - oneMinusCosT));
    double phi = 2 * Pi * u2;
    Vec3 w = f.toWorld({sinT * std::cos(phi), sinT * std::sin(phi), cosT});
    LocalHit h;
    Ray ray{ref, w};
    if (!intersect(ray, 0, Inf, h)) {
        // Grazing: take the tangent point.
        double tt = dot(-ref, w);
        p = normalize(ref + w * tt) * R_;
    } else {
        p = h.p;
    }
    n = normalize(p);
    pdfW = 1 / (2 * Pi * oneMinusCos);
    return true;
}

double SphereShape::pdfFrom(const Vec3& ref, const Vec3& p, const Vec3& n) const {
    double dc2 = lengthSq(ref);
    if (dc2 <= R_ * R_ * (1 + 1e-9)) return Shape::pdfFrom(ref, p, n);
    double sin2Max = R_ * R_ / dc2;
    double cosMax = safeSqrt(1 - sin2Max);
    return 1 / (2 * Pi * sin2Max / (1 + cosMax));
}

// ---------------------------------------------------------------- CylinderShape

bool CylinderShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    double a = r.d.x * r.d.x + r.d.y * r.d.y;
    if (a < 1e-300) return false;
    double b = 2 * (r.o.x * r.d.x + r.o.y * r.d.y);
    double c = r.o.x * r.o.x + r.o.y * r.o.y - R_ * R_;
    double t0, t1;
    if (solveQuadratic(a, b, c, t0, t1) == 0) return false;
    auto coeffs = [&](const Vec3& o, double& A, double& B, double& C) {
        A = a;
        B = 2 * (o.x * r.d.x + o.y * r.d.y);
        C = o.x * o.x + o.y * o.y - R_ * R_;
    };
    for (double t : {t0, t1}) {
        if (!mayLieIn(t, tmin, tmax)) continue;
        t = refineRoot(r, t, coeffs);
        if (t <= tmin || t >= tmax) continue;
        Vec3 p = r.at(t);
        if (p.z < z0_ || p.z > z1_) continue;
        h.t = t;
        h.p = p;
        h.n = normalize(Vec3(p.x, p.y, 0));
        h.prim = 0;
        return true;
    }
    return false;
}

AABB CylinderShape::bounds() const {
    AABB b;
    b.expand(Vec3(-R_, -R_, z0_));
    b.expand(Vec3(R_, R_, z1_));
    return b;
}

std::string CylinderShape::describe() const {
    char buf[128];
    std::snprintf(buf, sizeof buf, "cylinder(r=%g, z=[%g,%g])", R_, z0_, z1_);
    return buf;
}

void CylinderShape::sampleArea(double u1, double u2, double, Vec3& p, Vec3& n) const {
    double phi = 2 * Pi * u1;
    n = {std::cos(phi), std::sin(phi), 0};
    p = {R_ * n.x, R_ * n.y, z0_ + u2 * (z1_ - z0_)};
}

// ---------------------------------------------------------------- RoundWallShape

bool RoundWallShape::solid(const Vec3& p) const {
    if (p.z < 0 || p.z > H_) return false;
    double az = std::atan2(p.x, p.y);
    for (const auto& o : open_) {
        double dAz = std::remainder(az - o.azimuth, 2 * Pi);
        if (std::abs(dAz) * R_ < 0.5 * o.width && p.z > o.sill && p.z < o.top) return false;
    }
    return true;
}

bool RoundWallShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    double a = r.d.x * r.d.x + r.d.y * r.d.y;
    if (a < 1e-300) return false;
    double b = 2 * (r.o.x * r.d.x + r.o.y * r.d.y);
    double c = r.o.x * r.o.x + r.o.y * r.o.y - R_ * R_;
    double t0, t1;
    if (solveQuadratic(a, b, c, t0, t1) == 0) return false;
    auto coeffs = [&](const Vec3& o, double& A, double& B, double& C) {
        A = a;
        B = 2 * (o.x * r.d.x + o.y * r.d.y);
        C = o.x * o.x + o.y * o.y - R_ * R_;
    };
    for (double t : {t0, t1}) {
        if (!mayLieIn(t, tmin, tmax)) continue;
        t = refineRoot(r, t, coeffs);
        if (t <= tmin || t >= tmax) continue;
        Vec3 p = r.at(t);
        if (!solid(p)) continue;
        h.t = t;
        h.p = p;
        h.n = normalize(Vec3(p.x, p.y, 0));
        h.prim = 0;
        return true;
    }
    return false;
}

AABB RoundWallShape::bounds() const {
    AABB b;
    b.expand(Vec3(-R_, -R_, 0));
    b.expand(Vec3(R_, R_, H_));
    return b;
}

std::string RoundWallShape::describe() const {
    return "round-wall(r=" + std::to_string(R_) + ", h=" + std::to_string(H_) + ", " + std::to_string(open_.size()) +
           " openings)";
}

// ---------------------------------------------------------------- DomeShape

bool DomeShape::solid(const Vec3& p) const {
    if (p.z < 0) return false;
    if (w_ <= 0) return true;
    double el = std::asin(clampd(p.z / R_, -1, 1));
    if (el >= top_) return true;
    double dAz = std::remainder(std::atan2(p.x, p.y) - az_, 2 * Pi);
    double horiz = R_ * std::cos(el);
    // Constant linear width: the half-angle grows toward the zenith.
    return std::abs(std::sin(dAz)) * horiz >= 0.5 * w_ || std::cos(dAz) < 0;
}

bool DomeShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    double b = dot(r.o, r.d);
    Vec3 perp = r.o - r.d * b;
    double disc = R_ * R_ - lengthSq(perp);
    if (disc < 0) return false;
    double sq = std::sqrt(disc);
    double c = lengthSq(r.o) - R_ * R_;
    double q = -b - std::copysign(sq, b);
    double t0 = q, t1 = (q != 0) ? c / q : q;
    if (t0 > t1) std::swap(t0, t1);
    auto coeffs = [&](const Vec3& o, double& A, double& B, double& C) {
        A = dot(r.d, r.d);
        B = 2 * dot(o, r.d);
        C = lengthSq(o) - R_ * R_;
    };
    for (double t : {t0, t1}) {
        if (!mayLieIn(t, tmin, tmax)) continue;
        t = refineRoot(r, t, coeffs);
        if (t <= tmin || t >= tmax) continue;
        Vec3 p = r.at(t);
        if (!solid(p)) continue;
        h.t = t;
        h.p = p;
        h.n = normalize(p);
        h.prim = 0;
        return true;
    }
    return false;
}

AABB DomeShape::bounds() const {
    AABB b;
    b.expand(Vec3(-R_, -R_, 0));
    b.expand(Vec3(R_, R_, R_));
    return b;
}

std::string DomeShape::describe() const { return "dome(r=" + std::to_string(R_) + ", slit width " + std::to_string(w_) + ")"; }

// ---------------------------------------------------------------- MeshShape

MeshShape::MeshShape(std::vector<Vec3> positions, std::vector<std::array<uint32_t, 3>> triangles, std::string label)
    : pos_(std::move(positions)), tris_(std::move(triangles)), label_(std::move(label)) {
    std::vector<AABB> boxes(tris_.size());
    cdf_.resize(tris_.size());
    double acc = 0;
    for (size_t i = 0; i < tris_.size(); ++i) {
        const auto& t = tris_[i];
        for (uint32_t v : t)
            if (v >= pos_.size()) throw std::runtime_error("mesh: triangle index out of range");
        const Vec3 &a = pos_[t[0]], &b = pos_[t[1]], &c = pos_[t[2]];
        boxes[i].expand(a);
        boxes[i].expand(b);
        boxes[i].expand(c);
        double pad = 1e-12 * (1 + maxAbsComponent(boxes[i].hi) + maxAbsComponent(boxes[i].lo));
        boxes[i].lo -= Vec3(pad, pad, pad);
        boxes[i].hi += Vec3(pad, pad, pad);
        acc += 0.5 * length(cross(b - a, c - a));
        cdf_[i] = acc;
    }
    area_ = acc;
    bvh_.build(boxes);
}

bool MeshShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    bool found = false;
    double best = tmax;
    bvh_.traverse(r, tmin, best, [&](uint32_t i, double& tm) {
        const auto& t = tris_[i];
        const Vec3 &v0 = pos_[t[0]], &v1 = pos_[t[1]], &v2 = pos_[t[2]];
        Vec3 e1 = v1 - v0, e2 = v2 - v0;
        Vec3 pv = cross(r.d, e2);
        double det = dot(e1, pv);
        if (std::abs(det) < 1e-300) return;
        double inv = 1 / det;
        Vec3 tv = r.o - v0;
        double u = dot(tv, pv) * inv;
        if (u < 0 || u > 1) return;
        Vec3 qv = cross(tv, e1);
        double v = dot(r.d, qv) * inv;
        if (v < 0 || u + v > 1) return;
        double tt = dot(e2, qv) * inv;
        if (tt <= tmin || tt >= tm) return;
        tm = tt;
        found = true;
        h.t = tt;
        h.prim = i;
    });
    if (!found) return false;
    const auto& t = tris_[h.prim];
    h.p = r.at(h.t);
    h.n = normalize(cross(pos_[t[1]] - pos_[t[0]], pos_[t[2]] - pos_[t[0]]));
    return true;
}

std::string MeshShape::describe() const { return label_ + "(" + std::to_string(tris_.size()) + " triangles)"; }

void MeshShape::sampleArea(double u1, double u2, double u3, Vec3& p, Vec3& n) const {
    double target = u3 * area_;
    size_t i = size_t(std::lower_bound(cdf_.begin(), cdf_.end(), target) - cdf_.begin());
    if (i >= tris_.size()) i = tris_.size() - 1;
    const auto& t = tris_[i];
    double su = std::sqrt(u1);
    double b0 = 1 - su, b1 = u2 * su;
    p = pos_[t[0]] * b0 + pos_[t[1]] * b1 + pos_[t[2]] * (1 - b0 - b1);
    n = normalize(cross(pos_[t[1]] - pos_[t[0]], pos_[t[2]] - pos_[t[0]]));
}

// ---------------------------------------------------------------- MeshData

void MeshData::addTriangle(const Vec3& a, const Vec3& b, const Vec3& c) {
    uint32_t base = uint32_t(positions.size());
    positions.push_back(a);
    positions.push_back(b);
    positions.push_back(c);
    triangles.push_back({base, base + 1, base + 2});
}

void MeshData::addQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    uint32_t base = uint32_t(positions.size());
    positions.insert(positions.end(), {a, b, c, d});
    triangles.push_back({base, base + 1, base + 2});
    triangles.push_back({base, base + 2, base + 3});
}

void MeshData::append(const MeshData& o, const Transform& xf) {
    uint32_t base = uint32_t(positions.size());
    for (const Vec3& p : o.positions) positions.push_back(xf.point(p));
    for (const auto& t : o.triangles) triangles.push_back({t[0] + base, t[1] + base, t[2] + base});
}

MeshData loadObj(const std::string& path, double scale) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open OBJ file: " + path);
    std::string buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    buf.push_back('\n');
    MeshData m;
    const char* p = buf.c_str();
    const char* end = p + buf.size();
    std::vector<long> idx;
    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t')) ++p;
        if (p + 1 < end && p[0] == 'v' && (p[1] == ' ' || p[1] == '\t')) {
            char* q;
            Vec3 v;
            v.x = std::strtod(p + 2, &q);
            v.y = std::strtod(q, &q);
            v.z = std::strtod(q, &q);
            m.positions.push_back(v * scale);
            p = q;
        } else if (p + 1 < end && p[0] == 'f' && (p[1] == ' ' || p[1] == '\t')) {
            idx.clear();
            const char* q = p + 2;
            while (true) {
                while (*q == ' ' || *q == '\t') ++q;
                if (*q == '\n' || *q == '\r' || *q == '\0') break;
                char* e;
                long v = std::strtol(q, &e, 10);
                if (e == q) break;
                if (v < 0) v = long(m.positions.size()) + v + 1;
                idx.push_back(v - 1);
                q = e;
                while (*q && *q != ' ' && *q != '\t' && *q != '\n' && *q != '\r') ++q;  // skip /vt/vn
            }
            for (size_t i = 1; i + 1 < idx.size(); ++i)
                m.triangles.push_back({uint32_t(idx[0]), uint32_t(idx[i]), uint32_t(idx[i + 1])});
            p = q;
        }
        while (p < end && *p != '\n') ++p;
        ++p;
    }
    if (m.triangles.empty()) throw std::runtime_error("OBJ file has no faces: " + path);
    return m;
}

}  // namespace owe
