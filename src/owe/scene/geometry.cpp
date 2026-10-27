// Analytic shapes: optical sags, liquid waves, planes, spheres, cylinders, cones, torus patches,
// round walls and domes. Their algebra is shared with the GPU kernels (core/shared/surfaces.hpp).
#include "owe/scene/geometry.hpp"

#include <cstdio>
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
    return areaToSolidAngle(ref, p, n, area(), pdfW);
}

double Shape::pdfFrom(const Vec3& ref, const Vec3& p, const Vec3& n) const {
    double pdfW;
    return canSample() && area() > 0 && areaToSolidAngle(ref, p, n, area(), pdfW) ? pdfW : 0;
}

namespace {

// The first of a quadric's roots t0 ≤ t1 that, refined, lies in (tmin, tmax) and is accepted by
// solid(p) (the shape's trim); fills h with that hit and its normal normal(p).
template <class Solid, class Normal>
bool firstRefinedHit(const Ray& r, const Quadric& q, double t0, double t1, double tmin, double tmax, Solid solid,
                     Normal normal, LocalHit& h) {
    for (double t : {t0, t1}) {
        double tr;
        if (!refinedRootIn(q, r.o, r.d, t, tmin, tmax, tr)) continue;
        const Vec3 p = r.at(tr);
        if (!solid(p)) continue;
        h.t = tr;
        h.p = p;
        h.n = normal(p);
        h.prim = 0;
        return true;
    }
    return false;
}

// Bisects a sign change of f in [lo, hi] (f(lo) = flo) to the precision's bracket width.
template <class F>
double bisect(F f, double lo, double hi, double flo, int steps = kBisectSteps) {
    for (int it = 0; it < steps && hi - lo > kBisect * (1 + std::abs(hi)); ++it) {
        const double mid = 0.5 * (lo + hi);
        const double fm = f(mid);
        if ((fm <= 0) == (flo <= 0)) {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
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
        double z = sag(rMin_ + (rMax_ - rMin_) * i / N);
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
    double z = conicSag(c_, k_, r2);
    double rp = r2 * r2;  // r^4
    for (double a : A_) { z += a * rp; rp *= r2; }
    return z;
}

double SagSurface::dsag(double r) const {
    double d = conicSlope(c_, k_, r);
    double m = 4, rp = r * r * r;
    for (double a : A_) { d += a * m * rp; rp *= r * r; m += 2; }
    return d;
}

Vec3 SagSurface::normalAt(const Vec3& p) const {
    double r = std::sqrt(p.x * p.x + p.y * p.y);
    return sagNormal(p, r, dsag(r));
}

bool SagSurface::accept(const Vec3& p) const {
    double r2 = p.x * p.x + p.y * p.y;
    return r2 <= rMax_ * rMax_ && r2 >= rMin_ * rMin_;
}

bool SagSurface::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    return A_.empty() ? intersectQuadric(r, tmin, tmax, h) : intersectNumeric(r, tmin, tmax, h);
}

bool SagSurface::intersectQuadric(const Ray& ray, double tmin, double tmax, LocalHit& h) const {
    auto solid = [&](const Vec3& p) { return accept(p) && onConicSheet(c_, k_, p.z); };
    auto normal = [&](const Vec3& p) { return normalAt(p); };
    if (c_ == 0) {  // a plane: one exact root
        if (ray.d.z == 0) return false;
        const double t = -ray.o.z / ray.d.z;
        if (t <= tmin || t >= tmax || !solid(ray.at(t))) return false;
        h.t = t;
        h.p = ray.at(t);
        h.n = normal(h.p);
        h.prim = 0;
        return true;
    }
    const Quadric q = makeQuadric(kQuadSag, c_, c_ * (1 + k_), 0);
    double A, B, C, t0, t1;
    quadricCoeffs(q, ray.o, ray.d, A, B, C);
    return solveQuadratic(A, B, C, t0, t1) > 0 && firstRefinedHit(ray, q, t0, t1, tmin, tmax, solid, normal, h);
}

bool SagSurface::intersectNumeric(const Ray& ray, double tmin, double tmax, LocalHit& h) const {
    double ta = tmin, tb = tmax;
    if (!clipCylinder(ray.o, ray.d, rMax_, ta, tb) || !clipSlab(ray.o.z, ray.d.z, zMin_, zMax_, ta, tb)) return false;
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
            double t = bisect(f, tPrev, tCur, fPrev);
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
    p = sampleAnnulus(rMin_, rMax_, u1, u2);
    n = {0, 0, 1};
}

// ---------------------------------------------------------------- WaveSurface

WaveSurface::WaveSurface(std::vector<PlaneWave> waves, double halfX, double halfY, double margin, double spacing)
    : waves_(std::move(waves)), hx_(halfX), hy_(halfY), margin_(std::max(0.0, std::min({margin, halfX, halfY}))) {
    double slope = 0, curvature = 0, kMax = 0;
    for (const PlaneWave& w : waves_) {
        const double k = std::hypot(w.kx, w.ky);
        amp_ += std::abs(w.amplitude);
        slope += std::abs(w.amplitude) * k;
        curvature += std::abs(w.amplitude) * k * k;
        kMax = std::max(kMax, k);
    }
    // |∇(w g)| ≤ |∇g| + |g|·|∇w|, with |∂w/∂u| ≤ 1.5 / margin on each axis.
    const double taper = margin_ > 0 ? 1.5 * std::sqrt(2.0) / margin_ : 0;
    slope_ = slope + amp_ * taper;
    if (spacing < 0) spacing = waves_.size() <= 6 || kMax == 0 ? 0 : (2 * Pi / kMax) / 12;
    if (!(spacing > 0)) return;
    nx_ = std::clamp(int(std::ceil(2 * hx_ / spacing)) + 1, 2, 4096);
    ny_ = std::clamp(int(std::ceil(2 * hy_ / spacing)) + 1, 2, 4096);
    dx_ = 2 * hx_ / (nx_ - 1);
    dy_ = 2 * hy_ / (ny_ - 1);
    nodes_.resize(size_t(nx_) * ny_);
    double nodeSlope = 0;
    for (int j = 0; j < ny_; ++j)
        for (int i = 0; i < nx_; ++i) {
            auto& n = nodes_[size_t(j) * nx_ + i];
            analytic(-hx_ + i * dx_, -hy_ + j * dy_, n[0], n[1], n[2], n[3]);
            nodeSlope = std::max(nodeSlope, std::hypot(n[1], n[2]));
        }
    // Between nodes the slope exceeds its nodal maximum by at most the curvature bound times half a
    // cell diagonal; 5% more covers the spline's own deviation from the sum.
    const double hessian = curvature + 2 * taper * slope + (margin_ > 0 ? 6 * amp_ / (margin_ * margin_) : 0);
    slope_ = std::min(slope_, 1.05 * (nodeSlope + hessian * 0.5 * std::hypot(dx_, dy_)));
}

void WaveSurface::analytic(double x, double y, double& h, double& dhdx, double& dhdy, double& dhdxy) const {
    h = dhdx = dhdy = dhdxy = 0;
    if (std::abs(x) > hx_ || std::abs(y) > hy_) return;
    double g = 0, gx = 0, gy = 0, gxy = 0;
    for (const PlaneWave& w : waves_) {
        const double arg = w.kx * x + w.ky * y + w.phase;
        const double s = std::sin(arg), c = std::cos(arg);
        g += w.amplitude * s;
        gx += w.amplitude * c * w.kx;
        gy += w.amplitude * c * w.ky;
        gxy -= w.amplitude * s * w.kx * w.ky;
    }
    double wx, dwx, wy, dwy;
    waveTaper(x, hx_, margin_, wx, dwx);
    waveTaper(y, hy_, margin_, wy, dwy);
    h = wx * wy * g;
    dhdx = dwx * wy * g + wx * wy * gx;
    dhdy = wx * dwy * g + wx * wy * gy;
    dhdxy = dwx * dwy * g + dwx * wy * gy + wx * dwy * gx + wx * wy * gxy;
}

void WaveSurface::height(double x, double y, double& h, double& dhdx, double& dhdy) const {
    if (nodes_.empty()) {
        double hxy;
        analytic(x, y, h, dhdx, dhdy, hxy);
        return;
    }
    h = dhdx = dhdy = 0;
    if (std::abs(x) > hx_ || std::abs(y) > hy_) return;
    // Bicubic Hermite patch of the cell containing (x, y).
    const double fx = (x + hx_) / dx_, fy = (y + hy_) / dy_;
    const int i = std::clamp(int(fx), 0, nx_ - 2), j = std::clamp(int(fy), 0, ny_ - 2);
    Vec2 P, Q, dP, dQ, Pv, Qv, dPv, dQv;
    hermiteBases(fx - i, P, Q, dP, dQ);
    hermiteBases(fy - j, Pv, Qv, dPv, dQv);
    double f = 0, fu = 0, fv = 0;
    for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a) {
            const auto& n = nodes_[size_t(j + b) * nx_ + (i + a)];
            const double c0 = n[0], cu = n[1] * dx_, cv = n[2] * dy_, cuv = n[3] * dx_ * dy_;
            f += c0 * P[a] * Pv[b] + cu * Q[a] * Pv[b] + cv * P[a] * Qv[b] + cuv * Q[a] * Qv[b];
            fu += c0 * dP[a] * Pv[b] + cu * dQ[a] * Pv[b] + cv * dP[a] * Qv[b] + cuv * dQ[a] * Qv[b];
            fv += c0 * P[a] * dPv[b] + cu * Q[a] * dPv[b] + cv * P[a] * dQv[b] + cuv * Q[a] * dQv[b];
        }
    h = f;
    dhdx = fu / dx_;
    dhdy = fv / dy_;
}

Vec3 WaveSurface::normalAt(double x, double y) const {
    double h, gx, gy;
    height(x, y, h, gx, gy);
    return normalize(Vec3{-gx, -gy, 1});
}

bool WaveSurface::intersect(const Ray& ray, double tmin, double tmax, LocalHit& hit) const {
    const Vec3& d = ray.d;
    // The surface lies in the box |x| ≤ hx, |y| ≤ hy, |z| ≤ amplitude bound.
    double ta = tmin, tb = tmax;
    if (!clipSlab(ray.o.x, d.x, -hx_, hx_, ta, tb) || !clipSlab(ray.o.y, d.y, -hy_, hy_, ta, tb) ||
        !clipSlab(ray.o.z, d.z, -amp_, amp_, ta, tb))
        return false;
    // f(t) = z(t) − h(x(t), y(t)) changes by at most L per unit t: a step |f|/L cannot cross a root.
    const double L = std::abs(d.z) + slope_ * std::hypot(d.x, d.y);
    const double tol = 1e-12 * (1 + std::max({hx_, hy_, amp_}));
    auto f = [&](double t, double& dfdt) {
        const Vec3 p = ray.at(t);
        double h, gx, gy;
        height(p.x, p.y, h, gx, gy);
        dfdt = d.z - gx * d.x - gy * d.y;
        return p.z - h;
    };
    double t = ta, dfdt;
    double ft = f(t, dfdt);
    // A ray leaving the surface (its origin offset onto one side) must not find the surface it left.
    if (t <= tmin && std::abs(ft) < tol) {
        t = tmin + tol / std::max(L, 1e-300);
        ft = f(t, dfdt);
    }
    const double minStep = 1e-7 * (tb - ta), nearSurface = 1e-6 * (amp_ + tol);
    auto accept = [&](double tr) {
        const Vec3 p = ray.at(tr);
        if (!(tr > tmin && tr < tmax) || std::abs(p.x) > hx_ || std::abs(p.y) > hy_) return false;
        hit.t = tr;
        hit.p = p;
        hit.n = normalAt(p.x, p.y);
        hit.prim = 0;
        double heightValue, gx, gy;
        height(p.x, p.y, heightValue, gx, gy);
        hit.error = std::abs(p.z - heightValue) / std::sqrt(1 + gx * gx + gy * gy);
        if (ray.departedPrimitive != 0xFFFFFFFEu &&
            std::abs(tr * dot(d, hit.n)) <= ray.departureError + hit.error &&
            dot(d, hit.n) * dot(d, ray.departedNormal) >= 0) return false;
        return true;
    };
    for (int i = 0; i < 4096 && t < tb; ++i) {
        double step = std::max(std::abs(ft) / L, minStep);
        if (std::abs(ft) < nearSurface) {
            // Close to the surface: Newton from here reaches the root being approached.
            double tn = t;
            for (int k = 0; k < 8; ++k) {
                const double fn = f(tn, dfdt);
                const double arithmeticTol = 8 * std::numeric_limits<double>::epsilon() *
                    std::max({1e-12, amp_, maxAbsComponent(ray.at(tn))});
                if (std::abs(fn) < arithmeticTol || std::abs(dfdt) < 1e-12) break;
                tn -= fn / dfdt;
            }
            if (tn >= t - 1e3 * tol && std::abs(f(tn, dfdt)) < 1e3 * tol && accept(tn)) return true;
            step = std::max(minStep, 2 * std::abs(ft) / L);  // a grazing approach: move past it
        }
        const double tn = std::min(t + step, tb);
        const double fn = f(tn, dfdt);
        if ((fn < 0) != (ft < 0)) {
            // A crossing inside a minimum step: bisect it.
            auto value = [&](double x) { double dx; return f(x, dx); };
            if (accept(bisect(value, t, tn, ft))) return true;
        }
        t = tn;
        ft = fn;
    }
    return false;
}

AABB WaveSurface::bounds() const {
    AABB b;
    b.expand({-hx_, -hy_, -amp_});
    b.expand({hx_, hy_, amp_});
    return b;
}

std::string WaveSurface::describe() const {
    char buf[200];
    std::snprintf(buf, sizeof buf, "wave surface %.4g x %.4g m, %zu waves, |h| <= %.3g m, slope <= %.3g, %s", 2 * hx_, 2 * hy_,
                  waves_.size(), amp_, slope_, nodes_.empty() ? "analytic" : "bicubic spline");
    if (!nodes_.empty()) {
        std::string r = buf;
        std::snprintf(buf, sizeof buf, " (%dx%d nodes, %.3g mm)", nx_, ny_, 1e3 * dx_);
        return r + buf;
    }
    return buf;
}

// ---------------------------------------------------------------- PlaneShape

namespace {
std::shared_ptr<PlaneShape> makePlane(PlaneShape::Aperture ap, double a, double b, double hx = 0, double hy = 0) {
    return std::make_shared<PlaneShape>(ap, a, b, hx, hy);
}
}  // namespace

std::shared_ptr<PlaneShape> PlaneShape::disk(double rMax, double rMin) {
    if (!(rMax > 0) || rMin < 0 || rMin >= rMax) throw std::runtime_error("disk: invalid radii");
    return makePlane(Aperture::Disk, rMax, rMin);
}
std::shared_ptr<PlaneShape> PlaneShape::diskWithHole(double rMax, double holeR, double hx, double hy) {
    if (!(rMax > 0) || !(holeR > 0) || std::sqrt(hx * hx + hy * hy) + holeR >= rMax)
        throw std::runtime_error("disk with hole: the hole must lie inside the disk");
    return makePlane(Aperture::DiskHole, rMax, holeR, hx, hy);
}
std::shared_ptr<PlaneShape> PlaneShape::rect(double hx, double hy) {
    if (!(hx > 0 && hy > 0)) throw std::runtime_error("rect: invalid size");
    return makePlane(Aperture::Rect, hx, hy);
}
std::shared_ptr<PlaneShape> PlaneShape::ellipse(double ax, double ay) {
    if (!(ax > 0 && ay > 0)) throw std::runtime_error("ellipse: invalid size");
    return makePlane(Aperture::Ellipse, ax, ay);
}

bool PlaneShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    if (r.d.z == 0) return false;
    double t = -r.o.z / r.d.z;
    if (t <= tmin || t >= tmax) return false;
    Vec3 p = r.at(t);
    p.z = 0;
    if (!inPlaneAperture(uint(ap_), a_, b_, hx_, hy_, p)) return false;
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
        case Aperture::Disk:
        case Aperture::DiskHole: return Pi * (a_ * a_ - b_ * b_);
        case Aperture::Rect: return 4 * a_ * b_;
        case Aperture::Ellipse: return Pi * a_ * b_;
    }
    return 0;
}

void PlaneShape::sampleArea(double u1, double u2, double, Vec3& p, Vec3& n) const {
    n = {0, 0, 1};
    switch (ap_) {
        case Aperture::Disk: p = sampleAnnulus(b_, a_, u1, u2); return;
        case Aperture::Rect: p = sampleRect(a_, b_, u1, u2); return;
        case Aperture::Ellipse: p = sampleEllipse(a_, b_, u1, u2); return;
        case Aperture::DiskHole: p = {}; return;  // not sampleable (see canSample)
    }
}

// ---------------------------------------------------------------- SphereShape

bool SphereShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    double t0, t1;
    return sphereRoots(r.o, r.d, R_, t0, t1) &&
           firstRefinedHit(r, makeQuadric(kQuadSphere, R_ * R_, 0, 0), t0, t1, tmin, tmax,
                           [](const Vec3&) { return true; }, [&](const Vec3& p) { return normalize(p / R_); }, h);
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

// Seen from outside, the sphere is sampled uniformly within the cone it subtends.
bool SphereShape::sampleFrom(const Vec3& ref, double u1, double u2, double u3, Vec3& p, Vec3& n, double& pdfW) const {
    const double oneMinusCos = sphereConeSize(ref, R_);
    if (oneMinusCos == 0) return Shape::sampleFrom(ref, u1, u2, u3, p, n, pdfW);
    const Vec3 w = sampleSphereCone(ref, oneMinusCos, u1, u2);
    LocalHit h;
    if (intersect(Ray{ref, w}, 0, Inf, h)) p = h.p;
    else p = normalize(ref + w * dot(-ref, w)) * R_;  // grazing: the tangent point
    n = normalize(p);
    pdfW = 1 / (2 * Pi * oneMinusCos);
    return true;
}

double SphereShape::pdfFrom(const Vec3& ref, const Vec3& p, const Vec3& n) const {
    const double oneMinusCos = sphereConeSize(ref, R_);
    return oneMinusCos == 0 ? Shape::pdfFrom(ref, p, n) : 1 / (2 * Pi * oneMinusCos);
}

// ---------------------------------------------------------------- CylinderShape

bool CylinderShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    double t0, t1;
    return cylinderRoots(r.o, r.d, R_, t0, t1) &&
           firstRefinedHit(r, makeQuadric(kQuadCylinder, R_ * R_, 0, 0), t0, t1, tmin, tmax,
                           [&](const Vec3& p) { return p.z >= z0_ && p.z <= z1_; },
                           [](const Vec3& p) { return normalize(Vec3(p.x, p.y, 0)); }, h);
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

// ---------------------------------------------------------------- ConeShape

ConeShape::ConeShape(double r0, double z0, double r1, double z1) : r0_(r0), z0_(z0), r1_(r1), z1_(z1) {
    if (!(std::abs(z1 - z0) > 0) || r0 < 0 || r1 < 0) throw std::runtime_error("cone: needs z0 != z1 and radii >= 0");
    if (z0_ > z1_) {
        std::swap(z0_, z1_);
        std::swap(r0_, r1_);
    }
    k_ = (r1_ - r0_) / (z1_ - z0_);
}

bool ConeShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    // x² + y² = (r0 + k (z − z0))² along the ray, on the nappe where r0 + k (z − z0) ≥ 0.
    const Quadric q = makeQuadric(kQuadCone, r0_, z0_, k_);
    double A, B, C, t0, t1;
    quadricCoeffs(q, r.o, r.d, A, B, C);
    auto rho = [&](const Vec3& p) { return r0_ + k_ * (p.z - z0_); };
    return solveQuadratic(A, B, C, t0, t1) > 0 &&
           firstRefinedHit(r, q, t0, t1, tmin, tmax,
                           [&](const Vec3& p) { return p.z >= z0_ && p.z <= z1_ && rho(p) >= 0; },
                           [&](const Vec3& p) { return coneNormal(p, rho(p), k_); }, h);
}

AABB ConeShape::bounds() const {
    const double R = std::max(r0_, r1_);
    AABB b;
    b.expand(Vec3(-R, -R, z0_));
    b.expand(Vec3(R, R, z1_));
    return b;
}

std::string ConeShape::describe() const {
    char buf[128];
    std::snprintf(buf, sizeof buf, "cone(r=%g..%g, z=[%g,%g])", r0_, r1_, z0_, z1_);
    return buf;
}

// ---------------------------------------------------------------- TorusPatchShape

TorusPatchShape::TorusPatchShape(double R, double zc, double rho, double a0, double a1)
    : R_(R), zc_(zc), rho_(rho), a0_(std::min(a0, a1)), a1_(std::max(a0, a1)) {
    // The arc's centre may lie just across the axis (a large fillet); the arc itself must not.
    if (!(rho > 0) || a1_ - a0_ > Pi + 1e-12) throw std::runtime_error("torus patch: invalid arc");
    rMin_ = zMin_ = Inf;
    rMax_ = zMax_ = -Inf;
    auto add = [&](double a) {
        const double r = R_ + rho_ * std::cos(a), z = zc_ + rho_ * std::sin(a);
        rMin_ = std::min(rMin_, r);
        rMax_ = std::max(rMax_, r);
        zMin_ = std::min(zMin_, z);
        zMax_ = std::max(zMax_, z);
    };
    add(a0_);
    add(a1_);
    for (int k = -4; k <= 4; ++k) {  // extremes of cos and sin inside the range
        const double a = k * Pi / 2;
        if (a > a0_ && a < a1_) add(a);
    }
    if (rMin_ < -1e-12 * rho_) throw std::runtime_error("torus patch: the arc crosses the axis");
    rMin_ = std::max(0.0, rMin_);
}

bool TorusPatchShape::onArc(const Vec3& p) const { return torusOnArc(R_, zc_, a0_, a1_, p); }

bool TorusPatchShape::intersect(const Ray& ray, double tmin, double tmax, LocalHit& h) const {
    // Clip to the slab zMin ≤ z ≤ zMax and the cylinder ρ ≤ rMax.
    double ta = tmin, tb = tmax;
    const double pad = 1e-9 * (1 + rMax_ + std::abs(zMax_));
    if (!clipSlab(ray.o.z, ray.d.z, zMin_ - pad, zMax_ + pad, ta, tb) || !clipCylinder(ray.o, ray.d, rMax_ + pad, ta, tb))
        return false;
    // F(t) = (ρ(t) − R)² + (z(t) − zc)² − rho², sampled for sign changes and bisected.
    auto F = [&](double t) { return torusF(R_, zc_, rho_, ray.at(t)); };
    const int N = 64;
    double tPrev = ta, fPrev = F(ta);
    int sample = 1;
    double firstStep = ray.departedPrimitive == 0xFFFFFFFEu ? tb - ta :
        std::min((tb - ta) / N, std::max(std::ldexp((tb - ta) / N, -60),
            64 * std::numeric_limits<double>::epsilon() * (1 + std::abs(ta))));
    for (int guard = 0; sample <= N && guard < N + 64; ++guard) {
        const double uniform = ta + (tb - ta) * sample / N;
        const double tCur = std::min(uniform, ta + firstStep);
        // A departure's self-root and next grazing root can lie in the same
        // uniform interval. Resolve its beginning without moving the ray anchor.
        if (tCur == uniform) ++sample;
        else firstStep = std::min(tb - ta, firstStep * 2);
        const double fCur = F(tCur);
        if ((fPrev <= 0) != (fCur <= 0) || fCur == 0) {
            const double t = bisect(F, tPrev, tCur, fPrev, 100);
            const Vec3 p = ray.at(t);
            if (t > tmin && t < tmax && onArc(p)) {
                h.t = t;
                h.p = p;
                h.n = torusNormal(R_, zc_, p);
                h.prim = 0;
                return true;
            }
        }
        tPrev = tCur;
        fPrev = fCur;
    }
    return false;
}

AABB TorusPatchShape::bounds() const {
    AABB b;
    b.expand(Vec3(-rMax_, -rMax_, zMin_));
    b.expand(Vec3(rMax_, rMax_, zMax_));
    return b;
}

std::string TorusPatchShape::describe() const {
    char buf[160];
    std::snprintf(buf, sizeof buf, "torus-patch(R=%g, zc=%g, rho=%g, a=[%.1f,%.1f] deg)", R_, zc_, rho_, degrees(a0_),
                  degrees(a1_));
    return buf;
}

// ---------------------------------------------------------------- RoundWallShape

bool RoundWallShape::solid(const Vec3& p) const {
    if (p.z < 0 || p.z > H_) return false;
    for (const auto& o : open_)
        if (inWallOpening(R_, o.azimuth, o.width, o.sill, o.top, p)) return false;
    return true;
}

bool RoundWallShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    double t0, t1;
    return cylinderRoots(r.o, r.d, R_, t0, t1) &&
           firstRefinedHit(r, makeQuadric(kQuadCylinder, R_ * R_, 0, 0), t0, t1, tmin, tmax,
                           [&](const Vec3& p) { return solid(p); },
                           [](const Vec3& p) { return normalize(Vec3(p.x, p.y, 0)); }, h);
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

bool DomeShape::solid(const Vec3& p) const { return domeSolid(R_, az_, w_, top_, p); }

bool DomeShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    double t0, t1;
    return sphereRoots(r.o, r.d, R_, t0, t1) &&
           firstRefinedHit(r, makeQuadric(kQuadSphere, R_ * R_, 0, 0), t0, t1, tmin, tmax,
                           [&](const Vec3& p) { return solid(p); }, [](const Vec3& p) { return normalize(p); }, h);
}

AABB DomeShape::bounds() const {
    AABB b;
    b.expand(Vec3(-R_, -R_, 0));
    b.expand(Vec3(R_, R_, R_));
    return b;
}

std::string DomeShape::describe() const { return "dome(r=" + std::to_string(R_) + ", slit width " + std::to_string(w_) + ")"; }

}  // namespace owe
