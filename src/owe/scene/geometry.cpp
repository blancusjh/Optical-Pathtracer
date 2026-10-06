#include "owe/scene/geometry.hpp"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "owe/core/parallel.hpp"

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

// ---------------------------------------------------------------- WaveSurface

namespace {
// Smoothstep taper of coordinate u toward the edge at ±half over `margin`, and its derivative.
void waveTaper(double u, double half, double margin, double& w, double& dw) {
    w = 1;
    dw = 0;
    if (!(margin > 0)) return;
    const double e = half - std::abs(u);
    if (e >= margin) return;
    if (e <= 0) {
        w = 0;
        return;
    }
    const double t = e / margin;
    w = t * t * (3 - 2 * t);
    dw = -(u < 0 ? -1.0 : 1.0) * 6 * t * (1 - t) / margin;
}
}  // namespace

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
    const double u = fx - i, v = fy - j;
    // Value (P) and slope (Q) bases at the cell's two ends, and their derivatives.
    const double P[2] = {2 * u * u * u - 3 * u * u + 1, -2 * u * u * u + 3 * u * u};
    const double Q[2] = {u * u * u - 2 * u * u + u, u * u * u - u * u};
    const double dP[2] = {6 * u * u - 6 * u, -6 * u * u + 6 * u};
    const double dQ[2] = {3 * u * u - 4 * u + 1, 3 * u * u - 2 * u};
    const double Pv[2] = {2 * v * v * v - 3 * v * v + 1, -2 * v * v * v + 3 * v * v};
    const double Qv[2] = {v * v * v - 2 * v * v + v, v * v * v - v * v};
    const double dPv[2] = {6 * v * v - 6 * v, -6 * v * v + 6 * v};
    const double dQv[2] = {3 * v * v - 4 * v + 1, 3 * v * v - 2 * v};
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
    const Vec3& o = ray.o;
    const Vec3& d = ray.d;
    // The surface lies in the box |x| ≤ hx, |y| ≤ hy, |z| ≤ amplitude bound.
    double ta = tmin, tb = tmax;
    const double lo[3] = {-hx_, -hy_, -amp_}, hi[3] = {hx_, hy_, amp_};
    for (int k = 0; k < 3; ++k) {
        const double ok = k == 0 ? o.x : k == 1 ? o.y : o.z, dk = k == 0 ? d.x : k == 1 ? d.y : d.z;
        if (std::abs(dk) < 1e-300) {
            if (ok < lo[k] || ok > hi[k]) return false;
            continue;
        }
        double s0 = (lo[k] - ok) / dk, s1 = (hi[k] - ok) / dk;
        if (s0 > s1) std::swap(s0, s1);
        ta = std::max(ta, s0);
        tb = std::min(tb, s1);
    }
    if (!(ta < tb)) return false;
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
        return true;
    };
    for (int i = 0; i < 4096 && t < tb; ++i) {
        double step = std::max(std::abs(ft) / L, minStep);
        if (std::abs(ft) < nearSurface) {
            // Close to the surface: Newton from here reaches the root being approached.
            double tn = t;
            for (int k = 0; k < 8; ++k) {
                const double fn = f(tn, dfdt);
                if (std::abs(fn) < tol || std::abs(dfdt) < 1e-12) break;
                tn -= fn / dfdt;
            }
            if (tn >= t - 1e3 * tol && std::abs(f(tn, dfdt)) < 1e3 * tol) return accept(tn);
            step = std::max(minStep, 2 * std::abs(ft) / L);  // a grazing approach: move past it
        }
        const double tn = std::min(t + step, tb);
        const double fn = f(tn, dfdt);
        if ((fn < 0) != (ft < 0)) {
            // A crossing inside a minimum step: bisect it.
            double a = t, b = tn, fa = ft;
            for (int k = 0; k < 80 && b - a > tol; ++k) {
                const double m = 0.5 * (a + b);
                double dm;
                const double fm = f(m, dm);
                if ((fm < 0) == (fa < 0)) { a = m; fa = fm; } else { b = m; }
            }
            return accept(0.5 * (a + b));
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
    auto coeffs = [&](const Vec3& o, double& A, double& B, double& C) {
        const double rho = r0_ + k_ * (o.z - z0_);
        A = r.d.x * r.d.x + r.d.y * r.d.y - k_ * k_ * r.d.z * r.d.z;
        B = 2 * (o.x * r.d.x + o.y * r.d.y - k_ * rho * r.d.z);
        C = o.x * o.x + o.y * o.y - rho * rho;
    };
    double A, B, C, t0, t1;
    coeffs(r.o, A, B, C);
    const int n = solveQuadratic(A, B, C, t0, t1);
    for (int i = 0; i < n; ++i) {
        double t = i == 0 ? t0 : t1;
        if (!mayLieIn(t, tmin, tmax)) continue;
        t = refineRoot(r, t, coeffs);
        if (t <= tmin || t >= tmax) continue;
        const Vec3 p = r.at(t);
        if (p.z < z0_ || p.z > z1_) continue;
        const double rho = r0_ + k_ * (p.z - z0_);
        if (rho < 0) continue;
        h.t = t;
        h.p = p;
        h.n = normalize(Vec3(p.x, p.y, -rho * k_));
        h.prim = 0;
        return true;
    }
    return false;
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

bool TorusPatchShape::onArc(const Vec3& p) const {
    const double a = std::atan2(p.z - zc_, std::hypot(p.x, p.y) - R_);
    const double eps = 1e-9;
    for (double w : {a, a + 2 * Pi, a - 2 * Pi})
        if (w >= a0_ - eps && w <= a1_ + eps) return true;
    return false;
}

bool TorusPatchShape::intersect(const Ray& ray, double tmin, double tmax, LocalHit& h) const {
    const Vec3& o = ray.o;
    const Vec3& d = ray.d;
    // Clip to the slab zMin ≤ z ≤ zMax and the cylinder ρ ≤ rMax.
    double ta = tmin, tb = tmax;
    const double pad = 1e-9 * (1 + rMax_ + std::abs(zMax_));
    if (std::abs(d.z) < 1e-300) {
        if (o.z < zMin_ - pad || o.z > zMax_ + pad) return false;
    } else {
        double s0 = (zMin_ - pad - o.z) / d.z, s1 = (zMax_ + pad - o.z) / d.z;
        if (s0 > s1) std::swap(s0, s1);
        ta = std::max(ta, s0);
        tb = std::min(tb, s1);
    }
    const double a = d.x * d.x + d.y * d.y;
    if (a > 1e-300) {
        double q0, q1;
        if (solveQuadratic(a, 2 * (o.x * d.x + o.y * d.y), o.x * o.x + o.y * o.y - sqr(rMax_ + pad), q0, q1) < 2) return false;
        ta = std::max(ta, q0);
        tb = std::min(tb, q1);
    } else if (o.x * o.x + o.y * o.y > sqr(rMax_ + pad)) {
        return false;
    }
    if (!(ta < tb)) return false;
    // F(t) = (ρ(t) − R)² + (z(t) − zc)² − rho², sampled for sign changes and bisected.
    auto F = [&](double t) {
        const Vec3 p = ray.at(t);
        return sqr(std::hypot(p.x, p.y) - R_) + sqr(p.z - zc_) - rho_ * rho_;
    };
    const int N = 64;
    double tPrev = ta, fPrev = F(ta);
    for (int i = 1; i <= N; ++i) {
        const double tCur = ta + (tb - ta) * i / N;
        const double fCur = F(tCur);
        if ((fPrev <= 0) != (fCur <= 0) || fCur == 0) {
            double lo = tPrev, hi = tCur, flo = fPrev;
            for (int it = 0; it < 100 && hi - lo > 1e-15 * (1 + std::abs(hi)); ++it) {
                const double mid = 0.5 * (lo + hi);
                const double fm = F(mid);
                if ((fm <= 0) == (flo <= 0)) { lo = mid; flo = fm; } else { hi = mid; }
            }
            const double t = 0.5 * (lo + hi);
            const Vec3 p = ray.at(t);
            if (t > tmin && t < tmax && onArc(p)) {
                const double rho = std::hypot(p.x, p.y);
                const Vec3 c = rho > 0 ? Vec3(p.x * R_ / rho, p.y * R_ / rho, zc_) : Vec3(0, 0, zc_);
                h.t = t;
                h.p = p;
                h.n = normalize(p - c);
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

MeshShape::MeshShape(std::vector<Vec3> positions, std::vector<std::array<uint32_t, 3>> triangles, std::string label,
                     std::vector<Corners> shadingNormals, std::vector<UvCorners> uvs)
    : pos_(std::move(positions)), tris_(std::move(triangles)), normals_(std::move(shadingNormals)), uvs_(std::move(uvs)),
      label_(std::move(label)) {
    if (!normals_.empty() && normals_.size() != tris_.size())
        throw std::runtime_error("mesh: shading normals must be given for every triangle");
    if (!uvs_.empty() && uvs_.size() != tris_.size())
        throw std::runtime_error("mesh: texture coordinates must be given for every triangle");
    std::vector<AABB> boxes(tris_.size());
    cdf_.resize(tris_.size());
    parallelFor(tris_.size(), 1 << 15, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
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
            cdf_[i] = 0.5 * length(cross(b - a, c - a));
        }
    });
    double acc = 0;  // the running sum in triangle order, as a sequential pass
    for (double& a : cdf_) a = acc += a;
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

std::string MeshShape::describe() const {
    return label_ + "(" + std::to_string(tris_.size()) + " triangles" + (normals_.empty() ? "" : ", smooth") + ")";
}

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
    if (!normalIndex.empty()) normalIndex.resize(triangles.size(), {kNoNormal, kNoNormal, kNoNormal});
}

void MeshData::addQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    uint32_t base = uint32_t(positions.size());
    positions.insert(positions.end(), {a, b, c, d});
    triangles.push_back({base, base + 1, base + 2});
    triangles.push_back({base, base + 2, base + 3});
    if (!normalIndex.empty()) normalIndex.resize(triangles.size(), {kNoNormal, kNoNormal, kNoNormal});
}

void MeshData::append(const MeshData& o, const Transform& xf) {
    uint32_t base = uint32_t(positions.size());
    if (!o.normalIndex.empty() || !normalIndex.empty()) {
        normalIndex.resize(triangles.size(), {kNoNormal, kNoNormal, kNoNormal});
        const uint32_t nb = uint32_t(normals.size());
        for (const Vec3& n : o.normals) normals.push_back(xf.vector(n));
        for (size_t i = 0; i < o.triangles.size(); ++i) {
            std::array<uint32_t, 3> c{kNoNormal, kNoNormal, kNoNormal};
            if (i < o.normalIndex.size())
                for (int k = 0; k < 3; ++k)
                    if (o.normalIndex[i][k] != kNoNormal) c[k] = o.normalIndex[i][k] + nb;
            normalIndex.push_back(c);
        }
    }
    for (const Vec3& p : o.positions) positions.push_back(xf.point(p));
    for (const auto& t : o.triangles) triangles.push_back({t[0] + base, t[1] + base, t[2] + base});
}

namespace {

Vec3 faceNormal(const MeshData& m, const std::array<uint32_t, 3>& t) {
    const Vec3 &a = m.positions[t[0]], &b = m.positions[t[1]], &c = m.positions[t[2]];
    Vec3 n = cross(b - a, c - a);
    double l = length(n);
    return l > 0 ? n / l : Vec3();
}

// A shading normal for a corner of a triangle with (unit) normal fn: n itself when it lies on the
// triangle's front side, else fn (a corner normal behind its own face would shade it from behind).
Vec3 frontCorner(const Vec3& n, const Vec3& fn) {
    double l = length(n);
    if (!(l > 0) || dot(n, fn) <= 1e-6 * l) return fn;
    return n / l;
}

}  // namespace

std::vector<MeshShape::Corners> smoothNormals(const MeshData& m, double creaseAngle) {
    const size_t nt = m.triangles.size();
    // Vertices at one position are one vertex (files split vertices at texture seams).
    std::vector<uint32_t> weld(m.positions.size());
    {
        std::vector<uint32_t> order(m.positions.size());
        for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        auto key = [&](uint32_t i) { const Vec3& p = m.positions[i]; return std::array<double, 3>{p.x, p.y, p.z}; };
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key(a) < key(b) || (key(a) == key(b) && a < b); });
        for (size_t k = 0; k < order.size(); ++k)
            weld[order[k]] = (k > 0 && key(order[k]) == key(order[k - 1])) ? weld[order[k - 1]] : order[k];
    }
    // Triangles around each welded vertex (compressed rows).
    std::vector<uint32_t> start(m.positions.size() + 1, 0), around(3 * nt);
    for (const auto& t : m.triangles)
        for (uint32_t v : t) ++start[weld[v] + 1];
    for (size_t i = 1; i < start.size(); ++i) start[i] += start[i - 1];
    {
        std::vector<uint32_t> fill(start.begin(), start.end() - 1);
        for (uint32_t i = 0; i < nt; ++i)
            for (uint32_t v : m.triangles[i]) around[fill[weld[v]]++] = i;
    }
    std::vector<Vec3> fn(nt);
    std::vector<std::array<double, 3>> angle(nt);
    parallelFor(nt, 1 << 14, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const auto& t = m.triangles[i];
            fn[i] = faceNormal(m, t);
            for (int k = 0; k < 3; ++k) {
                Vec3 a = m.positions[t[(k + 1) % 3]] - m.positions[t[k]], b = m.positions[t[(k + 2) % 3]] - m.positions[t[k]];
                double la = length(a), lb = length(b);
                angle[i][k] = la > 0 && lb > 0 ? std::acos(clampd(dot(a, b) / (la * lb), -1, 1)) : 0;
            }
        }
    });
    const double cosCrease = std::cos(creaseAngle);
    std::vector<MeshShape::Corners> out(nt);
    parallelFor(nt, 1 << 14, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const auto& t = m.triangles[i];
            for (int k = 0; k < 3; ++k) {
                const uint32_t v = weld[t[k]];
                Vec3 sum;
                for (uint32_t s = start[v]; s < start[v + 1]; ++s) {
                    const uint32_t j = around[s];
                    if (dot(fn[j], fn[i]) < cosCrease) continue;
                    int corner = 0;  // j's corner at this vertex
                    while (corner < 2 && weld[m.triangles[j][corner]] != v) ++corner;
                    sum += fn[j] * angle[j][corner];
                }
                out[i][k] = frontCorner(sum, fn[i]);
            }
        }
    });
    return out;
}

std::vector<MeshShape::Corners> fileNormals(const MeshData& m) {
    if (m.normals.empty() || m.normalIndex.size() != m.triangles.size())
        throw std::runtime_error("the mesh has no vertex normals (OBJ vn)");
    std::vector<MeshShape::Corners> out(m.triangles.size());
    for (size_t i = 0; i < m.triangles.size(); ++i) {
        const Vec3 fn = faceNormal(m, m.triangles[i]);
        for (int k = 0; k < 3; ++k) {
            const uint32_t n = m.normalIndex[i][k];
            out[i][k] = frontCorner(n < m.normals.size() ? m.normals[n] : Vec3(), fn);
        }
    }
    return out;
}

namespace {

// One chunk of an OBJ file, parsed independently: its positions, and its faces with absolute
// indices, except relative (negative) ones, which are recorded against the chunk's own vertices
// until the chunks' vertex offsets are known.
struct ObjChunk {
    std::vector<Vec3> positions;
    std::vector<std::array<uint32_t, 3>> triangles;
    std::vector<std::pair<size_t, long>> relative;  // (triangle * 3 + corner, chunk-local vertex)
    std::vector<Vec3> normals;
    std::vector<std::array<uint32_t, 3>> normalIndex;     // per triangle (kNoNormal: none)
    std::vector<std::pair<size_t, long>> relativeNormal;  // as `relative`, for normals
};

const char* skipBlanks(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
    return p;
}

// A number as strtod would read it (leading blanks, optional '+'); 0 without advancing on failure.
double parseNumber(const char*& p, const char* end) {
    const char* q = skipBlanks(p, end);
    if (q < end && *q == '+') ++q;
    double v = 0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    auto r = std::from_chars(q, end, v);
    if (r.ec == std::errc() || r.ec == std::errc::result_out_of_range) p = r.ptr;
#else
    // Without floating-point from_chars (Apple's libc++): strtod on a terminated copy of the token.
    char buf[64];
    size_t n = 0;
    while (q + n < end && n < sizeof buf - 1 && q[n] != ' ' && q[n] != '\t' && q[n] != '\n' && q[n] != '\r') {
        buf[n] = q[n];
        ++n;
    }
    buf[n] = 0;
    char* e = buf;
    v = std::strtod(buf, &e);
    if (e != buf) p = q + (e - buf);
    else v = 0;
#endif
    return v;
}

void parseObjChunk(const char* p, const char* end, double scale, ObjChunk& out) {
    std::vector<long> idx, nidx;
    auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (p < end) {
        p = skipBlanks(p, end);
        if (p + 1 < end && p[0] == 'v' && (p[1] == ' ' || p[1] == '\t')) {
            const char* q = p + 2;
            Vec3 v;
            v.x = parseNumber(q, end);
            v.y = parseNumber(q, end);
            v.z = parseNumber(q, end);
            out.positions.push_back(v * scale);
            p = q;
        } else if (p + 2 < end && p[0] == 'v' && p[1] == 'n' && (p[2] == ' ' || p[2] == '\t')) {
            const char* q = p + 3;
            Vec3 n;
            n.x = parseNumber(q, end);
            n.y = parseNumber(q, end);
            n.z = parseNumber(q, end);
            out.normals.push_back(n);  // a direction: the scale does not apply
            p = q;
        } else if (p + 1 < end && p[0] == 'f' && (p[1] == ' ' || p[1] == '\t')) {
            idx.clear();
            nidx.clear();
            const char* q = p + 2;
            while (true) {
                q = skipBlanks(q, end);
                if (q >= end || *q == '\n' || *q == '\r') break;
                long v = 0;
                auto r = std::from_chars(q + (*q == '+'), end, v);
                if (r.ec != std::errc()) break;
                q = r.ptr;
                long vn = 0;  // v, v/vt, v//vn or v/vt/vn; 0: no normal
                if (q < end && *q == '/') {
                    ++q;
                    while (q < end && *q != '/' && !blank(*q)) ++q;  // vt
                    if (q < end && *q == '/') {
                        ++q;
                        auto rn = std::from_chars(q + (q < end && *q == '+'), end, vn);
                        if (rn.ec == std::errc()) q = rn.ptr;
                        else vn = 0;
                    }
                }
                idx.push_back(v);
                nidx.push_back(vn);
                while (q < end && !blank(*q)) ++q;
            }
            for (size_t i = 1; i + 1 < idx.size(); ++i) {
                const size_t c3[3] = {0, i, i + 1};
                std::array<uint32_t, 3> t{}, tn{};
                for (int k = 0; k < 3; ++k) {
                    const long corner = idx[c3[k]], normal = nidx[c3[k]];
                    if (corner < 0) {
                        out.relative.push_back({out.triangles.size() * 3 + size_t(k),
                                                long(out.positions.size()) + corner});
                    } else {
                        t[size_t(k)] = uint32_t(corner - 1);
                    }
                    tn[size_t(k)] = MeshData::kNoNormal;
                    if (normal < 0) {
                        out.relativeNormal.push_back({out.triangles.size() * 3 + size_t(k), long(out.normals.size()) + normal});
                    } else if (normal > 0) {
                        tn[size_t(k)] = uint32_t(normal - 1);
                    }
                }
                out.triangles.push_back(t);
                out.normalIndex.push_back(tn);
            }
            p = q;
        }
        while (p < end && *p != '\n') ++p;
        ++p;
    }
}

}  // namespace

// Positions, faces and vertex normals (texture coordinates are skipped). Large files are parsed
// in parallel chunks split at line breaks and joined in file order, so the mesh is exactly the one
// a sequential reading gives.
MeshData loadObj(const std::string& path, double scale) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open OBJ file: " + path);
    std::string buf(size_t(in.tellg()), '\0');
    in.seekg(0);
    in.read(buf.data(), std::streamsize(buf.size()));
    if (!in) throw std::runtime_error("cannot read OBJ file: " + path);
    buf.push_back('\n');
    const char* begin = buf.data();
    const char* end = begin + buf.size();
    const size_t parts = std::clamp<size_t>(buf.size() >> 22, 1, hardwareThreads());  // ≥ 4 MB each
    std::vector<const char*> cut(parts + 1, end);
    cut[0] = begin;
    for (size_t k = 1; k < parts; ++k) {
        const char* c = std::max(cut[k - 1], begin + buf.size() * k / parts);
        while (c < end && c[-1] != '\n') ++c;
        cut[k] = c;
    }
    std::vector<ObjChunk> chunks(parts);
    parallelParts(parts, parts, [&](size_t k, size_t, size_t) { parseObjChunk(cut[k], cut[k + 1], scale, chunks[k]); });

    MeshData m;
    size_t nv = 0, nt = 0, nn = 0;
    for (const ObjChunk& c : chunks) { nv += c.positions.size(); nt += c.triangles.size(); nn += c.normals.size(); }
    m.positions.reserve(nv);
    m.triangles.reserve(nt);
    m.normals.reserve(nn);
    if (nn > 0) m.normalIndex.reserve(nt);
    for (const ObjChunk& c : chunks) {
        const long vertexBase = long(m.positions.size()), normalBase = long(m.normals.size());
        const size_t triBase = m.triangles.size();
        m.positions.insert(m.positions.end(), c.positions.begin(), c.positions.end());
        m.triangles.insert(m.triangles.end(), c.triangles.begin(), c.triangles.end());
        for (auto [slot, local] : c.relative)
            m.triangles[triBase + slot / 3][slot % 3] = uint32_t(vertexBase + local);
        if (nn == 0) continue;
        m.normals.insert(m.normals.end(), c.normals.begin(), c.normals.end());
        m.normalIndex.insert(m.normalIndex.end(), c.normalIndex.begin(), c.normalIndex.end());
        for (auto [slot, local] : c.relativeNormal)
            m.normalIndex[triBase + slot / 3][slot % 3] = uint32_t(normalBase + local);
    }
    if (m.triangles.empty()) throw std::runtime_error("OBJ file has no faces: " + path);
    for (auto& t : m.normalIndex)
        for (uint32_t& n : t)
            if (n != MeshData::kNoNormal && n >= m.normals.size()) throw std::runtime_error("OBJ normal index out of range: " + path);
    return m;
}

}  // namespace owe
