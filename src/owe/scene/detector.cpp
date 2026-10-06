#include "owe/scene/detector.hpp"

#include <sstream>
#include <stdexcept>

namespace owe {

double Image::meanY() const {
    double s = 0;
    size_t n = size_t(width) * height;
    for (size_t i = 0; i < n; ++i) s += xyz[3 * i + 1];
    return n ? s / n : 0;
}

// ---------------------------------------------------------------- IdealObserver

std::string IdealObserver::describe() const {
    std::ostringstream os;
    os << "ideal observer '" << name << "' " << width << "x" << height << ", fovY=" << degrees(fovY)
       << " deg, pupil diameter=" << 2 * pupilRadius * 1e3 << " mm, focus="
       << (std::isinf(focusDistance) ? std::string("infinity") : std::to_string(focusDistance) + " m");
    if (pixelSigma > 0) os << ", pixel filter gaussian(" << pixelSigma << ")";
    if (guide.radius > 0) os << ", pupil samples guided to the exit pupil of '" << guide.instrument << "'";
    return os.str();
}

bool IdealObserver::guideCenter(const Vec3& dir, Vec3& center) const {
    double dn = dot(dir, fwd_);
    if (!(dn > 0)) return false;
    center = guide.center - dir * (dot(guide.center - position, fwd_) / dn);
    return lengthSq(center - position) < sqr(pupilRadius + guide.radius);
}

void IdealObserver::prepare(const World& world) {
    fwd_ = normalize(lookAt - position);
    Vec3 u = up;
    if (std::abs(dot(normalize(u), fwd_)) > 0.9999) u = std::abs(fwd_.z) < 0.9 ? Vec3(0, 0, 1) : Vec3(0, 1, 0);
    right_ = normalize(cross(fwd_, u));
    up_ = cross(right_, fwd_);
    tanY_ = std::tan(0.5 * fovY);
    tanX_ = tanY_ * double(width) / double(height);
    region = world.locate(position);
    // Guide the pupil samples to a telescope's exit pupil the eye is looking into, when that pupil
    // is much smaller than the eye's (else the eye's own pupil is the stop, and uniform is best).
    guide = {};
    constexpr double kMargin = 1.25;  // pupil aberration and field: beams scatter around the paraxial disk
    double nearest = Inf;
    for (const World::ExitPupil& e : world.exitPupils()) {
        const Transform xf = world.assemblyToWorld(e.assembly);
        const Vec3 center = xf.point({0, 0, e.z}), axis = normalize(xf.vector({0, 0, 1}));
        const double radius = e.radius * kMargin, distance = length(position - center);
        if (!(pupilRadius > 0 && radius < 0.7 * pupilRadius)) continue;
        if (dot(fwd_, axis) > -0.7 || distance > 0.5 || distance >= nearest) continue;  // not looking into it
        nearest = distance;
        guide = {center, radius, 0.9, e.instrument};
    }
}

bool IdealObserver::generate(double px, double py, Rng& rng, Ray& ray, double& weight) const {
    double sx = (2 * px / width - 1) * tanX_;
    double sy = (1 - 2 * py / height) * tanY_;
    Vec3 dirPinhole = normalize(right_ * sx + up_ * sy + fwd_);
    Vec3 q = position;
    weight = 1;
    if (pupilRadius > 0) {
        Vec3 c;
        const bool guided = guide.radius > 0 && guideCenter(dirPinhole, c);
        const bool fromGuide = guided && rng.uniform() < guide.share;
        Draw2 u(rng);
        Vec2 d = sampleUniformDiskConcentric(u.u1, u.u2);
        q = fromGuide ? c + (right_ * d.x + up_ * d.y) * guide.radius : position + (right_ * d.x + up_ * d.y) * pupilRadius;
        if (guided) {
            // Radiance averaged over the eye's pupil: (1/A_eye) / p(q), zero outside the eye. The
            // disk a sample came from is known to contain it (no rounding at its rim).
            if (fromGuide && lengthSq(q - position) > sqr(pupilRadius)) { weight = 0; return false; }
            double pEye = 1 / (Pi * sqr(pupilRadius));
            bool inGuide = fromGuide || lengthSq(q - c) <= sqr(guide.radius);
            double pGuide = inGuide ? 1 / (Pi * sqr(guide.radius)) : 0;
            weight = pEye / ((1 - guide.share) * pEye + guide.share * pGuide);
        }
    }
    Vec3 dir = dirPinhole;
    if (pupilRadius > 0 && std::isfinite(focusDistance)) {
        Vec3 F = position + (right_ * sx + up_ * sy + fwd_) * focusDistance;
        dir = normalize(F - q);
    }
    ray = Ray{q, dir};
    return true;
}

bool IdealObserver::connect(const Vec3& x, Rng& rng, Vec3& q, int& px, int& py, double& factor) const {
    q = position;
    if (pupilRadius > 0) {
        Draw2 u(rng);
        Vec2 d = sampleUniformDiskConcentric(u.u1, u.u2);
        q = position + (right_ * d.x + up_ * d.y) * pupilRadius;
    }
    Vec3 w = x - q;
    double dist2 = lengthSq(w);
    if (dist2 <= 0) return false;
    Vec3 om = w / std::sqrt(dist2);
    double c = dot(om, fwd_);
    if (c <= 1e-9) return false;
    double sx, sy;
    if (pupilRadius > 0 && std::isfinite(focusDistance)) {
        Vec3 F = q + om * (focusDistance / c);
        sx = dot(F - position, right_) / focusDistance;
        sy = dot(F - position, up_) / focusDistance;
    } else {
        sx = dot(om, right_) / c;
        sy = dot(om, up_) / c;
    }
    Vec2 o = filterOffset(rng);  // a direction just outside the field still reaches the border pixels
    double fx = (sx / tanX_ + 1) * 0.5 * width + o.x;
    double fy = (1 - sy / tanY_) * 0.5 * height + o.y;
    if (fx < 0 || fy < 0 || fx >= width || fy >= height) return false;
    px = int(fx);
    py = int(fy);
    double pixelArea = (2 * tanX_ / width) * (2 * tanY_ / height);
    factor = 1.0 / (dist2 * pixelArea * c * c * c);
    return true;
}

// ---------------------------------------------------------------- SurfaceSensor

std::string SurfaceSensor::describe() const {
    std::ostringstream os;
    os << "surface sensor '" << name << "' " << width << "x" << height << ", " << 2e3 * halfX << " x " << 2e3 * halfY
       << " mm" << (hasAim ? ", aimed" : ", hemispherical");
    if (pixelSigma > 0) os << ", pixel filter gaussian(" << pixelSigma << ")";
    return os.str();
}

void SurfaceSensor::prepare(const World& world) {
    const Boundary& b = world.boundaries().at(boundaryIndex);
    const auto& optics = world.optics()[b.optics];
    if (optics.type != SurfaceType::Detector && !(optics.type == SurfaceType::Diffuse && optics.detector >= 0))
        throw std::runtime_error("sensor '" + name + "' boundary is not a detector surface");
    toWorld_ = b.toWorld;
    toLocal_ = b.toLocal;
    normal_ = normalize(toWorld_.vector(Vec3(0, 0, 1)));
    region = b.front;
}

bool SurfaceSensor::generate(double px, double py, Rng& rng, Ray& ray, double& weight) const {
    // A filtered sample displaced beyond the sensor measures nothing: there is no sensor there.
    if (px < 0 || py < 0 || px >= width || py >= height) { weight = 0; return false; }
    if (flipX) px = width - px;
    if (flipY) py = height - py;
    Vec3 pl{(2 * px / width - 1) * halfX, (1 - 2 * py / height) * halfY, 0};
    Vec3 p = toWorld_.point(pl);
    Vec3 dir;
    if (hasAim) {
        Vec3 a1, a2;
        orthonormalBasis(aimNormal, a1, a2);
        bool useAim = aimShare >= 1 || rng.uniform() < aimShare;
        bool useFocus = focusShare > 0 && rng.uniform() < focusShare;
        const Vec3& c = useFocus ? focusCenter : aimCenter;
        double r = useFocus ? focusRadius : aimRadius;
        Draw2 u(rng);
        Vec2 d = sampleUniformDiskConcentric(u.u1, u.u2);
        Vec3 w = c + (a1 * d.x + a2 * d.y) * r - p;
        dir = w / length(w);
        if (!useAim) dir = Frame(normal_).toWorld(sampleCosineHemisphere(u.u1, u.u2));
        double cosS = dot(dir, normal_);
        if (cosS <= 0) { weight = 0; return false; }
        // Solid-angle density of a direction under uniform sampling of a disk ⟂ aimNormal.
        auto diskPdf = [&](const Vec3& center, double radius, bool generatingDisk) {
            double dn = dot(dir, aimNormal);
            if (std::abs(dn) <= 1e-12) return 0.0;
            double t = dot(center - p, aimNormal) / dn;
            if (t <= 0 || (!generatingDisk && lengthSq(p + dir * t - center) > radius * radius)) return 0.0;
            return t * t / (std::abs(dn) * Pi * radius * radius);
        };
        double pdfW = (1 - focusShare) * diskPdf(aimCenter, aimRadius, useAim && !useFocus);
        if (focusShare > 0) pdfW += focusShare * diskPdf(focusCenter, focusRadius, useAim && useFocus);
        pdfW = aimShare * pdfW + (1 - aimShare) * cosS * InvPi;
        if (!(pdfW > 0)) { weight = 0; return false; }
        weight = cosS / pdfW;
    } else {
        Frame f(normal_);
        Draw2 u(rng);
        dir = f.toWorld(sampleCosineHemisphere(u.u1, u.u2));
        weight = Pi;
    }
    ray = Ray{offsetOrigin(p, normal_, dir), dir};
    return true;
}

bool SurfaceSensor::pixelOfHit(const SurfaceHit& hit, Rng& rng, int& px, int& py, double& pixelArea) const {
    Vec3 pl = hit.pLocal;
    double fx = (pl.x / halfX + 1) * 0.5 * width;
    double fy = (1 - pl.y / halfY) * 0.5 * height;
    if (flipX) fx = width - fx;
    if (flipY) fy = height - fy;
    Vec2 o = filterOffset(rng);
    fx += o.x;
    fy += o.y;
    if (fx < 0 || fy < 0 || fx >= width || fy >= height) return false;
    px = int(fx);
    py = int(fy);
    pixelArea = (2 * halfX / width) * (2 * halfY / height);
    return true;
}

}  // namespace owe
