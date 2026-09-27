// Viewer camera motion in double precision. This is a virtual eye, never a moved sensor or lens.
#pragma once
#include <algorithm>
#include "owe/scene/detector.hpp"

namespace owe {
// Scale with the nearby geometry, while retaining a useful 2 cm/s base speed at glass.
// The user's speed remains an upper limit, so the wheel can always slow it further.
inline double navigationSpeed(double speed, bool precision, bool fast, double surfaceDistance = Inf,
                              double precisionFactor = 0.2, bool slowNearSurfaces = true) {
    if (slowNearSurfaces && std::isfinite(surfaceDistance)) speed = std::min(speed, std::max(0.02, surfaceDistance * 0.5));
    return speed * (precision ? precisionFactor : fast ? 5 : 1);
}
inline double navigationSensitivity(bool precision, double factor = 0.2) { return precision ? factor : 1; }

struct NavigationCamera {
    Vec3 position, forward{0, 1, 0}, up{0, 0, 1};
    double distance = 1, fov = radians(50), pupil = 0, focus = Inf;

    Vec3 right() const { return normalize(cross(forward, up)); }
    Vec3 pivot() const { return position + forward * distance; }
    double surfaceDistance(const World& world) const {
        SurfaceHit hit;
        return world.intersect({position, forward}, 1000, hit) ? hit.t : Inf;
    }

    static NavigationCamera from(const Detector& d, const World* world = nullptr) {
        NavigationCamera c;
        if (auto* o = dynamic_cast<const IdealObserver*>(&d)) {
            c.position = o->position;
            c.forward = o->forward();
            c.up = o->upVec();
            // Astronomical look-at coordinates must not set kilometre-sized navigation steps.
            c.distance = std::clamp(length(o->lookAt - o->position), 0.001, 20.0);
            c.fov = o->fovY;
            c.pupil = o->pupilRadius;
            c.focus = o->focusDistance;
        } else if (auto* s = dynamic_cast<const SurfaceSensor*>(&d)) {
            c.position = offsetOrigin(s->toWorld().t, s->normal(), s->normal());
            c.forward = s->normal();
            c.up = s->toWorld().vector({0, 1, 0});
            if (s->hasAim) c.distance = std::max(0.001, length(s->aimCenter - c.position));
        }
        if (world) {
            SurfaceHit hit;
            if (world->intersect(Ray{c.position, c.forward}, 1000, hit))
                c.distance = std::clamp(hit.t, 0.001, 20.0);
        }
        return c;
    }

    void apply(IdealObserver& o, const World& w) const {
        o.position = position;
        o.lookAt = position + forward * distance;
        o.up = up;
        o.fovY = fov;
        o.pupilRadius = pupil;
        o.focusDistance = focus;
        o.prepare(w); // also locates the eye's medium after crossing a surface
    }
    void turn(double yaw, double pitch, bool orbit) {
        Vec3 target = pivot();
        Mat3 y = Mat3::rotation(up, yaw);
        forward = normalize(y * forward);
        Mat3 p = Mat3::rotation(right(), pitch);
        forward = normalize(p * forward);
        up = normalize(p * up);
        up = normalize(cross(right(), forward));
        if (orbit) position = target - forward * distance;
    }
    void pan(double dx, double dy) { position += right() * dx + up * dy; }
    void dolly(double steps) {
        double next = std::clamp(distance * std::exp(-steps * 0.15), 1e-5, 1e9);
        position += forward * (distance - next);
        distance = next;
    }
    void move(double sideways, double ahead, double vertical, double amount, bool walk) {
        Vec3 f = forward, r = right(), u{0, 0, 1};
        if (walk) {
            f.z = r.z = 0;
            if (lengthSq(f) > 1e-12) f = normalize(f);
            if (lengthSq(r) > 1e-12) r = normalize(r);
        }
        Vec3 v = r * sideways + f * ahead + u * vertical;
        if (lengthSq(v) > 0) position += normalize(v) * amount;
    }
    void axis(Vec3 direction, Vec3 vertical) {
        Vec3 target = pivot();
        forward = normalize(direction);
        up = vertical;
        position = target - forward * distance;
    }
};
} // namespace owe
