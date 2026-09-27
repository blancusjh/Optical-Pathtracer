// Analytic reference scenes shared by the transport and backend-conformance tests.
#pragma once
#include <cmath>
#include <memory>

#include "owe/scene/scene.hpp"

// Irradiance at distance r from the axis, on a plane at height h below a Lambertian disk of
// radius a and radiance L (exact off-axis formula).
inline double diskIrradiance(double L, double a, double h, double r) {
    using owe::Pi;
    double s = h * h + r * r + a * a;
    return Pi * L / 2 * (1 - (h * h + r * r - a * a) / std::sqrt(s * s - 4 * r * r * a * a));
}
// Mean of diskIrradiance over a centred square of half-size `half`.
inline double meanDiskIrradiance(double L, double a, double h, double half) {
    int N = 200;
    double sum = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double x = -half + (i + 0.5) * 2 * half / N, y = -half + (j + 0.5) * 2 * half / N;
            sum += diskIrradiance(L, a, h, std::sqrt(x * x + y * y));
        }
    return sum / (N * N);
}
// Disk emitter (radiance 2) of radius a at height h above an upward-facing square sensor
// (half-size `half`), optionally aimed at the disk, optionally a diffuse screen (ρ = 0.7).
inline owe::Scene diskIrradianceScene(double a, double h, bool aimed, double half, bool screen = false) {
    using namespace owe;
    Scene sc;
    SurfaceOptics black;
    black.type = SurfaceType::Absorber;
    Emission e;
    e.radiance = Spectrum::constant(2.0);
    int body = sc.world.addBody("lamp", "sheet", -1, Transform::translate({0, 0, h}) * Transform::rotate({1, 0, 0}, Pi));
    sc.world.addBoundary(body, "face", PlaneShape::disk(a), Transform{}, kOutside, kOutside, sc.world.addOptics(black),
                         sc.world.addEmission(e));
    int sb = sc.world.addBody("sensor", "sensor", -1, Transform{});
    SurfaceOptics so;
    so.type = screen ? SurfaceType::Diffuse : SurfaceType::Detector;
    so.reflectance = Spectrum::constant(0.7);
    so.detector = 0;
    uint32_t bi = sc.world.addBoundary(sb, "pixels", PlaneShape::rect(half, half), Transform{}, kOutside, kOutside,
                                       sc.world.addOptics(so));
    auto d = std::make_unique<SurfaceSensor>();
    d->name = "E";
    d->width = d->height = 2;
    d->boundaryIndex = bi;
    d->halfX = d->halfY = half;
    if (aimed) {
        d->hasAim = true;
        d->aimCenter = {0, 0, h};
        d->aimNormal = {0, 0, 1};
        d->aimRadius = a;
        d->aimShare = screen ? 0.95 : 1;
    }
    sc.detectors.push_back(std::move(d));
    sc.build();
    return sc;
}
