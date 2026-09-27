#include <cstring>
#include "check.hpp"
#include "../apps/navigation.hpp"
#include "owe/loader/scene_loader.hpp"

using namespace owe;

TEST(navigation_precision_supports_millimetre_eye_alignment) {
    // Half-second taps: 10 cm across a room, 2 mm next to an eyepiece.
    CHECK_NEAR(0.5 * navigationSpeed(2, true, false, 2), 0.1, 1e-12);
    CHECK_NEAR(0.5 * navigationSpeed(2, true, false, 0.003), 0.002, 1e-12);
    // Wheel/slider choices and the precision factor remain effective at the glass.
    CHECK_NEAR(navigationSpeed(0.001, true, false, 0.003), 0.0002, 1e-12);
    CHECK_NEAR(navigationSpeed(2, true, false, 0.003, 0.5), 0.01, 1e-12);
    CHECK_NEAR(navigationSpeed(2, true, false, 0.003, 0.2, false), 0.4, 1e-12);
    CHECK_NEAR(navigationSpeed(2, true, true, 0.003), navigationSpeed(2, true, false, 0.003), 0);
    CHECK_NEAR(navigationSensitivity(true), 0.2, 0);
}

TEST(navigation_orbit_pan_dolly_and_walk) {
    NavigationCamera c;
    c.position = {0, -3, 1};
    c.distance = 3;
    Vec3 target = c.pivot();
    c.turn(0.7, 0.3, true);
    CHECK_NEAR(length(c.pivot() - target), 0, 1e-12);
    CHECK_NEAR(length(c.position - target), 3, 1e-12);
    c.pan(0.1, 0.2);
    CHECK_NEAR(length(c.pivot() - target), std::sqrt(0.05), 1e-12);
    target = c.pivot();
    c.dolly(2);
    CHECK_NEAR(length(c.pivot() - target), 0, 1e-12);
    CHECK(c.distance < 3);
    Vec3 p = c.position;
    c.turn(-0.4, -0.6, false);
    CHECK_NEAR(length(c.position - p), 0, 1e-12);
    c.move(1, 1, 0, 0.2, true);
    CHECK_NEAR(c.position.z, p.z, 1e-12);
    CHECK_NEAR(length(c.position - p), 0.2, 1e-12);
    for (int i = 0; i < 10000; ++i) c.turn(0.04, 0.03, false);
    CHECK_NEAR(length(c.forward), 1, 1e-12);
    CHECK_NEAR(dot(c.forward, c.up), 0, 1e-12);
}

TEST(navigation_preserves_optical_presets_and_finds_eye_medium) {
    Scene sc = loadScene("scenes/the_telescope.owe");
    int d = sc.findDetector("Eyepiece");
    const auto& base = dynamic_cast<const IdealObserver&>(*sc.detectors[size_t(d)]);
    NavigationCamera c = NavigationCamera::from(base);
    IdealObserver eye = base;
    c.apply(eye, sc.world);
    CHECK_NEAR(length(eye.forward() - base.forward()), 0, 1e-12);
    CHECK_NEAR(eye.pupilRadius, base.pupilRadius, 0);
    c.move(0, -1, 0, 0.02, false);
    c.apply(eye, sc.world);
    CHECK_NEAR(length(base.position - eye.position), 0.02, 1e-12);
    CHECK(eye.region == sc.world.locate(eye.position));

    Scene glass = loadSceneFromString(R"(
        body Ball { type = sphere radius = 1 medium = N-BK7 }
        observer Eye { position = (0, -2, 0) look_at = (0, 0, 0) }
    )");
    auto& e = dynamic_cast<IdealObserver&>(*glass.detectors[0]);
    auto outside = e.region;
    c = NavigationCamera::from(e);
    c.move(0, 1, 0, 2, false);
    c.apply(e, glass.world);
    CHECK(e.region != outside);
    CHECK(e.region == glass.world.locate({0, 0, 0}));
}
