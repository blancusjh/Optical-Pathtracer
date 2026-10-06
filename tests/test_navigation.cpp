#include <cstring>
#include "check.hpp"
#include "../apps/navigation.hpp"
#include "owe/backends/compare.hpp"
#include "owe/backends/registry.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/scene/builders.hpp"
#include "owe/scene/prescription.hpp"

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

// ---------------------------------------------------------------- free-view image formation
// The free eye sees through instruments by transport alone: no instrument is selected, no mode is
// enabled, and whatever image the optics form reaches the eye only because its rays enter the eye's
// pupil. These tests place ordinary observers and measure where the image appears.

namespace {

// Backend for the free-view checks: the GPU when it runs here (fast), else the reference.
std::string fastBackend() {
    return findBackend("gpu") && findBackend("gpu")->available() ? "gpu" : "cpu";
}

// Luminance-weighted centroid of the image (pixels), and the total.
struct Centroid {
    double x = 0, y = 0, total = 0;
};
Centroid centroid(const Image& img) {
    Centroid c;
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            double v = img.at(x, y).y;
            c.x += v * (x + 0.5);
            c.y += v * (y + 0.5);
            c.total += v;
        }
    if (c.total > 0) {
        c.x /= c.total;
        c.y /= c.total;
    }
    return c;
}

// Elevation (radians, + up) at which an observer sees image row y.
double elevation(const IdealObserver& eye, double y) {
    return std::atan((1 - 2 * y / eye.height) * eye.tanY());
}

Image renderEye(Scene& sc, int det, int spp) {
    RenderSettings rs;
    rs.backend = fastBackend();
    rs.integrator = "path";
    auto R = makeRenderer(sc, det, rs);
    R->runPass(spp);
    return R->resolve();
}

// A small lamp 30 mm above the axis of a free-standing f = 100 mm biconvex N-BK7 lens, and an
// ordinary eye on the axis somewhere else. mm throughout.
Scene lensScene(double lampY, double lampZ, double lampFacing, const Vec3& eyePos, double focus) {
    char text[1400];
    std::snprintf(text, sizeof text, R"(
        units = mm
        body Lamp { type = sheet  size = (6, 6)  position = (0, %g, %g)  axis = (0, %g, 0)  up = (0, 0, 1)
                    material = black  emission = blackbody(4500K, 2000) }
        body Lens { type = lens  medium = N-BK7  focal = 100  form = bi  diameter = 40  thickness = 6  rim = black
                    position = (0, 600, 0)  axis = (0, 1, 0) }
        observer Eye { position = (%g, %g, %g)  look_at = (0, 600, 0)  up = (0, 0, 1)  fov = 16deg  pupil = 3
                       focus = %g  resolution = (64, 64) }
    )", lampY, lampZ, lampFacing, eyePos.x, eyePos.y, eyePos.z, focus);
    return loadSceneFromString(text);
}

}  // namespace

TEST(free_eye_at_a_saved_observer_pose_is_that_observer) {
    // The viewer's free eye walked to exactly a saved observer's pose is that observer: the same
    // parameters, and statistically the same image (compared block by block over independent runs).
    Scene sc = loadScene("scenes/the_lens.owe");
    const int d = sc.findDetector("Eye");
    auto& saved = dynamic_cast<IdealObserver&>(*sc.detectors[size_t(d)]);
    saved.width = 48;
    saved.height = 30;
    saved.pixelSigma = 0.7;  // a scene's own pixel_filter, which the free eye must adopt
    saved.prepare(sc.world);
    auto eye = std::make_unique<IdealObserver>();
    adoptPixelResponse(*eye, saved);
    eye->width = saved.width;
    eye->height = saved.height;
    NavigationCamera::from(saved, &sc.world).apply(*eye, sc.world);
    CHECK_NEAR(length(eye->position - saved.position), 0, 1e-15);
    CHECK_NEAR(length(eye->forward() - saved.forward()), 0, 1e-12);
    CHECK_NEAR(length(eye->upVec() - saved.upVec()), 0, 1e-12);
    CHECK(eye->fovY == saved.fovY && eye->pupilRadius == saved.pupilRadius && eye->focusDistance == saved.focusDistance);
    CHECK(eye->pixelSigma == saved.pixelSigma && eye->region == saved.region);
    sc.detectors.push_back(std::move(eye));
    RenderSettings rs = sc.settingsFor(d);
    rs.backend = fastBackend();
    ComparisonReport rep = compareMeasurements(sc, d, rs, int(sc.detectors.size()) - 1, rs, 32, 6, 6);
    if (!rep.consistent()) std::fprintf(stderr, "%s", rep.text().c_str());
    CHECK(rep.consistent());
}

TEST(free_eye_sees_a_loose_lens_form_an_inverted_image_from_either_side) {
    // Lamp 600 mm before an f = 100 mm lens: a real, inverted image 0.2× as large forms 120 mm
    // behind it. An eye 400 mm behind the lens, focused on that aerial image, sees the lamp (30 mm
    // above the axis) below the axis. Swapping the sides (reciprocity) gives the same by symmetry.
    for (int side : {+1, -1}) {
        const double lensY = 600, lampY = lensY - side * 600, eyeY = lensY + side * 400;
        Scene sc = lensScene(lampY, 30, side, {0, eyeY, 0}, 280);
        auto& eye = dynamic_cast<IdealObserver&>(*sc.detectors[0]);
        Centroid c = centroid(renderEye(sc, 0, 128));
        CHECK(c.total > 0);
        // Paraxial prediction: image at 120 mm behind the lens, 6 mm below the axis, seen 280 mm away.
        double predicted = std::atan(-6.0 / 280.0), seen = elevation(eye, c.y);
        CHECK(seen < 0);  // inverted: the lamp is above the axis
        CHECK_NEAR(seen, predicted, 0.15 * std::abs(predicted));
        CHECK_NEAR(c.x, 32, 1.5);
    }
}

TEST(free_eye_sees_a_magnified_upright_virtual_image_close_to_a_lens) {
    // The same lens as a magnifier: a lamp 60 mm before it (inside the focal length) has an upright
    // virtual image 150 mm before the lens, 2.5× as large. An eye 60 mm behind the lens sees it higher
    // than the lamp itself would appear without the lens.
    const double lensY = 600;
    Scene sc = lensScene(lensY - 60, 8, +1, {0, lensY + 60, 0}, 210);
    auto& eye = dynamic_cast<IdealObserver&>(*sc.detectors[0]);
    Centroid c = centroid(renderEye(sc, 0, 128));
    CHECK(c.total > 0);
    double predicted = std::atan(20.0 / 210.0), naked = std::atan(8.0 / 120.0), seen = elevation(eye, c.y);
    CHECK(seen > naked * 1.2);  // magnified and upright
    CHECK_NEAR(seen, predicted, 0.15 * predicted);
}

TEST(free_eye_sees_the_real_image_a_concave_mirror_forms) {
    // A concave mirror (R = 400 mm, f = 200 mm) facing a lamp 600 mm away, 20 mm above its axis:
    // a real inverted image 0.5× as large forms 300 mm in front of the mirror, 10 mm below the axis.
    // An eye on the axis 900 mm from the mirror, focused on it, sees it below the axis.
    Scene sc = loadSceneFromString(R"(
        units = mm
        body Lamp   { type = sheet  size = (6, 6)  position = (0, 600, 20)  axis = (0, -1, 0)  up = (0, 0, 1)
                      material = black  emission = blackbody(4500K, 2000) }
        body Mirror { type = mirror  surface = sphere(R = 400)  diameter = 80  position = (0, 0, 0)  axis = (0, 1, 0) }
        observer Eye { position = (0, 900, 0)  look_at = (0, 0, 0)  up = (0, 0, 1)  fov = 8deg  pupil = 3  focus = 600
                       resolution = (64, 64) }
    )");
    auto& eye = dynamic_cast<IdealObserver&>(*sc.detectors[0]);
    Centroid c = centroid(renderEye(sc, 0, 128));
    CHECK(c.total > 0);
    double predicted = std::atan(-10.0 / 600.0), seen = elevation(eye, c.y);
    CHECK(seen < 0);
    CHECK_NEAR(seen, predicted, 0.15 * std::abs(predicted));
}

TEST(walking_across_a_telescope_exit_pupil_vignettes_as_the_pupils_overlap) {
    // An eye walking sideways across a refractor's exit pupil: the image dims as the fraction of the
    // eye's pupil that the emerging beam fills (the overlap of two discs), is lost once they no longer
    // overlap, and returns unchanged when the eye walks back. Nothing but the eye's position changes.
    Scene sc;
    Prescription p = loadPrescription("lenses/refractor_150mm.lens");
    solveAfocal(p, p.wavelength, catalogIndex());
    auto instrument = buildPrescription(sc.world, p, "glass", -1, Transform{}, 0.085, catalogIndex());
    SurfaceOptics black;
    black.type = SurfaceType::Absorber;
    Emission emission;
    emission.radiance = Spectrum::constant(1);
    buildSheet(sc.world, "object", 45.5, 45.5, sc.world.addOptics(black), sc.world.addEmission(emission), -1,
               Transform::translate({0, 0, -1e6}), true);
    auto observer = std::make_unique<IdealObserver>();
    observer->width = observer->height = 32;
    observer->fovY = radians(1.5);
    observer->up = {0, 1, 0};
    observer->pupilRadius = 0.0006;
    auto& eye = *observer;
    sc.detectors.push_back(std::move(observer));
    const Vec3 pupil{0, 0, instrument.paraxial.exitPupilZ};
    const double rx = instrument.paraxial.exitPupilRadius, re = eye.pupilRadius;
    CHECK(rx > 0.3 * re && rx < 3 * re);
    sc.build();
    RenderSettings rs;
    rs.backend = fastBackend();
    auto R = makeRenderer(sc, 0, rs);
    auto lens = [](double r1, double r2, double d) {  // area of intersection of two discs
        if (d >= r1 + r2) return 0.0;
        if (d <= std::abs(r1 - r2)) return Pi * sqr(std::min(r1, r2));
        double a = r1 * r1 * std::acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1));
        double b = r2 * r2 * std::acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2));
        return a + b - 0.5 * std::sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2));
    };
    auto measure = [&](double dx) {
        eye.position = pupil + Vec3(dx, 0, 0);
        eye.lookAt = eye.position + Vec3(0, 0, -1);
        eye.prepare(sc.world);
        CHECK(R->resetObserver());
        R->runPass(256);
        return R->resolve().meanY();
    };
    const double centred = measure(0);
    CHECK(centred > 0);
    double previous = 1;
    for (double f : {0.25, 0.5, 0.75, 1.0, 1.25}) {
        const double dx = f * (rx + re);
        const double seen = measure(dx) / centred, expected = lens(rx, re, dx) / lens(rx, re, 0);
        std::fprintf(stderr, "    eye %.2f mm off the exit pupil (radius %.3g mm; eye %.3g mm): %.4f of the centred image, "
                     "disc overlap %.4f\n", dx * 1e3, rx * 1e3, re * 1e3, seen, expected);
        CHECK(seen <= previous * 1.05);  // gradual dimming, never an increase beyond noise
        CHECK_NEAR(seen, expected, 0.12 + 0.1 * expected);
        previous = seen;
    }
    CHECK(measure(1.6 * (rx + re)) < 0.02 * centred);  // lost
    CHECK(measure(0) == centred);                       // walking back recovers it, bit for bit
}
