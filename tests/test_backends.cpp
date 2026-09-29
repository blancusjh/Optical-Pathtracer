// Backend conformance. Every check here runs on every backend in the build that can run on this
// machine — the reference ("cpu") always, "gpu" where a Vulkan device exists — through the same
// registry the applications use. A backend conforms when it reproduces analytic results and agrees
// statistically with the reference; a backend added to the registry is covered with no new tests.
#include <functional>

#include "analytic.hpp"
#include "check.hpp"
#include "owe/backends/compare.hpp"
#include "owe/backends/registry.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/scene/builders.hpp"
#include "owe/scene/prescription.hpp"
#include "owe/transport/optics.hpp"

using namespace owe;

namespace {

// f(name) for each available backend (other than the reference when candidatesOnly); the
// unavailable ones are reported as skipped.
void runOnBackends(const std::function<void(const std::string&)>& f, bool candidatesOnly) {
    for (const Backend* b : backends()) {
        if (candidatesOnly && b == &referenceBackend()) continue;
        std::string why;
        if (!b->available(&why)) {
            std::fprintf(stderr, "    [%s] skipped: %s\n", b->name().c_str(), why.c_str());
            continue;
        }
        std::fprintf(stderr, "    [%s]\n", b->name().c_str());
        f(b->name());
    }
}
void forEachBackend(const std::function<void(const std::string&)>& f) { runOnBackends(f, false); }
void forEachCandidate(const std::function<void(const std::string&)>& f) { runOnBackends(f, true); }

Image render(const Scene& sc, int det, const std::string& backend, int spp, uint64_t seed = 1,
             const std::string& integrator = "path") {
    RenderSettings rs;
    rs.backend = backend;
    rs.integrator = integrator;
    rs.spp = spp;
    rs.seed = seed;
    auto R = makeRenderer(sc, det, rs);
    R->runPass(spp);
    CHECK(R->stats().leaks == 0);
    return R->resolve();
}

// The candidate's image agrees with the reference's over independent seeds (block z-scores). The
// comparison must also be sharp enough to mean something: maxRelSe bounds the reference's
// relative standard error.
void checkAgainstReference(const Scene& sc, int det, const std::string& candidate, int spp, int runs,
                           double maxRelSe = 0.02, const std::string& integrator = "path") {
    RenderSettings reference;
    reference.backend = referenceBackend().name();
    reference.integrator = integrator;
    RenderSettings other = reference;
    other.backend = candidate;
    ComparisonReport rep = compareRenderers(sc, det, reference, other, spp, runs, 6);
    if (!rep.consistent() || !(rep.seA < maxRelSe * rep.meanA)) std::fprintf(stderr, "%s", rep.text().c_str());
    CHECK(rep.consistent());
    CHECK(rep.seA < maxRelSe * rep.meanA);
}

void checkReflectedMoonlight(const std::string& backend) {
    using namespace owe;
    Scene sc;
    sc.world.env.hasSun = true; sc.world.env.sunDir = {0, 0, -1};
    sc.world.env.sunRadiance = Spectrum::constant(1e5);
    SurfaceOptics white; white.type = SurfaceType::Diffuse; white.reflectance = Spectrum::constant(1);
    SurfaceOptics moon = white; moon.reflectance = Spectrum::constant(0.3);
    auto floor = sc.world.addOptics(white), lunar = sc.world.addOptics(moon);
    buildSheet(sc.world, "floor", 4, 4, floor, -1, -1, Transform{}, true);
    buildSphere(sc.world, "moon", 1e6, BodyMaterial{.optics = lunar}, -1, Transform::translate({0, 0, 1e9}));
    sc.indirectGuide = "moon";
    auto eye = std::make_unique<IdealObserver>();
    eye->position = {0, -1, 1}; eye->lookAt = {0, 0, 0}; eye->up = {0, 0, 1};
    eye->fovY = radians(0.01); eye->width = eye->height = 16;
    sc.detectors.push_back(std::move(eye)); sc.build();
    RenderSettings rs; rs.backend = backend;
    auto r = makeRenderer(sc, 0, rs); r->runPass(2048);
    // Full-phase Lambertian sphere: geometric albedo 2*rho/3. Finite-distance
    // and finite solar-disc corrections are below 0.5% for these dimensions.
    double solarE = 1e5 * Pi * sqr(std::sin(sc.world.env.sunAngularRadius));
    double expected = (2. / 3) * 0.3 * solarE * sqr(1e6 / 1e9) * InvPi;
    double measured = r->resolve().meanY();
    std::fprintf(stderr, "    %s reflected moonlight: %.9g, analytic %.9g\n", backend.c_str(), measured, expected);
    CHECK_REL(measured, expected, 0.015);
    CHECK(r->stats().inconsistencies == 0);
    CHECK(r->stats().leaks == 0);
    sc.indirectGuide.clear(); sc.build();
    CHECK(sc.world.optics()[floor].sampleAimShare == 0);
}

void checkDistantAndNearbyLights(const std::string& backend) {
    using namespace owe;
    Scene sc;
    SurfaceOptics white; white.type = SurfaceType::Diffuse; white.reflectance = Spectrum::constant(1);
    SurfaceOptics black; black.type = SurfaceType::Absorber;
    auto surface = sc.world.addOptics(white), absorber = sc.world.addOptics(black);
    buildSheet(sc.world, "screen", 4, 4, surface, -1, -1, Transform{}, true);
    Emission star; star.radiance = Spectrum::constant(1); star.distant = true;
    Emission lamp; lamp.radiance = Spectrum::constant(0.01);
    buildSphere(sc.world, "star", 5e9, BodyMaterial{.optics = absorber, .emission = sc.world.addEmission(star)}, -1,
                Transform::translate({0, 0, 1e13}));
    buildSphere(sc.world, "lamp", 0.01, BodyMaterial{.optics = absorber, .emission = sc.world.addEmission(lamp)}, -1,
                Transform::translate({1, 0, 2}));
    auto eye = std::make_unique<IdealObserver>();
    eye->position = {0, -1, 1}; eye->lookAt = {0, 0, 0}; eye->up = {0, 0, 1};
    eye->fovY = radians(0.01); eye->width = eye->height = 16;
    sc.detectors.push_back(std::move(eye)); sc.build();
    CHECK(sc.world.lights().size() == 2);
    CHECK_NEAR(sc.world.lightSelectPdf(0), 0.5, 1e-12);
    CHECK_NEAR(sc.world.lightSelectPdf(1), 0.5, 1e-12);
    RenderSettings rs; rs.backend = backend;
    auto r = makeRenderer(sc, 0, rs); r->runPass(512);
    double expected = sqr(5e9 / 1e13) + 0.01 * sqr(0.01) / 5 * (2 / std::sqrt(5.));
    double measured = r->resolve().meanY();
    std::fprintf(stderr, "    %s starlight + lamp: %.9g, analytic %.9g\n", backend.c_str(), measured, expected);
    CHECK_REL(measured, expected, 0.01);
    CHECK(r->stats().inconsistencies == 0);
    CHECK(r->stats().leaks == 0);
}

void checkUncoatedGlassReflectance(const std::string& backend) {
    using namespace owe;
    Scene sc;
    sc.world.env.skyModel = Environment::Sky::Uniform;
    sc.world.env.zenith = Spectrum::constant(1);
    Medium glass; glass.name = "test-glass";
    glass.index = IndexModel::constant(1.5).relativeToAir();
    glass.absorption = Spectrum::constant(1e6); // no return from transmitted light
    auto medium = sc.world.addMedium(glass);
    int body = sc.world.addBody("glass half-space", "interface", -1, Transform{});
    auto inside = sc.world.addRegion("glass", medium, body);
    sc.world.addBoundary(body, "front", PlaneShape::rect(1000, 1000), Transform{}, kOutside,
                         inside, sc.world.dielectricOptics());
    auto observer = std::make_unique<IdealObserver>();
    observer->name = "reflectance meter"; observer->lookAt = {0, 0, 0}; observer->up = {0, 1, 0};
    observer->width = observer->height = 16; observer->fovY = radians(0.01);
    auto& eye = *observer; sc.detectors.push_back(std::move(observer));
    eye.position = {0, 0, 1}; sc.build();
    RenderSettings rs; rs.backend = backend;
    auto renderer = makeRenderer(sc, 0, rs);
    for (double angle : {0., 60., 80.}) {
        eye.position = {std::sin(radians(angle)), 0, std::cos(radians(angle))};
        eye.prepare(sc.world); CHECK(renderer->resetObserver()); renderer->runPass(4096);
        double expected = fresnelDielectric(std::cos(radians(angle)), 1.5);
        double measured = renderer->resolve().meanY();
        std::fprintf(stderr, "    %s glass at %.0f deg: R=%.5f, Fresnel=%.5f\n", backend.c_str(), angle, measured, expected);
        CHECK_NEAR(measured, expected, 0.0025);
        CHECK(renderer->stats().inconsistencies == 0);
        CHECK(renderer->stats().leaks == 0);
    }
}

void checkPinholeIrradiance(const std::string& backend) {
    using namespace owe;
    double previous = 0;
    for (double diameter : {0.006, 0.012, 0.3, -0.006}) {
        bool wideScreen = diameter < 0;
        diameter = std::abs(diameter);
        double halfSize = wideScreen ? 1.5 : 1e-6;
        Scene sc;
        sc.world.env.skyModel = Environment::Sky::Uniform;
        sc.world.env.zenith = Spectrum::constant(1);
        double radius = diameter / 2, distance = 4;
        // Suppress light around the outer edge even for the cosine-mixture samples.
        buildStop(sc.world, "opaque wall", radius, 1e6, -1, Transform::translate({0, 0, distance}));
        int body = sc.world.addBody("meter", "sensor", -1, Transform{});
        SurfaceOptics detector; detector.type = SurfaceType::Detector; detector.detector = 0;
        if (wideScreen) {
            detector.type = SurfaceType::Diffuse;
            detector.reflectance = Spectrum::constant(1);
            detector.sampleAimCenter = {0, 0, distance};
            detector.sampleAimNormal = {0, 0, 1};
            detector.sampleAimRadius = radius;
            detector.sampleAimShare = 0.95;
        }
        auto boundary = sc.world.addBoundary(body, "pixels", PlaneShape::rect(halfSize, halfSize), Transform{},
                                             kOutside, kOutside, sc.world.addOptics(detector));
        auto meter = std::make_unique<SurfaceSensor>();
        meter->name = "irradiance"; meter->width = meter->height = 32;
        meter->boundaryIndex = boundary; meter->halfX = meter->halfY = halfSize;
        meter->hasAim = true; meter->aimCenter = {0, 0, distance};
        meter->aimNormal = {0, 0, 1}; meter->aimRadius = radius;
        if (wideScreen) meter->aimShare = 0.95;
        sc.detectors.push_back(std::move(meter)); sc.build();
        RenderSettings rs; rs.backend = backend;
        auto renderer = makeRenderer(sc, 0, rs); renderer->runPass(256);
        double expected = Pi * radius * radius / (distance * distance + radius * radius);
        if (wideScreen) {
            // Tiny aperture approximation integrated over the 3 m screen. The finite
            // aperture correction is < 1e-6 here. Off-axis pixels exercise cancellation
            // in reconstructing disk membership, hidden by a point-sized sensor.
            expected = 0;
            for (int y = 0; y < 256; ++y) for (int x = 0; x < 256; ++x) {
                double px = (2 * (x + 0.5) / 256 - 1) * halfSize;
                double py = (2 * (y + 0.5) / 256 - 1) * halfSize;
                expected += Pi * radius * radius * distance * distance /
                            sqr(distance * distance + px * px + py * py) / (256 * 256);
            }
        }
        double measured = renderer->resolve().meanY();
        std::fprintf(stderr, "    %s pinhole %.0f mm%s: E=%.8g, analytic=%.8g\n",
                     backend.c_str(), diameter * 1000, wideScreen ? " / 3 m screen" : "", measured, expected);
        CHECK_REL(measured, expected, 0.005);
        if (diameter == 0.012) CHECK_REL(measured / previous, 4, 0.005);
        previous = measured;
        CHECK(renderer->stats().inconsistencies == 0);
        CHECK(renderer->stats().leaks == 0);
        if (wideScreen) {
            // The same numerical trap occurs when a freely placed eye sees the screen.
            auto observer = std::make_unique<IdealObserver>();
            observer->position = {0, 0, 2}; observer->lookAt = {0, 0, 0};
            observer->up = {0, 1, 0}; observer->fovY = radians(30);
            observer->width = observer->height = 16;
            observer->prepare(sc.world); sc.detectors.push_back(std::move(observer));
            auto eyeRenderer = makeRenderer(sc, 1, rs); eyeRenderer->runPass(1024);
            auto image = eyeRenderer->resolve();
            double upper = radius * radius / (distance * distance + radius * radius);
            CHECK(image.meanY() > 0.9 * upper);
            CHECK(image.meanY() < 1.01 * upper);
            for (size_t i = 1; i < image.xyz.size(); i += 3) CHECK(image.xyz[i] < 1.05 * upper);
        }
    }
}

void checkFreeEyeThroughTelescope(const std::string& backend) {
    using namespace owe;
    Scene sc;
    Prescription p = loadPrescription("lenses/refractor_150mm.lens");
    solveAfocal(p, p.wavelength, catalogIndex());
    auto instrument = buildPrescription(sc.world, p, "glass", -1, Transform{}, 0.085, catalogIndex());

    // A distant luminous disc subtending Saturn's angular radius, and real glass surfaces.
    // No named eyepiece observer, display override, texture projection or cached image.
    SurfaceOptics black; black.type = SurfaceType::Absorber;
    auto absorber = sc.world.addOptics(black);
    Emission emission; emission.radiance = Spectrum::constant(1);
    buildSheet(sc.world, "object", 45.5, 45.5, absorber, sc.world.addEmission(emission), -1,
               Transform::translate({0, 0, -1e6}), true);
    auto observer = std::make_unique<IdealObserver>();
    observer->name = "arbitrary eye";
    observer->width = observer->height = 48;
    observer->fovY = radians(2);
    observer->up = {0, 1, 0};
    auto& eye = *observer;
    sc.detectors.push_back(std::move(observer));
    Vec3 pupil{0, 0, instrument.paraxial.exitPupilZ};
    eye.position = pupil; eye.lookAt = {0, 0, -1}; sc.build();
    RenderSettings rs; rs.backend = backend;
    auto renderer = makeRenderer(sc, 0, rs);
    double centred = 0;
    for (int pose = 0; pose < 6; ++pose) {
        eye.position = pupil;
        if (pose == 1) eye.position.x += 0.00025;
        if (pose == 2) eye.position.y -= 0.00025;
        if (pose == 3) eye.position.z += 0.005;
        eye.lookAt = eye.position + Vec3(pose == 5 ? std::tan(radians(0.15)) : 0, 0, -1);
        eye.pupilRadius = pose == 4 ? 0.0025 : 0.00055;
        eye.focusDistance = pose == 4 ? 3.5 : Inf;
        eye.prepare(sc.world);
        CHECK(renderer->resetObserver());
        renderer->runPass(256);
        Image img = renderer->resolve();
        double centre = 0, corners = 0;
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) {
                centre += img.at(x + 20, y + 20).y;
                corners += img.at(x, y).y;
            }
        CHECK(centre > 0.05);
        CHECK(centre > 20 * corners);
        if (pose == 0) centred = img.meanY();
        // A larger eye pupil still receives the image, with the correct pupil dilution.
        CHECK(img.meanY() > centred * 0.005);
        if (pose == 4) CHECK(img.meanY() < centred * 0.25);
    }

    // Blocking the objective must extinguish the image at the unchanged free-eye pose.
    renderer.reset();
    buildSheet(sc.world, "objective cap", 0.09, 0.09, absorber, -1, -1,
               Transform::translate({0, 0, -0.001}), true);
    sc.build();
    renderer = makeRenderer(sc, 0, rs); renderer->runPass(128);
    CHECK_NEAR(renderer->resolve().meanY(), 0, 0);
}

// An eye pupil (5 mm) much wider than the telescope's exit pupil (1 mm) guides its samples to the
// exit pupil. That changes the noise only: the measurement equals uniform pupil sampling's, at the
// exit pupil and behind it, with far less variance.
void checkGuidedPupilSampling(const std::string& backend) {
    using namespace owe;
    Scene sc;
    Prescription p = loadPrescription("lenses/refractor_150mm.lens");
    solveAfocal(p, p.wavelength, catalogIndex());
    auto instrument = buildPrescription(sc.world, p, "glass", -1, Transform{}, 0.085, catalogIndex());
    sc.world.addExitPupil({"glass", instrument.assembly, instrument.paraxial.exitPupilZ, instrument.paraxial.exitPupilRadius});
    SurfaceOptics black; black.type = SurfaceType::Absorber;
    Emission emission; emission.radiance = Spectrum::constant(1);
    buildSheet(sc.world, "object", 45.5, 45.5, sc.world.addOptics(black), sc.world.addEmission(emission), -1,
               Transform::translate({0, 0, -1e6}), true);
    auto observer = std::make_unique<IdealObserver>();
    observer->name = "eye";
    observer->width = observer->height = 16;
    observer->fovY = radians(2);
    observer->up = {0, 1, 0};
    observer->pupilRadius = 0.0025;
    auto& eye = *observer;
    sc.detectors.push_back(std::move(observer));
    for (double behind : {0.0, 0.004}) {
        eye.position = {0.0002, 0, instrument.paraxial.exitPupilZ + behind};
        eye.lookAt = eye.position + Vec3(0, 0, -1);
        sc.build();
        CHECK(eye.guide.radius > 0);
        CHECK(eye.describe().find("guided") != std::string::npos);
        auto stats = [&](bool guided, double& mean, double& se) {
            const int runs = 8;
            double s = 0, s2 = 0;
            for (int r = 0; r < runs; ++r) {
                eye.prepare(sc.world);
                if (!guided) eye.guide = {};
                double y = render(sc, 0, backend, 64, uint64_t(100 * guided + r + 1)).meanY();
                s += y;
                s2 += y * y;
            }
            mean = s / runs;
            se = std::sqrt(std::max(0.0, s2 / runs - mean * mean) / (runs - 1));
        };
        double mg, seg, mu, seu;
        stats(true, mg, seg);
        stats(false, mu, seu);
        std::fprintf(stderr, "      behind %.3f m: guided %.5g ± %.2g, uniform %.5g ± %.2g\n", behind, mg, seg, mu, seu);
        CHECK(mu > 0);
        CHECK(std::abs(mg - mu) < 4 * std::sqrt(seg * seg + seu * seu));
        CHECK(seg < 0.6 * seu);
    }
}

}  // namespace

// ---------------------------------------------------------------- analytic results, on every backend

TEST(backends_white_furnace_closed_diffuse_sphere) {
    forEachBackend([](const std::string& backend) {
            // Inside a closed sphere with albedo ρ and emitted radiance Le: L = Le / (1 − ρ).
            for (double rho : {0.0, 0.5, 0.8}) {
                Scene sc;
                SurfaceOptics o;
                o.type = SurfaceType::Diffuse;
                o.reflectance = Spectrum::constant(rho);
                uint32_t optics = sc.world.addOptics(o);
                Emission e;
                e.radiance = Spectrum::constant(1.0);
                e.front = false;
                e.back = true;
                int emission = sc.world.addEmission(e);
                int b = sc.world.addBody("shell", "sphere", -1, Transform{});
                uint32_t inside = sc.world.addRegion("shell.inside", sc.world.medium("air"), b);
                sc.world.addBoundary(b, "surface", std::make_shared<SphereShape>(1.0), Transform{}, kOutside, inside, optics, emission);
                auto obs = std::make_unique<IdealObserver>();
                obs->name = "eye";
                obs->width = obs->height = 16;
                obs->position = {0.2, 0.1, -0.3};
                obs->lookAt = {1, 0.3, 0};
                obs->fovY = radians(70);
                sc.detectors.push_back(std::move(obs));
                sc.build();
                Image img = render(sc, 0, backend, 1024);
                CHECK_NEAR(img.meanY(), 1.0 / (1 - rho), 0.01 / (1 - rho));
            }
    });
}

TEST(backends_dielectric_sphere_preserves_uniform_radiance) {
    forEachBackend([](const std::string& backend) {
            // A lossless dielectric in a uniform radiance field is invisible (n² law, Fresnel, TIR, dispersion).
            Scene sc;
            sc.world.env.skyModel = Environment::Sky::Uniform;
            sc.world.env.zenith = Spectrum::constant(1.0);
            BodyMaterial glass;
            glass.transparent = true;
            glass.medium = "N-SF11";
            glass.optics = sc.world.dielectricOptics();
            buildSphere(sc.world, "ball", 1.0, glass, -1, Transform{});
            auto obs = std::make_unique<IdealObserver>();
            obs->name = "eye";
            obs->width = obs->height = 24;
            obs->position = {0, -4, 0};
            obs->lookAt = {0, 0, 0};
            obs->fovY = radians(30);
            sc.detectors.push_back(std::move(obs));
            sc.build();
            Image img = render(sc, 0, backend, 1024);
            CHECK_NEAR(img.meanY(), 1.0, 0.004);
            double inside = 0;
            int n = 0;
            for (int y = 6; y < 18; ++y)
                for (int x = 6; x < 18; ++x) { inside += img.at(x, y).y; ++n; }
            CHECK_NEAR(inside / n, 1.0, 0.01);
    });
}

TEST(backends_uncoated_glass_reflected_energy_matches_fresnel) { forEachBackend(checkUncoatedGlassReflectance); }
TEST(backends_pinhole_irradiance_depends_on_aperture_area) { forEachBackend(checkPinholeIrradiance); }
TEST(backends_distant_stars_and_nearby_lamps_both_illuminate_surfaces) { forEachBackend(checkDistantAndNearbyLights); }
TEST(backends_reflected_moonlight_illuminates_ground) { forEachBackend(checkReflectedMoonlight); }
TEST(backends_telescope_forms_image_for_freely_placed_eyes) { forEachBackend(checkFreeEyeThroughTelescope); }
TEST(backends_guided_eye_pupil_measures_the_same_with_less_noise) { forEachBackend(checkGuidedPupilSampling); }

// Moving the eye (the viewer's navigation) restarts the estimate: the renderer after a reset equals
// a fresh renderer of the new pose and resolution, bit for bit.
TEST(backends_reset_observer_matches_fresh_render) {
    forEachBackend([](const std::string& backend) {
        Scene sc = loadScene("scenes/the_lens.owe");
        auto& eye = dynamic_cast<IdealObserver&>(*sc.detectors[0]);
        eye.width = 24; eye.height = 16; eye.prepare(sc.world);
        RenderSettings rs; rs.backend = backend;
        auto r = makeRenderer(sc, 0, rs);
        r->runPass(2);
        for (int i = 0; i < 3; ++i) {
            eye.position.x += 0.01;
            eye.lookAt.z += 0.01;
            eye.width = i == 1 ? 32 : 16;
            eye.height = i == 1 ? 24 : 12;
            eye.prepare(sc.world);
            CHECK(r->resetObserver());
            CHECK(r->samplesPerPixel() == 0 && r->stats().paths == 0);
            r->runPass(2);
            auto fresh = makeRenderer(sc, 0, rs);
            fresh->runPass(2);
            CHECK(r->resolve().xyz == fresh->resolve().xyz);
        }
    });
}

namespace {
Scene diffuseScreenScene() {
    Scene sc = loadSceneFromString(R"(
        material paint { type = diffuse reflectance = 0.7 back = black }
        sensor Screen { size = (2, 2) resolution = (24, 18) material = paint
            aim_center = (0, 0, 1) aim_axis = (0, 0, 1) aim_radius = 0.15 }
        observer Eye { position = (0, 0, 1) look_at = (0, 0, 0) up = (0, 1, 0)
            fov = 20deg resolution = (24, 18) }
    )");
    sc.world.env.skyModel = Environment::Sky::Uniform;
    sc.world.env.zenith = Spectrum::constant(1.0);
    sc.build();
    return sc;
}
}  // namespace

// A diffuse measurement screen under a uniform sky: its readout is the irradiance π, and an eye
// sees radiance ρ = 0.7 from every direction of the front hemisphere and nothing from behind.
TEST(backends_diffuse_screen_readout_and_observer) {
    forEachBackend([](const std::string& backend) {
        Scene sc = diffuseScreenScene();
        CHECK_NEAR(render(sc, 0, backend, 2048).meanY(), Pi, 0.05 * Pi);
        CHECK_NEAR(render(sc, 1, backend, 2048).meanY(), 0.7, 0.035);
        // The screen's sampling guide follows the eye when it moves.
        auto& eye = dynamic_cast<IdealObserver&>(*sc.detectors[1]);
        RenderSettings rs; rs.backend = backend;
        auto renderer = makeRenderer(sc, 1, rs); renderer->runPass(2);
        eye.position.x += 0.2; eye.prepare(sc.world);
        CHECK(renderer->resetObserver()); renderer->runPass(4);
        auto fresh = makeRenderer(sc, 1, rs); fresh->runPass(4);
        CHECK(renderer->resolve().xyz == fresh->resolve().xyz);
        eye.fovY = radians(0.2);
        for (double polar : {0., 45., 80., 85., 180.}) {
            double a = radians(polar);
            eye.position = Vec3(std::sin(a), 0, std::cos(a)) * 0.5;
            eye.prepare(sc.world);
            CHECK(renderer->resetObserver()); renderer->runPass(2048);
            CHECK_NEAR(renderer->resolve().meanY(), polar == 180 ? 0 : 0.7, 0.035);
        }
    });
}

// A flat, fan-triangulated plate with chamfered corners — the triangulation modelling tools give a
// bevelled board — has sliver triangles 0.3 mm wide and 1.4 m long. Under a uniform sky with a black
// ground below, its top reflects exactly ρ·L. A hit computed off the surface by more than the ray
// offset (float32 Möller–Trumbore loses precision as 1/area) starts the next ray under the plate,
// where it sees the black ground: this read 32% low on the GPU before the fix.
TEST(backends_sliver_triangles_are_lit_like_their_neighbours) {
    forEachBackend([](const std::string& backend) {
        Scene sc;
        sc.world.env.skyModel = Environment::Sky::Uniform;
        sc.world.env.zenith = Spectrum::constant(1.0);
        SurfaceOptics paint; paint.type = SurfaceType::Diffuse; paint.reflectance = Spectrum::constant(0.5);
        SurfaceOptics black; black.type = SurfaceType::Absorber;
        const double hx = 0.7, hy = 0.075, b = 0.0004;
        std::vector<Vec3> octagon = {{-hx + b, -hy, 0}, {hx - b, -hy, 0}, {hx, -hy + b, 0}, {hx, hy - b, 0},
                                     {hx - b, hy, 0},  {-hx + b, hy, 0}, {-hx, hy - b, 0}, {-hx, -hy + b, 0}};
        std::vector<std::array<uint32_t, 3>> fan;
        for (uint32_t k = 1; k + 1 < 8; ++k) fan.push_back({0, k, k + 1});
        int plate = sc.world.addBody("plate", "mesh", -1, Transform{});
        sc.world.addBoundary(plate, "surface", std::make_shared<MeshShape>(octagon, fan, "plate"), Transform{}, kOutside,
                             kOutside, sc.world.addOptics(paint));
        buildSheet(sc.world, "ground", 4, 4, sc.world.addOptics(black), -1, -1, Transform::translate({0, 0, -0.03}), true);
        // Look obliquely at the sliver (0, 3, 4) where it is 0.3 mm wide, over a ~0.1 mm footprint. (A
        // ray straight down hides the error: its numerator and denominator round alike.)
        auto eye = std::make_unique<IdealObserver>();
        eye->name = "oblique";
        eye->position = {0.2, -0.15, 0.3};
        eye->lookAt = {0.35, 0.03735, 0};
        eye->up = {0, 0, 1};
        eye->fovY = radians(0.01);
        eye->width = eye->height = 16;
        sc.detectors.push_back(std::move(eye));
        sc.build();
        // Every sample carries exactly ρ·L here, so the estimate has no noise.
        CHECK_NEAR(render(sc, 0, backend, 256).meanY(), 0.5, 0.005);
    });
}

// Every integrator a backend implements measures the same irradiance from a Lambertian disk onto a
// sensor below it: path tracing (unaimed and aimed at the disk) and particles.
TEST(backends_every_integrator_measures_lambertian_disk_irradiance) {
    const double a = 0.3, h = 0.5, L = 2.0, half = 0.15;
    const double expected = meanDiskIrradiance(L, a, h, half);
    forEachBackend([&](const std::string& backend) {
        for (bool aimed : {false, true}) {
            Scene sc = diskIrradianceScene(a, h, aimed, half);
            CHECK_NEAR(render(sc, 0, backend, 40000).meanY(), expected, 0.01 * expected);
        }
        if (!findBackend(backend)->supports("light")) return;
        Scene sc = diskIrradianceScene(a, h, false, half);
        CHECK_NEAR(render(sc, 0, backend, 400000, 1, "light").meanY(), expected, 0.01 * expected);
        // A diffuse measurement screen records the same incident irradiance with particles.
        Scene screen = diskIrradianceScene(a, h, true, half, true);
        CHECK_NEAR(render(screen, 0, backend, 400000, 1, "light").meanY(), expected, 0.02 * expected);
    });
}

// Inside a closed diffuse sphere of albedo ρ emitting Le: L = Le / (1 − ρ), with every integrator
// (particles reach the eye through connections from every diffuse vertex).
TEST(backends_every_integrator_passes_the_white_furnace) {
    forEachBackend([](const std::string& backend) {
        for (const char* integrator : {"light", "hybrid", "sppm", "bdpt", "vcm"}) {
            if (!findBackend(backend)->supports(integrator)) continue;
            // Bidirectional iterations cost a fixed overhead each: fewer, over more pixels.
            const bool bidirectional = std::string(integrator) == "bdpt" || std::string(integrator) == "vcm";
            for (double rho : {0.0, 0.5}) {
                Scene sc;
                SurfaceOptics o;
                o.type = SurfaceType::Diffuse;
                o.reflectance = Spectrum::constant(rho);
                Emission e;
                e.radiance = Spectrum::constant(1.0);
                e.front = false;
                e.back = true;
                int b = sc.world.addBody("shell", "sphere", -1, Transform{});
                uint32_t inside = sc.world.addRegion("shell.inside", sc.world.medium("air"), b);
                sc.world.addBoundary(b, "surface", std::make_shared<SphereShape>(1.0), Transform{}, kOutside, inside,
                                     sc.world.addOptics(o), sc.world.addEmission(e));
                auto obs = std::make_unique<IdealObserver>();
                obs->name = "eye";
                obs->width = obs->height = bidirectional ? 32 : 8;
                obs->position = {0.2, 0.1, -0.3};
                obs->lookAt = {1, 0.3, 0};
                obs->fovY = radians(70);
                sc.detectors.push_back(std::move(obs));
                sc.build();
                double measured = render(sc, 0, backend, bidirectional ? 256 : 4096, 1, integrator).meanY();
                CHECK_NEAR(measured, 1.0 / (1 - rho), 0.02 / (1 - rho));
            }
        }
    });
}

// ---------------------------------------------------------------- agreement with the reference

TEST(backends_agree_with_reference_on_a_caustic) {
    forEachCandidate([](const std::string& backend) {
        Scene sc = loadSceneFromString(R"(
            units = m
            material floor { type = diffuse  reflectance = 0.7 }
            material steel { type = conductor  metal = aluminium  roughness = 0.2 }
            body Floor { type = sheet  size = (3, 3)  material = floor }
            body Ball  { type = sphere radius = 0.3  position = (0, 0, 0.45)  medium = N-BK7 }
            body Can   { type = cylinder radius = 0.12 height = 0.5 position = (0.55, 0.2, 0)  material = steel }
            body Lamp  { type = sphere radius = 0.35  position = (0.2, 0.1, 1.8)  material = black  emission = blackbody(5000K, 4) }
            observer Eye { position = (0, -2.0, 1.1)  look_at = (0, 0, 0.3)  fov = 45deg  resolution = (36, 24) }
        )");
        // Every integrator both backends implement: path, and the caustic owned by particles.
        for (const char* integrator : {"path", "light", "hybrid"}) {
            if (!findBackend(backend)->supports(integrator) || !referenceBackend().supports(integrator)) continue;
            std::fprintf(stderr, "    %s\n", integrator);
            checkAgainstReference(sc, 0, backend, 256, 8, 0.02, integrator);
        }
    });
}

TEST(backends_agree_with_reference_on_diffuse_screens) {
    forEachCandidate([](const std::string& backend) {
        Scene sc = diffuseScreenScene();
        checkAgainstReference(sc, 0, backend, 512, 6, 0.06);
        checkAgainstReference(sc, 1, backend, 512, 6, 0.06);
    });
}

// A telescope at the eye (exit-pupil observer), a physical camera (surface sensor with pupil
// aiming), and an extreme-scale view: Saturn at 1.28e12 m through a 150 mm refractor.
TEST(backends_agree_with_reference_through_instruments_and_cameras) {
    forEachCandidate([](const std::string& backend) {
        struct Case {
            const char* scene;
            const char* detector;
            int width, height, spp;
            double maxRelSe;  // Saturn among stars is a noisy image
        };
        for (Case c : {Case{"scenes/the_telescope.owe", "Eyepiece", 36, 24, 128, 0.02},
                       Case{"scenes/the_temple.owe", "Cam", 36, 24, 128, 0.02},
                       Case{"scenes/the_observatory.owe", "SaturnEyepiece", 96, 64, 64, 0.03}}) {
            Scene sc = loadScene(c.scene);
            int det = sc.findDetector(c.detector);
            sc.useDetector(det);
            sc.detectors[size_t(det)]->width = c.width;
            sc.detectors[size_t(det)]->height = c.height;
            sc.build();
            std::fprintf(stderr, "    %s:%s\n", c.scene, c.detector);
            checkAgainstReference(sc, det, backend, c.spp, 8, c.maxRelSe);
        }
    });
}
