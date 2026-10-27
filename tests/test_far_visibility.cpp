// Distant light vertices must not bypass opaque matter near an observer. The
// visibility oracle is exact zero, independently of spectral/Monte Carlo error:
// a black screen covers the complete pupil and field. The open control proves
// the local far source actually illuminates the scene, and camera paths provide
// an independent zero measurement. No new visibility helper is called here.
#include <cmath>
#include <cstdio>
#include <string>

#include "check.hpp"
#include "owe/backends/registry.hpp"
#include "owe/loader/scene_loader.hpp"

using namespace owe;

namespace {
constexpr const char* fixture = "benchmarks/regressions/far_light_visibility.owe";

Scene farScene(bool blocked, bool directEmitter = false) {
    std::vector<std::string> edits;
    if (!blocked) edits.push_back("NearBlocker.position=(1,0,0.025)");
    if (directEmitter) {
        // Suppress the diffuse-vertex strategy and put the emitter in the field,
        // separately exercising the particle's directly visible emitter splat.
        edits.push_back("FarDiffuser.material=black");
        edits.push_back("Lamp.position=(0,0,999980)");
    }
    return loadSceneWithEdits(fixture, edits);
}

Image image(const Scene& scene, const std::string& backend, const char* integrator, int spp) {
    RenderSettings settings = scene.render;
    settings.backend = backend;
    settings.integrator = integrator;
    settings.threads = 2;
    auto renderer = makeRenderer(scene, 0, settings);
    renderer->runPass(spp);
    CHECK(renderer->stats().leaks == 0);
    CHECK(renderer->stats().inconsistencies == 0);
    return renderer->resolve();
}

void exactlyDark(const Image& image) {
    for (double value : image.xyz) CHECK(value == 0);
}

void checkVisibility(const std::string& backend, const char* integrator, bool directEmitter = false) {
    Scene blocked = farScene(true, directEmitter);
    Scene open = farScene(false, directEmitter);
    const Image camera = image(blocked, backend, "path", 8);
    const Image particles = image(blocked, backend, integrator, 128);
    const Image control = image(open, backend, integrator, 128);
    std::fprintf(stderr, "    [%s %s %s] blocked camera %.9g, particles %.9g, open %.9g\n",
                 backend.c_str(), integrator, directEmitter ? "emitter" : "diffuser",
                 camera.meanY(), particles.meanY(), control.meanY());
    exactlyDark(camera);
    exactlyDark(particles);
    CHECK(std::isfinite(control.meanY()));
    CHECK(control.meanY() > 1e-6);
}
} // namespace

TEST(backends_far_light_vertices_respect_near_observer_occlusion) {
    const Backend* gpu = findBackend("gpu");
    std::string why;
    if (!gpu || !gpu->available(&why)) {
        std::fprintf(stderr, "    [gpu] skipped: %s\n", why.c_str());
        return;
    }
    for (const char* integrator : {"light", "bdpt", "vcm"})
        if (gpu->supports(integrator)) checkVisibility("gpu", integrator);
}

TEST(backends_far_emitters_respect_near_observer_occlusion) {
    const Backend* gpu = findBackend("gpu");
    if (gpu && gpu->available() && gpu->supports("light")) checkVisibility("gpu", "light", true);
}

TEST(backends_far_emitters_respect_source_end_occlusion) {
    // Reversing a distant connection must preserve blockers near its source,
    // and terminate at the emitter rather than continuing beyond it. The same
    // seed gives paired controls without a reference or Monte Carlo tolerance.
    constexpr const char* source = "benchmarks/regressions/far_source_visibility.owe";
    for (const Backend* backend : backends()) {
        std::string why;
        if (!backend->available(&why)) {
            std::fprintf(stderr, "    [%s] skipped: %s\n", backend->name().c_str(), why.c_str());
            continue;
        }
        if (!backend->supports("light")) continue;
        Scene blocked = loadScene(source);
        Scene open = loadSceneWithEdits(source, {"SourceBlocker.position=(1,0,999999.5)"});
        Scene beyond = loadSceneWithEdits(source, {"SourceBlocker.position=(0,0,1000000.5)"});
        const Image black = image(blocked, backend->name(), "light", 128);
        const Image control = image(open, backend->name(), "light", 128);
        const Image behind = image(beyond, backend->name(), "light", 128);
        std::fprintf(stderr, "    [%s light] source-end blocked %.9g, open %.9g, beyond %.9g\n",
                     backend->name().c_str(), black.meanY(), control.meanY(), behind.meanY());
        exactlyDark(black);
        CHECK(std::isfinite(control.meanY()));
        CHECK(control.meanY() > 1e-6);
        CHECK(behind.xyz.size() == control.xyz.size());
        if (behind.xyz.size() == control.xyz.size())
            for (std::size_t i = 0; i < control.xyz.size(); ++i)
                CHECK_NEAR(behind.xyz[i], control.xyz[i], 1e-12 + 1e-6 * std::abs(control.xyz[i]));
    }
}

TEST(backends_camera_nee_respects_source_end_occlusion) {
    // The camera sees only the near diffuse receiver, so its nonzero open
    // signal exercises finite next-event estimation rather than emitter splats.
    // No sky or other source can illuminate the completely screened receiver.
    constexpr const char* source = "benchmarks/regressions/far_nee_visibility.owe";
    for (const Backend* backend : backends()) {
        std::string why;
        if (!backend->available(&why)) {
            std::fprintf(stderr, "    [%s] skipped: %s\n", backend->name().c_str(), why.c_str());
            continue;
        }
        Scene blocked = loadScene(source);
        Scene open = loadSceneWithEdits(source, {"SourceBlocker.position=(1,0,999999.95)"});
        Scene beyond = loadSceneWithEdits(source, {"SourceBlocker.position=(0,0,1000000.05)"});
        // VCM/BDPT have a separate camera-vertex NEE implementation; a path
        // pass alone cannot certify their finite source-end connections.
        for (const char* integrator : {"path", "bdpt", "vcm"}) {
            if (!backend->supports(integrator)) continue;
            const Image black = image(blocked, backend->name(), integrator, 128);
            const Image control = image(open, backend->name(), integrator, 128);
            const Image behind = image(beyond, backend->name(), integrator, 128);
            std::fprintf(stderr, "    [%s %s] NEE source-end blocked %.9g, open %.9g, beyond %.9g\n",
                         backend->name().c_str(), integrator, black.meanY(), control.meanY(), behind.meanY());
            exactlyDark(black);
            CHECK(std::isfinite(control.meanY()));
            CHECK(control.meanY() > 1e-6);
            CHECK(behind.xyz.size() == control.xyz.size());
            if (behind.xyz.size() == control.xyz.size())
                for (std::size_t i = 0; i < control.xyz.size(); ++i)
                    CHECK_NEAR(behind.xyz[i], control.xyz[i], 1e-12 + 1e-6 * std::abs(control.xyz[i]));
        }
    }
}

TEST(backends_surface_sensor_respects_exact_opaque_contact) {
    // Exercise Detector::generate through the public renderer. A sensor in the
    // same plane as a distinct opaque cover cannot measure the sky above it.
    // Raising only the sensor by either a resolved 100 nm or 1 mm exposes the
    // uniform hemisphere and supplies the analytic irradiance pi controls.
    for (const Backend* backend : backends()) {
        std::string why;
        if (!backend->available(&why)) {
            std::fprintf(stderr, "    [%s] skipped: %s\n", backend->name().c_str(), why.c_str());
            continue;
        }
        constexpr const char* sensor = "benchmarks/regressions/sensor_opaque_contact.owe";
        Scene contact = loadScene(sensor);
        const Image black = image(contact, backend->name(), "path", 64);
        exactlyDark(black);
        for (const char* gap : {"0.0000001", "0.001"}) {
            Scene separated = loadSceneWithEdits(sensor,
                {std::string("Screen.position=(0,0,") + gap + ")"});
            const Image control = image(separated, backend->name(), "path", 64);
            std::fprintf(stderr, "    [%s] sensor contact %.9g, gap %s m %.9g, analytic %.9g\n",
                         backend->name().c_str(), black.meanY(), gap, control.meanY(), Pi);
            CHECK_REL(control.meanY(), Pi, 0.02);
        }
    }
}

TEST(backends_particles_respect_surface_sensor_opaque_contact) {
    // Unlike the sky control, this finite disk emits particles onto the sensor.
    // Contact ties must never splat flux through the distinct opaque cover.
    constexpr const char* sensor = "benchmarks/regressions/sensor_particle_contact.owe";
    for (const Backend* backend : backends()) {
        std::string why;
        if (!backend->available(&why)) {
            std::fprintf(stderr, "    [%s] skipped: %s\n", backend->name().c_str(), why.c_str());
            continue;
        }
        // BDPT supports ideal observers, not physical surface sensors. Light
        // tracing supplies the finite-emitter particle oracle on both backends.
        for (const char* integrator : {"light"}) {
            if (!backend->supports(integrator)) continue;
            Scene contact = loadScene(sensor);
            Scene separated = loadSceneWithEdits(sensor, {"Screen.position=(0,0,0.001)"});
            // An incoming ray at two distinct ambient-facing coplanar sheets
            // may be reported ambiguous. That must remain diagnostic and must
            // never manufacture positive sensor energy. The open control has
            // resolved separation and no such ambiguity.
            RenderSettings settings = contact.render;
            settings.backend = backend->name();
            settings.integrator = integrator;
            settings.threads = 2;
            auto renderer = makeRenderer(contact, 0, settings);
            renderer->runPass(128);
            const Image black = renderer->resolve();
            const Image control = image(separated, backend->name(), integrator, 128);
            std::fprintf(stderr, "    [%s %s] finite-source sensor contact %.9g, separated %.9g, diagnostics %llu\n",
                         backend->name().c_str(), integrator, black.meanY(), control.meanY(),
                         static_cast<unsigned long long>(renderer->stats().inconsistencies));
            exactlyDark(black);
            CHECK(std::isfinite(control.meanY()));
            CHECK(control.meanY() > 1e-4);
        }
    }
}
