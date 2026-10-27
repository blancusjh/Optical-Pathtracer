// Bidirectional path tracing and vertex connection and merging on the GPU (shaders/vcm.slang). BDPT
// is unbiased: it must draw the path tracer's image. VCM adds merging, consistent (its bias shrinks
// with the radius): it must draw the same image, and also what path tracing cannot, a lens caustic
// seen through a glass plate. On a glossy reflection, where a photon map gathered at the glossy
// surface is poor, it must stay as good as camera paths.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "check.hpp"
#include "owe/backends/registry.hpp"
#include "owe/core/medium.hpp"
#include "owe/loader/scene_loader.hpp"

using namespace owe;

namespace {

bool gpuHas(const char* integrator) {
    const Backend* gpu = findBackend("gpu");
    return gpu && gpu->available() && gpu->supports(integrator);
}

Image render(const Scene& sc, const std::string& integrator, int spp, uint64_t seed = 1) {
    RenderSettings rs = sc.render;
    rs.backend = "gpu";
    rs.integrator = integrator;
    rs.seed = seed;
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(spp);
    return R->resolve();
}

// Preserve external debug settings, including when a regression throws.
class ScopedEnvironment {
  public:
    ScopedEnvironment(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        present_ = old != nullptr;
        if (old) old_ = old;
        set(value);
    }
    ~ScopedEnvironment() { set(present_ ? old_.c_str() : nullptr); }
    void set(const char* value) const {
#ifdef _WIN32
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value)
            setenv(name_.c_str(), value, 1);
        else
            unsetenv(name_.c_str());
#endif
    }

  private:
    std::string name_, old_;
    bool present_ = false;
};

void checkReplay(const Renderer& actual, const Renderer& reference, bool merges) {
    CHECK(actual.passes() == reference.passes());
    CHECK(actual.samplesPerPixel() == reference.samplesPerPixel());
    CHECK(actual.stats().paths == reference.stats().paths);
    CHECK(actual.stats().segments == reference.stats().segments);
    CHECK(actual.stats().inconsistencies == reference.stats().inconsistencies);
    CHECK(actual.stats().leaks == reference.stats().leaks);
    const Image a = actual.resolve(), b = reference.resolve();
    CHECK(b.meanY() > 0);
    // Disable the randomly chosen connection strategy in these replay checks:
    // cache insertion order is unspecified. Merging then differs only by float
    // summation order; light splats, camera streams, and the learned sun guide
    // must agree after rejected attempts.
    const double tolerance = merges ? b.meanY() * 2e-5 : 0;
    double worst = 0;
    for (size_t i = 0; i < a.xyz.size(); ++i) worst = std::max(worst, std::abs(a.xyz[i] - b.xyz[i]));
    std::fprintf(stderr, "    replay image max error %.6g (tolerance %.6g)\n", worst, tolerance);
    CHECK(worst <= tolerance);
}

// The largest deviation from 1 of a's over b's sums in 4 × 4 blocks (those with at least a quarter
// of an average block's light).
double worstBlock(const Image& a, const Image& b) {
    const int n = 4, bw = a.width / n, bh = a.height / n;
    double sa[16] = {}, sb[16] = {}, mean = 0;
    for (int y = 0; y < n * bh; ++y)
        for (int x = 0; x < n * bw; ++x) {
            sa[(y / bh) * n + x / bw] += a.at(x, y).y;
            sb[(y / bh) * n + x / bw] += b.at(x, y).y;
        }
    for (double v : sb) mean += v / 16;
    double worst = 0;
    for (int k = 0; k < 16; ++k)
        if (sb[k] >= 0.25 * mean) worst = std::max(worst, std::abs(sa[k] / sb[k] - 1));
    return worst;
}

// A diffuse ball on a floor before a wall, in sun and sky: direct light, interreflection, shadows.
Scene room() {
    return loadSceneFromString(R"(
        units = m
        world { sky = uniform(rgb(1, 1, 1, luminance = 0.05))  sun { elevation = 35deg  azimuth = 200deg  luminance = 2e4 } }
        material clay { type = diffuse  reflectance = 0.7 }
        material ground { type = diffuse  reflectance = 0.5 }
        body Ball { type = sphere  radius = 1  position = (0, 0, 0)  material = clay }
        body Floor { type = sheet  size = (12, 12)  position = (0, 0, -1.02)  material = ground }
        body Wall { type = sheet  size = (12, 8)  position = (0, 2.5, 2.98)  axis = (0, -1, 0)  up = (0, 0, 1)  material = ground }
        observer Eye { position = (0, -4, 0)  look_at = (0, 0, 0)  up = (0, 0, 1)  fov = 34deg  pupil = 1mm  focus = 4
                       resolution = (96, 96) }
        render { detector = Eye }
    )");
}

}  // namespace

TEST(vcm_cache_overflow_replays_the_pass_without_losing_vertices) {
    if (!gpuHas("vcm")) return;
    Scene sc = room();
    sc.detectors[0]->width = sc.detectors[0]->height = 16;
    RenderSettings rs = sc.render;
    rs.backend = "gpu";
    rs.maxDepth = 6;
    rs.photonsPerPixel = 4;
    rs.vcmWavelengthGroups = 4;
    rs.diagnosticVcmMask = 4;
    ScopedEnvironment limit("OWE_VCM_CACHE_MAX_CAPACITY", nullptr);
    ScopedEnvironment capacity("OWE_VCM_CACHE_CAPACITY", "32768");
    for (const char* integrator : {"bdpt", "vcm"}) {
        rs.integrator = integrator;
        capacity.set("32768");
        auto reference = makeRenderer(sc, 0, rs);
        capacity.set("4"); // One slot per wavelength group: guaranteed to overflow
                           // this sunlit room.
        auto replayed = makeRenderer(sc, 0, rs);
        reference->runPass(3);
        replayed->runPass(3);
        reference->runPass(5);
        replayed->runPass(5); // Exercise the guide/radius and normalisation across
                              // accepted passes.
        checkReplay(*replayed, *reference, rs.integrator == "vcm");
    }
}

TEST(vcm_cache_overflow_rejects_a_bounded_allocation_and_can_recover) {
    if (!gpuHas("vcm")) return;
    Scene sc = room();
    sc.detectors[0]->width = sc.detectors[0]->height = 16;
    RenderSettings rs = sc.render;
    rs.backend = "gpu";
    rs.integrator = "vcm";
    rs.maxDepth = 6;
    rs.photonsPerPixel = 4;
    rs.vcmWavelengthGroups = 4;
    rs.diagnosticVcmMask = 4;
    ScopedEnvironment limit("OWE_VCM_CACHE_MAX_CAPACITY", nullptr);
    ScopedEnvironment capacity("OWE_VCM_CACHE_CAPACITY", "32768");
    auto reference = makeRenderer(sc, 0, rs);
    capacity.set("4");
    limit.set("4");
    auto bounded = makeRenderer(sc, 0, rs);
    bool rejected = false;
    try {
        bounded->runPass(3);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("capacity limit") != std::string::npos;
    }
    CHECK(rejected);
    CHECK(bounded->passes() == 0);
    CHECK(bounded->samplesPerPixel() == 0);
    CHECK(bounded->stats().paths == 0);
    CHECK(bounded->stats().segments == 0);
    for (double value : bounded->resolve().xyz) CHECK(value == 0);
    // A failed growth retains valid old buffers and restores the guide/iteration
    // stream, so the same renderer can recover once the external allocation cap
    // is lifted.
    limit.set(nullptr);
    reference->runPass(3);
    bounded->runPass(3);
    checkReplay(*bounded, *reference, true);
}

TEST(vcm_cached_connections_cannot_escape_a_sealed_far_enclosure) {
    if (!gpuHas("vcm")) return;
    constexpr const char* fixture = "benchmarks/regressions/vcm_cached_enclosure_visibility.owe";
    const Scene sealed = loadScene(fixture);
    const Scene open = loadSceneWithEdits(fixture, {"Floor.position=(200,0,999996)"});
    // Leave only camera-to-cached-vertex connections enabled. The open control
    // must therefore illuminate the receiver through the far diffuse patch,
    // rather than finite NEE, merging, or a light-tracing splat. The lamp emits
    // away from the receiver; depth one prevents a camera-side diffuse bounce
    // at the far patch from reaching its emitting side.
    uint32_t mask = 27; // emission, NEE, merging, light tracing
    for (const char* integrator : {"bdpt", "vcm"}) {
        auto measure = [&](const Scene& scene) {
            // Isolate strategies explicitly for this renderer.
            RenderSettings settings = scene.render;
            settings.backend = "gpu";
            settings.integrator = integrator;
            settings.diagnosticVcmMask = mask;
            settings.photonsPerPixel = 1;
            settings.vcmWavelengthGroups = 1;
            auto renderer = makeRenderer(scene, 0, settings);
            renderer->runPass(32);
            CHECK(renderer->stats().inconsistencies == 0);
            CHECK(renderer->stats().leaks == 0);
            return renderer->resolve();
        };
        mask = 27;
        const Image black = measure(sealed);
        const Image control = measure(open);
        // Disabling the remaining strategy must also erase the open signal:
        // this proves that populated diffuse cache vertices drive the oracle.
        mask = 31;
        const Image noConnections = measure(open);
        std::fprintf(stderr, "    %s cached enclosure sealed %.9g, open %.9g, cache disabled %.9g\n",
                     integrator, black.meanY(), control.meanY(), noConnections.meanY());
        for (double value : black.xyz) CHECK(value == 0);
        for (double value : noConnections.xyz) CHECK(value == 0);
        CHECK(std::isfinite(control.meanY()));
        CHECK(control.meanY() > 1e-6);
    }
}

TEST(vcm_cached_diffuse_packets_match_coloured_textures_and_shading_normals) {
    if (!gpuHas("vcm")) return;
    // A coloured checker with relief exercises wavelength-dependent albedo and a shading
    // normal that differs from geometry. Observe both sides: the packet's common pdf must
    // retain the same front/back and normal-agreement rules as the scalar BSDF.
    for (bool back : {false, true}) {
        Scene sc = loadSceneFromString(std::string(R"(
            units = m
            world { sky = uniform(rgb(1, 1, 1, luminance = 1)) }
            material paint { type = diffuse
                texture = checker(a = rgb(0.75, 0.18, 0.08), b = rgb(0.08, 0.35, 0.7), scale = 0.18)
                relief = noise(scale = 0.15, depth = 0.01, octaves = 2) }
            body Sheet { type = sheet  size = (20, 20)  position = (0, 0, 0)  material = paint }
            observer Eye { position = (0, -0.2, )") + (back ? "-1" : "1") + R"()
                look_at = (0, 0, 0)  up = (0, 1, 0)  fov = 35deg  pupil = 1mm
                focus = 1  resolution = (32, 32) }
            render { detector = Eye }
        )");
        const Image reference = render(sc, "path", 1024, 7);
        for (const char* integrator : {"bdpt", "vcm"}) {
            const Image candidate = render(sc, integrator, 512, 7);
            double sumReference[3] = {}, sumCandidate[3] = {};
            for (size_t i = 0; i < reference.xyz.size(); ++i) {
                sumReference[i % 3] += reference.xyz[i];
                sumCandidate[i % 3] += candidate.xyz[i];
            }
            for (int c = 0; c < 3; ++c) {
                const double ratio = sumCandidate[c] / sumReference[c];
                std::fprintf(stderr, "    %s %s coloured diffuse channel %d ratio %.5f\n",
                             integrator, back ? "back" : "front", c, ratio);
                CHECK_NEAR(ratio, 1, 0.015);
            }
        }
    }
}

TEST(vcm_camera_cie_keeps_xyz_normalization_after_dispersion) {
    if (!gpuHas("vcm")) return;
    // A lossless dispersive plate in an isotropic sky cannot change its radiance:
    // reflection and transmission both lead back to the same illumination. The
    // glass terminates secondary wavelengths, so retaining the original packet
    // pdf in a colour cache would incorrectly divide the result by four.
    auto scene = [](bool glass) {
        return loadSceneFromString(std::string(R"(
            units = m
            world { sky = uniform(rgb(0.7, 0.3, 1, luminance = 1)) }
        )") + (glass ? R"(
            body Plate { type = box  size = (20, 20, 0.01)
                         position = (0, 0, 0)  medium = N-BK7 }
        )" : "") + R"(
            observer Eye { position = (0, 0, 1)  look_at = (0, 0, 0)
                up = (0, 1, 0)  fov = 20deg  pupil = 1mm
                focus = 1  resolution = (8, 8) }
            render { detector = Eye  max_depth = 16 }
        )");
    };
    const Image reference = render(scene(false), "path", 4096, 11);
    for (const char* integrator : {"bdpt", "vcm"}) {
        const Image through = render(scene(true), integrator, 4096, 11);
        double sumReference[3] = {}, sumThrough[3] = {};
        for (size_t i = 0; i < reference.xyz.size(); ++i) {
            sumReference[i % 3] += reference.xyz[i];
            sumThrough[i % 3] += through.xyz[i];
        }
        for (int c = 0; c < 3; ++c) {
            CHECK(sumReference[c] > 0);
            const double ratio = sumThrough[c] / sumReference[c];
            std::fprintf(stderr, "    %s dispersive sky channel %d ratio %.5f\n", integrator, c, ratio);
            CHECK_NEAR(ratio, 1, 0.015);
        }
    }
}

TEST(vcm_bdpt_and_vcm_draw_the_path_tracers_image) {
    if (!gpuHas("vcm")) return;
    const Scene sc = room();
    const Image path = render(sc, "path", 4096), bdpt = render(sc, "bdpt", 2048), vcm = render(sc, "vcm", 2048);
    const double bMean = bdpt.meanY() / path.meanY(), vMean = vcm.meanY() / path.meanY();
    const double bBlock = worstBlock(bdpt, path), vBlock = worstBlock(vcm, path);
    std::fprintf(stderr, "    against path: bdpt mean %.4f (worst block %.4f), vcm mean %.4f (worst block %.4f)\n", bMean, bBlock, vMean,
                 vBlock);
    CHECK_NEAR(bMean, 1.0, 0.003);
    CHECK(bBlock < 0.01);
    CHECK_NEAR(vMean, 1.0, 0.006);
    CHECK(vBlock < 0.015);
}

TEST(vcm_sees_a_lens_caustic_through_a_glass_plate) {
    // As sppm_sees_a_caustic_through_a_glass_plate_and_in_a_mirror (test_references.cpp): the caustic
    // of a lens on the floor seen through a glass plate is the direct view times the plate's Fresnel
    // transmittance. The eye looks through glass at a point lit through glass: merging's case.
    if (!gpuHas("vcm")) return;
    auto scene = [](const std::string& extra, const Vec3& eye) {
        char text[2000];
        std::snprintf(text, sizeof text, R"(
            units = mm
            world { sun { elevation = 90deg  azimuth = 0deg  radiance = 1 } }
            material floor { type = diffuse  reflectance = 0.5 }
            body Floor { type = sheet  size = (300, 300)  position = (0, 0, 0)  material = floor }
            body Lens { type = lens  medium = N-BK7  focal = 100  form = bi  diameter = 40  thickness = 8  rim = black
                        position = (0, 0, 100)  axis = (0, 0, 1) }
            %s
            observer Eye { position = (%g, %g, %g)  look_at = (0, 0, 0)  up = (0, 0, 1)  fov = 7deg  pupil = 2
                           focus = 335  resolution = (48, 48) }
            render { detector = Eye }
        )", extra.c_str(), eye.x, eye.y, eye.z);
        return loadSceneFromString(text);
    };
    const Vec3 eye{268.3, 0, 150};
    const Vec3 dir = normalize(Vec3(0, 0, 0) - eye);
    const Vec3 at = eye + dir * 120;
    char plate[300];
    std::snprintf(plate, sizeof plate,
                  "body Plate { type = box  size = (60, 60, 5)  position = (%g, %g, %g)  axis = (%g, %g, %g)  medium = N-BK7 }", at.x,
                  at.y, at.z, dir.x, dir.y, dir.z);
    const double direct = render(scene("", eye), "vcm", 2048).meanY();
    const double through = render(scene(plate, eye), "vcm", 2048).meanY();
    Medium glass;
    CHECK(catalogMedium("N-BK7", glass));
    double num = 0, den = 0;
    for (double l = LambdaMin; l <= LambdaMax; l += 1) {
        const double n = glass.n(l) / 1.000277, R = sqr((n - 1) / (n + 1));
        num += (1 - R) * (1 - R) / (1 - R * R) * cieY(l);
        den += cieY(l);
    }
    std::fprintf(stderr, "    through the plate: %.4f of the direct view (Fresnel: %.4f)\n", through / direct, num / den);
    CHECK_NEAR(through / direct, num / den, 0.015);
}

TEST(vcm_keeps_a_glossy_reflection_as_camera_paths_do) {
    // benchmarks/glossy_reflection.owe: a silver plate of GGX α 0.04 reflecting a sunlit checkered
    // wall. VCM must agree with the path tracer, and be less noisy than it at the same sample count.
    if (!gpuHas("vcm")) return;
    const Scene sc = loadScene("benchmarks/glossy_reflection.owe");
    const Image ref = render(sc, "path", 1024, 7);
    auto relMse = [&](const Image& a) {
        double mean = ref.meanY(), s = 0;
        for (size_t i = 1; i < a.xyz.size(); i += 3) s += sqr(a.xyz[i] - ref.xyz[i]) / (sqr(ref.xyz[i]) + 0.01 * mean * mean);
        return s / double(a.xyz.size() / 3);
    };
    const Image vcm = render(sc, "vcm", 64), path = render(sc, "path", 64);
    const double eV = relMse(vcm), eP = relMse(path);
    std::fprintf(stderr, "    at 64 samples against path × 1024: vcm relMSE %.4g, path %.4g; vcm mean %.4f\n", eV, eP,
                 vcm.meanY() / ref.meanY());
    CHECK(eV < eP);
    CHECK_NEAR(vcm.meanY() / ref.meanY(), 1.0, 0.015);
}

// Resolving a preview must neither consume nor duplicate the persistent device film.
// Exercise every accumulation route, including direct/medium and photon state lifetimes.
TEST(gpu_film_resolve_is_idempotent_across_passes) {
    if (!gpuHas("vcm")) return;
    Scene sc = room();
    sc.detectors[0]->width = sc.detectors[0]->height = 16;
    for (const char* integrator : {"path", "light", "hybrid", "sppm", "bdpt", "vcm"}) {
        RenderSettings rs = sc.render;
        rs.backend = "gpu";
        rs.integrator = integrator;
        rs.maxDepth = 6;
        auto renderer = makeRenderer(sc, 0, rs);
        for (double value : renderer->resolve().xyz) CHECK(value == 0);
        for (int spp : {3, 5}) {
            renderer->runPass(spp);
            const Image first = renderer->resolve();
            const auto paths = renderer->stats().paths;
            CHECK(first.xyz == renderer->resolve().xyz);
            CHECK(renderer->stats().paths == paths);
            CHECK(first.meanY() > 0);
            for (double value : first.xyz) CHECK(std::isfinite(value));
        }
        CHECK(renderer->passes() == 2);
        CHECK(renderer->samplesPerPixel() == 8);
    }
}
