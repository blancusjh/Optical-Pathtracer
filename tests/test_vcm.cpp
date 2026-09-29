// Bidirectional path tracing and vertex connection and merging on the GPU (shaders/vcm.slang). BDPT
// is unbiased: it must draw the path tracer's image. VCM adds merging, consistent (its bias shrinks
// with the radius): it must draw the same image, and also what path tracing cannot, a lens caustic
// seen through a glass plate. On a glossy reflection, where a photon map gathered at the glossy
// surface is poor, it must stay as good as camera paths.
#include <cmath>
#include <cstdio>

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
