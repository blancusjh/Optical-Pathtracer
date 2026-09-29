// Radiative transport against analytic limits.
#include "analytic.hpp"
#include "check.hpp"
#include "owe/scene/builders.hpp"
#include "owe/transport/optics.hpp"
#include "owe/backends/registry.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/transport/transport.hpp"

#include <array>

using namespace owe;

namespace {
Image renderScene(const Scene& sc, int det, int spp, const std::string& integrator = "path", int threads = 0,
                  uint64_t seed = 1) {
    RenderSettings rs;
    rs.spp = spp;
    rs.integrator = integrator;
    rs.threads = threads;
    rs.seed = seed;
    rs.backend = referenceBackend().name();  // the reference transport, every integrator
    auto R = makeRenderer(sc, det, rs);
    R->runPass(spp);
    CHECK(R->stats().inconsistencies == 0);
    CHECK(R->stats().leaks == 0);
    return R->resolve();
}
// Per-block means and standard errors over independent seeds (4×3 blocks of a 24×18 image).
struct BlockStats {
    double mean[12] = {}, se[12] = {};
    double imageMean = 0, imageSe = 0;
};
BlockStats blockStats(const Scene& sc, int det, int spp, const std::string& integrator, int seeds) {
    std::vector<std::array<double, 13>> runs;
    for (int k = 0; k < seeds; ++k) {
        Image img = renderScene(sc, det, spp, integrator, 0, 100 + uint64_t(k));
        std::array<double, 13> v{};
        for (int y = 0; y < 18; ++y)
            for (int x = 0; x < 24; ++x) v[(y / 6) * 4 + x / 6] += img.at(x, y).y;
        v[12] = img.meanY();
        runs.push_back(v);
    }
    BlockStats b;
    for (int i = 0; i < 13; ++i) {
        double m = 0, q = 0;
        for (auto& r : runs) m += r[i] / seeds;
        for (auto& r : runs) q += sqr(r[i] - m) / (seeds - 1);
        double se = std::sqrt(q / seeds);
        if (i < 12) { b.mean[i] = m; b.se[i] = se; } else { b.imageMean = m; b.imageSe = se; }
    }
    return b;
}
// Two unbiased estimators of the same image must agree: the image mean within 4 combined
// standard errors, and the 12 blocks as a sample (few seeds make block z-scores Student-t, so one
// block past 4 is chance; a bias shows in many blocks and in the sum of squares).
void checkAgreement(const BlockStats& a, const BlockStats& b) {
    CHECK_NEAR(a.imageMean, b.imageMean, 4 * std::hypot(a.imageSe, b.imageSe) + 1e-12);
    int beyond = 0;
    double chi2 = 0;
    for (int i = 0; i < 12; ++i) {
        double z = (a.mean[i] - b.mean[i]) / (std::hypot(a.se[i], b.se[i]) + 1e-300);
        beyond += std::abs(z) > 4;
        chi2 += z * z;
    }
    CHECK(beyond <= 1);
    CHECK(chi2 / 12 < 3);
}
}  // namespace

TEST(beer_lambert_attenuation_through_index_matched_slab) {
    Scene sc;
    Medium ink;
    ink.name = "ink";
    ink.absorption = Spectrum::constant(35.0);  // 1/m
    ink.index = IndexModel::ciddorAir();
    sc.world.addMedium(ink);
    BodyMaterial m;
    m.transparent = true;
    m.medium = "ink";
    m.optics = sc.world.nullOptics();
    double d = 0.04;
    buildBox(sc.world, "cell", {1, 1, d}, m, -1, Transform{});
    sc.world.build();
    Tracer tr(sc);
    Rng rng(1, 1);
    PathRecord r = tr.walk({{0, 0, -1}, {0, 0, 1}}, 0, 550, WalkMode::Stochastic, rng);
    CHECK(r.v.back().event == EventKind::Escape);
    // Origins are displaced ~1e-9 m off each crossed surface; the path length error is of that order.
    CHECK_NEAR(r.v.back().beta, std::exp(-35.0 * d), 1e-7);
}

TEST(glass_plate_transmission_with_multiple_reflections) {
    // Incoherent slab: T_total = (1−R)² τ / (1 − R² τ²), τ = exp(−α d), at normal incidence.
    Scene sc;
    Medium g;
    catalogMedium("N-BK7", g);
    g.name = "absorbing-glass";
    g.absorption = Spectrum::constant(20.0);
    sc.world.addMedium(g);
    BodyMaterial m;
    m.transparent = true;
    m.medium = "absorbing-glass";
    m.optics = sc.world.dielectricOptics();
    double d = 0.03, lambda = 550;
    buildBox(sc.world, "plate", {2, 2, d}, m, -1, Transform{});
    sc.world.build();
    double n = sc.world.media()[sc.world.medium("absorbing-glass")].n(lambda) / sc.world.media()[0].n(lambda);
    double R = sqr((n - 1) / (n + 1)), tau = std::exp(-20.0 * d);
    double expected = sqr(1 - R) * tau / (1 - R * R * tau * tau);
    Tracer tr(sc);
    double sum = 0;
    int N = 400000;
    for (int i = 0; i < N; ++i) {
        Rng rng(i, 5);
        PathRecord r = tr.walk({{0, 0, -1}, {0, 0, 1}}, 0, lambda, WalkMode::Stochastic, rng);
        if (r.v.back().event == EventKind::Escape && r.v.back().dOut.z > 0) sum += r.v.back().beta;
    }
    double est = sum / N;
    CHECK_NEAR(est, expected, 4 * std::sqrt(expected * (1 - expected) / N) + 1e-4);
}

TEST(white_furnace_closed_diffuse_sphere) {
    // Inside a closed sphere with albedo ρ and emitted radiance Le: L = Le / (1 − ρ).
    for (double rho : {0.0, 0.5, 0.8}) {
        Scene sc;
        SurfaceOptics o;
        o.type = SurfaceType::Diffuse;
        o.reflectance = Spectrum::constant(rho);
        BodyMaterial m;
        m.optics = sc.world.addOptics(o);
        Emission e;
        e.radiance = Spectrum::constant(1.0);
        e.front = false;
        e.back = true;
        m.emission = sc.world.addEmission(e);
        // Opaque interior would be wrong here: the observer lives inside, so keep the interior as air.
        int b = sc.world.addBody("shell", "sphere", -1, Transform{});
        uint32_t inside = sc.world.addRegion("shell.inside", sc.world.medium("air"), b);
        sc.world.addBoundary(b, "surface", std::make_shared<SphereShape>(1.0), Transform{}, kOutside, inside, m.optics,
                             m.emission);
        auto obs = std::make_unique<IdealObserver>();
        obs->name = "eye";
        obs->width = obs->height = 16;
        obs->position = {0.2, 0.1, -0.3};
        obs->lookAt = {1, 0.3, 0};
        obs->fovY = radians(70);
        sc.detectors.push_back(std::move(obs));
        sc.build();
        CHECK(sc.detectors[0]->region == inside);
        Image img = renderScene(sc, 0, 256);
        CHECK_NEAR(img.meanY(), 1.0 / (1 - rho), 0.01 / (1 - rho));
    }
}

TEST(dielectric_sphere_preserves_uniform_radiance) {
    // A lossless dielectric in a uniform radiance field is invisible (radiance conservation, n² law).
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
    Image img = renderScene(sc, 0, 256);
    CHECK_NEAR(img.meanY(), 1.0, 0.004);
    // Pixels are noisy in Y (one wavelength per path) but unbiased: the disk region alone must average to 1.
    double inside = 0;
    int n = 0;
    for (int y = 6; y < 18; ++y)
        for (int x = 6; x < 18; ++x) { inside += img.at(x, y).y; ++n; }
    CHECK_NEAR(inside / n, 1.0, 0.01);
}

TEST(underwater_radiance_obeys_n_squared_law) {
    // Looking straight up from under a flat water surface at a uniform sky L:
    // L_water = T(0) · (n_w/n_air)² · L.
    Scene sc;
    sc.world.env.skyModel = Environment::Sky::Uniform;
    sc.world.env.zenith = Spectrum::constant(1.0);
    int b = sc.world.addBody("lake", "box", -1, Transform{});
    Medium water;
    catalogMedium("water", water);
    water.absorption = Spectrum::constant(0);
    water.name = "clear-water";
    uint32_t wr = sc.world.addRegion("lake.water", sc.world.addMedium(water), b);
    uint32_t rock = sc.world.addRegion("lake.rock", sc.world.medium("opaque"), b);
    sc.world.addBoundary(b, "surface", PlaneShape::rect(50, 50), Transform{}, kOutside, wr, sc.world.dielectricOptics());
    sc.world.addBoundary(b, "bed", PlaneShape::rect(50, 50), Transform::translate({0, 0, -5}), wr, rock,
                         sc.world.absorberOptics());
    for (int s = 0; s < 4; ++s) {
        Transform t = Transform::rotate({0, 0, 1}, s * Pi / 2) * Transform::translate({50, 0, -2.5}) *
                      Transform::rotate({0, 1, 0}, Pi / 2);
        sc.world.addBoundary(b, "wall" + std::to_string(s), PlaneShape::rect(2.5, 50), t, rock, wr, sc.world.absorberOptics());
    }
    auto obs = std::make_unique<IdealObserver>();
    obs->name = "diver";
    obs->width = obs->height = 3;
    obs->position = {0, 0, -1};
    obs->lookAt = {0, 0, 0};
    obs->up = {0, 1, 0};
    obs->fovY = radians(0.5);
    sc.detectors.push_back(std::move(obs));
    sc.build();
    CHECK(sc.detectors[0]->region == wr);
    // Single wavelength check through the transport kernel: every sample sees the same value.
    Tracer tr(sc);
    TransportStats st;
    double lambda = 550;
    double nw = sc.world.indexOf(wr, lambda), na = sc.world.indexOf(0, lambda);
    double T = 1 - fresnelDielectric(1.0, nw / na);
    double sum = 0;
    int N = 20000;
    for (int i = 0; i < N; ++i) {
        Rng rng(i, 9);
        Wavelengths wl = Wavelengths::single(lambda);
        sum += tr.radiance({{0, 0, -1}, {0, 0, 1}}, wr, wl, rng, st)[0];
    }
    double expected = T * sqr(nw / na);
    CHECK_NEAR(sum / N, expected, 4 * std::sqrt(expected * expected * (1 - T) / T / N) + 1e-3);
    CHECK(st.inconsistencies == 0);
}

TEST(path_and_light_tracing_agree_through_a_pinhole_and_a_lens_pupil) {
    std::string text = R"(
        units = m
        material floor { type = diffuse  reflectance = rgb(0.9, 0.6, 0.3) }
        body Floor { type = sheet  size = (4, 4)  material = floor }
        body Lamp  { type = sphere radius = 0.25  position = (0.4, -0.2, 1.2)  material = black  emission = blackbody(4000K, 5) }
        observer Pin   { position = (0, -2.2, 1.4)  look_at = (0, 0, 0)  fov = 40deg  resolution = (24, 18) }
        observer Thin  { position = (0, -2.2, 1.4)  look_at = (0, 0, 0)  fov = 40deg  resolution = (24, 18)
                         pupil = 60mm  focus = 2.6 m }
    )";
    Scene sc = loadSceneFromString(text);
    for (int det : {0, 1}) {
        // Two independent unbiased estimators of the same measurement must converge together.
        BlockStats pt = blockStats(sc, det, 512, "path", 6);
        BlockStats lt = blockStats(sc, det, 2000, "light", 6);
        checkAgreement(pt, lt);
        CHECK(pt.imageSe < 0.01 * pt.imageMean);  // the comparison is sharp enough to be meaningful
    }
}

TEST(renders_are_bitwise_reproducible) {
    std::string text = R"(
        units = m
        world { sky = gradient(zenith = rgb(0.3,0.5,1, luminance = 1), horizon = rgb(1,1,1)) }
        body Ball { type = sphere radius = 0.5 medium = N-BK7 position = (0, 0, 0.5) }
        body Ground { type = sheet size = (10, 10) material = white }
        observer Eye { position = (0, -3, 1) look_at = (0, 0, 0.5) resolution = (32, 24) }
    )";
    Scene sc = loadSceneFromString(text);
    for (const char* integ : {"path", "light"}) {
        Image a = renderScene(sc, 0, 16, integ, 1, 7);
        Image b = renderScene(sc, 0, 16, integ, 5, 7);
        Image c = renderScene(sc, 0, 16, integ, 2, 8);
        CHECK(a.xyz == b.xyz);
        CHECK(a.xyz != c.xyz);
    }
}

TEST(hybrid_partition_matches_path_tracing_on_a_caustic) {
    // A glass ball focuses a large lamp onto a floor. Pure path tracing and the hybrid
    // (light tracing owns eye→D→S⁺→light) are both unbiased and must agree.
    std::string text = R"(
        units = m
        material floor { type = diffuse  reflectance = 0.7 }
        body Floor { type = sheet  size = (3, 3)  material = floor }
        body Ball  { type = sphere radius = 0.3  position = (0, 0, 0.45)  medium = N-BK7 }
        body Lamp  { type = sphere radius = 0.35  position = (0.2, 0.1, 1.8)  material = black  emission = blackbody(5000K, 4) }
        observer Eye { position = (0, -2.0, 1.1)  look_at = (0, 0, 0.3)  fov = 45deg  resolution = (24, 18) }
    )";
    Scene sc = loadSceneFromString(text);
    BlockStats pt = blockStats(sc, 0, 1024, "path", 6);
    BlockStats hy = blockStats(sc, 0, 256, "hybrid", 6);
    checkAgreement(pt, hy);
    CHECK(pt.imageSe < 0.01 * pt.imageMean);
}

TEST(diffuse_screen_records_irradiance_and_reflects_toward_observers) {
    Scene sc = diskIrradianceScene(0.3, 0.5, true, 0.15, true);
    double irradiance = meanDiskIrradiance(2, 0.3, 0.5, 0.15);
    CHECK_REL(renderScene(sc, 0, 40000).meanY(), irradiance, 0.02);
    CHECK_REL(renderScene(sc, 0, 200000, "light").meanY(), irradiance, 0.02);
    auto eye = std::make_unique<IdealObserver>();
    eye->name = "see-screen";
    eye->position = {0, 0, 0.25}; eye->lookAt = {0, 0, 0}; eye->up = {0, 1, 0};
    eye->fovY = radians(5); eye->width = eye->height = 12;
    sc.detectors.push_back(std::move(eye)); sc.build();
    double expected = 0.7 * diskIrradiance(2, 0.3, 0.5, 0) / Pi;
    CHECK_REL(renderScene(sc, 1, 1024).meanY(), expected, 0.02);
    // Light tracing reaches this 2 cm patch with few particles (~8% noise per render): judge the
    // mean of several seeds by its own standard error, plus 1% for the patch's irradiance falloff.
    double s = 0, s2 = 0;
    const int seeds = 12;
    for (int k = 0; k < seeds; ++k) {
        double y = renderScene(sc, 1, 4096, "light", 0, 1 + uint64_t(k)).meanY();
        s += y;
        s2 += y * y;
    }
    double mean = s / seeds, se = std::sqrt(std::max(0.0, s2 / seeds - mean * mean) / (seeds - 1));
    CHECK(se < 0.04 * expected);
    CHECK_NEAR(mean, expected, 4 * se + 0.01 * expected);

    // The SAME illuminated patch has the same radiance throughout its front hemisphere.
    // A small field keeps the footprint near the centre even at 85 degrees from the normal.
    auto& moving = static_cast<IdealObserver&>(*sc.detectors[1]);
    moving.fovY = radians(0.1);
    for (double polar : {0., 45., 80., 85.}) {
        for (double azimuth : {0., 120., 240.}) {
            double a = radians(polar), b = radians(azimuth);
            moving.position = Vec3(std::sin(a) * std::cos(b), std::sin(a) * std::sin(b), std::cos(a)) * 0.25;
            moving.prepare(sc.world);
            CHECK_REL(renderScene(sc, 1, 1024).meanY(), expected, 0.02);
        }
    }
    moving.position = {0, 0, -0.25}; moving.prepare(sc.world);
    CHECK_NEAR(renderScene(sc, 1, 128).meanY(), 0, 0); // no transmission of the front projection
}

TEST(optical_bench_projection_is_visible_from_free_front_directions) {
    Scene sc = loadScene("scenes/optical_bench.owe");
    // Delete every observer/readout. Image formation must depend only on physical bodies.
    sc.detectors.clear();
    Tracer tracer(sc);
    auto radianceAt = [&](Vec3 patch, double angle) {
        Vec3 eye = patch + Vec3(std::sin(angle), -std::cos(angle), 0) * 0.05;
        double sum = 0;
        TransportStats stats;
        for (int i = 0; i < 30000; ++i) {
            Rng rng(i, 101);
            auto wl = Wavelengths::single(550);
            sum += tracer.radiance({eye, normalize(patch - eye)}, sc.world.locate(eye), wl, rng, stats)[0];
        }
        CHECK(stats.leaks == 0);
        CHECK(stats.inconsistencies == 0);
        return sum / 30000;
    };
    // The stem moves from x=-6 mm on the object to x=+6 mm on the screen (inversion).
    Vec3 bright{0.006, 0.4038, 0.006}, dark{-0.012, 0.4038, 0.010};
    double reference = radianceAt(bright, 0);
    CHECK(reference > 5 * radianceAt(dark, 0));
    for (double angle : {-85., -45., 45., 85.})
        CHECK_REL(radianceAt(bright, radians(angle)), reference, 1e-6);
    CHECK(radianceAt(bright, Pi) < reference * 0.1);
}

TEST(diffuse_screen_aim_sampling_retains_full_hemisphere_and_matching_pdf) {
    SurfaceOptics o; o.type = SurfaceType::Diffuse; o.reflectance = Spectrum::constant(0.7);
    o.sampleAimCenter = {0, 0, 1}; o.sampleAimNormal = {0, 0, 1};
    o.sampleAimRadius = 0.15; o.sampleAimShare = 0.95;
    Interface it; it.optics = &o; it.n = {0, 0, 1};
    Rng rng(812, 3); double sum = 0, sq = 0;
    const int N = 200000;
    for (int i = 0; i < N; ++i) {
        Draw3 u(rng); ScatterSample s;
        CHECK(sampleScatter(it, {0, 0, -1}, u.u1, u.u2, u.u3, TransportMode::Radiance, s));
        double pdf; double f = evalScatter(it, {0, 0, -1}, s.wi, TransportMode::Radiance, pdf);
        CHECK_REL(pdf, s.pdf, 1e-10);
        CHECK_NEAR(f, 0.7 * InvPi, 1e-12);
        sum += s.weight; sq += s.weight * s.weight;
    }
    double mean = sum / N, se = std::sqrt((sq / N - mean * mean) / (N - 1));
    CHECK_NEAR(mean, 0.7, 5 * se);
}
