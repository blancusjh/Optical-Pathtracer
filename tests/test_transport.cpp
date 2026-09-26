// Radiative transport against analytic limits.
#include "check.hpp"
#include "owe/builders.hpp"
#include "owe/optics.hpp"
#include "owe/render.hpp"
#include "owe/scene_loader.hpp"

using namespace owe;

namespace {
Image renderScene(const Scene& sc, int det, int spp, const std::string& integrator = "path", int threads = 0,
                  uint64_t seed = 1) {
    RenderSettings rs;
    rs.spp = spp;
    rs.integrator = integrator;
    rs.threads = threads;
    rs.seed = seed;
    ProgressiveRenderer R(sc, det, rs);
    R.runPass(spp);
    CHECK(R.stats().inconsistencies == 0);
    CHECK(R.stats().leaks == 0);
    return R.resolve();
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
        sum += tr.radiance({{0, 0, -1}, {0, 0, 1}}, wr, lambda, rng, st);
    }
    double expected = T * sqr(nw / na);
    CHECK_NEAR(sum / N, expected, 4 * std::sqrt(expected * expected * (1 - T) / T / N) + 1e-3);
    CHECK(st.inconsistencies == 0);
}

namespace {
// Irradiance at radial offset r on a plane parallel to a Lambertian disk (radius a, height h, radiance L).
double diskIrradiance(double L, double a, double h, double r) {
    double s = h * h + r * r + a * a;
    return Pi * L / 2 * (1 - (h * h + r * r - a * a) / std::sqrt(s * s - 4 * r * r * a * a));
}
double meanDiskIrradiance(double L, double a, double h, double half) {
    int N = 200;
    double sum = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double x = -half + (i + 0.5) * 2 * half / N, y = -half + (j + 0.5) * 2 * half / N;
            sum += diskIrradiance(L, a, h, std::sqrt(x * x + y * y));
        }
    return sum / (N * N);
}
// Disk emitter of radius a at height h above an upward-facing square sensor (half-size `half`).
Scene diskIrradianceScene(double a, double h, bool aimed, double half) {
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
    so.type = SurfaceType::Detector;
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
    }
    sc.detectors.push_back(std::move(d));
    sc.build();
    return sc;
}
}  // namespace

TEST(irradiance_from_lambertian_disk_path_and_light_tracing) {
    double a = 0.3, h = 0.5, L = 2.0, half = 0.15;
    CHECK_NEAR(diskIrradiance(L, a, h, 0), Pi * L * a * a / (a * a + h * h), 1e-12);
    double expected = meanDiskIrradiance(L, a, h, half);
    for (bool aimed : {false, true}) {
        Scene sc = diskIrradianceScene(a, h, aimed, half);
        Image pt = renderScene(sc, 0, 40000);
        CHECK_NEAR(pt.meanY(), expected, 0.01 * expected);
    }
    Scene sc = diskIrradianceScene(a, h, false, half);
    Image lt = renderScene(sc, 0, 400000, "light");
    CHECK_NEAR(lt.meanY(), expected, 0.01 * expected);
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
        Image pt = renderScene(sc, det, 3072);
        Image lt = renderScene(sc, det, 12000, "light");
        CHECK_NEAR(lt.meanY(), pt.meanY(), 0.02 * pt.meanY());
        // Coarse spatial agreement (4×3 blocks).
        for (int by = 0; by < 3; ++by)
            for (int bx = 0; bx < 4; ++bx) {
                double sp = 0, sl = 0;
                for (int y = by * 6; y < by * 6 + 6; ++y)
                    for (int x = bx * 6; x < bx * 6 + 6; ++x) { sp += pt.at(x, y).y; sl += lt.at(x, y).y; }
                CHECK_NEAR(sl, sp, 0.08 * sp + 1e-9);  // ≈3σ at these sample counts
            }
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
