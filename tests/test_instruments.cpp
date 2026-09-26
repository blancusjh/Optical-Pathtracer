// Lenses and instruments: paraxial optics, real rays through physical bodies.
#include "check.hpp"
#include "owe/analysis.hpp"
#include "owe/builders.hpp"
#include "owe/optics.hpp"
#include "owe/render.hpp"
#include "owe/scene_loader.hpp"

using namespace owe;

namespace {
double relIndex(const std::string& glass, double l) {
    Medium g, air;
    catalogMedium(glass, g);
    catalogMedium("air", air);
    return g.n(l) / air.n(l);
}
// Thick-lens lensmaker formula.
double thickLensEFL(double n, double R1, double R2, double t) {
    double P = (n - 1) * (1 / R1 - 1 / R2 + (n - 1) * t / (n * R1 * R2));
    return 1 / P;
}
const char* kSinglet = R"(
name biconvex singlet
units mm
S1   50.0   8.0  N-BK7  12.5
S2  -50.0  40.0  air    12.5
)";
const char* kDoublet = R"(
name cemented achromat f=100
units mm
S1    44.78  6.0  N-BK7  12.5
S2   -44.78  2.5  F2     12.5
S3  -810.8  95.0  air    12.5
)";
}  // namespace

TEST(paraxial_efl_matches_thick_lens_formula) {
    Prescription p = parsePrescription(kSinglet);
    for (double l : {LambdaF, LambdaD, LambdaC}) {
        Paraxial px = paraxialAnalysis(p, l, catalogIndex());
        double n = relIndex("N-BK7", l);
        double f = thickLensEFL(n, 0.05, -0.05, 0.008);
        CHECK_REL(px.efl, f, 1e-12);
        // Back focal distance: f (1 − (n−1) t / (n R1)).
        CHECK_REL(px.bfl, f * (1 - (n - 1) * 0.008 / (n * 0.05)), 1e-12);
    }
}

TEST(designed_simple_lenses_have_requested_focal_length) {
    for (const char* form : {"bi", "plano"})
        for (double f : {0.05, 0.2, -0.08}) {
            LensSpec s = designSimpleLens(f, "N-BK7", 0.025, 0.004, form);
            Prescription p;
            PrescriptionSurface a, b;
            a.R = s.surfaces[0].R; a.t = s.thickness[0]; a.medium = "N-BK7"; a.sd = 0.0125;
            b.R = s.surfaces[1].R; b.t = 0.1; b.medium = "air"; b.sd = 0.0125;
            p.surfaces = {a, b};
            CHECK_REL(paraxialAnalysis(p, LambdaD, catalogIndex()).efl, f, 1e-12);
        }
}

TEST(near_axis_real_ray_reaches_paraxial_focus) {
    // Non-sequential trace through the lens *body* converges on the paraxial prediction.
    Prescription p = parsePrescription(kSinglet);
    Scene sc;
    BuiltInstrument bi = buildPrescription(sc.world, p, "L", -1, Transform{}, 0, catalogIndex());
    sc.world.build();
    Tracer tr(sc);
    Rng rng(1, 1);
    for (double l : {LambdaF, LambdaD, LambdaC}) {
        Paraxial px = paraxialAnalysis(p, l, catalogIndex());
        // Height 1 µm: third-order aberration (∝ h²) is then ~1e-11 m.
        PathRecord r = tr.walk({{0, 1e-6, -0.05}, {0, 0, 1}}, 0, l, WalkMode::PrimaryTransmission, rng);
        CHECK(r.v.size() == 4);  // start, S1, S2, escape
        const PathVertex& last = r.v[2];
        double zCross = last.p.z - last.p.y * last.dOut.z / last.dOut.y;
        CHECK_NEAR(zCross, bi.lastVertexZ + px.bfl, 5e-11);
    }
}

TEST(instruments_far_from_the_origin_keep_their_precision) {
    // The same lens 3 km from the world origin must focus identically (extreme spatial scale).
    Prescription p = parsePrescription(kSinglet);
    Vec3 far{3000.0, -2000.0, 850.0};
    Transform place = Transform::translate(far) * Transform::alignZ(normalize(Vec3(0.3, 0.8, 0.2)));
    Scene sc;
    BuiltInstrument bi = buildPrescription(sc.world, p, "L", -1, place, 0, catalogIndex());
    sc.world.build();
    Tracer tr(sc);
    Rng rng(1, 1);
    Paraxial px = paraxialAnalysis(p, LambdaD, catalogIndex());
    Vec3 axis = place.vector({0, 0, 1}), lateral = place.vector({0, 1, 0});
    PathRecord r = tr.walk({place.point({0, 2e-6, -0.05}), axis}, 0, LambdaD, WalkMode::PrimaryTransmission, rng);
    CHECK(r.v.size() == 4);
    const PathVertex& last = r.v[2];
    // Axis crossing in the lens frame.
    Vec3 q = place.inverse().point(last.p), d = place.inverse().vector(last.dOut);
    double zCross = q.z - q.y * d.z / d.y;
    CHECK_NEAR(zCross, bi.lastVertexZ + px.bfl, 2e-8);
    (void)lateral;
    // Edge rays still meet the rim rather than slipping past it.
    int rimHits = 0;
    for (int i = 0; i < 2000; ++i) {
        Rng g(i, 4);
        double a = 2 * Pi * g.uniform();
        Vec3 dl = normalize(Vec3(2 * std::cos(a), 2 * std::sin(a), 1));
        Vec3 entry{0.010 * std::cos(a), 0.010 * std::sin(a), 0.05 - std::sqrt(0.05 * 0.05 - 0.010 * 0.010)};
        PathRecord e = tr.walk({place.point(entry - dl * 0.01), place.vector(dl)}, 0, LambdaD, WalkMode::PrimaryTransmission, g);
        for (auto& v : e.v) rimHits += v.boundary != kNone && sc.world.boundaryLabel(v.boundary) == "L.L1.rim1";
    }
    CHECK(rimHits > 1900);
    // Rendering through it produces no region inconsistencies.
    auto obs = std::make_unique<IdealObserver>();
    obs->name = "eye";
    obs->width = obs->height = 24;
    obs->position = place.point({0, 0, 0.1});
    obs->lookAt = place.point({0, 0, -1});
    obs->fovY = radians(60);
    obs->pupilRadius = 0.002;
    sc.detectors.push_back(std::move(obs));
    sc.world.env.skyModel = Environment::Sky::Uniform;
    sc.world.env.zenith = Spectrum::constant(1);
    sc.build();
    RenderSettings rs;
    rs.spp = 64;
    ProgressiveRenderer R(sc, 0, rs);
    R.runPass(64);
    CHECK(R.stats().inconsistencies == 0);
}

TEST(spherical_and_chromatic_aberration_trends) {
    Prescription p = parsePrescription(kSinglet);
    LensReport rep = analyzeLens(p, {0}, {LambdaF, LambdaD, LambdaC}, 6);
    // Positive spherical singlet: undercorrected, marginal focus falls short, growing with zone².
    CHECK(rep.lsa.size() == 10);
    CHECK(rep.lsa.back().second < 0);
    CHECK(std::abs(rep.lsa.back().second) > 3 * std::abs(rep.lsa[4].second));
    CHECK_NEAR(rep.lsa.back().second / rep.lsa[4].second, sqr(rep.lsa.back().first / rep.lsa[4].first), 0.6);
    // Normal dispersion: blue focuses closer than red.
    CHECK(rep.chromaticFocalShift < 0);
    // An achromatic doublet of similar power reduces the F–C focal shift by an order of magnitude.
    LensReport ach = analyzeLens(parsePrescription(kDoublet), {0}, {LambdaF, LambdaD, LambdaC}, 6);
    double singletShiftPerPower = std::abs(rep.chromaticFocalShift) / rep.paraxial.efl;
    double achShiftPerPower = std::abs(ach.chromaticFocalShift) / ach.paraxial.efl;
    CHECK(achShiftPerPower < 0.1 * singletShiftPerPower);
    for (auto& s : ach.spots) CHECK(s.arrived == s.launched);
}

TEST(afocal_keplerian_telescope) {
    const char* kepler = R"(
units mm
O1   200.0   6.0  N-BK7  20
O2  -200.0 250.0  air    20
E1    25.0   5.0  N-BK7   8
E2   -25.0  30.0  air     8
)";
    Prescription p = parsePrescription(kepler);
    double gap = solveAfocal(p, LambdaD, catalogIndex());
    Paraxial px = paraxialAnalysis(p, LambdaD, catalogIndex());
    CHECK(px.afocal);
    double fo = thickLensEFL(relIndex("N-BK7", LambdaD), 0.2, -0.2, 0.006);
    double fe = thickLensEFL(relIndex("N-BK7", LambdaD), 0.025, -0.025, 0.005);
    CHECK_REL(px.angularMagnification, -fo / fe, 1e-9);
    CHECK(gap > 0.9 * (fo + fe) && gap < 1.1 * (fo + fe));
    // Exit pupil (Ramsden disc) lies behind the eyepiece with radius D/M.
    CHECK(px.exitPupilZ > px.length);
    CHECK_REL(px.exitPupilRadius, 0.020 / std::abs(px.angularMagnification), 0.02);
}

TEST(lens_rim_is_physical_matter) {
    // A ray entering the front face near the edge at a steep angle leaves through the rim.
    Scene sc;
    LensSpec s;
    SurfaceSpec a, b;
    a.R = 0.05; a.semiDiameter = 0.0125;
    b.R = -0.05; b.semiDiameter = 0.0125;
    s.surfaces = {a, b};
    s.media = {"N-BK7"};
    s.thickness = {0.008};
    s.rim = LensSpec::Rim::Polished;
    buildLens(sc.world, "L", s, -1, Transform{});
    sc.world.build();
    Tracer tr(sc);
    Rng rng(1, 1);
    Vec3 d = normalize(Vec3(0, 2, 1));
    Vec3 entry{0, 0.010, 0.05 - std::sqrt(0.05 * 0.05 - 0.010 * 0.010)};
    PathRecord r = tr.walk({entry - d * 0.01, d}, 0, 550, WalkMode::PrimaryTransmission, rng);
    CHECK(r.v.size() >= 3);
    CHECK(sc.world.boundaryLabel(r.v[1].boundary) == "L.S1");
    CHECK(sc.world.boundaryLabel(r.v[2].boundary) == "L.rim1");
}

TEST(ghost_reflections_emerge_from_non_sequential_transport) {
    // Double reflection inside a plate returns light parallel to the direct beam.
    Scene sc;
    BodyMaterial g;
    g.transparent = true;
    g.medium = "N-BK7";
    g.optics = sc.world.dielectricOptics();
    buildBox(sc.world, "plate", {2, 2, 0.01}, g, -1, Transform{});
    sc.world.build();
    Tracer tr(sc);
    int ghosts = 0, N = 100000;
    for (int i = 0; i < N; ++i) {
        Rng rng(i, 2);
        PathRecord r = tr.walk({{0, 0, -1}, normalize(Vec3(0, 0.05, 1))}, 0, 550, WalkMode::Stochastic, rng);
        int refl = 0;
        for (auto& v : r.v) refl += v.event == EventKind::Reflect;
        if (refl == 2 && r.v.back().dOut.z > 0) {
            ++ghosts;
            CHECK_NEAR(dot(r.v.back().dOut, normalize(Vec3(0, 0.05, 1))), 1.0, 1e-12);
        }
    }
    double R = fresnelDielectric(std::cos(std::atan(0.05)), relIndex("N-BK7", 550));
    double expected = (1 - R) * (1 - R) * R * R;
    CHECK_NEAR(double(ghosts) / N, expected, 5 * std::sqrt(expected / N) + 2e-5);
}

TEST(physical_camera_forms_an_upright_image) {
    // A bright disk up-right of the optical axis must appear up-right in the camera readout.
    const char* scene = R"(
units = m
world { sky = uniform(rgb(0.02, 0.02, 0.02)) }
body Lamp { type = disk radius = 0.12 position = (0.35, 6, 0.25) axis = (0, -1, 0) material = black
            emission = blackbody(5000K, 20) }
camera Cam { lens = "lenses/achromat_100mm.lens" sensor = (24mm, 16mm) resolution = (36, 24)
             position = (0, 0, 0) look_at = (0, 6, 0) focus = 6 m }
)";
    Scene sc = loadSceneFromString(scene, ".");
    RenderSettings rs;
    rs.spp = 96;
    ProgressiveRenderer R(sc, 0, rs);
    R.runPass(96);
    CHECK(R.stats().inconsistencies == 0);
    Image img = R.resolve();
    double quad[2][2] = {{0, 0}, {0, 0}};
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) quad[y < img.height / 2][x >= img.width / 2] += img.at(x, y).y;
    // quad[top][right]
    CHECK(quad[1][1] > 5 * (quad[0][0] + quad[0][1] + quad[1][0]));
}
