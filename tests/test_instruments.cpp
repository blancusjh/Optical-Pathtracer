// Lenses and instruments: paraxial optics, real rays through physical bodies.
#include "check.hpp"
#include "owe/analysis/analysis.hpp"
#include "owe/backends/registry.hpp"
#include "owe/scene/builders.hpp"
#include "owe/transport/optics.hpp"
#include "owe/loader/scene_loader.hpp"

using namespace owe;

TEST(observatory_free_eye_planet_signal_dominates_room_reflection) {
    Scene sc = loadScene("scenes/the_observatory.owe");
    sc.useDetector(sc.findDetector("Room"));
    IdealObserver eye = dynamic_cast<IdealObserver&>(*sc.detectors[sc.findDetector("Room")]);
    eye.width = eye.height = 100;
    Tracer tracer(sc); TransportStats stats;
    for (const char* object : {"Saturn", "Jupiter", "Moon"}) {
        auto& target = dynamic_cast<IdealObserver&>(*sc.detectors[sc.findDetector(std::string(object) + "Eyepiece")]);
        // Move the existing room eye; do not adopt the bookmark's aperture, FOV,
        // focus, exposure or sun-sampling settings. Also displace it from the nominal pupil.
        eye.position = target.position + target.right() * 0.0002;
        eye.lookAt = eye.position + target.forward(); eye.prepare(sc.world);
        double half = 50 * std::tan(radians(0.4)) / std::tan(eye.fovY / 2);
        double planet = 0, other = 0;
        for (int i = 0; i < 100000; ++i) {
            Rng rng(i, 27); Ray ray; double weight;
            eye.generate(50 + half * (2 * rng.uniform() - 1), 50 + half * (2 * rng.uniform() - 1), rng, ray, weight);
            auto wl = Wavelengths::single(550); PathRecord rec;
            tracer.radiance(ray, eye.region, wl, rng, stats, &rec);
            for (const auto& c : rec.c) {
                bool fromPlanet = false;
                for (size_t j = 1; j <= c.vertex && j < rec.v.size(); ++j) {
                    if (rec.v[j].event != EventKind::Diffuse) continue;
                    fromPlanet = sc.world.boundaryLabel(rec.v[j].boundary).rfind(object, 0) == 0;
                    break;
                }
                (fromPlanet ? planet : other) += c.value[0] * weight;
            }
        }
        std::fprintf(stderr, "    free %s eye: planet %.8g, other %.8g\n", object, planet / 100000, other / 100000);
        CHECK(planet > 0);
        CHECK(planet > 20 * other);
    }
}

TEST(observatory_window_transmits_candlelight_to_exterior_ground) {
    double signal[2]{};
    for (int sealed = 0; sealed < 2; ++sealed) {
        Scene sc = loadSceneWithEdits("scenes/the_observatory.owe", sealed ?
            std::vector<std::string>{"Wall.openings=[]"} : std::vector<std::string>{});
        Tracer tracer(sc); TransportStats stats; SurfaceHit ground;
        CHECK(sc.world.intersect({{-6 * std::sin(radians(70)), -6 * std::cos(radians(70)), 1000}, {0, 0, -1}}, 2000, ground));
        Ray ray{ground.p + Vec3(0, 0, 0.01), {0, 0, -1}};
        auto region = sc.world.locate(ray.o);
        for (int i = 0; i < 30000; ++i) {
            Rng rng(i, 17); auto wl = Wavelengths::single(550); PathRecord rec;
            tracer.radiance(ray, region, wl, rng, stats, &rec);
            for (const auto& c : rec.c) if (c.source.find("Flame") != std::string::npos) signal[sealed] += c.value[0];
        }
    }
    std::fprintf(stderr, "    exterior candlelight: window open %.9g, sealed %.9g\n", signal[0] / 30000, signal[1] / 30000);
    CHECK(signal[0] / 30000 > 1e-9);
    CHECK(signal[0] > 1000 * signal[1]);
}

TEST(pinhole_has_finite_geometric_blur_without_a_lens_focus) {
    for (double objectDistance : {12., 24.}) for (double screenDistance : {2., 3.9, 6.})
        for (double diameter : {0.006, 0.03, 0.3}) {
            Scene sc;
            buildStop(sc.world, "hole", diameter / 2, 10, -1, Transform{});
            int body = sc.world.addBody("screen", "sensor", -1, Transform::translate({0, 0, screenDistance}));
            SurfaceOptics detector; detector.type = SurfaceType::Detector;
            sc.world.addBoundary(body, "pixels", PlaneShape::rect(2, 2), Transform{}, kOutside, kOutside,
                                 sc.world.addOptics(detector));
            sc.build(); Tracer tracer(sc); Rng rng(19, 1);
            Vec3 object{0.2, 0, -objectDistance};
            Vec3 centre{-object.x * screenDistance / objectDistance, 0, screenDistance};
            double radius = diameter * 0.5 * (1 + screenDistance / objectDistance);
            double sumR2 = 0, maxR = 0;
            for (int i = 0; i < 4096; ++i) {
                Draw2 u(rng); Vec2 disk = sampleUniformDiskConcentric(u.u1, u.u2);
                Vec3 aperture{disk.x * diameter / 2, disk.y * diameter / 2, 0};
                auto path = tracer.walk({object, normalize(aperture - object)}, 0, 550,
                                        WalkMode::PrimaryTransmission, rng);
                CHECK(path.v.back().event == EventKind::Detect);
                double r = length(path.v.back().p - centre);
                CHECK(r <= radius * (1 + 1e-9));
                maxR = std::max(maxR, r); sumR2 += r * r;
            }
            CHECK_REL(maxR, radius, 0.005);
            CHECK_REL(std::sqrt(sumR2 / 4096), radius / std::sqrt(2), 0.02);
        }
}

TEST(moon_eyepiece_saved_glass_view_separates_reflection_from_transmission) {
    // Regression for the reported freely navigated view. Match the scene loaded by that
    // viewer (before the eyecups were added), rather than substituting a Saturn bookmark.
    auto sc = loadSceneWithEdits("scenes/the_observatory.owe",
        {"GreatRefractor.eyecup=false", "Refractor2.eyecup=false", "MoonScope.eyecup=false"});
    IdealObserver eye;
    eye.position = {-3.020303470341136, -1.096159549177377, 47.843021349537416};
    eye.lookAt = {-9.5845774247424593, -4.6348822980427569, 51.119390082637246};
    eye.up = {0.15135342771703073, 0.50563757519874331, 0.84936610626127584};
    eye.fovY = radians(80); eye.pupilRadius = 0.0025; eye.focusDistance = 3.5;
    eye.width = 80; eye.height = 50; eye.prepare(sc.world);
    const auto& reference = dynamic_cast<IdealObserver&>(*sc.detectors[sc.findDetector("MoonEyepiece")]);
    Vec3 offset = eye.position - reference.position;
    double axial = dot(offset, reference.forward());
    CHECK_NEAR(axial, 0.0188296038, 1e-9);
    CHECK_NEAR(length(offset - reference.forward() * axial), 0.0050488352, 1e-8);

    Tracer tracer(sc); Rng primaryRng(1, 1);
    auto primary = tracer.walk({eye.position, eye.forward()}, eye.region, 550,
                               WalkMode::PrimaryTransmission, primaryRng);
    CHECK(sc.world.boundaryLabel(primary.v.back().boundary) == "MoonScope.tube.wall");
    CHECK(primary.v.back().event == EventKind::Absorb);
    double moon[2]{}, reflectedRoom[2]{};
    for (int pose = 0; pose < 2; ++pose) {
        if (pose) {
            Vec3 direction = eye.lookAt - eye.position;
            eye.position = reference.position;
            eye.lookAt = eye.position + direction; // only translate; keep all eye settings
            eye.prepare(sc.world);
        }
        TransportStats stats;
        for (int i = 0; i < 100000; ++i) {
            Rng rng(i, 19); Ray ray; double weight;
            double x = eye.width * rng.uniform(), y = eye.height * rng.uniform();
            if (!eye.generate(x, y, rng, ray, weight)) continue;
            SurfaceHit first;
            if (!sc.world.intersect(ray, Inf, first) ||
                !sc.world.boundaryLabel(first.boundary).starts_with("MoonScope.L3.")) continue;
            auto wl = Wavelengths::single(550); PathRecord path;
            tracer.radiance(ray, eye.region, wl, rng, stats, &path);
            for (const auto& c : path.c) {
                bool reflected = false;
                for (size_t j = 1; j <= c.vertex && j < path.v.size(); ++j) {
                    const auto& v = path.v[j];
                    reflected |= v.event == EventKind::Reflect || v.event == EventKind::TIR;
                    if (v.event != EventKind::Diffuse) continue;
                    if (sc.world.boundaryLabel(v.boundary) == "Moon.surface") moon[pose] += c.value[0] * weight;
                    else {
                        CHECK(reflected);
                        reflectedRoom[pose] += c.value[0] * weight;
                    }
                    break;
                }
            }
        }
        CHECK(stats.leaks == 0);
        CHECK(stats.inconsistencies == 0);
    }
    CHECK(reflectedRoom[0] > 0);
    CHECK(moon[1] > 0);
    CHECK(moon[1] > 100 * moon[0]);
}

TEST(observatory_eyepiece_room_light_is_reflected_and_eyecup_reduces_it) {
    double planet[2]{}, room[2]{};
    for (int shield = 0; shield < 2; ++shield) {
        auto sc = loadSceneWithEdits("scenes/the_observatory.owe",
                                    {std::string("GreatRefractor.eyecup=") + (shield ? "true" : "false")});
        auto eye = dynamic_cast<IdealObserver&>(*sc.detectors[sc.findDetector("SaturnEyepiece")]);
        eye.pupilRadius = 0.0025; // the room observer's 5 mm pupil, not the small eyepiece preset
        eye.width = eye.height = 100; eye.prepare(sc.world);
        Tracer tracer(sc); TransportStats stats;
        for (int i = 0; i < 100000; ++i) {
            Rng rng(i, 19); Ray ray; double weight;
            double x = 40 + 20 * rng.uniform(), y = 40 + 20 * rng.uniform();
            if (!eye.generate(x, y, rng, ray, weight)) continue;
            auto wl = Wavelengths::single(550); PathRecord path;
            tracer.radiance(ray, eye.region, wl, rng, stats, &path);
            for (const auto& c : path.c) {
                bool reflected = false;
                for (size_t j = 1; j <= c.vertex && j < path.v.size(); ++j) {
                    const auto& v = path.v[j];
                    reflected |= v.event == EventKind::Reflect || v.event == EventKind::TIR;
                    if (v.event != EventKind::Diffuse) continue;
                    auto label = sc.world.boundaryLabel(v.boundary);
                    if (label.starts_with("Saturn")) planet[shield] += c.value[0] * weight;
                    else {
                        CHECK(reflected); // room geometry must not replace the transmitted celestial view
                        room[shield] += c.value[0] * weight;
                    }
                    break;
                }
            }
        }
        CHECK(stats.leaks == 0);
        CHECK(stats.inconsistencies == 0);
    }
    CHECK(planet[0] > 0 && room[0] > 0);
    CHECK_REL(planet[1], planet[0], 0.01);
    CHECK(room[1] < room[0] * 0.9);
}

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
    auto R = makeRenderer(sc, 0, rs);  // the reference backend
    R->runPass(64);
    CHECK(R->stats().inconsistencies == 0);
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
    auto R = makeRenderer(sc, 0, rs);  // the reference backend
    R->runPass(96);
    CHECK(R->stats().inconsistencies == 0);
    Image img = R->resolve();
    double quad[2][2] = {{0, 0}, {0, 0}};
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) quad[y < img.height / 2][x >= img.width / 2] += img.at(x, y).y;
    // quad[top][right]
    CHECK(quad[1][1] > 5 * (quad[0][0] + quad[0][1] + quad[1][0]));
}

TEST(camera_f_number_and_pupil_aim) {
    // f_number resizes the physical stop: the paraxial f-number of the built camera matches it.
    // Aiming sensor samples at the exit pupil (mixed with the whole rear opening) agrees with the
    // unaimed estimate within its noise. (A pupil-only aim is biased low by the ghost and veiling
    // light that leaves the rear element outside the pupil: ~1.8% at f/11 in the_temple.owe.)
    const char* scene = R"(
units = m
world { sky = uniform(rgb(0.5, 0.5, 0.5)) }
body Card { type = sheet size = (3, 3) position = (0.4, 5, 0.3) axis = (0, -1, 0) material = white
            emission = blackbody(5000K, 30) }
camera Cam { lens = "lenses/portrait_85mm.lens" sensor = (36mm, 24mm) resolution = (18, 12)
             position = (0, 0, 0) look_at = (0, 5, 0) focus = 5 m  f_number = 8 }
)";
    Scene sc = loadSceneFromString(scene, ".");
    CHECK(sc.notes["Cam"].find("f/8.00") != std::string::npos);
    auto* sensor = dynamic_cast<SurfaceSensor*>(sc.detectors[0].get());
    CHECK(sensor && sensor->focusShare > 0 && sensor->focusRadius < sensor->aimRadius);
    auto meanY = [&](double share, int spp, uint64_t seed) {
        sensor->focusShare = share;
        RenderSettings rs;
        rs.spp = spp;
        rs.seed = seed;
        auto R = makeRenderer(sc, 0, rs);  // the reference backend
        R->runPass(spp);
        return R->resolve().meanY();
    };
    double share = sensor->focusShare;
    // Independent estimates give the standard error of the unaimed (high-variance) sampler.
    std::vector<double> ref;
    for (uint64_t s = 1; s <= 4; ++s) ref.push_back(meanY(0, 1024, s));
    double m = 0, v = 0;
    for (double x : ref) m += x / ref.size();
    for (double x : ref) v += (x - m) * (x - m) / (ref.size() - 1);
    double aimed = meanY(share, 1024, 9);
    CHECK_NEAR(aimed, m, 5 * std::sqrt(v / ref.size()) + 1e-3 * m);
}

// An eye beside a telescope's small exit pupil focuses on the planet that part of its pupil receives,
// not on the black tube its pupil centre looks into (a free eye in the viewer, 2 mm off the axis).
TEST(accommodation_follows_the_light_the_pupil_receives) {
    Scene sc = loadScene("scenes/the_observatory.owe");
    auto& ep = dynamic_cast<IdealObserver&>(*sc.detectors[size_t(sc.findDetector("SaturnEyepiece"))]);
    for (double back : {0.0, 0.02})
        for (double side : {0.0, 0.001, 0.002}) {
            Vec3 eye = ep.position - ep.forward() * back + ep.right() * side;
            Accommodation a = accommodation(sc, eye, ep.forward(), sc.world.locate(eye), 0.0025);
            CHECK(a.found);
            CHECK(a.specular == 9);                // through the whole refractor
            CHECK(!std::isfinite(a.distance));    // Saturn's image is at infinity
        }
    // With nothing but optics between, the line of sight's own object: the room's wall.
    auto& room = dynamic_cast<IdealObserver&>(*sc.detectors[size_t(sc.findDetector("Room"))]);
    SurfaceHit hit;
    CHECK(sc.world.intersect(Ray{room.position, room.forward()}, Inf, hit));
    Accommodation a = accommodation(sc, room.position, room.forward(), room.region, 0.0025);
    CHECK(a.found && a.specular == 0);
    CHECK_REL(a.distance, hit.t, 1e-6);
}

