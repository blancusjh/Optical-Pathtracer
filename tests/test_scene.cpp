// Scene language and prescription views.
#include <fstream>

#include "check.hpp"
#include "owe/prescription.hpp"
#include "owe/scene_loader.hpp"

using namespace owe;

TEST(parser_handles_units_calls_tuples_and_blocks) {
    ValuePtr d = parseSceneText(R"(
        units = mm
        body A { type = lens  front = sphere(R = 48.0 mm)  thickness = 6.2  position = (0, 1, -2) m  tilt = (1deg, 0) }
    )", "t");
    CHECK(d->get("units")->str == "mm");
    CHECK(d->items.size() == 1);
    const Value& b = *d->items[0];
    CHECK(b.str == "body" && b.name == "A");
    CHECK(b.get("front")->kind == Value::Kind::Call);
    CHECK_NEAR(b.get("front")->get("R")->num, 0.048, 1e-15);
    CHECK(b.get("thickness")->unit == Value::Unit::None);
    CHECK_NEAR(b.get("position")->items[2]->num, -2.0, 1e-15);
    CHECK_NEAR(b.get("tilt")->items[0]->num, Pi / 180, 1e-15);
}

TEST(parser_reports_line_numbers) {
    bool threw = false;
    try {
        loadSceneFromString("units = m\n\nbody X { type = teapot }\nobserver E { position = (0,0,0) look_at = (0,1,0) }\n",
                            ".", "bad.owe");
    } catch (const std::exception& e) {
        threw = true;
        CHECK(std::string(e.what()).find("line 3") != std::string::npos);
        CHECK(std::string(e.what()).find("teapot") != std::string::npos);
    }
    CHECK(threw);
}

TEST(scene_builds_lens_with_physical_structure) {
    Scene s = loadSceneFromString(R"(
        units = mm
        body Singlet { type = lens  medium = N-BK7  front = sphere(R = 48)  back = conic(R = -120, k = -1.2)
                       thickness = 6.2  diameter = 25  front_diameter = 22  position = (0, 0, 100) }
        observer Eye { position = (0, 0, 0)  look_at = (0, 0, 1)  resolution = (8, 8) }
    )");
    int b = s.world.findBody("Singlet");
    CHECK(b >= 0);
    const Body& body = s.world.bodies()[b];
    CHECK(body.regions.size() == 1);
    // S1, S1-step (front aperture smaller than the rim), S2, rim1.
    CHECK(body.boundaries.size() == 4);
    CHECK_NEAR(body.params.at("edge_radius"), 0.0125, 1e-15);
    CHECK(s.detectors[0]->region == s.world.ambientRegion());
}

TEST(prescription_table_round_trip) {
    Prescription p = parsePrescription(R"(
name test
units mm
S1  40.0  5.0 N-BK7 10 k=-0.5 A4=1e-5
STO inf  3.0 air   6  stop
S3 -40.0 30.0 air  10
)");
    // Note: S3 follows air, so it is an air–air dummy surface in this toy table.
    CHECK(p.surfaces.size() == 3);
    CHECK(p.surfaces[1].stop);
    CHECK_NEAR(p.surfaces[0].A[0], 1e-5 * std::pow(1e-3, -3.0), 1e-3);
    Prescription q = parsePrescription(formatPrescription(p));
    for (size_t i = 0; i < p.surfaces.size(); ++i) {
        CHECK_REL(q.surfaces[i].R == Inf ? 1.0 : q.surfaces[i].R, p.surfaces[i].R == Inf ? 1.0 : p.surfaces[i].R, 1e-6);
        CHECK_NEAR(q.surfaces[i].t, p.surfaces[i].t, 1e-12);
        CHECK(q.surfaces[i].medium == p.surfaces[i].medium);
        CHECK(q.surfaces[i].stop == p.surfaces[i].stop);
        CHECK_REL(q.surfaces[i].k == 0 ? 1.0 : q.surfaces[i].k, p.surfaces[i].k == 0 ? 1.0 : p.surfaces[i].k, 1e-6);
    }
    CHECK_REL(q.surfaces[0].A[0], p.surfaces[0].A[0], 1e-5);
}

TEST(exit_pupil_placement_for_observers) {
    Scene s = loadSceneFromString(R"(
        units = mm
        body Scope { type = prescription  file = "lenses/kepler_16x.lens"  afocal = true
                     position = (0, 0, 0)  axis = (0, 1, 0) }
        observer Eye { position = exit_pupil("Scope")  look_at = (0, -1000, 0)  pupil = 2.5  resolution = (8, 8) }
    )", ".");
    const auto* eye = dynamic_cast<const IdealObserver*>(s.detectors[0].get());
    CHECK(eye != nullptr);
    // axis = (0,1,0) maps the instrument's +z (the direction light travels) to world +y,
    // so the exit pupil lies beyond the eyepiece at positive y.
    CHECK(eye->position.y > 0.3);
    CHECK_NEAR(eye->position.x, 0.0, 1e-12);
}

TEST(scene_edits_override_named_and_unnamed_blocks) {
    const std::string path = "build/test_edits.owe";
    std::ofstream(path) << R"(
units = m
world { sky = uniform(0.1)  sun { elevation = 30deg  azimuth = 0  luminance = 100  nee_share = 0.2 } }
body Box { type = box  size = (1, 1, 1)  position = (0, 5, 0)  material = white }
observer Eye { position = (0, 0, 0)  look_at = (0, 1, 0)  resolution = (8, 8)
               exposure = 3  white_balance = none  sun_share = 0.9 }
)";
    Scene a = loadSceneWithEdits(path, {"Box.position=(0, 7, 0)", "sun.nee_share=0.5", "render.spp=3"});
    CHECK(a.render.spp == 3);
    CHECK_NEAR(a.world.env.sunNeeShare, 0.5, 1e-15);
    CHECK(a.world.findBody("Box") >= 0);
    CHECK_NEAR(a.world.bodies()[size_t(a.world.findBody("Box"))].xf.point({0, 0, 0}).y, 7.0, 1e-12);
    // The observer carries its own display and sampling settings.
    RenderSettings rs = a.settingsFor(0);
    CHECK(!rs.autoExposure && rs.exposure == 3 && rs.whiteBalance == 0);
    a.useDetector(0);
    CHECK_NEAR(a.world.env.sunNeeShare, 0.9, 1e-15);
    bool threw = false;
    try {
        loadSceneWithEdits(path, {"Nothing.key=1"});
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}
