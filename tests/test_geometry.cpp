// Geometry kernel: exact surfaces, conic foci, acceleration structures.
#include "check.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/backends/registry.hpp"
#include "owe/scene/builders.hpp"
#include "owe/scene/geometry.hpp"
#include "owe/core/sampling.hpp"
#include "owe/transport/transport.hpp"

using namespace owe;

TEST(sphere_intersection_inside_and_outside) {
    SphereShape s(2.0);
    LocalHit h;
    CHECK(s.intersect({{0, 0, -10}, {0, 0, 1}}, 0, Inf, h));
    CHECK_NEAR(h.t, 8.0, 1e-12);
    CHECK_NEAR(h.n.z, -1.0, 1e-12);
    CHECK(s.intersect({{0, 0, 0}, {0, 1, 0}}, 0, Inf, h));
    CHECK_NEAR(h.t, 2.0, 1e-12);
    CHECK_NEAR(h.n.y, 1.0, 1e-12);
    CHECK(!s.intersect({{0, 3, -10}, {0, 0, 1}}, 0, Inf, h));
    // Far origin (precision): 1e6 m away.
    CHECK(s.intersect({{0, 0, -1e6}, {0, 0, 1}}, 0, Inf, h));
    CHECK_NEAR(h.p.z, -2.0, 1e-9);
}

TEST(planetary_sphere_preserves_twenty_centimetre_surface_clearance) {
    // World-space subtraction must keep the small height before constructing sphere coefficients.
    // Includes arbitrary planet placement; a z-only ground correction would fail the diagonal case.
    for (Vec3 n : {Vec3{0,0,1}, Vec3{1,0,0}, normalize(Vec3{1,2,3})}) {
        Scene sc;
        BodyMaterial material;
        SurfaceOptics black;
        black.type = SurfaceType::Absorber;
        material.optics = sc.world.addOptics(black);
        const double radius = 6371000;
        buildSphere(sc.world, "Earth", radius, material, -1, Transform::translate(-n * (radius + 0.2)));
        sc.world.build();
        SurfaceHit hit;
        CHECK(sc.world.intersect({n * 0.4, -n}, Inf, hit));
        CHECK_NEAR(hit.t, 0.6, 3e-9);
        CHECK_NEAR(length(hit.p + n * 0.2), 0, 3e-9);
        CHECK_NEAR(dot(hit.n, n), 1, 1e-12);
        SurfaceHit next;
        CHECK(!sc.world.intersect({offsetOrigin(hit.p, hit.n, n), n}, Inf, next));
        // From inside: the same full sphere must expose its far hemisphere, not a tangent plane.
        CHECK(sc.world.intersect({-n, -n}, Inf, next));
        CHECK_NEAR(next.t, 2 * radius - 0.8, 1e-8);
        CHECK_NEAR(dot(next.n, n), -1, 1e-12);
    }
}

TEST(quadric_hits_stay_precise_for_distant_origins) {
    // Rays arriving from 400 m must land on 5 cm surfaces to within an ulp of the origin (~1e-13 m),
    // otherwise the origin offset of the next segment can start inside the body.
    Rng rng(21, 22);
    CylinderShape cyl(0.05, 0, 1.5);
    SphereShape sph(0.05);
    SagSurface cap(1 / 0.03, -0.5, {}, 0.02);
    for (int i = 0; i < 5000; ++i) {
        Vec3 target{0.05 * (rng.uniform() - 0.5), 0.05 * (rng.uniform() - 0.5), 0.2 + rng.uniform()};
        Vec3 far = target + sampleUniformSphere(rng.uniform(), rng.uniform()) * 400.0;
        Ray r{far, normalize(target - far)};
        LocalHit h;
        if (cyl.intersect(r, 0, Inf, h)) CHECK_NEAR(std::hypot(h.p.x, h.p.y), 0.05, 1e-13);
        Ray rs{far - Vec3(0, 0, 0.7), r.d};
        if (sph.intersect(rs, 0, Inf, h)) CHECK_NEAR(length(h.p), 0.05, 1e-13);
        Ray rc{far - Vec3(0, 0, 0.7), r.d};
        if (cap.intersect(rc, 0, Inf, h)) CHECK_NEAR(cap.sag(std::hypot(h.p.x, h.p.y)), h.p.z, 1e-13);
    }
}

TEST(sag_surface_hits_lie_on_surface_with_correct_normal) {
    Rng rng(2, 3);
    struct Case { double c, k; std::vector<double> A; };
    std::vector<Case> cases = {{1 / 0.05, 0, {}}, {-1 / 0.08, -1, {}}, {1 / 0.1, -2.5, {}}, {1 / 0.07, 0.4, {}},
                               {1 / 0.06, -0.8, {3.0, -200.0}}, {0, 0, {50.0}}};
    for (auto& cs : cases) {
        SagSurface s(cs.c, cs.k, cs.A, 0.02);
        int hits = 0;
        for (int i = 0; i < 2000; ++i) {
            Vec3 o{(rng.uniform() - 0.5) * 0.04, (rng.uniform() - 0.5) * 0.04, -0.05};
            Vec3 d = normalize(Vec3((rng.uniform() - 0.5) * 0.4, (rng.uniform() - 0.5) * 0.4, 1));
            LocalHit h;
            if (!s.intersect({o, d}, 0, Inf, h)) continue;
            ++hits;
            double r = std::sqrt(h.p.x * h.p.x + h.p.y * h.p.y);
            CHECK(r <= 0.02 * (1 + 1e-12));
            CHECK_NEAR(s.sag(r), h.p.z, 1e-12);
            // Normal is the normalised gradient of z − sag(r), checked by finite differences.
            double e = 1e-7;
            auto F = [&](double x, double y) { return s.sag(std::sqrt(x * x + y * y)); };
            Vec3 g{-(F(h.p.x + e, h.p.y) - F(h.p.x - e, h.p.y)) / (2 * e), -(F(h.p.x, h.p.y + e) - F(h.p.x, h.p.y - e)) / (2 * e), 1};
            g = normalize(g);
            CHECK_NEAR(dot(g, h.n), 1.0, 1e-8);
        }
        CHECK(hits > 500);
    }
}

TEST(numeric_asphere_agrees_with_exact_quadric) {
    // A vanishing aspheric term forces the numeric path; it must agree with the closed form.
    SagSurface exact(1 / 0.05, -0.6, {}, 0.02), numeric(1 / 0.05, -0.6, {1e-30}, 0.02);
    Rng rng(4, 5);
    for (int i = 0; i < 2000; ++i) {
        Vec3 o{(rng.uniform() - 0.5) * 0.03, (rng.uniform() - 0.5) * 0.03, (rng.uniform() < 0.5 ? -0.03 : 0.03)};
        Vec3 d = normalize(Vec3((rng.uniform() - 0.5) * 0.6, (rng.uniform() - 0.5) * 0.6, o.z < 0 ? 1 : -1));
        LocalHit a, b;
        bool ha = exact.intersect({o, d}, 0, Inf, a), hb = numeric.intersect({o, d}, 0, Inf, b);
        CHECK(ha == hb);
        if (ha && hb) CHECK_NEAR(a.t, b.t, 1e-12);
    }
}

TEST(paraboloid_mirror_focuses_collimated_light_exactly) {
    // Concave paraboloid (k = −1) with vertex radius R: focus at R/2 for every zone.
    Scene sc;
    SurfaceSpec s;
    s.R = 1.0;
    s.k = -1;
    s.semiDiameter = 0.2;
    SurfaceOptics mir;
    mir.type = SurfaceType::Mirror;
    buildMirror(sc.world, "M", s, 0.02, sc.world.addOptics(mir), -1, Transform{});
    sc.world.build();
    Tracer tr(sc);
    Rng rng(1, 1);
    for (double h = 0.005; h < 0.2; h += 0.01) {
        PathRecord r = tr.walk({{h * 0.6, h * 0.8, 2.0}, {0, 0, -1}}, 0, 550, WalkMode::Stochastic, rng);
        CHECK(r.v.size() >= 3);
        const PathVertex& v = r.v[1];
        CHECK(v.event == EventKind::Reflect);
        // Distance of the reflected ray from the focal point.
        Vec3 F{0, 0, 0.5};
        Vec3 w = F - v.p;
        double miss = length(w - v.dOut * dot(w, v.dOut));
        CHECK_NEAR(miss, 0.0, 1e-12);
    }
}

TEST(ellipsoidal_mirror_images_focus_to_focus) {
    // Conic with vertex radius R and k = −e²: foci at a(1∓e) from the vertex, a = R/(1+k).
    double R = 0.3, k = -0.64, e = 0.8;
    double a = R / (1 + k);
    Vec3 F1{0, 0, a * (1 - e)}, F2{0, 0, a * (1 + e)};
    Scene sc;
    SurfaceSpec s;
    s.R = R;
    s.k = k;
    s.semiDiameter = 0.2;
    SurfaceOptics mir;
    mir.type = SurfaceType::Mirror;
    buildMirror(sc.world, "E", s, 0.01, sc.world.addOptics(mir), -1, Transform{});
    sc.world.build();
    Tracer tr(sc);
    Rng rng(3, 3);
    int n = 0;
    for (int i = 0; i < 400; ++i) {
        Vec3 d = sampleUniformSphere(rng.uniform(), rng.uniform());
        if (d.z > -0.2) continue;
        PathRecord r = tr.walk({F1, d}, 0, 550, WalkMode::Stochastic, rng);
        if (r.v.size() < 3 || r.v[1].event != EventKind::Reflect) continue;
        Vec3 w = F2 - r.v[1].p;
        CHECK_NEAR(length(w - r.v[1].dOut * dot(w, r.v[1].dOut)), 0.0, 1e-11);
        ++n;
    }
    CHECK(n > 50);
}

TEST(mesh_bvh_matches_brute_force) {
    Rng rng(7, 7);
    MeshData md;
    for (int i = 0; i < 3000; ++i) {
        Vec3 c{rng.uniform() * 10, rng.uniform() * 10, rng.uniform() * 10};
        md.addTriangle(c, c + Vec3(rng.uniform(), rng.uniform(), rng.uniform()) * 0.5,
                       c + Vec3(rng.uniform(), rng.uniform(), rng.uniform()) * 0.5);
    }
    MeshShape mesh(md.positions, md.triangles);
    for (int i = 0; i < 3000; ++i) {
        Ray r{{rng.uniform() * 10, rng.uniform() * 10, -1}, normalize(Vec3(rng.uniform() - 0.5, rng.uniform() - 0.5, 1))};
        LocalHit h;
        bool hit = mesh.intersect(r, 0, Inf, h);
        double best = Inf;
        for (size_t t = 0; t < md.triangles.size(); ++t) {
            MeshShape one({md.positions[md.triangles[t][0]], md.positions[md.triangles[t][1]], md.positions[md.triangles[t][2]]},
                          {{0, 1, 2}});
            LocalHit h1;
            if (one.intersect(r, 0, Inf, h1)) best = std::min(best, h1.t);
        }
        CHECK(hit == std::isfinite(best));
        if (hit) CHECK_NEAR(h.t, best, 1e-12);
        if (i > 300) break;  // brute force is slow; a few hundred rays suffice
    }
}

TEST(world_bvh_matches_brute_force_over_boundaries) {
    Scene sc;
    Rng rng(8, 8);
    BodyMaterial m;
    m.optics = sc.world.absorberOptics();
    for (int i = 0; i < 400; ++i)
        buildSphere(sc.world, "s" + std::to_string(i), 0.05 + 0.2 * rng.uniform(), m, -1,
                    Transform::translate({rng.uniform() * 10, rng.uniform() * 10, rng.uniform() * 10}));
    sc.world.build();
    for (int i = 0; i < 2000; ++i) {
        Ray r{{rng.uniform() * 10, rng.uniform() * 10, rng.uniform() * 10}, sampleUniformSphere(rng.uniform(), rng.uniform())};
        SurfaceHit h;
        bool hit = sc.world.intersect(r, Inf, h);
        double best = Inf;
        for (const auto& b : sc.world.boundaries()) {
            LocalHit lh;
            if (b.shape->intersect({b.toLocal.point(r.o), b.toLocal.vector(r.d)}, 0, Inf, lh)) best = std::min(best, lh.t);
        }
        CHECK(hit == std::isfinite(best));
        if (hit) CHECK_NEAR(h.t, best, 1e-12);
    }
}

TEST(point_location_identifies_regions) {
    Scene sc;
    BodyMaterial glass;
    glass.transparent = true;
    glass.medium = "N-BK7";
    glass.optics = sc.world.dielectricOptics();
    int b = buildSphere(sc.world, "ball", 1.0, glass, -1, Transform::translate({5, 0, 0}));
    sc.world.build();
    CHECK(sc.world.locate({5, 0.2, 0.1}) == sc.world.bodies()[b].regions[0]);
    CHECK(sc.world.locate({0, 0, 0}) == sc.world.ambientRegion());
}

TEST(dome_and_round_wall_are_exact_and_open_where_cut) {
    // Rays from inside the observatory: hits lie on the surfaces (to ~1e-12 m) and never inside
    // the slit or the window; rays aimed through the middle of an opening escape.
    const double R = 4.5, H = 3.2;
    DomeShape dome(R, radians(170), 1.8, radians(80));
    RoundWallShape wall(R, H, {{radians(250), 1.2, 1.2, 2.6}});
    Rng rng(5, 6);
    int domeHits = 0, wallHits = 0;
    for (int i = 0; i < 20000; ++i) {
        Vec3 o{rng.uniform() * 2 - 1, rng.uniform() * 2 - 1, 0.2 + rng.uniform()};
        Vec3 d = sampleUniformSphere(rng.uniform(), rng.uniform());
        LocalHit h;
        if (dome.intersect({o, d}, 0, Inf, h)) {
            domeHits++;
            CHECK_NEAR(length(h.p), R, 1e-11);
            CHECK(h.p.z >= -1e-12);
            // Not in the slit: lateral distance from the slit plane ≥ half width, or beyond its top.
            Vec3 s{std::sin(radians(170)), std::cos(radians(170)), 0};  // azimuth from +y toward +x
            Vec3 side = cross(Vec3(0, 0, 1), s);
            bool inSlit = std::abs(dot(h.p, side)) < 0.9 && dot(h.p, s) > 0 &&
                          std::atan2(h.p.z, std::abs(dot(h.p, s))) < radians(80);
            CHECK(!inSlit);
        }
        if (wall.intersect({o, d}, 0, Inf, h)) {
            wallHits++;
            CHECK_NEAR(std::hypot(h.p.x, h.p.y), R, 1e-11);
            CHECK(h.p.z >= -1e-12 && h.p.z <= H + 1e-12);
        }
    }
    CHECK(domeHits > 1000 && wallHits > 1000);
    LocalHit h;
    Vec3 up = normalize(Vec3(std::sin(radians(170)), std::cos(radians(170)), 1.0));
    CHECK(!dome.intersect({{0, 0, 0.5}, up}, 0, Inf, h));
    Vec3 win{std::sin(radians(250)), std::cos(radians(250)), 0};
    CHECK(!wall.intersect({{0, 0, 1.9}, win}, 0, Inf, h));
    CHECK(wall.intersect({{0, 0, 0.5}, win}, 0, Inf, h));
}

TEST(wave_surface_spline_follows_the_wave_sum_and_rays_find_it) {
    // A 48-wave ripple field: its bicubic spline must match the analytic sum in height and slope,
    // the slope bound must hold, and a ray must meet the surface where the sum says it is.
    auto waves = rippleField(48, 11, 0.015, 0.14, 0.22, radians(30), radians(110));
    WaveSurface exact(waves, 0.5, 0.5, 0.1, 0), spline(waves, 0.5, 0.5, 0.1);
    CHECK(!exact.sampled() && spline.sampled());
    Rng rng(5, 1);
    double worstH = 0, worstSlope = 0, maxSlope = 0;
    for (int i = 0; i < 20000; ++i) {
        double x = (2 * rng.uniform() - 1) * 0.5, y = (2 * rng.uniform() - 1) * 0.5;
        double h0, gx0, gy0, h1, gx1, gy1;
        exact.height(x, y, h0, gx0, gy0);
        spline.height(x, y, h1, gx1, gy1);
        worstH = std::max(worstH, std::abs(h1 - h0));
        worstSlope = std::max(worstSlope, std::hypot(gx1 - gx0, gy1 - gy0));
        maxSlope = std::max(maxSlope, std::hypot(gx1, gy1));
    }
    CHECK(worstH < 2e-3 * exact.amplitudeBound());
    CHECK(worstSlope < 0.01);
    CHECK(maxSlope <= spline.slopeBound());
    CHECK(spline.slopeBound() < exact.slopeBound());
    for (int i = 0; i < 2000; ++i) {
        Vec3 o{(2 * rng.uniform() - 1) * 0.3, (2 * rng.uniform() - 1) * 0.3, 0.5};
        Vec3 d = normalize(Vec3{(rng.uniform() - 0.5) * 0.8, (rng.uniform() - 0.5) * 0.8, -1});
        LocalHit h;
        CHECK(spline.intersect(Ray{o, d}, 0, 10, h));
        CHECK_NEAR(h.p.z, spline.height(h.p.x, h.p.y), 1e-9);
        CHECK_NEAR(h.p.z, exact.height(h.p.x, h.p.y), 2e-3 * exact.amplitudeBound());
        LocalHit up;  // and from just below it, back up along the same line
        CHECK(spline.intersect(Ray{h.p + d * 0.002, -d}, 0, 10, up));
        CHECK_NEAR(length(up.p - h.p), 0, 1e-7);
    }
}

TEST(lathe_cone_and_torus_patches_are_hit_exactly) {
    ConeShape cone(0.03, 0.01, 0.038, 0.095);
    TorusPatchShape torus(0.022, 0.009, 0.009, -Pi / 2, -0.08);
    Rng rng(9, 2);
    int coneHits = 0, torusHits = 0;
    for (int i = 0; i < 20000; ++i) {
        Vec3 o{(rng.uniform() - 0.5) * 0.3, -0.2, 0.05 + (rng.uniform() - 0.5) * 0.1};
        Vec3 target{(rng.uniform() - 0.5) * 0.08, (rng.uniform() - 0.5) * 0.08, rng.uniform() * 0.1};
        Ray r{o, normalize(target - o)};
        LocalHit h;
        if (cone.intersect(r, 0, 10, h)) {
            ++coneHits;
            double rho = 0.03 + (0.038 - 0.03) / (0.095 - 0.01) * (h.p.z - 0.01);
            CHECK_NEAR(std::hypot(h.p.x, h.p.y), rho, 1e-12);
            CHECK(h.p.z >= 0.01 && h.p.z <= 0.095);
            CHECK(dot(h.n, Vec3(h.p.x, h.p.y, 0)) > 0);  // away from the axis
        }
        Vec3 o2{(rng.uniform() - 0.5) * 0.2, (rng.uniform() - 0.5) * 0.2, -0.1};
        Vec3 t2{(rng.uniform() - 0.5) * 0.06, (rng.uniform() - 0.5) * 0.06, rng.uniform() * 0.02};
        Ray r2{o2, normalize(t2 - o2)};
        if (torus.intersect(r2, 0, 10, h)) {
            ++torusHits;
            CHECK_NEAR(sqr(std::hypot(h.p.x, h.p.y) - 0.022) + sqr(h.p.z - 0.009), 0.009 * 0.009, 1e-12);
            CHECK(h.p.z <= 0.009 + 1e-9);  // the lower quarter of the tube only
            // The first hit along the ray: nothing of the patch before it.
            LocalHit h2;
            CHECK(!torus.intersect(r2, 0, h.t * (1 - 1e-9), h2));
        }
    }
    CHECK(coneHits > 1000 && torusHits > 1000);
}

TEST(vessel_regions_are_consistent_through_glass_and_liquid) {
    // A tumbler with filleted corners and a round rim, holding water: every ray that crosses it meets
    // each boundary from the region it is in (no leaks, no inconsistencies), from every direction.
    Scene sc = loadSceneFromString(R"(
        units = m
        world { sky = uniform(1) }
        body Glass { type = vessel  outer = [(0, 0), (30mm, 0, 9mm), (38mm, 95mm)]
                     inner = [(0, 13mm), (27mm, 13mm, 7mm), (35mm, 95mm)]  rim = round  glass = N-BK7
                     liquid = water  level = 60mm }
        observer Eye { position = (0.04, -0.25, 0.12)  look_at = (0, 0, 0.05)  fov = 30deg  pupil = 0  resolution = (48, 48) }
    )");
    CHECK(sc.world.bodies().size() == 1);
    RenderSettings rs;
    rs.backend = "cpu";
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(64);
    CHECK(R->stats().leaks == 0);
    CHECK(R->stats().inconsistencies <= R->stats().segments / 1000000);
    CHECK(sc.world.locate({0, 0, 0.03}) != sc.world.locate({0, 0, 0.08}));  // water below, air above the level
}

TEST(contact_spawn_orders_glass_support_and_reflected_return) {
    for (double gap : {0.0, 1e-12, 1e-7, 1e-4}) {
        Scene scene;
        BodyMaterial glass;
        glass.transparent = true;
        glass.medium = "PMMA";
        glass.optics = scene.world.dielectricOptics();
        BodyMaterial support;
        SurfaceOptics diffuse;
        diffuse.type = SurfaceType::Diffuse;
        support.optics = scene.world.addOptics(diffuse);
        int g = buildBox(scene.world, "glass", {1, 1, .1}, glass, -1, Transform::translate({0,0,.05 + gap}));
        int s = buildBox(scene.world, "support", {1, 1, .2}, support, -1, Transform::translate({0,0,-.1}));
        scene.world.build();
        uint32_t glassRegion = scene.world.bodies()[g].regions[0];
        uint32_t glassBoundary = scene.world.bodies()[g].boundaries[0];
        uint32_t supportBoundary = scene.world.bodies()[s].boundaries[0];
        SurfaceHit exit;
        CHECK(scene.world.intersect(Ray{{0,0,.05 + gap}, {0,0,-1}, kNone, glassRegion}, Inf, exit));
        CHECK(exit.boundary == glassBoundary);
        CHECK(!exit.ambiguousContact);
        SurfaceHit next;
        CHECK(scene.world.intersect(scene.world.spawnRay(exit, {0,0,-1}, 0), Inf, next));
        CHECK(next.boundary == supportBoundary);
        CHECK_NEAR(next.t, gap, 2e-17);
        CHECK_NEAR(next.n.z, 1, 1e-12);
        // A support reflection must immediately enter the glass again at exact contact.
        SurfaceHit entry;
        CHECK(scene.world.intersect(scene.world.spawnRay(next, {0,0,1}, 0), Inf, entry));
        CHECK(entry.boundary == glassBoundary);
        CHECK_NEAR(entry.t, gap, 2e-17);
        CHECK_NEAR(entry.n.z, -1, 1e-12);
        // Transmitting at that entry suppresses only its self root, not its opposite face.
        SurfaceHit far;
        CHECK(scene.world.intersect(scene.world.spawnRay(entry, {0,0,1}, glassRegion), Inf, far));
        CHECK(far.boundary == glassBoundary);
        CHECK_NEAR(far.t, .1, 3e-17);
        // Reflecting at the glass exit stays in glass and never interacts with the support.
        CHECK(scene.world.intersect(scene.world.spawnRay(exit, {0,0,1}, glassRegion), Inf, far));
        CHECK(far.boundary == glassBoundary);
        CHECK_NEAR(far.t, .1, 3e-17);
        Tracer tracer(scene);
        CHECK(tracer.transmittance(Endpoint(exit, 0), {0,0,-1}, Inf,
                                  Wavelengths::sample(.5)).isZero());
    }
}

TEST(contact_spawn_keeps_resolved_overlap_and_interior_launch_diagnostic) {
    Scene scene;
    BodyMaterial glass;
    glass.transparent = true;
    glass.medium = "PMMA";
    glass.optics = scene.world.dielectricOptics();
    BodyMaterial support;
    support.optics = scene.world.absorberOptics();
    int g = buildBox(scene.world, "glass", {1,1,.1}, glass, -1, Transform::translate({0,0,.05 - 1e-5}));
    int s = buildBox(scene.world, "support", {1,1,.2}, support, -1, Transform::translate({0,0,-.1}));
    scene.world.build();
    uint32_t gr = scene.world.bodies()[g].regions[0];
    uint32_t sb = scene.world.bodies()[s].boundaries[0];
    SurfaceHit hit;
    CHECK(scene.world.intersect(Ray{{0,0,.05}, {0,0,-1}, kNone, gr}, Inf, hit));
    CHECK(hit.boundary == sb);  // The overlapping support is encountered before the glass exit.
    CHECK(scene.world.boundaries()[hit.boundary].front != gr);
    Tracer tracer(scene);
    Wavelengths wl = Wavelengths::sample(.5);
    Rng rng(1,2);
    TransportStats stats;
    tracer.radiance(Ray{{0,0,.05},{0,0,-1}}, gr, wl, rng, stats);
    CHECK(stats.inconsistencies > 0);
    // A detector incorrectly declared ambient while inside opaque matter remains an error.
    stats = TransportStats{};
    tracer.radiance(Ray{{0,0,-.1},{0,0,-1}}, 0, wl, rng, stats);
    CHECK(stats.inconsistencies > 0);
}

TEST(contact_spawn_preserves_nested_region_topology_and_thin_same_boundary) {
    Scene scene;
    BodyMaterial glass;
    glass.transparent = true;
    glass.medium = "PMMA";
    glass.optics = scene.world.dielectricOptics();
    int outer = buildSphere(scene.world, "outer", 1, glass, -1, Transform{});
    uint32_t outerRegion = scene.world.bodies()[outer].regions[0];
    scene.world.setImmersion(outerRegion);
    int inner = buildSphere(scene.world, "inner", .25, glass, -1, Transform::translate({.75,0,0}));
    scene.world.setImmersion(kOutside);
    scene.world.build();
    uint32_t innerRegion = scene.world.bodies()[inner].regions[0];
    SurfaceHit innerExit, outerExit;
    CHECK(scene.world.intersect(Ray{{.75,0,0},{1,0,0}, kNone, innerRegion}, Inf, innerExit));
    CHECK(innerExit.boundary == scene.world.bodies()[inner].boundaries[0]);
    CHECK(scene.world.intersect(scene.world.spawnRay(innerExit, {1,0,0}, outerRegion), Inf, outerExit));
    CHECK(outerExit.boundary == scene.world.bodies()[outer].boundaries[0]);
    CHECK_NEAR(outerExit.t, 0, 1e-15);
    SurfaceHit back;
    CHECK(scene.world.intersect(scene.world.spawnRay(outerExit, {-1,0,0}, outerRegion), Inf, back));
    CHECK(back.boundary == innerExit.boundary);
    CHECK_NEAR(back.t, 0, 1e-15);
    CHECK(scene.world.intersect(scene.world.spawnRay(back, {-1,0,0}, innerRegion), Inf, back));
    CHECK_NEAR(back.t, .5, 1e-15);

    Scene thin;
    glass.optics = thin.world.dielectricOptics();
    int body = buildBox(thin.world, "thin", {1,1,1e-10}, glass, -1, Transform{});
    thin.world.build();
    uint32_t region = thin.world.bodies()[body].regions[0];
    SurfaceHit entry;
    CHECK(thin.world.intersect(Ray{{0,0,1},{0,0,-1}, kNone, 0}, Inf, entry));
    CHECK(thin.world.intersect(thin.world.spawnRay(entry, {0,0,-1}, region), Inf, back));
    CHECK(back.boundary == entry.boundary);
    CHECK_NEAR(back.t, 1e-10, 1e-16);
}

TEST(contact_spawn_reports_ambiguous_distinct_incident_interfaces) {
    Scene scene;
    BodyMaterial black;
    black.optics = scene.world.absorberOptics();
    buildSphere(scene.world, "one", 1, black, -1, Transform{});
    buildSphere(scene.world, "two", 1, black, -1, Transform{});
    scene.world.build();
    SurfaceHit hit;
    CHECK(scene.world.intersect(Ray{{0,0,2},{0,0,-1}, kNone, 0}, Inf, hit));
    CHECK(hit.ambiguousContact);
    Tracer tracer(scene);
    Wavelengths wl = Wavelengths::sample(.5);
    Rng rng(1,2);
    TransportStats stats;
    tracer.radiance(Ray{{0,0,2},{0,0,-1}}, 0, wl, rng, stats);
    CHECK(stats.inconsistencies == 1);
}

TEST(contact_null_shadow_walk_preserves_region_order_and_rejects_overlap) {
    for (double overlap : {0.0, 1e-5}) {
        Scene scene;
        BodyMaterial material;
        material.transparent = true;
        material.medium = "PMMA";
        material.optics = scene.world.nullOptics();
        int top = buildSphere(scene.world, "top", .25, material, -1, Transform::translate({0,0,.25}));
        buildSphere(scene.world, "bottom", .25, material, -1, Transform::translate({0,0,-.25 + overlap}));
        scene.world.build();
        SurfaceHit exit;
        uint32_t r = scene.world.bodies()[top].regions[0];
        CHECK(scene.world.intersect(Ray{{0,0,.25},{0,0,-1}, kNone,r}, Inf, exit));
        Tracer tracer(scene);
        Wavelengths wl = Wavelengths::sample(.5);
        uint32_t final = kNone;
        Spec4 tr = tracer.transmittance(Endpoint(exit, 0), {0,0,-1}, Inf, wl, &final);
        if (overlap == 0) {
            CHECK_NEAR(tr[0], 1, 1e-12);
            CHECK(final == 0);
        } else {
            CHECK(tr.isZero());
        }
    }
}

TEST(finite_detector_connection_preserves_reciprocal_region_attenuation) {
    Scene scene;
    Medium absorbing;
    absorbing.name = "attenuating";
    absorbing.absorption = Spectrum::constant(.5);
    uint32_t medium = scene.world.addMedium(absorbing);
    int body = scene.world.addBody("volume", "sphere", -1, Transform{});
    uint32_t inside = scene.world.addRegion("volume.interior", medium, body);
    scene.world.addBoundary(body, "surface", std::make_shared<SphereShape>(1), Transform{},
                            0, inside, scene.world.nullOptics());
    scene.world.build();
    Tracer tracer(scene);
    Wavelengths wavelengths = Wavelengths::sample(.5);
    Spec4 through = tracer.connectionTransmittance(Endpoint({0,0,3}, 0), Endpoint({0,0,-3}, 0), wavelengths);
    Spec4 reverse = tracer.connectionTransmittance(Endpoint({0,0,-3}, 0), Endpoint({0,0,3}, 0), wavelengths);
    for (int k = 0; k < NW; ++k) {
        CHECK_NEAR(through[k], std::exp(-1), 1e-14);
        CHECK_NEAR(reverse[k], through[k], 1e-14);
    }
    Spec4 partial = tracer.connectionTransmittance(Endpoint({0,0,0}, inside), Endpoint({0,0,-3}, 0), wavelengths);
    CHECK_NEAR(partial[0], std::exp(-.5), 1e-14);
    CHECK(tracer.connectionTransmittance(Endpoint({0,0,0}, 0), Endpoint({0,0,-3}, 0), wavelengths).isZero());
}

TEST(contact_mesh_departure_retains_distinct_same_normal_faces) {
    // Both triangles belong to one boundary and have the same normal. Only
    // primitive identity separates the departing face from a close opaque layer.
    for (double gap : {0.0, 1e-15, 1e-10}) {
        Scene scene;
        const int body = scene.world.addBody("layers", "mesh", -1, Transform{});
        std::vector<Vec3> vertices{{-1,-1,0}, {1,-1,0}, {0,1,0},
                                   {-1.25,-1,-gap}, {1.25,-1,-gap}, {0,1.25,-gap}};
        auto mesh = std::make_shared<MeshShape>(vertices,
            std::vector<std::array<uint32_t,3>>{{0,1,2}, {3,4,5}});
        const uint32_t boundary = scene.world.addBoundary(body, "faces", mesh, Transform{},
                                                          0, 0, scene.world.absorberOptics());
        scene.world.build();
        SurfaceHit departure;
        departure.p = {0,0,0};
        departure.n = {0,0,1};
        departure.boundary = boundary;
        departure.prim = 0;
        departure.error = 1e-14;
        SurfaceHit next;
        CHECK(scene.world.intersect(scene.world.spawnRay(departure, {0,0,-1}, 0), Inf, next));
        CHECK(next.prim == 1);
        CHECK_NEAR(next.t, gap, 1e-25);
        Tracer tracer(scene);
        CHECK(tracer.connectionTransmittance(Endpoint(departure, 0), Endpoint({0,0,-1}, 0),
                                             Wavelengths::sample(.5)).isZero());
    }
}

TEST(contact_mesh_departure_skips_only_connected_coplanar_face) {
    for (bool splitVertices : {false, true}) {
        Scene scene;
        const int body = scene.world.addBody("quad", "mesh", -1, Transform{});
        auto vertices = std::vector<Vec3>{{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
        auto triangles = std::vector<std::array<uint32_t,3>>{{0,1,2}, {0,2,3}};
        if (splitVertices) {
            vertices.insert(vertices.end(), {{-1,-1,0}, {1,1,0}, {-1,1,0}});
            triangles[1] = {4,5,6};  // polygon soup: shared positions, distinct indices
        }
        auto mesh = std::make_shared<MeshShape>(vertices, triangles);
        const uint32_t boundary = scene.world.addBoundary(body, "face", mesh, Transform{},
                                                          0, 0, scene.world.absorberOptics());
        scene.world.build();
        SurfaceHit departure;
        CHECK(scene.world.intersect(Ray{{0,0,1},{0,0,-1}}, Inf, departure));
        CHECK(departure.boundary == boundary);
        SurfaceHit next;
        CHECK(!scene.world.intersect(scene.world.spawnRay(departure, {0,0,-1}, 0), Inf, next));
        CHECK(!scene.world.intersect(scene.world.spawnRay(departure, {0,0,1}, 0), Inf, next));
        Tracer tracer(scene);
        const Wavelengths wavelengths = Wavelengths::sample(.5);
        CHECK_NEAR(tracer.connectionTransmittance(Endpoint(departure, 0), Endpoint({0,0,-1}, 0), wavelengths)[0], 1, 1e-14);
        CHECK_NEAR(tracer.connectionTransmittance(Endpoint(departure, 0), Endpoint({0,0,1}, 0), wavelengths)[0], 1, 1e-14);
    }
}

TEST(finite_detector_connection_checks_only_forward_source_neighborhood) {
    for (double offset : {-1e-15, 0.0, 1e-15}) {
        Scene scene;
        const int body = scene.world.addBody("layers", "mesh", -1, Transform{});
        auto mesh = std::make_shared<MeshShape>(
            std::vector<Vec3>{{-1,-1,0}, {1,-1,0}, {0,1,0},
                              {-1.25,-1,offset}, {1.25,-1,offset}, {0,1.25,offset}},
            std::vector<std::array<uint32_t,3>>{{0,1,2}, {3,4,5}});
        const uint32_t boundary = scene.world.addBoundary(body, "faces", mesh, Transform{},
                                                          0, 0, scene.world.absorberOptics());
        scene.world.build();
        SurfaceHit source;
        source.p = {0,0,0};
        source.n = {0,0,1};
        source.boundary = boundary;
        source.prim = 0;
        source.error = 1e-14;
        Tracer tracer(scene);
        const Spec4 visibility = tracer.connectionTransmittance(Endpoint(source, 0), Endpoint({0,0,1}, 0),
                                                                Wavelengths::sample(.5));
        if (offset < 0) CHECK_NEAR(visibility[0], 1, 1e-14);
        else CHECK(visibility.isZero());
    }
}

TEST(contact_spawn_preserves_grazing_distant_emitter_departure) {
    World world;
    BodyMaterial material;
    material.optics = world.absorberOptics();
    const int body = buildSphere(world, "Lamp", 2, material, -1,
                                 Transform::translate({20,0,999980}));
    world.build();
    SurfaceHit emission;
    emission.p = {19.714002469769824, .44869003400739566, 999978.07207814837};
    emission.n = {-.14299876511508769, .22434501700369783, -.96396092582696036};
    emission.boundary = world.bodies()[body].boundaries[0];
    emission.error = 64 * std::numeric_limits<double>::epsilon() * maxAbsComponent(emission.p);
    const Vec3 direction{.98903240076991672, .0005071560992455376, -.14769784365381128};
    SurfaceHit self;
    // Transforming the sampled point back into the sphere places it 2.18e-11 m
    // inside; the outgoing grazing self-root lies 2.06e-8 m away. It is still
    // the departing face, while a genuine inward ray must reach the far face.
    CHECK(!world.intersect(world.spawnRay(emission, direction, 0), Inf, self));
    const uint32_t interior = world.bodies()[body].regions[0];
    CHECK(world.intersect(world.spawnRay(emission, -emission.n, interior), Inf, self));
    CHECK_NEAR(self.t, 4, 1e-9);
}

TEST(contact_wave_departure_keeps_sub_error_support_faces) {
    for (double gap : {0.0, 1e-15, 1e-10}) {
        Scene scene = loadSceneFromString(R"(
            units = m
            world { sky = uniform(1) }
            body Pool { type = waves size = (.12,.12) depth = .08 margin = .03 medium = water
                        waves = [wave(amplitude = .0005, wavelength = .02, direction = 0deg, phase = 0deg)] }
            observer Eye { position = (0,0,1) look_at = (0,0,0) pupil = 0 resolution = (1,1) }
        )");
        SurfaceHit departure;
        const Vec3 direction{0,0,-1};
        CHECK(scene.world.intersect(Ray{{-.052796776271980143,0,1},direction}, Inf, departure));
        const uint32_t region = scene.world.boundaries()[departure.boundary].back;
        const int support = scene.world.addBody("support", "sheet", -1, Transform{});
        const uint32_t boundary = scene.world.addBoundary(support, "face", PlaneShape::rect(.06,.06),
            Transform::translate({0,0,departure.p.z-gap}), region, region, scene.world.absorberOptics());
        scene.world.build();
        SurfaceHit next;
        CHECK(scene.world.intersect(scene.world.spawnRay(departure, direction, region), Inf, next));
        CHECK(next.boundary == boundary);
        CHECK_NEAR(next.t, gap, 1e-18);
    }
}

TEST(contact_torus_departure_keeps_next_grazing_root) {
    Scene scene;
    const int body = scene.world.addBody("rim", "torus", -1, Transform{});
    const uint32_t inside = scene.world.addRegion("rim.inside", scene.world.medium("N-BK7"), body);
    const uint32_t boundary = scene.world.addBoundary(body, "rim",
        std::make_shared<TorusPatchShape>(.0365,.095,.0015,0,Pi), Transform{},
        0, inside, scene.world.dielectricOptics());
    scene.world.build();
    SurfaceHit departure;
    departure.p = {.019602670396745223,.030927447989482597,.096495465186224261};
    const double radius = std::hypot(departure.p.x, departure.p.y);
    departure.n = normalize(departure.p - Vec3{departure.p.x*.0365/radius, departure.p.y*.0365/radius,.095});
    departure.boundary = boundary;
    departure.error = 1e-14;
    const Vec3 direction{-.30284486458648285,-.9514430080866626,-.055146988644892453};
    auto implicit = [&](double t) {
        const Vec3 p = departure.p + direction*t;
        return sqr(std::hypot(p.x,p.y)-.0365)+sqr(p.z-.095)-sqr(.0015);
    };
    double lo = 1e-5, hi = 1e-3;
    CHECK(implicit(lo) < 0 && implicit(hi) > 0);
    for (int i = 0; i < 60; ++i) {
        const double mid = .5*(lo+hi);
        if (implicit(mid) < 0) lo = mid; else hi = mid;
    }
    SurfaceHit next;
    CHECK(scene.world.intersect(scene.world.spawnRay(departure, direction, inside), Inf, next));
    CHECK(next.boundary == boundary);
    CHECK_NEAR(next.t, .5*(lo+hi), 2e-15);
    CHECK(next.t > 1e-4);
}

TEST(contact_mesh_carries_skinny_triangle_hit_residual) {
    // A fan triangle from the panel's flat back reconstructs a distant primary
    // hit 4.80e-14 m before its plane. Shape extent alone cannot bound this solve.
    for (double gap : {1e-15, 1e-10}) {
        Scene scene;
        const int body = scene.world.addBody("thin", "mesh", -1, Transform{});
        const uint32_t inside = scene.world.addRegion("thin.inside", scene.world.medium("PMMA"), body);
        auto mesh = std::make_shared<MeshShape>(
            std::vector<Vec3>{{0,0,0}, {-.150000006,.3010807149,0}, {-.150000006,.3016670454,0},
                              {0,0,gap}, {-.150000006,.3010807149,gap}, {-.150000006,.3016670454,gap}},
            std::vector<std::array<uint32_t,3>>{{0,1,2}, {3,5,4}});
        const uint32_t boundary = scene.world.addBoundary(body, "surface", mesh, Transform{},
                                                          0, inside, scene.world.dielectricOptics());
        scene.world.build();
        const Ray primary{{-.13555639704465883,-.25735279099073116,-.26475660100926879},
                          {.028307038741447871,.88247052564743556,.46951515728592741}};
        SurfaceHit entry, exit;
        CHECK(scene.world.intersect(primary, Inf, entry));
        CHECK(entry.prim == 0);
        CHECK(entry.error >= std::abs(entry.p.z));
        const Vec3 direction = normalize(Vec3{.018972048877950398,.5914526806867747,.80611648529837221});
        CHECK(scene.world.intersect(scene.world.spawnRay(entry, direction, inside), Inf, exit));
        CHECK(exit.boundary == boundary && exit.prim == 1);
        CHECK_NEAR(exit.t, gap/direction.z, 1e-18);  // distance from the native departing plane
    }
}

TEST(contact_null_film_does_not_revisit_past_entry) {
    for (double thickness : {1e-15, 1e-10}) {
        Scene scene;
        scene.world.env.skyModel = Environment::Sky::Uniform;
        scene.world.env.zenith = Spectrum::constant(1);
        Medium absorbing;
        absorbing.name = "film";
        absorbing.absorption = Spectrum::constant(1 / thickness);
        scene.world.addMedium(absorbing);
        BodyMaterial material;
        material.transparent = true;
        material.medium = "film";
        material.optics = scene.world.nullOptics();
        buildBox(scene.world, "film", {1,1,thickness}, material, -1, Transform::translate({0,0,.01}));
        scene.world.build();
        Tracer tracer(scene, 4);
        TransportStats stats;
        auto wavelengths = Wavelengths::sample(.5);
        Rng rng(1,2);
        const Spec4 radiance = tracer.radiance(Ray{{0,0,1},{0,0,-1}}, 0, wavelengths, rng, stats);
        for (int i = 0; i < NW; ++i) CHECK_NEAR(radiance[i], std::exp(-1), 1e-8);
        CHECK(stats.segments == 3);
        CHECK(stats.inconsistencies == 0 && stats.leaks == 0);
    }
}

TEST(contact_authored_sensor_rejects_strictly_past_cover) {
    std::vector<double> open;
    for (const char* gap : {"0", "0.000000000000001", "0.0000000000001", "0.000000001", "0.001"}) {
        Scene scene = loadSceneWithEdits("benchmarks/regressions/sensor_opaque_contact.owe",
                                         {std::string("Screen.position=(0,0,") + gap + ")"});
        auto settings = scene.render;
        settings.backend = "cpu";
        settings.threads = 1;
        auto renderer = makeRenderer(scene, 0, settings);
        renderer->runPass(64);
        const Image image = renderer->resolve();
        if (std::string(gap) == "0") {
            for (double value : image.xyz) CHECK(value == 0);
        } else {
            CHECK_REL(image.meanY(), Pi, .02);
            if (open.empty()) open = image.xyz;
            else CHECK(image.xyz == open);  // same RNG, all strictly positive clearances see the same sky
        }
        CHECK(renderer->stats().inconsistencies == 0 && renderer->stats().leaks == 0);
    }
}
