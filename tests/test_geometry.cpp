// Geometry kernel: exact surfaces, conic foci, acceleration structures.
#include "check.hpp"
#include "owe/builders.hpp"
#include "owe/geometry.hpp"
#include "owe/sampling.hpp"
#include "owe/transport.hpp"

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
