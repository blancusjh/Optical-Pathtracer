// Internals of the GPU backend that need no GPU: the flattened scene the kernels read. Its physics
// is tested with every other backend's in tests/test_backends.cpp.
#include <algorithm>
#include <cstring>

#include "../apps/navigation.hpp"
#include "check.hpp"
#include "owe/backends/gpu/gpu_scene.hpp"
#include "owe/backends/gpu/gpu_push.hpp"
#include "owe/backends/gpu/gpu_film.hpp"
#include "owe/backends/registry.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/scene/builders.hpp"

using namespace owe;

TEST(gpu_scene_layout_is_flat_and_aligned) {
    // Runs without a GPU: the flattened records must match the kernels' 16-byte layout.
    static_assert(sizeof(gpu::GNode) == 32);
    static_assert(sizeof(gpu::GBoundary) == 176);
    static_assert(sizeof(gpu::GTriangle) == 48);
    static_assert(sizeof(gpu::GMedium) == 48);
    static_assert(sizeof(gpu::GSpectrum) == 32);
    static_assert(sizeof(gpu::GOptics) == 144);
    static_assert(sizeof(gpu::GLight) == 16);
    static_assert(sizeof(gpu::GGlobals) % 16 == 0);
    Scene sc = loadScene("scenes/the_telescope.owe");
    int det = sc.findDetector("Eyepiece");
    gpu::GpuScene g = gpu::flattenScene(sc, det, sc.settingsFor(det));
    // Records: every world boundary at its own index, then one per mesh group.
    const size_t nb = sc.world.boundaries().size();
    CHECK(g.boundaries.size() == nb + g.groups->groups.size());
    CHECK(g.lights.size() == sc.world.lights().size());
    // Inner nodes retain their axis and encode the complete contiguous preorder subtree.
    // Check every child span, including relocated mesh hierarchies, without using a GPU.
    auto nodeSpan = [&](size_t i) -> size_t {
        const auto& node = g.nodes[i];
        return (node.b & gpu::kNodeInner) ? (node.b & ~gpu::kNodeInner) >> 2 : 1;
    };
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const auto& node = g.nodes[i];
        if ((node.b & gpu::kNodeInner) == 0) continue;
        CHECK((node.b & 3u) <= 2);
        CHECK(i + 1 < g.nodes.size());
        CHECK(node.a > i + 1 && node.a < g.nodes.size());
        CHECK(node.a == i + 1 + nodeSpan(i + 1));
        CHECK(nodeSpan(i) == 1 + nodeSpan(i + 1) + nodeSpan(node.a));
        CHECK(i + nodeSpan(i) <= g.nodes.size());
    }
    CHECK(nodeSpan(0) == g.top->nodeCount());
    // Every mesh belongs to exactly one group; the top level holds every other boundary and every
    // group exactly once; triangles name mesh boundaries of their group.
    std::vector<int> grouped(nb, 0), listed(g.boundaries.size(), 0);
    for (const auto& grp : g.groups->groups)
        for (uint32_t m : grp.members) grouped[m]++;
    for (uint32_t r : g.topRecord) listed[r]++;
    for (size_t i = 0; i < nb; ++i) {
        bool mesh = dynamic_cast<const MeshShape*>(sc.world.boundaries()[i].shape.get()) != nullptr;
        CHECK(grouped[i] == (mesh ? 1 : 0));
        CHECK(listed[i] == (mesh ? 0 : 1));
    }
    for (size_t k = 0; k < g.groups->groups.size(); ++k) {
        CHECK(listed[nb + k] == 1);
        const auto& grp = g.groups->groups[k];
        const size_t root = g.boundaries[nb + k].meshNodes;
        const size_t end = k + 1 < g.groups->groups.size()
                               ? g.boundaries[nb + k + 1].meshNodes : g.nodes.size();
        CHECK(root < end && end <= g.nodes.size());
        CHECK(root + nodeSpan(root) == end);
        for (uint32_t owner : grp.members) {
            CHECK(g.boundaries[owner].meshNodes == g.boundaries[nb + k].meshNodes);
            CHECK(g.boundaries[owner].meshNodes < g.nodes.size());
            for (int j = 0; j < 12; ++j)
                CHECK(g.boundaries[owner].toLocal[j] == g.boundaries[nb + k].toLocal[j]);
        }
        for (uint32_t t = grp.triBase; t < grp.triBase + grp.triCount; ++t) {
            const uint32_t owner = g.triangles[t].owner;
            CHECK(std::find(grp.members.begin(), grp.members.end(), owner) != grp.members.end());
        }
    }
    // Camera-relative: the observer sits at the origin, and every boundary box is finite or bounded.
    CHECK(g.globals.detPos[0] == 0 && g.globals.detPos[1] == 0 && g.globals.detPos[2] == 0);
    for (const auto& b : g.boundaries) CHECK(b.boxLo[0] <= b.boxHi[0] && b.boxLo[1] <= b.boxHi[1] && b.boxLo[2] <= b.boxHi[2]);
}

TEST(gpu_planetary_sphere_frame_preserves_height_and_observer_rebasing) {
    Scene sc = loadScene("benchmarks/regressions/planetary_ground.owe");
    const int det = sc.findDetector("Eye");
    gpu::GpuScene g = gpu::flattenScene(sc, det, sc.settingsFor(det));
    const auto& earth = g.boundaries[0];
    const size_t at = size_t(earth.data) * 4;
    CHECK(g.shapeData[at + 1] == 1);
    // Near pole translation is centimetres/metres; native shape centre stays planetary-size.
    CHECK(std::abs(earth.toLocal[11]) > 6e6);
    CHECK(std::abs(g.shapeData[at + 4 + 11]) < 2);
    auto& eye = static_cast<IdealObserver&>(*sc.detectors[size_t(det)]);
    eye.position.z += 0.1;
    eye.prepare(sc.world);
    gpu::updateObserver(g, sc.world, eye);
    CHECK(g.sphereFramesChanged);
    CHECK(g.shapeData[at + 1] == 1);
    CHECK_NEAR(g.shapeData[at + 4 + 11], 0.7, 1e-6);
    gpu::updateObserver(g, sc.world, eye);
    CHECK(!g.sphereFramesChanged);
    // Leaving the surface neighbourhood disables the auxiliary frame, including its upload.
    eye.position = {0,0,-6371000.2};
    eye.prepare(sc.world);
    gpu::updateObserver(g, sc.world, eye);
    CHECK(g.sphereFramesChanged);
    CHECK(g.shapeData[at + 1] == 0);
    gpu::updateObserver(g, sc.world, eye);
    CHECK(!g.sphereFramesChanged);
}

TEST(gpu_planetary_sphere_preserves_nearby_occlusion) {
    const Backend* gpuBackend = findBackend("gpu");
    std::string why;
    if (!gpuBackend || !gpuBackend->available(&why)) {
        std::fprintf(stderr, "    [gpu] skipped: %s\n", why.c_str());
        return;
    }
    // A luminous sheet is 5 cm below the real surface. Losing the centre's fractional part can
    // discard the near root and reveal it. Visibility is an analytic oracle, independent of CPU.
    for (Vec3 n : {Vec3{0,0,1}, normalize(Vec3{1,2,3})}) {
        Scene sc;
        SurfaceOptics black;
        black.type = SurfaceType::Absorber;
        BodyMaterial material;
        material.optics = sc.world.addOptics(black);
        buildSphere(sc.world, "Earth", 6371000, material, -1,
                    Transform::translate(-n * (6371000 + 0.2)));
        Emission emission;
        emission.radiance = Spectrum::constant(1);
        buildSheet(sc.world, "Witness", 0.1, 0.1, material.optics, sc.world.addEmission(emission), -1,
                   Transform::translate(n * -0.25) * Transform::alignZ(n));
        auto eye = std::make_unique<IdealObserver>();
        eye->name = "Eye";
        eye->position = {0,0,0};
        eye->lookAt = n * -0.2;
        eye->up = std::abs(n.z) > 0.9 ? Vec3{0,1,0} : Vec3{0,0,1};
        eye->width = eye->height = 8;
        eye->fovY = radians(1);
        eye->pupilRadius = 0;
        sc.detectors.push_back(std::move(eye));
        sc.build();
        RenderSettings settings;
        settings.backend = "gpu";
        auto renderer = makeRenderer(sc, 0, settings);
        renderer->runPass(16);
        CHECK_NEAR(renderer->resolve().meanY(), 0, 1e-8);
        CHECK(renderer->stats().inconsistencies == 0);
        CHECK(renderer->stats().leaks == 0);
        // Moving the observer exercises the conditional shape-data upload in resetObserver.
        auto& observer = static_cast<IdealObserver&>(*sc.detectors[0]);
        observer.position = n * 0.1;
        observer.prepare(sc.world);
        CHECK(renderer->resetObserver());
        renderer->runPass(16);
        CHECK_NEAR(renderer->resolve().meanY(), 0, 1e-8);
        CHECK(renderer->stats().inconsistencies == 0);
    }
}

TEST(gpu_planetary_sphere_keeps_stable_distant_origin_roots) {
    const Backend* gpuBackend = findBackend("gpu");
    std::string why;
    if (!gpuBackend || !gpuBackend->available(&why)) {
        std::fprintf(stderr, "    [gpu] skipped: %s\n", why.c_str());
        return;
    }
    for (bool planetary : {false, true}) {
        const double radius = planetary ? 6371000 : 1000;
        Scene sc;
        sc.world.env.skyModel = Environment::Sky::Uniform;
        sc.world.env.zenith = Spectrum::constant(1);
        SurfaceOptics diffuse;
        diffuse.type = SurfaceType::Diffuse;
        diffuse.reflectance = Spectrum::constant(0.3);
        BodyMaterial ground;
        ground.optics = sc.world.addOptics(diffuse);
        buildSphere(sc.world, "Earth", radius, ground, -1, Transform::translate({0,0,-radius-0.2}));
        if (planetary)
            buildStop(sc.world, "PinholeCeiling", 0.01, 100, -1, Transform::translate({0,0,-0.1}));
        SurfaceOptics mirror;
        mirror.type = SurfaceType::Mirror;
        mirror.reflectance = Spectrum::constant(1);
        buildSheet(sc.world, "DistantMirror", 100, 100, sc.world.addOptics(mirror), -1, -1,
                   Transform::translate({0,0,planetary ? 1e8 : 1e7}) * Transform::alignZ({0,0,-1}));
        auto eye = std::make_unique<IdealObserver>();
        eye->name = "Eye";
        eye->position = {0,0,0};
        eye->lookAt = {0,0,1};
        eye->up = {0,1,0};
        eye->width = eye->height = 8;
        eye->fovY = radians(planetary ? 1e-9 : 0.0001);
        eye->pupilRadius = 0;
        sc.detectors.push_back(std::move(eye));
        sc.build();
        // Detector is near the surface, enabling the auxiliary frame, but its reflected ray begins
        // 10000 radii away in the small case. The Earth case also checks the same surface position:
        // an old far-native hit at z=0 sees open sky, whereas the true z=-.2 point lies under the roof.
        auto g = gpu::flattenScene(sc, 0, sc.settingsFor(0));
        CHECK(g.shapeData[size_t(g.boundaries[0].data) * 4 + 1] == 1);
        RenderSettings settings;
        settings.backend = "gpu";
        auto renderer = makeRenderer(sc, 0, settings);
        renderer->runPass(128);
        const double expected = planetary ? 0.3 * 0.01 * 0.01 / (0.1 * 0.1 + 0.01 * 0.01) : 0.3;
        std::fprintf(stderr, "    far-origin %s surface meanY=%.8g, expected=%.8g\n",
                     planetary ? "Earth/pinhole" : "sphere", renderer->resolve().meanY(), expected);
        CHECK_NEAR(renderer->resolve().meanY(), expected, planetary ? 0.0015 : 0.01);
        CHECK(renderer->stats().inconsistencies == 0);
        CHECK(renderer->stats().leaks == 0);
    }
}


TEST(gpu_contact_glass_support_preserves_contact_and_resolved_gaps) {
    const Backend* backend = findBackend("gpu");
    std::string why;
    if (!backend || !backend->available(&why)) {
        std::fprintf(stderr, "    [gpu] skipped: %s\n", why.c_str());
        return;
    }
    for (double height : {.05, .0500001, .0501, .0499}) {
        Scene scene = loadScene("benchmarks/regressions/contact_glass_support.owe");
        int glass = scene.world.findBody("Glass");
        scene.world.bodies()[glass].xf.t.z = height;
        auto& eye = static_cast<IdealObserver&>(*scene.detectors[0]);
        eye.width = eye.height = 16;
        scene.build();
        RenderSettings settings = scene.render;
        settings.backend = "gpu";
        settings.integrator = "path";
        auto renderer = makeRenderer(scene, 0, settings);
        renderer->runPass(32);
        std::fprintf(stderr, "    [gpu contact %.9g] mean %.9g, inconsistencies %llu, leaks %llu\n",
            height, renderer->resolve().meanY(), (unsigned long long)renderer->stats().inconsistencies,
            (unsigned long long)renderer->stats().leaks);
        if (height >= .05) {
            CHECK(renderer->stats().inconsistencies == 0);
            CHECK(renderer->stats().leaks == 0);
            CHECK(renderer->resolve().meanY() > .1);
        } else {
            CHECK(renderer->stats().inconsistencies > 0);
        }
    }
}

TEST(gpu_contact_spawn_preserves_resolved_thin_same_boundary_exit) {
    const Backend* backend = findBackend("gpu");
    if (!backend || !backend->available()) return;
    Scene scene;
    scene.world.env.skyModel = Environment::Sky::Uniform;
    scene.world.env.zenith = Spectrum::constant(1);
    Medium absorbing;
    absorbing.name = "thin-attenuator";
    absorbing.absorption = Spectrum::constant(1e7);
    scene.world.addMedium(absorbing);
    BodyMaterial material;
    material.transparent = true;
    material.medium = absorbing.name;
    material.optics = scene.world.nullOptics();
    buildBox(scene.world, "film", {1,1,1e-7}, material, -1, Transform::translate({0,0,.01}));
    auto eye = std::make_unique<IdealObserver>();
    eye->name = "Eye";
    eye->position = {0,0,0};
    eye->lookAt = {0,0,.01};
    eye->width = eye->height = 8;
    eye->fovY = radians(.1);
    eye->pupilRadius = 0;
    scene.detectors.push_back(std::move(eye));
    scene.build();
    RenderSettings settings;
    settings.backend = "gpu";
    settings.integrator = "path";
    auto renderer = makeRenderer(scene, 0, settings);
    renderer->runPass(32);
    std::fprintf(stderr, "    [gpu thin contact] mean %.9g expected %.9g, inconsistencies %llu, leaks %llu\n",
        renderer->resolve().meanY(), std::exp(-1), (unsigned long long)renderer->stats().inconsistencies,
        (unsigned long long)renderer->stats().leaks);
    // Thickness is resolved in z although below the conservative frame error (large x/y extent).
    // The opposite crossing orientation must remain eligible on the SAME mesh boundary.
    CHECK_NEAR(renderer->resolve().meanY(), std::exp(-1), .015);
    CHECK(renderer->stats().inconsistencies == 0);
    CHECK(renderer->stats().leaks == 0);
}


TEST(gpu_contact_spawn_preserves_distinct_parallel_mesh_face) {
    const Backend* backend = findBackend("gpu");
    if (!backend || !backend->available()) return;
    for (bool covered : {false, true}) {
        Scene scene;
        scene.world.env.skyModel = Environment::Sky::Uniform;
        scene.world.env.zenith = Spectrum::constant(1);
        SurfaceOptics mirror;
        mirror.type = SurfaceType::Mirror;
        mirror.reflectance = Spectrum::constant(1);
        mirror.backAbsorbs = true;
        auto optics = scene.world.addOptics(mirror);
        MeshData mesh;
        mesh.addQuad({-.5,-.5,0}, {.5,-.5,0}, {.5,.5,0}, {-.5,.5,0});
        if (covered) mesh.addQuad({-.5,-.5,1e-7}, {.5,-.5,1e-7}, {.5,.5,1e-7}, {-.5,.5,1e-7});
        auto body = scene.world.addBody("parallel-open-faces", "mesh", -1, Transform{});
        scene.world.addBoundary(body, "surface", std::make_shared<MeshShape>(mesh.positions, mesh.triangles), Transform{}, 0, 0, optics);
        auto eye = std::make_unique<IdealObserver>();
        eye->name = "Eye";
        eye->position = {0,0,5e-8};
        eye->lookAt = {0,0,-1};
        eye->width = eye->height = 4;
        eye->fovY = radians(.1);
        eye->pupilRadius = 0;
        scene.detectors.push_back(std::move(eye));
        scene.build();
        RenderSettings settings;
        settings.backend = "gpu";
        settings.integrator = "path";
        auto renderer = makeRenderer(scene, 0, settings);
        renderer->runPass(32);
        const double y = renderer->resolve().meanY();
        std::fprintf(stderr, "    [parallel mesh %s] mean %.9g, issues %llu, leaks %llu\n",
                     covered ? "covered" : "open", y,
                     (unsigned long long)renderer->stats().inconsistencies,
                     (unsigned long long)renderer->stats().leaks);
        if (covered) CHECK_NEAR(y, 0, 0);
        else CHECK_NEAR(y, 1, .02);
        CHECK(renderer->stats().inconsistencies == 0);
        CHECK(renderer->stats().leaks == 0);
    }
}


TEST(gpu_contact_slivers_keep_rays_entering_beside_a_mesh_rim) {
    // A glass slab whose flat back is a fan of 150 mm × 0.6 mm slivers from its centre (as the
    // Invisible Window's panel): a ray entering a side face micrometres from the back must leave
    // through that back. In float, Möller–Trumbore's u + v ≤ 1 rejected the sliver there and the
    // ray escaped from inside the glass (the panel's leaks); the watertight test resolves it.
    const Backend* backend = findBackend("gpu");
    if (!backend || !backend->available()) return;
    Scene scene;
    scene.world.env.skyModel = Environment::Sky::Uniform;
    scene.world.env.zenith = Spectrum::constant(1);
    const int n = 512;  // rim vertices per side
    const double h = .15, depth = .02;
    std::vector<Vec3> rim;
    for (int side = 0; side < 4; ++side)
        for (int k = 0; k < n; ++k) {
            const double s = -h + 2 * h * k / n;
            rim.push_back(side == 0 ? Vec3(s, -h, 0) : side == 1 ? Vec3(h, s, 0) : side == 2 ? Vec3(-s, h, 0) : Vec3(-h, -s, 0));
        }
    MeshData mesh;
    const Vec3 up(0, 0, depth);
    for (size_t k = 0; k < rim.size(); ++k) {
        const Vec3 a = rim[k], b = rim[(k + 1) % rim.size()];
        mesh.addTriangle({0, 0, 0}, b, a);           // back: the fan
        mesh.addTriangle(up, a + up, b + up);        // front
        mesh.addQuad(a, b, b + up, a + up);          // side
    }
    BodyMaterial glass;
    glass.transparent = true;
    glass.medium = "PMMA";
    glass.optics = scene.world.dielectricOptics();
    buildMesh(scene.world, "fan-slab", mesh, glass, -1, Transform{});
    // A pinhole eye grazing the side face x = h at 20°, 5 µm above the back.
    const Vec3 target(h, -.1, 5e-6), dir = normalize(Vec3(-std::sin(radians(20)), 0, -std::cos(radians(20))));
    auto eye = std::make_unique<IdealObserver>();
    eye->name = "Eye";
    eye->position = target - dir * .5;
    eye->lookAt = target;
    eye->up = {0, 1, 0};
    eye->width = eye->height = 16;
    eye->fovY = radians(5e-4);
    eye->pupilRadius = 0;
    scene.detectors.push_back(std::move(eye));
    scene.build();
    RenderSettings settings;
    settings.backend = "gpu";
    settings.integrator = "path";
    auto renderer = makeRenderer(scene, 0, settings);
    renderer->runPass(16);
    std::fprintf(stderr, "    [fan slab] issues %llu, leaks %llu\n", (unsigned long long)renderer->stats().inconsistencies,
                 (unsigned long long)renderer->stats().leaks);
    CHECK(renderer->stats().inconsistencies == 0);
    CHECK(renderer->stats().leaks == 0);
}

TEST(gpu_contact_rays_grazing_a_concave_fold_meet_its_far_face) {
    // A mirror block whose top folds up by 7.5° (as a torus mesh's hole side): a ray reflected off
    // the flat half micrometres before the fold rises at 0.06° less than the fold and meets the far
    // half tens of micrometres later, a few hundredths of a micrometre above its plane at the end of
    // the spawn offset. World coordinates 3 m from the eye round to 0.1 µm, so the ordinary
    // traversal started behind that face and the ray ran on inside the block (the Peacock filaments'
    // inconsistencies); the block's own frame resolves the whole segment.
    const Backend* backend = findBackend("gpu");
    if (!backend || !backend->available()) return;
    Scene scene;
    scene.world.env.skyModel = Environment::Sky::Uniform;
    scene.world.env.zenith = Spectrum::constant(1);
    const double L = 1e-3, H = 1e-3, W = 1e-3, fold = radians(7.5);
    const Vec3 profile[5] = {{-L, 0, -H}, {L, 0, -H}, {L, 0, L * std::tan(fold)}, {0, 0, 0}, {-L, 0, 0}};
    MeshData mesh;
    const Vec3 side(0, W, 0);
    for (int k = 0; k < 5; ++k) {
        const Vec3 a = profile[k], b = profile[(k + 1) % 5];
        mesh.addQuad(a - side, b - side, b + side, a + side);
    }
    for (const Vec3& s : {side * -1.0, side})
        for (int k = 1; k < 4; ++k) mesh.addTriangle(profile[0] + s, profile[k] + s, profile[k + 1] + s);
    orientOutward(mesh);
    SurfaceOptics mirror;
    mirror.type = SurfaceType::Mirror;
    mirror.reflectance = Spectrum::constant(1);
    BodyMaterial block;
    block.optics = scene.world.addOptics(mirror);
    // Tilted so that every coordinate of a point seen from 3 m is large (the GPU's world frame is
    // relative to the eye, and its rounding then reaches the fold's normal from every axis).
    const Transform xf = Transform::translate({.5, .6, 1.5}) * Transform::rotate(normalize(Vec3(1, -2, 3)), radians(50));
    buildMesh(scene.world, "folded-mirror", mesh, block, -1, xf);
    // A pinhole eye whose image spans the 2 µm of the flat half before the fold.
    const double rise = fold - radians(.06), distance = 3;
    const Vec3 target = xf.point({-1e-6, 0, 0}), dir = xf.vector({std::cos(rise), 0, -std::sin(rise)});
    auto eye = std::make_unique<IdealObserver>();
    eye->name = "Eye";
    eye->position = target - dir * distance;
    eye->lookAt = target;
    eye->up = xf.vector({0, 0, 1});
    eye->width = eye->height = 16;
    eye->fovY = 2e-6 * std::sin(rise) / distance;
    eye->pupilRadius = 0;
    scene.detectors.push_back(std::move(eye));
    scene.build();
    RenderSettings settings;
    settings.backend = "gpu";
    settings.integrator = "path";
    auto renderer = makeRenderer(scene, 0, settings);
    renderer->runPass(16);
    std::fprintf(stderr, "    [concave fold] issues %llu, leaks %llu\n", (unsigned long long)renderer->stats().inconsistencies,
                 (unsigned long long)renderer->stats().leaks);
    CHECK(renderer->stats().inconsistencies == 0);
    CHECK(renderer->stats().leaks == 0);
}

TEST(gpu_contact_spawn_preserves_nearest_distinct_face_in_one_mesh) {
    const Backend* backend = findBackend("gpu");
    if (!backend || !backend->available()) return;
    Scene scene;
    scene.world.env.skyModel = Environment::Sky::Uniform;
    scene.world.env.zenith = Spectrum::constant(1);
    Medium absorbing;
    absorbing.name = "three-films";
    absorbing.absorption = Spectrum::constant(1e7);
    scene.world.addMedium(absorbing);
    BodyMaterial material;
    material.transparent = true;
    material.medium = absorbing.name;
    material.optics = scene.world.nullOptics();
    MeshData mesh;
    // One boundary describes three closed, disjoint components in the same medium.
    // Far-first input deliberately avoids relying on traversal order for the next face.
    for (double center : {4e-7, 2e-7, 0.0}) {
        MeshData box;
        Vec3 c[8];
        for (int i = 0; i < 8; ++i)
            c[i] = {(i & 1) ? .5 : -.5, (i & 2) ? .5 : -.5, center + ((i & 4) ? 5e-8 : -5e-8)};
        box.addQuad(c[0], c[2], c[3], c[1]);
        box.addQuad(c[4], c[5], c[7], c[6]);
        box.addQuad(c[0], c[1], c[5], c[4]);
        box.addQuad(c[2], c[6], c[7], c[3]);
        box.addQuad(c[0], c[4], c[6], c[2]);
        box.addQuad(c[1], c[3], c[7], c[5]);
        orientOutward(box);
        mesh.append(box, Transform{});
    }
    buildMesh(scene.world, "disjoint-films", mesh, material, -1, Transform::translate({0,0,.01}));
    auto eye = std::make_unique<IdealObserver>();
    eye->name = "Eye";
    eye->position = {.125,.05,0};
    eye->lookAt = {.125,.05,.01};
    eye->width = eye->height = 4;
    eye->fovY = radians(.1);
    eye->pupilRadius = 0;
    scene.detectors.push_back(std::move(eye));
    scene.build();
    RenderSettings settings;
    settings.backend = "gpu";
    settings.integrator = "path";
    auto renderer = makeRenderer(scene, 0, settings);
    renderer->runPass(32);
    const double y = renderer->resolve().meanY();
    std::fprintf(stderr, "    [three films] mean %.9g expected %.9g, issues %llu, leaks %llu\n",
                 y, std::exp(-3), (unsigned long long)renderer->stats().inconsistencies,
                 (unsigned long long)renderer->stats().leaks);
    CHECK_NEAR(y, std::exp(-3), .015);
    CHECK(renderer->stats().inconsistencies == 0);
    CHECK(renderer->stats().leaks == 0);
}
