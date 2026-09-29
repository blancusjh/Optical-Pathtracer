// Internals of the GPU backend that need no GPU: the flattened scene the kernels read. Its physics
// is tested with every other backend's in tests/test_backends.cpp.
#include <algorithm>
#include <cstring>

#include "../apps/navigation.hpp"
#include "check.hpp"
#include "owe/backends/gpu/gpu_scene.hpp"
#include "owe/loader/scene_loader.hpp"

using namespace owe;

TEST(gpu_scene_layout_is_flat_and_aligned) {
    // Runs without a GPU: the flattened records must match the kernels' 16-byte layout.
    static_assert(sizeof(gpu::GNode) == 32);
    static_assert(sizeof(gpu::GBoundary) == 176);
    static_assert(sizeof(gpu::GTriangle) == 48);
    static_assert(sizeof(gpu::GMedium) == 48);
    static_assert(sizeof(gpu::GSpectrum) == 32);
    static_assert(sizeof(gpu::GOptics) == 112);
    static_assert(sizeof(gpu::GLight) == 16);
    static_assert(sizeof(gpu::GGlobals) % 16 == 0);
    Scene sc = loadScene("scenes/the_telescope.owe");
    int det = sc.findDetector("Eyepiece");
    gpu::GpuScene g = gpu::flattenScene(sc, det, sc.settingsFor(det));
    // Records: every world boundary at its own index, then one per mesh group.
    const size_t nb = sc.world.boundaries().size();
    CHECK(g.boundaries.size() == nb + g.groups->groups.size());
    CHECK(g.lights.size() == sc.world.lights().size());
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
        for (uint32_t t = grp.triBase; t < grp.triBase + grp.triCount; ++t) {
            uint32_t owner;
            std::memcpy(&owner, &g.triangles[t].v1[3], 4);
            CHECK(std::find(grp.members.begin(), grp.members.end(), owner) != grp.members.end());
        }
    }
    // Camera-relative: the observer sits at the origin, and every boundary box is finite or bounded.
    CHECK(g.globals.detPos[0] == 0 && g.globals.detPos[1] == 0 && g.globals.detPos[2] == 0);
    for (const auto& b : g.boundaries) CHECK(b.boxLo[0] <= b.boxHi[0] && b.boxLo[1] <= b.boxHi[1] && b.boxLo[2] <= b.boxHi[2]);
}

// Moving the eye re-expresses the world relative to it (camera-relative float precision) without
// re-uploading meshes: the rebased records equal a fresh export, and triangles stay untouched.
TEST(gpu_camera_rebase_matches_fresh_export_including_meshes) {
    Scene sc = loadScene("scenes/the_temple.owe");
    int d = sc.findDetector("Wide");
    auto& eye = dynamic_cast<IdealObserver&>(*sc.detectors[size_t(d)]);
    auto g = gpu::flattenScene(sc, d, sc.render);
    auto triangles = g.triangles;
    NavigationCamera c = NavigationCamera::from(eye);
    for (int i = 0; i < 3; ++i) {
        c.pan(0.123, -0.09);
        c.turn(0.05, 0.02, false);
        c.apply(eye, sc.world);
        gpu::updateObserver(g, sc.world, eye);
        auto fresh = gpu::flattenScene(sc, d, sc.render);
        CHECK(std::memcmp(&g.globals, &fresh.globals, sizeof(g.globals)) == 0);
        CHECK(g.nodes.size() == fresh.nodes.size());
        CHECK(std::memcmp(g.nodes.data(), fresh.nodes.data(), g.nodes.size() * sizeof(gpu::GNode)) == 0);
        CHECK(std::memcmp(g.boundaries.data(), fresh.boundaries.data(), g.boundaries.size() * sizeof(gpu::GBoundary)) == 0);
        CHECK(std::memcmp(g.triangles.data(), triangles.data(), triangles.size() * sizeof(gpu::GTriangle)) == 0);
    }
}
