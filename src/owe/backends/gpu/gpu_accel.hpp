// Hardware ray traversal (VK_KHR_ray_query) for devices that have it: the same top-level entries as
// the software hierarchy, as acceleration structures. A mesh group is a triangle BLAS built straight
// from the kernels' triangle buffer (48-byte records read as three 16-byte vertices); an analytic
// boundary is a one-box BLAS whose candidates the kernel resolves with its own intersection code,
// so every precision measure of the software path (local frames, quadric re-solves, the sliver-safe
// triangle distance) is kept. Instances are camera-relative and carry the boundary record index;
// null boundaries have their own mask so that shadow rays can ignore them.
#pragma once

#include <vector>

#include "owe/backends/gpu/gpu_scene.hpp"
#include "owe/backends/gpu/vulkan_context.hpp"

namespace owe::gpu {

constexpr uint32_t kMaskMatter = 1, kMaskNull = 2;

class HardwareScene {
public:
    HardwareScene(Context& ctx, const World& world, const GpuScene& scene, const Buffer& triangles);
    ~HardwareScene();
    HardwareScene(const HardwareScene&) = delete;
    HardwareScene& operator=(const HardwareScene&) = delete;

    // The camera moved: instances follow the rebased records; the TLAS is rebuilt in place.
    void rebase(const GpuScene& scene);
    VkAccelerationStructureKHR tlas() const { return tlas_.handle; }

private:
    void buildTlas();

    Context& ctx_;
    std::vector<Accel> blas_;
    std::vector<uint32_t> instanceBlas_;   // per top-level entry: index into blas_
    std::vector<uint32_t> instanceMask_;
    Buffer aabbs_, instances_, scratch_;
    Accel tlas_;
    std::vector<uint32_t> records_;        // top-level entries (boundary records)
    std::vector<VkAccelerationStructureInstanceKHR> host_;
};

}  // namespace owe::gpu
