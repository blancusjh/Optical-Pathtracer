#include "owe/backends/gpu/gpu_accel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

#include "owe/scene/geometry.hpp"

namespace owe::gpu {

namespace {

constexpr VkDeviceSize kTriangleBytes = sizeof(GTriangle);

// A boundary's shape bounds in its own frame, rounded outward generously: the hardware reaches the
// frame with its own float transform, so the box must also admit rays our transform places just
// inside the surface. A spurious candidate costs one analytic test; a missed one would lose a hit.
VkAabbPositionsKHR localBox(const Shape& shape) {
    AABB b = shape.bounds();
    double lo[3] = {b.lo.x, b.lo.y, b.lo.z}, hi[3] = {b.hi.x, b.hi.y, b.hi.z};
    double size = 0, mag = 0;
    for (int a = 0; a < 3; ++a) {
        lo[a] = std::clamp(lo[a], -1e30, 1e30);
        hi[a] = std::clamp(hi[a], -1e30, 1e30);
        size = std::max(size, hi[a] - lo[a]);
        mag = std::max({mag, std::abs(lo[a]), std::abs(hi[a])});
    }
    const double pad = 1e-3 * size + 1e-6 * mag + 1e-9;
    auto down = [](double x) { float f = float(x); return double(f) > x ? std::nextafter(f, -INFINITY) : f; };
    auto up = [](double x) { float f = float(x); return double(f) < x ? std::nextafter(f, INFINITY) : f; };
    return {down(lo[0] - pad), down(lo[1] - pad), down(lo[2] - pad), up(hi[0] + pad), up(hi[1] + pad), up(hi[2] + pad)};
}

VkAccelerationStructureBuildSizesInfoKHR buildSizes(VkDevice dev, const VkAccelerationStructureBuildGeometryInfoKHR& info,
                                                    uint32_t primitives) {
    VkAccelerationStructureBuildSizesInfoKHR sz{};
    sz.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primitives, &sz);
    return sz;
}

// Serialises acceleration-structure builds (and their scratch memory) against each other, and
// makes their results visible to later builds and to the kernels.
void accelBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                       VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
}

VkDeviceSize alignUp(VkDeviceSize x, VkDeviceSize a) { return (x + a - 1) / a * a; }

}  // namespace

HardwareScene::HardwareScene(Context& ctx, const World& world, const GpuScene& scene, const Buffer& triangles) : ctx_(ctx) {
    VkDevice dev = ctx_.device();
    const size_t nb = world.boundaries().size();
    const VkDeviceSize align = ctx_.scratchAlignment();

    // One BLAS per mesh group, and one per distinct analytic box (the 3500 stars share one).
    struct Pending {
        VkAccelerationStructureGeometryKHR geometry{};
        VkAccelerationStructureBuildRangeInfoKHR range{};
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        bool triangles = false;
    };
    std::vector<Pending> pending;
    std::vector<VkAabbPositionsKHR> boxes;
    std::map<std::vector<float>, uint32_t> boxIndex;
    records_ = scene.topRecord;
    instanceBlas_.resize(records_.size());
    instanceMask_.resize(records_.size());
    std::vector<uint32_t> groupBlas(scene.groups->groups.size(), ~0u);
    for (size_t j = 0; j < records_.size(); ++j) {
        const uint32_t r = records_[j];
        if (r >= nb) {  // a mesh group
            const size_t k = r - nb;
            const MeshGroup& g = scene.groups->groups[k];
            instanceMask_[j] = g.null ? kMaskNull : kMaskMatter;
            if (groupBlas[k] == ~0u) {
                Pending p;
                p.triangles = true;
                p.geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                p.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                p.geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
                auto& t = p.geometry.geometry.triangles;
                t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                t.vertexData.deviceAddress = ctx_.address(triangles) + VkDeviceSize(g.triBase) * kTriangleBytes;
                t.vertexStride = 16;
                t.maxVertex = std::max(1u, 3 * g.triCount) - 1;
                t.indexType = VK_INDEX_TYPE_NONE_KHR;
                p.range.primitiveCount = g.triCount;
                groupBlas[k] = uint32_t(pending.size());
                pending.push_back(p);
            }
            instanceBlas_[j] = groupBlas[k];
        } else {  // an analytic boundary: one box in its own frame
            const Boundary& b = world.boundaries()[r];
            instanceMask_[j] = passable(world.optics()[b.optics].type) ? kMaskNull : kMaskMatter;
            VkAabbPositionsKHR box = localBox(*b.shape);
            std::vector<float> key = {box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ};
            auto it = boxIndex.find(key);
            if (it == boxIndex.end()) {
                Pending p;
                p.geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                p.geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
                p.geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
                p.geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
                p.range.primitiveCount = 1;
                p.range.primitiveOffset = uint32_t(boxes.size() * sizeof(VkAabbPositionsKHR));
                it = boxIndex.emplace(key, uint32_t(pending.size())).first;
                boxes.push_back(box);
                pending.push_back(p);
            }
            instanceBlas_[j] = it->second;
        }
    }
    if (!boxes.empty()) {
        aabbs_ = ctx_.createDeviceBuffer(boxes.size() * sizeof(VkAabbPositionsKHR));
        ctx_.upload(aabbs_, boxes.data(), boxes.size() * sizeof(VkAabbPositionsKHR));
        for (Pending& p : pending)
            if (!p.triangles) p.geometry.geometry.aabbs.data.deviceAddress = ctx_.address(aabbs_);
    }

    // Sizes, storage, and one scratch buffer: every box BLAS built together, each triangle BLAS alone.
    auto info = [](const VkAccelerationStructureGeometryKHR* g) {
        VkAccelerationStructureBuildGeometryInfoKHR bi{};
        bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = g;
        return bi;
    };
    VkDeviceSize boxScratch = 0, maxScratch = 0;
    blas_.resize(pending.size());
    for (size_t i = 0; i < pending.size(); ++i) {
        Pending& p = pending[i];
        auto bi = info(&p.geometry);
        p.sizes = buildSizes(dev, bi, p.range.primitiveCount);
        blas_[i] = ctx_.createAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, p.sizes.accelerationStructureSize);
        if (p.triangles) maxScratch = std::max(maxScratch, p.sizes.buildScratchSize);
        else boxScratch += alignUp(p.sizes.buildScratchSize, align);
    }
    // The TLAS reuses the scratch buffer.
    host_.resize(records_.size());
    VkAccelerationStructureGeometryKHR tg{};
    tg.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tg.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR ti = info(&tg);
    ti.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    auto tsz = buildSizes(dev, ti, uint32_t(records_.size()));
    maxScratch = std::max({maxScratch, boxScratch, tsz.buildScratchSize});
    scratch_ = ctx_.createDeviceBuffer(maxScratch + align);
    const VkDeviceAddress scratchBase = alignUp(ctx_.address(scratch_), align);

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> boxInfos;
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> boxRanges;
    VkDeviceSize offset = 0;
    for (size_t i = 0; i < pending.size(); ++i) {
        if (pending[i].triangles) continue;
        auto bi = info(&pending[i].geometry);
        bi.dstAccelerationStructure = blas_[i].handle;
        bi.scratchData.deviceAddress = scratchBase + offset;
        offset += alignUp(pending[i].sizes.buildScratchSize, align);
        boxInfos.push_back(bi);
        boxRanges.push_back(&pending[i].range);
    }
    if (!boxInfos.empty())
        ctx_.submit([&](VkCommandBuffer cmd) {
            vkCmdBuildAccelerationStructuresKHR(cmd, uint32_t(boxInfos.size()), boxInfos.data(), boxRanges.data());
            accelBarrier(cmd);
        });
    for (size_t i = 0; i < pending.size(); ++i) {
        if (!pending[i].triangles || pending[i].range.primitiveCount == 0) continue;
        auto bi = info(&pending[i].geometry);
        bi.dstAccelerationStructure = blas_[i].handle;
        bi.scratchData.deviceAddress = scratchBase;
        const VkAccelerationStructureBuildRangeInfoKHR* range = &pending[i].range;
        ctx_.submit([&](VkCommandBuffer cmd) {
            accelBarrier(cmd);
            vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bi, &range);
            accelBarrier(cmd);
        });
    }

    instances_ = ctx_.createDeviceBuffer(std::max<size_t>(1, records_.size()) * sizeof(VkAccelerationStructureInstanceKHR));
    tlas_ = ctx_.createAccel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tsz.accelerationStructureSize);
    for (size_t j = 0; j < records_.size(); ++j) {
        VkAccelerationStructureInstanceKHR& in = host_[j];
        in = {};
        in.instanceCustomIndex = records_[j];
        in.mask = instanceMask_[j];
        in.instanceShaderBindingTableRecordOffset = 0;
        in.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        in.accelerationStructureReference = blas_[instanceBlas_[j]].address;
    }
    rebase(scene);
}

HardwareScene::~HardwareScene() {
    vkDeviceWaitIdle(ctx_.device());
    ctx_.destroy(tlas_);
    for (Accel& a : blas_) ctx_.destroy(a);
    ctx_.destroy(aabbs_);
    ctx_.destroy(instances_);
    ctx_.destroy(scratch_);
}

void HardwareScene::rebase(const GpuScene& scene) {
    // Instance transforms are the records' own camera-relative frames (rows [R | t − camera]).
    for (size_t j = 0; j < records_.size(); ++j)
        std::memcpy(host_[j].transform.matrix, scene.boundaries[records_[j]].toWorld, sizeof(float) * 12);
    if (!host_.empty()) ctx_.upload(instances_, host_.data(), host_.size() * sizeof(VkAccelerationStructureInstanceKHR));
    buildTlas();
}

void HardwareScene::buildTlas() {
    VkAccelerationStructureGeometryKHR tg{};
    tg.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tg.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tg.geometry.instances.arrayOfPointers = VK_FALSE;
    tg.geometry.instances.data.deviceAddress = ctx_.address(instances_);
    VkAccelerationStructureBuildGeometryInfoKHR bi{};
    bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = &tg;
    bi.dstAccelerationStructure = tlas_.handle;
    bi.scratchData.deviceAddress = alignUp(ctx_.address(scratch_), ctx_.scratchAlignment());
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = uint32_t(records_.size());
    const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;
    ctx_.submit([&](VkCommandBuffer cmd) {
        accelBarrier(cmd);
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bi, &pr);
        accelBarrier(cmd);
    });
}

}  // namespace owe::gpu
