// Minimal Vulkan compute context: one device, one compute queue, storage buffers and one
// compute pipeline per kernel. Functions are loaded at run time through volk, so the engine
// links against no Vulkan library and runs wherever a driver (or MoltenVK) is installed.
#pragma once

#include <volk.h>

#include <array>
#include <cstddef>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "owe/render/backend.hpp"

namespace owe::gpu {

// True when a Vulkan loader and a usable compute device exist; `why` explains otherwise.
bool available(std::string* why = nullptr);
// Every Vulkan device, usable or not (the type says why not).
std::vector<DeviceInfo> devices();

void vkCheck(VkResult r, const char* what);

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // host-visible buffers stay mapped
};

// An acceleration structure (ray queries) and the buffer holding it.
struct Accel {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    Buffer storage;
    VkDeviceAddress address = 0;
};

constexpr uint32_t kNoBinding = ~0u;

class Context {
public:
    // Creates the instance and device (`device` < 0: the best one: discrete, then integrated, ...).
    explicit Context(int device);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    const DeviceInfo& info() const { return info_; }
    VkDevice device() const { return device_; }
    bool statistics() const { return statistics_; }  // OWE_GPU_STATS: report kernel statistics
    // Hardware ray traversal from compute shaders (VK_KHR_ray_query) is enabled on this device.
    bool rayQuery() const { return rayQuery_; }
    VkDeviceSize scratchAlignment() const { return scratchAlignment_; }
    VkDeviceSize maxStorageBufferRange() const { return props_.limits.maxStorageBufferRange; }
    // Pipelines compile through a cache kept on disk per device and driver (in the user's cache
    // directory; OWE_PIPELINE_CACHE=0 disables it), so a kernel compiles once rather than after
    // every eviction from the driver's own cache. Thread-safe.
    VkPipelineCache pipelineCache() const { return pipelineCache_; }
    // Writes the cache to disk if it grew since it was loaded or last saved, merged with what
    // other processes have written meanwhile.
    void savePipelineCache();

    Buffer createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags extraUsage = 0);  // storage + transfer, device-local
    VkDeviceAddress address(const Buffer& b) const;     // ray queries only
    Accel createAccel(VkAccelerationStructureTypeKHR type, VkDeviceSize size);
    void destroy(Accel& a);
    Buffer createHostBuffer(VkDeviceSize size, bool readback);  // mapped staging / readback
    void destroy(Buffer& b);
    // Copies host data into a device buffer (through a temporary staging buffer).
    void upload(const Buffer& dst, const void* data, size_t bytes);
    // The same at a byte offset into dst, through a staging buffer of at most 64 MB reused per slice.
    void uploadAt(const Buffer& dst, VkDeviceSize offset, const void* data, size_t bytes);

    // Records into the context's command buffer and submits it, waiting for completion.
    template <class F>
    void submit(F&& record, const char* label = "submission") {
        begin();
        record(cmd_);
        endAndWait(label);
    }

    // Bounded queue of submissions. Recording is immediate; resources must remain alive
    // until waitPending() or a later synchronous submit completes. Profiling stays synchronous.
    template <class F>
    void submitAsync(F&& record, const char* label = "submission") {
        if (profilePool_) { submit(std::forward<F>(record), label); return; }
        VkCommandBuffer cmd = beginAsync();
        record(cmd);
        endAsync();
    }
    void waitPending();

    // Optional diagnostic timestamps (OWE_GPU_PROFILE=1); inactive calls are no-ops.
    uint32_t beginKernelProfile(VkCommandBuffer cmd, const std::string& name);
    void endKernelProfile(VkCommandBuffer cmd, uint32_t query);

private:
    VkCommandBuffer beginAsync();
    void endAsync();
    void beginCommand(VkCommandBuffer cmd);                 // reset and begin a one-time recording
    void submitCommand(VkCommandBuffer cmd, VkFence fence);  // end and submit, signalling fence
    void begin();
    void endAndWait(const char* label);
    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback) const;
    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want,
                        VkMemoryPropertyFlags fallback);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    struct Submission {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool pending = false;
    };
    std::array<Submission, 3> submissions_{};
    size_t nextSubmission_ = 0;
    VkPhysicalDeviceMemoryProperties memProps_{};
    DeviceInfo info_;
    bool statistics_ = false;
    bool rayQuery_ = false;
    VkDeviceSize scratchAlignment_ = 256;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::string cachePath_;
    size_t cacheSavedBytes_ = 0;
    std::mutex cacheMutex_;
    VkPhysicalDeviceProperties props_{};
    static constexpr uint32_t kProfileQueries = 4096;
    VkQueryPool profilePool_ = VK_NULL_HANDLE;
    uint32_t timestampBits_ = 0, profileNext_ = 2;
    struct ProfileSpan { std::string name; uint32_t query; };
    struct ProfileTotal { uint64_t calls = 0; double gpuMs = 0, cpuMs = 0; };
    std::vector<ProfileSpan> profileSpans_;
    std::map<std::string, ProfileTotal> kernelProfile_, submissionProfile_;
    uint64_t profileOmitted_ = 0;
};

// A compute pipeline over `bindings` storage buffers (set 0, bindings 0..n-1) with push constants.
class Kernel {
public:
    // bindings storage buffers at 0 .. bindings−1, plus an acceleration structure at accelBinding.
    Kernel(Context& ctx, const unsigned char* spirv, size_t bytes, uint32_t bindings, uint32_t pushBytes,
           const char* name = "kernel", uint32_t accelBinding = kNoBinding);
    ~Kernel();
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    void bind(uint32_t binding, const Buffer& b);
    void bindAccel(uint32_t binding, VkAccelerationStructureKHR as);
    void dispatch(VkCommandBuffer cmd, const void* push, uint32_t gx, uint32_t gy);

private:
    void printStatistics(const char* name) const;
    Context& ctx_;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    uint32_t pushBytes_;
    std::string name_;
};

// A global memory barrier: accesses srcAccess at srcStage complete and are visible to dstAccess at dstStage.
void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);
// Serialises compute writes before the next compute dispatch or a transfer.
void computeBarrier(VkCommandBuffer cmd, bool toTransfer = false);
// Transfer writes (buffer fills, copies) before compute reads and writes.
void transferToCompute(VkCommandBuffer cmd);
// Copies the first `bytes` bytes of src into dst.
void copyBuffer(VkCommandBuffer cmd, const Buffer& src, const Buffer& dst, VkDeviceSize bytes);

// Process-wide contexts, one per requested device, created on first use.
std::shared_ptr<Context> acquireContext(int device);

}  // namespace owe::gpu
