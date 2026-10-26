#include "owe/backends/gpu/vulkan_context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <mutex>
#include <stdexcept>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace owe::gpu {

void vkCheck(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan: ") + what + " failed (VkResult " + std::to_string(int(r)) + ")");
}

namespace {

namespace fs = std::filesystem;

// Where the pipeline cache of this device and driver lives: $OWE_CACHE_DIR, else the platform's
// user cache directory, under owe/. Empty when caching is disabled or no directory is known.
std::string pipelineCachePath(const VkPhysicalDeviceProperties& p) {
    if (const char* e = std::getenv("OWE_PIPELINE_CACHE"); e && std::string(e) == "0") return {};
    fs::path dir;
    if (const char* e = std::getenv("OWE_CACHE_DIR"); e && *e) dir = e;
#ifdef _WIN32
    else if (const char* e = std::getenv("LOCALAPPDATA"); e && *e) dir = fs::path(e) / "owe";
#else
    else if (const char* e = std::getenv("XDG_CACHE_HOME"); e && *e) dir = fs::path(e) / "owe";
    else if (const char* e = std::getenv("HOME"); e && *e) dir = fs::path(e) / ".cache" / "owe";
#endif
    if (dir.empty()) return {};
    char name[96];
    std::snprintf(name, sizeof name, "pipelines-%04x-%04x-%08x-", p.vendorID, p.deviceID, p.driverVersion);
    std::string file = name;
    for (uint8_t b : p.pipelineCacheUUID) {
        std::snprintf(name, sizeof name, "%02x", b);
        file += name;
    }
    return (dir / (file + ".bin")).string();
}

// The cache file keeps the pipelines of every build of the kernels that used it, though only the
// current build's are looked up again: past this size it starts over from one process's pipelines.
constexpr size_t kMaxPipelineCacheBytes = size_t(128) << 20;

// The file's bytes when its header names this device and driver (a stale or foreign cache is ignored).
std::vector<char> readPipelineCache(const std::string& path, const VkPhysicalDeviceProperties& p) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() < 16 + VK_UUID_SIZE) return {};
    uint32_t h[4];
    std::memcpy(h, data.data(), sizeof h);
    if (h[0] < 16 + VK_UUID_SIZE || h[1] != VK_PIPELINE_CACHE_HEADER_VERSION_ONE || h[2] != p.vendorID || h[3] != p.deviceID ||
        std::memcmp(data.data() + 16, p.pipelineCacheUUID, VK_UUID_SIZE) != 0)
        return {};
    return data;
}

std::once_flag g_volkOnce;
VkResult g_volkResult = VK_ERROR_INITIALIZATION_FAILED;

#ifndef _WIN32
// Opens a Vulkan loader or ICD by path and hands its vkGetInstanceProcAddr to volk.
bool loadVulkanFrom(const char* path) {
    void* module = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!module) return false;
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(module, "vkGetInstanceProcAddr"));
    if (!gipa) return false;
    volkInitializeCustom(gipa);
    return true;
}
#endif

void loadVulkan() {
    std::call_once(g_volkOnce, [] {
#ifndef _WIN32
        // OWE_VULKAN_LIBRARY names a loader or a driver (e.g. libMoltenVK.dylib) explicitly.
        if (const char* p = std::getenv("OWE_VULKAN_LIBRARY"); p && loadVulkanFrom(p)) {
            g_volkResult = VK_SUCCESS;
            return;
        }
#endif
        g_volkResult = volkInitialize();
#ifdef __APPLE__
        // volk searches the default paths and /usr/local/lib; Homebrew on Apple silicon installs
        // the loader and MoltenVK (which is itself a complete Vulkan implementation) under /opt/homebrew.
        for (const char* p : {"/opt/homebrew/lib/libvulkan.1.dylib", "/opt/homebrew/lib/libvulkan.dylib",
                              "/opt/homebrew/lib/libMoltenVK.dylib", "/usr/local/lib/libMoltenVK.dylib"})
            if (g_volkResult != VK_SUCCESS && loadVulkanFrom(p)) g_volkResult = VK_SUCCESS;
#endif
    });
    if (g_volkResult != VK_SUCCESS)
        throw std::runtime_error(
            "no Vulkan loader found (install a GPU driver; on macOS: brew install molten-vk vulkan-loader, "
            "or set OWE_VULKAN_LIBRARY to libvulkan / libMoltenVK)");
}

// The items of a Vulkan two-call enumeration: f(args..., &count, nullptr), then f(args..., &count, items).
template <class T, class F, class... Args>
std::vector<T> vkList(F f, Args... args) {
    uint32_t n = 0;
    f(args..., &n, nullptr);
    std::vector<T> items(n);
    f(args..., &n, items.data());
    items.resize(n);
    return items;
}

// A pipeline cache holding `data` (the driver may refuse it: then empty); VK_NULL_HANDLE on failure.
VkPipelineCache createPipelineCache(VkDevice device, const std::vector<char>& data) {
    VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    cci.initialDataSize = data.size();
    cci.pInitialData = data.empty() ? nullptr : data.data();
    VkPipelineCache cache = VK_NULL_HANDLE;
    if (vkCreatePipelineCache(device, &cci, nullptr, &cache) == VK_SUCCESS) return cache;
    return data.empty() ? VK_NULL_HANDLE : createPipelineCache(device, {});
}

bool hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    return std::any_of(exts.begin(), exts.end(), [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
}

const char* typeName(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}
int typeRank(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 0;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 4;
        default: return 3;
    }
}

VkInstance createInstance(bool& portability) {
    loadVulkan();
    const auto exts = vkList<VkExtensionProperties>(vkEnumerateInstanceExtensionProperties, nullptr);
    std::vector<const char*> enable;
    portability = hasExtension(exts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (portability) enable.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "optical-world-engine";
    app.pEngineName = "owe";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = uint32_t(enable.size());
    ci.ppEnabledExtensionNames = enable.data();
    // MoltenVK (Apple silicon) is a portability implementation: it is only enumerated on request.
    if (portability) ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    VkInstance inst = VK_NULL_HANDLE;
    vkCheck(vkCreateInstance(&ci, nullptr, &inst), "vkCreateInstance");
    volkLoadInstanceOnly(inst);
    return inst;
}

struct Candidate {
    VkPhysicalDevice device;
    DeviceInfo info;
    int rank;
    bool usable;
    bool rayQuery = false;  // hardware ray traversal from compute shaders
    std::string why;
};

// Ray queries in compute shaders (VK_KHR_ray_query) over acceleration structures: the traversal
// hardware of NVIDIA RTX, AMD RDNA2+, Intel Arc. Every extension it depends on, and its features.
const char* const kRayQueryExtensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME};

// The features ray queries need, chained ahead of `next` for a query or a device request.
struct RayQueryFeatures {
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    void* chain(void* next) {
        bda.pNext = next;
        as.pNext = &bda;
        rq.pNext = &as;
        return &rq;
    }
};

bool supportsRayQuery(VkPhysicalDevice dev, const std::vector<VkExtensionProperties>& exts) {
    for (const char* e : kRayQueryExtensions)
        if (!hasExtension(exts, e)) return false;
    RayQueryFeatures f;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = f.chain(nullptr);
    vkGetPhysicalDeviceFeatures2(dev, &f2);
    return f.rq.rayQuery && f.as.accelerationStructure && f.bda.bufferDeviceAddress;
}

std::vector<Candidate> enumerate(VkInstance inst) {
    const auto devs = vkList<VkPhysicalDevice>(vkEnumeratePhysicalDevices, inst);
    std::vector<Candidate> out;
    for (uint32_t i = 0; i < devs.size(); ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        VkPhysicalDeviceFeatures f;
        vkGetPhysicalDeviceFeatures(devs[i], &f);
        const auto exts = vkList<VkExtensionProperties>(vkEnumerateDeviceExtensionProperties, devs[i], nullptr);
        Candidate c;
        c.device = devs[i];
        c.info.index = int(i);
        c.info.name = p.deviceName;
        c.info.type = typeName(p.deviceType);
        c.info.driver = "Vulkan " + std::to_string(VK_API_VERSION_MAJOR(p.apiVersion)) + "." +
                        std::to_string(VK_API_VERSION_MINOR(p.apiVersion)) + "." +
                        std::to_string(VK_API_VERSION_PATCH(p.apiVersion));
        c.info.portability = hasExtension(exts, "VK_KHR_portability_subset");
        c.rayQuery = p.apiVersion >= VK_API_VERSION_1_1 && supportsRayQuery(devs[i], exts);
        if (c.rayQuery) c.info.driver += ", ray queries";
        c.rank = typeRank(p.deviceType);
        c.usable = true;
        if (p.apiVersion < VK_API_VERSION_1_1) { c.usable = false; c.why = "needs Vulkan 1.1"; }
        if (!f.shaderInt64) { c.usable = false; c.why = "no 64-bit shader integers"; }
        // The kernels run in groups of 256 threads (every desktop GPU and Apple silicon allows 1024).
        if (p.limits.maxComputeWorkGroupInvocations < 256 || p.limits.maxComputeWorkGroupSize[0] < 256 ||
            p.limits.maxComputeWorkGroupSize[1] < 16) {
            c.usable = false;
            c.why = "compute groups of 256 threads are not supported";
        }
        const auto qs = vkList<VkQueueFamilyProperties>(vkGetPhysicalDeviceQueueFamilyProperties, devs[i]);
        if (std::none_of(qs.begin(), qs.end(), [](auto& q) { return (q.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0; })) {
            c.usable = false;
            c.why = "no compute queue";
        }
        out.push_back(c);
    }
    return out;
}

}  // namespace

Context::Context(int device) {
    bool portabilityInstance = false;
    instance_ = createInstance(portabilityInstance);
    auto cands = enumerate(instance_);
    const Candidate* pick = nullptr;
    if (device >= 0) {
        if (device >= int(cands.size())) throw std::runtime_error("no GPU device " + std::to_string(device));
        pick = &cands[size_t(device)];
        if (!pick->usable) throw std::runtime_error("GPU device " + pick->info.name + " is not usable: " + pick->why);
    } else {
        for (const Candidate& c : cands)
            if (c.usable && (!pick || c.rank < pick->rank)) pick = &c;
        if (!pick) throw std::runtime_error(cands.empty() ? "no Vulkan device found" : "no usable Vulkan device (" + cands[0].why + ")");
    }
    physical_ = pick->device;
    info_ = pick->info;
    rayQuery_ = pick->rayQuery;
    vkGetPhysicalDeviceMemoryProperties(physical_, &memProps_);

    const auto qs = vkList<VkQueueFamilyProperties>(vkGetPhysicalDeviceQueueFamilyProperties, physical_);
    const uint32_t nq = uint32_t(qs.size());
    // Prefer a compute queue without graphics (asynchronous compute), else any compute queue.
    int family = -1;
    for (uint32_t i = 0; i < nq; ++i)
        if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { family = int(i); break; }
    if (family < 0)
        for (uint32_t i = 0; i < nq; ++i)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = int(i); break; }
    queueFamily_ = uint32_t(family);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures features{};
    features.shaderInt64 = VK_TRUE;
    std::vector<const char*> exts;
    if (info_.portability) exts.push_back("VK_KHR_portability_subset");
    // OWE_GPU_STATS=1: report each kernel's compiled statistics (registers, spills, ...), where the
    // driver offers them. Development aid only; it does not change what the kernels compute.
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep{};
    pep.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
    if (const char* e = std::getenv("OWE_GPU_STATS"); e && *e && *e != '0') {
        if (hasExtension(vkList<VkExtensionProperties>(vkEnumerateDeviceExtensionProperties, physical_, nullptr), VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
            exts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
            pep.pipelineExecutableInfo = VK_TRUE;
            statistics_ = true;
        }
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    if (statistics_) dci.pNext = &pep;
    // Hardware ray traversal where the device has it (the kernels choose it per renderer).
    RayQueryFeatures rqf;
    rqf.bda.bufferDeviceAddress = rqf.as.accelerationStructure = rqf.rq.rayQuery = VK_TRUE;
    if (rayQuery_) {
        for (const char* e : kRayQueryExtensions) exts.push_back(e);
        dci.pNext = rqf.chain(const_cast<void*>(dci.pNext));
        VkPhysicalDeviceAccelerationStructurePropertiesKHR ap{};
        ap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &ap;
        vkGetPhysicalDeviceProperties2(physical_, &p2);
        scratchAlignment_ = std::max<VkDeviceSize>(1, ap.minAccelerationStructureScratchOffsetAlignment);
    }
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &features;
    dci.enabledExtensionCount = uint32_t(exts.size());
    dci.ppEnabledExtensionNames = exts.data();
    vkCheck(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice");
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queueFamily_;
    vkCheck(vkCreateCommandPool(device_, &pci, nullptr, &pool_), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    vkCheck(vkAllocateCommandBuffers(device_, &cai, &cmd_), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCheck(vkCreateFence(device_, &fci, nullptr, &fence_), "vkCreateFence");
    for (auto& submission : submissions_) {
        vkCheck(vkAllocateCommandBuffers(device_, &cai, &submission.command), "vkAllocateCommandBuffers async");
        vkCheck(vkCreateFence(device_, &fci, nullptr, &submission.fence), "vkCreateFence async");
    }

    vkGetPhysicalDeviceProperties(physical_, &props_);
    if (const char* e = std::getenv("OWE_GPU_PROFILE"); e && *e && *e != '0') {
        timestampBits_ = qs[queueFamily_].timestampValidBits;
        if (timestampBits_ && props_.limits.timestampPeriod > 0) {
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpi.queryCount = kProfileQueries;
            if (vkCreateQueryPool(device_, &qpi, nullptr, &profilePool_) != VK_SUCCESS)
                std::fprintf(stderr, "owe_gpu_profile unavailable: timestamp query allocation failed\n");
            else
                std::fprintf(stderr, "owe_gpu_profile enabled: diagnostic timestamps may affect performance\n");
        } else {
            std::fprintf(stderr, "owe_gpu_profile unavailable: compute queue has no timestamps\n");
        }
    }
    cachePath_ = pipelineCachePath(props_);
    if (!cachePath_.empty()) {
        std::vector<char> data = readPipelineCache(cachePath_, props_);
        if (data.size() > kMaxPipelineCacheBytes) data.clear();
        pipelineCache_ = createPipelineCache(device_, data);
        if (pipelineCache_) vkGetPipelineCacheData(device_, pipelineCache_, &cacheSavedBytes_, nullptr);
    }
}

void Context::savePipelineCache() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    if (!pipelineCache_) return;
    size_t bytes = 0;
    vkGetPipelineCacheData(device_, pipelineCache_, &bytes, nullptr);
    if (bytes <= cacheSavedBytes_) return;
    try {
        // Keep what other processes added since this one loaded the file.
        std::vector<char> disk = readPipelineCache(cachePath_, props_);
        if (disk.size() > kMaxPipelineCacheBytes) disk.clear();
        if (VkPipelineCache other = disk.empty() ? VK_NULL_HANDLE : createPipelineCache(device_, disk)) {
            vkMergePipelineCaches(device_, pipelineCache_, 1, &other);
            vkDestroyPipelineCache(device_, other, nullptr);
        }
        vkGetPipelineCacheData(device_, pipelineCache_, &bytes, nullptr);
        std::vector<char> data(bytes);
        if (vkGetPipelineCacheData(device_, pipelineCache_, &bytes, data.data()) != VK_SUCCESS) return;
        data.resize(bytes);
        const fs::path path(cachePath_);
        fs::create_directories(path.parent_path());
        // Written beside the target and renamed over it, so a reader never sees a partial file.
        const fs::path tmp = path.string() + ".tmp" + std::to_string(std::random_device{}());
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            out.write(data.data(), std::streamsize(data.size()));
            if (!out) {
                std::error_code ec;
                fs::remove(tmp, ec);
                return;
            }
        }
        fs::rename(tmp, path);
        cacheSavedBytes_ = bytes;
    } catch (const std::exception&) {
        // A cache that cannot be written only costs compile time.
    }
}

Context::~Context() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (profilePool_) {
            for (const auto& [name, total] : kernelProfile_)
                std::fprintf(stderr, "owe_gpu_profile kind=kernel name=%s calls=%llu gpu_ms=%.6f\n",
                             name.c_str(), static_cast<unsigned long long>(total.calls), total.gpuMs);
            for (const auto& [name, total] : submissionProfile_)
                std::fprintf(stderr, "owe_gpu_profile kind=submission name=%s calls=%llu gpu_ms=%.6f cpu_ms=%.6f\n",
                             name.c_str(), static_cast<unsigned long long>(total.calls), total.gpuMs, total.cpuMs);
            if (profileOmitted_)
                std::fprintf(stderr, "owe_gpu_profile omitted_kernel_spans=%llu (query capacity exceeded)\n",
                             static_cast<unsigned long long>(profileOmitted_));
            vkDestroyQueryPool(device_, profilePool_, nullptr);
        }
        if (pipelineCache_) {
            savePipelineCache();
            vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
        }
        for (const auto& submission : submissions_) vkDestroyFence(device_, submission.fence, nullptr);
        vkDestroyFence(device_, fence_, nullptr);
        vkDestroyCommandPool(device_, pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

uint32_t Context::memoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback) const {
    for (VkMemoryPropertyFlags flags : {want, fallback})
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memProps_.memoryTypes[i].propertyFlags & flags) == flags) return i;
    throw std::runtime_error("Vulkan: no suitable memory type");
}

Buffer Context::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want,
                             VkMemoryPropertyFlags fallback) {
    Buffer b;
    b.size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = b.size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device_, &bci, nullptr, &b.buffer), "vkCreateBuffer");
    try {
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, b.buffer, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        VkMemoryAllocateFlagsInfo flags{};
        flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) mai.pNext = &flags;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memoryType(req.memoryTypeBits, want, fallback);
        vkCheck(vkAllocateMemory(device_, &mai, nullptr, &b.memory), "vkAllocateMemory");
        vkCheck(vkBindBufferMemory(device_, b.buffer, b.memory, 0), "vkBindBufferMemory");
        if (memProps_.memoryTypes[mai.memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            vkCheck(vkMapMemory(device_, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
    } catch (...) {
        destroy(b);
        throw;
    }
    return b;
}

Buffer Context::createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags extraUsage) {
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | extraUsage;
    // With ray queries, any buffer may feed an acceleration-structure build (triangles, boxes, instances).
    if (rayQuery_)
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    return createBuffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
}

VkDeviceAddress Context::address(const Buffer& b) const {
    VkBufferDeviceAddressInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    ai.buffer = b.buffer;
    return vkGetBufferDeviceAddressKHR(device_, &ai);
}

Accel Context::createAccel(VkAccelerationStructureTypeKHR type, VkDeviceSize size) {
    Accel a;
    a.storage = createDeviceBuffer(size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    VkAccelerationStructureCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    ci.buffer = a.storage.buffer;
    ci.size = size;
    ci.type = type;
    vkCheck(vkCreateAccelerationStructureKHR(device_, &ci, nullptr, &a.handle), "vkCreateAccelerationStructureKHR");
    VkAccelerationStructureDeviceAddressInfoKHR ai{};
    ai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    ai.accelerationStructure = a.handle;
    a.address = vkGetAccelerationStructureDeviceAddressKHR(device_, &ai);
    return a;
}

void Context::destroy(Accel& a) {
    waitPending();
    if (a.handle) vkDestroyAccelerationStructureKHR(device_, a.handle, nullptr);
    destroy(a.storage);
    a = Accel{};
}

Buffer Context::createHostBuffer(VkDeviceSize size, bool readback) {
    const VkMemoryPropertyFlags base = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        readback ? base | VK_MEMORY_PROPERTY_HOST_CACHED_BIT : base, base);
}

void Context::destroy(Buffer& b) {
    waitPending();
    if (b.mapped) vkUnmapMemory(device_, b.memory);
    if (b.buffer) vkDestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device_, b.memory, nullptr);
    b = Buffer{};
}

void Context::upload(const Buffer& dst, const void* data, size_t bytes) { uploadAt(dst, 0, data, bytes); }

void Context::uploadAt(const Buffer& dst, VkDeviceSize offset, const void* data, size_t bytes) {
    if (bytes == 0) return;
    const size_t slice = std::min<size_t>(bytes, size_t(64) << 20);
    Buffer staging = createHostBuffer(slice, false);
    for (size_t done = 0; done < bytes; done += slice) {
        const size_t n = std::min(slice, bytes - done);
        std::memcpy(staging.mapped, static_cast<const char*>(data) + done, n);
        submit([&](VkCommandBuffer cmd) {
            VkBufferCopy region{0, offset + done, n};
            vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
        }, "upload");
    }
    destroy(staging);
}

void Context::waitPending() {
    for (auto& submission : submissions_) {
        if (!submission.pending) continue;
        vkCheck(vkWaitForFences(device_, 1, &submission.fence, VK_TRUE, UINT64_MAX), "wait async submission");
        submission.pending = false;
    }
}

VkCommandBuffer Context::beginAsync() {
    auto& submission = submissions_[nextSubmission_];
    if (submission.pending) {
        vkCheck(vkWaitForFences(device_, 1, &submission.fence, VK_TRUE, UINT64_MAX), "reuse async submission");
        submission.pending = false;
    }
    beginCommand(submission.command);
    return submission.command;
}

void Context::endAsync() {
    auto& submission = submissions_[nextSubmission_];
    submitCommand(submission.command, submission.fence);
    submission.pending = true;
    nextSubmission_ = (nextSubmission_ + 1) % submissions_.size();
}

void Context::beginCommand(VkCommandBuffer cmd) {
    vkCheck(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkCheck(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
}

void Context::submitCommand(VkCommandBuffer cmd, VkFence fence) {
    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    vkCheck(vkResetFences(device_, 1, &fence), "vkResetFences");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkCheck(vkQueueSubmit(queue_, 1, &si, fence), "vkQueueSubmit");
}

void Context::begin() {
    beginCommand(cmd_);
    if (profilePool_) {
        profileNext_ = 2;
        profileSpans_.clear();
        vkCmdResetQueryPool(cmd_, profilePool_, 0, kProfileQueries);
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, profilePool_, 0);
    }
}

uint32_t Context::beginKernelProfile(VkCommandBuffer cmd, const std::string& name) {
    if (!profilePool_) return ~0u;
    if (profileNext_ + 2 > kProfileQueries) {
        ++profileOmitted_;
        return ~0u;
    }
    const uint32_t query = profileNext_;
    profileNext_ += 2;
    profileSpans_.push_back({name, query});
    // Bottom-of-pipe on both sides excludes previously queued GPU work from this span.
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profilePool_, query);
    // Prevent the next dispatch from overlapping the starting timestamp. Profiling deliberately
    // serialises dispatches; its timings diagnose costs, not production throughput.
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 0, nullptr);
    return query;
}

void Context::endKernelProfile(VkCommandBuffer cmd, uint32_t query) {
    if (query != ~0u)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profilePool_, query + 1);
}

void Context::endAndWait(const char* label) {
    if (profilePool_)
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profilePool_, 1);
    const auto start = std::chrono::steady_clock::now();
    submitCommand(cmd_, fence_);
    vkCheck(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences (device lost?)");
    // This fence also covers earlier submissions on this queue.
    for (auto& submission : submissions_) submission.pending = false;
    if (profilePool_) {
        const double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::vector<uint64_t> ticks(profileNext_);
        vkCheck(vkGetQueryPoolResults(device_, profilePool_, 0, profileNext_, ticks.size() * sizeof(uint64_t),
                                     ticks.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT), "vkGetQueryPoolResults");
        const uint64_t mask = timestampBits_ == 64 ? ~uint64_t(0) : (uint64_t(1) << timestampBits_) - 1;
        const auto elapsed = [&](uint32_t a, uint32_t b) {
            return double((ticks[b] - ticks[a]) & mask) * double(props_.limits.timestampPeriod) * 1e-6;
        };
        auto& total = submissionProfile_[label];
        ++total.calls;
        total.gpuMs += elapsed(0, 1);
        total.cpuMs += cpuMs;
        for (const auto& span : profileSpans_) {
            auto& kernel = kernelProfile_[span.name];
            ++kernel.calls;
            kernel.gpuMs += elapsed(span.query, span.query + 1);
        }
    }
}

// ---------------------------------------------------------------- Kernel

Kernel::Kernel(Context& ctx, const unsigned char* spirv, size_t bytes, uint32_t bindings, uint32_t pushBytes,
               const char* name, uint32_t accelBinding)
    : ctx_(ctx), pushBytes_(pushBytes), name_(name) {
    VkDevice dev = ctx_.device();
    std::vector<uint32_t> words((bytes + 3) / 4);
    std::memcpy(words.data(), spirv, bytes);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = bytes;
    smi.pCode = words.data();
    vkCheck(vkCreateShaderModule(dev, &smi, nullptr, &module_), "vkCreateShaderModule");

    std::vector<VkDescriptorSetLayoutBinding> lb(bindings);
    for (uint32_t i = 0; i < bindings; ++i) {
        lb[i].binding = i;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    const bool accel = accelBinding != kNoBinding;
    if (accel) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = accelBinding;
        b.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        lb.push_back(b);
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = uint32_t(lb.size());
    dli.pBindings = lb.data();
    vkCheck(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &setLayout_), "vkCreateDescriptorSetLayout");

    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &setLayout_;
    pli.pushConstantRangeCount = pushBytes ? 1 : 0;
    pli.pPushConstantRanges = &pr;
    vkCheck(vkCreatePipelineLayout(dev, &pli, nullptr, &layout_), "vkCreatePipelineLayout");

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module_;
    cpi.stage.pName = "main";
    cpi.layout = layout_;
    if (ctx_.statistics()) cpi.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    vkCheck(vkCreateComputePipelines(dev, ctx_.pipelineCache(), 1, &cpi, nullptr, &pipeline_), "vkCreateComputePipelines");
    if (ctx_.statistics()) printStatistics(name);

    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings},
                                  {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = accel ? 2 : 1;
    dpi.pPoolSizes = ps;
    vkCheck(vkCreateDescriptorPool(dev, &dpi, nullptr, &pool_), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &setLayout_;
    vkCheck(vkAllocateDescriptorSets(dev, &dai, &set_), "vkAllocateDescriptorSets");
}

void Kernel::printStatistics(const char* name) const {
    VkDevice dev = ctx_.device();
    VkPipelineInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
    pi.pipeline = pipeline_;
    uint32_t n = 0;
    vkGetPipelineExecutablePropertiesKHR(dev, &pi, &n, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> props(n);
    for (auto& p : props) p.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    vkGetPipelineExecutablePropertiesKHR(dev, &pi, &n, props.data());
    for (uint32_t e = 0; e < n; ++e) {
        VkPipelineExecutableInfoKHR ei{};
        ei.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
        ei.pipeline = pipeline_;
        ei.executableIndex = e;
        uint32_t ns = 0;
        vkGetPipelineExecutableStatisticsKHR(dev, &ei, &ns, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(ns);
        for (auto& s : st) s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        vkGetPipelineExecutableStatisticsKHR(dev, &ei, &ns, st.data());
        std::fprintf(stderr, "kernel %s [%s, subgroup %u]:", name, props[e].name, props[e].subgroupSize);
        for (const auto& s : st) {
            std::fprintf(stderr, " %s=", s.name);
            switch (s.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: std::fprintf(stderr, "%u", s.value.b32); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::fprintf(stderr, "%lld", (long long)s.value.i64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::fprintf(stderr, "%llu", (unsigned long long)s.value.u64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: std::fprintf(stderr, "%g", s.value.f64); break;
                default: std::fprintf(stderr, "?");
            }
        }
        std::fprintf(stderr, "\n");
    }
}

Kernel::~Kernel() {
    ctx_.waitPending();
    VkDevice dev = ctx_.device();
    vkDeviceWaitIdle(dev);
    vkDestroyDescriptorPool(dev, pool_, nullptr);
    vkDestroyPipeline(dev, pipeline_, nullptr);
    vkDestroyPipelineLayout(dev, layout_, nullptr);
    vkDestroyDescriptorSetLayout(dev, setLayout_, nullptr);
    vkDestroyShaderModule(dev, module_, nullptr);
}

void Kernel::bind(uint32_t binding, const Buffer& b) {
    VkDescriptorBufferInfo bi{b.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set_;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
}

void Kernel::bindAccel(uint32_t binding, VkAccelerationStructureKHR as) {
    VkWriteDescriptorSetAccelerationStructureKHR wa{};
    wa.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    wa.accelerationStructureCount = 1;
    wa.pAccelerationStructures = &as;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.pNext = &wa;
    w.dstSet = set_;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
}

void Kernel::dispatch(VkCommandBuffer cmd, const void* push, uint32_t gx, uint32_t gy) {
    const uint32_t query = ctx_.beginKernelProfile(cmd, name_);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
    if (pushBytes_) vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes_, push);
    vkCmdDispatch(cmd, gx, gy, 1);
    ctx_.endKernelProfile(cmd, query);
}

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = srcAccess;
    mb.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void computeBarrier(VkCommandBuffer cmd, bool toTransfer) {
    if (toTransfer)
        memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_TRANSFER_READ_BIT);
    else
        memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void transferToCompute(VkCommandBuffer cmd) {
    memoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void copyBuffer(VkCommandBuffer cmd, const Buffer& src, const Buffer& dst, VkDeviceSize bytes) {
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &region);
}

std::shared_ptr<Context> acquireContext(int device) {
    static std::mutex m;
    static std::map<int, std::weak_ptr<Context>> cache;
    std::lock_guard<std::mutex> lock(m);
    if (auto c = cache[device].lock()) return c;
    auto c = std::make_shared<Context>(device);
    cache[device] = c;
    return c;
}

// ---------------------------------------------------------------- device listing

namespace {
// The candidate devices of a temporary instance (destroyed even when enumeration throws).
std::vector<Candidate> candidates() {
    bool portability = false;
    VkInstance inst = createInstance(portability);
    struct Destroy {
        VkInstance i;
        ~Destroy() { vkDestroyInstance(i, nullptr); }
    } destroy{inst};
    return enumerate(inst);
}
}  // namespace

bool available(std::string* why) {
    try {
        auto cands = candidates();
        for (auto& c : cands)
            if (c.usable) return true;
        if (why) *why = cands.empty() ? "no Vulkan device found" : "no usable Vulkan device: " + cands[0].why;
        return false;
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return false;
    }
}

std::vector<DeviceInfo> devices() {
    std::vector<DeviceInfo> out;
    try {
        for (auto& c : candidates()) {
            DeviceInfo d = c.info;
            if (!c.usable) d.type += " (unusable: " + c.why + ")";
            out.push_back(d);
        }
    } catch (const std::exception&) {
    }
    return out;
}

}  // namespace owe::gpu
