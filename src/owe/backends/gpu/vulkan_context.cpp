#include "owe/backends/gpu/vulkan_context.hpp"

#include <algorithm>
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
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
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

bool supportsRayQuery(VkPhysicalDevice dev, const std::vector<VkExtensionProperties>& exts) {
    for (const char* e : kRayQueryExtensions)
        if (!hasExtension(exts, e)) return false;
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bda{};
    bda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{};
    as.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    as.pNext = &bda;
    VkPhysicalDeviceRayQueryFeaturesKHR rq{};
    rq.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rq.pNext = &as;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &rq;
    vkGetPhysicalDeviceFeatures2(dev, &f2);
    return rq.rayQuery && as.accelerationStructure && bda.bufferDeviceAddress;
}

std::vector<Candidate> enumerate(VkInstance inst) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    std::vector<Candidate> out;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        VkPhysicalDeviceFeatures f;
        vkGetPhysicalDeviceFeatures(devs[i], &f);
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(devs[i], nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> exts(ne);
        vkEnumerateDeviceExtensionProperties(devs[i], nullptr, &ne, exts.data());
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
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qs.data());
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

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qs(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &nq, qs.data());
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
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> dexts(ne);
        vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ne, dexts.data());
        if (hasExtension(dexts, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
            exts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
            pep.pipelineExecutableInfo = VK_TRUE;
            statistics_ = true;
        }
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    if (statistics_) dci.pNext = &pep;
    // Hardware ray traversal where the device has it (the kernels choose it per renderer).
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bda{};
    bda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
    bda.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{};
    asf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asf.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR rqf{};
    rqf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rqf.rayQuery = VK_TRUE;
    if (rayQuery_) {
        for (const char* e : kRayQueryExtensions) exts.push_back(e);
        bda.pNext = const_cast<void*>(dci.pNext);
        asf.pNext = &bda;
        rqf.pNext = &asf;
        dci.pNext = &rqf;
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

    vkGetPhysicalDeviceProperties(physical_, &props_);
    cachePath_ = pipelineCachePath(props_);
    if (!cachePath_.empty()) {
        const std::vector<char> data = readPipelineCache(cachePath_, props_);
        VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        cci.initialDataSize = data.size();
        cci.pInitialData = data.empty() ? nullptr : data.data();
        if (vkCreatePipelineCache(device_, &cci, nullptr, &pipelineCache_) != VK_SUCCESS) {
            cci.initialDataSize = 0;  // the driver refused the file: start empty
            cci.pInitialData = nullptr;
            if (vkCreatePipelineCache(device_, &cci, nullptr, &pipelineCache_) != VK_SUCCESS) pipelineCache_ = VK_NULL_HANDLE;
        }
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
        const std::vector<char> disk = readPipelineCache(cachePath_, props_);
        if (!disk.empty()) {
            VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            cci.initialDataSize = disk.size();
            cci.pInitialData = disk.data();
            VkPipelineCache other = VK_NULL_HANDLE;
            if (vkCreatePipelineCache(device_, &cci, nullptr, &other) == VK_SUCCESS) {
                vkMergePipelineCaches(device_, pipelineCache_, 1, &other);
                vkDestroyPipelineCache(device_, other, nullptr);
            }
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
        if (pipelineCache_) {
            savePipelineCache();
            vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
        }
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
    if (b.mapped) vkUnmapMemory(device_, b.memory);
    if (b.buffer) vkDestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device_, b.memory, nullptr);
    b = Buffer{};
}

void Context::upload(const Buffer& dst, const void* data, size_t bytes) {
    if (bytes == 0) return;
    Buffer staging = createHostBuffer(bytes, false);
    std::memcpy(staging.mapped, data, bytes);
    submit([&](VkCommandBuffer cmd) {
        VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
    });
    destroy(staging);
}

void Context::begin() {
    vkCheck(vkResetCommandBuffer(cmd_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkCheck(vkBeginCommandBuffer(cmd_, &bi), "vkBeginCommandBuffer");
}

void Context::endAndWait() {
    vkCheck(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    vkCheck(vkResetFences(device_, 1, &fence_), "vkResetFences");
    vkCheck(vkQueueSubmit(queue_, 1, &si, fence_), "vkQueueSubmit");
    vkCheck(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences (device lost?)");
}

// ---------------------------------------------------------------- Kernel

Kernel::Kernel(Context& ctx, const unsigned char* spirv, size_t bytes, uint32_t bindings, uint32_t pushBytes,
               const char* name, uint32_t accelBinding)
    : ctx_(ctx), pushBytes_(pushBytes) {
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
    if (pushBytes_) vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes_, push);
    vkCmdDispatch(cmd, gx, gy, 1);
}

void computeBarrier(VkCommandBuffer cmd, bool toTransfer) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = toTransfer ? VK_ACCESS_TRANSFER_READ_BIT : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         toTransfer ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
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

bool available(std::string* why) {
    try {
        bool portability = false;
        VkInstance inst = createInstance(portability);
        auto cands = enumerate(inst);
        vkDestroyInstance(inst, nullptr);
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
        bool portability = false;
        VkInstance inst = createInstance(portability);
        for (auto& c : enumerate(inst)) {
            DeviceInfo d = c.info;
            if (!c.usable) d.type += " (unusable: " + c.why + ")";
            out.push_back(d);
        }
        vkDestroyInstance(inst, nullptr);
    } catch (const std::exception&) {
    }
    return out;
}

}  // namespace owe::gpu
