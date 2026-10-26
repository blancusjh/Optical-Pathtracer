// Internal renderer state shared by scene setup, transport passes and film resolution.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <string>
#include <vector>
#include <stdexcept>

#include "owe/backends/gpu/gpu_accel.hpp"
#include "owe/backends/gpu/gpu_renderer.hpp"
#include "owe/backends/gpu/gpu_scene.hpp"
#include "owe/backends/gpu/gpu_push.hpp"
#include "owe/backends/gpu/gpu_film.hpp"
#include "owe/backends/gpu/vulkan_context.hpp"


namespace owe::gpu::detail {

constexpr uint32_t kFlagPartition = 1;
constexpr uint32_t kFlagMediumOnly = 2;  // scene.slang: the path tracer's share of a VCM render with scattering media

struct PixelState {  // mirrors shaders/scene.slang
    uint32_t rng[4];
    float acc[4];
    float comp[4];
    uint32_t stats[4];
};
static_assert(sizeof(PixelState) == 64, "PixelState layout");

// Particle splats (shaders/particle.slang): per pixel, XYZ × kSplatBins 64-bit bins (lo, hi words).
constexpr int kSplatBins = 4;
constexpr VkDeviceSize kFilmPixelBytes = sizeof(FilmPixel);
constexpr uint32_t kSplatWords = 3 * kSplatBins * 2;
// SPPM (shaders/sppm.slang): visible points and pixel states of 5 and 2 float4, flux words per pixel.
constexpr VkDeviceSize kVisiblePointBytes = 5 * 16, kSppmPixelBytes = 2 * 16;
constexpr uint32_t kFluxWords = kSplatWords + 2;
constexpr uint32_t kGuideCells = 128 * 128;  // photon guiding map (shaders/particle.slang)
constexpr VkDeviceSize kLightVertexBytes = 6 * 16;  // shaders/vcm.slang's LightVertex
constexpr uint32_t kPathGroup = 16;       // path kernel groups: 16×16 pixels (path_kernel.slang)
constexpr uint32_t kParticleGroup = 256;  // particle kernel groups (particle_kernel.slang)
constexpr uint32_t kMaxGroups = 65535;

enum Binding : uint32_t {
    kGlobals, kNodes, kBoundaries, kShapeData, kTriangles, kRegions, kMedia, kSpectra, kScalars, kOptics, kEmissions,
    kLights, kPixels, kSplat, kCounters, kTop, kBindingCount,
    // SPPM kernels also bind these (shaders/sppm.slang).
    kVisiblePoints = kBindingCount, kSppmState, kFlux, kGridHead, kGridNext, kLevels, kGuide,
    kTexels,   // images for textures and environment maps (every kernel)
    kNormals,  // smooth meshes' shading normals (every kernel)
    // BDPT/VCM kernels (shaders/vcm.slang): the light vertex cache and its counters.
    kLightVertices, kVcmCounters, kFilm,
    kAllBindings
};
constexpr uint32_t kSceneBinding = 32;  // the acceleration structure (ray-query kernels), above all storage

// Hardware traversal where the device has it, unless OWE_GPU_RAY_QUERY=0 asks for the portable
// software traversal (both are validated against the reference; this chooses per renderer).
inline bool hardwareTraversalWanted() {
    const char* e = std::getenv("OWE_GPU_RAY_QUERY");
    return !(e && std::string(e) == "0");
}

// Workgroups covering n items in groups of `size`.
constexpr uint32_t groups(uint64_t n, uint32_t size) { return uint32_t((n + size - 1) / size); }
// Buckets of a spatial hash for n entries: a power of two from 1024, at most 2^30.
inline uint32_t hashSize(uint64_t n) {
    uint32_t size = 1024;
    while (size < n && size < (1u << 30)) size *= 2;
    return size;
}

// A kernel's SPIR-V words.
struct Spirv {
    const unsigned char* bytes;
    size_t size;
    const char* name;  // the kernel's build name (owe_kernel_<name>)
};

inline uint64_t word64(const uint32_t* w) { return uint64_t(w[0]) | (uint64_t(w[1]) << 32); }

class GpuRenderer : public Renderer {
public:
    GpuRenderer(const Scene&, int detectorIndex, const RenderSettings&);
    ~GpuRenderer() override;
    bool resetObserver() override;
    Image resolve() const override;
    std::string backend() const override;
private:
    void pass(int spp) override;
    void allocateImageBuffers();
    static uint64_t cacheEnvironment(const char* name);
    void allocateLightCache(uint64_t capacity);
    void bindAll();
    // Dispatches work items [0, total) of a kernel: samples per pixel over the image (perPixel),
    // or one thread per particle.
    void dispatchChunked(Kernel& kernel, PushConstants pc, uint64_t total, bool perPixel);
    // Fills whole buffers with zeros before the next compute work.
    void zeroBuffers(std::initializer_list<Binding> bindings);
    uint64_t photonsPerIteration() const;  // SPPM photons and VCM light paths per iteration
    float wavelengthRotation(uint64_t iteration) const;
    uint32_t vcmGroups() const;
    uint64_t vcmLightPaths() const;
    void validateVcmCounts(uint64_t lightPaths) const;
    void sizeLightCache();
    void passVcm(uint32_t iterations);
    void resetVcm();
    void resetSppm();
    void passSppm(uint32_t iterations);
    void clearFilm();
    void accumulateFilm(bool camera, bool light);
    void collectStats();
    void readFirstInconsistency(TransportStats&) const;
    PushConstants push(uint32_t spp) const;
    void passPath(uint32_t spp, uint32_t extraFlags = 0);
    void passParticles(uint32_t ppp);
    bool camera_ = true, particles_ = false, partition_ = false, sppm_ = false;  // vcm_ below
    int W_ = 0, H_ = 0;
    GpuScene gs_;
    std::shared_ptr<Context> ctx_;
    bool rayQuery_ = false;
    std::unique_ptr<HardwareScene> hardware_;
    std::unique_ptr<Kernel> pathKernel_, particleKernel_, filmKernel_;
    std::unique_ptr<Kernel> sppmCamera_, sppmPhoton_, sppmGrid_, sppmUpdate_, sppmGuide_;
    std::unique_ptr<Kernel> vcmLight_, vcmCamera_, vcmGrid_;
    struct KernelSlot {
        std::unique_ptr<Kernel>* kernel;
        bool traces;  // binds the acceleration structure (hardware traversal)
    };
    std::vector<KernelSlot> kernels_;  // this integrator's kernels
    Buffer buffers_[kAllBindings];
    Buffer filmReadback_, counterReadback_, sppmReadback_, vcmCounterReadback_, vcmGuideBackup_;
    // BDPT/VCM: whether it merges, the light vertex cache's capacity and its budget per light path
    // (grown before replaying a rejected pass that fills it).
    bool vcm_ = false, merging_ = false;
    bool mediumPartition_ = false;  // VCM/BDPT with scattering media: a path-traced pass for medium-scattered paths
    uint32_t cacheCapacity_ = 0;
    double vertexBudget_ = 6;
    bool vcmSizePending_ = false;
    // SPPM: iterations so far, photons per iteration, hash size, radius parameters, and the last
    // read-back pixel states (τ xyz, N; r, unused ×3).
    long long iterations_ = 0;
    uint64_t photonsPerIteration_ = 0;
    uint32_t gridSize_ = 0;
    float r0PerMetre_ = 0, r0Floor_ = 0;
    double particleCount_ = 0;
    uint32_t chunk_ = 1;
    uint32_t particleChunk_ = 16384;
};
} // namespace owe::gpu::detail
