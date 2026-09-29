// The GPU renderer: the flattened scene lives in device-local storage buffers. Each pass runs the
// path kernel (camera paths) and/or the particle kernel (light tracing) in short dispatches, so the
// display stays responsive and no OS watchdog fires; the pass's sums are then read back and
// accumulated in double on the host. As the CPU reference: path estimates are normalised per pixel
// by their sample count, particle splats by the number of particles, and hybrid adds the two.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>

#include "owe/backends/gpu/gpu_accel.hpp"
#include "owe/backends/gpu/gpu_renderer.hpp"
#include "owe/backends/gpu/gpu_scene.hpp"
#include "owe/backends/gpu/vulkan_context.hpp"

extern const unsigned char owe_kernel_path[];
extern const size_t owe_kernel_path_size;
extern const unsigned char owe_kernel_particle[];
extern const size_t owe_kernel_particle_size;
extern const unsigned char owe_kernel_path_rq[];
extern const size_t owe_kernel_path_rq_size;
extern const unsigned char owe_kernel_particle_rq[];
extern const size_t owe_kernel_particle_rq_size;
extern const unsigned char owe_kernel_sppm_camera[];
extern const size_t owe_kernel_sppm_camera_size;
extern const unsigned char owe_kernel_sppm_camera_rq[];
extern const size_t owe_kernel_sppm_camera_rq_size;
extern const unsigned char owe_kernel_sppm_photon[];
extern const size_t owe_kernel_sppm_photon_size;
extern const unsigned char owe_kernel_sppm_photon_rq[];
extern const size_t owe_kernel_sppm_photon_rq_size;
extern const unsigned char owe_kernel_sppm_grid[];
extern const size_t owe_kernel_sppm_grid_size;
extern const unsigned char owe_kernel_sppm_update[];
extern const size_t owe_kernel_sppm_update_size;
extern const unsigned char owe_kernel_sppm_guide[];
extern const size_t owe_kernel_sppm_guide_size;
extern const unsigned char owe_kernel_vcm_light[];
extern const size_t owe_kernel_vcm_light_size;
extern const unsigned char owe_kernel_vcm_light_rq[];
extern const size_t owe_kernel_vcm_light_rq_size;
extern const unsigned char owe_kernel_vcm_camera[];
extern const size_t owe_kernel_vcm_camera_size;
extern const unsigned char owe_kernel_vcm_camera_rq[];
extern const size_t owe_kernel_vcm_camera_rq_size;
extern const unsigned char owe_kernel_vcm_grid[];
extern const size_t owe_kernel_vcm_grid_size;

namespace owe::gpu {

namespace {

struct PushConstants {  // mirrors shaders/scene.slang
    uint32_t pass, spp, sampleBase, sampleCount;
    uint32_t seedLo, seedHi, width, height;
    uint32_t flags, baseHi, pad0, pad1;
    uint32_t extra[4];
};
constexpr uint32_t kFlagPartition = 1;

struct PixelState {  // mirrors shaders/scene.slang
    uint32_t rng[4];
    float acc[4];
    float comp[4];
    uint32_t stats[4];
};
static_assert(sizeof(PixelState) == 64, "PixelState layout");

// Particle splats (shaders/particle.slang): per pixel, XYZ × kSplatBins 64-bit bins (lo, hi words).
constexpr int kSplatBins = 4;
constexpr int kSplatExp0 = -80;
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
    kLightVertices, kVcmCounters,
    kAllBindings
};
constexpr uint32_t kSceneBinding = 32;  // the acceleration structure (ray-query kernels), above all storage

// Hardware traversal where the device has it, unless OWE_GPU_RAY_QUERY=0 asks for the portable
// software traversal (both are validated against the reference; this chooses per renderer).
bool hardwareTraversalWanted() {
    const char* e = std::getenv("OWE_GPU_RAY_QUERY");
    return !(e && std::string(e) == "0");
}

// Transfer writes (buffer fills) before compute reads and writes.
void transferToCompute(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr,
                         0, nullptr);
}

uint64_t word64(const uint32_t* w) { return uint64_t(w[0]) | (uint64_t(w[1]) << 32); }
uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

class GpuRenderer : public Renderer {
public:
    GpuRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings)
        : scene_(scene), settings_(settings), stats_() {
        if (detectorIndex < 0 || detectorIndex >= int(scene.detectors.size())) throw std::runtime_error("no such detector");
        det_ = scene.detectors[size_t(detectorIndex)].get();
        const std::string& I = settings_.integrator;
        if (I != "path" && I != "light" && I != "hybrid" && I != "sppm" && I != "bdpt" && I != "vcm")
            throw std::runtime_error("unknown integrator '" + I + "' (expected path, light, hybrid, sppm, bdpt or vcm)");
        if (I == "hybrid" && !det_->isVirtual())
            throw std::runtime_error("the hybrid integrator needs a virtual observer; use path or light for surface sensors");
        vcm_ = I == "bdpt" || I == "vcm";
        merging_ = I == "vcm";
        if (vcm_ && !dynamic_cast<const IdealObserver*>(det_))
            throw std::runtime_error("the " + I + " integrator needs an observer (eye); use path, light or sppm for sensors and cameras");
        if (vcm_)
            for (const Medium& m : scene.world.media())
                if (m.scatters())
                    throw std::runtime_error("the " + I + " integrator does not handle scattering media ('" + m.name +
                                             "'); use path or sppm");
        sppm_ = I == "sppm";
        camera_ = I == "path" || I == "hybrid";
        particles_ = I == "light" || I == "hybrid";
        partition_ = I == "hybrid";
        W_ = det_->width;
        H_ = det_->height;
        film_ = Film(W_, H_);
        lightFilm_ = Film(W_, H_);
        gs_ = flattenScene(scene, detectorIndex, settings_);
        ctx_ = acquireContext(settings_.device);
        rayQuery_ = ctx_->rayQuery() && hardwareTraversalWanted();
        const uint32_t accel = rayQuery_ ? kSceneBinding : kNoBinding;
        pathKernel_ = rayQuery_ ? std::make_unique<Kernel>(*ctx_, owe_kernel_path_rq, owe_kernel_path_rq_size, kAllBindings,
                                                           uint32_t(sizeof(PushConstants)), "path (ray queries)", accel)
                                : std::make_unique<Kernel>(*ctx_, owe_kernel_path, owe_kernel_path_size, kAllBindings,
                                                           uint32_t(sizeof(PushConstants)), "path");
        if (particles_)
            particleKernel_ =
                rayQuery_ ? std::make_unique<Kernel>(*ctx_, owe_kernel_particle_rq, owe_kernel_particle_rq_size, kAllBindings,
                                                     uint32_t(sizeof(PushConstants)), "particle (ray queries)", accel)
                          : std::make_unique<Kernel>(*ctx_, owe_kernel_particle, owe_kernel_particle_size, kAllBindings,
                                                     uint32_t(sizeof(PushConstants)), "particle");
        if (sppm_) {
            const uint32_t pb = uint32_t(sizeof(PushConstants));
            sppmCamera_ = rayQuery_ ? std::make_unique<Kernel>(*ctx_, owe_kernel_sppm_camera_rq, owe_kernel_sppm_camera_rq_size,
                                                               kAllBindings, pb, "sppm camera (ray queries)", accel)
                                    : std::make_unique<Kernel>(*ctx_, owe_kernel_sppm_camera, owe_kernel_sppm_camera_size,
                                                               kAllBindings, pb, "sppm camera");
            sppmPhoton_ = rayQuery_ ? std::make_unique<Kernel>(*ctx_, owe_kernel_sppm_photon_rq, owe_kernel_sppm_photon_rq_size,
                                                               kAllBindings, pb, "sppm photon (ray queries)", accel)
                                    : std::make_unique<Kernel>(*ctx_, owe_kernel_sppm_photon, owe_kernel_sppm_photon_size,
                                                               kAllBindings, pb, "sppm photon");
            sppmGrid_ = std::make_unique<Kernel>(*ctx_, owe_kernel_sppm_grid, owe_kernel_sppm_grid_size, kAllBindings, pb, "sppm grid");
            sppmUpdate_ =
                std::make_unique<Kernel>(*ctx_, owe_kernel_sppm_update, owe_kernel_sppm_update_size, kAllBindings, pb, "sppm update");
            buffers_[kLevels] = ctx_->createDeviceBuffer(8 * sizeof(int32_t));
            buffers_[kGuide] = ctx_->createDeviceBuffer((2 * kGuideCells + 1) * sizeof(uint32_t));
            sppmGuide_ = std::make_unique<Kernel>(*ctx_, owe_kernel_sppm_guide, owe_kernel_sppm_guide_size, kAllBindings, pb,
                                                  "sppm guide");
        }
        if (vcm_) {
            const uint32_t pb = uint32_t(sizeof(PushConstants));
            vcmLight_ = rayQuery_ ? std::make_unique<Kernel>(*ctx_, owe_kernel_vcm_light_rq, owe_kernel_vcm_light_rq_size, kAllBindings,
                                                             pb, "vcm light (ray queries)", accel)
                                  : std::make_unique<Kernel>(*ctx_, owe_kernel_vcm_light, owe_kernel_vcm_light_size, kAllBindings, pb,
                                                             "vcm light");
            vcmCamera_ = rayQuery_ ? std::make_unique<Kernel>(*ctx_, owe_kernel_vcm_camera_rq, owe_kernel_vcm_camera_rq_size,
                                                              kAllBindings, pb, "vcm camera (ray queries)", accel)
                                   : std::make_unique<Kernel>(*ctx_, owe_kernel_vcm_camera, owe_kernel_vcm_camera_size, kAllBindings,
                                                              pb, "vcm camera");
            vcmGrid_ = std::make_unique<Kernel>(*ctx_, owe_kernel_vcm_grid, owe_kernel_vcm_grid_size, kAllBindings, pb, "vcm grid");
            // Sun light paths are guided as SPPM's photons: the same map and its CDF kernel.
            buffers_[kGuide] = ctx_->createDeviceBuffer((2 * kGuideCells + 1) * sizeof(uint32_t));
            sppmGuide_ = std::make_unique<Kernel>(*ctx_, owe_kernel_sppm_guide, owe_kernel_sppm_guide_size, kAllBindings, pb,
                                                  "sppm guide");
            buffers_[kVcmCounters] = ctx_->createDeviceBuffer(16 * sizeof(uint32_t));
            vcmCounterReadback_ = ctx_->createHostBuffer(16 * sizeof(uint32_t), true);
        }
        auto make = [&](Binding b, const void* data, size_t bytes) {
            buffers_[b] = ctx_->createDeviceBuffer(bytes);
            ctx_->upload(buffers_[b], data, bytes);
        };
        auto vec = [&](Binding b, const auto& v) { make(b, v.data(), v.size() * sizeof(v[0])); };
        make(kGlobals, &gs_.globals, sizeof gs_.globals);
        vec(kNodes, gs_.nodes);
        vec(kBoundaries, gs_.boundaries);
        vec(kShapeData, gs_.shapeData);
        vec(kTriangles, gs_.triangles);
        vec(kRegions, gs_.regions);
        vec(kMedia, gs_.media);
        vec(kSpectra, gs_.spectra);
        vec(kScalars, gs_.scalars);
        vec(kOptics, gs_.optics);
        vec(kEmissions, gs_.emissions);
        vec(kLights, gs_.lights);
        vec(kTop, gs_.topRecord);
        vec(kTexels, gs_.texels);
        vec(kNormals, gs_.normals);
        // Integrator buffers a kernel does not use are bound as placeholders (every kernel has the
        // same layout).
        for (Binding b : {kVisiblePoints, kSppmState, kFlux, kGridHead, kGridNext, kLevels, kGuide, kLightVertices, kVcmCounters})
            if (!buffers_[b].buffer) buffers_[b] = ctx_->createDeviceBuffer(16);
        buffers_[kCounters] = ctx_->createDeviceBuffer(16 * sizeof(uint32_t));
        counterReadback_ = ctx_->createHostBuffer(16 * sizeof(uint32_t), true);
        if (rayQuery_) hardware_ = std::make_unique<HardwareScene>(*ctx_, scene.world, gs_, buffers_[kTriangles]);
        allocateImageBuffers();
    }

    ~GpuRenderer() override {
        hardware_.reset();
        for (Buffer& b : buffers_) ctx_->destroy(b);
        ctx_->destroy(pixelReadback_);
        ctx_->destroy(splatReadback_);
        ctx_->destroy(counterReadback_);
        ctx_->destroy(sppmReadback_);
        ctx_->destroy(vcmCounterReadback_);
    }

    bool resetObserver() override {
        auto* eye = dynamic_cast<const IdealObserver*>(det_);
        if (!eye) return false;
        updateObserver(gs_, scene_.world, *eye);
        ctx_->upload(buffers_[kGlobals], &gs_.globals, sizeof gs_.globals);
        ctx_->upload(buffers_[kOptics], gs_.optics.data(), gs_.optics.size() * sizeof(GOptics));
        ctx_->upload(buffers_[kBoundaries], gs_.boundaries.data(), gs_.boundaries.size() * sizeof(GBoundary));
        ctx_->upload(buffers_[kNodes], gs_.nodes.data(), gs_.top->nodes().size() * sizeof(GNode));  // top level only
        if (hardware_) hardware_->rebase(gs_);
        if (W_ != eye->width || H_ != eye->height) {
            W_ = eye->width;
            H_ = eye->height;
            allocateImageBuffers();
        }
        film_ = Film(W_, H_);
        lightFilm_ = Film(W_, H_);
        particleCount_ = 0;
        passes_ = 0;
        totalSpp_ = 0;
        seconds_ = 0;
        chunk_ = 1;
        stats_ = {};
        if (sppm_) resetSppm();
        if (vcm_) resetVcm();
        return true;
    }

    void runPass(int spp) override {
        if (spp <= 0) return;
        auto t0 = std::chrono::steady_clock::now();
        if (camera_) passPath(uint32_t(spp));
        if (particles_) passParticles(uint32_t(spp));
        if (sppm_) passSppm(uint32_t(spp));
        if (vcm_) passVcm(uint32_t(spp));
        passes_++;
        totalSpp_ += spp;
        seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    Image resolve() const override {
        Image img;
        img.width = W_;
        img.height = H_;
        img.xyz.assign(film_.xyz.size(), 0.0);
        const double lightNorm = particleCount_ > 0 ? 1.0 / particleCount_ : 0;
        for (size_t i = 0; i < size_t(W_) * H_; ++i) {
            double norm = film_.samples[i] > 0 ? 1.0 / film_.samples[i] : 0;
            for (int c = 0; c < 3; ++c)
                img.xyz[3 * i + c] = film_.xyz[3 * i + c] * norm + lightFilm_.xyz[3 * i + c] * lightNorm;
        }
        if (sppm_ && iterations_ > 0) {
            // L = L_direct / iterations (above) + τ / (N_emitted · π r²).
            const double emitted = double(photonsPerIteration_) * double(iterations_);
            for (size_t i = 0; i < size_t(W_) * H_; ++i) {
                const float* st = &sppmState_[8 * i];
                const double r = st[4];
                if (!(r > 0)) continue;
                const double k = 1 / (emitted * Pi * r * r);
                for (int c = 0; c < 3; ++c) img.xyz[3 * i + c] += double(st[c]) * k;
            }
        }
        return img;
    }

    int passes() const override { return passes_; }
    long long samplesPerPixel() const override { return totalSpp_; }
    double seconds() const override { return seconds_; }
    const TransportStats& stats() const override { return stats_; }
    const Detector& detector() const override { return *det_; }
    const RenderSettings& settings() const override { return settings_; }
    std::string backend() const override {
        return "gpu: " + ctx_->info().name + " (" + ctx_->info().driver + (ctx_->info().portability ? ", MoltenVK" : "") +
               "; Slang kernels, float32, camera-relative, " + (rayQuery_ ? "hardware" : "software") + " traversal)";
    }

private:
    void allocateImageBuffers() {
        for (Binding b : {kPixels, kSplat}) ctx_->destroy(buffers_[b]);
        ctx_->destroy(pixelReadback_);
        ctx_->destroy(splatReadback_);
        const VkDeviceSize pixels = VkDeviceSize(W_) * H_;
        const VkDeviceSize pixelBytes = pixels * sizeof(PixelState);
        const bool splats = particles_ || vcm_;
        const VkDeviceSize splatBytes = splats ? pixels * kSplatWords * sizeof(uint32_t) : 16;
        buffers_[kPixels] = ctx_->createDeviceBuffer(pixelBytes);
        buffers_[kSplat] = ctx_->createDeviceBuffer(splatBytes);
        pixelReadback_ = ctx_->createHostBuffer(pixelBytes, true);
        if (splats) splatReadback_ = ctx_->createHostBuffer(splatBytes, true);
        if (vcm_) {
            allocateLightCache(uint64_t(pixels) * vertexBudget_);
            resetVcm();
            vcmSizePending_ = true;  // sized by a counting pass once the kernels are bound
        }
        if (sppm_) {
            for (Binding b : {kVisiblePoints, kSppmState, kFlux, kGridHead, kGridNext}) ctx_->destroy(buffers_[b]);
            ctx_->destroy(sppmReadback_);
            gridSize_ = 1024;
            while (gridSize_ < 2 * pixels) gridSize_ *= 2;
            buffers_[kVisiblePoints] = ctx_->createDeviceBuffer(pixels * kVisiblePointBytes);
            buffers_[kSppmState] = ctx_->createDeviceBuffer(pixels * kSppmPixelBytes);
            buffers_[kFlux] = ctx_->createDeviceBuffer(pixels * kFluxWords * sizeof(uint32_t));
            buffers_[kGridHead] = ctx_->createDeviceBuffer(VkDeviceSize(gridSize_) * sizeof(uint32_t));
            buffers_[kGridNext] = ctx_->createDeviceBuffer(pixels * sizeof(uint32_t));
            sppmReadback_ = ctx_->createHostBuffer(pixels * kSppmPixelBytes, true);
            resetSppm();
        }
        bindAll();
    }

    // The light vertex cache for `capacity` vertices (split evenly among the wavelength groups), its
    // hash (twice as many buckets) and links.
    void allocateLightCache(uint64_t capacity) {
        for (Binding b : {kLightVertices, kGridHead, kGridNext}) ctx_->destroy(buffers_[b]);
        const uint32_t G = vcmGroups();
        cacheCapacity_ = uint32_t(std::min<uint64_t>((capacity + G - 1) / G, 0xFFFFFFF0u / G)) * G;
        gridSize_ = 1024;
        while (gridSize_ < 2 * uint64_t(cacheCapacity_) && gridSize_ < (1u << 30)) gridSize_ *= 2;
        buffers_[kLightVertices] = ctx_->createDeviceBuffer(VkDeviceSize(cacheCapacity_) * kLightVertexBytes);
        buffers_[kGridHead] = ctx_->createDeviceBuffer(VkDeviceSize(gridSize_) * sizeof(uint32_t));
        buffers_[kGridNext] = ctx_->createDeviceBuffer(VkDeviceSize(cacheCapacity_) * sizeof(uint32_t));
    }

    void bindAll() {
        for (Kernel* k : {pathKernel_.get(), particleKernel_.get(), sppmCamera_.get(), sppmPhoton_.get(), sppmGrid_.get(),
                          sppmUpdate_.get(), sppmGuide_.get(), vcmLight_.get(), vcmCamera_.get(), vcmGrid_.get()})
            if (k) {
                for (uint32_t b = 0; b < kAllBindings; ++b) k->bind(b, buffers_[b]);
                const bool traces = k != sppmGrid_.get() && k != sppmUpdate_.get() && k != sppmGuide_.get() && k != vcmGrid_.get();
                if (hardware_ && traces) k->bindAccel(kSceneBinding, hardware_->tlas());
            }
    }

    // Light subpaths [0, lightPaths) with push constants pc, in dispatches of about 50 ms.
    void dispatchLightPaths(const PushConstants& pc, uint64_t lightPaths) {
        PushConstants lp = pc;
        uint64_t done = 0;
        while (done < lightPaths) {
            auto s0 = std::chrono::steady_clock::now();
            uint64_t batch = 0;
            ctx_->submit([&](VkCommandBuffer cmd) {
                for (int k = 0; k < 2 && done + batch < lightPaths; ++k) {
                    uint64_t base = done + batch;
                    uint32_t count = uint32_t(std::min<uint64_t>(particleChunk_, lightPaths - base));
                    lp.sampleBase = uint32_t(base);
                    lp.baseHi = uint32_t(base >> 32);
                    lp.sampleCount = count;
                    vcmLight_->dispatch(cmd, &lp, (count + kParticleGroup - 1) / kParticleGroup, 1);
                    computeBarrier(cmd);
                    batch += count;
                }
            });
            done += batch;
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            double perPath = ms / double(std::max<uint64_t>(1, batch));
            particleChunk_ = uint32_t(std::clamp(50.0 / std::max(perPath, 1e-9), double(kParticleGroup),
                                                 double(kMaxGroups) * kParticleGroup));
        }
    }

    // Light subpaths per iteration: `photons` per pixel (MIS accounts for any number), a multiple of
    // the wavelength groups (each takes the same number).
    uint32_t vcmGroups() const { return uint32_t(std::clamp(settings_.vcmWavelengthGroups, 1, 8)); }
    uint64_t vcmLightPaths() const {
        const uint64_t G = vcmGroups();
        const uint64_t n = std::max<uint64_t>(1, uint64_t(std::llround(double(W_) * H_ * settings_.photonsPerPixel)));
        return (n + G - 1) / G * G;
    }

    // The cache's size: one counting pass of light subpaths (nothing stored or splatted) measures
    // how many vertices a light subpath that meets the world leaves. Guiding (particle.slang) comes
    // to send most sun paths where they meet it, so the cache holds that many for every path, and
    // 30% more.
    void sizeLightCache() {
        vcmSizePending_ = false;
        if (!(gs_.emittedPower > 0)) return;
        const uint64_t lightPaths = vcmLightPaths();
        PushConstants pc = push(1);
        pc.flags |= (1u << 16) | (vcmGroups() << 24);
        pc.pad1 = uint32_t(lightPaths / vcmGroups());
        ctx_->submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kVcmCounters].buffer, 0, VK_WHOLE_SIZE, 0);
            transferToCompute(cmd);
        });
        dispatchLightPaths(pc, lightPaths);
        ctx_->submit([&](VkCommandBuffer cmd) {
            computeBarrier(cmd, true);
            VkBufferCopy c{0, 0, 16 * sizeof(uint32_t)};
            vkCmdCopyBuffer(cmd, buffers_[kVcmCounters].buffer, vcmCounterReadback_.buffer, 1, &c);
        });
        const uint32_t* n = static_cast<const uint32_t*>(vcmCounterReadback_.mapped);
        const double perHit = n[2] > 0 ? double(n[0]) / double(n[2]) : 1.0;
        vertexBudget_ = std::max(1.0, 1.3 * perHit);
        allocateLightCache(uint64_t(double(lightPaths) * vertexBudget_) + 1024);
        bindAll();
    }

    // BDPT/VCM iterations (shaders/vcm.slang): light subpaths into the cache (and splatted to the
    // eye), the cache into the hash (vcm), then the camera paths. One iteration per sample.
    void passVcm(uint32_t iterations) {
        if (vcmSizePending_) sizeLightCache();
        const uint32_t gx = (uint32_t(W_) + kPathGroup - 1) / kPathGroup, gy = (uint32_t(H_) + kPathGroup - 1) / kPathGroup;
        const uint64_t lightPaths = vcmLightPaths();
        ctx_->submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kCounters].buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, buffers_[kSplat].buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, buffers_[kVcmCounters].buffer, 0, VK_WHOLE_SIZE, 0);
            transferToCompute(cmd);
        });
        // The merging radius at a point: `photonRadius` pixel footprints at its distance from the eye,
        // shrinking over iterations as r_i = r_1 i^((α−1)/2) (Georgiev et al.).
        const auto* eye = dynamic_cast<const IdealObserver*>(det_);
        const double pixelAngle = 2 * eye->tanY() / eye->height;
        PushConstants pc = push(iterations);
        if (const char* m = std::getenv("OWE_VCM_MASK")) pc.flags |= uint32_t(std::atoi(m) & 0xFF) << 8;  // debugging
        pc.extra[1] = gridSize_ - 1;
        pc.flags |= vcmGroups() << 24;
        pc.pad0 = cacheCapacity_ / vcmGroups();
        pc.pad1 = uint32_t(lightPaths / vcmGroups());
        const double offset = double(uint32_t(settings_.seed * 0x9E3779B97F4A7C15ull >> 32)) / 4294967296.0;
        for (uint32_t it = 0; it < iterations; ++it) {
            pc.pass = uint32_t(iterations_);
            pc.sampleBase = it;
            double u = offset + 0.6180339887498949 * double(iterations_ + 1);
            pc.extra[0] = floatBits(float(u - std::floor(u)));
            const double shrink = std::pow(double(iterations_ + 1), (settings_.photonAlpha - 1) / 2);
            pc.extra[2] = floatBits(merging_ ? float(settings_.photonRadius * pixelAngle * shrink) : 0.0f);
            pc.extra[3] = floatBits(1e-9f);
            // The iteration's start: the cache emptied (its offered count; drops accumulate over the
            // pass), the hash cleared, the guide's CDF from what it has learnt.
            auto begin = [&](VkCommandBuffer cmd) {
                vkCmdFillBuffer(cmd, buffers_[kVcmCounters].buffer, 0, 4, 0);
                vkCmdFillBuffer(cmd, buffers_[kVcmCounters].buffer, 16, 12 * sizeof(uint32_t), 0);
                if (merging_) vkCmdFillBuffer(cmd, buffers_[kGridHead].buffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
                transferToCompute(cmd);
                sppmGuide_->dispatch(cmd, &pc, 1, 1);
                computeBarrier(cmd);
            };
            auto light = [&](VkCommandBuffer cmd, uint64_t base, uint32_t count) {
                PushConstants lp = pc;
                lp.sampleBase = uint32_t(base);
                lp.baseHi = uint32_t(base >> 32);
                lp.sampleCount = count;
                vcmLight_->dispatch(cmd, &lp, (count + kParticleGroup - 1) / kParticleGroup, 1);
                computeBarrier(cmd);
            };
            auto finish = [&](VkCommandBuffer cmd) {
                if (merging_) {
                    vcmGrid_->dispatch(cmd, &pc, (cacheCapacity_ + kParticleGroup - 1) / kParticleGroup, 1);
                    computeBarrier(cmd);
                }
                vcmCamera_->dispatch(cmd, &pc, gx, gy);
                computeBarrier(cmd);
            };
            const bool emits = gs_.emittedPower > 0;
            if (!emits || lightPaths <= particleChunk_) {
                // One submission: small images spend more on submissions than on paths.
                ctx_->submit([&](VkCommandBuffer cmd) {
                    begin(cmd);
                    if (emits) light(cmd, 0, uint32_t(lightPaths));
                    finish(cmd);
                });
            } else {
                ctx_->submit(begin);
                dispatchLightPaths(pc, lightPaths);
                ctx_->submit(finish);
            }
            particleCount_ += double(lightPaths);
            ++iterations_;
        }
        ctx_->submit([&](VkCommandBuffer cmd) {
            computeBarrier(cmd, true);
            VkBufferCopy region{0, 0, pixelReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kPixels].buffer, pixelReadback_.buffer, 1, &region);
            VkBufferCopy splat{0, 0, splatReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kSplat].buffer, splatReadback_.buffer, 1, &splat);
            VkBufferCopy counters{0, 0, counterReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kCounters].buffer, counterReadback_.buffer, 1, &counters);
            VkBufferCopy cache{0, 0, 16 * sizeof(uint32_t)};
            vkCmdCopyBuffer(cmd, buffers_[kVcmCounters].buffer, vcmCounterReadback_.buffer, 1, &cache);
        });
        const uint32_t dropped = static_cast<const uint32_t*>(vcmCounterReadback_.mapped)[1];
        const PixelState* px = static_cast<const PixelState*>(pixelReadback_.mapped);
        TransportStats st;
        for (int i = 0; i < W_ * H_; ++i) {
            for (int c = 0; c < 3; ++c) film_.xyz[3 * size_t(i) + c] += px[i].acc[c];
            film_.samples[size_t(i)] += iterations;
            st.paths += px[i].stats[0];
            st.segments += px[i].stats[1];
            st.inconsistencies += px[i].stats[2];
            st.leaks += px[i].stats[3];
        }
        const uint32_t* w = static_cast<const uint32_t*>(splatReadback_.mapped);
        for (size_t i = 0; i < size_t(W_) * H_; ++i)
            for (int c = 0; c < 3; ++c) {
                const uint32_t* bins = w + i * kSplatWords + size_t(c) * kSplatBins * 2;
                double v = 0;
                for (int b = 0; b < kSplatBins; ++b)
                    if (bins[2 * b] | bins[2 * b + 1])
                        v += std::ldexp(double(bins[2 * b + 1]) * 4294967296.0 + double(bins[2 * b]), kSplatExp0 + 32 * b);
                lightFilm_.xyz[3 * i + size_t(c)] += v;
            }
        const uint32_t* n = static_cast<const uint32_t*>(counterReadback_.mapped);
        st.paths += word64(n);
        st.segments += word64(n + 2);
        st.inconsistencies += word64(n + 4);
        st.leaks += word64(n + 6);
        readFirstInconsistency(st);
        stats_.add(st);
        // A full cache dropped vertices (their connections and merges are missing): grow it for
        // the passes to come.
        if (dropped > 0) {
            std::fprintf(stderr, "vcm: the light vertex cache (%u vertices) was full; %u dropped this pass, growing it\n",
                         cacheCapacity_, dropped);
            vertexBudget_ *= 1.5;
            allocateLightCache(uint64_t(double(lightPaths) * vertexBudget_));
            bindAll();
        }
    }

    // A fresh BDPT/VCM estimate: the radius schedule restarts and the guide forgets.
    void resetVcm() {
        iterations_ = 0;
        ctx_->submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kGuide].buffer, 0, VK_WHOLE_SIZE, 0);
            transferToCompute(cmd);
        });
    }

    // A fresh SPPM estimate: no radius, flux or photon count yet; the eye's pixel angle sets the
    // initial radii, as a few pixel footprints at the distance of each pixel's first visible point.
    void resetSppm() {
        iterations_ = 0;
        photonsPerIteration_ = std::max<uint64_t>(1, uint64_t(std::llround(double(W_) * H_ * settings_.photonsPerPixel)));
        sppmState_.assign(size_t(W_) * H_ * 8, 0.0f);
        double pixelAngle = 0, radiusFloor = 0;
        if (auto* eye = dynamic_cast<const IdealObserver*>(det_)) {
            pixelAngle = 2 * eye->tanY() / eye->height;
        } else if (auto* s = dynamic_cast<const SurfaceSensor*>(det_)) {
            // A sensor's pixel seen from the pupil it looks through (a camera's exit pupil); a bare
            // screen looks at the whole hemisphere: a fixed radius of a few pixel sizes.
            const double pixel = 2 * s->halfY / s->height;
            const Vec3 aim = s->focusRadius > 0 ? s->focusCenter : s->aimCenter;
            const double d = s->hasAim ? length(aim - s->toWorld().t) : 0;
            if (d > 0) pixelAngle = pixel / d;
            else radiusFloor = settings_.photonRadius * pixel;
        }
        r0PerMetre_ = float(settings_.photonRadius * pixelAngle);
        r0Floor_ = float(std::max(radiusFloor, 1e-9));
        ctx_->submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kSppmState].buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, buffers_[kFlux].buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, buffers_[kGuide].buffer, 0, VK_WHOLE_SIZE, 0);
            transferToCompute(cmd);
        });
    }

    // SPPM iterations (shaders/sppm.slang): camera pass and hash, photons, update.
    void passSppm(uint32_t iterations) {
        const uint32_t gx = (uint32_t(W_) + kPathGroup - 1) / kPathGroup, gy = (uint32_t(H_) + kPathGroup - 1) / kPathGroup;
        const uint32_t pixelGroups = (uint32_t(W_) * uint32_t(H_) + kParticleGroup - 1) / kParticleGroup;
        ctx_->submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kCounters].buffer, 0, VK_WHOLE_SIZE, 0);
            transferToCompute(cmd);
        });
        PushConstants pc = push(iterations);
        pc.extra[1] = gridSize_ - 1;
        pc.extra[2] = floatBits(float(settings_.photonAlpha));
        pc.extra[3] = floatBits(r0PerMetre_);
        pc.pad0 = floatBits(r0Floor_);
        pc.pad1 = uint32_t(std::clamp(settings_.photonWavelengthGroups, 1, 255));
        // Hero wavelengths rotate over iterations by the golden ratio (a low-discrepancy sequence),
        // offset by the seed; every path and photon of one iteration uses them.
        const double offset = double(uint32_t(settings_.seed * 0x9E3779B97F4A7C15ull >> 32)) / 4294967296.0;
        for (uint32_t it = 0; it < iterations; ++it) {
            pc.pass = uint32_t(iterations_);
            pc.sampleBase = it;
            double u = offset + 0.6180339887498949 * double(iterations_ + 1);
            pc.extra[0] = floatBits(float(u - std::floor(u)));
            ctx_->submit([&](VkCommandBuffer cmd) {
                vkCmdFillBuffer(cmd, buffers_[kGridHead].buffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
                vkCmdFillBuffer(cmd, buffers_[kLevels].buffer, 0, 4, 0x7FFFFFFFu);
                vkCmdFillBuffer(cmd, buffers_[kLevels].buffer, 4, 4, 0x80000001u);
                transferToCompute(cmd);
                sppmCamera_->dispatch(cmd, &pc, gx, gy);
                computeBarrier(cmd);
                sppmGrid_->dispatch(cmd, &pc, pixelGroups, 1);
                sppmGuide_->dispatch(cmd, &pc, 1, 1);  // the photon guide learnt so far
                computeBarrier(cmd);
            });
            if (gs_.emittedPower > 0) {
                PushConstants pp = pc;
                uint64_t done = 0;
                while (done < photonsPerIteration_) {
                    auto s0 = std::chrono::steady_clock::now();
                    uint64_t batch = 0;
                    ctx_->submit([&](VkCommandBuffer cmd) {
                        for (int k = 0; k < 2 && done + batch < photonsPerIteration_; ++k) {
                            uint64_t base = done + batch;
                            uint32_t count = uint32_t(std::min<uint64_t>(particleChunk_, photonsPerIteration_ - base));
                            pp.sampleBase = uint32_t(base);
                            pp.baseHi = uint32_t(base >> 32);
                            pp.sampleCount = count;
                            sppmPhoton_->dispatch(cmd, &pp, (count + kParticleGroup - 1) / kParticleGroup, 1);
                            computeBarrier(cmd);
                            batch += count;
                        }
                    });
                    done += batch;
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
                    double perPhoton = ms / double(std::max<uint64_t>(1, batch));
                    particleChunk_ = uint32_t(std::clamp(50.0 / std::max(perPhoton, 1e-9), double(kParticleGroup),
                                                         double(kMaxGroups) * kParticleGroup));
                }
            }
            ctx_->submit([&](VkCommandBuffer cmd) {
                sppmUpdate_->dispatch(cmd, &pc, pixelGroups, 1);
                computeBarrier(cmd);
            });
            ++iterations_;
        }
        ctx_->submit([&](VkCommandBuffer cmd) {
            computeBarrier(cmd, true);
            VkBufferCopy region{0, 0, pixelReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kPixels].buffer, pixelReadback_.buffer, 1, &region);
            VkBufferCopy state{0, 0, sppmReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kSppmState].buffer, sppmReadback_.buffer, 1, &state);
            VkBufferCopy counters{0, 0, counterReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kCounters].buffer, counterReadback_.buffer, 1, &counters);
        });
        const PixelState* px = static_cast<const PixelState*>(pixelReadback_.mapped);
        TransportStats st;
        for (int i = 0; i < W_ * H_; ++i) {
            for (int c = 0; c < 3; ++c) film_.xyz[3 * size_t(i) + c] += px[i].acc[c];
            film_.samples[size_t(i)] += iterations;
            st.paths += px[i].stats[0];
            st.segments += px[i].stats[1];
            st.inconsistencies += px[i].stats[2];
            st.leaks += px[i].stats[3];
        }
        std::memcpy(sppmState_.data(), sppmReadback_.mapped, sppmState_.size() * sizeof(float));
        if (std::getenv("OWE_SPPM_DEBUG")) {
            Buffer rb = ctx_->createHostBuffer(32, true);
            ctx_->submit([&](VkCommandBuffer cmd) {
                VkBufferCopy c{0, 0, 32};
                vkCmdCopyBuffer(cmd, buffers_[kLevels].buffer, rb.buffer, 1, &c);
            });
            const int32_t* v = static_cast<const int32_t*>(rb.mapped);
            auto of = [](int32_t i) { int32_t j = i >= 0 ? i : i ^ 0x7FFFFFFF; float f; std::memcpy(&f, &j, 4); return f; };
            std::fprintf(stderr, "sppm: levels %d..%d, bounds (%g %g %g)..(%g %g %g), photons/iteration %llu\n", v[0], v[1],
                         of(v[2]), of(v[3]), of(v[4]), of(v[5]), of(v[6]), of(v[7]), (unsigned long long)photonsPerIteration_);
            ctx_->destroy(rb);
        }
        const uint32_t* n = static_cast<const uint32_t*>(counterReadback_.mapped);
        st.paths += word64(n);
        st.segments += word64(n + 2);
        st.inconsistencies += word64(n + 4);
        st.leaks += word64(n + 6);
        readFirstInconsistency(st);
        stats_.add(st);
    }

    // The first region inconsistency a kernel recorded (words 8–15 of the counters), into st.first.
    void readFirstInconsistency(TransportStats& st) const {
        const uint32_t* n = static_cast<const uint32_t*>(counterReadback_.mapped);
        if (n[8] == 0 || st.inconsistencies == 0) return;
        float p[3];
        std::memcpy(p, n + 13, sizeof p);
        st.first = {n[9], n[10], n[11], n[12] == kGpuNone ? kNone : n[12],
                    Vec3(p[0], p[1], p[2]) + gs_.origin};
    }

    PushConstants push(uint32_t spp) const {
        PushConstants pc{};
        pc.pass = uint32_t(passes_);
        pc.spp = spp;
        pc.seedLo = uint32_t(settings_.seed);
        pc.seedHi = uint32_t(settings_.seed >> 32);
        pc.width = uint32_t(W_);
        pc.height = uint32_t(H_);
        pc.flags = partition_ ? kFlagPartition : 0;
        return pc;
    }

    // Camera paths: one thread per pixel, `spp` samples in slices of chunk_.
    void passPath(uint32_t spp) {
        ctx_->submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kCounters].buffer, 0, VK_WHOLE_SIZE, 0);
            transferToCompute(cmd);
        });
        PushConstants pc = push(spp);
        const uint32_t gx = (uint32_t(W_) + kPathGroup - 1) / kPathGroup, gy = (uint32_t(H_) + kPathGroup - 1) / kPathGroup;
        // Samples per dispatch adapt to keep each submission near 50 ms; dispatches are batched
        // into submissions of ~100 ms.
        uint32_t done = 0;
        while (done < spp) {
            auto s0 = std::chrono::steady_clock::now();
            uint32_t batchSamples = 0;
            ctx_->submit([&](VkCommandBuffer cmd) {
                for (int k = 0; k < 2 && done + batchSamples < spp; ++k) {
                    pc.sampleBase = done + batchSamples;
                    pc.sampleCount = std::min(chunk_, spp - pc.sampleBase);
                    pathKernel_->dispatch(cmd, &pc, gx, gy);
                    computeBarrier(cmd);
                    batchSamples += pc.sampleCount;
                }
            });
            done += batchSamples;
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            double perSample = ms / std::max(1u, batchSamples);
            chunk_ = uint32_t(std::clamp(50.0 / std::max(perSample, 1e-3), 1.0, 4096.0));
        }
        ctx_->submit([&](VkCommandBuffer cmd) {
            computeBarrier(cmd, true);
            VkBufferCopy region{0, 0, pixelReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kPixels].buffer, pixelReadback_.buffer, 1, &region);
            VkBufferCopy counters{0, 0, counterReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kCounters].buffer, counterReadback_.buffer, 1, &counters);
        });
        const PixelState* px = static_cast<const PixelState*>(pixelReadback_.mapped);
        TransportStats st;
        for (int i = 0; i < W_ * H_; ++i) {
            film_.xyz[3 * size_t(i)] += px[i].acc[0];
            film_.xyz[3 * size_t(i) + 1] += px[i].acc[1];
            film_.xyz[3 * size_t(i) + 2] += px[i].acc[2];
            film_.samples[size_t(i)] += spp;
            st.paths += px[i].stats[0];
            st.segments += px[i].stats[1];
            st.inconsistencies += px[i].stats[2];
            st.leaks += px[i].stats[3];
        }
        readFirstInconsistency(st);
        stats_.add(st);
    }

    // Particles: W·H·ppp per pass, one thread each, in slices of particleChunk_.
    void passParticles(uint32_t ppp) {
        const uint64_t total = uint64_t(W_) * uint64_t(H_) * ppp;
        particleCount_ += double(total);
        if (!(gs_.emittedPower > 0)) return;  // nothing emits: the particle estimate is zero
        ctx_->submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kSplat].buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, buffers_[kCounters].buffer, 0, VK_WHOLE_SIZE, 0);
            transferToCompute(cmd);
        });
        PushConstants pc = push(ppp);
        uint64_t done = 0;
        while (done < total) {
            auto s0 = std::chrono::steady_clock::now();
            uint64_t batch = 0;
            ctx_->submit([&](VkCommandBuffer cmd) {
                for (int k = 0; k < 2 && done + batch < total; ++k) {
                    uint64_t base = done + batch;
                    uint32_t count = uint32_t(std::min<uint64_t>(particleChunk_, total - base));
                    pc.sampleBase = uint32_t(base);
                    pc.baseHi = uint32_t(base >> 32);
                    pc.sampleCount = count;
                    particleKernel_->dispatch(cmd, &pc, (count + kParticleGroup - 1) / kParticleGroup, 1);
                    computeBarrier(cmd);
                    batch += count;
                }
            });
            done += batch;
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            double perParticle = ms / double(std::max<uint64_t>(1, batch));
            particleChunk_ = uint32_t(std::clamp(50.0 / std::max(perParticle, 1e-9), double(kParticleGroup),
                                                 double(kMaxGroups) * kParticleGroup));
        }
        ctx_->submit([&](VkCommandBuffer cmd) {
            computeBarrier(cmd, true);
            VkBufferCopy splat{0, 0, splatReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kSplat].buffer, splatReadback_.buffer, 1, &splat);
            VkBufferCopy counters{0, 0, counterReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kCounters].buffer, counterReadback_.buffer, 1, &counters);
        });
        // The exact integer bins, converted to double: Σ_b (hi·2^32 + lo) · 2^(kSplatExp0 + 32 b).
        const uint32_t* w = static_cast<const uint32_t*>(splatReadback_.mapped);
        for (size_t i = 0; i < size_t(W_) * H_; ++i)
            for (int c = 0; c < 3; ++c) {
                const uint32_t* bins = w + i * kSplatWords + size_t(c) * kSplatBins * 2;
                double v = 0;
                for (int b = 0; b < kSplatBins; ++b)
                    if (bins[2 * b] | bins[2 * b + 1])
                        v += std::ldexp(double(bins[2 * b + 1]) * 4294967296.0 + double(bins[2 * b]), kSplatExp0 + 32 * b);
                lightFilm_.xyz[3 * i + size_t(c)] += v;
            }
        const uint32_t* n = static_cast<const uint32_t*>(counterReadback_.mapped);
        TransportStats st;
        st.paths = word64(n);
        st.segments = word64(n + 2);
        st.inconsistencies = word64(n + 4);
        st.leaks = word64(n + 6);
        readFirstInconsistency(st);
        stats_.add(st);
    }

    const Scene& scene_;
    const Detector* det_ = nullptr;
    RenderSettings settings_;
    bool camera_ = true, particles_ = false, partition_ = false, sppm_ = false;  // vcm_ below
    int W_ = 0, H_ = 0;
    GpuScene gs_;
    std::shared_ptr<Context> ctx_;
    bool rayQuery_ = false;
    std::unique_ptr<HardwareScene> hardware_;
    std::unique_ptr<Kernel> pathKernel_, particleKernel_;
    std::unique_ptr<Kernel> sppmCamera_, sppmPhoton_, sppmGrid_, sppmUpdate_, sppmGuide_;
    std::unique_ptr<Kernel> vcmLight_, vcmCamera_, vcmGrid_;
    Buffer buffers_[kAllBindings];
    Buffer pixelReadback_, splatReadback_, counterReadback_, sppmReadback_, vcmCounterReadback_;
    // BDPT/VCM: whether it merges, the light vertex cache's capacity and its budget per light path
    // (grown when a pass fills it).
    bool vcm_ = false, merging_ = false;
    uint32_t cacheCapacity_ = 0;
    double vertexBudget_ = 6;
    bool vcmSizePending_ = false;
    // SPPM: iterations so far, photons per iteration, hash size, radius parameters, and the last
    // read-back pixel states (τ xyz, N; r, unused ×3).
    long long iterations_ = 0;
    uint64_t photonsPerIteration_ = 0;
    uint32_t gridSize_ = 0;
    float r0PerMetre_ = 0, r0Floor_ = 0;
    std::vector<float> sppmState_;
    Film film_;       // camera-path estimates (per-pixel sample counts)
    Film lightFilm_;  // particle splats (normalised by the particle count)
    double particleCount_ = 0;
    int passes_ = 0;
    long long totalSpp_ = 0;
    double seconds_ = 0;
    TransportStats stats_;
    uint32_t chunk_ = 1;
    uint32_t particleChunk_ = 16384;
};

}  // namespace

std::unique_ptr<Renderer> makeRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings) {
    return std::make_unique<GpuRenderer>(scene, detectorIndex, settings);
}

}  // namespace owe::gpu
