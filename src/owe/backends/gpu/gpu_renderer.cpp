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

namespace owe::gpu {

namespace {

struct PushConstants {  // mirrors shaders/scene.slang
    uint32_t pass, spp, sampleBase, sampleCount;
    uint32_t seedLo, seedHi, width, height;
    uint32_t flags, baseHi, pad0, pad1;
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
constexpr uint32_t kParticleGroup = 64;
constexpr uint32_t kMaxGroups = 65535;

enum Binding : uint32_t {
    kGlobals, kNodes, kBoundaries, kShapeData, kTriangles, kRegions, kMedia, kSpectra, kScalars, kOptics, kEmissions,
    kLights, kPixels, kSplat, kCounters, kTop, kBindingCount
};
constexpr uint32_t kSceneBinding = kBindingCount;  // the acceleration structure (ray-query kernels)

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

class GpuRenderer : public Renderer {
public:
    GpuRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings)
        : scene_(scene), settings_(settings), stats_() {
        if (detectorIndex < 0 || detectorIndex >= int(scene.detectors.size())) throw std::runtime_error("no such detector");
        det_ = scene.detectors[size_t(detectorIndex)].get();
        const std::string& I = settings_.integrator;
        if (I != "path" && I != "light" && I != "hybrid")
            throw std::runtime_error("unknown integrator '" + I + "' (expected path, light or hybrid)");
        if (I == "hybrid" && !det_->isVirtual())
            throw std::runtime_error("the hybrid integrator needs a virtual observer; use path or light for surface sensors");
        camera_ = I != "light";
        particles_ = I != "path";
        partition_ = I == "hybrid";
        W_ = det_->width;
        H_ = det_->height;
        film_ = Film(W_, H_);
        lightFilm_ = Film(W_, H_);
        gs_ = flattenScene(scene, detectorIndex, settings_);
        ctx_ = acquireContext(settings_.device);
        rayQuery_ = ctx_->rayQuery() && hardwareTraversalWanted();
        const uint32_t accel = rayQuery_ ? kSceneBinding : kNoBinding;
        pathKernel_ = rayQuery_ ? std::make_unique<Kernel>(*ctx_, owe_kernel_path_rq, owe_kernel_path_rq_size, kBindingCount,
                                                           uint32_t(sizeof(PushConstants)), "path (ray queries)", accel)
                                : std::make_unique<Kernel>(*ctx_, owe_kernel_path, owe_kernel_path_size, kBindingCount,
                                                           uint32_t(sizeof(PushConstants)), "path");
        if (particles_)
            particleKernel_ =
                rayQuery_ ? std::make_unique<Kernel>(*ctx_, owe_kernel_particle_rq, owe_kernel_particle_rq_size, kBindingCount,
                                                     uint32_t(sizeof(PushConstants)), "particle (ray queries)", accel)
                          : std::make_unique<Kernel>(*ctx_, owe_kernel_particle, owe_kernel_particle_size, kBindingCount,
                                                     uint32_t(sizeof(PushConstants)), "particle");
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
        buffers_[kCounters] = ctx_->createDeviceBuffer(8 * sizeof(uint32_t));
        counterReadback_ = ctx_->createHostBuffer(8 * sizeof(uint32_t), true);
        if (rayQuery_) hardware_ = std::make_unique<HardwareScene>(*ctx_, scene.world, gs_, buffers_[kTriangles]);
        allocateImageBuffers();
    }

    ~GpuRenderer() override {
        hardware_.reset();
        for (Buffer& b : buffers_) ctx_->destroy(b);
        ctx_->destroy(pixelReadback_);
        ctx_->destroy(splatReadback_);
        ctx_->destroy(counterReadback_);
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
        return true;
    }

    void runPass(int spp) override {
        if (spp <= 0) return;
        auto t0 = std::chrono::steady_clock::now();
        if (camera_) passPath(uint32_t(spp));
        if (particles_) passParticles(uint32_t(spp));
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
        const VkDeviceSize pixelBytes = VkDeviceSize(W_) * H_ * sizeof(PixelState);
        const VkDeviceSize splatBytes = particles_ ? VkDeviceSize(W_) * H_ * kSplatWords * sizeof(uint32_t) : 16;
        buffers_[kPixels] = ctx_->createDeviceBuffer(pixelBytes);
        buffers_[kSplat] = ctx_->createDeviceBuffer(splatBytes);
        pixelReadback_ = ctx_->createHostBuffer(pixelBytes, true);
        if (particles_) splatReadback_ = ctx_->createHostBuffer(splatBytes, true);
        for (Kernel* k : {pathKernel_.get(), particleKernel_.get()})
            if (k) {
                for (uint32_t b = 0; b < kBindingCount; ++b) k->bind(b, buffers_[b]);
                if (hardware_) k->bindAccel(kSceneBinding, hardware_->tlas());
            }
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
        PushConstants pc = push(spp);
        const uint32_t gx = (uint32_t(W_) + 7) / 8, gy = (uint32_t(H_) + 7) / 8;
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
        stats_.add(st);
    }

    const Scene& scene_;
    const Detector* det_ = nullptr;
    RenderSettings settings_;
    bool camera_ = true, particles_ = false, partition_ = false;
    int W_ = 0, H_ = 0;
    GpuScene gs_;
    std::shared_ptr<Context> ctx_;
    bool rayQuery_ = false;
    std::unique_ptr<HardwareScene> hardware_;
    std::unique_ptr<Kernel> pathKernel_, particleKernel_;
    Buffer buffers_[kBindingCount];
    Buffer pixelReadback_, splatReadback_, counterReadback_;
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
