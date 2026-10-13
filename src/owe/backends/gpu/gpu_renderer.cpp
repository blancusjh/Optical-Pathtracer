#include "owe/backends/gpu/gpu_renderer_internal.hpp"

extern const unsigned char owe_kernel_film[];
extern const size_t owe_kernel_film_size;
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

namespace owe::gpu::detail {

GpuRenderer::GpuRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings)
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
    // With scattering media, VCM and BDPT take the paths without a medium scattering vertex
    // (media as extinction) and the path tracer, in a pass of its own, all the others.
    if (vcm_)
        for (const Medium& m : scene.world.media())
            if (m.scatters()) mediumPartition_ = true;
    sppm_ = I == "sppm";
    camera_ = I == "path" || I == "hybrid";
    particles_ = I == "light" || I == "hybrid";
    partition_ = I == "hybrid";
    W_ = det_->width;
    H_ = det_->height;

    gs_ = flattenScene(scene, detectorIndex, settings_);
    ctx_ = acquireContext(settings_.device);
    rayQuery_ = ctx_->rayQuery() && hardwareTraversalWanted();
    const uint32_t accel = rayQuery_ ? kSceneBinding : kNoBinding;
    // Only this integrator's kernels, compiled in parallel: each one is a separate driver
    // compilation of up to a minute when neither cache holds it.
    struct KernelJob {
        std::unique_ptr<Kernel>* slot;
        const unsigned char* spirv;
        size_t bytes;
        const char* name;
        bool traces;
    };
    std::vector<KernelJob> jobs;
    auto tracing = [&](std::unique_ptr<Kernel>& slot, const unsigned char* rq, size_t rqBytes, const char* rqName,
                       const unsigned char* sw, size_t swBytes, const char* swName) {
        jobs.push_back(rayQuery_ ? KernelJob{&slot, rq, rqBytes, rqName, true} : KernelJob{&slot, sw, swBytes, swName, false});
    };
    auto plain = [&](std::unique_ptr<Kernel>& slot, const unsigned char* code, size_t bytes, const char* name) {
        jobs.push_back(KernelJob{&slot, code, bytes, name, false});
    };
    plain(filmKernel_, owe_kernel_film, owe_kernel_film_size, "film accumulation");
    if (camera_ || mediumPartition_)
        tracing(pathKernel_, owe_kernel_path_rq, owe_kernel_path_rq_size, "path (ray queries)", owe_kernel_path,
                owe_kernel_path_size, "path");
    if (particles_)
        tracing(particleKernel_, owe_kernel_particle_rq, owe_kernel_particle_rq_size, "particle (ray queries)",
                owe_kernel_particle, owe_kernel_particle_size, "particle");
    if (sppm_) {
        tracing(sppmCamera_, owe_kernel_sppm_camera_rq, owe_kernel_sppm_camera_rq_size, "sppm camera (ray queries)",
                owe_kernel_sppm_camera, owe_kernel_sppm_camera_size, "sppm camera");
        tracing(sppmPhoton_, owe_kernel_sppm_photon_rq, owe_kernel_sppm_photon_rq_size, "sppm photon (ray queries)",
                owe_kernel_sppm_photon, owe_kernel_sppm_photon_size, "sppm photon");
        plain(sppmGrid_, owe_kernel_sppm_grid, owe_kernel_sppm_grid_size, "sppm grid");
        plain(sppmUpdate_, owe_kernel_sppm_update, owe_kernel_sppm_update_size, "sppm update");
        buffers_[kLevels] = ctx_->createDeviceBuffer(8 * sizeof(int32_t));
    }
    if (vcm_) {
        tracing(vcmLight_, owe_kernel_vcm_light_rq, owe_kernel_vcm_light_rq_size, "vcm light (ray queries)",
                owe_kernel_vcm_light, owe_kernel_vcm_light_size, "vcm light");
        tracing(vcmCamera_, owe_kernel_vcm_camera_rq, owe_kernel_vcm_camera_rq_size, "vcm camera (ray queries)",
                owe_kernel_vcm_camera, owe_kernel_vcm_camera_size, "vcm camera");
        plain(vcmGrid_, owe_kernel_vcm_grid, owe_kernel_vcm_grid_size, "vcm grid");
        buffers_[kVcmCounters] = ctx_->createDeviceBuffer(16 * sizeof(uint32_t));
        vcmCounterReadback_ = ctx_->createHostBuffer(16 * sizeof(uint32_t), true);
    }
    if (sppm_ || vcm_) {
        // SPPM's photons and VCM's sun light paths are guided by the same map and its CDF kernel.
        plain(sppmGuide_, owe_kernel_sppm_guide, owe_kernel_sppm_guide_size, "sppm guide");
        buffers_[kGuide] = ctx_->createDeviceBuffer((2 * kGuideCells + 1) * sizeof(uint32_t));
    }
    {
        std::vector<std::future<void>> compiling;
        for (const KernelJob& j : jobs)
            compiling.push_back(std::async(std::launch::async, [this, j, accel] {
                *j.slot = std::make_unique<Kernel>(*ctx_, j.spirv, j.bytes, kAllBindings, uint32_t(sizeof(PushConstants)),
                                                   j.name, j.traces ? accel : kNoBinding);
            }));
        std::exception_ptr failed;
        for (auto& c : compiling) try {
                c.get();
            } catch (...) {
                if (!failed) failed = std::current_exception();
            }
        if (failed) std::rethrow_exception(failed);
        ctx_->savePipelineCache();
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
    // Images: the header, then each image's words in place (never all of them copied at once).
    buffers_[kTexels] = ctx_->createDeviceBuffer(std::max<size_t>(gs_.texelWords, 4) * 4);
    ctx_->uploadAt(buffers_[kTexels], 0, gs_.texels.data(), gs_.texels.size() * 4);
    for (size_t i = 0; i < gs_.images.size(); ++i) {
        const ImageRGB& img = *gs_.images[i];
        const void* words = img.hdr ? static_cast<const void*>(img.rgb.data()) : static_cast<const void*>(img.srgb8.data());
        const size_t bytes = img.hdr ? img.rgb.size() * 4 : img.srgb8.size() * 4;
        ctx_->uploadAt(buffers_[kTexels], VkDeviceSize(gs_.texels[4 * i]) * 4, words, bytes);
    }
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

GpuRenderer::~GpuRenderer() {
    hardware_.reset();
    for (Buffer& b : buffers_) ctx_->destroy(b);
    ctx_->destroy(filmReadback_);
    ctx_->destroy(counterReadback_);
    ctx_->destroy(sppmReadback_);
    ctx_->destroy(vcmCounterReadback_);
    ctx_->destroy(vcmGuideBackup_);
}

void GpuRenderer::runPass(int spp) {
    if (spp <= 0) return;
    auto t0 = std::chrono::steady_clock::now();
    if (camera_) passPath(uint32_t(spp));
    if (particles_) passParticles(uint32_t(spp));
    if (sppm_) passSppm(uint32_t(spp));
    if (vcm_) passVcm(uint32_t(spp));
    if (mediumPartition_) passPath(uint32_t(spp), kFlagMediumOnly);
    passes_++;
    totalSpp_ += spp;
    seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool GpuRenderer::resetObserver() {
    auto* eye = dynamic_cast<const IdealObserver*>(det_);
    if (!eye) return false;
    updateObserver(gs_, scene_.world, *eye);
    if (gs_.sphereFramesChanged)
        ctx_->upload(buffers_[kShapeData], gs_.shapeData.data(), gs_.shapeData.size() * sizeof(float));
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

    clearFilm();
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

void GpuRenderer::allocateImageBuffers() {
    for (Binding b : {kPixels, kSplat, kFilm}) ctx_->destroy(buffers_[b]);
    ctx_->destroy(filmReadback_);
    const VkDeviceSize pixels = VkDeviceSize(W_) * H_;
    const VkDeviceSize pixelBytes = pixels * sizeof(PixelState);
    const bool splats = particles_ || vcm_;
    const VkDeviceSize splatBytes = splats ? pixels * kSplatWords * sizeof(uint32_t) : 16;
    buffers_[kPixels] = ctx_->createDeviceBuffer(pixelBytes);
    buffers_[kSplat] = ctx_->createDeviceBuffer(splatBytes);
    buffers_[kFilm] = ctx_->createDeviceBuffer(pixels * kFilmPixelBytes);
    filmReadback_ = ctx_->createHostBuffer(pixels * kFilmPixelBytes, true);
    clearFilm();
    if (vcm_) {
        // Optional initial capacity for diagnosing/reproducing cache overflow.
        // This overrides the counting heuristic only; retries still grow the
        // cache to hold every vertex.
        const uint64_t initial = cacheEnvironment("OWE_VCM_CACHE_CAPACITY");
        allocateLightCache(initial ? initial : uint64_t(pixels) * vertexBudget_);
        if (!vcmGuideBackup_.buffer) vcmGuideBackup_ = ctx_->createDeviceBuffer(buffers_[kGuide].size);
        resetVcm();
        vcmSizePending_ = initial == 0; // sized once the kernels are bound
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

void GpuRenderer::allocateLightCache(uint64_t capacity) {
    const uint32_t G = vcmGroups();
    const uint64_t limit = std::min<uint64_t>(uint64_t(kMaxGroups) * kParticleGroup, ctx_->maxStorageBufferRange() / kLightVertexBytes);
    const uint64_t configuredLimit = cacheEnvironment("OWE_VCM_CACHE_MAX_CAPACITY");
    const uint64_t boundedLimit = configuredLimit ? std::min(limit, configuredLimit) : limit;
    const uint64_t rounded = (std::max<uint64_t>(capacity, 1) + G - 1) / G * G;
    if (rounded > boundedLimit)
        throw std::runtime_error("vcm: complete light vertex cache needs " + std::to_string(rounded) +
                                 " vertices, exceeding capacity limit " + std::to_string(boundedLimit));
    uint32_t gridSize = 1024;
    while (gridSize < 2 * rounded && gridSize < (1u << 30)) gridSize *= 2;
    if (VkDeviceSize(gridSize) * sizeof(uint32_t) > ctx_->maxStorageBufferRange())
        throw std::runtime_error("vcm: light vertex hash exceeds the device storage buffer limit");
    // Keep the old cache usable if an allocation fails. No descriptors point to
    // a partially allocated replacement, and all replacement buffers are
    // cleaned up on failure.
    Buffer vertices, heads, next;
    try {
        vertices = ctx_->createDeviceBuffer(VkDeviceSize(rounded) * kLightVertexBytes);
        heads = ctx_->createDeviceBuffer(VkDeviceSize(gridSize) * sizeof(uint32_t));
        next = ctx_->createDeviceBuffer(VkDeviceSize(rounded) * sizeof(uint32_t));
    } catch (...) {
        ctx_->destroy(vertices);
        ctx_->destroy(heads);
        ctx_->destroy(next);
        throw;
    }
    for (Binding b : {kLightVertices, kGridHead, kGridNext}) ctx_->destroy(buffers_[b]);
    buffers_[kLightVertices] = vertices;
    buffers_[kGridHead] = heads;
    buffers_[kGridNext] = next;
    cacheCapacity_ = uint32_t(rounded);
    gridSize_ = gridSize;
}

std::string GpuRenderer::backend() const {
    return "gpu: " + ctx_->info().name + " (" + ctx_->info().driver + (ctx_->info().portability ? ", MoltenVK" : "") +
           "; Slang kernels, float32, camera-relative, " + (rayQuery_ ? "hardware" : "software") + " traversal)";
}

uint64_t GpuRenderer::cacheEnvironment(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return 0;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(value, &end, 10);
    if (*value == '-' || *end || n == 0 || n > 0xFFFFFFF0ull)
        throw std::runtime_error(std::string(name) + " must be a positive vertex count <= 4294967280");
    return uint64_t(n);
}

void GpuRenderer::bindAll() {
    for (Kernel* k : {pathKernel_.get(), particleKernel_.get(), sppmCamera_.get(), sppmPhoton_.get(), sppmGrid_.get(),
                      sppmUpdate_.get(), sppmGuide_.get(), vcmLight_.get(), vcmCamera_.get(), vcmGrid_.get(), filmKernel_.get()})
        if (k) {
            for (uint32_t b = 0; b < kAllBindings; ++b) k->bind(b, buffers_[b]);
            const bool traces = k != sppmGrid_.get() && k != sppmUpdate_.get() && k != sppmGuide_.get() &&
                                k != vcmGrid_.get() && k != filmKernel_.get();
            if (hardware_ && traces) k->bindAccel(kSceneBinding, hardware_->tlas());
        }
}

} // namespace owe::gpu::detail

namespace owe::gpu {
std::unique_ptr<Renderer> makeRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings) {
    return std::make_unique<detail::GpuRenderer>(scene, detectorIndex, settings);
}
} // namespace owe::gpu
