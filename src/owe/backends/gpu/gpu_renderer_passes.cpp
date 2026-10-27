#include "owe/backends/gpu/gpu_renderer_internal.hpp"

namespace owe::gpu::detail {

void GpuRenderer::zeroBuffers(std::initializer_list<Binding> bindings) {
    ctx_->submit([&](VkCommandBuffer cmd) {
        for (Binding b : bindings) vkCmdFillBuffer(cmd, buffers_[b].buffer, 0, VK_WHOLE_SIZE, 0);
        transferToCompute(cmd);
    });
}

// The first submission is timed and sets the chunk so a dispatch takes about 50 ms; the rest are
// queued, two dispatches per submission.
void GpuRenderer::dispatchChunked(Kernel& kernel, PushConstants pc, uint64_t total, bool perPixel) {
    uint32_t& chunk = perPixel ? chunk_ : particleChunk_;
    uint64_t done = 0;
    while (done < total) {
        const auto start = std::chrono::steady_clock::now();
        uint64_t batch = 0;
        auto record = [&](VkCommandBuffer cmd) {
            for (int k = 0; k < 2 && done + batch < total; ++k) {
                const uint64_t base = done + batch;
                pc.sampleBase = uint32_t(base);
                pc.baseHi = uint32_t(base >> 32);
                pc.sampleCount = uint32_t(std::min<uint64_t>(chunk, total - base));
                if (perPixel) kernel.dispatch(cmd, &pc, groups(W_, kPathGroup), groups(H_, kPathGroup));
                else kernel.dispatch(cmd, &pc, groups(pc.sampleCount, kParticleGroup), 1);
                computeBarrier(cmd);
                batch += pc.sampleCount;
            }
        };
        const bool calibrate = done == 0;
        if (calibrate) ctx_->submit(record, "calibrate dispatch");
        else ctx_->submitAsync(record, "queued dispatch");
        done += batch;
        if (calibrate) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            const double perItem = ms / double(std::max<uint64_t>(1, batch));
            chunk = perPixel ? uint32_t(std::clamp(50.0 / std::max(perItem, 1e-3), 1.0, 4096.0))
                             : uint32_t(std::clamp(50.0 / std::max(perItem, 1e-9), double(kParticleGroup),
                                                   double(kMaxGroups) * kParticleGroup));
        }
    }
}

uint64_t GpuRenderer::photonsPerIteration() const {
    return std::max<uint64_t>(1, uint64_t(std::llround(double(W_) * H_ * settings_.photonsPerPixel)));
}

// Hero wavelengths rotate over iterations by the golden ratio (a low-discrepancy sequence), offset
// by the seed; every path and photon of one iteration uses them.
float GpuRenderer::wavelengthRotation(uint64_t iteration) const {
    const double offset = double(uint32_t(settings_.seed * 0x9E3779B97F4A7C15ull >> 32)) / 4294967296.0;
    const double u = offset + 0.6180339887498949 * double(iteration + 1);
    return float(u - std::floor(u));
}

uint64_t GpuRenderer::vcmLightPaths() const {
    const uint64_t G = vcmGroups();
    return (photonsPerIteration() + G - 1) / G * G;
}

void GpuRenderer::validateVcmCounts(uint64_t lightPaths) const {
    const uint64_t verticesPerPath = uint64_t(std::max(settings_.maxDepth, 0)) + 1;
    if (lightPaths > 0xFFFFFFFFull / verticesPerPath)
        throw std::runtime_error("vcm: light path count and depth exceed the "
                                 "32-bit vertex counter range");
}

void GpuRenderer::sizeLightCache() {
    if (!(gs_.emittedPower > 0)) {
        vcmSizePending_ = false;
        return;
    }
    const uint64_t lightPaths = vcmLightPaths();
    validateVcmCounts(lightPaths);
    PushConstants pc = push(1);
    pc.flags |= (1u << 16) | (vcmGroups() << 24);
    pc.lightPathsPerGroup = uint32_t(lightPaths / vcmGroups());
    zeroBuffers({kVcmCounters});
    dispatchChunked(*vcmLight_, pc, lightPaths, false);
    ctx_->submit([&](VkCommandBuffer cmd) {
        computeBarrier(cmd, true);
        copyBuffer(cmd, buffers_[kVcmCounters], vcmCounterReadback_, 16 * sizeof(uint32_t));
    });
    const uint32_t* n = static_cast<const uint32_t*>(vcmCounterReadback_.mapped);
    const double perHit = n[2] > 0 ? double(n[0]) / double(n[2]) : 1.0;
    vertexBudget_ = std::max(1.0, 1.3 * perHit);
    allocateLightCache(uint64_t(double(lightPaths) * vertexBudget_) + 1024);
    bindAll();
    vcmSizePending_ = false;
}

uint32_t GpuRenderer::vcmGroups() const { return uint32_t(std::clamp(settings_.vcmWavelengthGroups, 1, 8)); }

// The guide may have been filled or restored by transfer, or learned by compute: both sources
// must be visible to the snapshot/restore copy.
static void guideToTransfer(VkCommandBuffer cmd) {
    memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

void GpuRenderer::snapshotVcmGuide() {
    ctx_->submit(
        [&](VkCommandBuffer cmd) {
            guideToTransfer(cmd);
            copyBuffer(cmd, buffers_[kGuide], vcmGuideBackup_, buffers_[kGuide].size);
        },
        "vcm guide snapshot");
}

void GpuRenderer::restoreVcmGuide() {
    ctx_->submit(
        [&](VkCommandBuffer cmd) {
            guideToTransfer(cmd);
            copyBuffer(cmd, vcmGuideBackup_, buffers_[kGuide], buffers_[kGuide].size);
            transferToCompute(cmd);
        },
        "vcm guide restore");
}

PushConstants GpuRenderer::vcmPush(uint32_t iterations, uint64_t lightPaths) {
    PushConstants pc = push(iterations);
    pc.flags |= (settings_.diagnosticVcmMask & 0xFFu) << 8;
    pc.flags |= vcmGroups() << 24;
    pc.lightPathsPerGroup = uint32_t(lightPaths / vcmGroups());
    pc.gridMask = gridSize_ - 1;
    pc.cacheCapacity = cacheCapacity_ / vcmGroups();
    return pc;
}

// The pass's iterations: per iteration, the light subpaths fill the cache, the cached vertices
// enter the merging grid, then the camera paths. Returns 0, or the cache capacity needed when it
// overflowed (the attempt is then abandoned).
uint64_t GpuRenderer::runVcmIterations(PushConstants& pc, uint32_t iterations, uint64_t lightPaths) {
    // The merging radius at a point: `photonRadius` pixel footprints at its distance from the
    // eye, shrinking over iterations as r_i = r_1 i^((α−1)/2) (Georgiev et al.).
    const double pixelAngle = dynamic_cast<const IdealObserver*>(det_)->pixelAngle();
    for (uint32_t it = 0; it < iterations; ++it) {
        const uint64_t iteration = uint64_t(iterations_) + it;
        pc.pass = uint32_t(iteration);
        pc.sampleBase = it;
        pc.wavelengthRotation = wavelengthRotation(iteration);
        const double shrink = std::pow(double(iteration + 1), (settings_.photonAlpha - 1) / 2);
        pc.mergeRadiusPerMetre = merging_ ? float(settings_.photonRadius * pixelAngle * shrink) : 0.0f;
        pc.mergeRadiusFloor = 1e-9f;
        // The iteration's start: the cache emptied, the hash cleared, the guide's CDF from what it
        // has learnt. No persistent host state advances until the pass succeeds.
        auto begin = [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kVcmCounters].buffer, 0, 4 * sizeof(uint32_t), 0);
            vkCmdFillBuffer(cmd, buffers_[kVcmCounters].buffer, 16, 12 * sizeof(uint32_t), 0);
            if (merging_) vkCmdFillBuffer(cmd, buffers_[kGridHead].buffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
            transferToCompute(cmd);
            sppmGuide_->dispatch(cmd, &pc, 1, 1);
            computeBarrier(cmd);
        };
        auto finish = [&](VkCommandBuffer cmd) {
            if (merging_) {
                vcmGrid_->dispatch(cmd, &pc, groups(cacheCapacity_, kParticleGroup), 1);
                computeBarrier(cmd);
            }
            vcmCamera_->dispatch(cmd, &pc, groups(W_, kPathGroup), groups(H_, kPathGroup));
            computeBarrier(cmd, true);
            copyBuffer(cmd, buffers_[kVcmCounters], vcmCounterReadback_, vcmCounterReadback_.size);
        };
        const bool emits = gs_.emittedPower > 0;
        if (!emits || lightPaths <= particleChunk_) {
            // One submission: small images spend more on submissions than on paths.
            ctx_->submit(
                [&](VkCommandBuffer cmd) {
                    begin(cmd);
                    if (emits) {
                        PushConstants lp = pc;
                        lp.sampleBase = lp.baseHi = 0;
                        lp.sampleCount = uint32_t(lightPaths);
                        vcmLight_->dispatch(cmd, &lp, groups(lightPaths, kParticleGroup), 1);
                        computeBarrier(cmd);
                    }
                    finish(cmd);
                },
                "vcm iteration");
        } else {
            ctx_->submit(begin, "vcm begin");
            dispatchChunked(*vcmLight_, pc, lightPaths, false);
            ctx_->submit(finish, "vcm grid/camera/counters");
        }
        const uint32_t* offered = static_cast<const uint32_t*>(vcmCounterReadback_.mapped);
        if (offered[1] > 0) {
            uint32_t largestGroup = 0;
            for (uint32_t g = 0; g < vcmGroups(); ++g) largestGroup = std::max(largestGroup, offered[4 + g]);
            return uint64_t(largestGroup) * vcmGroups();
        }
    }
    return 0;
}

// A pass is a transaction: a rejected attempt (the light vertex cache overflowed) must not teach
// the guide, advance the radius/wavelength/RNG schedule, or contribute to the accepted film or its
// statistics. It is replayed with a larger cache.
void GpuRenderer::passVcm(uint32_t iterations) {
    if (vcmSizePending_) sizeLightCache();
    const uint64_t lightPaths = vcmLightPaths();
    validateVcmCounts(lightPaths);
    snapshotVcmGuide();
    try {
        for (uint32_t attempt = 0;; ++attempt) {
            zeroBuffers({kCounters, kSplat, kVcmCounters});
            PushConstants pc = vcmPush(iterations, lightPaths);
            const uint64_t required = runVcmIterations(pc, iterations, lightPaths);
            if (required == 0) break;
            if (attempt >= 7)
                throw std::runtime_error("vcm: light vertex cache still overflowing "
                                         "after eight attempts; pass rejected");
            const uint64_t grown = std::max<uint64_t>(required, uint64_t(cacheCapacity_) + (uint64_t(cacheCapacity_) + 1) / 2);
            std::fprintf(stderr,
                         "vcm: light vertex cache (%u vertices) overflowed; "
                         "replaying pass with %llu vertices\n",
                         cacheCapacity_, static_cast<unsigned long long>(grown));
            allocateLightCache(grown);
            bindAll();
            restoreVcmGuide();
        }
        accumulateFilm(true, true);
    } catch (...) {
        restoreVcmGuide();
        throw;
    }
    particleCount_ += double(lightPaths) * iterations;
    iterations_ += iterations;
    collectStats();
}

void GpuRenderer::resetVcm() {
    iterations_ = 0;
    zeroBuffers({kGuide});
}

void GpuRenderer::resetSppm() {
    iterations_ = 0;
    photonsPerIteration_ = photonsPerIteration();
    double pixelAngle = 0, radiusFloor = 0;
    if (auto* eye = dynamic_cast<const IdealObserver*>(det_)) {
        pixelAngle = eye->pixelAngle();
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
    zeroBuffers({kSppmState, kFlux, kGuide});
}

void GpuRenderer::passSppm(uint32_t iterations) {
    const uint32_t pixelGroups = groups(uint64_t(W_) * uint64_t(H_), kParticleGroup);
    zeroBuffers({kCounters});
    PushConstants pc = push(iterations);
    pc.gridMask = gridSize_ - 1;
    pc.photonAlpha = float(settings_.photonAlpha);
    pc.initialRadiusPerMetre = r0PerMetre_;
    pc.initialRadiusFloor = r0Floor_;
    pc.wavelengthGroups = uint32_t(std::clamp(settings_.photonWavelengthGroups, 1, 255));
    for (uint32_t it = 0; it < iterations; ++it) {
        pc.pass = uint32_t(iterations_);
        pc.sampleBase = it;
        pc.wavelengthRotation = wavelengthRotation(uint64_t(iterations_));
        ctx_->submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, buffers_[kGridHead].buffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
            vkCmdFillBuffer(cmd, buffers_[kLevels].buffer, 0, 4, 0x7FFFFFFFu);
            vkCmdFillBuffer(cmd, buffers_[kLevels].buffer, 4, 4, 0x80000001u);
            transferToCompute(cmd);
            sppmCamera_->dispatch(cmd, &pc, groups(W_, kPathGroup), groups(H_, kPathGroup));
            computeBarrier(cmd);
            sppmGrid_->dispatch(cmd, &pc, pixelGroups, 1);
            sppmGuide_->dispatch(cmd, &pc, 1, 1);  // the photon guide learnt so far
            computeBarrier(cmd);
        });
        if (gs_.emittedPower > 0) dispatchChunked(*sppmPhoton_, pc, photonsPerIteration_, false);
        ctx_->submit([&](VkCommandBuffer cmd) {
            sppmUpdate_->dispatch(cmd, &pc, pixelGroups, 1);
            computeBarrier(cmd);
        });
        ++iterations_;
    }
    accumulateFilm(true, false);
    collectStats();
}

void GpuRenderer::passPath(uint32_t spp, uint32_t extraFlags) {
    zeroBuffers({kCounters});
    PushConstants pc = push(spp);
    pc.flags |= extraFlags;
    dispatchChunked(*pathKernel_, pc, spp, true);
    accumulateFilm(true, false);
    collectStats();
}

void GpuRenderer::passParticles(uint32_t ppp) {
    const uint64_t total = uint64_t(W_) * uint64_t(H_) * ppp;
    particleCount_ += double(total);
    if (!(gs_.emittedPower > 0)) return;  // nothing emits: the particle estimate is zero
    zeroBuffers({kSplat, kCounters});
    dispatchChunked(*particleKernel_, push(ppp), total, false);
    accumulateFilm(false, true);
    collectStats();
}

PushConstants GpuRenderer::push(uint32_t spp) const {
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

} // namespace owe::gpu::detail
