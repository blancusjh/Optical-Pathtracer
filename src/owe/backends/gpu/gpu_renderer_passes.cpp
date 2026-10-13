#include "owe/backends/gpu/gpu_renderer_internal.hpp"

namespace owe::gpu::detail {

void GpuRenderer::dispatchLightPaths(const PushConstants& pc, uint64_t lightPaths) {
    PushConstants lp = pc;
    uint64_t done = 0;
    while (done < lightPaths) {
        auto s0 = std::chrono::steady_clock::now();
        uint64_t batch = 0;
        auto record = [&](VkCommandBuffer cmd) {
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
        };
        const bool calibrate = done == 0;
        if (calibrate) ctx_->submit(record, "calibrate dispatch");
        else ctx_->submitAsync(record, "queued dispatch");
        done += batch;
        if (calibrate) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            double perPath = ms / double(std::max<uint64_t>(1, batch));
            particleChunk_ = uint32_t(std::clamp(50.0 / std::max(perPath, 1e-9), double(kParticleGroup),
                                                 double(kMaxGroups) * kParticleGroup));
        }
    }
}

uint64_t GpuRenderer::vcmLightPaths() const {
    const uint64_t G = vcmGroups();
    const uint64_t n = std::max<uint64_t>(1, uint64_t(std::llround(double(W_) * H_ * settings_.photonsPerPixel)));
    return (n + G - 1) / G * G;
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
    vcmSizePending_ = false;
}

uint32_t GpuRenderer::vcmGroups() const { return uint32_t(std::clamp(settings_.vcmWavelengthGroups, 1, 8)); }

void GpuRenderer::passVcm(uint32_t iterations) {
    if (vcmSizePending_) sizeLightCache();
    const uint32_t gx = (uint32_t(W_) + kPathGroup - 1) / kPathGroup, gy = (uint32_t(H_) + kPathGroup - 1) / kPathGroup;
    const uint64_t lightPaths = vcmLightPaths();
    validateVcmCounts(lightPaths);
    // A pass is a transaction. A rejected attempt must not teach the guide,
    // advance the radius/wavelength/RNG schedule, or contribute to either
    // accepted film or its statistics.
    auto guideToTransfer = [](VkCommandBuffer cmd) {
        // The guide may have been filled/restored by transfer or learned by compute.
        // Both sources must be visible to the snapshot/restore copy.
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    ctx_->submit(
        [&](VkCommandBuffer cmd) {
            guideToTransfer(cmd);
            VkBufferCopy guide{0, 0, buffers_[kGuide].size};
            vkCmdCopyBuffer(cmd, buffers_[kGuide].buffer, vcmGuideBackup_.buffer, 1, &guide);
        },
        "vcm guide snapshot");
    auto restoreGuide = [&] {
        ctx_->submit(
            [&](VkCommandBuffer cmd) {
                guideToTransfer(cmd);
                VkBufferCopy guide{0, 0, buffers_[kGuide].size};
                vkCmdCopyBuffer(cmd, vcmGuideBackup_.buffer, buffers_[kGuide].buffer, 1, &guide);
                transferToCompute(cmd);
            },
            "vcm guide restore");
    };
    // The merging radius at a point: `photonRadius` pixel footprints at its
    // distance from the eye, shrinking over iterations as r_i = r_1 i^((α−1)/2)
    // (Georgiev et al.).
    const auto* eye = dynamic_cast<const IdealObserver*>(det_);
    const double pixelAngle = 2 * eye->tanY() / eye->height;
    PushConstants pc = push(iterations);
    pc.flags |= (settings_.diagnosticVcmMask & 0xFFu) << 8;
    pc.flags |= vcmGroups() << 24;
    pc.lightPathsPerGroup = uint32_t(lightPaths / vcmGroups());
    const double offset = double(uint32_t(settings_.seed * 0x9E3779B97F4A7C15ull >> 32)) / 4294967296.0;
    try {
        for (uint32_t attempt = 0;; ++attempt) {
            ctx_->submit([&](VkCommandBuffer cmd) {
                vkCmdFillBuffer(cmd, buffers_[kCounters].buffer, 0, VK_WHOLE_SIZE, 0);
                vkCmdFillBuffer(cmd, buffers_[kSplat].buffer, 0, VK_WHOLE_SIZE, 0);
                vkCmdFillBuffer(cmd, buffers_[kVcmCounters].buffer, 0, VK_WHOLE_SIZE, 0);
                transferToCompute(cmd);
            });
            pc.gridMask = gridSize_ - 1;
            pc.cacheCapacity = cacheCapacity_ / vcmGroups();
            uint64_t requiredCapacity = 0;
            for (uint32_t it = 0; it < iterations; ++it) {
                const uint64_t iteration = uint64_t(iterations_) + it;
                pc.pass = uint32_t(iteration);
                pc.sampleBase = it;
                double u = offset + 0.6180339887498949 * double(iteration + 1);
                pc.wavelengthRotation = float(u - std::floor(u));
                const double shrink = std::pow(double(iteration + 1), (settings_.photonAlpha - 1) / 2);
                pc.mergeRadiusPerMetre = merging_ ? float(settings_.photonRadius * pixelAngle * shrink) : 0.0f;
                pc.mergeRadiusFloor = 1e-9f;
                // The iteration's start: the cache emptied, the hash cleared, the
                // guide's CDF from what it has learnt. No persistent host state
                // advances until the pass succeeds.
                auto begin = [&](VkCommandBuffer cmd) {
                    vkCmdFillBuffer(cmd, buffers_[kVcmCounters].buffer, 0, 4 * sizeof(uint32_t), 0);
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
                    computeBarrier(cmd, true);
                    VkBufferCopy cache{0, 0, vcmCounterReadback_.size};
                    vkCmdCopyBuffer(cmd, buffers_[kVcmCounters].buffer, vcmCounterReadback_.buffer, 1, &cache);
                };
                const bool emits = gs_.emittedPower > 0;
                if (!emits || lightPaths <= particleChunk_) {
                    // One submission: small images spend more on submissions than on
                    // paths.
                    ctx_->submit(
                        [&](VkCommandBuffer cmd) {
                            begin(cmd);
                            if (emits) light(cmd, 0, uint32_t(lightPaths));
                            finish(cmd);
                        },
                        "vcm iteration");
                } else {
                    ctx_->submit(begin, "vcm begin");
                    dispatchLightPaths(pc, lightPaths);
                    ctx_->submit(finish, "vcm grid/camera/counters");
                }
                const uint32_t* offered = static_cast<const uint32_t*>(vcmCounterReadback_.mapped);
                if (offered[1] > 0) {
                    uint32_t largestGroup = 0;
                    for (uint32_t g = 0; g < vcmGroups(); ++g) largestGroup = std::max(largestGroup, offered[4 + g]);
                    requiredCapacity = uint64_t(largestGroup) * vcmGroups();
                    break;
                }
            }
            if (requiredCapacity == 0) break;
            if (attempt >= 7)
                throw std::runtime_error("vcm: light vertex cache still overflowing "
                                         "after eight attempts; pass rejected");
            const uint64_t grown = std::max<uint64_t>(requiredCapacity, uint64_t(cacheCapacity_) + (uint64_t(cacheCapacity_) + 1) / 2);
            std::fprintf(stderr,
                         "vcm: light vertex cache (%u vertices) overflowed; "
                         "replaying pass with %llu vertices\n",
                         cacheCapacity_, static_cast<unsigned long long>(grown));
            allocateLightCache(grown);
            bindAll();
            restoreGuide();
        }
        accumulateFilm(true, true);
    } catch (...) {
        restoreGuide();
        throw;
    }
    particleCount_ += double(lightPaths) * iterations;
    iterations_ += iterations;
    collectStats();
}

void GpuRenderer::resetVcm() {
    iterations_ = 0;
    ctx_->submit([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, buffers_[kGuide].buffer, 0, VK_WHOLE_SIZE, 0);
        transferToCompute(cmd);
    });
}

void GpuRenderer::resetSppm() {
    iterations_ = 0;
    photonsPerIteration_ = std::max<uint64_t>(1, uint64_t(std::llround(double(W_) * H_ * settings_.photonsPerPixel)));
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

void GpuRenderer::passSppm(uint32_t iterations) {
    const uint32_t gx = (uint32_t(W_) + kPathGroup - 1) / kPathGroup, gy = (uint32_t(H_) + kPathGroup - 1) / kPathGroup;
    const uint32_t pixelGroups = (uint32_t(W_) * uint32_t(H_) + kParticleGroup - 1) / kParticleGroup;
    ctx_->submit([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, buffers_[kCounters].buffer, 0, VK_WHOLE_SIZE, 0);
        transferToCompute(cmd);
    });
    PushConstants pc = push(iterations);
    pc.gridMask = gridSize_ - 1;
    pc.photonAlpha = float(settings_.photonAlpha);
    pc.initialRadiusPerMetre = r0PerMetre_;
    pc.initialRadiusFloor = r0Floor_;
    pc.wavelengthGroups = uint32_t(std::clamp(settings_.photonWavelengthGroups, 1, 255));
    // Hero wavelengths rotate over iterations by the golden ratio (a low-discrepancy sequence),
    // offset by the seed; every path and photon of one iteration uses them.
    const double offset = double(uint32_t(settings_.seed * 0x9E3779B97F4A7C15ull >> 32)) / 4294967296.0;
    for (uint32_t it = 0; it < iterations; ++it) {
        pc.pass = uint32_t(iterations_);
        pc.sampleBase = it;
        double u = offset + 0.6180339887498949 * double(iterations_ + 1);
        pc.wavelengthRotation = float(u - std::floor(u));
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
                auto record = [&](VkCommandBuffer cmd) {
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
                };
                const bool calibrate = done == 0;
                if (calibrate) ctx_->submit(record, "calibrate photons");
                else ctx_->submitAsync(record, "queued photons");
                done += batch;
                if (calibrate) {
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
                    double perPhoton = ms / double(std::max<uint64_t>(1, batch));
                    particleChunk_ = uint32_t(std::clamp(50.0 / std::max(perPhoton, 1e-9), double(kParticleGroup),
                                                     double(kMaxGroups) * kParticleGroup));
                }
            }
        }
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
    ctx_->submit([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, buffers_[kCounters].buffer, 0, VK_WHOLE_SIZE, 0);
        transferToCompute(cmd);
    });
    PushConstants pc = push(spp);
    pc.flags |= extraFlags;
    const uint32_t gx = (uint32_t(W_) + kPathGroup - 1) / kPathGroup, gy = (uint32_t(H_) + kPathGroup - 1) / kPathGroup;
    // Samples per dispatch adapt to keep each submission near 50 ms; dispatches are batched
    // into submissions of ~100 ms.
    uint32_t done = 0;
    while (done < spp) {
        auto s0 = std::chrono::steady_clock::now();
        uint32_t batchSamples = 0;
        auto record = [&](VkCommandBuffer cmd) {
            for (int k = 0; k < 2 && done + batchSamples < spp; ++k) {
                pc.sampleBase = done + batchSamples;
                pc.sampleCount = std::min(chunk_, spp - pc.sampleBase);
                pathKernel_->dispatch(cmd, &pc, gx, gy);
                computeBarrier(cmd);
                batchSamples += pc.sampleCount;
            }
        };
        const bool calibrate = done == 0;
        if (calibrate) ctx_->submit(record, "calibrate dispatch");
        else ctx_->submitAsync(record, "queued dispatch");
        done += batchSamples;
        if (calibrate) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            double perSample = ms / std::max(1u, batchSamples);
            chunk_ = uint32_t(std::clamp(50.0 / std::max(perSample, 1e-3), 1.0, 4096.0));
        }
    }
    accumulateFilm(true, false);
    collectStats();
}

void GpuRenderer::passParticles(uint32_t ppp) {
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
        auto record = [&](VkCommandBuffer cmd) {
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
        };
        const bool calibrate = done == 0;
        if (calibrate) ctx_->submit(record, "calibrate dispatch");
        else ctx_->submitAsync(record, "queued dispatch");
        done += batch;
        if (calibrate) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
            double perParticle = ms / double(std::max<uint64_t>(1, batch));
            particleChunk_ = uint32_t(std::clamp(50.0 / std::max(perParticle, 1e-9), double(kParticleGroup),
                                                 double(kMaxGroups) * kParticleGroup));
        }
    }
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
