#include "owe/backends/gpu/gpu_renderer_internal.hpp"

namespace owe::gpu::detail {

Image GpuRenderer::resolve() const {
    Image img;
    img.width = W_;
    img.height = H_;
    ctx_->submit([&](VkCommandBuffer cmd) {
        computeBarrier(cmd, true);
        VkBufferCopy copy{0, 0, filmReadback_.size};
        vkCmdCopyBuffer(cmd, buffers_[kFilm].buffer, filmReadback_.buffer, 1, &copy);
        if (sppm_) {
            VkBufferCopy state{0, 0, sppmReadback_.size};
            vkCmdCopyBuffer(cmd, buffers_[kSppmState].buffer, sppmReadback_.buffer, 1, &state);
        }
    }, "resolve film");
    img.xyz.resize(size_t(W_) * H_ * 3);
    const float* film = static_cast<const float*>(filmReadback_.mapped);
    const double norm = totalSpp_ > 0 ? 1.0 / double(totalSpp_) : 0;
    const double lightNorm = particleCount_ > 0 ? 1.0 / particleCount_ : 0;
    for (size_t i = 0; i < size_t(W_) * H_; ++i)
        for (int c = 0; c < 3; ++c)
            img.xyz[3 * i + c] = double(film[16 * i + c]) * norm + double(film[16 * i + 8 + c]) * lightNorm;
    if (sppm_ && iterations_ > 0) {
        // L = L_direct / iterations (above) + τ / (N_emitted · π r²).
        const double emitted = double(photonsPerIteration_) * double(iterations_);
        for (size_t i = 0; i < size_t(W_) * H_; ++i) {
            const float* st = static_cast<const float*>(sppmReadback_.mapped) + 8 * i;
            const double r = st[4];
            if (!(r > 0)) continue;
            const double k = 1 / (emitted * Pi * r * r);
            for (int c = 0; c < 3; ++c) img.xyz[3 * i + c] += double(st[c]) * k;
        }
    }
    return img;
}

void GpuRenderer::collectStats() {
    const uint32_t* n = static_cast<const uint32_t*>(counterReadback_.mapped);
    TransportStats st;
    st.paths = word64(n);
    st.segments = word64(n + 2);
    st.inconsistencies = word64(n + 4);
    st.leaks = word64(n + 6);
    readFirstInconsistency(st);
    stats_.add(st);
}

void GpuRenderer::readFirstInconsistency(TransportStats& st) const {
    const uint32_t* n = static_cast<const uint32_t*>(counterReadback_.mapped);
    if (n[8] == 0 || st.inconsistencies == 0) return;
    float p[3];
    std::memcpy(p, n + 13, sizeof p);
    st.first = {n[9], n[10], n[11], n[12] == kGpuNone ? kNone : n[12],
                Vec3(p[0], p[1], p[2]) + gs_.origin};
}

void GpuRenderer::clearFilm() {
    ctx_->submit([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, buffers_[kFilm].buffer, 0, VK_WHOLE_SIZE, 0);
        transferToCompute(cmd);
    });
}

void GpuRenderer::accumulateFilm(bool camera, bool light) {
    PushConstants pc = push(0);
    pc.flags = (camera ? 1u : 0u) | (light ? 2u : 0u);
    ctx_->submit([&](VkCommandBuffer cmd) {
        computeBarrier(cmd);
        filmKernel_->dispatch(cmd, &pc, (uint32_t(W_) * uint32_t(H_) + 255) / 256, 1);
        computeBarrier(cmd, true);
        VkBufferCopy counters{0, 0, counterReadback_.size};
        vkCmdCopyBuffer(cmd, buffers_[kCounters].buffer, counterReadback_.buffer, 1, &counters);
    }, "accumulate film and read diagnostics");
}

} // namespace owe::gpu::detail
