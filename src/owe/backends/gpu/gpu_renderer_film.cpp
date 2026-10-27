#include "owe/backends/gpu/gpu_renderer_internal.hpp"

namespace owe::gpu::detail {

Image GpuRenderer::resolve() const {
    Image img;
    img.width = W_;
    img.height = H_;
    ctx_->submit([&](VkCommandBuffer cmd) {
        computeBarrier(cmd, true);
        copyBuffer(cmd, buffers_[kFilm], filmReadback_, filmReadback_.size);
        if (sppm_) copyBuffer(cmd, buffers_[kSppmState], sppmReadback_, sppmReadback_.size);
    }, "resolve film");
    img.xyz.resize(size_t(W_) * H_ * 3);
    const FilmPixel* film = static_cast<const FilmPixel*>(filmReadback_.mapped);
    const double norm = totalSpp_ > 0 ? 1.0 / double(totalSpp_) : 0;
    const double lightNorm = particleCount_ > 0 ? 1.0 / particleCount_ : 0;
    for (size_t i = 0; i < size_t(W_) * H_; ++i)
        for (int c = 0; c < 3; ++c)
            img.xyz[3 * i + c] = double(film[i].camera[c]) * norm + double(film[i].light[c]) * lightNorm;
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
    float f[9];  // position, origin, direction
    std::memcpy(f, n + 13, sizeof f);
    auto none = [](uint32_t v) { return v == kGpuNone ? kNone : v; };
    st.first = {n[9], n[10], n[11], none(n[12]), Vec3(f[0], f[1], f[2]) + gs_.origin, Vec3(f[3], f[4], f[5]) + gs_.origin,
                Vec3(f[6], f[7], f[8]), none(n[22]), none(n[23])};
}

void GpuRenderer::clearFilm() { zeroBuffers({kFilm}); }

void GpuRenderer::accumulateFilm(bool camera, bool light) {
    PushConstants pc = push(0);
    pc.flags = (camera ? 1u : 0u) | (light ? 2u : 0u);
    ctx_->submit([&](VkCommandBuffer cmd) {
        computeBarrier(cmd);
        filmKernel_->dispatch(cmd, &pc, groups(uint64_t(W_) * uint64_t(H_), 256), 1);
        computeBarrier(cmd, true);
        copyBuffer(cmd, buffers_[kCounters], counterReadback_, counterReadback_.size);
    }, "accumulate film and read diagnostics");
}

} // namespace owe::gpu::detail
