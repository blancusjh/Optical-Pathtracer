#include "owe/backends/gpu/gpu_backend.hpp"

#include <mutex>
#include <stdexcept>

#include "owe/backends/gpu/gpu_renderer.hpp"
#include "owe/backends/gpu/vulkan_context.hpp"

namespace owe {

namespace {

class GpuBackend final : public Backend {
public:
    std::string name() const override { return "gpu"; }
    std::string summary() const override {
        return "portable GPU: Vulkan compute with Slang kernels (NVIDIA, AMD, Intel; Apple silicon via MoltenVK); "
               "float32, camera-relative";
    }
    // Probing creates a Vulkan instance; the answer does not change while the process runs.
    bool available(std::string* why) const override {
        std::call_once(probed_, [this] { ok_ = gpu::available(&why_); });
        if (!ok_ && why) *why = why_;
        return ok_;
    }
    std::vector<DeviceInfo> devices() const override { return gpu::devices(); }
    std::vector<std::string> integrators() const override { return {"path", "light", "hybrid", "sppm", "bdpt", "vcm"}; }
    std::unique_ptr<Renderer> createRenderer(const Scene& scene, int detectorIndex,
                                             const RenderSettings& settings) const override {
        std::string why;
        if (!available(&why)) throw std::runtime_error("GPU backend unavailable: " + why);
        return gpu::makeRenderer(scene, detectorIndex, settings);
    }

private:
    mutable std::once_flag probed_;
    mutable bool ok_ = false;
    mutable std::string why_;
};

}  // namespace

const Backend& gpuBackend() {
    static const GpuBackend backend;
    return backend;
}

}  // namespace owe
