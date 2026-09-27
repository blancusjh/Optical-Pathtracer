// The GPU backend in a build configured without -DOWE_GPU=ON: listed, but unavailable.
#include <stdexcept>

#include "owe/backends/gpu/gpu_backend.hpp"

namespace owe {

namespace {

const char* kWhy = "this build has no GPU backend (configure with -DOWE_GPU=ON)";

class UnavailableGpuBackend final : public Backend {
public:
    std::string name() const override { return "gpu"; }
    std::string summary() const override { return "portable GPU (Vulkan compute, Slang kernels): not in this build"; }
    bool available(std::string* why) const override {
        if (why) *why = kWhy;
        return false;
    }
    std::vector<DeviceInfo> devices() const override { return {}; }
    std::vector<std::string> integrators() const override { return {"path", "light", "hybrid"}; }
    std::unique_ptr<Renderer> createRenderer(const Scene&, int, const RenderSettings&) const override {
        throw std::runtime_error(std::string("GPU backend unavailable: ") + kWhy);
    }
};

}  // namespace

const Backend& gpuBackend() {
    static const UnavailableGpuBackend backend;
    return backend;
}

}  // namespace owe
