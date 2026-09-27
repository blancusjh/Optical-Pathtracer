#include "owe/backends/registry.hpp"

#include <stdexcept>

#include "owe/backends/cpu/cpu_backend.hpp"
#include "owe/backends/gpu/gpu_backend.hpp"

namespace owe {

const std::vector<const Backend*>& backends() {
    static const std::vector<const Backend*> list = {&cpuBackend(), &gpuBackend()};
    return list;
}

const Backend* findBackend(const std::string& name) {
    for (const Backend* b : backends())
        if (b->name() == name) return b;
    return nullptr;
}

const Backend& backend(const std::string& name) {
    if (const Backend* b = findBackend(name)) return *b;
    std::string known;
    for (const Backend* b : backends()) known += (known.empty() ? "" : ", ") + b->name();
    throw std::runtime_error("unknown backend '" + name + "' (known: " + known + ")");
}

const Backend& referenceBackend() { return cpuBackend(); }

std::string adaptToBackend(RenderSettings& settings) {
    const Backend& b = backend(settings.backend);
    if (b.supports(settings.integrator)) return "";
    std::string note = "the " + b.name() + " backend implements the path integrator, not '" + settings.integrator +
                       "'; using path (same expected image, noisier directly seen caustics)";
    settings.integrator = "path";
    return note;
}

std::unique_ptr<Renderer> makeRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings) {
    return backend(settings.backend).createRenderer(scene, detectorIndex, settings);
}

}  // namespace owe
