// A backend is where light transport runs. Every backend estimates the same measurement — the
// transport defined by the reference engine (owe/transport) on the one world model (owe/scene) —
// so choosing one is a choice of hardware and precision, never of physics:
//
//   cpu  the reference: IEEE-754 double on host threads, bitwise reproducible (owe/backends/cpu)
//   gpu  portable Vulkan compute with Slang kernels, float32 camera-relative (owe/backends/gpu)
//
// Backends are independent of each other: each depends only on the scene model and this
// interface (enforced by the `layering` test), and each is validated against the reference
// statistically (owe/backends/compare.hpp) and by one conformance suite (tests/test_backends.cpp).
// The registry (owe/backends/registry.hpp) is the only code that names them.
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "owe/render/renderer.hpp"

namespace owe {

struct DeviceInfo {
    int index = 0;
    std::string name;
    std::string type;    // "cpu", "discrete", "integrated", "virtual", "other"
    std::string driver;  // e.g. thread count, or the Vulkan API version
    bool portability = false;  // a layered implementation (MoltenVK)
};

class Backend {
public:
    virtual ~Backend() = default;
    // The value of RenderSettings::backend and of --backend: "cpu", "gpu".
    virtual std::string name() const = 0;
    // One line: hardware, precision and role.
    virtual std::string summary() const = 0;
    // True when this build contains the backend and it can run here; `why` explains otherwise.
    virtual bool available(std::string* why = nullptr) const = 0;
    // Devices it can run on (RenderSettings::device selects one where that applies).
    virtual std::vector<DeviceInfo> devices() const = 0;
    // Integrators it implements ("path" always; "light", "hybrid").
    virtual std::vector<std::string> integrators() const = 0;
    // Throws if the backend is unavailable or the settings ask for something it does not implement.
    virtual std::unique_ptr<Renderer> createRenderer(const Scene& scene, int detectorIndex,
                                                     const RenderSettings& settings) const = 0;

    bool supports(const std::string& integrator) const {
        auto list = integrators();
        return std::find(list.begin(), list.end(), integrator) != list.end();
    }
};

}  // namespace owe
