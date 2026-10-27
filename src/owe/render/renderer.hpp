// The one rendering interface. A renderer is a progressive, unbiased estimator of one detector's
// image: each pass adds samples, and the raw accumulation is never altered by display processing.
// Renderers are created by a backend (owe/render/backend.hpp); front ends (the command line, the
// studio, the viewer, the tests) see only this interface, so changing where the transport runs is
// changing RenderSettings::backend and nothing else.
#pragma once

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "owe/scene/detector.hpp"
#include "owe/scene/scene.hpp"
#include "owe/scene/transport_stats.hpp"

namespace owe {

class Renderer {
public:
    // A renderer of the scene's detector `detectorIndex` that knows the integrators `known`.
    Renderer(const Scene& scene, int detectorIndex, const RenderSettings& settings, const std::vector<std::string>& known)
        : scene_(scene), settings_(settings) {
        if (detectorIndex < 0 || detectorIndex >= int(scene.detectors.size())) throw std::runtime_error("no such detector");
        det_ = scene.detectors[size_t(detectorIndex)].get();
        const std::string& I = settings.integrator;
        bool ok = false;
        std::string list;
        for (size_t i = 0; i < known.size(); ++i) {
            ok = ok || known[i] == I;
            list += (i == 0 ? "" : i + 1 == known.size() ? " or " : ", ") + known[i];
        }
        if (!ok) throw std::runtime_error("unknown integrator '" + I + "' (expected " + list + ")");
        if (I == "hybrid" && !det_->isVirtual())
            throw std::runtime_error("the hybrid integrator needs a virtual observer; use path or light for surface sensors");
    }
    virtual ~Renderer() = default;
    // Adds a pass of samplesPerPixel. path: spp; light: particles per pixel; hybrid: both, with a
    // disjoint path-space partition.
    void runPass(int samplesPerPixel) {
        if (samplesPerPixel <= 0) return;
        const auto start = std::chrono::steady_clock::now();
        pass(samplesPerPixel);
        passes_++;
        totalSpp_ += samplesPerPixel;
        seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    virtual Image resolve() const = 0;
    int passes() const { return passes_; }
    long long samplesPerPixel() const { return totalSpp_; }
    double seconds() const { return seconds_; }
    const TransportStats& stats() const { return stats_; }
    const Detector& detector() const { return *det_; }
    const RenderSettings& settings() const { return settings_; }
    // Human-readable identity of the backend for records and logs, e.g.
    // "cpu-reference (IEEE-754 double)" or "gpu: <device> (...; float32, camera-relative)".
    virtual std::string backend() const = 0;
    // The same scene and observer after a pose/resolution change. Discards accumulated samples.
    // Call only between passes, after preparing the observer. False means recreate the renderer.
    virtual bool resetObserver() { return false; }

protected:
    virtual void pass(int samplesPerPixel) = 0;
    // Forgets the passes, samples, time and statistics (a reset observer).
    void resetProgress() {
        passes_ = 0;
        totalSpp_ = 0;
        seconds_ = 0;
        stats_ = {};
    }

    const Scene& scene_;
    const Detector* det_ = nullptr;
    RenderSettings settings_;
    TransportStats stats_;
    int passes_ = 0;
    long long totalSpp_ = 0;
    double seconds_ = 0;
};

}  // namespace owe
