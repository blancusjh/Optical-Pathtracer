// The one rendering interface. A renderer is a progressive, unbiased estimator of one detector's
// image: each pass adds samples, and the raw accumulation is never altered by display processing.
// Renderers are created by a backend (owe/render/backend.hpp); front ends (the command line, the
// studio, the viewer, the tests) see only this interface, so changing where the transport runs is
// changing RenderSettings::backend and nothing else.
#pragma once

#include <string>

#include "owe/scene/detector.hpp"
#include "owe/scene/scene.hpp"
#include "owe/scene/transport_stats.hpp"

namespace owe {

class Renderer {
public:
    virtual ~Renderer() = default;
    // path: spp; light: particles per pixel; hybrid: both, with a disjoint path-space partition.
    virtual void runPass(int samplesPerPixel) = 0;
    virtual Image resolve() const = 0;
    virtual int passes() const = 0;
    virtual long long samplesPerPixel() const = 0;
    virtual double seconds() const = 0;
    virtual const TransportStats& stats() const = 0;
    virtual const Detector& detector() const = 0;
    virtual const RenderSettings& settings() const = 0;
    // Human-readable identity of the backend for records and logs, e.g.
    // "cpu-reference (IEEE-754 double)" or "gpu: <device> (...; float32, camera-relative)".
    virtual std::string backend() const = 0;
    // The same scene and observer after a pose/resolution change. Discards accumulated samples.
    // Call only between passes, after preparing the observer. False means recreate the renderer.
    virtual bool resetObserver() { return false; }
};

}  // namespace owe
