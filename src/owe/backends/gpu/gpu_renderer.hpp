// The GPU renderer (internal to the backend): the flattened scene in device-local storage buffers,
// kernels dispatched in bounded queued slices, accepted film accumulated on the device.
#pragma once

#include <memory>

#include "owe/render/renderer.hpp"

namespace owe::gpu {

std::unique_ptr<Renderer> makeRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings);

}  // namespace owe::gpu
