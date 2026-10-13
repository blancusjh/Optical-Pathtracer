// Reproducibility records: every render writes one (scene hash and edits, backend, integrator,
// sampling, seed, samples, statistics, display settings), whichever backend produced it.
#pragma once

#include <cstdint>
#include <string>

#include "owe/render/renderer.hpp"

namespace owe {

// Use the image actually written, without resolving (and reading back) the renderer again.
std::string renderMetadataJSON(const Scene& scene, const Renderer& r, const RenderSettings& s, const Image& img);
uint64_t fnv1a(const std::string& s);

}  // namespace owe
