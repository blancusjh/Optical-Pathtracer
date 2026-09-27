// Reproducibility records: every render writes one (scene hash and edits, backend, integrator,
// sampling, seed, samples, statistics, display settings), whichever backend produced it.
#pragma once

#include <cstdint>
#include <string>

#include "owe/render/renderer.hpp"

namespace owe {

std::string renderMetadataJSON(const Scene& scene, const Renderer& r, const RenderSettings& s);
uint64_t fnv1a(const std::string& s);

}  // namespace owe
