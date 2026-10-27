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
// A render's three files: PREFIX.png (displayed with s's exposure, white balance and tone),
// PREFIX.pfm (raw) and PREFIX.json (its record).
void writeRenderFiles(const std::string& prefix, const Scene& scene, const Renderer& r, const RenderSettings& s,
                      const Image& img);
// The display settings (exposure, auto exposure, white balance, tone) of a record as the render
// wrote it into s; false when the file cannot be read.
bool readDisplaySettings(const std::string& recordPath, RenderSettings& s);

}  // namespace owe
