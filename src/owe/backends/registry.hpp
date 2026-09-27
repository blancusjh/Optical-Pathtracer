// The backends compiled into this build, and the one way front ends obtain a renderer. Nothing
// outside this registry names a concrete backend: the command line, the studio, the viewer and the
// tests ask for one by name (RenderSettings::backend), so a render moves between CPU and GPU by
// changing that name alone.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "owe/render/backend.hpp"

namespace owe {

// Every backend in this build, the reference first. A backend compiled out is still listed (it
// reports itself unavailable and why).
const std::vector<const Backend*>& backends();
// Null when no backend has that name.
const Backend* findBackend(const std::string& name);
// Throws, listing the known names, when no backend has that name.
const Backend& backend(const std::string& name);
// The IEEE-754 double reference ("cpu") against which every other backend is validated.
const Backend& referenceBackend();
// Adapts settings to what their backend implements: an integrator it lacks becomes "path", which
// every backend implements and which estimates the same image (with noisier directly seen
// caustics). Returns a note describing the change, or "" when nothing changed.
std::string adaptToBackend(RenderSettings& settings);
// backend(settings.backend).createRenderer(...).
std::unique_ptr<Renderer> makeRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings);

}  // namespace owe
