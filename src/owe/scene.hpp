// A scene is a world plus the detectors observing it and default render settings.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "detector.hpp"
#include "world.hpp"

namespace owe {

struct RenderSettings {
    std::string detector;           // empty = first detector
    std::string integrator = "path";  // "path" (camera paths) or "light" (particle tracing)
    int spp = 16;                   // samples per pixel (path) or particles per pixel (light)
    uint64_t seed = 1;
    int maxDepth = 64;
    int rrDepth = 6;
    int threads = 0;                // 0 = hardware concurrency
    double exposure = 0;            // EV applied to the display image only
    bool autoExposure = true;
};

struct Scene {
    World world;
    std::vector<std::unique_ptr<Detector>> detectors;
    RenderSettings render;
    std::string sourcePath;
    std::string sourceText;  // exact scene description, hashed into render metadata

    int findDetector(const std::string& name) const {
        for (size_t i = 0; i < detectors.size(); ++i)
            if (detectors[i]->name == name) return int(i);
        return -1;
    }
    // Builds the world and prepares detectors. Call after construction or edits.
    void build() {
        world.build();
        for (auto& d : detectors) d->prepare(world);
    }
};

}  // namespace owe
