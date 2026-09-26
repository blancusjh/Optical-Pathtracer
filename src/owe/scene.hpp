// A scene is a world plus the detectors observing it and default render settings.
#pragma once

#include <map>
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
    double fresnelFloor = 0;        // sampling knob: min branch probability at smooth dielectrics
    double exposure = 0;            // EV applied to the display image only
    double whiteBalance = 0;        // display white point (Kelvin, Planckian); 0 = none
    bool autoExposure = true;
};

// Settings a detector carries itself. An eyepiece on Saturn wants a fixed exposure, daylight
// white and nearly all light samples on the sun; the candlelit room wants auto exposure, a warm
// white point and its samples on the candles. Display settings never touch the raw image; the
// sun share is a sampling choice (it changes noise, not the expected value).
struct DetectorSettings {
    bool hasExposure = false, hasWhiteBalance = false, hasSunShare = false;
    double exposure = 0, whiteBalance = 0, sunShare = 0;
};

struct Scene {
    World world;
    std::vector<std::unique_ptr<Detector>> detectors;
    RenderSettings render;
    std::string sourcePath;
    std::string sourceText;  // exact scene description, hashed into render metadata
    std::vector<std::string> edits;              // command-line / studio edits applied on top
    std::map<std::string, std::string> notes;    // human-readable facts per detector (cameras: EFL, f/N, DOF)
    std::map<std::string, DetectorSettings> display;  // per-detector overrides

    // Render settings for a detector: the render block, then the detector's own display keys.
    RenderSettings settingsFor(int di) const {
        RenderSettings rs = render;
        rs.detector = detectors[size_t(di)]->name;
        auto it = display.find(rs.detector);
        if (it != display.end()) {
            if (it->second.hasExposure) { rs.exposure = it->second.exposure; rs.autoExposure = false; }
            if (it->second.hasWhiteBalance) rs.whiteBalance = it->second.whiteBalance;
        }
        return rs;
    }

    int findDetector(const std::string& name) const {
        for (size_t i = 0; i < detectors.size(); ++i)
            if (detectors[i]->name == name) return int(i);
        return -1;
    }
    // Applies a detector's sampling override (sun share) to the world, rebuilding if it changes.
    void useDetector(int di) {
        auto it = display.find(detectors[size_t(di)]->name);
        if (it == display.end() || !it->second.hasSunShare || world.env.sunNeeShare == it->second.sunShare) return;
        world.env.sunNeeShare = it->second.sunShare;
        build();
    }
    // Builds the world and prepares detectors. Call after construction or edits.
    void build() {
        world.build();
        for (auto& d : detectors) d->prepare(world);
    }
};

}  // namespace owe
