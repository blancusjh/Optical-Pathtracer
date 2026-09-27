// A scene is a world plus the detectors observing it and default render settings.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "owe/scene/detector.hpp"
#include "owe/scene/world.hpp"

namespace owe {

struct RenderSettings {
    std::string detector;           // empty = first detector
    std::string integrator = "path";  // "path" (camera paths) or "light" (particle tracing)
    int spp = 16;                   // samples per pixel (path) or particles per pixel (light)
    uint64_t seed = 1;
    int maxDepth = 64;
    int rrDepth = 6;
    int threads = 0;                // cpu backend threads; 0 = hardware concurrency
    double fresnelFloor = 0;        // sampling knob: min branch probability at smooth dielectrics
    double exposure = 0;            // EV applied to the display image only
    double whiteBalance = 0;        // display white point (Kelvin, Planckian); 0 = none
    bool autoExposure = true;
    // Where the transport runs: a name in the backend registry (owe/backends/registry.hpp), "cpu"
    // (the IEEE-754 double reference) or "gpu" (portable Vulkan). Nothing else changes with it.
    std::string backend = "cpu";
    int device = -1;                // backend device index (GPU; -1 = the best one found)
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
    std::optional<double> defaultSunShare; // captured before the first detector sampling override
    std::string indirectGuide; // optional spherical reflector, an importance-sampling hint only
    std::vector<size_t> indirectGuidedOptics;

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
        if (!defaultSunShare) defaultSunShare = world.env.sunNeeShare;
        auto it = display.find(detectors[size_t(di)]->name);
        double share = it != display.end() && it->second.hasSunShare ? it->second.sunShare : *defaultSunShare;
        if (world.env.sunNeeShare == share) return;
        world.env.sunNeeShare = share;
        build();
    }
    // Builds the world and prepares detectors. Call after construction or edits.
    void build() {
        world.build();
        for (size_t i : indirectGuidedOptics) {
            world.optics().at(i).sampleAimRadius = 0;
            world.optics().at(i).sampleAimShare = 0;
        }
        indirectGuidedOptics.clear();
        if (!indirectGuide.empty()) {
            int body = world.findBody(indirectGuide);
            if (body < 0) throw std::runtime_error("unknown indirect_guide body: " + indirectGuide);
            const Boundary* target = nullptr;
            const SphereShape* sphere = nullptr;
            for (const auto& b : world.boundaries()) if (b.body == body) {
                if (auto s = dynamic_cast<const SphereShape*>(b.shape.get())) { target = &b; sphere = s; break; }
            }
            if (!sphere) throw std::runtime_error("indirect_guide must name a spherical body");
            Vec3 center = target->toWorld.point({0, 0, 0});
            Vec3 normal = lengthSq(center) > 0 ? normalize(-center) : Vec3(0, 0, 1);
            for (size_t i = 0; i < world.optics().size(); ++i) {
                auto& optics = world.optics()[i];
                if (i == target->optics || optics.type != SurfaceType::Diffuse || optics.sampleAimRadius > 0) continue;
                optics.sampleAimCenter = center;
                optics.sampleAimNormal = normal;
                optics.sampleAimRadius = sphere->radius() * 1.001;
                optics.sampleAimShare = 0.5; // retain the whole hemisphere, including every other path
                indirectGuidedOptics.push_back(i);
            }
        }
        for (auto& d : detectors) d->prepare(world);
    }
};

}  // namespace owe
