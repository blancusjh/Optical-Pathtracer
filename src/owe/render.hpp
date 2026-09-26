// Progressive rendering and reproducibility metadata.
#pragma once

#include <functional>
#include <string>

#include "transport.hpp"

namespace owe {

// Accumulates an unbiased estimate progressively: each pass adds samples;
// the raw accumulation is never altered by display processing.
class ProgressiveRenderer {
public:
    ProgressiveRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings);

    // path: spp; light: particles per pixel; hybrid: both, with a disjoint path-space partition.
    void runPass(int samplesPerPixel);
    Image resolve() const;
    int passes() const { return passes_; }
    long long samplesPerPixel() const { return totalSpp_; }
    double seconds() const { return seconds_; }
    const TransportStats& stats() const { return stats_; }
    const Detector& detector() const { return *det_; }
    const RenderSettings& settings() const { return settings_; }

private:
    void passPath(int spp);
    void passLight(int particlesPerPixel);

    const Scene& scene_;
    const Detector* det_;
    RenderSettings settings_;
    Tracer tracer_;
    Film film_;       // camera-path estimates (per-pixel sample counts)
    Film lightFilm_;  // particle splats (normalised by the particle count)
    double particles_ = 0;
    int passes_ = 0;
    long long totalSpp_ = 0;
    double seconds_ = 0;
    TransportStats stats_;
    int threads_;
};

// Colour-matching weights for one wavelength sample: cmf(λ)/pdf(λ)/∫ȳ.
void spectralWeights(double lambda, double pdf, double out[3]);

// Image output.
void writePFM(const std::string& path, const Image& img);                 // linear sRGB, raw
// Display rendering: exposure, optional white balance (Bradford adaptation from a Planckian white
// of `whiteKelvin` to D65; 0 = none), highlight roll-off, sRGB encoding. Never applied to PFM.
void writePNG(const std::string& path, const Image& img, double exposureEV, bool autoExposure, double whiteKelvin = 0);
std::string renderMetadataJSON(const Scene& scene, const ProgressiveRenderer& r, const RenderSettings& s);
uint64_t fnv1a(const std::string& s);

}  // namespace owe
