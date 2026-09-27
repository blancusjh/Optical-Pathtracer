// Statistical comparison of two renderers of the same measurement (e.g. the GPU backend against
// the CPU reference). Both are unbiased Monte-Carlo estimators, so their images differ by noise
// only: each is rendered with independent seeds, block means of X, Y and Z are formed per run, and
// the difference of the means is compared with its standard error estimated from the runs.
#pragma once

#include <string>
#include <vector>

#include "owe/render/renderer.hpp"

namespace owe {

struct ComparisonReport {
    int runs = 0, blocks = 0, tests = 0;  // tests: blocks × 3 channels with non-zero variance
    double meanA = 0, seA = 0, meanB = 0, seB = 0;  // image mean Y and its standard error
    double zImage = 0;           // (meanB − meanA) / combined standard error
    double maxAbsZ = 0;          // largest block |z|
    int over3 = 0, over4 = 0;    // block tests with |z| > 3 and > 4
    double chi2PerTest = 0;      // Σ z² / tests (≈ 1 when the estimators agree)
    double secondsA = 0, secondsB = 0;
    // The largest deviations: block position (in blocks), channel (0 X, 1 Y, 2 Z), both means, z.
    struct Outlier {
        int bx, by, channel;
        double a, b, z;
    };
    std::vector<Outlier> worst;
    std::string backendA, backendB;
    // Agreement: image mean within 4 σ, at most 1% of block tests beyond 4 σ, Σz²/n below 2
    // (runs are few, so block z follows a Student t with runs − 1 degrees of freedom).
    bool consistent() const;
    std::string text() const;
};

// Renders `runs` independent estimates of `sppPerRun` samples with settings a and with settings b
// (seeds differ between the two), in blocks of blockSize × blockSize pixels.
ComparisonReport compareRenderers(const Scene& scene, int detectorIndex, const RenderSettings& a,
                                  const RenderSettings& b, int sppPerRun, int runs, int blockSize);

}  // namespace owe
