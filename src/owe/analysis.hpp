// Optical diagnostics computed by tracing real rays through the same world
// representation used for rendering (no separate sequential ray tracer).
#pragma once

#include <string>
#include <vector>

#include "prescription.hpp"
#include "transport.hpp"

namespace owe {

struct SpotResult {
    double lambda = 0, fieldDeg = 0;
    int launched = 0, arrived = 0;
    double bestFocusZ = 0;        // relative to the last vertex
    double rmsBest = 0;           // RMS spot radius at best focus (m)
    double rmsParaxial = 0;       // RMS spot radius at the paraxial image plane (m)
    double centroidY = 0;         // at the paraxial image plane (m)
    std::vector<Vec2> spot;       // positions at the paraxial image plane relative to the centroid
    // Afocal systems: the emerging beam is (ideally) collimated.
    double apparentAngle = 0;     // mean emerging direction (rad)
    double angularRms = 0;        // RMS angular spread of the emerging beam (rad)
};

struct LensReport {
    Paraxial paraxial;
    std::vector<SpotResult> spots;
    std::vector<std::pair<double, double>> lsa;  // (pupil zone 0..1, longitudinal SA in m)
    double chromaticFocalShift = 0;               // best focus(F) − best focus(C), m
    std::vector<std::pair<double, double>> distortion;  // (field deg, percent)
    std::string text() const;
};

// Builds the prescription into a scratch world and traces bundles.
LensReport analyzeLens(const Prescription& p, const std::vector<double>& fieldsDeg, const std::vector<double>& lambdas,
                       int rings = 8);
// Traces a meridional fan through the lens for visualisation.
std::vector<PathRecord> lensFan(const Prescription& p, double fieldDeg, double lambda, int rays, Scene& sceneOut);

}  // namespace owe
