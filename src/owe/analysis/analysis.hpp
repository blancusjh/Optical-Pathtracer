// Optical diagnostics computed by tracing real rays through the same world
// representation used for rendering (no separate sequential ray tracer).
#pragma once

#include <string>
#include <vector>

#include "owe/scene/prescription.hpp"
#include "owe/transport/transport.hpp"

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

// Where an eye must focus to see sharply what lies in a direction: the object itself, or the
// image of it formed by the specular surfaces in between — a telescope's virtual image, a
// reflection in a mirror, the enlarged image in a magnifier. Found by real rays in the world: a
// line of sight is followed through the smooth surfaces (refracting, reflecting on total internal
// reflection or at mirrors) to the first surface that scatters light or emits it, and rays leaving
// the eye a little off it are required to meet it again there.
//
// The eye focuses on the light it receives. With a pupil (radius > 0) the lines of sight are tried
// across it, nearest the centre first, and the first that ends on something visible — a lit or
// emitting surface, or the sky — is used; one that ends on black matter (a telescope's tube seen
// from outside its exit pupil) is used only when nothing else is seen. An eye beside a telescope's
// small exit pupil thus focuses on the planet its pupil's edge receives, not on the tube wall.
struct Accommodation {
    bool found = false;       // false: nothing to focus on (every line of sight lost)
    double distance = Inf;    // eye focus distance (m); Inf: relaxed, at infinity (also for an
                              // image beyond infinity, which no eye can focus)
    double objectPath = Inf;  // length of the line of sight from the eye to the object (m)
    int specular = 0;         // smooth surfaces crossed or reflected on the way
    Vec3 pupilPoint;          // where on the pupil that line of sight starts
};
Accommodation accommodation(const Scene& scene, const Vec3& eye, const Vec3& direction, uint32_t region,
                            double pupilRadius = 0, double lambda = 555.0);

}  // namespace owe
