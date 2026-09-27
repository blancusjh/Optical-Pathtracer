// Inspection: reveal the transport paths behind any image, derived from the
// exact state being rendered (the same Tracer, world and random streams).
#pragma once

#include <string>
#include <vector>

#include "owe/transport/transport.hpp"

namespace owe {

// Plane onto which paths and a cross-section of the world are drawn.
struct ViewPlane {
    Vec3 origin{0, 0, 0};
    Vec3 u{0, 0, 1};  // horizontal axis in the drawing
    Vec3 v{0, 1, 0};  // vertical axis in the drawing
};

struct SvgOptions {
    int width = 1200;
    double margin = 0.05;         // fraction of extent
    bool drawSection = true;      // slice the world with the view plane
    int sectionLines = 700;
    double strokeWidth = 1.0;
    std::string title;
    // Optional explicit bounds in plane coordinates (u0,v0,u1,v1); empty = automatic.
    std::vector<double> bounds;
};

void writePathsSVG(const std::string& path, const World& world, const std::vector<PathRecord>& paths,
                   const ViewPlane& plane, const SvgOptions& opt);

// Approximate sRGB colour of a monochromatic wavelength, for drawing.
void wavelengthToRGB(double lambdaNm, double rgb[3]);

// "Why is this pixel this colour?"
struct PixelProbe {
    struct Group {
        std::string signature;
        double valueY = 0;   // contribution to the pixel's Y (luminance-like) estimate
        int paths = 0;
        size_t example = 0;  // index into records
    };
    int px = 0, py = 0, samples = 0;
    XYZ estimate;
    std::vector<Group> groups;          // sorted by contribution
    std::vector<PathRecord> records;
    std::string text(const World& w) const;
};
PixelProbe probePixel(const Scene& scene, int detectorIndex, int px, int py, int samples, uint64_t seed, int maxDepth = 64);

// "Where does its light go?"
struct EmissionProbe {
    struct Fate {
        std::string label;
        int rays = 0;
        double energy = 0;
    };
    std::vector<Fate> fates;
    std::vector<PathRecord> records;
    int rays = 0;
    std::string text() const;
};
EmissionProbe probeEmission(const Scene& scene, const Vec3& origin, const Vec3& direction, double coneHalfAngle,
                            int rays, const std::vector<double>& lambdas, WalkMode mode, uint64_t seed);

// Human-readable table of one recorded path.
std::string describePath(const World& w, const PathRecord& r);
std::string pathSignature(const World& w, const PathRecord& r, size_t upToVertex);
std::string pathsJSON(const World& w, const std::vector<PathRecord>& paths);

}  // namespace owe
