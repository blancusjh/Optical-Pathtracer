// The sequential lens table: a compact, literature-compatible *view* of
// rotationally symmetric optics. It is converted into real bodies (regions,
// boundaries, rims, stops) before any light is traced; the table never
// prescribes the order in which light meets surfaces.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "owe/scene/builders.hpp"
#include "owe/scene/world.hpp"

namespace owe {

struct PrescriptionSurface {
    std::string label;
    double R = Inf;            // metres
    double t = 0;              // distance to next vertex (metres)
    std::string medium = "air";  // medium after this surface
    double sd = 0;             // semi-diameter (metres)
    double k = 0;
    std::vector<double> A;     // A4, A6, ... (SI units: m^-3, m^-5, ...)
    bool stop = false;
    // A reflecting row (mirrors): R and t are signed in a fixed chart whose +z is the incoming light's
    // direction; reflection reverses the travel, so t is negative on a leg travelling −z.
    bool reflect = false;
    std::string material;      // the scene material of the reflecting face (Mirror or Conductor)
    double substrate = 0;      // axial depth of the mirror's backing
    double hole = 0;           // central bore diameter
    bool solve = false;        // the gap `afocal` adjusts (mirror systems)
    int spiderCount = 0;       // support = spider(count, width, angle, outer_radius): opaque struts
    double spiderWidth = 0, spiderAngle = 0, spiderOuter = 0;
};

struct Prescription {
    std::string name;
    std::string ambient = "air";
    double objectDistance = Inf;
    double wavelength = LambdaD;
    std::vector<PrescriptionSurface> surfaces;
    std::string source;
};

Prescription parsePrescription(const std::string& text, const std::string& origin = "<string>");
Prescription loadPrescription(const std::string& path);
std::string formatPrescription(const Prescription& p, double unit = 1e-3);

using IndexFn = std::function<double(const std::string& medium, double lambdaNm)>;
IndexFn catalogIndex();

struct Paraxial {
    double A = 1, B = 0, C = 0, D = 1;  // (y, n·u) system matrix, first to last vertex
    bool afocal = false;
    double efl = Inf, bfl = Inf, ffl = Inf;
    double angularMagnification = 0;    // afocal systems
    double imageDistance = Inf;         // from last vertex, for objectDistance
    int stopIndex = -1;
    double entrancePupilZ = 0, entrancePupilRadius = 0;  // z relative to first vertex
    double exitPupilZ = 0, exitPupilRadius = 0;          // z relative to first vertex
    double fNumber = Inf;
    double length = 0;                  // first to last vertex
};
Paraxial paraxialAnalysis(const Prescription& p, double lambdaNm, const IndexFn& index);
// Sets the largest air gap so that the system becomes afocal; returns the gap.
double solveAfocal(Prescription& p, double lambdaNm, const IndexFn& index);

struct BuiltInstrument {
    int assembly = -1;
    std::vector<int> bodies;
    std::vector<double> vertexZ;
    double lastVertexZ = 0;
    double rearNearestZ = 0;       // largest z occupied by the last lens group
    double rearEdgeRadius = 0;
    double maxRadius = 0;
    Paraxial paraxial;
};
// Builds lens groups (cemented where media are adjacent), stops and optional
// mounts (annular baffles out to tubeRadius) into a new assembly.
// Reflecting rows become mirrors with the scene material `materialOf(name)` (an ideal mirror when
// none is given).
using MaterialFn = std::function<uint32_t(const std::string& name)>;
BuiltInstrument buildPrescription(World& w, const Prescription& p, const std::string& name, int parentAssembly,
                                  const Transform& xf, double tubeRadius, const IndexFn& index,
                                  LensSpec::Rim rim = LensSpec::Rim::Black, const MaterialFn& materialOf = {});
// Whether any row reflects.
bool hasMirrors(const Prescription& p);

}  // namespace owe
