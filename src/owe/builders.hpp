// Builders turn physical descriptions into regions and boundaries.
// Every builder produces closed bodies: a ray that misses a clear aperture
// meets the rim, the mount or whatever else is physically there.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "world.hpp"

namespace owe {

// One optical surface of a rotationally symmetric element (lengths in metres).
struct SurfaceSpec {
    double R = Inf;                 // vertex radius of curvature (Inf = flat); R > 0: centre at +z
    double k = 0;                   // conic constant
    std::vector<double> A;          // even aspheric coefficients A4, A6, ...
    double semiDiameter = 0;        // mechanical semi-diameter of the polished surface
    double clearSemiDiameter = 0;   // annotation
    double curvature() const { return std::isinf(R) ? 0.0 : 1.0 / R; }
};

// A lens body: surfaces S0..Sn separated by n media along +z (singlet n = 1,
// cemented doublet n = 2, ...). Regions sharing a cemented surface share a boundary.
struct LensSpec {
    std::vector<SurfaceSpec> surfaces;
    std::vector<std::string> media;
    std::vector<double> thickness;  // vertex-to-vertex distances
    double edgeRadius = 0;          // mechanical radius of the rim (0 = largest semi-diameter)
    enum class Rim { Ground, Black, Polished } rim = Rim::Ground;
};

// How a solid body's surface interacts with light.
struct BodyMaterial {
    bool transparent = false;
    std::string medium = "opaque";  // interior medium when transparent
    uint32_t optics = 0;            // boundary optics (dielectric/rough dielectric for transparent bodies)
    int emission = -1;
};

int buildLens(World& w, const std::string& name, const LensSpec& spec, int assembly, const Transform& xf,
              uint32_t ambient = kOutside);
// Thick-lens helper: radii for a symmetric biconvex/biconcave (form = "bi") or
// plano-convex/concave (form = "plano") lens of focal length f at λ_d.
LensSpec designSimpleLens(double focal, const std::string& glass, double diameter, double thickness,
                          const std::string& form);

// Mirror: reflective sag surface on an opaque substrate. optics should be Conductor or Mirror.
int buildMirror(World& w, const std::string& name, const SurfaceSpec& front, double thickness, uint32_t optics,
                int assembly, const Transform& xf, double holeRadius = 0);
// Flat thin mirror (rectangular or elliptical), reflective on its front, black behind.
int buildFlatMirror(World& w, const std::string& name, double halfX, double halfY, bool elliptical, uint32_t optics,
                    int assembly, const Transform& xf);
// Aperture stop / iris / baffle: annular absorber in the z = 0 plane.
int buildStop(World& w, const std::string& name, double innerRadius, double outerRadius, int assembly,
              const Transform& xf);
// Absorbing tube (open cylinder), e.g. a lens barrel or telescope tube.
int buildTube(World& w, const std::string& name, double radius, double z0, double z1, int assembly,
              const Transform& xf, uint32_t optics);

int buildSphere(World& w, const std::string& name, double radius, const BodyMaterial& m, int assembly,
                const Transform& xf);
int buildBox(World& w, const std::string& name, const Vec3& size, const BodyMaterial& m, int assembly,
             const Transform& xf);
int buildCylinder(World& w, const std::string& name, double radius, double height, const BodyMaterial& m,
                  int assembly, const Transform& xf);
// Triangular prism: isosceles cross-section in the local x–y plane (apex toward +y), extruded along z.
int buildPrism(World& w, const std::string& name, double apexAngle, double sideLength, double length,
               const BodyMaterial& m, int assembly, const Transform& xf);
// Two-sided sheet (screen, paper, wall) in the local z = 0 plane.
int buildSheet(World& w, const std::string& name, double halfX, double halfY, uint32_t optics, int emission,
               int assembly, const Transform& xf, bool disk = false);
int buildMesh(World& w, const std::string& name, const MeshData& mesh, const BodyMaterial& m, int assembly,
              const Transform& xf);
// Drinking glass with water: z up, base at z = 0. Optionally a vertical rod (radius rodRadius,
// centred at rodX, rodY, rising to rodTop) stands in it, crossing the liquid surface through a
// matching hole so that all matter stays disjoint. rodMedium empty = opaque rod with rodOptics.
struct CupRod {
    double radius = 0, x = 0, y = 0, top = 0;
    std::string medium;        // transparent rod (e.g. "N-BK7"); empty = opaque
    uint32_t optics = 0;       // opaque rod surface
};
int buildCup(World& w, const std::string& name, double outerRadius, double wall, double base, double height,
             double waterLevel, const std::string& glass, const std::string& liquid, int assembly, const Transform& xf,
             const CupRod& rod = CupRod{});

// Procedural natural geometry.
struct TerrainSpec {
    double sizeX = 200, sizeY = 200;
    int resolution = 256;
    double amplitude = 30;     // metres
    double featureSize = 80;   // metres
    double ridge = 0.0;        // 0..1 blend toward ridged noise
    uint64_t seed = 1;
    double flatRadius = 0;     // flatten the terrain within this radius of the origin
};
double terrainHeight(const TerrainSpec& t, double x, double y);
MeshData makeTerrain(const TerrainSpec& t);
MeshData makeTreeFoliage(double height, double radius, int segments);
MeshData makeTreeTrunk(double height, double radius, int segments);
// Scatters conifers over a height field; returns the body index.
int buildForest(World& w, const std::string& name, const TerrainSpec& terrain, int count, double innerRadius,
                double outerRadius, uint64_t seed, uint32_t foliageOptics, uint32_t trunkOptics, int assembly,
                const Transform& xf);
// Fractal statue: a sphere carrying ever smaller copies of itself (depth levels, ratio r).
int buildFractalStatue(World& w, const std::string& name, double radius, int depth, double ratio, uint32_t optics,
                       uint32_t pedestalOptics, int assembly, const Transform& xf);

// Orients triangles of a convex mesh outward from its centroid.
void orientOutward(MeshData& m);

}  // namespace owe
