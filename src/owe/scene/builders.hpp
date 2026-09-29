// Builders turn physical descriptions into regions and boundaries.
// Every builder produces closed bodies: a ray that misses a clear aperture
// meets the rim, the mount or whatever else is physically there.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "owe/scene/world.hpp"

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
// A glass vessel (a tumbler, a bowl, a wine glass without its stem) turned on a lathe: outer and
// inner profiles in the (r, z) half-plane, each a polyline from the axis at the bottom up to the rim
// whose corners may be rounded by fillets (tangent circular arcs). Every straight piece revolves into
// a disc, cylinder or cone, every fillet into a torus patch: exact analytic surfaces with exact
// normals (never a mesh). The rim joins the two profiles' last points, flat or rounded (a half
// circle). A liquid fills the inner cavity up to `level` (0: empty), with a flat surface.
struct ProfilePoint {
    double r = 0, z = 0, fillet = 0;
};
struct VesselSpec {
    std::vector<ProfilePoint> outer, inner;
    bool roundRim = true;
    std::string glass = "N-BK7", liquid = "water";
    double level = 0;  // liquid height above the vessel's origin (0: empty)
};
int buildVessel(World& w, const std::string& name, const VesselSpec& spec, int assembly, const Transform& xf);

// A body of liquid under a wavy surface (WaveSurface): the analytic surface on top (air above), a
// flat floor at z = −depth and four walls below z = 0. The floor and walls meet the liquid as
// ordinary interfaces with the air, or as opaque matter (a seabed, a basin) when given optics
// (≥ 0; diffuse, facing the liquid).
struct WaterSpec {
    std::vector<PlaneWave> waves;
    double halfX = 0.5, halfY = 0.5, margin = 0.1, depth = 0.2;
    std::string medium = "water";
    int bottomOptics = -1, wallOptics = -1;
    // With an opaque floor or walls the liquid fills a hole in a solid block: `rim` of ground around
    // it at z = 0 (with the wall optics, else the floor's) and `base` of it beneath the floor, so the
    // basin is closed matter whichever side light arrives from.
    double rim = 0.25, base = 0.05;
};
int buildWater(World& w, const std::string& name, const WaterSpec& spec, int assembly, const Transform& xf);
// A deterministic multi-frequency ripple field: `count` plane waves with wavelengths log-uniform in
// [minWavelength, maxWavelength], directions within `spread` about `direction`, random phases, and
// amplitudes giving each component the slope `slope / √count` (so the total r.m.s. slope ≈ slope/√2).
std::vector<PlaneWave> rippleField(int count, uint64_t seed, double minWavelength, double maxWavelength, double slope,
                                   double direction, double spread);

int buildPrism(World& w, const std::string& name, double apexAngle, double sideLength, double length,
               const BodyMaterial& m, int assembly, const Transform& xf);
// Two-sided sheet (screen, paper, wall) in the local z = 0 plane.
int buildSheet(World& w, const std::string& name, double halfX, double halfY, uint32_t optics, int emission,
               int assembly, const Transform& xf, bool disk = false);
// A closed mesh body; `shading`: its shading normals per triangle corner (empty: flat).
int buildMesh(World& w, const std::string& name, const MeshData& mesh, const BodyMaterial& m, int assembly,
              const Transform& xf, std::vector<MeshShape::Corners> shading = {});
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
// Closed surface of revolution about +z from a profile of (r, z) points traversed from the
// bottom upward; the ends are closed on the axis automatically. Normals face outward.
MeshData revolveProfile(std::vector<std::pair<double, double>> profile, int segments);
MeshData makeTorus(double majorRadius, double minorRadius, int segU, int segV);
// Fluted column shaft (Doric): radius r0 at the foot, r1 at the neck, entasis as a fraction of r0.
MeshData makeFlutedShaft(double height, double r0, double r1, double entasis, int flutes, double fluteDepth,
                         int segPerFlute, int segZ);
// Cylindrical wall (open surface, radius R, height H) with rectangular openings, each given as
// (centre azimuth from +y toward +x, width, sill height, top height).
struct WallOpening {
    double azimuth, width, sill, top;
};
MeshData makeRoundWall(double R, double H, int segAz, int segZ, const std::vector<WallOpening>& openings);
// Hemispherical shell (open surface) of radius R, optionally with an observing slit of constant
// linear width centred on azimuth slitAz (from +y toward +x), open from the horizon to elevation slitTop.
MeshData makeDome(double R, int segAz, int segEl, double slitAz, double slitWidth, double slitTop);
MeshData makeTreeTrunk(double height, double radius, int segments);
// Scatters conifers over a height field; returns the body index.
int buildForest(World& w, const std::string& name, const TerrainSpec& terrain, int count, double innerRadius,
                double outerRadius, uint64_t seed, uint32_t foliageOptics, uint32_t trunkOptics, int assembly,
                const Transform& xf);
// Fractal statue: a sphere carrying ever smaller copies of itself (depth levels, ratio r).
int buildFractalStatue(World& w, const std::string& name, double radius, int depth, double ratio, uint32_t optics,
                       uint32_t pedestalOptics, int assembly, const Transform& xf);

// A starfield: `count` small emissive spheres at `distance`, each of angular radius
// `angularRadius`, above elevation `minElevation`. Colours follow random stellar temperatures
// and brightness a steep power law up to luminance `brightest`. Stars are left out of
// next-event estimation (their light on the scene is negligible); they are seen when hit.
int buildStarfield(World& w, const std::string& name, int count, uint64_t seed, double distance,
                   double angularRadius, double brightest, double minElevation, int assembly, const Transform& xf);

// Orients triangles of a convex mesh outward from its centroid.
void orientOutward(MeshData& m);

}  // namespace owe
