// The world: Medium → Region → Boundary → Body → Assembly.
//
// A region is a connected domain filled with one medium. A boundary is an
// oriented surface separating the region on its front (+normal) side from the
// region on its back side. Bodies group regions and boundaries into physical
// objects; assemblies position bodies mechanically. The order in which light
// meets boundaries is never prescribed: it is discovered by ray casting.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "owe/scene/bvh.hpp"
#include "owe/scene/geometry.hpp"
#include "owe/scene/image.hpp"
#include "owe/core/medium.hpp"
#include "owe/core/spectrum.hpp"

namespace owe {

constexpr uint32_t kOutside = 0xFFFFFFFFu;  // placeholder: "the region surrounding this body"
constexpr uint32_t kNone = 0xFFFFFFFEu;

// Spatially varying reflectance (solid textures in boundary-local coordinates, or an image).
struct Texture {
    enum class Kind { None, Checker, Noise, Rings, Terrain, Bands, Radial, Marble, Image } kind = Kind::None;
    Spectrum a = Spectrum::constant(0.8), b = Spectrum::constant(0.2), c = Spectrum::constant(0.9);
    double scale = 1.0;       // cell size / feature size in metres (image: the size one image covers)
    double param = 0.0;       // terrain: snow line height; rings/bands/marble: turbulence
    // Image: its linear RGB are the weights of a, b, c, which are the RGB reflectance basis spectra;
    // projected in boundary-local coordinates along z (planar: x across, y up) or along whichever
    // axis the normal favours, blended (triplanar: no seams on boxes or meshes without UVs), or
    // by a model's texture coordinates (uv: GPU; the CPU reference and bodies without texture
    // coordinates project it planar).
    std::shared_ptr<const ImageRGB> image;
    enum class Mapping { Planar, Triplanar, UV } mapping = Mapping::Planar;
    double eval(const Vec3& localP, const Vec3& localN, double lambdaNm) const;
    void weights(const Vec3& localP, const Vec3& localN, double w[3]) const;
    double mix(const double w[3], double lambdaNm) const {
        double r = w[0] * a.eval(lambdaNm);
        if (w[1] != 0) r += w[1] * b.eval(lambdaNm);
        if (w[2] != 0) r += w[2] * c.eval(lambdaNm);
        return r;
    }
};
double valueNoise3(const Vec3& p);
double fbm3(const Vec3& p, int octaves);

// What happens to light at a boundary.
enum class SurfaceType {
    Null,        // no optical effect; only the region changes (index-matched/virtual boundary)
    Dielectric,  // Fresnel interface between the media of the two regions (smooth or GGX-rough)
    Diffuse,     // opaque Lambertian
    Conductor,   // opaque metal: complex-index Fresnel, smooth or GGX-rough
    Mirror,      // opaque specular reflector with prescribed spectral reflectance
    Absorber,    // perfectly black matter
    Detector,    // sensitive surface of a detector; absorbs what it records
    StainedGlass // thin coloured-glass slab (GPU): Fresnel at both faces, straight and diffuse transmission
};
const char* surfaceTypeName(SurfaceType t);

// A thin slab of coloured glass (stained glass, opalescent shades). Its normal-incidence internal
// transmittance τ₀ = clamp(gain · colour, 0, 1)^density comes from `transmittance` (the material's
// texture or reflectance spectrum); at angle θₜ inside it is τ₀^(1/cos θₜ). Fresnel acts at both faces
// with incoherent inter-reflections; a fraction `haze` of the transmitted power leaves diffusely
// (opalescent glass), the rest straight through, without refraction or a change of region.
// `reflectionScale` · colour of the remaining absorbed budget is reflected diffusely. A mask image
// (raw red channel, nearest texel) below `maskThreshold` hands the point to the opaque `stone`.
struct StainedGlass {
    double gain = 1, density = 1, haze = 0, reflectionScale = 0, thickness = 0.003;
    Spectrum index = Spectrum::constant(1.52);
    std::shared_ptr<const ImageRGB> mask;
    double maskThreshold = 0.5;
    int stone = -1;  // optics index of the opaque material where the mask says stone
};

struct SurfaceOptics {
    std::string name;
    SurfaceType type = SurfaceType::Dielectric;
    double roughness = 0;                                 // GGX α; 0 = specular
    // A material map (GPU): with a texture, the roughness of each of its components (a, b, c),
    // mixed by the texture's weights as the reflectance is; empty: `roughness` everywhere.
    std::vector<double> roughnessMap;
    // Surface relief (GPU): a height depth·fbm(p / scale) over the surface, in the body's frame,
    // whose slopes tilt the shading normal (paper grain, a concrete's pores, hammered metal).
    double reliefScale = 0, reliefDepth = 0;
    int reliefOctaves = 3;
    Spectrum reflectance = Spectrum::constant(1.0);       // diffuse albedo / mirror reflectance
    Texture texture;                                      // overrides reflectance (diffuse) or tint (conductor)
    ConductorSpectrum conductor;                          // Conductor only
    StainedGlass glass;                                   // StainedGlass only (reflectance/texture: its colour)
    bool backAbsorbs = false;                             // thin opaque sheet with a black back
    int detector = -1;                                    // absorbing detector or diffuse measurement screen
    // Optional diffuse-screen sampling guide, in world coordinates. Mixed with cosine sampling
    // so indirect illumination remains supported; does not change the BRDF.
    Vec3 sampleAimCenter, sampleAimNormal{0, 0, 1};
    double sampleAimRadius = 0, sampleAimShare = 0;

    double albedo(const Vec3& localP, const Vec3& localN, double lambdaNm) const {
        return texture.kind == Texture::Kind::None ? reflectance.eval(lambdaNm) : texture.eval(localP, localN, lambdaNm);
    }
    bool isDelta() const {
        if (type == SurfaceType::StainedGlass) return glass.haze <= 0 && glass.reflectionScale <= 0;
        return (type == SurfaceType::Dielectric || type == SurfaceType::Conductor) ? roughness < 1e-3
                                                                                   : type == SurfaceType::Mirror;
    }
};

struct Emission {
    // Sampling hint only: keep astronomical sources from taking the entire finite-light
    // proposal budget through their enormous emitting area. Radiance is unchanged.
    bool distant = false;
    Spectrum radiance;  // spectral radiance, Lambertian
    bool front = true, back = false;
    // Sampling choice only: disabled emitters are still seen when hit by a path.
    bool nee = true;
};

struct Region {
    std::string name;
    uint32_t medium = 0;
    int body = -1;
};

struct Boundary {
    std::string name;
    std::shared_ptr<const Shape> shape;
    Transform local;  // shape frame → body frame
    uint32_t front = kOutside, back = kOutside;
    uint32_t optics = 0;
    int emission = -1;
    int body = -1;
    // Compiled state.
    Transform toWorld, toLocal;
    AABB worldBox;
};

struct Body {
    std::string name;
    std::string kind;  // builder that produced it ("lens", "prism", "mesh", ...)
    int assembly = -1;
    Transform xf;       // body frame → assembly frame
    uint32_t ambient = 0;  // region surrounding the body
    std::vector<uint32_t> regions, boundaries;
    std::map<std::string, double> params;  // builder parameters kept for diagnostics and prescription views
};

struct Assembly {
    std::string name;
    int parent = -1;
    Transform xf;  // assembly frame → parent frame
};

struct Environment {
    enum class Sky { None, Uniform, Gradient, Map } skyModel = Sky::None;
    // Map: an equirectangular HDR image of the sky about `up` (u: azimuth from mapFrame's first axis
    // toward its second, turned by mapRotation; v: angle from the zenith), in linear RGB; a texel's
    // radiance is its colour through the RGB illuminant basis, times mapScale. It is a light for
    // next-event estimation, sampled in proportion to luminance × sin θ.
    std::shared_ptr<const ImageRGB> map;
    double mapRotation = 0, mapScale = 1;
    struct MapDistribution {
        int W = 0, H = 0;
        std::vector<double> marginal;     // H + 1 row CDF
        std::vector<double> conditional;  // H × (W + 1) column CDFs
        std::vector<double> rowWeight;    // H: each row's total weight
        double total = 0;
    } mapDist;
    void prepareMap();  // builds mapDist
    void mapFrame(Vec3& e1, Vec3& e2) const;
    Vec3 mapRGB(const Vec3& dir) const;
    // A direction toward the sky drawn from mapDist, and its solid-angle pdf.
    bool sampleMap(double u1, double u2, Vec3& dir, double& pdf) const;
    double mapPdf(const Vec3& dir) const;
    Spectrum zenith = Spectrum::constant(0), horizon = Spectrum::constant(0), ground = Spectrum::constant(0);
    Vec3 up{0, 0, 1};
    bool hasSun = false;
    Vec3 sunDir{0, 0, 1};  // unit vector toward the sun
    double sunAngularRadius = radians(0.2665);
    Spectrum sunRadiance = Spectrum::constant(0);
    // Sampling knob: fraction of next-event light selections given to the sun (0 = by power).
    // Power estimates for a distant sun scale with the scene's extent, which is meaningless in
    // scenes that also contain astronomical bodies.
    double sunNeeShare = 0;

    double sky(const Vec3& dir, double lambdaNm) const;
    double sun(const Vec3& dir, double lambdaNm) const {
        return (hasSun && dot(dir, sunDir) >= std::cos(sunAngularRadius)) ? sunRadiance.eval(lambdaNm) : 0;
    }
    double radiance(const Vec3& dir, double lambdaNm) const { return sky(dir, lambdaNm) + sun(dir, lambdaNm); }
    double sunSolidAngle() const { return 2 * Pi * (1 - std::cos(sunAngularRadius)); }
};

struct SurfaceHit {
    double t = Inf;
    Vec3 p;        // world position
    Vec3 n;        // world geometric normal, pointing to the boundary's front region
    Vec3 pLocal;   // shape-local position
    Vec3 nLocal;
    uint32_t boundary = kNone;
    uint32_t prim = 0;
};

// A light that next-event estimation can sample.
struct LightEntry {
    enum class Kind { Boundary, Sun, Environment } kind = Kind::Boundary;
    uint32_t boundary = kNone;
    double power = 0;
};

class World {
public:
    World();

    // --- Construction ---
    uint32_t addMedium(const Medium& m);
    // Finds a medium by name, loading it from the built-in catalog on first use.
    uint32_t medium(const std::string& name);
    uint32_t addRegion(const std::string& name, uint32_t medium, int body = -1);
    uint32_t addOptics(const SurfaceOptics& o);
    uint32_t dielectricOptics(double roughness = 0);
    uint32_t absorberOptics();
    uint32_t nullOptics();
    int addEmission(const Emission& e);
    int addAssembly(const std::string& name, int parent, const Transform& xf);
    int addBody(const std::string& name, const std::string& kind, int assembly, const Transform& xf, uint32_t ambient = kOutside);
    uint32_t addBoundary(int body, const std::string& name, std::shared_ptr<const Shape> shape, const Transform& local,
                         uint32_t front, uint32_t back, uint32_t optics, int emission = -1);
    void setAmbientMedium(uint32_t medium) { regions_[0].medium = medium; }
    // Bodies added while set are immersed in `region` (their outside is that region, e.g. stones on a
    // seabed are surrounded by water); kOutside restores the ambient region.
    void setImmersion(uint32_t region) { immersion_ = region; }
    uint32_t immersion() const { return immersion_; }
    // Exit pupils of afocal instruments (telescopes): where an eye receives their light, on the
    // axis of the instrument's assembly (light leaves along +z). A sampling hint for observers
    // (IdealObserver::prepare); transport never reads it.
    struct ExitPupil {
        std::string instrument;
        int assembly = -1;
        double z = 0, radius = 0;
    };
    void addExitPupil(const ExitPupil& e) { exitPupils_.push_back(e); }
    const std::vector<ExitPupil>& exitPupils() const { return exitPupils_; }

    // Compiles transforms, resolves placeholders and builds acceleration structures.
    // Must be called after any structural or placement change.
    void build();

    // --- Queries ---
    bool intersect(const Ray& ray, double tmax, SurfaceHit& hit) const;
    uint32_t locate(const Vec3& p) const;  // region containing p
    Transform assemblyToWorld(int assembly) const;
    Transform bodyToWorld(int body) const;

    const Medium& mediumOf(uint32_t region) const { return media_[regions_[region].medium]; }
    double indexOf(uint32_t region, double lambdaNm) const { return mediumOf(region).n(lambdaNm); }

    std::vector<Medium>& media() { return media_; }
    const std::vector<Medium>& media() const { return media_; }
    const std::vector<Region>& regions() const { return regions_; }
    std::vector<Region>& regions() { return regions_; }
    const std::vector<Boundary>& boundaries() const { return boundaries_; }
    std::vector<Boundary>& boundaries() { return boundaries_; }
    const std::vector<SurfaceOptics>& optics() const { return optics_; }
    std::vector<SurfaceOptics>& optics() { return optics_; }
    const std::vector<Emission>& emissions() const { return emissions_; }
    const std::vector<Body>& bodies() const { return bodies_; }
    std::vector<Body>& bodies() { return bodies_; }
    const std::vector<Assembly>& assemblies() const { return assemblies_; }
    std::vector<Assembly>& assemblies() { return assemblies_; }
    int findBody(const std::string& name) const;
    int findAssembly(const std::string& name) const;
    int findRegion(const std::string& name) const;

    Environment env;
    uint32_t ambientRegion() const { return 0; }
    AABB bounds() const { return bounds_; }
    double sceneRadius() const;
    Vec3 sceneCenter() const { return bounds_.valid() ? bounds_.centroid() : Vec3(); }

    // Hierarchy over boundaries' world boxes (valid after build()).
    const Bvh& bvh() const { return bvh_; }

    // Lights for next-event estimation.
    const std::vector<LightEntry>& lights() const { return lights_; }
    double lightSelectPdf(size_t i) const { return lightPdf_[i]; }
    size_t pickLight(double u) const;
    double totalLightPower() const { return totalLightPower_; }
    // Emitted power estimate of the sky dome (used by light tracing to share particles with lights).
    double environmentPower() const { return envPower_; }
    // The environment map's light (next-event estimation), or -1.
    int environmentLight() const { return environmentLight_; }
    int lightOfBoundary(uint32_t b) const { return b < lightOfBoundary_.size() ? lightOfBoundary_[b] : -1; }
    std::string boundaryLabel(uint32_t b) const;
    std::string regionLabel(uint32_t r) const;

private:
    std::vector<Medium> media_;
    std::vector<Region> regions_;
    std::vector<SurfaceOptics> optics_;
    std::vector<Emission> emissions_;
    std::vector<Body> bodies_;
    std::vector<Assembly> assemblies_;
    std::vector<Boundary> boundaries_;
    Bvh bvh_;
    AABB bounds_;
    std::vector<LightEntry> lights_;
    std::vector<double> lightCdf_, lightPdf_;
    double totalLightPower_ = 0;
    double envPower_ = 0;
    int environmentLight_ = -1;
    double computeEnvironmentPower() const;
    std::vector<int> lightOfBoundary_;
    std::vector<ExitPupil> exitPupils_;
    uint32_t immersion_ = kOutside;
    bool built_ = false;
};

}  // namespace owe
