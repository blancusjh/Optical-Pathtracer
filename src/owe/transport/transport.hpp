// The reference transport engine (IEEE-754 double). Camera paths, light paths and diagnostic ray
// walks all use the same intersection, region bookkeeping and interface optics. It defines the
// transport every backend estimates: the cpu backend runs it on host threads, the gpu backend's
// kernels are a float32 port of it validated against it, and the lens analyser, the inspector and
// the loader (a camera's real focus) use its walks directly.
#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "owe/transport/scattering.hpp"
#include "owe/scene/scene.hpp"
#include "owe/scene/transport_stats.hpp"
#include "owe/core/wavelengths.hpp"

namespace owe {

// One recorded event along a path (inspection mode).
struct PathVertex {
    Vec3 p;
    Vec3 dOut;  // propagation direction after the event
    EventKind event = EventKind::None;
    uint32_t boundary = kNone;
    uint32_t regionFrom = kNone, regionTo = kNone;
    double nI = 0, nT = 0;       // refractive indices on both sides
    double thetaI = 0, thetaT = 0;  // radians
    double R = 0, T = 0;         // Fresnel/reflectance values at the event
    double opl = 0;              // accumulated optical path length Σ n·s (m)
    double beta = 0;             // path throughput after the event
};

struct PathContribution {
    size_t vertex = 0;     // vertex at which the contribution was collected
    Spec4 value;           // contribution per wavelength (before detector weight)
    std::string source;    // emitter label
    bool nee = false;      // collected by next-event estimation
};

struct PathRecord {
    double lambda = 0;     // hero wavelength
    Wavelengths wavelengths;
    std::vector<PathVertex> v;
    std::vector<PathContribution> c;
};

enum class WalkMode {
    Stochastic,         // Monte-Carlo branching exactly as in rendering
    PrimaryTransmission // deterministic: always refract at smooth dielectrics (weight × T), reflect on TIR
};

class Tracer {
public:
    Tracer(const Scene& scene, int maxDepth = 64, int rrDepth = 6) : scene_(scene), world_(scene.world),
        maxDepth_(maxDepth), rrDepth_(rrDepth) {}

    // Hybrid estimation: paths of the form  eye → D → S⁺ → light  (caustics seen directly;
    // D non-specular, S specular) are owned by light tracing, every other path by path
    // tracing. The partition is disjoint and complete, so the summed estimate is unbiased.
    void setCausticPartition(bool on) { partition_ = on; }
    // Minimum probability of choosing either branch at smooth dielectric interfaces (0 = exact
    // Fresnel probabilities). Unbiased for any value in [0, 0.5); raises ghost-path sampling.
    void setFresnelFloor(double f) { fresnelFloor_ = clampd(f, 0.0, 0.49); }

    // Spectral radiance (or its adjoint measurement) arriving along -ray.d at ray.o, at the
    // path's wavelengths. `wl` may have its secondary wavelengths terminated on return.
    Spec4 radiance(Ray ray, uint32_t region, Wavelengths& wl, Rng& rng, TransportStats& st,
                   PathRecord* rec = nullptr) const;

    // Emits one particle (four wavelengths) and splats its XYZ contributions into `film`.
    void traceParticle(const Detector& det, Wavelengths wl, Rng& rng, Film& film, TransportStats& st,
                       PathRecord* rec = nullptr) const;

    // Follows a single ray through the world and records every event.
    PathRecord walk(Ray ray, uint32_t region, double lambda, WalkMode mode, Rng& rng, int maxEvents = 256) const;

    // Transmittance along a straight segment through null boundaries; 0 if blocked.
    // dist may be infinite (escape to the environment). finalRegion receives the end region.
    Spec4 transmittance(Vec3 o, uint32_t region, const Vec3& dir, double dist, const Wavelengths& wl,
                        uint32_t* finalRegion = nullptr) const;

    Interface makeInterface(const SurfaceHit& hit, double lambda) const;

private:
    // surface: the hit's interface at the hero wavelength (null for medium scattering).
    Spec4 directLighting(const SurfaceHit& hit, const Interface* surface, const Vec3& d, const Wavelengths& wl, Rng& rng,
                         std::string* source, double g, uint32_t region) const;
    double lightPdf(const Vec3& ref, const SurfaceHit& hit) const;
    double sunPdf() const;

    const Scene& scene_;
    const World& world_;
    int maxDepth_, rrDepth_;
    bool partition_ = false;
    double fresnelFloor_ = 0;
};

}  // namespace owe
