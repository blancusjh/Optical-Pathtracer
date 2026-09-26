// The single transport engine. Camera paths, light paths and diagnostic ray
// walks all use the same intersection, region bookkeeping and interface optics.
#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "scattering.hpp"
#include "scene.hpp"

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
    double value = 0;      // contribution to the sample (before detector weight)
    std::string source;    // emitter label
    bool nee = false;      // collected by next-event estimation
};

struct PathRecord {
    double lambda = 0;
    std::vector<PathVertex> v;
    std::vector<PathContribution> c;
};

struct TransportStats {
    uint64_t paths = 0, segments = 0, inconsistencies = 0, leaks = 0;
    void add(const TransportStats& o) {
        paths += o.paths; segments += o.segments; inconsistencies += o.inconsistencies; leaks += o.leaks;
    }
};

enum class WalkMode {
    Stochastic,         // Monte-Carlo branching exactly as in rendering
    PrimaryTransmission // deterministic: always refract at smooth dielectrics (weight × T), reflect on TIR
};

class Tracer {
public:
    Tracer(const Scene& scene, int maxDepth = 64, int rrDepth = 6) : scene_(scene), world_(scene.world),
        maxDepth_(maxDepth), rrDepth_(rrDepth) {}

    // Spectral radiance (or its adjoint measurement) arriving along -ray.d at ray.o.
    double radiance(Ray ray, uint32_t region, double lambda, Rng& rng, TransportStats& st, PathRecord* rec = nullptr) const;

    // Emits one particle and deposits its contributions into `film` for detector `det`.
    // `scale` multiplies deposits (typically cmf(λ)/pdf(λ)/∫ȳ, one entry per XYZ channel).
    void traceParticle(const Detector& det, double lambda, const double scaleXYZ[3], Rng& rng, Film& film,
                       TransportStats& st, PathRecord* rec = nullptr) const;

    // Follows a single ray through the world and records every event.
    PathRecord walk(Ray ray, uint32_t region, double lambda, WalkMode mode, Rng& rng, int maxEvents = 256) const;

    // Transmittance along a straight segment through null boundaries; 0 if blocked.
    // dist may be infinite (escape to the environment). finalRegion receives the end region.
    double transmittance(Vec3 o, uint32_t region, const Vec3& dir, double dist, double lambda,
                         uint32_t* finalRegion = nullptr) const;

    Interface makeInterface(const SurfaceHit& hit, double lambda) const;

private:
    double directLighting(const Interface& it, const SurfaceHit& hit, const Vec3& d, double lambda, Rng& rng,
                          std::string* source, bool isMedium, double g, uint32_t region) const;
    double lightPdf(const Vec3& ref, const SurfaceHit& hit) const;
    double sunPdf() const;

    const Scene& scene_;
    const World& world_;
    int maxDepth_, rrDepth_;
};

}  // namespace owe
