// Surface scattering at boundaries. The same functions serve camera paths
// (radiance transport) and light paths (importance/flux transport).
#pragma once

#include "world.hpp"

namespace owe {

enum class TransportMode { Radiance, Importance };

enum class EventKind : uint8_t {
    None,
    Emit,
    Camera,
    Reflect,     // specular reflection (Fresnel branch or mirror)
    Refract,     // specular transmission
    TIR,         // total internal reflection
    Glossy,      // rough reflection
    GlossyT,     // rough transmission
    Diffuse,
    Scatter,     // volume scattering
    Pass,        // null boundary
    Absorb,
    Detect,
    Escape,
    Terminate    // Russian roulette or depth limit
};
const char* eventName(EventKind k);

// Local state at a boundary hit.
struct Interface {
    const SurfaceOptics* optics = nullptr;
    Vec3 n;              // world normal pointing to the front region
    Vec3 pLocal, nLocal; // shape-local position/normal (for textures)
    double nFront = 1, nBack = 1;  // refractive indices at the current wavelength
    double lambda = 550;
};

struct ScatterSample {
    Vec3 wi;               // outgoing propagation direction (world)
    double weight = 0;     // f·|cosθ|/pdf, including the radiance n² factor when applicable
    double pdf = 0;        // solid-angle pdf; meaningless for delta events
    bool delta = false;
    bool transmitted = false;
    EventKind event = EventKind::None;
    // Diagnostics for inspection.
    double R = 0, T = 0, cosI = 0, cosT = 0;
};

// Samples the continuation of a path arriving along propagation direction d.
bool sampleScatter(const Interface& it, const Vec3& d, double uc, double u1, double u2, TransportMode mode,
                   ScatterSample& s);
// Evaluates f(wo, wi) for non-delta components; wo = −d. Returns f (without cosine) and pdf of sampling wi.
double evalScatter(const Interface& it, const Vec3& d, const Vec3& wi, TransportMode mode, double& pdf);

// True when a dielectric interface bends different wavelengths differently, i.e. the
// index ratio differs between the hero and another wavelength.
bool isDispersive(const Interface& hero, const Interface& other);
// Throughput multiplier at another wavelength (interface `it`) for the direction the hero
// sampled (`s`). Only valid when the sample can be shared (see isDispersive).
double secondaryScatterWeight(const Interface& it, const Vec3& d, const ScatterSample& s, TransportMode mode);

}  // namespace owe
