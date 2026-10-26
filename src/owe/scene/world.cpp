#include "owe/scene/world.hpp"

#include <stdexcept>

#include "owe/core/sampling.hpp"

namespace owe {

const char* surfaceTypeName(SurfaceType t) {
    switch (t) {
        case SurfaceType::Null: return "null";
        case SurfaceType::Dielectric: return "dielectric";
        case SurfaceType::Diffuse: return "diffuse";
        case SurfaceType::Conductor: return "conductor";
        case SurfaceType::Mirror: return "mirror";
        case SurfaceType::Absorber: return "absorber";
        case SurfaceType::Detector: return "detector";
        case SurfaceType::StainedGlass: return "stained glass";
    }
    return "?";
}

// ---------------------------------------------------------------- textures

void Texture::weights(const Vec3& p, const Vec3& n, double w[3]) const {
    // The spatial pattern is independent of wavelength: it is computed once per hit as mixing
    // weights of the spectra a, b, c.
    Vec3 c(1, 0, 0);
    if (kind == Kind::Image && image) {
        // Centred: an image of the sheet's size fills the sheet (and tiles beyond it).
        if (mapping != Mapping::Triplanar) {
            c = image->bilinear(0.5 + p.x / scale, 0.5 - p.y / scale);
        } else {
            const Vec3 a = triplanarWeights(n);
            c = image->bilinear(0.5 + p.y / scale, 0.5 - p.z / scale) * a.x +
                image->bilinear(0.5 + p.x / scale, 0.5 - p.z / scale) * a.y +
                image->bilinear(0.5 + p.x / scale, 0.5 - p.y / scale) * a.z;
        }
        c = Vec3(saturate(c.x), saturate(c.y), saturate(c.z));
    } else if (kind != Kind::None && kind != Kind::Image) {
        c = proceduralWeights(uint(kind), scale, param, p, n);
    }
    w[0] = c.x;
    w[1] = c.y;
    w[2] = c.z;
}

double Texture::eval(const Vec3& p, const Vec3& n, double l) const {
    double w[3];
    weights(p, n, w);
    return mix(w, l);
}

// ---------------------------------------------------------------- environment

double Environment::sky(const Vec3& dir, double l) const {
    switch (skyModel) {
        case Sky::None: return 0;
        case Sky::Uniform: return zenith.eval(l);
        case Sky::Gradient: {
            double c = dot(dir, up);
            if (c < 0) return ground.eval(l);
            double t = std::pow(c, 0.45);
            return horizon.eval(l) * (1 - t) + zenith.eval(l) * t;
        }
        case Sky::Map: {
            if (!map) return 0;
            const Vec3 c = mapRGB(dir);
            return mapScale * Spectrum::rgbIlluminant(c.x, c.y, c.z).eval(l);
        }
    }
    return 0;
}

void Environment::mapFrame(Vec3& e1, Vec3& e2) const { envMapFrame(normalize(up), e1, e2); }

Vec3 Environment::mapRGB(const Vec3& dir) const {
    Vec3 e1, e2;
    mapFrame(e1, e2);
    const Vec2 uv = envMapUV(dir, up, e1, e2, mapRotation);
    return map->bilinear(uv.x, uv.y);
}

void Environment::prepareMap() {
    mapDist = {};
    if (skyModel != Sky::Map || !map) return;
    const int W = map->width, H = map->height;
    mapDist.W = W;
    mapDist.H = H;
    mapDist.conditional.assign(size_t(H) * (W + 1), 0.0);
    mapDist.marginal.assign(size_t(H) + 1, 0.0);
    mapDist.rowWeight.assign(size_t(H), 0.0);
    double sum = 0;
    for (int j = 0; j < H; ++j) {
        const double sinT = std::sin(Pi * (j + 0.5) / H);
        double* cdf = &mapDist.conditional[size_t(j) * (W + 1)];
        for (int i = 0; i < W; ++i) {
            const Vec3 c = map->texel(i, j);
            const double y = std::max(0.0, luminanceRec709(c.x, c.y, c.z));
            cdf[i + 1] = cdf[i] + y * sinT;
        }
        mapDist.rowWeight[size_t(j)] = cdf[W];
        sum += cdf[W];
    }
    mapDist.luminance = sum * (2 * Pi / W) * (Pi / H);  // texel solid angle = (2π/W)(π/H) sin θ
    // A floor of 0.1% of the mean keeps every direction with some radiance sampleable.
    const double floorW = 1e-3 * sum / (double(W) * H) + 1e-300;
    sum = 0;
    for (int j = 0; j < H; ++j) {
        const double sinT = std::sin(Pi * (j + 0.5) / H);
        double* cdf = &mapDist.conditional[size_t(j) * (W + 1)];
        double acc = 0;
        for (int i = 0; i < W; ++i) {
            acc += (cdf[i + 1] - cdf[i]) + floorW * sinT;
            cdf[i + 1] = acc;
        }
        cdf[0] = 0;
        mapDist.rowWeight[size_t(j)] = acc;
        sum += acc;
        mapDist.marginal[size_t(j) + 1] = sum;
    }
    mapDist.total = sum;
}

bool Environment::sampleMap(double u1, double u2, Vec3& dir, double& pdf) const {
    const auto& D = mapDist;
    if (!(D.total > 0)) return false;
    // Row by the marginal, column by the row's conditional, uniform within the texel.
    const double a = u1 * D.total;
    int j = int(std::upper_bound(D.marginal.begin(), D.marginal.end(), a) - D.marginal.begin()) - 1;
    j = std::clamp(j, 0, D.H - 1);
    const double rowW = D.rowWeight[size_t(j)];
    const double* cdf = &D.conditional[size_t(j) * (D.W + 1)];
    const double tRow = (a - D.marginal[size_t(j)]) / std::max(D.marginal[size_t(j) + 1] - D.marginal[size_t(j)], 1e-300);
    const double b = u2 * rowW;
    int i = int(std::upper_bound(cdf, cdf + D.W + 1, b) - cdf) - 1;
    i = std::clamp(i, 0, D.W - 1);
    const double tCol = (b - cdf[i]) / std::max(cdf[i + 1] - cdf[i], 1e-300);
    Vec3 e1, e2;
    mapFrame(e1, e2);
    double st;
    dir = envMapDirection((i + clamp(tCol, 0.0, 1.0)) / D.W, (j + clamp(tRow, 0.0, 1.0)) / D.H, normalize(up), e1, e2,
                          mapRotation, st);
    if (!(st > 0)) return false;
    pdf = envMapPdf((cdf[i + 1] - cdf[i]) / D.total, D.W, D.H, st);
    return pdf > 0;
}

double Environment::mapPdf(const Vec3& dir) const {
    const auto& D = mapDist;
    if (!(D.total > 0)) return 0;
    Vec3 e1, e2;
    mapFrame(e1, e2);
    const double c = clamp(dot(dir, up), -1.0, 1.0);
    const double st = std::sqrt(std::max(0.0, 1 - c * c));
    if (!(st > 0)) return 0;
    const Vec2 uv = envMapUV(dir, up, e1, e2, mapRotation);
    const int i = std::clamp(int(uv.x * D.W), 0, D.W - 1), j = std::clamp(int(uv.y * D.H), 0, D.H - 1);
    const double* cdf = &D.conditional[size_t(j) * (D.W + 1)];
    return envMapPdf((cdf[i + 1] - cdf[i]) / D.total, D.W, D.H, st);
}

// ---------------------------------------------------------------- world

World::World() {
    Medium air;
    catalogMedium("air", air);
    media_.push_back(air);
    regions_.push_back(Region{"ambient", 0, -1});
}

uint32_t World::addMedium(const Medium& m) {
    for (size_t i = 0; i < media_.size(); ++i)
        if (media_[i].name == m.name) { media_[i] = m; return uint32_t(i); }
    media_.push_back(m);
    return uint32_t(media_.size() - 1);
}

uint32_t World::medium(const std::string& name) {
    for (size_t i = 0; i < media_.size(); ++i)
        if (media_[i].name == name) return uint32_t(i);
    Medium m;
    if (!catalogMedium(name, m)) throw std::runtime_error("unknown medium: " + name);
    media_.push_back(m);
    return uint32_t(media_.size() - 1);
}

uint32_t World::addRegion(const std::string& name, uint32_t medium, int body) {
    if (medium >= media_.size()) throw std::runtime_error("addRegion: bad medium index");
    regions_.push_back(Region{name, medium, body});
    if (body >= 0) bodies_[body].regions.push_back(uint32_t(regions_.size() - 1));
    return uint32_t(regions_.size() - 1);
}

uint32_t World::addOptics(const SurfaceOptics& o) {
    optics_.push_back(o);
    return uint32_t(optics_.size() - 1);
}

uint32_t World::anonymousOptics(SurfaceType type, double roughness) {
    for (size_t i = 0; i < optics_.size(); ++i)
        if (optics_[i].type == type && optics_[i].roughness == roughness && optics_[i].name.empty()) return uint32_t(i);
    SurfaceOptics o;
    o.type = type;
    o.roughness = roughness;
    return addOptics(o);
}

int World::addEmission(const Emission& e) {
    emissions_.push_back(e);
    return int(emissions_.size() - 1);
}

int World::addAssembly(const std::string& name, int parent, const Transform& xf) {
    assemblies_.push_back(Assembly{name, parent, xf});
    return int(assemblies_.size() - 1);
}

int World::addBody(const std::string& name, const std::string& kind, int assembly, const Transform& xf, uint32_t ambient) {
    Body b;
    b.name = name;
    b.kind = kind;
    b.assembly = assembly;
    b.xf = xf;
    b.ambient = ambient != kOutside ? ambient : immersion_ != kOutside ? immersion_ : 0;
    bodies_.push_back(b);
    return int(bodies_.size() - 1);
}

uint32_t World::addBoundary(int body, const std::string& name, std::shared_ptr<const Shape> shape, const Transform& local,
                            uint32_t front, uint32_t back, uint32_t optics, int emission) {
    uint32_t amb = body >= 0 ? bodies_[body].ambient : 0;
    Boundary b;
    b.name = name;
    b.shape = std::move(shape);
    b.local = local;
    b.front = front == kOutside ? amb : front;
    b.back = back == kOutside ? amb : back;
    b.optics = optics;
    b.emission = emission;
    b.body = body;
    boundaries_.push_back(b);
    if (body >= 0) bodies_[body].boundaries.push_back(uint32_t(boundaries_.size() - 1));
    return uint32_t(boundaries_.size() - 1);
}

Transform World::assemblyToWorld(int a) const {
    Transform xf;
    int guard = 0;
    while (a >= 0) {
        if (++guard > 1000) throw std::runtime_error("assembly hierarchy contains a cycle");
        xf = assemblies_[a].xf * xf;
        a = assemblies_[a].parent;
    }
    return xf;
}

Transform World::bodyToWorld(int body) const {
    if (body < 0) return Transform{};
    return assemblyToWorld(bodies_[body].assembly) * bodies_[body].xf;
}

void World::build() {
    std::vector<AABB> boxes(boundaries_.size());
    bounds_ = AABB{};
    for (size_t i = 0; i < boundaries_.size(); ++i) {
        Boundary& b = boundaries_[i];
        b.toWorld = bodyToWorld(b.body) * b.local;
        b.toLocal = b.toWorld.inverse();
        b.worldBox = b.shape->bounds().transformed(b.toWorld);
        boxes[i] = b.worldBox;
        bounds_.expand(b.worldBox);
        if (b.optics >= optics_.size()) throw std::runtime_error("boundary '" + b.name + "' has no optics");
        const SurfaceOptics& o = optics_[b.optics];
        if (o.type == SurfaceType::Dielectric || o.type == SurfaceType::Null) {
            if (media_[regions_[b.front].medium].opaque || media_[regions_[b.back].medium].opaque)
                throw std::runtime_error("boundary '" + b.name + "' is transmissive but borders an opaque medium");
        }
    }
    bvh_.build(boxes, 2);
    buildLightTable();
}

// Every light next-event estimation draws from, with its selection probability.
void World::buildLightTable() {
    lights_.clear();
    lightOfBoundary_.assign(boundaries_.size(), -1);
    double radius = sceneRadius();
    for (size_t i = 0; i < boundaries_.size(); ++i) {
        const Boundary& b = boundaries_[i];
        if (b.emission < 0 || !b.shape->canSample() || !emissions_[b.emission].nee) continue;
        const Emission& e = emissions_[b.emission];
        double sides = (e.front ? 1 : 0) + (e.back ? 1 : 0);
        double p = std::max(1e-12, e.radiance.luminance()) * b.shape->area() * Pi * sides;
        lightOfBoundary_[i] = int(lights_.size());
        lights_.push_back(LightEntry{LightEntry::Kind::Boundary, uint32_t(i), p});
    }
    env.prepareMap();
    envPower_ = computeEnvironmentPower();
    environmentLight_ = -1;
    if (env.skyModel == Environment::Sky::Map && envPower_ > 0) {
        environmentLight_ = int(lights_.size());
        lights_.push_back(LightEntry{LightEntry::Kind::Environment, kNone, envPower_});
    }
    if (env.hasSun) {
        double p = std::max(1e-12, env.sunRadiance.luminance()) * env.sunSolidAngle() * Pi * radius * radius;
        lights_.push_back(LightEntry{LightEntry::Kind::Sun, kNone, p});
    }
    double total = 0, boundaryPower = 0, nearbyPower = 0, distantPower = 0;
    for (auto& l : lights_) {
        if (l.kind == LightEntry::Kind::Environment) continue;  // selected by its own share below
        total += l.power;
        if (l.kind == LightEntry::Kind::Boundary) {
            boundaryPower += l.power;
            const auto& e = emissions_[boundaries_[l.boundary].emission];
            (e.distant ? distantPower : nearbyPower) += l.power;
        }
    }
    totalLightPower_ = total;
    lightCdf_.clear();
    lightPdf_.clear();
    double share = env.hasSun && boundaryPower > 0 ? clamp(env.sunNeeShare, 0, 0.999) : 0;
    // A star's physical area at astronomical distance must not starve nearby lamps.
    // Sample both groups with full support; MIS uses this same selection probability.
    if (distantPower > 0 && env.hasSun && share == 0) share = 0.5;
    // The environment map takes its share of the combined power; the other lights share the rest
    // as they would without it.
    const double envShare = environmentLight_ >= 0 ? envPower_ / (envPower_ + total) : 0;
    double acc = 0;
    for (auto& l : lights_) {
        if (l.kind == LightEntry::Kind::Environment) {
            lightPdf_.push_back(envShare);
            acc += envShare;
            lightCdf_.push_back(acc);
            continue;
        }
        double p = total > 0 ? l.power / total : 0;
        if (share > 0) p = l.kind == LightEntry::Kind::Sun ? share : (1 - share) * l.power / boundaryPower;
        if (distantPower > 0 && l.kind == LightEntry::Kind::Boundary) {
            bool distant = emissions_[boundaries_[l.boundary].emission].distant;
            double groupShare = nearbyPower > 0 ? 0.5 : 1;
            p = (1 - share) * groupShare * l.power / (distant ? distantPower : nearbyPower);
        }
        p *= 1 - envShare;
        lightPdf_.push_back(p);
        acc += p;
        lightCdf_.push_back(acc);
    }
    if (!lightCdf_.empty()) lightCdf_.back() = 1.0;
}

double World::computeEnvironmentPower() const {
    if (env.skyModel == Environment::Sky::None) return 0;
    if (env.skyModel == Environment::Sky::Map) {
        const double R = sceneRadius();  // ∫ L_Y dω over the map, times π R²
        return env.mapScale * env.mapDist.luminance * Pi * R * R;
    }
    // Average luminance over the sphere of directions, times π R² · 4π.
    double sum = 0;
    int n = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 32; ++j) {
            double z = 1 - 2 * (i + 0.5) / 16, r = safeSqrt(1 - z * z), phi = 2 * Pi * (j + 0.5) / 32;
            Vec3 d{r * std::cos(phi), r * std::sin(phi), z};
            double y = 0;
            for (double l = 400; l <= 700; l += 50) y += env.sky(d, l);
            sum += y / 7;
            ++n;
        }
    double R = sceneRadius();
    return std::max(0.0, sum / n) * Pi * R * R * 4 * Pi;
}

size_t World::pickLight(double u) const {
    size_t i = size_t(std::upper_bound(lightCdf_.begin(), lightCdf_.end(), u) - lightCdf_.begin());
    return std::min(i, lights_.size() - 1);
}

double World::sceneRadius() const {
    if (!bounds_.valid()) return 1.0;
    return std::max(1e-3, 0.5 * length(bounds_.extent()));
}

namespace {
// Twofold doubles retain the low part of world-origin minus planet-centre subtraction. This
// uses IEEE double and fma rather than long double, which is not wider on every supported host.
struct Twofold { double hi, lo; };
Twofold add(Twofold a, Twofold b) {
    const double s = a.hi + b.hi, v = s - a.hi;
    const double e = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    const double h = s + e;
    return {h, e - (h - s)};
}
Twofold mul(Twofold a, Twofold b) {
    const double p = a.hi * b.hi;
    const double e = std::fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi + a.lo * b.lo;
    const double h = p + e;
    return {h, e - (h - p)};
}
double value(Twofold a) { return a.hi + a.lo; }

// An independent compensated quadratic for near-surface spheres. The shape and its native
// transform remain unchanged; coefficients are formed BEFORE subtraction is rounded to double.
bool intersectNearSphere(const Boundary& b, const SphereShape& sphere, const Ray& ray, double tmin, double tmax,
                         LocalHit& h) {
    Twofold A{0,0}, B{0,0}, C{0,0};
    const double radius = sphere.radius();
    Twofold rel[3];
    for (int k = 0; k < 3; ++k) {
        rel[k] = add({ray.o[k], 0}, {-b.toWorld.t[k], 0});
        A = add(A, mul({ray.d[k], 0}, {ray.d[k], 0}));
        B = add(B, mul(rel[k], {ray.d[k], 0}));
        C = add(C, mul(rel[k], rel[k]));
    }
    const Twofold r2 = mul({radius, 0}, {radius, 0});
    C = add(C, {-r2.hi, -r2.lo});
    const Twofold ac = mul(A, C);
    const double disc = value(add(mul(B, B), {-ac.hi, -ac.lo}));
    if (disc < 0) return false;
    const double bv = value(B), av = value(A);
    const double q = -bv - std::copysign(std::sqrt(disc), bv);
    double t0 = q / av, t1 = q != 0 ? value(C) / q : t0;
    if (t0 > t1) std::swap(t0, t1);
    const double t = t0 > tmin && t0 < tmax ? t0 : t1;
    if (!(t > tmin && t < tmax)) return false;
    const Vec3 p = ray.at(t);
    Vec3 n;
    for (int k = 0; k < 3; ++k) n[k] = value(add(rel[k], mul({ray.d[k], 0}, {t, 0}))) / radius;
    h.t = t;
    h.p = b.toLocal.point(p);
    h.n = normalize(b.toLocal.vector(n));
    h.prim = 0;
    return true;
}
bool needsNearSphere(const Boundary& b, const Ray& ray) {
    const auto* sphere = dynamic_cast<const SphereShape*>(b.shape.get());
    return sphere && sphere->radius() > 4096 * std::max(1e-9, std::abs(length(ray.o - b.toWorld.t) - sphere->radius()));
}
}  // namespace

bool World::intersect(const Ray& ray, double tmax, SurfaceHit& hit) const {
    bool found = false;
    double best = tmax;
    LocalHit lh, bestLocal;
    uint32_t bestB = kNone;
    bool bestNearSphere = false;
    bool bestMatches = false, ambiguous = false;
    double bestError = 0;
    const RaySlabs rs(ray);
    // Transform arithmetic bounds the local uncertainty, not the old transport offset.
    // Keep this interval distinct from a physical gap: positive roots are never clamped away.
    const double worldScale = std::max(1e-12, maxAbsComponent(ray.o));
    const double broadError = std::max(ray.departureError, contactError(std::max(worldScale, maxAbsComponent(bounds_.extent()))));
    bvh_.traverse(ray, -broadError, best, [&](uint32_t i, double& tm) {
        const Boundary& b = boundaries_[i];
        const bool nearSphere = needsNearSphere(b, ray);
        Ray lr{b.toLocal.point(ray.o), b.toLocal.vector(ray.d)};
        const double scale = nearSphere ? worldScale : std::max({worldScale, maxAbsComponent(lr.o),
            maxAbsComponent(b.toWorld.t), maxAbsComponent(b.shape->bounds().extent())});
        const double anchorError = std::max(ray.departureError,
            contactError(scale));
        double error = anchorError;
        const auto* mesh = dynamic_cast<const MeshShape*>(b.shape.get());
        const bool knownMeshDeparture = mesh && i == ray.departedBoundary &&
                                        ray.departedPrimitive < mesh->triangleCount();
        if (i == ray.departedBoundary) {
            lr.departedPrimitive = ray.departedPrimitive;
            lr.departedNormal = b.toLocal.vector(ray.departedNormal);
        }
        if (knownMeshDeparture) {
            const auto& tri = mesh->triangles()[ray.departedPrimitive];
            const auto& positions = mesh->positions();
            const Vec3 normal = normalize(cross(positions[tri[1]] - positions[tri[0]],
                                               positions[tri[2]] - positions[tri[0]]));
            lr.o -= normal * dot(lr.o - positions[tri[0]], normal);
        }
        lr.departureError = error;
        const bool spawned = ray.departedBoundary != kNone;
        // The native departing plane is known. Its past faces cannot be another
        // contact, even when their distance is smaller than the anchor error.
        double tmin = ray.forwardOnly || knownMeshDeparture ? std::nextafter(0.0, -Inf) : (spawned ? -error : 0.0);
        double tEnter;
        if (!rs.hit(b.worldBox, tmin, tm, tEnter)) return;
        auto solve = [&](double lo) {
            lh.error = 0;
            const bool solved = nearSphere ? intersectNearSphere(b, static_cast<const SphereShape&>(*b.shape), ray, lo, tm, lh)
                                          : b.shape->intersect(lr, lo, tm, lh);
            error = anchorError + lh.error;
            return solved;
        };
        if (!solve(tmin)) return;
        Vec3 n = b.toWorld.vector(lh.n);
        uint32_t incident = dot(ray.d, n) < 0 ? b.front : b.back;
        bool matches = ray.regionHint == kNone || incident == ray.regionHint;
        // At a contact already crossed in the outgoing direction, the opposite region is
        // no longer incident. Retain the body's later root rather than excluding the body.
        auto excludedRoot = [&] {
            // A grazing ray amplifies a small anchor displacement into a larger
            // root distance. Compare the displacement normal to the surface.
            const bool ownNear = std::abs(lh.t * dot(ray.d, normalize(n))) <= error;
            return spawned && ((std::abs(lh.t) <= error && !matches) ||
                (ownNear && !mesh && i == ray.departedBoundary &&
                 (b.front == b.back || lengthSq(ray.departedNormal) > 0) &&
                 dot(ray.d, n) * dot(ray.d, ray.departedNormal) >= 0));
        };
        bool unresolvedRoot = false;
        for (int retry = 0; excludedRoot(); ++retry) {
            if (retry == 8) { unresolvedRoot = true; break; }
            // Advance past only this root. A positive opposite face can be much
            // closer than the uncertainty of the incoming surface anchor.
            if (!solve(std::max(0.0, lh.t))) return;
            n = b.toWorld.vector(lh.n);
            incident = dot(ray.d, n) < 0 ? b.front : b.back;
            matches = ray.regionHint == kNone || incident == ray.regionHint;
        }
        const double t = std::max(0.0, lh.t);
        const bool tie = found && std::abs(t - bestLocal.t) <= std::max(error, bestError);
        const bool readout = optics_[b.optics].type == SurfaceType::Detector && b.front == b.back;
        const bool bestReadout = found && optics_[boundaries_[bestB].optics].type == SurfaceType::Detector &&
                                 boundaries_[bestB].front == boundaries_[bestB].back;
        if (tie && matches && bestMatches && readout != bestReadout) {
            if (readout) return;  // physical matter precedes a coincident zero-thickness readout
            ambiguous = false;
        } else if (tie && ray.regionHint != kNone && matches && bestMatches && i != bestB) ambiguous = true;
        if (tie && bestMatches && !matches) return;
        if (!tie) ambiguous = false;
        bool replace = !found || !tie;
        if (found && tie) {
            if (matches != bestMatches) replace = matches;
            else if (readout != bestReadout) replace = bestReadout;
            else replace = i < bestB;
        }
        if (replace) {
            bestLocal = lh;
            bestLocal.t = t;
            bestB = i;
            bestNearSphere = nearSphere;
            bestMatches = matches;
            bestError = error;
            found = true;
            ambiguous = ambiguous || unresolvedRoot;
        }
        // Leave a small window open so a coincident boundary with correct incidence can win.
        best = bestLocal.t;
        tm = std::min(tmax, best + std::max(error, bestError));
    });
    if (!found) return false;
    const Boundary& b = boundaries_[bestB];
    hit.t = bestLocal.t;
    hit.pLocal = bestLocal.p;
    hit.nLocal = bestLocal.n;
    hit.p = bestNearSphere ? ray.at(hit.t) : b.toWorld.point(bestLocal.p);
    hit.n = normalize(b.toWorld.vector(bestLocal.n));
    hit.boundary = bestB;
    hit.prim = bestLocal.prim;
    hit.error = bestError;
    hit.ambiguousContact = ambiguous;
    return true;
}

uint32_t World::locate(const Vec3& p) const {
    static const Vec3 dirs[5] = {normalize(Vec3(0.2113, 0.5774, 0.7887)), normalize(Vec3(-0.6, 0.3, -0.74)),
                                 normalize(Vec3(0.41, -0.83, 0.37)), normalize(Vec3(-0.12, -0.33, 0.93)),
                                 normalize(Vec3(0.91, 0.05, -0.41))};
    std::map<uint32_t, int> votes;
    for (const Vec3& d : dirs) {
        SurfaceHit h;
        uint32_t r = ambientRegion();
        if (intersect(Ray{p, d}, Inf, h)) {
            const Boundary& b = boundaries_[h.boundary];
            r = dot(d, h.n) < 0 ? b.front : b.back;
        }
        votes[r]++;
    }
    uint32_t best = ambientRegion();
    int bestVotes = -1;
    for (auto& [r, v] : votes)
        if (v > bestVotes) { best = r; bestVotes = v; }
    return best;
}

int World::findBody(const std::string& name) const {
    for (size_t i = 0; i < bodies_.size(); ++i)
        if (bodies_[i].name == name) return int(i);
    return -1;
}
int World::findAssembly(const std::string& name) const {
    for (size_t i = 0; i < assemblies_.size(); ++i)
        if (assemblies_[i].name == name) return int(i);
    return -1;
}
std::string World::boundaryLabel(uint32_t b) const {
    if (b >= boundaries_.size()) return "(none)";
    const Boundary& bd = boundaries_[b];
    std::string s = bd.body >= 0 ? bodies_[bd.body].name + "." : std::string();
    return s + bd.name;
}

std::string World::regionLabel(uint32_t r) const {
    if (r >= regions_.size()) return "(none)";
    return regions_[r].name + "[" + media_[regions_[r].medium].name + "]";
}

}  // namespace owe
