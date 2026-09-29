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
    }
    return "?";
}

// ---------------------------------------------------------------- textures

double valueNoise3(const Vec3& p) {
    double fx = std::floor(p.x), fy = std::floor(p.y), fz = std::floor(p.z);
    int ix = int(fx), iy = int(fy), iz = int(fz);
    double tx = p.x - fx, ty = p.y - fy, tz = p.z - fz;
    auto s = [](double t) { return t * t * (3 - 2 * t); };
    tx = s(tx); ty = s(ty); tz = s(tz);
    // Lattice value h(x,y,z) = hashCombine(hashCombine(x, y), z). hashCombine(a, b) mixes b on its
    // own first, so the y and z mixes are shared between corners (16 mixes instead of 32).
    constexpr uint64_t C = 0x9e3779b97f4a7c15ULL;
    uint64_t my[2], mz[2];
    for (int k = 0; k < 2; ++k) {
        my[k] = mixBits(uint64_t(uint32_t(iy + k))) + C;
        mz[k] = mixBits(uint64_t(uint32_t(iz + k))) + C;
    }
    double r = 0;
    for (int c = 0; c < 8; ++c) {
        int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
        double w = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
        uint64_t k = mixBits(mixBits(uint64_t(uint32_t(ix + dx)) ^ my[dy]) ^ mz[dz]);
        r += w * (double(k >> 11) * (1.0 / 9007199254740992.0));
    }
    return r;
}

double fbm3(const Vec3& p, int octaves) {
    double sum = 0, amp = 0.5, norm = 0;
    Vec3 q = p;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * valueNoise3(q);
        norm += amp;
        amp *= 0.5;
        q = q * 2.03 + Vec3(17.1, 5.3, 9.7);
    }
    return sum / norm;
}

void Texture::weights(const Vec3& p, const Vec3& n, double w[3]) const {
    // The spatial pattern is independent of wavelength: it is computed once per hit as mixing
    // weights of the spectra a, b, c.
    w[0] = 1; w[1] = 0; w[2] = 0;
    switch (kind) {
        case Kind::None: return;
        case Kind::Image: {
            if (!image) return;
            Vec3 c;
            // Centred: an image of the sheet's size fills the sheet (and tiles beyond it).
            if (mapping != Mapping::Triplanar) {
                c = image->bilinear(0.5 + p.x / scale, 0.5 - p.y / scale);
            } else {
                // Triplanar: projections along x, y and z blended by the normal's fourth powers.
                Vec3 a{std::pow(std::abs(n.x), 4), std::pow(std::abs(n.y), 4), std::pow(std::abs(n.z), 4)};
                const double sum = a.x + a.y + a.z;
                a = a / (sum > 0 ? sum : 1);
                c = image->bilinear(0.5 + p.y / scale, 0.5 - p.z / scale) * a.x +
                    image->bilinear(0.5 + p.x / scale, 0.5 - p.z / scale) * a.y +
                    image->bilinear(0.5 + p.x / scale, 0.5 - p.y / scale) * a.z;
            }
            w[0] = std::clamp(c.x, 0.0, 1.0);
            w[1] = std::clamp(c.y, 0.0, 1.0);
            w[2] = std::clamp(c.z, 0.0, 1.0);
            return;
        }
        case Kind::Checker: {
            Vec3 q = p / scale;
            long s = long(std::floor(q.x + 1e-9)) + long(std::floor(q.y + 1e-9)) + long(std::floor(q.z + 1e-9));
            if (s & 1) { w[0] = 0; w[1] = 1; }
            return;
        }
        case Kind::Noise: {
            double t = clampd((fbm3(p / scale, 6) - 0.25) * 2.0, 0, 1);
            w[0] = 1 - t; w[1] = t;
            return;
        }
        case Kind::Rings: {
            double r = std::sqrt(p.x * p.x + p.z * p.z) / scale + param * fbm3(p / scale, 4);
            double t = 0.5 + 0.5 * std::sin(2 * Pi * r);
            t = t * t;
            w[0] = 1 - t; w[1] = t;
            return;
        }
        case Kind::Bands: {
            double z = p.z / scale + param * (fbm3(Vec3(p.x, p.y, p.z * 4) / scale, 5) - 0.5);
            double t = 0.5 + 0.5 * std::sin(2 * Pi * z) * (0.6 + 0.4 * std::sin(2 * Pi * z * 0.37 + 1.3));
            double mix = clampd(fbm3(Vec3(0, 0, z * 3.1), 3) * 1.4 - 0.2, 0, 1);
            w[0] = (1 - t) * (1 - 0.35 * mix); w[1] = t * (1 - 0.35 * mix); w[2] = 0.35 * mix;
            return;
        }
        case Kind::Radial: {
            double r = std::sqrt(p.x * p.x + p.y * p.y) / scale;
            double t = clampd(fbm3(Vec3(r * 3.0, 0.5, 0.5), 6) * 1.6 - 0.3, 0, 1);
            w[0] = 1 - t; w[1] = t;
            return;
        }
        case Kind::Marble: {
            Vec3 q = p / scale;
            double v = std::abs(std::sin(q.x * 0.9 + q.y * 0.5 + q.z * 0.3 + param * 6 * fbm3(q * 0.7, 6)));
            double t = std::pow(1 - v, 6);  // thin dark veins
            double base = 0.94 + 0.06 * fbm3(q * 3.0, 4);
            w[0] = base * (1 - t); w[1] = t;
            return;
        }
        case Kind::Terrain: {
            // Grass on gentle slopes, rock on steep ones, snow above the snow line.
            double jitter = (fbm3(p / scale, 5) - 0.5);
            double slope = 1 - std::abs(n.z);
            double rock = clampd((slope + 0.3 * jitter - 0.25) * 6, 0, 1);
            double snow = 0;
            if (param > 0) snow = clampd((p.z - param + 40 * jitter) / 30.0, 0, 1) * clampd(1 - slope * 1.6, 0, 1);
            w[0] = (1 - rock) * (1 - snow); w[1] = rock * (1 - snow); w[2] = snow;
            return;
        }
    }
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
            double w[3], br, bg, bb;
            rgbToBasisWeights(c.x, c.y, c.z, w);
            rgbBasis(l, br, bg, bb);
            static const double norm = rgbIlluminantNorm();
            return mapScale * norm * (std::max(0.0, w[0]) * br + std::max(0.0, w[1]) * bg + std::max(0.0, w[2]) * bb) *
                   planck(l, 6504.0) * 1e-13;
        }
    }
    return 0;
}

void Environment::mapFrame(Vec3& e1, Vec3& e2) const {
    Vec3 u = normalize(up);
    e1 = std::abs(u.x) < 0.9 ? normalize(Vec3(1, 0, 0) - u * u.x) : normalize(Vec3(0, 1, 0) - u * u.y);
    e2 = cross(u, e1);
}

Vec3 Environment::mapRGB(const Vec3& dir) const {
    Vec3 e1, e2;
    mapFrame(e1, e2);
    const double c = clampd(dot(dir, up), -1.0, 1.0);
    const double phi = std::atan2(dot(dir, e2), dot(dir, e1)) - mapRotation;
    return map->bilinear(phi / (2 * Pi), std::acos(c) / Pi);
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
            const double y = std::max(0.0, 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z);
            cdf[i + 1] = cdf[i] + y * sinT;
        }
        mapDist.rowWeight[size_t(j)] = cdf[W];
        sum += cdf[W];
    }
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
    const double u = (i + clampd(tCol, 0.0, 1.0)) / D.W, v = (j + clampd(tRow, 0.0, 1.0)) / D.H;
    const double theta = Pi * v, phi = 2 * Pi * u + mapRotation;
    Vec3 e1, e2;
    mapFrame(e1, e2);
    const double st = std::sin(theta);
    dir = (e1 * std::cos(phi) + e2 * std::sin(phi)) * st + normalize(up) * std::cos(theta);
    if (!(st > 0)) return false;
    const double pCell = (cdf[i + 1] - cdf[i]) / D.total;
    pdf = pCell * D.W * D.H / (2 * Pi * Pi * st);
    return pdf > 0;
}

double Environment::mapPdf(const Vec3& dir) const {
    const auto& D = mapDist;
    if (!(D.total > 0)) return 0;
    Vec3 e1, e2;
    mapFrame(e1, e2);
    const double c = clampd(dot(dir, up), -1.0, 1.0);
    const double st = std::sqrt(std::max(0.0, 1 - c * c));
    if (!(st > 0)) return 0;
    double u = (std::atan2(dot(dir, e2), dot(dir, e1)) - mapRotation) / (2 * Pi);
    u -= std::floor(u);
    const double v = std::acos(c) / Pi;
    const int i = std::clamp(int(u * D.W), 0, D.W - 1), j = std::clamp(int(v * D.H), 0, D.H - 1);
    const double* cdf = &D.conditional[size_t(j) * (D.W + 1)];
    return (cdf[i + 1] - cdf[i]) / D.total * D.W * D.H / (2 * Pi * Pi * st);
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

uint32_t World::dielectricOptics(double roughness) {
    for (size_t i = 0; i < optics_.size(); ++i)
        if (optics_[i].type == SurfaceType::Dielectric && optics_[i].roughness == roughness && optics_[i].name.empty())
            return uint32_t(i);
    SurfaceOptics o;
    o.type = SurfaceType::Dielectric;
    o.roughness = roughness;
    return addOptics(o);
}

uint32_t World::absorberOptics() {
    for (size_t i = 0; i < optics_.size(); ++i)
        if (optics_[i].type == SurfaceType::Absorber && optics_[i].name.empty()) return uint32_t(i);
    SurfaceOptics o;
    o.type = SurfaceType::Absorber;
    return addOptics(o);
}

uint32_t World::nullOptics() {
    for (size_t i = 0; i < optics_.size(); ++i)
        if (optics_[i].type == SurfaceType::Null && optics_[i].name.empty()) return uint32_t(i);
    SurfaceOptics o;
    o.type = SurfaceType::Null;
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

    // Lights.
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
    double share = env.hasSun && boundaryPower > 0 ? clampd(env.sunNeeShare, 0, 0.999) : 0;
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
    built_ = true;
}

double World::computeEnvironmentPower() const {
    if (env.skyModel == Environment::Sky::None) return 0;
    if (env.skyModel == Environment::Sky::Map) {
        // ∫ L_Y dω over the map (texel luminance × solid angle), times π R².
        if (!env.map) return 0;
        const int W = env.map->width, H = env.map->height;
        double sum = 0;
        static const double norm = rgbIlluminantNorm();
        for (int j = 0; j < H; ++j) {
            const double dOmega = (2 * Pi / W) * (Pi / H) * std::sin(Pi * (j + 0.5) / H);
            for (int i = 0; i < W; ++i) {
                const Vec3 c = env.map->texel(i, j);
                sum += std::max(0.0, 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z) * dOmega;
            }
        }
        (void)norm;
        const double R = sceneRadius();
        return env.mapScale * sum * Pi * R * R;
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

bool World::intersect(const Ray& ray, double tmax, SurfaceHit& hit) const {
    bool found = false;
    double best = tmax;
    LocalHit lh, bestLocal;
    uint32_t bestB = kNone;
    const RaySlabs rs(ray);
    bvh_.traverse(ray, 0.0, best, [&](uint32_t i, double& tm) {
        const Boundary& b = boundaries_[i];
        // Leaves hold up to 16 overlapping boundaries (lens surfaces inside a tube); their own
        // boxes reject most before the ray is carried into their frames.
        double tEnter;
        if (!rs.hit(b.worldBox, 0.0, tm, tEnter)) return;
        Ray lr{b.toLocal.point(ray.o), b.toLocal.vector(ray.d)};
        if (b.shape->intersect(lr, 0.0, tm, lh)) {
            tm = lh.t;
            bestLocal = lh;
            bestB = i;
            found = true;
        }
    });
    if (!found) return false;
    const Boundary& b = boundaries_[bestB];
    hit.t = bestLocal.t;
    hit.pLocal = bestLocal.p;
    hit.nLocal = bestLocal.n;
    hit.p = b.toWorld.point(bestLocal.p);
    hit.n = normalize(b.toWorld.vector(bestLocal.n));
    hit.boundary = bestB;
    hit.prim = bestLocal.prim;
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
int World::findRegion(const std::string& name) const {
    for (size_t i = 0; i < regions_.size(); ++i)
        if (regions_[i].name == name) return int(i);
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
