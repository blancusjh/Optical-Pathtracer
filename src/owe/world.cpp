#include "world.hpp"

#include <stdexcept>

#include "sampling.hpp"

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
    auto h = [](int x, int y, int z) {
        uint64_t k = hashCombine(hashCombine(uint64_t(uint32_t(x)), uint64_t(uint32_t(y))), uint64_t(uint32_t(z)));
        return double(k >> 11) * (1.0 / 9007199254740992.0);
    };
    double fx = std::floor(p.x), fy = std::floor(p.y), fz = std::floor(p.z);
    int ix = int(fx), iy = int(fy), iz = int(fz);
    double tx = p.x - fx, ty = p.y - fy, tz = p.z - fz;
    auto s = [](double t) { return t * t * (3 - 2 * t); };
    tx = s(tx); ty = s(ty); tz = s(tz);
    double r = 0;
    for (int c = 0; c < 8; ++c) {
        int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
        double w = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
        r += w * h(ix + dx, iy + dy, iz + dz);
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

double Texture::eval(const Vec3& p, const Vec3& n, double l) const {
    switch (kind) {
        case Kind::None: return a.eval(l);
        case Kind::Checker: {
            Vec3 q = p / scale;
            long s = long(std::floor(q.x + 1e-9)) + long(std::floor(q.y + 1e-9)) + long(std::floor(q.z + 1e-9));
            return (s & 1) ? b.eval(l) : a.eval(l);
        }
        case Kind::Noise: {
            double t = fbm3(p / scale, 6);
            t = clampd((t - 0.25) * 2.0, 0, 1);
            return a.eval(l) * (1 - t) + b.eval(l) * t;
        }
        case Kind::Rings: {
            double r = std::sqrt(p.x * p.x + p.z * p.z) / scale + param * fbm3(p / scale, 4);
            double t = 0.5 + 0.5 * std::sin(2 * Pi * r);
            t = t * t;
            return a.eval(l) * (1 - t) + b.eval(l) * t;
        }
        case Kind::Terrain: {
            // Grass on gentle slopes, rock on steep ones, snow above the snow line.
            double jitter = (fbm3(p / scale, 5) - 0.5);
            double slope = 1 - std::abs(n.z);
            double rock = clampd((slope + 0.3 * jitter - 0.25) * 6, 0, 1);
            double base = a.eval(l) * (1 - rock) + b.eval(l) * rock;
            if (param > 0) {
                double snow = clampd((p.z - param + 40 * jitter) / 30.0, 0, 1) * clampd(1 - slope * 1.6, 0, 1);
                base = base * (1 - snow) + c.eval(l) * snow;
            }
            return base;
        }
    }
    return 0;
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
    }
    return 0;
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
    b.ambient = ambient == kOutside ? 0 : ambient;
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
        if (b.emission < 0 || !b.shape->canSample()) continue;
        const Emission& e = emissions_[b.emission];
        double sides = (e.front ? 1 : 0) + (e.back ? 1 : 0);
        double p = std::max(1e-12, e.radiance.luminance()) * b.shape->area() * Pi * sides;
        lightOfBoundary_[i] = int(lights_.size());
        lights_.push_back(LightEntry{LightEntry::Kind::Boundary, uint32_t(i), p});
    }
    if (env.hasSun) {
        double p = std::max(1e-12, env.sunRadiance.luminance()) * env.sunSolidAngle() * Pi * radius * radius;
        lights_.push_back(LightEntry{LightEntry::Kind::Sun, kNone, p});
    }
    double total = 0;
    for (auto& l : lights_) total += l.power;
    totalLightPower_ = total;
    lightCdf_.clear();
    lightPdf_.clear();
    double acc = 0;
    for (auto& l : lights_) {
        lightPdf_.push_back(l.power / total);
        acc += l.power / total;
        lightCdf_.push_back(acc);
    }
    if (!lightCdf_.empty()) lightCdf_.back() = 1.0;
    envPower_ = computeEnvironmentPower();
    built_ = true;
}

double World::computeEnvironmentPower() const {
    if (env.skyModel == Environment::Sky::None) return 0;
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
    bvh_.traverse(ray, 0.0, best, [&](uint32_t i, double& tm) {
        const Boundary& b = boundaries_[i];
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
