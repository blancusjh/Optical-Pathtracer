#include "builders.hpp"

#include <sstream>
#include <stdexcept>

#include "sampling.hpp"

namespace owe {

namespace {
Transform flipZ() { return Transform::rotate({1, 0, 0}, Pi); }  // maps +z to -z
std::string idx(const std::string& base, size_t i) { return base + std::to_string(i); }

uint32_t interiorRegion(World& w, int body, const std::string& name, const BodyMaterial& m) {
    return w.addRegion(name, w.medium(m.transparent ? m.medium : "opaque"), body);
}
}  // namespace

// ---------------------------------------------------------------- lenses

int buildLens(World& w, const std::string& name, const LensSpec& s, int assembly, const Transform& xf,
              uint32_t ambient) {
    size_t n = s.media.size();
    if (n == 0 || s.surfaces.size() != n + 1 || s.thickness.size() != n)
        throw std::runtime_error("lens '" + name + "': need n media, n thicknesses and n+1 surfaces");
    int body = w.addBody(name, "lens", assembly, xf, ambient);
    double edge = s.edgeRadius;
    for (auto& sf : s.surfaces) {
        if (!(sf.semiDiameter > 0)) throw std::runtime_error("lens '" + name + "': surface without semi-diameter");
        edge = std::max(edge, sf.semiDiameter);
    }
    std::vector<uint32_t> regions;
    for (size_t i = 0; i < n; ++i) regions.push_back(w.addRegion(name + ".glass" + std::to_string(i), w.medium(s.media[i]), body));
    uint32_t dielectric = w.dielectricOptics(0);
    uint32_t rimOptics = s.rim == LensSpec::Rim::Black ? w.absorberOptics()
                         : s.rim == LensSpec::Rim::Polished ? dielectric
                                                            : w.dielectricOptics(0.35);
    // Vertex positions and edge z of each surface.
    std::vector<double> zv(n + 1, 0.0), zEdge(n + 1);
    for (size_t i = 0; i < n; ++i) zv[i + 1] = zv[i] + s.thickness[i];
    Body& B = w.bodies()[body];
    for (size_t i = 0; i <= n; ++i) {
        const SurfaceSpec& sf = s.surfaces[i];
        auto shape = std::make_shared<SagSurface>(sf.curvature(), sf.k, sf.A, sf.semiDiameter);
        shape->clearSemiDiameter = sf.clearSemiDiameter;
        zEdge[i] = zv[i] + shape->sag(sf.semiDiameter);
        uint32_t front = i < n ? regions[i] : kOutside;  // +z side
        uint32_t back = i > 0 ? regions[i - 1] : kOutside;
        w.addBoundary(body, idx("S", i + 1), shape, Transform::translate({0, 0, zv[i]}), front, back, dielectric);
        if (sf.semiDiameter < edge) {
            // Flat annular step between the polished zone and the rim.
            w.addBoundary(body, idx("S", i + 1) + "-step", PlaneShape::disk(edge, sf.semiDiameter),
                          Transform::translate({0, 0, zEdge[i]}), front, back, rimOptics);
        }
        B.params["R" + std::to_string(i + 1)] = sf.R;
        B.params["sd" + std::to_string(i + 1)] = sf.semiDiameter;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!(zEdge[i + 1] > zEdge[i]))
            throw std::runtime_error("lens '" + name + "': non-positive edge thickness in element " + std::to_string(i + 1) +
                                     " (increase centre thickness or reduce semi-diameter)");
        w.addBoundary(body, idx("rim", i + 1), std::make_shared<CylinderShape>(edge, zEdge[i], zEdge[i + 1]), Transform{},
                      kOutside, regions[i], rimOptics);
        B.params["t" + std::to_string(i + 1)] = s.thickness[i];
    }
    B.params["edge_radius"] = edge;
    return body;
}

LensSpec designSimpleLens(double f, const std::string& glass, double diameter, double t, const std::string& form) {
    Medium m, air;
    if (!catalogMedium(glass, m)) throw std::runtime_error("unknown glass: " + glass);
    catalogMedium("air", air);
    double n = m.n(LambdaD) / air.n(LambdaD);  // index relative to the surrounding air
    LensSpec s;
    s.media = {glass};
    s.thickness = {t};
    SurfaceSpec a, b;
    a.semiDiameter = b.semiDiameter = diameter / 2;
    if (form == "plano") {
        // Curved face first; P = (n-1)/R exactly for a plano lens of any thickness.
        a.R = (n - 1) * f;
        b.R = Inf;
    } else if (form == "bi") {
        // 1/f = (n-1)(2/R - (n-1)t/(nR²)) → R² - 2(n-1)fR + (n-1)²tf/n = 0.
        double disc = 1 - t / (n * f);
        if (disc < 0) throw std::runtime_error("designSimpleLens: lens too thick for requested focal length");
        double R = (n - 1) * f * (1 + std::sqrt(disc));
        a.R = R;
        b.R = -R;
    } else {
        throw std::runtime_error("designSimpleLens: form must be 'bi' or 'plano'");
    }
    s.surfaces = {a, b};
    return s;
}

// ---------------------------------------------------------------- mirrors, stops, tubes

int buildMirror(World& w, const std::string& name, const SurfaceSpec& front, double thickness, uint32_t optics,
                int assembly, const Transform& xf, double holeRadius) {
    int body = w.addBody(name, "mirror", assembly, xf);
    uint32_t sub = w.addRegion(name + ".substrate", w.medium("opaque"), body);
    uint32_t black = w.absorberOptics();
    double sd = front.semiDiameter;
    auto face = std::make_shared<SagSurface>(front.curvature(), front.k, front.A, sd, holeRadius);
    // Reflective face: the substrate lies behind it (on the -z side).
    w.addBoundary(body, "face", face, Transform{}, kOutside, sub, optics);
    double zEdge = face->sag(sd);
    double zBack = std::min(0.0, zEdge) - thickness;
    w.addBoundary(body, "back", PlaneShape::disk(sd, holeRadius),
                  Transform::translate({0, 0, zBack}) * flipZ(), kOutside, sub, black);
    w.addBoundary(body, "rim", std::make_shared<CylinderShape>(sd, zBack, zEdge), Transform{}, kOutside, sub, black);
    if (holeRadius > 0) {
        double zHole = face->sag(holeRadius);
        auto bore = std::make_shared<CylinderShape>(holeRadius, zBack, zHole);
        // Bore normal must point into the hole (away from substrate): flip by using back = substrate on the outside.
        w.addBoundary(body, "bore", bore, Transform{}, sub, kOutside, black);
    }
    w.bodies()[body].params["R"] = front.R;
    w.bodies()[body].params["k"] = front.k;
    w.bodies()[body].params["sd"] = sd;
    return body;
}

int buildFlatMirror(World& w, const std::string& name, double hx, double hy, bool elliptical, uint32_t optics,
                    int assembly, const Transform& xf) {
    int body = w.addBody(name, "flat-mirror", assembly, xf);
    SurfaceOptics o = w.optics()[optics];
    o.backAbsorbs = true;
    uint32_t oi = w.addOptics(o);
    auto shape = elliptical ? PlaneShape::ellipse(hx, hy) : PlaneShape::rect(hx, hy);
    w.addBoundary(body, "face", shape, Transform{}, kOutside, kOutside, oi);
    return body;
}

int buildStop(World& w, const std::string& name, double inner, double outer, int assembly, const Transform& xf) {
    int body = w.addBody(name, "stop", assembly, xf);
    w.addBoundary(body, "blade", PlaneShape::disk(outer, inner), Transform{}, kOutside, kOutside, w.absorberOptics());
    w.bodies()[body].params["aperture_radius"] = inner;
    return body;
}

int buildTube(World& w, const std::string& name, double radius, double z0, double z1, int assembly,
              const Transform& xf, uint32_t optics) {
    int body = w.addBody(name, "tube", assembly, xf);
    w.addBoundary(body, "wall", std::make_shared<CylinderShape>(radius, z0, z1), Transform{}, kOutside, kOutside, optics);
    return body;
}

// ---------------------------------------------------------------- solids

int buildSphere(World& w, const std::string& name, double radius, const BodyMaterial& m, int assembly,
                const Transform& xf) {
    int body = w.addBody(name, "sphere", assembly, xf);
    uint32_t in = interiorRegion(w, body, name + ".interior", m);
    w.addBoundary(body, "surface", std::make_shared<SphereShape>(radius), Transform{}, kOutside, in, m.optics, m.emission);
    return body;
}

void orientOutward(MeshData& m) {
    Vec3 c;
    for (auto& p : m.positions) c += p;
    c = c / double(m.positions.size());
    for (auto& t : m.triangles) {
        Vec3 a = m.positions[t[0]], b = m.positions[t[1]], d = m.positions[t[2]];
        Vec3 n = cross(b - a, d - a);
        if (dot(n, (a + b + d) / 3.0 - c) < 0) std::swap(t[1], t[2]);
    }
}

int buildMesh(World& w, const std::string& name, const MeshData& mesh, const BodyMaterial& m, int assembly,
              const Transform& xf) {
    int body = w.addBody(name, "mesh", assembly, xf);
    uint32_t in = interiorRegion(w, body, name + ".interior", m);
    auto shape = std::make_shared<MeshShape>(mesh.positions, mesh.triangles, name);
    w.addBoundary(body, "surface", shape, Transform{}, kOutside, in, m.optics, m.emission);
    return body;
}

int buildBox(World& w, const std::string& name, const Vec3& size, const BodyMaterial& m, int assembly,
             const Transform& xf) {
    MeshData md;
    Vec3 h = size * 0.5;
    Vec3 c[8];
    for (int i = 0; i < 8; ++i) c[i] = {(i & 1) ? h.x : -h.x, (i & 2) ? h.y : -h.y, (i & 4) ? h.z : -h.z};
    md.addQuad(c[0], c[2], c[3], c[1]);  // -z
    md.addQuad(c[4], c[5], c[7], c[6]);  // +z
    md.addQuad(c[0], c[1], c[5], c[4]);  // -y
    md.addQuad(c[2], c[6], c[7], c[3]);  // +y
    md.addQuad(c[0], c[4], c[6], c[2]);  // -x
    md.addQuad(c[1], c[3], c[7], c[5]);  // +x
    orientOutward(md);
    int body = buildMesh(w, name, md, m, assembly, xf);
    w.bodies()[body].kind = "box";
    return body;
}

int buildCylinder(World& w, const std::string& name, double radius, double height, const BodyMaterial& m,
                  int assembly, const Transform& xf) {
    int body = w.addBody(name, "cylinder", assembly, xf);
    uint32_t in = interiorRegion(w, body, name + ".interior", m);
    w.addBoundary(body, "side", std::make_shared<CylinderShape>(radius, 0, height), Transform{}, kOutside, in, m.optics,
                  m.emission);
    w.addBoundary(body, "top", PlaneShape::disk(radius), Transform::translate({0, 0, height}), kOutside, in, m.optics,
                  m.emission);
    w.addBoundary(body, "bottom", PlaneShape::disk(radius), flipZ(), kOutside, in, m.optics, m.emission);
    return body;
}

int buildPrism(World& w, const std::string& name, double apex, double side, double length, const BodyMaterial& m,
               int assembly, const Transform& xf) {
    double hx = side * std::sin(apex / 2), hy = side * std::cos(apex / 2);
    Vec3 A{0, 2 * hy / 3, 0}, B{-hx, -hy / 3, 0}, C{hx, -hy / 3, 0};  // centroid at origin
    Vec3 dz{0, 0, length / 2};
    MeshData md;
    md.addTriangle(A - dz, B - dz, C - dz);
    md.addTriangle(A + dz, C + dz, B + dz);
    md.addQuad(A - dz, A + dz, B + dz, B - dz);
    md.addQuad(B - dz, B + dz, C + dz, C - dz);
    md.addQuad(C - dz, C + dz, A + dz, A - dz);
    orientOutward(md);
    int body = buildMesh(w, name, md, m, assembly, xf);
    w.bodies()[body].kind = "prism";
    w.bodies()[body].params["apex_deg"] = degrees(apex);
    return body;
}

int buildSheet(World& w, const std::string& name, double hx, double hy, uint32_t optics, int emission, int assembly,
               const Transform& xf, bool disk) {
    int body = w.addBody(name, "sheet", assembly, xf);
    std::shared_ptr<const Shape> s = disk ? std::static_pointer_cast<const Shape>(PlaneShape::disk(hx))
                                          : std::static_pointer_cast<const Shape>(PlaneShape::rect(hx, hy));
    w.addBoundary(body, "face", s, Transform{}, kOutside, kOutside, optics, emission);
    return body;
}

int buildCup(World& w, const std::string& name, double Ro, double wall, double base, double H, double level,
             const std::string& glass, const std::string& liquid, int assembly, const Transform& xf) {
    double Ri = Ro - wall;
    if (!(Ri > 0) || !(H > base) || level < 0 || base + level > H) throw std::runtime_error("cup '" + name + "': bad dimensions");
    int body = w.addBody(name, "cup", assembly, xf);
    uint32_t G = w.addRegion(name + ".glass", w.medium(glass), body);
    uint32_t d = w.dielectricOptics(0);
    w.addBoundary(body, "outer-wall", std::make_shared<CylinderShape>(Ro, 0, H), Transform{}, kOutside, G, d);
    w.addBoundary(body, "outer-bottom", PlaneShape::disk(Ro), flipZ(), kOutside, G, d);
    w.addBoundary(body, "rim", PlaneShape::disk(Ro, Ri), Transform::translate({0, 0, H}), kOutside, G, d);
    if (level > 0) {
        uint32_t L = w.addRegion(name + ".liquid", w.medium(liquid), body);
        w.addBoundary(body, "inner-wall-wet", std::make_shared<CylinderShape>(Ri, base, base + level), Transform{}, G, L, d);
        w.addBoundary(body, "inner-bottom", PlaneShape::disk(Ri), Transform::translate({0, 0, base}), L, G, d);
        w.addBoundary(body, "liquid-surface", PlaneShape::disk(Ri), Transform::translate({0, 0, base + level}), kOutside, L, d);
        if (base + level < H)
            w.addBoundary(body, "inner-wall-dry", std::make_shared<CylinderShape>(Ri, base + level, H), Transform{}, G,
                          kOutside, d);
    } else {
        w.addBoundary(body, "inner-wall", std::make_shared<CylinderShape>(Ri, base, H), Transform{}, G, kOutside, d);
        w.addBoundary(body, "inner-bottom", PlaneShape::disk(Ri), Transform::translate({0, 0, base}), kOutside, G, d);
    }
    return body;
}

// ---------------------------------------------------------------- natural geometry

namespace {
double rawTerrainHeight(const TerrainSpec& t, double x, double y) {
    Vec3 q{x / t.featureSize + double(t.seed % 1000) * 13.37, y / t.featureSize + double(t.seed % 777) * 7.1, 0.5};
    double smooth = fbm3(q, 7);
    double ridged = 1 - std::abs(2 * fbm3(q * 0.7 + Vec3(3.3, 1.1, 0), 6) - 1);
    double h = (1 - t.ridge) * smooth + t.ridge * ridged * ridged;
    return t.amplitude * (h - 0.45) * 2;
}
}  // namespace

double terrainHeight(const TerrainSpec& t, double x, double y) {
    double z = rawTerrainHeight(t, x, y);
    if (t.flatRadius > 0) {
        // Blend smoothly to a level plateau around the origin (a site for instruments).
        double r = std::sqrt(x * x + y * y);
        double s = clampd((r - t.flatRadius) / t.flatRadius, 0, 1);
        s = s * s * (3 - 2 * s);
        z = rawTerrainHeight(t, 0, 0) * (1 - s) + z * s;
    }
    return z;
}

MeshData makeTerrain(const TerrainSpec& t) {
    MeshData m;
    int N = std::max(2, t.resolution);
    for (int j = 0; j <= N; ++j)
        for (int i = 0; i <= N; ++i) {
            double x = (double(i) / N - 0.5) * t.sizeX, y = (double(j) / N - 0.5) * t.sizeY;
            m.positions.push_back({x, y, terrainHeight(t, x, y)});
        }
    auto id = [&](int i, int j) { return uint32_t(j * (N + 1) + i); };
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            m.triangles.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
            m.triangles.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
        }
    return m;
}

namespace {
// Closed surface of revolution about +z from a profile r(z) (z increasing; r > 0 except the apex),
// with a flat bottom cap. Outward normals.
MeshData revolve(const std::vector<std::pair<double, double>>& profile, int seg) {
    MeshData m;
    auto ring = [&](double r, double z, int s) {
        double a = 2 * Pi * s / seg;
        return Vec3(r * std::cos(a), r * std::sin(a), z);
    };
    for (int s = 0; s < seg; ++s) {
        for (size_t k = 0; k + 1 < profile.size(); ++k) {
            auto [r0, z0] = profile[k];
            auto [r1, z1] = profile[k + 1];
            Vec3 a = ring(r0, z0, s), b = ring(r0, z0, s + 1), c = ring(r1, z1, s + 1), d = ring(r1, z1, s);
            if (r1 <= 0) m.addTriangle(a, b, c);
            else if (r0 <= 0) m.addTriangle(a, c, d);
            else m.addQuad(a, b, c, d);
        }
        auto [rb, zb] = profile.front();
        m.addTriangle(ring(rb, zb, s + 1), ring(rb, zb, s), Vec3(0, 0, zb));  // bottom cap, facing −z
    }
    return m;
}
}  // namespace

MeshData makeTreeFoliage(double height, double radius, int seg) {
    // A conifer silhouette: three tiers with downward-facing ledges; base at z = 0.
    const double H = height, R = radius;
    return revolve({{R, 0.0},
                    {0.42 * R, 0.36 * H},
                    {0.78 * R, 0.36 * H},
                    {0.30 * R, 0.68 * H},
                    {0.55 * R, 0.68 * H},
                    {0.0, H}},
                   seg);
}

MeshData makeTreeTrunk(double height, double radius, int seg) {
    // Closed cylinder: side, top and bottom caps.
    return revolve({{radius, 0.0}, {radius, height}, {0.0, height}}, seg);
}

int buildForest(World& w, const std::string& name, const TerrainSpec& terrain, int count, double r0, double r1,
                uint64_t seed, uint32_t foliageOptics, uint32_t trunkOptics, int assembly, const Transform& xf) {
    int body = w.addBody(name, "forest", assembly, xf);
    uint32_t inside = w.addRegion(name + ".wood", w.medium("opaque"), body);
    const int variants = 4;
    std::vector<std::shared_ptr<MeshShape>> foliage, trunks;
    std::vector<double> heights;
    for (int v = 0; v < variants; ++v) {
        double h = 6 + 4.0 * v;
        MeshData f = makeTreeFoliage(h * 0.88, h * 0.22, 10);
        // Trunk from 0.4 m below the ground reference up to just under the foliage base.
        MeshData t = makeTreeTrunk(h * 0.12 + 0.39, h * 0.025, 6);
        foliage.push_back(std::make_shared<MeshShape>(f.positions, f.triangles, "foliage"));
        trunks.push_back(std::make_shared<MeshShape>(t.positions, t.triangles, "trunk"));
        heights.push_back(h);
    }
    Rng rng(seed, 77);
    for (int i = 0; i < count; ++i) {
        double r = std::sqrt(r0 * r0 + rng.uniform() * (r1 * r1 - r0 * r0));
        double a = 2 * Pi * rng.uniform();
        double x = r * std::cos(a), y = r * std::sin(a);
        if (std::abs(x) > terrain.sizeX / 2 || std::abs(y) > terrain.sizeY / 2) continue;
        double z = terrainHeight(terrain, x, y);
        int v = int(rng.uniform() * variants) % variants;
        Transform place = Transform::translate({x, y, z}) * Transform::rotate({0, 0, 1}, 2 * Pi * rng.uniform());
        w.addBoundary(body, "tree" + std::to_string(i) + ".foliage", foliage[v],
                      place * Transform::translate({0, 0, heights[v] * 0.12}), kOutside, inside, foliageOptics);
        w.addBoundary(body, "tree" + std::to_string(i) + ".trunk", trunks[v], place * Transform::translate({0, 0, -0.4}),
                      kOutside, inside, trunkOptics);
    }
    return body;
}

int buildFractalStatue(World& w, const std::string& name, double radius, int depth, double ratio, uint32_t optics,
                       uint32_t pedestalOptics, int assembly, const Transform& xf) {
    int body = w.addBody(name, "fractal-statue", assembly, xf);
    uint32_t inside = w.addRegion(name + ".stone", w.medium("opaque"), body);
    // Pedestal.
    MeshData ped;
    {
        double hw = radius * 0.9, h = radius * 0.8;
        Vec3 c[8];
        for (int i = 0; i < 8; ++i) c[i] = {(i & 1) ? hw : -hw, (i & 2) ? hw : -hw, (i & 4) ? 0.0 : -h};
        ped.addQuad(c[0], c[2], c[3], c[1]);
        ped.addQuad(c[4], c[5], c[7], c[6]);
        ped.addQuad(c[0], c[1], c[5], c[4]);
        ped.addQuad(c[2], c[6], c[7], c[3]);
        ped.addQuad(c[0], c[4], c[6], c[2]);
        ped.addQuad(c[1], c[3], c[7], c[5]);
        orientOutward(ped);
    }
    w.addBoundary(body, "pedestal", std::make_shared<MeshShape>(ped.positions, ped.triangles, "pedestal"),
                  Transform::translate({0, 0, -radius * 0.02}), kOutside, inside, pedestalOptics);
    static const Vec3 dirs[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    std::vector<std::shared_ptr<SphereShape>> shapes;
    double r = radius;
    for (int l = 0; l <= depth; ++l) { shapes.push_back(std::make_shared<SphereShape>(r)); r *= ratio; }
    size_t count = 0;
    std::function<void(const Vec3&, int, int)> rec = [&](const Vec3& c, int level, int parentDir) {
        w.addBoundary(body, "sphere-L" + std::to_string(level) + "-" + std::to_string(count++), shapes[level],
                      Transform::translate(c), kOutside, inside, optics);
        if (level == depth) return;
        double rl = shapes[level]->radius(), rc = shapes[level + 1]->radius();
        for (int k = 0; k < 6; ++k) {
            if (parentDir >= 0 && k == (parentDir ^ 1)) continue;  // skip the direction back to the parent
            if (level == 0 && k == 5) continue;                    // nothing sinks into the pedestal
            rec(c + dirs[k] * (rl + rc), level + 1, k);
        }
    };
    rec(Vec3(0, 0, radius), 0, -1);
    w.bodies()[body].params["depth"] = depth;
    w.bodies()[body].params["ratio"] = ratio;
    w.bodies()[body].params["spheres"] = double(count);
    return body;
}

}  // namespace owe
