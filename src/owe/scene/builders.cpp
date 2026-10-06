#include "owe/scene/builders.hpp"

#include <array>
#include <sstream>
#include <stdexcept>

#include "owe/core/sampling.hpp"

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
              const Transform& xf, std::vector<MeshShape::Corners> shading) {
    int body = w.addBody(name, "mesh", assembly, xf);
    uint32_t in = interiorRegion(w, body, name + ".interior", m);
    auto shape = std::make_shared<MeshShape>(mesh.positions, mesh.triangles, name, std::move(shading), mesh.uvs);
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

namespace {
// A point or direction in the (r, z) half-plane.
struct P2 {
    double x = 0, y = 0;
    P2() = default;
    P2(double a, double b) : x(a), y(b) {}
    P2 operator+(const P2& o) const { return {x + o.x, y + o.y}; }
    P2 operator-(const P2& o) const { return {x - o.x, y - o.y}; }
    P2 operator-() const { return {-x, -y}; }
    P2 operator*(double k) const { return {x * k, y * k}; }
};
double length(const P2& p) { return std::hypot(p.x, p.y); }
P2 normalize(const P2& p) { return p * (1 / length(p)); }

// A piece of a lathe profile: a line from a to b, or an arc of radius rho about c from angle a0 to a1
// (counterclockwise in (r, z) if a1 > a0). P2 components: (r, z).
struct ProfilePiece {
    bool arc = false;
    P2 a, b;           // line ends; for an arc its end points too
    P2 c;              // arc centre
    double rho = 0, a0 = 0, a1 = 0;
    P2 at(double u) const {  // u ∈ [0, 1] along the piece
        if (!arc) return a + (b - a) * u;
        const double t = a0 + (a1 - a0) * u;
        return c + P2(std::cos(t), std::sin(t)) * rho;
    }
    P2 tangent(double u) const {  // direction of travel
        if (!arc) return normalize(b - a);
        const double t = a0 + (a1 - a0) * u, s = a1 > a0 ? 1.0 : -1.0;
        return P2(-std::sin(t), std::cos(t)) * s;
    }
};

ProfilePiece lineOf(const P2& a, const P2& b) {
    ProfilePiece p;
    p.a = a;
    p.b = b;
    return p;
}

// Polyline with fillets at its interior corners → lines and tangent arcs.
std::vector<ProfilePiece> filletProfile(const std::vector<ProfilePoint>& pts, const std::string& what) {
    if (pts.size() < 2) throw std::runtime_error("vessel " + what + " profile needs at least two points");
    std::vector<ProfilePiece> out;
    P2 cursor(pts[0].r, pts[0].z);
    for (size_t i = 1; i < pts.size(); ++i) {
        const P2 P(pts[i].r, pts[i].z);
        const double f = i + 1 < pts.size() ? pts[i].fillet : 0;
        if (!(f > 0)) {
            if (length(P - cursor) > 1e-12) out.push_back(lineOf(cursor, P));
            cursor = P;
            continue;
        }
        const P2 N(pts[i + 1].r, pts[i + 1].z);
        const P2 u = normalize(cursor - P), v = normalize(N - P);
        const double half = 0.5 * std::acos(std::clamp(u.x * v.x + u.y * v.y, -1.0, 1.0));
        if (!(half > 1e-6 && half < Pi / 2 - 1e-9)) throw std::runtime_error("vessel " + what + " profile: a fillet on a straight corner");
        const double d = f / std::tan(half);
        if (d > length(cursor - P) + 1e-12 || d > length(N - P) + 1e-12)
            throw std::runtime_error("vessel " + what + " profile: fillet too large for its corner");
        const P2 t1 = P + u * d, t2 = P + v * d, c = P + normalize(u + v) * (f / std::sin(half));
        if (length(t1 - cursor) > 1e-12) out.push_back(lineOf(cursor, t1));
        ProfilePiece arc{true, t1, t2, c, f, std::atan2(t1.y - c.y, t1.x - c.x), std::atan2(t2.y - c.y, t2.x - c.x)};
        // Take the short way round (a fillet turns by less than π).
        if (arc.a1 - arc.a0 > Pi) arc.a1 -= 2 * Pi;
        if (arc.a0 - arc.a1 > Pi) arc.a1 += 2 * Pi;
        out.push_back(arc);
        cursor = t2;
    }
    return out;
}

// Splits a piece at height z (if it crosses it); returns the pieces below and above.
void splitAt(const ProfilePiece& p, double z, std::vector<ProfilePiece>& below, std::vector<ProfilePiece>& above) {
    const double za = p.a.y, zb = p.b.y;
    const double lo = std::min(za, zb), hi = std::max(za, zb);
    if (!p.arc) {
        if (hi <= z) { below.push_back(p); return; }
        if (lo >= z) { above.push_back(p); return; }
        const P2 m = p.a + (p.b - p.a) * ((z - za) / (zb - za));
        ProfilePiece p1 = p, p2 = p;
        p1.b = m;
        p2.a = m;
        (za < z ? below : above).push_back(p1);
        (za < z ? above : below).push_back(p2);
        return;
    }
    // An arc: find where it crosses z (at most once for a fillet whose angle range avoids a turning
    // point of z, which we check by sampling).
    double zmin = Inf, zmax = -Inf;
    for (int i = 0; i <= 64; ++i) { double y = p.at(i / 64.0).y; zmin = std::min(zmin, y); zmax = std::max(zmax, y); }
    if (zmax <= z) { below.push_back(p); return; }
    if (zmin >= z) { above.push_back(p); return; }
    const double s = (z - p.c.y) / p.rho;  // sin t = s
    double best = -1;
    for (double t : {std::asin(std::clamp(s, -1.0, 1.0)), Pi - std::asin(std::clamp(s, -1.0, 1.0))})
        for (double w : {t, t + 2 * Pi, t - 2 * Pi, t + 4 * Pi, t - 4 * Pi}) {
            const double u = (w - p.a0) / (p.a1 - p.a0);
            if (u > 0 && u < 1) best = u;
        }
    if (best < 0) { (zmax <= z ? below : above).push_back(p); return; }
    const double tm = p.a0 + (p.a1 - p.a0) * best;
    ProfilePiece p1 = p, p2 = p;
    p1.a1 = tm;
    p1.b = p.at(best);
    p2.a0 = tm;
    p2.a = p.at(best);
    (za < z ? below : above).push_back(p1);
    (za < z ? above : below).push_back(p2);
}

// The revolved surface of a piece, and whether its own normal points to the side `sideNormal`
// (a direction in (r, z) at the piece's middle) indicates.
std::shared_ptr<const Shape> revolve(const ProfilePiece& p, Transform& local, const P2& sideNormal, bool& alongNormal) {
    local = Transform{};
    P2 n;  // the shape's own normal at the middle, in (r, z)
    std::shared_ptr<const Shape> shape;
    const P2 mid = p.at(0.5);
    if (p.arc) {
        shape = std::make_shared<TorusPatchShape>(p.c.x, p.c.y, p.rho, p.a0, p.a1);
        n = normalize(mid - p.c);
    } else if (std::abs(p.a.y - p.b.y) < 1e-12) {
        const double r0 = std::min(p.a.x, p.b.x), r1 = std::max(p.a.x, p.b.x);
        shape = PlaneShape::disk(r1, r0);
        local = Transform::translate({0, 0, p.a.y});
        n = P2(0, 1);
    } else if (std::abs(p.a.x - p.b.x) < 1e-12) {
        shape = std::make_shared<CylinderShape>(p.a.x, p.a.y, p.b.y);
        n = P2(1, 0);
    } else {
        auto cone = std::make_shared<ConeShape>(p.a.x, p.a.y, p.b.x, p.b.y);
        const double k = (p.b.x - p.a.x) / (p.b.y - p.a.y);
        n = normalize(P2(1, -k));
        shape = cone;
    }
    alongNormal = n.x * sideNormal.x + n.y * sideNormal.y > 0;
    return shape;
}
}  // namespace

int buildVessel(World& w, const std::string& name, const VesselSpec& s, int assembly, const Transform& xf) {
    auto outer = filletProfile(s.outer, "outer"), inner = filletProfile(s.inner, "inner");
    int body = w.addBody(name, "vessel", assembly, xf);
    const uint32_t glass = w.addRegion(name + ".glass", w.medium(s.glass), body);
    const uint32_t d = w.dielectricOptics(0);
    uint32_t liquid = kOutside;
    if (s.level > 0) liquid = w.addRegion(name + ".liquid", w.medium(s.liquid), body);
    // Walking a profile from the axis up to the rim, the air lies on the right of the outer one and
    // the cavity on the left of the inner one.
    auto rightOf = [](const ProfilePiece& p) { P2 t = p.tangent(0.5); return P2(t.y, -t.x); };
    auto add = [&](const ProfilePiece& p, const P2& sideNormal, uint32_t sideRegion, uint32_t otherRegion,
                   const char* what) {
        Transform local;
        bool along;
        auto shape = revolve(p, local, sideNormal, along);
        w.addBoundary(body, what, shape, local, along ? sideRegion : otherRegion, along ? otherRegion : sideRegion, d);
    };
    for (const ProfilePiece& p : outer) add(p, rightOf(p), kOutside, glass, "outer");
    // The rim: from the outer profile's last point to the inner's, the air above (on its right).
    const P2 o(s.outer.back().r, s.outer.back().z), in(s.inner.back().r, s.inner.back().z);
    if (s.roundRim && length(o - in) > 1e-9) {
        const P2 c = (o + in) * 0.5;
        ProfilePiece rim{true, o, in, c, 0.5 * length(o - in), std::atan2(o.y - c.y, o.x - c.x), 0};
        rim.a1 = rim.a0 + Pi;  // over the top: counterclockwise from the outer to the inner side
        if (rim.at(0.5).y < c.y) rim.a1 = rim.a0 - Pi;
        rim.b = rim.at(1);
        add(rim, normalize(rim.at(0.5) - c), kOutside, glass, "rim");
    } else if (length(o - in) > 1e-9) {
        ProfilePiece rim = lineOf(o, in);
        add(rim, rightOf(rim), kOutside, glass, "rim");
    }
    // The inner profile, wet below the liquid's level and dry above it.
    std::vector<ProfilePiece> wet, dry;
    for (const ProfilePiece& p : inner) {
        if (s.level > 0) splitAt(p, s.level, wet, dry);
        else dry.push_back(p);
    }
    for (const ProfilePiece& p : wet) add(p, -rightOf(p), liquid, glass, "inner-wet");
    for (const ProfilePiece& p : dry) add(p, -rightOf(p), kOutside, glass, "inner-dry");
    if (s.level > 0) {
        // The liquid's flat surface spans the cavity at its level.
        double rl = -1;
        for (const ProfilePiece& p : inner)
            for (int i = 0; i < 4096; ++i) {
                const P2 a = p.at(i / 4096.0), b = p.at((i + 1) / 4096.0);
                if ((a.y - s.level) * (b.y - s.level) <= 0 && a.y != b.y) {
                    const double r = a.x + (b.x - a.x) * (s.level - a.y) / (b.y - a.y);
                    rl = std::max(rl, r);
                }
            }
        if (!(rl > 0)) throw std::runtime_error("vessel '" + name + "': the liquid level is outside the cavity");
        w.addBoundary(body, "liquid-surface", PlaneShape::disk(rl), Transform::translate({0, 0, s.level}), kOutside, liquid, d);
    }
    return body;
}

int buildWater(World& w, const std::string& name, const WaterSpec& s, int assembly, const Transform& xf) {
    int body = w.addBody(name, "water", assembly, xf);
    uint32_t liquid = w.addRegion(name + ".liquid", w.medium(s.medium), body);
    uint32_t basin = kOutside;
    if (s.bottomOptics >= 0 || s.wallOptics >= 0) basin = w.addRegion(name + ".basin", w.medium("opaque"), body);
    const uint32_t interface = w.dielectricOptics(0);
    w.addBoundary(body, "surface", std::make_shared<WaveSurface>(s.waves, s.halfX, s.halfY, s.margin), Transform{},
                  kOutside, liquid, interface);
    // Floor and walls face the liquid (normals inward).
    auto side = [&](int optics) { return optics >= 0 ? std::pair{basin, uint32_t(optics)} : std::pair{kOutside, interface}; };
    auto [floorBack, floorOptics] = side(s.bottomOptics);
    w.addBoundary(body, "floor", PlaneShape::rect(s.halfX, s.halfY), Transform::translate({0, 0, -s.depth}), liquid,
                  floorBack, floorOptics);
    auto [wallBack, wallOptics] = side(s.wallOptics);
    const double hz = s.depth / 2;
    struct Wall {
        Vec3 at, inward;
        double half;
    };
    for (const Wall& wl : {Wall{{s.halfX, 0, -hz}, {-1, 0, 0}, s.halfY}, Wall{{-s.halfX, 0, -hz}, {1, 0, 0}, s.halfY},
                           Wall{{0, s.halfY, -hz}, {0, -1, 0}, s.halfX}, Wall{{0, -s.halfY, -hz}, {0, 1, 0}, s.halfX}})
        w.addBoundary(body, "wall", PlaneShape::rect(wl.half, hz), Transform::translate(wl.at) * Transform::alignZ(wl.inward, {0, 0, 1}),
                      liquid, wallBack, wallOptics);
    if (basin != kOutside) {
        // The block the basin is cut into: a rim of ground at z = 0 around the opening, its outer
        // sides and its underside, all facing the air.
        const uint32_t ground = uint32_t(s.wallOptics >= 0 ? s.wallOptics : s.bottomOptics);
        const double ox = s.halfX + s.rim, oy = s.halfY + s.rim, zb = -s.depth - s.base, oz = -zb / 2;
        for (auto [cx, cy, hx, hy] : {std::array<double, 4>{0, (oy + s.halfY) / 2, ox, (oy - s.halfY) / 2},
                                      std::array<double, 4>{0, -(oy + s.halfY) / 2, ox, (oy - s.halfY) / 2},
                                      std::array<double, 4>{(ox + s.halfX) / 2, 0, (ox - s.halfX) / 2, s.halfY},
                                      std::array<double, 4>{-(ox + s.halfX) / 2, 0, (ox - s.halfX) / 2, s.halfY}})
            w.addBoundary(body, "rim", PlaneShape::rect(hx, hy), Transform::translate({cx, cy, 0}), kOutside, basin, ground);
        for (const Wall& wl : {Wall{{ox, 0, zb / 2}, {1, 0, 0}, oy}, Wall{{-ox, 0, zb / 2}, {-1, 0, 0}, oy},
                               Wall{{0, oy, zb / 2}, {0, 1, 0}, ox}, Wall{{0, -oy, zb / 2}, {0, -1, 0}, ox}})
            w.addBoundary(body, "side", PlaneShape::rect(wl.half, oz), Transform::translate(wl.at) * Transform::alignZ(wl.inward, {0, 0, 1}),
                          kOutside, basin, ground);
        w.addBoundary(body, "underside", PlaneShape::rect(ox, oy), Transform::translate({0, 0, zb}) * Transform::alignZ({0, 0, -1}),
                      kOutside, basin, ground);
    }
    return body;
}

std::vector<PlaneWave> rippleField(int count, uint64_t seed, double minWavelength, double maxWavelength, double slope,
                                   double direction, double spread) {
    std::vector<PlaneWave> waves;
    Rng rng(seed, 0x7269707063ull);
    for (int i = 0; i < count; ++i) {
        const double u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
        const double lambda = minWavelength * std::pow(maxWavelength / minWavelength, u1);
        const double k = 2 * Pi / lambda, theta = direction + (u2 - 0.5) * spread;
        PlaneWave pw;
        pw.kx = k * std::cos(theta);
        pw.ky = k * std::sin(theta);
        pw.amplitude = slope / (k * std::sqrt(double(count)));
        pw.phase = 2 * Pi * u3;
        waves.push_back(pw);
    }
    return waves;
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
             const std::string& glass, const std::string& liquid, int assembly, const Transform& xf, const CupRod& rod) {
    double Ri = Ro - wall;
    if (!(Ri > 0) || !(H > base) || level < 0 || base + level > H) throw std::runtime_error("cup '" + name + "': bad dimensions");
    bool hasRod = rod.radius > 0;
    if (hasRod && (std::hypot(rod.x, rod.y) + rod.radius >= Ri || rod.top <= base + level))
        throw std::runtime_error("cup '" + name + "': the rod must stand inside the cup and rise above the liquid");
    int body = w.addBody(name, "cup", assembly, xf);
    uint32_t G = w.addRegion(name + ".glass", w.medium(glass), body);
    uint32_t d = w.dielectricOptics(0);
    w.addBoundary(body, "outer-wall", std::make_shared<CylinderShape>(Ro, 0, H), Transform{}, kOutside, G, d);
    w.addBoundary(body, "outer-bottom", PlaneShape::disk(Ro), flipZ(), kOutside, G, d);
    w.addBoundary(body, "rim", PlaneShape::disk(Ro, Ri), Transform::translate({0, 0, H}), kOutside, G, d);
    uint32_t L = kOutside;
    if (level > 0) {
        L = w.addRegion(name + ".liquid", w.medium(liquid), body);
        w.addBoundary(body, "inner-wall-wet", std::make_shared<CylinderShape>(Ri, base, base + level), Transform{}, G, L, d);
        w.addBoundary(body, "inner-bottom", PlaneShape::disk(Ri), Transform::translate({0, 0, base}), L, G, d);
        std::shared_ptr<const Shape> surface =
            hasRod ? std::static_pointer_cast<const Shape>(PlaneShape::diskWithHole(Ri, rod.radius, rod.x, rod.y))
                   : std::static_pointer_cast<const Shape>(PlaneShape::disk(Ri));
        w.addBoundary(body, "liquid-surface", surface, Transform::translate({0, 0, base + level}), kOutside, L, d);
        if (base + level < H)
            w.addBoundary(body, "inner-wall-dry", std::make_shared<CylinderShape>(Ri, base + level, H), Transform{}, G,
                          kOutside, d);
    } else {
        w.addBoundary(body, "inner-wall", std::make_shared<CylinderShape>(Ri, base, H), Transform{}, G, kOutside, d);
        w.addBoundary(body, "inner-bottom", PlaneShape::disk(Ri), Transform::translate({0, 0, base}), kOutside, G, d);
    }
    if (hasRod) {
        bool clear = !rod.medium.empty();
        uint32_t R = w.addRegion(name + ".rod", w.medium(clear ? rod.medium : "opaque"), body);
        uint32_t ro = clear ? d : rod.optics;
        double z0 = base + 1e-3;  // rests 1 mm above the bottom: bodies never share faces
        Transform at = Transform::translate({rod.x, rod.y, 0});
        uint32_t below = level > 0 ? L : kOutside;
        double zs = level > 0 ? base + level : z0;
        w.addBoundary(body, "rod-bottom", PlaneShape::disk(rod.radius), at * Transform::translate({0, 0, z0}) * flipZ(), below,
                      R, ro);
        if (level > 0)
            w.addBoundary(body, "rod-wet", std::make_shared<CylinderShape>(rod.radius, z0, zs), at, below, R, ro);
        w.addBoundary(body, "rod-dry", std::make_shared<CylinderShape>(rod.radius, zs, rod.top), at, kOutside, R, ro);
        w.addBoundary(body, "rod-top", PlaneShape::disk(rod.radius), at * Transform::translate({0, 0, rod.top}), kOutside, R,
                      ro);
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

MeshData revolveProfile(std::vector<std::pair<double, double>> profile, int seg) {
    // Close the profile on the axis so the surface is watertight. Traversing the profile from
    // the bottom of the axis outward, up, and back to the axis keeps normals outward, including
    // overhangs and ledges (the normal of a segment is Δz·r̂ − Δr·ẑ).
    if (profile.size() < 2) throw std::runtime_error("lathe profile needs at least two points");
    if (profile.front().first > 0) profile.insert(profile.begin(), {0.0, profile.front().second});
    if (profile.back().first > 0) profile.push_back({0.0, profile.back().second});
    MeshData m;
    auto ring = [&](double r, double z, int s) {
        double a = 2 * Pi * s / seg;
        return Vec3(r * std::cos(a), r * std::sin(a), z);
    };
    for (int s = 0; s < seg; ++s)
        for (size_t k = 0; k + 1 < profile.size(); ++k) {
            auto [r0, z0] = profile[k];
            auto [r1, z1] = profile[k + 1];
            if (r0 <= 0 && r1 <= 0) continue;
            Vec3 a = ring(r0, z0, s), b = ring(r0, z0, s + 1), c = ring(r1, z1, s + 1), d = ring(r1, z1, s);
            if (r1 <= 0) m.addTriangle(a, b, c);
            else if (r0 <= 0) m.addTriangle(a, c, d);
            else m.addQuad(a, b, c, d);
        }
    return m;
}

MeshData makeTorus(double R, double r, int segU, int segV) {
    MeshData m;
    auto P = [&](int i, int j) {
        double u = 2 * Pi * i / segU, v = 2 * Pi * j / segV;
        return Vec3((R + r * std::cos(v)) * std::cos(u), (R + r * std::cos(v)) * std::sin(u), r * std::sin(v));
    };
    for (int i = 0; i < segU; ++i)
        for (int j = 0; j < segV; ++j) m.addQuad(P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1));
    return m;
}

MeshData makeFlutedShaft(double height, double r0, double r1, double entasis, int flutes, double fluteDepth,
                         int segPerFlute, int segZ) {
    MeshData m;
    int segT = flutes * segPerFlute;
    auto radius = [&](double theta, double z) {
        double t = z / height;
        double rz = r0 + (r1 - r0) * t + entasis * r0 * std::sin(Pi * t);
        // Shallow concave flutes meeting in sharp arrises (Doric).
        double f = std::fmod(theta * flutes / (2 * Pi), 1.0);
        double x = 2 * f - 1;  // −1 … 1 across one flute
        return rz * (1 - fluteDepth * (1 - x * x));
    };
    auto P = [&](int i, int j) {
        double th = 2 * Pi * i / segT, z = height * j / segZ;
        double r = radius(th, z);
        return Vec3(r * std::cos(th), r * std::sin(th), z);
    };
    for (int i = 0; i < segT; ++i) {
        for (int j = 0; j < segZ; ++j) m.addQuad(P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1));
        m.addTriangle(Vec3(0, 0, height), P(i, segZ), P(i + 1, segZ));  // top cap
        m.addTriangle(Vec3(0, 0, 0), P(i + 1, 0), P(i, 0));             // bottom cap
    }
    return m;
}

MeshData makeRoundWall(double R, double H, int segAz, int segZ, const std::vector<WallOpening>& openings) {
    MeshData m;
    auto P = [&](int i, int j) {
        double az = 2 * Pi * i / segAz;
        return Vec3(R * std::sin(az), R * std::cos(az), H * j / segZ);
    };
    for (int i = 0; i < segAz; ++i)
        for (int j = 0; j < segZ; ++j) {
            double azMid = 2 * Pi * (i + 0.5) / segAz, zMid = H * (j + 0.5) / segZ;
            bool open = false;
            for (const auto& o : openings) {
                double dAz = std::remainder(azMid - o.azimuth, 2 * Pi);
                if (std::abs(dAz) * R < 0.5 * o.width && zMid > o.sill && zMid < o.top) open = true;
            }
            if (!open) m.addQuad(P(i, j), P(i, j + 1), P(i + 1, j + 1), P(i + 1, j));
        }
    return m;
}

MeshData makeDome(double R, int segAz, int segEl, double slitAz, double slitWidth, double slitTop) {
    MeshData m;
    auto P = [&](int i, int j) {
        double az = 2 * Pi * i / segAz, el = 0.5 * Pi * j / segEl;
        return Vec3(R * std::cos(el) * std::sin(az), R * std::cos(el) * std::cos(az), R * std::sin(el));
    };
    for (int i = 0; i < segAz; ++i)
        for (int j = 0; j < segEl; ++j) {
            if (slitWidth > 0) {
                double azMid = 2 * Pi * (i + 0.5) / segAz;
                double dAz = std::remainder(azMid - slitAz, 2 * Pi);
                double elMid = 0.5 * Pi * (j + 0.5) / segEl;
                // The slit keeps a constant linear width, so its angular width grows toward the zenith.
                double halfW = std::asin(std::min(1.0, 0.5 * slitWidth / std::max(1e-9, R * std::cos(elMid))));
                if (std::abs(dAz) < halfW && elMid < slitTop) continue;
            }
            Vec3 a = P(i, j), b = P(i + 1, j), c = P(i + 1, j + 1), d = P(i, j + 1);
            if (j + 1 == segEl) m.addTriangle(a, b, c);  // c == d at the zenith
            else m.addQuad(a, b, c, d);
        }
    return m;
}

MeshData makeTreeFoliage(double height, double radius, int seg) {
    // A conifer silhouette: three tiers with downward-facing ledges; base at z = 0.
    const double H = height, R = radius;
    return revolveProfile({{R, 0.0},
                    {0.42 * R, 0.36 * H},
                    {0.78 * R, 0.36 * H},
                    {0.30 * R, 0.68 * H},
                    {0.55 * R, 0.68 * H},
                    {0.0, H}},
                   seg);
}

MeshData makeTreeTrunk(double height, double radius, int seg) {
    // Closed cylinder: side, top and bottom caps.
    return revolveProfile({{radius, 0.0}, {radius, height}, {0.0, height}}, seg);
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

int buildStarfield(World& w, const std::string& name, int count, uint64_t seed, double distance, double angularRadius,
                   double brightest, double minElevation, int assembly, const Transform& xf) {
    int body = w.addBody(name, "starfield", assembly, xf);
    uint32_t inside = w.addRegion(name + ".stellar", w.medium("opaque"), body);
    auto shape = std::make_shared<SphereShape>(distance * std::tan(angularRadius));
    SurfaceOptics black;
    black.type = SurfaceType::Absorber;
    black.name = name;
    uint32_t oi = w.addOptics(black);
    Rng rng(seed, 1234);
    double zMin = std::sin(minElevation);
    for (int i = 0; i < count; ++i) {
        double z = zMin + (1 - zMin) * rng.uniform(), phi = 2 * Pi * rng.uniform();
        double r = safeSqrt(1 - z * z);
        Vec3 dir{r * std::cos(phi), r * std::sin(phi), z};
        // Most stars are faint: luminance ∝ u⁴ spans several magnitudes.
        double y = brightest * std::pow(rng.uniform(), 4.0) + brightest * 1e-3;
        double T = 3000 + 9000 * std::pow(rng.uniform(), 1.6);
        Emission e;
        e.radiance = Spectrum::blackbody(T, y);
        e.distant = true;
        w.addBoundary(body, "star" + std::to_string(i), shape, Transform::translate(dir * distance), kOutside, inside, oi,
                      w.addEmission(e));
    }
    w.bodies()[body].params["stars"] = count;
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
                  Transform::translate({0, 0, -radius * 1e-6}), kOutside, inside, pedestalOptics);
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
            // Tangent spheres would touch at a single point (a degenerate contact); a 1e-7
            // relative clearance keeps every body disjoint without any visible gap.
            rec(c + dirs[k] * ((rl + rc) * (1 + 1e-7)), level + 1, k);
        }
    };
    rec(Vec3(0, 0, radius), 0, -1);
    w.bodies()[body].params["depth"] = depth;
    w.bodies()[body].params["ratio"] = ratio;
    w.bodies()[body].params["spheres"] = double(count);
    return body;
}

}  // namespace owe
