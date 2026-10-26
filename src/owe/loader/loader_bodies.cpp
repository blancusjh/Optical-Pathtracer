// Placement and bodies of the .owe language: one builder per body type.
#include <fstream>
#include "owe/loader/loader.hpp"

namespace owe::detail {

// ------------------------------------------------------------ placement
Transform Loader::placement(const Value& b) {
    Transform xf;
    Vec3 pos = b.has("position") ? vecLen(need(b, "position")) : Vec3();
    Transform R;
    if (auto r = b.get("rotate")) {
        auto s = seq(*r, 3);
        R = Transform::rotate({0, 0, 1}, ang(*s[2])) * Transform::rotate({0, 1, 0}, ang(*s[1])) *
            Transform::rotate({1, 0, 0}, ang(*s[0]));
    }
    Vec3 up = upOr(b);
    if (auto a = b.get("axis")) R = Transform::alignZ(vecNum(*a), up) * R;
    else if (auto la = b.get("look_at")) R = Transform::alignZ(vecLen(*la) - pos, up) * R;
    // Instruments receive light travelling along local +z: point their input side at a target.
    else if (auto pa = b.get("point_at")) R = Transform::alignZ(pos - vecLen(*pa), up) * R;
    xf = Transform::translate(pos) * R;
    if (auto d = b.get("decenter")) {
        auto s = seq(*d, 2);
        xf = xf * Transform::translate({len(*s[0]), len(*s[1]), 0});
    }
    if (auto t = b.get("tilt")) {
        auto s = seq(*t, 2);
        xf = xf * Transform::rotate({1, 0, 0}, ang(*s[0])) * Transform::rotate({0, 1, 0}, ang(*s[1]));
    }
    return xf;
}

Vec3 Loader::onTerrain(const Value& v) {
    if (v.items.size() < 3) fail(v, "on_terrain(\"Terrain\", x, y [, height above ground])");
    std::string n = str(*v.items[0]);
    auto it = terrains_.find(n);
    if (it == terrains_.end()) fail(v, "on_terrain: no terrain named '" + n + "' (define it first)");
    const Transform& xf = it->second.second;
    double x = len(*v.items[1]), y = len(*v.items[2]);
    double dz = v.items.size() > 3 ? len(*v.items[3]) : 0.0;
    return {x, y, terrainHeight(it->second.first, x - xf.t.x, y - xf.t.y) + xf.t.z + dz};
}

Vec3 Loader::exitPupil(const Value& v) {
    if (v.items.empty()) fail(v, "exit_pupil needs an instrument name");
    std::string n = str(*v.items[0]);
    auto it = instruments_.find(n);
    if (it == instruments_.end()) fail(v, "exit_pupil: no instrument named '" + n + "' (define it first)");
    double offset = v.items.size() > 1 ? len(*v.items[1]) : 0.0;
    const Instrument& ins = it->second;
    return w_.assemblyToWorld(ins.assembly).point({0, 0, ins.paraxial.exitPupilZ + offset});
}

// ------------------------------------------------------------ surfaces
SurfaceSpec Loader::surface(const Value& v, double semiDiameter) const {
    SurfaceSpec s;
    s.semiDiameter = semiDiameter;
    if (v.kind != Value::Kind::Call) fail(v, "expected a surface: plane(), sphere(R=...), conic(R=..., k=...), asphere(...)");
    if (v.str == "plane" || v.str == "flat") return s;
    auto R = v.get("R");
    if (!R && !v.items.empty()) R = v.items[0];
    if (!R) fail(v, "surface needs R");
    s.R = lenOrInf(*R);
    if (v.str == "sphere") return s;
    if (v.str == "conic" || v.str == "asphere") {
        s.k = getNum(v, "k", 0);
        for (int order = 4; order <= 20; order += 2) {
            if (auto a = v.get("A" + std::to_string(order))) {
                size_t slot = size_t((order - 4) / 2);
                if (s.A.size() <= slot) s.A.resize(slot + 1, 0.0);
                s.A[slot] = num(*a) * std::pow(unit_, 1 - order);
            }
        }
        return s;
    }
    fail(v, "unknown surface '" + v.str + "'");
}

void Loader::body(const Value& b, int assembly) {
    if (b.name.empty()) fail(b, "bodies need a name");
    // repeat = (count, (dx, dy, dz)): identical copies Name_0 … Name_{n−1} offset in the parent frame.
    if (auto r = b.get("repeat")) {
        auto q = seq(*r, 2);
        int n = int(num(*q[0]));
        Vec3 step = vecLen(*q[1]);
        if (n < 1) fail(*r, "repeat count must be positive");
        for (int i = 0; i < n; ++i)
            bodyImpl(b, assembly, b.name + "_" + std::to_string(i), Transform::translate(step * double(i)));
        return;
    }
    bodyImpl(b, assembly, b.name, Transform{});
}

void Loader::bodyImpl(const Value& b, int assembly, const std::string& name, const Transform& pre) {
    std::string type = getStr(b, "type", "");
    if (!names_.insert(name).second) fail(b, "duplicate body name '" + name + "'");
    Transform xf = pre * placement(b);
    // `in = Name`: the body sits inside another body's transparent medium (stones under water).
    struct Immersion {
        World& w;
        ~Immersion() { w.setImmersion(kOutside); }
    } immersion{w_};
    if (auto in = b.get("in")) {
        int host = w_.findBody(str(*in));
        if (host < 0) fail(*in, "'in' names a body defined earlier (with a transparent medium)");
        uint32_t region = kOutside;
        for (uint32_t r = 0; r < w_.regions().size(); ++r)
            if (w_.regions()[r].body == host && !w_.media()[w_.regions()[r].medium].opaque) { region = r; break; }
        if (region == kOutside) fail(*in, "body '" + str(*in) + "' has no transparent medium to be inside");
        w_.setImmersion(region);
    }
    // One builder per body type.
    using Build = void (Loader::*)(const Value&, const std::string&, int, const Transform&);
    static const std::map<std::string, Build> builders = {
        {"lens", &Loader::lensBody},
        {"prescription", &Loader::prescriptionBody},
        {"mirror", &Loader::mirrorBody},
        {"flat_mirror", &Loader::flatMirrorBody},
        {"stop", &Loader::stopBody},
        {"iris", &Loader::stopBody},
        {"aperture", &Loader::stopBody},
        {"tube", &Loader::tubeBody},
        {"sphere", &Loader::sphereBody},
        {"box", &Loader::boxBody},
        {"cylinder", &Loader::cylinderBody},
        {"prism", &Loader::prismBody},
        {"lathe", &Loader::latheBody},
        {"torus", &Loader::torusBody},
        {"ring", &Loader::torusBody},
        {"column", &Loader::columnBody},
        {"dome", &Loader::domeBody},
        {"round_wall", &Loader::roundWallBody},
        {"starfield", &Loader::starfieldBody},
        {"sheet", &Loader::sheetBody},
        {"screen", &Loader::sheetBody},
        {"disk", &Loader::sheetBody},
        {"mesh", &Loader::meshBody},
        {"vessel", &Loader::vesselBody},
        {"waves", &Loader::wavesBody},
        {"cup", &Loader::cupBody},
        {"terrain", &Loader::terrainBody},
        {"forest", &Loader::forestBody},
        {"fractal_statue", &Loader::statueBody},
        {"statue", &Loader::statueBody}};
    auto builder = builders.find(type);
    if (builder == builders.end()) fail(b, "unknown body type '" + type + "'");
    (this->*builder->second)(b, name, assembly, xf);
}

void Loader::mirrorBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    SurfaceSpec s = surface(need(b, "surface"), getRadius(b, "", 0.05));
    uint32_t o = materialOr(b, "material", "aluminium");
    buildMirror(w_, name, s, getLen(b, "thickness", 0.1 * s.semiDiameter * 2), o, assembly, xf,
                getRadius(b, "hole_", 0));
}

void Loader::flatMirrorBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    uint32_t o = materialOr(b, "material", "aluminium");
    if (auto e = b.get("ellipse")) {
        const Vec2 h = halfSize(*e);
        buildFlatMirror(w_, name, h.x, h.y, true, o, assembly, xf);
    } else {
        const Vec2 h = halfSize(need(b, "size"));
        buildFlatMirror(w_, name, h.x, h.y, false, o, assembly, xf);
    }
}

void Loader::stopBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    double inner = getRadius(b, "aperture_", -1);
    if (inner < 0) inner = len(need(b, "aperture")) / 2;
    buildStop(w_, name, inner, getRadius(b, "outer_", inner * 4), assembly, xf);
}

void Loader::tubeBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    buildTube(w_, name, getRadius(b, "", 0.05), 0, len(need(b, "length")), assembly, xf,
              materialOr(b, "material", "black"));
}

void Loader::sphereBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    buildSphere(w_, name, getRadius(b, "", 0.01), solidMaterial(b), assembly, xf);
}

void Loader::boxBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    buildBox(w_, name, vecLen(need(b, "size")), solidMaterial(b), assembly, xf);
}

void Loader::cylinderBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    buildCylinder(w_, name, getRadius(b, "", 0.01), len(need(b, "height")), solidMaterial(b), assembly, xf);
}

void Loader::prismBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    buildPrism(w_, name, getAng(b, "apex", radians(60)), len(need(b, "side")), len(need(b, "length")),
               solidMaterial(b), assembly, xf);
}

void Loader::latheBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    std::vector<std::pair<double, double>> prof;
    for (auto& pt : need(b, "profile").items) {
        auto q = seq(*pt, 2);
        prof.push_back({len(*q[0]), len(*q[1])});
    }
    int id = buildMesh(w_, name, revolveProfile(prof, int(getNum(b, "segments", 64))), solidMaterial(b), assembly, xf);
    w_.bodies()[id].kind = "lathe";
}

void Loader::torusBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    double R = getRadius(b, "", 0.1), r = getRadius(b, "tube_", R * 0.05);
    int id = buildMesh(w_, name, makeTorus(R, r, int(getNum(b, "segments", 96)), int(getNum(b, "tube_segments", 16))),
                       solidMaterial(b), assembly, xf);
    w_.bodies()[id].kind = "torus";
}

void Loader::domeBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    // Exact hemisphere with an observing slit (open surface, same finish both sides).
    double R = getRadius(b, "", 3);
    auto shape = std::make_shared<DomeShape>(R, getAng(b, "slit_azimuth", 0), getLen(b, "slit_width", 0),
                                             getAng(b, "slit_top", radians(90)));
    uint32_t o = materialOr(b, "material", "white");
    int id = w_.addBody(name, "dome", assembly, xf);
    w_.addBoundary(id, "shell", shape, Transform{}, kOutside, kOutside, o);
}

void Loader::roundWallBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    std::vector<RoundWallShape::Opening> ops;
    if (auto list = b.get("openings"))
        for (auto& o : list->items) {
            auto q = seq(*o, 4);
            ops.push_back({ang(*q[0]), len(*q[1]), len(*q[2]), len(*q[3])});
        }
    auto shape = std::make_shared<RoundWallShape>(getRadius(b, "", 3), len(need(b, "height")), ops);
    uint32_t o = materialOr(b, "material", "white");
    int id = w_.addBody(name, "round-wall", assembly, xf);
    w_.addBoundary(id, "wall", shape, Transform{}, kOutside, kOutside, o);
}

void Loader::starfieldBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    buildStarfield(w_, name, int(getNum(b, "count", 2000)), uint64_t(getNum(b, "seed", 1)), getLen(b, "distance", 1e13),
                   getAng(b, "angular_radius", radians(0.02)), getNum(b, "brightest", 5.0),
                   getAng(b, "min_elevation", radians(-5)), assembly, xf);
}

void Loader::sheetBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    BodyMaterial m = solidMaterial(b);
    if (b.has("inner_radius") || b.has("inner_diameter")) {
        double r = getRadius(b, "", 0.1), ri = getRadius(b, "inner_", 0);
        int id = w_.addBody(name, "sheet", assembly, xf);
        w_.addBoundary(id, "face", PlaneShape::disk(r, ri), Transform{}, kOutside, kOutside, m.optics, m.emission);
    } else if (getStr(b, "type", "") == "disk" || b.has("radius") || b.has("diameter")) {
        double r = getRadius(b, "", 0.1);
        buildSheet(w_, name, r, r, m.optics, m.emission, assembly, xf, true);
    } else {
        const Vec2 h = halfSize(need(b, "size"));
        buildSheet(w_, name, h.x, h.y, m.optics, m.emission, assembly, xf, false);
    }
}

void Loader::meshBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    std::string file = path(str(need(b, "file")));
    if (auto opt = b.get("optional"); opt && boolean(*opt) && !std::ifstream(file)) {
        std::fprintf(stderr, "warning: optional mesh '%s' not found (%s); body skipped\n", name.c_str(), file.c_str());
        return;
    }
    MeshData md;
    try {
        md = loadModel(file, getNum(b, "scale", 1.0) * unit_);
        // part = "Glass" | ["Stone", "Lead"]: some of a glTF model's materials. Placement
        // (model_up, fit_height, ground) measures the whole model, so its parts, loaded as
        // separate bodies, stay together.
        if (auto p = b.get("part")) {
            std::vector<std::string> names;
            if (p->kind == Value::Kind::List)
                for (const ValuePtr& it : p->items) names.push_back(str(*it));
            else
                names.push_back(str(*p));
            md = selectParts(md, names);
        }
    } catch (const std::runtime_error& e) {
        fail(b, e.what());
    }
    prepareImportedMesh(b, md);
    BodyMaterial m = solidMaterial(b);
    if (w_.optics()[m.optics].texture.mapping == Texture::Mapping::UV && md.uvs.empty())
        fail(b, "the material's image is mapped by texture coordinates, which " + file + " does not have");
    bool closed = m.transparent;
    if (auto c = b.get("closed")) closed = boolean(*c);
    std::vector<MeshShape::Corners> shading = meshNormals(b, md);
    if (closed) {
        buildMesh(w_, name, md, m, assembly, xf, std::move(shading));
    } else {
        // Scanned models are rarely watertight or consistently wound: represent an opaque
        // model as a shell with air on both sides, so orientation and holes cannot matter.
        int id = w_.addBody(name, "mesh", assembly, xf);
        w_.addBoundary(id, "surface",
                       std::make_shared<MeshShape>(md.positions, md.triangles, name, std::move(shading), md.uvs),
                       Transform{}, kOutside, kOutside, m.optics, m.emission);
    }
}

void Loader::vesselBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    // body Glass { type = vessel  outer = [(r, z), (r, z, fillet), …]  inner = [...]  rim = round | flat
    //              glass = N-BK7  liquid = water  level = 70mm }
    VesselSpec vs;
    auto profile = [&](const char* key) {
        std::vector<ProfilePoint> pts;
        for (const ValuePtr& it : need(b, key).items) {
            if (!it || it->kind != Value::Kind::Tuple || it->items.size() < 2 || it->items.size() > 3)
                fail(b, std::string(key) + " must be a list of (r, z) or (r, z, fillet)");
            ProfilePoint q;
            q.r = len(*it->items[0]);
            q.z = len(*it->items[1]);
            if (it->items.size() == 3) q.fillet = len(*it->items[2]);
            pts.push_back(q);
        }
        return pts;
    };
    vs.outer = profile("outer");
    vs.inner = profile("inner");
    vs.roundRim = getStr(b, "rim", "round") == "round";
    vs.glass = getStr(b, "glass", "N-BK7");
    vs.liquid = getStr(b, "liquid", "water");
    vs.level = getLen(b, "level", 0);
    w_.medium(vs.glass);
    if (vs.level > 0) w_.medium(vs.liquid);
    try {
        buildVessel(w_, name, vs, assembly, xf);
    } catch (const std::exception& e) {
        fail(b, e.what());
    }
}

void Loader::wavesBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    // body Pool { type = waves  size = (w, l)  depth = d  margin = m  medium = water  bottom = sand
    //             walls = tile  waves = [wave(amplitude, wavelength, direction, phase), ...]
    //                         | ripples(count, seed, wavelength = (min, max), slope, direction, spread) }
    WaterSpec ws;
    const Vec2 h = halfSize(need(b, "size"));
    ws.halfX = h.x;
    ws.halfY = h.y;
    ws.depth = getLen(b, "depth", 0.2);
    ws.margin = getLen(b, "margin", 0.2 * std::min(ws.halfX, ws.halfY));
    ws.medium = getStr(b, "medium", "water");
    ws.rim = getLen(b, "rim", 0.25 * std::max(ws.halfX, ws.halfY));
    ws.base = getLen(b, "base", 0.25 * ws.depth);
    w_.medium(ws.medium);
    if (auto m = b.get("bottom")) ws.bottomOptics = int(materialRef(*m));
    if (auto m = b.get("walls")) ws.wallOptics = int(materialRef(*m));
    for (int o : {ws.bottomOptics, ws.wallOptics})
        if (o >= 0 && w_.optics()[size_t(o)].type == SurfaceType::Dielectric) fail(b, "bottom and walls must be opaque materials");
    const Value& wv = need(b, "waves");
    if (wv.kind == Value::Kind::Call && wv.str == "ripples") {
        auto range = seq(need(wv, "wavelength"), 2);
        ws.waves = rippleField(int(getNum(wv, "count", 24)), uint64_t(getNum(wv, "seed", 1)), len(*range[0]),
                               len(*range[1]), getNum(wv, "slope", 0.1), getAng(wv, "direction", 0),
                               getAng(wv, "spread", radians(90)));
    } else {
        for (const ValuePtr& it : wv.items) {
            if (!it || it->kind != Value::Kind::Call || it->str != "wave") fail(wv, "waves must be a list of wave(...) or ripples(...)");
            PlaneWave pw;
            const double k = 2 * Pi / len(need(*it, "wavelength")), dir = getAng(*it, "direction", 0);
            pw.amplitude = len(need(*it, "amplitude"));
            pw.kx = k * std::cos(dir);
            pw.ky = k * std::sin(dir);
            pw.phase = getAng(*it, "phase", 0);
            ws.waves.push_back(pw);
        }
    }
    buildWater(w_, name, ws, assembly, xf);
}

void Loader::cupBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    CupRod rod;
    if (auto r = b.get("rod")) {
        // rod = rod(diameter = 7mm, at = (x, y), top = 0.16, medium = N-BK7 | material = ...)
        rod.radius = getRadius(*r, "", 0.0035);
        if (auto at = r->get("at")) { const Vec2 q = lengths2(*at); rod.x = q.x; rod.y = q.y; }
        rod.top = getLen(*r, "top", getLen(b, "height", 0.11) + 0.05);
        rod.medium = getStr(*r, "medium", "");
        if (rod.medium.empty()) rod.optics = materialOr(*r, "material", "white");
        else w_.medium(rod.medium);
    }
    buildCup(w_, name, getRadius(b, "outer_", 0.035), getLen(b, "wall", 0.0025), getLen(b, "base", 0.008),
             getLen(b, "height", 0.11), getLen(b, "level", 0.07), getStr(b, "glass", "N-BK7"),
             getStr(b, "liquid", "water"), assembly, xf, rod);
}

void Loader::terrainBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    TerrainSpec t = terrainSpec(b);
    terrains_[name] = {t, xf};
    // A reference-only terrain is the height field that on_terrain() and forests stand on,
    // without matter: a detailed mesh elsewhere in the scene is the visible ground.
    if (auto r = b.get("reference_only"); r && boolean(*r)) return;
    BodyMaterial m = solidMaterial(b);
    int id = buildMesh(w_, name, makeTerrain(t), m, assembly, xf);
    w_.bodies()[id].kind = "terrain";
}

void Loader::forestBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    std::string tn = str(need(b, "terrain"));
    auto it = terrains_.find(tn);
    if (it == terrains_.end()) fail(b, "forest: unknown terrain '" + tn + "'");
    uint32_t fo = materialOr(b, "foliage", "white");
    uint32_t tr = materialOr(b, "trunk", "grey");
    buildForest(w_, name, it->second.first, int(getNum(b, "count", 200)), getLen(b, "inner_radius", 10),
                getLen(b, "outer_radius", 80), uint64_t(getNum(b, "seed", 1)), fo, tr, assembly, xf);
}

void Loader::statueBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    uint32_t o = materialOr(b, "material", "white");
    uint32_t po = b.has("pedestal") ? materialRef(need(b, "pedestal")) : o;
    buildFractalStatue(w_, name, getRadius(b, "", 0.1), int(getNum(b, "depth", 4)), getNum(b, "ratio", 1.0 / 3),
                       o, po, assembly, xf);
}

// Imported models arrive in arbitrary frames: `model_up` names the model axis that points up,
// `fit_height` rescales to a physical height, and the result stands on the body origin.
// normals = flat (the default: each triangle its own plane) | smooth | smooth(crease = 30deg) |
// file (the OBJ's vn). Smooth normals interpolate across every edge whose faces meet at less
// than the crease angle; the file's normals are the model maker's.
std::vector<MeshShape::Corners> Loader::meshNormals(const Value& b, const MeshData& md) {
    auto v = b.get("normals");
    if (!v || (v->kind == Value::Kind::Ident && v->str == "flat")) return {};
    if (v->str == "smooth" && (v->kind == Value::Kind::Ident || v->kind == Value::Kind::Call)) {
        double crease = v->kind == Value::Kind::Call ? getAng(*v, "crease", radians(30)) : radians(30);
        if (!(crease > 0 && crease <= Pi)) fail(*v, "crease must be an angle in (0, 180deg]");
        return smoothNormals(md, crease);
    }
    if (v->kind == Value::Kind::Ident && v->str == "file") {
        try {
            return fileNormals(md);
        } catch (const std::exception& e) {
            fail(*v, e.what());
        }
    }
    fail(*v, "normals must be flat, smooth, smooth(crease = ...) or file");
}

void Loader::prepareImportedMesh(const Value& b, MeshData& md) {
    if (auto u = b.get("model_up")) {
        // The model's up, measured or a named axis, turned to +z by the minimal rotation (which
        // keeps the facing).
        Vec3 up;
        if (u->kind == Value::Kind::Tuple) {
            up = normalize(vecNum(*u));
        } else {
            static const std::map<std::string, Vec3> axes = {{"x", {1, 0, 0}}, {"-x", {-1, 0, 0}}, {"y", {0, 1, 0}},
                                                             {"-y", {0, -1, 0}}, {"z", {0, 0, 1}}, {"-z", {0, 0, -1}}};
            auto a = axes.find(str(*u));
            if (a == axes.end()) fail(*u, "model_up must be one of x, -x, y, -y, z, -z");
            up = a->second;
        }
        const Transform R = Transform::rotateTo(up, {0, 0, 1});
        for (auto& p : md.positions) p = R.point(p);
        for (auto& n : md.normals) n = R.vector(n);
    }
    if (b.has("fit_height") || b.has("ground")) {
        AABB box;
        for (auto& p : md.positions) box.expand(p);
        double s = b.has("fit_height") ? len(need(b, "fit_height")) / box.extent().z : 1.0;
        Vec3 foot{box.centroid().x, box.centroid().y, box.lo.z};
        for (auto& p : md.positions) p = (p - foot) * s;
    }
}

void Loader::columnBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    // Doric column: fluted shaft with entasis, echinus (lathe) and square abacus, stacked
    // with 10 µm clearances so the three solids never share a face.
    double H = len(need(b, "height"));
    double r0 = getRadius(b, "", H / 14);
    double capH = getLen(b, "capital_height", 2 * r0 * 0.5);
    double shaftH = H - capH;
    double r1 = r0 * getNum(b, "taper", 0.78);
    uint32_t o = materialOr(b, "material", "white");
    int body = w_.addBody(name, "column", assembly, xf);
    uint32_t inside = w_.addRegion(name + ".stone", w_.medium("opaque"), body);
    MeshData shaft = makeFlutedShaft(shaftH, r0, r1, getNum(b, "entasis", 0.02), int(getNum(b, "flutes", 20)),
                                     getNum(b, "flute_depth", 0.035), 8, 24);
    w_.addBoundary(body, "shaft", std::make_shared<MeshShape>(shaft.positions, shaft.triangles, "shaft"), Transform{},
                   kOutside, inside, o);
    double gap = 1e-5, eH = capH * 0.45, aH = capH * 0.55 - 2 * gap;
    double a = r0 * 1.22;  // half-width of the abacus
    MeshData ech = revolveProfile({{r1 * 0.98, 0}, {r1 * 1.02, eH * 0.15}, {a * 0.98, eH * 0.8}, {a * 0.99, eH}}, 64);
    w_.addBoundary(body, "echinus", std::make_shared<MeshShape>(ech.positions, ech.triangles, "echinus"),
                   Transform::translate({0, 0, shaftH + gap}), kOutside, inside, o);
    const MeshData ab = makeBoxMesh({-a, -a, 0}, {a, a, aH});
    w_.addBoundary(body, "abacus", std::make_shared<MeshShape>(ab.positions, ab.triangles, "abacus"),
                   Transform::translate({0, 0, shaftH + eH + 2 * gap}), kOutside, inside, o);
}

TerrainSpec Loader::terrainSpec(const Value& b) const {
    TerrainSpec t;
    if (auto s = b.get("size")) {
        auto q = seq(*s, 2);
        t.sizeX = len(*q[0]);
        t.sizeY = len(*q[1]);
    }
    t.resolution = int(getNum(b, "resolution", 256));
    t.amplitude = getLen(b, "amplitude", 30);
    t.featureSize = getLen(b, "feature", 80);
    t.ridge = getNum(b, "ridge", 0);
    t.seed = uint64_t(getNum(b, "seed", 1));
    t.flatRadius = getLen(b, "flat_radius", 0);
    return t;
}

void Loader::lensBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    LensSpec spec;
    double r = getRadius(b, "", 0.0125);
    if (b.has("focal")) {
        spec = designSimpleLens(len(need(b, "focal")), getStr(b, "medium", "N-BK7"), 2 * r, getLen(b, "thickness", 0.2 * r),
                                getStr(b, "form", "bi"));
    } else if (b.has("surfaces")) {
        const Value& S = need(b, "surfaces");
        for (auto& s : S.items) spec.surfaces.push_back(surface(*s, r));
        for (auto& m : need(b, "media").items) spec.media.push_back(str(*m));
        for (auto& t : need(b, "thicknesses").items) spec.thickness.push_back(len(*t));
    } else {
        spec.media = {str(need(b, "medium"))};
        spec.surfaces = {surface(need(b, "front"), getRadius(b, "front_", r)), surface(need(b, "back"), getRadius(b, "back_", r))};
        spec.thickness = {len(need(b, "thickness"))};
    }
    spec.rim = rimKind(b, "ground");
    spec.edgeRadius = getRadius(b, "edge_", 0);
    for (auto& m : spec.media) w_.medium(m);
    buildLens(w_, name, spec, assembly, xf);
}

void Loader::prescriptionBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
    Prescription p = loadPrescription(path(str(need(b, "file"))));
    IndexFn index = catalogIndex();
    if (auto a = b.get("afocal"); a && boolean(*a)) solveAfocal(p, p.wavelength, index);
    double tube = 0;
    if (auto t = b.get("tube"); t && t->kind == Value::Kind::Ident) {
        if (boolean(*t)) {
            tube = maxSemiDiameter(p) * 1.1 + 2e-3;
        }
    } else if (auto tr = b.get("tube_radius")) {
        tube = len(*tr);
    }
        if (tube > 0 && hasMirrors(p))
        fail(b, "tube = true is not supported for mirror prescriptions yet: use tube = false and a housing");
    auto mirrorMaterial = [&](const std::string& n) -> uint32_t {
        auto it = materials_.find(n);
        if (it == materials_.end()) fail(b, "the lens file's mirror material '" + n + "' is not defined in the scene");
        const SurfaceType t = w_.optics()[it->second].type;
        if (t != SurfaceType::Mirror && t != SurfaceType::Conductor) fail(b, "mirror material '" + n + "' must be a mirror or conductor");
        return it->second;
    };
    BuiltInstrument bi = buildPrescription(w_, p, name, assembly, xf, tube, index,
                                           rimKind(b, "black"),
                                           mirrorMaterial);
    if (tube > 0) {
        double z0 = -0.01 * bi.lastVertexZ - 1e-3, z1 = bi.lastVertexZ + 1e-3;
        // Blackened inside; an optional finish (brass, paint) on a slightly larger outer shell.
        buildTube(w_, name + ".tube", tube, z0, z1, bi.assembly, Transform{}, materials_.at("black"));
        if (b.has("tube_material"))
            buildTube(w_, name + ".tube-finish", tube * 1.03, z0, z1, bi.assembly, Transform{},
                      materialRef(need(b, "tube_material")));
    }
    if (auto cup = b.get("eyecup"); cup && boolean(*cup)) {
        // A physical light shield on the eye side. The virtual observer has no face
        // to shade the glass; without a cup, bright room light reflects into the eye.
        double inner = bi.rearEdgeRadius + 0.001;
        double outer = inner + 0.002;
        double z0 = bi.rearNearestZ + 0.0001;
        double z1 = std::max(z0 + 0.001, bi.paraxial.exitPupilZ + 0.002);
        buildTube(w_, name + ".eyecup-inner", inner, z0, z1, bi.assembly, Transform{}, materials_.at("black"));
        buildTube(w_, name + ".eyecup-outer", outer, z0, z1, bi.assembly, Transform{}, materials_.at("black"));
        buildStop(w_, name + ".eyecup-base", bi.rearEdgeRadius * (1 + 1e-9), outer, bi.assembly,
                  Transform::translate({0, 0, z0}));
        buildStop(w_, name + ".eyecup-lip", inner, outer, bi.assembly,
                  Transform::translate({0, 0, z1}));
    }
    instruments_[name] = Instrument{bi.assembly, bi.paraxial};
    if (auto a = b.get("afocal"); a && boolean(*a) && bi.paraxial.exitPupilRadius > 0)
        w_.addExitPupil({name, bi.assembly, bi.paraxial.exitPupilZ, bi.paraxial.exitPupilRadius});
}

}  // namespace owe::detail
