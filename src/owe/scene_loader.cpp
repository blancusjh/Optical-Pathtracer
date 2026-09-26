#include "scene_loader.hpp"

#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>

#include "builders.hpp"

namespace owe {

Transform cameraPlacement(const Vec3& position, const Vec3& lookAt, const Vec3& up) {
    Vec3 f = normalize(lookAt - position);
    return Transform::translate(position) * Transform::alignZ(-f, up);
}

int addPhysicalCamera(Scene& scene, const std::string& name, const Prescription& lensIn, const CameraParams& c,
                      const Transform& placement, int parentAssembly) {
    World& w = scene.world;
    Prescription lens = lensIn;
    lens.objectDistance = c.focusDistance;
    IndexFn index = catalogIndex();
    Paraxial px = paraxialAnalysis(lens, lens.wavelength, index);
    if (px.afocal || std::isinf(px.imageDistance) || px.imageDistance <= 0)
        throw std::runtime_error("camera '" + name + "': lens does not form a real image for the requested focus");
    // The housing must enclose both the lens and the sensor; mounts close the gap between them.
    double maxR = 0;
    for (auto& s : lens.surfaces) maxR = std::max(maxR, s.sd);
    double halfDiag = 0.5 * std::sqrt(sqr(c.sensorWidth) + sqr(c.sensorHeight));
    double tubeR = c.housing ? std::max(maxR * 1.08 + 1e-3, halfDiag * 1.05) : 0.0;
    int cam = w.addAssembly(name, parentAssembly, placement);
    BuiltInstrument bi = buildPrescription(w, lens, name + ".lens", cam, Transform{}, tubeR, index);
    double zSensor = bi.lastVertexZ + px.imageDistance + c.sensorShift;
    if (c.housing) {
        uint32_t black = w.absorberOptics();
        double zFront = -1e-3;
        buildTube(w, name + ".housing", tubeR, zFront - 0.02 * maxR, zSensor + 2e-3, cam, Transform{}, black);
        buildStop(w, name + ".back", 0, tubeR, cam, Transform::translate({0, 0, zSensor + 2e-3}));
        w.bodies().back().kind = "housing";
    }
    // Sensor surface facing the lens: rotated π about y, so sensor x = −x_cam, y = y_cam, normal −z.
    // The lens inverts the image; reading rows bottom-up (flipY) yields an upright, unmirrored picture.
    int sensorBody = w.addBody(name + ".sensor", "sensor", cam, Transform::translate({0, 0, zSensor}));
    int detIndex = int(scene.detectors.size());
    SurfaceOptics so;
    so.name = name + ".sensor";
    so.type = SurfaceType::Detector;
    so.detector = detIndex;
    uint32_t oi = w.addOptics(so);
    uint32_t bidx = w.addBoundary(sensorBody, "pixels", PlaneShape::rect(c.sensorWidth / 2, c.sensorHeight / 2),
                                  Transform::rotate({0, 1, 0}, Pi), kOutside, kOutside, oi);
    auto det = std::make_unique<SurfaceSensor>();
    det->name = name;
    det->width = c.width;
    det->height = c.height;
    det->boundaryIndex = bidx;
    det->halfX = c.sensorWidth / 2;
    det->halfY = c.sensorHeight / 2;
    det->flipY = true;
    det->hasAim = true;
    Transform camToWorld = w.assemblyToWorld(cam);
    det->aimCenter = camToWorld.point({0, 0, bi.rearNearestZ});
    det->aimNormal = normalize(camToWorld.vector({0, 0, 1}));
    det->aimRadius = std::max(halfDiag, bi.rearEdgeRadius) * 1.0001;
    scene.detectors.push_back(std::move(det));
    w.bodies()[sensorBody].params["z"] = zSensor;
    return detIndex;
}

namespace {

struct Instrument {
    int assembly = -1;
    Paraxial paraxial;
};

class Loader {
public:
    Loader(Scene& s, std::string baseDir) : scene_(s), w_(s.world), base_(std::move(baseDir)) { registerDefaults(); }

    void run(const Value& doc) {
        for (auto& [k, v] : doc.named) {
            if (k == "units") unit_ = lengthUnit(*v);
            else fail(*v, "unknown top-level setting '" + k + "'");
        }
        for (auto& b : doc.items) statement(*b, -1);
    }

private:
    // ------------------------------------------------------------ errors & values
    [[noreturn]] void fail(const Value& v, const std::string& m) const {
        throw std::runtime_error(scene_.sourcePath + ": line " + std::to_string(v.line) + ": " + m);
    }
    double lengthUnit(const Value& v) {
        std::string s = str(v);
        if (s == "m") return 1;
        if (s == "mm") return 1e-3;
        if (s == "cm") return 1e-2;
        if (s == "um") return 1e-6;
        if (s == "km") return 1e3;
        fail(v, "unknown length unit '" + s + "'");
    }
    std::string str(const Value& v) const {
        if (v.kind == Value::Kind::String || v.kind == Value::Kind::Ident) return v.str;
        fail(v, "expected a name or string");
    }
    double num(const Value& v) const {
        if (v.kind != Value::Kind::Number) fail(v, "expected a number");
        return v.num;
    }
    double len(const Value& v) const {
        if (v.kind != Value::Kind::Number) fail(v, "expected a length");
        if (v.unit == Value::Unit::Length) return v.num;
        if (v.unit != Value::Unit::None) fail(v, "expected a length unit");
        return v.num * unit_;
    }
    double ang(const Value& v) const {  // unitless angles are degrees
        if (v.kind != Value::Kind::Number) fail(v, "expected an angle");
        if (v.unit == Value::Unit::Angle) return v.num;
        if (v.unit != Value::Unit::None) fail(v, "expected an angle unit");
        return radians(v.num);
    }
    double nm(const Value& v) const {  // wavelengths: unitless = nm
        if (v.unit == Value::Unit::Length) return v.num * 1e9;
        return num(v);
    }
    bool boolean(const Value& v) const {
        std::string s = str(v);
        if (s == "true" || s == "yes" || s == "on") return true;
        if (s == "false" || s == "no" || s == "off") return false;
        fail(v, "expected true or false");
    }
    std::vector<ValuePtr> seq(const Value& v, size_t n) const {
        if ((v.kind != Value::Kind::Tuple && v.kind != Value::Kind::List) || (n && v.items.size() != n))
            fail(v, "expected a tuple of " + std::to_string(n) + " values");
        return v.items;
    }
    Vec3 vecLen(const Value& v) {
        if (v.kind == Value::Kind::Call && v.str == "exit_pupil") return exitPupil(v);
        auto s = seq(v, 3);
        return {len(*s[0]), len(*s[1]), len(*s[2])};
    }
    Vec3 vecNum(const Value& v) const {
        auto s = seq(v, 3);
        return {num(*s[0]), num(*s[1]), num(*s[2])};
    }
    double getLen(const Value& b, const std::string& k, double def) const { auto v = b.get(k); return v ? len(*v) : def; }
    double getNum(const Value& b, const std::string& k, double def) const { auto v = b.get(k); return v ? num(*v) : def; }
    double getAng(const Value& b, const std::string& k, double def) const { auto v = b.get(k); return v ? ang(*v) : def; }
    std::string getStr(const Value& b, const std::string& k, const std::string& def) const {
        auto v = b.get(k);
        return v ? str(*v) : def;
    }
    const Value& need(const Value& b, const std::string& k) const {
        auto v = b.get(k);
        if (!v) fail(b, "'" + b.str + " " + b.name + "' needs '" + k + "'");
        return *v;
    }
    // Diameter or radius: accepts `diameter`/`<prefix>diameter` or `radius`.
    double getRadius(const Value& b, const std::string& prefix, double def) const {
        if (auto v = b.get(prefix + "diameter")) return len(*v) / 2;
        if (auto v = b.get(prefix + "radius")) return len(*v);
        return def;
    }

    std::string path(const std::string& p) const {
        if (!p.empty() && p[0] == '/') return p;
        return base_ + "/" + p;
    }

    // ------------------------------------------------------------ spectra
    Spectrum spectrum(const Value& v, bool emission) const {
        if (v.kind == Value::Kind::Number) return Spectrum::constant(v.num);
        if (v.kind == Value::Kind::Call) {
            auto arg = [&](size_t i, const std::string& k) -> ValuePtr {
                if (auto x = v.get(k)) return x;
                return i < v.items.size() ? v.items[i] : nullptr;
            };
            double scale = 1;
            if (auto s = v.get("scale")) scale = num(*s);
            if (v.str == "rgb") {
                if (v.items.size() != 3) fail(v, "rgb needs three components");
                double r = num(*v.items[0]), g = num(*v.items[1]), b = num(*v.items[2]);
                if (auto y = v.get("luminance")) scale = num(*y);
                return (emission ? Spectrum::rgbIlluminant(r, g, b) : Spectrum::rgbReflectance(r, g, b)).scaled(scale);
            }
            if (v.str == "blackbody") {
                auto T = arg(0, "T");
                if (!T) fail(v, "blackbody needs a temperature");
                double y = 1;
                if (auto L = arg(1, "luminance")) y = num(*L);
                return Spectrum::blackbody(num(*T), y).scaled(scale);
            }
            if (v.str == "constant") {
                auto c = arg(0, "value");
                if (!c) fail(v, "constant needs a value");
                return Spectrum::constant(num(*c) * scale);
            }
            if (v.str == "tabulated") {
                auto L = arg(0, "lambda"), V = arg(1, "values");
                if (!L || !V) fail(v, "tabulated needs lambda=[...] and values=[...]");
                std::vector<double> ls, vs;
                for (auto& x : L->items) ls.push_back(nm(*x));
                for (auto& x : V->items) vs.push_back(num(*x));
                return Spectrum::tabulated(ls, vs).scaled(scale);
            }
            if (v.str == "d65" || v.str == "daylight") {
                double y = 1;
                if (auto L = arg(0, "luminance")) y = num(*L);
                return Spectrum::rgbIlluminant(1, 1, 1).scaled(y * scale);
            }
        }
        fail(v, "expected a spectrum: number, rgb(...), blackbody(T, luminance), tabulated(...) or constant(...)");
    }

    Texture texture(const Value& v) const {
        Texture t;
        if (v.kind != Value::Kind::Call) fail(v, "expected a texture call");
        if (v.str == "checker") t.kind = Texture::Kind::Checker;
        else if (v.str == "noise") t.kind = Texture::Kind::Noise;
        else if (v.str == "rings" || v.str == "wood") t.kind = Texture::Kind::Rings;
        else if (v.str == "terrain") t.kind = Texture::Kind::Terrain;
        else fail(v, "unknown texture '" + v.str + "'");
        if (auto a = v.get("a")) t.a = spectrum(*a, false);
        if (auto b = v.get("b")) t.b = spectrum(*b, false);
        if (auto c = v.get("c")) t.c = spectrum(*c, false);
        if (auto s = v.get("scale")) t.scale = len(*s);
        if (auto p = v.get("snow_line")) t.param = len(*p);
        if (auto p = v.get("turbulence")) t.param = num(*p);
        return t;
    }

    // ------------------------------------------------------------ registries
    void registerDefaults() {
        auto add = [&](const std::string& n, SurfaceOptics o) { o.name = n; materials_[n] = w_.addOptics(o); };
        SurfaceOptics o;
        o.type = SurfaceType::Diffuse; o.reflectance = Spectrum::constant(0.8); add("white", o);
        o.reflectance = Spectrum::constant(0.18); add("grey", o);
        o = SurfaceOptics{}; o.type = SurfaceType::Absorber; add("black", o);
        o = SurfaceOptics{}; o.type = SurfaceType::Mirror; o.reflectance = Spectrum::constant(1.0); add("ideal_mirror", o);
        for (const std::string m : {"aluminium", "silver", "gold", "copper"}) {
            o = SurfaceOptics{};
            o.type = SurfaceType::Conductor;
            catalogConductor(m, o.conductor);
            add(m, o);
        }
        materials_["aluminum"] = materials_["aluminium"];
    }

    uint32_t materialRef(const Value& v) {
        if (v.kind == Value::Kind::Block) return defineMaterial(v, "");
        std::string n = str(v);
        auto it = materials_.find(n);
        if (it == materials_.end()) fail(v, "unknown material '" + n + "'");
        return it->second;
    }

    uint32_t defineMaterial(const Value& b, const std::string& name) {
        SurfaceOptics o;
        o.name = name.empty() ? "anonymous" : name;
        std::string type = getStr(b, "type", "diffuse");
        if (type == "diffuse" || type == "lambertian") {
            o.type = SurfaceType::Diffuse;
            o.reflectance = b.has("reflectance") ? spectrum(need(b, "reflectance"), false) : Spectrum::constant(0.8);
            if (auto t = b.get("texture")) o.texture = texture(*t);
        } else if (type == "mirror") {
            o.type = SurfaceType::Mirror;
            o.reflectance = b.has("reflectance") ? spectrum(need(b, "reflectance"), false) : Spectrum::constant(1.0);
        } else if (type == "conductor" || type == "metal") {
            o.type = SurfaceType::Conductor;
            std::string m = getStr(b, "metal", "aluminium");
            if (!catalogConductor(m, o.conductor)) fail(b, "unknown metal '" + m + "'");
            o.roughness = getNum(b, "roughness", 0);
            o.reflectance = b.has("tint") ? spectrum(need(b, "tint"), false) : Spectrum::constant(1.0);
        } else if (type == "dielectric") {
            o.type = SurfaceType::Dielectric;
            o.roughness = getNum(b, "roughness", 0);
        } else if (type == "absorber" || type == "black") {
            o.type = SurfaceType::Absorber;
        } else if (type == "null") {
            o.type = SurfaceType::Null;
        } else {
            fail(b, "unknown material type '" + type + "'");
        }
        if (auto bk = b.get("back")) o.backAbsorbs = str(*bk) == "black";
        uint32_t id = w_.addOptics(o);
        if (!name.empty()) materials_[name] = id;
        return id;
    }

    void defineMedium(const Value& b) {
        Medium m;
        if (auto base = b.get("base")) {
            if (!catalogMedium(str(*base), m)) fail(*base, "unknown base medium");
        }
        m.name = b.name;
        if (auto ix = b.get("index")) m.index = indexModel(*ix);
        if (auto a = b.get("absorption")) m.absorption = spectrum(*a, false);  // σₐ in 1/m
        if (auto t = b.get("internal_transmittance")) {
            // (τ, thickness): Beer–Lambert coefficient α = −ln τ / d, spectral if τ is a spectrum.
            auto s = seq(*t, 2);
            double d = len(*s[1]);
            if (s[0]->kind == Value::Kind::Number) m.absorption = Spectrum::constant(-std::log(num(*s[0])) / d);
            else {
                Spectrum tau = spectrum(*s[0], false);
                std::vector<double> ls, vs;
                for (double l = LambdaMin; l <= LambdaMax; l += 5) { ls.push_back(l); vs.push_back(-std::log(std::max(1e-12, tau(l))) / d); }
                m.absorption = Spectrum::tabulated(ls, vs);
            }
        }
        if (auto s = b.get("scattering")) m.scattering = spectrum(*s, false);
        if (auto g = b.get("g")) m.g = num(*g);
        w_.addMedium(m);
    }

    IndexModel indexModel(const Value& v) const {
        if (v.kind == Value::Kind::Number) return IndexModel::constant(v.num);
        if (v.kind != Value::Kind::Call) fail(v, "expected an index model");
        auto list = [&](const std::string& k) {
            std::vector<double> r;
            auto x = v.get(k);
            if (!x) fail(v, "missing '" + k + "'");
            for (auto& e : x->items) r.push_back(num(*e));
            return r;
        };
        IndexModel m;
        if (v.str == "constant") m = IndexModel::constant(num(*v.items.at(0)));
        else if (v.str == "cauchy") m = IndexModel::cauchy(getNum(v, "A", 1.5), getNum(v, "B", 0), getNum(v, "C", 0));
        else if (v.str == "sellmeier") m = IndexModel::sellmeier(list("B"), list("C"));
        else if (v.str == "tabulated") {
            std::vector<double> ls;
            for (auto& e : need(v, "lambda").items) ls.push_back(nm(*e));
            m = IndexModel::tabulated(ls, list("n"));
        } else fail(v, "unknown index model '" + v.str + "'");
        if (auto r = v.get("relative_to_air"); r && boolean(*r)) m = m.relativeToAir();
        return m;
    }

    // ------------------------------------------------------------ placement
    Transform placement(const Value& b) {
        Transform xf;
        Vec3 pos = b.has("position") ? vecLen(need(b, "position")) : Vec3();
        Transform R;
        if (auto r = b.get("rotate")) {
            auto s = seq(*r, 3);
            R = Transform::rotate({0, 0, 1}, ang(*s[2])) * Transform::rotate({0, 1, 0}, ang(*s[1])) *
                Transform::rotate({1, 0, 0}, ang(*s[0]));
        }
        Vec3 up = b.has("up") ? vecNum(need(b, "up")) : w_.env.up;
        if (auto a = b.get("axis")) R = Transform::alignZ(vecNum(*a), up) * R;
        else if (auto la = b.get("look_at")) R = Transform::alignZ(vecLen(*la) - pos, up) * R;
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

    Vec3 exitPupil(const Value& v) {
        if (v.items.empty()) fail(v, "exit_pupil needs an instrument name");
        std::string n = str(*v.items[0]);
        auto it = instruments_.find(n);
        if (it == instruments_.end()) fail(v, "exit_pupil: no instrument named '" + n + "' (define it first)");
        double offset = v.items.size() > 1 ? len(*v.items[1]) : 0.0;
        const Instrument& ins = it->second;
        return w_.assemblyToWorld(ins.assembly).point({0, 0, ins.paraxial.exitPupilZ + offset});
    }

    // ------------------------------------------------------------ surfaces
    SurfaceSpec surface(const Value& v, double semiDiameter) const {
        SurfaceSpec s;
        s.semiDiameter = semiDiameter;
        if (v.kind != Value::Kind::Call) fail(v, "expected a surface: plane(), sphere(R=...), conic(R=..., k=...), asphere(...)");
        if (v.str == "plane" || v.str == "flat") return s;
        auto R = v.get("R");
        if (!R && !v.items.empty()) R = v.items[0];
        if (!R) fail(v, "surface needs R");
        s.R = R->kind == Value::Kind::Number && std::isinf(R->num) ? Inf : len(*R);
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

    BodyMaterial solidMaterial(const Value& b) {
        BodyMaterial m;
        if (auto med = b.get("medium")) {
            m.transparent = true;
            m.medium = str(*med);
            w_.medium(m.medium);  // validate / load
            if (w_.media()[w_.medium(m.medium)].opaque) {
                m.transparent = false;
                m.optics = materials_.at("black");
            } else {
                m.optics = w_.dielectricOptics(getNum(b, "roughness", 0));
            }
        } else if (auto mat = b.get("material")) {
            m.optics = materialRef(*mat);
            if (w_.optics()[m.optics].type == SurfaceType::Dielectric) fail(*mat, "use 'medium' for transparent bodies");
        } else {
            m.optics = materials_.at("white");
        }
        if (auto e = b.get("emission")) {
            Emission em;
            em.radiance = spectrum(*e, true);
            std::string sides = getStr(b, "emission_sides", "front");
            em.front = sides == "front" || sides == "both";
            em.back = sides == "back" || sides == "both";
            m.emission = w_.addEmission(em);
        }
        return m;
    }

    // ------------------------------------------------------------ statements
    void statement(const Value& b, int assembly) {
        const std::string& kw = b.str;
        if (kw == "medium") defineMedium(b);
        else if (kw == "material") defineMaterial(b, b.name);
        else if (kw == "world") worldBlock(b);
        else if (kw == "body") body(b, assembly);
        else if (kw == "assembly") assemblyBlock(b, assembly);
        else if (kw == "observer") observer(b);
        else if (kw == "camera") camera(b, assembly);
        else if (kw == "sensor") sensor(b, assembly);
        else if (kw == "render") renderBlock(b);
        else fail(b, "unknown block '" + kw + "'");
    }

    void worldBlock(const Value& b) {
        Environment& e = w_.env;
        if (auto m = b.get("medium")) w_.setAmbientMedium(w_.medium(str(*m)));
        if (auto u = b.get("up")) e.up = normalize(vecNum(*u));
        if (auto s = b.get("sky")) {
            if (s->kind == Value::Kind::Call && s->str == "gradient") {
                e.skyModel = Environment::Sky::Gradient;
                e.zenith = spectrum(need(*s, "zenith"), true);
                e.horizon = spectrum(need(*s, "horizon"), true);
                e.ground = s->has("ground") ? spectrum(need(*s, "ground"), true) : Spectrum::constant(0);
            } else if (s->kind == Value::Kind::Call && s->str == "uniform") {
                e.skyModel = Environment::Sky::Uniform;
                e.zenith = spectrum(*s->items.at(0), true);
            } else if (s->kind == Value::Kind::Ident && s->str == "none") {
                e.skyModel = Environment::Sky::None;
            } else {
                e.skyModel = Environment::Sky::Uniform;
                e.zenith = spectrum(*s, true);
            }
        }
        for (auto& c : b.items) {
            if (c->str != "sun") fail(*c, "unknown block in world: " + c->str);
            e.hasSun = true;
            if (auto d = c->get("direction")) e.sunDir = normalize(vecNum(*d));
            else {
                double el = getAng(*c, "elevation", radians(45)), az = getAng(*c, "azimuth", radians(0));
                e.sunDir = normalize(Vec3(std::cos(el) * std::sin(az), std::cos(el) * std::cos(az), std::sin(el)));
            }
            e.sunAngularRadius = getAng(*c, "angular_diameter", radians(0.533)) / 2;
            if (auto r = c->get("radiance")) e.sunRadiance = spectrum(*r, true);
            else e.sunRadiance = Spectrum::blackbody(getNum(*c, "temperature", 5778), getNum(*c, "luminance", 1e5));
        }
    }

    void assemblyBlock(const Value& b, int parent) {
        int a = w_.addAssembly(b.name, parent, placement(b));
        for (auto& c : b.items) statement(*c, a);
    }

    void body(const Value& b, int assembly) {
        std::string type = getStr(b, "type", "");
        const std::string& name = b.name;
        if (name.empty()) fail(b, "bodies need a name");
        if (!names_.insert(name).second) fail(b, "duplicate body name '" + name + "'");
        Transform xf = placement(b);
        if (type == "lens") lensBody(b, assembly, xf);
        else if (type == "prescription") prescriptionBody(b, assembly, xf);
        else if (type == "mirror") {
            SurfaceSpec s = surface(need(b, "surface"), getRadius(b, "", 0.05));
            uint32_t o = b.has("material") ? materialRef(need(b, "material")) : materials_.at("aluminium");
            buildMirror(w_, name, s, getLen(b, "thickness", 0.1 * s.semiDiameter * 2), o, assembly, xf,
                        getRadius(b, "hole_", 0));
        } else if (type == "flat_mirror") {
            uint32_t o = b.has("material") ? materialRef(need(b, "material")) : materials_.at("aluminium");
            if (auto e = b.get("ellipse")) {
                auto s = seq(*e, 2);
                buildFlatMirror(w_, name, len(*s[0]) / 2, len(*s[1]) / 2, true, o, assembly, xf);
            } else {
                auto s = seq(need(b, "size"), 2);
                buildFlatMirror(w_, name, len(*s[0]) / 2, len(*s[1]) / 2, false, o, assembly, xf);
            }
        } else if (type == "stop" || type == "iris" || type == "aperture") {
            double inner = getRadius(b, "aperture_", -1);
            if (inner < 0) inner = len(need(b, "aperture")) / 2;
            buildStop(w_, name, inner, getRadius(b, "outer_", inner * 4), assembly, xf);
        } else if (type == "tube") {
            buildTube(w_, name, getRadius(b, "", 0.05), 0, len(need(b, "length")), assembly, xf,
                      b.has("material") ? materialRef(need(b, "material")) : materials_.at("black"));
        } else if (type == "sphere") {
            buildSphere(w_, name, getRadius(b, "", 0.01), solidMaterial(b), assembly, xf);
        } else if (type == "box") {
            buildBox(w_, name, vecLen(need(b, "size")), solidMaterial(b), assembly, xf);
        } else if (type == "cylinder") {
            buildCylinder(w_, name, getRadius(b, "", 0.01), len(need(b, "height")), solidMaterial(b), assembly, xf);
        } else if (type == "prism") {
            buildPrism(w_, name, getAng(b, "apex", radians(60)), len(need(b, "side")), len(need(b, "length")),
                       solidMaterial(b), assembly, xf);
        } else if (type == "sheet" || type == "screen" || type == "disk") {
            BodyMaterial m = solidMaterial(b);
            if (type == "disk" || b.has("radius") || b.has("diameter")) {
                double r = getRadius(b, "", 0.1);
                buildSheet(w_, name, r, r, m.optics, m.emission, assembly, xf, true);
            } else {
                auto s = seq(need(b, "size"), 2);
                buildSheet(w_, name, len(*s[0]) / 2, len(*s[1]) / 2, m.optics, m.emission, assembly, xf, false);
            }
        } else if (type == "mesh") {
            MeshData md = loadObj(path(str(need(b, "file"))), getNum(b, "scale", 1.0) * unit_);
            buildMesh(w_, name, md, solidMaterial(b), assembly, xf);
        } else if (type == "cup") {
            buildCup(w_, name, getRadius(b, "outer_", 0.035), getLen(b, "wall", 0.0025), getLen(b, "base", 0.008),
                     getLen(b, "height", 0.11), getLen(b, "level", 0.07), getStr(b, "glass", "N-BK7"),
                     getStr(b, "liquid", "water"), assembly, xf);
        } else if (type == "terrain") {
            TerrainSpec t = terrainSpec(b);
            terrains_[name] = t;
            BodyMaterial m = solidMaterial(b);
            int id = buildMesh(w_, name, makeTerrain(t), m, assembly, xf);
            w_.bodies()[id].kind = "terrain";
        } else if (type == "forest") {
            std::string tn = str(need(b, "terrain"));
            auto it = terrains_.find(tn);
            if (it == terrains_.end()) fail(b, "forest: unknown terrain '" + tn + "'");
            uint32_t fo = b.has("foliage") ? materialRef(need(b, "foliage")) : materials_.at("white");
            uint32_t tr = b.has("trunk") ? materialRef(need(b, "trunk")) : materials_.at("grey");
            buildForest(w_, name, it->second, int(getNum(b, "count", 200)), getLen(b, "inner_radius", 10),
                        getLen(b, "outer_radius", 80), uint64_t(getNum(b, "seed", 1)), fo, tr, assembly, xf);
        } else if (type == "fractal_statue" || type == "statue") {
            uint32_t o = b.has("material") ? materialRef(need(b, "material")) : materials_.at("white");
            uint32_t po = b.has("pedestal") ? materialRef(need(b, "pedestal")) : o;
            buildFractalStatue(w_, name, getRadius(b, "", 0.1), int(getNum(b, "depth", 4)), getNum(b, "ratio", 1.0 / 3),
                               o, po, assembly, xf);
        } else {
            fail(b, "unknown body type '" + type + "'");
        }
    }

    TerrainSpec terrainSpec(const Value& b) const {
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

    void lensBody(const Value& b, int assembly, const Transform& xf) {
        LensSpec spec;
        std::string rim = getStr(b, "rim", "ground");
        spec.rim = rim == "black" ? LensSpec::Rim::Black : rim == "polished" ? LensSpec::Rim::Polished : LensSpec::Rim::Ground;
        double r = getRadius(b, "", 0.0125);
        if (b.has("focal")) {
            spec = designSimpleLens(len(need(b, "focal")), getStr(b, "medium", "N-BK7"), 2 * r, getLen(b, "thickness", 0.2 * r),
                                    getStr(b, "form", "bi"));
            spec.rim = rim == "black" ? LensSpec::Rim::Black : rim == "polished" ? LensSpec::Rim::Polished : LensSpec::Rim::Ground;
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
        spec.edgeRadius = getRadius(b, "edge_", 0);
        for (auto& m : spec.media) w_.medium(m);
        buildLens(w_, b.name, spec, assembly, xf);
    }

    void prescriptionBody(const Value& b, int assembly, const Transform& xf) {
        Prescription p = loadPrescription(path(str(need(b, "file"))));
        IndexFn index = catalogIndex();
        if (auto a = b.get("afocal"); a && boolean(*a)) solveAfocal(p, p.wavelength, index);
        double tube = 0;
        if (auto t = b.get("tube"); t && t->kind == Value::Kind::Ident) {
            if (boolean(*t)) {
                double maxR = 0;
                for (auto& s : p.surfaces) maxR = std::max(maxR, s.sd);
                tube = maxR * 1.1 + 2e-3;
            }
        } else if (auto tr = b.get("tube_radius")) {
            tube = len(*tr);
        }
        std::string rim = getStr(b, "rim", "black");
        BuiltInstrument bi = buildPrescription(w_, p, b.name, assembly, xf, tube, index,
                                               rim == "ground" ? LensSpec::Rim::Ground
                                               : rim == "polished" ? LensSpec::Rim::Polished : LensSpec::Rim::Black);
        if (tube > 0) {
            double z0 = -0.01 * bi.lastVertexZ - 1e-3, z1 = bi.lastVertexZ + 1e-3;
            buildTube(w_, b.name + ".tube", tube, z0, z1, bi.assembly, Transform{}, materials_.at("black"));
        }
        instruments_[b.name] = Instrument{bi.assembly, bi.paraxial};
    }

    // ------------------------------------------------------------ detectors
    std::pair<int, int> resolution(const Value& b) const {
        auto v = b.get("resolution");
        if (!v) return {320, 240};
        auto s = seq(*v, 2);
        return {int(num(*s[0])), int(num(*s[1]))};
    }

    void observer(const Value& b) {
        auto o = std::make_unique<IdealObserver>();
        o->name = b.name;
        auto [w, h] = resolution(b);
        o->width = w;
        o->height = h;
        o->position = vecLen(need(b, "position"));
        if (auto la = b.get("look_at")) o->lookAt = vecLen(*la);
        else if (auto d = b.get("direction")) o->lookAt = o->position + normalize(vecNum(*d));
        else fail(b, "observer needs look_at or direction");
        o->up = b.has("up") ? vecNum(need(b, "up")) : w_.env.up;
        o->fovY = getAng(b, "fov", radians(40));
        o->pupilRadius = getRadius(b, "pupil_", -1);
        if (o->pupilRadius < 0) o->pupilRadius = b.has("pupil") ? len(need(b, "pupil")) / 2 : 0;
        if (auto f = b.get("focus")) o->focusDistance = (f->kind == Value::Kind::Number && std::isinf(f->num)) ? Inf : len(*f);
        scene_.detectors.push_back(std::move(o));
    }

    void camera(const Value& b, int assembly) {
        Prescription p = loadPrescription(path(str(need(b, "lens"))));
        CameraParams c;
        if (auto s = b.get("sensor")) {
            auto q = seq(*s, 2);
            c.sensorWidth = len(*q[0]);
            c.sensorHeight = len(*q[1]);
        }
        auto [w, h] = resolution(b);
        c.width = w;
        c.height = h;
        if (auto f = b.get("focus")) c.focusDistance = (f->kind == Value::Kind::Number && std::isinf(f->num)) ? Inf : len(*f);
        c.sensorShift = getLen(b, "sensor_shift", 0);
        if (auto hs = b.get("housing")) c.housing = boolean(*hs);
        Vec3 pos = vecLen(need(b, "position"));
        Vec3 la = vecLen(need(b, "look_at"));
        Vec3 up = b.has("up") ? vecNum(need(b, "up")) : w_.env.up;
        addPhysicalCamera(scene_, b.name, p, c, cameraPlacement(pos, la, up), assembly);
    }

    void sensor(const Value& b, int assembly) {
        auto s = seq(need(b, "size"), 2);
        double hx = len(*s[0]) / 2, hy = len(*s[1]) / 2;
        int body = w_.addBody(b.name, "sensor", assembly, placement(b));
        int detIndex = int(scene_.detectors.size());
        SurfaceOptics so;
        so.name = b.name;
        so.type = SurfaceType::Detector;
        so.detector = detIndex;
        uint32_t bi = w_.addBoundary(body, "pixels", PlaneShape::rect(hx, hy), Transform{}, kOutside, kOutside, w_.addOptics(so));
        auto d = std::make_unique<SurfaceSensor>();
        d->name = b.name;
        auto [rw, rh] = resolution(b);
        d->width = rw;
        d->height = rh;
        d->boundaryIndex = bi;
        d->halfX = hx;
        d->halfY = hy;
        if (auto ac = b.get("aim_center")) {
            d->hasAim = true;
            d->aimCenter = vecLen(*ac);
            d->aimNormal = normalize(vecNum(need(b, "aim_axis")));
            d->aimRadius = getRadius(b, "aim_", 0.01);
        }
        scene_.detectors.push_back(std::move(d));
    }

    void renderBlock(const Value& b) {
        RenderSettings& r = scene_.render;
        r.detector = getStr(b, "detector", r.detector);
        r.integrator = getStr(b, "integrator", r.integrator);
        r.spp = int(getNum(b, "spp", r.spp));
        r.seed = uint64_t(getNum(b, "seed", double(r.seed)));
        r.maxDepth = int(getNum(b, "max_depth", r.maxDepth));
        r.exposure = getNum(b, "exposure", r.exposure);
        if (auto a = b.get("auto_exposure")) r.autoExposure = boolean(*a);
    }

    Scene& scene_;
    World& w_;
    std::string base_;
    double unit_ = 1.0;
    std::map<std::string, uint32_t> materials_;
    std::map<std::string, TerrainSpec> terrains_;
    std::map<std::string, Instrument> instruments_;
    std::set<std::string> names_;
};

}  // namespace

Scene loadSceneFromString(const std::string& text, const std::string& baseDir, const std::string& origin) {
    Scene scene;
    scene.sourcePath = origin;
    scene.sourceText = text;
    ValuePtr doc = parseSceneText(text, origin);
    Loader ld(scene, baseDir);
    ld.run(*doc);
    scene.build();
    if (scene.detectors.empty()) throw std::runtime_error(origin + ": scene defines no observer, camera or sensor");
    return scene;
}

Scene loadScene(const std::string& p) {
    std::ifstream in(p);
    if (!in) throw std::runtime_error("cannot open scene: " + p);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string dir = ".";
    auto slash = p.find_last_of('/');
    if (slash != std::string::npos) dir = p.substr(0, slash);
    return loadSceneFromString(ss.str(), dir, p);
}

}  // namespace owe
