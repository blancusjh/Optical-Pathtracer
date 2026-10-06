#include "owe/loader/scene_loader.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>

#include "owe/analysis/analysis.hpp"
#include "owe/scene/builders.hpp"

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
    if (c.fNumber > 0) {
        // Resize the physical aperture stop so that the entrance pupil gives EFL / (2N). The
        // entrance pupil scales linearly with the stop, paraxially. Lenses without a stop get
        // an iris 2 mm in front of their first surface.
        Prescription inf = lens;
        inf.objectDistance = Inf;
        int stop = -1;
        for (size_t i = 0; i < lens.surfaces.size(); ++i)
            if (lens.surfaces[i].stop) stop = int(i);
        if (stop < 0) {
            PrescriptionSurface iris;
            iris.label = "IRIS";
            iris.t = 2e-3;
            iris.stop = true;
            iris.sd = lens.surfaces[0].sd;
            lens.surfaces.insert(lens.surfaces.begin(), iris);
            inf.surfaces = lens.surfaces;
            stop = 0;
        }
        Paraxial p0 = paraxialAnalysis(inf, inf.wavelength, index);
        double target = std::abs(p0.efl) / (2 * c.fNumber);
        lens.surfaces[size_t(stop)].sd *= target / p0.entrancePupilRadius;
    }
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
    // Focus like a real camera: at the plane of least blur for this aperture (traced with real
    // rays through the lens bodies), not at the paraxial image. The correction is the on-axis
    // best-focus shift measured for a distant object.
    double focusShift = 0;
    if (c.realFocus) {
        Prescription probe = lens;
        probe.objectDistance = Inf;
        LensReport rep = analyzeLens(probe, {0}, {probe.wavelength}, 6);
        if (!rep.spots.empty() && rep.spots[0].arrived >= 3 && !std::isinf(rep.paraxial.imageDistance))
            focusShift = rep.spots[0].bestFocusZ - rep.paraxial.imageDistance;
    }
    double zSensor = bi.lastVertexZ + px.imageDistance + focusShift + c.sensorShift;
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
    det->aimNormal = normalize(camToWorld.vector({0, 0, 1}));
    det->aimCenter = camToWorld.point({0, 0, bi.rearNearestZ});
    det->aimRadius = std::max(halfDiag, bi.rearEdgeRadius) * 1.0001;
    // Image-forming light reaches the sensor from the exit pupil (the stop imaged by the rear
    // group), so 85% of samples aim there; at f/11 that wastes ~30× fewer rays on the iris. The
    // rest cover the whole rear opening, which also carries ghost reflections and veiling glare.
    double zExit = bi.vertexZ.empty() ? 0 : bi.vertexZ[0] + px.exitPupilZ;
    if (px.exitPupilRadius > 0 && zExit < zSensor - 1e-3) {
        det->focusCenter = camToWorld.point({0, 0, zExit});
        det->focusRadius = px.exitPupilRadius * 1.3;
        det->focusShare = 0.85;
    }
    scene.detectors.push_back(std::move(det));
    w.bodies()[sensorBody].params["z"] = zSensor;
    {
        // Depth of field from thin-lens relations with a circle of confusion of 1/1500 of the diagonal.
        Prescription inf = lens;
        inf.objectDistance = Inf;
        Paraxial pi = paraxialAnalysis(inf, inf.wavelength, index);
        double f = std::abs(pi.efl), N = pi.fNumber, coc = 2 * halfDiag / 1500;
        double Hf = f * f / (N * coc) + f;
        char buf[512];
        double s = c.focusDistance;
        if (std::isinf(s)) {
            std::snprintf(buf, sizeof buf, "EFL %.1f mm, f/%.2f, focus infinity; sharp beyond %.2f m (hyperfocal)", f * 1e3, N, Hf);
        } else {
            double nearD = s * (Hf - f) / (Hf + s - 2 * f);
            double farD = s < Hf ? s * (Hf - f) / (Hf - s) : Inf;
            char far[32] = "infinity";
            if (!std::isinf(farD)) std::snprintf(far, sizeof far, "%.3f m", farD);
            std::snprintf(buf, sizeof buf, "EFL %.1f mm, f/%.2f, focus %.3f m; depth of field %.3f m to %s (CoC %.0f µm)",
                          f * 1e3, N, s, nearD, far, coc * 1e6);
        }
        scene.notes[name] = buf;
    }
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
        if (v.kind == Value::Kind::Call && v.str == "on_terrain") return onTerrain(v);
        if (v.kind == Value::Kind::Call && v.str == "sky") {
            // sky(azimuth, elevation, distance): azimuth from +y toward +x, as for the sun.
            if (v.items.size() != 3) fail(v, "sky(azimuth, elevation, distance)");
            double az = ang(*v.items[0]), el = ang(*v.items[1]), d = len(*v.items[2]);
            return Vec3(std::cos(el) * std::sin(az), std::cos(el) * std::cos(az), std::sin(el)) * d;
        }
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
        else if (v.str == "bands") t.kind = Texture::Kind::Bands;
        else if (v.str == "radial") t.kind = Texture::Kind::Radial;
        else if (v.str == "marble") t.kind = Texture::Kind::Marble;
        else if (v.str == "image") {
            // image("file.jpg", scale = 0.5, mapping = planar | triplanar | uv): an sRGB colour image
            // as reflectance (linear RGB through the RGB reflectance basis), one image per `scale`,
            // or placed by a model's own texture coordinates (uv).
            t.kind = Texture::Kind::Image;
            auto f = v.get("file");
            if (!f && !v.items.empty()) f = v.items[0];
            if (!f) fail(v, "image texture needs a file");
            t.image = image(str(*f), v);
            t.a = Spectrum::rgbReflectance(1, 0, 0);
            t.b = Spectrum::rgbReflectance(0, 1, 0);
            t.c = Spectrum::rgbReflectance(0, 0, 1);
            std::string m = getStr(v, "mapping", "planar");
            if (m == "triplanar") t.mapping = Texture::Mapping::Triplanar;
            else if (m == "uv") t.mapping = Texture::Mapping::UV;
            else if (m != "planar") fail(v, "mapping must be planar, triplanar or uv");
        }
        else fail(v, "unknown texture '" + v.str + "'");
        if (auto a = v.get("a")) t.a = spectrum(*a, false);
        if (auto b = v.get("b")) t.b = spectrum(*b, false);
        if (auto c = v.get("c")) t.c = spectrum(*c, false);
        if (auto s = v.get("scale")) t.scale = len(*s);
        if (auto p = v.get("snow_line")) t.param = len(*p);
        if (auto p = v.get("turbulence")) t.param = num(*p);
        return t;
    }

    // Images are shared by every texture and environment that names them.
    std::shared_ptr<const ImageRGB> image(const std::string& file, const Value& where) const {
        const std::string path = std::filesystem::path(file).is_absolute() ? file : (std::filesystem::path(base_) / file).string();
        auto it = images_.find(path);
        if (it != images_.end()) return it->second;
        try {
            return images_[path] = loadImage(path);
        } catch (const std::exception& e) {
            fail(where, e.what());
        }
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
            roughness(b, o);
            o.reflectance = b.has("tint") ? spectrum(need(b, "tint"), false) : Spectrum::constant(1.0);
        } else if (type == "dielectric") {
            o.type = SurfaceType::Dielectric;
            roughness(b, o);
        } else if (type == "stained_glass") {
            stainedGlass(b, o);
        } else if (type == "absorber" || type == "black") {
            o.type = SurfaceType::Absorber;
        } else if (type == "null") {
            o.type = SurfaceType::Null;
        } else {
            fail(b, "unknown material type '" + type + "'");
        }
        if (auto bk = b.get("back")) o.backAbsorbs = str(*bk) == "black";
        relief(b, o);
        uint32_t id = w_.addOptics(o);
        if (!name.empty()) materials_[name] = id;
        return id;
    }

    // type = stained_glass (thin coloured-glass slab; GPU): transmittance = <spectrum> | image(…),
    // gain, density, thickness, index, haze (diffusely transmitted fraction), reflectance (the same
    // image) with reflection_scale, glass_mask = image(…, mapping = uv), mask_threshold, stone_material.
    void stainedGlass(const Value& b, SurfaceOptics& o) {
        o.type = SurfaceType::StainedGlass;
        StainedGlass& g = o.glass;
        std::string colourImage;
        if (auto t = b.get("transmittance")) {
            if (t->kind == Value::Kind::Call && t->str == "image") {
                o.texture = texture(*t);
                colourImage = o.texture.image ? o.texture.image->path : "";
            } else {
                o.reflectance = spectrum(*t, false);
            }
        } else {
            o.reflectance = Spectrum::constant(1.0);
        }
        g.gain = getNum(b, "gain", 1);
        g.density = getNum(b, "density", 1);
        g.haze = getNum(b, "haze", 0);
        if (b.has("thickness")) g.thickness = len(need(b, "thickness"));
        if (!(g.gain > 0) || !(g.density > 0) || !(g.haze >= 0 && g.haze <= 1) || !(g.thickness > 0))
            fail(b, "stained_glass needs gain > 0, density > 0, 0 ≤ haze ≤ 1 and thickness > 0");
        if (b.has("haze_angle")) fail(need(b, "haze_angle"), "haze_angle is not supported yet: haze transmits diffusely");
        if (auto ix = b.get("index")) {
            const IndexModel m = indexModel(*ix);
            std::vector<double> ls, ns;
            for (double l = 360; l <= 830; l += 10) ls.push_back(l), ns.push_back(m.n(l));
            g.index = Spectrum::tabulated(ls, ns);
        }
        if (auto r = b.get("reflectance")) {
            const bool same = r->kind == Value::Kind::Call && r->str == "image" && o.texture.image &&
                              texture(*r).image == o.texture.image;
            if (!same) fail(*r, "a stained_glass reflectance must be its transmittance image (scaled by reflection_scale)");
            g.reflectionScale = getNum(b, "reflection_scale", 1);
        } else if (b.has("reflection_scale")) {
            fail(need(b, "reflection_scale"), "reflection_scale needs reflectance = <the transmittance image>");
        }
        if (b.has("roughness") && getNum(b, "roughness", 0) > 0)
            std::fprintf(stderr, "note: %s: stained_glass roughness is not supported yet; its reflection stays specular\n",
                         o.name.c_str());
        if (auto m = b.get("glass_mask")) {
            if (m->kind != Value::Kind::Call || m->str != "image") fail(*m, "glass_mask must be image(\"file\", mapping = uv)");
            auto f = m->get("file");
            if (!f && !m->items.empty()) f = m->items[0];
            if (!f) fail(*m, "glass_mask needs a file");
            g.mask = image(str(*f), *m);
            g.maskThreshold = getNum(b, "mask_threshold", 0.5);
            const uint32_t stone = b.has("stone_material") ? materialRef(need(b, "stone_material")) : materials_.at("black");
            const SurfaceType st = w_.optics()[stone].type;
            if (st != SurfaceType::Diffuse && st != SurfaceType::Conductor && st != SurfaceType::Mirror &&
                st != SurfaceType::Absorber)
                fail(b, "stone_material must be opaque (diffuse, conductor, mirror or absorber)");
            g.stone = int(stone);
        }
    }

    // roughness = α, or a material map: roughness = (α_a, α_b[, α_c]) with a texture, one α per
    // texture component (polished and honed stone, clear and etched glass). A conductor's texture
    // tints it per component; a dielectric's only carries the map.
    void roughness(const Value& b, SurfaceOptics& o) {
        if (auto t = b.get("texture")) o.texture = texture(*t);
        auto r = b.get("roughness");
        if (!r) return;
        if (r->kind != Value::Kind::Tuple) {
            o.roughness = num(*r);
            return;
        }
        if (r->items.size() < 2 || r->items.size() > 3) fail(*r, "a roughness map is (a, b) or (a, b, c)");
        if (o.texture.kind == Texture::Kind::None) fail(*r, "a roughness per texture component needs a texture");
        for (const ValuePtr& v : r->items) o.roughnessMap.push_back(num(*v));
        o.roughnessMap.resize(3, o.roughnessMap.back());
        o.roughness = o.roughnessMap[0];  // what backends without material maps use
    }

    // relief = noise(scale = 0.5mm, depth = 0.03mm, octaves = 3): a surface relief finer than
    // the geometry, which tilts the shading normal by its slopes (depth / scale).
    void relief(const Value& b, SurfaceOptics& o) {
        auto r = b.get("relief");
        if (!r) return;
        if (r->kind != Value::Kind::Call || r->str != "noise") fail(*r, "relief must be noise(scale =, depth =)");
        o.reliefScale = len(need(*r, "scale"));
        o.reliefDepth = len(need(*r, "depth"));
        o.reliefOctaves = int(getNum(*r, "octaves", 3));
        if (!(o.reliefScale > 0) || !(o.reliefDepth >= 0) || o.reliefOctaves < 1 || o.reliefOctaves > 8)
            fail(*r, "relief needs scale > 0, depth ≥ 0 and 1–8 octaves");
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

    Vec3 onTerrain(const Value& v) {
        if (v.items.size() < 3) fail(v, "on_terrain(\"Terrain\", x, y [, height above ground])");
        std::string n = str(*v.items[0]);
        auto it = terrains_.find(n);
        if (it == terrains_.end()) fail(v, "on_terrain: no terrain named '" + n + "' (define it first)");
        const Transform& xf = it->second.second;
        double x = len(*v.items[1]), y = len(*v.items[2]);
        double dz = v.items.size() > 3 ? len(*v.items[3]) : 0.0;
        return {x, y, terrainHeight(it->second.first, x - xf.t.x, y - xf.t.y) + xf.t.z + dz};
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
            } else if (auto sf = b.get("surface")) {
                // surface = null: an index-matched medium (haze or smoke in air) whose boundary only
                // changes the region; light crosses it, shadow rays too, attenuated by the medium.
                if (str(*sf) != "null") fail(*sf, "a medium body's surface can only be null");
                SurfaceOptics o;
                o.name = "null";
                o.type = SurfaceType::Null;
                m.optics = w_.addOptics(o);
            } else if (b.has("texture") || b.has("relief")) {
                // Its own surface: a material map over the glass (an etched pattern), or a relief.
                SurfaceOptics o;
                o.type = SurfaceType::Dielectric;
                roughness(b, o);
                relief(b, o);
                m.optics = w_.addOptics(o);
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
            } else if (s->kind == Value::Kind::Call && s->str == "map") {
                // map("sky.hdr", rotation = 0deg, scale = 1): an equirectangular HDR environment
                // (zenith at the top row, azimuth from +x toward +y, turned by rotation).
                e.skyModel = Environment::Sky::Map;
                auto f = s->get("file");
                if (!f && !s->items.empty()) f = s->items[0];
                if (!f) fail(*s, "map needs a file");
                e.map = image(str(*f), *s);
                e.mapRotation = getAng(*s, "rotation", 0);
                e.mapScale = getNum(*s, "scale", 1);
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
            e.sunNeeShare = getNum(*c, "nee_share", 0);
            if (auto r = c->get("radiance")) e.sunRadiance = spectrum(*r, true);
            else e.sunRadiance = Spectrum::blackbody(getNum(*c, "temperature", 5778), getNum(*c, "luminance", 1e5));
        }
    }

    void assemblyBlock(const Value& b, int parent) {
        int a = w_.addAssembly(b.name, parent, placement(b));
        for (auto& c : b.items) statement(*c, a);
    }

    void body(const Value& b, int assembly) {
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

    void bodyImpl(const Value& b, int assembly, const std::string& name, const Transform& pre) {
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
        if (type == "lens") lensBody(b, name, assembly, xf);
        else if (type == "prescription") prescriptionBody(b, name, assembly, xf);
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
        } else if (type == "lathe") {
            std::vector<std::pair<double, double>> prof;
            for (auto& pt : need(b, "profile").items) {
                auto q = seq(*pt, 2);
                prof.push_back({len(*q[0]), len(*q[1])});
            }
            int id = buildMesh(w_, name, revolveProfile(prof, int(getNum(b, "segments", 64))), solidMaterial(b), assembly, xf);
            w_.bodies()[id].kind = "lathe";
        } else if (type == "torus" || type == "ring") {
            double R = getRadius(b, "", 0.1), r = getRadius(b, "tube_", R * 0.05);
            int id = buildMesh(w_, name, makeTorus(R, r, int(getNum(b, "segments", 96)), int(getNum(b, "tube_segments", 16))),
                               solidMaterial(b), assembly, xf);
            w_.bodies()[id].kind = "torus";
        } else if (type == "column") {
            columnBody(b, name, assembly, xf);
        } else if (type == "dome") {
            // Exact hemisphere with an observing slit (open surface, same finish both sides).
            double R = getRadius(b, "", 3);
            auto shape = std::make_shared<DomeShape>(R, getAng(b, "slit_azimuth", 0), getLen(b, "slit_width", 0),
                                                     getAng(b, "slit_top", radians(90)));
            uint32_t o = b.has("material") ? materialRef(need(b, "material")) : materials_.at("white");
            int id = w_.addBody(name, "dome", assembly, xf);
            w_.addBoundary(id, "shell", shape, Transform{}, kOutside, kOutside, o);
        } else if (type == "round_wall") {
            std::vector<RoundWallShape::Opening> ops;
            if (auto list = b.get("openings"))
                for (auto& o : list->items) {
                    auto q = seq(*o, 4);
                    ops.push_back({ang(*q[0]), len(*q[1]), len(*q[2]), len(*q[3])});
                }
            auto shape = std::make_shared<RoundWallShape>(getRadius(b, "", 3), len(need(b, "height")), ops);
            uint32_t o = b.has("material") ? materialRef(need(b, "material")) : materials_.at("white");
            int id = w_.addBody(name, "round-wall", assembly, xf);
            w_.addBoundary(id, "wall", shape, Transform{}, kOutside, kOutside, o);
        } else if (type == "starfield") {
            buildStarfield(w_, name, int(getNum(b, "count", 2000)), uint64_t(getNum(b, "seed", 1)), getLen(b, "distance", 1e13),
                           getAng(b, "angular_radius", radians(0.02)), getNum(b, "brightest", 5.0),
                           getAng(b, "min_elevation", radians(-5)), assembly, xf);
        } else if (type == "sheet" || type == "screen" || type == "disk") {
            BodyMaterial m = solidMaterial(b);
            if (b.has("inner_radius") || b.has("inner_diameter")) {
                double r = getRadius(b, "", 0.1), ri = getRadius(b, "inner_", 0);
                int id = w_.addBody(name, "sheet", assembly, xf);
                w_.addBoundary(id, "face", PlaneShape::disk(r, ri), Transform{}, kOutside, kOutside, m.optics, m.emission);
            } else if (type == "disk" || b.has("radius") || b.has("diameter")) {
                double r = getRadius(b, "", 0.1);
                buildSheet(w_, name, r, r, m.optics, m.emission, assembly, xf, true);
            } else {
                auto s = seq(need(b, "size"), 2);
                buildSheet(w_, name, len(*s[0]) / 2, len(*s[1]) / 2, m.optics, m.emission, assembly, xf, false);
            }
        } else if (type == "mesh") {
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
        } else if (type == "vessel") {
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
        } else if (type == "waves") {
            // body Pool { type = waves  size = (w, l)  depth = d  margin = m  medium = water  bottom = sand
            //             walls = tile  waves = [wave(amplitude, wavelength, direction, phase), ...]
            //                         | ripples(count, seed, wavelength = (min, max), slope, direction, spread) }
            WaterSpec ws;
            auto sz = seq(need(b, "size"), 2);
            ws.halfX = len(*sz[0]) / 2;
            ws.halfY = len(*sz[1]) / 2;
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
        } else if (type == "cup") {
            CupRod rod;
            if (auto r = b.get("rod")) {
                // rod = rod(diameter = 7mm, at = (x, y), top = 0.16, medium = N-BK7 | material = ...)
                rod.radius = getRadius(*r, "", 0.0035);
                if (auto at = r->get("at")) { auto q = seq(*at, 2); rod.x = len(*q[0]); rod.y = len(*q[1]); }
                rod.top = getLen(*r, "top", getLen(b, "height", 0.11) + 0.05);
                rod.medium = getStr(*r, "medium", "");
                if (rod.medium.empty()) rod.optics = r->has("material") ? materialRef(need(*r, "material")) : materials_.at("white");
                else w_.medium(rod.medium);
            }
            buildCup(w_, name, getRadius(b, "outer_", 0.035), getLen(b, "wall", 0.0025), getLen(b, "base", 0.008),
                     getLen(b, "height", 0.11), getLen(b, "level", 0.07), getStr(b, "glass", "N-BK7"),
                     getStr(b, "liquid", "water"), assembly, xf, rod);
        } else if (type == "terrain") {
            TerrainSpec t = terrainSpec(b);
            terrains_[name] = {t, xf};
            // A reference-only terrain is the height field that on_terrain() and forests stand on,
            // without matter: a detailed mesh elsewhere in the scene is the visible ground.
            if (auto r = b.get("reference_only"); r && boolean(*r)) return;
            BodyMaterial m = solidMaterial(b);
            int id = buildMesh(w_, name, makeTerrain(t), m, assembly, xf);
            w_.bodies()[id].kind = "terrain";
        } else if (type == "forest") {
            std::string tn = str(need(b, "terrain"));
            auto it = terrains_.find(tn);
            if (it == terrains_.end()) fail(b, "forest: unknown terrain '" + tn + "'");
            uint32_t fo = b.has("foliage") ? materialRef(need(b, "foliage")) : materials_.at("white");
            uint32_t tr = b.has("trunk") ? materialRef(need(b, "trunk")) : materials_.at("grey");
            buildForest(w_, name, it->second.first, int(getNum(b, "count", 200)), getLen(b, "inner_radius", 10),
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

    // Imported models arrive in arbitrary frames: `model_up` names the model axis that points up,
    // `fit_height` rescales to a physical height, and the result stands on the body origin.
    // normals = flat (the default: each triangle its own plane) | smooth | smooth(crease = 30deg) |
    // file (the OBJ's vn). Smooth normals interpolate across every edge whose faces meet at less
    // than the crease angle; the file's normals are the model maker's.
    std::vector<MeshShape::Corners> meshNormals(const Value& b, const MeshData& md) {
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

    void prepareImportedMesh(const Value& b, MeshData& md) {
        if (auto u = b.get("model_up"); u && u->kind == Value::Kind::Tuple) {
            // A measured up vector: apply the minimal rotation taking it to +z (keeps the facing).
            Vec3 up = normalize(vecNum(*u)), z{0, 0, 1};
            Vec3 axis = cross(up, z);
            Transform R;
            if (length(axis) > 1e-12) R = Transform::rotate(axis, std::acos(clampd(dot(up, z), -1, 1)));
            else if (up.z < 0) R = Transform::rotate({1, 0, 0}, Pi);
            for (auto& p : md.positions) p = R.point(p);
            for (auto& n : md.normals) n = R.vector(n);
        } else if (auto u = b.get("model_up")) {
            std::string a = str(*u);
            Transform R;
            if (a == "y") R = Transform::rotate({1, 0, 0}, Pi / 2);
            else if (a == "-y") R = Transform::rotate({1, 0, 0}, -Pi / 2);
            else if (a == "-z") R = Transform::rotate({1, 0, 0}, Pi);
            else if (a == "x") R = Transform::rotate({0, 1, 0}, -Pi / 2);
            else if (a == "-x") R = Transform::rotate({0, 1, 0}, Pi / 2);
            else if (a != "z") fail(*u, "model_up must be one of x, -x, y, -y, z, -z");
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

    void columnBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
        // Doric column: fluted shaft with entasis, echinus (lathe) and square abacus, stacked
        // with 10 µm clearances so the three solids never share a face.
        double H = len(need(b, "height"));
        double r0 = getRadius(b, "", H / 14);
        double capH = getLen(b, "capital_height", 2 * r0 * 0.5);
        double shaftH = H - capH;
        double r1 = r0 * getNum(b, "taper", 0.78);
        uint32_t o = b.has("material") ? materialRef(need(b, "material")) : materials_.at("white");
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
        MeshData ab;
        {
            Vec3 c[8];
            for (int i = 0; i < 8; ++i) c[i] = {(i & 1) ? a : -a, (i & 2) ? a : -a, (i & 4) ? aH : 0.0};
            ab.addQuad(c[0], c[2], c[3], c[1]);
            ab.addQuad(c[4], c[5], c[7], c[6]);
            ab.addQuad(c[0], c[1], c[5], c[4]);
            ab.addQuad(c[2], c[6], c[7], c[3]);
            ab.addQuad(c[0], c[4], c[6], c[2]);
            ab.addQuad(c[1], c[3], c[7], c[5]);
            orientOutward(ab);
        }
        w_.addBoundary(body, "abacus", std::make_shared<MeshShape>(ab.positions, ab.triangles, "abacus"),
                       Transform::translate({0, 0, shaftH + eH + 2 * gap}), kOutside, inside, o);
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

    void lensBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
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
        buildLens(w_, name, spec, assembly, xf);
    }

    void prescriptionBody(const Value& b, const std::string& name, int assembly, const Transform& xf) {
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
        BuiltInstrument bi = buildPrescription(w_, p, name, assembly, xf, tube, index,
                                               rim == "ground" ? LensSpec::Rim::Ground
                                               : rim == "polished" ? LensSpec::Rim::Polished : LensSpec::Rim::Black);
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

    // ------------------------------------------------------------ detectors
    std::pair<int, int> resolution(const Value& b) const {
        auto v = b.get("resolution");
        if (!v) return {320, 240};
        auto s = seq(*v, 2);
        return {int(num(*s[0])), int(num(*s[1]))};
    }

    // Detector keys: `exposure = EV` (fixed, replaces auto exposure), `white_balance = K | none`,
    // `tone = agx | standard` (the display view transform), `sun_share = s` (the sun's share of
    // light samples while this detector renders).
    void detectorDisplay(const Value& b) {
        DetectorSettings d;
        if (auto e = b.get("exposure")) { d.hasExposure = true; d.exposure = num(*e); }
        if (auto wb = b.get("white_balance")) {
            d.hasWhiteBalance = true;
            d.whiteBalance = (wb->kind == Value::Kind::Ident && wb->str == "none") ? 0 : num(*wb);
        }
        if (auto t = b.get("tone")) { d.hasTone = true; d.tone = toneName(*t); }
        if (auto ss = b.get("sun_share")) { d.hasSunShare = true; d.sunShare = num(*ss); }
        if (d.hasExposure || d.hasWhiteBalance || d.hasSunShare || d.hasTone) scene_.display[b.name] = d;
    }
    std::string toneName(const Value& v) const {
        std::string t = str(v);
        if (t != "agx" && t != "standard") fail(v, "tone must be agx or standard");
        return t;
    }

    // `pixel_filter = box | gaussian | gaussian(sigma)`: the pixel response (Detector::pixelSigma,
    // in pixels). Observers default to gaussian, sensors to box.
    void pixelFilter(const Value& b, Detector& d) {
        auto f = b.get("pixel_filter");
        if (!f) return;
        if (f->kind == Value::Kind::Ident && f->str == "box") d.pixelSigma = 0;
        else if (f->kind == Value::Kind::Ident && f->str == "gaussian") d.pixelSigma = IdealObserver::kDefaultPixelSigma;
        else if (f->kind == Value::Kind::Call && f->str == "gaussian" && f->items.size() == 1) d.pixelSigma = num(*f->items[0]);
        else fail(*f, "pixel_filter must be box, gaussian or gaussian(sigma in pixels)");
        if (!(d.pixelSigma >= 0 && d.pixelSigma <= 8)) fail(*f, "pixel filter sigma must be between 0 and 8 pixels");
    }

    void observer(const Value& b) {
        detectorDisplay(b);
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
        pixelFilter(b, *o);
        scene_.detectors.push_back(std::move(o));
    }

    void camera(const Value& b, int assembly) {
        detectorDisplay(b);
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
        c.fNumber = getNum(b, "f_number", 0);
        if (auto rf = b.get("real_focus")) c.realFocus = boolean(*rf);
        if (auto hs = b.get("housing")) c.housing = boolean(*hs);
        Vec3 pos = vecLen(need(b, "position"));
        Vec3 la = vecLen(need(b, "look_at"));
        Vec3 up = b.has("up") ? vecNum(need(b, "up")) : w_.env.up;
        addPhysicalCamera(scene_, b.name, p, c, cameraPlacement(pos, la, up), assembly);
        pixelFilter(b, *scene_.detectors.back());
    }

    void sensor(const Value& b, int assembly) {
        auto s = seq(need(b, "size"), 2);
        double hx = len(*s[0]) / 2, hy = len(*s[1]) / 2;
        int body = w_.addBody(b.name, "sensor", assembly, placement(b));
        int detIndex = int(scene_.detectors.size());
        SurfaceOptics so;
        so.type = SurfaceType::Detector;
        if (auto material = b.get("material")) {
            so = w_.optics()[materialRef(*material)];
            if (so.type != SurfaceType::Diffuse) fail(*material, "sensor material must be diffuse (a measurement screen)");
        }
        so.name = b.name;
        so.detector = detIndex;
        if (so.type == SurfaceType::Diffuse && b.has("aim_center")) {
            so.sampleAimCenter = vecLen(need(b, "aim_center"));
            so.sampleAimNormal = normalize(vecNum(need(b, "aim_axis")));
            so.sampleAimRadius = getRadius(b, "aim_", 0.01);
            if (!(so.sampleAimRadius > 0)) fail(b, "screen aim radius must be positive");
            so.sampleAimShare = 0.95;
        }
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
            if (so.type == SurfaceType::Diffuse) d->aimShare = 0.95;
        }
        pixelFilter(b, *d);
        scene_.detectors.push_back(std::move(d));
    }

    void renderBlock(const Value& b) {
        RenderSettings& r = scene_.render;
        scene_.indirectGuide = getStr(b, "indirect_guide", "");
        r.detector = getStr(b, "detector", r.detector);
        r.integrator = getStr(b, "integrator", r.integrator);
        r.photonRadius = getNum(b, "photon_radius", r.photonRadius);
        r.photonAlpha = getNum(b, "photon_alpha", r.photonAlpha);
        r.photonsPerPixel = getNum(b, "photons", r.photonsPerPixel);
        r.vcmWavelengthGroups = int(getNum(b, "vcm_groups", r.vcmWavelengthGroups));
        if (r.vcmWavelengthGroups < 1 || r.vcmWavelengthGroups > 8) fail(b, "vcm_groups must be 1 to 8");
        if (!(r.photonRadius > 0) || !(r.photonAlpha > 0 && r.photonAlpha <= 1) || !(r.photonsPerPixel > 0))
            fail(b, "photon_radius and photons must be positive, photon_alpha in (0, 1]");
        r.spp = int(getNum(b, "spp", r.spp));
        r.seed = uint64_t(getNum(b, "seed", double(r.seed)));
        r.maxDepth = int(getNum(b, "max_depth", r.maxDepth));
        r.exposure = getNum(b, "exposure", r.exposure);
        r.fresnelFloor = getNum(b, "fresnel_floor", r.fresnelFloor);
        if (auto wbv = b.get("white_balance")) r.whiteBalance = (wbv->kind == Value::Kind::Ident && wbv->str == "none") ? 0 : num(*wbv);
        if (auto a = b.get("auto_exposure")) r.autoExposure = boolean(*a);
        if (auto t = b.get("tone")) r.tone = toneName(*t);
    }

    Scene& scene_;
    World& w_;
    std::string base_;
    double unit_ = 1.0;
    std::map<std::string, uint32_t> materials_;
    mutable std::map<std::string, std::shared_ptr<const ImageRGB>> images_;
    std::map<std::string, std::pair<TerrainSpec, Transform>> terrains_;
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

void applySceneEdit(Value& doc, const std::string& edit) {
    auto dot = edit.find('.');
    auto eq = edit.find('=');
    if (dot == std::string::npos || eq == std::string::npos || dot > eq)
        throw std::runtime_error("scene edit must look like Block.key=value: '" + edit + "'");
    std::string block = edit.substr(0, dot);
    std::string assignment = edit.substr(dot + 1);
    ValuePtr parsed = parseSceneText(assignment, "edit '" + edit + "'");
    if (parsed->named.size() != 1) throw std::runtime_error("scene edit must assign exactly one key: '" + edit + "'");
    std::function<bool(Value&)> visit = [&](Value& v) {
        bool hit = false;
        for (auto& c : v.items)
            if (c->kind == Value::Kind::Block) {
                // Named blocks match by name; unnamed ones (sun) by keyword.
                if (c->name == block || (c->name.empty() && c->str == block)) {
                    c->named.push_back(parsed->named[0]);
                    hit = true;
                }
                hit = visit(*c) || hit;
            }
        return hit;
    };
    if (block == "render" || block == "world") {
        for (auto& c : doc.items)
            if (c->str == block) { c->named.push_back(parsed->named[0]); return; }
        auto blk = std::make_shared<Value>();
        blk->kind = Value::Kind::Block;
        blk->str = block;
        blk->named.push_back(parsed->named[0]);
        doc.items.push_back(blk);
        return;
    }
    if (!visit(doc)) throw std::runtime_error("scene edit: no block named '" + block + "'");
}

Scene loadSceneWithEdits(const std::string& p, const std::vector<std::string>& edits) {
    std::ifstream in(p);
    if (!in) throw std::runtime_error("cannot open scene: " + p);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string dir = ".";
    auto slash = p.find_last_of('/');
    if (slash != std::string::npos) dir = p.substr(0, slash);
    Scene scene;
    scene.sourcePath = p;
    scene.sourceText = ss.str();
    scene.edits = edits;
    ValuePtr doc = parseSceneText(scene.sourceText, p);
    for (auto& e : edits) applySceneEdit(*doc, e);
    Loader ld(scene, dir);
    ld.run(*doc);
    scene.build();
    if (scene.detectors.empty()) throw std::runtime_error(p + ": scene defines no observer, camera or sensor");
    return scene;
}

Scene loadScene(const std::string& p) { return loadSceneWithEdits(p, {}); }

}  // namespace owe
