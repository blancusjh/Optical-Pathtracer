// Materials, spectra, textures, media and refractive indices of the .owe language.
#include <cstdio>
#include "owe/loader/loader.hpp"

namespace owe::detail {

// ------------------------------------------------------------ spectra
Spectrum Loader::spectrum(const Value& v, bool emission) const {
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

Texture Loader::texture(const Value& v) const {
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
        t.image = image(str(fileArg(v, "image texture")), v);
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
// An image is read in two steps: its size now, from the file's header, and its texels once the
// whole scene is known (fitTextureBudget), so that each is decoded at the size it will keep.
std::shared_ptr<const ImageRGB> Loader::image(const std::string& file, const Value& where) const {
    const std::string path = this->path(file);
    auto it = images_.find(path);
    if (it != images_.end()) return it->second;
    auto img = std::make_shared<ImageRGB>();
    img->path = path;
    if (!imageSize(path, img->width, img->height) || img->width <= 0 || img->height <= 0)
        fail(where, "cannot read image " + path);
    img->hdr = imageIsHdr(path);
    images_[path] = img;
    return img;
}

// ------------------------------------------------------------ registries
void Loader::registerDefaults() {
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

uint32_t Loader::materialRef(const Value& v) {
    if (v.kind == Value::Kind::Block) return defineMaterial(v, "");
    std::string n = str(v);
    auto it = materials_.find(n);
    if (it == materials_.end()) fail(v, "unknown material '" + n + "'");
    return it->second;
}

uint32_t Loader::defineMaterial(const Value& b, const std::string& name) {
    SurfaceOptics o;
    o.name = name.empty() ? "anonymous" : name;
    std::string type = getStr(b, "type", "diffuse");
    if (type == "diffuse" || type == "lambertian") {
        o.type = SurfaceType::Diffuse;
        o.reflectance = spectrumOr(b, "reflectance", Spectrum::constant(0.8), false);
        if (auto t = b.get("texture")) o.texture = texture(*t);
    } else if (type == "mirror") {
        o.type = SurfaceType::Mirror;
        o.reflectance = spectrumOr(b, "reflectance", Spectrum::constant(1.0), false);
    } else if (type == "conductor" || type == "metal") {
        o.type = SurfaceType::Conductor;
        std::string m = getStr(b, "metal", "aluminium");
        if (!catalogConductor(m, o.conductor)) fail(b, "unknown metal '" + m + "'");
        roughness(b, o);
        o.reflectance = spectrumOr(b, "tint", Spectrum::constant(1.0), false);
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
    imageMaps(b, o);
    uint32_t id = w_.addOptics(o);
    if (!name.empty()) materials_[name] = id;
    return id;
}

// type = stained_glass (thin coloured-glass slab; GPU): transmittance = <spectrum> | image(…),
// gain, density, thickness, index, haze (diffusely transmitted fraction), reflectance (the same
// image) with reflection_scale, glass_mask = image(…, mapping = uv), mask_threshold, stone_material.
void Loader::stainedGlass(const Value& b, SurfaceOptics& o) {
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
        g.index = Spectrum::sampled([&](double l) { return m.n(l); }, 10);
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
        g.mask = image(str(fileArg(*m, "glass_mask")), *m);
        g.maskThreshold = getNum(b, "mask_threshold", 0.5);
        const uint32_t stone = materialOr(b, "stone_material", "black");
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
void Loader::roughness(const Value& b, SurfaceOptics& o) {
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

// normal_map = image("file", mapping = uv | triplanar | planar, scale =), normal_strength,
// normal_convention = opengl | directx; roughness_map = image(…), roughness_channel = r | g | b.
void Loader::imageMaps(const Value& b, SurfaceOptics& o) {
    auto map = [&](const char* key, ImageMap& out) {
        auto v = b.get(key);
        if (!v) return;
        if (v->kind != Value::Kind::Call || v->str != "image") fail(*v, std::string(key) + " must be image(\"file\", mapping = …)");
        const Texture t = texture(*v);
        out.image = t.image;
        out.mapping = t.mapping;
        out.scale = t.scale;
    };
    map("normal_map", o.normalImage);
    map("roughness_map", o.roughnessImage);
    o.normalStrength = getNum(b, "normal_strength", 1);
    const std::string conv = getStr(b, "normal_convention", "opengl");
    if (conv != "opengl" && conv != "directx") fail(b, "normal_convention must be opengl or directx");
    o.normalDirectX = conv == "directx";
    const std::string ch = getStr(b, "roughness_channel", "g");
    if (ch != "r" && ch != "g" && ch != "b") fail(b, "roughness_channel must be r, g or b");
    o.roughnessChannel = ch == "r" ? 0 : ch == "g" ? 1 : 2;
}

// relief = noise(scale = 0.5mm, depth = 0.03mm, octaves = 3): a surface relief finer than
// the geometry, which tilts the shading normal by its slopes (depth / scale).
void Loader::relief(const Value& b, SurfaceOptics& o) {
    auto r = b.get("relief");
    if (!r) return;
    if (r->kind != Value::Kind::Call || r->str != "noise") fail(*r, "relief must be noise(scale =, depth =)");
    o.reliefScale = len(need(*r, "scale"));
    o.reliefDepth = len(need(*r, "depth"));
    o.reliefOctaves = int(getNum(*r, "octaves", 3));
    if (!(o.reliefScale > 0) || !(o.reliefDepth >= 0) || o.reliefOctaves < 1 || o.reliefOctaves > 8)
        fail(*r, "relief needs scale > 0, depth ≥ 0 and 1–8 octaves");
}

void Loader::defineMedium(const Value& b) {
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
            const Spectrum tau = spectrum(*s[0], false);
            m.absorption = Spectrum::sampled([&](double l) { return -std::log(std::max(1e-12, tau(l))) / d; }, 5);
        }
    }
    if (auto s = b.get("scattering")) m.scattering = spectrum(*s, false);
    if (auto g = b.get("g")) m.g = num(*g);
    w_.addMedium(m);
}

IndexModel Loader::indexModel(const Value& v) const {
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

BodyMaterial Loader::solidMaterial(const Value& b) {
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
        } else if (b.has("texture") || b.has("relief") || b.has("normal_map") || b.has("roughness_map")) {
            // Its own surface: a material map over the glass (an etched pattern), a relief, or
            // image maps.
            SurfaceOptics o;
            o.type = SurfaceType::Dielectric;
            roughness(b, o);
            relief(b, o);
            imageMaps(b, o);
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

}  // namespace owe::detail
