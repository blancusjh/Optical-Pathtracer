// The .owe scene language: loading a document into a Scene, its top-level statements (world,
// assemblies, render settings) and the scene's texture budget. Materials, bodies and detectors are
// in loader_materials.cpp, loader_bodies.cpp and loader_detectors.cpp.
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <sstream>

#include "owe/core/parallel.hpp"
#include "owe/core/units.hpp"
#include "owe/loader/loader.hpp"

namespace owe {

namespace detail {

std::string resolvePath(const std::string& baseDir, const std::string& p) {
    const std::filesystem::path path(p);
    return path.is_absolute() ? p : (std::filesystem::path(baseDir) / path).string();
}

void Loader::run(const Value& doc) {
    for (auto& [k, v] : doc.named) {
        if (k == "units") {
            unit_ = lengthUnitMetres(str(*v));
            if (!(unit_ > 0)) fail(*v, "unknown length unit '" + str(*v) + "'");
        }
        else fail(*v, "unknown top-level setting '" + k + "'");
    }
    for (auto& b : doc.items) statement(*b, -1);
}

Vec3 Loader::vecLen(const Value& v) {
    if (v.kind == Value::Kind::Call && v.str == "exit_pupil") return exitPupil(v);
    if (v.kind == Value::Kind::Call && v.str == "on_terrain") return onTerrain(v);
    if (v.kind == Value::Kind::Call && v.str == "sky") {
        // sky(azimuth, elevation, distance): azimuth from +y toward +x, as for the sun.
        if (v.items.size() != 3) fail(v, "sky(azimuth, elevation, distance)");
        return directionFromAzimuthElevation(ang(*v.items[0]), ang(*v.items[1])) * len(*v.items[2]);
    }
    auto s = seq(v, 3);
    return {len(*s[0]), len(*s[1]), len(*s[2])};
}

// ------------------------------------------------------------ statements
void Loader::statement(const Value& b, int assembly) {
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

void Loader::worldBlock(const Value& b) {
    Environment& e = w_.env;
    if (auto m = b.get("medium")) w_.setAmbientMedium(w_.medium(str(*m)));
    if (auto u = b.get("up")) e.up = normalize(vecNum(*u));
    if (auto s = b.get("sky")) {
        if (s->kind == Value::Kind::Call && s->str == "gradient") {
            e.skyModel = Environment::Sky::Gradient;
            e.zenith = spectrum(need(*s, "zenith"), true);
            e.horizon = spectrum(need(*s, "horizon"), true);
            e.ground = spectrumOr(*s, "ground", Spectrum::constant(0), true);
        } else if (s->kind == Value::Kind::Call && s->str == "uniform") {
            e.skyModel = Environment::Sky::Uniform;
            e.zenith = spectrum(*s->items.at(0), true);
        } else if (s->kind == Value::Kind::Call && s->str == "map") {
            // map("sky.hdr", rotation = 0deg, scale = 1): an equirectangular HDR environment
            // (zenith at the top row, azimuth from +x toward +y, turned by rotation).
            e.skyModel = Environment::Sky::Map;
            e.map = image(str(fileArg(*s, "map")), *s);
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
        else e.sunDir = directionFromAzimuthElevation(getAng(*c, "azimuth", 0), getAng(*c, "elevation", radians(45)));
        e.sunAngularRadius = getAng(*c, "angular_diameter", radians(0.533)) / 2;
        e.sunNeeShare = getNum(*c, "nee_share", 0);
        if (auto r = c->get("radiance")) e.sunRadiance = spectrum(*r, true);
        else e.sunRadiance = Spectrum::blackbody(getNum(*c, "temperature", 5778), getNum(*c, "luminance", 1e5));
    }
}

void Loader::assemblyBlock(const Value& b, int parent) {
    int a = w_.addAssembly(b.name, parent, placement(b));
    for (auto& c : b.items) statement(*c, a);
}

void Loader::renderBlock(const Value& b) {
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

// The scene's images over the texture budget (OWE_TEXTURE_BUDGET_MB, 2048 by default; 0: none)
// are halved, largest first, until they fit. The files stay at their full resolution; a
// render at a few megapixels cannot show an 8192² photograph's finest texels anyway.
void Loader::fitTextureBudget() {
    double mb = 2048;
    if (const char* e = std::getenv("OWE_TEXTURE_BUDGET_MB"); e && *e) mb = std::atof(e);
    const size_t budget = mb > 0 ? size_t(mb * 1024 * 1024) : std::numeric_limits<size_t>::max();
    // Plan the halvings on the sizes alone: the largest image first, until all fit.
    struct Plan {
        std::shared_ptr<const ImageRGB> img;
        int w, h, halvings = 0;
        size_t bytes() const { return size_t(w) * size_t(h) * (img->hdr ? 12 : 4); }
    };
    std::vector<Plan> plans;
    size_t total = 0;
    for (auto& [path, img] : images_) {
        plans.push_back({img, img->width, img->height});
        total += plans.back().bytes();
    }
    const size_t before = total;
    while (total > budget) {
        Plan* largest = nullptr;
        for (Plan& p : plans)
            if (!largest || p.bytes() > largest->bytes()) largest = &p;
        if (!largest || (largest->w < 2 && largest->h < 2)) break;
        total -= largest->bytes();
        largest->w = std::max(1, largest->w / 2);
        largest->h = std::max(1, largest->h / 2);
        largest->halvings++;
        total += largest->bytes();
    }
    // Decode and halve the images in parallel, largest first, while those at full size fit a
    // memory cap (OWE_TEXTURE_DECODE_MB; one image at a time when it alone exceeds it): decoding
    // is most of a scene's loading. The images are the loader's own, shared read-only once loaded.
    double capMb = 4096;
    if (const char* e = std::getenv("OWE_TEXTURE_DECODE_MB"); e && *e) capMb = std::atof(e);
    const size_t cap = size_t(std::max(0.0, capMb) * 1024 * 1024);
    auto fullBytes = [](const Plan& p) {  // the decoder's buffer and the texels it becomes
        return size_t(p.img->width) * size_t(p.img->height) * (p.img->hdr ? 24 : 7);
    };
    std::vector<Plan*> order;
    for (Plan& p : plans) order.push_back(&p);
    std::stable_sort(order.begin(), order.end(), [&](const Plan* a, const Plan* b) { return fullBytes(*a) > fullBytes(*b); });
    std::mutex m;
    std::condition_variable cv;
    size_t next = 0, inFlight = 0;
    bool failed = false;
    parallelWorkers(std::min<size_t>(hardwareThreads(), order.size()), [&](size_t) {
        for (;;) {
            Plan* p;
            size_t need;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] {
                    return failed || next >= order.size() || inFlight == 0 || inFlight + fullBytes(*order[next]) <= cap;
                });
                if (failed || next >= order.size()) return;
                p = order[next++];
                need = fullBytes(*p);
                inFlight += need;
            }
            ImageRGB& img = const_cast<ImageRGB&>(*p->img);
            try {
                ImageRGB decoded = decodeImage(img.path);
                for (int k = 0; k < p->halvings; ++k) halveImage(decoded);
                img = std::move(decoded);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(m);
                failed = true;
                cv.notify_all();
                throw std::runtime_error(img.path + ": " + e.what());
            }
            std::lock_guard<std::mutex> lk(m);
            inFlight -= need;
            cv.notify_all();
        }
    });
    if (before > budget) {
        std::fprintf(stderr, "textures: %.0f MB over the %.0f MB budget, now %.0f MB (OWE_TEXTURE_BUDGET_MB):\n",
                     before / 1048576.0, mb, total / 1048576.0);
        for (const Plan& p : plans)
            if (p.halvings > 0)
                std::fprintf(stderr, "  %s → %d×%d\n", std::filesystem::path(p.img->path).filename().string().c_str(), p.w, p.h);
    }
}

}  // namespace detail

namespace {

// A parsed document (with its edits applied) as a scene: every statement, the texture budget, the
// built world.
Scene load(Scene scene, const Value& doc, const std::string& baseDir) {
    detail::Loader ld(scene, baseDir);
    ld.run(doc);
    ld.fitTextureBudget();
    scene.build();
    if (scene.detectors.empty()) throw std::runtime_error(scene.sourcePath + ": scene defines no observer, camera or sensor");
    return scene;
}

}  // namespace

Scene loadSceneFromString(const std::string& text, const std::string& baseDir, const std::string& origin) {
    Scene scene;
    scene.sourcePath = origin;
    scene.sourceText = text;
    ValuePtr doc = parseSceneText(text, origin);
    return load(std::move(scene), *doc, baseDir);
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
    Scene scene;
    scene.sourcePath = p;
    scene.sourceText = ss.str();
    scene.edits = edits;
    ValuePtr doc = parseSceneText(scene.sourceText, p);
    for (auto& e : edits) applySceneEdit(*doc, e);
    const std::string dir = std::filesystem::path(p).parent_path().string();
    return load(std::move(scene), *doc, dir.empty() ? "." : dir);
}

Scene loadScene(const std::string& p) { return loadSceneWithEdits(p, {}); }

}  // namespace owe
