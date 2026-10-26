// The .owe loader's internal state: one Loader turns a parsed scene document into a Scene. Its
// parts live in loader_*.cpp: values and statements (scene_loader.cpp), materials and media
// (loader_materials.cpp), placement and bodies (loader_bodies.cpp), detectors and cameras
// (loader_detectors.cpp).
#pragma once

#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "owe/analysis/analysis.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/scene/builders.hpp"

namespace owe::detail {

// A path in the scene (absolute, or relative to the scene's directory).
std::string resolvePath(const std::string& baseDir, const std::string& p);

struct Instrument {
    int assembly = -1;
    Paraxial paraxial;
};

class Loader {
public:
    Loader(Scene& s, std::string baseDir) : scene_(s), w_(s.world), base_(std::move(baseDir)) { registerDefaults(); }
    void run(const Value& doc);
private:
    // ------------------------------------------------------------ errors & values
    [[noreturn]] void fail(const Value& v, const std::string& m) const {
        throw std::runtime_error(scene_.sourcePath + ": line " + std::to_string(v.line) + ": " + m);
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
    Vec3 vecLen(const Value& v);
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
    // A file named in the scene, relative to the scene's directory.
    std::string path(const std::string& p) const { return resolvePath(base_, p); }
    // A material named by key, else the predefined material `def`.
    uint32_t materialOr(const Value& b, const std::string& key, const std::string& def) {
        return b.has(key) ? materialRef(need(b, key)) : materials_.at(def);
    }
    Spectrum spectrumOr(const Value& b, const std::string& key, const Spectrum& def, bool emission) const {
        auto v = b.get(key);
        return v ? spectrum(*v, emission) : def;
    }
    // The file of image("file", …) or map("file", …): its `file` or its first argument.
    const Value& fileArg(const Value& call, const std::string& what) const {
        if (auto f = call.get("file")) return *f;
        if (!call.items.empty()) return *call.items[0];
        fail(call, what + " needs a file");
    }
    // A pair of lengths (x, y), and half of a (width, height) size.
    Vec2 lengths2(const Value& v) const {
        auto s = seq(v, 2);
        return {len(*s[0]), len(*s[1])};
    }
    Vec2 halfSize(const Value& v) const { return lengths2(v) / 2; }
    // A length that may be `inf` (a focus at infinity).
    double lenOrInf(const Value& v) const { return v.kind == Value::Kind::Number && std::isinf(v.num) ? Inf : len(v); }
    // A block's `up`, else the world's.
    Vec3 upOr(const Value& b) const { return b.has("up") ? vecNum(need(b, "up")) : w_.env.up; }
    // A lens rim finish: ground, polished or black.
    LensSpec::Rim rimKind(const Value& b, const std::string& def) const {
        const std::string rim = getStr(b, "rim", def);
        if (rim == "black") return LensSpec::Rim::Black;
        if (rim == "polished") return LensSpec::Rim::Polished;
        if (rim != "ground") fail(b, "rim must be ground, polished or black");
        return LensSpec::Rim::Ground;
    }
    // ------------------------------------------------------------ spectra
    Spectrum spectrum(const Value& v, bool emission) const;
    Texture texture(const Value& v) const;
    // Images are shared by every texture and environment that names them.
    // An image is read in two steps: its size now, from the file's header, and its texels once the
    // whole scene is known (fitTextureBudget), so that each is decoded at the size it will keep.
    std::shared_ptr<const ImageRGB> image(const std::string& file, const Value& where) const;
    // ------------------------------------------------------------ registries
    void registerDefaults();
    uint32_t materialRef(const Value& v);
    uint32_t defineMaterial(const Value& b, const std::string& name);
    // type = stained_glass (thin coloured-glass slab; GPU): transmittance = <spectrum> | image(…),
    // gain, density, thickness, index, haze (diffusely transmitted fraction), reflectance (the same
    // image) with reflection_scale, glass_mask = image(…, mapping = uv), mask_threshold, stone_material.
    void stainedGlass(const Value& b, SurfaceOptics& o);
    // roughness = α, or a material map: roughness = (α_a, α_b[, α_c]) with a texture, one α per
    // texture component (polished and honed stone, clear and etched glass). A conductor's texture
    // tints it per component; a dielectric's only carries the map.
    void roughness(const Value& b, SurfaceOptics& o);
    // normal_map = image("file", mapping = uv | triplanar | planar, scale =), normal_strength,
    // normal_convention = opengl | directx; roughness_map = image(…), roughness_channel = r | g | b.
    void imageMaps(const Value& b, SurfaceOptics& o);
    // relief = noise(scale = 0.5mm, depth = 0.03mm, octaves = 3): a surface relief finer than
    // the geometry, which tilts the shading normal by its slopes (depth / scale).
    void relief(const Value& b, SurfaceOptics& o);
    void defineMedium(const Value& b);
    IndexModel indexModel(const Value& v) const;
    // ------------------------------------------------------------ placement
    Transform placement(const Value& b);
    Vec3 onTerrain(const Value& v);
    Vec3 exitPupil(const Value& v);
    // ------------------------------------------------------------ surfaces
    SurfaceSpec surface(const Value& v, double semiDiameter) const;
    BodyMaterial solidMaterial(const Value& b);
    // ------------------------------------------------------------ statements
    void statement(const Value& b, int assembly);
    void worldBlock(const Value& b);
    void assemblyBlock(const Value& b, int parent);
    void body(const Value& b, int assembly);
    void bodyImpl(const Value& b, int assembly, const std::string& name, const Transform& pre);
    // The builders of the body types (bodyImpl's table).
    void mirrorBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void flatMirrorBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void stopBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void tubeBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void sphereBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void boxBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void cylinderBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void prismBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void latheBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void torusBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void domeBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void roundWallBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void starfieldBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void sheetBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void meshBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void vesselBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void wavesBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void cupBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void terrainBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void forestBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void statueBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    // Imported models arrive in arbitrary frames: `model_up` names the model axis that points up,
    // `fit_height` rescales to a physical height, and the result stands on the body origin.
    // normals = flat (the default: each triangle its own plane) | smooth | smooth(crease = 30deg) |
    // file (the OBJ's vn). Smooth normals interpolate across every edge whose faces meet at less
    // than the crease angle; the file's normals are the model maker's.
    std::vector<MeshShape::Corners> meshNormals(const Value& b, const MeshData& md);
    void prepareImportedMesh(const Value& b, MeshData& md);
    void columnBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    TerrainSpec terrainSpec(const Value& b) const;
    void lensBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    void prescriptionBody(const Value& b, const std::string& name, int assembly, const Transform& xf);
    // ------------------------------------------------------------ detectors
    std::pair<int, int> resolution(const Value& b) const;
    // Detector keys: `exposure = EV` (fixed, replaces auto exposure), `white_balance = K | none`,
    // `tone = agx | standard` (the display view transform), `sun_share = s` (the sun's share of
    // light samples while this detector renders).
    void detectorDisplay(const Value& b);
    std::string toneName(const Value& v) const;
    // `pixel_filter = box | gaussian | gaussian(sigma)`: the pixel response (Detector::pixelSigma,
    // in pixels). Observers default to gaussian, sensors to box.
    void pixelFilter(const Value& b, Detector& d);
    void observer(const Value& b);
    void camera(const Value& b, int assembly);
    void sensor(const Value& b, int assembly);
    void renderBlock(const Value& b);
    Scene& scene_;
    World& w_;
    std::string base_;
    double unit_ = 1.0;
    std::map<std::string, uint32_t> materials_;
    mutable std::map<std::string, std::shared_ptr<const ImageRGB>> images_;
public:
    // The scene's images over the texture budget (OWE_TEXTURE_BUDGET_MB, 2048 by default; 0: none)
    // are halved, largest first, until they fit. The files stay at their full resolution; a
    // render at a few megapixels cannot show an 8192² photograph's finest texels anyway.
    void fitTextureBudget();
private:
    std::map<std::string, std::pair<TerrainSpec, Transform>> terrains_;
    std::map<std::string, Instrument> instruments_;
    std::set<std::string> names_;
};

}  // namespace owe::detail
