// Imported models: glTF 2.0 files placed by their node transforms and turned from glTF's y-up to
// the engine's z-up, their materials as selectable parts, and their texture coordinates carrying
// a picture onto the surface (GPU).
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "check.hpp"
#include "owe/backends/registry.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/scene/detector.hpp"
#include "owe/scene/geometry.hpp"

using namespace owe;
namespace fs = std::filesystem;

namespace {

std::string base64(const std::string& s) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    for (size_t i = 0; i < s.size(); i += 3) {
        uint32_t v = uint32_t(uint8_t(s[i])) << 16;
        if (i + 1 < s.size()) v |= uint32_t(uint8_t(s[i + 1])) << 8;
        if (i + 2 < s.size()) v |= uint32_t(uint8_t(s[i + 2]));
        o += k[(v >> 18) & 63];
        o += k[(v >> 12) & 63];
        o += i + 1 < s.size() ? k[(v >> 6) & 63] : '=';
        o += i + 2 < s.size() ? k[v & 63] : '=';
    }
    return o;
}

// A unit square standing on glTF's ground (x across, y up, facing +z), bottom edge centred on the
// origin, two triangles of the material "Painted" textured with `image`; the node places it with
// `node` (glTF node JSON keys).
std::string quadGltf(const std::string& image, const std::string& node) {
    const float pos[12] = {-0.5f, 0, 0, 0.5f, 0, 0, 0.5f, 1, 0, -0.5f, 1, 0};
    const float uv[8] = {0, 1, 1, 1, 1, 0, 0, 0};  // glTF: v down the image
    const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
    std::string bin(reinterpret_cast<const char*>(pos), sizeof pos);
    bin += std::string(reinterpret_cast<const char*>(uv), sizeof uv);
    bin += std::string(reinterpret_cast<const char*>(idx), sizeof idx);
    return R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0)" + node +
           R"(}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]}],)"
           R"("materials":[{"name":"Painted","pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}],)"
           R"("textures":[{"source":0}],"images":[{"uri":")" + image + R"("}],)"
           R"("buffers":[{"byteLength":92,"uri":"data:application/octet-stream;base64,)" + base64(bin) + R"("}],)"
           R"("bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":32},)"
           R"({"buffer":0,"byteOffset":80,"byteLength":12}],)"
           R"("accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-0.5,0,0],"max":[0.5,1,0]},)"
           R"({"bufferView":1,"componentType":5126,"count":4,"type":"VEC2"},)"
           R"({"bufferView":2,"componentType":5123,"count":6,"type":"SCALAR"}]})";
}

// The same square with per-vertex normals (glTF's +z, its front) and no material.
std::string normalQuadGltf(const std::string& node) {
    const float pos[12] = {-0.5f, 0, 0, 0.5f, 0, 0, 0.5f, 1, 0, -0.5f, 1, 0};
    const float nrm[12] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
    std::string bin(reinterpret_cast<const char*>(pos), sizeof pos);
    bin += std::string(reinterpret_cast<const char*>(nrm), sizeof nrm);
    bin += std::string(reinterpret_cast<const char*>(idx), sizeof idx);
    return R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0)" + node +
           R"(}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"indices":2}]}],)"
           R"("buffers":[{"byteLength":108,"uri":"data:application/octet-stream;base64,)" + base64(bin) + R"("}],)"
           R"("bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":48},)"
           R"({"buffer":0,"byteOffset":96,"byteLength":12}],)"
           R"("accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-0.5,0,0],"max":[0.5,1,0]},)"
           R"({"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},)"
           R"({"bufferView":2,"componentType":5123,"count":6,"type":"SCALAR"}]})";
}

fs::path writeTemp(const std::string& name, const std::string& text) {
    const fs::path p = fs::temp_directory_path() / name;
    std::ofstream(p, std::ios::binary) << text;
    return p;
}

// An uncompressed 2 × 2 RGB PNG (stored deflate block, CRCs computed): red, green / blue, white.
std::string quadrantsPng() {
    auto crc = [](const std::string& s) {
        uint32_t c = 0xFFFFFFFFu;
        for (unsigned char b : s) {
            c ^= b;
            for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        }
        return ~c;
    };
    auto be = [](uint32_t v) { return std::string{char(v >> 24), char(v >> 16), char(v >> 8), char(v)}; };
    auto chunk = [&](const std::string& type, const std::string& data) {
        return be(uint32_t(data.size())) + type + data + be(crc(type + data));
    };
    const std::string raw = std::string("\0\xff\0\0\0\xff\0", 7) + std::string("\0\0\0\xff\xff\xff\xff", 7);
    uint32_t a = 1, b = 0;
    for (unsigned char c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    const std::string z = std::string("\x78\x01\x01", 3) + char(raw.size()) + char(0) + char(~raw.size()) + char(0xff) + raw +
                          be((b << 16) | a);
    return std::string("\x89PNG\r\n\x1a\n", 8) + chunk("IHDR", be(2) + be(2) + std::string("\x08\x02\0\0\0", 5)) +
           chunk("IDAT", z) + chunk("IEND", "");
}

// A 2 × 2 PNG of one colour (the same encoder as quadrantsPng).
std::string solidPng(unsigned char r, unsigned char g, unsigned char b) {
    auto crc = [](const std::string& s) {
        uint32_t c = 0xFFFFFFFFu;
        for (unsigned char x : s) {
            c ^= x;
            for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        }
        return ~c;
    };
    auto be = [](uint32_t v) { return std::string{char(v >> 24), char(v >> 16), char(v >> 8), char(v)}; };
    auto chunk = [&](const std::string& type, const std::string& data) {
        return be(uint32_t(data.size())) + type + data + be(crc(type + data));
    };
    const std::string row = std::string(1, '\0') + char(r) + char(g) + char(b) + char(r) + char(g) + char(b);
    const std::string raw = row + row;
    uint32_t a = 1, bb = 0;
    for (unsigned char c : raw) { a = (a + c) % 65521; bb = (bb + a) % 65521; }
    const std::string z = std::string("\x78\x01\x01", 3) + char(raw.size()) + char(0) + char(~raw.size()) + char(0xff) + raw +
                          be((bb << 16) | a);
    return std::string("\x89PNG\r\n\x1a\n", 8) + chunk("IHDR", be(2) + be(2) + std::string("\x08\x02\0\0\0", 5)) +
           chunk("IDAT", z) + chunk("IEND", "");
}

// Mean Y over the middle of a GPU render of a scene's first detector.
double gpuMeanY(const std::string& text, int spp) {
    const Scene sc = loadSceneFromString(text);
    RenderSettings rs;
    rs.backend = "gpu";
    rs.integrator = "path";
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(spp);
    const Image im = R->resolve();
    double s = 0;
    int n = 0;
    for (int y = im.height / 4; y < 3 * im.height / 4; ++y)
        for (int x = im.width / 4; x < 3 * im.width / 4; ++x, ++n) s += im.at(x, y).y;
    return s / n;
}

}  // namespace

TEST(models_gltf_is_placed_z_up_with_parts_and_uvs) {
    // The node lifts the square 2 m along glTF's y (up) and mirrors it in x: the engine sees it
    // standing 2 m up along z, still facing the viewer on −y (the mirror's reversed winding undone).
    const fs::path f = writeTemp("owe_test_quad.gltf", quadGltf("owe_test_quad.png", R"(,"translation":[0,2,0],"scale":[-1,1,1])"));
    const MeshData m = loadModel(f.string());
    CHECK(m.triangles.size() == 2);
    CHECK(m.partNames.size() == 1 && m.partNames[0] == "Painted");
    CHECK(m.partImages.size() == 1 && m.partImages[0] == (fs::temp_directory_path() / "owe_test_quad.png").string());
    CHECK(m.uvs.size() == 2 && m.normals.empty());
    AABB box;
    for (const Vec3& p : m.positions) box.expand(p);
    CHECK_NEAR(box.lo.z, 2.0, 1e-9);
    CHECK_NEAR(box.hi.z, 3.0, 1e-9);
    CHECK_NEAR(box.hi.y - box.lo.y, 0.0, 1e-9);
    for (const auto& t : m.triangles) {
        const Vec3 n = cross(m.positions[t[1]] - m.positions[t[0]], m.positions[t[2]] - m.positions[t[0]]);
        CHECK(n.y < 0);
    }
    // Each corner keeps its own texture coordinates through the reversed winding: the corner at
    // the model's bottom-left (+x after the mirror) is the image's bottom-left, (0, 1).
    for (size_t k = 0; k < 2; ++k)
        for (int c = 0; c < 3; ++c) {
            const Vec3& p = m.positions[m.triangles[k][size_t(c)]];
            const auto& uv = m.uvs[k][size_t(c)];
            CHECK_NEAR(uv[0], p.x > 0 ? 0.0 : 1.0, 1e-6);
            CHECK_NEAR(uv[1], p.z < 2.5 ? 1.0 : 0.0, 1e-6);
        }
    const MeshData sel = selectParts(m, {"Painted"});
    CHECK(sel.triangles.size() == 2 && sel.uvs.size() == 2);
    bool threw = false;
    try {
        selectParts(m, {"Glass"});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(models_gltf_normals_follow_rotated_and_stretched_nodes) {
    // Normals transform by the inverse transpose of the node's matrix: after a quarter turn about
    // glTF's y and a stretch along x, every file normal still points along its face's front.
    for (const char* node : {R"(,"rotation":[0,0.70710678,0,0.70710678],"scale":[2,1,1])",
                             R"(,"rotation":[0.38268343,0,0,0.92387953],"scale":[1,3,-1])"}) {
        const MeshData m = loadModel(writeTemp("owe_test_normal_quad.gltf", normalQuadGltf(node)).string());
        CHECK(m.normals.size() == 4 && m.normalIndex.size() == 2);
        for (size_t t = 0; t < m.triangles.size(); ++t) {
            const auto& tri = m.triangles[t];
            const Vec3 face = normalize(cross(m.positions[tri[1]] - m.positions[tri[0]], m.positions[tri[2]] - m.positions[tri[0]]));
            for (uint32_t k : m.normalIndex[t]) CHECK(dot(m.normals[k], face) > 0.999);
        }
    }
}

TEST(models_gltf_texture_follows_its_uvs_on_the_gpu) {
    const Backend* gpu = findBackend("gpu");
    if (!gpu || !gpu->available()) return;
    writeTemp("owe_test_quadrants.png", quadrantsPng());
    const fs::path f = writeTemp("owe_test_quadrants.gltf", quadGltf("owe_test_quadrants.png", ""));
    const Scene sc = loadSceneFromString(R"(
        units = m
        world { sky = uniform(rgb(1, 1, 1, luminance = 1)) }
        material painted { type = diffuse  texture = image(")" + (fs::temp_directory_path() / "owe_test_quadrants.png").string() +
                                         R"(", mapping = uv) }
        body Quad { type = mesh  file = ")" + f.string() + R"("  part = "Painted"  material = painted }
        observer Eye { position = (0, -3, 0.5)  look_at = (0, 0, 0.5)  up = (0, 0, 1)  fov = 25deg  pupil = 1mm  focus = 3
                       resolution = (64, 64) }
        render { detector = Eye }
    )");
    RenderSettings rs;
    rs.backend = "gpu";
    rs.integrator = "path";
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(16);
    const Image im = R->resolve();
    // Mean XYZ of each quadrant of the square (it spans the middle ~48% of the frame).
    auto mean = [&](double cx, double cy) {
        Vec3 s;
        for (int y = int((cy - 0.06) * 64); y < int((cy + 0.06) * 64); ++y)
            for (int x = int((cx - 0.06) * 64); x < int((cx + 0.06) * 64); ++x) {
                const XYZ c = im.at(x, y);
                s += Vec3(c.x, c.y, c.z);
            }
        return s;
    };
    const Vec3 red = mean(0.4, 0.37), green = mean(0.6, 0.37), blue = mean(0.4, 0.63), white = mean(0.6, 0.63);
    auto chroma = [](const Vec3& c) { return c / (c.x + c.y + c.z); };
    std::fprintf(stderr, "    quadrant chromaticity x: %.3f %.3f %.3f %.3f  z: %.3f %.3f %.3f %.3f\n", chroma(red).x,
                 chroma(green).x, chroma(blue).x, chroma(white).x, chroma(red).z, chroma(green).z, chroma(blue).z,
                 chroma(white).z);
    CHECK(chroma(red).x > chroma(green).x && chroma(red).x > chroma(blue).x && chroma(red).x > chroma(white).x);
    CHECK(chroma(green).y > chroma(red).y && chroma(green).y > chroma(blue).y && chroma(green).y > chroma(white).y);
    CHECK(chroma(blue).z > chroma(red).z && chroma(blue).z > chroma(green).z && chroma(blue).z > chroma(white).z);
    CHECK(white.y > red.y && white.y > green.y && white.y > blue.y);
}

// A normal map tilting a Lambertian floor's shading normal 30° toward the sun (elevation 60°)
// raises its radiance by 1/cos 30°; a roughness map of value 1 gives α = 1, as roughness = 1 does.
TEST(models_normal_and_roughness_maps_on_the_gpu) {
    const Backend* gpu = findBackend("gpu");
    if (!gpu || !gpu->available()) return;
    const std::string tilt = (fs::temp_directory_path() / "owe_test_tilt.png").string();
    const std::string one = (fs::temp_directory_path() / "owe_test_rough1.png").string();
    writeTemp("owe_test_tilt.png", solidPng(191, 128, 238));  // (sin 30°, 0, cos 30°) encoded
    writeTemp("owe_test_rough1.png", solidPng(255, 255, 255));
    auto scene = [](const std::string& material) {
        return R"(
            units = m
            world { sky = uniform(rgb(0, 0, 0))  sun { elevation = 60deg  azimuth = 90deg  luminance = 1e4 } }
            material floor { )" + material + R"( }
            body Floor { type = sheet  size = (4, 4)  position = (0, 0, 0)  material = floor }
            observer Eye { position = (0, 0, 3)  look_at = (0, 0, 0)  up = (0, 1, 0)  fov = 20deg  pupil = 1mm  resolution = (16, 16) }
            render { detector = Eye }
        )";
    };
    const double flat = gpuMeanY(scene("type = diffuse  reflectance = 0.8"), 32);
    const double tilted = gpuMeanY(scene("type = diffuse  reflectance = 0.8  normal_map = image(\"" + tilt + "\", mapping = planar, scale = 1)"), 32);
    std::fprintf(stderr, "    tilted / flat = %.4f (1/cos 30° = 1.1547)\n", tilted / flat);
    CHECK(std::abs(tilted / flat - 1 / std::cos(Pi / 6)) < 0.01);
    const double rough = gpuMeanY(scene("type = conductor  metal = gold  roughness = 1"), 64);
    const double mapped = gpuMeanY(scene("type = conductor  metal = gold  roughness_map = image(\"" + one + "\", mapping = planar)"), 64);
    std::fprintf(stderr, "    roughness map / roughness = %.4f\n", mapped / rough);
    CHECK(std::abs(mapped / rough - 1) < 0.01);
    // The same tilt by texture coordinates on a glTF quad facing −y (u along +x, the picture's top
    // up): the shading normal turns 30° toward +x, straight at a sun at azimuth 150° on the horizon.
    const fs::path quad = writeTemp("owe_test_tilt_quad.gltf", quadGltf("owe_test_tilt.png", ""));
    auto quadScene = [&](const std::string& material) {
        return R"(
            units = m
            world { sky = uniform(rgb(0, 0, 0))  sun { elevation = 0deg  azimuth = 150deg  luminance = 1e4 } }
            material painted { )" + material + R"( }
            body Quad { type = mesh  file = ")" + quad.string() + R"("  part = "Painted"  material = painted }
            observer Eye { position = (0, -3, 0.5)  look_at = (0, 0, 0.5)  up = (0, 0, 1)  fov = 16deg  pupil = 1mm  resolution = (16, 16) }
            render { detector = Eye }
        )";
    };
    const double quadFlat = gpuMeanY(quadScene("type = diffuse  reflectance = 0.8"), 32);
    const double quadTilted = gpuMeanY(quadScene("type = diffuse  reflectance = 0.8  normal_map = image(\"" + tilt + "\", mapping = uv)"), 32);
    std::fprintf(stderr, "    uv-mapped tilted / flat = %.4f (1.1547)\n", quadTilted / quadFlat);
    CHECK(std::abs(quadTilted / quadFlat - 1 / std::cos(Pi / 6)) < 0.01);
}
