// glTF 2.0 models: cgltf parses the file and its buffers; this turns the meshes of its scene into
// one polygon soup in the engine's frame, keeping normals, texture coordinates and materials.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "owe/scene/geometry.hpp"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

namespace owe {

namespace {

const char* resultName(cgltf_result r) {
    switch (r) {
        case cgltf_result_data_too_short: return "data too short";
        case cgltf_result_unknown_format: return "unknown format";
        case cgltf_result_invalid_json: return "invalid JSON";
        case cgltf_result_invalid_gltf: return "invalid glTF";
        case cgltf_result_invalid_options: return "invalid options";
        case cgltf_result_file_not_found: return "file not found";
        case cgltf_result_io_error: return "I/O error";
        case cgltf_result_out_of_memory: return "out of memory";
        case cgltf_result_legacy_gltf: return "glTF 1.0 (only 2.0 is read)";
        default: return "error";
    }
}

// The image a material's base colour comes from, as a path beside the model; empty when it has
// none or the image is embedded in a buffer.
std::string baseColorImage(const cgltf_material& m, const std::filesystem::path& dir) {
    if (!m.has_pbr_metallic_roughness) return {};
    const cgltf_texture* t = m.pbr_metallic_roughness.base_color_texture.texture;
    if (!t || !t->image || !t->image->uri || std::strncmp(t->image->uri, "data:", 5) == 0) return {};
    std::string uri = t->image->uri;
    cgltf_decode_uri(uri.data());
    uri.resize(std::strlen(uri.c_str()));
    return (dir / uri).lexically_normal().string();
}

}  // namespace

MeshData loadGltf(const std::string& path, double scale) {
    cgltf_options options{};
    cgltf_data* raw = nullptr;
    cgltf_result r = cgltf_parse_file(&options, path.c_str(), &raw);
    if (r != cgltf_result_success) throw std::runtime_error("cannot read glTF file " + path + ": " + resultName(r));
    std::unique_ptr<cgltf_data, void (*)(cgltf_data*)> data(raw, cgltf_free);
    r = cgltf_load_buffers(&options, raw, path.c_str());
    if (r != cgltf_result_success) throw std::runtime_error("cannot load the buffers of " + path + ": " + resultName(r));

    MeshData m;
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    for (size_t i = 0; i < raw->materials_count; ++i) {
        const cgltf_material& mat = raw->materials[i];
        m.partNames.push_back(mat.name && *mat.name ? mat.name : "material" + std::to_string(i));
        m.partImages.push_back(baseColorImage(mat, dir));
    }
    const uint32_t unnamed = uint32_t(raw->materials_count);  // primitives without a material
    bool anyNormals = false, anyUvs = false, anyUnnamed = false;
    std::vector<float> buf;
    std::vector<uint32_t> idx;

    auto addPrimitive = [&](const cgltf_primitive& prim, const float M[16]) {
        if (prim.type != cgltf_primitive_type_triangles) return;
        if (prim.has_draco_mesh_compression)
            throw std::runtime_error("glTF file " + path + " is Draco-compressed; export it uncompressed");
        const cgltf_accessor *pos = nullptr, *nrm = nullptr, *tex = nullptr;
        int uvSet = 0;
        if (prim.material && prim.material->has_pbr_metallic_roughness)
            uvSet = prim.material->pbr_metallic_roughness.base_color_texture.texcoord;
        for (size_t a = 0; a < prim.attributes_count; ++a) {
            const cgltf_attribute& at = prim.attributes[a];
            if (at.type == cgltf_attribute_type_position) pos = at.data;
            else if (at.type == cgltf_attribute_type_normal) nrm = at.data;
            else if (at.type == cgltf_attribute_type_texcoord && at.index == uvSet) tex = at.data;
        }
        if (!pos || pos->count == 0) return;
        const size_t nv = pos->count;
        // Column-major M; glTF's y-up turned to z-up: (x, y, z) → (x, −z, y). Normals transform
        // by the cofactor matrix (the inverse transpose up to the determinant's scale).
        const double a00 = M[0], a01 = M[4], a02 = M[8], a10 = M[1], a11 = M[5], a12 = M[9], a20 = M[2], a21 = M[6],
                     a22 = M[10];
        const double det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
        auto zUp = [](double x, double y, double z) { return Vec3(x, -z, y); };
        const size_t base = m.positions.size(), nbase = m.normals.size();
        buf.resize(nv * 3);
        cgltf_accessor_unpack_floats(pos, buf.data(), nv * 3);
        for (size_t k = 0; k < nv; ++k) {
            const double x = buf[3 * k], y = buf[3 * k + 1], z = buf[3 * k + 2];
            m.positions.push_back(zUp(a00 * x + a01 * y + a02 * z + M[12], a10 * x + a11 * y + a12 * z + M[13],
                                      a20 * x + a21 * y + a22 * z + M[14]) *
                                  scale);
        }
        if (nrm && nrm->count == nv) {
            anyNormals = true;
            cgltf_accessor_unpack_floats(nrm, buf.data(), nv * 3);
            const double c00 = a11 * a22 - a12 * a21, c01 = a12 * a20 - a10 * a22, c02 = a10 * a21 - a11 * a20,
                         c10 = a02 * a21 - a01 * a22, c11 = a00 * a22 - a02 * a20, c12 = a01 * a20 - a00 * a21,
                         c20 = a01 * a12 - a02 * a11, c21 = a02 * a10 - a00 * a12, c22 = a00 * a11 - a01 * a10;
            const double s = det < 0 ? -1 : 1;
            for (size_t k = 0; k < nv; ++k) {
                const double x = buf[3 * k], y = buf[3 * k + 1], z = buf[3 * k + 2];
                Vec3 n = zUp(c00 * x + c10 * y + c20 * z, c01 * x + c11 * y + c21 * z, c02 * x + c12 * y + c22 * z) * s;
                const double l = length(n);
                m.normals.push_back(l > 0 ? n / l : Vec3());
            }
        } else {
            nrm = nullptr;
        }
        std::vector<float> uv;
        if (tex && tex->count == nv) {
            anyUvs = true;
            uv.resize(nv * 2);
            cgltf_accessor_unpack_floats(tex, uv.data(), nv * 2);
        }
        const size_t ni = prim.indices ? prim.indices->count : nv;
        idx.resize(ni);
        if (prim.indices) cgltf_accessor_unpack_indices(prim.indices, idx.data(), sizeof(uint32_t), ni);
        else for (size_t k = 0; k < ni; ++k) idx[k] = uint32_t(k);
        uint32_t part = unnamed;
        if (prim.material) part = uint32_t(cgltf_material_index(raw, prim.material));
        else anyUnnamed = true;
        for (size_t t = 0; t + 2 < ni; t += 3) {
            // A mirroring transform reverses the winding; restore counter-clockwise fronts.
            std::array<uint32_t, 3> c{idx[t], idx[t + 1], idx[t + 2]};
            if (det < 0) std::swap(c[1], c[2]);
            for (uint32_t v : c)
                if (v >= nv) throw std::runtime_error("glTF index out of range in " + path);
            m.triangles.push_back({uint32_t(base + c[0]), uint32_t(base + c[1]), uint32_t(base + c[2])});
            m.normalIndex.push_back(nrm ? std::array<uint32_t, 3>{uint32_t(nbase + c[0]), uint32_t(nbase + c[1]),
                                                                  uint32_t(nbase + c[2])}
                                        : std::array<uint32_t, 3>{MeshData::kNoNormal, MeshData::kNoNormal, MeshData::kNoNormal});
            MeshShape::UvCorners w{};
            if (!uv.empty())
                for (int k = 0; k < 3; ++k) w[size_t(k)] = {uv[2 * c[size_t(k)]], uv[2 * c[size_t(k)] + 1]};
            m.uvs.push_back(w);
            m.part.push_back(part);
        }
    };

    // The default scene (or the first), walked from its roots with each node's world transform.
    const cgltf_scene* scene = raw->scene ? raw->scene : (raw->scenes_count ? &raw->scenes[0] : nullptr);
    std::vector<const cgltf_node*> stack;
    if (scene) {
        for (size_t i = 0; i < scene->nodes_count; ++i) stack.push_back(scene->nodes[i]);
    } else {
        for (size_t i = 0; i < raw->nodes_count; ++i)
            if (!raw->nodes[i].parent) stack.push_back(&raw->nodes[i]);
    }
    while (!stack.empty()) {
        const cgltf_node* node = stack.back();
        stack.pop_back();
        for (size_t i = 0; i < node->children_count; ++i) stack.push_back(node->children[i]);
        if (!node->mesh) continue;
        float M[16];
        cgltf_node_transform_world(node, M);
        for (size_t p = 0; p < node->mesh->primitives_count; ++p) addPrimitive(node->mesh->primitives[p], M);
    }
    if (m.triangles.empty()) throw std::runtime_error("glTF file has no triangles: " + path);
    if (!anyNormals) {
        m.normals.clear();
        m.normalIndex.clear();
    }
    if (!anyUvs) m.uvs.clear();
    if (anyUnnamed) {
        m.partNames.push_back("unnamed");
        m.partImages.emplace_back();
    }
    return m;
}

MeshData loadModel(const std::string& path, double scale) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return ext == ".gltf" || ext == ".glb" ? loadGltf(path, scale) : loadObj(path, scale);
}

MeshData selectParts(const MeshData& m, const std::vector<std::string>& names) {
    std::vector<char> keep(m.partNames.size(), 0);
    for (const std::string& n : names) {
        auto it = std::find(m.partNames.begin(), m.partNames.end(), n);
        if (it == m.partNames.end()) {
            std::string all;
            for (const std::string& p : m.partNames) all += (all.empty() ? "" : ", ") + p;
            throw std::runtime_error("no part named '" + n + "' (parts: " + (all.empty() ? "none" : all) + ")");
        }
        keep[size_t(it - m.partNames.begin())] = 1;
    }
    MeshData out;
    out.positions = m.positions;
    out.normals = m.normals;
    out.partNames = m.partNames;
    out.partImages = m.partImages;
    for (size_t t = 0; t < m.triangles.size(); ++t) {
        if (!keep[m.part[t]]) continue;
        out.triangles.push_back(m.triangles[t]);
        if (!m.normalIndex.empty()) out.normalIndex.push_back(m.normalIndex[t]);
        if (!m.uvs.empty()) out.uvs.push_back(m.uvs[t]);
        out.part.push_back(m.part[t]);
    }
    if (out.triangles.empty()) throw std::runtime_error("the selected parts have no triangles");
    return out;
}

}  // namespace owe
