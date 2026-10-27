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
std::string textureImage(const cgltf_texture* t, const std::filesystem::path& dir) {
    if (!t || !t->image || !t->image->uri || std::strncmp(t->image->uri, "data:", 5) == 0) return {};
    std::string uri = t->image->uri;
    cgltf_decode_uri(uri.data());
    uri.resize(std::strlen(uri.c_str()));
    return (dir / uri).lexically_normal().string();
}

// One glTF file turned into a polygon soup: its materials as parts, then the triangle primitives
// of the meshes its scene's nodes place, in the engine's frame.
class GltfImporter {
public:
    GltfImporter(const cgltf_data& raw, const std::string& path, double scale) : raw_(raw), path_(path), scale_(scale) {}

    MeshData run() {
        materials();
        for (const cgltf_node* node : meshNodes()) {
            float M[16];
            cgltf_node_transform_world(node, M);
            for (size_t p = 0; p < node->mesh->primitives_count; ++p) primitive(node->mesh->primitives[p], M);
        }
        return finish();
    }

private:
    // Each material is a part, with the images of its base colour, normal and roughness maps.
    void materials() {
        const std::filesystem::path dir = std::filesystem::path(path_).parent_path();
        for (size_t i = 0; i < raw_.materials_count; ++i) {
            const cgltf_material& mat = raw_.materials[i];
            m_.partNames.push_back(mat.name && *mat.name ? mat.name : "material" + std::to_string(i));
            const cgltf_pbr_metallic_roughness* pbr = mat.has_pbr_metallic_roughness ? &mat.pbr_metallic_roughness : nullptr;
            m_.partImages.push_back(pbr ? textureImage(pbr->base_color_texture.texture, dir) : "");
            m_.partNormalImages.push_back(textureImage(mat.normal_texture.texture, dir));
            m_.partRoughnessImages.push_back(pbr ? textureImage(pbr->metallic_roughness_texture.texture, dir) : "");
        }
    }

    // The nodes with a mesh of the default scene (or the first, or every root), walked from its roots.
    std::vector<const cgltf_node*> meshNodes() const {
        const cgltf_scene* scene = raw_.scene ? raw_.scene : (raw_.scenes_count ? &raw_.scenes[0] : nullptr);
        std::vector<const cgltf_node*> stack, out;
        if (scene) {
            for (size_t i = 0; i < scene->nodes_count; ++i) stack.push_back(scene->nodes[i]);
        } else {
            for (size_t i = 0; i < raw_.nodes_count; ++i)
                if (!raw_.nodes[i].parent) stack.push_back(&raw_.nodes[i]);
        }
        while (!stack.empty()) {
            const cgltf_node* node = stack.back();
            stack.pop_back();
            for (size_t i = 0; i < node->children_count; ++i) stack.push_back(node->children[i]);
            if (node->mesh) out.push_back(node);
        }
        return out;
    }

    // A triangle primitive placed by world matrix M (column-major).
    void primitive(const cgltf_primitive& prim, const float M[16]) {
        if (prim.type != cgltf_primitive_type_triangles) return;
        if (prim.has_draco_mesh_compression)
            throw std::runtime_error("glTF file " + path_ + " is Draco-compressed; export it uncompressed");
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
        const double rows[3][3] = {{M[0], M[4], M[8]}, {M[1], M[5], M[9]}, {M[2], M[6], M[10]}};
        const Mat3 A = Mat3::fromRows(rows);
        const bool mirrored = A.determinant() < 0;
        const size_t base = m_.positions.size(), nbase = m_.normals.size();
        positions(pos, A, Vec3(M[12], M[13], M[14]));
        if (nrm && nrm->count == nv) normals(nrm, A, mirrored);
        else nrm = nullptr;
        std::vector<float> uv;
        if (tex && tex->count == nv) {
            anyUvs_ = true;
            uv.resize(nv * 2);
            cgltf_accessor_unpack_floats(tex, uv.data(), nv * 2);
        }
        const size_t ni = prim.indices ? prim.indices->count : nv;
        idx_.resize(ni);
        if (prim.indices) cgltf_accessor_unpack_indices(prim.indices, idx_.data(), sizeof(uint32_t), ni);
        else for (size_t k = 0; k < ni; ++k) idx_[k] = uint32_t(k);
        uint32_t part = uint32_t(raw_.materials_count);  // primitives without a material: "unnamed"
        if (prim.material) part = uint32_t(cgltf_material_index(&raw_, prim.material));
        else anyUnnamed_ = true;
        for (size_t t = 0; t + 2 < ni; t += 3) {
            // A mirroring transform reverses the winding; restore counter-clockwise fronts.
            std::array<uint32_t, 3> c{idx_[t], idx_[t + 1], idx_[t + 2]};
            if (mirrored) std::swap(c[1], c[2]);
            for (uint32_t v : c)
                if (v >= nv) throw std::runtime_error("glTF index out of range in " + path_);
            m_.triangles.push_back({uint32_t(base + c[0]), uint32_t(base + c[1]), uint32_t(base + c[2])});
            m_.normalIndex.push_back(nrm ? std::array<uint32_t, 3>{uint32_t(nbase + c[0]), uint32_t(nbase + c[1]),
                                                                   uint32_t(nbase + c[2])}
                                         : std::array<uint32_t, 3>{MeshData::kNoNormal, MeshData::kNoNormal, MeshData::kNoNormal});
            MeshShape::UvCorners w{};
            if (!uv.empty())
                for (int k = 0; k < 3; ++k) w[size_t(k)] = {uv[2 * c[size_t(k)]], uv[2 * c[size_t(k)] + 1]};
            m_.uvs.push_back(w);
            m_.part.push_back(part);
        }
    }

    // glTF's y-up turned to z-up: (x, y, z) → (x, −z, y).
    static Vec3 zUp(const Vec3& v) { return Vec3(v.x, -v.z, v.y); }

    void positions(const cgltf_accessor* pos, const Mat3& A, const Vec3& translation) {
        const size_t nv = pos->count;
        buf_.resize(nv * 3);
        cgltf_accessor_unpack_floats(pos, buf_.data(), nv * 3);
        for (size_t k = 0; k < nv; ++k)
            m_.positions.push_back(zUp(A * Vec3(buf_[3 * k], buf_[3 * k + 1], buf_[3 * k + 2]) + translation) * scale_);
    }

    // Normals transform by the cofactor matrix (the inverse transpose up to the determinant's scale).
    void normals(const cgltf_accessor* nrm, const Mat3& A, bool mirrored) {
        const size_t nv = nrm->count;
        anyNormals_ = true;
        buf_.resize(nv * 3);
        cgltf_accessor_unpack_floats(nrm, buf_.data(), nv * 3);
        const Mat3 C = A.cofactor();
        const double s = mirrored ? -1 : 1;
        for (size_t k = 0; k < nv; ++k) {
            const Vec3 n = zUp(C * Vec3(buf_[3 * k], buf_[3 * k + 1], buf_[3 * k + 2])) * s;
            const double l = length(n);
            m_.normals.push_back(l > 0 ? n / l : Vec3());
        }
    }

    MeshData finish() {
        if (m_.triangles.empty()) throw std::runtime_error("glTF file has no triangles: " + path_);
        if (!anyNormals_) {
            m_.normals.clear();
            m_.normalIndex.clear();
        }
        if (!anyUvs_) m_.uvs.clear();
        if (anyUnnamed_) {
            m_.partNames.push_back("unnamed");
            m_.partImages.emplace_back();
            m_.partNormalImages.emplace_back();
            m_.partRoughnessImages.emplace_back();
        }
        return std::move(m_);
    }

    const cgltf_data& raw_;
    const std::string& path_;
    const double scale_;
    MeshData m_;
    bool anyNormals_ = false, anyUvs_ = false, anyUnnamed_ = false;
    std::vector<float> buf_;
    std::vector<uint32_t> idx_;
};

}  // namespace

MeshData loadGltf(const std::string& path, double scale) {
    cgltf_options options{};
    cgltf_data* raw = nullptr;
    cgltf_result r = cgltf_parse_file(&options, path.c_str(), &raw);
    if (r != cgltf_result_success) throw std::runtime_error("cannot read glTF file " + path + ": " + resultName(r));
    std::unique_ptr<cgltf_data, void (*)(cgltf_data*)> data(raw, cgltf_free);
    r = cgltf_load_buffers(&options, raw, path.c_str());
    if (r != cgltf_result_success) throw std::runtime_error("cannot load the buffers of " + path + ": " + resultName(r));
    return GltfImporter(*raw, path, scale).run();
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
    out.partNormalImages = m.partNormalImages;
    out.partRoughnessImages = m.partRoughnessImages;
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
