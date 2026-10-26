// Triangle meshes: intersection through their BVH, area sampling, face identity, and the mesh data
// builders and model files produce (shading normals included).
#include <stdexcept>

#include "owe/core/parallel.hpp"
#include "owe/scene/geometry.hpp"

namespace owe {

// ---------------------------------------------------------------- MeshShape

MeshShape::MeshShape(std::vector<Vec3> positions, std::vector<std::array<uint32_t, 3>> triangles, std::string label,
                     std::vector<Corners> shadingNormals, std::vector<UvCorners> uvs)
    : pos_(std::move(positions)), tris_(std::move(triangles)), normals_(std::move(shadingNormals)), uvs_(std::move(uvs)),
      label_(std::move(label)) {
    if (!normals_.empty() && normals_.size() != tris_.size())
        throw std::runtime_error("mesh: shading normals must be given for every triangle");
    if (!uvs_.empty() && uvs_.size() != tris_.size())
        throw std::runtime_error("mesh: texture coordinates must be given for every triangle");
    std::vector<AABB> boxes(tris_.size());
    cdf_.resize(tris_.size());
    parallelFor(tris_.size(), 1 << 15, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const auto& t = tris_[i];
            for (uint32_t v : t)
                if (v >= pos_.size()) throw std::runtime_error("mesh: triangle index out of range");
            const Vec3 &a = pos_[t[0]], &b = pos_[t[1]], &c = pos_[t[2]];
            boxes[i].expand(a);
            boxes[i].expand(b);
            boxes[i].expand(c);
            double pad = 1e-12 * (1 + maxAbsComponent(boxes[i].hi) + maxAbsComponent(boxes[i].lo));
            boxes[i].lo -= Vec3(pad, pad, pad);
            boxes[i].hi += Vec3(pad, pad, pad);
            cdf_[i] = 0.5 * length(cross(b - a, c - a));
        }
    });
    double acc = 0;  // the running sum in triangle order, as a sequential pass
    for (double& a : cdf_) a = acc += a;
    area_ = acc;
    bvh_.build(boxes);
}

uint32_t MeshShape::sampledPrimitive(double u) const {
    const auto& cdf = areaCdf();
    if (cdf.empty()) return ~0u;
    return uint32_t(std::min(size_t(std::lower_bound(cdf.begin(), cdf.end(), u * area()) - cdf.begin()),
                             cdf.size() - 1));
}

bool MeshShape::sameFace(uint32_t first, uint32_t second) const {
    if (first >= tris_.size() || second >= tris_.size()) return false;
    if (first == second) return true;
    const auto& a = tris_[first];
    const auto& b = tris_[second];
    auto sharedVertex = [&](uint32_t v) {
        for (uint32_t u : a)
            if (pos_[v].x == pos_[u].x && pos_[v].y == pos_[u].y && pos_[v].z == pos_[u].z) return true;
        return false;
    };
    int shared = 0;
    for (uint32_t v : b)
        if (sharedVertex(v)) ++shared;
    if (shared < 2) return false;
    const Vec3 normal = cross(pos_[a[1]] - pos_[a[0]], pos_[a[2]] - pos_[a[0]]);
    if (dot(normal, cross(pos_[b[1]] - pos_[b[0]], pos_[b[2]] - pos_[b[0]])) <= 0) return false;
    // Adjacent exactly coplanar triangles form the same face. Do not use a
    // distance tolerance: a distinct parallel layer can be closer than it.
    for (uint32_t v : b)
        if (!sharedVertex(v) && dot(normal, pos_[v] - pos_[a[0]]) != 0) return false;
    return true;
}

bool MeshShape::intersect(const Ray& r, double tmin, double tmax, LocalHit& h) const {
    bool found = false;
    double best = tmax;
    bvh_.traverse(r, tmin, best, [&](uint32_t i, double& tm) {
        const auto& t = tris_[i];
        const Vec3 &v0 = pos_[t[0]], &v1 = pos_[t[1]], &v2 = pos_[t[2]];
        Vec3 e1 = v1 - v0, e2 = v2 - v0;
        Vec3 pv = cross(r.d, e2);
        double det = dot(e1, pv);
        if (std::abs(det) < 1e-300) return;
        double inv = 1 / det;
        Vec3 tv = r.o - v0;
        double u = dot(tv, pv) * inv;
        if (u < 0 || u > 1) return;
        Vec3 qv = cross(tv, e1);
        double v = dot(r.d, qv) * inv;
        if (v < 0 || u + v > 1) return;
        double tt = dot(e2, qv) * inv;
        if (tt <= tmin || tt >= tm) return;
        // Near a grazing departure, t can be large even for a tiny plane error.
        // Primitive identity and normal displacement identify the actual self face.
        if (sameFace(r.departedPrimitive, i) &&
            std::abs(dot(tv, normalize(cross(e1, e2)))) <= r.departureError) return;
        tm = tt;
        found = true;
        h.t = tt;
        h.prim = i;
    });
    if (!found) return false;
    const auto& t = tris_[h.prim];
    h.p = r.at(h.t);
    h.n = normalize(triangleNormal(pos_[t[0]], pos_[t[1]], pos_[t[2]]));
    h.error = std::abs(dot(h.p - pos_[t[0]], h.n));
    return true;
}

std::string MeshShape::describe() const {
    return label_ + "(" + std::to_string(tris_.size()) + " triangles" + (normals_.empty() ? "" : ", smooth") + ")";
}

void MeshShape::sampleArea(double u1, double u2, double u3, Vec3& p, Vec3& n) const {
    double target = u3 * area_;
    size_t i = size_t(std::lower_bound(cdf_.begin(), cdf_.end(), target) - cdf_.begin());
    if (i >= tris_.size()) i = tris_.size() - 1;
    const auto& t = tris_[i];
    p = sampleTriangle(pos_[t[0]], pos_[t[1]], pos_[t[2]], u1, u2);
    n = normalize(triangleNormal(pos_[t[0]], pos_[t[1]], pos_[t[2]]));
}

// ---------------------------------------------------------------- MeshData

void MeshData::addTriangle(const Vec3& a, const Vec3& b, const Vec3& c) {
    uint32_t base = uint32_t(positions.size());
    positions.push_back(a);
    positions.push_back(b);
    positions.push_back(c);
    triangles.push_back({base, base + 1, base + 2});
    if (!normalIndex.empty()) normalIndex.resize(triangles.size(), {kNoNormal, kNoNormal, kNoNormal});
}

void MeshData::addQuad(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    uint32_t base = uint32_t(positions.size());
    positions.insert(positions.end(), {a, b, c, d});
    triangles.push_back({base, base + 1, base + 2});
    triangles.push_back({base, base + 2, base + 3});
    if (!normalIndex.empty()) normalIndex.resize(triangles.size(), {kNoNormal, kNoNormal, kNoNormal});
}

void MeshData::append(const MeshData& o, const Transform& xf) {
    uint32_t base = uint32_t(positions.size());
    if (!o.normalIndex.empty() || !normalIndex.empty()) {
        normalIndex.resize(triangles.size(), {kNoNormal, kNoNormal, kNoNormal});
        const uint32_t nb = uint32_t(normals.size());
        for (const Vec3& n : o.normals) normals.push_back(xf.vector(n));
        for (size_t i = 0; i < o.triangles.size(); ++i) {
            std::array<uint32_t, 3> c{kNoNormal, kNoNormal, kNoNormal};
            if (i < o.normalIndex.size())
                for (int k = 0; k < 3; ++k)
                    if (o.normalIndex[i][k] != kNoNormal) c[k] = o.normalIndex[i][k] + nb;
            normalIndex.push_back(c);
        }
    }
    for (const Vec3& p : o.positions) positions.push_back(xf.point(p));
    for (const auto& t : o.triangles) triangles.push_back({t[0] + base, t[1] + base, t[2] + base});
}

namespace {

Vec3 faceNormal(const MeshData& m, const std::array<uint32_t, 3>& t) {
    const Vec3 &a = m.positions[t[0]], &b = m.positions[t[1]], &c = m.positions[t[2]];
    Vec3 n = cross(b - a, c - a);
    double l = length(n);
    return l > 0 ? n / l : Vec3();
}

// A shading normal for a corner of a triangle with (unit) normal fn: n itself when it lies on the
// triangle's front side, else fn (a corner normal behind its own face would shade it from behind).
Vec3 frontCorner(const Vec3& n, const Vec3& fn) {
    double l = length(n);
    if (!(l > 0) || dot(n, fn) <= 1e-6 * l) return fn;
    return n / l;
}

}  // namespace

std::vector<MeshShape::Corners> smoothNormals(const MeshData& m, double creaseAngle) {
    const size_t nt = m.triangles.size();
    // Vertices at one position are one vertex (files split vertices at texture seams).
    std::vector<uint32_t> weld(m.positions.size());
    {
        std::vector<uint32_t> order(m.positions.size());
        for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        auto key = [&](uint32_t i) { const Vec3& p = m.positions[i]; return std::array<double, 3>{p.x, p.y, p.z}; };
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key(a) < key(b) || (key(a) == key(b) && a < b); });
        for (size_t k = 0; k < order.size(); ++k)
            weld[order[k]] = (k > 0 && key(order[k]) == key(order[k - 1])) ? weld[order[k - 1]] : order[k];
    }
    // Triangles around each welded vertex (compressed rows).
    std::vector<uint32_t> start(m.positions.size() + 1, 0), around(3 * nt);
    for (const auto& t : m.triangles)
        for (uint32_t v : t) ++start[weld[v] + 1];
    for (size_t i = 1; i < start.size(); ++i) start[i] += start[i - 1];
    {
        std::vector<uint32_t> fill(start.begin(), start.end() - 1);
        for (uint32_t i = 0; i < nt; ++i)
            for (uint32_t v : m.triangles[i]) around[fill[weld[v]]++] = i;
    }
    std::vector<Vec3> fn(nt);
    std::vector<std::array<double, 3>> angle(nt);
    parallelFor(nt, 1 << 14, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const auto& t = m.triangles[i];
            fn[i] = faceNormal(m, t);
            for (int k = 0; k < 3; ++k) {
                Vec3 a = m.positions[t[(k + 1) % 3]] - m.positions[t[k]], b = m.positions[t[(k + 2) % 3]] - m.positions[t[k]];
                double la = length(a), lb = length(b);
                angle[i][k] = la > 0 && lb > 0 ? std::acos(clamp(dot(a, b) / (la * lb), -1, 1)) : 0;
            }
        }
    });
    const double cosCrease = std::cos(creaseAngle);
    std::vector<MeshShape::Corners> out(nt);
    parallelFor(nt, 1 << 14, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const auto& t = m.triangles[i];
            for (int k = 0; k < 3; ++k) {
                const uint32_t v = weld[t[k]];
                Vec3 sum;
                for (uint32_t s = start[v]; s < start[v + 1]; ++s) {
                    const uint32_t j = around[s];
                    if (dot(fn[j], fn[i]) < cosCrease) continue;
                    int corner = 0;  // j's corner at this vertex
                    while (corner < 2 && weld[m.triangles[j][corner]] != v) ++corner;
                    sum += fn[j] * angle[j][corner];
                }
                out[i][k] = frontCorner(sum, fn[i]);
            }
        }
    });
    return out;
}

std::vector<MeshShape::Corners> fileNormals(const MeshData& m) {
    if (m.normals.empty() || m.normalIndex.size() != m.triangles.size())
        throw std::runtime_error("the mesh has no vertex normals (OBJ vn)");
    std::vector<MeshShape::Corners> out(m.triangles.size());
    for (size_t i = 0; i < m.triangles.size(); ++i) {
        const Vec3 fn = faceNormal(m, m.triangles[i]);
        for (int k = 0; k < 3; ++k) {
            const uint32_t n = m.normalIndex[i][k];
            out[i][k] = frontCorner(n < m.normals.size() ? m.normals[n] : Vec3(), fn);
        }
    }
    return out;
}

}  // namespace owe
