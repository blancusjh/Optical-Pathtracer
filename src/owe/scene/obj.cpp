// Wavefront OBJ models: positions, faces and vertex normals, parsed in parallel chunks.
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "owe/core/parallel.hpp"
#include "owe/scene/geometry.hpp"

namespace owe {

namespace {

// One chunk of an OBJ file, parsed independently: its positions, and its faces with absolute
// indices, except relative (negative) ones, which are recorded against the chunk's own vertices
// until the chunks' vertex offsets are known.
struct ObjChunk {
    std::vector<Vec3> positions;
    std::vector<std::array<uint32_t, 3>> triangles;
    std::vector<std::pair<size_t, long>> relative;  // (triangle * 3 + corner, chunk-local vertex)
    std::vector<Vec3> normals;
    std::vector<std::array<uint32_t, 3>> normalIndex;     // per triangle (kNoNormal: none)
    std::vector<std::pair<size_t, long>> relativeNormal;  // as `relative`, for normals
};

const char* skipBlanks(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
    return p;
}

// A number as strtod would read it (leading blanks, optional '+'); 0 without advancing on failure.
double parseNumber(const char*& p, const char* end) {
    const char* q = skipBlanks(p, end);
    if (q < end && *q == '+') ++q;
    double v = 0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    auto r = std::from_chars(q, end, v);
    if (r.ec == std::errc() || r.ec == std::errc::result_out_of_range) p = r.ptr;
#else
    // Without floating-point from_chars (Apple's libc++): strtod on a terminated copy of the token.
    char buf[64];
    size_t n = 0;
    while (q + n < end && n < sizeof buf - 1 && q[n] != ' ' && q[n] != '\t' && q[n] != '\n' && q[n] != '\r') {
        buf[n] = q[n];
        ++n;
    }
    buf[n] = 0;
    char* e = buf;
    v = std::strtod(buf, &e);
    if (e != buf) p = q + (e - buf);
    else v = 0;
#endif
    return v;
}

void parseObjChunk(const char* p, const char* end, double scale, ObjChunk& out) {
    std::vector<long> idx, nidx;
    auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (p < end) {
        p = skipBlanks(p, end);
        if (p + 1 < end && p[0] == 'v' && (p[1] == ' ' || p[1] == '\t')) {
            const char* q = p + 2;
            Vec3 v;
            v.x = parseNumber(q, end);
            v.y = parseNumber(q, end);
            v.z = parseNumber(q, end);
            out.positions.push_back(v * scale);
            p = q;
        } else if (p + 2 < end && p[0] == 'v' && p[1] == 'n' && (p[2] == ' ' || p[2] == '\t')) {
            const char* q = p + 3;
            Vec3 n;
            n.x = parseNumber(q, end);
            n.y = parseNumber(q, end);
            n.z = parseNumber(q, end);
            out.normals.push_back(n);  // a direction: the scale does not apply
            p = q;
        } else if (p + 1 < end && p[0] == 'f' && (p[1] == ' ' || p[1] == '\t')) {
            idx.clear();
            nidx.clear();
            const char* q = p + 2;
            while (true) {
                q = skipBlanks(q, end);
                if (q >= end || *q == '\n' || *q == '\r') break;
                long v = 0;
                auto r = std::from_chars(q + (*q == '+'), end, v);
                if (r.ec != std::errc()) break;
                q = r.ptr;
                long vn = 0;  // v, v/vt, v//vn or v/vt/vn; 0: no normal
                if (q < end && *q == '/') {
                    ++q;
                    while (q < end && *q != '/' && !blank(*q)) ++q;  // vt
                    if (q < end && *q == '/') {
                        ++q;
                        auto rn = std::from_chars(q + (q < end && *q == '+'), end, vn);
                        if (rn.ec == std::errc()) q = rn.ptr;
                        else vn = 0;
                    }
                }
                idx.push_back(v);
                nidx.push_back(vn);
                while (q < end && !blank(*q)) ++q;
            }
            for (size_t i = 1; i + 1 < idx.size(); ++i) {
                const size_t c3[3] = {0, i, i + 1};
                std::array<uint32_t, 3> t{}, tn{};
                for (int k = 0; k < 3; ++k) {
                    const long corner = idx[c3[k]], normal = nidx[c3[k]];
                    if (corner < 0) {
                        out.relative.push_back({out.triangles.size() * 3 + size_t(k),
                                                long(out.positions.size()) + corner});
                    } else {
                        t[size_t(k)] = uint32_t(corner - 1);
                    }
                    tn[size_t(k)] = MeshData::kNoNormal;
                    if (normal < 0) {
                        out.relativeNormal.push_back({out.triangles.size() * 3 + size_t(k), long(out.normals.size()) + normal});
                    } else if (normal > 0) {
                        tn[size_t(k)] = uint32_t(normal - 1);
                    }
                }
                out.triangles.push_back(t);
                out.normalIndex.push_back(tn);
            }
            p = q;
        }
        while (p < end && *p != '\n') ++p;
        ++p;
    }
}

}  // namespace

// Positions, faces and vertex normals (texture coordinates are skipped). Large files are parsed
// in parallel chunks split at line breaks and joined in file order, so the mesh is exactly the one
// a sequential reading gives.
MeshData loadObj(const std::string& path, double scale) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open OBJ file: " + path);
    std::string buf(size_t(in.tellg()), '\0');
    in.seekg(0);
    in.read(buf.data(), std::streamsize(buf.size()));
    if (!in) throw std::runtime_error("cannot read OBJ file: " + path);
    buf.push_back('\n');
    const char* begin = buf.data();
    const char* end = begin + buf.size();
    const size_t parts = std::clamp<size_t>(buf.size() >> 22, 1, hardwareThreads());  // ≥ 4 MB each
    std::vector<const char*> cut(parts + 1, end);
    cut[0] = begin;
    for (size_t k = 1; k < parts; ++k) {
        const char* c = std::max(cut[k - 1], begin + buf.size() * k / parts);
        while (c < end && c[-1] != '\n') ++c;
        cut[k] = c;
    }
    std::vector<ObjChunk> chunks(parts);
    parallelParts(parts, parts, [&](size_t k, size_t, size_t) { parseObjChunk(cut[k], cut[k + 1], scale, chunks[k]); });

    MeshData m;
    size_t nv = 0, nt = 0, nn = 0;
    for (const ObjChunk& c : chunks) { nv += c.positions.size(); nt += c.triangles.size(); nn += c.normals.size(); }
    m.positions.reserve(nv);
    m.triangles.reserve(nt);
    m.normals.reserve(nn);
    if (nn > 0) m.normalIndex.reserve(nt);
    for (const ObjChunk& c : chunks) {
        const long vertexBase = long(m.positions.size()), normalBase = long(m.normals.size());
        const size_t triBase = m.triangles.size();
        m.positions.insert(m.positions.end(), c.positions.begin(), c.positions.end());
        m.triangles.insert(m.triangles.end(), c.triangles.begin(), c.triangles.end());
        for (auto [slot, local] : c.relative)
            m.triangles[triBase + slot / 3][slot % 3] = uint32_t(vertexBase + local);
        if (nn == 0) continue;
        m.normals.insert(m.normals.end(), c.normals.begin(), c.normals.end());
        m.normalIndex.insert(m.normalIndex.end(), c.normalIndex.begin(), c.normalIndex.end());
        for (auto [slot, local] : c.relativeNormal)
            m.normalIndex[triBase + slot / 3][slot % 3] = uint32_t(normalBase + local);
    }
    if (m.triangles.empty()) throw std::runtime_error("OBJ file has no faces: " + path);
    for (auto& t : m.normalIndex)
        for (uint32_t& n : t)
            if (n != MeshData::kNoNormal && n >= m.normals.size()) throw std::runtime_error("OBJ normal index out of range: " + path);
    return m;
}

}  // namespace owe
