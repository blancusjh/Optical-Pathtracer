#include "owe/scene/prescription.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace owe {

namespace {
double parseNumber(const std::string& s, const std::string& ctx) {
    if (s == "inf" || s == "infinity" || s == "INF" || s == "Infinity") return Inf;
    if (s == "-inf") return -Inf;
    size_t pos = 0;
    double v;
    try { v = std::stod(s, &pos); } catch (...) { throw std::runtime_error(ctx + ": expected a number, got '" + s + "'"); }
    if (pos != s.size()) throw std::runtime_error(ctx + ": trailing characters in '" + s + "'");
    return v;
}
}  // namespace

Prescription parsePrescription(const std::string& text, const std::string& origin) {
    Prescription p;
    p.source = text;
    std::istringstream in(text);
    std::string line;
    double unit = 1e-3;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream ls(line);
        std::vector<std::string> tok;
        std::string t;
        while (ls >> t) tok.push_back(t);
        if (tok.empty()) continue;
        std::string ctx = origin + ":" + std::to_string(lineNo);
        const std::string& key = tok[0];
        if (key == "name") {
            std::string rest = line.substr(line.find("name") + 4);
            auto a = rest.find_first_not_of(" \t\"");
            auto b = rest.find_last_not_of(" \t\"");
            p.name = a == std::string::npos ? "" : rest.substr(a, b - a + 1);
        } else if (key == "units") {
            if (tok.size() < 2) throw std::runtime_error(ctx + ": units needs a value");
            if (tok[1] == "mm") unit = 1e-3;
            else if (tok[1] == "m") unit = 1;
            else if (tok[1] == "cm") unit = 1e-2;
            else if (tok[1] == "um") unit = 1e-6;
            else if (tok[1] == "in") unit = 0.0254;
            else throw std::runtime_error(ctx + ": unknown unit " + tok[1]);
        } else if (key == "ambient") {
            p.ambient = tok.at(1);
        } else if (key == "object_distance") {
            p.objectDistance = parseNumber(tok.at(1), ctx) * unit;
        } else if (key == "wavelength") {
            p.wavelength = parseNumber(tok.at(1), ctx);
        } else {
            if (tok.size() < 5) throw std::runtime_error(ctx + ": surface lines need: label R t medium semi-diameter");
            PrescriptionSurface s;
            s.label = key;
            s.R = parseNumber(tok[1], ctx) * unit;
            s.t = parseNumber(tok[2], ctx) * unit;
            s.medium = tok[3];
            s.sd = parseNumber(tok[4], ctx) * unit;
            for (size_t i = 5; i < tok.size(); ++i) {
                const std::string& x = tok[i];
                if (x == "stop") { s.stop = true; continue; }
                if (x == "reflect") { s.reflect = true; continue; }
                if (x == "solve") { s.solve = true; continue; }
                auto eq = x.find('=');
                if (eq == std::string::npos) throw std::runtime_error(ctx + ": unexpected token '" + x + "'");
                std::string k = x.substr(0, eq);
                if (k == "material") { s.material = x.substr(eq + 1); continue; }
                if (k == "out_axis") throw std::runtime_error(ctx + ": out_axis (folded, non-coaxial mirrors) is not supported yet");
                if (k == "support") {
                    // spider(count=N,width=w,angle=deg,outer_radius=r)
                    std::string v = x.substr(eq + 1);
                    if (v.rfind("spider(", 0) != 0 || v.back() != ')') throw std::runtime_error(ctx + ": support must be spider(...)");
                    std::istringstream args(v.substr(7, v.size() - 8));
                    std::string kv;
                    while (std::getline(args, kv, ',')) {
                        auto e = kv.find('=');
                        if (e == std::string::npos) throw std::runtime_error(ctx + ": spider takes name=value pairs");
                        std::string n = kv.substr(0, e);
                        double val = parseNumber(kv.substr(e + 1), ctx);
                        if (n == "count") s.spiderCount = int(val);
                        else if (n == "width") s.spiderWidth = val * unit;
                        else if (n == "angle") s.spiderAngle = val * Pi / 180;
                        else if (n == "outer_radius") s.spiderOuter = val * unit;
                        else throw std::runtime_error(ctx + ": unknown spider parameter '" + n + "'");
                    }
                    if (s.spiderCount < 1 || !(s.spiderWidth > 0) || !(s.spiderOuter > 0))
                        throw std::runtime_error(ctx + ": spider needs count >= 1, width > 0 and outer_radius > 0");
                    continue;
                }
                double v = parseNumber(x.substr(eq + 1), ctx);
                if (k == "substrate") { s.substrate = v * unit; continue; }
                if (k == "hole") { s.hole = v * unit; continue; }
                if (k == "k") s.k = v;
                else if (k.size() > 1 && k[0] == 'A') {
                    int order = std::stoi(k.substr(1));
                    if (order < 4 || order % 2) throw std::runtime_error(ctx + ": aspheric terms are A4, A6, A8, ...");
                    size_t slot = size_t((order - 4) / 2);
                    if (s.A.size() <= slot) s.A.resize(slot + 1, 0.0);
                    s.A[slot] = v * std::pow(unit, 1 - order);  // coefficient of r^order: [length^(1-order)]
                } else throw std::runtime_error(ctx + ": unknown key '" + k + "'");
            }
            if (std::isinf(s.t)) throw std::runtime_error(ctx + ": thickness must be finite");
            if (s.reflect && (s.material.empty() || !(s.substrate > 0)))
                throw std::runtime_error(ctx + ": a reflecting row needs material= and substrate= > 0");
            if (s.reflect && s.hole >= 2 * s.sd) throw std::runtime_error(ctx + ": the hole must be smaller than the mirror");
            p.surfaces.push_back(s);
        }
    }
    if (p.surfaces.empty()) throw std::runtime_error(origin + ": prescription has no surfaces");
    if (hasMirrors(p)) {
        // Coaxial mirrors: each leg's t carries the sign of its travel; glass only on +z legs.
        double dir = 1;
        for (size_t i = 0; i < p.surfaces.size(); ++i) {
            const PrescriptionSurface& sf = p.surfaces[i];
            if (sf.reflect) {
                if (sf.medium != p.ambient) throw std::runtime_error(origin + ": " + sf.label + ": a mirror reflects into the ambient medium");
                dir = -dir;
            } else if (dir < 0) {
                throw std::runtime_error(origin + ": " + sf.label + ": refraction on a leg travelling -z is not supported yet");
            }
            if (i + 1 < p.surfaces.size() && sf.t * dir <= 0)
                throw std::runtime_error(origin + ": " + sf.label + ": t must have the sign of the travel (" +
                                         (dir > 0 ? "+" : "-") + "z) after this row");
        }
    }
    return p;
}

bool hasMirrors(const Prescription& p) {
    for (const auto& s : p.surfaces)
        if (s.reflect) return true;
    return false;
}

Prescription loadPrescription(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open lens file: " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return parsePrescription(ss.str(), path);
}

std::string formatPrescription(const Prescription& p, double unit) {
    std::ostringstream os;
    os << "# " << (p.name.empty() ? "unnamed" : p.name) << "\n";
    os << "units " << (unit == 1e-3 ? "mm" : "m") << "\n";
    os << std::left << std::setw(8) << "# label" << std::right << std::setw(14) << "R" << std::setw(12) << "t"
       << "  " << std::left << std::setw(14) << "medium" << std::right << std::setw(10) << "sd" << "\n";
    for (const auto& s : p.surfaces) {
        os << std::left << std::setw(8) << s.label << std::right << std::setw(14)
           << (std::isinf(s.R) ? std::string("inf") : std::to_string(s.R / unit)) << std::setw(12) << s.t / unit << "  "
           << std::left << std::setw(14) << s.medium << std::right << std::setw(10) << s.sd / unit;
        if (s.k != 0) os << "  k=" << s.k;
        for (size_t i = 0; i < s.A.size(); ++i)
            if (s.A[i] != 0) os << "  A" << 4 + 2 * i << "=" << s.A[i] / std::pow(unit, -3.0 - 2.0 * double(i));
        if (s.stop) os << "  stop";
        os << "\n";
    }
    return os.str();
}

IndexFn catalogIndex() {
    return [](const std::string& name, double l) {
        Medium m;
        if (!catalogMedium(name, m)) throw std::runtime_error("unknown medium in prescription: " + name);
        return m.n(l);
    };
}

namespace {
struct M2 {
    double a = 1, b = 0, c = 0, d = 1;
    M2 operator*(const M2& o) const {
        return {a * o.a + b * o.c, a * o.b + b * o.d, c * o.a + d * o.c, c * o.b + d * o.d};
    }
};
// Element list: refraction at surface i then transfer t_i (except after the last).
// Mirror systems are unfolded: transfers are |t| along the travel, and a mirror met travelling
// d = ±1 acts as a thin lens of power −2 d n / R (fixed-chart R): phi is each surface's power.
struct ParaxialSystem {
    std::vector<double> c, nBefore, nAfter, t, phi, z;  // z: vertex positions in the fixed chart
    double dirLast = 1;                                 // travel after the last surface
};
ParaxialSystem makeSystem(const Prescription& p, double l, const IndexFn& index) {
    ParaxialSystem s;
    double nPrev = index(p.ambient, l);
    double dir = 1, z = 0;
    for (const auto& sf : p.surfaces) {
        const double c = std::isinf(sf.R) ? 0.0 : 1.0 / sf.R;
        s.c.push_back(c);
        s.nBefore.push_back(nPrev);
        double n = index(sf.medium, l);
        s.nAfter.push_back(n);
        s.phi.push_back(sf.reflect ? -2 * dir * n * c : (n - nPrev) * c);
        if (sf.reflect) dir = -dir;
        s.t.push_back(std::abs(sf.t));
        s.z.push_back(z);
        z += sf.t;
        nPrev = n;
    }
    s.dirLast = dir;
    return s;
}
// Matrix from just before surface i0 to just after surface i1 (inclusive), transfers between them.
M2 systemMatrix(const ParaxialSystem& s, size_t i0, size_t i1) {
    M2 m;
    for (size_t i = i0; i <= i1; ++i) {
        M2 R{1, 0, -s.phi[i], 1};
        m = R * m;
        if (i < i1) {
            M2 T{1, s.t[i] / s.nAfter[i], 0, 1};
            m = T * m;
        }
    }
    return m;
}
}  // namespace

Paraxial paraxialAnalysis(const Prescription& p, double l, const IndexFn& index) {
    Paraxial r;
    ParaxialSystem s = makeSystem(p, l, index);
    size_t N = p.surfaces.size();
    M2 m = systemMatrix(s, 0, N - 1);
    r.A = m.a; r.B = m.b; r.C = m.c; r.D = m.d;
    double n0 = s.nBefore[0], nk = s.nAfter[N - 1];
    std::vector<double> z(N, 0.0);
    for (size_t i = 1; i < N; ++i) z[i] = z[i - 1] + s.t[i - 1];  // unfolded
    r.length = z[N - 1];
    double scale = 1e-12 * (1 + std::abs(m.a) + std::abs(m.d));
    r.afocal = std::abs(m.c) * std::max(1.0, r.length) < scale * 1e3 || std::abs(m.c) < 1e-9;
    if (!r.afocal) {
        r.efl = -nk / m.c;          // image-space focal length n'/φ
        r.bfl = -nk * m.a / m.c;    // from last vertex (ray y=1, u=0: y' = A, n'u' = C)
        r.ffl = n0 * m.d / m.c;     // from first vertex, negative = in front
    } else {
        r.angularMagnification = m.d * n0 / nk;
    }
    // Image of the object point.
    if (std::isinf(p.objectDistance)) {
        r.imageDistance = r.afocal ? Inf : r.bfl;
    } else {
        double u = 1e-3;
        double y0 = p.objectDistance * u;  // height at first vertex of a ray from the axial object point
        double yk = m.a * y0 + m.b * n0 * u;
        double nuk = m.c * y0 + m.d * n0 * u;
        r.imageDistance = nuk == 0 ? Inf : -yk * nk / nuk;
    }
    // Aperture stop: explicit, else the surface limiting a marginal ray from the object.
    int stop = -1;
    for (size_t i = 0; i < N; ++i)
        if (p.surfaces[i].stop) stop = int(i);
    if (stop < 0) {
        double best = -1;
        double y = 1, nu = std::isinf(p.objectDistance) ? 0 : n0 / p.objectDistance;
        for (size_t i = 0; i < N; ++i) {
            double ratio = std::abs(y) / std::max(p.surfaces[i].sd, 1e-12);
            if (ratio > best) { best = ratio; stop = int(i); }
            nu -= s.phi[i] * y;
            if (i + 1 < N) y += s.t[i] * nu / s.nAfter[i];
        }
    }
    r.stopIndex = stop;
    double stopR = p.surfaces[size_t(stop)].sd;
    // Exit pupil: image of the stop by the surfaces after it (the stop surface itself included when powered).
    {
        double y = 0, nu = 1e-3 * s.nAfter[size_t(stop)];
        double yM = stopR, nuM = 0;  // marginal-type ray from the stop edge
        for (size_t i = size_t(stop) + 1; i < N; ++i) {
            y += s.t[i - 1] * nu / s.nAfter[i - 1];
            yM += s.t[i - 1] * nuM / s.nAfter[i - 1];
            double phi = s.phi[i];
            nu -= phi * y;
            nuM -= phi * yM;
        }
        // Chief ray (through the stop centre) crosses the axis at the exit pupil; the fixed chart
        // places it along the last leg's travel.
        double u = nu / nk;
        double zLast = s.z[N - 1];
        if (size_t(stop) == N - 1) { r.exitPupilZ = zLast; r.exitPupilRadius = stopR; }
        else if (u != 0) {
            double dz = -y / u;
            r.exitPupilZ = zLast + s.dirLast * dz;
            // Height of the stop-edge ray at the pupil plane gives the pupil radius.
            r.exitPupilRadius = std::abs(yM + dz * nuM / nk);
        }
    }
    // Entrance pupil: image of the stop by the surfaces before it, traced backwards.
    {
        double y = 0, nu = 1e-3 * s.nBefore[size_t(stop)];
        double yM = stopR, nuM = 0;
        // Going backwards: refraction at surface i maps n'u' → n u = n'u' + φ y; transfer uses the index before i.
        for (int i = stop - 1; i >= 0; --i) {
            y -= s.t[size_t(i)] * nu / s.nAfter[size_t(i)];
            yM -= s.t[size_t(i)] * nuM / s.nAfter[size_t(i)];
            double phi = s.phi[size_t(i)];
            nu += phi * y;
            nuM += phi * yM;
        }
        if (stop == 0) { r.entrancePupilZ = 0; r.entrancePupilRadius = stopR; }
        else {
            double u = nu / n0;
            if (u != 0) {
                double dz = -y / u;  // along +z from first vertex
                r.entrancePupilZ = dz;
                r.entrancePupilRadius = std::abs(yM + dz * nuM / n0);
            }
        }
    }
    if (!r.afocal && r.entrancePupilRadius > 0) r.fNumber = std::abs(r.efl) / (2 * r.entrancePupilRadius);
    return r;
}

double solveAfocal(Prescription& p, double l, const IndexFn& index) {
    // The gap marked `solve` (mirror systems: exactly one), else the largest air gap.
    size_t gap = p.surfaces.size();
    double best = -1;
    int marked = 0;
    for (size_t i = 0; i + 1 < p.surfaces.size(); ++i)
        if (p.surfaces[i].solve) { gap = i; ++marked; }
    if (marked > 1) throw std::runtime_error("solveAfocal: more than one gap is marked solve");
    if (marked == 0 && hasMirrors(p)) throw std::runtime_error("solveAfocal: mark the gap to adjust with solve");
    if (marked == 0)
        for (size_t i = 0; i + 1 < p.surfaces.size(); ++i)
            if (p.surfaces[i].medium == p.ambient && p.surfaces[i].t > best) { best = p.surfaces[i].t; gap = i; }
    if (gap == p.surfaces.size()) throw std::runtime_error("solveAfocal: no air gap to adjust");
    // C is affine in the gap's length; its sign (the leg's travel) is kept.
    double old = p.surfaces[gap].t;
    const double sign = old < 0 ? -1 : 1;
    p.surfaces[gap].t = 0;
    double c0 = paraxialAnalysis(p, l, index).C;
    p.surfaces[gap].t = sign;
    double c1 = paraxialAnalysis(p, l, index).C;
    if (c1 == c0) { p.surfaces[gap].t = old; throw std::runtime_error("solveAfocal: gap does not affect power"); }
    double g = -c0 / (c1 - c0);
    if (!(g > 0)) { p.surfaces[gap].t = old; throw std::runtime_error("solveAfocal: no positive afocal separation"); }
    p.surfaces[gap].t = sign * g;
    return g;
}

BuiltInstrument buildPrescription(World& w, const Prescription& p, const std::string& name, int parent,
                                  const Transform& xf, double tubeRadius, const IndexFn& index, LensSpec::Rim rim,
                                  const MaterialFn& materialOf) {
    BuiltInstrument bi;
    bi.assembly = w.addAssembly(name, parent, xf);
    bi.paraxial = paraxialAnalysis(p, p.wavelength, index);
    size_t N = p.surfaces.size();
    bi.vertexZ.assign(N, 0.0);
    for (size_t i = 1; i < N; ++i) bi.vertexZ[i] = bi.vertexZ[i - 1] + p.surfaces[i - 1].t;
    bi.lastVertexZ = bi.vertexZ[N - 1];
    size_t i = 0;
    int group = 0;
    double dir = 1;  // travel along the fixed chart's z before surface i
    while (i < N) {
        const auto& s = p.surfaces[i];
        std::string before = i == 0 ? p.ambient : p.surfaces[i - 1].medium;
        if (s.reflect) {
            // The mirror's own frame has its reflecting face toward +z: met travelling +z, it is
            // turned over (and its sag with it).
            SurfaceSpec f;
            f.R = dir > 0 ? -s.R : s.R;
            f.k = s.k;
            f.A = s.A;
            if (dir > 0)
                for (double& a : f.A) a = -a;
            f.semiDiameter = s.sd;
            uint32_t optics;
            if (materialOf) optics = materialOf(s.material);
            else {
                SurfaceOptics o;
                o.type = SurfaceType::Mirror;
                optics = w.addOptics(o);
            }
            Transform at = Transform::translate({0, 0, bi.vertexZ[i]});
            if (dir > 0) at = at * Transform::rotate({1, 0, 0}, Pi);
            bi.bodies.push_back(buildMirror(w, name + "." + s.label, f, s.substrate, optics, bi.assembly, at, s.hole / 2));
            bi.maxRadius = std::max(bi.maxRadius, s.sd);
            // Struts behind the mirror (on the side away from its face), one substrate deep.
            for (int k = 0; k < s.spiderCount; ++k) {
                const double a = s.spiderAngle + 2 * Pi * k / s.spiderCount;
                const double zBack = std::min(0.0, SagSurface(f.curvature(), f.k, f.A, f.semiDiameter).sag(f.semiDiameter)) - s.substrate;
                Transform strut = at * Transform::rotate({0, 0, 1}, a) *
                                  Transform::translate({0.5 * s.spiderOuter, 0, zBack - 0.5 * s.substrate - 1e-6});
                BodyMaterial m;
                m.optics = w.absorberOptics();
                bi.bodies.push_back(buildBox(w, name + "." + s.label + ".strut" + std::to_string(k),
                                             {s.spiderOuter, s.spiderWidth, s.substrate}, m, bi.assembly, strut));
            }
            dir = -dir;
            ++i;
            continue;
        }
        if (before == p.ambient && s.medium == p.ambient) {
            if (s.stop && s.sd > 0) {
                double outer = tubeRadius > s.sd ? tubeRadius : s.sd * 3;
                int b = buildStop(w, name + "." + s.label, s.sd, outer, bi.assembly,
                                  Transform::translate({0, 0, bi.vertexZ[i]}));
                bi.bodies.push_back(b);
            }
            ++i;
            continue;
        }
        if (before != p.ambient) throw std::runtime_error("prescription: group does not start in the ambient medium");
        size_t j = i;
        while (j < N && p.surfaces[j].medium != p.ambient) ++j;
        if (j >= N) throw std::runtime_error("prescription: last surface must exit into the ambient medium");
        LensSpec spec;
        spec.rim = rim;
        for (size_t k = i; k <= j; ++k) {
            SurfaceSpec sf;
            sf.R = p.surfaces[k].R;
            sf.k = p.surfaces[k].k;
            sf.A = p.surfaces[k].A;
            sf.semiDiameter = p.surfaces[k].sd;
            spec.surfaces.push_back(sf);
            if (k < j) {
                spec.media.push_back(p.surfaces[k].medium);
                spec.thickness.push_back(p.surfaces[k].t);
            }
        }
        std::string label = name + ".L" + std::to_string(++group);
        int b = buildLens(w, label, spec, bi.assembly, Transform::translate({0, 0, bi.vertexZ[i]}));
        bi.bodies.push_back(b);
        double edge = w.bodies()[b].params["edge_radius"];
        bi.maxRadius = std::max(bi.maxRadius, edge);
        // Extent of the group along z.
        double zFirstEdge = bi.vertexZ[i] + SagSurface(spec.surfaces.front().curvature(), spec.surfaces.front().k,
                                                     spec.surfaces.front().A, spec.surfaces.front().semiDiameter)
                                                .sag(spec.surfaces.front().semiDiameter);
        double zLastEdge = bi.vertexZ[j] + SagSurface(spec.surfaces.back().curvature(), spec.surfaces.back().k,
                                                    spec.surfaces.back().A, spec.surfaces.back().semiDiameter)
                                               .sag(spec.surfaces.back().semiDiameter);
        bi.rearNearestZ = std::max({bi.vertexZ[j], zLastEdge, bi.vertexZ[i], zFirstEdge});
        bi.rearEdgeRadius = edge;
        if (tubeRadius > edge) {
            // The mount's bore clears the rim by 1e-9 relative (picometres) so the two bodies do
            // not share an edge; rays then never meet an ambiguous rim/mount junction.
            int m = buildStop(w, label + "-mount", edge * (1 + 1e-9), tubeRadius, bi.assembly,
                              Transform::translate({0, 0, 0.5 * (zFirstEdge + zLastEdge)}));
            bi.bodies.push_back(m);
        }
        i = j + 1;
    }
    return bi;
}

}  // namespace owe
