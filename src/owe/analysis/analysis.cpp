#include "owe/analysis/analysis.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace owe {

namespace {

struct LensWorld {
    Scene scene;
    BuiltInstrument bi;
    size_t surfaces = 0, mirrors = 0;
};

void makeLensWorld(const Prescription& p, LensWorld& lw) {
    lw.bi = buildPrescription(lw.scene.world, p, "lens", -1, Transform{}, 0.0, catalogIndex(), LensSpec::Rim::Black);
    lw.scene.world.build();
    lw.surfaces = 0;
    lw.mirrors = 0;
    for (size_t i = 0; i < p.surfaces.size(); ++i) {
        std::string before = i == 0 ? p.ambient : p.surfaces[i - 1].medium;
        if (p.surfaces[i].reflect) lw.mirrors++;
        else if (!(before == p.ambient && p.surfaces[i].medium == p.ambient)) lw.surfaces++;
    }
}

struct TracedRay {
    Vec3 p, d;
    bool ok = false;
};

TracedRay traceThrough(const LensWorld& lw, const Vec3& origin, const Vec3& dir, double lambda) {
    Tracer tr(lw.scene);
    Rng rng(1, 1);
    PathRecord rec = tr.walk(Ray{origin, dir}, lw.scene.world.ambientRegion(), lambda, WalkMode::PrimaryTransmission, rng, 64);
    TracedRay r;
    if (rec.v.size() < 2 || rec.v.back().event != EventKind::Escape) return r;
    size_t refr = 0, refl = 0;
    for (auto& v : rec.v) {
        if (v.event == EventKind::Refract) refr++;
        else if (v.event == EventKind::Reflect && lw.mirrors > 0) refl++;
        else if (v.event != EventKind::Emit && v.event != EventKind::Escape) return r;
    }
    if (refr != lw.surfaces || refl != lw.mirrors) return r;
    const PathVertex& last = rec.v[rec.v.size() - 2];
    r.p = last.p;
    r.d = last.dOut;
    r.ok = r.d.z > 0;
    return r;
}

Vec3 launchOrigin(double zEP, double x, double y, const Vec3& d, double zStart) {
    Vec3 P{x, y, zEP};
    return P + d * ((zStart - zEP) / d.z);
}

}  // namespace

LensReport analyzeLens(const Prescription& p, const std::vector<double>& fields, const std::vector<double>& lambdas,
                       int rings) {
    LensReport rep;
    LensWorld lw;
    makeLensWorld(p, lw);
    const Paraxial& px = lw.bi.paraxial;
    rep.paraxial = px;
    double zStart = -0.05 - 0.1 * px.length;
    double rEP = px.entrancePupilRadius;
    double zImg = lw.bi.lastVertexZ + (std::isinf(px.imageDistance) ? 0 : px.imageDistance);

    for (double lambda : lambdas)
        for (double f : fields) {
            SpotResult s;
            s.lambda = lambda;
            s.fieldDeg = f;
            Vec3 d{0, std::sin(radians(f)), std::cos(radians(f))};
            std::vector<TracedRay> rays;
            for (int k = 0; k <= rings; ++k) {
                int n = k == 0 ? 1 : 6 * k;
                for (int j = 0; j < n; ++j) {
                    double rr = rEP * double(k) / rings * 0.999;
                    double a = 2 * Pi * j / n;
                    TracedRay t = traceThrough(lw, launchOrigin(px.entrancePupilZ, rr * std::cos(a), rr * std::sin(a), d, zStart),
                                               d, lambda);
                    s.launched++;
                    if (t.ok) rays.push_back(t);
                }
            }
            s.arrived = int(rays.size());
            if (px.afocal && rays.size() >= 3) {
                double n = double(rays.size()), mx = 0, my = 0;
                for (auto& r : rays) { mx += r.d.x / r.d.z / n; my += r.d.y / r.d.z / n; }
                double v = 0;
                for (auto& r : rays) {
                    double ax = std::atan(r.d.x / r.d.z) - std::atan(mx), ay = std::atan(r.d.y / r.d.z) - std::atan(my);
                    v += ax * ax + ay * ay;
                }
                s.apparentAngle = std::atan(my);
                s.angularRms = std::sqrt(v / n);
            } else if (rays.size() >= 3) {
                // Best focus: minimise the spot variance over z (closed form).
                double n = double(rays.size());
                Vec2 ma{}, mb{};
                std::vector<Vec2> A(rays.size()), B(rays.size());
                for (size_t i = 0; i < rays.size(); ++i) {
                    const auto& r = rays[i];
                    B[i] = {r.d.x / r.d.z, r.d.y / r.d.z};
                    A[i] = {r.p.x - B[i].x * r.p.z, r.p.y - B[i].y * r.p.z};
                    ma.x += A[i].x / n; ma.y += A[i].y / n; mb.x += B[i].x / n; mb.y += B[i].y / n;
                }
                double cov = 0, varb = 0;
                for (size_t i = 0; i < rays.size(); ++i) {
                    cov += (A[i].x - ma.x) * (B[i].x - mb.x) + (A[i].y - ma.y) * (B[i].y - mb.y);
                    varb += sqr(B[i].x - mb.x) + sqr(B[i].y - mb.y);
                }
                double zBest = varb > 0 ? -cov / varb : zImg;
                auto rmsAt = [&](double z, double* cy, std::vector<Vec2>* pts) {
                    Vec2 c{};
                    for (size_t i = 0; i < rays.size(); ++i) { c.x += (A[i].x + B[i].x * z) / n; c.y += (A[i].y + B[i].y * z) / n; }
                    double v = 0;
                    for (size_t i = 0; i < rays.size(); ++i) {
                        double dx = A[i].x + B[i].x * z - c.x, dy = A[i].y + B[i].y * z - c.y;
                        v += dx * dx + dy * dy;
                        if (pts) pts->push_back({dx, dy});
                    }
                    if (cy) *cy = c.y;
                    return std::sqrt(v / n);
                };
                s.bestFocusZ = zBest - lw.bi.lastVertexZ;
                s.rmsBest = rmsAt(zBest, nullptr, nullptr);
                s.rmsParaxial = rmsAt(zImg, &s.centroidY, &s.spot);
            }
            rep.spots.push_back(s);
        }

    // Longitudinal spherical aberration on axis at the first wavelength.
    double l0 = lambdas.empty() ? p.wavelength : lambdas.front();
    if (!px.afocal) {
        Paraxial pl = paraxialAnalysis(p, l0, catalogIndex());
        double zPar = lw.bi.lastVertexZ + pl.bfl;
        for (int k = 1; k <= 10; ++k) {
            double h = 0.1 * k * 0.999;
            TracedRay t = traceThrough(lw, launchOrigin(px.entrancePupilZ, 0, h * rEP, {0, 0, 1}, zStart), {0, 0, 1}, l0);
            if (!t.ok || t.d.y == 0) continue;
            double zc = t.p.z - t.p.y * t.d.z / t.d.y;
            rep.lsa.push_back({h, zc - zPar});
        }
    }
    // Chromatic focal shift (best focus on axis).
    {
        auto focusAt = [&](double l) {
            for (auto& s : rep.spots)
                if (s.lambda == l && s.fieldDeg == 0 && s.arrived >= 3) return s.bestFocusZ;
            return std::nan("");
        };
        double fF = focusAt(LambdaF), fC = focusAt(LambdaC);
        rep.chromaticFocalShift = (std::isnan(fF) || std::isnan(fC)) ? std::nan("") : fF - fC;
    }
    // Distortion from chief rays.
    if (!px.afocal)
        for (double f : fields) {
            if (f == 0) continue;
            Vec3 d{0, std::sin(radians(f)), std::cos(radians(f))};
            TracedRay t = traceThrough(lw, launchOrigin(px.entrancePupilZ, 0, 0, d, zStart), d, l0);
            if (!t.ok) continue;
            double yReal = t.p.y + t.d.y / t.d.z * (zImg - t.p.z);
            double yPar = px.efl * std::tan(radians(f));
            rep.distortion.push_back({f, 100.0 * (yReal - yPar) / yPar});
        }
    return rep;
}

std::string LensReport::text() const {
    std::ostringstream os;
    char buf[256];
    const Paraxial& p = paraxial;
    os << "Paraxial (first-order) properties at the design wavelength\n";
    if (p.afocal) {
        std::snprintf(buf, sizeof buf, "  afocal system, angular magnification %.4f\n", p.angularMagnification);
        os << buf;
    } else {
        std::snprintf(buf, sizeof buf, "  EFL %.4f mm   BFL %.4f mm   FFL %.4f mm   f/%.3f\n", p.efl * 1e3, p.bfl * 1e3,
                      p.ffl * 1e3, p.fNumber);
        os << buf;
    }
    std::snprintf(buf, sizeof buf, "  stop: surface %d   entrance pupil z=%.3f mm r=%.3f mm   exit pupil z=%.3f mm r=%.3f mm\n",
                  p.stopIndex + 1, p.entrancePupilZ * 1e3, p.entrancePupilRadius * 1e3, p.exitPupilZ * 1e3,
                  p.exitPupilRadius * 1e3);
    os << buf;
    if (!p.afocal) {
        std::snprintf(buf, sizeof buf, "  image distance from last vertex: %.4f mm\n", p.imageDistance * 1e3);
        os << buf;
    }
    if (!spots.empty() && p.afocal) {
        os << "\nReal-ray emerging beams (non-sequential transport, primary transmission branch)\n";
        os << "   lambda   field    arrived   apparent angle   magnification   RMS angular spread\n";
        for (const auto& s : spots) {
            if (s.arrived < 3) {
                std::snprintf(buf, sizeof buf, "  %6.1fnm %6.2f° %4d/%-4d  (vignetted)\n", s.lambda, s.fieldDeg, s.arrived, s.launched);
            } else {
                double m = s.fieldDeg != 0 ? std::tan(s.apparentAngle) / std::tan(radians(s.fieldDeg)) : std::nan("");
                std::snprintf(buf, sizeof buf, "  %6.1fnm %6.2f° %4d/%-4d %12.3f° %14.3f %15.3f arcmin\n", s.lambda, s.fieldDeg,
                              s.arrived, s.launched, degrees(s.apparentAngle), m, degrees(s.angularRms) * 60);
            }
            os << buf;
        }
    } else if (!spots.empty()) {
        os << "\nReal-ray spots (non-sequential transport, primary transmission branch)\n";
        os << "   lambda   field   arrived   RMS@paraxial   best focus dz   RMS@best     centroid y\n";
        for (const auto& s : spots) {
            std::snprintf(buf, sizeof buf, "  %6.1fnm %6.2f° %4d/%-4d %10.2f µm %13.4f mm %9.2f µm %12.4f mm\n", s.lambda,
                          s.fieldDeg, s.arrived, s.launched, s.rmsParaxial * 1e6,
                          (s.bestFocusZ - (std::isinf(p.imageDistance) ? 0 : p.imageDistance)) * 1e3, s.rmsBest * 1e6,
                          s.centroidY * 1e3);
            os << buf;
        }
    }
    if (!lsa.empty()) {
        os << "\nLongitudinal spherical aberration (axis crossing − paraxial focus)\n";
        for (auto& [h, v] : lsa) {
            std::snprintf(buf, sizeof buf, "  zone %.2f: %+.4f mm\n", h, v * 1e3);
            os << buf;
        }
    }
    if (!std::isnan(chromaticFocalShift)) {
        std::snprintf(buf, sizeof buf, "\nChromatic focal shift z(F) − z(C): %+.4f mm\n", chromaticFocalShift * 1e3);
        os << buf;
    }
    if (!distortion.empty()) {
        os << "\nDistortion (chief ray vs f·tanθ)\n";
        for (auto& [f, d] : distortion) {
            std::snprintf(buf, sizeof buf, "  %6.2f°: %+.4f %%\n", f, d);
            os << buf;
        }
    }
    return os.str();
}

std::vector<PathRecord> lensFan(const Prescription& p, double fieldDeg, double lambda, int rays, Scene& sceneOut) {
    BuiltInstrument bi = buildPrescription(sceneOut.world, p, "lens", -1, Transform{}, 0.0, catalogIndex(), LensSpec::Rim::Black);
    sceneOut.world.build();
    const Paraxial& px = bi.paraxial;
    double zStart = -0.02 - 0.1 * px.length;
    Vec3 d{0, std::sin(radians(fieldDeg)), std::cos(radians(fieldDeg))};
    Tracer tr(sceneOut);
    std::vector<PathRecord> out;
    for (int i = 0; i < rays; ++i) {
        double y = px.entrancePupilRadius * (rays == 1 ? 0.0 : (2.0 * i / (rays - 1) - 1)) * 0.999;
        Rng rng(7, uint64_t(i));
        Vec3 P{0, y, px.entrancePupilZ};
        Vec3 o = P + d * ((zStart - px.entrancePupilZ) / d.z);
        PathRecord rec = tr.walk(Ray{o, d}, sceneOut.world.ambientRegion(), lambda, WalkMode::PrimaryTransmission, rng, 64);
        // Clip escaping segments at a sensible distance behind the lens.
        if (rec.v.size() >= 2 && rec.v.back().event == EventKind::Escape) {
            const PathVertex& last = rec.v[rec.v.size() - 2];
            double zEnd = bi.lastVertexZ + (std::isinf(px.imageDistance) ? 0.3 * px.length + 0.02 : px.imageDistance * 1.15);
            if (last.dOut.z > 0) rec.v.back().p = last.p + last.dOut * ((zEnd - last.p.z) / last.dOut.z);
        }
        out.push_back(rec);
    }
    return out;
}

namespace {

bool specularEvent(EventKind k) {
    return k == EventKind::Refract || k == EventKind::Reflect || k == EventKind::TIR || k == EventKind::Pass;
}

// A line of sight up to the object: the events (boundary, kind) crossed, and either the object
// point or, for light from beyond the world, the final direction.
struct Sight {
    std::vector<std::pair<uint32_t, EventKind>> events;
    Vec3 lastPoint, lastDir;  // after the final specular event (the eye if there is none)
    Vec3 object;              // the object point (when !atInfinity)
    bool atInfinity = false, ok = false;
    bool dark = false;        // ends on matter that neither scatters nor emits (black paint)
    double length = 0;
    int specular = 0;
};

Sight followSight(const Scene& scene, const Tracer& tr, const Vec3& eye, const Vec3& dir, uint32_t region, double lambda) {
    Rng rng(1, 1);  // PrimaryTransmission walks are deterministic through smooth surfaces
    PathRecord rec = tr.walk(Ray{eye, dir}, region, lambda, WalkMode::PrimaryTransmission, rng, 64);
    Sight s;
    s.lastPoint = eye;
    s.lastDir = dir;
    for (size_t i = 1; i < rec.v.size(); ++i) {
        const PathVertex& v = rec.v[i];
        if (v.event == EventKind::Escape) {
            s.atInfinity = s.ok = true;
            return s;
        }
        s.length += length(v.p - rec.v[i - 1].p);
        if (specularEvent(v.event)) {
            s.events.push_back({v.boundary, v.event});
            if (v.event != EventKind::Pass) s.specular++;
            s.lastPoint = v.p;
            s.lastDir = v.dOut;
            continue;
        }
        // The first surface that scatters, emits or measures: the object.
        s.events.push_back({v.boundary, v.event});
        s.object = v.p;
        s.ok = v.event != EventKind::Terminate && v.event != EventKind::None;
        s.dark = v.event == EventKind::Absorb && v.boundary != kNone &&
                 scene.world.boundaries()[v.boundary].emission < 0;
        return s;
    }
    return s;
}

// The focus for the line of sight from `eye` along `dir` (see accommodation()).
Accommodation focusAlong(const Scene& scene, const Tracer& tr, const Sight& chief, const Vec3& eye, const Vec3& dir,
                         uint32_t region, double lambda) {
    Accommodation a;
    a.found = true;
    a.pupilPoint = eye;
    a.specular = chief.specular;
    a.objectPath = chief.atInfinity ? Inf : chief.length;
    if (chief.specular == 0) {
        a.distance = chief.atInfinity ? Inf : chief.length;
        return a;
    }
    // Paraxial rays from the eye at height h along a, aimed to cross the line of sight at distance
    // 1/v (vergence v; 0: parallel). Their miss at the object (a transverse distance, or for an
    // object at infinity an angle) is affine in v: solve it for zero along two directions and
    // take the mean vergence, the circle of least confusion of an astigmatic image.
    Vec3 a1, a2;
    orthonormalBasis(dir, a1, a2);
    auto miss = [&](const Vec3& axis, double h, double v, Vec3& out) {
        Sight m = followSight(scene, tr, eye + axis * h, normalize(dir - axis * (v * h)), region, lambda);
        if (!m.ok || m.atInfinity != chief.atInfinity || m.events.size() != chief.events.size()) return false;
        for (size_t k = 0; k < m.events.size(); ++k)
            if (m.events[k] != chief.events[k]) return false;
        Vec3 e;
        if (chief.atInfinity) {
            e = m.lastDir - chief.lastDir;
        } else {
            // Where the ray crosses the plane through the object normal to the line of sight.
            double c = dot(m.lastDir, chief.lastDir);
            if (!(c > 0)) return false;
            e = m.lastPoint + m.lastDir * (dot(chief.object - m.lastPoint, chief.lastDir) / c) - chief.object;
        }
        out = e - chief.lastDir * dot(e, chief.lastDir);
        return true;
    };
    double sum = 0;
    int solved = 0;
    for (const Vec3& axis : {a1, a2}) {
        for (double h : {1e-4, 1e-5, 1e-6}) {
            // Two vergences bracket the answer's scale: 0 and that of the nearest optics.
            double v0 = 0, v1 = 1 / std::max(1e-3, length(chief.lastPoint - eye));
            Vec3 e0, e1;
            if (!miss(axis, h, v0, e0) || !miss(axis, h, v1, e1)) continue;
            bool ok = true;
            for (int it = 0; it < 3 && ok; ++it) {
                Vec3 de = e1 - e0;
                double dd = dot(de, de);
                if (!(dd > 0)) { ok = false; break; }
                double v = v1 - dot(e1, de) / dd * (v1 - v0);  // least-squares secant step
                Vec3 e;
                if (!miss(axis, h, v, e)) { ok = false; break; }
                v0 = v1; e0 = e1;
                v1 = v; e1 = e;
            }
            if (!ok || !std::isfinite(v1)) continue;
            sum += v1;
            solved++;
            break;
        }
    }
    if (solved == 0) {
        a.found = false;
        return a;
    }
    double v = sum / solved;
    // Collimated light (v = 0) or converging light (v < 0): the relaxed eye, focused at infinity.
    a.distance = v > 1e-9 ? 1 / v : Inf;
    return a;
}

}  // namespace

Accommodation accommodation(const Scene& scene, const Vec3& eye, const Vec3& direction, uint32_t region,
                            double pupilRadius, double lambda) {
    const Tracer tr(scene);
    const Vec3 dir = normalize(direction);
    // Pupil points: the centre, then rings an eighth of the radius apart (fine enough to find a
    // 1 mm exit pupil in a 10 mm eye pupil).
    std::vector<Vec3> points{eye};
    if (pupilRadius > 0) {
        Vec3 b1, b2;
        orthonormalBasis(dir, b1, b2);
        constexpr int kRings = 8;
        for (int ring = 1; ring <= kRings; ++ring) {
            int n = 6 * ring;
            double r = pupilRadius * ring / kRings;
            for (int k = 0; k < n; ++k) {
                double phi = 2 * Pi * (k + 0.5 * (ring % 2)) / n;
                points.push_back(eye + (b1 * std::cos(phi) + b2 * std::sin(phi)) * r);
            }
        }
    }
    // Lines of sight grouped by what they end on. A group's representative is its line through the
    // middle of the bundle it forms on the pupil (the member nearest the members' centroid): at a
    // telescope, the centre of the exit pupil's footprint, not its aberrated rim.
    struct Group {
        uint32_t boundary;
        bool atInfinity;
        int specular;
        std::vector<size_t> members;
        std::vector<Sight> sights;
        std::vector<uint32_t> starts;
        size_t point = 0;  // representative: index into members
        int count() const { return int(members.size()); }
    };
    std::vector<Group> groups;
    for (size_t i = 0; i < points.size(); ++i) {
        const uint32_t start = i == 0 ? region : scene.world.locate(points[i]);
        Sight s = followSight(scene, tr, points[i], dir, start, lambda);
        if (!s.ok) continue;
        uint32_t boundary = s.atInfinity || s.events.empty() ? kNone : s.events.back().first;
        auto g = std::find_if(groups.begin(), groups.end(), [&](const Group& x) {
            return x.boundary == boundary && x.atInfinity == s.atInfinity && x.specular == s.specular;
        });
        if (g == groups.end()) {
            groups.push_back({boundary, s.atInfinity, s.specular, {}, {}, {}});
            g = groups.end() - 1;
        }
        g->members.push_back(i);
        g->sights.push_back(s);
        g->starts.push_back(start);
    }
    for (Group& g : groups) {
        Vec3 centroid;
        for (size_t m : g.members) centroid += points[m] / double(g.members.size());
        double nearest = Inf;
        for (size_t k = 0; k < g.members.size(); ++k) {
            double d = lengthSq(points[g.members[k]] - centroid);
            if (d < nearest) { nearest = d; g.point = k; }
        }
    }
    // The eye focuses on what sends it light: the object seen through the largest share of the
    // pupil, weighted by its luminance (a few deterministic radiance samples; black matter and
    // unlit surfaces bring none). Without light anywhere, the centre's line of sight.
    const Group* best = nullptr;
    double bestScore = 0;
    for (const Group& g : groups) {
        if (g.sights[g.point].dark) continue;
        const size_t q = g.members[g.point];
        Rng rng(0x5eed1e55ULL, q);
        TransportStats st;
        double y = 0;
        constexpr int kSamples = 24;
        for (int k = 0; k < kSamples; ++k) {
            Wavelengths wl = Wavelengths::sample((k + 0.5) / kSamples);
            Spec4 L = tr.radiance(Ray{points[q], dir}, g.starts[g.point], wl, rng, st);
            double v = wl.toXYZ(L).y;
            if (std::isfinite(v)) y += v / kSamples;
        }
        double score = y * g.count();
        if (score > bestScore) {
            bestScore = score;
            best = &g;
        }
    }
    if (!best) {  // nothing sends light: the line of sight nearest the centre
        for (const Group& g : groups)
            if (!best || g.members.front() < best->members.front()) best = &g;
        if (!best) return Accommodation{};
        const size_t q = best->members.front();
        return focusAlong(scene, tr, best->sights.front(), points[q], dir, best->starts.front(), lambda);
    }
    const size_t q = best->members[best->point];
    return focusAlong(scene, tr, best->sights[best->point], points[q], dir, best->starts[best->point], lambda);
}

}  // namespace owe
