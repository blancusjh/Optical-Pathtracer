#include "owe/analysis/analysis.hpp"

#include <cstdio>
#include <sstream>

namespace owe {

namespace {

struct LensWorld {
    Scene scene;
    BuiltInstrument bi;
    size_t surfaces = 0;
};

void makeLensWorld(const Prescription& p, LensWorld& lw) {
    lw.bi = buildPrescription(lw.scene.world, p, "lens", -1, Transform{}, 0.0, catalogIndex(), LensSpec::Rim::Black);
    lw.scene.world.build();
    lw.surfaces = 0;
    for (size_t i = 0; i < p.surfaces.size(); ++i) {
        std::string before = i == 0 ? p.ambient : p.surfaces[i - 1].medium;
        if (!(before == p.ambient && p.surfaces[i].medium == p.ambient)) lw.surfaces++;
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
    size_t refr = 0;
    for (auto& v : rec.v) {
        if (v.event == EventKind::Refract) refr++;
        else if (v.event != EventKind::Emit && v.event != EventKind::Escape) return r;
    }
    if (refr != lw.surfaces) return r;
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

}  // namespace owe
