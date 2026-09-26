#include "inspect.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "render.hpp"

namespace owe {

void wavelengthToRGB(double l, double rgb[3]) {
    XYZ c{cieX(l), cieY(l), cieZ(l)};
    xyzToLinearSRGB(c, rgb);
    double m = 0;
    for (int i = 0; i < 3; ++i) { rgb[i] = std::max(0.0, rgb[i]); m = std::max(m, rgb[i]); }
    // Fade toward the ends of the visible range.
    double fade = clampd(std::min((l - 380) / 40.0, (780 - l) / 60.0), 0.25, 1.0);
    for (int i = 0; i < 3; ++i) {
        double v = m > 0 ? rgb[i] / m : 0;
        rgb[i] = std::pow(v, 1 / 2.2) * fade;
    }
}

namespace {

const char* opticsColor(SurfaceType t) {
    switch (t) {
        case SurfaceType::Dielectric: return "#2f6fb5";
        case SurfaceType::Absorber: return "#202020";
        case SurfaceType::Diffuse: return "#9a7b4f";
        case SurfaceType::Mirror:
        case SurfaceType::Conductor: return "#7d7d7d";
        case SurfaceType::Detector: return "#1f8a4c";
        case SurfaceType::Null: return "#c8c8c8";
    }
    return "#000";
}

std::string hexColor(const double rgb[3]) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "#%02x%02x%02x", unsigned(255 * clampd(rgb[0], 0, 1)) & 0xFFu,
                  unsigned(255 * clampd(rgb[1], 0, 1)) & 0xFFu, unsigned(255 * clampd(rgb[2], 0, 1)) & 0xFFu);
    return buf;
}

}  // namespace

void writePathsSVG(const std::string& path, const World& world, const std::vector<PathRecord>& paths,
                   const ViewPlane& plane, const SvgOptions& opt) {
    Vec3 u = normalize(plane.u), v = normalize(plane.v);
    auto proj = [&](const Vec3& p) { return Vec2{dot(p - plane.origin, u), dot(p - plane.origin, v)}; };
    double a0 = Inf, a1 = -Inf, b0 = Inf, b1 = -Inf;
    auto grow = [&](const Vec2& q) {
        a0 = std::min(a0, q.x); a1 = std::max(a1, q.x);
        b0 = std::min(b0, q.y); b1 = std::max(b1, q.y);
    };
    if (opt.bounds.size() == 4) {
        a0 = opt.bounds[0]; b0 = opt.bounds[1]; a1 = opt.bounds[2]; b1 = opt.bounds[3];
    } else {
        for (const auto& r : paths)
            for (const auto& vx : r.v) grow(proj(vx.p));
        if (!(a1 > a0) || !(b1 > b0) || paths.empty()) {
            AABB wb = world.bounds();
            for (int i = 0; i < 8; ++i)
                grow(proj({(i & 1) ? wb.hi.x : wb.lo.x, (i & 2) ? wb.hi.y : wb.lo.y, (i & 4) ? wb.hi.z : wb.lo.z}));
        }
        double ea = (a1 - a0) * opt.margin + 1e-9, eb = (b1 - b0) * opt.margin + 1e-9;
        double ext = std::max(a1 - a0, b1 - b0);
        if (b1 - b0 < 0.05 * ext) { double c = 0.5 * (b0 + b1); b0 = c - 0.025 * ext; b1 = c + 0.025 * ext; }
        a0 -= ea; a1 += ea; b0 -= eb; b1 += eb;
    }
    double W = opt.width;
    double H = std::max(80.0, W * (b1 - b0) / (a1 - a0));
    auto sx = [&](double a) { return (a - a0) / (a1 - a0) * W; };
    auto sy = [&](double b) { return H - (b - b0) / (b1 - b0) * H; };

    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H << "\" viewBox=\"0 0 " << W
        << " " << H << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"#fbfaf7\"/>\n";

    if (opt.drawSection) {
        // Slice the world with the view plane by casting dense lines within it.
        out << "<g opacity=\"0.9\">\n";
        std::map<std::string, std::vector<Vec2>> pts;
        int L = opt.sectionLines;
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < L; ++i) {
                double f = (i + 0.5) / L;
                Vec3 o, d;
                double len;
                if (pass == 0) {
                    o = plane.origin + u * a0 + v * (b0 + f * (b1 - b0));
                    d = u;
                    len = a1 - a0;
                } else {
                    o = plane.origin + v * b0 + u * (a0 + f * (a1 - a0));
                    d = v;
                    len = b1 - b0;
                }
                double travelled = 0;
                for (int k = 0; k < 64; ++k) {
                    SurfaceHit h;
                    if (!world.intersect(Ray{o, d}, len - travelled, h)) break;
                    travelled += h.t;
                    const Boundary& bd = world.boundaries()[h.boundary];
                    pts[opticsColor(world.optics()[bd.optics].type)].push_back(proj(h.p));
                    o = h.p + d * surfaceEpsilon(h.p) * 4;
                }
            }
        double r = std::max(0.6, W / 1400.0);
        for (auto& [color, list] : pts) {
            out << "<g fill=\"" << color << "\">";
            for (auto& q : list) out << "<circle cx=\"" << sx(q.x) << "\" cy=\"" << sy(q.y) << "\" r=\"" << r << "\"/>";
            out << "</g>\n";
        }
        out << "</g>\n";
    }

    out << "<g fill=\"none\" stroke-linejoin=\"round\" stroke-width=\"" << opt.strokeWidth << "\" stroke-opacity=\"0.75\">\n";
    for (const auto& r : paths) {
        if (r.v.size() < 2) continue;
        double rgb[3];
        wavelengthToRGB(r.lambda, rgb);
        out << "<polyline stroke=\"" << hexColor(rgb) << "\" points=\"";
        for (const auto& vx : r.v) {
            Vec2 q = proj(vx.p);
            out << sx(q.x) << "," << sy(q.y) << " ";
        }
        out << "\"/>\n";
    }
    out << "</g>\n";
    // Scale bar.
    double span = a1 - a0;
    double bar = std::pow(10.0, std::floor(std::log10(span * 0.25)));
    out << "<g font-family=\"sans-serif\" font-size=\"13\" fill=\"#333\">";
    out << "<line x1=\"20\" y1=\"" << H - 18 << "\" x2=\"" << 20 + bar / span * W << "\" y2=\"" << H - 18
        << "\" stroke=\"#333\" stroke-width=\"2\"/>";
    char lab[64];
    if (bar >= 1) std::snprintf(lab, sizeof lab, "%g m", bar);
    else if (bar >= 1e-3) std::snprintf(lab, sizeof lab, "%g mm", bar * 1e3);
    else std::snprintf(lab, sizeof lab, "%g µm", bar * 1e6);
    out << "<text x=\"20\" y=\"" << H - 24 << "\">" << lab << "</text>";
    if (!opt.title.empty()) out << "<text x=\"20\" y=\"22\" font-size=\"16\">" << opt.title << "</text>";
    out << "</g>\n</svg>\n";
}

std::string pathSignature(const World& w, const PathRecord& r, size_t upTo) {
    std::string s;
    for (size_t i = 1; i <= upTo && i < r.v.size(); ++i) {
        const PathVertex& v = r.v[i];
        if (v.event == EventKind::Pass) continue;
        if (!s.empty()) s += " → ";
        s += eventName(v.event);
        if (v.boundary != kNone) s += "@" + w.boundaryLabel(v.boundary);
    }
    return s.empty() ? "(direct)" : s;
}

PixelProbe probePixel(const Scene& scene, int di, int px, int py, int samples, uint64_t seed, int maxDepth) {
    PixelProbe pr;
    pr.px = px;
    pr.py = py;
    pr.samples = samples;
    const Detector& det = *scene.detectors.at(size_t(di));
    Tracer tr(scene, maxDepth);
    TransportStats st;
    std::map<std::string, PixelProbe::Group> groups;
    Rng rng(hashCombine(seed, uint64_t(py) * det.width + px), 0xabcdefULL);
    for (int s = 0; s < samples; ++s) {
        double lambda = sampleVisibleWavelength(rng.uniform());
        double pdf = visibleWavelengthPdf(lambda);
        Ray ray;
        double weight;
        PathRecord rec;
        if (!det.generate(px + rng.uniform(), py + rng.uniform(), rng, ray, weight) || weight <= 0) continue;
        tr.radiance(ray, det.region, lambda, rng, st, &rec);
        double w3[3];
        spectralWeights(lambda, pdf, w3);
        size_t index = pr.records.size();
        for (const auto& c : rec.c) {
            double val = c.value * weight;
            pr.estimate.x += val * w3[0] / samples;
            pr.estimate.y += val * w3[1] / samples;
            pr.estimate.z += val * w3[2] / samples;
            std::string sig = pathSignature(scene.world, rec, c.vertex) + (c.nee ? " ⇢ " : " ← ") + c.source;
            auto& g = groups[sig];
            if (g.paths == 0) { g.signature = sig; g.example = index; }
            g.paths++;
            g.valueY += val * w3[1] / samples;
        }
        pr.records.push_back(std::move(rec));
    }
    for (auto& [k, g] : groups) pr.groups.push_back(g);
    std::sort(pr.groups.begin(), pr.groups.end(), [](const auto& a, const auto& b) { return a.valueY > b.valueY; });
    return pr;
}

std::string PixelProbe::text(const World& w) const {
    std::ostringstream os;
    char buf[512];
    std::snprintf(buf, sizeof buf, "Pixel (%d, %d), %d samples: XYZ = (%.5g, %.5g, %.5g)\n", px, py, samples, estimate.x,
                  estimate.y, estimate.z);
    os << buf;
    os << "Why is this pixel this colour? Transport classes ranked by contribution to Y\n"
       << "(→ separates events along the camera path; ⇢ light reached by next-event estimation, ← emitter hit)\n\n";
    double total = std::max(1e-300, estimate.y);
    int shown = 0;
    for (const auto& g : groups) {
        if (shown++ >= 12) break;
        std::snprintf(buf, sizeof buf, "%6.2f%%  (%4d paths)  ", 100 * g.valueY / total, g.paths);
        os << buf << g.signature << "\n";
    }
    if (!groups.empty()) {
        os << "\nRepresentative path of the dominant class:\n" << describePath(w, records[groups[0].example]);
    }
    return os.str();
}

EmissionProbe probeEmission(const Scene& scene, const Vec3& origin, const Vec3& direction, double cone, int rays,
                            const std::vector<double>& lambdas, WalkMode mode, uint64_t seed) {
    EmissionProbe ep;
    ep.rays = rays;
    Tracer tr(scene);
    uint32_t region = scene.world.locate(origin);
    Frame f(normalize(direction));
    std::map<std::string, EmissionProbe::Fate> fates;
    for (int i = 0; i < rays; ++i) {
        Rng rng(hashCombine(seed, uint64_t(i)), 3);
        Vec3 d = f.toWorld(sampleUniformCone(rng.uniform(), rng.uniform(), std::cos(cone)));
        double lambda = lambdas.empty() ? sampleVisibleWavelength(rng.uniform()) : lambdas[size_t(i) % lambdas.size()];
        PathRecord rec = tr.walk(Ray{origin, d}, region, lambda, mode, rng, 128);
        const PathVertex& last = rec.v.back();
        std::string label = eventName(last.event);
        if (last.boundary != kNone) label += " at " + scene.world.boundaryLabel(last.boundary);
        auto& fate = fates[label];
        fate.label = label;
        fate.rays++;
        fate.energy += last.beta;
        ep.records.push_back(std::move(rec));
    }
    for (auto& [k, v] : fates) ep.fates.push_back(v);
    std::sort(ep.fates.begin(), ep.fates.end(), [](const auto& a, const auto& b) { return a.rays > b.rays; });
    return ep;
}

std::string EmissionProbe::text() const {
    std::ostringstream os;
    char buf[256];
    os << "Where does its light go? " << rays << " rays\n";
    for (const auto& f : fates) {
        std::snprintf(buf, sizeof buf, "  %6.2f%% of rays, %6.2f%% of energy : ", 100.0 * f.rays / rays, 100.0 * f.energy / rays);
        os << buf << f.label << "\n";
    }
    return os.str();
}

std::string describePath(const World& w, const PathRecord& r) {
    std::ostringstream os;
    char buf[512];
    std::snprintf(buf, sizeof buf, "  λ = %.2f nm\n", r.lambda);
    os << buf;
    os << "   #  event            boundary                        regions                               n_i      n_t    θi°    θt°      R        T     OPL[mm]    β\n";
    for (size_t i = 0; i < r.v.size(); ++i) {
        const PathVertex& v = r.v[i];
        std::string regions = (v.regionFrom == kNone ? std::string("-") : w.regionLabel(v.regionFrom));
        if (v.regionTo != v.regionFrom && v.regionTo != kNone) regions += " → " + w.regionLabel(v.regionTo);
        std::snprintf(buf, sizeof buf, "  %2zu  %-16s %-31s %-36s %7.5f %7.5f %6.2f %6.2f %8.5f %8.5f %9.3f %8.4g\n", i,
                      eventName(v.event), v.boundary == kNone ? "-" : w.boundaryLabel(v.boundary).c_str(), regions.c_str(),
                      v.nI, v.nT, degrees(v.thetaI), degrees(v.thetaT), v.R, v.T, v.opl * 1e3, v.beta);
        os << buf;
    }
    return os.str();
}

std::string pathsJSON(const World& w, const std::vector<PathRecord>& paths) {
    std::ostringstream os;
    os.precision(10);
    os << "[\n";
    for (size_t k = 0; k < paths.size(); ++k) {
        const PathRecord& r = paths[k];
        os << "  {\"lambda_nm\": " << r.lambda << ", \"vertices\": [";
        for (size_t i = 0; i < r.v.size(); ++i) {
            const PathVertex& v = r.v[i];
            os << (i ? ", " : "") << "{\"p\": [" << v.p.x << ", " << v.p.y << ", " << v.p.z << "], \"event\": \""
               << eventName(v.event) << "\", \"boundary\": \"" << (v.boundary == kNone ? "" : w.boundaryLabel(v.boundary))
               << "\", \"n_i\": " << v.nI << ", \"n_t\": " << v.nT << ", \"theta_i_deg\": " << degrees(v.thetaI)
               << ", \"theta_t_deg\": " << degrees(v.thetaT) << ", \"R\": " << v.R << ", \"T\": " << v.T
               << ", \"optical_path_m\": " << v.opl << ", \"beta\": " << v.beta << "}";
        }
        os << "]}" << (k + 1 < paths.size() ? "," : "") << "\n";
    }
    os << "]\n";
    return os.str();
}

}  // namespace owe
