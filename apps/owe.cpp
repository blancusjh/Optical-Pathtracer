// owe — command-line front end of the Optical World Engine (CPU reference backend).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "owe/analysis.hpp"
#include "owe/inspect.hpp"
#include "owe/render.hpp"
#include "owe/scene_loader.hpp"
#include "owe/version.hpp"

using namespace owe;

namespace {

struct Args {
    std::vector<std::string> positional;
    std::map<std::string, std::vector<std::string>> opts;
    bool has(const std::string& k) const { return opts.count(k) > 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = opts.find(k);
        return it == opts.end() || it->second.empty() ? def : it->second[0];
    }
    const std::vector<std::string>& all(const std::string& k) const {
        static const std::vector<std::string> empty;
        auto it = opts.find(k);
        return it == opts.end() ? empty : it->second;
    }
};

// Options that take N values.
const std::map<std::string, int> kArity = {
    {"--detector", 1}, {"--spp", 1},     {"--passes", 1},   {"--seed", 1},     {"--integrator", 1}, {"--threads", 1},
    {"--max-depth", 1}, {"--out", 1},    {"--exposure", 1}, {"--pixel", 2},    {"--samples", 1},    {"--svg", 1},
    {"--json", 1},     {"--from", 1},    {"--dir", 1},      {"--toward", 1},   {"--cone", 1},       {"--rays", 1},
    {"--lambda", 1},   {"--plane", 1},   {"--origin", 1},   {"--fields", 1},   {"--resolution", 1}, {"--field", 1},
    {"--bounds", 1},   {"--title", 1},      {"--fresnel-floor", 1}};

Args parse(int argc, char** argv, int start) {
    Args a;
    for (int i = start; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) {
            auto it = kArity.find(s);
            int n = it == kArity.end() ? 0 : it->second;
            auto& v = a.opts[s];
            for (int k = 0; k < n; ++k) {
                if (i + 1 >= argc) throw std::runtime_error("option " + s + " needs a value");
                v.push_back(argv[++i]);
            }
        } else {
            a.positional.push_back(s);
        }
    }
    return a;
}

std::vector<double> numbers(const std::string& s) {
    std::vector<double> r;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) r.push_back(std::stod(item));
    return r;
}

Vec3 vec(const std::string& s) {
    auto v = numbers(s);
    if (v.size() != 3) throw std::runtime_error("expected x,y,z but got '" + s + "'");
    return {v[0], v[1], v[2]};
}

ViewPlane plane(const Args& a, const ViewPlane& def) {
    ViewPlane p = def;
    if (a.has("--plane")) {
        std::string s = a.get("--plane");
        auto axis = [&](char c) -> Vec3 {
            switch (c) {
                case 'x': return {1, 0, 0};
                case 'y': return {0, 1, 0};
                case 'z': return {0, 0, 1};
            }
            throw std::runtime_error("--plane expects two axis letters, e.g. zy");
        };
        if (s.size() != 2) throw std::runtime_error("--plane expects two axis letters, e.g. zy");
        p.u = axis(s[0]);
        p.v = axis(s[1]);
    }
    if (a.has("--origin")) p.origin = vec(a.get("--origin"));
    return p;
}

int detectorIndex(const Scene& s, const Args& a) {
    std::string name = a.get("--detector", s.render.detector);
    if (name.empty()) return 0;
    int i = s.findDetector(name);
    if (i < 0) throw std::runtime_error("no detector named '" + name + "'");
    return i;
}

void usage() {
    std::puts(
        "owe " OWE_VERSION " — Optical World Engine (CPU reference tracer)\n"
        "\n"
        "  owe render <scene.owe> [--detector NAME] [--spp N] [--passes K] [--integrator path|light]\n"
        "                         [--seed S] [--threads T] [--max-depth D] [--resolution WxH]\n"
        "                         [--out PREFIX] [--exposure EV] [--no-auto-exposure] [--fresnel-floor P]\n"
        "      Progressive spectral render. Writes PREFIX.png (display), PREFIX.pfm (raw linear) and\n"
        "      PREFIX.json (reproducibility record) after every pass.\n"
        "\n"
        "  owe probe <scene.owe> --pixel X Y [--detector NAME] [--samples N] [--svg F] [--json F] [--plane uv]\n"
        "      Why is this pixel this colour? Ranks transport classes and shows representative paths.\n"
        "\n"
        "  owe emit <scene.owe> --from x,y,z (--dir x,y,z | --toward x,y,z) [--cone DEG] [--rays N]\n"
        "                       [--lambda nm[,nm...]] [--primary] [--svg F] [--json F] [--plane uv] [--origin x,y,z]\n"
        "      Where does its light go? Traces an emission ensemble and reports every fate.\n"
        "\n"
        "  owe lens <file.lens> [--fields 0,5,10] [--afocal] [--svg F] [--field DEG] [--rays N]\n"
        "      Paraxial and real-ray analysis of a lens prescription built as physical bodies.\n"
        "\n"
        "  owe glass [NAME]    Refractive index model, n at spectral lines, Abbe number.\n"
        "  owe info <scene.owe>  The world's ontology: media, regions, boundaries, bodies, detectors.\n");
}

int cmdRender(const Args& a) {
    if (a.positional.empty()) throw std::runtime_error("render: missing scene file");
    Scene scene = loadScene(a.positional[0]);
    RenderSettings rs = scene.render;
    if (a.has("--spp")) rs.spp = std::stoi(a.get("--spp"));
    if (a.has("--seed")) rs.seed = std::stoull(a.get("--seed"));
    if (a.has("--integrator")) rs.integrator = a.get("--integrator");
    if (a.has("--threads")) rs.threads = std::stoi(a.get("--threads"));
    if (a.has("--max-depth")) rs.maxDepth = std::stoi(a.get("--max-depth"));
    if (a.has("--exposure")) rs.exposure = std::stod(a.get("--exposure"));
    if (a.has("--fresnel-floor")) rs.fresnelFloor = std::stod(a.get("--fresnel-floor"));
    if (a.has("--no-auto-exposure")) rs.autoExposure = false;
    int di = detectorIndex(scene, a);
    if (a.has("--resolution")) {
        std::string r = a.get("--resolution");
        auto x = r.find('x');
        if (x == std::string::npos) throw std::runtime_error("--resolution expects WxH");
        scene.detectors[di]->width = std::stoi(r.substr(0, x));
        scene.detectors[di]->height = std::stoi(r.substr(x + 1));
        scene.build();
    }
    int passes = std::max(1, std::stoi(a.get("--passes", "1")));
    std::string out = a.get("--out", "render");
    ProgressiveRenderer R(scene, di, rs);
    std::fprintf(stderr, "%s\n", scene.detectors[di]->describe().c_str());
    int done = 0;
    for (int p = 0; p < passes; ++p) {
        int spp = rs.spp * (p + 1) / passes - done;
        done += spp;
        if (spp <= 0) continue;
        R.runPass(spp);
        Image img = R.resolve();
        writePNG(out + ".png", img, rs.exposure, rs.autoExposure);
        writePFM(out + ".pfm", img);
        std::ofstream(out + ".json") << renderMetadataJSON(scene, R, rs);
        std::fprintf(stderr, "pass %d/%d: %lld spp, %.2f s, mean Y %.5g\n", p + 1, passes, R.samplesPerPixel(),
                     R.seconds(), img.meanY());
    }
    const TransportStats& st = R.stats();
    std::fprintf(stderr, "paths %llu, segments %llu, region inconsistencies %llu, leaks %llu\n",
                 (unsigned long long)st.paths, (unsigned long long)st.segments, (unsigned long long)st.inconsistencies,
                 (unsigned long long)st.leaks);
    if (st.inconsistencies) {
        const auto& f = st.first;
        std::fprintf(stderr,
                     "first inconsistency: ray in %s, last at %s, met %s (whose side there is %s) at (%.9g, %.9g, %.9g)\n",
                     scene.world.regionLabel(f.rayRegion).c_str(),
                     f.previous == kNone ? "its origin" : scene.world.boundaryLabel(f.previous).c_str(),
                     scene.world.boundaryLabel(f.boundary).c_str(), scene.world.regionLabel(f.boundaryRegion).c_str(), f.p.x,
                     f.p.y, f.p.z);
    }
    std::printf("%s.png %s.pfm %s.json\n", out.c_str(), out.c_str(), out.c_str());
    return 0;
}

int cmdProbe(const Args& a) {
    if (a.positional.empty() || a.all("--pixel").size() != 2) throw std::runtime_error("probe: need scene and --pixel X Y");
    Scene scene = loadScene(a.positional[0]);
    int di = detectorIndex(scene, a);
    int px = std::stoi(a.all("--pixel")[0]), py = std::stoi(a.all("--pixel")[1]);
    PixelProbe pr = probePixel(scene, di, px, py, std::stoi(a.get("--samples", "2000")), scene.render.seed);
    std::cout << pr.text(scene.world);
    if (a.has("--svg")) {
        std::vector<PathRecord> shown;
        for (auto& r : pr.records)
            if (!r.c.empty() && shown.size() < 400) shown.push_back(r);
        ViewPlane def;
        const Detector& d = *scene.detectors[di];
        if (auto* o = dynamic_cast<const IdealObserver*>(&d)) { def.origin = o->position; def.u = o->forward(); def.v = o->upVec(); }
        SvgOptions so;
        so.title = "paths contributing to pixel (" + std::to_string(px) + ", " + std::to_string(py) + ")";
        writePathsSVG(a.get("--svg"), scene.world, shown, plane(a, def), so);
        std::cout << "wrote " << a.get("--svg") << "\n";
    }
    if (a.has("--json")) std::ofstream(a.get("--json")) << pathsJSON(scene.world, pr.records);
    return 0;
}

int cmdEmit(const Args& a) {
    if (a.positional.empty() || !a.has("--from")) throw std::runtime_error("emit: need scene and --from");
    Scene scene = loadScene(a.positional[0]);
    Vec3 from = vec(a.get("--from"));
    Vec3 dir;
    if (a.has("--dir")) dir = vec(a.get("--dir"));
    else if (a.has("--toward")) dir = vec(a.get("--toward")) - from;
    else throw std::runtime_error("emit: need --dir or --toward");
    std::vector<double> lambdas;
    if (a.has("--lambda")) lambdas = numbers(a.get("--lambda"));
    EmissionProbe ep = probeEmission(scene, from, dir, radians(std::stod(a.get("--cone", "1"))),
                                     std::stoi(a.get("--rays", "500")), lambdas,
                                     a.has("--primary") ? WalkMode::PrimaryTransmission : WalkMode::Stochastic,
                                     scene.render.seed);
    std::cout << ep.text();
    if (!ep.records.empty()) std::cout << "\nFirst ray in detail:\n" << describePath(scene.world, ep.records[0]);
    if (a.has("--svg")) {
        ViewPlane def;
        def.origin = from;
        Vec3 d = normalize(dir), b1, b2;
        orthonormalBasis(d, b1, b2);
        def.u = d;
        def.v = std::abs(dot(scene.world.env.up, d)) < 0.9 ? normalize(scene.world.env.up - d * dot(scene.world.env.up, d)) : b1;
        SvgOptions so;
        so.title = a.get("--title", "emission ensemble");
        if (a.has("--bounds")) so.bounds = numbers(a.get("--bounds"));
        writePathsSVG(a.get("--svg"), scene.world, ep.records, plane(a, def), so);
        std::cout << "wrote " << a.get("--svg") << "\n";
    }
    if (a.has("--json")) std::ofstream(a.get("--json")) << pathsJSON(scene.world, ep.records);
    return 0;
}

int cmdLens(const Args& a) {
    if (a.positional.empty()) throw std::runtime_error("lens: missing prescription file");
    Prescription p = loadPrescription(a.positional[0]);
    if (a.has("--afocal")) {
        double g = solveAfocal(p, p.wavelength, catalogIndex());
        std::printf("afocal separation solved: %.4f mm\n", g * 1e3);
    }
    std::cout << formatPrescription(p) << "\n";
    std::vector<double> fields = a.has("--fields") ? numbers(a.get("--fields")) : std::vector<double>{0, 5, 10};
    LensReport rep = analyzeLens(p, fields, {LambdaF, LambdaD, LambdaC});
    std::cout << rep.text();
    if (a.has("--svg")) {
        std::vector<PathRecord> all;
        std::vector<double> fs = a.has("--field") ? numbers(a.get("--field")) : std::vector<double>{fields.front(), fields.back()};
        Scene sc;
        int rays = std::stoi(a.get("--rays", "11"));
        for (double f : fs)
            for (double l : {LambdaF, LambdaD, LambdaC}) {
                Scene tmp;
                auto fan = lensFan(p, f, l, rays, tmp);
                all.insert(all.end(), fan.begin(), fan.end());
            }
        buildPrescription(sc.world, p, "lens", -1, Transform{}, 0.0, catalogIndex());
        sc.world.build();
        ViewPlane vp;
        vp.u = {0, 0, 1};
        vp.v = {0, 1, 0};
        SvgOptions so;
        so.title = p.name.empty() ? a.positional[0] : p.name;
        writePathsSVG(a.get("--svg"), sc.world, all, vp, so);
        std::cout << "wrote " << a.get("--svg") << "\n";
    }
    return 0;
}

int cmdGlass(const Args& a) {
    if (a.positional.empty()) {
        std::puts("media:");
        for (auto& n : catalogMediumNames()) {
            Medium m;
            catalogMedium(n, m);
            if (m.opaque) continue;
            std::printf("  %-14s n_d = %.5f  V_d = %6.2f  %s\n", n.c_str(), m.index.nRelative(LambdaD),
                        abbeNumber(m.index), m.index.describe().c_str());
        }
        std::puts("metals:");
        for (auto& n : catalogConductorNames()) std::printf("  %s\n", n.c_str());
        return 0;
    }
    Medium m;
    if (!catalogMedium(a.positional[0], m)) throw std::runtime_error("unknown medium " + a.positional[0]);
    std::printf("%s: %s\n", m.name.c_str(), m.index.describe().c_str());
    for (double l : {404.656, 435.835, LambdaF, 546.074, LambdaD, LambdaC, 706.519})
        std::printf("  λ = %8.3f nm   n = %.6f   (absolute %.6f)   α = %.4g /m\n", l, m.index.nRelative(l), m.n(l),
                    m.absorption.eval(l));
    std::printf("  Abbe number V_d = %.3f\n", abbeNumber(m.index));
    return 0;
}

int cmdInfo(const Args& a) {
    if (a.positional.empty()) throw std::runtime_error("info: missing scene file");
    Scene s = loadScene(a.positional[0]);
    const World& w = s.world;
    std::printf("media (%zu):\n", w.media().size());
    for (auto& m : w.media())
        std::printf("  %-16s n_d=%.6f %s%s\n", m.name.c_str(), m.n(LambdaD), m.index.describe().c_str(), m.opaque ? " [opaque]" : "");
    std::printf("regions (%zu):\n", w.regions().size());
    for (size_t i = 0; i < w.regions().size(); ++i) std::printf("  %3zu %s\n", i, w.regionLabel(uint32_t(i)).c_str());
    std::printf("assemblies (%zu):\n", w.assemblies().size());
    for (auto& as : w.assemblies())
        std::printf("  %s%s\n", as.name.c_str(), as.parent >= 0 ? (" in " + w.assemblies()[as.parent].name).c_str() : "");
    std::printf("bodies (%zu):\n", w.bodies().size());
    for (auto& b : w.bodies()) {
        std::printf("  %-28s %-14s regions=%zu boundaries=%zu", b.name.c_str(), b.kind.c_str(), b.regions.size(), b.boundaries.size());
        int shown = 0;
        for (auto& [k, v] : b.params)
            if (shown++ < 6) std::printf(" %s=%g", k.c_str(), v);
        std::printf("\n");
    }
    std::printf("boundaries (%zu):\n", w.boundaries().size());
    size_t shown = 0;
    for (size_t i = 0; i < w.boundaries().size(); ++i) {
        if (shown++ >= 60) { std::printf("  ... %zu more\n", w.boundaries().size() - 60); break; }
        const Boundary& b = w.boundaries()[i];
        std::printf("  %-34s %-10s front=%-26s back=%-26s %s\n", w.boundaryLabel(uint32_t(i)).c_str(),
                    surfaceTypeName(w.optics()[b.optics].type), w.regionLabel(b.front).c_str(), w.regionLabel(b.back).c_str(),
                    b.shape->describe().c_str());
    }
    std::printf("lights for next-event estimation: %zu\n", w.lights().size());
    std::printf("detectors:\n");
    for (auto& d : s.detectors) std::printf("  %s (starts in %s)\n", d->describe().c_str(), w.regionLabel(d->region).c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    try {
        Args a = parse(argc, argv, 2);
        if (cmd == "render") return cmdRender(a);
        if (cmd == "probe") return cmdProbe(a);
        if (cmd == "emit") return cmdEmit(a);
        if (cmd == "lens") return cmdLens(a);
        if (cmd == "glass") return cmdGlass(a);
        if (cmd == "info") return cmdInfo(a);
        if (cmd == "help" || cmd == "--help" || cmd == "-h") { usage(); return 0; }
        std::fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
