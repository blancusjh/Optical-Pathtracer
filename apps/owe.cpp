// owe — command-line front end of the Optical World Engine. It renders through the backend registry
// (owe/backends/registry.hpp): --backend chooses where the transport runs, nothing else changes.
#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "owe/analysis/analysis.hpp"
#include "owe/analysis/inspect.hpp"
#include "owe/backends/compare.hpp"
#include "owe/backends/registry.hpp"
#include "owe/core/version.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/render/output.hpp"
#include "owe/render/record.hpp"
#include "viewer.hpp"

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
    {"--bounds", 1},   {"--title", 1},      {"--fresnel-floor", 1}, {"--set", 1}, {"--white-balance", 1},
    {"--repeat", 1}, {"--backend", 1}, {"--device", 1}, {"--runs", 1}, {"--block", 1},
    {"--view", 1}, {"--scale", 1}, {"--tour-seconds", 1}, {"--screenshot", 1}, {"--after", 1}, {"--reference", 1}};

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

void applyResolution(Scene& scene, int di, const Args& a) {
    if (!a.has("--resolution")) return;
    std::string r = a.get("--resolution");
    auto x = r.find('x');
    if (x == std::string::npos) throw std::runtime_error("--resolution expects WxH");
    scene.detectors[size_t(di)]->width = std::stoi(r.substr(0, x));
    scene.detectors[size_t(di)]->height = std::stoi(r.substr(x + 1));
    scene.build();
}

// --backend NAME and --device N; an integrator the backend lacks becomes path (with a note).
void applyBackend(RenderSettings& rs, const Args& a) {
    if (a.has("--backend")) rs.backend = a.get("--backend");
    if (a.has("--device")) rs.device = std::stoi(a.get("--device"));
    std::string why;
    if (!backend(rs.backend).available(&why)) throw std::runtime_error(rs.backend + " backend unavailable: " + why);
    std::string note = adaptToBackend(rs);
    if (!note.empty()) std::fprintf(stderr, "note: %s\n", note.c_str());
}

std::string backendNames() {
    std::string names;
    for (const Backend* b : backends()) names += (names.empty() ? "" : "|") + b->name();
    return names;
}

int detectorIndex(const Scene& s, const Args& a) {
    std::string name = a.get("--detector", s.render.detector);
    if (name.empty()) return 0;
    int i = s.findDetector(name);
    if (i < 0) throw std::runtime_error("no detector named '" + name + "'");
    return i;
}

void usage() {
    std::printf(
        "owe " OWE_VERSION " — Optical World Engine\n"
        "\n"
        "  owe render <scene.owe> [--detector NAME] [--spp N] [--passes K] [--integrator path|light]\n"
        "                         [--seed S] [--threads T] [--max-depth D] [--resolution WxH]\n"
        "                         [--out PREFIX] [--exposure EV] [--no-auto-exposure] [--fresnel-floor P]\n"
        "                         [--white-balance KELVIN] [--backend %s] [--device N]\n"
        "      Progressive spectral render. Writes PREFIX.png (display), PREFIX.pfm (raw linear) and\n"
        "      PREFIX.json (reproducibility record) after every pass.\n"
        "\n"
        "  --backend chooses where the transport runs (render, studio, bench, view, compare):\n"
        "      cpu  the reference: IEEE-754 double on host threads (default)\n"
        "      gpu  portable Vulkan compute (NVIDIA, AMD, Intel; Apple silicon via MoltenVK), float32\n"
        "      Both implement every integrator; the measurement is the same.\n"
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
        "  owe studio <scene.owe> [--detector NAME] [--resolution WxH] [--out PREFIX] [--backend NAME]\n"
        "                         [--set Block.key=value ...]\n"
        "      Interactive session: tune aperture, focus, placement, backend or any scene value and\n"
        "      refine the image pass by pass (reads commands from stdin; type help).\n"
        "\n"
        "  owe view [scene.owe|directory ...] [--view file.owe:DETECTOR] [--backend NAME]\n"
        "           [--device N] [--scale S] [--tour] [--tour-seconds S] [--set Block.key=value ...]\n"
        "           [--screenshot FILE.png|DIRECTORY/] [--after SECONDS]\n"
        "      Interactive window: browse scenes and detectors, watch progressive rendering.\n"
        "      Arrow keys switch scenes/views; G changes backend; Space pauses; T tours; S saves; Q quits.\n"
        "      Middle drag orbits; Shift+middle pans; wheel dollies; right mouse + WASD flies.\n"
        "      Shift+F toggles flight; Esc releases the mouse; Home restores the saved view.\n"
        "\n"
        "  Every command accepts --set Block.key=value to edit the scene without rewriting it,\n"
        "  e.g. --set Cam.f_number=2.8 --set \"Cam.focus=3.5 m\".\n"
        "\n"
        "  owe bench [scene.owe[:DETECTOR] ...] [--spp N] [--resolution WxH] [--repeat R] [--threads T]\n"
        "                                       [--integrator path|light|hybrid] [--backend NAME] [--json F]\n"
        "      Throughput on fixed scenes and seeds (Mpaths/s, Msegments/s) with a digest of the raw\n"
        "      image, so a change that alters any result bit shows up. Default: the canonical suite.\n"
        "\n"
        "  owe compare <scene.owe>[:DETECTOR] [--backend NAME] [--reference NAME] [--integrator path|light|hybrid]\n"
        "                                     [--spp N] [--runs K] [--resolution WxH] [--block B] [--device N]\n"
        "      Statistical test of a backend (default: gpu) against the reference (cpu): K independent renders\n"
        "      per backend, block means of X, Y, Z compared with their standard errors (exit 3 if inconsistent).\n"
        "\n"
        "  owe backends [NAME]  The backends in this build, their devices and integrators (exit 1 if NAME\n"
        "                       is unavailable here).\n"
        "  owe glass [NAME]    Refractive index model, n at spectral lines, Abbe number.\n"
        "  owe info <scene.owe>  The world's ontology: media, regions, boundaries, bodies, detectors.\n",
        backendNames().c_str());
}

int cmdView(const Args& a) {
    ViewerOptions o;
    o.paths = a.positional;
    o.view = a.get("--view");
    o.backend = a.get("--backend");
    o.device = std::stoi(a.get("--device", "-1"));
    o.scale = std::stod(a.get("--scale", "0"));
    o.edits = a.all("--set");
    o.tour = a.has("--tour");
    o.tourSeconds = std::stod(a.get("--tour-seconds", "10"));
    o.screenshot = a.get("--screenshot");
    o.after = std::stod(a.get("--after", "5"));
    return runViewer(o);
}

int cmdRender(const Args& a) {
    if (a.positional.empty()) throw std::runtime_error("render: missing scene file");
    Scene scene = loadSceneWithEdits(a.positional[0], a.all("--set"));
    int di = detectorIndex(scene, a);
    scene.useDetector(di);
    RenderSettings rs = scene.settingsFor(di);
    if (a.has("--spp")) rs.spp = std::stoi(a.get("--spp"));
    if (a.has("--seed")) rs.seed = std::stoull(a.get("--seed"));
    if (a.has("--integrator")) rs.integrator = a.get("--integrator");
    if (a.has("--threads")) rs.threads = std::stoi(a.get("--threads"));
    if (a.has("--max-depth")) rs.maxDepth = std::stoi(a.get("--max-depth"));
    if (a.has("--exposure")) rs.exposure = std::stod(a.get("--exposure"));
    if (a.has("--fresnel-floor")) rs.fresnelFloor = std::stod(a.get("--fresnel-floor"));
    if (a.has("--white-balance")) rs.whiteBalance = std::stod(a.get("--white-balance"));
    if (a.has("--no-auto-exposure")) rs.autoExposure = false;
    applyBackend(rs, a);
    applyResolution(scene, di, a);
    int passes = std::max(1, std::stoi(a.get("--passes", "1")));
    std::string out = a.get("--out", "render");
    auto Rp = makeRenderer(scene, di, rs);
    Renderer& R = *Rp;
    std::fprintf(stderr, "%s\n", R.backend().c_str());
    std::fprintf(stderr, "%s\n", scene.detectors[di]->describe().c_str());
    if (scene.notes.count(scene.detectors[di]->name)) std::fprintf(stderr, "%s\n", scene.notes[scene.detectors[di]->name].c_str());
    int done = 0;
    for (int p = 0; p < passes; ++p) {
        int spp = rs.spp * (p + 1) / passes - done;
        done += spp;
        if (spp <= 0) continue;
        R.runPass(spp);
        Image img = R.resolve();
        writePNG(out + ".png", img, rs.exposure, rs.autoExposure, rs.whiteBalance);
        writePFM(out + ".pfm", img);
        std::ofstream(out + ".json") << renderMetadataJSON(scene, R, rs);
        std::fprintf(stderr, "pass %d/%d: %lld spp, %.2f s, mean Y %.5g\n", p + 1, passes, R.samplesPerPixel(),
                     R.seconds(), img.meanY());
    }
    const TransportStats& st = R.stats();
    std::fprintf(stderr, "paths %llu, segments %llu, region inconsistencies %llu, leaks %llu\n",
                 (unsigned long long)st.paths, (unsigned long long)st.segments, (unsigned long long)st.inconsistencies,
                 (unsigned long long)st.leaks);
    if (st.inconsistencies && st.first.boundary != kNone) {  // the GPU backend counts but does not record them
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
    Scene scene = loadSceneWithEdits(a.positional[0], a.all("--set"));
    int di = detectorIndex(scene, a);
    scene.useDetector(di);
    applyResolution(scene, di, a);
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
    Scene scene = loadSceneWithEdits(a.positional[0], a.all("--set"));
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
    Scene s = loadSceneWithEdits(a.positional[0], a.all("--set"));
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
    for (auto& d : s.detectors) {
        std::printf("  %s (starts in %s)\n", d->describe().c_str(), w.regionLabel(d->region).c_str());
        if (s.notes.count(d->name)) std::printf("    %s\n", s.notes[d->name].c_str());
    }
    return 0;
}

// The backends in this build: availability, devices, integrators.
int cmdBackends(const Args& a) {
    std::string only = a.positional.empty() ? "" : a.positional[0];
    if (!only.empty()) backend(only);  // unknown names are an error
    bool ok = true;
    for (const Backend* b : backends()) {
        if (!only.empty() && b->name() != only) continue;
        std::string why;
        bool here = b->available(&why);
        ok = ok && here;
        std::string integrators;
        for (auto& i : b->integrators()) integrators += (integrators.empty() ? "" : ", ") + i;
        std::printf("%s%s — %s\n", b->name().c_str(), b == &referenceBackend() ? " (reference)" : "", b->summary().c_str());
        std::printf("  integrators: %s\n", integrators.c_str());
        if (!here) std::printf("  unavailable: %s\n", why.c_str());
        for (auto& d : b->devices())
            std::printf("  [%d] %s — %s, %s%s\n", d.index, d.name.c_str(), d.type.c_str(), d.driver.c_str(),
                        d.portability ? ", portability (MoltenVK)" : "");
    }
    return only.empty() || ok ? 0 : 1;  // exit status answers "can NAME run here?"
}

// Statistical comparison of a backend with the reference on one scene.
int cmdCompare(const Args& a) {
    if (a.positional.empty()) throw std::runtime_error("compare: missing scene file");
    std::string path = a.positional[0], det;
    if (auto c = path.rfind(':'); c != std::string::npos && path.substr(c).find('/') == std::string::npos) {
        det = path.substr(c + 1);
        path = path.substr(0, c);
    }
    Scene scene = loadSceneWithEdits(path, a.all("--set"));
    int di = det.empty() ? detectorIndex(scene, a) : scene.findDetector(det);
    if (di < 0) throw std::runtime_error("no detector named '" + det + "'");
    scene.useDetector(di);
    if (!a.has("--resolution")) {
        scene.detectors[size_t(di)]->width = 96;
        scene.detectors[size_t(di)]->height = 64;
        scene.build();
    }
    applyResolution(scene, di, a);
    RenderSettings reference = scene.settingsFor(di);
    reference.backend = backend(a.get("--reference", referenceBackend().name())).name();
    reference.integrator = a.get("--integrator", "path");
    if (a.has("--threads")) reference.threads = std::stoi(a.get("--threads"));
    RenderSettings candidate = reference;
    std::string other;  // default: the first backend besides the reference
    for (const Backend* b : backends())
        if (b != &backend(reference.backend) && other.empty()) other = b->name();
    candidate.backend = backend(a.get("--backend", other)).name();
    if (a.has("--device")) candidate.device = std::stoi(a.get("--device"));
    for (RenderSettings* rs : {&reference, &candidate})
        if (!backend(rs->backend).supports(rs->integrator))
            throw std::runtime_error("the " + rs->backend + " backend does not implement the " + rs->integrator + " integrator");
    int spp = std::stoi(a.get("--spp", "256")), runs = std::stoi(a.get("--runs", "8")), block = std::stoi(a.get("--block", "8"));
    std::printf("%s [%s] %dx%d, %s integrator, %d runs × %d spp per backend, %dx%d-pixel blocks\n", path.c_str(),
                scene.detectors[size_t(di)]->name.c_str(), scene.detectors[size_t(di)]->width,
                scene.detectors[size_t(di)]->height, reference.integrator.c_str(), runs, spp, block, block);
    ComparisonReport rep = compareRenderers(scene, di, reference, candidate, spp, runs, block);
    std::fputs(rep.text().c_str(), stdout);
    if (a.has("--verbose"))
        for (auto& o : rep.worst)
            std::printf("  block (%d, %d) %c: A %.6g  B %.6g  z %+.2f\n", o.bx, o.by, "XYZ"[o.channel], o.a, o.b, o.z);
    return rep.consistent() ? 0 : 3;
}

// Throughput benchmark over fixed scenes and seeds. The digest hashes the raw XYZ accumulation,
// so an optimisation that changes any result bit is visible at once.
int cmdBench(const Args& a) {
    struct Case {
        std::string scene, detector;
    };
    std::vector<Case> cases;
    for (const std::string& p : a.positional) {
        auto c = p.rfind(':');
        cases.push_back(c == std::string::npos ? Case{p, ""} : Case{p.substr(0, c), p.substr(c + 1)});
    }
    if (cases.empty())
        cases = {{"scenes/the_lens.owe", "Eye"},          {"scenes/the_prism.owe", "Eye"},
                 {"scenes/glass_of_water.owe", "Eye"},    {"scenes/the_statue.owe", "Magnifier"},
                 {"scenes/the_telescope.owe", "Eyepiece"}, {"scenes/the_temple.owe", "Wide"},
                 {"scenes/the_temple.owe", "Cam"},        {"scenes/the_observatory.owe", "SaturnEyepiece"},
                 {"scenes/camera_obscura.owe", "Inside"}};
    const int spp = std::stoi(a.get("--spp", "16"));
    const std::string res = a.get("--resolution", "256x192");
    const int repeat = std::max(1, std::stoi(a.get("--repeat", "1")));
    auto x = res.find('x');
    if (x == std::string::npos) throw std::runtime_error("--resolution expects WxH");
    const int W = std::stoi(res.substr(0, x)), H = std::stoi(res.substr(x + 1));
    std::ostringstream js;
    js << "[\n";
    std::printf("%-26s %-15s %9s %4s %9s %9s %8s %12s %16s\n", "scene", "detector", "res", "spp", "Mpaths/s", "Msegs/s",
                "seconds", "mean Y", "digest");
    std::fflush(stdout);
    double totalPaths = 0, totalSeconds = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case& c = cases[ci];
        Scene scene = loadSceneWithEdits(c.scene, a.all("--set"));
        int di = c.detector.empty() ? detectorIndex(scene, a) : scene.findDetector(c.detector);
        if (di < 0) throw std::runtime_error("no detector named '" + c.detector + "' in " + c.scene);
        scene.useDetector(di);
        scene.detectors[size_t(di)]->width = W;
        scene.detectors[size_t(di)]->height = H;
        scene.build();
        RenderSettings rs = scene.settingsFor(di);
        rs.integrator = a.get("--integrator", "path");
        if (a.has("--seed")) rs.seed = std::stoull(a.get("--seed"));
        if (a.has("--threads")) rs.threads = std::stoi(a.get("--threads"));
        applyBackend(rs, a);
        double best = Inf;
        Image img;
        TransportStats st;
        for (int r = 0; r < repeat; ++r) {
            auto R = makeRenderer(scene, di, rs);
            if (ci == 0 && r == 0) std::printf("backend: %s\n", R->backend().c_str());
            R->runPass(spp);
            best = std::min(best, R->seconds());
            img = R->resolve();
            st = R->stats();
        }
        std::string bytes(reinterpret_cast<const char*>(img.xyz.data()), img.xyz.size() * sizeof(double));
        char digest[32];
        std::snprintf(digest, sizeof digest, "%016llx", (unsigned long long)fnv1a(bytes));
        std::string name = c.scene.substr(c.scene.find_last_of('/') + 1);
        double mp = st.paths / best * 1e-6, ms = st.segments / best * 1e-6;
        totalPaths += double(st.paths);
        totalSeconds += best;
        std::printf("%-26s %-15s %9s %4d %9.3f %9.3f %8.3f %12.6g %16s\n", name.c_str(),
                    scene.detectors[size_t(di)]->name.c_str(), res.c_str(), spp, mp, ms, best, img.meanY(), digest);
        std::fflush(stdout);
        js << "  {\"scene\": \"" << c.scene << "\", \"detector\": \"" << scene.detectors[size_t(di)]->name
           << "\", \"resolution\": \"" << res << "\", \"spp\": " << spp << ", \"integrator\": \"" << rs.integrator
           << "\", \"seconds\": " << best << ", \"paths\": " << st.paths << ", \"segments\": " << st.segments
           << ", \"mean_Y\": " << img.meanY() << ", \"digest\": \"" << digest << "\"}"
           << (ci + 1 < cases.size() ? "," : "") << "\n";
    }
    js << "]\n";
    std::printf("total: %.3f s, %.3f Mpaths/s\n", totalSeconds, totalPaths / totalSeconds * 1e-6);
    if (a.has("--json")) std::ofstream(a.get("--json")) << js.str();
    return 0;
}

// Interactive session: edit the scene (aperture, focus, placement, anything), watch the
// estimate refine pass by pass. Reads commands from stdin, so it can also be scripted.
int cmdStudio(const Args& a) {
    if (a.positional.empty()) throw std::runtime_error("studio: missing scene file");
    const std::string path = a.positional[0];
    std::vector<std::string> edits = a.all("--set");
    std::string detName = a.get("--detector");
    std::string out = a.get("--out", "studio");
    int width = 0, height = 0;
    if (a.has("--resolution")) {
        std::string r = a.get("--resolution");
        auto x = r.find('x');
        if (x == std::string::npos) throw std::runtime_error("--resolution expects WxH");
        width = std::stoi(r.substr(0, x));
        height = std::stoi(r.substr(x + 1));
    }
    double exposure = 0, whiteBalance = 0;
    bool autoExp = true, userExposure = false;  // set by the exposure command; cleared by detector
    std::string backendName = a.get("--backend");  // set by the backend command
    std::unique_ptr<Scene> scene;
    std::unique_ptr<Renderer> R;
    bool interactive = isatty(0);
    auto rebuild = [&] {
        R.reset();
        scene = std::make_unique<Scene>(loadSceneWithEdits(path, edits));
        if (detName.empty()) detName = scene->render.detector.empty() ? scene->detectors[0]->name : scene->render.detector;
        int di = scene->findDetector(detName);
        if (di < 0) throw std::runtime_error("no detector named '" + detName + "'");
        scene->useDetector(di);
        if (width > 0) {
            scene->detectors[size_t(di)]->width = width;
            scene->detectors[size_t(di)]->height = height;
            scene->build();
        }
        RenderSettings rs = scene->settingsFor(di);
        if (!backendName.empty()) rs.backend = backendName;
        if (a.has("--device")) rs.device = std::stoi(a.get("--device"));
        std::string note = adaptToBackend(rs);
        if (!note.empty()) std::printf("note: %s\n", note.c_str());
        whiteBalance = a.has("--white-balance") ? std::stod(a.get("--white-balance")) : rs.whiteBalance;
        if (!userExposure) {
            exposure = a.has("--exposure") ? std::stod(a.get("--exposure")) : rs.exposure;
            autoExp = rs.autoExposure && !a.has("--no-auto-exposure");
        }
        R = makeRenderer(*scene, di, rs);
        std::printf("%s\n%s\n", R->backend().c_str(), scene->detectors[size_t(di)]->describe().c_str());
        if (scene->notes.count(detName)) std::printf("%s\n", scene->notes[detName].c_str());
    };
    auto help = [] {
        std::puts("commands:\n"
                  "  set Block.key=value     edit the scene (e.g. set Cam.f_number=2.8, set Cam.focus=3.2 m)\n"
                  "  unset Block.key         drop edits of that key\n"
                  "  edits                   list current edits\n"
                  "  detector NAME           render another observer/camera/sensor\n"
                  "  size WxH                image size\n"
                  "  backend NAME            where the transport runs (owe backends lists them)\n"
                  "  render [SPP] [PASSES]   refine the current estimate (edits restart it)\n"
                  "  reset                   restart accumulation\n"
                  "  exposure EV | auto      display exposure\n"
                  "  info                    detector and camera state (EFL, f-number, depth of field)\n"
                  "  probe X Y [N]           why is this pixel this colour?\n"
                  "  quit");
    };
    try { rebuild(); } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    if (interactive) help();
    std::string line;
    while (true) {
        if (interactive) { std::printf("owe> "); std::fflush(stdout); }
        if (!std::getline(std::cin, line)) break;
        std::istringstream ls(line);
        std::string cmd;
        ls >> cmd;
        if (cmd.empty() || cmd[0] == '#') continue;
        try {
            if (cmd == "quit" || cmd == "exit") break;
            else if (cmd == "help") help();
            else if (cmd == "set") {
                std::string rest;
                std::getline(ls, rest);
                rest.erase(0, rest.find_first_not_of(" \t"));
                // Accept "Cam.f_number=2.8" and "Cam f_number 2.8".
                if (rest.find('=') == std::string::npos) {
                    std::istringstream rs(rest);
                    std::string blk, key, val;
                    rs >> blk >> key;
                    std::getline(rs, val);
                    rest = blk + "." + key + "=" + val;
                }
                edits.push_back(rest);
                rebuild();
            } else if (cmd == "unset") {
                std::string key;
                ls >> key;
                edits.erase(std::remove_if(edits.begin(), edits.end(),
                                           [&](const std::string& e) { return e.rfind(key + "=", 0) == 0 || e.rfind(key + " =", 0) == 0; }),
                            edits.end());
                rebuild();
            } else if (cmd == "edits") {
                for (auto& e : edits) std::printf("  %s\n", e.c_str());
            } else if (cmd == "detector") {
                ls >> detName;
                userExposure = false;
                rebuild();
            } else if (cmd == "size") {
                std::string wh;
                ls >> wh;
                auto x = wh.find('x');
                width = std::stoi(wh.substr(0, x));
                height = std::stoi(wh.substr(x + 1));
                rebuild();
            } else if (cmd == "backend") {
                std::string name;
                ls >> name;
                std::string why;
                if (!backend(name).available(&why)) throw std::runtime_error(name + " unavailable: " + why);
                backendName = name;
                rebuild();
            } else if (cmd == "reset") {
                rebuild();
            } else if (cmd == "exposure") {
                std::string v;
                ls >> v;
                autoExp = v == "auto";
                if (!autoExp) exposure = std::stod(v);
                userExposure = true;
                writePNG(out + ".png", R->resolve(), exposure, autoExp, whiteBalance);
            } else if (cmd == "info") {
                std::printf("%s\n", R->detector().describe().c_str());
                if (scene->notes.count(detName)) std::printf("%s\n", scene->notes[detName].c_str());
                std::printf("%lld spp accumulated, %.1f s\n", R->samplesPerPixel(), R->seconds());
            } else if (cmd == "render") {
                int spp = 16, passes = 1;
                ls >> spp >> passes;
                if (spp <= 0) spp = 16;
                if (passes <= 0) passes = 1;
                for (int p = 0; p < passes; ++p) {
                    R->runPass(spp);
                    Image img = R->resolve();
                    writePNG(out + ".png", img, exposure, autoExp, whiteBalance);
                    writePFM(out + ".pfm", img);
                    std::ofstream(out + ".json") << renderMetadataJSON(*scene, *R, R->settings());
                    std::printf("  %lld spp  %.1f s  mean Y %.5g  -> %s.png\n", R->samplesPerPixel(), R->seconds(), img.meanY(),
                                out.c_str());
                    std::fflush(stdout);
                }
            } else if (cmd == "probe") {
                int x, y, n = 1000;
                ls >> x >> y >> n;
                if (n <= 0) n = 1000;
                PixelProbe pr = probePixel(*scene, scene->findDetector(detName), x, y, n, scene->render.seed);
                std::cout << pr.text(scene->world);
            } else {
                std::printf("unknown command '%s' (help lists commands)\n", cmd.c_str());
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            if (cmd == "set" && !edits.empty()) {
                edits.pop_back();  // keep the session usable after a bad edit
                try { rebuild(); } catch (...) {}
            }
        }
    }
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
        if (cmd == "studio") return cmdStudio(a);
        if (cmd == "view") return cmdView(a);
        if (cmd == "bench") return cmdBench(a);
        if (cmd == "backends") return cmdBackends(a);
        if (cmd == "compare") return cmdCompare(a);
        if (cmd == "help" || cmd == "--help" || cmd == "-h") { usage(); return 0; }
        std::fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
