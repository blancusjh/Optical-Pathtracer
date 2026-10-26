#include "owe/render/record.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "owe/core/json.hpp"
#include "owe/core/version.hpp"
#include "owe/render/exposure.hpp"
#include "owe/render/output.hpp"

namespace owe {

uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}


std::string renderMetadataJSON(const Scene& scene, const Renderer& r, const RenderSettings& s, const Image& img) {
    const World& w = scene.world;
    std::ostringstream os;
    char hash[32];
    std::snprintf(hash, sizeof hash, "%016llx", (unsigned long long)fnv1a(scene.sourceText));
    os << "{\n";
    os << "  \"engine\": \"optical-world-engine " << OWE_VERSION << "\",\n";
    os << "  \"backend\": \"" << jsonEscape(r.backend()) << "\",\n";
    os << "  \"scene\": {\"path\": \"" << jsonEscape(scene.sourcePath) << "\", \"fnv1a64\": \"" << hash << "\", \"edits\": [";
    for (size_t i = 0; i < scene.edits.size(); ++i) os << (i ? ", " : "") << "\"" << jsonEscape(scene.edits[i]) << "\"";
    os << "]},\n";
    if (const std::string note = scene.noteFor(r.detector().name); !note.empty())
        os << "  \"camera\": \"" << jsonEscape(note) << "\",\n";
    os << "  \"detector\": {\"name\": \"" << jsonEscape(r.detector().name) << "\", \"description\": \""
       << jsonEscape(r.detector().describe()) << "\", \"quantity\": \""
       << (r.detector().quantity() == Detector::Quantity::Radiance ? "radiance [W m^-2 sr^-1] as CIE XYZ"
                                                                    : "irradiance [W m^-2] as CIE XYZ")
       << "\", \"pixel_response\": \"square"
       << (r.detector().pixelSigma > 0 ? " * gaussian(sigma " + std::to_string(r.detector().pixelSigma) + " px)" : "")
       << "\"},\n";
    if (auto* eye = dynamic_cast<const IdealObserver*>(&r.detector())) {
        os << std::setprecision(17);
        auto vec = [&](const Vec3& v) { os << "[" << v.x << ", " << v.y << ", " << v.z << "]"; };
        os << "  \"observer_pose\": {\"position_m\": "; vec(eye->position);
        os << ", \"look_at_m\": "; vec(eye->lookAt);
        os << ", \"up\": "; vec(eye->up);
        os << ", \"fov_deg\": " << degrees(eye->fovY) << ", \"pupil_radius_m\": " << eye->pupilRadius;
        os << ", \"focus_distance_m\": ";
        if (std::isfinite(eye->focusDistance)) os << eye->focusDistance; else os << "null";
        os << ", \"width\": " << eye->width << ", \"height\": " << eye->height;
        if (eye->guide.radius > 0)  // a sampling choice (variance only), recorded for exact reproduction
            os << ", \"pupil_guide\": {\"instrument\": \"" << jsonEscape(eye->guide.instrument)
               << "\", \"radius_m\": " << eye->guide.radius << ", \"share\": " << eye->guide.share << "}";
        os << "},\n";
        os << std::setprecision(6);
    }
    os << "  \"integrator\": \"" << s.integrator << "\",\n";
    os << "  \"sampling\": {\"wavelengths\": \"4 per path (hero + 3 stratified; secondaries terminated at dispersive "
          "interfaces), visible-importance pdf on [" << LambdaMin << ", "
       << LambdaMax << "] nm\", \"fresnel\": \"stochastic branch selection\", \"nee\": true, \"mis\": \"power heuristic\","
       << " \"russian_roulette_depth\": " << s.rrDepth << ", \"fresnel_floor\": " << s.fresnelFloor
       << (s.integrator == "hybrid" ? ", \"partition\": \"light tracing owns eye-D-S+-light paths, path tracing all others\"" : "")
       << (s.integrator == "sppm" ? ", \"photons\": {\"radius_pixels\": " + std::to_string(s.photonRadius) +
                                        ", \"alpha\": " + std::to_string(s.photonAlpha) +
                                        ", \"per_pixel_and_iteration\": " + std::to_string(s.photonsPerPixel) +
                                        ", \"wavelengths\": \"shared by every path and photon of an iteration\"}"
                                  : std::string())
       << (s.integrator == "bdpt" || s.integrator == "vcm"
               ? std::string(", \"bidirectional\": {\"light_vertex_cache\": true, \"connections_per_vertex\": 1, "
                             "\"mis\": \"balance heuristic over all strategies\"") +
                     (s.integrator == "vcm" ? ", \"merging\": {\"radius_pixels\": " + std::to_string(s.photonRadius) +
                                                  ", \"alpha\": " + std::to_string(s.photonAlpha) + "}"
                                            : std::string()) +
                     "}"
               : std::string())
       << "},\n";
    os << "  \"seed\": " << s.seed << ",\n";
    os << "  \"samples_per_pixel\": " << r.samplesPerPixel() << ",\n";
    os << "  \"passes\": " << r.passes() << ",\n";
    os << "  \"max_depth\": " << s.maxDepth << ",\n";
    os << "  \"display\": {\"exposure_ev\": " << s.exposure << ", \"auto_exposure\": " << (s.autoExposure ? "true" : "false")
       << ", \"auto_gain_ev\": " << (s.autoExposure ? meterEV(img) : 0)
       << ", \"white_balance_kelvin\": " << s.whiteBalance << ", \"tone\": \"" << s.tone << "\"},\n";
    os << "  \"render_seconds\": " << r.seconds() << ",\n";
    os << "  \"statistics\": {\"paths\": " << r.stats().paths << ", \"segments\": " << r.stats().segments
       << ", \"region_inconsistencies\": " << r.stats().inconsistencies << ", \"leaks\": " << r.stats().leaks;
    if (r.stats().inconsistencies) {
        const auto& first = r.stats().first;
        os << ", \"first_inconsistency\": {\"boundary\": \"" << jsonEscape(w.boundaryLabel(first.boundary))
           << "\", \"ray_region\": \"" << jsonEscape(w.regionLabel(first.rayRegion))
           << "\", \"expected_region\": \"" << jsonEscape(w.regionLabel(first.boundaryRegion))
           << "\", \"previous_boundary\": \"" << jsonEscape(w.boundaryLabel(first.previous))
           << "\", \"position_m\": [" << std::setprecision(17) << first.p.x << ", " << first.p.y << ", "
           << first.p.z << "]}" << std::setprecision(6);
    }
    os << "},\n";
    os << "  \"mean_Y\": " << img.meanY() << ",\n";
    os << "  \"world\": {\"media\": [";
    for (size_t i = 0; i < w.media().size(); ++i)
        os << (i ? ", " : "") << "{\"name\": \"" << jsonEscape(w.media()[i].name) << "\", \"index\": \""
           << jsonEscape(w.media()[i].index.describe()) << "\", \"n_d\": " << w.media()[i].n(LambdaD) << "}";
    os << "], \"regions\": " << w.regions().size() << ", \"boundaries\": " << w.boundaries().size()
       << ", \"bodies\": " << w.bodies().size() << ", \"assemblies\": " << w.assemblies().size() << "}\n";
    os << "}\n";
    return os.str();
}

void writeRenderFiles(const std::string& prefix, const Scene& scene, const Renderer& r, const RenderSettings& s,
                      const Image& img) {
    writePNG(prefix + ".png", img, s.exposure, s.autoExposure, s.whiteBalance, toneFromName(s.tone));
    writePFM(prefix + ".pfm", img);
    std::ofstream(prefix + ".json") << renderMetadataJSON(scene, r, s, img);
}

bool readDisplaySettings(const std::string& recordPath, RenderSettings& s) {
    std::ifstream in(recordPath);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string j = ss.str();
    // The keys renderMetadataJSON writes in its "display" object, each once in the record.
    auto value = [&](const char* key) -> std::string {
        auto p = j.find(std::string("\"") + key + "\": ");
        if (p == std::string::npos) return {};
        p += std::strlen(key) + 4;
        if (j[p] == '"') return j.substr(p + 1, j.find('"', p + 1) - p - 1);
        return j.substr(p, j.find_first_of(",}\n", p) - p);
    };
    if (auto v = value("exposure_ev"); !v.empty()) s.exposure = std::stod(v);
    if (auto v = value("white_balance_kelvin"); !v.empty()) s.whiteBalance = std::stod(v);
    if (auto v = value("auto_exposure"); !v.empty()) s.autoExposure = v == "true";
    if (auto v = value("tone"); !v.empty()) s.tone = v;
    return true;
}

}  // namespace owe
