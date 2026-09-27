#include "owe/render/record.hpp"

#include <cstdio>
#include <iomanip>
#include <sstream>

#include "owe/core/version.hpp"
#include "owe/render/output.hpp"

namespace owe {

uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

namespace {
std::string jsonEscape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') { r += '\\'; r += c; }
        else if (c == '\n') r += "\\n";
        else if ((unsigned char)c < 0x20) { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); r += buf; }
        else r += c;
    }
    return r;
}
}  // namespace

std::string renderMetadataJSON(const Scene& scene, const Renderer& r, const RenderSettings& s) {
    const World& w = scene.world;
    std::ostringstream os;
    char hash[32];
    std::snprintf(hash, sizeof hash, "%016llx", (unsigned long long)fnv1a(scene.sourceText));
    Image img = r.resolve();
    os << "{\n";
    os << "  \"engine\": \"optical-world-engine " << OWE_VERSION << "\",\n";
    os << "  \"backend\": \"" << jsonEscape(r.backend()) << "\",\n";
    os << "  \"scene\": {\"path\": \"" << jsonEscape(scene.sourcePath) << "\", \"fnv1a64\": \"" << hash << "\", \"edits\": [";
    for (size_t i = 0; i < scene.edits.size(); ++i) os << (i ? ", " : "") << "\"" << jsonEscape(scene.edits[i]) << "\"";
    os << "]},\n";
    auto note = scene.notes.find(r.detector().name);
    if (note != scene.notes.end()) os << "  \"camera\": \"" << jsonEscape(note->second) << "\",\n";
    os << "  \"detector\": {\"name\": \"" << jsonEscape(r.detector().name) << "\", \"description\": \""
       << jsonEscape(r.detector().describe()) << "\", \"quantity\": \""
       << (r.detector().quantity() == Detector::Quantity::Radiance ? "radiance [W m^-2 sr^-1] as CIE XYZ"
                                                                    : "irradiance [W m^-2] as CIE XYZ")
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
        os << ", \"width\": " << eye->width << ", \"height\": " << eye->height << "},\n";
        os << std::setprecision(6);
    }
    os << "  \"integrator\": \"" << s.integrator << "\",\n";
    os << "  \"sampling\": {\"wavelengths\": \"4 per path (hero + 3 stratified; secondaries terminated at dispersive "
          "interfaces), visible-importance pdf on [" << LambdaMin << ", "
       << LambdaMax << "] nm\", \"fresnel\": \"stochastic branch selection\", \"nee\": true, \"mis\": \"power heuristic\","
       << " \"russian_roulette_depth\": " << s.rrDepth << ", \"fresnel_floor\": " << s.fresnelFloor
       << (s.integrator == "hybrid" ? ", \"partition\": \"light tracing owns eye-D-S+-light paths, path tracing all others\"" : "")
       << "},\n";
    os << "  \"seed\": " << s.seed << ",\n";
    os << "  \"samples_per_pixel\": " << r.samplesPerPixel() << ",\n";
    os << "  \"passes\": " << r.passes() << ",\n";
    os << "  \"max_depth\": " << s.maxDepth << ",\n";
    os << "  \"display\": {\"exposure_ev\": " << s.exposure << ", \"auto_exposure\": " << (s.autoExposure ? "true" : "false")
       << ", \"auto_gain_ev\": " << (s.autoExposure ? autoExposureEV(img) : 0)
       << ", \"white_balance_kelvin\": " << s.whiteBalance << "},\n";
    os << "  \"render_seconds\": " << r.seconds() << ",\n";
    os << "  \"statistics\": {\"paths\": " << r.stats().paths << ", \"segments\": " << r.stats().segments
       << ", \"region_inconsistencies\": " << r.stats().inconsistencies << ", \"leaks\": " << r.stats().leaks;
    if (r.stats().inconsistencies)
        os << ", \"first_inconsistency\": {\"boundary\": \"" << jsonEscape(w.boundaryLabel(r.stats().first.boundary))
           << "\", \"ray_region\": \"" << jsonEscape(w.regionLabel(r.stats().first.rayRegion)) << "\"}";
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

}  // namespace owe
