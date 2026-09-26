#include "render.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "version.hpp"

#ifdef OWE_HAVE_ZLIB
#include <zlib.h>
#endif

namespace owe {

void spectralWeights(double lambda, double pdf, double out[3]) {
    double inv = 1.0 / (pdf * cieYIntegral());
    out[0] = cieX(lambda) * inv;
    out[1] = cieY(lambda) * inv;
    out[2] = cieZ(lambda) * inv;
}

ProgressiveRenderer::ProgressiveRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings)
    : scene_(scene), settings_(settings), tracer_(scene, settings.maxDepth, settings.rrDepth) {
    if (detectorIndex < 0 || detectorIndex >= int(scene.detectors.size())) throw std::runtime_error("no such detector");
    det_ = scene.detectors[detectorIndex].get();
    film_ = Film(det_->width, det_->height);
    lightFilm_ = Film(det_->width, det_->height);
    tracer_.setFresnelFloor(settings.fresnelFloor);
    threads_ = settings.threads > 0 ? settings.threads : int(std::max(1u, std::thread::hardware_concurrency()));
    if (settings.integrator != "path" && settings.integrator != "light" && settings.integrator != "hybrid")
        throw std::runtime_error("unknown integrator '" + settings.integrator + "' (expected path, light or hybrid)");
    if (settings.integrator == "hybrid") {
        if (!det_->isVirtual())
            throw std::runtime_error("the hybrid integrator needs a virtual observer; use path or light for surface sensors");
        tracer_.setCausticPartition(true);
    }
}

void ProgressiveRenderer::runPass(int spp) {
    auto t0 = std::chrono::steady_clock::now();
    if (settings_.integrator == "light") passLight(spp);
    else if (settings_.integrator == "path") passPath(spp);
    else {
        passPath(spp);
        passLight(spp);
    }
    passes_++;
    totalSpp_ += spp;
    seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void ProgressiveRenderer::passPath(int spp) {
    const int W = det_->width, H = det_->height;
    const int tile = 16;
    const int tilesX = (W + tile - 1) / tile, tilesY = (H + tile - 1) / tile;
    std::atomic<int> next{0};
    std::vector<TransportStats> stats(threads_);
    auto worker = [&](int tid) {
        for (;;) {
            int t = next.fetch_add(1);
            if (t >= tilesX * tilesY) break;
            int tx = t % tilesX, ty = t / tilesX;
            for (int y = ty * tile; y < std::min(H, (ty + 1) * tile); ++y)
                for (int x = tx * tile; x < std::min(W, (tx + 1) * tile); ++x) {
                    uint64_t pixel = uint64_t(y) * W + x;
                    Rng rng(hashCombine(settings_.seed, pixel), uint64_t(passes_));
                    double acc[3] = {0, 0, 0};
                    // Hero wavelengths are stratified across the pixel's samples in this pass
                    // (with a per-pixel random rotation), which suppresses colour noise.
                    double rot = rng.uniform();
                    for (int s = 0; s < spp; ++s) {
                        double u = (s + rng.uniform()) / spp + rot;
                        Wavelengths wl = Wavelengths::sample(u - std::floor(u));
                        Ray ray;
                        double weight;
                        double px = x + rng.uniform(), py = y + rng.uniform();
                        if (!det_->generate(px, py, rng, ray, weight) || weight <= 0) continue;
                        Spec4 L = tracer_.radiance(ray, det_->region, wl, rng, stats[tid]);
                        if (L.isZero()) continue;
                        XYZ c = wl.toXYZ(L * weight);
                        if (!std::isfinite(c.x + c.y + c.z)) continue;  // never poison the accumulation
                        acc[0] += c.x;
                        acc[1] += c.y;
                        acc[2] += c.z;
                    }
                    film_.add(x, y, acc[0], acc[1], acc[2]);
                    film_.samples[pixel] += spp;
                }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < threads_; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();
    for (auto& s : stats) stats_.add(s);
}

void ProgressiveRenderer::passLight(int ppp) {
    const int W = det_->width, H = det_->height;
    const uint64_t total = uint64_t(W) * H * uint64_t(ppp);
    // Fixed lanes make the result independent of the thread count.
    const int lanes = 16;
    const uint64_t chunk = 4096;
    const uint64_t chunks = (total + chunk - 1) / chunk;
    std::vector<Film> films(lanes, Film(W, H));
    std::vector<TransportStats> stats(lanes);
    std::atomic<int> nextLane{0};
    auto worker = [&] {
        for (;;) {
            int lane = nextLane.fetch_add(1);
            if (lane >= lanes) break;
            for (uint64_t c = uint64_t(lane); c < chunks; c += lanes) {
                Rng rng(hashCombine(settings_.seed ^ 0x5bd1e995ULL, c), uint64_t(passes_));
                uint64_t end = std::min(total, (c + 1) * chunk);
                for (uint64_t i = c * chunk; i < end; ++i)
                    tracer_.traceParticle(*det_, Wavelengths::sample(rng.uniform()), rng, films[lane], stats[lane]);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < std::min(threads_, lanes); ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    for (int l = 0; l < lanes; ++l) {
        lightFilm_.merge(films[l]);
        stats_.add(stats[l]);
    }
    particles_ += double(total);
}

Image ProgressiveRenderer::resolve() const {
    Image img;
    img.width = film_.width;
    img.height = film_.height;
    img.xyz.assign(film_.xyz.size(), 0.0);
    size_t n = size_t(img.width) * img.height;
    double lightNorm = particles_ > 0 ? 1.0 / particles_ : 0;
    for (size_t i = 0; i < n; ++i) {
        double pathNorm = film_.samples[i] > 0 ? 1.0 / film_.samples[i] : 0;
        for (int c = 0; c < 3; ++c)
            img.xyz[3 * i + c] = film_.xyz[3 * i + c] * pathNorm + lightFilm_.xyz[3 * i + c] * lightNorm;
    }
    return img;
}

// ---------------------------------------------------------------- output

void writePFM(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "PF\n" << img.width << " " << img.height << "\n-1.0\n";
    std::vector<float> row(size_t(img.width) * 3);
    for (int y = img.height - 1; y >= 0; --y) {  // PFM stores bottom-to-top
        for (int x = 0; x < img.width; ++x) {
            double rgb[3];
            xyzToLinearSRGB(img.at(x, y), rgb);
            for (int c = 0; c < 3; ++c) row[3 * x + c] = float(rgb[c]);
        }
        out.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size() * sizeof(float)));
    }
}

namespace {
uint32_t crc32(const unsigned char* data, size_t n, uint32_t crc = 0) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}
void be32(std::vector<unsigned char>& v, uint32_t x) {
    v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}
void chunk(std::ofstream& out, const char* type, const std::vector<unsigned char>& data) {
    std::vector<unsigned char> buf;
    be32(buf, uint32_t(data.size()));
    buf.insert(buf.end(), type, type + 4);
    buf.insert(buf.end(), data.begin(), data.end());
    uint32_t c = crc32(buf.data() + 4, buf.size() - 4);
    be32(buf, c);
    out.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(buf.size()));
}
double srgbEncode(double v) {
    v = clampd(v, 0, 1);
    return v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1 / 2.4) - 0.055;
}
}  // namespace

namespace {
// Bradford chromatic adaptation matrix from white (Xw, Yw, Zw) to D65.
void bradfordToD65(const XYZ& w, double M[3][3]) {
    const double B[3][3] = {{0.8951, 0.2664, -0.1614}, {-0.7502, 1.7135, 0.0367}, {0.0389, -0.0685, 1.0296}};
    const double Bi[3][3] = {{0.9869929, -0.1470543, 0.1599627}, {0.4323053, 0.5183603, 0.0492912},
                             {-0.0085287, 0.0400428, 0.9684867}};
    const double d65[3] = {0.95047, 1.0, 1.08883};
    double src[3], dst[3];
    for (int i = 0; i < 3; ++i) {
        src[i] = B[i][0] * w.x + B[i][1] * w.y + B[i][2] * w.z;
        dst[i] = B[i][0] * d65[0] + B[i][1] * d65[1] + B[i][2] * d65[2];
    }
    double T[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) T[i][j] = dst[i] / src[i] * B[i][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M[i][j] = Bi[i][0] * T[0][j] + Bi[i][1] * T[1][j] + Bi[i][2] * T[2][j];
}
}  // namespace

void writePNG(const std::string& path, const Image& img, double exposureEV, bool autoExposure, double whiteKelvin) {
    double scale = std::pow(2.0, exposureEV);
    double wb[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    if (whiteKelvin > 0) bradfordToD65(Spectrum::blackbody(whiteKelvin, 1.0).toXYZ(), wb);
    if (autoExposure) {
        // Map the log-average luminance to middle grey.
        double s = 0;
        size_t n = 0;
        for (int y = 0; y < img.height; ++y)
            for (int x = 0; x < img.width; ++x) {
                double Y = img.at(x, y).y;
                if (Y > 0 && std::isfinite(Y)) { s += std::log(Y + 1e-12); n++; }
            }
        double avg = n ? std::exp(s / n) : 1;
        scale *= 0.18 / std::max(avg, 1e-12);
    }
    std::vector<unsigned char> raw;
    raw.reserve(size_t(img.height) * (img.width * 3 + 1));
    for (int y = 0; y < img.height; ++y) {
        raw.push_back(0);
        for (int x = 0; x < img.width; ++x) {
            XYZ c0 = img.at(x, y);
            XYZ c{(wb[0][0] * c0.x + wb[0][1] * c0.y + wb[0][2] * c0.z) * scale,
                  (wb[1][0] * c0.x + wb[1][1] * c0.y + wb[1][2] * c0.z) * scale,
                  (wb[2][0] * c0.x + wb[2][1] * c0.y + wb[2][2] * c0.z) * scale};
            double rgb[3];
            xyzToLinearSRGB(c, rgb);
            // Gentle highlight roll-off (display only).
            for (int k = 0; k < 3; ++k) {
                double v = std::max(0.0, rgb[k]);
                v = v / (1 + v / 4.0) * 1.25;
                raw.push_back((unsigned char)std::lround(255 * srgbEncode(v)));
            }
        }
    }
    std::vector<unsigned char> z;
#ifdef OWE_HAVE_ZLIB
    uLongf zlen = compressBound(uLong(raw.size()));
    z.resize(zlen);
    if (compress2(z.data(), &zlen, raw.data(), uLong(raw.size()), 9) != Z_OK) throw std::runtime_error("zlib failure");
    z.resize(zlen);
#else
    // zlib stream with stored (uncompressed) deflate blocks.
    z = {0x78, 0x01};
    size_t pos = 0;
    uint32_t a = 1, b = 0;
    for (unsigned char ch : raw) { a = (a + ch) % 65521; b = (b + a) % 65521; }
    do {
        size_t len = std::min<size_t>(65535, raw.size() - pos);
        bool last = pos + len == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(len & 0xFF); z.push_back(len >> 8);
        z.push_back(~len & 0xFF); z.push_back((~len >> 8) & 0xFF);
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
    } while (pos < raw.size());
    be32(z, (b << 16) | a);
#endif

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    out.write(reinterpret_cast<const char*>(sig), 8);
    std::vector<unsigned char> ihdr;
    be32(ihdr, uint32_t(img.width));
    be32(ihdr, uint32_t(img.height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    chunk(out, "IHDR", ihdr);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
}

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

std::string renderMetadataJSON(const Scene& scene, const ProgressiveRenderer& r, const RenderSettings& s) {
    const World& w = scene.world;
    std::ostringstream os;
    char hash[32];
    std::snprintf(hash, sizeof hash, "%016llx", (unsigned long long)fnv1a(scene.sourceText));
    Image img = r.resolve();
    os << "{\n";
    os << "  \"engine\": \"optical-world-engine " << OWE_VERSION << "\",\n";
    os << "  \"backend\": \"cpu-reference (IEEE-754 double)\",\n";
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
