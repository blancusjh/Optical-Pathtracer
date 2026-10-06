#include "owe/backends/cpu/cpu_backend.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "owe/transport/transport.hpp"

namespace owe {

namespace {

// Camera paths are scheduled in 16×16 tiles over the threads; each pixel draws from its own
// stream (seed, pixel, pass), so the result does not depend on the scheduling. Particles are
// traced in fixed lanes, merged in lane order.
class CpuRenderer final : public Renderer {
public:
    CpuRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings);

    void runPass(int samplesPerPixel) override;
    Image resolve() const override;
    bool resetObserver() override;
    int passes() const override { return passes_; }
    long long samplesPerPixel() const override { return totalSpp_; }
    double seconds() const override { return seconds_; }
    const TransportStats& stats() const override { return stats_; }
    const Detector& detector() const override { return *det_; }
    const RenderSettings& settings() const override { return settings_; }
    std::string backend() const override { return "cpu-reference (IEEE-754 double)"; }

private:
    void passPath(int spp);
    void passLight(int particlesPerPixel);

    const Scene& scene_;
    const Detector* det_;
    RenderSettings settings_;
    Tracer tracer_;
    Film film_;       // camera-path estimates (per-pixel sample counts)
    Film lightFilm_;  // particle splats (normalised by the particle count)
    double particles_ = 0;
    int passes_ = 0;
    long long totalSpp_ = 0;
    double seconds_ = 0;
    TransportStats stats_;
    int threads_;
};

CpuRenderer::CpuRenderer(const Scene& scene, int detectorIndex, const RenderSettings& settings)
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

bool CpuRenderer::resetObserver() {
    if (!dynamic_cast<const IdealObserver*>(det_)) return false;
    film_ = Film(det_->width, det_->height);
    lightFilm_ = Film(det_->width, det_->height);
    particles_ = seconds_ = 0;
    passes_ = 0;
    totalSpp_ = 0;
    stats_ = {};
    return true;
}

void CpuRenderer::runPass(int spp) {
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

void CpuRenderer::passPath(int spp) {
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
                        Vec2 f = det_->filterOffset(rng);
                        if (!det_->generate(px + f.x, py + f.y, rng, ray, weight) || weight <= 0) continue;
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

void CpuRenderer::passLight(int ppp) {
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

Image CpuRenderer::resolve() const {
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

// "model name" from /proc/cpuinfo where there is one.
std::string hostCpuName() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("model name", 0) == 0 && line.find(':') != std::string::npos) {
            std::string name = line.substr(line.find(':') + 1);
            name.erase(0, name.find_first_not_of(" \t"));
            if (!name.empty()) return name;
        }
    return "host CPU";
}

class CpuBackend final : public Backend {
public:
    std::string name() const override { return "cpu"; }
    std::string summary() const override {
        return "the reference transport on host threads; IEEE-754 double, bitwise reproducible";
    }
    bool available(std::string*) const override { return true; }
    std::vector<DeviceInfo> devices() const override {
        DeviceInfo d;
        d.name = hostCpuName();
        d.type = "cpu";
        d.driver = std::to_string(std::max(1u, std::thread::hardware_concurrency())) + " hardware threads";
        return {d};
    }
    std::vector<std::string> integrators() const override { return {"path", "light", "hybrid"}; }
    std::unique_ptr<Renderer> createRenderer(const Scene& scene, int detectorIndex,
                                             const RenderSettings& settings) const override {
        for (const SurfaceOptics& o : scene.world.optics())
            if (o.type == SurfaceType::StainedGlass)
                throw std::runtime_error("material '" + o.name + "' is stained glass, which only the GPU backend renders (--backend gpu)");
        return std::make_unique<CpuRenderer>(scene, detectorIndex, settings);
    }
};

}  // namespace

const Backend& cpuBackend() {
    static const CpuBackend backend;
    return backend;
}

}  // namespace owe
