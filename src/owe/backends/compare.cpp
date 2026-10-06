#include "owe/backends/compare.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "owe/backends/registry.hpp"

namespace owe {

namespace {

// Per-run block means (X, Y, Z per block) and the image mean Y.
struct Runs {
    std::vector<std::vector<double>> blocks;  // [run][3 * block + channel]
    std::vector<double> image;                // [run]
    double seconds = 0;
    std::string backend;
};

Runs collect(const Scene& scene, int det, RenderSettings rs, uint64_t seedBase, int spp, int runs, int B) {
    Runs r;
    for (int k = 0; k < runs; ++k) {
        rs.seed = seedBase + uint64_t(k);
        auto R = makeRenderer(scene, det, rs);
        R->runPass(spp);
        r.seconds += R->seconds();
        r.backend = R->backend();
        Image img = R->resolve();
        const int bx = (img.width + B - 1) / B, by = (img.height + B - 1) / B;
        std::vector<double> v(size_t(3 * bx * by), 0.0);
        std::vector<int> n(size_t(bx * by), 0);
        for (int y = 0; y < img.height; ++y)
            for (int x = 0; x < img.width; ++x) {
                size_t b = size_t((y / B) * bx + x / B);
                XYZ c = img.at(x, y);
                v[3 * b] += c.x;
                v[3 * b + 1] += c.y;
                v[3 * b + 2] += c.z;
                n[b]++;
            }
        for (size_t b = 0; b < n.size(); ++b)
            for (int c = 0; c < 3; ++c) v[3 * b + c] /= std::max(1, n[b]);
        r.blocks.push_back(std::move(v));
        r.image.push_back(img.meanY());
    }
    return r;
}

void meanSe(const std::vector<double>& xs, double& m, double& se) {
    const double n = double(xs.size());
    m = 0;
    for (double x : xs) m += x / n;
    double q = 0;
    for (double x : xs) q += (x - m) * (x - m) / (n - 1);
    se = std::sqrt(q / n);
}

}  // namespace

ComparisonReport compareRenderers(const Scene& scene, int det, const RenderSettings& a, const RenderSettings& b,
                                  int spp, int runs, int B) {
    return compareMeasurements(scene, det, a, det, b, spp, runs, B);
}

ComparisonReport compareMeasurements(const Scene& scene, int det, const RenderSettings& a, int detB,
                                     const RenderSettings& b, int spp, int runs, int B) {
    if (runs < 3) throw std::runtime_error("compare: need at least 3 runs per renderer");
    const Detector &da = *scene.detectors.at(size_t(det)), &db = *scene.detectors.at(size_t(detB));
    if (da.width != db.width || da.height != db.height)
        throw std::runtime_error("compare: the two detectors must have the same resolution");
    Runs ra = collect(scene, det, a, 1000, spp, runs, B);
    Runs rb = collect(scene, detB, b, 2000, spp, runs, B);
    ComparisonReport rep;
    rep.runs = runs;
    rep.backendA = ra.backend;
    rep.backendB = rb.backend;
    rep.secondsA = ra.seconds;
    rep.secondsB = rb.seconds;
    meanSe(ra.image, rep.meanA, rep.seA);
    meanSe(rb.image, rep.meanB, rep.seB);
    double se = std::hypot(rep.seA, rep.seB);
    rep.zImage = se > 0 ? (rep.meanB - rep.meanA) / se : (rep.meanB == rep.meanA ? 0 : INFINITY);
    const size_t nv = ra.blocks[0].size();
    const int bxCount = (scene.detectors[size_t(det)]->width + B - 1) / B;
    rep.blocks = int(nv / 3);
    double chi2 = 0;
    for (size_t i = 0; i < nv; ++i) {
        std::vector<double> xa, xb;
        for (auto& v : ra.blocks) xa.push_back(v[i]);
        for (auto& v : rb.blocks) xb.push_back(v[i]);
        double ma, sa, mb, sb;
        meanSe(xa, ma, sa);
        meanSe(xb, mb, sb);
        double s = std::hypot(sa, sb);
        if (!(s > 0)) {
            if (ma != mb) { rep.maxAbsZ = INFINITY; rep.over3++; rep.over4++; rep.tests++; }
            continue;
        }
        double z = (mb - ma) / s;
        rep.worst.push_back({int(i / 3) % bxCount, int(i / 3) / bxCount, int(i % 3), ma, mb, z});
        rep.tests++;
        chi2 += z * z;
        rep.maxAbsZ = std::max(rep.maxAbsZ, std::abs(z));
        if (std::abs(z) > 3) rep.over3++;
        if (std::abs(z) > 4) rep.over4++;
    }
    rep.chi2PerTest = rep.tests ? chi2 / rep.tests : 0;
    std::sort(rep.worst.begin(), rep.worst.end(), [](auto& p, auto& q) { return std::abs(p.z) > std::abs(q.z); });
    if (rep.worst.size() > 8) rep.worst.resize(8);
    return rep;
}

bool ComparisonReport::consistent() const {
    return std::abs(zImage) < 4 && over4 <= std::max(1, tests / 100) && chi2PerTest < 2;
}

std::string ComparisonReport::text() const {
    std::ostringstream os;
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "A: %s\n   mean Y %.6g ± %.2g   (%.2f s)\n"
                  "B: %s\n   mean Y %.6g ± %.2g   (%.2f s)\n"
                  "image:  B/A = %.5f, z = %+.2f\n"
                  "blocks: %d × XYZ over %d runs: %d tests, Σz²/n = %.3f, max |z| = %.2f, |z|>3: %d, |z|>4: %d\n"
                  "verdict: %s\n",
                  backendA.c_str(), meanA, seA, secondsA, backendB.c_str(), meanB, seB, secondsB,
                  meanA != 0 ? meanB / meanA : 0.0, zImage, blocks, runs, tests, chi2PerTest, maxAbsZ, over3, over4,
                  consistent() ? "consistent (differences are Monte-Carlo noise)" : "INCONSISTENT");
    os << buf;
    return os.str();
}

}  // namespace owe
