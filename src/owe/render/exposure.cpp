#include "owe/render/exposure.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace owe {

double meterEV(const Image& img, const MeterOptions& o) {
    if (img.width <= 0 || img.height <= 0 || img.xyz.empty()) return 0;
    // Block means: a block's mean is an unbiased estimate of its luminance with far less noise
    // than any pixel, so percentiles taken over blocks are not moved by fireflies or by the empty
    // pixels of a one-sample preview.
    const int bs = std::max(1, std::min(img.width, img.height) / std::max(1, o.blocks));
    const int bw = (img.width + bs - 1) / bs, bh = (img.height + bs - 1) / bs;
    std::vector<double> sum(size_t(bw) * bh, 0.0);
    std::vector<int> count(size_t(bw) * bh, 0);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            double Y = img.at(x, y).y;
            if (!std::isfinite(Y)) continue;
            size_t b = size_t(y / bs) * bw + size_t(x / bs);
            sum[b] += std::max(0.0, Y);
            count[b]++;
        }
    struct Block {
        double logY, weight;
    };
    std::vector<Block> blocks;
    blocks.reserve(sum.size());
    double brightest = 0;
    for (size_t b = 0; b < sum.size(); ++b)
        if (count[b] > 0) brightest = std::max(brightest, sum[b] / count[b]);
    if (!(brightest > 0)) return 0;
    // Blocks without light sit 40 stops below the brightest: dark, but finite in the averages.
    const double floorY = brightest * std::exp2(-40.0);
    // Centre weighting falls smoothly from 1 at the centre to ~0.36 in the corners: there is no
    // window whose edge a bright object could cross.
    const double cx = 0.5 * img.width, cy = 0.5 * img.height, half = 0.5 * std::hypot(img.width, img.height);
    double totalWeight = 0;
    for (int by = 0; by < bh; ++by)
        for (int bx = 0; bx < bw; ++bx) {
            size_t b = size_t(by) * bw + bx;
            if (count[b] == 0) continue;
            double px = std::min(double(img.width), (bx + 0.5) * bs), py = std::min(double(img.height), (by + 0.5) * bs);
            double r = std::hypot(px - cx, py - cy) / half;
            double w = (0.35 + 0.65 * std::exp(-r * r / (2 * 0.35 * 0.35))) * count[b];
            blocks.push_back({std::log2(std::max(sum[b] / count[b], floorY)), w});
            totalWeight += w;
        }
    std::sort(blocks.begin(), blocks.end(), [](const Block& a, const Block& b) { return a.logY < b.logY; });
    double median = blocks.back().logY;
    for (double c = 0; const Block& b : blocks)
        if ((c += b.weight / totalWeight) >= 0.5) {
            median = b.logY;
            break;
        }
    // Key: the weighted mean of log luminance over the band [low, high] of the weighted distribution
    // (partial overlap at the band's edges counts in proportion), each value limited to
    // [median − keyBelow, median + keyAbove] so that a large lamp moves the key by a bounded amount.
    double c0 = 0, keySum = 0, keyWeight = 0;
    for (const Block& b : blocks) {
        double c1 = c0 + b.weight / totalWeight;
        double overlap = std::min(c1, o.highPercentile) - std::max(c0, o.lowPercentile);
        if (overlap > 0) {
            keySum += overlap * std::clamp(b.logY, median - o.keyBelow, median + o.keyAbove);
            keyWeight += overlap;
        }
        c0 = c1;
    }
    const double keyLog = keyWeight > 0 ? keySum / keyWeight : median;
    // Highlights: the share of the frame that the key's exposure pushes beyond white.
    // Lit surfaces (a candle-lit wall, a projected image, sunlit stone) lie a few stops above the
    // key; an emitter seen directly (a flame, a lamp, the sun) lies ten or more. Surfaces should keep
    // their detail; emitters may saturate, as in any photograph. So a pixel counts in full from 2 EV
    // above white, and less and less from `emitterStart` above white to nothing 4 EV further.
    // Pixels count only inside blocks whose mean is itself beyond white (noise cannot lift a block
    // mean there), so the share follows a bright object's exact coverage. A share beyond pullStart
    // lowers the exposure in proportion, by maxHighlightPull from fullPullShare on. All of it is
    // continuous in a bright object's position, size and brightness: nothing steps when a lamp
    // enters the view or crosses its centre.
    const double whiteLog = keyLog + std::log2(o.highlightValue / o.key);
    const double whiteY = std::exp2(whiteLog);
    double share = 0;
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            size_t b = size_t(y / bs) * bw + size_t(x / bs);
            if (!(sum[b] > whiteY * count[b])) continue;
            double Y = img.at(x, y).y;
            if (!(Y > whiteY) || !std::isfinite(Y)) continue;
            double above = std::log2(Y / whiteY);
            share += std::min(1.0, 0.5 * above) * std::clamp((o.emitterStart + 4 - above) / 4, 0.0, 1.0);
        }
    share /= double(img.width) * img.height;  // by area: a lit wall at the frame's edge counts in full
    const double pull = std::clamp((share - o.pullStart) / (o.fullPullShare - o.pullStart), 0.0, 1.0);
    return std::log2(o.key) - keyLog - o.maxHighlightPull * pull;
}

const char* exposureModeName(ExposureMode m) {
    switch (m) {
        case ExposureMode::Manual: return "manual";
        case ExposureMode::Locked: return "locked";
        case ExposureMode::Adaptive: return "adaptive";
    }
    return "?";
}

bool parseExposureMode(const std::string& s, ExposureMode& m) {
    if (s == "manual") m = ExposureMode::Manual;
    else if (s == "locked" || s == "lock") m = ExposureMode::Locked;
    else if (s == "adaptive" || s == "auto") m = ExposureMode::Adaptive;
    else return false;
    return true;
}

void ExposureControl::reset() {
    pendingLock_ = true;
    snapNext_ = true;
    haveReading_ = false;
    adapting_ = false;
}

void ExposureControl::setMode(ExposureMode m) {
    if (m == mode_) return;
    const double resolved = resolvedEV();
    if (m == ExposureMode::Manual) {
        compensation_ = resolved;
    } else if (mode_ == ExposureMode::Manual) {
        // From manual: keep today's brightness as the automatic gain, with no compensation.
        compensation_ = 0;
        current_ = target_ = resolved;
        pendingLock_ = false;  // Locked keeps it until asked to meter again
        snapNext_ = false;     // Adaptive moves from it gradually
        haveReading_ = false;
    }
    // Locked ↔ Adaptive keep the automatic gain and the compensation as they are.
    if (m == ExposureMode::Adaptive) {
        snapNext_ = false;
        target_ = current_;
    }
    adapting_ = false;
    mode_ = m;
}

void ExposureControl::update(const Image* fresh, long long spp, double dt) {
    dt = std::max(0.0, dt);
    if (fresh && fresh->width > 0) {
        const double m = meterEV(*fresh);
        lastMeter_ = m;
        if (mode_ == ExposureMode::Locked && pendingLock_) {
            current_ = m;
            if (spp >= lockSamples) pendingLock_ = false;
        } else if (mode_ == ExposureMode::Adaptive) {
            if (snapNext_) {
                current_ = target_ = m;
                if (spp >= lockSamples) snapNext_ = false;
            } else if (!haveReading_) {
                target_ = m;
            } else {
                // Successive readings are smoothed over UI time, so a single noisy frame moves the
                // target only in proportion to how long it is on screen.
                double a = 1 - std::exp(-(sinceReading_ + dt) / rates.meterSmoothing);
                target_ += a * (m - target_);
            }
            haveReading_ = true;
            sinceReading_ = -dt;  // dt is counted below
        }
    }
    sinceReading_ += dt;
    if (mode_ != ExposureMode::Adaptive || snapNext_ || !haveReading_ || dt <= 0) return;
    const double delta = target_ - current_;
    if (!adapting_ && std::abs(delta) > rates.deadBand) adapting_ = true;
    if (!adapting_) return;
    const double rate = delta > 0 ? rates.brighten : rates.darken;
    const double step = std::min(rate * dt, std::abs(delta) * (1 - std::exp(-dt / rates.timeConstant)));
    current_ += std::copysign(step, delta);
    if (std::abs(target_ - current_) <= rates.settle) adapting_ = false;
}

}  // namespace owe
