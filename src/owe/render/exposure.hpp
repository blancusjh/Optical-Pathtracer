// Display exposure: the gain between the raw estimate and the display. It never touches the raw
// accumulation, the sources or any measurement; it only decides how bright the picture is shown.
//
// Three modes, as on a camera:
//   Manual    a fixed exposure chosen by the user;
//   Locked    meter once (a new view), then keep that exposure while navigating;
//   Adaptive  follow the meter gradually, in UI time, with bounded rates and a dead band.
//
// The meter is robust by construction. It averages block means (not pixels, so Monte-Carlo noise
// and fireflies are averaged away before any statistic is taken) in log luminance over a band of
// weighted percentiles, each limited to a range about the median, with a smooth centre weighting
// and no hard window. Lit surfaces beyond white pull the exposure down in proportion to the share of
// the frame they cover, by a bounded amount; emitters seen directly (ten or more stops above the
// key) do not: a lamp or the sun entering the view saturates, as it should, instead of darkening
// the whole room. Every statistic is continuous in what the image shows: nothing steps when a
// lamp crosses the view.
#pragma once

#include <string>

#include "owe/scene/detector.hpp"

namespace owe {

struct MeterOptions {
    double lowPercentile = 0.40;         // the band of block luminances averaged (in log) for the key
    double highPercentile = 0.97;
    double keyAbove = 4, keyBelow = 16;  // EV: values limited to this range about the median
    double key = 0.18;                   // the key maps to middle grey
    double highlightValue = 0.9;         // "white": the key's exposure puts it here. A share of the
    double pullStart = 0.01;             // frame beyond it (above pullStart) lowers the exposure in
    double fullPullShare = 0.06;         // proportion, up to maxHighlightPull EV from fullPullShare
    double maxHighlightPull = 4.0;       // on; what lies more than emitterStart EV above white (an
    double emitterStart = 6.0;           // emitter seen directly) counts less, nothing from +4 EV more
    int blocks = 48;                     // blocks along the image's shorter side
};

// Gain in EV (a factor 2^EV on the raw values) that the meter asks for; 0 for an image without light.
double meterEV(const Image& img, const MeterOptions& options = {});

enum class ExposureMode { Manual, Locked, Adaptive };
const char* exposureModeName(ExposureMode m);
// "manual", "locked", "adaptive" (also "auto" for adaptive); false if unknown.
bool parseExposureMode(const std::string& s, ExposureMode& m);

struct AdaptationRates {
    double brighten = 1.5;      // EV/s at most when the scene gets darker (the display brightens)
    double darken = 2.5;        // EV/s at most when the scene gets brighter
    double timeConstant = 0.5;  // s: exponential approach inside the rate limits
    double deadBand = 0.35;     // EV: differences below this do not start an adaptation ...
    double settle = 0.05;       // ... and one ends within this of its target
    double meterSmoothing = 0.25;  // s: time constant smoothing successive meter readings
};

// The exposure state of one display. Time is UI time (seconds between updates), never passes.
class ExposureControl {
public:
    ExposureControl() = default;
    explicit ExposureControl(ExposureMode mode, double compensation = 0) : mode_(mode), compensation_(compensation) {}

    // A new view: Locked meters it once, Adaptive snaps to its first reading. Manual is unchanged.
    void reset();
    // Meter the next image again (Locked), or snap to it (Adaptive).
    void remeter() { reset(); }
    // Switch mode keeping the displayed brightness (the resolved EV) unchanged.
    void setMode(ExposureMode m);
    ExposureMode mode() const { return mode_; }

    // Manual: the exposure. Locked, Adaptive: the compensation added to the meter.
    double compensation() const { return compensation_; }
    void setCompensation(double ev) { compensation_ = ev; }

    AdaptationRates rates;
    // An image may lock the exposure once it has at least this many samples per pixel; earlier
    // (noisier) images meter provisionally.
    long long lockSamples = 4;

    // Advances by dt seconds of UI time. `fresh` is a newly displayed image (or null) with its
    // samples per pixel.
    void update(const Image* fresh, long long spp, double dt);

    // The gain applied to the raw estimate, in EV: what the display and every export use.
    double resolvedEV() const { return mode_ == ExposureMode::Manual ? compensation_ : compensation_ + current_; }
    // The automatic part (Locked, Adaptive), and what the meter reads for the latest image.
    double automaticEV() const { return mode_ == ExposureMode::Manual ? 0 : current_; }
    double lastMeterEV() const { return lastMeter_; }
    bool locked() const { return mode_ == ExposureMode::Locked && !pendingLock_; }
    bool adapting() const { return adapting_; }

private:
    ExposureMode mode_ = ExposureMode::Locked;
    double compensation_ = 0;
    double current_ = 0;       // automatic gain now shown (EV)
    double target_ = 0;        // smoothed meter reading (Adaptive)
    double lastMeter_ = 0;
    double sinceReading_ = 0;  // UI time since the last meter reading (s)
    bool haveReading_ = false;
    bool pendingLock_ = true;  // Locked: not metered yet for this view
    bool snapNext_ = true;     // Adaptive: a new view takes its first readings directly
    bool adapting_ = false;
};

}  // namespace owe
