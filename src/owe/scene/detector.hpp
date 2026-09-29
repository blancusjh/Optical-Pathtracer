// Detectors: anything that measures light. A surface sensor is matter in the
// world (a boundary with Detector optics); an ideal observer is a virtual eye
// (a pupil plus a perfect angular retina) used to look *through* instruments.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "owe/core/sampling.hpp"
#include "owe/core/spectrum.hpp"
#include "owe/scene/world.hpp"

namespace owe {

// Accumulates XYZ tristimulus values of the measured spectral quantity.
struct Film {
    int width = 0, height = 0;
    std::vector<double> xyz;      // 3 per pixel
    std::vector<double> samples;  // samples per pixel (path tracing)
    Film() = default;
    Film(int w, int h) : width(w), height(h), xyz(size_t(w) * h * 3, 0.0), samples(size_t(w) * h, 0.0) {}
    void add(int x, int y, double X, double Y, double Z) {
        size_t i = size_t(y) * width + x;
        xyz[3 * i] += X;
        xyz[3 * i + 1] += Y;
        xyz[3 * i + 2] += Z;
    }
    void merge(const Film& o) {
        for (size_t i = 0; i < xyz.size(); ++i) xyz[i] += o.xyz[i];
        for (size_t i = 0; i < samples.size(); ++i) samples[i] += o.samples[i];
    }
};

// Resolved image: per-pixel XYZ of the measured quantity.
struct Image {
    int width = 0, height = 0;
    std::vector<double> xyz;
    XYZ at(int x, int y) const {
        size_t i = size_t(y) * width + x;
        return {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    }
    double meanY() const;
};

class Detector {
public:
    enum class Quantity { Radiance, Irradiance };
    virtual ~Detector() = default;

    std::string name;
    int width = 256, height = 256;
    uint32_t region = 0;  // region in which camera paths start
    // Pixel response: the pixel's square convolved with a Gaussian of pixelSigma pixels (0: the
    // square alone). Camera samples and particle splats are displaced by the same Gaussian, so the
    // path, light and hybrid estimates measure one filtered image. With the square alone, a source
    // smaller than a pixel images as a pixel-aligned square; the Gaussian images it as a round spot
    // centred where it falls.
    double pixelSigma = 0;
    // A displacement (in pixels) drawn from the pixel response's Gaussian; none when pixelSigma is 0.
    Vec2 filterOffset(Rng& rng) const {
        if (!(pixelSigma > 0)) return {};
        Draw2 u(rng);
        Vec2 g = sampleStandardNormal2(u.u1, u.u2);
        return {g.x * pixelSigma, g.y * pixelSigma};
    }

    virtual Quantity quantity() const = 0;
    virtual std::string describe() const = 0;
    virtual void prepare(const World& world) = 0;
    // Generates a measurement ray for continuous raster position (px, py).
    // weight is the importance-to-pdf ratio; false means a zero-valued sample.
    virtual bool generate(double px, double py, Rng& rng, Ray& ray, double& weight) const = 0;

    // Light-tracing connection for virtual observers: from world point x, choose a
    // pupil point, report the pixel and the factor W such that the pixel receives
    // β·f·|cos_x|·W·Tr / N. Returns false if x cannot be seen.
    virtual bool connect(const Vec3& x, Rng& rng, Vec3& pupilPoint, int& px, int& py, double& factor) const {
        (void)x; (void)rng; (void)pupilPoint; (void)px; (void)py; (void)factor;
        return false;
    }
    // Surface sensors: pixel receiving a particle that hit `hit`.
    virtual bool pixelOfHit(const SurfaceHit& hit, Rng& rng, int& px, int& py, double& pixelArea) const {
        (void)hit; (void)rng; (void)px; (void)py; (void)pixelArea;
        return false;
    }
    virtual int boundary() const { return -1; }
    virtual bool isVirtual() const { return false; }
};

// A virtual eye: circular pupil (radius ≥ 0) and a perfect retina mapping
// directions (focus at infinity) or a focal plane (finite focus) to pixels.
// It measures radiance averaged over the pupil and the pixel's angular footprint
// (by default a soft one: see pixelSigma).
class IdealObserver : public Detector {
public:
    static constexpr double kDefaultPixelSigma = 0.4;  // with the square: ~0.5 px, the usual Gaussian reconstruction
    IdealObserver() { pixelSigma = kDefaultPixelSigma; }

    Vec3 position{0, 0, 0}, lookAt{0, 1, 0}, up{0, 0, 1};
    double fovY = radians(40);
    double pupilRadius = 0;
    double focusDistance = Inf;

    // Pupil sampling hint, chosen by prepare(): an eye looking into a telescope whose exit pupil is
    // much smaller than its own pupil receives the instrument's light only through that small
    // disk. Most pupil samples are then drawn there, and weighted by the mixture density, so the
    // estimate is unchanged and only its noise falls.
    struct PupilGuide {
        Vec3 center;        // the instrument's exit pupil (world)
        double radius = 0;  // sampled disk radius (the exit pupil's, with a margin); 0: no guide
        double share = 0;   // fraction of pupil samples drawn from the guide
        std::string instrument;
    };
    PupilGuide guide;
    // Where rays along `dir` through the guide's centre cross the eye's pupil plane; false if that
    // disk misses the eye's pupil.
    bool guideCenter(const Vec3& dir, Vec3& center) const;

    Quantity quantity() const override { return Quantity::Radiance; }
    std::string describe() const override;
    void prepare(const World& world) override;
    bool generate(double px, double py, Rng& rng, Ray& ray, double& weight) const override;
    bool connect(const Vec3& x, Rng& rng, Vec3& pupilPoint, int& px, int& py, double& factor) const override;
    bool isVirtual() const override { return true; }

    Vec3 forward() const { return fwd_; }
    Vec3 right() const { return right_; }
    Vec3 upVec() const { return up_; }
    double tanX() const { return tanX_; }
    double tanY() const { return tanY_; }

private:
    Vec3 fwd_, right_, up_;
    double tanX_ = 1, tanY_ = 1;
};

// A rectangular sensor that is part of the world (a boundary with Detector optics).
// It measures irradiance on its front face. `aim` is an importance-sampling hint:
// a disk through which all light reaching the sensor must pass (e.g. the rear
// opening of a camera). It changes variance, never the expected value, provided
// the enclosure is closed elsewhere.
class SurfaceSensor : public Detector {
public:
    uint32_t boundaryIndex = 0;
    double halfX = 0.01, halfY = 0.01;
    // Readout convention: which physical corner is pixel (0,0). Relabelling pixels
    // never mirrors the optical image; it only orients the stored picture.
    bool flipX = false, flipY = false;
    bool hasAim = false;
    Vec3 aimCenter, aimNormal{0, 0, 1};
    double aimRadius = 0;
    double aimShare = 1;  // <1 retains cosine-hemisphere support for indirect light on screens
    // Optional focus disk (parallel to the aim disk) sampled with probability focusShare: a
    // camera aims most samples at its exit pupil, where image-forming light comes from, and the
    // rest at the whole rear opening, which also carries ghosts and veiling glare. Directions are
    // weighted by the mixture density, so the estimate stays unbiased.
    Vec3 focusCenter;
    double focusRadius = 0, focusShare = 0;

    Quantity quantity() const override { return Quantity::Irradiance; }
    std::string describe() const override;
    void prepare(const World& world) override;
    bool generate(double px, double py, Rng& rng, Ray& ray, double& weight) const override;
    bool pixelOfHit(const SurfaceHit& hit, Rng& rng, int& px, int& py, double& pixelArea) const override;
    int boundary() const override { return int(boundaryIndex); }
    const Transform& toWorld() const { return toWorld_; }
    Vec3 normal() const { return normal_; }

private:
    Transform toWorld_, toLocal_;
    Vec3 normal_;
};

}  // namespace owe
