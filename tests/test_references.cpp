// Quantitative forward-light references (the caustic programme's step A): light traced forward from
// the sun through a glass lens, a water surface and a concave mirror onto a receiving plane. Each
// measures integrated power on a surface sensor (particle tracing on every backend) and compares it
// with a power budget computed here independently of the transport code: the sun's irradiance,
// the clear aperture, textbook Fresnel transmittance at the actual angles and Beer–Lambert
// absorption, integrated over wavelength with the CIE ȳ weight the detector reports in.
#include <algorithm>
#include <cmath>

#include "check.hpp"
#include "owe/analysis/inspect.hpp"
#include "owe/backends/registry.hpp"
#include "owe/core/medium.hpp"
#include "owe/core/sampling.hpp"
#include "owe/loader/scene_loader.hpp"

using namespace owe;

namespace {

std::vector<std::string> availableBackends() {
    std::vector<std::string> r;
    for (const Backend* b : backends())
        if (b->available()) r.push_back(b->name());
    return r;
}

// Unpolarised Fresnel reflectance from index n1 into n2 at incidence cosine ci (textbook form).
double fresnel(double n1, double n2, double ci) {
    double si2 = 1 - ci * ci, st2 = sqr(n1 / n2) * si2;
    if (st2 >= 1) return 1;
    double ct = std::sqrt(1 - st2);
    double rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct), rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct);
    return 0.5 * (rs * rs + rp * rp);
}

// ∫ f(λ) ȳ(λ) dλ / ∫ ȳ(λ) dλ: what a detector's Y channel reports for a spectral quantity f.
template <class F>
double yWeighted(F f) {
    double num = 0, den = 0;
    for (double l = LambdaMin; l <= LambdaMax; l += 0.5) {
        num += f(l) * cieY(l);
        den += cieY(l);
    }
    return num / den;
}

// Total power on a sensor: Σ irradiance × pixel area (Y channel).
double sensorPower(const Image& img, double width, double height) {
    double s = 0;
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) s += img.at(x, y).y;
    return s * (width / img.width) * (height / img.height);
}

double peak(const Image& img) {
    double m = 0;
    for (size_t i = 1; i < img.xyz.size(); i += 3) m = std::max(m, img.xyz[i]);
    return m;
}

Image renderLight(const Scene& sc, const std::string& backend, int particlesPerPixel) {
    RenderSettings rs;
    rs.backend = backend;
    rs.integrator = "light";
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(particlesPerPixel);
    return R->resolve();
}

// Irradiance of a sun of unit (spectrally constant) radiance on a plane facing it: L·π·sin²θ_s.
double sunIrradiance(const Scene& sc) { return Pi * sqr(std::sin(sc.world.env.sunAngularRadius)); }

}  // namespace

TEST(reference_lens_focuses_sunlight_with_its_fresnel_losses) {
    // Overhead sun, an f = 100 mm biconvex N-BK7 lens of 40 mm clear aperture, and a sensor at its
    // focus. Every watt through the aperture reaches the sensor except what the two surfaces
    // reflect: P = E_sun · π·20² mm² · (1 − R₁)(1 − R₂), with R at each surface's local incidence.
    Scene sc = loadSceneFromString(R"(
        units = mm
        world { sun { elevation = 90deg  azimuth = 0deg  radiance = 1 } }
        body Lens { type = lens  medium = N-BK7  focal = 100  form = bi  diameter = 40  thickness = 8  rim = black
                    position = (0, 0, 100)  axis = (0, 0, 1) }
        sensor Focus { size = (24, 24)  resolution = (48, 48)  position = (0, 0, -1.5)  axis = (0, 0, 1)  up = (0, 1, 0) }
        render { detector = Focus }
    )");
    Medium glass;
    CHECK(catalogMedium("N-BK7", glass));
    Medium air;
    CHECK(catalogMedium("air", air));
    // Radii of the symmetric biconvex lens from the lens maker's equation with thickness (as built).
    const double t = 8, f = 100;
    const double transmitted = yWeighted([&](double l) {
        const double n = glass.n(l) / air.n(l);
        const double nd = glass.n(LambdaD) / air.n(LambdaD);
        // 1/f = (n−1)(2/R − (n−1)t/(n R²)) → R.
        const double a = 2 * (nd - 1), b = -sqr(nd - 1) * t / nd;
        const double R = (a * f + std::sqrt(sqr(a * f) + 4 * b * f)) / 2;
        // Average (1 − R₁)(1 − R₂) over the aperture area; the second surface is met at the angle
        // between the refracted ray and its normal (paraxial-thin estimate of the ray height there).
        double sum = 0, weight = 0;
        for (int i = 0; i < 400; ++i) {
            const double h = 20.0 * (i + 0.5) / 400;
            const double c1 = std::sqrt(1 - sqr(h / R));
            const double s2 = std::sqrt(1 - c1 * c1) / n, th1 = std::asin(std::sqrt(1 - c1 * c1)), th2 = std::asin(s2);
            const double tilt = th1 - th2;  // the ray's angle to the axis inside the glass
            const double h2 = h - t * std::tan(tilt);
            const double c3 = std::cos(std::asin(h2 / R) + tilt);  // incidence at the back surface
            sum += h * (1 - fresnel(1, n, c1)) * (1 - fresnel(n, 1, c3));
            weight += h;
        }
        return sum / weight;
    });
    const double expected = sunIrradiance(sc) * Pi * 20.0 * 20.0 * transmitted;
    for (const std::string& backend : availableBackends()) {
        Image img = renderLight(sc, backend, 400);
        const double power = sensorPower(img, 24, 24);
        std::fprintf(stderr, "    [%s] lens: power %.6g, expected %.6g (T = %.4f), ratio %.4f, concentration %.0f×\n",
                     backend.c_str(), power, expected, transmitted, power / expected, peak(img) / sunIrradiance(sc));
        CHECK_NEAR(power / expected, 1, 0.01);
        // The sun's image (0.93 mm across) plus spherical aberration, averaged over 0.5 mm pixels.
        CHECK(peak(img) > 150 * sunIrradiance(sc));
    }
}

TEST(reference_water_surface_transmits_the_fresnel_share) {
    // Sun at 40° elevation over a 10 cm layer of water (Daimon & Masumura index, Pope & Fry
    // absorption); a sensor under the layer receives E = E_sun·cos θᵢ · T(θᵢ)·T(θₜ)·exp(−σₐ d/cos θₜ).
    Scene sc = loadSceneFromString(R"(
        units = m
        world { sun { elevation = 40deg  azimuth = 0deg  radiance = 1 } }
        body Water { type = box  size = (1.2, 1.2, 0.1)  position = (0, 0, 0.3)  medium = water }
        sensor Floor { size = (0.2, 0.2)  resolution = (16, 16)  position = (0, 0, 0)  axis = (0, 0, 1)  up = (0, 1, 0) }
        render { detector = Floor }
    )");
    Medium water, air;
    CHECK(catalogMedium("water", water));
    CHECK(catalogMedium("air", air));
    const double ci = std::cos(radians(50));
    const double expected = sunIrradiance(sc) * yWeighted([&](double l) {
        const double n = water.n(l) / air.n(l);
        const double ct = std::sqrt(1 - (1 - ci * ci) / (n * n));
        return ci * (1 - fresnel(1, n, ci)) * (1 - fresnel(n, 1, ct)) * std::exp(-water.absorption.eval(l) * 0.1 / ct);
    });
    for (const std::string& backend : availableBackends()) {
        Image img = renderLight(sc, backend, 4000);
        const double E = sensorPower(img, 0.2, 0.2) / 0.04;
        std::fprintf(stderr, "    [%s] water: irradiance %.6g, expected %.6g, ratio %.4f\n", backend.c_str(), E, expected,
                     E / expected);
        CHECK_NEAR(E / expected, 1, 0.01);
    }
}

TEST(reference_concave_mirror_concentrates_sunlight) {
    // A spherical mirror (R = 400 mm, 100 mm aperture, reflectance 0.9) under an overhead sun; a
    // 10 mm sensor at its focus facing it, which shades 100 mm² of the mirror:
    // P = E_sun · (π·50² − 10·10) mm² · 0.9.
    Scene sc = loadSceneFromString(R"(
        units = mm
        world { sun { elevation = 90deg  azimuth = 0deg  radiance = 1 } }
        material coating { type = mirror  reflectance = 0.9 }
        body Mirror { type = mirror  surface = sphere(R = 400)  diameter = 100  position = (0, 0, 0)  axis = (0, 0, 1)
                      material = coating }
        sensor Focus { size = (10, 10)  resolution = (20, 20)  position = (0, 0, 199)  axis = (0, 0, -1)  up = (0, 1, 0) }
        render { detector = Focus }
    )");
    const double expected = sunIrradiance(sc) * (Pi * 50 * 50 - 100) * 0.9;
    for (const std::string& backend : availableBackends()) {
        Image img = renderLight(sc, backend, 2000);
        const double power = sensorPower(img, 10, 10);
        std::fprintf(stderr, "    [%s] mirror: power %.6g, expected %.6g, ratio %.4f, concentration %.0f×\n",
                     backend.c_str(), power, expected, power / expected, peak(img) / sunIrradiance(sc));
        CHECK_NEAR(power / expected, 1, 0.01);
        CHECK(peak(img) > 1000 * sunIrradiance(sc));
    }
}

namespace {

// A lens caustic on a diffuse floor (sun, f = 100 mm lens 100 mm above it), optionally seen through
// a 5 mm glass plate across the line of sight, or in a flat mirror (reflectance 0.9).
Scene causticScene(const std::string& extra, const Vec3& eye, const Vec3& target) {
    char text[2000];
    std::snprintf(text, sizeof text, R"(
        units = mm
        world { sun { elevation = 90deg  azimuth = 0deg  radiance = 1 } }
        material floor { type = diffuse  reflectance = 0.5 }
        body Floor { type = sheet  size = (300, 300)  position = (0, 0, 0)  material = floor }
        body Lens { type = lens  medium = N-BK7  focal = 100  form = bi  diameter = 40  thickness = 8  rim = black
                    position = (0, 0, 100)  axis = (0, 0, 1) }
        %s
        observer Eye { position = (%g, %g, %g)  look_at = (%g, %g, %g)  up = (0, 0, 1)  fov = 7deg  pupil = 2
                       focus = 335  resolution = (48, 48) }
        render { detector = Eye }
    )", extra.c_str(), eye.x, eye.y, eye.z, target.x, target.y, target.z);
    return loadSceneFromString(text);
}

double sppmSum(const Scene& sc, int iterations) {
    RenderSettings rs;
    rs.backend = "gpu";
    rs.integrator = "sppm";
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(iterations);
    return R->resolve().meanY();
}

}  // namespace

TEST(sppm_sees_a_caustic_through_a_glass_plate_and_in_a_mirror) {
    // The caustic seen through extra optics is the same caustic, attenuated by exactly what those
    // optics transmit: a plate at normal incidence passes (1 − R)²/(1 − R²) (incoherent multiple
    // reflections), a mirror its reflectance. Neither path can be found by path tracing (a point on
    // the floor lit through the lens by a 0.5° sun) nor by particles connected to the eye (the eye
    // looks through glass or a mirror): this is what photon mapping is for.
    const Backend* gpu = findBackend("gpu");
    if (!gpu || !gpu->available() || !gpu->supports("sppm")) return;
    const Vec3 eye{268.3, 0, 150}, spot{0, 0, 0};  // 335 mm from the spot, looking down at ~29°
    const Vec3 dir = normalize(spot - eye);
    const Vec3 platePos = eye + dir * 120;
    char plate[300];
    std::snprintf(plate, sizeof plate,
                  "body Plate { type = box  size = (60, 60, 5)  position = (%g, %g, %g)  axis = (%g, %g, %g)  medium = N-BK7 }",
                  platePos.x, platePos.y, platePos.z, dir.x, dir.y, dir.z);
    const double direct = sppmSum(causticScene("", eye, spot), 2048);
    const double throughPlate = sppmSum(causticScene(plate, eye, spot), 2048);
    Medium glass, air;
    CHECK(catalogMedium("N-BK7", glass));
    CHECK(catalogMedium("air", air));
    const double T = yWeighted([&](double l) {
        double R = fresnel(1, glass.n(l) / air.n(l), 1);
        return (1 - R) * (1 - R) / (1 - R * R);
    });
    std::fprintf(stderr, "    through the plate: %.4f of the direct view (Fresnel: %.4f)\n", throughPlate / direct, T);
    CHECK_NEAR(throughPlate / direct, T, 0.015);

    // The mirror: an eye beside the lens looks into a vertical mirror at x = 250 mm and sees the spot
    // as an eye at the mirror image of its position (x = 300) would see it directly, times 0.9.
    const Vec3 viaMirrorEye{200, 0, 150}, mirroredSpot{500, 0, 0}, virtualEye{300, 0, 150};
    const std::string mirror =
        "material coating { type = mirror  reflectance = 0.9 }\n"
        "body Mirror { type = flat_mirror  size = (80, 80)  position = (250, 0, 125)  axis = (-1, 0, 0)  up = (0, 0, 1)  "
        "material = coating }";
    const double inMirror = sppmSum(causticScene(mirror, viaMirrorEye, mirroredSpot), 2048);
    const double straight = sppmSum(causticScene("", virtualEye, spot), 2048);
    std::fprintf(stderr, "    in the mirror: %.4f of the direct view (reflectance 0.9)\n", inMirror / straight);
    CHECK_NEAR(inMirror / straight, 0.9, 0.015);
}

namespace {

// Snell refraction of unit direction d at a surface of unit normal n (either orientation), from
// index n1 into n2; false on total internal reflection.
bool refract2(const Vec3& d, Vec3 n, double n1, double n2, Vec3& out) {
    if (dot(d, n) > 0) n = -n;
    const double ci = -dot(d, n), eta = n1 / n2, k = 1 - eta * eta * (1 - ci * ci);
    if (k < 0) return false;
    out = normalize(d * eta + n * (eta * ci - std::sqrt(k)));
    return true;
}

// Where the ray o + t·d (t > 0) crosses the line through a and b (in the plane z = 0).
bool crossLine(const Vec3& o, const Vec3& d, const Vec3& a, const Vec3& b, Vec3& hit) {
    const Vec3 e = b - a, n{-e.y, e.x, 0};
    const double den = dot(d, n);
    if (std::abs(den) < 1e-15) return false;
    const double t = dot(a - o, n) / den;
    if (!(t > 1e-9)) return false;
    hit = o + d * t;
    const double s = dot(hit - a, e) / dot(e, e);
    return s >= 0 && s <= 1;
}

}  // namespace

TEST(reference_prism_deviates_each_wavelength_as_snell_predicts) {
    // A 60° N-SF11 prism (the spectroscope's) and a ray at 63.2° incidence on one face — close to
    // minimum deviation at the d line. The engine's transport (the walk the renderer uses, at single
    // wavelengths) must send each wavelength out where an independent two-face Snell trace of the
    // same triangle, with n(λ) of N-SF11 relative to air, puts it: to 1e-9 rad.
    Scene sc = loadSceneFromString(R"(
        units = mm
        body Prism { type = prism  apex = 60deg  side = 50  length = 40  medium = N-SF11  position = (0, 0, 0) }
        observer Eye { position = (0, -300, 0)  look_at = (0, 0, 0) }
    )");
    const double apex = radians(60), side = 0.05;  // the world is in metres
    const double hx = side * std::sin(apex / 2), hy = side * std::cos(apex / 2);
    const Vec3 A{0, 2 * hy / 3, 0}, B{-hx, -hy / 3, 0}, C{hx, -hy / 3, 0};  // as buildPrism lays it out
    const Vec3 d0{std::cos(radians(33.2)), std::sin(radians(33.2)), 0};
    const Vec3 origin = (A + B) * 0.5 - d0 * 0.2;
    Medium glass, air;
    CHECK(catalogMedium("N-SF11", glass));
    CHECK(catalogMedium("air", air));
    std::vector<double> deviation;
    for (double lambda : {404.7, 486.1, 546.1, 587.6, 656.3, 706.5}) {
        // Analytic: in through AB, out through AC.
        const double n = glass.n(lambda) / air.n(lambda);
        Vec3 p1, p2, d1, d2;
        CHECK(crossLine(origin, d0, A, B, p1));
        CHECK(refract2(d0, normalize(Vec3{-(B - A).y, (B - A).x, 0}), 1, n, d1));
        CHECK(crossLine(p1, d1, A, C, p2));
        CHECK(refract2(d1, normalize(Vec3{-(C - A).y, (C - A).x, 0}), n, 1, d2));
        // The engine: a single ray at this wavelength through the world (deterministic refraction).
        EmissionProbe e = probeEmission(sc, origin, d0, 0, 1, {lambda}, WalkMode::PrimaryTransmission, 1);
        CHECK(!e.records.empty() && e.records[0].v.size() >= 2);
        if (e.records.empty() || e.records[0].v.size() < 2) continue;
        const Vec3 out = e.records[0].v.back().dOut;
        const double measured = std::atan2(cross(d0, out).z, dot(d0, out));
        const double expected = std::atan2(cross(d0, d2).z, dot(d0, d2));
        CHECK_NEAR(measured, expected, 1e-9);
        deviation.push_back(-degrees(measured));
        std::fprintf(stderr, "    %.1f nm: n = %.5f, deviation %.4f° (analytic %.4f°)\n", lambda, n, -degrees(measured),
                     -degrees(expected));
    }
    // Normal dispersion: blue is deviated most; the d line near the classical 66.4° minimum.
    for (size_t i = 1; i < deviation.size(); ++i) CHECK(deviation[i] < deviation[i - 1]);
    CHECK_NEAR(deviation[3], 66.4, 0.2);
}

TEST(reference_ripple_caustics_follow_the_ray_map_jacobian) {
    // Sunlight through a rippled water surface h = a sin(kx) (λ_w = 2 cm, a = 0.5 mm: crests of 2 cm
    // radius, which focus ~8 cm deep) onto the flat bottom of an 8 cm layer, and out into a sensor
    // just below it. The irradiance E(u) on the sensor is the pushforward of the sunlight through the
    // surface-to-sensor ray map: E(u) = Σ E₀ T₁T₂ e^{−σₐℓ} / |∂u/∂x| over the map's preimages, with
    // the fold caustics where the Jacobian vanishes. Here the map is traced analytically (Snell and
    // Fresnel at the analytic surface and the flat bottom, the sun's 0.53° disc, CIE-weighted
    // wavelengths) and histogrammed; the renderer's particles must land the same.
    const Backend* gpu = findBackend("gpu");
    const std::string backend = gpu && gpu->available() ? "gpu" : "cpu";
    const double a = 0.0005, lambdaW = 0.02, k = 2 * Pi / lambdaW, D = 0.08, s = 0.001;
    char text[1200];
    std::snprintf(text, sizeof text, R"(
        units = m
        world { sun { elevation = 90deg  azimuth = 0deg  radiance = 1 } }
        body Pool { type = waves  size = (0.12, 0.12)  depth = %g  margin = 0.03  medium = water
                    waves = [wave(amplitude = %g, wavelength = %g, direction = 0deg, phase = 0deg)] }
        sensor Floor { size = (0.04, 0.02)  resolution = (160, 4)  position = (0, 0, %g)  axis = (0, 0, 1)  up = (0, 1, 0) }
        render { detector = Floor }
    )", D, a, lambdaW, -D - s);
    Scene sc = loadSceneFromString(text);
    Image img = renderLight(sc, backend, backend == "gpu" ? 60000 : 20000);
    std::vector<double> rendered(160, 0.0);
    for (int x = 0; x < 160; ++x)
        for (int y = 0; y < 4; ++y) rendered[size_t(x)] += img.at(x, y).y / 4;

    // The analytic map, Monte-Carlo integrated.
    Medium water, air;
    CHECK(catalogMedium("water", water));
    CHECK(catalogMedium("air", air));
    std::vector<double> cdf, lambdas;  // wavelengths drawn with the CIE ȳ weight
    for (double l = LambdaMin; l <= LambdaMax; l += 0.5) {
        lambdas.push_back(l);
        cdf.push_back((cdf.empty() ? 0 : cdf.back()) + cieY(l));
    }
    const double thetaS = sc.world.env.sunAngularRadius, cone = 2 * Pi * (1 - std::cos(thetaS));
    const double X = 0.03, Y = 0.012;  // the window of entry points (every preimage of the sensor)
    const int N = 4000000;
    std::vector<double> analytic(160, 0.0);
    Rng rng(99, 7);
    for (int i = 0; i < N; ++i) {
        const double x = (2 * rng.uniform() - 1) * X, y = (2 * rng.uniform() - 1) * Y;
        const size_t li = size_t(std::upper_bound(cdf.begin(), cdf.end(), rng.uniform() * cdf.back()) - cdf.begin());
        const double l = lambdas[std::min(li, lambdas.size() - 1)];
        const Vec3 local = sampleUniformCone(rng.uniform(), rng.uniform(), std::cos(thetaS));
        const Vec3 d0 = normalize(Vec3{local.x, local.y, -local.z});  // downward, within the sun's disc
        const double h = a * std::sin(k * x), n = water.n(l) / air.n(l);
        const Vec3 normal = normalize(Vec3{-a * k * std::cos(k * x), 0, 1});
        Vec3 d1, d2;
        if (!refract2(d0, normal, 1, n, d1) || !refract2(d1, {0, 0, 1}, n, 1, d2)) continue;
        const double c1 = std::abs(dot(d0, normal)), c2 = std::abs(d1.z);
        const double t1 = (-D - h) / d1.z;  // to the bottom
        const Vec3 q{x + d1.x * t1, y + d1.y * t1, -D};
        const double t2 = -s / d2.z;
        const double u = q.x + d2.x * t2, v = q.y + d2.y * t2;
        if (std::abs(v) >= 0.01 || std::abs(u) >= 0.02) continue;
        // Flux per unit horizontal area within the cone: L cosθ dω; the sample carries L cosθ Ω A/N.
        const double w = std::abs(d0.z) * cone * (2 * X) * (2 * Y) / N * (1 - fresnel(1, n, c1)) *
                         (1 - fresnel(n, 1, c2)) * std::exp(-water.absorption.eval(l) * t1);
        analytic[size_t((u + 0.02) / 0.04 * 160)] += w;
    }
    const double binArea = (0.04 / 160) * 0.02;
    double sumA = 0, sumR = 0, l1 = 0, peakA = 0, peakR = 0;
    int argA = 0, argR = 0;
    for (int b = 0; b < 160; ++b) {
        analytic[size_t(b)] /= binArea;
        sumA += analytic[size_t(b)];
        sumR += rendered[size_t(b)];
        l1 += std::abs(analytic[size_t(b)] - rendered[size_t(b)]);
        if (analytic[size_t(b)] > peakA) { peakA = analytic[size_t(b)]; argA = b; }
        if (rendered[size_t(b)] > peakR) { peakR = rendered[size_t(b)]; argR = b; }
    }
    const double E0 = sunIrradiance(sc);
    std::fprintf(stderr, "    [%s] mean irradiance %.5g (analytic %.5g), L1 difference %.2f%%, peak %.1f E₀ at bin %d "
                 "(analytic %.1f E₀ at %d)\n", backend.c_str(), sumR / 160, sumA / 160, 100 * l1 / sumA, peakR / E0, argR,
                 peakA / E0, argA);
    CHECK_NEAR(sumR / sumA, 1, 0.01);
    CHECK(l1 / sumA < 0.06);
    CHECK(std::abs(argA - argR) <= 1 || std::abs(std::abs(argA - argR) - 80) <= 1);  // the same fold (or its twin a wave away)
    CHECK(peakA > 3 * E0);  // the fold concentrates the sunlight
}
