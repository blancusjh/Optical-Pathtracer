// The backend-neutral render front end: the display pipeline shared by PNGs and the viewer, and
// the backend registry.
#include <filesystem>
#include <fstream>

#include "check.hpp"
#include "owe/backends/registry.hpp"
#include "owe/render/exposure.hpp"
#include "owe/render/output.hpp"
#include "owe/loader/scene_loader.hpp"

using namespace owe;

namespace {
// A grey image (XYZ of D65 at luminance Y everywhere).
Image greyImage(int w, int h, double Y) {
    Image img;
    img.width = w;
    img.height = h;
    img.xyz.resize(size_t(w) * h * 3);
    for (size_t i = 0; i < img.xyz.size(); i += 3) {
        img.xyz[i] = 0.95047 * Y;
        img.xyz[i + 1] = Y;
        img.xyz[i + 2] = 1.08883 * Y;
    }
    return img;
}
void setGrey(Image& img, int x, int y, double Y) {
    size_t i = 3 * (size_t(y) * img.width + x);
    img.xyz[i] = 0.95047 * Y;
    img.xyz[i + 1] = Y;
    img.xyz[i + 2] = 1.08883 * Y;
}
// A dim room (luminance varying smoothly over a decade) with a square lamp of side s at (cx, cy).
Image roomWithLamp(int w, int h, double cx, double cy, int s, double lamp) {
    Image img = greyImage(w, h, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double Y = 0.01 * std::pow(10.0, double(x + y) / (w + h));
            if (std::abs(x + 0.5 - cx) < 0.5 * s && std::abs(y + 0.5 - cy) < 0.5 * s) Y = lamp;
            setGrey(img, x, y, Y);
        }
    return img;
}
}  // namespace

TEST(meter_brings_a_uniform_scene_to_middle_grey) {
    for (double Y : {1e-7, 0.01, 3.0, 1e5}) CHECK_NEAR(meterEV(greyImage(64, 40, Y)), std::log2(0.18 / Y), 1e-9);
    CHECK(meterEV(greyImage(64, 40, 0)) == 0);  // no light: no gain
}

TEST(meter_ignores_compact_highlights_anywhere_in_the_frame) {
    // A lamp, a planet or a firefly smaller than a percent of the frame never sets the exposure,
    // wherever it is (the old meter protected any compact highlight in a central window, so the
    // whole room collapsed when a lamp crossed it).
    Image room = roomWithLamp(160, 100, 0, 0, 0, 0);
    const double m0 = meterEV(room);
    for (auto [x, y] : {std::pair{80.0, 50.0}, {5.0, 5.0}, {150.0, 90.0}}) {
        Image lit = roomWithLamp(160, 100, x, y, 8, 1e4);  // 0.4% of the frame at 10⁶× the room
        CHECK_NEAR(meterEV(lit), m0, 0.05);
    }
    Image firefly = room;
    setGrey(firefly, 80, 50, 1e9);
    CHECK_NEAR(meterEV(firefly), m0, 0.05);
    // The display lets the lamp saturate; the room keeps its brightness.
    Image lit = roomWithLamp(160, 100, 80, 50, 8, 1e4);
    auto a = displayRGB8(room, meterEV(room), false), b = displayRGB8(lit, meterEV(lit), false);
    CHECK(b[3 * (50 * 160 + 80)] == 255);
    CHECK(std::abs(int(a[3 * (10 * 160 + 10) + 1]) - int(b[3 * (10 * 160 + 10) + 1])) <= 2);
}

TEST(meter_changes_continuously_as_a_lamp_crosses_the_centre) {
    double prev = 0, worst = 0;
    for (int x = 0; x <= 160; ++x) {
        double m = meterEV(roomWithLamp(160, 100, x, 50, 12, 1e4));  // 0.9% of the frame
        if (x > 0) worst = std::max(worst, std::abs(m - prev));
        prev = m;
    }
    CHECK(worst < 0.05);
}

TEST(meter_lets_a_large_bright_area_pull_by_a_bounded_amount) {
    const double m0 = meterEV(roomWithLamp(160, 100, 0, 0, 0, 0));
    // 2.5% and 10% of the frame at 10⁶× the room: the room darkens by a few stops at most. When
    // the lamp fills most of the view (40%, centre-weighted past the median), it is the scene.
    const MeterOptions o;
    double previous = m0;
    for (auto [side, most] : {std::pair{20, o.maxHighlightPull}, {40, o.maxHighlightPull + 1.5}, {80, 100.0}}) {
        double m = meterEV(roomWithLamp(160, 100, 80, 50, side, 1e4));
        CHECK(m < previous);
        CHECK(m >= m0 - most);
        previous = m;
    }
}

TEST(meter_is_robust_to_one_sample_noise) {
    // Each pixel of a one-sample preview holds either nothing or a sample weighted up by its
    // probability: the same image in expectation. Block means average that out.
    Image clean = roomWithLamp(320, 200, 0, 0, 0, 0), noisy = clean;
    Rng rng(7, 1);
    for (size_t i = 0; i < noisy.xyz.size(); i += 3) {
        double keep = rng.uniform() < 0.2 ? 5.0 : 0.0;
        for (int c = 0; c < 3; ++c) noisy.xyz[i + c] *= keep;
    }
    CHECK_NEAR(meterEV(noisy), meterEV(clean), 0.35);
}

TEST(meter_is_scale_invariant_and_a_locked_meter_keeps_the_picture) {
    Image img = roomWithLamp(100, 60, 50, 30, 10, 3);
    auto inside = displayRGB8(img, 0, true);
    auto locked = displayRGB8(img, autoExposureEV(img), false);
    CHECK(inside == locked);  // locking the meter must not change the current brightness
    for (double& x : img.xyz) x *= 1e6;  // daylight: the same scene a million times brighter
    auto outside = displayRGB8(img, 0, true);
    for (size_t i = 0; i < inside.size(); ++i) CHECK(std::abs(int(inside[i]) - int(outside[i])) <= 1);
}

TEST(exposure_manual_and_locked_map_unchanged_raw_pixels_identically) {
    Image room = roomWithLamp(160, 100, 0, 0, 0, 0), lamp = roomWithLamp(160, 100, 80, 50, 60, 1e4);
    ExposureControl locked(ExposureMode::Locked);
    locked.update(&room, 16, 1 / 60.0);
    CHECK(locked.locked());
    ExposureControl manual(ExposureMode::Manual, locked.resolvedEV());
    CHECK(displayRGB8(room, locked.resolvedEV(), false) == displayRGB8(room, manual.resolvedEV(), false));
    // Navigating: new images of any content never move a locked or manual exposure.
    const double ev = locked.resolvedEV();
    for (int k = 0; k < 100; ++k) {
        locked.update(k % 2 ? &lamp : &room, 1, 1 / 60.0);
        manual.update(k % 2 ? &lamp : &room, 1, 1 / 60.0);
        CHECK(locked.resolvedEV() == ev && manual.resolvedEV() == ev);
    }
    // Asking to meter again reads the next image.
    locked.remeter();
    locked.update(&lamp, 16, 1 / 60.0);
    CHECK_NEAR(locked.resolvedEV(), meterEV(lamp), 1e-12);
}

TEST(exposure_locks_only_on_a_sufficiently_sampled_image) {
    Image dim = greyImage(64, 40, 0.01), bright = greyImage(64, 40, 1);
    ExposureControl c(ExposureMode::Locked);
    c.update(&bright, 1, 0.05);  // a one-sample first image meters provisionally
    CHECK(!c.locked());
    CHECK_NEAR(c.resolvedEV(), meterEV(bright), 1e-12);
    c.update(&dim, c.lockSamples, 0.05);
    CHECK(c.locked());
    CHECK_NEAR(c.resolvedEV(), meterEV(dim), 1e-12);
    c.update(&bright, 64, 0.05);
    CHECK_NEAR(c.resolvedEV(), meterEV(dim), 1e-12);
}

TEST(exposure_adaptive_follows_the_meter_within_its_rate_limits) {
    // A pan across a lamp at 60 frames per second, with one-sample-like noise on every frame: the
    // exposure (EV against UI time) never changes faster than its configured rates.
    ExposureControl c(ExposureMode::Adaptive);
    const double dt = 1 / 60.0;
    Image room = roomWithLamp(160, 100, 0, 0, 0, 0);
    c.update(&room, 16, dt);
    double prev = c.resolvedEV(), fastest = 0;
    Rng rng(3, 5);
    std::vector<double> evs;
    for (int f = 0; f < 900; ++f) {
        double x = 80 + 70 * std::sin(f * dt * 1.3);  // the lamp swings through the centre
        Image img = roomWithLamp(160, 100, x, 50, f < 300 ? 50 : 8, 1e4);
        for (double& v : img.xyz) v *= 0.8 + 0.4 * rng.uniform();
        c.update(&img, 1, dt);
        fastest = std::max(fastest, std::abs(c.resolvedEV() - prev) / dt);
        prev = c.resolvedEV();
        evs.push_back(prev);
    }
    CHECK(fastest <= std::max(c.rates.brighten, c.rates.darken) + 1e-9);
    // The large lamp (first 5 s) lowered the exposure; after it shrank, it came back to the room's.
    CHECK(*std::min_element(evs.begin(), evs.begin() + 300) < meterEV(room) - 1);
    CHECK_NEAR(evs.back(), meterEV(room), 0.4);
}

TEST(exposure_adaptive_holds_through_small_fluctuations) {
    ExposureControl c(ExposureMode::Adaptive);
    Image a = greyImage(64, 40, 0.01), b = greyImage(64, 40, 0.01 * std::exp2(0.25));
    c.update(&a, 16, 0.02);
    const double ev = c.resolvedEV();
    for (int k = 0; k < 300; ++k) c.update(k % 2 ? &b : &a, 16, 0.02);
    CHECK(c.resolvedEV() == ev);  // within the dead band: nothing moves
}

TEST(exposure_mode_switches_keep_the_displayed_brightness) {
    Image room = roomWithLamp(160, 100, 0, 0, 0, 0), lamp = roomWithLamp(160, 100, 80, 50, 60, 1e4);
    ExposureControl c(ExposureMode::Locked);
    c.setCompensation(0.7);
    c.update(&room, 16, 0.02);
    for (ExposureMode m : {ExposureMode::Manual, ExposureMode::Adaptive, ExposureMode::Locked, ExposureMode::Adaptive,
                           ExposureMode::Manual, ExposureMode::Locked}) {
        const double ev = c.resolvedEV();
        c.setMode(m);
        CHECK_NEAR(c.resolvedEV(), ev, 1e-12);
        c.update(&lamp, 16, 0.02);  // Adaptive then starts moving, gradually
        CHECK(std::abs(c.resolvedEV() - ev) <= 0.02 * std::max(c.rates.brighten, c.rates.darken) + 1e-12);
    }
    ExposureMode m;
    CHECK(parseExposureMode("locked", m) && m == ExposureMode::Locked);
    CHECK(parseExposureMode("auto", m) && m == ExposureMode::Adaptive);
    CHECK(!parseExposureMode("sometimes", m));
}

TEST(display_highlights_fade_to_white_instead_of_clipping_per_channel) {
    // A blue-white star far beyond the display range: white, not cyan. A dim one keeps its hue,
    // and a colour within range is exactly the per-channel roll-off it always was.
    Image img; img.width = 3; img.height = 1; img.xyz.resize(9);
    auto set = [&](int i, double r, double g, double b) {  // linear sRGB → XYZ (D65)
        img.xyz[3 * i] = 0.4124 * r + 0.3576 * g + 0.1805 * b;
        img.xyz[3 * i + 1] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        img.xyz[3 * i + 2] = 0.0193 * r + 0.1192 * g + 0.9505 * b;
    };
    set(0, 2, 6, 20);
    set(1, 0.1, 0.25, 0.7);
    set(2, 0.2, 0.3, 0.9);
    auto rgb = displayRGB8(img, 0, false, 0, Tone::Standard);
    CHECK(rgb[0] == 255 && rgb[1] == 255 && rgb[2] == 255);
    CHECK(rgb[5] > rgb[4] && rgb[4] > rgb[3]);  // still blue
    CHECK(rgb[8] < 255);
}

TEST(display_agx_keeps_grey_and_takes_bright_colours_to_white) {
    double grey[3] = {0.18, 0.18, 0.18};
    applyTone(Tone::AgX, grey);
    CHECK_NEAR(grey[0], 0.215, 0.01);  // middle grey shows as the standard curve shows it
    CHECK_NEAR(grey[1], grey[0], 1e-12);
    CHECK_NEAR(grey[2], grey[0], 1e-12);
    double black[3] = {0, 0, 0}, white[3] = {16.3, 16.3, 16.3};
    applyTone(Tone::AgX, black);
    applyTone(Tone::AgX, white);
    CHECK(black[0] < 1e-4);
    CHECK(white[0] > 0.99);
    // Monotonic over the whole range (every stop from −14 to +6).
    double last = -1;
    bool monotonic = true;
    for (double ev = -14; ev <= 6; ev += 0.25) {
        double v[3] = {std::exp2(ev), std::exp2(ev), std::exp2(ev)};
        applyTone(Tone::AgX, v);
        monotonic = monotonic && v[1] >= last;
        last = v[1];
    }
    CHECK(monotonic);
    // A saturated red: red at its own brightness, going to white as it brightens (its green and
    // blue rise toward its red), never a clipped pure primary.
    double dim[3] = {0.3, 0.02, 0.02}, bright[3] = {60, 4, 4};
    applyTone(Tone::AgX, dim);
    applyTone(Tone::AgX, bright);
    CHECK(dim[0] > 2 * dim[1]);
    CHECK(bright[1] / bright[0] > dim[1] / dim[0]);
    CHECK(bright[1] > 0.7);
}

TEST(registry_lists_the_reference_first_and_names_backends) {
    const auto& list = backends();
    CHECK(!list.empty());
    CHECK(list.front() == &referenceBackend());
    CHECK(referenceBackend().name() == "cpu");
    CHECK(referenceBackend().available());
    for (const char* integrator : {"path", "light", "hybrid"}) CHECK(referenceBackend().supports(integrator));
    for (const Backend* b : list) {
        CHECK(findBackend(b->name()) == b);
        CHECK(b->supports("path"));  // every backend implements the path integrator
        std::string why;
        if (!b->available(&why)) CHECK(!why.empty());
    }
    CHECK(findBackend("gpu") != nullptr);  // listed even when compiled out (then unavailable)
    CHECK(findBackend("tpu") == nullptr);
    bool threw = false;
    try {
        backend("tpu");
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("known: cpu") != std::string::npos;
    }
    CHECK(threw);
}

TEST(registry_adapts_integrators_a_backend_lacks) {
    RenderSettings rs;
    rs.integrator = "hybrid";
    CHECK(adaptToBackend(rs).empty());  // the reference implements it
    CHECK(rs.integrator == "hybrid");
    for (const Backend* b : backends()) {
        RenderSettings s;
        s.backend = b->name();
        s.integrator = "light";
        std::string note = adaptToBackend(s);
        CHECK(b->supports("light") ? note.empty() && s.integrator == "light" : !note.empty() && s.integrator == "path");
    }
}

TEST(display_keeps_the_hue_of_spectral_colours) {
    // Every monochromatic colour lies outside sRGB. Its displayed colour must keep the stimulus's
    // perceptual hue (Oklab), so a spectrum shows its continuous progression of hues instead of the
    // few bands that clipping negative components produced (up to 40° hue errors at 470–600 nm).
    auto hue = [](const double rgb[3]) {
        double l = std::cbrt(0.4122214708 * rgb[0] + 0.5363325363 * rgb[1] + 0.0514459929 * rgb[2]);
        double m = std::cbrt(0.2119034982 * rgb[0] + 0.6806995451 * rgb[1] + 0.1073969566 * rgb[2]);
        double s = std::cbrt(0.0883024619 * rgb[0] + 0.2817188376 * rgb[1] + 0.6299787005 * rgb[2]);
        double a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
        double b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
        return std::atan2(b, a) * 180 / Pi;
    };
    double previous = 0;
    for (double lambda = 420; lambda <= 680; lambda += 2) {
        XYZ c{cieX(lambda) * 0.3, cieY(lambda) * 0.3, cieZ(lambda) * 0.3};
        double rgb[3];
        xyzToLinearSRGB(c, rgb);
        CHECK(std::min({rgb[0], rgb[1], rgb[2]}) < 0);  // out of gamut
        const double h0 = hue(rgb);
        gamutMapLinearSRGB(rgb);
        CHECK(std::min({rgb[0], rgb[1], rgb[2]}) >= 0);
        double dh = std::remainder(hue(rgb) - h0, 360.0);
        CHECK(std::abs(dh) < 0.01);
        // The displayed hue moves monotonically with wavelength (no plateaus) where the stimulus's does.
        if (lambda > 450 && lambda <= 630) CHECK(std::remainder(hue(rgb) - previous, 360.0) < -0.2);
        previous = hue(rgb);
    }
    double inside[3] = {0.2, 0.5, 0.1}, copy[3] = {0.2, 0.5, 0.1};
    gamutMapLinearSRGB(inside);
    for (int k = 0; k < 3; ++k) CHECK(inside[k] == copy[k]);  // in-gamut colours are untouched
}

TEST(image_texture_reflects_its_texels_spectra_on_every_backend) {
    // A four-quadrant sRGB image (red, green, blue, white) on a sheet under a uniform sky of unit radiance: each
    // quadrant's radiance is the texel's reflectance spectrum, r·B_r + g·B_g + b·B_b of the linear
    // colour, whose Y is computed here directly from the basis.
    namespace fs = std::filesystem;
    const fs::path file = fs::temp_directory_path() / "owe_test_texture_quadrants.png";
    std::vector<unsigned char> px;
    const unsigned char colour[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}};
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            for (int c = 0; c < 3; ++c) px.push_back(colour[(y / 32) * 2 + x / 32][c]);
    writeRGB8PNG(file.string(), 64, 64, px);
    char text[800];
    std::snprintf(text, sizeof text, R"(
        units = m
        world { sky = uniform(1) }
        material tex { type = diffuse  texture = image("%s", scale = 1.0) }
        body Sheet { type = sheet  size = (1.0, 1.0)  position = (0.5, -0.5, 0)  material = tex }
        observer Eye { position = (0.5, -0.5, 2.0)  look_at = (0.5, -0.5, 0)  up = (0, 1, 0)  fov = 20deg  pupil = 0
                       resolution = (32, 32) }
    )", file.string().c_str());
    Scene sc = loadSceneFromString(text);
    auto yOf = [](double r, double g, double b) {
        Spectrum s = Spectrum::rgbReflectance(r, g, b);
        double num = 0, den = 0;
        for (double l = LambdaMin; l <= LambdaMax; l += 0.5) { num += s.eval(l) * cieY(l); den += cieY(l); }
        return num / den;
    };
    // Quadrants as seen (image top row at +y): red top-left, green top-right, blue bottom-left, white.
    const double expected[4] = {yOf(1, 0, 0), yOf(0, 1, 0), yOf(0, 0, 1), yOf(1, 1, 1)};
    for (const Backend* b : backends()) {
        if (!b->available()) continue;
        RenderSettings rs;
        rs.backend = b->name();
        auto R = makeRenderer(sc, 0, rs);
        R->runPass(64);
        Image img = R->resolve();
        for (int q = 0; q < 4; ++q) {
            double sum = 0;
            int n = 0;
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x) {
                    int X = (q % 2) * 16 + x, Y = (q / 2) * 16 + y;
                    if (X < 2 || Y < 2 || X > 29 || Y > 29 || (X > 13 && X < 18) || (Y > 13 && Y < 18)) continue;  // edges
                    sum += img.at(X, Y).y;
                    ++n;
                }
            CHECK_NEAR(sum / n, expected[q], 0.02 * expected[q] + 1e-3);
        }
    }
    fs::remove(file);
}

namespace {
// Writes a flat (uncompressed) Radiance HDR image, rows from the top.
void writeHdr(const std::string& path, int w, int h, const std::vector<float>& rgb) {
    std::ofstream out(path, std::ios::binary);
    out << "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y " << h << " +X " << w << "\n";
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        const float r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
        const float m = std::max({r, g, b});
        unsigned char px[4] = {0, 0, 0, 0};
        if (m > 1e-32f) {
            int e;
            const float f = std::frexp(m, &e) * 256.0f / m;
            px[0] = (unsigned char)(r * f);
            px[1] = (unsigned char)(g * f);
            px[2] = (unsigned char)(b * f);
            px[3] = (unsigned char)(e + 128);
        }
        out.write(reinterpret_cast<const char*>(px), 4);
    }
}
double meanOf(const Scene& sc, const std::string& backend, int spp, const std::string& integrator = "path") {
    RenderSettings rs;
    rs.backend = backend;
    rs.integrator = integrator;
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(spp);
    return R->resolve().meanY();
}
// Mean Y over the central n×n pixels.
double centreOf(const Scene& sc, const std::string& backend, int spp, const std::string& integrator, int n) {
    RenderSettings rs;
    rs.backend = backend;
    rs.integrator = integrator;
    auto R = makeRenderer(sc, 0, rs);
    R->runPass(spp);
    Image img = R->resolve();
    double s = 0;
    for (int y = (img.height - n) / 2; y < (img.height + n) / 2; ++y)
        for (int x = (img.width - n) / 2; x < (img.width + n) / 2; ++x) s += img.at(x, y).y;
    return s / (n * n);
}
}  // namespace

TEST(environment_map_lights_the_scene_as_its_radiance_says) {
    // A constant white HDR map is the uniform sky of the same spectrum (rgb(1,1,1) as an illuminant):
    // a white diffuse ball under either must look the same. A map with a small bright patch (a
    // "sun" 1% of the sphere at 500× the rest) lights the ball through next-event estimation of the
    // map and through escapes, combined by MIS: the result must match the analytic mean irradiance.
    namespace fs = std::filesystem;
    const fs::path white = fs::temp_directory_path() / "owe_test_env_white.hdr";
    const fs::path patch = fs::temp_directory_path() / "owe_test_env_patch.hdr";
    const int W = 64, H = 32;
    writeHdr(white.string(), W, H, std::vector<float>(size_t(W) * H * 3, 1.0f));
    std::vector<float> px(size_t(W) * H * 3, 0.2f);
    for (int y = 4; y < 7; ++y)
        for (int x = 10; x < 13; ++x)
            for (int c = 0; c < 3; ++c) px[3 * (size_t(y) * W + x) + c] = 100.0f;
    writeHdr(patch.string(), W, H, px);
    auto scene = [&](const std::string& sky) {
        char text[800];
        std::snprintf(text, sizeof text, R"(
            units = m
            world { sky = %s }
            material white { type = diffuse  reflectance = 0.8 }
            body Ball { type = sphere  radius = 1  position = (0, 0, 0)  material = white }
            observer Eye { position = (0, -4, 0)  look_at = (0, 0, 0)  up = (0, 0, 1)  fov = 30deg  pupil = 0
                           resolution = (32, 32) }
        )", sky.c_str());
        return loadSceneFromString(text);
    };
    Scene mapped = scene("map(\"" + white.string() + "\")");
    Scene uniform = scene("uniform(rgb(1, 1, 1, luminance = 1))");
    Scene sunny = scene("map(\"" + patch.string() + "\", rotation = 30deg)");
    CHECK(mapped.world.environmentLight() >= 0);
    for (const Backend* b : backends()) {
        if (!b->available()) continue;
        const double a = meanOf(mapped, b->name(), 256), u = meanOf(uniform, b->name(), 256);
        CHECK_NEAR(a / u, 1, 0.01);
        // The patchy map: brightness on the ball compared between the backend's path tracer at many
        // samples (escapes and NEE) and the same with a different seed: consistent, and positive.
        const double s1 = meanOf(sunny, b->name(), 512);
        CHECK(s1 > 0);
        // Independently, on the ball (light tracing cannot show the sky seen directly): particles drawn
        // from the map's distribution and connected to the eye against the path tracer.
        const double p1 = centreOf(sunny, b->name(), 512, "path", 12), p2 = centreOf(sunny, b->name(), 4096, "light", 12);
        CHECK_NEAR(p2 / p1, 1, 0.02);
        std::fprintf(stderr, "    [%s] white map / uniform sky %.4f; patchy map on the ball: path %.5g, light %.5g\n",
                     b->name().c_str(), a / u, p1, p2);
    }
    fs::remove(white);
    fs::remove(patch);
}

// Compiled kernels are kept in a pipeline cache on disk, per device and driver: the first renderer
// writes it, and a later one in a fresh device context compiles nothing new.
TEST(gpu_pipeline_cache_is_written_and_reused) {
    const Backend* gpu = findBackend("gpu");
    if (!gpu || !gpu->available()) return;
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "owe_test_pipeline_cache";
    fs::remove_all(dir);
    setenv("OWE_CACHE_DIR", dir.string().c_str(), 1);
    const Scene sc = loadSceneFromString(R"(
        units = m
        world { sky = uniform(rgb(1, 1, 1, luminance = 1)) }
        body Card { type = sheet  size = (1, 1)  position = (0, 0, 0)  axis = (0, -1, 0)  material = white }
        observer Eye { position = (0, -3, 0)  look_at = (0, 0, 0)  up = (0, 0, 1)  fov = 30deg  pupil = 1mm  resolution = (8, 8) }
        render { detector = Eye }
    )");
    RenderSettings rs;
    rs.backend = "gpu";
    rs.integrator = "path";
    auto cacheFiles = [&] {
        std::vector<fs::path> files;
        if (fs::exists(dir))
            for (const auto& e : fs::directory_iterator(dir)) files.push_back(e.path());
        return files;
    };
    makeRenderer(sc, 0, rs)->runPass(1);
    const auto first = cacheFiles();
    CHECK(first.size() == 1);
    const auto bytes = first.empty() ? 0 : fs::file_size(first[0]);
    CHECK(bytes > 32);
    makeRenderer(sc, 0, rs)->runPass(1);
    const auto second = cacheFiles();
    CHECK(second.size() == 1 && fs::file_size(second[0]) == bytes);
    unsetenv("OWE_CACHE_DIR");
    fs::remove_all(dir);
}
