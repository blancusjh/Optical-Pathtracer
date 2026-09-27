// The backend-neutral render front end: the display pipeline shared by PNGs and the viewer, and
// the backend registry.
#include "check.hpp"
#include "owe/backends/registry.hpp"
#include "owe/render/output.hpp"

using namespace owe;

TEST(auto_exposure_preserves_a_small_planet_in_a_wide_view) {
    Image img; img.width = 160; img.height = 100; img.xyz.resize(160 * 100 * 3);
    for (size_t i = 0; i < img.xyz.size(); i += 3) {
        img.xyz[i] = 0.95047e-7; img.xyz[i+1] = 1e-7; img.xyz[i+2] = 1.08883e-7;
    }
    for (int y = 49; y <= 50; ++y) for (int x = 79; x <= 80; ++x) {
        size_t i = 3 * (y * 160 + x);
        img.xyz[i] = 0.95047e-3; img.xyz[i+1] = 1e-3; img.xyz[i+2] = 1.08883e-3;
    }
    double gain = autoExposureEV(img);
    CHECK_NEAR(std::exp2(gain), 600, 1e-8);
    img.xyz[1] = 1; // an isolated firefly must not override the resolved small object
    CHECK_NEAR(autoExposureEV(img), gain, 1e-10);
    img.xyz[4] = img.xyz[3 * 160 + 1] = 1; // a peripheral lamp does not set the fixation meter
    CHECK_NEAR(autoExposureEV(img), gain, 1e-10);
    auto rgb = displayRGB8(img, 0, true);
    CHECK(rgb[3 * (50 * 160 + 80) + 1] > 180);
    CHECK(rgb[3 * (20 * 160 + 20) + 1] < 10);
}

TEST(auto_exposure_preserves_projection_detail_and_adapts_to_daylight) {
    Image img; img.width = 100; img.height = 1; img.xyz.resize(300);
    for (int i = 0; i < 100; ++i) {
        double y = i < 90 ? 1e-6 : 0.1 + 0.01 * (i - 90);
        img.xyz[3*i] = 0.95047 * y; img.xyz[3*i+1] = y; img.xyz[3*i+2] = 1.08883 * y;
    }
    auto inside = displayRGB8(img, 0, true);
    auto locked = displayRGB8(img, autoExposureEV(img), false);
    CHECK(inside == locked); // locking the meter must not change the current brightness
    CHECK(inside[3*99+1] < 250);
    CHECK(inside[3*99+1] > inside[3*90+1] + 20);
    for (double& x : img.xyz) x *= 1e6;
    auto outside = displayRGB8(img, 0, true);
    for (size_t i = 0; i < inside.size(); ++i) CHECK(std::abs(int(inside[i]) - int(outside[i])) <= 1);
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
