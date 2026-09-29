// Exercise the viewer's asynchronous worker directly, including coalesced camera motion.
// Including the implementation keeps this test out of the engine's public API.
#include "../apps/viewer.cpp"

namespace {
void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
template<class Predicate>
owe::Snapshot until(owe::RenderWorker& worker, Predicate done) {
    owe::Snapshot s;
    auto start = std::chrono::steady_clock::now();
    do {
        worker.poll(s);
        if (s.state == "error") throw std::runtime_error(s.error);
        if (done(s)) return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (std::chrono::steady_clock::now() - start < std::chrono::seconds(30));
    throw std::runtime_error("viewer did not deliver the requested camera frame");
}
}

int main(int argc, char** argv) {
    using namespace owe;
    try {
        RenderWorker worker;
        worker.setMaxSpp(2);
        ViewRequest req;
        req.scenePath = "scenes/the_lens.owe";
        req.detector = "Eye";
        req.backend = argc > 1 ? argv[1] : "cpu";
        auto id = worker.request(req);
        auto original = until(worker, [&](const Snapshot& s) { return s.request == id && s.spp >= 2; });
        NavigationCamera camera = original.camera;
        camera.pan(0.02, 0.01);
        worker.navigate(camera, true);
        auto moving = until(worker, [&](const Snapshot& s) {
            return s.exploring && s.spp > 0 && length(s.camera.position - camera.position) < 1e-12 && s.width == 320;
        });
        require(moving.image.width == 320, "movement preview resolution");
        for (int i = 0; i < 12; ++i) {
            camera.turn(0.005, 0.002, true);
            worker.navigate(camera, true);
        }
        worker.navigate(camera, false);
        auto settled = until(worker, [&](const Snapshot& s) {
            return length(s.camera.position - camera.position) < 1e-12 && s.spp >= 2 && s.image.width == original.image.width;
        });
        require(settled.image.xyz != original.image.xyz, "camera movement must change the rendered image");
        id = worker.request(req);
        auto reset = until(worker, [&](const Snapshot& s) { return s.request == id && s.spp >= 2; });
        require(!reset.exploring, "reset must return to the saved detector");
        require(length(reset.camera.position - original.camera.position) == 0, "navigation modified the saved observer");
        require(reset.image.xyz == original.image.xyz, "reset must reproduce the original view");
        worker.focus(reset.camera, 0.5, 0.5);
        auto focused = until(worker, [&](const Snapshot& s) { return s.focusVersion > reset.focusVersion && s.spp >= 2; });
        require(std::isfinite(focused.camera.focus), "surface focus must set a finite eye focus distance");
        require(length(focused.camera.position - reset.camera.position) == 0, "surface focus must not teleport the eye");
        req.scenePath = "scenes/optical_bench.owe";
        req.detector = "Sensor";
        req.scale = 0.1;
        id = worker.request(req);
        auto sensor = until(worker, [&](const Snapshot& s) { return s.request == id && s.spp >= 2; });
        camera = sensor.camera;
        camera.move(0, -1, 1, 0.1, false);
        worker.navigate(camera, false);
        until(worker, [&](const Snapshot& s) { return s.exploring && s.spp >= 2 && length(s.camera.position - camera.position) < 1e-12; });
        id = worker.request(req);
        auto back = until(worker, [&](const Snapshot& s) { return s.request == id && s.spp >= 2; });
        require(back.description.find("surface sensor") != std::string::npos, "navigation moved a physical sensor");
        require(sensor.image.xyz == back.image.xyz, "physical sensor changed after exploring");
        req.scenePath = "scenes/the_observatory.owe";
        req.detector = "SaturnEyepiece";
        req.eyepieces = {"SaturnEyepiece", "JupiterEyepiece", "MoonEyepiece"};
        req.scale = 0.1;
        id = worker.request(req);
        auto telescope = until(worker, [&](const Snapshot& s) { return s.request == id && s.spp >= 2; });
        require(telescope.eyepieces.size() == 3, "missing telescope interaction targets");
        require(length(telescope.eyepieces[0].position - telescope.camera.position) == 0, "eyepiece target is not at exit pupil");
        auto catalog = scanScenes({req.scenePath});
        require(std::count_if(catalog.views.begin(), catalog.views.end(), [](const ViewEntry& v) { return v.eyepiece; }) == 3,
                "catalog must recognize all exit-pupil views");
        worker.navigate(telescope.camera, false);
        auto eye = until(worker, [&](const Snapshot& s) { return s.exploring && s.spp >= 2; });
        require(eye.image.meanY() > 0, "free eye at telescope exit pupil sees no planet");
        // Free transport must use world sampling, regardless of the previously selected view.
        Scene independent = loadScene(req.scenePath);
        independent.detectors.clear();
        auto independentEye = std::make_unique<IdealObserver>();
        independentEye->width = eye.width; independentEye->height = eye.height;
        eye.camera.apply(*independentEye, independent.world);
        independent.detectors.push_back(std::move(independentEye));
        RenderSettings rs = independent.render; rs.backend = req.backend;
        auto fresh = makeRenderer(independent, 0, rs);
        // Match the worker's pass schedule: RNG streams and wavelength strata are per pass.
        fresh->runPass(1); fresh->runPass(1);
        require(eye.image.xyz == fresh->resolve().xyz,
                "free eye retained transport state from a named eyepiece view");

        // Start in the room, then move a free eye beside (not exactly on) the nominal pupil.
        req.detector = "Room";
        id = worker.request(req);
        auto room = until(worker, [&](const Snapshot& s) { return s.request == id && s.spp >= 2; });
        camera = room.camera;
        camera.position = telescope.camera.position + telescope.camera.right() * 0.00025;
        camera.forward = telescope.camera.forward;
        camera.up = telescope.camera.up;
        worker.navigate(camera, false);
        auto walked = until(worker, [&](const Snapshot& s) { return s.exploring && s.spp >= 2; });
        require(walked.camera.fov == room.camera.fov && walked.camera.pupil == room.camera.pupil &&
                walked.camera.focus == room.camera.focus, "approaching optics adopted an eyepiece preset");
        auto& freeEye = static_cast<IdealObserver&>(*independent.detectors[0]);
        freeEye.width = walked.width; freeEye.height = walked.height;
        camera.apply(freeEye, independent.world);
        require(fresh->resetObserver(), "independent eye cannot move");
        fresh->runPass(1); fresh->runPass(1);
        require(walked.image.xyz == fresh->resolve().xyz,
                "walking from the room differs from independent physical eye transport");

        // An accommodating eye at the eyepiece focuses on Saturn's image, which the afocal
        // telescope forms at infinity; Shift+click on the eyepiece does the same, not on the glass.
        worker.navigate(camera, false, true);
        auto relaxed = until(worker, [&](const Snapshot& s) { return s.exploring && s.accommodated && s.spp >= 2; });
        require(!std::isfinite(relaxed.camera.focus), "an eye at the eyepiece must focus Saturn's image at infinity");
        worker.focus(camera, 0.5, 0.5);
        auto clicked = until(worker, [&](const Snapshot& s) { return s.focusVersion > relaxed.focusVersion && s.spp >= 2; });
        require(!std::isfinite(clicked.camera.focus), "focusing on the eyepiece must focus through it, not on its glass");
        camera.position = room.camera.position;
        camera.forward = room.camera.forward;
        camera.up = room.camera.up;
        worker.navigate(camera, false, true);
        auto wall = until(worker, [&](const Snapshot& s) {
            return s.exploring && s.accommodated && s.spp >= 2 && length(s.camera.position - camera.position) < 1e-12;
        });
        SurfaceHit hit;
        require(independent.world.intersect(Ray{camera.position, camera.forward}, Inf, hit) &&
                    std::abs(wall.camera.focus - hit.t) < 1e-9 * hit.t,
                "with nothing in between, the eye focuses on the surface it looks at");
        std::puts("Viewer navigation: preview, coalesced movement, refinement, reset and sensor preservation passed.");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "viewer navigation: %s\n", e.what());
        return 1;
    }
}
