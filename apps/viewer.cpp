// owe view — the interactive window. A render worker thread owns the scene and the renderer
// (from any backend in the registry) and refines the current view pass by pass; the UI thread shows
// the latest estimate through the same display pipeline as the PNG writer, and sends requests:
// another view, another backend or resolution, scene edits, a pixel probe, a save.
#include "viewer.hpp"
#include "navigation.hpp"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <utility>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#include "owe/analysis/analysis.hpp"
#include "owe/analysis/inspect.hpp"
#include "owe/backends/registry.hpp"
#include "owe/render/exposure.hpp"
#include "owe/render/output.hpp"
#include "owe/render/record.hpp"
#include "owe/loader/scene_loader.hpp"
#include "owe/loader/scene_parser.hpp"

namespace owe {

bool viewerAvailable(std::string*) { return true; }

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// ---------------------------------------------------------------- the catalogue of views

struct ViewEntry {
    int scene = 0;
    std::string detector, kind;     // kind: observer, camera, sensor
    double fNumber = 0, focus = 0;  // camera values as written in the scene (0 = unset)
    bool eyepiece = false;
};

struct SceneEntry {
    std::string path, file, title, blurb, error, defaultDetector;
    std::vector<int> views;  // indices into Catalog::views
};

struct Catalog {
    std::vector<SceneEntry> scenes;
    std::vector<ViewEntry> views;

    // "file.owe" or "file.owe:Detector" (matched by path or file name); -1 if absent.
    int find(const std::string& spec) const {
        std::string file = spec, det;
        if (auto c = spec.rfind(':'); c != std::string::npos && spec.substr(c).find('/') == std::string::npos) {
            file = spec.substr(0, c);
            det = spec.substr(c + 1);
        }
        for (size_t s = 0; s < scenes.size(); ++s) {
            const SceneEntry& e = scenes[s];
            std::error_code ec;
            bool same = e.file == file || e.path == file || fs::equivalent(e.path, file, ec);
            if (!same || e.views.empty()) continue;
            const std::string& want = det.empty() ? e.defaultDetector : det;
            for (int v : e.views)
                if (views[size_t(v)].detector == want) return v;
            return det.empty() ? e.views.front() : -1;
        }
        return -1;
    }
};

std::string titleCase(const std::string& s) {
    std::string r = s;
    bool start = true;
    for (char& c : r) {
        if (std::isalpha((unsigned char)c)) {
            c = start ? char(std::toupper((unsigned char)c)) : char(std::tolower((unsigned char)c));
            start = false;
        } else {
            start = c == ' ' || c == '/' || c == '-';
        }
    }
    return r;
}

// Title and description from the scene file's leading comment block.
void describe(const std::string& text, SceneEntry& e) {
    std::istringstream in(text);
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] != '#') break;
        std::string t = line.substr(1);
        t.erase(0, t.find_first_not_of(" \t"));
        if (first) {
            auto dash = t.find(" — ");
            e.title = titleCase(dash == std::string::npos ? t : t.substr(0, dash));
            if (dash != std::string::npos) e.blurb = t.substr(dash + std::strlen(" — "));
            first = false;
        } else {
            if (t.empty()) break;
            e.blurb += (e.blurb.empty() ? "" : " ") + t;
        }
    }
    if (e.title.empty()) e.title = fs::path(e.path).stem().string();
}

Catalog scanScenes(const std::vector<std::string>& paths) {
    std::vector<std::string> files;
    for (const std::string& p : paths) {
        std::error_code ec;
        if (fs::is_directory(p, ec)) {
            std::vector<std::string> found;
            // Groups of scenes live in subdirectories (scenes/glass/...).
            for (auto& de : fs::recursive_directory_iterator(p, ec))
                if (de.path().extension() == ".owe") found.push_back(de.path().string());
            std::sort(found.begin(), found.end());
            files.insert(files.end(), found.begin(), found.end());
        } else {
            files.push_back(p);
        }
    }
    Catalog cat;
    for (const std::string& f : files) {
        SceneEntry e;
        e.path = f;
        e.file = fs::path(f).filename().string();
        std::ifstream in(f, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        describe(text, e);
        const int si = int(cat.scenes.size());
        try {
            ValuePtr doc = parseSceneText(text, f);
            for (const ValuePtr& b : doc->items) {
                if (!b || b->kind != Value::Kind::Block) continue;
                if (b->str == "observer" || b->str == "camera" || b->str == "sensor") {
                    ViewEntry v;
                    v.scene = si;
                    v.detector = b->name;
                    v.kind = b->str;
                    if (auto p = b->get("position"))
                        v.eyepiece = p->kind == Value::Kind::Call && p->str == "exit_pupil";
                    if (auto x = b->get("f_number"); x && x->kind == Value::Kind::Number) v.fNumber = x->num;
                    if (auto x = b->get("focus"); x && x->kind == Value::Kind::Number) v.focus = x->num;
                    e.views.push_back(int(cat.views.size()));
                    cat.views.push_back(v);
                } else if (b->str == "render") {
                    if (auto d = b->get("detector"); d && (d->kind == Value::Kind::Ident || d->kind == Value::Kind::String))
                        e.defaultDetector = d->str;
                }
            }
        } catch (const std::exception& ex) {
            e.error = ex.what();
        }
        cat.scenes.push_back(e);
    }
    return cat;
}

// ---------------------------------------------------------------- the render worker

struct ViewRequest {
    std::string scenePath, detector;
    std::vector<std::string> edits;
    std::string backend = "cpu";
    int device = -1;
    double scale = 1;
    bool pathOnly = false;  // path integrator instead of the scene's own (also while exploring)
    std::optional<NavigationCamera> camera;
    std::vector<std::string> eyepieces;
};

struct EyepieceTarget {
    std::string detector;
    Vec3 position;
    double pupilDiameter = 0;
    Vec3 towardObjective;
};

struct NavigationRequest {
    NavigationCamera camera;
    bool preview = false;
    bool accommodate = false;       // the eye focuses on what it looks at, through any optics
    std::optional<Vec2> focusPoint; // normalized image coordinates
};

// What the UI shows; written by the worker under the mutex.
struct Snapshot {
    uint64_t request = 0;        // the request this state belongs to
    uint64_t version = 0;        // bumps with every new image
    std::string state = "idle";  // loading, rendering, paused, converged, error
    std::string phase;           // while loading: what the worker is doing
    std::string error, message;
    Image image;
    NavigationCamera camera;
    bool exploring = false;
    bool accommodated = false;   // camera.focus was set by the eye's accommodation
    double surfaceDistance = Inf;
    uint64_t focusVersion = 0;
    std::vector<EyepieceTarget> eyepieces;
    long long spp = 0;
    int passes = 0;
    double seconds = 0, lastPass = 0;
    TransportStats stats;
    std::string backend, integrator, description, note;
    int width = 0, height = 0;
    double exposure = 0, whiteBalance = 0;  // the scene's own display settings for this detector
    bool autoExposure = true;
    std::string tone = "agx";
    uint64_t probeVersion = 0;
    int probeX = -1, probeY = -1;
    std::string probeText;
};

class RenderWorker {
public:
    RenderWorker() : thread_([this] { loop(); }) {}
    ~RenderWorker() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    uint64_t request(const ViewRequest& r) {
        std::lock_guard<std::mutex> lk(m_);
        pending_ = r;
        navigation_.reset();
        cv_.notify_all();
        snap_.request = ++requestId_;
        snap_.state = "loading";
        snap_.phase = "waiting for the current pass";
        snap_.error.clear();
        snap_.message.clear();
        snap_.image = Image{};
        snap_.spp = 0;
        snap_.probeText.clear();
        snap_.version++;
        save_.reset();
        probe_.reset();
        return requestId_;
    }
    void navigate(const NavigationCamera& camera, bool preview, bool accommodate = false) {
        std::lock_guard<std::mutex> lk(m_);
        navigation_ = NavigationRequest{camera, preview, accommodate, std::nullopt};
        paused_ = false;
        save_.reset();
        probe_.reset();
        cv_.notify_all();
    }
    void focus(const NavigationCamera& camera, double x, double y) {
        std::lock_guard<std::mutex> lk(m_);
        navigation_ = NavigationRequest{camera, false, false, Vec2{x, y}};
        paused_ = false;
        cv_.notify_all();
    }
    void setPaused(bool p) {
        std::lock_guard<std::mutex> lk(m_);
        paused_ = p;
        if (snap_.state == "rendering" || snap_.state == "paused") snap_.state = p ? "paused" : "rendering";
        cv_.notify_all();
    }
    void setMaxSpp(long long n) {
        std::lock_guard<std::mutex> lk(m_);
        maxSpp_ = n;
        cv_.notify_all();
    }
    void save(const std::string& prefix, double ev, bool autoEv, double wb, Tone tone) {
        std::lock_guard<std::mutex> lk(m_);
        save_ = SaveJob{prefix, ev, autoEv, wb, tone};
        cv_.notify_all();
    }
    void probe(int x, int y) {
        std::lock_guard<std::mutex> lk(m_);
        probe_ = std::make_pair(x, y);
        cv_.notify_all();
    }
    // Copies the state; the image only when it changed since out.version. True if it changed.
    bool poll(Snapshot& out) {
        std::lock_guard<std::mutex> lk(m_);
        bool changed = out.version != snap_.version;
        // Copy metadata on every frame, but copy the image only after a render pass.
        Image image = std::move(snap_.image);
        Image keep = std::move(out.image);
        out = snap_;
        if (changed) out.image = image;
        else out.image = std::move(keep);
        snap_.image = std::move(image);
        return changed;
    }

private:
    struct SaveJob {
        std::string prefix;
        double ev;
        bool autoEv;
        double wb;
        Tone tone;
    };

    void loop() {
        std::unique_ptr<Scene> scene;
        std::string scenePath;
        std::vector<std::string> sceneEdits;
        std::map<std::string, std::pair<int, int>> native;  // detector → its own resolution
        std::unique_ptr<Renderer> R;
        int det = -1, spp = 1, explorer = -1;
        int nativeWidth = 0, nativeHeight = 0;
        bool exploring = false, preview = false;
        NavigationCamera camera;
        uint64_t current = 0;
        const int threads = std::max(1, int(std::thread::hardware_concurrency()) - 1);  // one core for the UI
        const double kTargetPass = 0.12;  // seconds: ~8 image updates per second
        auto converged = [&] { return maxSpp_ > 0 && R && R->samplesPerPixel() >= maxSpp_; };

        for (;;) {
            std::optional<ViewRequest> req;
            std::optional<NavigationRequest> nav;
            std::optional<SaveJob> save;
            std::optional<std::pair<int, int>> probe;
            bool paused;
            long long maxSpp;
            uint64_t reqId;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return quit_ || pending_ || navigation_ || save_ || probe_ || (R && !paused_ && !converged()); });
                if (quit_) return;
                req = std::exchange(pending_, std::nullopt);
                nav = std::exchange(navigation_, std::nullopt);
                save = std::exchange(save_, std::nullopt);
                probe = std::exchange(probe_, std::nullopt);
                paused = paused_;
                maxSpp = maxSpp_;
                reqId = requestId_;
                if (!req && R) snap_.state = paused ? "paused" : converged() ? "converged" : "rendering";
            }

            if (req) {
                R.reset();  // the renderer refers to the scene: release it first
                {
                    std::lock_guard<std::mutex> lk(m_);
                    if (requestId_ != reqId) continue;
                }
                auto phase = [&](const std::string& what) {
                    std::lock_guard<std::mutex> lk(m_);
                    if (snap_.request == reqId) snap_.phase = what;
                };
                try {
                    if (!scene || scenePath != req->scenePath || sceneEdits != req->edits) {
                        phase("reading the scene and its models");
                        scene.reset();
                        auto sc = std::make_unique<Scene>(loadSceneWithEdits(req->scenePath, req->edits));
                        explorer = -1;
                        native.clear();
                        for (auto& d : sc->detectors) native[d->name] = {d->width, d->height};
                        scene = std::move(sc);
                        scenePath = req->scenePath;
                        sceneEdits = req->edits;
                    }
                    int d = req->detector.empty()
                                ? (scene->render.detector.empty() ? 0 : scene->findDetector(scene->render.detector))
                                : scene->findDetector(req->detector);
                    if (d < 0) throw std::runtime_error("no detector named '" + req->detector + "'");
                    Detector& base = *scene->detectors[size_t(d)];
                    auto [nw, nh] = native[base.name];
                    nativeWidth = base.width = std::max(8, int(std::lround(nw * req->scale)));
                    nativeHeight = base.height = std::max(8, int(std::lround(nh * req->scale)));
                    scene->useDetector(d);
                    base.prepare(scene->world);
                    camera = req->camera.value_or(NavigationCamera::from(base, &scene->world));
                    exploring = req->camera.has_value();
                    preview = false;
                    RenderSettings rs = scene->settingsFor(d);
                    if (exploring) {
                        if (explorer < 0) {
                            explorer = int(scene->detectors.size());
                            scene->detectors.push_back(std::make_unique<IdealObserver>());
                        }
                        auto& eye = static_cast<IdealObserver&>(*scene->detectors[size_t(explorer)]);
                        eye.name = base.name + " (explore)";
                        adoptPixelResponse(eye, base);
                        eye.width = nativeWidth;
                        eye.height = nativeHeight;
                        camera.apply(eye, scene->world);
                        d = explorer;
                        scene->useDetector(d);
                    }
                    Detector& D = *scene->detectors[size_t(d)];
                    rs.backend = req->backend;
                    rs.device = req->device;
                    rs.threads = threads;
                    // Exploring keeps the scene's integrator: caustics seen from a free eye are formed
                    // by light tracing, as in the scene's own views, instead of rare camera paths.
                    if (req->pathOnly) rs.integrator = "path";
                    adaptToBackend(rs);  // e.g. hybrid → path on a backend without it
                    phase("preparing the " + rs.backend + " renderer");
                    R = makeRenderer(*scene, d, rs);
                    det = d;
                    spp = 1;
                    current = reqId;
                    std::lock_guard<std::mutex> lk(m_);
                    if (snap_.request == reqId) {
                        snap_.state = "rendering";
                        snap_.camera = camera;
                        snap_.surfaceDistance = camera.surfaceDistance(scene->world);
                        snap_.exploring = exploring;
                        snap_.accommodated = false;  // the view's own focus
                        snap_.eyepieces.clear();
                        for (const auto& name : req->eyepieces) {
                            int index = scene->findDetector(name);
                            if (index >= 0) {
                                auto* eye = dynamic_cast<IdealObserver*>(scene->detectors[size_t(index)].get());
                                if (eye) snap_.eyepieces.push_back({name, eye->position, 2 * eye->pupilRadius, eye->forward()});
                            }
                        }
                        snap_.backend = R->backend();
                        snap_.integrator = rs.integrator;
                        snap_.description = D.describe();
                        auto note = scene->notes.find(D.name);
                        snap_.note = note == scene->notes.end() ? "" : note->second;
                        snap_.width = D.width;
                        snap_.height = D.height;
                        snap_.exposure = rs.exposure;
                        snap_.autoExposure = rs.autoExposure;
                        snap_.whiteBalance = rs.whiteBalance;
                        snap_.tone = rs.tone;
                        snap_.spp = 0;
                        snap_.passes = 0;
                        snap_.seconds = snap_.lastPass = 0;
                        snap_.stats = TransportStats{};
                        snap_.image = Image{};
                        snap_.probeText.clear();
                        snap_.version++;
                    }
                } catch (const std::exception& e) {
                    R.reset();
                    scene.reset();
                    scenePath.clear();
                    std::lock_guard<std::mutex> lk(m_);
                    if (snap_.request == reqId) {
                        snap_.state = "error";
                        snap_.error = e.what();
                    }
                }
                continue;
            }

            if (nav && scene && R) {
                try {
                    camera = nav->camera;
                    preview = nav->preview;
                    // The eye focuses where the object's image is: through a telescope, a mirror or
                    // a magnifier that is not the glass surface in front of it.
                    auto accommodate = [&](const Vec3& dir) {
                        Accommodation a = accommodation(*scene, camera.position, dir, scene->world.locate(camera.position),
                                                        camera.pupil);
                        if (a.found) camera.focus = a.distance;
                        return a.found;
                    };
                    if (nav->focusPoint) {
                        double tanY = std::tan(camera.fov * 0.5);
                        Vec3 dir = normalize(camera.forward + camera.right() * ((2 * nav->focusPoint->x - 1) * tanY * nativeWidth / nativeHeight)
                                             + camera.up * ((1 - 2 * nav->focusPoint->y) * tanY));
                        SurfaceHit hit;
                        if (scene->world.intersect(Ray{camera.position, dir}, Inf, hit)) {
                            camera.forward = dir;
                            camera.up = normalize(cross(camera.right(), camera.forward));
                            camera.distance = std::max(1e-5, hit.t);  // orbit about the surface clicked
                            if (!accommodate(dir)) camera.focus = camera.distance;
                        }
                    } else if (nav->accommodate && camera.pupil > 0) {
                        accommodate(camera.forward);
                    }
                    RenderSettings rs = R->settings();
                    if (!exploring) {
                        const std::string name = R->detector().name + " (explore)";
                        const int base = det;
                        R.reset();
                        if (explorer < 0) {
                            explorer = int(scene->detectors.size());
                            scene->detectors.push_back(std::make_unique<IdealObserver>());
                        }
                        scene->detectors[size_t(explorer)]->name = name;
                        adoptPixelResponse(static_cast<IdealObserver&>(*scene->detectors[size_t(explorer)]),
                                           *scene->detectors[size_t(base)]);
                        det = explorer;
                        exploring = true;
                        scene->useDetector(det);
                    }
                    auto& eye = static_cast<IdealObserver&>(*scene->detectors[size_t(det)]);
                    eye.width = preview ? std::min(nativeWidth, 320) : nativeWidth;
                    eye.height = std::max(8, int(std::lround(double(nativeHeight) * eye.width / nativeWidth)));
                    camera.apply(eye, scene->world);
                    // The current integrator (the scene's, or path if asked); a free eye is an
                    // observer, so light and hybrid both apply.
                    if (!R || !R->resetObserver()) R = makeRenderer(*scene, det, rs);
                    spp = 1;
                    paused = false;
                    std::lock_guard<std::mutex> lk(m_);
                    snap_.exploring = true;
                    snap_.camera = camera;
                    snap_.surfaceDistance = camera.surfaceDistance(scene->world);
                    if (nav->focusPoint) snap_.focusVersion++;
                    snap_.accommodated = nav->accommodate || nav->focusPoint.has_value();
                    snap_.description = eye.describe();
                    snap_.integrator = rs.integrator;
                    snap_.width = eye.width;
                    snap_.height = eye.height;
                    snap_.spp = 0;
                    snap_.passes = 0;
                    snap_.seconds = 0;
                    snap_.stats = {};
                    snap_.probeText.clear();
                    snap_.state = "rendering";
                } catch (const std::exception& e) {
                    R.reset();
                    std::lock_guard<std::mutex> lk(m_);
                    snap_.state = "error";
                    snap_.error = e.what();
                    continue;
                }
            }

            if (save && R) {
                std::string msg;
                try {
                    fs::create_directories(fs::path(save->prefix).parent_path());
                    Image img = R->resolve();
                    writePNG(save->prefix + ".png", img, save->ev, save->autoEv, save->wb, save->tone);
                    writePFM(save->prefix + ".pfm", img);
                    RenderSettings settings = R->settings();
                    settings.exposure = save->ev;
                    settings.autoExposure = save->autoEv;
                    settings.whiteBalance = save->wb;
                    settings.tone = save->tone == Tone::AgX ? "agx" : "standard";
                    std::ofstream(save->prefix + ".json") << renderMetadataJSON(*scene, *R, settings);
                    msg = "saved " + save->prefix + ".png, .pfm, .json";
                } catch (const std::exception& e) {
                    msg = std::string("save failed: ") + e.what();
                }
                std::lock_guard<std::mutex> lk(m_);
                snap_.message = msg;
            }

            if (probe && R && scene) {
                std::string text;
                try {
                    PixelProbe pr = probePixel(*scene, det, probe->first, probe->second, 2000, scene->render.seed);
                    text = pr.text(scene->world);
                } catch (const std::exception& e) {
                    text = std::string("probe failed: ") + e.what();
                }
                std::lock_guard<std::mutex> lk(m_);
                snap_.probeText = text;
                snap_.probeX = probe->first;
                snap_.probeY = probe->second;
                snap_.probeVersion++;
            }

            if (R && !paused && !(maxSpp > 0 && R->samplesPerPixel() >= maxSpp)) {
                int n = preview ? 1 : spp;
                if (maxSpp > 0) n = int(std::min<long long>(n, maxSpp - R->samplesPerPixel()));
                auto t0 = Clock::now();
                try {
                    R->runPass(n);
                } catch (const std::exception& e) {
                    R.reset();
                    std::lock_guard<std::mutex> lk(m_);
                    snap_.state = "error";
                    snap_.error = e.what();
                    continue;
                }
                double dt = since(t0);
                Image img = R->resolve();
                {
                    std::lock_guard<std::mutex> lk(m_);
                    if (snap_.request == current) {
                        snap_.image = std::move(img);
                        snap_.version++;
                        snap_.spp = R->samplesPerPixel();
                        snap_.passes = R->passes();
                        snap_.seconds = R->seconds();
                        snap_.lastPass = dt;
                        snap_.stats = R->stats();
                        snap_.state = paused_ ? "paused" : converged() ? "converged" : "rendering";
                    }
                }
                // Size passes to keep the image updating a few times per second.
                spp = std::clamp(int(n * kTargetPass / std::max(dt, 1e-3)), std::max(1, n / 2), std::max(1, 2 * n));
                spp = std::clamp(spp, 1, 4096);
            }
        }
    }

    std::mutex m_;
    std::condition_variable cv_;
    bool quit_ = false, paused_ = false;
    long long maxSpp_ = 4096;
    uint64_t requestId_ = 0;
    std::optional<ViewRequest> pending_;
    std::optional<NavigationRequest> navigation_;
    std::optional<SaveJob> save_;
    std::optional<std::pair<int, int>> probe_;
    Snapshot snap_;
    std::thread thread_;
};

// ---------------------------------------------------------------- the window

ImFont* loadFont(const std::vector<const char*>& candidates, float size) {
    for (const char* p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec)) {
            if (ImFont* f = ImGui::GetIO().Fonts->AddFontFromFileTTF(p, size)) return f;
        }
    }
    return nullptr;
}

std::string fileSafe(const std::string& s) {
    std::string r;
    for (char c : s) r += std::isalnum((unsigned char)c) ? c : '_';
    return r;
}

struct ImageDraw {
    SDL_Texture* texture = nullptr;
    SDL_FRect rect{};
};

void drawImageTexture(const ImDrawList*, const ImDrawCmd* command) {
    auto* state = static_cast<ImGui_ImplSDLRenderer3_RenderState*>(ImGui::GetPlatformIO().Renderer_RenderState);
    auto* image = static_cast<const ImageDraw*>(command->UserCallbackData);
    SDL_Rect clip{int(command->ClipRect.x), int(command->ClipRect.y),
                  int(command->ClipRect.z - command->ClipRect.x), int(command->ClipRect.w - command->ClipRect.y)};
    SDL_SetRenderClipRect(state->Renderer, &clip);
    SDL_RenderTexture(state->Renderer, image->texture, nullptr, &image->rect);
}

class Viewer {
public:
    explicit Viewer(const ViewerOptions& o) : opt_(o) {
        std::vector<std::string> paths = o.paths;
        if (paths.empty()) paths.push_back(fs::exists("scenes") ? "scenes" : ".");
        cat_ = scanScenes(paths);
        if (!o.view.empty() && cat_.find(o.view) < 0) {
            // A scene outside the listed paths: add it.
            std::string file = o.view.substr(0, o.view.rfind(':') == std::string::npos ? o.view.size() : o.view.rfind(':'));
            Catalog extra = scanScenes({file});
            for (auto& v : extra.views) v.scene += int(cat_.scenes.size());
            for (auto& s : extra.scenes) {
                for (int& v : s.views) v += int(cat_.views.size());
                cat_.scenes.push_back(s);
            }
            cat_.views.insert(cat_.views.end(), extra.views.begin(), extra.views.end());
        }
        if (cat_.views.empty()) throw std::runtime_error("no scene views found; pass a directory containing .owe scenes");
        if (!o.view.empty() && cat_.find(o.view) < 0) throw std::runtime_error("no view matching '" + o.view + "'");
        for (const Backend* b : backends()) {
            std::string why;
            if (b->available(&why)) {
                usable_.push_back(b);
                devices_[b->name()] = b->devices();  // probing a GPU creates an instance: once
            } else {
                unusable_.push_back({b, why});
            }
        }
        if (!o.backend.empty()) {
            std::string why;
            if (!backend(o.backend).available(&why)) throw std::runtime_error(o.backend + " backend unavailable: " + why);
            backend_ = o.backend;
        } else {
            // The fastest available: a backend other than the reference when it has a discrete GPU
            // (or Apple silicon, through MoltenVK). A laptop's integrated GPU (Intel, AMD) is slower
            // than the reference on its CPU cores (2-7× on an i7-12650H), so the CPU starts there;
            // G or the Render panel still switches.
            backend_ = referenceBackend().name();
            for (const Backend* b : usable_) {
                const auto& ds = devices_[b->name()];
                if (b != &referenceBackend() &&
                    std::any_of(ds.begin(), ds.end(), [](const DeviceInfo& d) { return d.type == "discrete" || d.portability; }))
                    backend_ = b->name();
            }
        }
        const auto& devices = devices_[backend_];
        if (o.device >= 0 && std::none_of(devices.begin(), devices.end(), [&](const auto& d) { return d.index == o.device; }))
            throw std::runtime_error("invalid device index for the " + backend_ + " backend");
        device_ = o.device;
        // The reference renders in double on the CPU: start it at half resolution.
        scale_ = float(o.scale > 0 ? o.scale : (backend_ == referenceBackend().name() ? 0.5 : 1.0));
        tour_ = o.tour;
        tourSeconds_ = float(o.tourSeconds);
        int start = o.view.empty() ? -1 : cat_.find(o.view);
        if (start < 0 && !cat_.views.empty()) start = cat_.find(cat_.scenes[size_t(cat_.views.front().scene)].path);
        if (start >= 0) {
            if (!o.edits.empty()) edits_[cat_.scenes[size_t(cat_.views[size_t(start)].scene)].path] = o.edits;
            select(start);
        }
    }

    int run();

private:
    const ViewEntry* view() const { return current_ >= 0 ? &cat_.views[size_t(current_)] : nullptr; }
    const SceneEntry* sceneOf(int v) const { return &cat_.scenes[size_t(cat_.views[size_t(v)].scene)]; }

    void select(int v) {
        if (v < 0 || v >= int(cat_.views.size())) return;
        stopFlight();
        navigationReady_ = exploring_ = motionPending_ = false;
        current_ = v;
        viewStart_ = Clock::now();
        adoptSceneDisplay_ = true;
        probeOpen_ = false;
        focusSeen_ = snap_.focusVersion;
        const ViewEntry& e = cat_.views[size_t(v)];
        fNumber_ = float(e.fNumber);
        focus_ = float(e.focus);
        for (const std::string& ed : edits_[sceneOf(v)->path]) {  // camera edits already applied win
            if (ed.rfind(e.detector + ".f_number=", 0) == 0) fNumber_ = std::stof(ed.substr(ed.find('=') + 1));
            if (ed.rfind(e.detector + ".focus=", 0) == 0) focus_ = std::stof(ed.substr(ed.find('=') + 1));
        }
        send();
    }
    void send() {
        const ViewEntry* e = view();
        if (!e) return;
        ViewRequest r;
        r.scenePath = sceneOf(current_)->path;
        r.detector = e->detector;
        r.edits = edits_[r.scenePath];
        r.backend = backend_;
        r.device = device_;
        r.scale = scale_;
        r.pathOnly = pathOnly_;
        for (int vi : sceneOf(current_)->views)
            if (cat_.views[size_t(vi)].eyepiece) r.eyepieces.push_back(cat_.views[size_t(vi)].detector);
        if (exploring_ && navigationReady_) r.camera = camera_;
        requestId_ = worker_.request(r);
        requestStart_ = Clock::now();
        worker_.setPaused(paused_ = false);
    }
    void switchBackend(const std::string& name) {
        if (name == backend_) return;
        backend_ = name;
        device_ = -1;
        send();
    }
    void step(int dv) {
        if (cat_.views.empty()) return;
        int n = int(cat_.views.size());
        select(((current_ < 0 ? 0 : current_) + dv + n) % n);
    }
    void stepScene(int ds) {
        if (cat_.scenes.empty() || current_ < 0) return;
        int n = int(cat_.scenes.size());
        int s = cat_.views[size_t(current_)].scene;
        for (int k = 0; k < n; ++k) {
            s = (s + ds + n) % n;
            const SceneEntry& e = cat_.scenes[size_t(s)];
            if (e.views.empty()) continue;
            int want = cat_.find(e.path);
            select(want >= 0 ? want : e.views.front());
            return;
        }
    }
    void setEdit(const std::string& key, const std::string& value) {
        auto& ed = edits_[sceneOf(current_)->path];
        ed.erase(std::remove_if(ed.begin(), ed.end(), [&](const std::string& s) { return s.rfind(key + "=", 0) == 0; }), ed.end());
        if (!value.empty()) ed.push_back(key + "=" + value);
        send();
    }
    void saveCurrent() {
        const ViewEntry* e = view();
        if (!e || snap_.request != requestId_ || snap_.spp <= 0 || motionPending_) return;
        std::string prefix = "out/view_" + fileSafe(fs::path(sceneOf(current_)->path).stem().string()) + "_" + fileSafe(e->detector);
        if (exploring_) prefix += "_explore";
        // The file shows exactly what the window shows: the resolved exposure, not a new meter reading.
        worker_.save(prefix, exposure_.resolvedEV(), false, whiteBalance_, filmic_ ? Tone::AgX : Tone::Standard);
    }

    void stopFlight() {
        flight_ = lookHeld_ = false;
        if (window_) SDL_SetWindowRelativeMouseMode(window_, false);
    }
    void moved() {
        exploring_ = motionPending_ = true;
        paused_ = tour_ = false;
        lastMotion_ = Clock::now();
        worker_.navigate(camera_, true, accommodate_);
    }
    // Field of view as a zoom: the eye stays where it is; finer angular detail per pixel.
    void zoom(double factor) {
        camera_.fov = std::clamp(camera_.fov * factor, radians(0.1), radians(120));
        moved();
    }
    void navigationInput(bool hovered, float imageHeight);
    int nearbyEyepiece() const {
        if (!navigationReady_ || snap_.request != requestId_) return -1;
        int result = -1;
        double best = 0.25;
        for (const auto& target : snap_.eyepieces) {
            double distance = length(camera_.position - target.position);
            if (distance >= best) continue;
            for (int vi : sceneOf(current_)->views) {
                if (cat_.views[size_t(vi)].detector == target.detector) { result = vi; best = distance; break; }
            }
        }
        return result;
    }
    void drawNavigation();
    void refreshTexture(SDL_Renderer* ren);
    // Manual, Locked or Adaptive; the displayed brightness does not change at the switch.
    void setExposureMode(ExposureMode m) {
        exposure_.setMode(m);
        if (m != ExposureMode::Manual) autoMode_ = m;
        displayDirty_ = true;
    }
    void addExposure(double ev) {
        exposure_.setCompensation(exposure_.compensation() + ev);
        displayDirty_ = true;
    }
    void drawSidebar(float width);
    void drawImage();
    void drawProbe();
    void handleKeys();
    bool screenshot(SDL_Renderer* ren, const std::string& path);

    ViewerOptions opt_;
    Catalog cat_;
    RenderWorker worker_;
    Snapshot snap_;
    uint64_t requestId_ = 0;
    int current_ = -1;
    std::vector<const Backend*> usable_;                           // available here, reference first
    std::vector<std::pair<const Backend*, std::string>> unusable_;  // and why the others are not
    std::map<std::string, std::vector<DeviceInfo>> devices_;        // per usable backend
    std::string backend_;
    bool pathOnly_ = false, paused_ = false;
    int device_ = -1;
    float scale_ = 1;
    int maxSppIndex_ = 2;  // 256, 1024, 4096, 16384, unlimited
    // Display.
    // Display. Exposure is display state only (owe/render/exposure.hpp): in free navigation it is
    // locked by default, so a lamp entering the view saturates instead of re-metering the room.
    bool adoptSceneDisplay_ = true, pixelated_ = false;
    float whiteBalance_ = 0;
    bool filmic_ = true;  // the display's view transform: AgX, else standard
    ExposureControl exposure_{ExposureMode::Locked};
    ExposureMode autoMode_ = ExposureMode::Locked;  // the automatic mode a view without a fixed exposure uses
    Clock::time_point lastFrame_ = Clock::now();    // UI time for adaptation
    double shownEv_ = 0;
    bool displayDirty_ = true;
    // Texture holding the displayed image.
    ImageDraw imageDraw_;
    SDL_Texture* tex_ = nullptr;
    int texW_ = 0, texH_ = 0;
    uint64_t shownVersion_ = ~0ull;
    // Camera controls, scene edits, tour, probe, panels.
    float fNumber_ = 0, focus_ = 0;
    std::map<std::string, std::vector<std::string>> edits_;
    char editBuf_[256] = {};
    bool tour_ = false;
    float tourSeconds_ = 10;
    Clock::time_point viewStart_ = Clock::now();
    Clock::time_point requestStart_ = Clock::now();
    bool probeOpen_ = false;
    uint64_t probeSeen_ = 0;
    uint64_t focusSeen_ = 0;
    bool showPanel_ = true, quit_ = false;
    ImFont* mono_ = nullptr;
    float uiScale_ = 1;
    int shots_ = 0;
    SDL_Window* window_ = nullptr;
    NavigationCamera camera_;
    bool navigationReady_ = false, exploring_ = false, motionPending_ = false;
    bool flight_ = false, lookHeld_ = false, imageHovered_ = false, walk_ = false;
    bool accommodate_ = true;  // the free eye focuses on what it looks at (through any optics)
    int drag_ = 0;
    float speed_ = 1;
    float precisionFactor_ = 0.2f;
    bool slowNearSurfaces_ = true;
    float relativeX_ = 0, relativeY_ = 0;
    Clock::time_point lastMotion_ = Clock::now();
};

void Viewer::refreshTexture(SDL_Renderer* ren) {
    bool fresh = worker_.poll(snap_);
    if (snap_.request == requestId_ && snap_.focusVersion != focusSeen_) {
        focusSeen_ = snap_.focusVersion;
        camera_ = snap_.camera;
        speed_ = float(std::clamp(camera_.distance * 0.5, 0.02, 10.0));
    }
    if (!navigationReady_ && snap_.request == requestId_ &&
        (snap_.state == "rendering" || snap_.state == "paused" || snap_.state == "converged")) {
        camera_ = snap_.camera;
        navigationReady_ = true;
        speed_ = float(std::clamp(camera_.distance * 0.5, 0.02, 10.0));
    }
    if (snap_.request == requestId_ && adoptSceneDisplay_ && snap_.state != "loading" && snap_.state != "error") {
        // A new view starts with the scene's own display settings for its detector: a fixed
        // exposure is manual; otherwise the automatic mode in use meters the new view (once, when
        // locked).
        exposure_.setMode(snap_.autoExposure ? autoMode_ : ExposureMode::Manual);
        exposure_.setCompensation(snap_.exposure);
        exposure_.reset();
        whiteBalance_ = float(snap_.whiteBalance);
        filmic_ = snap_.tone != "standard";
        adoptSceneDisplay_ = false;
        displayDirty_ = true;
    }
    // The exposure advances in UI time (not render passes); it meters only images of this request.
    const double dt = std::min(since(lastFrame_), 0.25);
    lastFrame_ = Clock::now();
    const Image& img = snap_.image;
    const bool imageReady = img.width > 0 && img.height > 0 && !img.xyz.empty() && snap_.request == requestId_;
    exposure_.update(fresh && imageReady ? &img : nullptr, snap_.spp, dt);
    if (exposure_.resolvedEV() != shownEv_) displayDirty_ = true;
    if (!fresh && !displayDirty_) return;
    if (img.width <= 0 || img.height <= 0 || img.xyz.empty()) {
        if (tex_) SDL_DestroyTexture(tex_);
        tex_ = nullptr;
        texW_ = texH_ = 0;
        return;
    }
    shownEv_ = exposure_.resolvedEV();
    std::vector<unsigned char> rgb = displayRGB8(img, shownEv_, false, whiteBalance_, filmic_ ? Tone::AgX : Tone::Standard);
    if (!tex_ || texW_ != img.width || texH_ != img.height) {
        if (tex_) SDL_DestroyTexture(tex_);
        tex_ = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, img.width, img.height);
        texW_ = img.width;
        texH_ = img.height;
    }
    SDL_SetTextureScaleMode(tex_, pixelated_ ? SDL_SCALEMODE_NEAREST : SDL_SCALEMODE_LINEAR);
    SDL_UpdateTexture(tex_, nullptr, rgb.data(), img.width * 3);
    shownVersion_ = snap_.version;
    displayDirty_ = false;
}

void Viewer::navigationInput(bool hovered, float imageHeight) {
    if (!navigationReady_ || ImGui::GetIO().WantTextInput) return;
    ImGuiIO& io = ImGui::GetIO();
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        lookHeld_ = true;
        SDL_SetWindowRelativeMouseMode(window_, true);
        tour_ = false;
    }
    if (lookHeld_ && !ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        lookHeld_ = false;
        SDL_SetWindowRelativeMouseMode(window_, flight_);
    }
    bool changed = false;
    double sensitivity = navigationSensitivity(io.KeyCtrl, precisionFactor_);
    // Zoom ([ and ], or Alt + wheel): a narrower field shows finer angular detail per pixel, as
    // a planet seen through an eyepiece needs; the eye itself does not move.
    const bool wheelZoom = io.KeyAlt && io.MouseWheel != 0 && (hovered || flight_ || lookHeld_);
    if (wheelZoom) zoom(std::pow(1.25, -double(io.MouseWheel)));
    if (!io.WantCaptureKeyboard || flight_ || lookHeld_) {
        if (ImGui::IsKeyPressed(ImGuiKey_RightBracket)) zoom(1 / 1.25);
        if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket)) zoom(1.25);
    }
    if (flight_ || lookHeld_) {
        if (relativeX_ != 0 || relativeY_ != 0) {
            camera_.turn(-relativeX_ * 0.003 * sensitivity, -relativeY_ * 0.003 * sensitivity, false);
            changed = true;
        }
        if (io.MouseWheel != 0 && !wheelZoom) speed_ = std::clamp(speed_ * std::exp(io.MouseWheel * 0.25f), 0.00001f, 100.0f);
        auto held = [](ImGuiKey key) { return ImGui::IsKeyDown(key) ? 1.0 : 0.0; };
        double x = held(ImGuiKey_D) - held(ImGuiKey_A);
        double y = held(ImGuiKey_W) - held(ImGuiKey_S);
        double z = held(ImGuiKey_E) - held(ImGuiKey_Q);
        double dt = std::min(double(io.DeltaTime), 0.05);
        if (x || y || z) {
            camera_.move(x, y, z, navigationSpeed(speed_, io.KeyCtrl, io.KeyShift, snap_.surfaceDistance,
                                                 precisionFactor_, slowNearSurfaces_) * dt, walk_);
            changed = true;
        }
        double roll = held(ImGuiKey_C) - held(ImGuiKey_Z);
        if (roll) {
            camera_.up = normalize(Mat3::rotation(camera_.forward, roll * dt * sensitivity) * camera_.up);
            changed = true;
        }
    } else {
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) drag_ = 1;
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Middle)) drag_ = 0;
        if (drag_ && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
            if (io.KeyShift) {
                double perPixel = 2 * camera_.distance * std::tan(camera_.fov * 0.5) / std::max(1.0f, imageHeight);
                camera_.pan(-io.MouseDelta.x * perPixel * sensitivity, io.MouseDelta.y * perPixel * sensitivity);
            } else if (io.KeyCtrl) {
                camera_.dolly(-io.MouseDelta.y * 0.05 * sensitivity);
            } else {
                camera_.turn(-io.MouseDelta.x * 0.006, -io.MouseDelta.y * 0.006, true);
            }
            changed = true;
        }
        if (hovered && io.MouseWheel != 0 && !wheelZoom) { camera_.dolly(io.MouseWheel * sensitivity); changed = true; }
    }
    if (changed) moved();
}

void Viewer::drawNavigation() {
    if (!ImGui::CollapsingHeader("Explore", ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (snap_.request == requestId_ && !snap_.eyepieces.empty()) {
        ImGui::TextUnformatted("Look through a telescope");
        // select() queues a new worker request; the polled target list stays stable this frame.
        for (const auto& target : snap_.eyepieces) {
            if (ImGui::Button(("Look through " + target.detector).c_str())) {
                for (int vi : sceneOf(current_)->views) {
                    if (cat_.views[size_t(vi)].detector == target.detector) { select(vi); tour_ = false; break; }
                }
            }
            ImGui::SetItemTooltip("Places your eye at the computed exit pupil, aligns the telescope axis, and adopts its eye focus, field of view and exposure. Eye pupil %.2f mm.", target.pupilDiameter * 1000);
        }
        int nearby = nearbyEyepiece();
        if (nearby >= 0) {
            const auto& name = cat_.views[size_t(nearby)].detector;
            ImGui::TextWrapped("Nearby: %s. Enter to look through, or stay free and zoom with [ ].", name.c_str());
            for (const auto& target : snap_.eyepieces) if (target.detector == name) {
                Vec3 offset = camera_.position - target.position;
                double axial = dot(offset, target.towardObjective);
                double lateral = length(offset - target.towardObjective * axial);
                ImGui::TextWrapped("From exit pupil: %.2f mm %s, %.2f mm off axis",
                                   1000 * std::abs(axial), axial >= 0 ? "toward glass" : "behind", 1000 * lateral);
                ImGui::SetItemTooltip("The exit pupil is where the instrument's outgoing image bundle meets the eye; it is separate from the glass surface. These distances only measure your free position.");
                ImGui::Text("View direction: %.2f deg from axis",
                            degrees(std::acos(clampd(dot(camera_.forward, target.towardObjective), -1, 1))));
            }
        }
        ImGui::Separator();
    }
    ImGui::TextWrapped("Middle drag: orbit | Shift + middle: pan | Wheel: dolly");
    ImGui::TextWrapped("[ / ] or Alt + wheel: zoom (field of view)");
    ImGui::TextWrapped("Shift + click: set orbit center and eye focus");
    ImGui::TextWrapped("Hold right mouse: look + WASD move, Q/E down/up");
    ImGui::TextWrapped("Shift: faster | Ctrl: precision move/look | Z/C: roll | Esc: release");
    ImGui::BeginDisabled(!navigationReady_);
    if (ImGui::Button(flight_ ? "Stop flying" : "Fly (Shift+F)")) {
        flight_ = !flight_;
        SDL_SetWindowRelativeMouseMode(window_, flight_);
        tour_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset (Home)")) select(current_);
    ImGui::Checkbox("walk at constant height", &walk_);
    ImGui::SetItemTooltip("WASD stays horizontal; Q/E changes height. Movement has no collisions.");
    ImGui::SliderFloat("speed (m/s)", &speed_, 0.00001f, 100.0f, "%.5g", ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("slow near surfaces", &slowNearSurfaces_);
    ImGui::SliderFloat("Ctrl speed / look", &precisionFactor_, 0.02f, 1.0f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ImGui::Text("Ctrl movement: %.3g mm/s", 1000 * navigationSpeed(speed_, true, false, snap_.surfaceDistance,
                                                                  precisionFactor_, slowNearSurfaces_));
    float fov = float(degrees(camera_.fov));
    if (ImGui::SliderFloat("field of view", &fov, 0.1f, 120, "%.3g deg", ImGuiSliderFlags_Logarithmic)) {
        camera_.fov = radians(fov);
        moved();
    }
    ImGui::SetItemTooltip("A zoom: the eye stays in place. Narrow it to see a planet's detail through an eyepiece.");
    float pupil = float(camera_.pupil * 2000);
    if (ImGui::SliderFloat("eye pupil", &pupil, 0, 10, "%.2f mm")) { camera_.pupil = pupil / 2000; moved(); }
    ImGui::SetItemTooltip("Eye pupil diameter. Small pupils help align with telescope exit pupils; 0 is a pinhole.");
    if (ImGui::Checkbox("eye accommodates", &accommodate_)) {
        // Manual focus continues from where the eye was focused.
        if (!accommodate_ && snap_.request == requestId_ && snap_.accommodated) camera_.focus = snap_.camera.focus;
        moved();
    }
    ImGui::SetItemTooltip("Like a real eye, it focuses on what is at the centre of the view. Through a telescope, "
                          "a mirror or a magnifier that is the image they form, not the glass in front of you. "
                          "Turn off to focus by hand.");
    if (accommodate_) {
        double f = snap_.request == requestId_ && snap_.accommodated ? snap_.camera.focus : camera_.focus;
        if (std::isfinite(f)) ImGui::TextDisabled("eye focus: %.4g m", f);
        else ImGui::TextDisabled("eye focus: infinity");
    } else {
        bool infinity = !std::isfinite(camera_.focus);
        if (ImGui::Checkbox("eye focus at infinity", &infinity)) {
            camera_.focus = infinity ? Inf : camera_.distance;
            moved();
        }
        if (!infinity) {
            float focus = float(camera_.focus);
            if (ImGui::SliderFloat("eye focus", &focus, 0.001f, 1000, "%.4g m", ImGuiSliderFlags_Logarithmic)) {
                camera_.focus = focus;
                moved();
            }
        }
    }
    float distance = float(camera_.distance);
    if (ImGui::SliderFloat("orbit distance", &distance, 0.00001f, 100, "%.5g m", ImGuiSliderFlags_Logarithmic))
        camera_.distance = distance;
    if (ImGui::TreeNode("Position / orientation")) {
        double xyz[] = {camera_.position.x, camera_.position.y, camera_.position.z};
        if (ImGui::InputScalarN("position (m)", ImGuiDataType_Double, xyz, 3, nullptr, nullptr, "%.9g",
                               ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (std::isfinite(xyz[0]) && std::isfinite(xyz[1]) && std::isfinite(xyz[2])) {
                camera_.position = {xyz[0], xyz[1], xyz[2]};
                moved();
            }
        }
        ImGui::Text("looking %.3f, %.3f, %.3f", camera_.forward.x, camera_.forward.y, camera_.forward.z);
        ImGui::TextWrapped("Numpad 1/3/7: front/right/top. Ctrl reverses the view.");
        ImGui::TreePop();
    }
    ImGui::EndDisabled();
    ImGui::TextWrapped("Choose an Eyepiece view below to align with a telescope, then move freely from there.");
    ImGui::Separator();
}

void Viewer::drawSidebar(float width) {
    ImGui::BeginChild("sidebar", ImVec2(width, 0), ImGuiChildFlags_Borders);
    const ViewEntry* v = view();
    drawNavigation();

    if (ImGui::CollapsingHeader("Views", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled("Left/Right: scene  Up/Down: view  T: tour");
        for (size_t s = 0; s < cat_.scenes.size(); ++s) {
            const SceneEntry& e = cat_.scenes[s];
            bool holds = v && v->scene == int(s);
            if (holds) ImGui::SetNextItemOpen(true, ImGuiCond_Appearing);
            bool open = ImGui::TreeNodeEx(e.path.c_str(), holds ? ImGuiTreeNodeFlags_Selected : 0, "%s", e.title.c_str());
            if (!e.blurb.empty() && ImGui::BeginItemTooltip()) {
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30);
                ImGui::TextUnformatted(e.blurb.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            if (!open) continue;
            if (!e.error.empty()) ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%s", e.error.c_str());
            for (int vi : e.views) {
                const ViewEntry& ve = cat_.views[size_t(vi)];
                std::string label = ve.detector + (ve.detector == e.defaultDetector ? "  *" : "");
                if (ImGui::Selectable((label + "##" + std::to_string(vi)).c_str(), vi == current_)) select(vi);
                ImGui::SameLine(width * 0.62f);
                ImGui::TextDisabled("%s", ve.kind == "sensor" ? "readout" : ve.kind.c_str());
                if (ve.kind == "sensor") ImGui::SetItemTooltip("Incident irradiance on the sensor, not a camera viewpoint. Select an observer to see the physical screen.");
            }
            ImGui::TreePop();
        }
    }

    if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
        // One button per backend: the same scene and view, transported elsewhere.
        for (size_t i = 0; i < usable_.size(); ++i) {
            if (i) ImGui::SameLine();
            const Backend* b = usable_[i];
            if (ImGui::RadioButton(b->name().c_str(), backend_ == b->name())) switchBackend(b->name());
            ImGui::SetItemTooltip("%s", b->summary().c_str());
        }
        for (const auto& [b, why] : unusable_) {
            ImGui::SameLine();
            ImGui::BeginDisabled();
            ImGui::RadioButton(b->name().c_str(), false);
            ImGui::EndDisabled();
            ImGui::SetItemTooltip("%s", why.c_str());
        }
        const auto& devices = devices_[backend_];
        if (devices.size() > 1) {
            std::string cur = "best available";
            for (const auto& d : devices) if (d.index == device_) cur = d.name;
            if (ImGui::BeginCombo("device", cur.c_str())) {
                if (ImGui::Selectable("best available", device_ < 0)) { device_ = -1; send(); }
                for (auto& d : devices)
                    if (ImGui::Selectable(d.name.c_str(), device_ == d.index)) { device_ = d.index; send(); }
                ImGui::EndCombo();
            }
        }
        ImGui::SliderFloat("resolution", &scale_, 0.1f, 1.5f, "%.2f ×");
        if (ImGui::IsItemDeactivatedAfterEdit()) send();
        ImGui::SetItemTooltip("relative to the detector's own resolution");
        if (backend(backend_).supports("hybrid") || backend(backend_).supports("light")) {
            if (ImGui::Checkbox("path integrator only", &pathOnly_)) send();
            ImGui::SetItemTooltip("otherwise the scene's integrator (light or hybrid) is used");
        }
        const char* stops[] = {"256", "1024", "4096", "16384", "never"};
        if (ImGui::Combo("stop at spp", &maxSppIndex_, stops, 5)) {
            const long long n[] = {256, 1024, 4096, 16384, 0};
            worker_.setMaxSpp(n[maxSppIndex_]);
        }
        if (ImGui::Button(paused_ ? "Resume" : "Pause")) worker_.setPaused(paused_ = !paused_);
        ImGui::SameLine();
        if (ImGui::Button("Restart")) send();
        ImGui::SameLine();
        if (ImGui::Button("Save")) saveCurrent();
        ImGui::SetItemTooltip("out/view_<scene>_<detector>.png, .pfm, .json");
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
        int mode = int(exposure_.mode());
        bool changed = ImGui::RadioButton("manual", &mode, int(ExposureMode::Manual));
        ImGui::SetItemTooltip("A fixed exposure.");
        ImGui::SameLine();
        changed |= ImGui::RadioButton("locked", &mode, int(ExposureMode::Locked));
        ImGui::SetItemTooltip("Meter each new view once, then keep that exposure while you move (L meters again).");
        ImGui::SameLine();
        changed |= ImGui::RadioButton("adaptive", &mode, int(ExposureMode::Adaptive));
        ImGui::SetItemTooltip("Follow the meter gradually, at most %.1f EV/s brighter and %.1f EV/s darker.",
                              exposure_.rates.brighten, exposure_.rates.darken);
        if (changed) setExposureMode(ExposureMode(mode));
        float ev = float(exposure_.compensation());
        const bool manual = exposure_.mode() == ExposureMode::Manual;
        if (ImGui::SliderFloat(manual ? "exposure" : "compensation", &ev, -20, 32, "%+.1f EV")) {
            exposure_.setCompensation(ev);
            displayDirty_ = true;
        }
        if (manual) {
            ImGui::Text("shown: %+.1f EV", exposure_.resolvedEV());
        } else {
            ImGui::Text("meter %+.1f EV · shown %+.1f EV%s", exposure_.automaticEV(), exposure_.resolvedEV(),
                        exposure_.adapting() ? " · adapting" : exposure_.locked() ? " · locked" : "");
            ImGui::SameLine();
            if (ImGui::SmallButton("meter now")) exposure_.remeter();
        }
        if (ImGui::SliderFloat("white", &whiteBalance_, 0, 12000, whiteBalance_ > 0 ? "%.0f K" : "off")) displayDirty_ = true;
        ImGui::SetItemTooltip("white point (Planckian); 0 = none");
        if (ImGui::Checkbox("filmic", &filmic_)) displayDirty_ = true;
        ImGui::SetItemTooltip("AgX view transform: bright colours go gradually to white over 16.5 stops;\n"
                              "off: linear to white with a short roll-off");
        ImGui::SameLine();
        if (ImGui::Checkbox("pixelated", &pixelated_)) displayDirty_ = true;
        ImGui::SameLine();
        if (ImGui::Button("scene defaults")) adoptSceneDisplay_ = true;
        ImGui::TextDisabled("display only: the raw estimate is never altered");
    }

    if (v && v->kind == "camera" && ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("f-number", &fNumber_, 1.0f, 32.0f, "f/%.1f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemDeactivatedAfterEdit()) setEdit(v->detector + ".f_number", std::to_string(fNumber_));
        ImGui::SliderFloat("focus", &focus_, 0.3f, 1000.0f, "%.2f m", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemDeactivatedAfterEdit()) setEdit(v->detector + ".focus", std::to_string(focus_) + " m");
        ImGui::TextDisabled("the stop resizes and the sensor moves to real focus");
    }

    if (v && ImGui::CollapsingHeader("Scene edits")) {
        ImGui::TextDisabled("Block.key=value, e.g. Lens.medium=N-SF11");
        bool apply = ImGui::InputText("##edit", editBuf_, sizeof editBuf_, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        apply |= ImGui::Button("Apply");
        if (apply && std::strchr(editBuf_, '=')) {
            std::string e = editBuf_;
            setEdit(e.substr(0, e.find('=')), e.substr(e.find('=') + 1));
            editBuf_[0] = 0;
        }
        auto& ed = edits_[sceneOf(current_)->path];
        for (size_t i = 0; i < ed.size(); ++i) {
            ImGui::PushID(int(i));
            bool remove = ImGui::SmallButton("x");
            ImGui::SameLine();
            ImGui::TextUnformatted(ed[i].c_str());
            ImGui::PopID();
            if (remove) {
                ed.erase(ed.begin() + long(i));
                send();
                break;
            }
        }
    }

    if (ImGui::CollapsingHeader("Tour")) {
        ImGui::Checkbox("cycle through every view", &tour_);
        ImGui::SliderFloat("seconds per view", &tourSeconds_, 2, 120, "%.0f s");
        if (tour_) ImGui::Text("next view in %.0f s", std::max(0.0, tourSeconds_ - since(viewStart_)));
    }

    ImGui::Separator();
    if (v) {
        const SceneEntry& e = *sceneOf(current_);
        ImGui::Text("%s · %s", e.title.c_str(), v->detector.c_str());
        ImGui::TextWrapped("%s", snap_.description.c_str());
        if (!snap_.note.empty()) ImGui::TextWrapped("%s", snap_.note.c_str());
    }
    if (snap_.request == requestId_) {
        if (snap_.state == "error") ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "error: %s", snap_.error.c_str());
        ImGui::Text("%s — %lld spp, %d passes, %.1f s", snap_.state.c_str(), snap_.spp, snap_.passes, snap_.seconds);
        if (snap_.seconds > 0)
            ImGui::Text("%.2f M paths/s, %.2f M segments/s", snap_.stats.paths / snap_.seconds * 1e-6,
                        snap_.stats.segments / snap_.seconds * 1e-6);
        ImGui::Text("%dx%d, %s integrator", snap_.width, snap_.height, snap_.integrator.c_str());
        if (snap_.stats.inconsistencies || snap_.stats.leaks)
            ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "region inconsistencies %llu, leaks %llu",
                               (unsigned long long)snap_.stats.inconsistencies, (unsigned long long)snap_.stats.leaks);
        ImGui::TextWrapped("%s", snap_.backend.c_str());
    } else {
        ImGui::TextWrapped("loading… %s", snap_.request == requestId_ ? snap_.phase.c_str() : "");
    }
    if (!snap_.message.empty()) ImGui::TextWrapped("%s", snap_.message.c_str());
    ImGui::TextDisabled("click a pixel: why is it this colour?");
    ImGui::TextDisabled("Space pause  R restart  S save  G next backend  H panel  F full screen");
    ImGui::EndChild();
}

void Viewer::drawImage() {
    ImGui::BeginChild("image", ImVec2(0, 0), 0, ImGuiWindowFlags_NoScrollbar);
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    bool current = snap_.request == requestId_;
    if (tex_ && texW_ > 0 && avail.x > 1 && avail.y > 1) {
        float s = std::min(avail.x / texW_, avail.y / texH_);
        ImVec2 size(texW_ * s, texH_ * s);
        ImVec2 pad((avail.x - size.x) * 0.5f, (avail.y - size.y) * 0.5f);
        ImGui::SetCursorScreenPos(ImVec2(origin.x + pad.x, origin.y + pad.y));
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        // Draw the viewport as an SDL texture copy. The software triangle path used
        // by ImGui::Image can lose half the image at fractional scaled coordinates.
        ImGui::InvisibleButton("rendered scene", size);
        imageDraw_ = {tex_, SDL_FRect{p0.x, p0.y, size.x, size.y}};
        ImGui::GetWindowDrawList()->AddCallback(drawImageTexture, &imageDraw_);
        imageHovered_ = ImGui::IsItemHovered();
        navigationInput(imageHovered_, size.y);
        if (imageHovered_ && !flight_ && !lookHeld_ && drag_ == 0 && current && !snap_.image.xyz.empty()) {
            ImVec2 m = ImGui::GetMousePos();
            int px = std::clamp(int((m.x - p0.x) / s), 0, texW_ - 1), py = std::clamp(int((m.y - p0.y) / s), 0, texH_ - 1);
            if (px < snap_.image.width && py < snap_.image.height) {
                XYZ c = snap_.image.at(px, py);
                ImGui::SetTooltip("pixel (%d, %d)\nX %.4g  Y %.4g  Z %.4g\nclick: why is this pixel this colour?", px, py, c.x, c.y,
                                  c.z);
                if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                    if (ImGui::GetIO().KeyShift) {
                        worker_.focus(camera_, (px + 0.5) / texW_, (py + 0.5) / texH_);
                        exploring_ = true;
                        motionPending_ = tour_ = false;
                    } else {
                        probeSeen_ = snap_.probeVersion;
                        worker_.probe(px, py);
                        probeOpen_ = true;
                    }
                }
            }
        }
    }
    // Status overlay.
    const ViewEntry* v = view();
    if (v) {
        std::string line = sceneOf(current_)->title + " · " + v->detector + "   " + backend_ + "   ";
        char buf[160];
        if (!current || snap_.state == "loading") {
            double waited = since(requestStart_);
            std::snprintf(buf, sizeof buf, "loading… %s (%.0f s)", current ? snap_.phase.c_str() : "", waited);
            // A GPU driver compiles the kernels the first time it sees them (after an update too),
            // alone on one core: minutes on an integrated GPU, with nothing else to show for it.
            if (current && waited > 5 && snap_.phase.rfind("preparing the gpu", 0) == 0)
                line += std::string(buf) + " — the GPU driver is compiling the kernels (first run after a "
                        "driver update; minutes on an integrated GPU, cached afterwards)";
            else
                line += buf;
            buf[0] = 0;
        } else
            std::snprintf(buf, sizeof buf, "%lld spp · %.1f s%s", snap_.spp, snap_.seconds,
                          snap_.state == "paused" ? " · paused" : snap_.state == "converged" ? " · done" : "");
        line += buf;
        if (!exploring_ && v->kind == "sensor") line += "   irradiance readout";
        if (exploring_) {
            line += motionPending_ ? "   moving (preview)" : "   exploring";
            char fov[48];
            std::snprintf(fov, sizeof fov, " · fov %.3g°", degrees(camera_.fov));
            line += fov;
        }
        char ev[64];
        std::snprintf(ev, sizeof ev, "   %s %+.1f EV%s", exposureModeName(exposure_.mode()), exposure_.resolvedEV(),
                      exposure_.adapting() ? " (adapting)" : "");
        line += ev;
        if (flight_ || lookHeld_) line += "   WASD move | Q/E down/up | Esc release";
        if (tour_) line += "   tour";
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 ts = ImGui::CalcTextSize(line.c_str());
        ImVec2 a(origin.x + 8 * uiScale_, origin.y + 8 * uiScale_);
        dl->AddRectFilled(ImVec2(a.x - 6 * uiScale_, a.y - 4 * uiScale_), ImVec2(a.x + ts.x + 6 * uiScale_, a.y + ts.y + 4 * uiScale_),
                          IM_COL32(0, 0, 0, 150), 4 * uiScale_);
        dl->AddText(a, IM_COL32(235, 235, 235, 255), line.c_str());
    }
    ImGui::EndChild();
}

void Viewer::drawProbe() {
    if (!probeOpen_) return;
    ImGui::SetNextWindowSize(ImVec2(900 * uiScale_, 420 * uiScale_), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Why is this pixel this colour?", &probeOpen_)) {
        if (snap_.probeVersion == probeSeen_) {
            ImGui::Text("tracing 2000 paths through the pixel…");
        } else {
            ImGui::Text("pixel (%d, %d) — transport classes ranked by contribution", snap_.probeX, snap_.probeY);
            ImGui::BeginChild("probe", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar);
            if (mono_) ImGui::PushFont(mono_, 0.0f);
            ImGui::TextUnformatted(snap_.probeText.c_str());
            if (mono_) ImGui::PopFont();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void Viewer::handleKeys() {
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) stopFlight();
    if (io.WantTextInput) return;
    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        int nearby = nearbyEyepiece();
        if (nearby >= 0) { select(nearby); tour_ = false; return; }
    }
    if (navigationReady_ && io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        flight_ = !flight_;
        if (window_) SDL_SetWindowRelativeMouseMode(window_, flight_);
        tour_ = false;
    }
    // Movement keys must not also trigger save, exposure or quit shortcuts.
    if (flight_ || lookHeld_ || (imageHovered_ && io.MouseDown[ImGuiMouseButton_Right])) return;
    if (ImGui::IsKeyPressed(ImGuiKey_Home, false)) select(current_);
    if (navigationReady_ && !io.WantCaptureKeyboard) {
        bool axis = true;
        const double side = io.KeyCtrl ? -1 : 1;
        if (ImGui::IsKeyPressed(ImGuiKey_Keypad1)) camera_.axis({0, side, 0}, {0, 0, 1});
        else if (ImGui::IsKeyPressed(ImGuiKey_Keypad3)) camera_.axis({-side, 0, 0}, {0, 0, 1});
        else if (ImGui::IsKeyPressed(ImGuiKey_Keypad7)) camera_.axis({0, 0, -side}, {0, 1, 0});
        else axis = false;
        if (axis) moved();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) stepScene(+1);
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) stepScene(-1);
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) step(+1);
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) step(-1);
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) worker_.setPaused(paused_ = !paused_);
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) send();
    if (ImGui::IsKeyPressed(ImGuiKey_S, false)) saveCurrent();
    if (ImGui::IsKeyPressed(ImGuiKey_G, false) && usable_.size() > 1) {
        size_t i = 0;
        while (i < usable_.size() && usable_[i]->name() != backend_) ++i;
        switchBackend(usable_[(i + 1) % usable_.size()]->name());
    }
    if (ImGui::IsKeyPressed(ImGuiKey_T, false)) {
        tour_ = !tour_;
        viewStart_ = Clock::now();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_H, false)) showPanel_ = !showPanel_;
    if (ImGui::IsKeyPressed(ImGuiKey_A, false)) setExposureMode(ExposureMode((int(exposure_.mode()) + 1) % 3));
    if (ImGui::IsKeyPressed(ImGuiKey_L, false)) exposure_.remeter();
    if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd)) addExposure(0.5);
    if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract)) addExposure(-0.5);
    if (ImGui::IsKeyPressed(ImGuiKey_Q, false)) quit_ = true;
}

bool Viewer::screenshot(SDL_Renderer* ren, const std::string& path) {
    SDL_Surface* s = SDL_RenderReadPixels(ren, nullptr);
    if (!s) return false;
    SDL_Surface* rgb = SDL_ConvertSurface(s, SDL_PIXELFORMAT_RGB24);
    SDL_DestroySurface(s);
    if (!rgb) return false;
    std::vector<unsigned char> px(size_t(rgb->w) * rgb->h * 3);
    for (int y = 0; y < rgb->h; ++y)
        std::memcpy(&px[size_t(y) * rgb->w * 3], static_cast<const unsigned char*>(rgb->pixels) + size_t(y) * rgb->pitch,
                    size_t(rgb->w) * 3);
    std::error_code ec;
    if (fs::path(path).has_parent_path()) fs::create_directories(fs::path(path).parent_path(), ec);
    writeRGB8PNG(path, rgb->w, rgb->h, px);
    SDL_DestroySurface(rgb);
    return true;
}

int Viewer::run() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "owe view: cannot open a window: %s\n", SDL_GetError());
        return 2;
    }
    float dpi = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    uiScale_ = dpi > 0 ? dpi : 1.0f;
    SDL_Window* win = SDL_CreateWindow("Optical World Engine", int(opt_.width * uiScale_), int(opt_.height * uiScale_),
                                       SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN);
    if (!win) {
        std::fprintf(stderr, "owe view: %s\n", SDL_GetError());
        SDL_Quit();
        return 2;
    }
    window_ = win;
    SDL_SetWindowMinimumSize(win, 900, 600);
    SDL_Renderer* ren = SDL_CreateRenderer(win, nullptr);
    if (!ren) {
        std::fprintf(stderr, "owe view: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 2;
    }
    SDL_SetRenderVSync(ren, 1);
    SDL_SetWindowPosition(win, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(win);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // no imgui.ini next to the scenes
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(uiScale_);
    style.FontScaleDpi = uiScale_;
    style.FontSizeBase = 16.0f;
    style.WindowRounding = 0;
    ImFont* ui = loadFont({"/usr/share/fonts/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
                           "/usr/share/fonts/liberation/LiberationSans-Regular.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                           "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/segoeui.ttf"},
                          16.0f);
    if (!ui) io.Fonts->AddFontDefaultVector();
    mono_ = loadFont({"/usr/share/fonts/noto/NotoSansMono-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                      "/usr/share/fonts/liberation/LiberationMono-Regular.ttf", "/System/Library/Fonts/Menlo.ttc",
                      "C:/Windows/Fonts/consola.ttf"},
                     15.0f);
    ImGui_ImplSDL3_InitForSDLRenderer(win, ren);
    ImGui_ImplSDLRenderer3_Init(ren);

    worker_.setMaxSpp(4096);
    const bool shotDir = !opt_.screenshot.empty() && opt_.screenshot.back() == '/';
    bool fullscreen = false;
    int result = 0;
    while (!quit_) {
        SDL_Event ev;
        relativeX_ = relativeY_ = 0;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL3_ProcessEvent(&ev);
            if (ev.type == SDL_EVENT_MOUSE_MOTION && (flight_ || lookHeld_)) {
                relativeX_ += ev.motion.xrel;
                relativeY_ += ev.motion.yrel;
            }
            if (ev.type == SDL_EVENT_WINDOW_FOCUS_LOST) { stopFlight(); drag_ = 0; }
            if (ev.type == SDL_EVENT_QUIT) quit_ = true;
            if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && ev.window.windowID == SDL_GetWindowID(win)) quit_ = true;
            if (ev.type == SDL_EVENT_KEY_DOWN && !io.WantTextInput && !flight_ && !lookHeld_ &&
                !(ev.key.mod & SDL_KMOD_SHIFT) && (ev.key.key == SDLK_F || ev.key.key == SDLK_F11)) {
                fullscreen = !fullscreen;
                SDL_SetWindowFullscreen(win, fullscreen);
            }
        }
        if (SDL_GetWindowFlags(win) & SDL_WINDOW_MINIMIZED) {
            SDL_Delay(20);
            continue;
        }
        refreshTexture(ren);

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        handleKeys();
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("owe", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                         ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar();
        if (showPanel_) {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8 * uiScale_, 8 * uiScale_));
            drawSidebar(std::min(380 * uiScale_, vp->WorkSize.x * 0.45f));
            ImGui::PopStyleVar();
            ImGui::SameLine(0, 0);
        }
        drawImage();
        if (motionPending_ && since(lastMotion_) > 0.25) {
            worker_.navigate(camera_, false, accommodate_);
            motionPending_ = false;
        }
        ImGui::End();
        drawProbe();
        ImGui::Render();

        SDL_SetRenderScale(ren, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
        SDL_SetRenderDrawColorFloat(ren, 0.06f, 0.06f, 0.07f, 1.0f);
        SDL_RenderClear(ren);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), ren);

        // Tour and screenshots: a view is due once it has rendered for its time (and shows an image).
        const bool shown = snap_.request == requestId_ && snap_.spp > 0;
        const double wait = opt_.screenshot.empty() ? tourSeconds_ : opt_.after;
        if (shown && !opt_.screenshot.empty() && since(viewStart_) >= wait) {
            std::string path = opt_.screenshot;
            if (shotDir) {
                const ViewEntry& e = cat_.views[size_t(current_)];
                char name[256];
                std::snprintf(name, sizeof name, "%02d_%s_%s.png", shots_,
                              fileSafe(fs::path(sceneOf(current_)->path).stem().string()).c_str(), fileSafe(e.detector).c_str());
                path += name;
            }
            if (!screenshot(ren, path)) {
                std::fprintf(stderr, "owe view: screenshot failed: %s\n", SDL_GetError());
                result = 2;
                quit_ = true;
            } else {
                std::fprintf(stderr, "wrote %s\n", path.c_str());
            }
            ++shots_;
            if (!shotDir || !tour_ || shots_ >= int(cat_.views.size())) quit_ = true;
            else step(+1);
        } else if (tour_ && opt_.screenshot.empty() && shown && since(viewStart_) >= tourSeconds_) {
            step(+1);
        }
        if (!opt_.screenshot.empty() && snap_.request == requestId_ && snap_.state == "error") {
            std::fprintf(stderr, "owe view: %s\n", snap_.error.c_str());
            result = 2;
            quit_ = true;
        }
        SDL_RenderPresent(ren);
        SDL_Delay(1);
    }

    stopFlight();
    window_ = nullptr;
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    if (tex_) SDL_DestroyTexture(tex_);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return result;
}

}  // namespace

int runViewer(const ViewerOptions& options) {
    if (!options.backend.empty()) backend(options.backend);  // throws for unknown names
    if (!std::isfinite(options.scale) || options.scale < 0 || options.scale > 1.5)
        throw std::runtime_error("--scale must be between 0 (automatic) and 1.5");
    if (!std::isfinite(options.tourSeconds) || options.tourSeconds <= 0 || !std::isfinite(options.after) || options.after < 0)
        throw std::runtime_error("tour duration must be positive and screenshot delay nonnegative");
    if (options.device < -1) throw std::runtime_error("--device must be -1 (automatic) or a device index");
    Viewer v(options);
    return v.run();
}

}  // namespace owe
