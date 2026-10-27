// Observers, cameras and sensors of the .owe language, and the display settings they carry.
#include <tuple>

#include "owe/loader/loader.hpp"

namespace owe {

Transform cameraPlacement(const Vec3& position, const Vec3& lookAt, const Vec3& up) {
    Vec3 f = normalize(lookAt - position);
    return Transform::translate(position) * Transform::alignZ(-f, up);
}

namespace {
// Resizes the physical aperture stop so that the entrance pupil gives EFL / (2N). The entrance
// pupil scales linearly with the stop, paraxially. Lenses without a stop get an iris 2 mm in front
// of their first surface.
void setFNumber(Prescription& lens, double fNumber, const IndexFn& index) {
    int stop = stopIndex(lens);
    if (stop < 0) {
        PrescriptionSurface iris;
        iris.label = "IRIS";
        iris.t = 2e-3;
        iris.stop = true;
        iris.sd = lens.surfaces[0].sd;
        lens.surfaces.insert(lens.surfaces.begin(), iris);
        stop = 0;
    }
    Paraxial p0 = paraxialAnalysis(atInfinity(lens), lens.wavelength, index);
    double target = std::abs(p0.efl) / (2 * fNumber);
    lens.surfaces[size_t(stop)].sd *= target / p0.entrancePupilRadius;
}

// Focus like a real camera: at the plane of least blur for this aperture (traced with real rays
// through the lens bodies), not at the paraxial image. The correction is the on-axis best-focus
// shift measured for a distant object.
double realFocusShift(const Prescription& lens) {
    LensReport rep = analyzeLens(atInfinity(lens), {0}, {lens.wavelength}, 6);
    if (!rep.spots.empty() && rep.spots[0].arrived >= 3 && !std::isinf(rep.paraxial.imageDistance))
        return rep.spots[0].bestFocusZ - rep.paraxial.imageDistance;
    return 0;
}

// Depth of field from thin-lens relations with a circle of confusion of 1/1500 of the diagonal.
std::string depthOfFieldNote(const Prescription& lens, double focusDistance, double halfDiag, const IndexFn& index) {
    Paraxial pi = paraxialAnalysis(atInfinity(lens), lens.wavelength, index);
    double f = std::abs(pi.efl), N = pi.fNumber, coc = 2 * halfDiag / 1500;
    double Hf = f * f / (N * coc) + f;
    char buf[512];
    double s = focusDistance;
    if (std::isinf(s)) {
        std::snprintf(buf, sizeof buf, "EFL %.1f mm, f/%.2f, focus infinity; sharp beyond %.2f m (hyperfocal)", f * 1e3, N, Hf);
    } else {
        double nearD = s * (Hf - f) / (Hf + s - 2 * f);
        double farD = s < Hf ? s * (Hf - f) / (Hf - s) : Inf;
        char far[32] = "infinity";
        if (!std::isinf(farD)) std::snprintf(far, sizeof far, "%.3f m", farD);
        std::snprintf(buf, sizeof buf, "EFL %.1f mm, f/%.2f, focus %.3f m; depth of field %.3f m to %s (CoC %.0f µm)",
                      f * 1e3, N, s, nearD, far, coc * 1e6);
    }
    return buf;
}

// The sensor at zSensor in camera assembly cam, facing the lens; its samples aim at the rear of the
// lens, most of them at the exit pupil. Returns the detector's index.
int addCameraSensor(Scene& scene, const std::string& name, const CameraParams& c, int cam, double zSensor,
                    const BuiltInstrument& bi, const Paraxial& px, double halfDiag) {
    World& w = scene.world;
    // Sensor surface facing the lens: rotated π about y, so sensor x = −x_cam, y = y_cam, normal −z.
    // The lens inverts the image; reading rows bottom-up (flipY) yields an upright, unmirrored picture.
    int sensorBody = w.addBody(name + ".sensor", "sensor", cam, Transform::translate({0, 0, zSensor}));
    int detIndex = int(scene.detectors.size());
    SurfaceOptics so;
    so.name = name + ".sensor";
    so.type = SurfaceType::Detector;
    so.detector = detIndex;
    uint32_t oi = w.addOptics(so);
    uint32_t bidx = w.addBoundary(sensorBody, "pixels", PlaneShape::rect(c.sensorWidth / 2, c.sensorHeight / 2),
                                  Transform::rotate({0, 1, 0}, Pi), kOutside, kOutside, oi);
    auto det = std::make_unique<SurfaceSensor>();
    det->name = name;
    det->width = c.width;
    det->height = c.height;
    det->boundaryIndex = bidx;
    det->halfX = c.sensorWidth / 2;
    det->halfY = c.sensorHeight / 2;
    det->flipY = true;
    det->hasAim = true;
    Transform camToWorld = w.assemblyToWorld(cam);
    det->aimNormal = normalize(camToWorld.vector({0, 0, 1}));
    det->aimCenter = camToWorld.point({0, 0, bi.rearNearestZ});
    det->aimRadius = std::max(halfDiag, bi.rearEdgeRadius) * 1.0001;
    // Image-forming light reaches the sensor from the exit pupil (the stop imaged by the rear
    // group), so 85% of samples aim there; at f/11 that wastes ~30× fewer rays on the iris. The
    // rest cover the whole rear opening, which also carries ghost reflections and veiling glare.
    double zExit = bi.vertexZ.empty() ? 0 : bi.vertexZ[0] + px.exitPupilZ;
    if (px.exitPupilRadius > 0 && zExit < zSensor - 1e-3) {
        det->focusCenter = camToWorld.point({0, 0, zExit});
        det->focusRadius = px.exitPupilRadius * 1.3;
        det->focusShare = 0.85;
    }
    scene.detectors.push_back(std::move(det));
    w.bodies()[sensorBody].params["z"] = zSensor;
    return detIndex;
}
}  // namespace

int addPhysicalCamera(Scene& scene, const std::string& name, const Prescription& lensIn, const CameraParams& c,
                      const Transform& placement, int parentAssembly) {
    World& w = scene.world;
    Prescription lens = lensIn;
    lens.objectDistance = c.focusDistance;
    IndexFn index = catalogIndex();
    if (c.fNumber > 0) setFNumber(lens, c.fNumber, index);
    Paraxial px = paraxialAnalysis(lens, lens.wavelength, index);
    if (px.afocal || std::isinf(px.imageDistance) || px.imageDistance <= 0)
        throw std::runtime_error("camera '" + name + "': lens does not form a real image for the requested focus");
    // The housing must enclose both the lens and the sensor; mounts close the gap between them.
    const double maxR = maxSemiDiameter(lens);
    double halfDiag = 0.5 * std::sqrt(sqr(c.sensorWidth) + sqr(c.sensorHeight));
    double tubeR = c.housing ? std::max(maxR * 1.08 + 1e-3, halfDiag * 1.05) : 0.0;
    int cam = w.addAssembly(name, parentAssembly, placement);
    BuiltInstrument bi = buildPrescription(w, lens, name + ".lens", cam, Transform{}, tubeR, index);
    double focusShift = c.realFocus ? realFocusShift(lens) : 0;
    double zSensor = bi.lastVertexZ + px.imageDistance + focusShift + c.sensorShift;
    if (c.housing) {
        uint32_t black = w.absorberOptics();
        double zFront = -1e-3;
        buildTube(w, name + ".housing", tubeR, zFront - 0.02 * maxR, zSensor + 2e-3, cam, Transform{}, black);
        buildStop(w, name + ".back", 0, tubeR, cam, Transform::translate({0, 0, zSensor + 2e-3}));
        w.bodies().back().kind = "housing";
    }
    int detIndex = addCameraSensor(scene, name, c, cam, zSensor, bi, px, halfDiag);
    scene.notes[name] = depthOfFieldNote(lens, c.focusDistance, halfDiag, index);
    return detIndex;
}

namespace detail {

// ------------------------------------------------------------ detectors
std::pair<int, int> Loader::resolution(const Value& b) const {
    auto v = b.get("resolution");
    if (!v) return {320, 240};
    auto s = seq(*v, 2);
    return {int(num(*s[0])), int(num(*s[1]))};
}

// Detector keys: `exposure = EV` (fixed, replaces auto exposure), `white_balance = K | none`,
// `tone = agx | standard` (the display view transform), `sun_share = s` (the sun's share of
// light samples while this detector renders).
void Loader::detectorDisplay(const Value& b) {
    DetectorSettings d;
    if (auto e = b.get("exposure")) { d.hasExposure = true; d.exposure = num(*e); }
    if (auto wb = b.get("white_balance")) {
        d.hasWhiteBalance = true;
        d.whiteBalance = (wb->kind == Value::Kind::Ident && wb->str == "none") ? 0 : num(*wb);
    }
    if (auto t = b.get("tone")) { d.hasTone = true; d.tone = toneName(*t); }
    if (auto ss = b.get("sun_share")) { d.hasSunShare = true; d.sunShare = num(*ss); }
    if (d.hasExposure || d.hasWhiteBalance || d.hasSunShare || d.hasTone) scene_.display[b.name] = d;
}

std::string Loader::toneName(const Value& v) const {
    std::string t = str(v);
    if (t != "agx" && t != "standard") fail(v, "tone must be agx or standard");
    return t;
}

// `pixel_filter = box | gaussian | gaussian(sigma)`: the pixel response (Detector::pixelSigma,
// in pixels). Observers default to gaussian, sensors to box.
void Loader::pixelFilter(const Value& b, Detector& d) {
    auto f = b.get("pixel_filter");
    if (!f) return;
    if (f->kind == Value::Kind::Ident && f->str == "box") d.pixelSigma = 0;
    else if (f->kind == Value::Kind::Ident && f->str == "gaussian") d.pixelSigma = IdealObserver::kDefaultPixelSigma;
    else if (f->kind == Value::Kind::Call && f->str == "gaussian" && f->items.size() == 1) d.pixelSigma = num(*f->items[0]);
    else fail(*f, "pixel_filter must be box, gaussian or gaussian(sigma in pixels)");
    if (!(d.pixelSigma >= 0 && d.pixelSigma <= 8)) fail(*f, "pixel filter sigma must be between 0 and 8 pixels");
}

void Loader::observer(const Value& b) {
    detectorDisplay(b);
    auto o = std::make_unique<IdealObserver>();
    o->name = b.name;
    auto [w, h] = resolution(b);
    o->width = w;
    o->height = h;
    o->position = vecLen(need(b, "position"));
    if (auto la = b.get("look_at")) o->lookAt = vecLen(*la);
    else if (auto d = b.get("direction")) o->lookAt = o->position + normalize(vecNum(*d));
    else fail(b, "observer needs look_at or direction");
    o->up = upOr(b);
    o->fovY = getAng(b, "fov", radians(40));
    o->pupilRadius = getRadius(b, "pupil_", -1);
    if (o->pupilRadius < 0) o->pupilRadius = b.has("pupil") ? len(need(b, "pupil")) / 2 : 0;
    if (auto f = b.get("focus")) o->focusDistance = lenOrInf(*f);
    pixelFilter(b, *o);
    scene_.detectors.push_back(std::move(o));
}

void Loader::camera(const Value& b, int assembly) {
    detectorDisplay(b);
    Prescription p = loadPrescription(path(str(need(b, "lens"))));
    CameraParams c;
    if (auto s = b.get("sensor")) {
        const Vec2 size = lengths2(*s);
        c.sensorWidth = size.x;
        c.sensorHeight = size.y;
    }
    auto [w, h] = resolution(b);
    c.width = w;
    c.height = h;
    if (auto f = b.get("focus")) c.focusDistance = lenOrInf(*f);
    c.sensorShift = getLen(b, "sensor_shift", 0);
    c.fNumber = getNum(b, "f_number", 0);
    if (auto rf = b.get("real_focus")) c.realFocus = boolean(*rf);
    if (auto hs = b.get("housing")) c.housing = boolean(*hs);
    Vec3 pos = vecLen(need(b, "position"));
    Vec3 la = vecLen(need(b, "look_at"));
    Vec3 up = upOr(b);
    addPhysicalCamera(scene_, b.name, p, c, cameraPlacement(pos, la, up), assembly);
    pixelFilter(b, *scene_.detectors.back());
}

void Loader::sensor(const Value& b, int assembly) {
    const Vec2 half = halfSize(need(b, "size"));
    int body = w_.addBody(b.name, "sensor", assembly, placement(b));
    int detIndex = int(scene_.detectors.size());
    SurfaceOptics so;
    so.type = SurfaceType::Detector;
    if (auto material = b.get("material")) {
        so = w_.optics()[materialRef(*material)];
        if (so.type != SurfaceType::Diffuse) fail(*material, "sensor material must be diffuse (a measurement screen)");
    }
    so.name = b.name;
    so.detector = detIndex;
    auto d = std::make_unique<SurfaceSensor>();
    // aim_center, aim_axis, aim_radius: the disc (a pupil) the sensor's samples aim at; a diffuse
    // measurement screen also guides its scattered light there.
    if (auto ac = b.get("aim_center")) {
        d->hasAim = true;
        d->aimCenter = vecLen(*ac);
        d->aimNormal = normalize(vecNum(need(b, "aim_axis")));
        d->aimRadius = getRadius(b, "aim_", 0.01);
        if (so.type == SurfaceType::Diffuse) {
            if (!(d->aimRadius > 0)) fail(b, "screen aim radius must be positive");
            so.sampleAimCenter = d->aimCenter;
            so.sampleAimNormal = d->aimNormal;
            so.sampleAimRadius = d->aimRadius;
            so.sampleAimShare = d->aimShare = 0.95;
        }
    }
    d->boundaryIndex = w_.addBoundary(body, "pixels", PlaneShape::rect(half.x, half.y), Transform{}, kOutside, kOutside,
                                      w_.addOptics(so));
    d->name = b.name;
    std::tie(d->width, d->height) = resolution(b);
    d->halfX = half.x;
    d->halfY = half.y;
    pixelFilter(b, *d);
    scene_.detectors.push_back(std::move(d));
}

}  // namespace detail

}  // namespace owe
