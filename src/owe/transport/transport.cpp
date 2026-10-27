#include "owe/transport/transport.hpp"

#include "owe/core/shared/optics.hpp"

namespace owe {

namespace {

struct LightSample {
    Vec3 wi;
    double dist = Inf;
    Spec4 Le;
    double pdf = 0;  // solid angle, including light selection
    uint32_t boundary = kNone;
    SurfaceHit endpoint;
    uint32_t region = kNone;
};

double sampledAnchorError(const Boundary& boundary, const Vec3& local, const Vec3& world) {
    return contactError(std::max({maxAbsComponent(world), maxAbsComponent(local), maxAbsComponent(boundary.toWorld.t),
                  maxAbsComponent(boundary.shape->bounds().extent())}));
}

// A point drawn on an emitting boundary, with the anchor its visibility queries need.
SurfaceHit emitterPoint(const Boundary& b, uint32_t boundary, const Vec3& pLocal, const Vec3& nLocal, double u3) {
    SurfaceHit e;
    e.p = b.toWorld.point(pLocal);
    e.n = normalize(b.toWorld.vector(nLocal));
    e.boundary = boundary;
    e.prim = b.shape->sampledPrimitive(u3);
    e.error = sampledAnchorError(b, pLocal, e.p);
    return e;
}

bool sampleLightFrom(const World& w, const Vec3& p, const Wavelengths& wl, Rng& rng, LightSample& ls) {
    if (w.lights().empty()) return false;
    size_t li = w.pickLight(rng.uniform());
    double sel = w.lightSelectPdf(li);
    const LightEntry& L = w.lights()[li];
    double u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
    if (L.kind == LightEntry::Kind::Sun) {
        ls.wi = w.env.sampleSun(u1, u2);
        ls.Le = wl.eval(w.env.sunRadiance);
        ls.pdf = sel / w.env.sunSolidAngle();
        return true;
    }
    if (L.kind == LightEntry::Kind::Environment) {
        double pdfDir;
        if (!w.env.sampleMap(u1, u2, ls.wi, pdfDir)) return false;
        for (int i = 0; i < NW; ++i) ls.Le[i] = w.env.sky(ls.wi, wl.lambda[size_t(i)]);
        ls.pdf = sel * pdfDir;
        return true;
    }
    const Boundary& b = w.boundaries()[L.boundary];
    Vec3 pL, nL;
    double pdfW;
    if (!b.shape->sampleFrom(b.toLocal.point(p), u1, u2, u3, pL, nL, pdfW)) return false;
    ls.endpoint = emitterPoint(b, L.boundary, pL, nL, u3);
    Vec3 d = ls.endpoint.p - p;
    ls.dist = length(d);
    if (!(ls.dist > 0)) return false;
    ls.wi = d / ls.dist;
    const Emission& e = w.emissions()[b.emission];
    double cosL = dot(ls.endpoint.n, -ls.wi);
    ls.Le = ((cosL > 0 && e.front) || (cosL < 0 && e.back)) ? wl.eval(e.radiance) : Spec4(0.0);
    ls.pdf = sel * pdfW;
    ls.boundary = L.boundary;
    ls.region = cosL > 0 ? b.front : b.back;
    return true;
}

Spec4 sigmaT(const Medium& m, const Wavelengths& wl) {
    Spec4 r;
    for (int i = 0; i < NW; ++i) r[i] = m.absorption.eval(wl.lambda[size_t(i)]) + m.scattering.eval(wl.lambda[size_t(i)]);
    return r;
}

Spec4 expNeg(const Spec4& s, double t) {
    Spec4 r;
    for (int i = 0; i < NW; ++i) r[i] = std::exp(-s[i] * t);
    return r;
}

// Zero the throughput of wavelengths whose pdf was cleared.
void maskTerminated(const Wavelengths& wl, Spec4& beta) {
    for (int i = 1; i < NW; ++i)
        if (wl.pdf[size_t(i)] == 0) beta[i] = 0;
}

// What the medium of the current region does to a segment of length seg (Inf: unbounded): a
// scattering point at t < seg, distance-sampled with the hero's extinction (the secondaries
// reweighted), or passage with Beer–Lambert attenuation. End: the path cannot go on (an unbounded
// segment through absorbing or scattering matter).
enum class MediumEvent { Pass, Scatter, End };
MediumEvent sampleMedium(const Medium& m, const Wavelengths& wl, double seg, Rng& rng, Spec4& beta, double& t) {
    if (m.scatters()) {
        Spec4 sT = sigmaT(m, wl);
        t = -std::log(1 - rng.uniform()) / sT[0];
        if (t < seg) {
            double pdf0 = sT[0] * std::exp(-sT[0] * t);
            for (int i = 0; i < NW; ++i)
                beta[i] *= m.scattering.eval(wl.lambda[size_t(i)]) * std::exp(-sT[i] * t) / pdf0;
            return MediumEvent::Scatter;
        }
        if (std::isinf(seg)) return MediumEvent::End;
        double pass0 = std::exp(-sT[0] * seg);
        for (int i = 0; i < NW; ++i) beta[i] *= std::exp(-sT[i] * seg) / pass0;
    } else if (m.absorbs()) {
        if (std::isinf(seg)) return MediumEvent::End;
        for (int i = 0; i < NW; ++i) beta[i] *= std::exp(-m.absorption.eval(wl.lambda[size_t(i)]) * seg);
    }
    return MediumEvent::Pass;
}

// The sides of a boundary met along d. A ray arriving from a region other than the boundary's on
// that side is counted as an inconsistency and continues from the boundary's region.
struct Crossing {
    uint32_t inc, other;  // the region the ray must arrive from, and the one beyond
    bool fromFront;
};
Crossing meetBoundary(const World& w, const SurfaceHit& hit, const Ray& ray, uint32_t& region, uint32_t previous,
                      TransportStats& st) {
    const Boundary& b = w.boundaries()[hit.boundary];
    Crossing c;
    c.fromFront = dot(ray.d, hit.n) < 0;
    c.inc = c.fromFront ? b.front : b.back;
    c.other = c.fromFront ? b.back : b.front;
    if (c.inc != region || hit.ambiguousContact) {
        st.noteInconsistency({hit.boundary, region, c.inc, previous, hit.p, ray.o, ray.d, ray.departedPrimitive, hit.prim});
        region = c.inc;
    }
    return c;
}

// Appends an event to an inspection record (nothing without one); the caller adds its optics.
PathVertex* record(PathRecord* rec, EventKind ev, const Vec3& p, const Vec3& dOut, double opl, double beta,
                   uint32_t boundary = kNone, uint32_t from = kNone, uint32_t to = kNone, double nI = 0, double nT = 0) {
    if (!rec) return nullptr;
    PathVertex v;
    v.p = p;
    v.dOut = dOut;
    v.event = ev;
    v.boundary = boundary;
    v.regionFrom = from;
    v.regionTo = to;
    v.nI = nI;
    v.nT = nT;
    v.opl = opl;
    v.beta = beta;
    rec->v.push_back(v);
    return &rec->v.back();
}
// A scattering event's angles and Fresnel split.
void recordScatter(PathVertex* v, const ScatterSample& s) {
    if (!v) return;
    v->thetaI = std::acos(clamp(s.cosI, -1, 1));
    v->thetaT = std::acos(clamp(s.cosT, -1, 1));
    v->R = s.R;
    v->T = s.T;
}
// Where an escaping ray is drawn: beyond the world.
Vec3 escapePoint(const World& w, const Ray& ray) {
    return ray.o + ray.d * (2 * w.sceneRadius() + length(ray.o - w.sceneCenter()));
}

}  // namespace

Interface Tracer::makeInterface(const SurfaceHit& hit, double lambda) const {
    const Boundary& b = world_.boundaries()[hit.boundary];
    Interface it;
    it.optics = &world_.optics()[b.optics];
    it.n = hit.n;
    it.p = hit.p;
    it.pLocal = hit.pLocal;
    it.nLocal = hit.nLocal;
    it.nFront = world_.indexOf(b.front, lambda);
    it.nBack = world_.indexOf(b.back, lambda);
    it.lambda = lambda;
    it.fresnelFloor = fresnelFloor_;
    if (it.optics->texture.kind != Texture::Kind::None) {
        it.optics->texture.weights(it.pLocal, it.nLocal, it.tex);
        it.texReady = true;
    }
    return it;
}

// The same interface at another wavelength: only the indices change.
static Interface atWavelength(const Interface& base, const World& w, const SurfaceHit& hit, double lambda) {
    Interface it = base;
    const Boundary& b = w.boundaries()[hit.boundary];
    it.nFront = w.indexOf(b.front, lambda);
    it.nBack = w.indexOf(b.back, lambda);
    it.lambda = lambda;
    return it;
}

namespace {

// f(wo = −d, wi) at every live wavelength of the interface `base` (the hero's); heroPdf: the hero's
// pdf of sampling wi.
Spec4 evalScatterSpectral(const Interface& base, const World& w, const SurfaceHit& hit, const Wavelengths& wl,
                          const Vec3& d, const Vec3& wi, TransportMode mode, double& heroPdf) {
    Spec4 f;
    heroPdf = 0;
    for (int i = 0; i < NW; ++i) {
        if (wl.pdf[size_t(i)] == 0) continue;
        double pdf;
        f[i] = evalScatter(i == 0 ? base : atWavelength(base, w, hit, wl.lambda[size_t(i)]), d, wi, mode, pdf);
        if (i == 0) heroPdf = pdf;
    }
    return f;
}

// At a dispersive dielectric the secondary wavelengths would refract elsewhere: they end, and the
// hero carries the path for all four.
void terminateIfDispersive(const Interface& it, const World& w, const SurfaceHit& hit, Wavelengths& wl, Spec4& beta) {
    if (it.optics->type != SurfaceType::Dielectric || wl.secondaryTerminated()) return;
    for (int i = 1; i < NW; ++i)
        if (wl.pdf[size_t(i)] != 0 && isDispersive(it, atWavelength(it, w, hit, wl.lambda[size_t(i)]))) {
            wl.terminateSecondary();
            maskTerminated(wl, beta);
            return;
        }
}

// The throughput after scattering sample s, drawn at the hero wavelength, at every live wavelength.
void weighScatter(Spec4& beta, const Interface& it, const World& w, const SurfaceHit& hit, const Wavelengths& wl,
                  const Vec3& d, const ScatterSample& s, TransportMode mode) {
    beta[0] *= s.weight;
    for (int i = 1; i < NW; ++i)
        if (wl.pdf[size_t(i)] != 0)
            beta[i] *= secondaryScatterWeight(atWavelength(it, w, hit, wl.lambda[size_t(i)]), d, s, mode);
}

}  // namespace

Spec4 Tracer::transmittance(const Endpoint& start, const Vec3& dir, double dist, const Wavelengths& wl,
                            uint32_t* finalRegion, const Endpoint* endpoint) const {
    uint32_t region = start.region;
    Spec4 Tr(1.0);
    Ray ray{start.p, dir, start.boundary, region, start.normal, start.primitive, start.error};
    double remaining = dist;
    const double endError = endpoint ? std::max(endpoint->error, contactError(
        std::max({dist, maxAbsComponent(endpoint->p), maxAbsComponent(start.p)}))) : 0;
    const double margin = std::isinf(dist) || endpoint ? 0 : 1e-7 * dist + 1e-12;
    auto finish = [&](const Spec4& value) {
        if (finalRegion) *finalRegion = region;
        return value;
    };
    for (int guard = 0; guard < 256; ++guard) {
        SurfaceHit h;
        double tmax = std::isinf(remaining) ? Inf : (endpoint ? remaining + endError : remaining - margin);
        if (tmax <= 0) return finish(Tr);
        ray.regionHint = region;
        bool found = world_.intersect(ray, tmax, h);
        double seg = found ? std::min(h.t, remaining) : remaining;
        const Medium& m = world_.mediumOf(region);
        if (m.absorbs() || m.scatters()) {
            if (std::isinf(seg)) return Spec4(0.0);
            Tr *= expNeg(sigmaT(m, wl), seg);
        }
        if (!found) {
            if (std::isinf(dist) && region != world_.ambientRegion()) return Spec4(0.0);
            return finish(Tr);
        }
        bool endpointFace = endpoint && h.boundary == endpoint->boundary;
        if (endpointFace)
            endpointFace = world_.boundaries()[h.boundary].shape->sameFace(endpoint->primitive, h.prim);
        if (endpointFace && !h.ambiguousContact &&
            std::abs(h.t - remaining) <= endError && dot(h.n, endpoint->normal) > 0 && region == endpoint->region)
            return finish(Tr);  // exclude only the destination's own face, never its neighborhood
        // A broad numerical query can discover matter beyond the source. The source-local
        // forward proof checked the actual open segment.
        if (endpoint && dot(h.p - endpoint->p, dir) > 0) return finish(Tr);
        const Boundary& b = world_.boundaries()[h.boundary];
        uint32_t incident = dot(dir, h.n) < 0 ? b.front : b.back;
        if (incident != region || h.ambiguousContact) return Spec4(0.0);
        if (world_.optics()[b.optics].type != SurfaceType::Null) return Spec4(0.0);
        region = dot(dir, h.n) < 0 ? b.back : b.front;
        ray = world_.spawnRay(h, dir, region);
        if (!std::isinf(remaining)) remaining -= h.t;
    }
    // Exhausting the progress guard is not evidence that the remaining segment is clear.
    return finish(Spec4(0.0));
}

Spec4 Tracer::connectionTransmittance(Endpoint source, const Endpoint& target, const Wavelengths& wl) const {
    const Vec3 delta = source.p - target.p;
    const double distance = length(delta);
    if (!(distance > 0)) return Spec4(0.0);
    const Vec3 direction = delta / distance;
    const double error = contactError(std::max({distance, maxAbsComponent(source.p), maxAbsComponent(target.p)}));
    if (source.boundary != kNone) {
        const Boundary& b = world_.boundaries()[source.boundary];
        source.error = std::max(source.error, sampledAnchorError(b, b.toLocal.point(source.p), source.p));
    }
    // Check the source neighborhood before excluding its own face in the reciprocal walk.
    SurfaceHit contact;
    Ray proof{source.p, -direction, source.boundary, source.region, source.normal, source.primitive, source.error};
    proof.forwardOnly = true;
    if (world_.intersect(proof, std::max(error, source.error), contact)) {
        if (contact.ambiguousContact || world_.optics()[world_.boundaries()[contact.boundary].optics].type != SurfaceType::Null)
            return Spec4(0.0);
    }
    uint32_t arrival = kNone;
    Spec4 tr = transmittance(target, direction, distance, wl, &arrival, &source);
    return arrival == source.region ? tr : Spec4(0.0);
}

double Tracer::lightPdf(const Vec3& ref, const SurfaceHit& hit) const {
    int li = world_.lightOfBoundary(hit.boundary);
    if (li < 0) return 0;
    const Boundary& b = world_.boundaries()[hit.boundary];
    return world_.lightSelectPdf(size_t(li)) * b.shape->pdfFrom(b.toLocal.point(ref), hit.pLocal, hit.nLocal);
}

double Tracer::sunPdf() const {
    if (!world_.env.hasSun) return 0;
    size_t li = world_.lights().size() - 1;  // the sun is always the last light
    return world_.lightSelectPdf(li) / world_.env.sunSolidAngle();
}

Spec4 Tracer::directLighting(const SurfaceHit& hit, const Interface* surface, const Vec3& d, const Wavelengths& wl,
                             Rng& rng, std::string* source, double g, uint32_t region) const {
    LightSample ls;
    if (!sampleLightFrom(world_, hit.p, wl, rng, ls) || ls.Le.isZero() || ls.pdf <= 0) return Spec4(0.0);
    Spec4 f;
    double pdfScatter = 0, cosFactor = 1;
    uint32_t startRegion = region;
    if (!surface) {
        pdfScatter = hgPhase(dot(-d, ls.wi), g);
        f = Spec4(pdfScatter);
    } else {
        f = evalScatterSpectral(*surface, world_, hit, wl, d, ls.wi, TransportMode::Radiance, pdfScatter);
        if (f.isZero()) return Spec4(0.0);
        cosFactor = std::abs(dot(ls.wi, hit.n));
        const Boundary& b = world_.boundaries()[hit.boundary];
        startRegion = dot(ls.wi, hit.n) > 0 ? b.front : b.back;
    }
    const Endpoint start = surface ? Endpoint(hit, startRegion) : Endpoint(hit.p, startRegion);
    const Spec4 Tr = std::isfinite(ls.dist)
        ? connectionTransmittance(Endpoint(ls.endpoint, ls.region), start, wl)
        : transmittance(start, ls.wi, Inf, wl);
    if (Tr.isZero()) return Spec4(0.0);
    double w = powerHeuristic(ls.pdf, pdfScatter);
    if (source) *source = ls.boundary == kNone ? std::string("sun") : world_.boundaryLabel(ls.boundary);
    return f * Tr * ls.Le * (cosFactor * w / ls.pdf);
}

struct Tracer::CameraPath {
    Ray ray;
    uint32_t region;
    uint32_t lastBoundary = kNone;
    PathRecord* rec;
    Spec4 L = Spec4(0.0), beta = Spec4(1.0);
    bool specular = true;
    double prevPdf = 0, opl = 0;
    Vec3 prevP;

    CameraPath(const Ray& r, uint32_t reg, PathRecord* record) : ray(r), region(reg), rec(record), prevP(r.o) {}
    void contribute(size_t vertex, const Spec4& value, const std::string& src, bool nee) {
        L += value;
        if (rec && value.sum() > 0) rec->c.push_back(PathContribution{vertex, value, src, nee});
    }
    // The last recorded vertex (0 without a record), and a pointer to it.
    size_t lastVertex() const { return rec ? rec->v.size() - 1 : 0; }
    PathVertex* vertex() const { return rec ? &rec->v.back() : nullptr; }
    // Russian roulette against the throughput; false when the path ends.
    bool survives(Rng& rng) {
        double q = std::min(1.0, beta.maxValue());
        if (rng.uniform() >= q) return false;
        beta *= 1 / q;
        return true;
    }
};

// A scattering event in the medium at distance t: next-event estimation from there, then a new
// direction from the phase function.
bool Tracer::scatterInMedium(const Medium& med, double nMed, double t, int depth, Wavelengths& wl, Rng& rng,
                             CameraPath& path) const {
    Vec3 p = path.ray.at(t);
    path.opl += nMed * t;
    record(path.rec, EventKind::Scatter, p, path.ray.d, path.opl, path.beta[0], kNone, path.region, path.region, nMed, nMed);
    if (depth >= maxDepth_) return false;
    SurfaceHit mh;
    mh.p = p;
    std::string src;
    Spec4 dl = directLighting(mh, nullptr, path.ray.d, wl, rng, path.rec ? &src : nullptr, med.g, path.region);
    if (!dl.isZero()) path.contribute(path.lastVertex(), path.beta * dl, src, true);
    Vec3 wi;
    Draw2 u(rng);
    path.prevPdf = sampleHg(-path.ray.d, med.g, u.u1, u.u2, wi);
    path.specular = false;
    path.prevP = p;
    path.ray = Ray{p, wi};
    return depth < rrDepth_ || path.survives(rng);
}

// The sky and the sun met leaving the world, weighted against next-event estimation unless the
// last scatter was specular.
void Tracer::skyMet(const Wavelengths& wl, double nMed, CameraPath& path) const {
    const Ray& ray = path.ray;
    Spec4 sky, sun;
    for (int i = 0; i < NW; ++i) {
        sky[i] = world_.env.sky(ray.d, wl.lambda[size_t(i)]);
        sun[i] = world_.env.sun(ray.d, wl.lambda[size_t(i)]);
    }
    double w = (path.specular || sun.isZero()) ? 1.0 : powerHeuristic(path.prevPdf, sunPdf());
    // An environment map is also a light for next-event estimation: MIS.
    const int envLight = world_.environmentLight();
    double wSky = 1;
    if (envLight >= 0 && !path.specular)
        wSky = powerHeuristic(path.prevPdf, world_.lightSelectPdf(size_t(envLight)) * world_.env.mapPdf(ray.d));
    record(path.rec, EventKind::Escape, escapePoint(world_, ray), ray.d, path.opl, path.beta[0], kNone, path.region,
           path.region, nMed, nMed);
    path.contribute(path.lastVertex(), path.beta * (sky * wSky + sun * w), sun.isZero() ? "sky" : "sun+sky", false);
}

// Emission met at hit, weighted against next-event estimation unless the last scatter was specular.
void Tracer::emissionMet(const SurfaceHit& hit, bool fromFront, const Wavelengths& wl, CameraPath& path) const {
    const Boundary& b = world_.boundaries()[hit.boundary];
    if (b.emission < 0) return;
    const Emission& e = world_.emissions()[b.emission];
    if (!(fromFront ? e.front : e.back)) return;
    double w = 1;
    if (!path.specular) {
        double pl = lightPdf(path.prevP, hit);
        if (pl > 0) w = powerHeuristic(path.prevPdf, pl);
    }
    // The emitter's label only names the contribution in a recorded path (no allocation otherwise).
    path.contribute(path.lastVertex(), path.beta * wl.eval(e.radiance) * w,
                    path.rec ? world_.boundaryLabel(hit.boundary) : std::string(), false);
}

// At a scattering surface: next-event estimation (unless it is specular), then a scattering sample
// that may cross into the region `other`.
bool Tracer::scatterAtSurface(const SurfaceHit& hit, const SurfaceOptics& o, uint32_t other, int depth,
                              Wavelengths& wl, Rng& rng, CameraPath& path) const {
    PathVertex* v = path.vertex();
    const size_t vIndex = path.lastVertex();
    const uint32_t inc = path.region;
    Interface it = makeInterface(hit, wl.hero());
    terminateIfDispersive(it, world_, hit, wl, path.beta);
    if (!o.isDelta()) {
        std::string src;
        Spec4 dl = directLighting(hit, &it, path.ray.d, wl, rng, path.rec ? &src : nullptr, 0, path.region);
        if (!dl.isZero()) path.contribute(vIndex, path.beta * dl, src, true);
    }
    ScatterSample s;
    Draw3 u(rng);
    bool ok = sampleScatter(it, path.ray.d, u.u1, u.u2, u.u3, TransportMode::Radiance, s);
    if (v) {
        v->event = ok ? s.event : EventKind::Absorb;
        v->dOut = ok ? s.wi : path.ray.d;
        v->regionTo = ok && s.transmitted ? other : inc;
        v->beta = ok ? path.beta[0] * s.weight : 0;
        recordScatter(v, s);
    }
    if (!ok || s.weight <= 0) return false;
    weighScatter(path.beta, it, world_, hit, wl, path.ray.d, s, TransportMode::Radiance);
    if (s.transmitted) path.region = other;
    path.specular = s.delta;
    path.prevPdf = s.pdf;
    path.prevP = hit.p;
    path.ray = world_.spawnRay(hit, s.wi, path.region);
    if (depth >= rrDepth_ && !path.survives(rng)) {
        record(path.rec, EventKind::Terminate, path.ray.o, path.ray.d, path.opl, 0, kNone, path.region, path.region);
        return false;
    }
    return true;
}

Spec4 Tracer::radiance(Ray ray, uint32_t region, Wavelengths& wl, Rng& rng, TransportStats& st, PathRecord* rec) const {
    CameraPath path(ray, region, rec);
    maskTerminated(wl, path.beta);
    const double hero = wl.hero();
    st.paths++;
    if (rec) rec->lambda = hero;
    record(rec, EventKind::Camera, ray.o, ray.d, 0, 1, kNone, region, region);

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        path.ray.regionHint = path.region;
        bool found = world_.intersect(path.ray, Inf, hit);
        st.segments++;
        const Medium& med = world_.mediumOf(path.region);
        const double nMed = med.n(hero);
        double t;
        const MediumEvent event = sampleMedium(med, wl, found ? hit.t : Inf, rng, path.beta, t);
        if (event == MediumEvent::End) break;
        if (event == MediumEvent::Scatter) {
            if (!scatterInMedium(med, nMed, t, depth, wl, rng, path)) break;
            continue;
        }
        if (!found) {
            if (path.region == world_.ambientRegion()) skyMet(wl, nMed, path);
            else st.leaks++;
            break;
        }

        path.opl += nMed * hit.t;
        const Crossing c = meetBoundary(world_, hit, path.ray, path.region, path.lastBoundary, st);
        if (hit.ambiguousContact) break;
        const SurfaceOptics& o = world_.optics()[world_.boundaries()[hit.boundary].optics];
        path.lastBoundary = hit.boundary;
        PathVertex* v = rec ? record(rec, EventKind::None, hit.p, path.ray.d, path.opl, path.beta[0], hit.boundary, c.inc,
                                     c.inc, nMed, world_.indexOf(c.other, hero))
                            : nullptr;
        emissionMet(hit, c.fromFront, wl, path);
        if (o.type == SurfaceType::Null) {
            if (v) { v->event = EventKind::Pass; v->regionTo = c.other; v->T = 1; }
            path.region = c.other;
            path.ray = world_.spawnRay(hit, path.ray.d, path.region);
            if (depth >= 4 * maxDepth_) break;
            continue;
        }
        if (o.type == SurfaceType::Absorber || o.type == SurfaceType::Detector) {
            if (v) v->event = o.type == SurfaceType::Detector ? EventKind::Detect : EventKind::Absorb;
            break;
        }
        if (depth >= maxDepth_) {
            if (v) v->event = EventKind::Terminate;
            break;
        }
        if (!scatterAtSurface(hit, o, c.other, depth, wl, rng, path)) break;
    }
    if (rec) rec->wavelengths = wl;
    return path.L;
}

PathRecord Tracer::walk(Ray ray, uint32_t region, double lambda, WalkMode mode, Rng& rng, int maxEvents) const {
    PathRecord rec;
    rec.lambda = lambda;
    double beta = 1, opl = 0;
    record(&rec, EventKind::Emit, ray.o, ray.d, 0, 1, kNone, region, region, world_.indexOf(region, lambda),
           world_.indexOf(region, lambda));
    for (int ev = 0; ev < maxEvents; ++ev) {
        SurfaceHit hit;
        ray.regionHint = region;
        bool found = world_.intersect(ray, Inf, hit);
        const Medium& med = world_.mediumOf(region);
        double nMed = med.n(lambda);
        if (!found) {
            record(&rec, EventKind::Escape, escapePoint(world_, ray), ray.d, opl, beta, kNone, region, region, nMed, nMed);
            return rec;
        }
        if (med.absorbs()) beta *= std::exp(-med.absorption.eval(lambda) * hit.t);
        opl += nMed * hit.t;
        const Boundary& b = world_.boundaries()[hit.boundary];
        bool fromFront = dot(ray.d, hit.n) < 0;
        uint32_t inc = fromFront ? b.front : b.back;
        uint32_t other = fromFront ? b.back : b.front;
        region = inc;
        const SurfaceOptics& o = world_.optics()[b.optics];
        double n2 = world_.indexOf(other, lambda);
        if (o.type == SurfaceType::Null) {
            record(&rec, EventKind::Pass, hit.p, ray.d, opl, beta, hit.boundary, inc, other, nMed, n2)->T = 1;
            region = other;
            ray = world_.spawnRay(hit, ray.d, region);
            continue;
        }
        if (o.type == SurfaceType::Absorber || o.type == SurfaceType::Detector ||
            (mode == WalkMode::PrimaryTransmission && o.type == SurfaceType::Diffuse)) {
            EventKind k = o.type == SurfaceType::Detector ? EventKind::Detect
                          : o.type == SurfaceType::Diffuse  ? EventKind::Diffuse
                                                             : EventKind::Absorb;
            PathVertex* v = record(&rec, k, hit.p, ray.d, opl, beta, hit.boundary, inc, inc, nMed, n2);
            v->thetaI = v->thetaT = std::acos(clamp(std::abs(dot(ray.d, hit.n)), -1, 1));
            return rec;
        }
        Interface it = makeInterface(hit, lambda);
        ScatterSample s;
        bool ok;
        if (mode == WalkMode::PrimaryTransmission && o.type == SurfaceType::Dielectric) {
            Draw2 u(rng);
            ok = sampleScatter(it, ray.d, 1.0, u.u1, u.u2, TransportMode::Importance, s);
            if (!ok) {
                Draw2 v(rng);
                ok = sampleScatter(it, ray.d, 0.0, v.u1, v.u2, TransportMode::Importance, s);
            }
            if (ok && s.delta) s.weight = s.transmitted ? s.T : s.R;
        } else {
            Draw3 u(rng);
            ok = sampleScatter(it, ray.d, u.u1, u.u2, u.u3, TransportMode::Importance, s);
        }
        recordScatter(record(&rec, ok ? s.event : EventKind::Absorb, hit.p, ok ? s.wi : ray.d, opl, ok ? beta * s.weight : 0,
                             hit.boundary, inc, ok && s.transmitted ? other : inc, nMed, n2), s);
        if (!ok) return rec;
        beta *= s.weight;
        if (s.transmitted) region = other;
        ray = world_.spawnRay(hit, s.wi, region);
    }
    record(&rec, EventKind::Terminate, ray.o, ray.d, opl, beta, kNone, region, region);
    return rec;
}

}  // namespace owe
