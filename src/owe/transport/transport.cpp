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

Spec4 Tracer::radiance(Ray ray, uint32_t region, Wavelengths& wl, Rng& rng, TransportStats& st, PathRecord* rec) const {
    Spec4 L(0.0), beta(1.0);
    uint32_t lastBoundary = kNone;
    maskTerminated(wl, beta);
    double opl = 0;
    bool specular = true;
    double prevPdf = 0;
    Vec3 prevP = ray.o;
    const double hero = wl.hero();
    // Caustic-partition bookkeeping (see setCausticPartition).
    int scatterEvents = 0, nonDeltaEvents = 0, deltaSinceNonDelta = 0;
    bool firstIsNonDelta = false;
    auto ownedByLightTracing = [&] {
        return partition_ && nonDeltaEvents == 1 && firstIsNonDelta && deltaSinceNonDelta >= 1;
    };
    auto noteScatter = [&](bool delta) {
        if (delta) ++deltaSinceNonDelta;
        else {
            if (scatterEvents == 0) firstIsNonDelta = true;
            ++nonDeltaEvents;
            deltaSinceNonDelta = 0;
        }
        ++scatterEvents;
    };
    // Russian roulette against the throughput; false when the path ends.
    auto survives = [&] {
        double q = std::min(1.0, beta.maxValue());
        if (rng.uniform() >= q) return false;
        beta *= 1 / q;
        return true;
    };
    st.paths++;
    if (rec) rec->lambda = hero;
    record(rec, EventKind::Camera, ray.o, ray.d, 0, 1, kNone, region, region);
    auto contribute = [&](size_t vertex, const Spec4& value, const std::string& src, bool nee) {
        L += value;
        if (rec && value.sum() > 0) rec->c.push_back(PathContribution{vertex, value, src, nee});
    };
    auto lastVertex = [&] { return rec ? rec->v.size() - 1 : 0; };

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        ray.regionHint = region;
        bool found = world_.intersect(ray, Inf, hit);
        st.segments++;
        const Medium& med = world_.mediumOf(region);
        double nMed = med.n(hero);
        double t;
        const MediumEvent event = sampleMedium(med, wl, found ? hit.t : Inf, rng, beta, t);
        if (event == MediumEvent::End) break;
        if (event == MediumEvent::Scatter) {
            Vec3 p = ray.at(t);
            opl += nMed * t;
            record(rec, EventKind::Scatter, p, ray.d, opl, beta[0], kNone, region, region, nMed, nMed);
            if (depth >= maxDepth_) break;
            SurfaceHit mh;
            mh.p = p;
            std::string src;
            Spec4 dl = directLighting(mh, nullptr, ray.d, wl, rng, rec ? &src : nullptr, med.g, region);
            if (!dl.isZero()) contribute(lastVertex(), beta * dl, src, true);
            Vec3 wi;
            Draw2 u(rng);
            prevPdf = sampleHg(-ray.d, med.g, u.u1, u.u2, wi);
            specular = false;
            prevP = p;
            noteScatter(false);
            ray = Ray{p, wi};
            if (depth >= rrDepth_ && !survives()) break;
            continue;
        }

        if (!found) {
            if (region == world_.ambientRegion() && !ownedByLightTracing()) {
                Spec4 sky, sun;
                for (int i = 0; i < NW; ++i) {
                    sky[i] = world_.env.sky(ray.d, wl.lambda[size_t(i)]);
                    sun[i] = world_.env.sun(ray.d, wl.lambda[size_t(i)]);
                }
                double w = (specular || sun.isZero()) ? 1.0 : powerHeuristic(prevPdf, sunPdf());
                // An environment map is also a light for next-event estimation: MIS.
                const int envLight = world_.environmentLight();
                double wSky = 1;
                if (envLight >= 0 && !specular)
                    wSky = powerHeuristic(prevPdf, world_.lightSelectPdf(size_t(envLight)) * world_.env.mapPdf(ray.d));
                record(rec, EventKind::Escape, escapePoint(world_, ray), ray.d, opl, beta[0], kNone, region, region, nMed, nMed);
                contribute(lastVertex(), beta * (sky * wSky + sun * w), sun.isZero() ? "sky" : "sun+sky", false);
            } else if (region != world_.ambientRegion()) {
                st.leaks++;
            }
            break;
        }

        opl += nMed * hit.t;
        const Crossing c = meetBoundary(world_, hit, ray, region, lastBoundary, st);
        if (hit.ambiguousContact) break;
        const Boundary& b = world_.boundaries()[hit.boundary];
        const SurfaceOptics& o = world_.optics()[b.optics];
        lastBoundary = hit.boundary;
        PathVertex* v = rec ? record(rec, EventKind::None, hit.p, ray.d, opl, beta[0], hit.boundary, c.inc, c.inc, nMed,
                                     world_.indexOf(c.other, hero))
                            : nullptr;
        const size_t vIndex = lastVertex();

        // Emission.
        if (b.emission >= 0) {
            const Emission& e = world_.emissions()[b.emission];
            if (((c.fromFront && e.front) || (!c.fromFront && e.back)) && !ownedByLightTracing()) {
                double w = 1;
                if (!specular) {
                    double pl = lightPdf(prevP, hit);
                    if (pl > 0) w = powerHeuristic(prevPdf, pl);
                }
                // The emitter's label only names the contribution in a recorded path (no allocation otherwise).
                contribute(vIndex, beta * wl.eval(e.radiance) * w, rec ? world_.boundaryLabel(hit.boundary) : std::string(), false);
            }
        }

        if (o.type == SurfaceType::Null) {
            if (v) { v->event = EventKind::Pass; v->regionTo = c.other; v->T = 1; }
            region = c.other;
            ray = world_.spawnRay(hit, ray.d, region);
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

        Interface it = makeInterface(hit, hero);
        terminateIfDispersive(it, world_, hit, wl, beta);
        if (!o.isDelta()) {
            std::string src;
            Spec4 dl = directLighting(hit, &it, ray.d, wl, rng, rec ? &src : nullptr, 0, region);
            if (!dl.isZero()) contribute(vIndex, beta * dl, src, true);
        }
        ScatterSample s;
        Draw3 u(rng);
        bool ok = sampleScatter(it, ray.d, u.u1, u.u2, u.u3, TransportMode::Radiance, s);
        if (v) {
            v->event = ok ? s.event : EventKind::Absorb;
            v->dOut = ok ? s.wi : ray.d;
            v->regionTo = ok && s.transmitted ? c.other : c.inc;
            v->beta = ok ? beta[0] * s.weight : 0;
            recordScatter(v, s);
        }
        if (!ok || s.weight <= 0) break;
        weighScatter(beta, it, world_, hit, wl, ray.d, s, TransportMode::Radiance);
        if (s.transmitted) region = c.other;
        specular = s.delta;
        prevPdf = s.pdf;
        prevP = hit.p;
        noteScatter(s.delta);
        ray = world_.spawnRay(hit, s.wi, region);
        if (depth >= rrDepth_ && !survives()) {
            record(rec, EventKind::Terminate, ray.o, ray.d, opl, 0, kNone, region, region);
            break;
        }
    }
    if (rec) rec->wavelengths = wl;
    return L;
}

void Tracer::traceParticle(const Detector& det, Wavelengths wl, Rng& rng, Film& film, TransportStats& st,
                           PathRecord* rec) const {
    const World& w = world_;
    double skyPower = w.environmentPower();
    double lightPower = w.lights().empty() ? 0.0 : w.totalLightPower();
    if (skyPower + lightPower <= 0) return;
    st.paths++;
    const double hero = wl.hero();
    double pSky = skyPower / (skyPower + lightPower);
    auto deposit = [&](int px, int py, const Spec4& value) {
        XYZ c = wl.toXYZ(value);
        film.add(px, py, c.x, c.y, c.z);
    };
    // A vertex x seen by a virtual observer through point q: its connection's pixel and factor.
    struct Connection {
        Vec3 q, dir;
        int px, py;
        double factor;
    };
    auto connectObserver = [&](const Vec3& x, Connection& k) {
        if (!det.isVirtual() || !det.connect(x, rng, k.q, k.px, k.py, k.factor)) return false;
        k.dir = normalize(k.q - x);
        return true;
    };
    Ray ray;
    uint32_t region;
    Spec4 beta;
    uint32_t emitterBoundary = kNone;
    bool fromSky = rng.uniform() < pSky;
    size_t li = 0;
    if (!fromSky) li = w.pickLight(rng.uniform());
    // An environment map is both the sky and a light in the list: either way the photon is a sky
    // photon, emitted with the combined probability of the two choices.
    const int envLight = w.environmentLight();
    if (!fromSky && int(li) == envLight) fromSky = true;
    const double pSkyPhoton = pSky + (envLight >= 0 ? (1 - pSky) * w.lightSelectPdf(size_t(envLight)) : 0.0);
    if (fromSky || w.lights()[li].kind == LightEntry::Kind::Sun) {
        // Distant emission (sky dome or sun): pick a direction, then a disk covering the world.
        double R = w.sceneRadius() * 1.001;
        Vec3 toward;  // direction from which the light arrives
        double pdfDir, sel;
        Draw2 u(rng);
        if (fromSky) {
            if (w.env.skyModel == Environment::Sky::Map) {
                // The map's own distribution (luminance × sin θ).
                if (!w.env.sampleMap(u.u1, u.u2, toward, pdfDir)) return;
            } else {
                toward = sampleUniformSphere(u.u1, u.u2);
                pdfDir = 1 / (4 * Pi);
            }
            for (int i = 0; i < NW; ++i) beta[i] = w.env.sky(toward, wl.lambda[size_t(i)]);
            sel = pSkyPhoton;
        } else {
            toward = w.env.sampleSun(u.u1, u.u2);
            pdfDir = 1 / w.env.sunSolidAngle();
            beta = wl.eval(w.env.sunRadiance);
            sel = (1 - pSky) * w.lightSelectPdf(li);
        }
        Vec3 dir = -toward;
        Vec3 a1, a2;
        orthonormalBasis(dir, a1, a2);
        Draw2 ud(rng);
        Vec2 dk = sampleUniformDiskConcentric(ud.u1, ud.u2);
        ray = Ray{w.sceneCenter() - dir * (1.01 * R) + (a1 * dk.x + a2 * dk.y) * R, dir};
        region = w.ambientRegion();
        beta *= Pi * R * R / (pdfDir * sel);
    } else {
        double sel = (1 - pSky) * w.lightSelectPdf(li);
        const LightEntry& L = w.lights()[li];
        const Boundary& b = w.boundaries()[L.boundary];
        const Emission& e = w.emissions()[b.emission];
        Spec4 Le = wl.eval(e.radiance);
        Vec3 pL, nL;
        Draw3 ua(rng);
        b.shape->sampleArea(ua.u1, ua.u2, ua.u3, pL, nL);
        const SurfaceHit emission = emitterPoint(b, L.boundary, pL, nL, ua.u3);
        double pdfA = 1.0 / b.shape->area();
        // Emitter seen directly by a virtual observer (path tracing owns this under the partition).
        Connection k;
        if (!partition_ && connectObserver(emission.p, k)) {
            double cosL = dot(emission.n, k.dir);
            if ((cosL > 0 && e.front) || (cosL < 0 && e.back)) {
                Spec4 Tr = connectionTransmittance(Endpoint(emission, cosL > 0 ? b.front : b.back), Endpoint(k.q, det.region), wl);
                if (!Tr.isZero()) deposit(k.px, k.py, Le * Tr * (std::abs(cosL) * k.factor / (sel * pdfA)));
            }
        }
        double sidePdf = 1;
        bool front = e.front;
        if (e.front && e.back) {
            front = rng.uniform() < 0.5;
            sidePdf = 0.5;
        }
        Draw2 uc(rng);
        Vec3 dir = Frame(front ? emission.n : -emission.n).toWorld(sampleCosineHemisphere(uc.u1, uc.u2));
        // β = Le·cos / (sel · pdfA · sidePdf · cos/π)
        beta = Le * (Pi / (sel * pdfA * sidePdf));
        region = front ? b.front : b.back;
        ray = w.spawnRay(emission, dir, region);
        emitterBoundary = L.boundary;
    }
    maskTerminated(wl, beta);
    if (beta.isZero()) return;
    if (rec) rec->lambda = hero;
    record(rec, EventKind::Emit, ray.o, ray.d, 0, beta[0], emitterBoundary, region, region);
    double opl = 0;
    uint32_t lastBoundary = emitterBoundary;
    int targetBoundary = det.boundary();
    int nonDeltaEvents = 0, deltaEvents = 0;
    // Under the caustic partition, connect only at the first non-specular vertex after ≥1 specular one.
    auto mayConnect = [&] { return !partition_ || (nonDeltaEvents == 0 && deltaEvents >= 1); };
    // surface: the hit's interface at the hero wavelength (null for medium scattering).
    auto connect = [&](const SurfaceHit& hit, const Interface* surface, const Vec3& d, double g) {
        Connection k;
        if (!connectObserver(hit.p, k)) return;
        Spec4 f;
        double cosX = 1;
        uint32_t startRegion = region;
        if (!surface) {
            f = Spec4(hgPhase(dot(-d, k.dir), g));
        } else {
            double pdf;
            f = evalScatterSpectral(*surface, w, hit, wl, d, k.dir, TransportMode::Importance, pdf);
            if (f.isZero()) return;
            cosX = std::abs(dot(k.dir, hit.n));
            const Boundary& b = w.boundaries()[hit.boundary];
            startRegion = dot(k.dir, hit.n) > 0 ? b.front : b.back;
        }
        const Endpoint start = surface ? Endpoint(hit, startRegion) : Endpoint(hit.p, startRegion);
        Spec4 Tr = connectionTransmittance(start, Endpoint(k.q, det.region), wl);
        if (Tr.isZero()) return;
        deposit(k.px, k.py, beta * f * Tr * (cosX * k.factor));
    };

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        ray.regionHint = region;
        bool found = w.intersect(ray, Inf, hit);
        st.segments++;
        const Medium& med = w.mediumOf(region);
        double nMed = med.n(hero);
        double t;
        const MediumEvent event = sampleMedium(med, wl, found ? hit.t : Inf, rng, beta, t);
        if (event == MediumEvent::End) break;
        if (event == MediumEvent::Scatter) {
            Vec3 p = ray.at(t);
            opl += nMed * t;
            record(rec, EventKind::Scatter, p, ray.d, opl, beta[0], kNone, region, region, nMed, nMed);
            SurfaceHit mh;
            mh.p = p;
            if (mayConnect()) connect(mh, nullptr, ray.d, med.g);
            ++nonDeltaEvents;
            if (partition_) break;  // nothing further can be connected
            if (depth >= maxDepth_) break;
            Vec3 wi;
            Draw2 u(rng);
            sampleHg(-ray.d, med.g, u.u1, u.u2, wi);
            ray = Ray{p, wi};
            if (depth >= rrDepth_) {
                const double q = 0.95;  // any survival probability in (0,1] is unbiased
                if (rng.uniform() >= q) break;
                beta *= 1 / q;
            }
            continue;
        }
        if (!found) {
            record(rec, EventKind::Escape, escapePoint(w, ray), ray.d, opl, beta[0], kNone, region, region, nMed, nMed);
            break;
        }
        opl += nMed * hit.t;
        const Crossing c = meetBoundary(w, hit, ray, region, lastBoundary, st);
        if (hit.ambiguousContact) break;
        const SurfaceOptics& o = w.optics()[w.boundaries()[hit.boundary].optics];
        lastBoundary = hit.boundary;
        const double nOther = w.indexOf(c.other, hero);
        if (o.type == SurfaceType::Null) {
            PathVertex* v = record(rec, EventKind::Pass, hit.p, ray.d, opl, beta[0], hit.boundary, c.inc, c.other, nMed, nOther);
            if (v) v->T = 1;
            region = c.other;
            ray = world_.spawnRay(hit, ray.d, region);
            if (depth >= 4 * maxDepth_) break;
            continue;
        }
        // A diffuse measurement screen records incident flux and then scatters the reflected
        // fraction. An absorbing detector records flux and terminates the path as before.
        if (c.fromFront && int(hit.boundary) == targetBoundary) {
            int px, py;
            double area;
            if (det.pixelOfHit(hit, rng, px, py, area)) deposit(px, py, beta * (1 / area));
        }
        if (o.type == SurfaceType::Detector || o.type == SurfaceType::Absorber) {
            record(rec, o.type == SurfaceType::Detector ? EventKind::Detect : EventKind::Absorb, hit.p, ray.d, opl, beta[0],
                   hit.boundary, c.inc, c.inc, nMed, nMed);
            break;
        }
        Interface it = makeInterface(hit, hero);
        if (!o.isDelta()) {
            if (mayConnect()) connect(hit, &it, ray.d, 0);
            ++nonDeltaEvents;
            if (partition_) break;  // later vertices are owned by path tracing
        }
        if (depth >= maxDepth_) break;
        terminateIfDispersive(it, w, hit, wl, beta);
        ScatterSample s;
        Draw3 u(rng);
        bool ok = sampleScatter(it, ray.d, u.u1, u.u2, u.u3, TransportMode::Importance, s);
        recordScatter(record(rec, ok ? s.event : EventKind::Absorb, hit.p, ok ? s.wi : ray.d, opl, ok ? beta[0] * s.weight : 0,
                             hit.boundary, c.inc, ok && s.transmitted ? c.other : c.inc, nMed, nOther), s);
        if (!ok || s.weight <= 0) break;
        weighScatter(beta, it, w, hit, wl, ray.d, s, TransportMode::Importance);
        if (s.delta) ++deltaEvents;
        if (s.transmitted) region = c.other;
        ray = world_.spawnRay(hit, s.wi, region);
        // Particles carry flux, so roulette against the per-event throughput.
        const double q = std::min(1.0, s.weight);
        if (depth >= rrDepth_ && q < 1) {
            if (rng.uniform() >= q) break;
            beta *= 1 / q;
        }
    }
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
