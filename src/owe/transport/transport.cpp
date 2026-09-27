#include "owe/transport/transport.hpp"

#include "owe/transport/optics.hpp"

namespace owe {

namespace {

struct LightSample {
    Vec3 wi;
    double dist = Inf;
    Spec4 Le;
    double pdf = 0;  // solid angle, including light selection
    uint32_t boundary = kNone;
};

bool sampleLightFrom(const World& w, const Vec3& p, const Wavelengths& wl, Rng& rng, LightSample& ls) {
    if (w.lights().empty()) return false;
    size_t li = w.pickLight(rng.uniform());
    double sel = w.lightSelectPdf(li);
    const LightEntry& L = w.lights()[li];
    double u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
    if (L.kind == LightEntry::Kind::Sun) {
        double cosMax = std::cos(w.env.sunAngularRadius);
        Frame f(w.env.sunDir);
        ls.wi = f.toWorld(sampleUniformCone(u1, u2, cosMax));
        ls.dist = Inf;
        ls.Le = wl.eval(w.env.sunRadiance);
        ls.pdf = sel / w.env.sunSolidAngle();
        ls.boundary = kNone;
        return true;
    }
    const Boundary& b = w.boundaries()[L.boundary];
    Vec3 pL, nL;
    double pdfW;
    if (!b.shape->sampleFrom(b.toLocal.point(p), u1, u2, u3, pL, nL, pdfW)) return false;
    Vec3 pW = b.toWorld.point(pL);
    Vec3 nW = normalize(b.toWorld.vector(nL));
    Vec3 d = pW - p;
    ls.dist = length(d);
    if (!(ls.dist > 0)) return false;
    ls.wi = d / ls.dist;
    const Emission& e = w.emissions()[b.emission];
    double cosL = dot(nW, -ls.wi);
    ls.Le = ((cosL > 0 && e.front) || (cosL < 0 && e.back)) ? wl.eval(e.radiance) : Spec4(0.0);
    ls.pdf = sel * pdfW;
    ls.boundary = L.boundary;
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

void pushVertex(PathRecord* rec, const Vec3& p, const Vec3& dOut, EventKind ev, uint32_t boundary, uint32_t from,
                uint32_t to, double nI, double nT, double cosI, double cosT, double R, double T, double opl, double beta) {
    if (!rec) return;
    PathVertex v;
    v.p = p;
    v.dOut = dOut;
    v.event = ev;
    v.boundary = boundary;
    v.regionFrom = from;
    v.regionTo = to;
    v.nI = nI;
    v.nT = nT;
    v.thetaI = std::acos(clampd(cosI, -1, 1));
    v.thetaT = std::acos(clampd(cosT, -1, 1));
    v.R = R;
    v.T = T;
    v.opl = opl;
    v.beta = beta;
    rec->v.push_back(v);
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

Spec4 Tracer::transmittance(Vec3 o, uint32_t region, const Vec3& dir, double dist, const Wavelengths& wl,
                            uint32_t* finalRegion) const {
    Spec4 Tr(1.0);
    double remaining = dist;
    double margin = std::isinf(dist) ? 0 : 1e-7 * dist + 1e-12;
    for (int guard = 0; guard < 256; ++guard) {
        SurfaceHit h;
        double tmax = std::isinf(remaining) ? Inf : remaining - margin;
        if (tmax <= 0) break;
        bool found = world_.intersect(Ray{o, dir}, tmax, h);
        double seg = found ? h.t : remaining;
        const Medium& m = world_.mediumOf(region);
        if (m.absorbs() || m.scatters()) {
            if (std::isinf(seg)) return Spec4(0.0);
            Tr *= expNeg(sigmaT(m, wl), seg);
        }
        if (!found) {
            if (std::isinf(dist) && region != world_.ambientRegion()) return Spec4(0.0);
            if (finalRegion) *finalRegion = region;
            return Tr;
        }
        const Boundary& b = world_.boundaries()[h.boundary];
        if (world_.optics()[b.optics].type != SurfaceType::Null) return Spec4(0.0);
        region = dot(dir, h.n) < 0 ? b.back : b.front;
        o = offsetOrigin(h.p, h.n, dir);
        if (!std::isinf(remaining)) remaining -= h.t;
    }
    if (finalRegion) *finalRegion = region;
    return Tr;
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
    Vec3 origin = hit.p;
    if (!surface) {
        double ph = hgPhase(dot(-d, ls.wi), g);
        f = Spec4(ph);
        pdfScatter = ph;
    } else {
        const Interface& base = *surface;
        for (int i = 0; i < NW; ++i) {
            if (wl.pdf[size_t(i)] == 0) continue;
            Interface it = i == 0 ? base : atWavelength(base, world_, hit, wl.lambda[size_t(i)]);
            double pdf;
            f[i] = evalScatter(it, d, ls.wi, TransportMode::Radiance, pdf);
            if (i == 0) pdfScatter = pdf;
        }
        if (f.isZero()) return Spec4(0.0);
        cosFactor = std::abs(dot(ls.wi, hit.n));
        const Boundary& b = world_.boundaries()[hit.boundary];
        startRegion = dot(ls.wi, hit.n) > 0 ? b.front : b.back;
        origin = offsetOrigin(hit.p, hit.n, ls.wi);
    }
    Spec4 Tr = transmittance(origin, startRegion, ls.wi, ls.dist, wl);
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
    st.paths++;
    if (rec) {
        rec->lambda = hero;
        pushVertex(rec, ray.o, ray.d, EventKind::Camera, kNone, region, region, 0, 0, 1, 1, 0, 0, 0, 1);
    }
    auto contribute = [&](size_t vertex, const Spec4& value, const std::string& src, bool nee) {
        L += value;
        if (rec && value.sum() > 0) rec->c.push_back(PathContribution{vertex, value, src, nee});
    };

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = world_.intersect(ray, Inf, hit);
        st.segments++;
        const Medium& med = world_.mediumOf(region);
        double nMed = med.n(hero);
        double seg = found ? hit.t : Inf;

        if (med.scatters()) {
            // Distance sampling with the hero's extinction; secondaries are reweighted.
            Spec4 sT = sigmaT(med, wl);
            double t = -std::log(1 - rng.uniform()) / sT[0];
            if (t < seg) {
                Vec3 p = ray.at(t);
                opl += nMed * t;
                double pdf0 = sT[0] * std::exp(-sT[0] * t);
                for (int i = 0; i < NW; ++i)
                    beta[i] *= med.scattering.eval(wl.lambda[size_t(i)]) * std::exp(-sT[i] * t) / pdf0;
                pushVertex(rec, p, ray.d, EventKind::Scatter, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta[0]);
                if (depth >= maxDepth_) break;
                SurfaceHit mh;
                mh.p = p;
                std::string src;
                Spec4 dl = directLighting(mh, nullptr, ray.d, wl, rng, rec ? &src : nullptr, med.g, region);
                if (!dl.isZero()) contribute(rec ? rec->v.size() - 1 : 0, beta * dl, src, true);
                Vec3 wi;
                Draw2 u(rng);
                double ph = sampleHg(-ray.d, med.g, u.u1, u.u2, wi);
                prevPdf = ph;
                specular = false;
                prevP = p;
                noteScatter(false);
                ray = Ray{p, wi};
                if (depth >= rrDepth_) {
                    double q = std::min(1.0, beta.maxValue());
                    if (rng.uniform() >= q) break;
                    beta *= 1 / q;
                }
                continue;
            }
            if (std::isinf(seg)) break;
            double pass0 = std::exp(-sT[0] * seg);
            for (int i = 0; i < NW; ++i) beta[i] *= std::exp(-sT[i] * seg) / pass0;
        } else if (med.absorbs()) {
            if (!found) break;  // infinite path through an absorbing medium
            for (int i = 0; i < NW; ++i) beta[i] *= std::exp(-med.absorption.eval(wl.lambda[size_t(i)]) * seg);
        }

        if (!found) {
            if (region == world_.ambientRegion() && !ownedByLightTracing()) {
                Spec4 sky, sun;
                for (int i = 0; i < NW; ++i) {
                    sky[i] = world_.env.sky(ray.d, wl.lambda[size_t(i)]);
                    sun[i] = world_.env.sun(ray.d, wl.lambda[size_t(i)]);
                }
                double w = (specular || sun.isZero()) ? 1.0 : powerHeuristic(prevPdf, sunPdf());
                if (rec)
                    pushVertex(rec, ray.o + ray.d * (2 * world_.sceneRadius() + length(ray.o - world_.sceneCenter())),
                               ray.d, EventKind::Escape, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta[0]);
                contribute(rec ? rec->v.size() - 1 : 0, beta * (sky + sun * w), sun.isZero() ? "sky" : "sun+sky", false);
            } else if (region != world_.ambientRegion()) {
                st.leaks++;
            }
            break;
        }

        opl += nMed * hit.t;
        const Boundary& b = world_.boundaries()[hit.boundary];
        bool fromFront = dot(ray.d, hit.n) < 0;
        uint32_t inc = fromFront ? b.front : b.back;
        uint32_t other = fromFront ? b.back : b.front;
        if (inc != region) {
            st.noteInconsistency(hit.boundary, region, inc, lastBoundary, hit.p);
            region = inc;
        }
        const SurfaceOptics& o = world_.optics()[b.optics];
        lastBoundary = hit.boundary;
        size_t vIndex = rec ? rec->v.size() : 0;
        if (rec) pushVertex(rec, hit.p, ray.d, EventKind::None, hit.boundary, inc, inc, nMed, world_.indexOf(other, hero),
                            1, 1, 0, 0, opl, beta[0]);

        // Emission.
        if (b.emission >= 0) {
            const Emission& e = world_.emissions()[b.emission];
            if (((fromFront && e.front) || (!fromFront && e.back)) && !ownedByLightTracing()) {
                double w = 1;
                if (!specular) {
                    double pl = lightPdf(prevP, hit);
                    if (pl > 0) w = powerHeuristic(prevPdf, pl);
                }
                contribute(vIndex, beta * wl.eval(e.radiance) * w, world_.boundaryLabel(hit.boundary), false);
            }
        }

        if (o.type == SurfaceType::Null) {
            if (rec) { rec->v[vIndex].event = EventKind::Pass; rec->v[vIndex].regionTo = other; rec->v[vIndex].T = 1; }
            region = other;
            ray.o = offsetOrigin(hit.p, hit.n, ray.d);
            if (depth >= 4 * maxDepth_) break;
            continue;
        }
        if (o.type == SurfaceType::Absorber || o.type == SurfaceType::Detector) {
            if (rec) rec->v[vIndex].event = o.type == SurfaceType::Detector ? EventKind::Detect : EventKind::Absorb;
            break;
        }
        if (depth >= maxDepth_) {
            if (rec) rec->v[vIndex].event = EventKind::Terminate;
            break;
        }

        Interface it = makeInterface(hit, hero);
        if (o.type == SurfaceType::Dielectric && !wl.secondaryTerminated()) {
            for (int i = 1; i < NW; ++i)
                if (wl.pdf[size_t(i)] != 0 && isDispersive(it, atWavelength(it, world_, hit, wl.lambda[size_t(i)]))) {
                    wl.terminateSecondary();
                    maskTerminated(wl, beta);
                    break;
                }
        }
        if (!o.isDelta()) {
            std::string src;
            Spec4 dl = directLighting(hit, &it, ray.d, wl, rng, rec ? &src : nullptr, 0, region);
            if (!dl.isZero()) contribute(vIndex, beta * dl, src, true);
        }
        ScatterSample s;
        Draw3 u(rng);
        bool ok = sampleScatter(it, ray.d, u.u1, u.u2, u.u3, TransportMode::Radiance, s);
        if (rec) {
            PathVertex& v = rec->v[vIndex];
            v.event = ok ? s.event : EventKind::Absorb;
            v.dOut = ok ? s.wi : ray.d;
            v.regionTo = ok && s.transmitted ? other : inc;
            v.thetaI = std::acos(clampd(s.cosI, 0, 1));
            v.thetaT = std::acos(clampd(s.cosT, 0, 1));
            v.R = s.R;
            v.T = s.T;
            v.beta = ok ? beta[0] * s.weight : 0;
        }
        if (!ok || s.weight <= 0) break;
        beta[0] *= s.weight;
        for (int i = 1; i < NW; ++i)
            if (wl.pdf[size_t(i)] != 0)
                beta[i] *= secondaryScatterWeight(atWavelength(it, world_, hit, wl.lambda[size_t(i)]), ray.d, s,
                                                  TransportMode::Radiance);
        if (s.transmitted) region = other;
        specular = s.delta;
        prevPdf = s.pdf;
        prevP = hit.p;
        noteScatter(s.delta);
        ray = Ray{offsetOrigin(hit.p, hit.n, s.wi), s.wi};
        if (depth >= rrDepth_) {
            double q = std::min(1.0, beta.maxValue());
            if (rng.uniform() >= q) {
                if (rec) pushVertex(rec, ray.o, ray.d, EventKind::Terminate, kNone, region, region, 0, 0, 1, 1, 0, 0, opl, 0);
                break;
            }
            beta *= 1 / q;
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
    Ray ray;
    uint32_t region;
    Spec4 beta;
    uint32_t emitterBoundary = kNone;
    bool fromSky = rng.uniform() < pSky;
    size_t li = 0;
    if (!fromSky) li = w.pickLight(rng.uniform());
    if (fromSky || w.lights()[li].kind == LightEntry::Kind::Sun) {
        // Distant emission (sky dome or sun): pick a direction, then a disk covering the world.
        double R = w.sceneRadius() * 1.001;
        Vec3 toward;  // direction from which the light arrives
        double pdfDir, sel;
        if (fromSky) {
            Draw2 u(rng);
            toward = sampleUniformSphere(u.u1, u.u2);
            pdfDir = 1 / (4 * Pi);
            for (int i = 0; i < NW; ++i) beta[i] = w.env.sky(toward, wl.lambda[size_t(i)]);
            sel = pSky;
        } else {
            double cosMax = std::cos(w.env.sunAngularRadius);
            Frame f(w.env.sunDir);
            Draw2 u(rng);
            toward = f.toWorld(sampleUniformCone(u.u1, u.u2, cosMax));
            pdfDir = 1 / w.env.sunSolidAngle();
            beta = wl.eval(w.env.sunRadiance);
            sel = (1 - pSky) * w.lightSelectPdf(li);
        }
        Vec3 dir = -toward;
        Vec3 a1, a2;
        orthonormalBasis(dir, a1, a2);
        Draw2 ud(rng);
        Vec2 dk = sampleUniformDiskConcentric(ud.u1, ud.u2);
        Vec3 o = w.sceneCenter() - dir * (1.01 * R) + (a1 * dk.x + a2 * dk.y) * R;
        ray = Ray{o, dir};
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
        Vec3 p = b.toWorld.point(pL), n = normalize(b.toWorld.vector(nL));
        double pdfA = 1.0 / b.shape->area();
        // Emitter seen directly by a virtual observer (path tracing owns this under the partition).
        if (det.isVirtual() && !partition_) {
            Vec3 q;
            int px, py;
            double factor;
            if (det.connect(p, rng, q, px, py, factor)) {
                Vec3 wv = q - p;
                double dist = length(wv);
                Vec3 wc = wv / dist;
                double cosL = dot(n, wc);
                if ((cosL > 0 && e.front) || (cosL < 0 && e.back)) {
                    uint32_t startRegion = cosL > 0 ? b.front : b.back, endRegion = kNone;
                    Spec4 Tr = transmittance(offsetOrigin(p, n, wc), startRegion, wc, dist, wl, &endRegion);
                    if (!Tr.isZero() && endRegion == det.region)
                        deposit(px, py, Le * Tr * (std::abs(cosL) * factor / (sel * pdfA)));
                }
            }
        }
        double sidePdf = 1;
        bool front = e.front;
        if (e.front && e.back) {
            front = rng.uniform() < 0.5;
            sidePdf = 0.5;
        }
        Vec3 ns = front ? n : -n;
        Frame f(ns);
        Draw2 uc(rng);
        Vec3 dir = f.toWorld(sampleCosineHemisphere(uc.u1, uc.u2));
        // β = Le·cos / (sel · pdfA · sidePdf · cos/π)
        beta = Le * (Pi / (sel * pdfA * sidePdf));
        ray = Ray{offsetOrigin(p, n, dir), dir};
        region = front ? b.front : b.back;
        emitterBoundary = L.boundary;
    }
    maskTerminated(wl, beta);
    if (beta.isZero()) return;
    if (rec) {
        rec->lambda = hero;
        pushVertex(rec, ray.o, ray.d, EventKind::Emit, emitterBoundary, region, region, 0, 0, 1, 1, 0, 0, 0, beta[0]);
    }
    double opl = 0;
    uint32_t lastBoundary = emitterBoundary;
    int targetBoundary = det.boundary();
    int nonDeltaEvents = 0, deltaEvents = 0;
    // Under the caustic partition, connect only at the first non-specular vertex after ≥1 specular one.
    auto mayConnect = [&] { return !partition_ || (nonDeltaEvents == 0 && deltaEvents >= 1); };
    // surface: the hit's interface at the hero wavelength (null for medium scattering).
    auto connect = [&](const SurfaceHit& hit, const Interface* surface, const Vec3& d, double g) {
        if (!det.isVirtual()) return;
        Vec3 q;
        int px, py;
        double factor;
        if (!det.connect(hit.p, rng, q, px, py, factor)) return;
        Vec3 wv = q - hit.p;
        double dist = length(wv);
        Vec3 wc = wv / dist;
        Spec4 f;
        double cosX = 1;
        Vec3 origin = hit.p;
        uint32_t startRegion = region;
        if (!surface) {
            f = Spec4(hgPhase(dot(-d, wc), g));
        } else {
            const Interface& base = *surface;
            for (int i = 0; i < NW; ++i) {
                if (wl.pdf[size_t(i)] == 0) continue;
                double pdfDummy;
                f[i] = evalScatter(i == 0 ? base : atWavelength(base, w, hit, wl.lambda[size_t(i)]), d, wc,
                                   TransportMode::Importance, pdfDummy);
            }
            if (f.isZero()) return;
            cosX = std::abs(dot(wc, hit.n));
            const Boundary& b = w.boundaries()[hit.boundary];
            startRegion = dot(wc, hit.n) > 0 ? b.front : b.back;
            origin = offsetOrigin(hit.p, hit.n, wc);
        }
        uint32_t endRegion = kNone;
        Spec4 Tr = transmittance(origin, startRegion, wc, dist, wl, &endRegion);
        if (Tr.isZero() || endRegion != det.region) return;
        deposit(px, py, beta * f * Tr * (cosX * factor));
    };

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = w.intersect(ray, Inf, hit);
        st.segments++;
        const Medium& med = w.mediumOf(region);
        double nMed = med.n(hero);
        double seg = found ? hit.t : Inf;
        if (med.scatters()) {
            Spec4 sT = sigmaT(med, wl);
            double t = -std::log(1 - rng.uniform()) / sT[0];
            if (t < seg) {
                Vec3 p = ray.at(t);
                opl += nMed * t;
                double pdf0 = sT[0] * std::exp(-sT[0] * t);
                for (int i = 0; i < NW; ++i)
                    beta[i] *= med.scattering.eval(wl.lambda[size_t(i)]) * std::exp(-sT[i] * t) / pdf0;
                pushVertex(rec, p, ray.d, EventKind::Scatter, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta[0]);
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
            if (std::isinf(seg)) break;
            double pass0 = std::exp(-sT[0] * seg);
            for (int i = 0; i < NW; ++i) beta[i] *= std::exp(-sT[i] * seg) / pass0;
        } else if (med.absorbs()) {
            if (!found) break;
            for (int i = 0; i < NW; ++i) beta[i] *= std::exp(-med.absorption.eval(wl.lambda[size_t(i)]) * seg);
        }
        if (!found) {
            pushVertex(rec, ray.o + ray.d * (2 * w.sceneRadius() + length(ray.o - w.sceneCenter())), ray.d,
                       EventKind::Escape, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta[0]);
            break;
        }
        opl += nMed * hit.t;
        const Boundary& b = w.boundaries()[hit.boundary];
        bool fromFront = dot(ray.d, hit.n) < 0;
        uint32_t inc = fromFront ? b.front : b.back;
        uint32_t other = fromFront ? b.back : b.front;
        if (inc != region) {
            st.noteInconsistency(hit.boundary, region, inc, lastBoundary, hit.p);
            region = inc;
        }
        const SurfaceOptics& o = w.optics()[b.optics];
        lastBoundary = hit.boundary;
        if (o.type == SurfaceType::Null) {
            pushVertex(rec, hit.p, ray.d, EventKind::Pass, hit.boundary, inc, other, nMed, w.indexOf(other, hero), 1, 1, 0,
                       1, opl, beta[0]);
            region = other;
            ray.o = offsetOrigin(hit.p, hit.n, ray.d);
            if (depth >= 4 * maxDepth_) break;
            continue;
        }
        // A diffuse measurement screen records incident flux and then scatters the reflected
        // fraction. An absorbing detector records flux and terminates the path as before.
        if (fromFront && int(hit.boundary) == targetBoundary) {
            int px, py;
            double area;
            if (det.pixelOfHit(hit, px, py, area)) deposit(px, py, beta * (1 / area));
        }
        if (o.type == SurfaceType::Detector) {
            pushVertex(rec, hit.p, ray.d, EventKind::Detect, hit.boundary, inc, inc, nMed, nMed, 1, 1, 0, 0, opl, beta[0]);
            break;
        }
        if (o.type == SurfaceType::Absorber) {
            pushVertex(rec, hit.p, ray.d, EventKind::Absorb, hit.boundary, inc, inc, nMed, nMed, 1, 1, 0, 0, opl, beta[0]);
            break;
        }
        Interface it = makeInterface(hit, hero);
        if (!o.isDelta()) {
            if (mayConnect()) connect(hit, &it, ray.d, 0);
            ++nonDeltaEvents;
            if (partition_) break;  // later vertices are owned by path tracing
        }
        if (depth >= maxDepth_) break;
        if (o.type == SurfaceType::Dielectric && !wl.secondaryTerminated()) {
            for (int i = 1; i < NW; ++i)
                if (wl.pdf[size_t(i)] != 0 && isDispersive(it, atWavelength(it, world_, hit, wl.lambda[size_t(i)]))) {
                    wl.terminateSecondary();
                    maskTerminated(wl, beta);
                    break;
                }
        }
        ScatterSample s;
        Draw3 u(rng);
        bool ok = sampleScatter(it, ray.d, u.u1, u.u2, u.u3, TransportMode::Importance, s);
        pushVertex(rec, hit.p, ok ? s.wi : ray.d, ok ? s.event : EventKind::Absorb, hit.boundary, inc,
                   ok && s.transmitted ? other : inc, nMed, w.indexOf(other, hero), s.cosI, s.cosT, s.R, s.T, opl,
                   ok ? beta[0] * s.weight : 0);
        if (!ok || s.weight <= 0) break;
        double w0 = s.weight;
        beta[0] *= s.weight;
        for (int i = 1; i < NW; ++i)
            if (wl.pdf[size_t(i)] != 0)
                beta[i] *= secondaryScatterWeight(atWavelength(it, w, hit, wl.lambda[size_t(i)]), ray.d, s,
                                                  TransportMode::Importance);
        if (s.delta) ++deltaEvents;
        if (s.transmitted) region = other;
        ray = Ray{offsetOrigin(hit.p, hit.n, s.wi), s.wi};
        if (depth >= rrDepth_) {
            // Particles carry flux, so roulette against the per-event throughput.
            double q = std::min(1.0, w0);
            if (q < 1) {
                if (rng.uniform() >= q) break;
                beta *= 1 / q;
            }
        }
    }
}

PathRecord Tracer::walk(Ray ray, uint32_t region, double lambda, WalkMode mode, Rng& rng, int maxEvents) const {
    PathRecord rec;
    rec.lambda = lambda;
    double beta = 1, opl = 0;
    pushVertex(&rec, ray.o, ray.d, EventKind::Emit, kNone, region, region, world_.indexOf(region, lambda),
               world_.indexOf(region, lambda), 1, 1, 0, 0, 0, 1);
    for (int ev = 0; ev < maxEvents; ++ev) {
        SurfaceHit hit;
        bool found = world_.intersect(ray, Inf, hit);
        const Medium& med = world_.mediumOf(region);
        double nMed = med.n(lambda);
        if (!found) {
            pushVertex(&rec, ray.o + ray.d * (2 * world_.sceneRadius() + length(ray.o - world_.sceneCenter())), ray.d,
                       EventKind::Escape, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta);
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
            pushVertex(&rec, hit.p, ray.d, EventKind::Pass, hit.boundary, inc, other, nMed, n2, 1, 1, 0, 1, opl, beta);
            region = other;
            ray.o = offsetOrigin(hit.p, hit.n, ray.d);
            continue;
        }
        if (o.type == SurfaceType::Absorber || o.type == SurfaceType::Detector ||
            (mode == WalkMode::PrimaryTransmission && o.type == SurfaceType::Diffuse)) {
            EventKind k = o.type == SurfaceType::Detector ? EventKind::Detect
                          : o.type == SurfaceType::Diffuse  ? EventKind::Diffuse
                                                             : EventKind::Absorb;
            double ci = std::abs(dot(ray.d, hit.n));
            pushVertex(&rec, hit.p, ray.d, k, hit.boundary, inc, inc, nMed, n2, ci, ci, 0, 0, opl, beta);
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
        pushVertex(&rec, hit.p, ok ? s.wi : ray.d, ok ? s.event : EventKind::Absorb, hit.boundary, inc,
                   ok && s.transmitted ? other : inc, nMed, n2, s.cosI, s.cosT, s.R, s.T, opl, ok ? beta * s.weight : 0);
        if (!ok) return rec;
        beta *= s.weight;
        if (s.transmitted) region = other;
        ray = Ray{offsetOrigin(hit.p, hit.n, s.wi), s.wi};
    }
    pushVertex(&rec, ray.o, ray.d, EventKind::Terminate, kNone, region, region, 0, 0, 1, 1, 0, 0, opl, beta);
    return rec;
}

}  // namespace owe
