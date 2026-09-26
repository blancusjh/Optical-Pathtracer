#include "transport.hpp"

#include "optics.hpp"

namespace owe {

namespace {

struct LightSample {
    Vec3 wi;
    double dist = Inf;
    double Le = 0;
    double pdf = 0;  // solid angle, including light selection
    uint32_t boundary = kNone;
};

bool sampleLightFrom(const World& w, const Vec3& p, double lambda, Rng& rng, LightSample& ls) {
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
        ls.Le = w.env.sunRadiance.eval(lambda);
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
    ls.Le = ((cosL > 0 && e.front) || (cosL < 0 && e.back)) ? e.radiance.eval(lambda) : 0;
    ls.pdf = sel * pdfW;
    ls.boundary = L.boundary;
    return true;
}

double sigmaT(const Medium& m, double lambda) {
    return m.absorption.eval(lambda) + m.scattering.eval(lambda);
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
    it.pLocal = hit.pLocal;
    it.nLocal = hit.nLocal;
    it.nFront = world_.indexOf(b.front, lambda);
    it.nBack = world_.indexOf(b.back, lambda);
    it.lambda = lambda;
    return it;
}

double Tracer::transmittance(Vec3 o, uint32_t region, const Vec3& dir, double dist, double lambda,
                             uint32_t* finalRegion) const {
    double Tr = 1;
    double remaining = dist;
    double margin = std::isinf(dist) ? 0 : 1e-7 * dist + 1e-12;
    for (int guard = 0; guard < 256; ++guard) {
        SurfaceHit h;
        double tmax = std::isinf(remaining) ? Inf : remaining - margin;
        if (tmax <= 0) break;
        bool found = world_.intersect(Ray{o, dir}, tmax, h);
        double seg = found ? h.t : remaining;
        const Medium& m = world_.mediumOf(region);
        double st = sigmaT(m, lambda);
        if (st > 0) {
            if (std::isinf(seg)) return 0;
            Tr *= std::exp(-st * seg);
        }
        if (!found) {
            if (std::isinf(dist) && region != world_.ambientRegion()) return 0;
            if (finalRegion) *finalRegion = region;
            return Tr;
        }
        const Boundary& b = world_.boundaries()[h.boundary];
        if (world_.optics()[b.optics].type != SurfaceType::Null) return 0;
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

double Tracer::directLighting(const Interface& it, const SurfaceHit& hit, const Vec3& d, double lambda, Rng& rng,
                              std::string* source, bool isMedium, double g, uint32_t region) const {
    LightSample ls;
    if (!sampleLightFrom(world_, hit.p, lambda, rng, ls) || ls.Le <= 0 || ls.pdf <= 0) return 0;
    double f, pdfScatter, cosFactor;
    uint32_t startRegion;
    Vec3 origin;
    if (isMedium) {
        f = hgPhase(dot(-d, ls.wi), g);
        pdfScatter = f;
        cosFactor = 1;
        startRegion = region;
        origin = hit.p;
    } else {
        f = evalScatter(it, d, ls.wi, TransportMode::Radiance, pdfScatter);
        if (f <= 0) return 0;
        cosFactor = std::abs(dot(ls.wi, hit.n));
        const Boundary& b = world_.boundaries()[hit.boundary];
        startRegion = dot(ls.wi, hit.n) > 0 ? b.front : b.back;
        origin = offsetOrigin(hit.p, hit.n, ls.wi);
    }
    double Tr = transmittance(origin, startRegion, ls.wi, ls.dist, lambda);
    if (Tr <= 0) return 0;
    double w = powerHeuristic(ls.pdf, pdfScatter);
    if (source) *source = ls.boundary == kNone ? std::string("sun") : world_.boundaryLabel(ls.boundary);
    return f * cosFactor * ls.Le * Tr * w / ls.pdf;
}

double Tracer::radiance(Ray ray, uint32_t region, double lambda, Rng& rng, TransportStats& st, PathRecord* rec) const {
    double L = 0, beta = 1, opl = 0;
    bool specular = true;
    double prevPdf = 0;
    Vec3 prevP = ray.o;
    st.paths++;
    if (rec) {
        rec->lambda = lambda;
        pushVertex(rec, ray.o, ray.d, EventKind::Camera, kNone, region, region, 0, 0, 1, 1, 0, 0, 0, 1);
    }
    auto contribute = [&](double value, const std::string& src, bool nee) {
        // value already includes beta
        if (rec && value > 0) rec->c.push_back(PathContribution{rec->v.size() - 1, value, src, nee});
    };

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = world_.intersect(ray, Inf, hit);
        st.segments++;
        const Medium& med = world_.mediumOf(region);
        double nMed = med.n(lambda);
        double seg = found ? hit.t : Inf;

        if (med.scatters()) {
            double sS = med.scattering.eval(lambda), sA = med.absorption.eval(lambda), sT = sS + sA;
            double t = -std::log(1 - rng.uniform()) / sT;
            if (t < seg) {
                Vec3 p = ray.at(t);
                opl += nMed * t;
                beta *= sS / sT;
                pushVertex(rec, p, ray.d, EventKind::Scatter, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta);
                if (depth >= maxDepth_) break;
                SurfaceHit mh;
                mh.p = p;
                Interface dummy;
                std::string src;
                double dl = directLighting(dummy, mh, ray.d, lambda, rng, rec ? &src : nullptr, true, med.g, region);
                if (dl > 0) { L += beta * dl; contribute(beta * dl, src, true); }
                Vec3 wi;
                double ph = sampleHg(-ray.d, med.g, rng.uniform(), rng.uniform(), wi);
                prevPdf = ph;
                specular = false;
                prevP = p;
                ray = Ray{p, wi};
                if (depth >= rrDepth_) {
                    double q = std::min(1.0, beta);
                    if (rng.uniform() >= q) break;
                    beta /= q;
                }
                continue;
            }
        } else if (med.absorbs()) {
            if (!found) break;  // infinite path through an absorbing medium
            beta *= std::exp(-med.absorption.eval(lambda) * seg);
        }

        if (!found) {
            if (region == world_.ambientRegion()) {
                double sky = world_.env.sky(ray.d, lambda);
                double sun = world_.env.sun(ray.d, lambda);
                double w = (specular || sun <= 0) ? 1.0 : powerHeuristic(prevPdf, sunPdf());
                double v = beta * (sky + sun * w);
                L += v;
                if (rec) {
                    pushVertex(rec, ray.o + ray.d * (2 * world_.sceneRadius() + length(ray.o - world_.sceneCenter())),
                               ray.d, EventKind::Escape, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta);
                    contribute(v, sun > 0 ? "sun+sky" : "sky", false);
                }
            } else {
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
            st.inconsistencies++;
            region = inc;
        }

        // Emission.
        if (b.emission >= 0) {
            const Emission& e = world_.emissions()[b.emission];
            if ((fromFront && e.front) || (!fromFront && e.back)) {
                double Le = e.radiance.eval(lambda);
                double w = 1;
                if (!specular) {
                    double pl = lightPdf(prevP, hit);
                    if (pl > 0) w = powerHeuristic(prevPdf, pl);
                }
                double v = beta * Le * w;
                L += v;
                if (rec) {
                    pushVertex(rec, hit.p, ray.d, EventKind::Emit, hit.boundary, inc, inc, nMed, nMed, 1, 1, 0, 0, opl, beta);
                    contribute(v, world_.boundaryLabel(hit.boundary), false);
                    rec->v.pop_back();  // re-added below as the actual interaction
                }
            }
        }

        const SurfaceOptics& o = world_.optics()[b.optics];
        if (o.type == SurfaceType::Null) {
            double n2 = world_.indexOf(other, lambda);
            pushVertex(rec, hit.p, ray.d, EventKind::Pass, hit.boundary, inc, other, nMed, n2, 1, 1, 0, 1, opl, beta);
            region = other;
            ray.o = offsetOrigin(hit.p, hit.n, ray.d);
            if (depth >= 4 * maxDepth_) break;
            continue;
        }
        if (o.type == SurfaceType::Absorber || o.type == SurfaceType::Detector) {
            pushVertex(rec, hit.p, ray.d, o.type == SurfaceType::Detector ? EventKind::Detect : EventKind::Absorb,
                       hit.boundary, inc, inc, nMed, nMed, 1, 1, 0, 0, opl, beta);
            break;
        }
        if (depth >= maxDepth_) {
            pushVertex(rec, hit.p, ray.d, EventKind::Terminate, hit.boundary, inc, inc, nMed, nMed, 1, 1, 0, 0, opl, beta);
            break;
        }

        Interface it = makeInterface(hit, lambda);
        size_t vIndex = rec ? rec->v.size() : 0;
        if (rec) pushVertex(rec, hit.p, ray.d, EventKind::None, hit.boundary, inc, inc, nMed, nMed, 1, 1, 0, 0, opl, beta);
        if (!o.isDelta()) {
            std::string src;
            double dl = directLighting(it, hit, ray.d, lambda, rng, rec ? &src : nullptr, false, 0, region);
            if (dl > 0) {
                L += beta * dl;
                if (rec) rec->c.push_back(PathContribution{vIndex, beta * dl, src, true});
            }
        }
        ScatterSample s;
        bool ok = sampleScatter(it, ray.d, rng.uniform(), rng.uniform(), rng.uniform(), TransportMode::Radiance, s);
        if (rec) {
            PathVertex& v = rec->v[vIndex];
            v.event = ok ? s.event : EventKind::Absorb;
            v.dOut = ok ? s.wi : ray.d;
            v.regionTo = ok && s.transmitted ? other : inc;
            v.nT = world_.indexOf(other, lambda);
            v.thetaI = std::acos(clampd(s.cosI, 0, 1));
            v.thetaT = std::acos(clampd(s.cosT, 0, 1));
            v.R = s.R;
            v.T = s.T;
            v.beta = ok ? beta * s.weight : 0;
        }
        if (!ok || s.weight <= 0) break;
        beta *= s.weight;
        if (s.transmitted) region = other;
        specular = s.delta;
        prevPdf = s.pdf;
        prevP = hit.p;
        ray = Ray{offsetOrigin(hit.p, hit.n, s.wi), s.wi};
        if (depth >= rrDepth_) {
            double q = std::min(1.0, beta);
            if (rng.uniform() >= q) {
                if (rec) pushVertex(rec, ray.o, ray.d, EventKind::Terminate, kNone, region, region, 0, 0, 1, 1, 0, 0, opl, 0);
                break;
            }
            beta /= q;
        }
    }
    return L;
}

void Tracer::traceParticle(const Detector& det, double lambda, const double scale[3], Rng& rng, Film& film,
                           TransportStats& st, PathRecord* rec) const {
    const World& w = world_;
    double skyPower = w.environmentPower();
    double lightPower = w.lights().empty() ? 0.0 : w.totalLightPower();
    if (skyPower + lightPower <= 0) return;
    st.paths++;
    double pSky = skyPower / (skyPower + lightPower);
    auto deposit = [&](int px, int py, double value) {
        film.add(px, py, value * scale[0], value * scale[1], value * scale[2]);
    };
    Ray ray;
    uint32_t region;
    double beta;
    uint32_t emitterBoundary = kNone;
    bool fromSky = rng.uniform() < pSky;
    size_t li = 0;
    if (!fromSky) li = w.pickLight(rng.uniform());
    if (fromSky || w.lights()[li].kind == LightEntry::Kind::Sun) {
        // Distant emission (sky dome or sun): pick a direction, then a disk covering the world.
        double R = w.sceneRadius() * 1.001;
        Vec3 toward;  // direction from which the light arrives
        double pdfDir, L;
        double sel;
        if (fromSky) {
            toward = sampleUniformSphere(rng.uniform(), rng.uniform());
            pdfDir = 1 / (4 * Pi);
            L = w.env.sky(toward, lambda);
            sel = pSky;
        } else {
            double cosMax = std::cos(w.env.sunAngularRadius);
            Frame f(w.env.sunDir);
            toward = f.toWorld(sampleUniformCone(rng.uniform(), rng.uniform(), cosMax));
            pdfDir = 1 / w.env.sunSolidAngle();
            L = w.env.sunRadiance.eval(lambda);
            sel = (1 - pSky) * w.lightSelectPdf(li);
        }
        Vec3 dir = -toward;
        Vec3 a1, a2;
        orthonormalBasis(dir, a1, a2);
        Vec2 dk = sampleUniformDiskConcentric(rng.uniform(), rng.uniform());
        Vec3 o = w.sceneCenter() - dir * (1.01 * R) + (a1 * dk.x + a2 * dk.y) * R;
        ray = Ray{o, dir};
        region = w.ambientRegion();
        beta = L * Pi * R * R / (pdfDir * sel);
    } else {
        double sel = (1 - pSky) * w.lightSelectPdf(li);
        const LightEntry& L = w.lights()[li];
        const Boundary& b = w.boundaries()[L.boundary];
        const Emission& e = w.emissions()[b.emission];
        Vec3 pL, nL;
        b.shape->sampleArea(rng.uniform(), rng.uniform(), rng.uniform(), pL, nL);
        Vec3 p = b.toWorld.point(pL), n = normalize(b.toWorld.vector(nL));
        double pdfA = 1.0 / b.shape->area();
        // Emitter seen directly by a virtual observer.
        if (det.isVirtual()) {
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
                    double Tr = transmittance(offsetOrigin(p, n, wc), startRegion, wc, dist, lambda, &endRegion);
                    if (Tr > 0 && endRegion == det.region)
                        deposit(px, py, e.radiance.eval(lambda) * std::abs(cosL) * factor * Tr / (sel * pdfA));
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
        Vec3 dir = f.toWorld(sampleCosineHemisphere(rng.uniform(), rng.uniform()));
        // β = Le·cos / (sel · pdfA · sidePdf · cos/π)
        beta = e.radiance.eval(lambda) * Pi / (sel * pdfA * sidePdf);
        ray = Ray{offsetOrigin(p, n, dir), dir};
        region = front ? b.front : b.back;
        emitterBoundary = L.boundary;
    }
    if (beta <= 0) return;
    if (rec) {
        rec->lambda = lambda;
        pushVertex(rec, ray.o, ray.d, EventKind::Emit, emitterBoundary, region, region, 0, 0, 1, 1, 0, 0, 0, beta);
    }
    double opl = 0;
    int targetBoundary = det.boundary();
    auto connect = [&](const SurfaceHit& hit, const Vec3& d, const Interface* it, bool isMedium, double g) {
        if (!det.isVirtual()) return;
        Vec3 q;
        int px, py;
        double factor;
        if (!det.connect(hit.p, rng, q, px, py, factor)) return;
        Vec3 wv = q - hit.p;
        double dist = length(wv);
        Vec3 wc = wv / dist;
        double f, pdfDummy, cosX;
        Vec3 origin;
        uint32_t startRegion;
        if (isMedium) {
            f = hgPhase(dot(-d, wc), g);
            cosX = 1;
            origin = hit.p;
            startRegion = region;
        } else {
            f = evalScatter(*it, d, wc, TransportMode::Importance, pdfDummy);
            if (f <= 0) return;
            cosX = std::abs(dot(wc, hit.n));
            const Boundary& b = w.boundaries()[hit.boundary];
            startRegion = dot(wc, hit.n) > 0 ? b.front : b.back;
            origin = offsetOrigin(hit.p, hit.n, wc);
        }
        uint32_t endRegion = kNone;
        double Tr = transmittance(origin, startRegion, wc, dist, lambda, &endRegion);
        if (Tr <= 0 || endRegion != det.region) return;
        deposit(px, py, beta * f * cosX * factor * Tr);
    };

    for (int depth = 0;; ++depth) {
        SurfaceHit hit;
        bool found = w.intersect(ray, Inf, hit);
        st.segments++;
        const Medium& med = w.mediumOf(region);
        double nMed = med.n(lambda);
        double seg = found ? hit.t : Inf;
        if (med.scatters()) {
            double sS = med.scattering.eval(lambda), sA = med.absorption.eval(lambda), sT = sS + sA;
            double t = -std::log(1 - rng.uniform()) / sT;
            if (t < seg) {
                Vec3 p = ray.at(t);
                opl += nMed * t;
                beta *= sS / sT;
                pushVertex(rec, p, ray.d, EventKind::Scatter, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta);
                SurfaceHit mh;
                mh.p = p;
                connect(mh, ray.d, nullptr, true, med.g);
                if (depth >= maxDepth_) break;
                Vec3 wi;
                sampleHg(-ray.d, med.g, rng.uniform(), rng.uniform(), wi);
                ray = Ray{p, wi};
                if (depth >= rrDepth_) {
                    double q = std::min(1.0, beta / 1.0);
                    if (q < 1) {
                        if (rng.uniform() >= q) break;
                        beta /= q;
                    }
                }
                continue;
            }
        } else if (med.absorbs()) {
            if (!found) break;
            beta *= std::exp(-med.absorption.eval(lambda) * seg);
        }
        if (!found) {
            pushVertex(rec, ray.o + ray.d * (2 * w.sceneRadius() + length(ray.o - w.sceneCenter())), ray.d,
                       EventKind::Escape, kNone, region, region, nMed, nMed, 1, 1, 0, 0, opl, beta);
            break;
        }
        opl += nMed * hit.t;
        const Boundary& b = w.boundaries()[hit.boundary];
        bool fromFront = dot(ray.d, hit.n) < 0;
        uint32_t inc = fromFront ? b.front : b.back;
        uint32_t other = fromFront ? b.back : b.front;
        if (inc != region) {
            st.inconsistencies++;
            region = inc;
        }
        const SurfaceOptics& o = w.optics()[b.optics];
        if (o.type == SurfaceType::Null) {
            pushVertex(rec, hit.p, ray.d, EventKind::Pass, hit.boundary, inc, other, nMed, w.indexOf(other, lambda), 1, 1, 0,
                       1, opl, beta);
            region = other;
            ray.o = offsetOrigin(hit.p, hit.n, ray.d);
            if (depth >= 4 * maxDepth_) break;
            continue;
        }
        if (o.type == SurfaceType::Detector) {
            pushVertex(rec, hit.p, ray.d, EventKind::Detect, hit.boundary, inc, inc, nMed, nMed, 1, 1, 0, 0, opl, beta);
            if (fromFront && int(hit.boundary) == targetBoundary) {
                int px, py;
                double area;
                if (det.pixelOfHit(hit, px, py, area)) deposit(px, py, beta / area);
            }
            break;
        }
        if (o.type == SurfaceType::Absorber) {
            pushVertex(rec, hit.p, ray.d, EventKind::Absorb, hit.boundary, inc, inc, nMed, nMed, 1, 1, 0, 0, opl, beta);
            break;
        }
        Interface it = makeInterface(hit, lambda);
        if (!o.isDelta()) connect(hit, ray.d, &it, false, 0);
        if (depth >= maxDepth_) break;
        ScatterSample s;
        bool ok = sampleScatter(it, ray.d, rng.uniform(), rng.uniform(), rng.uniform(), TransportMode::Importance, s);
        pushVertex(rec, hit.p, ok ? s.wi : ray.d, ok ? s.event : EventKind::Absorb, hit.boundary, inc,
                   ok && s.transmitted ? other : inc, nMed, w.indexOf(other, lambda), s.cosI, s.cosT, s.R, s.T, opl,
                   ok ? beta * s.weight : 0);
        if (!ok || s.weight <= 0) break;
        beta *= s.weight;
        if (s.transmitted) region = other;
        ray = Ray{offsetOrigin(hit.p, hit.n, s.wi), s.wi};
        if (depth >= rrDepth_) {
            // Particles carry flux, so roulette against the throughput relative to the start.
            double q = std::min(1.0, s.weight);
            if (q < 1) {
                if (rng.uniform() >= q) break;
                beta /= q;
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
            ok = sampleScatter(it, ray.d, 1.0, rng.uniform(), rng.uniform(), TransportMode::Importance, s);
            if (!ok) ok = sampleScatter(it, ray.d, 0.0, rng.uniform(), rng.uniform(), TransportMode::Importance, s);
            if (ok && s.delta) s.weight = s.transmitted ? s.T : s.R;
        } else {
            ok = sampleScatter(it, ray.d, rng.uniform(), rng.uniform(), rng.uniform(), TransportMode::Importance, s);
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
