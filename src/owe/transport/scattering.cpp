#include "owe/transport/scattering.hpp"

#include <complex>

#include "owe/transport/optics.hpp"

namespace owe {

const char* eventName(EventKind k) {
    switch (k) {
        case EventKind::None: return "none";
        case EventKind::Emit: return "emit";
        case EventKind::Camera: return "camera";
        case EventKind::Reflect: return "reflect";
        case EventKind::Refract: return "refract";
        case EventKind::TIR: return "TIR";
        case EventKind::Glossy: return "glossy";
        case EventKind::GlossyT: return "glossy-transmit";
        case EventKind::Diffuse: return "diffuse";
        case EventKind::Scatter: return "scatter";
        case EventKind::Pass: return "pass";
        case EventKind::Absorb: return "absorb";
        case EventKind::Detect: return "detect";
        case EventKind::Escape: return "escape";
        case EventKind::Terminate: return "terminate";
    }
    return "?";
}

namespace {

double screenAimPdf(const Interface& it, const Vec3& wi, bool generatingDisk = false) {
    const auto& o = *it.optics;
    if (!(o.sampleAimRadius > 0)) return 0;
    double dn = dot(wi, o.sampleAimNormal);
    if (std::abs(dn) < 1e-12) return 0;
    double t = dot(o.sampleAimCenter - it.p, o.sampleAimNormal) / dn;
    if (t <= 0 || (!generatingDisk && lengthSq(it.p + wi * t - o.sampleAimCenter) > sqr(o.sampleAimRadius))) return 0;
    return t * t / (std::abs(dn) * Pi * sqr(o.sampleAimRadius));
}

std::complex<double> conductorEta(const SurfaceOptics& o, double lambda, double nMedium) {
    return std::complex<double>(o.conductor.n.eval(lambda), o.conductor.k.eval(lambda)) / nMedium;
}

// --- rough dielectric (Walter et al. 2007, pbrt-v4 formulation) in local frame; eta = nBack/nFront.
bool sampleRoughDielectric(const GGX& mf, const Vec3& wo, double eta, double uc, double u1, double u2,
                           TransportMode mode, ScatterSample& s, Vec3& wiLocal) {
    Vec3 wm = mf.sampleWm(wo, u1, u2);
    double R = fresnelDielectric(dot(wo, wm), eta), T = 1 - R;
    s.R = R;
    s.T = T;
    if (uc < R / (R + T)) {
        wiLocal = reflect(-wo, wm);
        if (!sameHemisphere(wo, wiLocal)) return false;
        s.pdf = mf.pdf(wo, wm) / (4 * std::abs(dot(wo, wm))) * R / (R + T);
        double f = mf.D(wm) * mf.G(wo, wiLocal) * R / (4 * absCosTheta(wiLocal) * absCosTheta(wo));
        s.weight = f * absCosTheta(wiLocal) / s.pdf;
        s.event = EventKind::Glossy;
        s.transmitted = false;
    } else {
        double etap;
        if (!refractLocal(wo, wm, eta, etap, wiLocal)) return false;
        if (sameHemisphere(wo, wiLocal) || wiLocal.z == 0) return false;
        double denom = sqr(dot(wiLocal, wm) + dot(wo, wm) / etap);
        double dwmdwi = std::abs(dot(wiLocal, wm)) / denom;
        s.pdf = mf.pdf(wo, wm) * dwmdwi * T / (R + T);
        double ft = T * mf.D(wm) * mf.G(wo, wiLocal) *
                    std::abs(dot(wiLocal, wm) * dot(wo, wm) / (cosTheta(wiLocal) * cosTheta(wo) * denom));
        if (mode == TransportMode::Radiance) ft /= sqr(etap);
        s.weight = ft * absCosTheta(wiLocal) / s.pdf;
        s.event = EventKind::GlossyT;
        s.transmitted = true;
    }
    s.delta = false;
    return std::isfinite(s.weight) && s.pdf > 0;
}

double evalRoughDielectric(const GGX& mf, const Vec3& wo, const Vec3& wi, double eta, TransportMode mode, double& pdf) {
    pdf = 0;
    double co = cosTheta(wo), ci = cosTheta(wi);
    if (co == 0 || ci == 0) return 0;
    bool refl = co * ci > 0;
    double etap = 1;
    if (!refl) etap = co > 0 ? eta : 1 / eta;
    Vec3 wm = wi * etap + wo;
    if (lengthSq(wm) == 0) return 0;
    wm = normalize(wm);
    if (wm.z < 0) wm = -wm;
    if (dot(wm, wi) * ci < 0 || dot(wm, wo) * co < 0) return 0;
    double R = fresnelDielectric(dot(wo, wm), eta), T = 1 - R;
    if (refl) {
        pdf = mf.pdf(wo, wm) / (4 * std::abs(dot(wo, wm))) * R / (R + T);
        return mf.D(wm) * mf.G(wo, wi) * R / std::abs(4 * ci * co);
    }
    double denom = sqr(dot(wi, wm) + dot(wo, wm) / etap);
    pdf = mf.pdf(wo, wm) * std::abs(dot(wi, wm)) / denom * T / (R + T);
    double ft = mf.D(wm) * T * mf.G(wo, wi) * std::abs(dot(wi, wm) * dot(wo, wm) / (denom * ci * co));
    if (mode == TransportMode::Radiance) ft /= sqr(etap);
    return ft;
}

}  // namespace

bool sampleScatter(const Interface& it, const Vec3& d, double uc, double u1, double u2, TransportMode mode,
                   ScatterSample& s) {
    const SurfaceOptics& o = *it.optics;
    Frame frame(it.n);
    Vec3 wo = frame.toLocal(-d);
    bool fromFront = wo.z > 0;
    s = ScatterSample{};
    s.cosI = std::abs(wo.z);

    switch (o.type) {
        case SurfaceType::Null:
        case SurfaceType::Absorber:
        case SurfaceType::Detector:
        case SurfaceType::StainedGlass:  // GPU only (the CPU backend refuses such scenes)
            return false;

        case SurfaceType::Dielectric: {
            double eta = it.nBack / it.nFront;
            if (o.roughness < 1e-3 || eta == 1) {
                double ni = fromFront ? it.nFront : it.nBack;
                double nt = fromFront ? it.nBack : it.nFront;
                double R = fresnelDielectric(wo.z, eta);
                s.R = R;
                s.T = 1 - R;
                s.delta = true;
                s.pdf = 0;
                double pR = R;
                if (it.fresnelFloor > 0 && R < 1) pR = clampd(R, it.fresnelFloor, 1 - it.fresnelFloor);
                if (uc < pR) {
                    s.wi = frame.toWorld(Vec3(-wo.x, -wo.y, wo.z));
                    s.weight = R / pR;
                    s.event = R >= 1 ? EventKind::TIR : EventKind::Reflect;
                    s.transmitted = false;
                    s.cosT = s.cosI;
                } else {
                    Vec3 wt;
                    if (!refractDirection(d, it.n, ni, nt, wt)) return false;
                    s.wi = wt;
                    s.weight = (1 - R) / (1 - pR) * (mode == TransportMode::Radiance ? sqr(ni / nt) : 1.0);
                    s.event = EventKind::Refract;
                    s.transmitted = true;
                    s.cosT = std::abs(dot(wt, it.n));
                }
                return true;
            }
            GGX mf(o.roughness);
            Vec3 wiL;
            if (!sampleRoughDielectric(mf, wo, eta, uc, u1, u2, mode, s, wiL)) return false;
            s.wi = frame.toWorld(wiL);
            s.cosT = std::abs(wiL.z);
            return true;
        }

        case SurfaceType::Diffuse: {
            if (!fromFront && o.backAbsorbs) return false;
            double albedo = surfaceAlbedo(it);
            if (albedo <= 0) return false;
            Vec3 w = sampleCosineHemisphere(u1, u2);
            if (!fromFront) w.z = -w.z;
            s.wi = frame.toWorld(w);
            s.pdf = std::abs(w.z) * InvPi;
            s.weight = albedo;
            double share = fromFront && mode == TransportMode::Radiance ? o.sampleAimShare : 0;
            if (share > 0) {
                if (uc < share) {
                    Vec3 a, b;
                    orthonormalBasis(o.sampleAimNormal, a, b);
                    Vec2 disk = sampleUniformDiskConcentric(u1, u2);
                    s.wi = normalize(o.sampleAimCenter + (a * disk.x + b * disk.y) * o.sampleAimRadius - it.p);
                }
                double c = dot(s.wi, it.n);
                if (c <= 0) return false;
                s.pdf = (1 - share) * c * InvPi + share * screenAimPdf(it, s.wi, uc < share);
                s.weight = albedo * c * InvPi / s.pdf;
            }
            s.delta = false;
            s.event = EventKind::Diffuse;
            s.R = albedo;
            s.cosT = std::abs(dot(s.wi, it.n));
            return s.pdf > 0;
        }

        case SurfaceType::Mirror: {
            if (!fromFront && o.backAbsorbs) return false;
            double r = o.reflectance.eval(it.lambda);
            if (r <= 0) return false;
            s.wi = frame.toWorld(Vec3(-wo.x, -wo.y, wo.z));
            s.weight = r;
            s.delta = true;
            s.event = EventKind::Reflect;
            s.R = r;
            s.cosT = s.cosI;
            return true;
        }

        case SurfaceType::Conductor: {
            if (!fromFront && o.backAbsorbs) return false;
            double nMed = fromFront ? it.nFront : it.nBack;
            auto eta = conductorEta(o, it.lambda, nMed);
            Vec3 w = fromFront ? wo : Vec3(wo.x, wo.y, -wo.z);  // work in the upper hemisphere
            if (o.roughness < 1e-3) {
                double F = fresnelConductor(w.z, eta) * o.reflectance.eval(it.lambda);
                s.wi = frame.toWorld(Vec3(-wo.x, -wo.y, wo.z));
                s.weight = F;
                s.delta = true;
                s.event = EventKind::Reflect;
                s.R = F;
                s.cosT = s.cosI;
                return F > 0;
            }
            GGX mf(o.roughness);
            Vec3 wm = mf.sampleWm(w, u1, u2);
            Vec3 wiL = reflect(-w, wm);
            if (wiL.z <= 0) return false;
            double pdf = mf.pdf(w, wm) / (4 * std::abs(dot(w, wm)));
            double F = fresnelConductor(std::abs(dot(w, wm)), eta) * o.reflectance.eval(it.lambda);
            double f = mf.D(wm) * F * mf.G(w, wiL) / (4 * wiL.z * w.z);
            if (!(pdf > 0)) return false;
            if (!fromFront) wiL.z = -wiL.z;
            s.wi = frame.toWorld(wiL);
            s.pdf = pdf;
            s.weight = f * std::abs(wiL.z) / pdf;
            s.delta = false;
            s.event = EventKind::Glossy;
            s.R = F;
            s.cosT = std::abs(wiL.z);
            return std::isfinite(s.weight);
        }
    }
    return false;
}

double evalScatter(const Interface& it, const Vec3& d, const Vec3& wiWorld, TransportMode mode, double& pdf) {
    pdf = 0;
    const SurfaceOptics& o = *it.optics;
    if (o.isDelta()) return 0;
    Frame frame(it.n);
    Vec3 wo = frame.toLocal(-d);
    Vec3 wi = frame.toLocal(wiWorld);
    bool fromFront = wo.z > 0;
    switch (o.type) {
        case SurfaceType::Diffuse: {
            if (!fromFront && o.backAbsorbs) return 0;
            if (!sameHemisphere(wo, wi)) return 0;
            pdf = std::abs(wi.z) * InvPi;
            double share = fromFront && mode == TransportMode::Radiance ? o.sampleAimShare : 0;
            if (share > 0) pdf = (1 - share) * pdf + share * screenAimPdf(it, wiWorld);
            return surfaceAlbedo(it) * InvPi;
        }
        case SurfaceType::Conductor: {
            if (!fromFront && o.backAbsorbs) return 0;
            if (!sameHemisphere(wo, wi)) return 0;
            if (!fromFront) { wo.z = -wo.z; wi.z = -wi.z; }
            GGX mf(o.roughness);
            Vec3 wm = wo + wi;
            if (lengthSq(wm) == 0) return 0;
            wm = normalize(wm);
            double nMed = fromFront ? it.nFront : it.nBack;
            double F = fresnelConductor(std::abs(dot(wo, wm)), conductorEta(o, it.lambda, nMed)) *
                       o.reflectance.eval(it.lambda);
            pdf = mf.pdf(wo, wm) / (4 * std::abs(dot(wo, wm)));
            return mf.D(wm) * F * mf.G(wo, wi) / (4 * wi.z * wo.z);
        }
        case SurfaceType::Dielectric: {
            GGX mf(o.roughness);
            return evalRoughDielectric(mf, wo, wi, it.nBack / it.nFront, mode, pdf);
        }
        default: return 0;
    }
}

double surfaceAlbedo(const Interface& it) {
    const SurfaceOptics& o = *it.optics;
    if (o.texture.kind == Texture::Kind::None) return o.reflectance.eval(it.lambda);
    if (it.texReady) return o.texture.mix(it.tex, it.lambda);
    return o.texture.eval(it.pLocal, it.nLocal, it.lambda);
}

bool isDispersive(const Interface& hero, const Interface& other) {
    double a = hero.nBack / hero.nFront, b = other.nBack / other.nFront;
    return std::abs(a - b) > 1e-12 * std::abs(a);
}

double secondaryScatterWeight(const Interface& it, const Vec3& d, const ScatterSample& s, TransportMode mode) {
    const SurfaceOptics& o = *it.optics;
    if (!s.delta) {
        double pdf;
        double f = evalScatter(it, d, s.wi, mode, pdf);
        return s.pdf > 0 ? f * std::abs(dot(s.wi, it.n)) / s.pdf : 0.0;
    }
    double cosI = std::abs(dot(d, it.n));
    switch (o.type) {
        case SurfaceType::Dielectric:
            return s.weight;  // non-dispersive: identical Fresnel split and n² factor
        case SurfaceType::Mirror:
            return o.reflectance.eval(it.lambda);
        case SurfaceType::Conductor: {
            bool fromFront = dot(d, it.n) < 0;
            double nMed = fromFront ? it.nFront : it.nBack;
            return fresnelConductor(cosI, conductorEta(o, it.lambda, nMed)) * o.reflectance.eval(it.lambda);
        }
        default: return 0;
    }
}

}  // namespace owe
