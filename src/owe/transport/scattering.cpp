#include "owe/transport/scattering.hpp"

#include "owe/core/shared/optics.hpp"

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
    return diskAimPdf(it.p, wi, o.sampleAimCenter, o.sampleAimNormal, o.sampleAimRadius, generatingDisk);
}

Vec2 conductorEta(const SurfaceOptics& o, double lambda, double nMedium) {
    return Vec2(o.conductor.n.eval(lambda), o.conductor.k.eval(lambda)) / nMedium;
}

// A lobe sample (local frame) as the interface's world sample.
void fromLobe(const Frame& frame, const LobeSample& l, EventKind event, ScatterSample& s) {
    s.wi = frame.toWorld(l.wi);
    s.weight = l.weight;
    s.pdf = l.pdf;
    s.delta = false;
    s.transmitted = l.transmitted;
    s.event = event;
    s.R = l.R;
    s.T = l.T;
    s.cosT = std::abs(l.wi.z);
}

// A specular reflection about the frame's normal with the given weight.
bool specularReflection(const Frame& frame, const Vec3& wo, double weight, ScatterSample& s) {
    s.wi = frame.toWorld(Vec3(-wo.x, -wo.y, wo.z));
    s.weight = weight;
    s.delta = true;
    s.event = EventKind::Reflect;
    s.R = weight;
    s.cosT = s.cosI;
    return weight > 0;
}

}  // namespace

bool sampleScatter(const Interface& it, const Vec3& d, double uc, double u1, double u2, TransportMode mode,
                   ScatterSample& s) {
    const SurfaceOptics& o = *it.optics;
    const bool radiance = mode == TransportMode::Radiance;
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
            if (o.roughness >= kSmoothRoughness && eta != 1) {
                LobeSample l;
                if (!sampleRoughDielectric(ggxAlpha(o.roughness), wo, eta, uc, u1, u2, radiance, l)) return false;
                fromLobe(frame, l, l.transmitted ? EventKind::GlossyT : EventKind::Glossy, s);
                return true;
            }
            double ni = fromFront ? it.nFront : it.nBack;
            double nt = fromFront ? it.nBack : it.nFront;
            double R = fresnelDielectric(wo.z, eta);
            double pR = fresnelBranchProbability(R, it.fresnelFloor);
            s.R = R;
            s.T = 1 - R;
            s.delta = true;
            if (uc < pR) {
                specularReflection(frame, wo, R / pR, s);
                s.R = R;
                s.event = R >= 1 ? EventKind::TIR : EventKind::Reflect;
                return true;
            }
            if (!refractDirection(d, it.n, ni, nt, s.wi)) return false;
            s.weight = (1 - R) / (1 - pR) * (radiance ? sqr(ni / nt) : 1.0);
            s.event = EventKind::Refract;
            s.transmitted = true;
            s.cosT = std::abs(dot(s.wi, it.n));
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
            double share = fromFront && radiance ? o.sampleAimShare : 0;
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
            s.event = EventKind::Diffuse;
            s.R = albedo;
            s.cosT = std::abs(dot(s.wi, it.n));
            return s.pdf > 0;
        }

        case SurfaceType::Mirror:
            if (!fromFront && o.backAbsorbs) return false;
            return specularReflection(frame, wo, o.reflectance.eval(it.lambda), s);

        case SurfaceType::Conductor: {
            if (!fromFront && o.backAbsorbs) return false;
            const Vec2 eta = conductorEta(o, it.lambda, fromFront ? it.nFront : it.nBack);
            const double tint = o.reflectance.eval(it.lambda);
            Vec3 w = fromFront ? wo : Vec3(wo.x, wo.y, -wo.z);  // work in the upper hemisphere
            if (o.roughness < kSmoothRoughness) return specularReflection(frame, wo, fresnelConductor(w.z, eta) * tint, s);
            LobeSample l;
            if (!sampleRoughConductor(ggxAlpha(o.roughness), w, eta, tint, u1, u2, l)) return false;
            if (!fromFront) l.wi.z = -l.wi.z;
            fromLobe(frame, l, EventKind::Glossy, s);
            return true;
        }
    }
    return false;
}

double evalScatter(const Interface& it, const Vec3& d, const Vec3& wiWorld, TransportMode mode, double& pdf) {
    pdf = 0;
    const SurfaceOptics& o = *it.optics;
    if (o.isDelta()) return 0;
    const bool radiance = mode == TransportMode::Radiance;
    Frame frame(it.n);
    Vec3 wo = frame.toLocal(-d);
    Vec3 wi = frame.toLocal(wiWorld);
    bool fromFront = wo.z > 0;
    switch (o.type) {
        case SurfaceType::Diffuse: {
            if (!fromFront && o.backAbsorbs) return 0;
            if (!sameHemisphere(wo, wi)) return 0;
            pdf = std::abs(wi.z) * InvPi;
            double share = fromFront && radiance ? o.sampleAimShare : 0;
            if (share > 0) pdf = (1 - share) * pdf + share * screenAimPdf(it, wiWorld);
            return surfaceAlbedo(it) * InvPi;
        }
        case SurfaceType::Conductor: {
            if (!fromFront && o.backAbsorbs) return 0;
            if (!sameHemisphere(wo, wi)) return 0;
            if (!fromFront) { wo.z = -wo.z; wi.z = -wi.z; }
            const Vec2 eta = conductorEta(o, it.lambda, fromFront ? it.nFront : it.nBack);
            return evalRoughConductor(ggxAlpha(o.roughness), wo, wi, eta, o.reflectance.eval(it.lambda), pdf);
        }
        case SurfaceType::Dielectric:
            return evalRoughDielectric(ggxAlpha(o.roughness), wo, wi, it.nBack / it.nFront, radiance, pdf);
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
