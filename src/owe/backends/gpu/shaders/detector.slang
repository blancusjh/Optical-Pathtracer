// Measurement rays of the detectors (camera-relative): float port of detector.cpp.
#pragma once
#include "owe/core/shared/optics.hpp"
#include "world.slang"

// A displacement (pixels) drawn from the pixel response's Gaussian (Detector::filterOffset).
float2 filterOffset(inout Rng rng) {
    const float sigma = asfloat(globals().target.z);
    if (!(sigma > 0)) return 0;
    float u2 = rng.uniform();  // Draw2 order: last argument first
    float u1 = rng.uniform();
    return sampleStandardNormal2(u1, u2) * sigma;
}

// Returns false for a zero-valued sample. weight is the importance-to-pdf ratio.
bool generateRay(float px, float py, inout Rng rng, out float3 o, out float3 d, out float weight) {
    GGlobals G = globals();
    const float W = float(G.limits.z), H = float(G.limits.w);
    o = 0;
    d = float3(0, 0, 1);
    weight = 0;
    if (G.counts.w == 0) {
        // Ideal observer at the origin.
        float sx = (2 * px / W - 1) * G.lens.x;
        float sy = (1 - 2 * py / H) * G.lens.y;
        float3 fwd = G.fwd.xyz, right = G.right.xyz, up = G.upv.xyz;
        float3 dirPinhole = normalize(right * sx + up * sy + fwd);
        float3 q = 0;
        float pupil = G.lens.z, focus = G.lens.w;
        weight = 1;
        if (pupil > 0) {
            // IdealObserver::generate: pupil samples guided to a telescope's exit pupil.
            const float gr = G.guide.w, share = asfloat(G.target.w);
            float3 c = 0;
            bool guided = false;
            if (gr > 0) {
                float dn = dot(dirPinhole, fwd);
                c = G.guide.xyz - dirPinhole * (dot(G.guide.xyz, fwd) / dn);
                guided = dn > 0 && dot(c, c) < (pupil + gr) * (pupil + gr);
            }
            const bool fromGuide = guided && rng.uniform() < share;
            float u2 = rng.uniform();
            float u1 = rng.uniform();
            float2 dk = sampleUniformDiskConcentric(u1, u2);
            q = fromGuide ? c + (right * dk.x + up * dk.y) * gr : (right * dk.x + up * dk.y) * pupil;
            if (guided) {
                if (fromGuide && dot(q, q) > pupil * pupil) return false;
                float pEye = 1 / (Pi * pupil * pupil);
                float3 e = q - c;
                bool inGuide = fromGuide || dot(e, e) <= gr * gr;
                float pGuide = inGuide ? 1 / (Pi * gr * gr) : 0;
                weight = pEye / ((1 - share) * pEye + share * pGuide);
            }
        }
        float3 dir = dirPinhole;
        if (pupil > 0 && focus > 0) {
            float3 F = (right * sx + up * sy + fwd) * focus;
            dir = normalize(F - q);
        }
        o = q;
        d = dir;
        return true;
    }
    // Surface sensor centred at the origin. A filtered sample displaced beyond it measures nothing.
    if (!(px >= 0 && py >= 0 && px < W && py < H)) return false;
    if (G.det.y != 0) px = W - px;
    if (G.det.z != 0) py = H - py;
    float3 pl = float3((2 * px / W - 1) * G.sensorSize.x, (1 - 2 * py / H) * G.sensorSize.y, 0);
    float3 p = float3(dot(G.fwd.xyz, pl), dot(G.right.xyz, pl), dot(G.upv.xyz, pl));  // rows of R
    float3 nrm = G.sensorNormal.xyz;
    float3 dir;
    if (G.det.w != 0) {
        float3 aimN = G.aimNormal.xyz;
        float3 a1, a2;
        orthonormalBasis(aimN, a1, a2);
        float share = G.aimCenter.w;
        float aimShare = G.sensorNormal.w;
        bool useAim = aimShare >= 1 || rng.uniform() < aimShare;
        bool useFocus = share > 0 && rng.uniform() < share;
        float3 c = useFocus ? G.focusCenter.xyz : G.aimCenter.xyz;
        float r = useFocus ? G.sensorSize.w : G.sensorSize.z;
        float u2 = rng.uniform();
        float u1 = rng.uniform();
        float2 dk = sampleUniformDiskConcentric(u1, u2);
        float3 w = c + (a1 * dk.x + a2 * dk.y) * r - p;
        dir = w / length(w);
        if (!useAim) dir = Frame(nrm).toWorld(sampleCosineHemisphere(u1, u2));
        float cosS = dot(dir, nrm);
        if (cosS <= 0) return false;
        // (The generating disc is known to contain its own sample: see diskAimPdf.)
        float pdfW = (1 - share) * diskAimPdf(p, dir, G.aimCenter.xyz, aimN, G.sensorSize.z, useAim && !useFocus);
        if (share > 0) pdfW += share * diskAimPdf(p, dir, G.focusCenter.xyz, aimN, G.sensorSize.w, useAim && useFocus);
        pdfW = aimShare * pdfW + (1 - aimShare) * cosS * InvPi;
        if (!(pdfW > 0)) return false;
        weight = cosS / pdfW;
    } else {
        Frame f = Frame(nrm);
        float u2 = rng.uniform();
        float u1 = rng.uniform();
        dir = f.toWorld(sampleCosineHemisphere(u1, u2));
        weight = Pi;
    }
    o = p;  // startPath retains the sensor boundary and processes contacts before any offset
    d = dir;
    return true;
}
