// Measurement rays of the detectors (camera-relative): float port of detector.cpp.
#pragma once
#include "world.slang"

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
        if (pupil > 0) {
            float u2 = rng.uniform();
            float u1 = rng.uniform();
            float2 dk = sampleUniformDiskConcentric(u1, u2);
            q = (right * dk.x + up * dk.y) * pupil;
        }
        float3 dir = dirPinhole;
        if (pupil > 0 && focus > 0) {
            float3 F = (right * sx + up * sy + fwd) * focus;
            dir = normalize(F - q);
        }
        o = q;
        d = dir;
        weight = 1;
        return true;
    }
    // Surface sensor centred at the origin.
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
        float pdfW = 0;
        [ForceUnroll] for (int k = 0; k < 2; ++k) {
            float3 center = k == 0 ? G.aimCenter.xyz : G.focusCenter.xyz;
            float radius = k == 0 ? G.sensorSize.z : G.sensorSize.w;
            float mix = k == 0 ? 1 - share : share;
            if (mix <= 0) continue;
            float dn = dot(dir, aimN);
            if (abs(dn) <= 1e-12) continue;
            float t = dot(center - p, aimN) / dn;
            float3 hitP = p + dir * t - center;
            // The generating component is known to contain its own sample. Rebuilding
            // its disk point loses precision for millimetre pupils metres away; a false
            // rejection would leave only the cosine PDF and create enormous weights.
            bool generatingDisk = useAim && ((k == 1) == useFocus);
            if (t <= 0 || (!generatingDisk && dot(hitP, hitP) > radius * radius)) continue;
            pdfW += mix * t * t / (abs(dn) * kPi * radius * radius);
        }
        pdfW = aimShare * pdfW + (1 - aimShare) * cosS * kInvPi;
        if (!(pdfW > 0)) return false;
        weight = cosS / pdfW;
    } else {
        Frame f = Frame(nrm);
        float u2 = rng.uniform();
        float u1 = rng.uniform();
        dir = f.toWorld(sampleCosineHemisphere(u1, u2));
        weight = kPi;
    }
    o = offsetOrigin(p, nrm, dir, max(G.sensorSize.x, G.sensorSize.y));
    d = dir;
    return true;
}
