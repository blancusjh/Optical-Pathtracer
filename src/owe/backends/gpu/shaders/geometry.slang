// Shape intersection and sampling in shape-local coordinates, reading the shapes' records. The
// algebra is the reference's (owe/core/shared/surfaces.hpp): quadric roots are re-solved from the
// approximate hit point as on the CPU, which keeps hits precise when the ray starts far from the
// shape (a telescope looking at Saturn).
#pragma once
#include "owe/core/shared/surfaces.hpp"
#include "common.slang"
#include "scene.slang"

struct LocalHit {
    float t;
    float3 p;
    float3 n;
    uint prim;
    uint owner;  // mesh groups: the boundary the hit triangle belongs to (else kGpuNone)
    uint slot;   // meshes: the hit triangle's record in g_triangles (set by mesh intersection only)
};

// ---------------------------------------------------------------- sag surfaces

struct Sag {
    float c, k, rMax, rMin;
    uint data;  // asphere: zMin, zMax, nA at data + 1, coefficients from data + 2
    uint nA;

    float sag(float r) {
        float r2 = r * r;
        float z = conicSag(c, k, r2);
        float rp = r2 * r2;
        for (uint i = 0; i < nA; ++i) {
            z += g_shapeData[data + 2 + i / 4][i % 4] * rp;
            rp *= r2;
        }
        return z;
    }
    float dsag(float r) {
        float r2 = r * r;
        float d = conicSlope(c, k, r);
        float m = 4;
        float rp = r2 * r;
        for (uint i = 0; i < nA; ++i) {
            d += g_shapeData[data + 2 + i / 4][i % 4] * m * rp;
            rp *= r2;
            m += 2;
        }
        return d;
    }
    float3 normalAt(float3 p) {
        float r = sqrt(p.x * p.x + p.y * p.y);
        return sagNormal(p, r, dsag(r));
    }
    bool accept(float3 p) {
        float r2 = p.x * p.x + p.y * p.y;
        return r2 <= rMax * rMax && r2 >= rMin * rMin;
    }
};

Sag loadSag(uint data, bool asphere) {
    float4 a = g_shapeData[data];
    Sag s;
    s.c = a.x;
    s.k = a.y;
    s.rMax = a.z;
    s.rMin = a.w;
    s.data = data;
    s.nA = asphere ? uint(g_shapeData[data + 1].z) : 0;
    return s;
}

// The first of a quadric's roots t0 ≤ t1 that, refined, lies in (tmin, tmax), as the reference's
// firstRefinedHit; the shape's trim and normal are its caller's.
bool refinedRoot(Quadric q, float3 o, float3 d, float t0, float t1, int index, float tmin, float tmax, out float t) {
    return refinedRootIn(q, o, d, index == 0 ? t0 : t1, tmin, tmax, t);
}

bool intersectSagQuadric(Sag s, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float t0, t1;
    if (s.c == 0) {  // a plane: one exact root
        if (d.z == 0) return false;
        t0 = -o.z / d.z;
        float3 p = o + d * t0;
        if (t0 <= tmin || t0 >= tmax || !s.accept(p)) return false;
        h.t = t0;
        h.p = p;
        h.n = s.normalAt(p);
        h.prim = 0;
        return true;
    }
    Quadric q = makeQuadric(kQuadSag, s.c, s.c * (1 + s.k), 0);
    float A, B, C;
    quadricCoeffs(q, o, d, A, B, C);
    if (solveQuadratic(A, B, C, t0, t1) == 0) return false;
    [ForceUnroll] for (int i = 0; i < 2; ++i) {
        float t;
        if (!refinedRoot(q, o, d, t0, t1, i, tmin, tmax, t)) continue;
        float3 p = o + d * t;
        if (!s.accept(p) || !onConicSheet(s.c, s.k, p.z)) continue;
        h.t = t;
        h.p = p;
        h.n = s.normalAt(p);
        h.prim = 0;
        return true;
    }
    return false;
}

bool intersectSagNumeric(Sag s, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float zMin = g_shapeData[s.data + 1].x, zMax = g_shapeData[s.data + 1].y;
    float ta = tmin, tb = tmax;
    if (!clipCylinder(o, d, s.rMax, ta, tb) || !clipSlab(o.z, d.z, zMin, zMax, ta, tb)) return false;
    const int N = 96;
    float tPrev = ta;
    float3 pp = o + d * ta;
    float fPrev = s.sag(sqrt(pp.x * pp.x + pp.y * pp.y)) - pp.z;
    for (int i = 1; i <= N; ++i) {
        float tCur = ta + (tb - ta) * i / N;
        float3 pc = o + d * tCur;
        float fCur = s.sag(sqrt(pc.x * pc.x + pc.y * pc.y)) - pc.z;
        if ((fPrev <= 0 && fCur >= 0) || (fPrev >= 0 && fCur <= 0)) {
            if (fPrev == 0 && tPrev <= tmin) { tPrev = tCur; fPrev = fCur; continue; }
            float lo = tPrev, hi = tCur, flo = fPrev;
            for (int it = 0; it < kBisectSteps && hi - lo > kBisect * (1 + abs(hi)); ++it) {
                float mid = 0.5 * (lo + hi);
                float3 pm = o + d * mid;
                float fm = s.sag(sqrt(pm.x * pm.x + pm.y * pm.y)) - pm.z;
                if ((fm <= 0) == (flo <= 0)) { lo = mid; flo = fm; } else { hi = mid; }
            }
            float t = 0.5 * (lo + hi);
            float3 p = o + d * t;
            if (t > tmin && t < tmax && s.accept(p)) {
                h.t = t;
                h.p = p;
                h.n = s.normalAt(p);
                h.prim = 0;
                return true;
            }
        }
        tPrev = tCur;
        fPrev = fCur;
    }
    return false;
}

// ---------------------------------------------------------------- liquid surfaces

// WaveSurface (geometry.cpp): z = w(x) w(y) Σ aᵢ sin(kᵢ·(x, y) + φᵢ) over |x| ≤ hx, |y| ≤ hy, either
// summed here or (many waves) as its bicubic Hermite spline. Data: (hx, hy, margin, waves),
// (amplitude bound, slope bound, dx, dy), (nx, ny, sampled, -), then the waves (a, kx, ky, φ) or the
// nodes (h, ∂h/∂x·dx, ∂h/∂y·dy, ∂²h/∂x∂y·dx·dy).
// Height and gradient (dh/dx, dh/dy) of the surface at (x, y).
float3 waveHeight(uint data, float x, float y) {
    float4 hdr = g_shapeData[data];
    if (abs(x) > hdr.x || abs(y) > hdr.y) return 0;
    float4 grid = g_shapeData[data + 2];
    if (grid.z != 0) {
        float4 sp = g_shapeData[data + 1];
        uint nx = uint(grid.x), ny = uint(grid.y);
        float fx = (x + hdr.x) / sp.z, fy = (y + hdr.y) / sp.w;
        uint i = min(uint(max(fx, 0.0)), nx - 2), j = min(uint(max(fy, 0.0)), ny - 2);
        float2 P, Q, dP, dQ, Pv, Qv, dPv, dQv;
        hermiteBases(fx - float(i), P, Q, dP, dQ);
        hermiteBases(fy - float(j), Pv, Qv, dPv, dQv);
        float f = 0, fu = 0, fv = 0;
        [ForceUnroll] for (int b = 0; b < 2; ++b)
            [ForceUnroll] for (int a = 0; a < 2; ++a) {
                float4 n = g_shapeData[data + 3 + (j + uint(b)) * nx + i + uint(a)];
                f += n.x * P[a] * Pv[b] + n.y * Q[a] * Pv[b] + n.z * P[a] * Qv[b] + n.w * Q[a] * Qv[b];
                fu += n.x * dP[a] * Pv[b] + n.y * dQ[a] * Pv[b] + n.z * dP[a] * Qv[b] + n.w * dQ[a] * Qv[b];
                fv += n.x * P[a] * dPv[b] + n.y * Q[a] * dPv[b] + n.z * P[a] * dQv[b] + n.w * Q[a] * dQv[b];
            }
        return float3(f, fu / sp.z, fv / sp.w);
    }
    float g = 0, gx = 0, gy = 0;
    uint n = uint(hdr.w);
    for (uint k = 0; k < n; ++k) {
        float4 w = g_shapeData[data + 3 + k];
        float arg = w.y * x + w.z * y + w.w;
        float s = sin(arg), c = cos(arg);
        g += w.x * s;
        gx += w.x * c * w.y;
        gy += w.x * c * w.z;
    }
    float wx, dwx, wy, dwy;
    waveTaper(x, hdr.x, hdr.z, wx, dwx);
    waveTaper(y, hdr.y, hdr.z, wy, dwy);
    return float3(wx * wy * g, dwx * wy * g + wx * wy * gx, wx * dwy * g + wx * wy * gy);
}

// The same Lipschitz-safe march and Newton refinement as the reference, in float.
bool intersectWaves(uint data, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float4 hdr = g_shapeData[data];
    float amp = g_shapeData[data + 1].x, slope = g_shapeData[data + 1].y;
    float ta = tmin, tb = tmax;
    if (!clipSlab(o.x, d.x, -hdr.x, hdr.x, ta, tb) || !clipSlab(o.y, d.y, -hdr.y, hdr.y, ta, tb) ||
        !clipSlab(o.z, d.z, -amp, amp, ta, tb))
        return false;
    const float L = abs(d.z) + slope * length(d.xy);
    const float scale = 1 + max(max(hdr.x, hdr.y), amp);
    const float tol = 2e-7 * scale, nearSurface = 1e-4 * (amp + tol), minStep = 1e-5 * (tb - ta);
    float t = ta;
    float3 hv = waveHeight(data, o.x + d.x * t, o.y + d.y * t);
    float ft = o.z + d.z * t - hv.x;
    for (int i = 0; i < 1024 && t < tb; ++i) {
        float step = max(abs(ft) / L, minStep);
        if (abs(ft) < nearSurface) {
            float tn = t;
            for (int k = 0; k < 6; ++k) {
                float3 hn = waveHeight(data, o.x + d.x * tn, o.y + d.y * tn);
                float fn = o.z + d.z * tn - hn.x;
                float dfdt = d.z - hn.y * d.x - hn.z * d.y;
                if (abs(fn) < tol || abs(dfdt) < 1e-9) break;
                tn -= fn / dfdt;
            }
            float3 hn = waveHeight(data, o.x + d.x * tn, o.y + d.y * tn);
            if (tn >= t - 10 * tol && abs(o.z + d.z * tn - hn.x) < 20 * tol) {
                float3 p = o + d * tn;
                if (!(tn > tmin && tn < tmax) || abs(p.x) > hdr.x || abs(p.y) > hdr.y) return false;
                h.t = tn;
                h.p = p;
                h.n = normalize(float3(-hn.y, -hn.z, 1));
                h.prim = 0;
                return true;
            }
            step = max(minStep, 2 * abs(ft) / L);
        }
        float tn = min(t + step, tb);
        float3 hn = waveHeight(data, o.x + d.x * tn, o.y + d.y * tn);
        float fn = o.z + d.z * tn - hn.x;
        if ((fn < 0) != (ft < 0)) {
            float a = t, b = tn, fa = ft;
            for (int k = 0; k < 40 && b - a > tol; ++k) {
                float m = 0.5 * (a + b);
                float3 hm = waveHeight(data, o.x + d.x * m, o.y + d.y * m);
                float fm = o.z + d.z * m - hm.x;
                if ((fm < 0) == (fa < 0)) { a = m; fa = fm; } else { b = m; }
            }
            float tr = 0.5 * (a + b);
            float3 p = o + d * tr;
            if (!(tr > tmin && tr < tmax) || abs(p.x) > hdr.x || abs(p.y) > hdr.y) return false;
            float3 hr = waveHeight(data, p.x, p.y);
            h.t = tr;
            h.p = p;
            h.n = normalize(float3(-hr.y, -hr.z, 1));
            h.prim = 0;
            return true;
        }
        t = tn;
        ft = fn;
    }
    return false;
}

// ---------------------------------------------------------------- lathe surfaces

// ConeShape (geometry.cpp): x² + y² = (r0 + k (z − z0))², z0 ≤ z ≤ z1. Data: (r0, z0, r1, z1).
bool intersectCone(uint data, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float4 c = g_shapeData[data];
    const float k = (c.z - c.x) / (c.w - c.y);
    Quadric q = makeQuadric(kQuadCone, c.x, c.y, k);
    float A, B, C, t0, t1;
    quadricCoeffs(q, o, d, A, B, C);
    if (solveQuadratic(A, B, C, t0, t1) == 0) return false;
    [ForceUnroll] for (int i = 0; i < 2; ++i) {
        float t;
        if (!refinedRoot(q, o, d, t0, t1, i, tmin, tmax, t)) continue;
        float3 p = o + d * t;
        if (p.z < c.y || p.z > c.w) continue;
        float rho = c.x + k * (p.z - c.y);
        if (rho < 0) continue;
        h.t = t;
        h.p = p;
        h.n = coneNormal(p, rho, k);
        h.prim = 0;
        return true;
    }
    return false;
}

// TorusPatchShape: (√(x²+y²) − R)² + (z − zc)² = rho² for arc angles a0 ≤ a ≤ a1.
// Data: (R, zc, rho, -), (a0, a1, rMax, -), (zMin, zMax, -, -).

bool intersectTorus(uint data, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float4 c = g_shapeData[data], a = g_shapeData[data + 1], zr = g_shapeData[data + 2];
    float ta = tmin, tb = tmax;
    const float pad = 1e-6 * (1 + a.z + abs(zr.y));
    if (!clipSlab(o.z, d.z, zr.x - pad, zr.y + pad, ta, tb)) return false;
    // Continue from the slab's entry: a ray from far away (a sun photon starts metres off) keeps
    // its precision there, where the cylinder test would otherwise lose it to cancellation.
    const float t0 = ta;
    const float3 o2 = o + d * t0;
    float ca = ta - t0, cb = tb - t0;  // the bracket about o2
    if (!clipCylinder(o2, d, a.z + pad, ca, cb)) return false;
    ta = t0 + ca;
    tb = t0 + cb;
    const int N = 64;
    float tPrev = ta, fPrev = torusF(c.x, c.y, c.z, o2 + d * (ta - t0));
    for (int i = 1; i <= N; ++i) {
        float tCur = ta + (tb - ta) * float(i) / N;
        float fCur = torusF(c.x, c.y, c.z, o2 + d * (tCur - t0));
        if ((fPrev <= 0) != (fCur <= 0) || fCur == 0) {
            float lo = tPrev - t0, hi = tCur - t0, flo = fPrev;
            for (int it = 0; it < kBisectSteps && hi - lo > kBisect * (1 + abs(hi)); ++it) {
                float mid = 0.5 * (lo + hi);
                float fm = torusF(c.x, c.y, c.z, o2 + d * mid);
                if ((fm <= 0) == (flo <= 0)) { lo = mid; flo = fm; } else { hi = mid; }
            }
            float3 p = o2 + d * (0.5 * (lo + hi));
            float t = t0 + 0.5 * (lo + hi);
            if (t > tmin && t < tmax && torusOnArc(c.x, c.y, a.x, a.y, p)) {
                h.t = t;
                h.p = p;
                h.n = torusNormal(c.x, c.y, p);
                h.prim = 0;
                return true;
            }
        }
        tPrev = tCur;
        fPrev = fCur;
    }
    return false;
}

// ---------------------------------------------------------------- planes, spheres, cylinders

bool intersectPlane(uint data, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    if (d.z == 0) return false;
    float t = -o.z / d.z;
    if (t <= tmin || t >= tmax) return false;
    float3 p = o + d * t;
    p.z = 0;
    float4 a = g_shapeData[data];
    float2 hole = uint(a.x) == kApDiskHole ? g_shapeData[data + 1].xy : float2(0, 0);
    if (!inPlaneAperture(uint(a.x), a.y, a.z, hole.x, hole.y, p)) return false;
    h.t = t;
    h.p = p;
    h.n = float3(0, 0, 1);
    h.prim = 0;
    return true;
}

bool roundWallSolid(uint data, float3 p) {
    float4 a = g_shapeData[data];  // R, H, n
    if (p.z < 0 || p.z > a.y) return false;
    uint n = uint(a.z);
    for (uint i = 0; i < n; ++i) {
        float4 op = g_shapeData[data + 1 + i];  // azimuth, width, sill, top
        if (inWallOpening(a.x, op.x, op.y, op.z, op.w, p)) return false;
    }
    return true;
}

// Spheres, open cylinders, round walls with openings and domes: refined quadric roots, then the trim.
bool intersectRevolved(uint kind, uint data, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float4 a = g_shapeData[data];
    const bool spherical = kind == kShapeSphere || kind == kShapeDome;
    float t0, t1;
    if (!(spherical ? sphereRoots(o, d, a.x, t0, t1) : cylinderRoots(o, d, a.x, t0, t1))) return false;
    Quadric q = makeQuadric(spherical ? kQuadSphere : kQuadCylinder, a.x * a.x, 0, 0);
    [ForceUnroll] for (int i = 0; i < 2; ++i) {
        float t;
        if (!refinedRoot(q, o, d, t0, t1, i, tmin, tmax, t)) continue;
        float3 p = o + d * t;
        bool solid = true;
        if (kind == kShapeCylinder) solid = !(p.z < a.y || p.z > a.z);
        else if (kind == kShapeRoundWall) solid = roundWallSolid(data, p);
        else if (kind == kShapeDome) solid = domeSolid(a.x, a.y, a.z, a.w, p);  // R, azimuth, width, top
        if (!solid) continue;
        h.t = t;
        h.p = p;
        h.n = spherical ? normalize(p / a.x) : normalize(float3(p.x, p.y, 0));
        h.prim = 0;
        return true;
    }
    return false;
}
bool intersectSphere(float R, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float t0, t1;
    if (!sphereRoots(o, d, R, t0, t1)) return false;
    Quadric q = makeQuadric(kQuadSphere, R * R, 0, 0);
    [ForceUnroll] for (int i = 0; i < 2; ++i) {
        float t;
        if (!refinedRoot(q, o, d, t0, t1, i, tmin, tmax, t)) continue;
        h.t = t;
        h.p = o + d * t;
        h.n = normalize(h.p / R);
        h.prim = 0;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- BVH traversal and meshes

struct RaySlabs {
    float3 o, inv;
    __init(float3 origin, float3 dir) {
        o = origin;
        inv = 1.0 / dir;
    }
    // Slab test; a NaN (origin on a slab plane, zero direction component) never narrows the
    // interval. The far bound is widened by a few ulps so float traversal stays conservative.
    bool hit(float3 lo, float3 hi, float tmin, float tmax, out float tEnter) {
        float3 t0 = (lo - o) * inv, t1 = (hi - o) * inv;
        float3 tn = min(t0, t1), tf = max(t0, t1);
        [ForceUnroll] for (int a = 0; a < 3; ++a) {
            tmin = tn[a] > tmin ? tn[a] : tmin;
            // Contact neighborhoods include negative t. Widen toward +infinity
            // on either side of zero; multiplying a negative far bound by >1
            // would shrink the interval and reject a touching slab.
            float f = tf[a] * (tf[a] >= 0 ? 1.0000004 : 0.9999996);
            tmax = f < tmax ? f : tmax;
        }
        tEnter = tmin;
        return tmin <= tmax;
    }
};

static const int kStack = 64;

// Möller–Trumbore's inside test, with the distance taken to the triangle's plane: Möller–Trumbore's
// own dot(e2, qv) / det loses precision in proportion to 1 / area, which on sliver triangles
// placed float hit points microns off the surface, beyond the ray offset, so the next ray began
// inside the body. (The double-precision reference does not need this.)
bool hitTriangle(GTriangle tri, float3 o, float3 d, float tmin, float tmax, out float t) {
    t = 0;
    float3 v0 = tri.v0.xyz;
    if (!lineCrossesTriangle(o, d, v0, tri.v1.xyz, tri.v2.xyz)) return false;
    float3 np = triangleNormal(v0, tri.v1.xyz, tri.v2.xyz);
    t = -dot(o - v0, np) / dot(d, np);
    return t > tmin && t < tmax;
}

bool intersectMesh(GBoundary b, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    RaySlabs rs = RaySlabs(o, d);
    uint stackNode[kStack];
    float stackT[kStack];
    int sp = 0;
    uint cur = b.meshNodes;
    float best = tmax;
    bool found = false;
    uint bestSlot = 0;
    float tEnter;
    GNode root = g_nodes[cur];
    if (!rs.hit(root.lo.xyz, root.hi.xyz, tmin, best, tEnter)) return false;
    while (true) {
        GNode node = g_nodes[cur];
        uint na = asuint(node.lo.w), nb = asuint(node.hi.w);
        bool descend = false;
        if ((nb & kNodeInner) == 0) {
            for (uint i = 0; i < nb; ++i) {
                float tt;
                if (!hitTriangle(g_triangles[na + i], o, d, tmin, best, tt)) continue;
                best = tt;
                found = true;
                bestSlot = na + i;
            }
        } else {
            uint first = cur + 1, second = na;
            if (rs.inv[nb & 3u] < 0) { uint tmp = first; first = second; second = tmp; }
            GNode n0 = g_nodes[first], n1 = g_nodes[second];
            float t0, t1;
            bool h0 = rs.hit(n0.lo.xyz, n0.hi.xyz, tmin, best, t0);
            bool h1 = rs.hit(n1.lo.xyz, n1.hi.xyz, tmin, best, t1);
            if (h0 && h1) {
                if (t1 < t0) { uint tmp = first; first = second; second = tmp; float tt = t0; t0 = t1; t1 = tt; }
                if (sp < kStack) { stackNode[sp] = second; stackT[sp] = t1; ++sp; }
                cur = first;
                descend = true;
            } else if (h0 || h1) {
                cur = h0 ? first : second;
                descend = true;
            }
        }
        if (descend) continue;
        while (sp > 0 && stackT[sp - 1] > best) --sp;
        if (sp == 0) break;
        --sp;
        cur = stackNode[sp];
    }
    if (!found) return false;
    GTriangle tri = g_triangles[bestSlot];
    h.t = best;
    h.p = o + d * best;
    h.n = normalize(triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz));
    h.prim = tri.primitive;
    h.owner = tri.owner;
    h.slot = bestSlot;
    return true;
}

// Unit vectors in octahedral coordinates, 2 × 16-bit snorm (x low); as gpu_scene.cpp's octEncode.
uint octEncode(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 e = n.xy;
    if (n.z < 0) e = (1 - abs(n.yx)) * float2(n.x >= 0 ? 1.0 : -1.0, n.y >= 0 ? 1.0 : -1.0);
    int2 q = int2(round(clamp(e, -1.0, 1.0) * 32767.0));
    return (uint(q.x) & 0xFFFFu) | (uint(q.y) << 16);
}
float3 octDecode(uint w) {
    float2 e = float2(float(int(w << 16) >> 16), float(int(w) >> 16)) / 32767.0;
    float3 n = float3(e.x, e.y, 1 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0 ? -t : t;
    n.y += n.y >= 0 ? -t : t;
    return normalize(n);
}

// Local point p's barycentric coordinates in a triangle (false for a degenerate one).
bool triangleBarycentrics(GTriangle tri, float3 p, out float3 b) {
    b = float3(1, 0, 0);
    float3 e1 = tri.v1.xyz - tri.v0.xyz, e2 = tri.v2.xyz - tri.v0.xyz, q = p - tri.v0.xyz;
    float d00 = dot(e1, e1), d01 = dot(e1, e2), d11 = dot(e2, e2), d20 = dot(q, e1), d21 = dot(q, e2);
    float den = d00 * d11 - d01 * d01;
    if (!(den > 0)) return false;
    float b1 = saturate((d11 * d20 - d01 * d21) / den), b2 = saturate((d00 * d21 - d01 * d20) / den);
    b = float3(max(0.0, 1 - b1 - b2), b1, b2);
    return true;
}

// The shading normal of a smooth triangle at local point p (its corner normals interpolated with
// p's barycentric coordinates), on the side of the geometric normal n; n itself for a flat one.
float3 meshShadingNormal(uint slot, float3 p, float3 n) {
    GTriangle tri = g_triangles[slot];
    uint cw = tri.corners;
    if (cw == kGpuNone || (cw & kCornerNormals) == 0) return n;
    float3 b;
    if (!triangleBarycentrics(tri, p, b)) return n;
    uint w = cw & kCornerIndex;
    float3 ns = octDecode(g_normals[w]) * b.x + octDecode(g_normals[w + 1]) * b.y + octDecode(g_normals[w + 2]) * b.z;
    float l = length(ns);
    ns = l > 0 ? ns / l : n;
    return dot(ns, n) > 1e-4 ? ns : n;
}

// The texture coordinates of a textured triangle at local point p (false: it has none).
bool meshUv(uint slot, float3 p, out float2 uv) {
    uv = 0;
    GTriangle tri = g_triangles[slot];
    uint cw = tri.corners;
    if (cw == kGpuNone || (cw & kCornerUvs) == 0) return false;
    float3 b;
    triangleBarycentrics(tri, p, b);
    uint w = (cw & kCornerIndex) + ((cw & kCornerNormals) != 0 ? 3 : 0);
    uv = float2(asfloat(g_normals[w]), asfloat(g_normals[w + 1])) * b.x +
         float2(asfloat(g_normals[w + 2]), asfloat(g_normals[w + 3])) * b.y +
         float2(asfloat(g_normals[w + 4]), asfloat(g_normals[w + 5])) * b.z;
    return true;
}

// A textured triangle's derivatives of position along its texture coordinates, in its local frame
// (false: no texture coordinates, or degenerate ones).
bool meshUvFrame(uint slot, out float3 dpdu, out float3 dpdv) {
    dpdu = 0;
    dpdv = 0;
    GTriangle tri = g_triangles[slot];
    uint cw = tri.corners;
    if (cw == kGpuNone || (cw & kCornerUvs) == 0) return false;
    uint w = (cw & kCornerIndex) + ((cw & kCornerNormals) != 0 ? 3 : 0);
    float2 t0 = float2(asfloat(g_normals[w]), asfloat(g_normals[w + 1]));
    float2 t1 = float2(asfloat(g_normals[w + 2]), asfloat(g_normals[w + 3]));
    float2 t2 = float2(asfloat(g_normals[w + 4]), asfloat(g_normals[w + 5]));
    float2 du = t1 - t0, dv = t2 - t0;
    float3 e1 = tri.v1.xyz - tri.v0.xyz, e2 = tri.v2.xyz - tri.v0.xyz;
    float det = du.x * dv.y - du.y * dv.x;
    if (abs(det) < 1e-20) return false;
    dpdu = (e1 * dv.y - e2 * du.y) / det;
    dpdv = (e2 * du.x - e1 * dv.x) / det;
    return true;
}

// Boundaries a shadow ray crosses (null) or is attenuated by (stained glass) rather than stopped.
bool isPassableType(uint type) { return type == kSurfNull || type == kSurfStainedGlass; }
bool isNullBoundary(uint boundary) { return isPassableType(g_optics[g_boundaries[boundary].optics].a.x); }

// Any hit on matter (a triangle whose boundary is not null) in (tmin, tmax); no ordering, first
// found wins. sawNull reports crossing a null boundary's triangle.
bool occludedMesh(GBoundary b, float3 o, float3 d, float tmin, float tmax, inout bool sawNull) {
    RaySlabs rs = RaySlabs(o, d);
    uint stackNode[kStack];
    int sp = 0;
    uint cur = b.meshNodes;
    float tEnter;
    GNode root = g_nodes[cur];
    if (!rs.hit(root.lo.xyz, root.hi.xyz, tmin, tmax, tEnter)) return false;
    while (true) {
        GNode node = g_nodes[cur];
        uint na = asuint(node.lo.w), nb = asuint(node.hi.w);
        if ((nb & kNodeInner) == 0) {
            for (uint i = 0; i < nb; ++i) {
                GTriangle tri = g_triangles[na + i];
                float tt;
                if (!hitTriangle(tri, o, d, tmin, tmax, tt)) continue;
                if (!isNullBoundary(tri.owner)) return true;
                sawNull = true;
            }
        } else {
            GNode n0 = g_nodes[cur + 1], n1 = g_nodes[na];
            float t0, t1;
            bool h0 = rs.hit(n0.lo.xyz, n0.hi.xyz, tmin, tmax, t0);
            bool h1 = rs.hit(n1.lo.xyz, n1.hi.xyz, tmin, tmax, t1);
            if (h0 || h1) {
                // Nearer child first: occluders near the origin end the search soonest.
                uint first = cur + 1, second = na;
                if (h0 && h1 && t1 < t0) { first = na; second = cur + 1; }
                else if (!h0) { first = na; }
                if (h0 && h1 && sp < kStack) stackNode[sp++] = second;
                cur = first;
                continue;
            }
        }
        if (sp == 0) break;
        cur = stackNode[--sp];
    }
    return false;
}

bool intersectShape(GBoundary b, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    uint kind = b.shape, data = b.data;
    switch (kind) {
        case kShapeSag: return intersectSagQuadric(loadSag(data, false), o, d, tmin, tmax, h);
        case kShapeAsphere: return intersectSagNumeric(loadSag(data, true), o, d, tmin, tmax, h);
        case kShapePlane: return intersectPlane(data, o, d, tmin, tmax, h);
        case kShapeSphere:
        case kShapeCylinder:
        case kShapeRoundWall:
        case kShapeDome: return intersectRevolved(kind, data, o, d, tmin, tmax, h);
        case kShapeWaves: return intersectWaves(data, o, d, tmin, tmax, h);
        case kShapeCone: return intersectCone(data, o, d, tmin, tmax, h);
        case kShapeTorus: return intersectTorus(data, o, d, tmin, tmax, h);
#ifndef OWE_RAY_QUERY
        // With hardware traversal a mesh is triangle geometry of the acceleration structure and
        // never reaches here; its software traversal inlined at every call site would only spill
        // (~4 KB of local memory per thread).
        case kShapeMesh: return intersectMesh(b, o, d, tmin, tmax, h);
#endif
        default: return false;
    }
}

bool hasSphereFrame(GBoundary b) {
    return b.shape == kShapeSphere && g_shapeData[b.data].y != 0;
}

bool usesSphereFrame(GBoundary b, float3 o) {
    if (!hasSphereFrame(b)) return false;
    uint data = b.data;
    float3 lo;
    [ForceUnroll] for (int k = 0; k < 3; ++k) {
        float4 row = g_shapeData[data + 1 + k];
        lo[k] = dot(row.xyz, o) + row.w;
    }
    // Far-origin light paths keep the existing closest-approach sphere roots. Their normalized
    // quadratic discriminant would subtract large nearly equal terms in the pole frame.
    return maxAbsComponent(lo) < g_shapeData[data].x / 8;
}

// Same complete analytic sphere, expressed about the pole nearest the detector. Dividing its
// equation by 2R keeps both the planetary radius squared and subtractive cancellation out of
// the coefficients: |p|²/(2R) + p.z = 0. Both roots remain available (including the far hemisphere).
[noinline]
bool intersectBoundary(GBoundary b, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    if (!usesSphereFrame(b, o)) return intersectShape(b, xfPoint(b.toLocal, o), xfVector(b.toLocal, d), tmin, tmax, h);
    uint data = b.data;
    float R = g_shapeData[data].x;
    float4 toPole[3], fromPole[3];
    [ForceUnroll] for (int k = 0; k < 3; ++k) {
        toPole[k] = g_shapeData[data + 1 + k];
        fromPole[k] = g_shapeData[data + 4 + k];
    }
    float3 lo = xfPoint(toPole, o), ld = xfVector(toPole, d);
    float A = dot(ld, ld) / (2 * R);
    float B = ld.z + dot(lo, ld) / R;
    float C = lo.z + dot(lo, lo) / (2 * R);
    float disc = B * B - 4 * A * C;
    if (disc < 0) return false;
    float q = -0.5 * (B + copysign(sqrt(disc), B));
    float t0 = q / A, t1 = q != 0 ? C / q : t0;
    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
    float t = 0;
    bool found = false;
    [ForceUnroll] for (int root = 0; root < 2 && !found; ++root) {
        float tr = root == 0 ? t0 : t1;
        if (!mayLieIn(tr, tmin, tmax)) continue;
        [ForceUnroll] for (int step = 0; step < 2; ++step) {
            float3 p = lo + ld * tr;
            float derivative = ld.z + dot(p, ld) / R;
            if (derivative == 0) break;
            float correction = -(p.z + dot(p, p) / (2 * R)) / derivative;
            if (abs(correction) < kRefine * (1 + abs(tr))) tr += correction;
        }
        if (tr > tmin && tr < tmax) { t = tr; found = true; }
    }
    if (!found) return false;
    float3 p = lo + ld * t;
    float3 n = normalize(float3(p.xy / R, 1 + p.z / R));
    h.t = t;
    // Native coordinates remain authoritative for textures and emitter PDFs. World reconstruction
    // projects the hit on the pole-anchored sphere, preserving its small surface displacement.
    h.p = xfPoint(b.toLocal, o + d * t);
    h.n = normalize(xfVector(b.toLocal, xfVector(fromPole, n)));
    h.prim = 0;
    return true;
}

// ---------------------------------------------------------------- emitter sampling (local frame)

// Uniform area sampling; false when the shape has no area sampler.
bool sampleArea(GBoundary b, float u1, float u2, float u3, out float3 p, out float3 n, out uint primitive) {
    primitive = 0;
    uint kind = b.shape, data = b.data;
    float4 a = g_shapeData[data];
    p = 0;
    n = float3(0, 0, 1);
    switch (kind) {
        case kShapeSag:  // flat only (sampleable flag)
            p = sampleAnnulus(a.w, a.z, u1, u2);
            return true;
        case kShapePlane: {
            uint ap = uint(a.x);
            if (ap == kApDisk) p = sampleAnnulus(a.z, a.y, u1, u2);
            else if (ap == kApRect) p = sampleRect(a.y, a.z, u1, u2);
            else if (ap == kApEllipse) p = sampleEllipse(a.y, a.z, u1, u2);
            else return false;
            return true;
        }
        case kShapeSphere:
            n = sampleUniformSphere(u1, u2);
            p = n * a.x;
            return true;
        case kShapeCylinder: {
            float phi = 2 * Pi * u1;
            n = float3(cos(phi), sin(phi), 0);
            p = float3(a.x * n.x, a.x * n.y, a.y + u2 * (a.z - a.y));
            return true;
        }
        case kShapeMesh: {
            uint cdf = asuint(a.x), count = b.meshCount;
            uint lo = 0, len = count;  // first CDF entry ≥ u3 (std::lower_bound)
            while (len > 0) {
                uint half = len / 2;
                if (g_scalars[cdf + lo + half] < u3) { lo += half + 1; len -= half + 1; }
                else len = half;
            }
            uint i = min(lo, count - 1);
            GTriangle tri = g_triangles[b.meshTris + i];
            primitive = b.meshTris + i;  // transient departure identity is a global triangle slot
            p = sampleTriangle(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz, u1, u2);
            n = normalize(triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz));
            return true;
        }
        default: return false;
    }
}

bool sampleArea(GBoundary b, float u1, float u2, float u3, out float3 p, out float3 n) {
    uint primitive;
    return sampleArea(b, u1, u2, u3, p, n, primitive);
}

// Point on the shape as seen from `ref` with a solid-angle pdf (false on failure). Local frame. A
// sphere seen from outside is sampled within the cone it subtends.
bool sampleFrom(GBoundary b, float3 ref, float u1, float u2, float u3, out float3 p, out float3 n, out float pdfW,
                out uint primitive) {
    primitive = 0;
    pdfW = 0;
    p = 0;
    n = float3(0, 0, 1);
    if (b.shape == kShapeSphere) {
        float R = g_shapeData[b.data].x;
        float oneMinusCos = sphereConeSize(ref, R);
        if (oneMinusCos > 0) {
            float3 w = sampleSphereCone(ref, oneMinusCos, u1, u2);
            LocalHit h;
            if (intersectSphere(R, ref, w, 0, infinity(), h)) p = h.p;
            else p = normalize(ref + w * dot(-ref, w)) * R;  // grazing: the tangent point
            n = normalize(p);
            pdfW = 1 / (2 * Pi * oneMinusCos);
            return true;
        }
    }
    if (b.sampleable == 0 || b.area <= 0) return false;
    return sampleArea(b, u1, u2, u3, p, n, primitive) && areaToSolidAngle(ref, p, n, b.area, pdfW);
}

bool sampleFrom(GBoundary b, float3 ref, float u1, float u2, float u3, out float3 p, out float3 n, out float pdfW) {
    uint primitive;
    return sampleFrom(b, ref, u1, u2, u3, p, n, pdfW, primitive);
}

float pdfFrom(GBoundary b, float3 ref, float3 p, float3 n) {
    if (b.shape == kShapeSphere) {
        float oneMinusCos = sphereConeSize(ref, g_shapeData[b.data].x);
        if (oneMinusCos > 0) return 1 / (2 * Pi * oneMinusCos);
    }
    float pdfW;
    return b.sampleable != 0 && b.area > 0 && areaToSolidAngle(ref, p, n, b.area, pdfW) ? pdfW : 0.0;
}
