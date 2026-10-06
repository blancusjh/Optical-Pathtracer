// Shape intersection and sampling in shape-local coordinates: float port of geometry.cpp.
// Quadric roots are re-solved from the approximate hit point exactly as on the CPU, which keeps
// hits precise when the ray starts far from the shape (a telescope looking at Saturn).
#pragma once
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

int solveQuadratic(float a, float b, float c, out float t0, out float t1) {
    t0 = 0;
    t1 = 0;
    if (abs(a) < 1e-35 || abs(a) < 1e-14 * abs(b)) {
        if (b == 0) return 0;
        t0 = t1 = -c / b;
        return 1;
    }
    float disc = b * b - 4 * a * c;
    if (disc < 0) return 0;
    float s = sqrt(disc);
    float q = -0.5 * (b + copySign(s, b));
    t0 = q / a;
    t1 = (q != 0) ? c / q : t0;
    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
    return 2;
}

static const float kRefine = 1e-3;

// Quadric coefficients of a ray (origin o, direction d) against the shapes that need refinement.
static const uint kQuadSag = 0, kQuadSphere = 1, kQuadCylinder = 2;
struct Quadric {
    uint kind;
    float c, ck, R2;
    void coeffs(float3 o, float3 d, out float A, out float B, out float C) {
        switch (kind) {
            case kQuadSag:
                A = c * (d.x * d.x + d.y * d.y) + ck * d.z * d.z;
                B = 2 * (c * (o.x * d.x + o.y * d.y) + ck * o.z * d.z - d.z);
                C = c * (o.x * o.x + o.y * o.y) + ck * o.z * o.z - 2 * o.z;
                break;
            case kQuadSphere:
                A = dot(d, d);
                B = 2 * dot(o, d);
                C = dot(o, o) - R2;
                break;
            default:
                A = d.x * d.x + d.y * d.y;
                B = 2 * (o.x * d.x + o.y * d.y);
                C = o.x * o.x + o.y * o.y - R2;
                break;
        }
    }
    // Re-solves from the approximate hit point, where every quantity is small. In float the first
    // solution can be off by ~1e-5 of the distance (the discriminant cancels), so corrections up to
    // kRefine (1 + |t|) are accepted, far below the separation of the two roots of any real hit.
    float refine(float3 o, float3 d, float t) {
        float3 o1 = o + d * t;
        float A, B, C;
        coeffs(o1, d, A, B, C);
        float u0, u1;
        if (solveQuadratic(A, B, C, u0, u1) == 0) return t;
        float du = abs(u0) < abs(u1) ? u0 : u1;
        return abs(du) < kRefine * (1 + abs(t)) ? t + du : t;
    }
};
// refine() moves a root by less than kRefine (1 + |t|): a root farther than that outside
// (tmin, tmax) stays outside, so its refinement is skipped.
bool mayLieIn(float t, float tmin, float tmax) {
    float m = kRefine * (1 + abs(t));
    return !(t + m <= tmin || t - m >= tmax);
}

// ---------------------------------------------------------------- sag surfaces

struct Sag {
    float c, k, rMax, rMin;
    uint data;  // asphere: zMin, zMax, nA at data + 1, coefficients from data + 2
    uint nA;

    float sag(float r) {
        float r2 = r * r;
        float z = c * r2 / (1 + safeSqrt(1 - (1 + k) * c * c * r2));
        float rp = r2 * r2;
        for (uint i = 0; i < nA; ++i) {
            z += g_shapeData[data + 2 + i / 4][i % 4] * rp;
            rp *= r2;
        }
        return z;
    }
    float dsag(float r) {
        float r2 = r * r;
        float root = safeSqrt(1 - (1 + k) * c * c * r2);
        float d = root > 0 ? c * r / root : (c != 0 ? infinity() : 0);
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
        if (r < 1e-30) return float3(0, 0, 1);
        float s = dsag(r);
        return normalize(float3(-s * p.x / r, -s * p.y / r, 1));
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

bool intersectSagQuadric(Sag s, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    Quadric q;
    q.kind = kQuadSag;
    q.c = s.c;
    q.ck = s.c * (1 + s.k);
    q.R2 = 0;
    float t0, t1;
    int n;
    if (s.c == 0) {
        if (d.z == 0) return false;
        t0 = t1 = -o.z / d.z;
        n = 1;
    } else {
        float A, B, C;
        q.coeffs(o, d, A, B, C);
        n = solveQuadratic(A, B, C, t0, t1);
    }
    for (int i = 0; i < n; ++i) {
        float t = i == 0 ? t0 : t1;
        if (s.c != 0) {
            if (!mayLieIn(t, tmin, tmax)) continue;
            t = q.refine(o, d, t);
        }
        if (t <= tmin || t >= tmax) continue;
        float3 p = o + d * t;
        if (!s.accept(p)) continue;
        if (s.c != 0 && (1 + s.k) * s.c * p.z > 1 + 1e-6) continue;  // wrong sheet of the quadric
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
    float a = d.x * d.x + d.y * d.y;
    float b = 2 * (o.x * d.x + o.y * d.y);
    float c = o.x * o.x + o.y * o.y - s.rMax * s.rMax;
    if (a < 1e-35) {
        if (c > 0) return false;
    } else {
        float q0, q1;
        if (solveQuadratic(a, b, c, q0, q1) < 2) return false;
        ta = max(ta, q0);
        tb = min(tb, q1);
    }
    if (abs(d.z) < 1e-35) {
        if (o.z < zMin || o.z > zMax) return false;
    } else {
        float s0 = (zMin - o.z) / d.z, s1 = (zMax - o.z) / d.z;
        if (s0 > s1) { float tmp = s0; s0 = s1; s1 = tmp; }
        ta = max(ta, s0);
        tb = min(tb, s1);
    }
    if (!(ta < tb)) return false;
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
            for (int it = 0; it < 40 && hi - lo > 2e-7 * (1 + abs(hi)); ++it) {
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
void waveTaper(float u, float half, float margin, out float w, out float dw) {
    w = 1;
    dw = 0;
    if (!(margin > 0)) return;
    float e = half - abs(u);
    if (e >= margin) return;
    if (e <= 0) {
        w = 0;
        return;
    }
    float t = e / margin;
    w = t * t * (3 - 2 * t);
    dw = -(u < 0 ? -1.0 : 1.0) * 6 * t * (1 - t) / margin;
}

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
        float u = fx - float(i), v = fy - float(j);
        float2 P = float2(2 * u * u * u - 3 * u * u + 1, -2 * u * u * u + 3 * u * u);
        float2 Q = float2(u * u * u - 2 * u * u + u, u * u * u - u * u);
        float2 dP = float2(6 * u * u - 6 * u, -6 * u * u + 6 * u);
        float2 dQ = float2(3 * u * u - 4 * u + 1, 3 * u * u - 2 * u);
        float2 Pv = float2(2 * v * v * v - 3 * v * v + 1, -2 * v * v * v + 3 * v * v);
        float2 Qv = float2(v * v * v - 2 * v * v + v, v * v * v - v * v);
        float2 dPv = float2(6 * v * v - 6 * v, -6 * v * v + 6 * v);
        float2 dQv = float2(3 * v * v - 4 * v + 1, 3 * v * v - 2 * v);
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
    float3 lo = float3(-hdr.x, -hdr.y, -amp), hi = float3(hdr.x, hdr.y, amp);
    [ForceUnroll] for (int k = 0; k < 3; ++k) {
        if (abs(d[k]) < 1e-35) {
            if (o[k] < lo[k] || o[k] > hi[k]) return false;
            continue;
        }
        float s0 = (lo[k] - o[k]) / d[k], s1 = (hi[k] - o[k]) / d[k];
        ta = max(ta, min(s0, s1));
        tb = min(tb, max(s0, s1));
    }
    if (!(ta < tb)) return false;
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
    float t0, t1;
    float rho0 = c.x + k * (o.z - c.y);
    float A = d.x * d.x + d.y * d.y - k * k * d.z * d.z;
    float B = 2 * (o.x * d.x + o.y * d.y - k * rho0 * d.z);
    float C = o.x * o.x + o.y * o.y - rho0 * rho0;
    int n = solveQuadratic(A, B, C, t0, t1);
    for (int i = 0; i < n; ++i) {
        float t = i == 0 ? t0 : t1;
        if (!mayLieIn(t, tmin, tmax)) continue;
        // Re-solve from the approximate hit, as the reference does (float precision far away).
        float3 p1 = o + d * t;
        float rho1 = c.x + k * (p1.z - c.y);
        float B1 = 2 * (p1.x * d.x + p1.y * d.y - k * rho1 * d.z);
        float C1 = p1.x * p1.x + p1.y * p1.y - rho1 * rho1;
        float u0, u1;
        if (solveQuadratic(A, B1, C1, u0, u1) > 0) {
            float du = abs(u0) < abs(u1) ? u0 : u1;
            if (abs(du) < 1e-3 * (1 + abs(t))) t += du;
        }
        if (t <= tmin || t >= tmax) continue;
        float3 p = o + d * t;
        if (p.z < c.y || p.z > c.w) continue;
        float rho = c.x + k * (p.z - c.y);
        if (rho < 0) continue;
        h.t = t;
        h.p = p;
        h.n = normalize(float3(p.x, p.y, -rho * k));
        h.prim = 0;
        return true;
    }
    return false;
}

// TorusPatchShape: (√(x²+y²) − R)² + (z − zc)² = rho² for arc angles a0 ≤ a ≤ a1.
// Data: (R, zc, rho, -), (a0, a1, rMax, -), (zMin, zMax, -, -).
bool torusOnArc(float a, float a0, float a1) {
    const float eps = 1e-5;
    return (a >= a0 - eps && a <= a1 + eps) || (a + 2 * kPi >= a0 - eps && a + 2 * kPi <= a1 + eps) ||
           (a - 2 * kPi >= a0 - eps && a - 2 * kPi <= a1 + eps);
}
float torusF(float4 c, float3 p) { return sqr(length(p.xy) - c.x) + sqr(p.z - c.y) - c.z * c.z; }

bool intersectTorus(uint data, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float4 c = g_shapeData[data], a = g_shapeData[data + 1], zr = g_shapeData[data + 2];
    float ta = tmin, tb = tmax;
    const float pad = 1e-6 * (1 + a.z + abs(zr.y));
    if (abs(d.z) < 1e-35) {
        if (o.z < zr.x - pad || o.z > zr.y + pad) return false;
    } else {
        float s0 = (zr.x - pad - o.z) / d.z, s1 = (zr.y + pad - o.z) / d.z;
        ta = max(ta, min(s0, s1));
        tb = min(tb, max(s0, s1));
    }
    if (!(ta < tb)) return false;
    // Continue from the slab's entry: a ray from far away (a sun photon starts metres off) keeps
    // its precision there, where the cylinder test would otherwise lose it to cancellation.
    const float t0 = ta;
    const float3 o2 = o + d * t0;
    float A = d.x * d.x + d.y * d.y;
    if (A > 1e-30) {
        float q0, q1;
        if (solveQuadratic(A, 2 * (o2.x * d.x + o2.y * d.y), o2.x * o2.x + o2.y * o2.y - sqr(a.z + pad), q0, q1) < 2)
            return false;
        ta = max(ta, t0 + q0);
        tb = min(tb, t0 + q1);
    } else if (o2.x * o2.x + o2.y * o2.y > sqr(a.z + pad)) {
        return false;
    }
    if (!(ta < tb)) return false;
    const int N = 64;
    float tPrev = ta, fPrev = torusF(c, o2 + d * (ta - t0));
    for (int i = 1; i <= N; ++i) {
        float tCur = ta + (tb - ta) * float(i) / N;
        float fCur = torusF(c, o2 + d * (tCur - t0));
        if ((fPrev <= 0) != (fCur <= 0) || fCur == 0) {
            float lo = tPrev - t0, hi = tCur - t0, flo = fPrev;
            for (int it = 0; it < 40 && hi - lo > 2e-7 * (1 + abs(hi)); ++it) {
                float mid = 0.5 * (lo + hi);
                float fm = torusF(c, o2 + d * mid);
                if ((fm <= 0) == (flo <= 0)) { lo = mid; flo = fm; } else { hi = mid; }
            }
            float3 p = o2 + d * (0.5 * (lo + hi));
            float t = t0 + 0.5 * (lo + hi);
            float rho = length(p.xy);
            if (t > tmin && t < tmax && torusOnArc(atan2(p.z - c.y, rho - c.x), a.x, a.y)) {
                float3 ctr = rho > 0 ? float3(p.xy * (c.x / rho), c.y) : float3(0, 0, c.y);
                h.t = t;
                h.p = p;
                h.n = normalize(p - ctr);
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

static const uint kApDisk = 0, kApRect = 1, kApEllipse = 2, kApDiskHole = 3;

bool intersectPlane(uint data, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    if (d.z == 0) return false;
    float t = -o.z / d.z;
    if (t <= tmin || t >= tmax) return false;
    float3 p = o + d * t;
    p.z = 0;
    float4 a = g_shapeData[data];
    switch (uint(a.x)) {
        case kApDisk: {
            float r2 = p.x * p.x + p.y * p.y;
            if (r2 > a.y * a.y || r2 < a.z * a.z) return false;
            break;
        }
        case kApRect:
            if (abs(p.x) > a.y || abs(p.y) > a.z) return false;
            break;
        case kApEllipse:
            if (sqr(p.x / a.y) + sqr(p.y / a.z) > 1) return false;
            break;
        default: {
            float2 hole = g_shapeData[data + 1].xy;
            if (p.x * p.x + p.y * p.y > a.y * a.y || sqr(p.x - hole.x) + sqr(p.y - hole.y) < a.z * a.z) return false;
            break;
        }
    }
    h.t = t;
    h.p = p;
    h.n = float3(0, 0, 1);
    h.prim = 0;
    return true;
}

// Sphere roots in the geometric formulation (accurate for distant origins), ascending.
void sphereRoots(float3 o, float3 d, float R, out float t0, out float t1, out bool any) {
    float b = dot(o, d);
    float3 perp = o - d * b;
    float disc = R * R - dot(perp, perp);
    any = disc >= 0;
    t0 = t1 = 0;
    if (!any) return;
    float s = sqrt(disc);
    float c = dot(o, o) - R * R;
    float q = -b - copySign(s, b);
    t0 = q;
    t1 = (q != 0) ? c / q : q;
    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
}

bool intersectSphere(float R, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float t0, t1;
    bool any;
    sphereRoots(o, d, R, t0, t1, any);
    if (!any) return false;
    Quadric q;
    q.kind = kQuadSphere;
    q.c = 0;
    q.ck = 0;
    q.R2 = R * R;
    float t = 0;
    bool ok = false;
    [ForceUnroll] for (int i = 0; i < 2 && !ok; ++i) {
        float tr = i == 0 ? t0 : t1;
        if (!mayLieIn(tr, tmin, tmax)) continue;
        t = q.refine(o, d, tr);
        ok = !(t <= tmin || t >= tmax);
    }
    if (!ok) return false;
    h.t = t;
    h.p = o + d * t;
    h.n = normalize(h.p / R);
    h.prim = 0;
    return true;
}

bool roundWallSolid(uint data, float3 p) {
    float4 a = g_shapeData[data];  // R, H, n
    if (p.z < 0 || p.z > a.y) return false;
    float az = atan2(p.x, p.y);
    uint n = uint(a.z);
    for (uint i = 0; i < n; ++i) {
        float4 op = g_shapeData[data + 1 + i];  // azimuth, width, sill, top
        float dAz = remainderf(az - op.x, 2 * kPi);
        if (abs(dAz) * a.x < 0.5 * op.y && p.z > op.z && p.z < op.w) return false;
    }
    return true;
}

bool domeSolid(float4 a, float3 p) {  // R, azimuth, width, top
    if (p.z < 0) return false;
    if (a.z <= 0) return true;
    float el = asin(clamp(p.z / a.x, -1.0, 1.0));
    if (el >= a.w) return true;
    float dAz = remainderf(atan2(p.x, p.y) - a.y, 2 * kPi);
    float horiz = a.x * cos(el);
    return abs(sin(dAz)) * horiz >= 0.5 * a.z || cos(dAz) < 0;
}

// Cylinder-like walls (open cylinder, round wall with openings) and the dome share this loop.
bool intersectCylinderLike(uint kind, uint data, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    float4 a = g_shapeData[data];
    float t0, t1;
    Quadric q;
    q.c = 0;
    q.ck = 0;
    q.R2 = a.x * a.x;
    if (kind == kShapeDome) {
        bool any;
        sphereRoots(o, d, a.x, t0, t1, any);
        if (!any) return false;
        q.kind = kQuadSphere;
    } else {
        // Roots of A t² + 2 b t + c = 0 in the geometric form (as for the sphere): the discriminant
        // b² − A c = A (R² − |perp|²), with perp the closest approach to the axis, does not cancel
        // when the ray starts far away.
        float2 o2 = o.xy, d2 = d.xy;
        float A = dot(d2, d2);
        if (A < 1e-35) return false;
        float b = dot(o2, d2);
        float2 perp = o2 - d2 * (b / A);
        float disc = A * (a.x * a.x - dot(perp, perp));
        if (disc < 0) return false;
        float qq = -(b + copySign(sqrt(disc), b));
        float c = dot(o2, o2) - a.x * a.x;
        t0 = qq / A;
        t1 = qq != 0 ? c / qq : t0;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        q.kind = kQuadCylinder;
    }
    [ForceUnroll] for (int i = 0; i < 2; ++i) {
        float t = i == 0 ? t0 : t1;
        if (!mayLieIn(t, tmin, tmax)) continue;
        t = q.refine(o, d, t);
        if (t <= tmin || t >= tmax) continue;
        float3 p = o + d * t;
        bool solid;
        if (kind == kShapeCylinder) solid = !(p.z < a.y || p.z > a.z);
        else if (kind == kShapeRoundWall) solid = roundWallSolid(data, p);
        else solid = domeSolid(a, p);
        if (!solid) continue;
        h.t = t;
        h.p = p;
        h.n = kind == kShapeDome ? normalize(p) : normalize(float3(p.x, p.y, 0));
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
            float f = tf[a] * 1.0000004;
            tmax = f < tmax ? f : tmax;
        }
        tEnter = tmin;
        return tmin <= tmax;
    }
};

static const int kStack = 64;

// A triangle's (unnormalised) normal as the cross product of its two shorter edges. For a sliver —
// a long thin triangle, as fan-triangulated bevels produce — the two long edges are nearly
// parallel and their cross product cancels catastrophically in float; the shorter pair does not.
float3 triangleNormal(float3 v0, float3 v1, float3 v2) {
    float3 a = v1 - v0, b = v2 - v0, c = v2 - v1;
    float la = dot(a, a), lb = dot(b, b), lc = dot(c, c);
    if (lc >= la && lc >= lb) return cross(a, b);
    return la >= lb ? cross(b, c) : cross(a, c);
}

// Möller–Trumbore's inside test, with the distance taken to the triangle's plane: Möller–Trumbore's
// own dot(e2, qv) / det loses precision in proportion to 1 / area, which on sliver triangles
// placed float hit points microns off the surface, beyond the ray offset, so the next ray began
// inside the body. (The double-precision reference does not need this.)
bool hitTriangle(GTriangle tri, float3 o, float3 d, float tmin, float tmax, out float t) {
    t = 0;
    float3 v0 = tri.v0.xyz;
    float3 e1 = tri.v1.xyz - v0, e2 = tri.v2.xyz - v0;
    float3 pv = cross(d, e2);
    float det = dot(e1, pv);
    if (det == 0) return false;
    float inv = 1 / det;
    float3 tv = o - v0;
    float u = dot(tv, pv) * inv;
    if (u < 0 || u > 1) return false;
    float3 qv = cross(tv, e1);
    float v = dot(d, qv) * inv;
    if (v < 0 || u + v > 1) return false;
    float3 np = triangleNormal(v0, tri.v1.xyz, tri.v2.xyz);
    t = -dot(tv, np) / dot(d, np);
    return t > tmin && t < tmax;
}

bool intersectMesh(GBoundary b, float3 o, float3 d, float tmin, float tmax, inout LocalHit h) {
    RaySlabs rs = RaySlabs(o, d);
    uint stackNode[kStack];
    float stackT[kStack];
    int sp = 0;
    uint cur = b.mesh.y;
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
    h.prim = asuint(tri.v0.w);
    h.owner = asuint(tri.v1.w);
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
    uint cw = asuint(tri.v2.w);
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
    uint cw = asuint(tri.v2.w);
    if (cw == kGpuNone || (cw & kCornerUvs) == 0) return false;
    float3 b;
    triangleBarycentrics(tri, p, b);
    uint w = (cw & kCornerIndex) + ((cw & kCornerNormals) != 0 ? 3 : 0);
    uv = float2(asfloat(g_normals[w]), asfloat(g_normals[w + 1])) * b.x +
         float2(asfloat(g_normals[w + 2]), asfloat(g_normals[w + 3])) * b.y +
         float2(asfloat(g_normals[w + 4]), asfloat(g_normals[w + 5])) * b.z;
    return true;
}

// Boundaries a shadow ray crosses (null) or is attenuated by (stained glass) rather than stopped.
bool isPassableType(uint type) { return type == kSurfNull || type == kSurfStainedGlass; }
bool isNullBoundary(uint boundary) { return isPassableType(g_optics[g_boundaries[boundary].links.z].a.x); }

// Any hit on matter (a triangle whose boundary is not null) in (tmin, tmax); no ordering, first
// found wins. sawNull reports crossing a null boundary's triangle.
bool occludedMesh(GBoundary b, float3 o, float3 d, float tmin, float tmax, inout bool sawNull) {
    RaySlabs rs = RaySlabs(o, d);
    uint stackNode[kStack];
    int sp = 0;
    uint cur = b.mesh.y;
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
                if (!isNullBoundary(asuint(tri.v1.w))) return true;
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
    uint kind = asuint(b.boxLo.w), data = asuint(b.boxHi.w);
    switch (kind) {
        case kShapeSag: return intersectSagQuadric(loadSag(data, false), o, d, tmin, tmax, h);
        case kShapeAsphere: return intersectSagNumeric(loadSag(data, true), o, d, tmin, tmax, h);
        case kShapePlane: return intersectPlane(data, o, d, tmin, tmax, h);
        case kShapeSphere: return intersectSphere(g_shapeData[data].x, o, d, tmin, tmax, h);
        case kShapeCylinder:
        case kShapeRoundWall:
        case kShapeDome: return intersectCylinderLike(kind, data, o, d, tmin, tmax, h);
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

// ---------------------------------------------------------------- emitter sampling (local frame)

// Uniform area sampling; false when the shape has no area sampler.
bool sampleArea(GBoundary b, float u1, float u2, float u3, out float3 p, out float3 n) {
    uint kind = asuint(b.boxLo.w), data = asuint(b.boxHi.w);
    float4 a = g_shapeData[data];
    p = 0;
    n = float3(0, 0, 1);
    switch (kind) {
        case kShapeSag: {  // flat only (sampleable flag)
            float r = sqrt(a.w * a.w + u1 * (a.z * a.z - a.w * a.w));
            float phi = 2 * kPi * u2;
            p = float3(r * cos(phi), r * sin(phi), 0);
            return true;
        }
        case kShapePlane: {
            uint ap = uint(a.x);
            if (ap == kApDisk) {
                float r = sqrt(a.z * a.z + u1 * (a.y * a.y - a.z * a.z));
                float phi = 2 * kPi * u2;
                p = float3(r * cos(phi), r * sin(phi), 0);
            } else if (ap == kApRect) {
                p = float3((2 * u1 - 1) * a.y, (2 * u2 - 1) * a.z, 0);
            } else if (ap == kApEllipse) {
                float r = sqrt(u1), phi = 2 * kPi * u2;
                p = float3(a.y * r * cos(phi), a.z * r * sin(phi), 0);
            } else {
                return false;
            }
            return true;
        }
        case kShapeSphere:
            n = sampleUniformSphere(u1, u2);
            p = n * a.x;
            return true;
        case kShapeCylinder: {
            float phi = 2 * kPi * u1;
            n = float3(cos(phi), sin(phi), 0);
            p = float3(a.x * n.x, a.x * n.y, a.y + u2 * (a.z - a.y));
            return true;
        }
        case kShapeMesh: {
            uint cdf = asuint(a.x), count = b.mesh.w;
            uint lo = 0, len = count;  // first CDF entry ≥ u3 (std::lower_bound)
            while (len > 0) {
                uint half = len / 2;
                if (g_scalars[cdf + lo + half] < u3) { lo += half + 1; len -= half + 1; }
                else len = half;
            }
            uint i = min(lo, count - 1);
            GTriangle tri = g_triangles[b.mesh.z + i];
            float su = sqrt(u1);
            float b0 = 1 - su, b1 = u2 * su;
            p = tri.v0.xyz * b0 + tri.v1.xyz * b1 + tri.v2.xyz * (1 - b0 - b1);
            n = normalize(triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz));
            return true;
        }
        default: return false;
    }
}

// Point on the shape as seen from `ref` with a solid-angle pdf (false on failure). Local frame.
bool sampleFrom(GBoundary b, float3 ref, float u1, float u2, float u3, out float3 p, out float3 n, out float pdfW) {
    uint kind = asuint(b.boxLo.w);
    float area = b.misc.y;
    pdfW = 0;
    p = 0;
    n = float3(0, 0, 1);
    if (kind == kShapeSphere) {
        float R = g_shapeData[asuint(b.boxHi.w)].x;
        float dc2 = dot(ref, ref);
        if (dc2 > R * R * (1 + 1e-6)) {
            float dc = sqrt(dc2);
            float sin2Max = R * R / dc2;
            float cosMax = safeSqrt(1 - sin2Max);
            float oneMinusCos = sin2Max / (1 + cosMax);
            Frame f = Frame(-ref / dc);
            float oneMinusCosT = u1 * oneMinusCos;
            float cosT = 1 - oneMinusCosT;
            float sinT = safeSqrt(oneMinusCosT * (2 - oneMinusCosT));
            float phi = 2 * kPi * u2;
            float3 w = f.toWorld(float3(sinT * cos(phi), sinT * sin(phi), cosT));
            LocalHit h;
            if (!intersectSphere(R, ref, w, 0, infinity(), h)) {
                float tt = dot(-ref, w);
                p = normalize(ref + w * tt) * R;
            } else {
                p = h.p;
            }
            n = normalize(p);
            pdfW = 1 / (2 * kPi * oneMinusCos);
            return true;
        }
    }
    if (asuint(b.misc.z) == 0 || area <= 0) return false;
    if (!sampleArea(b, u1, u2, u3, p, n)) return false;
    float3 w = p - ref;
    float d2 = dot(w, w);
    if (d2 <= 0) return false;
    float cosL = abs(dot(n, w)) / sqrt(d2);
    if (cosL <= 1e-7) return false;
    pdfW = d2 / (cosL * area);
    return true;
}

float pdfFrom(GBoundary b, float3 ref, float3 p, float3 n) {
    uint kind = asuint(b.boxLo.w);
    float area = b.misc.y;
    if (kind == kShapeSphere) {
        float R = g_shapeData[asuint(b.boxHi.w)].x;
        float dc2 = dot(ref, ref);
        if (dc2 > R * R * (1 + 1e-6)) {
            float sin2Max = R * R / dc2;
            float cosMax = safeSqrt(1 - sin2Max);
            return 1 / (2 * kPi * sin2Max / (1 + cosMax));
        }
    }
    if (asuint(b.misc.z) == 0 || area <= 0) return 0;
    float3 w = p - ref;
    float d2 = dot(w, w);
    float cosL = abs(dot(n, w)) / sqrt(d2);
    if (cosL <= 1e-7) return 0;
    return d2 / (cosL * area);
}
