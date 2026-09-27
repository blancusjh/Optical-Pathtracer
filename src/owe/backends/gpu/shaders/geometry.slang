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
    return true;
}

bool isNullBoundary(uint boundary) { return g_optics[g_boundaries[boundary].links.z].a.x == kSurfNull; }

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
        case kShapeMesh: return intersectMesh(b, o, d, tmin, tmax, h);
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
