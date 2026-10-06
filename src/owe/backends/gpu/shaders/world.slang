// Closest-hit queries against the world: the boundary hierarchy (camera-relative), each candidate
// carried into its shape's frame with one float rotation and translation.
#pragma once
#include "geometry.slang"

struct SurfaceHit {
    float t;
    float3 p;       // camera-relative world position
    float3 n;       // world geometric normal, toward the front region
    float3 ns;      // world shading normal (smooth meshes; else n), on n's side
    float3 pLocal;  // shape-local position and normal (textures, emitter pdfs)
    float3 nLocal;
    float2 uv;      // texture coordinates (textured meshes, where hasUv)
    bool hasUv;
    uint boundary;
    float scale;    // float error scale of the boundary's local frame (ray offsets)
};

#ifdef OWE_RAY_QUERY
// Hardware traversal (VK_KHR_ray_query): the device walks its acceleration structures; analytic
// boundaries come back as box candidates that the kernel's own intersection code resolves (in the
// boundary's frame, reached with the record's transform exactly as in the software path).

RayDesc makeRay(float3 o, float3 d, float tmax) {
    RayDesc ray;
    ray.Origin = o;
    ray.Direction = d;
    ray.TMin = 0;
    ray.TMax = min(tmax, 3.0e38);
    return ray;
}

bool intersectWorld(float3 o, float3 d, float tmax, out SurfaceHit hit) {
    hit = (SurfaceHit)0;
    if (globals().counts.x == 0) return false;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g_scene, RAY_FLAG_NONE, 0xFF, makeRay(o, d, tmax));
    LocalHit bestLocal = (LocalHit)0;
    uint bestB = 0;
    while (q.Proceed()) {
        // Triangles are opaque (the hardware commits them); only analytic boxes come back here.
        if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) {
            q.CommitNonOpaqueTriangleHit();
            continue;
        }
        uint bi = q.CandidateInstanceID();
        GBoundary b = g_boundaries[bi];
        float best = q.CommittedStatus() == COMMITTED_NOTHING ? tmax : q.CommittedRayT();
        LocalHit lh;
        lh.owner = kGpuNone;
        if (intersectShape(b, xfPoint(b.toLocal, o), xfVector(b.toLocal, d), 0, best, lh)) {
            bestLocal = lh;
            bestB = bi;
            q.CommitProceduralPrimitiveHit(lh.t);
        }
    }
    uint status = q.CommittedStatus();
    if (status == COMMITTED_NOTHING) return false;
    if (status == COMMITTED_TRIANGLE_HIT) {
        GBoundary g = g_boundaries[q.CommittedInstanceID()];
        GTriangle tri = g_triangles[g.mesh.z + q.CommittedPrimitiveIndex()];
        float3 lo = xfPoint(g.toLocal, o), ld = xfVector(g.toLocal, d);
        // The distance re-solved against the triangle's plane, exactly as the software path does.
        float3 np = triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz);
        bestLocal.t = -dot(lo - tri.v0.xyz, np) / dot(ld, np);
        bestLocal.p = lo + ld * bestLocal.t;
        bestLocal.n = normalize(np);
        bestLocal.prim = asuint(tri.v0.w);
        bestLocal.slot = g.mesh.z + q.CommittedPrimitiveIndex();
        bestB = asuint(tri.v1.w);
    }
    GBoundary b = g_boundaries[bestB];
    hit.t = bestLocal.t;
    hit.pLocal = bestLocal.p;
    hit.nLocal = bestLocal.n;
    hit.p = xfPoint(b.toWorld, bestLocal.p);
    hit.n = normalize(xfVector(b.toWorld, bestLocal.n));
    hit.ns = hit.n;
    if (asuint(b.boxLo.w) == kShapeMesh) {
        float3 ns = meshShadingNormal(bestLocal.slot, bestLocal.p, bestLocal.n);
        if (any(ns != bestLocal.n)) hit.ns = normalize(xfVector(b.toWorld, ns));
        hit.hasUv = meshUv(bestLocal.slot, bestLocal.p, hit.uv);
    }
    hit.boundary = bestB;
    hit.scale = b.misc.x;
    return true;
}

// Any hit on the given instances in (0, tmax): the first committed hit ends the search.
bool anyHit(float3 o, float3 d, float tmax, uint mask) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(g_scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, mask, makeRay(o, d, tmax));
    while (q.Proceed()) {
        if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) {
            q.CommitNonOpaqueTriangleHit();
            continue;
        }
        GBoundary b = g_boundaries[q.CandidateInstanceID()];
        LocalHit lh;
        lh.owner = kGpuNone;
        if (intersectShape(b, xfPoint(b.toLocal, o), xfVector(b.toLocal, d), 0, tmax, lh))
            q.CommitProceduralPrimitiveHit(lh.t);
    }
    return q.CommittedStatus() != COMMITTED_NOTHING;
}

bool occluded(float3 o, float3 d, float tmax, out bool sawNull) {
    sawNull = false;
    if (globals().counts.x == 0) return false;
    if (anyHit(o, d, tmax, kMaskMatter)) return true;
    sawNull = globals().target.y != 0 && anyHit(o, d, tmax, kMaskNull);
    return false;
}

#else
// Software traversal (every device): the top-level hierarchy, then each candidate in its frame.

bool intersectWorld(float3 o, float3 d, float tmax, out SurfaceHit hit) {
    hit = (SurfaceHit)0;
    if (globals().counts.x == 0) return false;
    RaySlabs rs = RaySlabs(o, d);
    uint stackNode[kStack];
    float stackT[kStack];
    int sp = 0;
    uint cur = 0;
    float best = tmax;
    bool found = false;
    uint bestB = 0;
    LocalHit bestLocal = (LocalHit)0;
    float tEnter;
    GNode root = g_nodes[0];
    if (!rs.hit(root.lo.xyz, root.hi.xyz, 0, best, tEnter)) return false;
    while (true) {
        GNode node = g_nodes[cur];
        uint na = asuint(node.lo.w), nb = asuint(node.hi.w);
        bool descend = false;
        if ((nb & kNodeInner) == 0) {
            for (uint i = 0; i < nb; ++i) {
                uint bi = g_top[na + i];
                float tb;
                if (!rs.hit(g_boundaries[bi].boxLo.xyz, g_boundaries[bi].boxHi.xyz, 0, best, tb)) continue;
                GBoundary b = g_boundaries[bi];
                float3 lo = xfPoint(b.toLocal, o);
                float3 ld = xfVector(b.toLocal, d);
                LocalHit lh;
                lh.owner = kGpuNone;
                if (intersectShape(b, lo, ld, 0, best, lh)) {
                    best = lh.t;
                    bestLocal = lh;
                    bestB = lh.owner != kGpuNone ? lh.owner : bi;  // a mesh group reports its member
                    found = true;
                }
            }
        } else {
            uint first = cur + 1, second = na;
            if (rs.inv[nb & 3u] < 0) { uint tmp = first; first = second; second = tmp; }
            GNode n0 = g_nodes[first], n1 = g_nodes[second];
            float t0, t1;
            bool h0 = rs.hit(n0.lo.xyz, n0.hi.xyz, 0, best, t0);
            bool h1 = rs.hit(n1.lo.xyz, n1.hi.xyz, 0, best, t1);
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
    GBoundary b = g_boundaries[bestB];
    hit.t = bestLocal.t;
    hit.pLocal = bestLocal.p;
    hit.nLocal = bestLocal.n;
    hit.p = xfPoint(b.toWorld, bestLocal.p);
    hit.n = normalize(xfVector(b.toWorld, bestLocal.n));
    hit.ns = hit.n;
    if (asuint(b.boxLo.w) == kShapeMesh) {
        float3 ns = meshShadingNormal(bestLocal.slot, bestLocal.p, bestLocal.n);
        if (any(ns != bestLocal.n)) hit.ns = normalize(xfVector(b.toWorld, ns));
        hit.hasUv = meshUv(bestLocal.slot, bestLocal.p, hit.uv);
    }
    hit.boundary = bestB;
    hit.scale = b.misc.x;
    return true;
}

// Is the open segment (0, tmax) blocked by matter — a boundary that is not null? Shadow rays need
// only this, so the first opaque hit ends the search (no ordering). sawNull: a null boundary lies
// on the segment; the caller then walks the segment in order (regions and media change there).
bool occluded(float3 o, float3 d, float tmax, out bool sawNull) {
    sawNull = false;
    if (globals().counts.x == 0) return false;
    RaySlabs rs = RaySlabs(o, d);
    uint stackNode[kStack];
    int sp = 0;
    uint cur = 0;
    float tEnter;
    GNode root = g_nodes[0];
    if (!rs.hit(root.lo.xyz, root.hi.xyz, 0, tmax, tEnter)) return false;
    while (true) {
        GNode node = g_nodes[cur];
        uint na = asuint(node.lo.w), nb = asuint(node.hi.w);
        if ((nb & kNodeInner) == 0) {
            for (uint i = 0; i < nb; ++i) {
                uint bi = g_top[na + i];
                float tb;
                if (!rs.hit(g_boundaries[bi].boxLo.xyz, g_boundaries[bi].boxHi.xyz, 0, tmax, tb)) continue;
                GBoundary b = g_boundaries[bi];
                float3 lo = xfPoint(b.toLocal, o);
                float3 ld = xfVector(b.toLocal, d);
                if (asuint(b.boxLo.w) == kShapeMesh) {
                    if (occludedMesh(b, lo, ld, 0, tmax, sawNull)) return true;
                    continue;
                }
                LocalHit lh;
                lh.owner = kGpuNone;
                if (!intersectShape(b, lo, ld, 0, tmax, lh)) continue;
                if (!isPassableType(g_optics[b.links.z].a.x)) return true;
                sawNull = true;
            }
        } else {
            GNode n0 = g_nodes[cur + 1], n1 = g_nodes[na];
            float t0, t1;
            bool h0 = rs.hit(n0.lo.xyz, n0.hi.xyz, 0, tmax, t0);
            bool h1 = rs.hit(n1.lo.xyz, n1.hi.xyz, 0, tmax, t1);
            if (h0 || h1) {
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

#endif

// Moves a new ray origin off a surface along the new ray itself (no lateral error); grazing
// directions fall back to the normal. The displacement follows float precision: a few hundred
// ulps of the larger of the point's and the local frame's coordinates, far below optical
// feature sizes (40 nm for an eyepiece lens 2 cm from the eye, 20 µm at 10 m).
float surfaceEpsilon(float3 p, float scale) { return 2e-6 * max(maxAbsComponent(p), scale) + 1e-9; }
float3 offsetOrigin(float3 p, float3 n, float3 dir, float scale) {
    float e = surfaceEpsilon(p, scale);
    float c = dot(dir, n);
    if (abs(c) > 0.05) return p + dir * (e / abs(c));
    return c >= 0 ? p + n * e : p - n * e;
}
