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
    uint tri;       // meshes: the hit triangle's record (normal maps' tangent frame); else kGpuNone
    uint boundary;
    uint prim;      // meshes: global triangle slot (area sampling may use its separate copy)
    float scale;    // float error scale of the boundary's local frame (ray offsets)
    bool ambiguousContact;
};

struct Endpoint {
    float3 p, normal;
    float scale, error;
    uint region, boundary, primitive;
};
Endpoint endpoint(float3 p, uint region, float scale = 0, uint boundary = kGpuNone,
                  float3 normal = float3(0), uint primitive = kGpuNone) {
    Endpoint e;
    e.p = p; e.region = region; e.scale = scale; e.error = 0;
    e.boundary = boundary; e.normal = normal; e.primitive = primitive;
    return e;
}
Endpoint endpoint(SurfaceHit h, uint region) {
    return endpoint(h.p, region, h.scale, h.boundary, h.n, h.prim);
}

float3 pointOnTrianglePlane(GTriangle tri, float3 p) {
    float3 n = triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz);
    float3 q = p - tri.v0.xyz, a = abs(n);
    if (a.x >= a.y && a.x >= a.z && a.x > 0) p.x = tri.v0.x - (q.y * n.y + q.z * n.z) / n.x;
    else if (a.y >= a.z && a.y > 0) p.y = tri.v0.y - (q.x * n.x + q.z * n.z) / n.y;
    else if (a.z > 0) p.z = tri.v0.z - (q.x * n.x + q.y * n.y) / n.z;
    return p;
}

// Where the ray lo + t ld meets triangle tri's plane (false if parallel to it).
bool trianglePlaneDistance(GTriangle tri, float3 lo, float3 ld, out float t) {
    float3 np = triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz);
    float denom = dot(ld, np);
    t = -dot(lo - tri.v0.xyz, np) / denom;
    return denom != 0;
}
// The ray lo + t ld at triangle tri (record slot), in the triangle's frame.
LocalHit triangleHit(GTriangle tri, uint slot, float3 lo, float3 ld, float t) {
    LocalHit lh = (LocalHit)0;
    lh.t = t;
    lh.p = lo + ld * t;
    lh.n = normalize(triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz));
    lh.prim = tri.primitive;
    lh.owner = tri.owner;
    lh.slot = slot;
    return lh;
}

// A native far-origin root may round away the centre's low part. Recover its spherical direction
// on the SAME pole-anchored sphere used by nearby rays, rather than reverting to a shifted planet.
// Stable sag reconstruction also places the hit on the surface when a huge t cannot encode its
// centimetre-scale final displacement. Both hemispheres remain analytic; no plane replaces it.
void reconstructBoundaryHit(GBoundary b, float3 o, float3 d, LocalHit lh, inout SurfaceHit hit) {
    if (!hasSphereFrame(b)) {
        if (b.shape == kShapeMesh) {
            lh.p = pointOnTrianglePlane(g_triangles[lh.slot], lh.p);
            hit.pLocal = lh.p;
        }
        hit.p = xfPoint(b.toWorld, lh.p);
        hit.n = normalize(xfVector(b.toWorld, lh.n));
        hit.scale = b.scale;
        return;
    }
    uint data = b.data;
    float R = g_shapeData[data].x;
    float4 toPole[3], fromPole[3];
    [ForceUnroll] for (int k = 0; k < 3; ++k) {
        toPole[k] = g_shapeData[data + 1 + k];
        fromPole[k] = g_shapeData[data + 4 + k];
    }
    float3 p;
    bool north;
    if (usesSphereFrame(b, o)) {
        p = xfPoint(toPole, o) + xfVector(toPole, d) * lh.t;
        north = p.z >= -R;
    } else {
        float3 n = normalize(xfVector(toPole, xfVector(b.toWorld, lh.n)));
        p = float3(n.xy * R, 0);
        north = n.z >= 0;
    }
    float2 xy = p.xy / R;
    float r2 = dot(xy, xy);
    if (r2 > 1) { p.xy /= sqrt(r2); xy = p.xy / R; r2 = 1; }
    float nz = safeSqrt(1 - r2);
    p.z = north ? -R * r2 / (1 + nz) : -R * (1 + nz);
    float3 normal = normalize(float3(xy, north ? nz : -nz));
    hit.p = xfPoint(fromPole, p);
    hit.n = normalize(xfVector(fromPole, normal));
    hit.pLocal = xfPoint(b.toLocal, hit.p);
    hit.nLocal = normalize(xfVector(b.toLocal, hit.n));
    // The point was reconstructed on the surface from small pole coordinates. Its precision now
    // follows those coordinates, not the millions-of-metres native centre or the ray's old origin.
    hit.scale = max(maxAbsComponent(p), max(abs(fromPole[0].w), max(abs(fromPole[1].w), abs(fromPole[2].w))));
}

// The surface hit of the ray o + t d at the winning local hit lh on boundary record bi: the world
// point and normal reconstructed (reconstructBoundaryHit), a mesh's shading normal and texture
// coordinates interpolated.
SurfaceHit completeHit(uint bi, LocalHit lh, float3 o, float3 d) {
    GBoundary b = g_boundaries[bi];
    SurfaceHit hit = (SurfaceHit)0;
    hit.t = lh.t;
    hit.pLocal = lh.p;
    hit.nLocal = lh.n;
    reconstructBoundaryHit(b, o, d, lh, hit);
    hit.ns = hit.n;
    hit.tri = kGpuNone;
    if (b.shape == kShapeMesh) {
        float3 ns = meshShadingNormal(lh.slot, lh.p, lh.n);
        if (any(ns != lh.n)) hit.ns = normalize(xfVector(b.toWorld, ns));
        hit.hasUv = meshUv(lh.slot, lh.p, hit.uv);
        hit.tri = lh.slot;
    }
    hit.boundary = bi;
    hit.prim = b.shape == kShapeMesh ? lh.slot : lh.prim;
    return hit;
}

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

[noinline]
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
        if (intersectBoundary(b, o, d, 0, best, lh)) {
            bestLocal = lh;
            bestB = bi;
            q.CommitProceduralPrimitiveHit(lh.t);
        }
    }
    uint status = q.CommittedStatus();
    if (status == COMMITTED_NOTHING) return false;
    if (status == COMMITTED_TRIANGLE_HIT) {
        GBoundary g = g_boundaries[q.CommittedInstanceID()];
        uint slot = g.meshTris + q.CommittedPrimitiveIndex();
        GTriangle tri = g_triangles[slot];
        float3 lo = xfPoint(g.toLocal, o), ld = xfVector(g.toLocal, d);
        // The distance re-solved against the triangle's plane, exactly as the software path does.
        float t;
        trianglePlaneDistance(tri, lo, ld, t);
        bestLocal = triangleHit(tri, slot, lo, ld, t);
        bestB = tri.owner;
    }
    hit = completeHit(bestB, bestLocal, o, d);
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
        if (intersectBoundary(b, o, d, 0, tmax, lh))
            q.CommitProceduralPrimitiveHit(lh.t);
    }
    return q.CommittedStatus() != COMMITTED_NOTHING;
}

[noinline]
bool occluded(float3 o, float3 d, float tmax, out bool sawNull) {
    sawNull = false;
    if (globals().counts.x == 0) return false;
    if (anyHit(o, d, tmax, kMaskMatter)) return true;
    sawNull = globals().target.y != 0 && anyHit(o, d, tmax, kMaskNull);
    return false;
}

#else
// Software traversal (every device): the top-level hierarchy, then each candidate in its frame.

[noinline]
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
                LocalHit lh;
                lh.owner = kGpuNone;
                if (intersectBoundary(b, o, d, 0, best, lh)) {
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
    hit = completeHit(bestB, bestLocal, o, d);
    return true;
}

// Is the open segment (0, tmax) blocked by matter — a boundary that is not null? Shadow rays need
// only this, so the first opaque hit ends the search (no ordering). sawNull: a null boundary lies
// on the segment; the caller then walks the segment in order (regions and media change there).
[noinline]
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
                if (b.shape == kShapeMesh) {
                    if (occludedMesh(b, lo, ld, 0, tmax, sawNull)) return true;
                    continue;
                }
                LocalHit lh;
                lh.owner = kGpuNone;
                if (!intersectBoundary(b, o, d, 0, tmax, lh)) continue;
                if (!isPassableType(g_optics[b.optics].a.x)) return true;
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

// The region a ray along d meets hit from: the front's if it arrives against the normal.
uint incidentRegion(SurfaceHit hit, float3 d) {
    GBoundary b = g_boundaries[hit.boundary];
    return dot(d, hit.n) < 0 ? b.front : b.back;
}
