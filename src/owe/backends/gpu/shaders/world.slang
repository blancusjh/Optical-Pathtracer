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
        GTriangle tri = g_triangles[g.meshTris + q.CommittedPrimitiveIndex()];
        float3 lo = xfPoint(g.toLocal, o), ld = xfVector(g.toLocal, d);
        // The distance re-solved against the triangle's plane, exactly as the software path does.
        float3 np = triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz);
        bestLocal.t = -dot(lo - tri.v0.xyz, np) / dot(ld, np);
        bestLocal.p = lo + ld * bestLocal.t;
        bestLocal.n = normalize(np);
        bestLocal.prim = tri.primitive;
        bestLocal.slot = g.meshTris + q.CommittedPrimitiveIndex();
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

// Contact queries retain an exact surface anchor until the outgoing numerical offset's
// short segment is known clear. The current region determines zero-distance interface order.
float contactError(float3 p, float scale) {
    return 8 * 1.1920928955078125e-7 * max(1e-12, max(maxAbsComponent(p), scale));
}

// Mesh sampling records and traversal records can have different slots for the same
// original face. Connected exactly coplanar facets also belong to that face; separate
// owners or parallel layers do not. Departure and finite endpoint recognition share it.
[noinline]
bool sameMeshFace(uint previousSlot, uint currentSlot) {
    if (previousSlot == kGpuNone || currentSlot == kGpuNone) return false;
    GTriangle previous = g_triangles[previousSlot], current = g_triangles[currentSlot];
    if (previous.owner != current.owner) return false;
    if (previous.primitive == current.primitive) return true;
    uint shared = 0;
    float3 pv[3] = {previous.v0.xyz, previous.v1.xyz, previous.v2.xyz};
    float3 cv[3] = {current.v0.xyz, current.v1.xyz, current.v2.xyz};
    [ForceUnroll] for (int i = 0; i < 3; ++i)
        [ForceUnroll] for (int j = 0; j < 3; ++j) if (all(pv[i] == cv[j])) ++shared;
    float3 pn = triangleNormal(pv[0], pv[1], pv[2]);
    float3 cn = triangleNormal(cv[0], cv[1], cv[2]);
    return shared >= 2 && all(cross(pn, cn) == 0) && dot(pn, cn) > 0 &&
           dot(pn, cv[0] - pv[0]) == 0;
}

bool contactAccepted(uint bi, LocalHit lh, float3 d, Endpoint departure, bool forwardOnly) {
    if ((forwardOnly || bi == departure.boundary) && lh.t < 0) return false;
    GBoundary b = g_boundaries[bi];
    float3 n = xfVector(b.toWorld, lh.n);
    if (bi == departure.boundary) {
        bool sameFace = dot(d, n) * dot(d, departure.normal) >= 0;
        if (b.shape == kShapeMesh && departure.primitive != kGpuNone) {
            sameFace = sameMeshFace(departure.primitive, lh.slot);
        }
        // A grazing self-root may have a large ray parameter although its displacement
        // normal to the departed face is within the anchor's rounding error. A mesh face
        // has only one plane crossing, so its exact primitive identity is sufficient.
        bool nearSelf = b.shape == kShapeMesh || abs(lh.t * dot(d, departure.normal)) <= departure.error;
        if (sameFace && nearSelf) return false;
    }
    if (abs(lh.t * dot(d, normalize(n))) > departure.error) return lh.t >= 0;
    uint incident = dot(d, n) < 0 ? b.front : b.back;
    return incident == departure.region;
}

// The region a ray in `region` is in after meeting boundary bi: the far side's if it can pass into
// it, else unchanged (opaque matter, or a sheet with the same region on both sides); kGpuNone if the
// ray meets the boundary from another region.
uint contactOutcome(uint bi, float3 n, float3 d, uint region) {
    GBoundary b = g_boundaries[bi];
    bool entering = dot(d, xfVector(b.toWorld, n)) < 0;
    if ((entering ? b.front : b.back) != region) return kGpuNone;
    uint type = g_optics[b.optics].a.x;
    return type == kSurfDielectric || isPassableType(type) ? (entering ? b.back : b.front) : region;
}

void considerContact(uint bi, LocalHit lh, float3 d, Endpoint departure, bool forwardOnly,
                     inout bool found, inout uint bestB, inout LocalHit bestLocal, inout bool ambiguous) {
    if (!contactAccepted(bi, lh, d, departure, forwardOnly)) return;
    float t = max(0, lh.t);
    bool tie = found && abs(t - max(0, bestLocal.t)) <= departure.error;
    GBoundary candidate = g_boundaries[bi];
    GBoundary previous = g_boundaries[bestB];
    bool readout = g_optics[candidate.optics].a.x == kSurfDetector && candidate.front == candidate.back;
    bool previousReadout = found && g_optics[previous.optics].a.x == kSurfDetector && previous.front == previous.back;
    if (tie && readout != previousReadout) {
        if (readout) return;
        ambiguous = false;
    } else if (tie && bi != bestB) {
        // Two interfaces met at once are ordered by stable ID when either leaves the ray in the same
        // region (touching opaque faces, sheets with air on both sides); otherwise the order would
        // decide the region, and the path stops as ambiguous.
        uint outcome = contactOutcome(bi, lh.n, d, departure.region);
        if (outcome == kGpuNone || outcome != contactOutcome(bestB, bestLocal.n, d, departure.region)) ambiguous = true;
    }
    // Order candidates by distance, then physical matter before readout, then stable ID.
    // Repeated roots on one boundary keep the nearer root even inside the tie window.
    bool replace = !found;
    if (found && bi == bestB) replace = t < max(0, bestLocal.t);
    else if (found && !tie) replace = t < max(0, bestLocal.t);
    else if (found && readout != previousReadout) replace = previousReadout;
    else if (found) replace = bi < bestB;
    if (replace) {
        if (!tie) ambiguous = false;
        bestLocal = lh;
        bestB = bi;
        found = true;
    }
}

void fillContactHit(uint bestB, LocalHit bestLocal, float3 o, float3 d, bool ambiguous, out SurfaceHit hit) {
    hit = completeHit(bestB, bestLocal, o, d);
    hit.t = max(0, bestLocal.t);
    hit.ambiguousContact = ambiguous;
}

// The next float above t (the next candidate root after a rejected one).
float nextUp(float t) { return t == 0 ? asfloat(1u) : asfloat(asuint(t) + (t > 0 ? 1u : uint(-1))); }

// The departed mesh face supplies an exact native plane even when transforming the
// rounded world anchor back to this frame leaves a small normal displacement.
float3 meshDepartureAnchor(float3 lo, uint owner, uint departed, uint primitive) {
    if (owner != departed || primitive == kGpuNone) return lo;
    return pointOnTrianglePlane(g_triangles[primitive], lo);
}

float contactSceneSpan(float3 o) {
    GNode root = g_nodes[0];
    float3 delta = max(abs(root.lo.xyz - o), abs(root.hi.xyz - o));
    return min(1.7e38, maxAbsComponent(delta)) * 1.7320508;
}
float contactRayError(float3 o, float3 d, float scale, float3 normal) {
    float e = contactError(o, scale), c = abs(dot(d, normal));
    return c > 0 ? min(e / c, max(e, contactSceneSpan(o))) : e;
}
float spawnAdvance(float3 o, float3 normal, float3 d, float scale, float limit) {
    float c = abs(dot(d, normal));
    // Exactly tangent rays remain on their known anchor; no invented normal side is chosen.
    if (c == 0) return 0;
    return min(limit, min(surfaceEpsilon(o, scale) / c, contactSceneSpan(o)));
}

[noinline]
void contactMesh(GBoundary b, float3 o, float3 d, float window, uint departed, uint region, float error, float lag, float3 departingNormal, uint departedPrimitive, bool forwardOnly,
                 inout bool found, inout uint bestB, inout LocalHit bestLocal, inout bool ambiguous, uint onlyOwner = kGpuNone) {
    Endpoint departure = endpoint(o, region, 0, departed, departingNormal, departedPrimitive);
    departure.error = error;

    float3 lo = xfPoint(b.toLocal, o), ld = xfVector(b.toLocal, d);
    // A hardware triangle candidate belongs to the lagged world ray. Projection onto
    // the departed native plane changes that ray, so its complete native hierarchy
    // must supply membership and candidate coverage for this owner.
    if (onlyOwner != kGpuNone) lo = meshDepartureAnchor(lo, onlyOwner, departed, departedPrimitive);
    RaySlabs rs = RaySlabs(lo, ld);
    uint cur = b.meshNodes;
    uint rootTag = asuint(g_nodes[cur].hi.w);
    uint end = cur + ((rootTag & kNodeInner) != 0 ? (rootTag & ~kNodeInner) >> 2 : 1);
    // Nodes are contiguous preorder. An inner node carries its subtree span,
    // letting a miss skip that subtree without a per-ray software stack. This
    // preserves the former left-first candidate order, including contact ties.
    while (cur < end) {
        GNode node = g_nodes[cur];
        float enter;
        uint na = asuint(node.lo.w), nb = asuint(node.hi.w);
        if (rs.hit(node.lo.xyz, node.hi.xyz, -lag, window + lag, enter)) {
            if ((nb & kNodeInner) != 0) {
                cur = cur + 1;
                continue;
            }
            for (uint i = 0; i < nb; ++i) {
                GTriangle tri = g_triangles[na + i];
                uint owner = tri.owner;
                if (onlyOwner != kGpuNone && owner != onlyOwner) continue;
                if (onlyOwner == kGpuNone && owner == departed && departedPrimitive != kGpuNone) continue;
                float3 anchor = meshDepartureAnchor(lo, tri.owner, departed, departedPrimitive);
                float t;
                if (!hitTriangle(tri, anchor, ld, -lag, window + lag, t)) continue;
                LocalHit lh = (LocalHit)0;
                lh.t = t;
                lh.p = anchor + ld * t;
                lh.n = normalize(triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz));
                lh.prim = tri.primitive;
                lh.owner = tri.owner;
                lh.slot = na + i;
                considerContact(lh.owner, lh, d, departure, forwardOnly,
                                found, bestB, bestLocal, ambiguous);
            }
        }
        cur += (nb & kNodeInner) != 0 ? (nb & ~kNodeInner) >> 2 : 1;
    }
}


#ifdef OWE_RAY_QUERY
[noinline]
bool intersectContactNeighborhood(float3 o, float3 d, float window, uint departed, uint region,
                                  float scale, float3 departingNormal, uint departedPrimitive, bool forwardOnly, out SurfaceHit hit) {
    hit = (SurfaceHit)0;
    if (globals().counts.x == 0) return false;
    const float error = contactError(o, scale);

    Endpoint departure = endpoint(o, region, 0, departed, departingNormal, departedPrimitive);
    departure.error = error;
    const float lag = contactRayError(o, d, scale, departingNormal);
    bool found = false, ambiguous = false;
    uint bestB = 0;
    LocalHit bestLocal = (LocalHit)0;
    bool nativeDeparture = departed != kGpuNone && departedPrimitive != kGpuNone &&
                           g_boundaries[departed].shape == kShapeMesh;
    if (nativeDeparture)
        contactMesh(g_boundaries[departed], o, d, window, departed, region, error, lag,
                    departingNormal, departedPrimitive, forwardOnly,
                    found, bestB, bestLocal, ambiguous, departed);
    // This finite neighborhood is deliberately not committed/pruned: all simultaneous
    // incident interfaces must be examined. Ordinary distant traversal remains unchanged.
    // Start hardware traversal after the native owner walk so its traversal state is
    // not live across the software BVH walk and its nested calls.
    RayDesc ray = makeRay(o - d * lag, d, window + 2 * lag);
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(g_scene, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);
    while (query.Proceed()) {
        uint instance = query.CandidateInstanceID();
        GBoundary b = g_boundaries[instance];
        LocalHit lh = (LocalHit)0;
        lh.owner = kGpuNone;
        uint bi = instance;
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            if (!intersectBoundary(b, o, d, -lag, window + lag, lh)) continue;
            if (!contactAccepted(bi, lh, d, departure, forwardOnly)) {
                if (!intersectBoundary(b, o, d, nextUp(lh.t), window + error, lh)) continue;
            }
        } else {
            uint slot = b.meshTris + query.CandidatePrimitiveIndex();
            GTriangle tri = g_triangles[slot];
            bi = tri.owner;
            if (nativeDeparture && bi == departed) continue;
            float3 lo = meshDepartureAnchor(xfPoint(b.toLocal, o), bi, departed, departedPrimitive);
            float3 ld = xfVector(b.toLocal, d);
            // Distinct owners retain the unmodified native anchor. Hardware established
            // membership on the lagged query; do not duplicate a strict edge test here.
            // The departed owner is queried completely in its corrected native frame above.
            float3 np = triangleNormal(tri.v0.xyz, tri.v1.xyz, tri.v2.xyz);
            float denom = dot(ld, np);
            if (denom == 0) continue;
            float t = -dot(lo - tri.v0.xyz, np) / denom;
            if (!(t > -lag && t < window + lag)) continue;
            lh.t = t;
            lh.p = lo + ld * t;
            lh.n = normalize(np);
            lh.prim = tri.primitive;
            lh.slot = slot;
        }
        considerContact(bi, lh, d, departure, forwardOnly, found, bestB, bestLocal, ambiguous);
    }
    if (found) fillContactHit(bestB, bestLocal, o, d, ambiguous, hit);
    return found;
}
#else

[noinline]
bool intersectContactNeighborhood(float3 o, float3 d, float window, uint departed, uint region,
                                  float scale, float3 departingNormal, uint departedPrimitive, bool forwardOnly, out SurfaceHit hit) {
    hit = (SurfaceHit)0;
    if (globals().counts.x == 0) return false;
    const float error = contactError(o, scale);

    Endpoint departure = endpoint(o, region, 0, departed, departingNormal, departedPrimitive);
    departure.error = error;
    const float lag = contactRayError(o, d, scale, departingNormal);
    RaySlabs rs = RaySlabs(o, d);
    uint stackNode[kStack];
    int sp = 0;
    uint cur = 0;
    bool found = false, ambiguous = false;
    uint bestB = 0;
    LocalHit bestLocal = (LocalHit)0;
    bool nativeDeparture = departed != kGpuNone && departedPrimitive != kGpuNone &&
                           g_boundaries[departed].shape == kShapeMesh;
    if (nativeDeparture)
        contactMesh(g_boundaries[departed], o, d, window, departed, region, error, lag,
                    departingNormal, departedPrimitive, forwardOnly,
                    found, bestB, bestLocal, ambiguous, departed);
    while (true) {
        GNode node = g_nodes[cur];
        float enter;
        uint na = asuint(node.lo.w), nb = asuint(node.hi.w);
        if (rs.hit(node.lo.xyz, node.hi.xyz, -lag, window + lag, enter)) {
            if ((nb & kNodeInner) != 0) {
                if (sp < kStack) stackNode[sp++] = na;
                cur = cur + 1;
                continue;
            }
            for (uint i = 0; i < nb; ++i) {
                uint bi = g_top[na + i];
                GBoundary b = g_boundaries[bi];
                if (!rs.hit(b.boxLo.xyz, b.boxHi.xyz, -lag, window + lag, enter)) continue;
                if (b.shape == kShapeMesh) {
                    contactMesh(b, o, d, window, departed, region, error, lag, departingNormal, departedPrimitive, forwardOnly,
                                found, bestB, bestLocal, ambiguous);
                    continue;
                }
                LocalHit lh = (LocalHit)0;
                lh.owner = kGpuNone;
                if (!intersectBoundary(b, o, d, -lag, window + lag, lh)) continue;
                if (!contactAccepted(bi, lh, d, departure, forwardOnly)) {
                    if (!intersectBoundary(b, o, d, nextUp(lh.t), window + error, lh)) continue;
                }
                considerContact(bi, lh, d, departure, forwardOnly,
                                found, bestB, bestLocal, ambiguous);
            }
        }
        if (sp == 0) break;
        cur = stackNode[--sp];
    }
    if (found) fillContactHit(bestB, bestLocal, o, d, ambiguous, hit);
    return found;
}

#endif

// The nearest accepted crossing of one mesh owner within window of a spawn anchor, walked in
// the owner's native frame (whose coordinates are small) instead of the world's.
[noinline]
bool intersectOwnerNative(uint owner, float3 o, float3 d, float window, uint departed, uint region, float scale,
                          float3 departingNormal, uint departedPrimitive, bool forwardOnly, out SurfaceHit hit) {
    hit = (SurfaceHit)0;
    bool found = false, ambiguous = false;
    uint bestB = 0;
    LocalHit bestLocal = (LocalHit)0;
    contactMesh(g_boundaries[owner], o, d, window, departed, region, contactError(o, scale),
                contactRayError(o, d, scale, departingNormal), departingNormal, departedPrimitive, forwardOnly,
                found, bestB, bestLocal, ambiguous, owner);
    if (found) fillContactHit(bestB, bestLocal, o, d, ambiguous, hit);
    return found;
}

// Invoked with the surface anchor, outgoing side's region, and the departing boundary only.
// Its opposite face and every distinct contacting boundary remain eligible.
bool intersectSpawnWorld(float3 o, float3 d, float tmax, uint departed, uint region,
                         float3 normal, float scale, uint departedPrimitive, out SurfaceHit hit, bool forwardOnly = false,
                         float certifiedAdvance = -1) {
    if (departed == kGpuNone) {
        // A primary/non-surface ray uses the original traversal directly, including its
        // infinite bound. Contact ordering below is needed only for a readout or mismatch.
        if (!intersectWorld(o, d, tmax, hit)) return false;
    } else {
        // A caller may reuse an empty contact interval proved for this exact anchor,
        // direction and region. Keep the full distance and the hit-order repair below.
        float advance = certifiedAdvance >= 0 ? certifiedAdvance : spawnAdvance(o, normal, d, scale, tmax);
        float3 shifted = o + d * advance;
        if (certifiedAdvance < 0 && intersectContactNeighborhood(o, d, advance, departed, region, scale, normal, departedPrimitive, forwardOnly, hit)) return true;
        float limit = isinf(tmax) ? tmax : max(0, tmax - advance);
        if (!intersectWorld(shifted, d, limit, hit)) return false;
        hit.t += advance;
    }
    GBoundary b = g_boundaries[hit.boundary];
    uint incident = dot(d, hit.n) < 0 ? b.front : b.back;
    // Past the spawn offset the traversal starts from a world point rounded to the world's float
    // spacing, which can lie behind a face the ray only grazes (a concave fold of the face it
    // left), so the ray arrives inside that mesh. Its own frame resolves the whole segment.
    if (incident != region && departed != kGpuNone && b.shape == kShapeMesh) {
        SurfaceHit native;
        if (intersectOwnerNative(hit.boundary, o, d, hit.t, departed, region, scale, normal, departedPrimitive,
                                 forwardOnly, native)) {
            GBoundary nb = g_boundaries[native.boundary];
            if ((dot(d, native.n) < 0 ? nb.front : nb.back) == region && native.t < hit.t) {
                hit = native;
                b = nb;
                incident = region;
            }
        }
    }
    if (incident != region || g_optics[b.optics].a.x == kSurfDetector) {
        SurfaceHit ordered;
        // A raw primary hit may select the entering opaque neighbor before the glass exit.
        // Repair only unresolved local ordering, never a resolved overlap or interior launch.
        if (intersectContactNeighborhood(hit.p, d, contactError(hit.p, hit.scale),
                                         kGpuNone, region, hit.scale, float3(0), kGpuNone, false, ordered)) {
            GBoundary candidate = g_boundaries[ordered.boundary];
            uint ci = dot(d, ordered.n) < 0 ? candidate.front : candidate.back;
            if (ci == region) {
                ordered.t += hit.t;
                hit = ordered;
            }
        }
    }
    return true;
}
