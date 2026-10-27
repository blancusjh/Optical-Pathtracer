// Contact resolution: the next crossing of a ray that leaves a surface (docs/TRANSPORT_CONTACTS.md).
//
// One rule: a ray leaving an anchor in region R meets next the nearest boundary crossing met from
// R's side. Float makes three parts of it explicit.
//  - Near the anchor a crossing's side cannot be told from its distance (the anchor carries its
//    frame's rounding error), so there the region decides: a candidate counts only if it is met from
//    R, and the face the ray left never does (contactAccepted). Every candidate within the spawn
//    offset is examined, each in its own frame (intersectContactNeighborhood).
//  - Beyond the offset the ordinary nearest-hit traversal runs (intersectWorld).
//  - That hit, found in the world frame, may be met from another region: the ray grazed a concave
//    fold, or chose one of two tied faces. It is then found again where float is exact (resolveFarHit).
// Crossings at one distance are ordered: matter before a readout, then stable ID when the order
// cannot change the ray's region; otherwise the contact is ambiguous and the path stops.
#pragma once
#include "world.slang"

// ---------------------------------------------------------------- the error model

// The spawn offset: a few hundred ulps of the larger of the point's and its frame's coordinates,
// far below optical feature sizes (40 nm for an eyepiece lens 2 cm from the eye, 20 µm at 10 m).
float surfaceEpsilon(float3 p, float scale) { return 2e-6 * max(maxAbsComponent(p), scale) + 1e-9; }

// The anchor's rounding error, along the normal of the face it is on.
float contactError(float3 p, float scale) {
    return 8 * 1.1920928955078125e-7 * max(1e-12, max(maxAbsComponent(p), scale));
}
float contactSceneSpan(float3 o) {
    GNode root = g_nodes[0];
    float3 delta = max(abs(root.lo.xyz - o), abs(root.hi.xyz - o));
    return min(1.7e38, maxAbsComponent(delta)) * 1.7320508;
}
// That error carried along the ray: how far before the anchor a candidate may still lie.
float contactRayError(float3 o, float3 d, float scale, float3 normal) {
    float e = contactError(o, scale), c = abs(dot(d, normal));
    return c > 0 ? min(e / c, max(e, contactSceneSpan(o))) : e;
}
// How far along the ray the neighbourhood is examined before ordinary traversal takes over.
float spawnAdvance(float3 o, float3 normal, float3 d, float scale, float limit) {
    float c = abs(dot(d, normal));
    // Exactly tangent rays remain on their known anchor; no invented normal side is chosen.
    if (c == 0) return 0;
    return min(limit, min(surfaceEpsilon(o, scale) / c, contactSceneSpan(o)));
}

// The anchor a query leaves from: the point, its region, the face it left and its rounding error.
Endpoint departureFrom(float3 o, uint region, float scale, uint departed, float3 normal, uint primitive) {
    Endpoint e = endpoint(o, region, scale, departed, normal, primitive);
    e.error = contactError(o, scale);
    return e;
}

// ---------------------------------------------------------------- which crossings count, in what order

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

// Is crossing lh of boundary bi a candidate for a ray leaving `departure` along d? Not the departed
// face itself; beyond the anchor's error any crossing ahead; within it, only one met from the region.
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

// The candidates of one query so far: the nearest accepted crossing, and whether another at its
// distance leaves the order undecided.
struct ContactSet {
    bool found, ambiguous;
    uint boundary;
    LocalHit nearest;
};

bool isReadout(GBoundary b) { return g_optics[b.optics].a.x == kSurfDetector && b.front == b.back; }

void considerContact(uint bi, LocalHit lh, float3 d, Endpoint departure, bool forwardOnly, inout ContactSet s) {
    if (!contactAccepted(bi, lh, d, departure, forwardOnly)) return;
    float t = max(0, lh.t), nearest = max(0, s.nearest.t);
    bool tie = s.found && abs(t - nearest) <= departure.error;
    bool readout = isReadout(g_boundaries[bi]);
    bool previousReadout = s.found && isReadout(g_boundaries[s.boundary]);
    if (tie && readout != previousReadout) {
        if (readout) return;
        s.ambiguous = false;
    } else if (tie && bi != s.boundary) {
        // Ordered by stable ID when either interface leaves the ray in the same region (touching
        // opaque faces, sheets with air on both sides); else the order would decide the region.
        uint outcome = contactOutcome(bi, lh.n, d, departure.region);
        if (outcome == kGpuNone || outcome != contactOutcome(s.boundary, s.nearest.n, d, departure.region))
            s.ambiguous = true;
    }
    // Order candidates by distance, then physical matter before readout, then stable ID.
    // Repeated roots on one boundary keep the nearer root even inside the tie window.
    bool replace = !s.found;
    if (s.found && (bi == s.boundary || !tie)) replace = t < nearest;
    else if (s.found && readout != previousReadout) replace = previousReadout;
    else if (s.found) replace = bi < s.boundary;
    if (replace) {
        if (!tie) s.ambiguous = false;
        s.nearest = lh;
        s.boundary = bi;
        s.found = true;
    }
}

bool contactHit(ContactSet s, float3 o, float3 d, out SurfaceHit hit) {
    hit = (SurfaceHit)0;
    if (!s.found) return false;
    hit = completeHit(s.boundary, s.nearest, o, d);
    hit.t = max(0, s.nearest.t);
    hit.ambiguousContact = s.ambiguous;
    return true;
}

// ---------------------------------------------------------------- the neighbourhood of the anchor

// The next float above t (the next candidate root after a rejected one).
float nextUp(float t) { return t == 0 ? asfloat(1u) : asfloat(asuint(t) + (t > 0 ? 1u : uint(-1))); }

// An analytic boundary's candidates within the window; a rejected nearer root yields the next one.
void considerAnalytic(uint bi, float3 d, float window, float lag, Endpoint departure, bool forwardOnly, inout ContactSet s) {
    GBoundary b = g_boundaries[bi];
    LocalHit lh = (LocalHit)0;
    lh.owner = kGpuNone;
    if (!intersectBoundary(b, departure.p, d, -lag, window + lag, lh)) return;
    if (!contactAccepted(bi, lh, d, departure, forwardOnly) &&
        !intersectBoundary(b, departure.p, d, nextUp(lh.t), window + departure.error, lh)) return;
    considerContact(bi, lh, d, departure, forwardOnly, s);
}

// The departed mesh face supplies an exact native plane even when transforming the
// rounded world anchor back to this frame leaves a small normal displacement.
float3 meshDepartureAnchor(float3 lo, uint owner, uint departed, uint primitive) {
    if (owner != departed || primitive == kGpuNone) return lo;
    return pointOnTrianglePlane(g_triangles[primitive], lo);
}
bool departsMesh(Endpoint departure) {
    return departure.boundary != kGpuNone && departure.primitive != kGpuNone &&
           g_boundaries[departure.boundary].shape == kShapeMesh;
}

// The candidates of one mesh group within the window, in its native frame: every owner but the
// departed one, or (onlyOwner) that owner alone with the anchor on its departed face's plane.
[noinline]
void contactMesh(GBoundary b, float3 d, float window, float lag, Endpoint departure, bool forwardOnly,
                 inout ContactSet s, uint onlyOwner = kGpuNone) {
    float3 lo = xfPoint(b.toLocal, departure.p), ld = xfVector(b.toLocal, d);
    // A hardware triangle candidate belongs to the lagged world ray. Projection onto
    // the departed native plane changes that ray, so its complete native hierarchy
    // must supply membership and candidate coverage for this owner.
    if (onlyOwner != kGpuNone) lo = meshDepartureAnchor(lo, onlyOwner, departure.boundary, departure.primitive);
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
                if (onlyOwner == kGpuNone && owner == departure.boundary && departure.primitive != kGpuNone) continue;
                float3 anchor = meshDepartureAnchor(lo, owner, departure.boundary, departure.primitive);
                float t;
                if (!hitTriangle(tri, anchor, ld, -lag, window + lag, t)) continue;
                considerContact(owner, triangleHit(tri, na + i, anchor, ld, t), d, departure, forwardOnly, s);
            }
        }
        cur += (nb & kNodeInner) != 0 ? (nb & ~kNodeInner) >> 2 : 1;
    }
}

// The nearest accepted crossing within `window` of the anchor o: every candidate examined (none
// pruned, since all simultaneous interfaces must be seen), each in its own frame.
[noinline]
bool intersectContactNeighborhood(float3 o, float3 d, float window, uint departed, uint region,
                                  float scale, float3 departingNormal, uint departedPrimitive, bool forwardOnly, out SurfaceHit hit) {
    hit = (SurfaceHit)0;
    if (globals().counts.x == 0) return false;
    const Endpoint departure = departureFrom(o, region, scale, departed, departingNormal, departedPrimitive);
    const float lag = contactRayError(o, d, scale, departingNormal);
    ContactSet s = (ContactSet)0;
    // The departed mesh, whole, in its corrected native frame first; traversal then skips it.
    const bool nativeDeparture = departsMesh(departure);
    if (nativeDeparture) contactMesh(g_boundaries[departed], d, window, lag, departure, forwardOnly, s, departed);
#ifdef OWE_RAY_QUERY
    // Hardware traversal starts after the native owner walk, so its state is not live across it.
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(g_scene, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, makeRay(o - d * lag, d, window + 2 * lag));
    while (query.Proceed()) {
        uint instance = query.CandidateInstanceID();
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            considerAnalytic(instance, d, window, lag, departure, forwardOnly, s);
            continue;
        }
        GBoundary b = g_boundaries[instance];
        uint slot = b.meshTris + query.CandidatePrimitiveIndex();
        GTriangle tri = g_triangles[slot];
        if (nativeDeparture && tri.owner == departed) continue;
        // Hardware established membership on the lagged ray; distinct owners keep the native anchor.
        float3 lo = xfPoint(b.toLocal, o), ld = xfVector(b.toLocal, d);
        float t;
        if (!trianglePlaneDistance(tri, lo, ld, t) || !(t > -lag && t < window + lag)) continue;
        considerContact(tri.owner, triangleHit(tri, slot, lo, ld, t), d, departure, forwardOnly, s);
    }
#else
    RaySlabs rs = RaySlabs(o, d);
    uint stackNode[kStack];
    int sp = 0;
    uint cur = 0;
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
                if (b.shape == kShapeMesh) contactMesh(b, d, window, lag, departure, forwardOnly, s);
                else considerAnalytic(bi, d, window, lag, departure, forwardOnly, s);
            }
        }
        if (sp == 0) break;
        cur = stackNode[--sp];
    }
#endif
    return contactHit(s, o, d, hit);
}

// ---------------------------------------------------------------- the far hit

// One mesh owner's nearest accepted crossing within `window` of the anchor, in its native frame.
[noinline]
bool intersectOwnerNative(uint owner, float3 o, float3 d, float window, uint departed, uint region, float scale,
                          float3 departingNormal, uint departedPrimitive, bool forwardOnly, out SurfaceHit hit) {
    const Endpoint departure = departureFrom(o, region, scale, departed, departingNormal, departedPrimitive);
    ContactSet s = (ContactSet)0;
    contactMesh(g_boundaries[owner], d, window, contactRayError(o, d, scale, departingNormal), departure,
                forwardOnly, s, owner);
    return contactHit(s, o, d, hit);
}

// A hit found by world-frame traversal, met from another region than the ray's (or on a readout),
// found again where float is exact; only a crossing met from the ray's region replaces it.
//  - Its own mesh over the whole segment, natively: past the spawn offset the traversal starts from
//    a world point rounded to the world's float spacing, which can lie behind a face the ray only
//    grazes (a concave fold of the face it left), so the ray arrives inside that mesh.
//  - Then the candidates within the hit's own error: a raw hit may select the entering opaque
//    neighbour before the glass exit. Only unresolved local order is repaired, never a resolved
//    overlap or an interior launch.
void resolveFarHit(float3 o, float3 d, uint departed, uint region, float3 normal, float scale,
                   uint departedPrimitive, bool forwardOnly, inout SurfaceHit hit) {
    GBoundary b = g_boundaries[hit.boundary];
    if (incidentRegion(hit, d) != region && departed != kGpuNone && b.shape == kShapeMesh) {
        SurfaceHit native;
        if (intersectOwnerNative(hit.boundary, o, d, hit.t, departed, region, scale, normal, departedPrimitive,
                                 forwardOnly, native) &&
            incidentRegion(native, d) == region && native.t < hit.t) {
            hit = native;
            b = g_boundaries[hit.boundary];
        }
    }
    if (incidentRegion(hit, d) == region && g_optics[b.optics].a.x != kSurfDetector) return;
    SurfaceHit ordered;
    if (intersectContactNeighborhood(hit.p, d, contactError(hit.p, hit.scale), kGpuNone, region, hit.scale,
                                     float3(0), kGpuNone, false, ordered) &&
        incidentRegion(ordered, d) == region) {
        ordered.t += hit.t;
        hit = ordered;
    }
}

// The next crossing of a ray leaving anchor o in `region`: from boundary `departed` (its opposite
// face and every distinct contacting boundary remain eligible), or from no surface (the eye, a
// volume point), which needs no neighbourhood.
bool intersectSpawnWorld(float3 o, float3 d, float tmax, uint departed, uint region,
                         float3 normal, float scale, uint departedPrimitive, out SurfaceHit hit, bool forwardOnly = false,
                         float certifiedAdvance = -1) {
    if (departed == kGpuNone) {
        if (!intersectWorld(o, d, tmax, hit)) return false;
    } else {
        // A caller may reuse an empty contact interval proved for this exact anchor,
        // direction and region. Keep the full distance and the far-hit resolution below.
        float advance = certifiedAdvance >= 0 ? certifiedAdvance : spawnAdvance(o, normal, d, scale, tmax);
        if (certifiedAdvance < 0 && intersectContactNeighborhood(o, d, advance, departed, region, scale, normal, departedPrimitive, forwardOnly, hit)) return true;
        float limit = isinf(tmax) ? tmax : max(0, tmax - advance);
        if (!intersectWorld(o + d * advance, d, limit, hit)) return false;
        hit.t += advance;
    }
    resolveFarHit(o, d, departed, region, normal, scale, departedPrimitive, forwardOnly, hit);
    return true;
}
