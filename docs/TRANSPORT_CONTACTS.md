# Surface contacts and finite visibility

The transport invariant is that a numeric ray offset must not skip a distinct
physical interface. A glass surface can touch a support, a detector can have an
opaque cover, and two sides of a thin layer can lie closer than a conventional
ray epsilon. Suppressing all nearby intersections changes the measured light.

## Departure from a surface

Spawned rays retain the exact surface anchor, incident region, geometric normal
and departed boundary/primitive. CPU rays also carry intersection uncertainty.
Only the departed face, including a coplanar mesh facet sharing its edge, can be
excluded as a self-hit. Distinct parallel faces and opposite sides of the same
analytic body remain eligible. Mesh identity must refer to the actual triangle
slot when GPU BVH grouping has reordered triangles.

CPU traversal orders hits in the anchor's uncertainty interval using the known
region. A grazing self-root is tested by its displacement along the surface
normal, rather than by ray distance alone. Numerical surface solvers contribute
their residual to the anchor uncertainty and must continue to later roots after
an excluded root.

GPU traversal first examines a bounded neighborhood of the exact anchor. Once
that interval is proven clear, the ordinary offset and accelerated traversal can
be used for the rest of the ray. Hardware ray queries must inspect all contact
candidates before pruning; software traversal applies the same selection rules.
The offset follows the ray, including at grazing incidence: a normal displacement
would leave the interval just checked and could cross an untested thin layer.
Its length is bounded by the scene and finite connection span. Candidate contact
uncertainty is still measured along the surface normal, rather than widened to
that entire grazing-ray interval. Distinct faces of one mesh retain their actual
nearest-positive-hit order.
The near-surface planetary sphere representation remains analytic and retains
small clearances without subtracting two planetary-scale squared lengths.

An authored sensor plane starts a forward-only query. Matter at exact contact
still covers it, but a resolved positive gap cannot pull matter behind the plane
into the outgoing ray. Known mesh departures are reconstructed on their native
triangle plane; their measured residual contributes to the carried uncertainty.

Projecting a departure onto its native plane changes the ray used to intersect
that mesh. A triangle accepted by a hardware query on the earlier ray cannot
establish membership on the corrected ray. The departing mesh therefore gets
a complete native BVH query with the corrected anchor, before world or group
pruning. Its owner record points to the existing shared group root; traversal
filters triangles by owner, without adding another hierarchy. Hardware queries
exclude that owner after the native query and retain their membership checks for
distinct owners. Software traversal now performs the same owner-first query.
Checking only a previously reported triangle could miss another valid face, so
the correction covers the complete owner rather than widening edge tolerances.

The current performance implementation preserves that coverage while reducing
query state and repeated work:

- `transmittanceSpawn` saves an empty first contact interval and passes its
  `certifiedAdvance` to the first `intersectSpawnWorld` call. The remaining query
  includes the finite endpoint error band; hit distances regain the saved advance
  before attenuation and endpoint recognition. Pending passable contacts use the
  saved hit, and subsequent null/stained-glass crossings require fresh proofs.
- `contactMesh` walks contiguous preorder nodes without a private traversal
  stack. Inner-node `b` retains `kNodeInner` and the low two axis bits, with the
  subtree node count in bits 2–30. A miss advances by that count; an inner hit
  advances to its left child. Leaf counts are unchanged. Relative spans survive
  scene rebasing and preserve the previous left-first contact candidate order.
- Hardware contact traversal starts its ray query after the native owner walk,
  reducing how long traversal state remains live across nested calls.
- `RaySlabs::hit` widens each far bound toward positive infinity on either side
  of zero. Negative contact distances therefore retain conservative coverage;
  multiplying a negative far bound by a factor above one would narrow it.
- `intersectBoundary` shares the complete analytic solver through a `noinline`
  call, including planetary sphere behavior. The small `intersectSpawnWorld`
  wrapper permits inlining; its traversal and solver children remain isolated.

Triangles are crossed by the watertight test (Woop, Benthin and Wald 2013,
`lineCrossesTriangle` in `core/shared/surfaces.hpp`, both backends): each edge's
side is decided from that edge's own two vertices, so faces sharing an edge agree
on it and a long sliver is resolved near its far edge. Möller–Trumbore's u + v ≤ 1
cancels there in float: the Invisible Window panel's fan-triangulated back let
rays entering a side face micrometres from the rim escape from inside the glass.

Two distinct, simultaneously incident physical interfaces are ambiguous if the
region does not establish their order. Keep the diagnostic and stop that path.
A zero-thickness detector readout cannot take precedence over coincident opaque
matter. Real scene overlaps need scene repairs: the observatory cabinet shelves,
for example, originally overlapped their stiles and duplicated back faces with
different wood materials. Their recipe now fits the shelves between the stiles.

## Connections to a known endpoint

A distance-scaled endpoint margin can hide a blocker near a distant light or
observer. Finite connections therefore need both endpoint positions and surface
identities. The endpoint-aware walk excludes the intended endpoint's own face,
checks neighboring matter, verifies the arrival region, and caps attenuation at
the physical connection length. Matter beyond the endpoint must not block it.
Infinite sky and Sun connections have no finite endpoint to exclude.

Detector connections, finite next-event light samples and VCM cached-vertex
connections use this walk. Cached GPU vertices preserve primitive identity in
the upper 30 bits of their flag word, without changing the low flags or record size.

Both CPU and GPU endpoint-face recognition require an unambiguous hit before
excluding the destination face. A coincident ambiguous interface retains the
contact failure even when its boundary matches the endpoint.

GPU departure filtering and finite endpoint exclusion share `sameMeshFace` in
`world.slang`. It recognizes the same original triangle across sampling and BVH
record slots, or adjacent triangles with an exact shared edge, coplanarity and
matching orientation. Both records must belong to the same boundary. Distinct
parallel layers and reversed facets remain separate faces; no distance tolerance
collapses them into an endpoint.

The fixtures under `benchmarks/regressions/` and the geometry, GPU and
`test_far_visibility.cpp` tests cover exact contacts, resolved gaps, thin layers,
grazing roots, invalid overlaps, detector covers, and blockers at either end of
long connections. Each dark-image oracle needs an illuminated open control.
Raw images and diagnostics are both necessary: a sky-only render that misses all
geometry can misleadingly report zero inconsistencies.

Current validation and remaining production diagnostics are recorded in
[STATUS.md](STATUS.md). Earlier implementation experiments, measurements and
validation limits are retained in the [contact archive](archive/2026-10/TRANSPORT_CONTACTS.md).

## API and uncertainty

CPU and GPU visibility calls carry an `Endpoint` containing position, region,
boundary, geometric normal, primitive identity and intersection uncertainty.
A missing boundary means an authored point rather than a known surface. Callers
construct records from intersections or light samples instead of passing a list
of unrelated sentinel values.

Each backend uses one `contactError` helper for its arithmetic precision.
The CPU shape interface supplies face identity directly; visibility traversal
never downcasts a shape to discover whether it is a mesh. A distinct parallel
face, including another face in the same mesh, remains eligible for intersection.
