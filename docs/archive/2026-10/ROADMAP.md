Historical record, retained for provenance. Current state: [STATUS.md](../../STATUS.md)

# Roadmap

Updated 2026-10-10 against the implementation and the September–October project discussions.
New transport work targets the GPU; the CPU remains a reference for the physics it implements.

## Direction and next milestone

Build an explorable optical world in which images emerge from matter and light transport.
Preserve Medium → Region → Boundary → Body → Assembly, analytic optical surfaces, and the
separation of physics, estimators, backends, measurements and display. A saved instrument view
only positions the observer. Every flagship needs both independent optical validation and a
convincing, converged appearance; CPU/GPU agreement alone is not physical proof.

The gallery and engine milestone advances **fast, verified, beautiful optical worlds**:

1. Preserve the validated contact, finite-connection, VCM recovery and planetary-precision
   checkpoint; investigate remaining production-scene diagnostics without hiding them.
2. Use the completed corrected twelve-view captures and independent-seed checks to guide
   contact-traversal profiling. References remain noisy; longer captures alone are not acceptance.
3. Deliver bounded, measured engine improvements with final validation; pursue the full
   4× equal-quality target as follow-up work.
4. The four curated scan scenes now have seven locally promoted production-resolution
   views, with composition, lighting and material reviews, metadata and credits.

The completed release unit is the reviewed seven-view gallery plus tested engine improvements.
The 4× target remains follow-up work. Reuse of a verified first contact interval and
stack-free native mesh traversal pass final validation, with bounded measured gains
recorded in [PERFORMANCE_STATUS.md](PERFORMANCE_STATUS.md). All seven gallery views are
reviewed and locally promoted with verified archives, finite raw images and sidecars;
see the [portable gallery record](../../../benchmarks/results/20261010_gallery.json). The earlier seven benchmark sweeps remain
[historical evidence](../../../benchmarks/results/20261009_corrected_checkpoint.json), including
the regression and unresolved production diagnostics.

See [NEXT_ITERATION.md](NEXT_ITERATION.md) for the measurement contract and current status.
The [full twelve-case baseline](BENCHMARK_BASELINE_20261008.md) is captured at 16/64 spp with
256-spp references; its noise and transport diagnostics prevent a converged equal-quality claim.
The [performance measurements](PERFORMANCE_STATUS.md) identify the VCM camera kernel as the
dominant Peacock Lamp cost. Its later CIE cache measured 3.65% less transport time in an
unprofiled paired 64-spp capture. The requested full-suite 4× result remains unproved.
An isolated v1.0 CLI is built and its eight canonical inputs pass the compatibility audit.
Historical scan map/material support is incompatible with the current workload; those cases
cannot supply an equal-quality historical speedup. See the
[scene and historical audit](SCENE_READINESS_20261008.md).
Design desired scenes first and let their physical requirements define missing interfaces.
The inverse-caustic solver is an authoring tool: solve once and render its resulting glass geometry.

Follow-up capability work prioritizes difficult surface paths (manifold NEE and physical-camera
BDPT/VCM) or volumetric VCM according to the next flagship's measured limitations. Then extend
heterogeneous media, instruments and coherent sky illumination. Longer-range work includes
GRIN/mirages, polarization, atmospheric scattering, interactive ray inspection and optical
design, transient measurements and multi-GPU scheduling. Diffraction and interference remain
outside the agreed scope. The numbered engine list below is a capability backlog, not an
instruction to implement every feature before measuring performance.

## Done in this version

- **VCM cache recovery:** an overflowing pass is discarded, the cache grows, and the same
  sampling schedule is replayed before accumulation. Allocation rejection preserves a usable
  renderer. Forced-overflow and recovery regressions cover hardware and software traversal.
- **Performance measurement:** recursive scene/asset identities, frozen references, raw-image
  error comparisons, wall/transport timing, host RSS and optional GPU kernel timestamps.
  All twelve fixed production cases completed the initial frozen baseline capture.
  The camera-side diffuse-response and CIE caches have bounded measured benefits on the lamp;
  see [PERFORMANCE_STATUS.md](PERFORMANCE_STATUS.md) for its scope and limitations.
- **Planetary surface precision:** compensated CPU sphere intersections and GPU pole-relative
  intersections/reconstruction preserve small surface clearances, including hits from distant
   rays. The sphere stays analytic and complete. Integrated targeted contact and endpoint
   checks pass on hardware and software traversal; see the [regressions](../../../benchmarks/regressions/README.md).
- **Pipeline caching.** Kernels compile through a cache kept on disk (`~/.cache/owe/`, per
  device and driver), and a renderer compiles only its integrator's kernels, in parallel.
  Historical v1.0 captures reduced a cold VCM start from 131 s to 61 s and reported 0.4 s
  for later starts. These are earlier measurements; current contact shaders and driver
  compilation costs need separate measurement.
- **`owe model <file>`:** a model's parts, sizes and textures, with the paths a scene writes to
  load them.
- **Stained glass** (`type = stained_glass`): a thin slab with Fresnel at both faces and
  internal transmittance from a picture.
  - Covers opalescent `haze`, diffuse reflection, and glass masks that hand masked points to an
    opaque stone material.
  - Shadow rays cross it, so sunlight through a window is direct lighting.
  - Specification: `docs/scenes/peacock_lamp.txt`.
- **Mirrors in lens prescriptions:** coaxial `reflect` rows with a bore and spider struts.
  Paraxial data come from the unfolded system, and `afocal` adjusts the gap marked `solve`.
  Specification: `docs/scenes/gregorian.txt`.
- **VCM and BDPT with scattering media,** by partition:
  - the path tracer takes the paths that scatter in a medium;
  - VCM or BDPT takes the rest, through media as extinction.
- **VCM wavelength groups:** `vcm_groups` selects one to eight groups; adaptive spectral
  sampling and photon-radius improvements remain separate work.
- **`surface = null`** for index-matched media: haze or smoke in air whose boundary only changes
  the region.
- **Scenes:**
  - the Kunstkammer, the Gregorian, the Peacock lamp and the invisible window, all
    built on museum scans;
  - an 85 mm physical camera in the study.
- **Texture memory:** byte-backed LDR images, a configurable 2048 MB default texture budget,
  sequential decode/reduction and staged GPU upload. October 7 scene records report Kunstkammer
  at 960×600 using about 4.4 GB peak host memory; this is historical evidence, not a new benchmark.
- **Material maps:** GPU image normal and roughness maps, UV tangent frames and planar/triplanar
  mapping; glTF inspection exposes the corresponding image paths. Curated Poly Haven assets
  and scene revisions already exist locally.

## Engine: what remains

At the earlier corrected checkpoint on 9 October, CPU-only contact/finite-connection
coverage passed 157/157 registered tests
(GPU transport unavailable and skipped). The earlier full hardware run passed 156/157;
the remaining smooth-glass failure was traced to a mesh-owner query using the wrong contact
anchor. A complete query of that owner at the corrected native anchor passes the unchanged
smooth-glass assertion and five focused controls. That checkpoint passed 25/25 targeted hardware checks and 21/21 software checks. The bounded software SPPM smoke completes
16 passes with zero inconsistencies or leaks. The long software smooth-glass diagnostic was terminated without a conformance
result; the smoke does not replace that full assertion. The corrected executable and source
record were [frozen](../../../benchmarks/evidence/performance_20261008/corrected_final_manifest_20261009.json) for
the corrected checkpoint benchmark references. These counts do not validate the later
contact performance changes; current milestone validation is recorded separately. See
[the contact design](../../TRANSPORT_CONTACTS.md) for the current rules and remaining migration.

1. **Loading:**
   - Smaller kernels. Find what inflates them (unrolled hero-wavelength loops, BSDF and texture
     switches inlined at every call site), then try `[noinline]` and specialization constants.
   - Scene-side caching: write generated meshes and mesh-group BVHs to disk keyed by content
     hash. The Kunstkammer's scans take ~14 s to load.
2. **Stained glass, later parts:**
   - roughness of the reflection;
   - a forward-scattering `haze_angle` lobe;
   - reciprocal diffuse transmission (the opalescent lobe uses the incident angle's T).
3. **Mirrors, later parts:**
   - folded systems (`out_axis`, the Newtonian);
   - `tube = true` for reflectors;
   - refraction on a leg travelling −z;
   - annular-pupil guiding.
4. **Volume vertices in VCM** (merging in media, the UPBP family), in place of the partition,
   where caustics are seen through haze.
5. **Sun from photographed skies:** move the sun's disc out of an HDR sky into an analytic sun.
6. **OpenEXR and a celestial frame,** for NASA's Deep Star Maps.
7. **Heterogeneous media** (density grids, delta tracking): smoke.
8. **Manifold next-event estimation:** glints (eye → specular chain → sun) and telescope stars.
9. **BDPT and VCM with physical cameras.**
10. **Smaller items:**
    - OBJ `vt` coordinates;
    - efficiency-aware MIS and an adaptive photon radius;
    - region inconsistencies where scans and blocks touch.

## Scenes: what remains

- **Gallery upkeep:** the seven-view refresh is complete, including Window Wall.
  The [gallery workflow](../../GALLERY_WORKFLOW.md) retains raw images, binary/assets/settings
  and written reviews. Fine grain, optical softness and production transport diagnostics
  remain explicit; acceptance does not certify convergence or clean conformance.
  Source attribution and portable asset setup are in [credits](../../ASSET_CREDITS.md) and
  the workflow; scene reviews live in `docs/scenes/the_*.txt`.
- **The invisible window:**
  - the optional colour version: three panels behind red, green and blue filter glasses;
  - the Sacred Heart photograph's source rights remain unknown. The user deferred that
    investigation; it does not block requested local render/review/gallery work. Preserve the
    unknown status in [asset credits](../../ASSET_CREDITS.md), without claiming clearance.
- **Further scene targets** (scattering currently falls back to the path-tracing partition;
  efficient volumetric caustics still need volume vertices):
  - forest sunbeams in fog;
  - a crystal with rainbow beams in haze;
  - the observatory with the Jagiellonian globe under a photographed night sky.

## Known limitations

- The CPU reference ignores texture coordinates (it projects `uv` images planar), smooth normals
  and material maps, and refuses stained glass.
