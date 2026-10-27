# Boundary and planetary precision reproductions

These small, independent scenes require no downloaded assets and stay outside the
default benchmark suite. Run commands from the repository root. They are
diagnostic fixtures, not assertions that a particular diagnostic count or image
will occur on every backend. Both CPU and GPU cases below were checked on
2026-10-08. GPU validation used device 1, an NVIDIA GeForce RTX 4060 Laptop GPU,
Vulkan 1.4.351, float32 camera-relative hardware traversal, and the default
pipeline cache. Change `--device 1` for a different device enumeration. These
small runs diagnose correctness; their timing is not a performance benchmark.

The current departure and endpoint rules are documented in
[Surface contacts and finite visibility](../../docs/TRANSPORT_CONTACTS.md).

## Glass touching an opaque support

The glass bottom and support top share z=0. A ray leaving glass downward should
encounter the opaque support before it can escape. Independent boundaries name
ambient as their exterior; offsetting a transmitted ray can skip the support's
coincident entry face. A later backside hit then reports that its incident region
is opaque while the ray still claims ambient.

```sh
build/owe info benchmarks/regressions/contact_glass_support.owe
build/owe render benchmarks/regressions/contact_glass_support.owe --backend cpu --out /tmp/contact_cpu
build/owe render benchmarks/regressions/contact_glass_support.owe --backend cpu --set 'Glass.position=(0,0,0.0501)' --out /tmp/contact_gap_cpu
build/owe render benchmarks/regressions/contact_glass_support.owe --backend gpu --device 1 --out /tmp/contact_gpu
build/owe render benchmarks/regressions/contact_glass_support.owe --backend gpu --device 1 --set 'Glass.position=(0,0,0.0500001)' --out /tmp/contact_gap_100nm
build/owe render benchmarks/regressions/contact_glass_support.owe --backend gpu --device 1 --set 'Glass.position=(0,0,0.0501)' --out /tmp/contact_gap_100um
```

The 100 nm gap was below the original GPU offset at this camera distance; 100 um
was substantially larger. Compare region inconsistencies, leaks, first reported
boundary, and linear image output. Exact contact can also affect CPU transport.
Zero leaks alone does not prove correctness: an invalid region may absorb the ray
before it escapes. Glass/support hit ties and stochastic Fresnel branches can
change counts; do not require CPU and GPU to report identical counts.

Historical contact baseline: CPU loading and the commands above were checked on
2026-10-08 with the existing
build, at 64x64, 16 spp, seed 7. Exact contact reported 136,572 inconsistencies,
zero leaks, and first Glass-to-Support backside hit at z=-0.2. The 100 um gap
reported zero inconsistencies and zero leaks. Both renders took about 0.05 s.
These are observations from this build, not fixed acceptance counts.

Historical GPU contact observations at the same resolution, samples and seed:

| Contact variant | Inconsistencies | Leaks | Mean Y |
| --- | ---: | ---: | ---: |
| Exact contact | 178638 | 1 | 0.16448 |
| 100 nm gap | 60286 | 0 | 0.080020 |
| 100 um gap | 0 | 0 | 0.25123 |

Exact contact first reported a ray still in Glass.interior meeting the support's
ambient side at z=4.76837159e-8. The 100 nm gap first reported ambient meeting
the support's opaque backside at z=-0.19999994. These two signatures expose
both coincident-hit ordering and entry-face skipping. The larger-gap GPU control
matched the CPU control's displayed mean and path/segment counts. Moving the
glass is a diagnostic control, not an authorized fix to a frozen physical scene.

Relevant code: [independent region assignment](../../src/owe/scene/world.cpp),
[GPU ray offsets](../../src/owe/backends/gpu/shaders/world.slang),
[GPU incidence correction and leak accounting](../../src/owe/backends/gpu/shaders/transport.slang),
[CPU ray offsets](../../src/owe/core/math.hpp).
This isolates the contact mechanism seen at Panel/FrameBelow and Floor/ShutterL
in the Invisible Window; it does not reproduce every diagnostic in that scene.

The contact correction proves the ray-collinear departure interval clear before
advancing, and orders distinct touching interfaces by their incident region.
It excludes the departed mesh face (including connected exactly coplanar facets),
preserves a positive next face of that same boundary, and retains diagnostics for
resolved overlaps. A normal-direction offset cannot be justified by checking a
short interval along a grazing ray; this was the cause of the remaining rare
glass/support failures after the first face-identity correction.

Isolated hardware-path validation on the RTX 4060, 16x16, 32 spp, seed 7:
exact contact and 100 nm gap both had mean Y 0.245508630; the 100 um gap had
mean Y 0.245923583. All three had zero inconsistencies and zero leaks.
A 100 um overlap (`Glass.position=(0,0,0.0499)`) retained 23,220 inconsistencies.
These are observed values, not portable fixed acceptance counts.

The fresh test objects ran all four `gpu_contact_` tests and the sensor test:

```sh
build/owe_tests gpu_contact_ backends_surface_sensor_respects_exact_opaque_contact
OWE_GPU_RAY_QUERY=0 build/owe_tests gpu_contact_ backends_surface_sensor_respects_exact_opaque_contact
```

Hardware and software traversal both passed 5/5. Independent hardware controls measured 0.367854484 for an absorbing
100 nm film (expected exp(-1)), exact zero behind a second parallel face in the
same mesh, and 0.0497896136 through three ordered films (expected exp(-3)).
The authored surface sensor was dark at exact opaque contact and measured
approximately pi after a positive 100 nm or 1 mm separation. Raw records and
source/SPIR-V/binary identities are in
`out/performance_20261008/contact_collinear_probe` and
`out/performance_20261008/contact_collinear_sw_probe`. Software film mean was
0.367669591, with the same exact-dark and gap/contact diagnostics. These isolated
proofs override only the corresponding path kernel; integrated other-estimator
validation is tracked separately. Software cold pipeline compilation used about
3.79 GiB peak child RSS and completed without an OOM; this is a compilation
observation, not render throughput.

Float32 cannot establish topology below the local representable geometry and
hit-reconstruction uncertainty. Ordering uses local positional error projected
onto the candidate normal; it does not declare arbitrary nearby bodies to be
touching or clear their diagnostics. Independent exactly coincident plausible
material interfaces remain ambiguous. The CPU uses its own double-precision
error interval and preserves strictly positive authored sensor separations.

## Planetary sphere beneath a local observer

The sphere's physical surface is z=-0.2. The red witness sphere is 0.2 m above
the ground at its lowest point. The control substitutes a local box with the
same top surface; over this small field its curvature difference is negligible.

```sh
build/owe info benchmarks/regressions/planetary_ground.owe
build/owe render benchmarks/regressions/planetary_ground.owe --backend cpu --out /tmp/planet_cpu
build/owe render benchmarks/regressions/planetary_ground.owe --backend gpu --device 1 --out /tmp/planet_gpu
build/owe render benchmarks/regressions/planetary_ground.owe --backend cpu --set Earth.type=box --set 'Earth.size=(200,200,1)' --set 'Earth.position=(0,0,-0.7)' --out /tmp/local_cpu
build/owe render benchmarks/regressions/planetary_ground.owe --backend gpu --device 1 --set Earth.type=box --set 'Earth.size=(200,200,1)' --set 'Earth.position=(0,0,-0.7)' --out /tmp/local_gpu
```

Expected failure signatures include ground height/shadow differences or boundary
diagnostics for the planetary sphere versus its local-ground control. Nonzero
inconsistency/leak counts are not guaranteed in this minimal scene. Inspect the
linear images as well as statistics; no quantitative tolerance has yet been
established. Increasing samples cannot repair a systematic intersection error.

Historical pre-fix planetary baseline: CPU loading and both CPU renders were
checked on 2026-10-08 with the existing
build. The sphere reported 6,206 inconsistencies, zero leaks, and first
Earth-to-Earth backside hit; mean Y was 0.027082. The local box reported zero
inconsistencies and zero leaks; mean Y was 0.029430. Both took about 0.04 s.
The original CPU therefore also had a large-frame surface issue; double precision alone
does not make its spherical control a correctness oracle. Its offset depends
on the world hit point rather than the millions-of-metres local frame, so a
rounding residual can exceed that displacement. This is a candidate explanation
for CPU self-hits, not a separately verified intersection root cause.

Historical pre-fix GPU sphere validation reported 149 inconsistencies, zero leaks, and mean Y
0.029955. The first inconsistency was ambient meeting Earth's opaque backside
at approximately (-487519.406,-1196734.45,-132427.1), far beyond the local
observer's intended footprint. The local-box GPU control reported zero
inconsistencies and zero leaks, mean Y 0.029430, and matched the CPU box control's
displayed mean and path/segment counts. The smaller GPU diagnostic count does
not establish a more accurate sphere: each backend follows different erroneous
paths, and this experiment does not yet quantify image error against a reference.

The original native [GPU frame export](../../src/owe/backends/gpu/gpu_scene.cpp) sets the boundary
scale to camera/body distance plus local extent. For this Earth it is about
12,742,000 m; [surfaceEpsilon](../../src/owe/backends/gpu/shaders/world.slang)
therefore yields about 25.5 m. Float32 spacing near 6,371,000 m is 0.5 m, so the
0.2 m centre displacement is not preserved. [Sphere intersection](../../src/owe/backends/gpu/shaders/geometry.slang)
cannot recover information already lost during float conversion. The CPU uses
double precision and a different origin offset.

The correction retains the full analytic sphere. GPU intersection adds a pole
frame computed in double before float export, preserving small surface heights
without subtracting planetary centres. Nearby rays solve the translated sphere
equation; distant rays retain stable closest-approach roots. Both reconstruct
the world hit on the same pole-anchored sphere using a stable spherical sag,
with point-scale offsets. Native texture and emitter-sampling frames are retained.
Observer rebasing updates the auxiliary data only when its records change.
The CPU independently forms compensated double sphere coefficients directly from
world coordinates using `std::fma`, with no dependency on wider `long double`.
See [GPU intersection](../../src/owe/backends/gpu/shaders/geometry.slang),
[world hit reconstruction](../../src/owe/backends/gpu/shaders/world.slang), and
[CPU compensated intersection](../../src/owe/scene/world.cpp).

Post-fix validation, same 64x64, 16 spp, seed 7:

| Backend / ground | Inconsistencies | Leaks | Mean Y | Segments |
| --- | ---: | ---: | ---: | ---: |
| CPU / sphere | 0 | 0 | 0.029429 | 133330 |
| GPU hardware / sphere | 0 | 0 | 0.029429 | 133330 |
| GPU software / sphere | 0 | 0 | 0.029429 | 133330 |
| CPU / local box | 0 | 0 | 0.029430 | 133325 |
| GPU hardware / local box | 0 | 0 | 0.029430 | 133325 |

The sphere stays curved; a few path differences from the box control are expected.
At the planetary-only milestone, contact was also rechecked: CPU still reported 136572 inconsistencies / zero
leaks, GPU hardware 178638 / one leak, with unchanged means. Contact ownership
was not changed by the planetary correction.

Focused tests run with both hardware and software traversal:

```sh
build/owe_tests planetary_sphere vcm_cache_overflow vcm_cached_diffuse_packets
OWE_GPU_RAY_QUERY=0 build/owe_tests planetary_sphere vcm_cache_overflow vcm_cached_diffuse_packets
```

Both passed 7/7. Tests include the analytic 0.2 m ground clearance, outgoing
self-hit rejection, the far hemisphere, diagonal planet placement, observer
rebasing, and two independent GPU visibility checks. One looks at the planet
through a distant mirror and a pinhole ceiling: a falsely reconstructed z=0
surface sees the open sky, whereas the real z=-0.2 surface lies beneath the
ceiling. Pre-fix mean Y was 0.300000; corrected hardware and software mean was
0.0027513322 versus analytic 0.002970297. A smaller sphere also tests roots from
10000 radii away. Forced VCM cache replay/rejection/recovery and coloured diffuse
packet agreement remain covered after the geometry change.

Validation records are in `out/performance_20261008/planetary_final_{hardware,software}_tests.log`,
`planetary_fixed_{cpu,gpu,software}.{json,pfm,png,log}`, and
`planetary_box_{cpu,gpu}.{json,pfm,png,log}`. These outputs are local artifacts,
not checked-in fixture dependencies.

This fixture addresses the Gregorian's former Earth boundary, not lunar sampling
or the central eyepiece ghost. Those remain separate unresolved observations;
the black-Moon path report was made with the old Earth sphere, whereas the final
optical views used VCM and local ground.

## Mesh departure membership (2026-10-09)

The integrated hardware checkpoint passed 156/157 tests; the remaining smooth
closed-glass mesh test reported five inconsistencies. Its mesh is a valid closed
manifold. The captured photon departed original triangle 40 at a grazing outward
angle. Reconstructing its native departure plane changed the anchor by 29.8 nm.
Hardware accepted triangle 69 on the lagged world ray, but exact float replay
showed that the corrected native ray misses that triangle. Reusing hardware
membership after changing the ray produced the false boundary crossing.

The correction queries the departing mesh's native BVH from its reconstructed
anchor and considers only that owner. Other bodies retain hardware traversal.
This complete query also supplies neighboring triangles that hardware may not
have returned; it does not merely reject stale candidates. Owner records retain
an existing group-BVH root, with root/frame ownership checked by the layout test.
See [native contact query](../../src/owe/backends/gpu/shaders/world.slang) and
[group-root export](../../src/owe/backends/gpu/gpu_scene.cpp).

An isolated immutable hardware proof ran these unchanged tests:

```sh
build/owe_tests gpu_scene_layout_is_flat_and_aligned gpu_contact_ \
  shading_smooth_glass_mesh_refracts_like_the_glass_ball
```

All six passed. Smooth-glass inconsistencies fell from five to zero, while the
flat/smooth image errors stayed 0.0192/0.0057. Exact glass/support contact and
resolved gaps remained clean; real overlap retained 23,220 diagnostics. A thin
absorbing film matched `exp(-1)`, three films matched `exp(-3)`, and the separate
parallel-face opaque control stayed exactly black. No assertions or diagnostic
counters were relaxed.

Exact source, shader/uploader/binary hashes and logs are preserved locally in
`out/performance_20261008/native_owner_hardware_proof_20261009`. The command above
is the reproducible test selection; that proof used a temporary relink with its
recorded uploader and three hardware shaders, before the subsequent integrated
software-parity build. Broad integrated promotion requires the later validation
records. Cold-start-inclusive proof wall time was 308.11 s, with child peak RSS
6,222,928 KiB; these are startup diagnostics, not warm transport benchmarks.

The attempted earlier software smooth-glass three-image diagnostic was stopped
after about 21 minutes without a result. It is explicitly not a passed or failed
conformance result; its logs and termination record remain preserved. Focused
software contact/endpoint validation is separate. The native owner query adds a
bounded mesh traversal; its performance cost requires measurement on the final
validated candidate.

### Final integrated coverage

The integrated candidate passed 25/25 hardware and 21/21 software traversal
checks, with exact selected names/counts verified. Both cover contact/gaps/true
overlap, thin absorption and nearest faces, finite emitter/diffuse/NEE visibility
at both endpoints, sensor contact, cached VCM enclosure, cache replay/recovery,
CIE dispersion, diffuse packets and planetary precision. Hardware additionally
passes the unchanged 2048-sample smooth-glass comparison and all VCM optics tests.
The preceding full CPU checkpoint was 157/157; the preceding full hardware
checkpoint was 156/157 before the now-corrected smooth-glass failure.

Exact commands, selected test names and resource records are in
`out/performance_20261008/contact_native_owner_{hardware,software}_targeted_tests.json`;
raw output is in the corresponding `.log` files. For a compact contact rerun:

```sh
out/performance_20261008/owe_tests_corrected_final gpu_contact_
OWE_GPU_RAY_QUERY=0 out/performance_20261008/owe_tests_corrected_final gpu_contact_
```

A separately bounded software SPPM smoke rendered the extracted smooth mesh at
96x96, 16 iterations, depth 64, one photon per pixel, seed 1, in sixteen short
passes. It completed all passes: 294,912 paths, 721,434 segments, zero
inconsistencies/leaks, mean Y 0.2278. The 420 s limit included cold startup;
actual total wall was 162.05 s, while reported transport was 1.53 s. This smoke
checks completion and basic transport, not the 2048-sample comparison. Exact
arguments and status are in `native_owner_software_sppm_smoke_status.json`, with
per-pass outputs/logs under the same local prefix.

| Validation | Cold-start-inclusive wall | Peak child RSS |
| --- | ---: | ---: |
| Hardware targeted 25 tests | 727.26 s | 9,588,176 KiB |
| Software targeted 21 tests | 488.28 s | 8,911,124 KiB |
| Software SPPM smoke | 162.05 s | 8,728,168 KiB |

These figures describe validation startup, not paired warm transport performance.
Driver compilation remains costly. The frozen corrected CLI/tests, kernel/source
hashes, complete source archive (including new files) and tracked source patch are
recorded in `out/performance_20261008/corrected_final_manifest_20261009.json`.
Production-scene overlap diagnostics and gallery readiness remain separate from
these asset-free engine regressions.
