# Architecture and status

This document maps the Optical World Engine vision, section by section, to the code. Status
markers: **done** (implemented and validated), **partial** (a foundation exists; named parts
are missing), **planned** (not started; the design accommodates it).

## Source map

The code is layered. Each directory may include only the layers listed above it (and itself);
backends never include each other, and the GPU backend never includes the reference transport.
These rules are checked on every include by the `layering` test (`cmake/CheckLayers.cmake`).

| Directory | Role |
|---|---|
| `src/owe/core/` | Double-precision vectors, rigid transforms, AABBs, origin displacement; PCG32 streams and warps; CIE 1931, Planck, spectra, hero wavelengths; n(λ) models, glass/metal catalogs, media |
| `src/owe/scene/` | The world model: sag surfaces (conic + even asphere), planes with apertures, spheres, cylinders, meshes, dome with slit, round wall with openings; BVHs; media → regions → boundaries → bodies → assemblies, environment, lights; detectors (films, ideal observers, surface sensors); builders (lenses, mirrors, tubes, terrain, forest, lathe profiles, columns, starfields); sequential prescriptions and paraxial analysis; transport statistics |
| `src/owe/transport/` | The reference transport, IEEE-754 double: interface optics (Snell, Fresnel, GGX — scene-free kernels), boundary scattering, and the `Tracer` (camera paths, particles, diagnostic walks with path records) |
| `src/owe/analysis/` | Real-ray lens diagnostics through the world representation; pixel interrogation, emission probes, SVG/JSON path export — all on reference walks |
| `src/owe/loader/` | The `.owe` language, scene edits (`Block.key=value`); physical cameras (f-number, real focus traced by the analyser, depth of field) |
| `src/owe/render/` | Backend-neutral rendering: the `Renderer` and `Backend` interfaces, image output (display pipeline, PNG, PFM), reproducibility records |
| `src/owe/backends/cpu/` | The reference backend: the reference transport scheduled on host threads |
| `src/owe/backends/gpu/` | The portable GPU backend: the world flattened for kernels (camera-relative, float, 16-byte records; mesh groups), hardware acceleration structures where the device has ray queries, the Vulkan compute host (volk-loaded; MoltenVK on macOS), the renderer, and the Slang kernels in `shaders/` (a float port of geometry, traversal, spectra, optics, camera paths and particles, detectors) |
| `src/owe/backends/` | The registry — the only code that names concrete backends — and the statistical comparison of two backends |
| `apps/` | `owe` (command line and the `studio` session) and `owe view` (SDL3 and Dear ImGui); they reach backends only through the registry |
| `tests/` | Validation of the physics against analytic limits; `test_backends.cpp`, the conformance suite every backend runs |
| `tools/` | Scene-model generators (Python, Blender) and checks, gallery rendering |

## Backends

There is one measurement and several places to run it. A **backend** implements the `Backend`
interface (`src/owe/render/backend.hpp`): a name, availability (and why not), devices, the
integrators it implements, and a factory for `Renderer`s — progressive, unbiased estimators of a
detector's image. Front ends choose a backend by name through the registry
(`src/owe/backends/registry.hpp`): `RenderSettings::backend` (`--backend` on `render`, `studio`,
`bench`, `view` and `compare`; `backend NAME` in the studio; a button or **G** in the viewer) is
the only thing that changes. Image output, records, the display pipeline and every command are
shared. An integrator a backend does not implement falls back to `path`, which every backend
implements and which estimates the same image (`adaptToBackend`, which says so).

| | `cpu` (reference) | `gpu` |
|---|---|---|
| Where | host threads | Vulkan compute: NVIDIA, AMD, Intel; Apple silicon via MoltenVK |
| Arithmetic | IEEE-754 double | float32, camera-relative (XIV) |
| Integrators | path, light, hybrid | path, light, hybrid |
| Reproducibility | bitwise for a seed, any thread count or compiler | bitwise on a given device and driver |
| Code | `backends/cpu` + the reference transport | `backends/gpu` (C++ host + Slang kernels) |

**Independence.** Each backend depends on the scene model and the render interface only; neither
includes the other, and the GPU backend does not include the reference transport — it is a port
of it, not a client (the `layering` test enforces this). Each backend is its own library
(`owe_backend_cpu`, `owe_backend_gpu`); a build without `-DOWE_GPU=ON` contains a GPU backend
that reports itself unavailable, so `--backend gpu` explains what is missing.

**Conformance.** `tests/test_backends.cpp` runs every check on every backend available on the
machine: analytic results (white furnace, an invisible dielectric sphere in a uniform field,
Fresnel reflectance of uncoated glass, pinhole irradiance ∝ aperture area, starlight plus a lamp,
reflected moonlight, a diffuse screen's readout and its radiance across the hemisphere, a free
eye behind a telescope), bit-exact observer resets, and — for every backend other than the
reference — block-wise statistical agreement with the reference on a caustic, diffuse screens,
a telescope, a physical camera and Saturn through the observatory's refractor. `owe compare`
runs the same test on any scene and detector.

**Adding a backend** (e.g. a native Metal or CUDA host, or a wavefront CPU tracer): implement
`Backend` and `Renderer` under `src/owe/backends/<name>/`, add it to the registry and give its
directory a rule in `cmake/CheckLayers.cmake`. The conformance suite then covers it with no new
tests, and every front end offers it.

## Section by section

**I. Scope — partial.** Reflection, refraction, dispersion, absorption, scattering
(homogeneous media), Fresnel splitting, TIR, surface roughness, caustics, multiple
reflections/refractions, vignetting, aberrations, finite apertures, real sensors and pupils
all emerge from the transport. Not yet: polarisation, anisotropic media.

**II. Regions and boundaries — done.** `World` holds media → regions → boundaries → bodies →
assemblies. Boundaries carry front/back regions; the incident region is decided by the side a
ray arrives from, so propagation order is discovered, never prescribed. Rays that reach a
boundary from a region other than the one they travel in (overlapping or unclosed matter) are
counted as *region inconsistencies*; the render log names the first one (boundary, regions,
position), which has repeatedly located modelling errors in practice. At junction curves where
two analytic surfaces meet (a lens face and its rim) such events occur at ~1 per 10⁷–10⁸
segments; the engine continues with the geometric side. The mechanism is understood: a ray
that strikes a surface exactly on its edge circle and turns outward starts its next segment a
fraction of a nanometre outside the rim cylinder. Watertight edge handling is future work.

**III. Universal non-sequential ray — done.** Ray state: position, direction, four
wavelengths (hero + stratified), region, per-wavelength throughput, optical path length
Σ n·s, optional history (`PathRecord`: event, boundary, regions, n_i, n_t, θ_i, θ_t, R, T,
OPL, throughput). One engine handles every event type.

**IV. Interfaces obey optics — done.** Snell and exact unpolarised Fresnel per wavelength;
dispersion from n(λ); Sellmeier, Cauchy, tabulated and Ciddor models; Schott glasses are
relative to air (absolute index = n_rel · n_air(λ)). Planned: birefringence.

**V. Optical surfaces — partial.** Planes, spheres, cylinders, conic sags (paraboloid,
ellipsoid, hyperboloid caps) with closed-form intersection, even aspheres with bracketed
root finding, triangle meshes (OBJ import), apertures (disk, annulus, rectangle, ellipse,
off-centre hole). Mechanical extent and clear aperture are separate. Physical rims and edge
steps are generated for every lens. Planned: cones, full closed quadrics, XY/Zernike
freeforms, general trimming curves.

**VI. Lens definition — done.** `body X { type = lens ... }` with front/back surfaces or a
surface list with media (cemented groups), plus `focal =` design helpers. Sequential tables
(`lenses/*.lens`) are converted into bodies; `owe lens` prints the table view and paraxial
data.

**VII. Natural scenes — partial.** Meshes coexist with analytic optics; procedural terrain
with slope/altitude texturing, conifer forests, solid textures (checker, noise, wood, marble,
latitude bands, radial rings). Architecture is built from lathe profiles, tori, fluted Doric
columns (entasis, echinus, abacus), boxes and prisms, placed in rows with `repeat`. Imported
scans (OBJ) are reoriented (`model_up`, including a measured up vector), scaled to a physical
height and stood on the ground; opaque imported meshes are shells with air on both sides, so
museum scans with holes and mixed winding cause no region inconsistencies. `the_temple.owe`
and `the_observatory.owe` are built this way. Detailed scene models — jointed paving, branching
olives, cypresses and conifers, brick-and-stone architecture, bound books, a woven carpet,
engraved instruments, slope-grounded landscapes — are generated deterministically into `models/`
by `tools/` and loaded as ordinary meshes; where a detailed relief replaces a procedural terrain,
the terrain stays as a `reference_only` height field for placements, adding no overlapping matter. Not yet: image textures, smooth shading normals,
clouds, participating atmosphere over large scenes (`the_temple.owe` fakes distant haze with
the sky's below-horizon colour).

**VIII. Instruments inside the world — done.** A camera is lens + housing + mounts + stop +
sensor (`addPhysicalCamera`); `f_number` rescales the physical stop through the paraxial
entrance pupil, and the sensor sits at the real best focus for the requested object distance.
Sensor samples are drawn from a mixture: 85% aimed at the paraxial exit pupil (×1.3 margin),
15% at the whole rear opening, weighted by the mixture density. Aiming at the pupil alone was
measured to lose ~1.8% of the light at f/11 in `the_temple.owe` — the ghost and veiling light
of the uncoated lens leaves the rear element outside the pupil — so the mixture keeps it. A
telescope is an assembly built from its prescription; observers are placed with
`exit_pupil("Name")`; `the_observatory.owe` has three telescopes aimed at the Moon, Jupiter
and Saturn, each with an eye at its exit pupil. `the_telescope.owe` shows the world naked,
through the eyepiece, and from beside the instrument, with no change of physics. Planned: a
physical eye model (cornea, lens, curved retina) replacing the ideal observer.

**IX. Multiple detectors — partial.** Any number of ideal observers and surface sensors;
sensors are matter. They absorb by default; diffuse measurement screens record incident
irradiance while reflecting light toward observers. Screen aperture guides use a full-support
mixture on CPU and GPU. Planned: curved sensors, spectral detectors (per-band films),
photodiode arrays, detailed sensor coatings.

**X. Rendering algorithm — partial.** Progressive spectral path tracing with NEE, MIS (power
heuristic), stochastic Fresnel branching (with an optional unbiased branch-probability floor
for ghost exploration), Russian roulette, deterministic seeds, raw linear output; particle
tracing; a hybrid estimator with a disjoint path-space partition. Planned: Tier II BDPT,
Tier III VCM (the camera obscura's inside view is the motivating hard case), Tier IV path
guiding, ReSTIR, manifold NEE.

**XI. Interactive convergence — partial.** `--passes` refines an unbiased accumulation;
display processing (exposure, white balance) never touches the raw data. `owe studio` is a
command-line session that edits any scene value (aperture, focus, placement, glass, backend),
rebuilds the world and refines the image pass by pass, writing it after each pass. `owe view`
is a window over every scene and detector: progressive refinement on either backend, scene
edits, pixel probes, and a free virtual eye (orbit, pan, fly) that looks through instruments
by light transport alone. There is no denoiser.

**XII. Performance — partial.** `owe bench` measures throughput (M paths/s, M segments/s) on a
fixed suite of scenes and seeds and prints a digest of the raw image, so an optimisation that
changes any result bit is visible at once. CPU work so far, all bit-identical: per-boundary box
culling inside BVH leaves, slab tests with the near/far planes chosen once per ray, culling of
deferred BVH nodes against the closest hit found since, skipping quadric root refinement for
roots that cannot enter (tmin, tmax), shared lattice hashes in the value noise, and texture
weights computed once per hit — 1.19× on the suite at 960×540 (up to 1.33× on mesh- and
instrument-heavy scenes; 16 threads, i7-12650H). On the suite with the detailed models (up to
9 M triangles per scene) the GPU backend (XIII) is 13× faster than the 16-thread CPU with its
portable software traversal and 32× with hardware traversal (RTX 4060 Laptop; 134.8 s → 10.1 s →
4.2 s). Planned: wavefront GPU kernels (material sorting), a parallel build of the merged mesh
hierarchies, SIMD/wide BVH on the CPU.

**XIII. GPU architecture — partial.** One of the backends described above (`-DOWE_GPU=ON`, `--backend gpu`):
Vulkan compute, so it runs on NVIDIA, AMD and Intel GPUs and on Apple silicon through MoltenVK,
with kernels written in Slang and compiled to SPIR-V at build time (embedded in the binary).
The host loads Vulkan at run time through volk, so nothing links against a Vulkan SDK. The
kernel is a float port of the reference transport — the same shapes (quadric re-solve
included), two-level BVH (world boundaries, then mesh triangles in the shape's frame),
spectra, index models, textures, Fresnel/GGX scattering, hero wavelengths, NEE with MIS,
Russian roulette, ideal observers and surface sensors — and draws random numbers in the CPU's
order from the same PCG32 streams, so GPU and CPU paths coincide until float rounding makes a
decision differ. Camera paths run one thread per pixel; a pass is split into ~50 ms dispatches
(no OS watchdog) with each pixel's random stream and compensated sums kept in device memory, so
the result does not depend on how a pass is split. The light and hybrid integrators add a
particle kernel (one thread per particle, its own PCG32 stream; the hybrid partition is the
reference's). Particles splat onto pixels from many threads in no fixed order; float atomics
would make the image depend on that order, so each splat's 24-bit significand is added as an
integer into 64-bit fixed-point bins keyed by its exponent (32-bit atomics with carries, which
every Vulkan device has) — exact, associative, and read back as double. Renders of every
integrator are therefore bitwise reproducible on a given device and driver. Float, not double, because Apple GPUs have no double at all and consumer GPUs run it at
1/32–1/64 rate; see XIV for how precision is kept.

Performance. (1) Meshes that share a frame form one *mesh group* with a single hierarchy and a
per-triangle owner: modelled scenes are many meshes spanning the same space (bricks, boards,
carpet threads ... one mesh per finish), and separate hierarchies overlap completely — a camera
ray in the observatory descended into 22 of them before its hit. The groups depend only on the
meshes and their frames, so they are built once per scene and cached (weakly held, so a freed
scene never matches). (2) Traversal is hardware where the device has it (`VK_KHR_ray_query`):
mesh groups become triangle BLASes built straight from the kernels' triangle buffer, analytic
boundaries one-box BLASes whose candidates the kernel resolves with its own analytic code, and
hits on triangles are re-solved against the triangle's plane as in software. Instances are
camera-relative; null boundaries have their own mask. Elsewhere the portable software traversal
runs — a variant of the same kernels, selected per renderer. (3) Shadow rays ask only whether any
matter blocks them (first hit ends the search), and are traced after the scatter direction is
drawn so the surface interface is not live across their traversal. None of this changes an image.
(Path regeneration — a lane starting its next sample as soon as its path ends — measured ~5%
faster with hardware traversal, but its interleaved loop is miscompiled by Mesa's Intel compiler,
which loses the device; portability wins, so a pixel's samples run one after another.)
`OWE_GPU_STATS=1` prints each kernel's compiled statistics (registers,
local memory) where the driver offers them. Validation: `owe compare` renders a scene
with both backends over independent seeds and tests block means of X, Y and Z against their
standard errors, and the conformance suite runs every analytic check on it. Planned: wavefront kernels, hardware
ray queries for triangle meshes, a native Metal host (Slang also emits Metal) if MoltenVK
proves limiting — each a backend or a backend capability behind the same interface.

**XIV. Extreme spatial scale — partial.** `the_observatory.owe` puts a 16 mm eyepiece and
Saturn (1.28·10¹² m away) in one world, with the Moon, Jupiter and an Earth sphere of radius
6371 km beneath the terrain; stars are emitters at 10¹³ m. Double precision everywhere; rigid hierarchical
transforms; the ray-origin displacement scales as 1e-11 × coordinate magnitude (10 pm at the
origin, 1 nm at 100 m) and moves along the new ray, so there is no lateral error. Quadric
intersections (spheres, cylinders, conic sags) re-solve from the approximate hit point, so a
ray arriving from hundreds of metres lands on a centimetre-scale surface to within an ulp of
its origin rather than ~1e-10 m inside it. Regression tests cover a lens 3 km from the origin
and distant-origin quadric hits. The GPU backend works in float, camera-relative: positions are
differenced with the detector's origin in double on the host and then rounded, each boundary's
frame is reached by one float rotation plus a translation that already includes the camera
offset, bounding boxes are rounded outward, quadric roots use the geometric (closest-approach)
form and are re-solved from the approximate hit, triangle hit distances are taken from the plane
through each triangle's two shorter edges (Möller–Trumbore's own distance loses precision as
1/area: on the 0.3 mm-wide sliver triangles of fan-triangulated bevels it placed float hits up to
38 µm off the surface, so the next ray began inside the body and fan diagonals rendered as dark
lines — a conformance test now covers it), and ray offsets scale with the float error of
the frame they were computed in (2e-6 × the larger of the point's and the local frame's
coordinates: 40 nm for an eyepiece lens 2 cm from the eye, 70 µm for a tree trunk 34 m away).
Saturn at 1.28·10¹² m through the 150 mm refractor renders consistently with the double
reference. Planned: optics-aware LOD (only exact geometry exists today, so there is nothing to
LOD).

**XV. Fractal statue — done.** `fractal_statue` builds exact nested spheres; `the_statue.owe`
observes it directly, up close, through a hand lens in the world and through a telescope.

**XVI. Inspection — partial.** Paths with λ, n, θ_i, θ_t, R, T, regions and OPL from the exact
rendering state; SVG path drawings over a sliced cross-section of the world; JSON export.
Planned: interactive overlay, chief/marginal ray labelling, ray density maps.

**XVII. Interrogate a pixel / a point — done.** `owe probe` (transport classes ranked by
contribution, representative path) and `owe emit` (fates of an emission ensemble).

**XVIII. Design from inside the world — partial.** Spot size, best focus, LSA, chromatic
focal shift, distortion, afocal beam spread, pupils, all traced through the world's bodies.
Editing R, spacing, glass, tilt (`tilt`), decenter (`decenter`) and apertures is done in the
scene file, with `--set Block.key=value`, or live in `owe studio`. Planned: ray fans, pupil
maps.

**XIX. Inverse design — planned.** Paraxial and real-ray merit functions exist; an optimiser
and differentiable transport do not. Nothing in the representation precludes them.

**XX. Historical instruments — partial.** Camera obscura (a pinhole room, and the same room with
a 300 mm f = 3 m lens), Keplerian and Galilean telescopes,
Petzval-type objective, simple magnifier, achromats. Planned: Newtonian (parabolic mirror and
flat diagonal already exist as bodies), Cooke triplet, Double-Gauss, Tessar, microscope.

**XXI. Natural phenomena — partial.** Underwater refraction (n² law validated), prism
dispersion, glass/water, caustics, internal reflections, lens ghosts. Planned: rainbows and
halos (need droplet/crystal ensembles), mirages (need GRIN).

**XXII. GRIN media — planned.** The transport loop's segment step is the extension point:
replace straight segments by an eikonal integrator inside regions flagged as graded.

**XXIII. Time — partial.** Optical path length Σ n·s is accumulated per path (t = OPL/c).
Planned: transient films.

**XXIV. Distributed rendering — partial.** Work is split into independent, deterministically
seeded units (pixels per pass; fixed particle lanes), which is the property distribution
needs. Planned: multi-GPU/multi-node scheduling.

**XXV. Reproducibility — done.** Every render writes a JSON record with scene FNV-1a hash,
backend, integrator, spectral sampling, seed, samples, passes, depth, time and statistics;
results are bitwise identical across thread counts and compilers. (Sampling calls used to pass
several `rng.uniform()` draws as arguments, whose evaluation order C++ leaves unspecified: GCC
and Clang produced different images from the same seed. The order is now explicit — the one
GCC used, so earlier records still reproduce — and the GPU kernels follow it.) GPU renders are
bitwise reproducible on a given device and driver.

**XXVI. Validation — done for the implemented physics.** See `tests/` and the README. Every
backend runs the conformance suite (`tests/test_backends.cpp`: analytic limits, and statistical
agreement with the reference for the others; `owe compare` on any scene); a backend that cannot
run on the machine is reported as skipped. External comparisons with PBRT/Mitsuba/LuxCore are
planned.

**XXVII. Canonical demonstrations.** Implemented: The Lens, The Statue, The Telescope/The
Mountain Observatory, The Camera Obscura, The Prism, The Glass of Water, The Optical Bench,
The Ghost; plus The Temple (a physical 85 mm camera with tunable aperture and focus) and The
Observatory at night (planets and the Moon through refractors). For the uncoated Petzval objective in `the_ghost.owe`, double-reflection ghosts
are present (an emission probe finds ~0.4% of the lantern's flux in them) but spread over the
whole frame as a veil; a sharper demonstration needs a design with compact ghosts plus
BDPT. Not yet: The City Through Glass.

**XXVIII–XXX. User experience and the whole.** The command-line tools expose the same world
to rendering, inspection and design; `owe view` is the interactive window. Choosing where the
transport runs is one setting, never a different program.

## Known limitations

* Physical-camera detectors absorb; standalone measurement screens can have diffuse reflectance.
* No antireflection coatings: every air–glass surface reflects its uncoated Fresnel share,
  which produces the veiling glare visible through the telescopes.
* The light tracer covers directly visible emitters and connections to ideal observers, not
  to surface sensors behind lenses (those receive particles directly).
* The holed-disk aperture cannot be sampled by next-event estimation.
* Mesh normals are geometric (faceted shading).
* The sun is a directional source with its irradiance at Earth, so the albedos of Jupiter and
  Saturn in `the_observatory.owe` carry the (1 AU / r)² factor explicitly.
* Diffraction is not modelled: a telescope's resolution is set by its geometric aberrations
  (for the 150 mm refractor, ~0.8″ at d-light — close to its real Airy limit by coincidence).
* Transport through dispersive lenses carries one wavelength per path, so camera and eyepiece
  views need many samples per pixel for smooth colour.
* GPU backend: hardware traversal needs `VK_KHR_ray_query` (NVIDIA RTX, AMD RDNA2+, Intel Arc);
  MoltenVK does not offer it, so Apple silicon uses the software traversal. Light tracing of
  distant sources (sun, sky) in scenes of astronomical extent loses float precision far from the
  camera (and is inefficient on every backend: particles start on a disk the size of the scene).
  Float
  precision (ray offsets of 2e-6 of the distance to the camera skip features thinner than that);
  mesh emitters are sampled from a float area CDF; the first run compiles the kernel in the
  driver (~10 s on an RTX 4060; cached afterwards).
