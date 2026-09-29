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
| Integrators | path, light, hybrid | path, light, hybrid, sppm, bdpt, vcm |
| Mesh shading | flat | flat or smooth normals |
| Material maps | first roughness | roughness and tint per texture component |
| Reproducibility | bitwise for a seed, any thread count or compiler | bitwise on a given device and driver |
| Code | `backends/cpu` + the reference transport | `backends/gpu` (C++ host + Slang kernels) |

**Development.** New transport (SPPM, shading normals, and the estimators still to come) is built
on the GPU backend alone; the CPU reference keeps what it has, and scenes using GPU-only features
render there without them (meshes flat; `sppm` falls back to `path`).

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
steps are generated for every lens. Lathe surfaces for glassware (`type = vessel`): outer and
inner (r, z) profiles with filleted corners revolve into discs, cylinders, cone frusta and torus
patches, all analytic with exact normals (torus patches are bracketed along the ray from the slab
entry, so a sun photon arriving from metres away keeps float precision at a fillet). Liquid
surfaces (`type = waves`): an analytic sum of plane waves, or for many waves its C¹ bicubic Hermite
spline with exact nodal slopes, intersected by a Lipschitz-bounded march that cannot skip a
crossing, closed by a floor and walls or cut into a solid basin; bodies can stand inside another's
medium (`in = Sea`). Planned: full closed quadrics, XY/Zernike freeforms, general trimming curves.

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
the terrain stays as a `reference_only` height field for placements, adding no overlapping matter.
Photographs texture surfaces (`image(...)`, planar, triplanar, or by a glTF model's texture
coordinates; linear RGB through the smooth RGB reflectance basis) and light worlds (`sky = map("sky.hdr")`, an equirectangular HDR
environment sampled in proportion to luminance × sin θ, with MIS). Meshes shade with smooth
normals where asked (`normals = smooth | file`, GPU): scattering happens in the frame of the
interpolated normal while the geometric normal decides sides, and a sample the two disagree about
(grazing directions on strongly curved coarse meshes) is dropped, never a region crossed that the
geometry does not cross; particles and photons carry Veach's adjoint correction
(|wo·ns| |wi·n| / |wo·n| |wi·ns|), and photon gathers the camera's |wi·ns| over the photons'
|wi·n| per geometric area. Verified (tests/test_shading.cpp): a 32 × 16 UV sphere shades like the
sphere (L1 0.2% against 3.2% flat); a two-triangle square with leaning corner normals, lit only by
a mirrored sunbeam, draws (wi·ns)/(wi·n) to 0.3% (SPPM) and 0.1% (particles), against 60% and 49%
off without the corrections; path, particle and photon estimators agree on a coarse smooth ball;
a smooth glass ball's caustic is the analytic ball's, with no region inconsistencies. Textures
are material maps too (GPU): a roughness per texture component, mixed like the colour, and a
conductor's tint per component (on a checker map of polished and rough silver, each set of squares
matches the all-polished or all-rough sheet within 1%). A procedural relief (`relief = noise(...)`,
GPU) tilts the shading normal by the analytic gradient of its height; seen from above under a 30°
sun, 4 × 4 pixel blocks match (ns·s)/(n·s) predicted from the reference's own noise within 1.2%
(the relief varies them by 7.3%). Not yet:
clouds, participating atmosphere over large scenes (`the_temple.owe` fakes distant haze with the
sky's below-horizon colour).

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
through the eyepiece, and from beside the instrument, with no change of physics. An afocal
instrument registers its exit pupil with the world; an observer whose pupil is much wider than
an exit pupil it looks into (a 5 mm eye at a 1 mm Ramsden disc receives the image through 4% of
its pupil) draws 90% of its pupil samples from that disc, projected along each pixel's direction
onto the eye's pupil, and weights them by the mixture density: the same measurement, ~10× less
variance (conformance test `backends_guided_eye_pupil_...`). Eyes whose pupil already matches
(the eyepiece presets) are unchanged. `accommodation()` (owe/analysis) finds where an eye must
focus to see a line of sight sharply: it follows the line through the smooth surfaces to the first
surface that scatters or emits, and solves for the vergence at which paraxial rays from the pupil
meet it again there — the object itself, or its image in a mirror, a magnifier or a telescope
(Saturn through the refractor: infinity; the statue through the 16× telescope: its image 1.78 m
away at the d line, L/M²). The eye focuses on the light it receives: lines of sight are traced
across the pupil and grouped by what they reach, each group is weighted by its luminance (a few
radiance samples) and its share of the pupil, and the winner's central line is solved. An eye 2 mm
beside the refractor's 1 mm exit pupil thus focuses on Saturn, which its pupil's edge receives,
and not on the black tube its pupil centre looks into (which had blurred the planet into a disc). Planned: a physical eye model (cornea, lens, curved retina) replacing
the ideal observer.

**IX. Multiple detectors — partial.** Any number of ideal observers and surface sensors;
sensors are matter. They absorb by default; diffuse measurement screens record incident
irradiance while reflecting light toward observers. Screen aperture guides use a full-support
mixture on CPU and GPU. A pixel's response is its square, optionally convolved with a Gaussian
(`pixel_filter`; observers default to σ = 0.4 px, ~0.5 px in all, the usual reconstruction
width): camera samples are displaced by the Gaussian, and particle splats by the same Gaussian
before they are binned, so path, light and hybrid still estimate one image. With the square
alone, anything smaller than a pixel — a star, a focused caustic — images as a pixel-aligned
square whose shape depends on where it falls; with the Gaussian it images as a round spot at its
true sub-pixel position. Sensors keep the square (a physical pixel integrates its area). Planned: curved sensors, spectral detectors (per-band films),
photodiode arrays, detailed sensor coatings.

**X. Rendering algorithm — partial.** Progressive spectral path tracing with NEE, MIS (power
heuristic), stochastic Fresnel branching (with an optional unbiased branch-probability floor
for ghost exploration), Russian roulette, deterministic seeds, raw linear output; particle
tracing; a hybrid estimator with a disjoint path-space partition; stochastic progressive photon
mapping (`sppm`, GPU; Hachisuka & Jensen 2009): a camera pass finds each pixel's first non-specular
point through any chain of glass, mirrors and water (direct light there by NEE and one
scatter-sampled segment, MIS-weighted, as the path tracer), photons deposit only after a scattering
event (no double counting) on a disc in the point's tangent plane with a matching orientation (no
leakage), found through a multi-level spatial hash (radii from millimetres to kilometres), with
exact fixed-point flux accumulation (bitwise reproducible) and progressive radii. Spectra: an
iteration splits its paths and photons into wavelength groups whose hero wavelengths interleave; a
point the eye sees directly takes every photon at the photon's own wavelengths, a point seen
through glass (its wavelengths fixed by dispersion) only its group's, scaled accordingly. Sun
photons are guided: a map over the sun's emission disc counts where photons from the uniform share
proved useful, and three quarters of later photons are drawn from it, weighted by the mixture
density (unbiased; it learns windows and lenses far from the caustics they form: colour speckle
in a window-lit caustic vanished at the same time budget). Verified: white furnace; the image
mean against the unbiased estimators (bias −0.6% at 256, −0.3% at 4096 iterations on a coarse
96×64 image, shrinking as r² ∝ i^(−1/3)); a lens caustic seen through a glass plate and in a
mirror equals the direct view times Fresnel transmittance (0.9269 against 0.9187) and the
reflectance (0.9000 against 0.9); caustics neither path nor light tracing can find. Forward-light
references (particles onto sensors, both backends) match analytic power budgets within 1%: a lens
(Fresnel losses at the actual incidences), a water surface (Fresnel and Pope–Fry absorption), a
concave mirror; ripple caustics match the Monte-Carlo-integrated Jacobian of the surface-to-floor
ray map (L1 difference 1.4% over 160 bins, peak 9.2 E₀ against 9.1). Bidirectional path tracing
(`bdpt`) and vertex connection and merging (`vcm`, Georgiev et al. 2012) run on the GPU with a
light vertex cache (Davidovič et al. 2014): per iteration one light subpath per pixel caches its
non-specular vertices with the partial MIS quantities of Georgiev's recursive formulation and
splats to the eye; each camera vertex then takes next-event estimation, a connection to one vertex
drawn from the cache (weighted by the cache's size per light path), and, for `vcm`, merges the
cached vertices within a radius of a few pixel footprints at its distance from the eye (a function
of position, so both path ends evaluate merging's MIS factor alike; it shrinks as SPPM's). Every
strategy is weighted by the balance heuristic, next-event pdfs of sphere lights (cone sampling)
are evaluated from the actual first light vertex, distances accumulate across null boundaries,
and shading normals enter as in path tracing. Verified: BDPT draws the path tracer's image (mean
1.0000, worst 4 × 4 block 0.05%), VCM converges to it (0.9998, 0.15%); a lens caustic seen through
a glass plate is 0.913 of the direct view (Fresnel 0.919); on a glossy silver plate VCM's error is
7.5× the path tracer's lower at equal samples, where SPPM (gathering at the glossy plate) is 20×
worse than path tracing. On the caustic benchmarks VCM matches or beats SPPM where the caustic is
seen through glass or in a mirror, and trails it for dispersion (an iteration's paths share four
wavelengths; SPPM spreads sixteen). Sun light subpaths are guided as SPPM's photons (the same map
over the emission disc, taught by light paths whose vertices were splatted, connected or merged;
the camera side evaluates the mixture density where its ray crosses the disc, so MIS stays exact):
on the prism benchmark this took VCM's error from 6.5 to 0.089 (SPPM 0.29). An iteration traces
`photons` light subpaths per pixel; the cache is sized by a counting pass for every light path
meeting the world (what guiding tends to), with 30% to spare. Sun and sky particles re-find their
first hit from just before it: from a disc tens of metres out, float error left photons starting
under the surface they scattered from (the study's region inconsistencies fell 6×). Scattering
media are not handled by `bdpt`/`vcm`. Planned: wavelength groups for VCM, ReSTIR, manifold NEE.

**XI. Interactive convergence — partial.** `--passes` refines an unbiased accumulation;
display processing (exposure, white balance, gamut mapping) never touches the raw data. Exposure is
display state with three modes (`owe/render/exposure.hpp`): manual, locked (meter a new view once,
then keep it while navigating; the viewer's default) and adaptive (follow the meter in UI time, rate
limited in EV/s, with a dead band and smoothed readings). The meter is robust by construction (block
means, a percentile band in log luminance with a smooth centre weight; only lit surfaces, not
directly seen emitters, pull it down, in proportion to their area) and continuous in what the image
shows; exports use the resolved exposure the window shows. Every monochromatic colour lies outside
sRGB: the display reduces such a colour's chroma at constant Oklab lightness and hue instead of
clipping its negative component, which had collapsed whole wavelength ranges into one hue (a
spectrum showed as a few bands). The view transform is AgX by default (a filmic sigmoid over 16.5
stops in log2, with an inset of the primaries): colours beyond the display's range go gradually to
white, as a caustic's core or a sun glint does on film, where the standard curve (linear to white,
a short roll-off) clips them 4 stops sooner; middle grey shows the same in both. `owe display`
redisplays a raw PFM with its record's exposure and any view transform. `owe studio` is a
command-line session that edits any scene value (aperture, focus, placement, glass, backend),
rebuilds the world and refines the image pass by pass, writing it after each pass. `owe view`
is a window over every scene and detector: progressive refinement on either backend, scene
edits, pixel probes, and a free virtual eye (orbit, pan, fly, zoom) that looks through
instruments by light transport alone. The free eye accommodates (`accommodation()`, VIII), so
walking up to an eyepiece brings the planet into focus rather than the glass, and it keeps the
scene's integrator, so caustics stay formed by light tracing. The free eye takes the pixel response
of the view it starts from, so at a saved observer's pose it is that observer (tested block by block
over independent runs). Free-view image formation is verified quantitatively: a loose lens seen
from either side forms its inverted image where paraxial optics puts it, a magnifier its upright
virtual image, a concave mirror its real image, and an eye walking across a telescope's exit pupil
sees the image dim as the overlap of the two discs (measured 0.693, 0.386, 0.139, 0 against 0.692,
0.395, 0.146, 0) and return bit for bit when it walks back. There is no denoiser.

**XII. Performance — partial.** `owe bench` measures throughput (M paths/s, M segments/s) on a
fixed suite of scenes and seeds and prints a digest of the raw image, so an optimisation that
changes any result bit is visible at once. CPU work so far, all bit-identical: per-boundary box
culling inside BVH leaves, slab tests with the near/far planes chosen once per ray, culling of
deferred BVH nodes against the closest hit found since, skipping quadric root refinement for
roots that cannot enter (tmin, tmax), shared lattice hashes in the value noise, and texture
weights computed once per hit — 1.19× on the suite at 960×540 (up to 1.33× on mesh- and
instrument-heavy scenes; 16 threads, i7-12650H). On the suite with the detailed models (up to
9 M triangles per scene) the GPU backend (XIII) is 14× faster than the 16-thread CPU with its
portable software traversal and 35× with hardware traversal (RTX 4060 Laptop; 134.8 s → 9.6 s →
3.8 s). Setup is parallel and exact: OBJ files are parsed in chunks split at line breaks and
joined in file order (`std::from_chars`), and hierarchies are built from compact records
partitioned in place — upper levels split level by level with parallel binning and a parallel
partition that makes the same exchanges as the two-pointer sweep, subtrees below on all threads,
the SAH evaluated with prefix sweeps — laid out depth-first in one copy. Every hierarchy is
identical to the sequential build's (tested against it), so no image changes: the telescope's
7.7 M-triangle landscape loads in 1.2 s instead of 6.8 s and its first GPU renderer takes 1.8 s
instead of 5.0 s. Planned: wavefront GPU kernels (material sorting), SIMD/wide BVH on the CPU.

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
drawn so the surface interface is not live across their traversal. (4) Kernels run in groups of
256 threads: the driver sets a kernel's register budget from its group size (NVIDIA: 168 per
thread for 64-thread groups, 128 for 256), and the larger group keeps a third more threads
resident to hide latency, for a small spill (−5% time with hardware traversal, −7% software).
(5) The hardware-traversal kernels leave out the software mesh traversal: a mesh never reaches
the analytic shape code there, but that code, inlined at every traversal site with its stacks,
had spilled ~4 KB of local memory per thread (now 0.1–0.4 KB). None of this changes an image.
(Path regeneration — a lane starting its next sample as soon as its path ends — measured ~5%
faster with hardware traversal, less than (4) and (5), but its interleaved loop is miscompiled
by Mesa's Intel compiler, which loses the device; portability wins, so a pixel's samples run
one after another.) `OWE_GPU_STATS=1` prints each kernel's compiled statistics (registers,
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

**XV. Fractal statue — done.** `fractal_statue` builds exact nested spheres; `the_telescope.owe`
places one on a mountain shoulder 450 m away, observed with the naked eye and through the telescope.

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
dispersion (deviation of every wavelength equal to an independent two-face Snell trace to
10⁻⁹ rad: 74.90° at 404.7 nm to 64.67° at 706.5 nm for 60° N-SF11), glass/water, caustics,
internal reflections, lens ghosts; ripple caustic networks with dispersion fringes
(`the_shallows.owe`), a tumbler of water in window sunlight (`glass/glass_in_sunlight.owe`), a prism in
sunlight with its beams grazing the floor (`prism_in_sunlight.owe`), a glass of tea whose caustic
is amber by Beer–Lambert absorption (`glass/tea_glass.owe`). Planned: rainbows and halos (need
droplet/crystal ensembles), mirages (need GRIN).

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

**XXVII. Canonical demonstrations.** Implemented: The Lens, The Telescope/The Mountain
Observatory, The Camera Obscura, The Glass of Water, The Optical Bench; the caustic studies after
photographs (a prism, glasses of water and tea, rippled shallows); The Temple (a physical 85 mm
camera with tunable aperture and focus); The Observatory at night (planets and the Moon through
refractors); and The Study, all of it in one room. Lens ghosts (double Fresnel reflections inside
an objective) are physical paths in every camera but, for the Petzval designs here, spread over
the frame as a faint veil. Not yet: The City Through Glass.

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
* Pipelines compile from cold in the driver after every kernel change (the VCM kernels take about
  a minute on an RTX 4060); later runs start in under a second from the driver's cache. See
  docs/ROADMAP.md.
* Smooth mesh normals are GPU-only; like every shading-normal renderer, a coarse smooth mesh
  keeps its polygonal silhouette and loses a little light at grazing directions where the
  interpolated and the geometric surface disagree (the shadow-terminator band).
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
