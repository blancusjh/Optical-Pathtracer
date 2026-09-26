# Architecture and status

This document maps the Optical World Engine vision, section by section, to the code. Status
markers: **done** (implemented and validated), **partial** (a foundation exists; named parts
are missing), **planned** (not started; the design accommodates it).

## Source map

| File | Role |
|---|---|
| `src/owe/math.hpp` | Double-precision vectors, rigid transforms, AABBs, origin displacement |
| `src/owe/sampling.hpp` | PCG32 streams, warps, Henyey–Greenstein |
| `src/owe/spectrum.*`, `wavelengths.hpp` | CIE 1931 fit, Planck, spectra, hero-wavelength sampling |
| `src/owe/medium.*` | n(λ) models, glass/metal catalogs, media |
| `src/owe/optics.hpp` | Snell, Fresnel (dielectric, conductor), GGX — scene-free kernels |
| `src/owe/geometry.*`, `bvh.*` | Sag surfaces (conic + even asphere), planes with apertures, spheres, cylinders, meshes |
| `src/owe/world.*` | The ontology: media, regions, boundaries, bodies, assemblies, environment, lights |
| `src/owe/scattering.*` | Boundary scattering (sample/eval), secondary-wavelength weights |
| `src/owe/transport.*` | The one transport engine: camera paths, particles, diagnostic walks |
| `src/owe/detector.*` | Films, ideal observer (virtual eye), surface sensors |
| `src/owe/render.*` | Progressive renderer, image output, reproducibility records |
| `src/owe/builders.*` | Lenses, mirrors, stops, tubes, solids, cup, terrain, forest, fractal statue |
| `src/owe/prescription.*` | Sequential tables as a view; paraxial analysis; afocal solve |
| `src/owe/analysis.*` | Real-ray lens diagnostics through the world representation |
| `src/owe/inspect.*` | Pixel interrogation, emission probes, SVG/JSON path export |
| `src/owe/scene_parser.*`, `scene_loader.*` | The `.owe` language; physical cameras |
| `apps/owe.cpp` | Command-line front end |

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
with slope/altitude texturing, conifer forests, solid textures (checker, noise, wood). Not yet:
image textures, smooth shading normals, clouds, participating atmosphere over large scenes.

**VIII. Instruments inside the world — done.** A camera is lens + housing + mounts + stop +
sensor (`addPhysicalCamera`); a telescope is an assembly built from its prescription;
observers are placed with `exit_pupil("Name")`. `the_telescope.owe` shows the world naked,
through the eyepiece, and from beside the instrument, with no change of physics. Planned: a
physical eye model (cornea, lens, curved retina) replacing the ideal observer.

**IX. Multiple detectors — partial.** Any number of ideal observers and surface sensors;
sensors are matter (they absorb what they record). Planned: curved sensors, spectral
detectors (per-band films), photodiode arrays, sensor reflectance.

**X. Rendering algorithm — partial.** Progressive spectral path tracing with NEE, MIS (power
heuristic), stochastic Fresnel branching (with an optional unbiased branch-probability floor
for ghost exploration), Russian roulette, deterministic seeds, raw linear output; particle
tracing; a hybrid estimator with a disjoint path-space partition. Planned: Tier II BDPT,
Tier III VCM (the camera obscura's inside view is the motivating hard case), Tier IV path
guiding, ReSTIR, manifold NEE.

**XI. Interactive convergence — partial.** `--passes` refines an unbiased accumulation;
display processing never touches the raw data. There is no interactive viewer or denoiser.

**XII. Performance — planned.** The CPU reference traces ~2–3 M paths/s on four cores. The
real-time targets belong to the GPU backend.

**XIII. GPU architecture — planned.** Not started: this environment has no Vulkan SDK or GPU.
The reference is organised to make the port mechanical: scattering and Fresnel kernels
(`optics.hpp`, `scattering.cpp`) are pure functions of an `Interface`; shapes are small
closed-form or bracketed intersectors suited to intersection shaders; the world is flattened
into arrays at `build()`; randomness is counter-based per pixel and pass. The CPU tracer is the
oracle: the plan is to compare GPU and CPU on every scene in `tests/` and `scenes/`.

**XIV. Extreme spatial scale — partial.** Double precision everywhere; rigid hierarchical
transforms; the ray-origin displacement scales as 1e-11 × coordinate magnitude (10 pm at the
origin, 1 nm at 100 m) and moves along the new ray, so there is no lateral error. Quadric
intersections (spheres, cylinders, conic sags) re-solve from the approximate hit point, so a
ray arriving from hundreds of metres lands on a centimetre-scale surface to within an ulp of
its origin rather than ~1e-10 m inside it. Regression tests cover a lens 3 km from the origin
and distant-origin quadric hits. Planned: camera-relative coordinates for the GPU,
optics-aware LOD (only exact geometry exists today, so there is nothing to LOD).

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
scene file. Planned: live editing, ray fans, pupil maps.

**XIX. Inverse design — planned.** Paraxial and real-ray merit functions exist; an optimiser
and differentiable transport do not. Nothing in the representation precludes them.

**XX. Historical instruments — partial.** Camera obscura, Keplerian and Galilean telescopes,
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
integrator, spectral sampling, seed, samples, passes, depth, time and statistics; results are
bitwise identical across thread counts.

**XXVI. Validation — done for the implemented physics.** See `tests/` and the README.
External comparisons with PBRT/Mitsuba/LuxCore are planned.

**XXVII. Canonical demonstrations.** Implemented: The Lens, The Statue, The Telescope/The
Mountain Observatory, The Camera Obscura, The Prism, The Glass of Water, The Optical Bench,
The Ghost. For the uncoated Petzval objective in `the_ghost.owe`, double-reflection ghosts
are present (an emission probe finds ~0.4% of the lantern's flux in them) but spread over the
whole frame as a veil; a sharper demonstration needs a design with compact ghosts plus
BDPT. Not yet: The City Through Glass.

**XXVIII–XXX. User experience and the whole.** The command-line tools expose the same world
to rendering, inspection and design. The interactive application is future work.

## Known limitations

* Single-sided detectors do not reflect (real sensors reflect a few percent).
* No antireflection coatings: every air–glass surface reflects its uncoated Fresnel share,
  which produces the veiling glare visible through the telescopes.
* The light tracer covers directly visible emitters and connections to ideal observers, not
  to surface sensors behind lenses (those receive particles directly).
* The holed-disk aperture cannot be sampled by next-event estimation.
* Mesh normals are geometric (faceted shading).
