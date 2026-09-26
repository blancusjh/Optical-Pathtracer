# Optical World Engine

*Mundus ipse systema opticum est* — the world itself is the optical system.

A physically grounded, spectral, non-sequential light-transport engine in which lenses,
telescopes, prisms, droplets and cameras are simply matter placed in a world. Images are not
assembled from effects: the engine defines matter and geometry, propagates light, and lets
images emerge.

This repository currently contains the **CPU high-precision reference tracer** (C++20,
IEEE-754 double throughout) that the vision document calls for (§XIII): the implementation
against which any future GPU backend must be validated. It is complete enough to render the
canonical demonstrations, to analyse lenses, and to interrogate every pixel.

| | |
|---|---|
| ![The Lens](docs/gallery/the_lens.png) | ![Through the telescope](docs/gallery/telescope_eyepiece.png) |
| **The Lens** — a loose magnifier over a page: magnified view, dispersion-tinted caustic in its shadow. | **The Telescope** — the eye's pupil placed at the telescope's computed exit pupil: the statue on the ridge, magnified 16× and inverted. |
| ![The Prism](docs/gallery/the_prism.png) | ![Glass of water](docs/gallery/glass_of_water.png) |
| **The Prism** — lamp → slit → collimator → N-SF11 prism → screen, plus a secondary spectrum from an internal reflection. | **The Glass of Water** — a rod broken at the water line; the card behind compressed by the water cylinder. |

| | |
|---|---|
| ![The temple](docs/gallery/temple_wide.png) | ![The west colonnade](docs/gallery/temple_colonnade.png) |
| **The Temple** — a Doric peristyle temple at golden hour, with a museum scan of Athena on her pedestal. | Fluted columns with entasis along the sunlit west colonnade. |
| ![85 mm at f/2](docs/gallery/temple_cam_f2.png) | ![85 mm at f/11](docs/gallery/temple_cam_f11.png) |
| **A physical camera**: 85 mm Petzval-type portrait lens at f/2, focused on Athena (DOF 10.9–13.2 m). | The same camera at f/11 focused at 25 m: the stop is smaller and the depth of field reaches the temple. |
| ![The observatory](docs/gallery/observatory_room.png) | ![The desk](docs/gallery/observatory_desk.png) |
| **The Observatory** — a domed observatory at night, lit by candles; two refractors aim through the slit. | The astronomer's desk: books, an armillary, a compass under glass. |
| ![Saturn](docs/gallery/observatory_saturn.png) | ![Jupiter](docs/gallery/observatory_jupiter.png) |
| **Saturn at 126×** through the 150 mm f/15 achromat: the planet and its rings are bodies at 1.28·10¹² m; the Cassini division and the globe's shadow on the rings follow from geometry, the violet halo from the achromat's secondary spectrum. | **Jupiter** through the second refractor. |
| ![The Moon](docs/gallery/observatory_moon.png) | ![The slit](docs/gallery/observatory_slit.png) |
| **The Moon at 16×** through the small telescope at the window, with the eyepiece field stop; its phase follows from where the sun is below the horizon. | The dome slit with the night sky, seen by the naked eye. |

More in [`docs/gallery`](docs/gallery) (each PNG has a JSON record for reproduction).

## The one idea

There is one world model and one transport engine:

```
Medium  →  Region  →  Boundary  →  Body  →  Assembly
```

* A **medium** is constitutive data: n(λ) (Sellmeier, Cauchy, tabulated, Ciddor air; catalog
  glasses relative to air), absorption σₐ(λ), scattering σₛ(λ), phase anisotropy.
* A **region** is a connected domain filled with one medium.
* A **boundary** is an oriented surface separating the region on its front from the region
  on its back, with surface optics (Fresnel dielectric, rough GGX dielectric, Lambertian,
  conductor with complex n + ik, mirror, absorber, detector, null).
* A **body** is a physical object built from regions and boundaries — a lens is a volume of
  glass bounded by its polished surfaces, edge steps and rim; a cemented doublet is two
  regions sharing a boundary.
* An **assembly** positions bodies mechanically (a telescope, a camera).

The order in which light meets surfaces is never prescribed; it is discovered by ray casting.
Ghost paths such as S₄ → S₇ → S₄ → S₂ are simply paths. The same `Tracer` serves path
tracing, light tracing, the lens analyser and the inspector.

## Build and test

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
(cd build && ctest --output-on-failure)     # or: build/owe_tests [filter]
```

Requirements: a C++20 compiler and CMake ≥ 3.20. zlib is optional (compressed PNGs).

## Quick start

```sh
build/owe render scenes/the_lens.owe --spp 128 --out lens          # lens.png, lens.pfm, lens.json
build/owe render scenes/the_telescope.owe --detector Eyepiece --spp 256 --passes 8 --out scope
build/owe probe  scenes/the_lens.owe --pixel 320 110 --svg why.svg  # why is this pixel this colour?
build/owe emit   scenes/the_prism.owe --from -0.3195,0,0.1 --dir 1,0,0 --cone 0.25 --svg beam.svg
build/owe lens   lenses/kepler_16x.lens --afocal --fields 0,0.3,0.6
build/owe glass  N-SF11
build/owe info   scenes/the_telescope.owe                           # the world's ontology
build/owe render scenes/the_temple.owe --detector Cam --set Cam.f_number=5.6 --set "Cam.focus=20 m"
build/owe studio scenes/the_temple.owe --detector Cam                # interactive: set, render, probe
```

### Cameras: aperture and focus

A `camera` is a physical camera: lens bodies from a prescription, a housing, mounts, an
aperture stop and a sensor surface. `f_number` resizes the physical stop, and `focus` moves
the sensor to the real plane of best focus for that object distance. Both are ordinary scene
values, so they can be changed from the command line with `--set` (any `Block.key=value`) or
live in the studio:

```
$ build/owe studio scenes/the_temple.owe --detector Cam --resolution 300x200
camera 'Cam' ...
EFL 86.3 mm, f/2.00, focus 11.950 m; depth of field 10.945 m to 13.158 m (CoC 29 µm)
> render 16 4                       # 4 passes of 16 spp; studio.png/.pfm/.json after each
> set Cam.f_number=11               # the stop closes; the scene is rebuilt and refined afresh
> set Cam.focus=25 m
> render 32 4
> probe 150 100                     # why is this pixel this colour?
> detector Wide                     # any observer, camera or sensor
```

Commands: `set`, `unset`, `edits`, `detector`, `size WxH`, `render [SPP] [PASSES]`, `reset`,
`exposure EV|auto`, `info`, `probe X Y [N]`, `quit`. The same commands can be piped from a
file for scripted sweeps.

### `render` — progressive spectral rendering

Writes `PREFIX.png` (display), `PREFIX.pfm` (raw linear, never touched by display
processing) and `PREFIX.json` (reproducibility record: scene hash, integrator, sampling
strategy, seed, samples, render time, statistics) after every pass, so image quality grows
with observation time. Integrators:

* `path` — unidirectional spectral path tracing with next-event estimation and MIS.
* `light` — particle tracing: emitters (area lights, sun, sky) → world → sensors, with
  connections to virtual eyes.
* `hybrid` — a disjoint partition of path space: light tracing owns *eye → diffuse →
  specular⁺ → light* paths (caustics seen directly), path tracing everything else. Both
  are unbiased on their partitions, so the sum is unbiased.

Renders are bit-for-bit reproducible for a given seed, independent of thread count.

### `probe` — "Why is this pixel this colour?"

```
$ owe probe scenes/the_lens.owe --pixel 320 110
 67.04%  refract@Magnifier.S2 → refract@Magnifier.S1 → diffuse@Table.surface → escape ← sky
 19.66%  reflect@Magnifier.S2 → escape ← sky
  8.50%  refract@Magnifier.S2 → reflect@Magnifier.S1 → refract@Magnifier.S2 → escape ← sky
  ...
Representative path of the dominant class:
   #  event     boundary        regions                              n_i     n_t     θi°    θt°   R        T      OPL[mm]
   1  refract   Magnifier.S2    ambient[air] → Magnifier.glass0      1.00028 1.51413 49.18  30.00 0.05818  0.94182 312.438
```

Every value comes from the exact transport state being rendered, not from a separately
drawn diagram. `--svg` draws the contributing paths over a cross-section of the world.

### `emit` — "Where does its light go?"

Launches an ensemble from a point and reports every fate (detected, absorbed where,
escaped), with per-vertex tables and SVG/JSON path output. `--primary` follows the
transmitted branch deterministically (for design work); otherwise Fresnel branching is
stochastic exactly as in rendering.

### `lens` — optical design diagnostics

Loads a sequential prescription, builds it as physical bodies and reports paraxial
properties (EFL, BFL, FFL, f-number, pupils; angular magnification for afocal systems) and
real-ray results traced *through those bodies*: spot sizes, best focus, longitudinal
spherical aberration, chromatic focal shift, distortion, or angular beam spread for
telescopes. `--afocal` solves the tube length of a telescope.

## Scenes

The `.owe` language (full reference in [`docs/SCENE_FORMAT.md`](docs/SCENE_FORMAT.md)):

```
units = mm
body CrownSinglet {
    type = lens
    medium = N-BK7
    front = sphere(R = 48)
    back  = asphere(R = -120, k = -1.2, A4 = 2e-6)
    thickness = 6.2
    diameter = 25
    position = (0, 0, 100)
}
body Scope { type = prescription  file = "lenses/kepler_16x.lens"  afocal = true  tube = true
             position = (0, 0, 1500)  point_at = (0, 20000, 3000) }
observer Eye { position = exit_pupil("Scope")  look_at = (0, 20000, 3000)  pupil = 3 }
```

Included: `the_lens`, `the_telescope` (alpine landscape, forest, statue, telescope),
`the_statue` (fractal statue: direct, close, hand lens, telescope), `the_prism`,
`glass_of_water`, `optical_bench`, `camera_obscura`, `the_ghost`, and two scenes built for
their own sake:

* `the_temple` — a Doric peristyle temple (6 × 9 fluted columns with entasis, entablature,
  pedimented roof) on a hillside at golden hour, with cypresses, olives, amphorae and a marble
  Athena. Observers `Wide` and `Colonnade`; the physical camera `Cam` (85 mm portrait lens).
  The statue is a museum scan (Three D Scans) fetched by `tools/fetch_assets.sh`; without it
  the pedestal stands empty.
* `the_observatory` — a domed observatory at night. Inside: two refractors on iron piers aimed
  through the slit, a small brass telescope on a tripod at the window, a desk with books, an
  armillary of nested rings, a compass under glass, a globe and candles. Outside: the Moon,
  Jupiter and Saturn (with its C, B and A rings) as bodies at their real sizes and distances,
  lit by the sun below the horizon, and 3500 stars. Observers `Room`, `Desk`, `Slit`, and the
  eyepieces `SaturnEyepiece`, `JupiterEyepiece`, `MoonEyepiece`, placed at each telescope's
  computed exit pupil.

Lens files in `lenses/` use a literature-style table (label, R, t, medium, semi-diameter,
`k=`, `A4=`, `stop`); they include an 85 mm f/1.9 Petzval-type portrait lens and a 150 mm f/15
achromatic refractor with a 16 mm Plössl eyepiece.

## Validation

Correctness is established against analytic limits, never by "looks plausible"
(`tests/`, 47 tests, ~10 s):

* **Interfaces** — mirror law; Snell for random rays; Fresnel at normal incidence, Brewster
  angle, Rs + Ts = Rp + Tp = 1; critical angle and TIR; conductor limits.
* **Materials** — Schott n_d and V_d for eight glasses, fused silica, CaF₂, sapphire,
  diamond, water; air n − 1.
* **Geometry** — conic/asphere hits lie on the sag with gradient normals; numeric asphere
  intersection equals the closed-form quadric; a paraboloid focuses every zone to R/2 to
  1e-12 m; an ellipsoidal mirror images focus to focus; quadric hits from 400 m away land
  on 5 cm surfaces to 1e-13 m; BVHs equal brute force.
* **Transport** — Beer–Lambert; an absorbing glass plate with incoherent multiple
  reflections (T = (1−R)²τ / (1 − R²τ²)); white furnace L = Lₑ/(1−ρ); a lossless glass
  sphere is invisible in a uniform field; the n² radiance law under water; irradiance from a
  Lambertian disk (exact off-axis formula) by path *and* light tracing; path tracing, light
  tracing and the hybrid agree within 4σ over independent seeds; bitwise reproducibility.
* **Instruments** — paraxial EFL/BFL equal the thick-lens formulas; a near-axis real ray
  through the lens *body* reaches the paraxial focus to 5e-11 m; the same lens 3 km from the
  origin keeps that precision; spherical aberration and chromatic trends; an achromat
  reduces the F–C focal shift > 10×; afocal Keplerian telescope magnification and Ramsden
  disc; rims and ghost reflections as matter; a physical camera forms an upright image; a
  camera's `f_number` gives that paraxial f-number, and aiming its samples at the exit pupil
  agrees with unaimed sampling.
* **Scene building** — the dome and round wall are exact (hits on the surface to 1e-11 m,
  never inside the slit or window); scene edits override named and unnamed blocks.

## Status

What exists, what is partial and what is ahead is mapped section by section to the vision
in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). In short: the CPU reference engine,
geometrical optics with spectral transport, the ontology, instruments inside the world,
inspection and lens diagnostics are implemented, with a command-line studio for tuning a
scene while it refines; the Vulkan/GPU backend, bidirectional estimators (BDPT/VCM), GRIN
media, thin-film coatings, polarisation, optimisation and a graphical interactive application
are not yet.
