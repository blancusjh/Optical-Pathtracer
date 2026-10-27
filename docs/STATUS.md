# Current status

Updated 2026-10-13. This is the current project checkpoint; earlier claims and
capture logs are retained in the [October archive](archive/2026-10/ROADMAP.md).

## Maintenance changes

- Visibility uses endpoint records on CPU and GPU, keeping region, primitive,
  normal and numerical uncertainty together. CPU contact bounds share one helper;
  shape face identity no longer requires a mesh cast in the visibility loop.
  GPU calls use one internal scalar adapter to avoid driver spills at the
  non-inlined visibility boundary; callers retain endpoint records.
- Boundary, triangle, film and push-constant fields share host/shader definitions.
  Push values have names and types; host offsets are checked at compile time.
- GPU dispatch batches use a bounded queue after a synchronous calibration batch.
  Readbacks and VCM capacity decisions still synchronize. Accepted camera and
  light sums stay in compensated float accumulators on the GPU until resolve.
  Particle bins remain exact integers during transport. This replaces host double
  accumulation, so image agreement must be checked within floating-point tolerance.
- Environment overrides for VCM strategy masking and SPPM debug readback are
  removed. Strategy-isolation tests use explicit renderer settings.
- New gallery promotion requires cleared rights, zero leaks and at most one region
  inconsistency per million paths (zero until 10-26), with preflight checks for every
  selected entry before copying.
- CI runs Python tooling/asset tests and GPU contact regressions. Required GPU
  tests fail when the device is unavailable; unmatched test filters also fail.
- The four empty root files and obsolete `build-dev` directory are removed.
  Scratch evidence linked by the documentation is preserved under
  [benchmarks/evidence](../benchmarks/evidence). Raw rendering scratch remains local.

Validation passed: **158/158** hardware tests, **16/16** selected software-traversal
tests, then final film/reset checks (**2 hardware, 1 software**) after the shared
film layout change. Its shader bytes and checked render output are unchanged.
The final CPU-only build and seven tooling/viewer CTest groups pass; these include
**46 Python tool tests** and **3 asset tests**. Full records are in the
[validation manifest](../benchmarks/evidence/maintenance_20261012/validation.json).

Measured against the saved pre-maintenance workspace executable on the RTX 4060,
with one warmup and three timed repeats per variant. Ratios above 1 are faster.
These are fixed-budget diagnostics, **not v1.0 or equal-quality acceptance**.

| Case | Transport speedup | Whole-process speedup |
|---|---:|---:|
| Lens / path | 1.046× | 0.757× |
| Glass / light | 1.001× | 0.767× |
| Peacock / VCM | 0.977× | 0.925× |
| Window / SPPM | 1.004× | 0.984× |
| Glass / light, 1024×768, 16 passes | 1.246× | 0.989× |

The small cases are near parity in transport; the larger many-pass case measures
the film change more directly. Whole-process regressions remain visible and
startup/loading still needs work. Diagnostic counts are unchanged. The largest
fixed-seed relative pixel L2 difference is 4.14e-05 (VCM); this is not an unbiasedness proof.
See the [four-case record](../benchmarks/results/20261013_maintenance.json) and
[film-workload record](../benchmarks/results/20261013_film_workload.json).

## Open acceptance criteria

| Work | Current evidence and remaining requirement |
|---|---|
| 4× over v1.0 at equal image error | **Not achieved.** The 10-09 checkpoint was about 2.96× slower on the two compatible comparisons. Later local gains do not establish a full-suite result. Compare the same scene/assets, estimator, seeds and error threshold. |
| Camera-kernel cost | The historical Peacock profile assigned about 79% of kernel time to camera paths. Submission and film changes do not remove that bottleneck. Specialized kernels and a wavefront prototype still need separate measurements. |
| Production transport | Fixed 10-26: (1) the Window panel's fan of 150 mm × 0.6 mm sliver triangles let float Möller–Trumbore miss a face near its far edge; both backends now use the watertight triangle test in `core/shared`. (2) A ray grazing a concave fold of a mesh (the Peacock filament tori) could start the ordinary GPU traversal behind the next face, rounded to the eye-relative float spacing; a wrong-sided hit is now walked again in the mesh's own frame. (3) The Peacock lead-in wires ended inside their filaments; they now stop 30 µm short. (4) Ties between two faces that cannot change the ray's region (touching opaque bodies, air-sided sheets) are ordered by stable ID on the GPU instead of stopping the path; ties into different regions still stop. Each has a regression test. Gallery captures 10-26: Window Room 10 inconsistencies in 6.3e8 paths and 1 leak, Wall 26 in 1.25e9 and 0; Peacock Room 276 and Shade 234 in 5.9e8 each, 0 leaks. Open: the Window Room leak; rare float32 cases at sub-millimetre bodies (a torus facet edge, a lead-wire cap); the Peacock window sill sunk 2 cm into its wall. Kunstkammer, Gregorian and other earlier captures still record diagnostics ([inventory](gallery/README.md)). |
| Loading | Disk pipeline caching exists (bounded at 128 MB since 10-26: it kept every build's pipelines and had reached 342 MB). Each tracing kernel was one inlined `main` of 2–3 MB SPIR-V, half of it spectrum evaluation; with spectra evaluated by a call (10-26) kernels are 1–1.4 MB and the uncached VCM kernels of a 64×48 prism render compile in 68 s instead of 217 s (NVIDIA's own 1 GB shader cache hides compiles until it is full). Loading was measured 10-26: meshes, parsing and the world build take under 0.5 s of the Window scene's load; decoding image textures (16k JPEGs of 2–3 s each, mostly halved afterwards by the texture budget) took the rest, one image at a time. Images are now decoded in parallel under a 4 GB cap (`OWE_TEXTURE_DECODE_MB`): Kunstkammer 30 → 7.7 s, Peacock 16.7 → 5.7 s, Gregorian 13.3 → 5.7 s, Window 11.3 → 5.6 s; peak memory 3.2 → 4.4 GB. A persistent cache of the budgeted texels would remove most of the rest; mesh/BVH caching is not where the time goes. |
| Gallery quality | Promoted 10-26 under the rate gate: Peacock Room and Shade, Window Wall. The Peacock views keep a few dozen isolated bright pixels on the shade (rare high-weight VCM paths); Kunstkammer has isolated bright noise. Window Room is withheld (one leak). Captures before 10-26 shade glTF models with wrong file normals wherever a node is rotated or non-uniformly scaled (fixed, with a regression test): the Kunstkammer astrolabe and globe and the Gregorian chest need recapturing. |
| Window photograph | Done 10-26: permission confirmed by the user (use as the solver target and publication of the derived renders; the photograph itself is not redistributed); the Wall view is in the gallery. |
| Manifold next-event estimation | Unimplemented. Needs a separate estimator, PDF/MIS derivation and independent glint/caustic validation. |
| Code structure | Done 10-26. Formulas both backends evaluate live once in `core/shared` (C++ double, Slang float). GPU contact resolution is one file around its one rule (`contact.slang`: a ray leaving region R meets next the nearest crossing met from R's side), and the CPU's `World::intersect` states the same rule as a candidate step and an ordering rule. The CPU is the double-precision reference for path tracing and the lens analyser; light tracing and the bidirectional estimators exist on the GPU only. Long functions are split into named steps: the GPU kernels (VCM strategies, path steps, emitters, per-material samplers, observer and sensor rays), the CPU path tracer, glTF import, BVH build, prescriptions, cameras, lens analysis and path drawings; only `walkTransmittance` (101 lines, a no-inline kernel boundary kept scalar to avoid register spills) exceeds 100 lines. Same-seed images are bit-identical across the restructuring where the estimator is deterministic (CPU; GPU path, light and SPPM). |

## Evidence and reproduction

- [Benchmark contract](BENCHMARKS.md), [suite definitions](../benchmarks/suites.json),
  [corrected checkpoint](../benchmarks/results/20261009_corrected_checkpoint.json).
- [Historical performance measurements](archive/2026-10/PERFORMANCE_STATUS.md),
  [gallery records and diagnostics](../benchmarks/results/20261010_gallery.json).
- [Contact invariants](TRANSPORT_CONTACTS.md), [regression fixtures](../benchmarks/regressions/README.md).
- [Gallery workflow](GALLERY_WORKFLOW.md), [capture diagnostics](gallery/README.md),
  [asset credits](ASSET_CREDITS.md).
- Current scene specifications: [Gregorian](scenes/the_gregorian.md),
  [Peacock Lamp](scenes/the_peacock_lamp.md), [Kunstkammer](scenes/the_kunstkammer.md),
  [Invisible Window](scenes/the_invisible_window.md).

Build with `cmake --build build -j 4`; run `ctest --test-dir build --output-on-failure`.
Python tests require NumPy and Pillow; install `tools/requirements-test.txt` into the
interpreter selected by CMake (`-DPython3_EXECUTABLE=/path/to/python`).
GPU tests require a Vulkan device. The software-traversal conformance run uses
`OWE_GPU_RAY_QUERY=0`; the normal run covers hardware traversal where available.
