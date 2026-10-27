Historical record, retained for provenance. Current state: [STATUS.md](../../STATUS.md)

# Validated gallery engine — 2026-10-09 measurements

The selected engine passed the full hardware suite **157/157** and targeted
software traversal checks **21/21**, with no skips. The unchanged hardware
smooth-glass comparison passes. Layering and the three viewer checks also pass.

Two serial, unprofiled repeats on the RTX 4060 compared identical seed-1 settings
against the frozen corrected checkpoint. These are fixed-budget native transport
speedups, not comparisons against v1 or evidence that the full 4× target is met.

| View | Estimator / samples | Resolution | Speedup range |
|---|---|---|---:|
| Window / Wall | SPPM / 2; 16 photons per pixel | 960×600 | 2.258× |
| Saturn / SaturnEyepiece | path / 16 | 256×192 | 1.319–1.327× |
| lens / Eye | path / 64 | 256×192 | 1.094–1.095× |
| Peacock / Room | VCM / 16 | 960×600 | 1.069–1.071× |

Every case in both repeats matches or stays within 0.3% of the better shared
solver's transport time. Window, Saturn and lens raw images are exact against
the corrected checkpoint. Peacock VCM differences remain within the observed
repeat variation: mean absolute differences 2.612–2.615e-7, maximum absolute
differences 1.72–2.40e-4. Paths, segments, inconsistency and leak counts are
unchanged in every pair; parallel first-diagnostic identity can vary. Existing
production diagnostics remain unresolved.

[Recorded engine results](../../../benchmarks/results/20261009_gallery_engine.json)
retain immutable binary/kernel/input identities, commands, image hashes,
reference errors and validation scope. Cold driver startup is included in wall
measurements but excluded from these transport speedups. The measured Window
input is preserved before its later gallery-setting adoption; gallery results
are recorded separately. Performance against v1 and the larger 4× objective
remain follow-up work.

# Historical corrected checkpoint results — 2026-10-09

These measurements precede the current contact optimization and final gallery revisions.
The active milestone is tracked in [NEXT_ITERATION.md](NEXT_ITERATION.md).

The seven planned benchmark sweeps completed with exit code zero: two canonical
8192-sample reference seeds, historical/current canonical budgets 16/64/256/1024,
two scan 256-iteration reference seeds and current scan budgets 16/64. Total sweep
wall time was 7.82 hours. No further capture or gallery promotion is part of this
checkpoint. The earlier measurements below remain historical, not current performance claims.

[Repository-contained results](../../../benchmarks/results/20261009_corrected_checkpoint.json)
retain commands, binary identity, validation coverage, native reference statistics,
raw-image hashes, candidate/historical measurements and independent-seed errors.
Large raw images and frozen binaries remain locally under `out/performance_20261008/`.

**The 4× goal is not met.** The strict observed equal-error comparison matches only
2/8 canonical cases. Their diagnostic partial geometric-mean speedup is 0.338×
(about 2.96 times slower); no complete canonical or twelve-case speedup exists.
Six candidate targets were not reached within the measured budgets. Finite-reference
noise, nonmonotonic bright-path errors and changed visibility also limit interpretation.
Historical scan map/material support remains incompatible, so no historical scan
speedup is reported. Passing process exit codes do not establish clean transport.

The table uses the seed-1000 native transport time and diagnostics. Seed-pair relMSE
is reported in both directions because the denominator depends on the reference.
These are full unmasked linear-image errors, not confidence intervals or proofs of convergence.

| Fixed view | Reference samples | Transport seconds | Seed-pair relMSE, both directions | Inconsistencies / leaks |
|---|---:|---:|---:|---:|
| the_lens | 8192 | 26.66 | 18.151 / 11.383 | 63 / 11 |
| prism_in_sunlight | 8192 | 25.72 | 3382.4 / 4074.1 | 21 / 5 |
| glass_of_water | 8192 | 114.19 | 0.18597 / 0.18366 | 50 / 1 |
| study_magnifier | 8192 | 218.87 | 13.637 / 4.4272 | 69 / 0 |
| the_telescope | 8192 | 79.38 | 0.0060613 / 0.0060632 | 50 / 1 |
| study_camera | 8192 | 151.25 | 115.95 / 108.69 | 204 / 0 |
| observatory_saturn | 8192 | 2178.05 | 0.093941 / 0.21006 | 5078 / 1 |
| camera_obscura | 8192 | 90.16 | 4184.7 / 3607.4 | 12 / 0 |
| the_kunstkammer | 256 | 81.83 | 0.23776 / 0.2107 | 174 / 0 |
| the_gregorian | 256 | 41.37 | 19.908 / 19.218 | 19 / 0 |
| the_peacock_lamp | 256 | 479.49 | 7.5758 / 39.853 | 1824773 / 0 |
| the_invisible_window | 256 | 8104.42 | 0.0061405 / 0.0062644 | 569 / 66 |

At this checkpoint, the Window reference alone took about 135 minutes. Source inspection
identified repeated contact-interval traversal: `transmittanceSpawn` proved the target
interval, then its first ordered `intersectSpawnWorld` call proved it again. Certified
interval reuse and native mesh-owner BVH traversal were proposed here and have since
been implemented in the current milestone. An endpoint-verified interior occlusion fast
path remained a separate proposal. The historical table does not measure those changes;
their performance and final validation must use the later executable and paired evidence.
The previously proposed grid-level packing must not reuse primitive-identity bits
now occupied by finite-connection correctness data.

The earlier integrated checkpoint passed 25/25 targeted hardware and 21/21 targeted software
checks, plus a 16-total-iteration software SPPM smoke with zero inconsistencies/leaks.
The unchanged full smooth-glass assertion passes on hardware. The software smoke does
not replace its 2048-sample assertion, and these targeted runs are not a new full GPU
suite. Production diagnostics above remain unresolved and preclude a clean release claim.

---

# Historical GPU performance baseline — 2026-10-08

The sections below retain the 8 October experiment narrative, including ideas that were
subsequently implemented or superseded. References to “next” candidates describe that
experiment date, not the current milestone.

Measured on the working tree at that date with the VCM cache replay fix, using the NVIDIA GeForce RTX 4060 Laptop GPU, Vulkan 1.4.351, hardware ray queries. This was a baseline for that experiment; it is not a comparison with an older implementation and establishes no 4× speedup.

```sh
OWE_GPU_PROFILE=1 build/owe render scenes/the_peacock_lamp.owe \
  --backend gpu --device 1 --detector Room --integrator vcm \
  --resolution 960x600 --spp 16 --max-depth 16 \
  --out out/performance_20261008/peacock_profile
```

The second run used the same command with `OWE_GPU_PROFILE=0` and output prefix `peacock_profile_off`. Scene seed: 17; scene FNV-1a hash: `056e69c67cc6642c`. Existing pipeline caches stayed enabled. No GPU tests ran concurrently.

| Measurement | Profiling on | Profiling off |
|---|---:|---:|
| Renderer time | 18.3854 s | 18.3927 s |
| End-to-end wall time | 38.707296 s | 38.126636 s |
| Peak process RSS | 2,674,892 KiB | 2,681,152 KiB |
| Mean Y | 1.20054e-5 | 1.20042e-5 |

Both runs accepted 18,432,000 paths and 114,068,071 segments, with zero leaks and 116,207 region inconsistencies. The first inconsistency was at a lead side boundary (`Lead1A.side` / `Lead2A.side`); this scene therefore is not a clean geometry-conformance reference. Neither run reported cache overflow. The 0.04% renderer-time difference between this single on/off pair does not establish a measurable profiling penalty or speed gain.

GPU timestamp breakdown for the profiled run:

| Kernel | Dispatches | GPU time | Share of kernel time |
|---|---:|---:|---:|
| VCM camera, ray queries | 16 | 14,483.837 ms | 79.148% |
| VCM light, ray queries | 84 | 3,799.112 ms | 20.761% |
| VCM grid | 16 | 16.198 ms | 0.089% |
| Guide CDF | 16 | 0.502 ms | 0.003% |

Kernel total: 18.299649 s. Submission timestamps contain those kernel timings; **do not add kernel and submission totals**. The camera/grid/counter submissions took 14.500073 s GPU / 14.505498 s host elapsed; light submissions took 3.799121 s GPU / 3.815970 s host elapsed. Final film readback took 7.003 ms GPU / 7.427 ms host elapsed. Scene uploads took 158.388 ms GPU. The approximately 20 s difference between wall and renderer times includes startup and output work; these measurements do not isolate its individual causes. Texture budgeting reported 1,924 MB after reducing a 6,532 MB source set to fit the 2,048 MB budget.

The next optimization target is the camera kernel, which accounts for 79% of measured kernel time. First distinguish its continuation traversal, next-event visibility, cached-vertex connections, and merging costs; then optimize the dominant camera-side work while preserving sampling and MIS. Grid construction, guide updates, and final readback are too small here to explain a substantial renderer speedup. Retain the overflow/recovery regressions and compare image quality at matched time for any estimator-changing experiment.

## Visual check of the bounded reference

The later 960×600, 256-spp baseline reference was opened for inspection. Composition: the full
shade and base have clear silhouettes, balanced by the book; the large dark upper-left area
remains a deliberate but strong negative space. Lighting: the shade is the clear subject and
its cast light reads on the wall, while much of the bronze base remains nearly black.
Materials: the glass pattern and wood grain are legible, but the background still has visible
sampling grain. These observations do not establish convergence or resolve the recorded lead
contact inconsistencies. Scene geometry, materials and lighting were kept fixed for benchmarking.

Artifacts: [profile log](../../../benchmarks/evidence/performance_20261008/peacock_profile.stderr.log), [profile metadata](../../../benchmarks/evidence/performance_20261008/peacock_profile.json), [profile-off log](../../../benchmarks/evidence/performance_20261008/peacock_profile_off.stderr.log), and [profile-off metadata](../../../benchmarks/evidence/performance_20261008/peacock_profile_off.json). `/usr/bin/time` was unavailable; the launcher used Python `time.perf_counter()` and `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` for wall time and peak RSS. Renderer and GPU timings come from the renderer itself.

## Measured camera optimization — 2026-10-08

The first tables above describe the initial profiled scene baseline with seed 17. This additional comparison uses seed **1** for both frozen executables and the same scene, detector, 960×600 resolution, VCM, 16 spp, max depth 16, and default 2,048 MB texture budget. The baseline already includes correct cache-overflow replay. The camera candidate freezes the shader change before the separate planetary-geometry patch, so this comparison isolates the camera optimization.

`cameraDiffuseResponse` caches each diffuse camera hit's four spectral BSDF values. `evalCameraVcm` shares its direction/hemisphere/shading-normal/pdf checks across wavelengths for next-event estimation, light-vertex connection and merging. Other materials keep their existing spectral BSDF evaluation. The change preserves RNG draws, visibility queries, path sampling, MIS weights and the photon set.

| Bounded measurement | Frozen baseline | Frozen diffuse-cache candidate |
|---|---:|---:|
| Transport time, 16 spp | 18.3959 s | 15.5010 s |
| End-to-end wall time | 40.973923 s | 35.130355 s |
| Full-image relMSE against frozen VCM 256-spp reference | 0.9352956 | 0.9267776 |
| Diagnostic relMSE with glint mask | 0.8942389 | 0.8857074 |
| Mean/reference ratio, raw linear RGB | 0.8988885 | 0.8988176 |

Observed transport speedup: **1.187×**, or **15.7% less transport time**. This is one bounded baseline/candidate pair, not a 4× result or a time-to-target curve. The candidate capture enabled diagnostic GPU timestamps; the bounded baseline disabled them. The earlier on/off pair found no measurable timestamp cost at this budget, but repeat unprofiled measurements remain appropriate before treating the ratio as a stable performance guarantee. The baseline capture ran during some host shader compilation, so wall-time differences are not attributable solely to this optimization.

The candidate camera kernel took **11.591910 s** GPU, its light kernel **3.793790 s**, grid **16.195 ms**, and guide **0.500 ms**. Kernel and submission timings remain nested. Accepted counts match the bounded baseline exactly: 18,432,000 paths, 114,051,169 segments, 116,278 region inconsistencies, zero leaks. Neither lamp capture reported overflow replay.

Correctness evidence for the isolated camera candidate:

- All five existing `vcm_` tests passed, including cache overflow/recovery, agreement with path tracing, a glass-plate caustic, and glossy reflection. The first test took 95.76 s including cold pipeline compilation; the remaining steady tests completed in approximately 16 s.
- A 32×32 coloured-checker/relief room, 64 spp, seed 7, produced **bitwise equal raw PFM images and equal statistics** with `OWE_VCM_MASK=12` (connections and merging disabled).
- With `OWE_VCM_MASK=4` (merging retained), the maximum raw image difference was **1.0431e-7**, mean relative absolute difference **5.98e-9**, and statistics matched exactly. The remaining difference is compatible with unordered cache/hash float summation.
- The new integrated regression, `vcm_cached_diffuse_packets_match_coloured_textures_and_shading_normals`, passed with the final planetary-geometry correction in both hardware and software traversal. It checks BDPT/VCM XYZ means against path tracing for a coloured checker with relief viewed from both sides; the worst printed ratio deviation was 0.00144 (approximately 0.14%). The complete final focused subset passed 7/7 in each traversal mode; this correctness result is separate from the frozen shader performance comparison.

The frozen 64-spp baseline took 63.2678 s but had full-image relMSE 4.030536 against the same finite 256-spp reference (diagnostic masked relMSE 0.2563183). That nonmonotonic full-image result prevents a reliable convergence or time-to-quality claim from these two budgets. Existing scene region inconsistencies and the finite stochastic reference also limit interpretation; the measured optimization preserves the estimator rather than resolving those limitations.

The next camera candidate is to cache the four wavelengths' **CIE XYZ colour-matching values** once at path start. `Wavelengths.toXYZ` currently recomputes the same CIE lobes for every contribution, including every accepted merge. Reusing only those direction-independent values while retaining the current wavelength pdfs, hero termination, multiplication and accumulation order could remove repeated transcendental work. It is not implemented or credited in the speedup above; extra register pressure and exact/matched-error checks need measurement first.

Artifacts: [bounded baseline summary](../../../benchmarks/evidence/performance_20261008/bounded_baseline/summary.json), [candidate comparison](../../../benchmarks/evidence/performance_20261008/camera_candidate_comparison.json), [candidate profiler log](../../../benchmarks/evidence/performance_20261008/camera_candidate_profile.stderr.log), [masked comparison](../../../benchmarks/evidence/performance_20261008/camera_mask_comparison.json), and [frozen candidate test log](../../../benchmarks/evidence/performance_20261008/camera_candidate_vcm_tests.stderr.log). Frozen executables are `out/performance_20261008/owe_baseline` and `out/performance_20261008/owe_camera_candidate`.

## Historical integrated sanity result — 2026-10-08

After integrating the diffuse cache and the corrected planetary sphere reconstruction, the final focused suites passed **7/7 with hardware traversal and 7/7 with software traversal**. Logs: [hardware tests](../../../benchmarks/evidence/performance_20261008/planetary_final_hardware_tests.log), [software tests](../../../benchmarks/evidence/performance_20261008/planetary_final_software_tests.log). These include cache overflow/rejection/recovery and the new coloured diffuse front/back shading-normal regression.

A single final **profiling-off** lamp render used the same seed-1, Room, 960×600, VCM, 16-spp, max-depth-16 settings as the frozen bounded baseline:

| Measurement | Frozen bounded baseline | Final integrated renderer |
|---|---:|---:|
| Transport time | 18.3959 s | 16.1294 s |
| Full-image relMSE vs frozen 256-spp reference | 0.9352956 | 0.9258913 |
| Mean/reference ratio, raw linear RGB | 0.8988885 | 0.8988852 |

Observed integrated speedup: **1.1405×**, or **12.3% less transport time**. Accepted path/segment counts and diagnostics match the baseline. This is one sanity point, not a stable time-to-target curve or a 4× result; it supersedes the isolated shader result when describing the final combined implementation. Differences between the isolated and integrated captures are not sufficient to assign a precise cost to the geometry correction.

Artifacts: [integrated render log](../../../benchmarks/evidence/performance_20261008/integrated_lamp16.log), [renderer metadata](../../../benchmarks/evidence/performance_20261008/integrated_lamp16.json), and [reference error metrics](../../../benchmarks/evidence/performance_20261008/integrated_lamp16.error.json). No further optimization is credited beyond the measured diffuse packet cache; CIE-value caching remains a proposed next experiment.

## Camera CIE cache — 2026-10-08

The next isolated GPU candidate caches the four wavelengths' CIE XYZ colour-matching values once per valid camera path. Each contribution still divides by the **current** packet pdf, including hero termination on either the camera or light side, and retains the original multiplication and wavelength accumulation order. Sampling, traversal, MIS and the light cache are unchanged. This costs twelve additional live float values in the camera kernel; the measured benefit below is modest.

The frozen candidate contains a separate CPU contact prototype, but GPU attribution was checked directly: **14 of 16 embedded kernels are byte-identical** to `owe_integrated_baseline`; only the two VCM camera variants differ. Camera ray-query SPIR-V shrank from 3,696,204 to 3,675,724 bytes, and the duplicated CIE evaluation sequences decreased from twenty to four. Static instruction counts establish that repeated code was removed; they are not runtime operation counts.

All captures used RTX 4060 device 1, Room, 960×600, VCM, depth 16, seed 1, the default 2,048 MB texture budget and retained pipeline caches. There were no concurrent GPU jobs or heavy host builds. The comparison uses the new [full integrated 256-spp baseline](../../../benchmarks/evidence/performance_20261008/full_integrated_baseline256/summary.json), rather than the earlier pre-diffuse baseline:

| Capture | Transport time | Full-image relMSE vs integrated reference256 | Raw RGB mean/reference |
|---|---:|---:|---:|
| Recent integrated baseline, 16 spp | 15.7980 s | 0.9317588 | 0.8988544 |
| CIE candidate, 16 spp, profiling off | 15.1529 s | 0.9323029 | 0.8987993 |
| Recent integrated baseline, 64 spp | 55.2453 s | 4.0214925 | 0.8870851 |
| CIE candidate, 64 spp, profiling off | 53.2466 s | 4.0214716 | 0.8870876 |
| Fresh paired integrated baseline, 64 spp, profiling off | 55.2653 s | 4.0217266 | 0.8870591 |

The fresh 64-spp pair measured **1.0379×**, or **3.65% less transport time**. The recent 16-spp comparison measured 4.08% less transport time. During the paired 64-spp renders, device samples showed 72°C for the candidate and 73°C for the baseline, both at 2,085 MHz graphics / 8,001 MHz memory and 99–100% utilization. End-to-end wall times were 71.55 s and 73.57 s respectively; peak process RSS was approximately 2.78 GB for both. These are bounded captures, not a timing distribution or a 4× result.

The 16-spp profiled candidate took 15.1519 s transport, with **11.173876 s camera**, **3.875368 s light**, 16.213 ms grid and 0.500 ms guide GPU time. Profiling off took 15.1529 s. Kernel timestamps are nested within submission timestamps and must not be added to them. The camera remains the dominant optimization target.

Correctness evidence:

- The frozen hardware VCM subset passed **7/7**, including overflow/rejection/recovery, coloured diffuse/shading normals, path agreement, glass-plate caustics and glossy reflection.
- The new `vcm_camera_cie_keeps_xyz_normalization_after_dispersion` test checks a lossless BK7 plate in an isotropic coloured sky. BDPT and VCM XYZ ratios against path tracing were 0.99962, 1.00013 and 0.99940; a stale pre-termination pdf would incorrectly scale the hero-only result by one quarter.
- The coloured fixture with `OWE_VCM_MASK=12` remained **bitwise equal** between frozen integrated baseline and candidate. With merging retained (`OWE_VCM_MASK=4`), maximum raw-image difference was 1.0431e-7 and relative mean absolute difference 7.32e-9. Statistics matched in both cases.
- Production accepted path/segment counts, inconsistencies and leaks matched. At 64 spp these were 73,728,000 paths, 456,252,136 segments, 466,568 inconsistencies and zero leaks. Differences in stochastic cache ordering can still affect individual connections and image error.

The finite 256-spp reference, retained lead-contact inconsistencies and nonmonotonic full-image error at 64 spp prevent a time-to-target claim. The 16-spp candidate error is slightly higher than its baseline, so this capture establishes preserved expected behaviour and a transport-time reduction, not improved quality. The CIE candidate has not yet been separately tested with software traversal; the next integrated hardware/software validation must include its new dispersion test.

A next bounded candidate can reuse the grid level already computed during insertion, storing it in unused light-vertex flag bits to avoid repeated square-root/logarithm work during camera merging. It requires no additional buffer memory and will be measured separately. Hashing wavelength groups separately cannot improve the current default single-group production scene and is deferred.

Artifacts: [candidate comparison](../../../benchmarks/evidence/performance_20261008/cie_candidate_comparison.json), [embedded kernel identity check](../../../benchmarks/evidence/performance_20261008/cie_embedded_kernel_comparison.json), [hardware tests](../../../benchmarks/evidence/performance_20261008/cie_vcm_hardware_tests.log), [raw masked comparisons](../../../benchmarks/evidence/performance_20261008/cie_mask_comparison.json), and [camera profile](../../../benchmarks/evidence/performance_20261008/cie_profile16.stderr.log). Frozen executables: `owe_integrated_baseline`, `owe_cie_candidate` and `owe_tests_cie_candidate` under `out/performance_20261008`.
