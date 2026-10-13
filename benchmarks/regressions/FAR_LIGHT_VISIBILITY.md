# Distant particle connections and near-observer occlusion

`far_light_visibility.owe` is a standalone, asset-free physical oracle. An observer
at the origin looks at a diffuse sheet 1,000,000 m away. An off-axis local emitter
illuminates that sheet. A 20 mm opaque black screen 25 mm in front of the observer
covers the entire 1 mm pupil and field; there is no other light source. Every
physical estimator must therefore produce **exactly zero XYZ in every pixel**.
Moving the screen to x=1 m supplies the positive open control. The source geometry
is metres across and twenty metres from the diffuser, avoiding sub-ULP far-origin
features and unrelated emitter-spawn offset failures on float32.

The former CPU finite visibility ray shortened its far-to-near segment by
`1e-7 * distance`, skipping the final 0.1 m here. The GPU shortened it by
`1e-5 * distance`, skipping the final 10 m. Thus a distant diffuse-vertex splat can
skip the nearby blocker even though independently generated camera paths hit it.
Endpoint region agreement and zero diagnostic counts do not catch this error:
the thin screen has ambient on both sides but still absorbs the ray.

Run from the repository root, with a built renderer:

```sh
build/owe render benchmarks/regressions/far_light_visibility.owe --backend cpu \
  --integrator light --out /tmp/far_blocked
build/owe render benchmarks/regressions/far_light_visibility.owe --backend cpu \
  --integrator light --set 'NearBlocker.position=(1,0,0.025)' --out /tmp/far_open
build/owe render benchmarks/regressions/far_light_visibility.owe --backend cpu \
  --integrator path --spp 8 --out /tmp/far_camera_blocked
build/owe render benchmarks/regressions/far_light_visibility.owe --backend gpu \
  --device 1 --integrator vcm --out /tmp/far_vcm_blocked
```

Use the same open-screen edit for GPU light/BDPT/VCM controls. The black oracle
needs no reference image, error metric, tone mapping, denoising or noise tolerance;
the control must have nonzero finite radiance. A separate direct-emitter variant
uses `--set 'FarDiffuser.material=black' --set 'Lamp.position=(0,0,999980)'` to
exercise the particle's emitter-to-observer connection without a diffuse bounce.

Historical evidence, 8 October 2026, frozen `owe_integrated_baseline` SHA-256
`b08531232c341ba367a1b433fd4771851e944d399868b9a38455f28381abca7a`,
scene FNV `3bac0dd9cc2eece7`, seed11, max depth3, 16×16, CPU two threads:

| Estimator | Screen | Samples | Mean linear Y | Inconsistencies / leaks |
|---|---|---:|---:|---:|
| light | blocked | 128 | 0.00283122 | 0 / 0 |
| light | open | 128 | 0.00283122 | 0 / 0 |
| path | blocked | 8 | 0 | 0 / 0 |
| light, direct-emitter variant | blocked | 128 | 0.0412923 | 0 / 0 |
| light, direct-emitter variant | open | 128 | 0.0412923 | 0 / 0 |

Blocked and open particle renders are identical despite the opaque screen. Raw
artifacts/logs are under `out/performance_20261008/far_visibility/integrated_cpu_*`.
These tiny diagnostic timings are not performance evidence.

Frozen integrated GPU hardware traversal, device1, same scene/seed, 32 samples
(64 for the sensor), also fails the zero-light oracle:

| Estimator / variant | Blocked mean linear Y | Open mean linear Y |
|---|---:|---:|
| light, diffuse vertex | 0.00306109 | 0.00306109 |
| BDPT, diffuse vertex | 0.0000660490 | 0.00287344 |
| VCM, diffuse vertex | 0.0000574576 | 0.00287610 |
| light, direct emitter | 0.0411807 | 0.0411807 |
| path, surface sensor / cover | 3.14135 | 3.14135 |

Blocked GPU camera-path control returns exactly zero. Every GPU reproduction
reports zero inconsistencies and zero leaks. The surface-sensor case is the
independent `sensor_opaque_contact.owe` oracle: a sensor at an opaque cover must
measure zero; raising it 1 mm measures uniform-sky irradiance pi. Its old displaced
launch skips the distinct cover. Artifacts are `integrated_gpu_*` in the same
output directory. Fixed CPU/GPU hardware/software validation is pending the
contact/reciprocal-visibility build.

`tests/test_far_visibility.cpp` uses the public renderer and loader interfaces,
checks path and particle/BDPT/VCM blocked images against exact zero, and requires a
positive open control. It exercises diffuse-vertex and direct-emitter strategies
independently. GPU checks skip explicitly when unavailable and run under the
existing hardware/software backend test selections when available.

The sensor camera regression now includes a 100 nm positive gap as well as the
1 mm control. Both resolved gaps must measure pi without inconsistencies or
leaks; exact contact must remain dark. This distinguishes suppressing a true
contact from incorrectly rejecting a separate nearby surface.

The isolated collinear hardware-path checkpoint passes this sensor test on CPU
and RTX 4060: exact-contact mean Y is zero, while both positive gaps give
3.14135048 (CPU) and 3.14135061 (GPU), with zero inconsistencies/leaks. Together
with four GPU contact/attenuation tests, the explicitly selected suite passed
5/5. Logs, SPV, shader/test source snapshot and binary/link-input hashes are in
`out/performance_20261008/contact_collinear_probe`. Only the embedded hardware
path shader and two test objects were replaced in the temporary binaries;
this does not certify particle/BDPT/VCM kernels or the software driver.

The corresponding portable BVH path shader also passes the same five tests on
the RTX 4060 with `OWE_GPU_RAY_QUERY=0`; this is software traversal on GPU
hardware, not a CPU Vulkan driver. It exited normally with peak host RSS
3,972,936 KiB (3.79 GiB), measured by an isolated Python child-resource helper.
Logs, memory sidecar and identities are in
`out/performance_20261008/contact_collinear_sw_probe`. Cold shader compilation
is included in this memory measurement; neither probe is a throughput claim.

The additional `sensor_particle_contact.owe` fixture replaces the sky with a
finite downward-facing emitting disk. It checks actual physical-sensor particle
hits using CPU/GPU light tracing, with exact-zero covered signal
and a positive control after raising the sensor by 1 mm. The frozen integrated
CPU light estimator gives 0.0877554 in both configurations (128 samples, seed11).
An unresolved incoming contact may report an ambiguity, but it must not splat
energy through the opaque cover. The open control has resolved separation and
must remain free of contact diagnostics. BDPT requires an ideal observer and is
not a supported physical-sensor estimator; this fixture does not request it.

`far_nee_visibility.owe` exercises a different strategy: the camera sees a near
diffuse receiver illuminated by a finite source 1,000,000 m away. The source is
completely screened 5 cm before its emitting face. Physical camera-path radiance
must be exactly zero; the side-open and beyond-source controls must retain equal
positive images. The high source radiance compensates inverse-square dilution,
without introducing sky or another source. It tests finite next-event estimation,
which the direct-emitter particle oracle does not exercise.

The frozen integrated CPU path estimator returns **0.00254645 in all three
configurations**, with bitwise identical PFM SHA-256
`a84dff1a3d3857c8ad8e83efc2f6d1c115a7404a67ae807b81eef7a8079d41fb`.
Each uses 16×16, 128 spp, seed 11, depth 2, two threads; all report 32,768 paths,
65,536 segments and zero inconsistencies/leaks. Fixture FNV is
`d755e3bba440d7f0`. Evidence is preserved as `integrated_cpu_nee_*` in the same
output directory, including `integrated_cpu_nee_evidence.json` with binary,
fixture and raw-image hashes. The regression
`backends_camera_nee_respects_source_end_occlusion` checks available path, BDPT
and VCM camera estimators. The fixed CPU path candidate passes: the blocked image has exactly zero
XYZ, and the open and beyond-source images match pixelwise with mean Y
0.00254644605 and zero inconsistencies/leaks. The independently run focused
CPU suite passed 21/21; GPU cases explicitly skipped in that CPU-only build.
Its log is `out/performance_20261008/contact_nee_cpu_focused_tests.log`; the
candidate checkpoint then named `owe_cpu_contact_nee_candidate` had SHA-256
`89ca02065dc4013191b33acdf55c30420db8bc85f8dafdbaffbb915c3fa4bd71`.
That generic executable path was subsequently replaced; the hash and log identify
this historical validation checkpoint, not the current executable at that path.
GPU reproduction and fixed hardware/software validation remain pending. The
expanded BDPT/VCM test has not run on a fixed GPU build; the CPU path result does
not certify VCM's separate direct-NEE or cached-vertex connection implementations.
